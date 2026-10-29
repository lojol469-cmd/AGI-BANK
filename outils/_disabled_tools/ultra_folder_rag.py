import os
import json
import pandas as pd
import pdfplumber
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

# 🧠 Client d'embeddings (BGE-M3 sur le port 8081)
class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        all_embeddings = []
        for text in texts:
            safe_text = text[:800] # Troncature
            try:
                r = requests.post("http://127.0.0.1:8081/v1/embeddings", json={"input": [safe_text], "model": "bge-m3"}, timeout=30)
                if r.status_code == 200:
                    all_embeddings.append(r.json()["data"][0]["embedding"])
                else:
                    all_embeddings.append([0.0] * 1024)
            except:
                all_embeddings.append([0.0] * 1024)
        return all_embeddings
    def embed_query(self, text):
        return self.embed_documents([text])[0]

embedding_model = LocalEmbeddings()

class UltraFolderRAG(BaseTool):
    @property
    def name(self) -> str:
        return "ultra_folder_rag"

    @property
    def description(self) -> str:
        return (
            "🔒 OUTIL PREMIUM (RAG sur Dossier).\n"
            "1️⃣ Pour INDEXER un dossier : {\"action\": \"index_folder\", \"path\": \"C:\\\\chemin\\\\dossier\"}\n"
            "2️⃣ Pour CHERCHER dans le dossier indexé : {\"action\": \"search\", \"path\": \"C:\\\\chemin\\\\dossier\", \"query\": \"ta question\"}\n\n"
            "⚠️ L'outil découpe les fichiers Excel et PDF et les stocke dans une base FAISS locale. "
            "Tu DOIS d'abord indexer, puis chercher."
        )

    @property
    def capabilities(self) -> list:
        return ["rag", "dossier", "excel", "pdf", "audit", "premium"]

    def can_handle(self, query: str, context: dict = None) -> float:
        if any(kw in query.lower() for kw in ["dossier", "audit", "fichier excel", "indexe", "analyse fichier"]):
            return 0.99
        return 0.0

    def _read_excel(self, filepath):
        for engine in ['calamine', 'xlrd', 'openpyxl', None]:
            try:
                return pd.read_excel(filepath, engine=engine) if engine else pd.read_excel(filepath)
            except:
                pass
        raise Exception("Impossible de lire l'Excel.")

    def execute(self, query: str, context: dict = None) -> dict:
        print(f"💎 [Ultra Folder RAG] Requête reçue.")
        
        clean_input = query.replace("```json", "").replace("```", "").strip()
        try:
            payload = json.loads(clean_input)
        except json.JSONDecodeError:
            return {"success": False, "error": "❌ JSON invalide."}

        action = payload.get("action", "")
        path = payload.get("path")
        
        if not path or not os.path.exists(path):
            return {"success": False, "error": f"❌ Le chemin '{path}' n'existe pas."}

        folder_name = os.path.basename(path).replace(" ", "_")
        faiss_dir = os.path.join(r"D:\kibali-IA\rag_folders", folder_name)

        # ==========================================
        # ACTION 1 : INDEXER LE DOSSIER
        # ==========================================
        if action == "index_folder":
            print(f"   📂 Lecture et découpage du dossier : {path}...")
            all_docs = []
            
            for root, _, files in os.walk(path):
                for f in files:
                    file_path = os.path.join(root, f)
                    ext = f.lower().split('.')[-1]
                    
                    # PDF
                    if ext == 'pdf':
                        try:
                            with pdfplumber.open(file_path) as pdf:
                                for i, page in enumerate(pdf.pages):
                                    text = page.extract_text()
                                    if text and text.strip():
                                        all_docs.append(Document(
                                            page_content=f"Fichier: {f} (Page {i+1})\n{text}",
                                            metadata={"file": f, "page": i+1}
                                        ))
                        except: pass
                    
                    # EXCEL / CSV
                    elif ext in ['xlsx', 'xls', 'csv']:
                        try:
                            if ext == 'csv':
                                df = pd.read_csv(file_path)
                            else:
                                df = self._read_excel(file_path)
                            
                            # Nettoyage des nombres
                            for col in df.columns:
                                if df[col].dtype == 'object':
                                    cleaned = df[col].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
                                    try: df[col] = pd.to_numeric(cleaned, errors='ignore')
                                    except: pass

                            # On découpe le dataframe en petits morceaux de 10 lignes
                            for i in range(0, len(df), 10):
                                chunk_df = df.iloc[i:i+10]
                                text_data = chunk_df.to_string()
                                all_docs.append(Document(
                                    page_content=f"Fichier Excel: {f} (Lignes {i} à {i+10})\n{text_data}",
                                    metadata={"file": f, "row_start": i}
                                ))
                        except Exception as e:
                            print(f"⚠️ Erreur lecture {f}: {e}")

            if not all_docs:
                return {"success": False, "error": "❌ Aucun document lisible trouvé dans ce dossier."}

            print(f"   🧠 Indexation de {len(all_docs)} chunks dans FAISS...")
            os.makedirs(faiss_dir, exist_ok=True)
            
            # On indexe par batch de 50 pour ne pas saturer le serveur
            vectordb = None
            for i in range(0, len(all_docs), 50):
                batch = all_docs[i:i+50]
                if vectordb is None:
                    vectordb = FAISS.from_documents(batch, embedding_model)
                else:
                    vectordb.add_documents(batch)
                print(f"      ... {i+len(batch)}/{len(all_docs)} chunks indexés")

            vectordb.save_local(faiss_dir)
            return {"success": True, "output": f"✅ Dossier '{folder_name}' indexé avec succès ({len(all_docs)} chunks). Tu peux maintenant poser des questions avec l'action 'search'."}

        # ==========================================
        # ACTION 2 : RECHERCHER DANS LE DOSSIER
        # ==========================================
        elif action == "search":
            search_query = payload.get("query", "")
            if not os.path.exists(faiss_dir):
                return {"success": False, "error": f"❌ Ce dossier n'a pas encore été indexé. Lance d'abord l'action 'index_folder'."}

            print(f"   🔍 Recherche de '{search_query}' dans FAISS...")
            try:
                vectordb = FAISS.load_local(faiss_dir, embedding_model, allow_dangerous_deserialization=True)
                results = vectordb.similarity_search(search_query, k=4)
                
                output = f"🔍 **Résultats pour : '{search_query}'**\n\n"
                for i, res in enumerate(results, 1):
                    output += f"--- Extrait {i} ({res.metadata.get('file')}) ---\n"
                    output += f"{res.page_content[:1000]}\n\n"
                    
                return {"success": True, "output": output}
            except Exception as e:
                return {"success": False, "error": f"❌ Erreur de recherche : {e}"}

        return {"success": False, "error": "Action inconnue. Utilise 'index_folder' ou 'search'."}