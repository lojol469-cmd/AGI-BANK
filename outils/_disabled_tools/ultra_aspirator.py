import json
import os
import requests
import zipfile
import io
import ast
import time
import shutil
import stat
import sys
import contextlib
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

# 🧠 Embeddings (avec troncature pour éviter l'erreur 512 tokens)
class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        all_embeddings = []
        for text in texts:
            safe_text = text[:800]
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

def on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)
    except: pass

def chunk_python_code(code, filename):
    docs = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1
                end_line = node.end_lineno
                source_code = "\n".join(code.splitlines()[start_line:end_line])
                if len(source_code) > 1000: source_code = source_code[:1000] + "\n...[Tronqué]..."
                
                doc = Document(
                    page_content=f"Composant: {node.name}\nFichier: {filename}\n\n{source_code}",
                    metadata={"file": filename, "name": node.name, "type": type(node).__name__}
                )
                docs.append(doc)
    except: pass
    return docs

class UltraAspirator(BaseTool):
    @property
    def name(self) -> str:
        return "ultra_aspirator"

    @property
    def description(self) -> str:
        return (
            "🔒 OUTIL PREMIUM ULTRA (Aspirateur GitHub + Code RAG).\n"
            "Permet de télécharger un projet GitHub, de l'analyser, et de trouver du code réutilisable.\n\n"
            "⚠️ MÉTHODE 1 (Recommandée) : Envoie un JSON strict.\n"
            "{\"action\": \"aspire_and_search\", \"url\": \"https://github.com/user/repo\", \"query\": \"ce que tu cherches\"}\n\n"
            "🚨 MÉTHODE 2 (Si tu n'arrives pas à faire le JSON) : Copie-colle EXACTEMENT ce script et exécute-le avec l'outil `python_executor` :\n"
            "```python\n"
            "import json, sys\n"
            "sys.path.insert(0, r'D:\\kibali-IA')\n"
            "from outils.ultra_aspirator import UltraAspirator\n"
            "tool = UltraAspirator()\n"
            "req = json.dumps({'action': 'aspire_and_search', 'url': 'https://github.com/psf/requests', 'query': 'connection http'})\n"
            "print(tool.execute(req))\n"
            "```\n"
            "Fais ça si tu bloques, ça marchera à 100%."
        )

    @property
    def capabilities(self) -> list:
        return ["aspiration", "github", "code", "creation_logiciel", "premium"]

    def can_handle(self, query: str, context: dict = None) -> float:
        if any(kw in query.lower() for kw in ["aspire", "github", "projet", "code source", "répare", "analyse le code", "fonction"]):
            return 0.99
        return 0.0

    def execute(self, query: str, context: dict = None) -> dict:
        print(f"💎 [Ultra Aspirator] Requête reçue : {query}")
        
        # 🚀 Si l'IA a utilisé la Méthode 2 (Code Python)
        if "import json, sys" in query or "UltraAspirator()" in query:
            try:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(query, globals())
                return {"success": True, "output": output.getvalue()}
            except Exception as e:
                return {"success": False, "error": f"Erreur script Python : {e}"}

        # 📦 Si l'IA a utilisé la Méthode 1 (JSON)
        clean_input = query.replace("```json", "").replace("```", "").strip()
        try:
            payload = json.loads(clean_input)
        except json.JSONDecodeError:
            return {"success": False, "error": "❌ JSON invalide. Utilise la Méthode 2 (le script Python)."}

        action = payload.get("action", "")
        url = payload.get("url")
        search_query = payload.get("query", "")

        if action == "aspire_and_search":
            if not url or "github.com" not in url:
                return {"success": False, "error": "❌ URL GitHub invalide."}
            
            timestamp = int(time.time())
            repo_name = url.rstrip('/').split('/')[-1].replace('.git', '')
            repo_dir = os.path.join(r"D:\kibali-IA\temp_repos", f"{repo_name}_{timestamp}")
            os.makedirs(repo_dir, exist_ok=True)
            
            print(f"   📥 Téléchargement de {url}...")
            base_url = url.rstrip('/')
            r = None
            for branch in ['main', 'master']:
                zip_url = f"{base_url}/archive/refs/heads/{branch}.zip"
                r = requests.get(zip_url, stream=True, timeout=30)
                if r.status_code == 200: break
            
            if r is None or r.status_code != 200:
                return {"success": False, "error": "❌ Impossible de télécharger l'archive GitHub."}

            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(repo_dir)
            
            for folder in os.listdir(repo_dir):
                inner = os.path.join(repo_dir, folder)
                if os.path.isdir(inner):
                    for item in os.listdir(inner):
                        shutil.move(os.path.join(inner, item), repo_dir)
                    os.rmdir(inner)
                    break

            print("   ✂️ Analyse du code (AST) et FAISS...")
            all_docs = []
            for root, _, files in os.walk(repo_dir):
                if ".git" in root: continue
                for f in files:
                    if f.endswith('.py'):
                        file_path = os.path.join(root, f)
                        rel_path = os.path.relpath(file_path, repo_dir)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as file:
                                code = file.read()
                                all_docs.extend(chunk_python_code(code, rel_path))
                        except: pass

            if not all_docs:
                return {"success": False, "error": "❌ Aucun composant Python trouvé dans ce projet."}

            vectordb = FAISS.from_documents(all_docs[:100], embedding_model)
            
            print(f"   🔍 Recherche sémantique : '{search_query}'...")
            results = vectordb.similarity_search(search_query, k=3)
            
            try: shutil.rmtree(repo_dir, onerror=on_rm_error)
            except: pass

            output_text = f"✅ Projet '{repo_name}' analysé ({len(all_docs)} composants trouvés).\n"
            output_text += f"🔍 Voici le code le plus pertinent pour '{search_query}' :\n\n"
            
            for i, res in enumerate(results, 1):
                output_text += f"--- Code {i} ({res.metadata['name']}) ---\n"
                output_text += f"Fichier : {res.metadata['file']}\n"
                output_text += f"{res.page_content[:800]}\n\n"

            return {"success": True, "output": output_text}

        return {"success": False, "error": "Action inconnue. Utilise 'aspire_and_search'."}