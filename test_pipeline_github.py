import os
import sys
import requests
import zipfile
import io
import ast
import time
import shutil
import stat
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 🐍 1. FORCER L'UTILISATION DU PYTHON PORTABLE
PYTHON_EXE = r"D:\kibali-IA\env_portable\python.exe"
if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable # Fallback
print(f"🐍 Python Portable utilisé : {PYTHON_EXE}")

# 🧠 2. EMBEDDINGS ROBUSTES (Sans dépasser la limite de 512 tokens)
class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        all_embeddings = []
        print(f"🧠 Génération des embeddings pour {len(texts)} éléments...")
        
        for i, text in enumerate(texts):
            # On tronque à 800 caractères pour être sûr de ne pas dépasser 512 tokens
            safe_text = text[:800] 
            
            try:
                r = requests.post(
                    "http://127.0.0.1:8081/v1/embeddings", 
                    json={"input": [safe_text], "model": "bge-m3"}, 
                    timeout=30
                )
                
                if r.status_code != 200:
                    print(f"❌ Erreur serveur: {r.text}")
                    all_embeddings.append([0.0] * 1024)
                    continue
                    
                data = r.json()
                all_embeddings.append(data["data"][0]["embedding"])
                
                if i % 10 == 0:
                    print(f"   ... {i}/{len(texts)} complétés")
                    time.sleep(0.05)
                    
            except Exception as e:
                print(f"❌ Exception : {e}")
                all_embeddings.append([0.0] * 1024)
                
        return all_embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

embedding_model = LocalEmbeddings()

def on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        os.unlink(path)
    except Exception:
        pass

# 📥 3. TÉLÉCHARGEMENT VIA L'API GITHUB (Sans Git)
def download_github_zip(repo_url, target_dir):
    """Télécharge un repo GitHub en ZIP via l'API, sans utiliser Git."""
    print(f"📥 Téléchargement de {repo_url} via l'API GitHub...")
    
    # On nettoie l'URL et on prépare le lien de l'archive
    base_url = repo_url.rstrip('/')
    
    # On essaie la branche 'main' puis 'master'
    for branch in ['main', 'master']:
        zip_url = f"{base_url}/archive/refs/heads/{branch}.zip"
        r = requests.get(zip_url, stream=True, timeout=30)
        if r.status_code == 200:
            break
    
    if r.status_code != 200:
        print(f"❌ Impossible de télécharger l'archive (Code: {r.status_code})")
        return False

    # Extraction du ZIP
    print("📦 Extraction de l'archive...")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(target_dir)
    
    # Le ZIP crée un dossier à l'intérieur (ex: requests-main). On doit le déplacer
    extracted_folders = os.listdir(target_dir)
    if extracted_folders:
        inner_folder = os.path.join(target_dir, extracted_folders[0])
        if os.path.isdir(inner_folder):
            for item in os.listdir(inner_folder):
                shutil.move(os.path.join(inner_folder, item), target_dir)
            os.rmdir(inner_folder)
            
    return True

def chunk_python_code(code, filename):
    docs = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1
                end_line = node.end_lineno
                source_code = "\n".join(code.splitlines()[start_line:end_line])
                
                # 🛡️ On coupe à 1000 caractères pour éviter l'erreur 512 tokens
                if len(source_code) > 1000:
                    source_code = source_code[:1000] + "\n...[Code tronqué]..."
                
                doc = Document(
                    page_content=f"Composant: {node.name}\nFichier: {filename}\n\n{source_code}",
                    metadata={"file": filename, "name": node.name, "type": type(node).__name__}
                )
                docs.append(doc)
    except Exception:
        pass
    return docs

def test_pipeline(repo_url, project_name):
    print(f"\n{'='*50}\n🚀 TEST PIPELINE : {project_name}\n{'='*50}")
    
    timestamp = int(time.time())
    repo_dir = os.path.join(r"D:\kibali-IA\temp_repos", f"{project_name}_{timestamp}")
    faiss_dir = os.path.join(r"D:\kibali-IA\code_rag", f"{project_name}_{timestamp}")
    
    os.makedirs(repo_dir, exist_ok=True)
    
    # 1. TELECHARGEMENT
    if not download_github_zip(repo_url, repo_dir):
        return
    
    # 2. CHUNKING
    print("✂️ Découpage du code (AST)...")
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
                        docs = chunk_python_code(code, rel_path)
                        all_docs.extend(docs)
                except: pass
                
    print(f"🧩 {len(all_docs)} composants trouvés (fonctions/classes).")
    
    if not all_docs:
        print("❌ Aucun composant Python valide trouvé.")
        return

    # 3. INDEXATION FAISS
    print("🧠 Indexation dans FAISS (Embeddings BGE-M3)...")
    # On limite à 50 docs pour ce test (en prod on fera tout)
    vectordb = FAISS.from_documents(all_docs[:50], embedding_model)
    
    os.makedirs(faiss_dir, exist_ok=True)
    vectordb.save_local(faiss_dir)
    print(f"💾 Index sauvegardé dans : {faiss_dir}")

    # 4. RECHERCHE
    query = "connection session http"
    print(f"\n🔍 Test de recherche sémantique : '{query}'")
    results = vectordb.similarity_search(query, k=3)
    
    for i, res in enumerate(results, 1):
        print(f"\n--- Résultat {i} ---")
        print(f"Fichier : {res.metadata['file']}")
        print(f"Composant : {res.metadata['name']}")
        print(f"Extrait :\n{res.page_content[:200]}...")

    print("\n🧹 Nettoyage du dossier temporaire...")
    try:
        shutil.rmtree(repo_dir, onerror=on_rm_error)
    except:
        pass

if __name__ == "__main__":
    test_pipeline("https://github.com/psf/requests", "requests_light")