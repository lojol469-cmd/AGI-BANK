import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        r = requests.post("http://127.0.0.1:8081/v1/embeddings", json={"input": texts, "model": "bge-m3"})
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]
    def embed_query(self, text):
        return self.embed_documents([text])[0]

def init_memory():
    # On cherche le dossier de mémoire (généralement dans kibali_data)
    path = os.path.join(os.getcwd(), "kibali_data", "chat_memory")
    
    if not os.path.exists(path):
        print(f"📁 Création du dossier de mémoire : {path}")
        os.makedirs(path, exist_ok=True)
        
        # On crée un document fantôme pour initialiser FAISS
        doc = Document(page_content="Initialisation de la mémoire Kibali", metadata={"source": "system"})
        vectordb = FAISS.from_documents([doc], LocalEmbeddings())
        vectordb.save_local(path)
        print("✅ Mémoire conversationnelle initialisée avec succès !")
    else:
        print(f"✅ Le dossier de mémoire existe déjà : {path}")

if __name__ == "__main__":
    init_memory()