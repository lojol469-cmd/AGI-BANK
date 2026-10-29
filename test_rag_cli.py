import requests
import json

RAG_URL = "http://127.0.0.1:8765/search_docs"

queries = [
    "article 221 taux normal 18% TVA",
    "taux zero exportation exoneration article 221 CGI",
    "Code général des impôts Gabon TVA exonération",
    "TVA Gabon exonération article",
    "Loi de finances 2025 TVA exonération Gabon"
]

print("🚀 TEST DES 5 METHODES DE RECHERCHE RAG\n")

for i, q in enumerate(queries, 1):
    print(f"--- MÉTHODE {i} : '{q}' ---")
    try:
        payload = {"query": q, "top_k": 1}
        res = requests.post(RAG_URL, json=payload, timeout=10)
        
        if res.status_code == 200:
            data = res.json()
            
            # 🛠️ CORRECTION ICI : On gère le cas où le serveur renvoie une string directement
            if isinstance(data, dict):
                text_result = data.get("docs", data.get("documents", str(data)))
            else:
                text_result = str(data)
            
            # On affiche juste les 300 premiers caractères pour voir le sujet
            print(f"✅ Résultat : {text_result[:300]}...\n")
        else:
            print(f"❌ Erreur HTTP {res.status_code}\n")
            
    except Exception as e:
        print(f"❌ Erreur : {e}\n")

print("🏁 Fin des tests.")