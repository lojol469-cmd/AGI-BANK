import requests
import json

base_url = "http://127.0.0.1:8765"
query = "audit de sécurité"

print("1. Test de connexion au serveur RAG...")
try:
    res_health = requests.get(f"{base_url}/health")
    print(f"   Status /health : {res_health.status_code}")
except Exception as e:
    print(f"   ERREUR : Le serveur n'est pas accessible. ({e})")
    exit()

print("\n2. Recherche de l'outil pour 'audit de sécurité'...")
res_search = requests.post(
    f"{base_url}/search_tools", 
    json={"query": query, "top_k": 1}
)

if res_search.status_code == 200:
    data = res_search.json()
    print(f"   Succès ! Réponse :\n{json.dumps(data, indent=2, ensure_ascii=False)[:500]}...")
else:
    print(f"   Erreur {res_search.status_code} : {res_search.text}")