import requests
import json

print("Téléchargement des schémas d'outils depuis le serveur RAG...")
try:
    r = requests.get("http://127.0.0.1:8765/tool_schemas", timeout=10)
    r.raise_for_status()
    data = r.json()
    
    with open("kibali_tools.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"✅ {data.get('count', 0)} outils sauvegardés dans kibali_tools.json")
except Exception as e:
    print(f"❌ Erreur : {e}")