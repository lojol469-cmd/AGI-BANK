import requests
import json

url = "http://127.0.0.1:8765"

print("1. Récupération du schéma OpenAPI...")
r = requests.get(f"{url}/openapi.json")
if r.status_code == 200:
    data = r.json()
    # Afficher les routes disponibles
    print("Routes :", list(data.get("paths", {}).keys()))
    
    # Chercher le format attendu pour /execute_tool
    if "/execute_tool" in data.get("paths", {}):
        print("\nSchéma de /execute_tool :")
        print(json.dumps(data["paths"]["/execute_tool"], indent=2)[:1000])
else:
    print("Impossible de lire OpenAPI")

print("\n2. Récupération du schéma de l'outil tva_gabon...")
r = requests.get(f"{url}/tool_schema/tva_gabon")
if r.status_code == 200:
    print(json.dumps(r.json(), indent=2)[:500])
else:
    print("Impossible de lire le schéma de tva_gabon")

print("\n3. Test d'exécution direct...")
# Test 1 : avec tool et kwargs
payload1 = {"tool": "tva_gabon", "kwargs": {}}
r1 = requests.post(f"{url}/execute_tool", json=payload1)
print(f"Test 1 {payload1} -> {r1.status_code} : {r1.text[:200]}")

# Test 2 : avec tool_name et args
payload2 = {"tool_name": "tva_gabon", "args": {}}
r2 = requests.post(f"{url}/execute_tool", json=payload2)
print(f"Test 2 {payload2} -> {r2.status_code} : {r2.text[:200]}")