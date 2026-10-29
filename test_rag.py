# -*- coding: utf-8 -*-
import json
import urllib.request
import urllib.error

RAG_URL = "http://127.0.0.1:8765/execute_tool"
TOOL_NAME = "rag_smart_search"
QUERY = "définition de l'impôt sur les sociétés IS RDC"

# Liste des formats à tester
payloads = [
    # Format 1 : args
    {"tool_name": TOOL_NAME, "args": {"query": QUERY}},
    # Format 2 : arguments
    {"tool_name": TOOL_NAME, "arguments": {"query": QUERY}},
    # Format 3 : query direct
    {"tool_name": TOOL_NAME, "query": QUERY},
    # Format 4 : input direct
    {"tool_name": TOOL_NAME, "input": QUERY},
    # Format 5 : tool + args
    {"tool": TOOL_NAME, "args": {"query": QUERY}},
]

def test_payload(payload, index):
    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(RAG_URL, data=data, headers={'Content-Type': 'application/json'})
        
        print(f"\n--- Test Format {index} ---")
        print(f"Payload envoyé : {json.dumps(payload, ensure_ascii=False)}")
        
        with urllib.request.urlopen(req, timeout=15) as response:
            status = response.status
            body = response.read().decode('utf-8')
            print(f"✅ SUCCÈS (Status {status})")
            print(f"Réponse : {body[:500]}...")
            return True
            
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8')
        print(f"❌ ÉCHEC (Status {e.code})")
        print(f"Erreur : {body}")
        return False
    except Exception as e:
        print(f"❌ ERREUR RÉSEAU : {e}")
        return False

if __name__ == "__main__":
    print(f"🔍 Test de connexion au RAG : {RAG_URL}")
    print(f"🎯 Outil ciblé : {TOOL_NAME}")
    print(f"🔎 Recherche : '{QUERY}'")
    
    for i, payload in enumerate(payloads, 1):
        if test_payload(payload, i):
            print("\n✅✅✅ FORMAT TROUVÉ ! C'est le format numéro", i, "✅✅✅")
            break