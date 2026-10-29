# -*- coding: utf-8 -*-
import requests
import json

RAG_URL = "http://127.0.0.1:8765"

def inspect():
    print("🔍 Inspection profonde du serveur RAG...\n")
    
    # 1. Récupération de tous les outils
    try:
        r = requests.get(f"{RAG_URL}/tool_schemas", timeout=10)
        tools = r.json().get("tools", [])
    except Exception as e:
        print(f"❌ Impossible de contacter le RAG : {e}")
        return

    # 2. Filtrage des outils de recherche documentaire
    print("--- OUTILS DE RECHERCHE DOCUMENTAIRE DISPONIBLES ---")
    search_keywords = ["rag", "doc", "search", "fiscal", "article", "chunk", "module"]
    doc_tools = []
    
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        
        if any(k in name.lower() or k in desc.lower() for k in search_keywords):
            doc_tools.append(fn)
            print(f"🔹 {name}")
            print(f"   Desc: {desc[:150]}...")
            print(f"   Args: {json.dumps(fn.get('parameters', {}).get('properties', {}), indent=2)}\n")

    # 3. Test réel de recherche (query = "TVA")
    print("\n--- TEST RÉEL DE RECHERCHE (Query: 'TVA') ---")
    test_query = "TVA"
    
    endpoints_to_test = [
        {"tool_name": "rag_search", "args": {"query": test_query, "k": 3}},
        {"tool_name": "fiscal_search", "args": {"query": test_query, "use_web": False}},
        {"tool_name": "rag_smart_search", "args": {"query": test_query, "top_k": 3}},
        {"tool_name": "search_chunks", "args": {"query": test_query, "top_k": 3}},
        {"tool_name": "report_dgi_article", "args": {"query": test_query, "k": 3}}
    ]

    for ep in endpoints_to_test:
        print(f"▶️ Test de l'outil : {ep['tool_name']}...")
        try:
            resp = requests.post(f"{RAG_URL}/execute_tool", json=ep, timeout=30)
            data = resp.json()
            
            # Extraction du résultat (parfois imbriqué dans "result" en string JSON)
            result = data.get("result", data)
            if isinstance(result, str):
                try: result = json.loads(result)
                except: pass
                
            # Vérification s'il y a des résultats
            if isinstance(result, dict) and "results" in result:
                results = result["results"]
                if results:
                    print(f"  ✅ SUCCÈS ! {len(results)} résultat(s) trouvé(s).")
                    first_res = results[0]
                    text_preview = str(first_res.get("text", first_res))[:200].replace('\n', ' ')
                    print(f"  📄 Aperçu : {text_preview}...\n")
                else:
                    print(f"  ⚠️ Réponse valide, mais liste de résultats vide.\n")
            else:
                print(f"  ⚠️ Réponse brute : {str(result)[:200]}...\n")
                
        except Exception as e:
            print(f"  ❌ Erreur lors de l'appel : {e}\n")

if __name__ == "__main__":
    inspect()