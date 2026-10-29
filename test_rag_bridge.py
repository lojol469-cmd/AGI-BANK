# -*- coding: utf-8 -*-
import json
import requests
from typing import Dict, Any

class RagBridgeTool:
    @property
    def name(self) -> str:
        return "rag_bridge"

    @property
    def description(self) -> str:
        return """Passerelle centrale vers le RAG Kibali. Permet de chercher des OUTILS, des DOCUMENTS (textes fiscaux/lois), ou d'interroger la MÉMOIRE.
        Actions possibles :
        - action='search_tools' : Cherche un outil par mots-clés (ex: 'email', 'audit excel').
        - action='run_tool' : Exécute un outil spécifique. (args: tool_name, tool_args)
        - action='search_docs' : Cherche dans la base documentaire (ex: 'TVA', 'article 12').
        - action='recall_memory' : Cherche dans les situations passées (ex: 'erreur python')."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string", 
                    "enum": ["search_tools", "run_tool", "search_docs", "recall_memory"],
                    "description": "L'action à effectuer."
                },
                "query": {
                    "type": "string", 
                    "description": "Mots-clés de recherche (pour search_tools, search_docs, recall_memory)."
                },
                "tool_name": {
                    "type": "string",
                    "description": "Le nom exact de l'outil à exécuter (si action='run_tool')."
                },
                "tool_args": {
                    "type": "string",
                    "description": "Arguments pour l'outil en format JSON (si action='run_tool'). Ex: '{\"limit\": 5}'"
                }
            },
            "required": ["action"]
        }

    def execute(self, action: str, query: str = "", tool_name: str = "", tool_args: str = "{}", **kwargs) -> Dict[str, Any]:
        RAG_URL = "http://127.0.0.1:8765/execute_tool"
        
        if action == "search_tools":
            target_tool = "search_tools"
            args = {"query": query, "top_k": 5}
        elif action == "search_docs":
            # On utilise l'outil de recherche RAG intelligent du serveur
            target_tool = "rag_smart_search" 
            args = {"query": query, "top_k": 3}
        elif action == "recall_memory":
            target_tool = "recall_situations"
            args = {"problem": query, "top_k": 3}
        elif action == "run_tool":
            if not tool_name:
                return {"success": False, "error": "Le paramètre 'tool_name' est requis."}
            target_tool = "run_tool"
            try:
                parsed_args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
            except:
                parsed_args = {}
            args = {"tool_name": tool_name, "arguments": json.dumps(parsed_args)}
        else:
            return {"success": False, "error": f"Action inconnue : {action}"}

        payload = {"tool_name": target_tool, "args": args}
        
        try:
            response = requests.post(RAG_URL, json=payload, timeout=120)
            response.raise_for_status()
            res = response.json()
            
            # Nettoyage de la réponse
            output = res.get("result", res)
            if isinstance(output, str):
                try: output = json.loads(output)
                except: pass
                
            # Formatage propre pour le LLM
            if action in ["search_tools", "search_docs", "recall_memory"] and isinstance(output, dict) and "results" in output:
                results = output["results"]
                if not results: return {"success": True, "output": "⚠️ Aucun résultat trouvé."}
                
                formatted = f"🔍 Résultats pour '{query}' :\n\n"
                for i, r in enumerate(results, 1):
                    # Gestion des différents formats de retour (outils vs docs vs mémoire)
                    title = r.get("name") or r.get("source") or r.get("problem") or "Inconnu"
                    text = r.get("description") or r.get("text") or r.get("solution") or str(r)
                    score = r.get("score", "")
                    
                    formatted += f"[{i}] {title} (Score: {score})\n"
                    formatted += f"📝 {text[:1000]}\n" + "-"*40 + "\n"
                return {"success": True, "output": formatted}
                
            return {"success": True, "output": json.dumps(output, ensure_ascii=False, indent=2)}

        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Serveur RAG introuvable sur le port 8765."}
        except Exception as e:
            return {"success": False, "error": str(e)}

# =====================================================================
# SCRIPT DE TEST
# =====================================================================
if __name__ == "__main__":
    tool = RagBridgeTool()
    
    print("\n--- TEST 1 : Recherche d'outils ---")
    res = tool.execute(action="search_tools", query="audit excel")
    print(res.get("output") or res.get("error"))
    
    print("\n--- TEST 2 : Recherche documentaire (Fiscalité) ---")
    res = tool.execute(action="search_docs", query="TVA")
    print(res.get("output") or res.get("error"))

    print("\n--- TEST 3 : Exécution d'un outil (avec bonne catégorie) ---")
    res = tool.execute(action="run_tool", tool_name="list_tools", tool_args='{"category": "files"}')
    print(res.get("output") or res.get("error"))