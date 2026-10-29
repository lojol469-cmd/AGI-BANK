# -*- coding: utf-8 -*-
import json
import requests
import os
import sys
from typing import Dict, Any, List

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from . import BaseTool

class RagBridgeTool(BaseTool):
    @property
    def name(self) -> str:
        return "rag_bridge"

    @property
    def description(self) -> str:
        return """Passerelle vers le RAG Kibali (390 outils métier, documents, mémoire).
        Actions :
        - action='search_tools' : Cherche un outil (ex: 'audit excel', 'email').
        - action='run_tool' : Exécute un outil (args: tool_name, tool_args).
        - action='search_docs' : Cherche dans les documents/lois (ex: 'TVA', 'article 12').
        - action='search_memory' : Cherche dans les situations passées."""

    @property
    def capabilities(self) -> List[str]:
        return ["rag", "documents", "tools", "search", "fiscal", "memory"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string", 
                    "enum": ["search_tools", "run_tool", "search_docs", "search_memory"],
                    "description": "L'action à effectuer."
                },
                "query": {"type": "string", "description": "Mots-clés."},
                "tool_name": {"type": "string", "description": "Nom de l'outil RAG."},
                "tool_args": {"type": "string", "description": "Arguments JSON de l'outil."}
            },
            "required": ["action"]
        }

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        q = query.lower()
        # Score élevé si l'utilisateur demande une recherche de doc ou d'outil
        if any(k in q for k in ["rag", "document", "loi", "article", "fiscal", "mémoire", "outil", "cherche"]):
            return 0.85
        return 0.1

    def execute(self, action: str, query: str = "", tool_name: str = "", tool_args: str = "{}", **kwargs) -> Dict[str, Any]:
        RAG_URL = "http://127.0.0.1:8765/execute_tool"
        
        if action == "search_tools":
            target_tool = "search_tools"
            args = {"query": query, "top_k": 5}
        elif action == "search_docs":
            target_tool = "rag_search" 
            args = {"query": query, "k": 3}
        elif action == "search_memory":
            target_tool = "recall_situations"
            args = {"problem": query, "top_k": 3}
        elif action == "run_tool":
            if not tool_name: return {"success": False, "error": "tool_name requis."}
            target_tool = "run_tool"
            try: parsed_args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
            except: parsed_args = {}
            args = {"tool_name": tool_name, "arguments": json.dumps(parsed_args)}
        else:
            return {"success": False, "error": f"Action inconnue : {action}"}

        payload = {"tool_name": target_tool, "args": args}
        try:
            response = requests.post(RAG_URL, json=payload, timeout=120)
            response.raise_for_status()
            res = response.json()
            
            output = res.get("result", res)
            if isinstance(output, str):
                try: output = json.loads(output)
                except: pass
                
            return self._format_response(output, action, query)
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Serveur RAG introuvable."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _format_response(self, output: Any, action: str, query: str) -> Dict[str, Any]:
        if isinstance(output, str):
            return {"success": True, "output": output if output.strip() else "⚠️ Aucun résultat."}
        if isinstance(output, dict) and "results" in output:
            results = output["results"]
            if not results: return {"success": True, "output": "⚠️ Aucun résultat."}
            formatted = f"🔍 Résultats pour '{query}' :\n\n"
            for i, r in enumerate(results, 1):
                source = r.get("source", r.get("name", "Inconnu"))
                text = r.get("text", r.get("description", str(r)))
                formatted += f"[{i}] {source}\n{text[:1500]}\n" + "-"*40 + "\n"
            return {"success": True, "output": formatted}
        return {"success": True, "output": json.dumps(output, ensure_ascii=False, indent=2)}