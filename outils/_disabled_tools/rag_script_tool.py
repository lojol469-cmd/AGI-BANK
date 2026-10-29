# -*- coding: utf-8 -*-
import subprocess
import os
import sys
from typing import Dict, Any, List

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from . import BaseTool

class RagScriptTool(BaseTool):
    @property
    def name(self) -> str:
        return "execute_rag_python"

    @property
    def description(self) -> str:
        return """Exécute du code Python pour interagir avec le RAG Kibali (390 outils, docs, mémoire).
        Un SDK est automatiquement pré-importé :
        - `rag_tools()` : Retourne la liste de tous les outils RAG (JSON).
        - `rag_run(tool_name, **kwargs)` : Exécute un outil RAG.
        Utilise `print()` pour afficher les résultats."""

    @property
    def capabilities(self) -> List[str]:
        return ["rag", "python", "search", "tools", "memory", "documents"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "Le code Python à exécuter. Ex: `tools = rag_tools(); print([t['function']['name'] for t in tools if 'email' in t['function']['name']])`"
                }
            },
            "required": ["query"]
        }

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        q = query.lower()
        if any(k in q for k in ["rag", "outil", "recherche", "tool", "document", "fiscal", "mémoire", "imap", "email"]):
            return 0.9
        return 0.1

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        python_exe = r"D:\kibali-IA\env_portable\python.exe"
        workspace = r"D:\kibali-IA\workspace"
        os.makedirs(workspace, exist_ok=True)
        
        prelude = """
import requests
import json

RAG_BASE = "http://127.0.0.1:8765"
RAG_EXEC = f"{RAG_BASE}/execute_tool"

def rag_tools():
    try:
        r = requests.get(f"{RAG_BASE}/tool_schemas", timeout=10)
        return r.json().get("tools", [])
    except Exception as e:
        print(f"[SDK Erreur] rag_tools: {e}")
        return []

def rag_run(tool_name, **kwargs):
    try:
        payload = {"tool_name": tool_name, "args": kwargs}
        r = requests.post(RAG_EXEC, json=payload, timeout=120)
        return r.json()
    except Exception as e:
        print(f"[SDK Erreur] rag_run: {e}")
        return {"error": str(e)}

# --- CODE DE L'AGI ---
"""
        full_code = prelude + "\n" + query
        script_path = os.path.join(workspace, "temp_rag_script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_code)
            
        try:
            result = subprocess.run([python_exe, script_path], capture_output=True, text=True, timeout=120, cwd=workspace)
            output = result.stdout
            if result.stderr: output += f"\n[STDERR]\n{result.stderr}"
            return {"success": result.returncode == 0, "output": output or "Succès (aucune sortie)."}
        except Exception as e:
            return {"success": False, "output": str(e)}