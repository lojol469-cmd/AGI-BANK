# -*- coding: utf-8 -*-
import os
import sys
import json
from typing import Dict, Any, List

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from . import BaseTool, run_in_portable

class UltraLibInstaller(BaseTool):
    @property
    def name(self) -> str:
        return "ultra_lib_installer"

    @property
    def description(self) -> str:
        return (
            "🔒 OUTIL PREMIUM (Installateur de Librairies).\n"
            "Permet d'installer n'importe quelle librairie Python manquante (ex: python-docx, pywin32, pandas).\n"
            "⚠️ FORMAT STRICT REQUIS : {\"libs\": [\"python-docx\", \"pywin32\"]}"
        )

    @property
    def capabilities(self) -> List[str]:
        return ["installation", "pip", "librairies", "premium", "système"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Le JSON ou le nom de la librairie à installer."}
            },
            "required": ["query"]
        }

    def can_handle(self, query: str, context: dict = None) -> float:
        if any(kw in query.lower() for kw in ["install", "pip", "manquant", "module not found", "no module", "librairie"]):
            return 0.99
        return 0.0

    def execute(self, query: str, context: dict = None) -> dict:
        print(f"💎 [Ultra Lib Installer] Requête reçue : {query}")
        
        script_code = f"""
import json
import subprocess
import sys

clean_input = {json.dumps(query)}.replace("```json", "").replace("```", "").strip()
try:
    payload = json.loads(clean_input)
    libs = payload.get("libs", [])
except json.JSONDecodeError:
    libs = [clean_input] if clean_input else []

if not libs:
    print(json.dumps({{"success": False, "output": "❌ Aucune librairie spécifiée. Exemple: {{\\"libs\\": [\\"python-docx\\"]}}"}}))
    sys.exit(0)

python_exe = sys.executable
results = []
for lib in libs:
    try:
        res = subprocess.run(
            [python_exe, "-m", "pip", "install", lib], 
            capture_output=True, text=True, timeout=120
        )
        if res.returncode == 0:
            results.append(f"✅ {lib} installé avec succès.")
        else:
            results.append(f"❌ Échec de {lib}: {{res.stderr[:200]}}")
    except Exception as e:
        results.append(f"❌ Erreur sur {lib}: {{str(e)}}")

print(json.dumps({{"success": True, "output": "\\n".join(results)}}))
"""
        return run_in_portable(script_code)