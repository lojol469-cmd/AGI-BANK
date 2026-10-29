# -*- coding: utf-8 -*-
import os
import sys
import re
import glob
import time
import subprocess
from typing import Dict, Any, List

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from . import BaseTool

class GuiLauncherTool(BaseTool):
    @property
    def name(self) -> str:
        return "gui_launcher_tool"

    @property
    def description(self) -> str:
        return (
            "Gère les applications PyQt6. "
            "Actions : 'create' (app_name, code), 'read' (app_name), 'launch' (app_name), 'list', 'delete' (app_name). "
            "Pour AMÉLIORER une app : utilise 'read', modifie le code, puis 'create'. "
            "NE METS AUCUNE BALISE MARKDOWN DANS LE CODE."
        )

    @property
    def capabilities(self) -> List[str]:
        return ["create_app", "launch_app", "list_apps", "read_app", "delete_app", "pyqt6"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action":   {"type": "string",  "enum": ["create", "launch", "list", "read", "delete"]},
                "app_name": {"type": "string",  "description": "Nom de l'app (sans .py)"},
                "code":     {"type": "string",  "description": "Code Python complet et brut (pour create)"}
            },
            "required": ["action"]
        }

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        q = query.lower()
        if any(kw in q for kw in ["code une application", "crée une app", "pyqt6", "interface graphique", "lance l'app", "améliore l'app", "modifie l'application"]):
            return 0.9
        return 0.0

    def _get_paths(self):
        apps_dir = r"D:\kibali-IA\generated_apps"
        os.makedirs(apps_dir, exist_ok=True)
        pyw_exe = r"D:\kibali-IA\env_portable\pythonw.exe"
        if not os.path.exists(pyw_exe):
            pyw_exe = r"D:\kibali-IA\env_portable\python.exe"
        return apps_dir, pyw_exe

    def execute(self, action: str, app_name: str = "", code: str = "", **kwargs) -> Dict[str, Any]:
        apps_dir, pyw_exe = self._get_paths()
        action = action.lower()
        
        if app_name:
            app_name = re.sub(r"[^a-zA-Z0-9_]", "", app_name)

        try:
            if action == "list":
                apps = [os.path.basename(f).replace(".py", "") for f in glob.glob(os.path.join(apps_dir, "*.py"))]
                return {"success": True, "output": f"Applications : {', '.join(apps) if apps else 'Aucune'}"}

            elif action == "read":
                if not app_name: return {"success": False, "output": "Nom de l'app manquant."}
                app_path = os.path.join(apps_dir, f"{app_name}.py")
                if not os.path.exists(app_path): return {"success": False, "output": f"App '{app_name}' introuvable."}
                with open(app_path, "r", encoding="utf-8") as f:
                    return {"success": True, "output": f.read()}

            elif action == "launch":
                if not app_name: return {"success": False, "output": "Nom de l'app manquant."}
                app_path = os.path.join(apps_dir, f"{app_name}.py")
                if not os.path.exists(app_path): return {"success": False, "output": f"App '{app_name}' introuvable."}
                subprocess.Popen([pyw_exe, app_path], shell=False)
                return {"success": True, "output": f"🚀 Application '{app_name}' lancée."}

            elif action == "delete":
                if not app_name: return {"success": False, "output": "Nom de l'app manquant."}
                app_path = os.path.join(apps_dir, f"{app_name}.py")
                if os.path.exists(app_path):
                    os.remove(app_path)
                    return {"success": True, "output": f"🗑️ Application '{app_name}' supprimée."}
                return {"success": False, "output": "App introuvable."}

            elif action == "create":
                if not code: return {"success": False, "output": "Code Python manquant."}
                
                code = re.sub(r'^```[a-zA-Z]*\n', '', code, flags=re.MULTILINE)
                code = code.replace("```", "").strip()

                if not app_name: app_name = f"app_{int(time.time())}"
                app_path = os.path.join(apps_dir, f"{app_name}.py")

                # --- INJECTION DU LOGO DANS PYQT6 ---
                # Chercher l'initialisation de la fenêtre et ajouter setWindowIcon
                if "QApplication" in code and "setWindowIcon" not in code:
                    logo_path = r"D:\CPT_Analysis_Studio_PORTABLE\kibali_loops\tool\kibali-IA\kibali_logo.svg"
                    # On cherche 'app = QApplication(sys.argv)' ou similaire
                    match = re.search(r"app\s*=\s*QApplication\(.*?\)", code)
                    if match:
                        inject_code = f"\n        from PyQt6.QtGui import QIcon\n        app.setWindowIcon(QIcon(r'{logo_path}'))\n"
                        code = code[:match.end()] + inject_code + code[match.end():]

                try:
                    compile(code, app_path, "exec")
                except SyntaxError as se:
                    extrait = code[:200].replace('\n', '\\n')
                    return {"success": False, "output": f"SyntaxError ligne {se.lineno}. Code : {extrait}... Corrige-le."}

                with open(app_path, "w", encoding="utf-8") as f:
                    f.write(code)

                return {"success": True, "output": f"✅ Application '{app_name}' créée/mise à jour avec succès."}

            return {"success": False, "output": f"Action inconnue : {action}"}
        except Exception as e:
            return {"success": False, "output": f"Erreur outil: {str(e)}"}