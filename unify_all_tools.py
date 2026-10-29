# -*- coding: utf-8 -*-
import pathlib

# Le code parfait, basé sur la logique smolagents native
standard_code = '''# -*- coding: utf-8 -*-
import os
import subprocess
import tempfile

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class {CLASS_NAME}(BaseTool):
    @property
    def name(self) -> str:
        return "{TOOL_NAME}"

    @property
    def description(self) -> str:
        return "{DESCRIPTION}"

    @property
    def capabilities(self) -> list:
        return {CAPABILITIES}

    def can_handle(self, query: str, context: dict = None) -> float:
        if any(kw in query.lower() for kw in {CAPABILITIES}):
            return 0.8
        return 0.0

    def execute(self, query: str, context: dict = None) -> dict:
        # Nettoyage des balises markdown
        code = query.replace("```python", "").replace("```", "").strip()

        # L'environnement portable
        portable_python = r"D:\\kibali-IA\\env_portable\\python.exe"
        if not os.path.exists(portable_python):
            portable_python = r"D:\\CPT_Analysis_Studio_PORTABLE\\environment\\python.exe"

        try:
            # Écriture du code BRUT généré par l'IA
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(code)
                tmp_path = f.name

            # Exécution avec le Python Portable
            result = subprocess.run([portable_python, tmp_path], capture_output=True, text=True, timeout=120)
            os.unlink(tmp_path)

            if result.returncode == 0:
                return {"success": True, "output": result.stdout or "✅ Exécuté avec succès."}
            else:
                # On renvoie l'erreur pour que l'IA se corrige elle-même (Self-Healing)
                return {"success": False, "error": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}
'''

tools_to_create = [
    {
        "file": r"D:\kibali-IA\outils\agi_coder_tool.py",
        "class_name": "AgiCoderTool",
        "tool_name": "agi_coder_tool",
        "description": "Outil pour coder. Exécute du code Python brut avec les libs (pandas, torch, etc.).",
        "capabilities": '["coder", "python", "script", "projet", "calcul"]'
    },
    {
        "file": r"D:\kibali-IA\outils\python_executor_tool.py",
        "class_name": "PythonExecutorTool",
        "tool_name": "python_executor",
        "description": "Exécute du code Python brut.",
        "capabilities": '["python", "script", "exécuter"]'
    },
    {
        "file": r"D:\kibali-IA\outils\gui_launcher_tool.py",
        "class_name": "GuiLauncherTool",
        "tool_name": "gui_launcher_tool",
        "description": "Exécute du code Python pour ouvrir des fichiers, créer des interfaces PyQt6, ou piloter Excel.",
        "capabilities": '["gui", "pyqt6", "ouvrir", "excel", "word", "application"]'
    },
    {
        "file": r"D:\kibali-IA\outils\excel_com_tool.py",
        "class_name": "ExcelComTool",
        "tool_name": "excel_com_tool",
        "description": "Exécute du code Python brut pour automatiser Excel (win32com).",
        "capabilities": '["excel", "vba", "macro", "cellule"]'
    }
]

for t in tools_to_create:
    target = pathlib.Path(t["file"])
    
    # On formate le code avec les infos de l'outil
    final_code = standard_code.replace("{CLASS_NAME}", t["class_name"])
    final_code = final_code.replace("{TOOL_NAME}", t["tool_name"])
    final_code = final_code.replace("{DESCRIPTION}", t["description"])
    final_code = final_code.replace("{CAPABILITIES}", t["capabilities"])
    
    target.write_text(final_code, encoding="utf-8")
    print(f"✅ Outil standardisé : {target.name}")