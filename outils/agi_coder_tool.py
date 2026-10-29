# -*- coding: utf-8 -*-
import os
import subprocess
import tempfile

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class AgiCoderTool(BaseTool):
    @property
    def name(self) -> str:
        return "agi_coder_tool"

    @property
    def description(self) -> str:
        return "Outil pour coder. Exécute du code Python brut avec les libs (pandas, torch, etc.)."

    @property
    def capabilities(self) -> list:
        return ["coder", "python", "script", "projet", "calcul"]

    def can_handle(self, query: str, context: dict = None) -> float:
        if any(kw in query.lower() for kw in ["coder", "python", "script", "projet", "calcul"]):
            return 0.8
        return 0.0

    def execute(self, query: str, context: dict = None) -> dict:
        # Nettoyage des balises markdown
        code = query.replace("```python", "").replace("```", "").strip()

        # L'environnement portable
        portable_python = r"D:\kibali-IA\env_portable\python.exe"
        if not os.path.exists(portable_python):
            portable_python = r"D:\CPT_Analysis_Studio_PORTABLE\environment\python.exe"

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
