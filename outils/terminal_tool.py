import subprocess
import os
from . import BaseTool

class TerminalTool(BaseTool):
    @property
    def name(self): 
        return "execute_python"
    
    @property
    def description(self): 
        return "Exécute du code Python sur la machine. Utilise 'query' pour passer ton code Python. L'environnement a accès à des centaines de bibliothèques (pandas, torch, etc.) et s'exécute dans le dossier D:\\kibali-IA\\workspace."
    
    @property
    def capabilities(self): 
        return ["execute", "terminal", "code", "system"]
    
    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "Le code Python complet et exécutable à lancer (pas besoin de balises markdown)."
                }
            },
            "required": ["query"]
        }
        
    def can_handle(self, query: str, context: dict = None) -> float:
        # Score élevé si l'utilisateur demande explicitement de coder ou manipuler des fichiers
        if any(k in query.lower() for k in ["code", "python", "script", "terminal", "dossier", "fichier", "execute", "lance", "installe"]):
            return 0.95
        return 0.1
            
    def execute(self, query: str, context: dict = None) -> dict:
        python_exe = r"D:\kibali-IA\env_portable\python.exe"
        workspace = r"D:\kibali-IA\workspace"
        os.makedirs(workspace, exist_ok=True)
        
        script_path = os.path.join(workspace, "temp_script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(query)
            
        try:
            # Timeout de 120s pour éviter que l'AGI ne bloque le serveur
            result = subprocess.run(
                [python_exe, script_path], 
                capture_output=True, 
                text=True, 
                timeout=120, 
                cwd=workspace
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\n[ERREUR / STDERR]\n{result.stderr}"
                
            return {
                "success": result.returncode == 0, 
                "output": output or "Succès (le script n'a produit aucune sortie)."
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "Le script a dépassé le temps imparti (120s)."}
        except Exception as e:
            return {"success": False, "output": str(e)}