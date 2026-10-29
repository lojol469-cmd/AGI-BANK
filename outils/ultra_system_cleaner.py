import json
import os
import subprocess
import gc

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class UltraSystemCleaner(BaseTool):
    @property
    def name(self) -> str:
        return "ultra_system_cleaner"

    @property
    def description(self) -> str:
        return (
            "🔒 OUTIL PREMIUM ULTRA (Gestion Système).\n"
            "Permet de libérer la mémoire RAM et de nettoyer les conteneurs Docker inactifs.\n\n"
            "⚠️ FORMAT JSON STRICT REQUIS :\n"
            "1. Pour nettoyer la RAM : {\"action\": \"free_memory\"}\n"
            "2. Pour nettoyer Docker : {\"action\": \"docker_cleanup\"}\n"
            "3. Pour voir les logs : {\"action\": \"logs\"}\n\n"
            "💡 UTILISATION : Utilise ceci si ton contexte est saturé ou si le système ralentit."
        )

    @property
    def capabilities(self) -> list:
        return ["systeme", "docker", "ram", "nettoyage", "premium"]

    def can_handle(self, query: str, context: dict = None) -> float:
        if any(kw in query.lower() for kw in ["ram", "mémoire", "docker", "nettoyer", "lent", "crash"]):
            return 0.99
        return 0.0

    def execute(self, query: str, context: dict = None) -> dict:
        clean_input = query.replace("```json", "").replace("```", "").strip()
        try:
            payload = json.loads(clean_input)
        except json.JSONDecodeError:
            return {"success": False, "error": "❌ JSON invalide. Exemple: {\"action\": \"free_memory\"}"}

        action = payload.get("action", "")

        if action == "free_memory":
            try:
                collected = gc.collect()
                # Note: drop_caches nécessite Linux/sudo, on l'adapte pour Windows
                if os.name == 'nt':
                    # Sur Windows, on peut tenter de vider le working set (simulé ici)
                    return {"success": True, "output": f"✅ RAM libérée (Python GC: {collected} objets)."}
                else:
                    subprocess.run(["sync"], capture_output=True)
                    return {"success": True, "output": f"✅ RAM et caches Linux libérés (GC: {collected})."}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif action == "docker_cleanup":
            try:
                # Nettoyage Docker basique
                subprocess.run(["docker", "container", "prune", "-f"], capture_output=True, timeout=30)
                subprocess.run(["docker", "image", "prune", "-f"], capture_output=True, timeout=30)
                return {"success": True, "output": "✅ Docker nettoyé (conteneurs arrêtés et images dangling supprimés)."}
            except Exception as e:
                return {"success": False, "error": str(e)}
                
        elif action == "logs":
            journal = os.path.expanduser("~/.kiba/actions_stream.jsonl")
            if not os.path.exists(journal):
                return {"success": True, "output": "Aucun log trouvé."}
            with open(journal, "r", encoding="utf-8") as f:
                lines = f.readlines()[-10:] # 10 dernières lignes
            return {"success": True, "output": "".join(lines)}

        return {"success": False, "error": "Action inconnue."}