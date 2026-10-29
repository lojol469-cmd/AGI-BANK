import requests
import json

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class ServerMemory(BaseTool):
    @property
    def name(self) -> str:
        return "server_memory"

    @property
    def description(self) -> str:
        return "Accède à la mémoire épisodique du serveur (souvenirs des conversations passées)."

    @property
    def capabilities(self) -> list:
        return ["mémoire", "souvenirs", "contexte", "historique"]

    def can_handle(self, query: str, context: dict = None) -> float:
        keywords = ["souvenir", "mémoire", "rappelle", "hier", "précédent", "passé", "conversation"]
        if any(kw in query.lower() for kw in keywords):
            return 0.9
        return 0.0

    def execute(self, query: str, context: dict = None) -> dict:
        try:
            # Recherche dans la mémoire
            res = requests.post("http://127.0.0.1:8765/memory/search", json={"query": query, "top_k": 5, "threshold": 0.3}, timeout=15)
            if res.status_code == 200:
                return {"success": True, "output": json.dumps(res.json(), indent=2, ensure_ascii=False)}
            return {"success": False, "error": f"Erreur Memory RAG: {res.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}