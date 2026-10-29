import requests
import json

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class ServerDocSearch(BaseTool):
    @property
    def name(self) -> str:
        return "server_doc_search"

    @property
    def description(self) -> str:
        return (
            "Recherche dans la base documentaire massive (Code Général des Impôts, Lois de Finances).\n"
            "⚠️ IMPORTANT : TU DOIS RÉPONDRE AVEC UN OBJET JSON STRICT.\n"
            "Format attendu : {\"query\": \"mots-clés précis\", \"top_k\": 3}\n\n"
            "EXEMPLES DE BONNES REQUÊTES JSON :\n"
            "1. {\"query\": \"taux TVA première nécessité\", \"top_k\": 3}\n"
            "2. {\"query\": \"article 221 exonération TVA\", \"top_k\": 3}\n\n"
            "💡 INSTRUCTION POUR LES CALCULS :\n"
            "Si l'utilisateur demande de CALCULER quelque chose (ex: 'Calcule la TVA pour 500 000 FCFA') :\n"
            "1. Cherche d'abord le taux avec cet outil.\n"
            "2. Dès que tu as lu le taux (ex: 5%), TU DOIS OBLIGATOIREMENT appeler l'outil `python_executor` pour faire le calcul (ex: `print(500000 * 0.05)`).\n"
            "Ne réponds JAMAIS avec le calcul à la main, utilise le code Python."
        )

    @property
    def capabilities(self) -> list:
        return ["documents", "recherche", "rag", "lois", "fiscalite"]

    def can_handle(self, query: str, context: dict = None) -> float:
        if any(kw in query.lower() for kw in ["document", "pdf", "recherche", "tva", "fiscalité", "cepp", "loi", "impôt", "exonération"]):
            return 0.98
        return 0.0

    def execute(self, query: str, context: dict = None) -> dict:
        # Nettoyage des balises markdown
        clean_input = query.replace("```json", "").replace("```", "").strip()
        
        print(f"📥 [RAG] L'IA a envoyé : {clean_input}")

        try:
            # On essaie de lire ce que l'IA a écrit comme du JSON
            payload = json.loads(clean_input)
            if "top_k" not in payload:
                payload["top_k"] = 3
        except json.JSONDecodeError:
            # Fallback si l'IA n'a pas écrit du JSON valide
            payload = {"query": query, "top_k": 3}

        # APPEL AU SERVEUR RAG (Port 8765)
        try:
            res = requests.post("http://127.0.0.1:8765/search_docs", json=payload, timeout=15)
            
            if res.status_code != 200:
                return {"success": False, "error": f"Erreur serveur RAG ({res.status_code}): {res.text}"}
            
            text_result = res.text
            
            if not text_result or text_result.strip() == "" or "Aucun document" in text_result:
                return {"success": True, "output": "❌ Aucun document trouvé. L'IA doit reformuler son JSON avec d'autres mots-clés."}

            return {"success": True, "output": text_result}

        except Exception as e:
            return {"success": False, "error": f"Erreur de communication : {str(e)}"}