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
            "Recherche dans la base documentaire massive (6834 documents) via le serveur RAG FAISS. "
            "⚠️ RÈGLES DE RECHERCHE STRICTES :\n"
            "1. Ne cherche JAMAIS avec des termes vagues (ex: 'TVA Gabon' renvoie du hors-sujet).\n"
            "2. Cherche TOUJOURS avec des numéros d'articles ou des termes juridiques précis "
            "(ex: 'article 221 taux normal 18% TVA', 'CEPP réglementation hydrocarbures').\n"
            "3. Si tu ne trouves pas la réponse dans les extraits, reformule ta requête avec des synonymes ou des termes plus techniques."
        )

    @property
    def capabilities(self) -> list:
        return ["documents", "recherche", "rag", "faiss", "investigation", "lois", "fiscalite"]

    def can_handle(self, query: str, context: dict = None) -> float:
        if any(kw in query.lower() for kw in ["document", "pdf", "recherche", "fichier", "tva", "fiscalité", "loi", "cepp", "pétrole"]):
            return 0.95
        return 0.0

    def execute(self, query: str, context: dict = None) -> dict:
        # 🛡️ GARDE-FOU CONTRE LA PARESSE
        cleaned_query = query.strip().lower()
        stop_words = ["oui", "non", "ok", "d'accord", "yes", "no", "vas-y", "cherche", "recherche"]
        if len(cleaned_query) < 5 or cleaned_query in stop_words:
            return {
                "success": False, 
                "error": "❌ REQUÊTE INVALIDE. Tu as envoyé un mot vide. Regarde l'historique et utilise le VRAI SUJET (ex: 'CEPP réglementation hydrocarbures'). Réessaie immédiatement."
            }

        try:
            # On demande à FAISS de fouiller les 4 meilleurs extraits
            payload = {"query": query, "top_k": 4} 
            res = requests.post("http://127.0.0.1:8765/search_docs", json=payload, timeout=15)
            
            if res.status_code != 200:
                return {"success": False, "error": f"Erreur RAG ({res.status_code}): {res.text}"}
            
            data = res.json()
            
            # Extraction des morceaux (selon le format du serveur : "docs", "documents" ou "results")
            chunks = data.get("docs", data.get("documents", data.get("results", [])))
            if not chunks and isinstance(data, list):
                chunks = data

            if not chunks:
                return {"success": True, "output": "❌ Aucun document trouvé. Change tes mots-clés (ex: utilise des termes juridiques ou des numéros d'articles)."}

            # Formatage de la réponse pour l'agent
            response_text = f"🔍 **Résultats de recherche FAISS pour : '{query}'**\n\n"
            
            count = 0
            for chunk in chunks:
                if isinstance(chunk, dict):
                    text = chunk.get("page_content", chunk.get("text", chunk.get("content", str(chunk))))
                else:
                    text = str(chunk)
                
                # On tronque chaque extrait à 1000 caractères pour ne pas exploser la mémoire du LLM
                if len(text) > 1000:
                    text = text[:1000] + "..."
                    
                response_text += f"📄 **Extrait {count+1}** :\n{text}\n\n"
                count += 1
                
                # On s'arrête à 3 extraits (env. 3000 caractères = 750 tokens)
                if count >= 3:
                    break

            response_text += "---\n"
            response_text += "🧠 **ORDRE SYSTÈME (AGENT AUTONOME)** : Si la réponse exacte n'est pas dans ces extraits, "
            response_text += "TU NE DOIS PAS abandonner. Tu dois IMMÉDIATEMENT relancer `server_doc_search` avec une requête plus précise (ex: si tu as cherché 'CEPP', cherche 'CEPP article loi hydrocarbures')."

            return {"success": True, "output": response_text}
            
        except Exception as e:
            return {"success": False, "error": f"Erreur : {str(e)}"}