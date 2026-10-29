import requests
import json
import re
import time

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class DeepResearchAgent(BaseTool):
    @property
    def name(self) -> str:
        return "deep_research_agent"

    @property
    def description(self) -> str:
        return (
            "Un sous-agent de recherche autonome. Envoie-lui un SUJET (ex: 'TVA Gabon exonération'). "
            "Il va fouiller dans les 6834 documents, relancer des recherches si nécessaire, "
            "et te renvoyer un RAPPORT COMPLET et PROPRE. Tu n'as pas besoin de lire les documents toi-même."
        )

    @property
    def capabilities(self) -> list:
        return ["recherche_profonde", "rag", "documents", "investigation"]

    def can_handle(self, query: str, context: dict = None) -> float:
        # Il gère tout ce qui touche à la recherche documentaire
        if any(kw in query.lower() for kw in ["document", "pdf", "recherche", "tva", "fiscalité", "cepp", "loi"]):
            return 0.98
        return 0.0

    def _call_rag(self, query):
        """Appelle le serveur RAG sur le port 8765"""
        try:
            payload = {"query": query, "top_k": 3}
            res = requests.post("http://127.0.0.1:8765/search_docs", json=payload, timeout=15)
            if res.status_code == 200:
                data = res.json()
                # Extraction propre du texte
                docs = data.get("docs", data.get("documents", data.get("results", [])))
                if not docs and isinstance(data, list): docs = data
                
                text = ""
                for d in docs:
                    if isinstance(d, dict):
                        text += str(d.get("page_content", d.get("text", d.get("content", str(d))))) + "\n\n"
                    else:
                        text += str(d) + "\n\n"
                return text.strip()
            return ""
        except Exception as e:
            return f"Erreur RAG: {e}"

    def _call_llm(self, prompt):
        """Appelle llama-server pour faire réfléchir le sous-agent"""
        try:
            res = requests.post(
                "http://127.0.0.1:8080/v1/chat/completions",
                json={
                    "model": "qwen3-8b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                timeout=60
            )
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            return f"Erreur LLM ({res.status_code})"
        except Exception as e:
            return f"Erreur LLM: {e}"

    def execute(self, query: str, context: dict = None) -> dict:
        print(f"🕵️‍♂️ [Sub-Agent] Début de la recherche pour : {query}")
        
        current_search = query
        collected_docs = []
        
        # Boucle de recherche autonome (max 2 itérations pour ne pas être trop long)
        for iteration in range(2):
            print(f"🔎 [Sub-Agent] Recherche {iteration+1} : '{current_search}'")
            docs = self._call_rag(current_search)
            
            if docs:
                collected_docs.append(docs[:2000]) # On garde un max de 2000 chars par recherche
            else:
                print(f"⚠️ [Sub-Agent] Aucun document trouvé pour '{current_search}'")

            # Demander au LLM s'il a assez d'infos ou s'il doit relancer une recherche
            prompt = f"""Tu es un chercheur documentaire. L'utilisateur veut savoir : "{query}"
Voici ce que tu as trouvé pour l'instant :
{' '.join(collected_docs)}

As-tu assez d'informations pour répondre de manière précise et complète à la question ?
- Si OUI, rédige un rapport synthétique et précis. Commence par "RAPPORT FINAL:".
- Si NON, dis exactement ce qu'il te manque et donne les mots-clés d'une NOUVELLE recherche. Commence par "NOUVELLE RECHERCHE:"."""

            response = self._call_llm(prompt)
            
            if "RAPPORT FINAL:" in response:
                print("✅ [Sub-Agent] Rapport final généré !")
                return {"success": True, "output": response.split("RAPPORT FINAL:")[1].strip()}
            
            elif "NOUVELLE RECHERCHE:" in response:
                new_query = response.split("NOUVELLE RECHERCHE:")[1].strip()
                print(f"🔄 [Sub-Agent] Le chercheur n'est pas satisfait. Nouvelle requête : {new_query}")
                current_search = new_query
            else:
                # Si le LLM ne respecte pas le format, on s'arrête
                return {"success": True, "output": response}

        # Si après 2 boucles on n'a pas de rapport final, on synthétise de force
        print("⚠️ [Sub-Agent] Limite de recherches atteinte. Synthèse finale forcée.")
        final_prompt = f"Fais un rapport final clair et complet avec ce que tu as trouvé pour répondre à : {query}\n\nCONTEXTE:\n{' '.join(collected_docs)}"
        final_report = self._call_llm(final_prompt)
        return {"success": True, "output": final_report}