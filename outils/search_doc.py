# -*- coding: utf-8 -*-
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List
import os
import sys

# Sécurité pour les imports relatifs (comme dans ton __init__.py)
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from . import BaseTool

class SearchDocTool(BaseTool):
    """
    Outil de recherche dans la base documentaire RAG (FAISS + BGE).
    Interroge le serveur RAG local (port 8765) via l'outil 'report_dgi_article'.
    """
    
    @property
    def name(self) -> str:
        return "search_doc"

    @property
    def description(self) -> str:
        return "Recherche des articles de loi, textes fiscaux et documents comptables (IRPP, IS, TVA, CEPP, etc.) dans la base RAG FAISS. Retourne les extraits de texte avec leurs scores de pertinence."

    @property
    def capabilities(self) -> List[str]:
        return ["rag_search", "document_search", "fiscalite", "droit", "faiss"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "Mots-clés précis à rechercher (ex: 'IRPP', 'TVA', 'IS', 'article 12')."
                }
            },
            "required": ["query"]
        }

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        q = query.lower()
        keywords = ["irpp", "is", "tva", "impot", "impôt", "fiscal", "loi", "article", "code", "dgi", "cepp", "texte", "document"]
        
        # Si un mot-clé fiscal est présent, on priorise fortement cet outil
        if any(kw in q for kw in keywords):
            return 0.95
            
        if "recherche" in q or "trouve" in q or "cherche" in q:
            return 0.7
            
        return 0.1

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        RAG_URL = "http://127.0.0.1:8765/execute_tool"
        TOOL_NAME = "report_dgi_article"
        
        # Nettoyage de la query pour ne garder que les mots-clés essentiels
        query = query.strip().strip('"').strip("'")
        if not query:
            return {"success": False, "error": "La requête est vide."}

        payload = {
            "tool_name": TOOL_NAME,
            "args": {"query": query, "k": 3} # Demande les 3 meilleurs résultats
        }
        
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(RAG_URL, data=data, headers={'Content-Type': 'application/json'})
            
            # TIMEOUT DE 120 SECONDES pour laisser le temps à FAISS/BGE
            with urllib.request.urlopen(req, timeout=120) as response:
                res = json.loads(response.read().decode('utf-8'))
                
                # Le serveur renvoie {"ok": true, "result": "{...json imbriqué...}"}
                if isinstance(res, dict) and "result" in res:
                    inner = res["result"]
                    if isinstance(inner, str):
                        try:
                            inner = json.loads(inner)
                        except:
                            pass
                    
                    if isinstance(inner, dict) and "results" in inner:
                        results = inner["results"]
                        if not results:
                            return {"success": True, "output": "⚠️ Aucun document trouvé pour cette recherche."}
                        
                        # Formatage propre pour le LLM
                        formatted_output = f"🔍 Résultats de recherche pour '{query}' :\n\n"
                        for i, r in enumerate(results, 1):
                            source = r.get("source", "Inconnu")
                            score = r.get("score", 0.0)
                            text = r.get("text", "")
                            formatted_output += f"📄 [Source: {source}] (Score de pertinence: {score:.4f})\n"
                            formatted_output += f"📝 Texte: {text[:1000]}...\n" if len(text) > 1000 else f"📝 Texte: {text}\n"
                            formatted_output += "-" * 50 + "\n\n"
                            
                        return {"success": True, "output": formatted_output}
                
                return {"success": False, "error": f"Format de réponse RAG inattendu : {res}"}
                
        except urllib.error.HTTPError as e:
            return {"success": False, "error": f"Erreur HTTP {e.code} : {e.read().decode('utf-8')}"}
        except Exception as e:
            return {"success": False, "error": f"Erreur de connexion au RAG local : {str(e)}"}