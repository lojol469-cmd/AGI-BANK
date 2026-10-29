# -*- coding: utf-8 -*-
import os
import sys
import json
import requests
from typing import Dict, Any, List

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from . import BaseTool

class UltraPremiumRAG(BaseTool):
    @property
    def name(self) -> str:
        return "ultra_premium_rag"

    @property
    def description(self) -> str:
        return (
            "🔒 OUTIL PREMIUM (Réservé à Kibali Ultra).\n"
            "Recherche intelligente dans les documents (Code Général des Impôts, Lois).\n"
            "Nettoie automatiquement votre question et tente une 2ème recherche de secours si nécessaire."
        )

    @property
    def capabilities(self) -> List[str]:
        return ["documents", "recherche", "rag", "premium", "fiscalite", "lois"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "La recherche fiscale ou juridique."}
            },
            "required": ["query"]
        }

    def can_handle(self, query: str, context: dict = None) -> float:
        if any(kw in query.lower() for kw in ["document", "pdf", "recherche", "tva", "fiscalité", "cepp", "loi", "impôt", "exonération", "calcul"]):
            return 0.99
        return 0.0

    def _call_rag(self, query):
        try:
            payload = {"query": query, "top_k": 2}
            res = requests.post("http://127.0.0.1:8765/search_docs", json=payload, timeout=15)
            if res.status_code == 200:
                return res.text
            return ""
        except:
            return ""

    def execute(self, query: str, context: dict = None) -> dict:
        print(f"💎 [Premium RAG] Requête reçue : {query}")

        # 1. NETTOYAGE PREMIUM
        clean_query = query.replace("cherche", "").replace("trouve", "").replace("documents", "").replace("quelle est", "").replace("?", "").strip()
        
        # 2. PREMIÈRE RECHERCHE
        text_result = self._call_rag(clean_query)
        
        # 3. RECHERCHE DE SECOURS
        if not text_result or len(text_result) < 50 or "Aucun document" in text_result:
            print("   ⚠️ Première recherche infructueuse. Lancement de la recherche de secours...")
            fallback_query = clean_query + " article 221 loi de finances CGI"
            text_result = self._call_rag(fallback_query)

        if not text_result or len(text_result) < 50:
            return {"success": True, "output": "❌ Aucun document trouvé, même avec la recherche de secours."}

        # 4. BOUCLIER ANTI-EXPLOSION
        if len(text_result) > 2500:
            text_result = text_result[:2500] + "\n\n...[Texte tronqué par la sécurité Premium]..."

        return {"success": True, "output": text_result}