# -*- coding: utf-8 -*-
import os
import sys
import json
from typing import Dict, Any, List

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from . import BaseTool, run_in_portable

class UniversalSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "universal_search"

    @property
    def description(self) -> str:
        return """Recherche UNIVERSELLE. Fouille le Web, les Docs RAG et les Outils RAG.
        Renvoie TOUJOURS le texte et des images d'illustration pour le sujet demandé.
        - source='web' : actualités et infos internet (inclut les images).
        - source='rag_docs' : cherche dans les lois/documents locaux.
        - source='rag_tools' : trouve un outil dans le RAG."""

    @property
    def capabilities(self) -> List[str]:
        return ["search", "web", "images", "rag", "docs", "tools"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "La recherche."},
                "source": {
                    "type": "string",
                    "enum": ["web", "rag_docs", "rag_tools"],
                    "description": "La source."
                }
            },
            "required": ["query"]
        }

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        q = query.lower()
        if any(k in q for k in ["cherche", "recherche", "internet", "actualité", "image", "photo", "document", "loi", "rag", "outil"]):
            return 0.95
        return 0.1

    def execute(self, query: str, source: str = "web", **kwargs) -> Dict[str, Any]:
        if source == "web":
            return self._search_web_isolated(query)
        elif source in ["rag_docs", "rag_tools"]:
            return self._search_rag(query, source)
        return {"success": False, "output": f"Source inconnue : {source}"}

    def _search_web_isolated(self, query: str) -> Dict[str, Any]:
        """Cherche le texte ET les images sur le web en une seule passe."""
        
        script_code = f"""
import json
import sys
import time

try:
    from duckduckgo_search import DDGS
except ImportError:
    print(json.dumps({{"success": False, "output": "ERREUR : duckduckgo_search n'est pas installé dans env_portable."}}))
    sys.exit(0)

query = {json.dumps(query)}
max_retries = 3
text_results = []
image_results = []

# Recherche du texte (avec retry)
for attempt in range(max_retries):
    try:
        with DDGS() as ddgs:
            text_results = list(ddgs.text(query, max_results=4))
        break
    except Exception as e:
        if attempt == max_retries - 1:
            pass
        else:
            time.sleep(2)

# Recherche des images (avec retry)
for attempt in range(max_retries):
    try:
        with DDGS() as ddgs:
            image_results = list(ddgs.images(query, max_results=4))
        break
    except Exception as e:
        if attempt == max_retries - 1:
            pass
        else:
            time.sleep(2)

if not text_results and not image_results:
    print(json.dumps({{"success": True, "output": "Aucun résultat trouvé sur internet.", "images": []}}))
    sys.exit(0)

# Formatage du texte pour le LLM
out_text = f"### 🌐 Résultats Web pour '{{query}}'\\n\\n"
if text_results:
    for r in text_results:
        out_text += f"**{{r['title']}}**\\n{{r['body']}}\\n🔗 [Source]({{r['href']}})\\n\\n"
else:
    out_text += "Aucun texte trouvé, mais voici des images."

# Extraction des URLs d'images
img_urls = [r['image'] for r in image_results if 'image' in r]

# On renvoie le JSON structuré
print(json.dumps({{
    "success": True, 
    "output": out_text,
    "images": img_urls
}}))
"""
        return run_in_portable(script_code, timeout=90)

    def _search_rag(self, query: str, mode: str) -> Dict[str, Any]:
        import requests
        RAG_URL = "http://127.0.0.1:8765/execute_tool"
        target_tool = "search_tools" if mode == "rag_tools" else "rag_search"
        args = {"query": query, "top_k": 4} if mode == "rag_tools" else {"query": query, "k": 4}
        
        try:
            resp = requests.post(RAG_URL, json={"tool_name": target_tool, "args": args}, timeout=120)
            resp.raise_for_status()
            res = resp.json()
            
            output = res.get("result", res)
            if isinstance(output, str):
                try: output = json.loads(output)
                except: pass
                
            if isinstance(output, dict) and "results" in output:
                results = output["results"]
                if not results: return {"success": True, "output": "Aucun résultat dans le RAG.", "images": []}
                
                formatted = f"### 📚 Résultats RAG ({mode})\n\n"
                for i, r in enumerate(results, 1):
                    name = r.get("name", r.get("source", "Inconnu"))
                    text = r.get("description", r.get("text", str(r)))
                    formatted += f"**[{i}] {name}**\n{text[:1000]}\n" + "-"*40 + "\n"
                return {"success": True, "output": formatted, "images": []}
            
            return {"success": True, "output": str(output), "images": []}
        except Exception as e:
            return {"success": False, "output": f"Erreur RAG : {str(e)}", "images": []}