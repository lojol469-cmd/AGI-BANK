# -*- coding: utf-8 -*-
import sys
import time
import random
import re
import requests
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

try:
    from ddgs import DDGS
    DDGS_OK = True
except ImportError:
    DDGS_OK = False

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
]

class WebSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return ("Recherche des informations sur internet sur PLUSIEURS sources à la fois. "
                "Idéal pour les actualités, les résultats sportifs, ou les faits récents.")

    @property
    def capabilities(self) -> list:
        return ["web_search", "internet", "scraping", "news"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La requête de recherche (ex: 'Ballon d'or 2026', 'Actualité Barcelone')"
                }
            },
            "required": ["query"]
        }

    def can_handle(self, query: str, context: dict = None) -> float:
        q = query.lower()
        if any(kw in q for kw in ["internet", "cherche", "actualité", "ballon d'or", "news", "web", "météo", "qui a gagné", "résultat"]):
            return 0.95
        return 0.0

    def _scrape_url(self, url: str, max_chars: int = 1200) -> str:
        if not BS4_OK: return ""
        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code != 200: return ""
            
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
                
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text[:max_chars] + "..." if len(text) > max_chars else text
        except Exception:
            return ""

    def execute(self, query: str, context: dict = None) -> dict:
        print(f"[web_search] Recherche Multi-Sources : {query}")
        sys.stdout.flush()

        links = []
        if DDGS_OK:
            try:
                with DDGS() as ddgs:
                    # 1. On cherche D'ABORD dans les NEWS (Presse, Actualités)
                    news_results = list(ddgs.news(query, max_results=3))
                    for r in news_results:
                        links.append({
                            "title": r.get("title", "").strip(),
                            "url": r.get("url", "").strip(),
                            "snippet": r.get("body", "").strip()
                        })
                    
                    # 2. On complète avec une recherche WEB classique (Blogs, Wikipedia...)
                    text_results = list(ddgs.text(query, max_results=3))
                    for r in text_results:
                        url = r.get("href", "").strip()
                        # On évite de remettre un lien Wikipedia si on en a déjà un
                        if not any(l["url"] == url for l in links):
                            links.append({
                                "title": r.get("title", "").strip(),
                                "url": url,
                                "snippet": r.get("body", "").strip()
                            })
            except Exception as e:
                print(f"[web_search] DDGS echec: {e}")

        if not links:
            return {"success": False, "output": "Impossible de trouver des sources sur internet pour le moment."}

        # On garde les 5 meilleurs liens
        links = links[:5]
        print(f"   -> Scraping de {len(links)} sources variées en parallèle...")
        
        def process_link(link):
            content = self._scrape_url(link["url"])
            if not content:
                content = link["snippet"] + "..." if link["snippet"] else "[Contenu bloqué par le site]"
            
            return f"### Source : {link['title']}\nURL: {link['url']}\nExtrait: {content}\n"

        with ThreadPoolExecutor(max_workers=4) as executor:
            scraped_results = list(executor.map(process_link, links))

        final_output = f"Voici les résultats de recherche multi-sources pour '{query}' :\n\n"
        final_output += "\n---\n".join(scraped_results)
        
        return {"success": True, "output": final_output}