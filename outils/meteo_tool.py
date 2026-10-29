# -*- coding: utf-8 -*-
from . import BaseTool
import requests

class MeteoTool(BaseTool):
    @property
    def name(self) -> str:
        return "meteo_tool"

    @property
    def description(self) -> str:
        return "Donne la météo actuelle pour une ville (ex: Paris, Libreville)."

    @property
    def capabilities(self) -> list:
        return ["weather", "meteo", "temperature"]

    def can_handle(self, query: str, context: dict = None) -> float:
        q = query.lower()
        if any(kw in q for kw in ["météo", "meteo", "temps qu'il fait", "température"]):
            return 0.95
        return 0.0

    def execute(self, query: str, context: dict = None) -> dict:
        try:
            # Remplacer par une vraie API
            return {"success": True, "output": f"🌤️ Il fait beau à {query}."}
        except Exception as e:
            return {"success": False, "error": str(e)}