# -*- coding: utf-8 -*-
import datetime

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class DateTimeTool(BaseTool):
    @property
    def name(self) -> str:
        return "datetime_tool"

    @property
    def description(self) -> str:
        return "Donne la date, l'heure exacte et l'époque actuelle. Utilise cet outil pour savoir quel jour on est."

    @property
    def capabilities(self) -> list:
        return ["date", "heure", "temps", "aujourd'hui", "maintenant", "époque", "année", "jour", "mois"]

    def can_handle(self, query: str, context: dict = None) -> float:
        query_lower = query.lower()
        score = 0.0
        keywords = ["date", "heure", "temps", "aujourd'hui", "maintenant", "quelle époque", "quel jour", "quelle année", "quand"]
        for kw in keywords:
            if kw in query_lower:
                score += 0.35
        return min(score, 1.0)

    def execute(self, query: str, context: dict = None) -> dict:
        now = datetime.datetime.now()
        
        # Traduction manuelle en français pour éviter les problèmes de locale Windows
        jours_fr = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        mois_fr = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
        
        jour_semaine = jours_fr[now.weekday()]
        mois = mois_fr[now.month - 1]
        
        formatted = f"📅 **Date :** {jour_semaine} {now.day} {mois} {now.year}\n"
        formatted += f"🕒 **Heure :** {now.strftime('%H:%M:%S')}\n"
        formatted += f"🌍 **Fuseau horaire :** {now.astimezone().tzname()}\n"
        formatted += f"⏱️ **Timestamp Unix :** {int(now.timestamp())}\n\n"
        formatted += f"💡 *Nous sommes actuellement en l'an {now.year}.*"
        
        return {
            "success": True,
            "output": formatted
        }