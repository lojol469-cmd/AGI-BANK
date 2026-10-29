# -*- coding: utf-8 -*-
import os
import json
from typing import Dict, Any

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class CadViewerTool(BaseTool):
    @property
    def name(self) -> str:
        return "cad_viewer_tool"

    @property
    def description(self) -> str:
        return (
            "AFFICHE UN PLAN D'ARCHITECTURE (CAD) DANS L'INTERFACE FLUTTER. "
            "Utilise cet outil si l'utilisateur demande de voir un plan, un dessin technique ou le CAD Viewer.\n\n"
            "MANUEL D'EXÉCUTION :\n"
            "- Tu peux charger un fichier JSON existant en fournissant son chemin absolu.\n"
            "- Si l'utilisateur ne fournit pas de fichier, tu peux générer un plan de démonstration (un carré avec une porte) en laissant 'file_path' vide."
        )

    @property
    def capabilities(self) -> list:
        return ["load_cad", "display_json", "cad"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Chemin absolu vers le fichier JSON du plan CAD (laisser vide pour un plan de démonstration)."
                }
            },
            "required": []
        }

    def can_handle(self, query: str, context: dict = None) -> float:
        return 0.85

    def execute(self, query: str, context: dict = None) -> dict:
        actions = []
        file_path = context.get("file_path", "") if context else ""
        
        if file_path and os.path.exists(file_path):
            # On charge le fichier JSON existant
            tag = f'<ui_action plugin="system" param="load_cad" value="{file_path}" />'
            actions.append(tag)
            msg = f"✅ Plan CAD chargé depuis {file_path}"
        else:
            # Génération d'un plan de démonstration (un carré avec une porte)
            demo_data = {
                "walls": [{"type": "polyline", "points": [[0, 0], [10000, 0], [10000, 10000], [0, 10000], [0, 0]]}],
                "rooms": [{"x": 0, "y": 0, "w": 10000, "h": 10000, "name": "Salon"}],
                "doors": [{"x": 5000, "y": 0, "width": 900}]
            }
            # On sauvegarde temporairement pour que Flutter puisse le lire
            temp_path = r"D:\kibali-IA\outputs\demo_cad.json"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(demo_data, f)
            
            tag = f'<ui_action plugin="system" param="load_cad" value="{temp_path}" />'
            actions.append(tag)
            msg = "✅ Plan CAD de démonstration généré et chargé."
            
        # On bascule aussi sur l'onglet CAD (index 5)
        actions.append('<ui_action plugin="system" param="switch_tab" value="5" />')

        return {
            "success": True,
            "output": msg,
            "ui_tags": "\n".join(actions)
        }