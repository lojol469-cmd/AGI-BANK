# -*- coding: utf-8 -*-
import os
import json
from typing import Dict, Any

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class UIControllerTool(BaseTool):
    @property
    def name(self) -> str:
        return "control_flutter_interface"

    @property
    def description(self) -> str:
        return (
            "CONTRÔLE L'INTERFACE GRAPHIQUE FLUTTER (CEPP, TABFISCAL, ONGLETS).\n"
            "MANUEL D'EXÉCUTION :\n"
            "- Pour changer d'onglet : action='switch_tab', target_tab='CEPP' (ou 'CAD', 'TABFISCAL', etc.)\n"
            "- Pour mettre à jour le CEPP : action='update_cepp', price=90.0, production=8000000, costs=2500000, rmp_rate=12.5, cost_oil_cap=70.0\n"
            "- Pour afficher un tableau TABFISCAL : action='display_tabfiscal', title='Titre', columns=['A','B'], rows=[['1','2']]\n"
            "- Pour envoyer un message live : action='send_live_event', message='Votre message'"
        )

    @property
    def capabilities(self) -> list:
        return ["switch_tab", "update_cepp", "display_tabfiscal", "live_event"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["switch_tab", "update_cepp", "display_tabfiscal", "send_live_event"],
                    "description": "L'action à effectuer."
                },
                "target_tab": {
                    "type": "string",
                    "enum": ["Terminal", "CEPP", "TABFISCAL", "Fichiers", "Projects", "CAD"],
                    "description": "Onglet cible (si action='switch_tab')."
                },
                "message": {
                    "type": "string",
                    "description": "Message à afficher (si action='send_live_event')."
                },
                "price": {"type": "number", "description": "Prix du baril (si action='update_cepp')."},
                "production": {"type": "number", "description": "Production (si action='update_cepp')."},
                "costs": {"type": "number", "description": "Coûts (si action='update_cepp')."},
                "rmp_rate": {"type": "number", "description": "Redevance (si action='update_cepp')."},
                "cost_oil_cap": {"type": "number", "description": "Plafond Cost Oil (si action='update_cepp')."},
                "title": {"type": "string", "description": "Titre (si action='display_tabfiscal')."},
                "columns": {"type": "array", "items": {"type": "string"}},
                "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
            },
            "required": ["action"]
        }

    def can_handle(self, query: str, context: dict = None) -> float:
        return 0.95

    def execute(self, query: str, context: dict = None) -> dict:
        actions = []
        
        if context:
            action = context.get("action")
            
            if action == "switch_tab":
                tab_map = {"Terminal": 0, "CEPP": 1, "TABFISCAL": 2, "Fichiers": 3, "Projects": 4, "CAD": 5}
                target = context.get("target_tab", "Terminal")
                actions.append(f'<ui_action plugin="system" param="switch_tab" value="{tab_map.get(target, 0)}" />')
                return_msg = f"✅ Onglet {target} activé."
            
            elif action == "send_live_event":
                msg = context.get("message", "Événement")
                safe_msg = msg.replace('"', '&quot;')
                actions.append(f'<ui_action plugin="system" param="live_event" value="{safe_msg}" />')
                return_msg = f"✅ Événement live envoyé : {msg}"

            elif action == "update_cepp":
                if "price" in context: actions.append(f'<ui_action plugin="cepp" param="price" value="{context["price"]}" />')
                if "production" in context: actions.append(f'<ui_action plugin="cepp" param="production" value="{context["production"]}" />')
                if "costs" in context: actions.append(f'<ui_action plugin="cepp" param="costs" value="{context["costs"]}" />')
                if "rmp_rate" in context: actions.append(f'<ui_action plugin="cepp" param="rmp_rate" value="{context["rmp_rate"]}" />')
                if "cost_oil_cap" in context: actions.append(f'<ui_action plugin="cepp" param="cost_oil_cap" value="{context["cost_oil_cap"]}" />')
                actions.append('<ui_action plugin="system" param="switch_tab" value="1" />') # Bascule auto sur CEPP
                return_msg = "✅ Paramètres CEPP mis à jour."

            elif action == "display_tabfiscal":
                title = context.get("title", "Analyse")
                cols = context.get("columns", [])
                rows = context.get("rows", [])
                table_json = json.dumps({"columns": cols, "rows": rows}).replace('"', '\\"')
                actions.append(f'<ui_action plugin="tabfiscal" param="title" value="{title}" />')
                actions.append(f'<ui_action plugin="tabfiscal" param="table" value="{table_json}" />')
                actions.append('<ui_action plugin="system" param="switch_tab" value="2" />') # Bascule auto sur TABFISCAL
                return_msg = "✅ Tableau TABFISCAL mis à jour."

            else:
                return_msg = "❌ Action non reconnue."
        else:
            return_msg = "❌ Aucun contexte fourni."

        return {"success": True, "output": return_msg, "ui_tags": "\n".join(actions)}