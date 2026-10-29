# -*- coding: utf-8 -*-
import os
import sys
import json
from typing import Dict, Any, List

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from . import BaseTool, run_in_portable

class TranslationTool(BaseTool):
    @property
    def name(self) -> str:
        return "translation_tool"

    @property
    def description(self) -> str:
        return "Traduit un texte d'une langue à une autre. Ex: traduis 'bonjour' en anglais."

    @property
    def capabilities(self) -> List[str]:
        return ["traduction", "langue", "texte"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Le texte à traduire."},
                "target_lang": {"type": "string", "description": "Code de la langue cible (ex: 'en' pour anglais, 'fr' pour français)."}
            },
            "required": ["text", "target_lang"]
        }

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        q = query.lower()
        if any(k in q for k in ["traduis", "traduire", "translate", "traduction"]):
            return 0.95
        return 0.0

    def execute(self, text: str = "", target_lang: str = "en", **kwargs) -> Dict[str, Any]:
        if not text:
            return {"success": False, "output": "❌ Aucun texte à traduire fourni."}

        script_code = f"""
import json
import sys

try:
    from argostranslate import translate
except ImportError:
    print(json.dumps({{"success": False, "output": "argostranslate n'est pas installé dans env_portable."}}))
    sys.exit(0)

text = {json.dumps(text)}
target_lang = {json.dumps(target_lang)}

try:
    # Traduction automatique (suppose que les modèles sont installés)
    translated_text = translate.translate(text, "auto", target_lang)
    print(json.dumps({{"success": True, "output": f"Traduction : {{translated_text}}"}}))
except Exception as e:
    print(json.dumps({{"success": False, "output": f"Erreur de traduction (modèle manquant ?) : {{str(e)}}"}}))
"""
        return run_in_portable(script_code)