CURRENT_DATE = "2026-10-28"

import json
import sys

try:
    from argostranslate import translate
except ImportError:
    print(json.dumps({"success": False, "output": "argostranslate n'est pas installé dans env_portable."}))
    sys.exit(0)

text = "L'intelligence artificielle va changer le monde"
target_lang = "en"

try:
    # Traduction automatique (suppose que les modèles sont installés)
    translated_text = translate.translate(text, "auto", target_lang)
    print(json.dumps({"success": True, "output": f"Traduction : {translated_text}"}))
except Exception as e:
    print(json.dumps({"success": False, "output": f"Erreur de traduction (modèle manquant ?) : {str(e)}"}))
