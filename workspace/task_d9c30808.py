CURRENT_DATE = "2026-10-28"

import json
import sys
import time

try:
    from duckduckgo_search import DDGS
except ImportError:
    print(json.dumps({"success": False, "output": "ERREUR : duckduckgo_search n'est pas installé dans env_portable."}))
    sys.exit(0)

query = "Ballon d'Or 2026"
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
    print(json.dumps({"success": True, "output": "Aucun résultat trouvé sur internet.", "images": []}))
    sys.exit(0)

# Formatage du texte pour le LLM
out_text = f"### 🌐 Résultats Web pour '{query}'\n\n"
if text_results:
    for r in text_results:
        out_text += f"**{r['title']}**\n{r['body']}\n🔗 [Source]({r['href']})\n\n"
else:
    out_text += "Aucun texte trouvé, mais voici des images."

# Extraction des URLs d'images
img_urls = [r['image'] for r in image_results if 'image' in r]

# On renvoie le JSON structuré
print(json.dumps({
    "success": True, 
    "output": out_text,
    "images": img_urls
}))
