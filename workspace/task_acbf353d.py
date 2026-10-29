
import json
try:
    from duckduckgo_search import DDGS
except ImportError:
    print(json.dumps({"success": False, "output": "La librairie duckduckgo_search n'est pas installée dans env_portable."}))
    exit(0)

query = "chat mignon"
mode = "images"

try:
    with DDGS() as ddgs:
        if mode == "images":
            results = list(ddgs.images(query, max_results=4))
            if not results:
                print(json.dumps({"success": True, "output": "Aucune image trouvée."}))
            else:
                out = "### 🖼️ Images trouvées\n\n"
                for r in results:
                    out += f"![{r['title']}]({r['image']})\n\n"
                print(json.dumps({"success": True, "output": out}))
        else:
            results = list(ddgs.text(query, max_results=4))
            if not results:
                print(json.dumps({"success": True, "output": "Aucun résultat web trouvé."}))
            else:
                out = f"### 🌐 Résultats Web pour '{query}'\n\n"
                for r in results:
                    out += f"**{r['title']}**\n{r['body']}\n🔗 [Source]({r['href']})\n\n"
                print(json.dumps({"success": True, "output": out}))
except Exception as e:
    print(json.dumps({"success": False, "output": f"Erreur Web : {str(e)}"}))
