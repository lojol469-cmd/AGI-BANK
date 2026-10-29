# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\kibali-IA")

QUERY = "actualités Ballon d'Or 2026"

print(f"🧪 TEST DES OUTILS DE RECHERCHE WEB AVEC LA REQUÊTE : '{QUERY}'\n")

# ==========================================
# TEST 1 : universal_search
# ==========================================
try:
    from outils.universal_search import UniversalSearchTool
    print("="*60)
    print("TEST 1 : universal_search")
    print("="*60)
    tool = UniversalSearchTool()
    res = tool.execute(query=QUERY, source="web")
    
    if res.get("success"):
        print("✅ SUCCÈS")
        print("Texte :", res.get("output", "")[:400] + "...")
        print("Images :", len(res.get("images", [])), "trouvée(s)")
    else:
        print("❌ ÉCHEC :", res.get("output"))
except Exception as e:
    print("❌ ERREUR D'IMPORT :", e)


# ==========================================
# TEST 2 : web_search_pro
# ==========================================
try:
    from outils.web_search_pro import WebSearchProTool
    print("\n" + "="*60)
    print("TEST 2 : web_search_pro")
    print("="*60)
    tool = WebSearchProTool()
    res = tool.execute(query=QUERY, search_images=True)
    
    if res.get("success"):
        print("✅ SUCCÈS")
        print("Texte :", res.get("output", "")[:400] + "...")
    else:
        print("❌ ÉCHEC :", res.get("output"))
except Exception as e:
    print("❌ ERREUR D'IMPORT :", e)


# ==========================================
# TEST 3 : web_search (L'original)
# ==========================================
try:
    from outils.web_search_tool import WebSearchTool
    print("\n" + "="*60)
    print("TEST 3 : web_search (original)")
    print("="*60)
    tool = WebSearchTool()
    res = tool.execute(query=QUERY)
    
    if res.get("success"):
        print("✅ SUCCÈS")
        print("Texte :", res.get("output", "")[:400] + "...")
    else:
        print("❌ ÉCHEC :", res.get("output"))
except Exception as e:
    print("❌ ERREUR D'IMPORT :", e)

print("\n" + "="*60)
print("FIN DES TESTS")