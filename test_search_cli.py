# -*- coding: utf-8 -*-
import sys
import json

sys.path.insert(0, r"D:\kibali-IA")
from outils.universal_search import UniversalSearchTool

def run_test():
    tool = UniversalSearchTool()
    print("🧪 Test CLI de UniversalSearchTool\n")

    print("--- TEST 1 : Recherche Web (Texte) ---")
    res_web = tool.execute(query="dernières actualités intelligence artificielle", source="web")
    if res_web.get("success"):
        print(res_web["output"][:800] + "\n[...]\n")
    else:
        print(f"❌ ERREUR : {res_web.get('output')}")

    print("\n--- TEST 2 : Recherche d'Images ---")
    res_img = tool.execute(query="chat mignon", source="images")
    if res_img.get("success"):
        print(res_img["output"][:800] + "\n[...]\n")
    else:
        print(f"❌ ERREUR : {res_img.get('output')}")

if __name__ == "__main__":
    run_test()