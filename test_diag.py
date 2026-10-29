import sys
sys.path.insert(0, r"D:\kibali-IA")
from outils import ToolManager

q = "fais une recherche web sur le ballon d or de cette année"
print("QUERY:", q)

tm = ToolManager()
ctx = {"has_pdfs": False, "vectordb_available": False, "web_enabled": True, "query": q}

sel = tm.get_relevant_tools(q, ctx, max_tools=5)
print("SELECTED:", [t.name for t in sel])

if not sel:
    print(">>> PROBLEME: can_handle ne matche pas")
    print(">>> Outils disponibles:", list(tm.tools.keys())[:10])
else:
    for t in sel:
        r = t.execute(q, ctx)
        ok = isinstance(r, dict) and r.get("success")
        print(t.name, "->", "OK" if ok else "FAIL")
        if ok:
            out = str(r.get("output", ""))
            print("   LEN:", len(out))
            print("   PREVIEW:", out[:300])
        else:
            print("   ERR:", r.get("error") if isinstance(r, dict) else r)