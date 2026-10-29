import sys
sys.path.insert(0, r"D:\kibali-IA")

# Charger l app pour avoir les helpers
import app
from outils import ToolManager

tm = ToolManager()
q = "fais une recherche web sur le ballon d or de cette année"
ctx = {"has_pdfs": False, "vectordb_available": False, "web_enabled": True, "query": q}

BLACKLIST = {"control_flutter_interface","server_memory","kibali_server_tools",
             "ultra_system_cleaner","ultra_lib_installer","gui_launcher_tool",
             "excel_com_tool","excel_organizer","cad_viewer_tool",
             "deep_research_agent","ultra_report_master"}
RAG_TOOLS = {"ultra_premium_rag","ultra_folder_rag","pdf_document_search",
             "rag_connector_tool","server_doc_search","ultra_aspirator"}

ordered, scores = app.select_tools_hybrid(
    q, tm, ctx,
    wants_web=True, wants_rag=False,
    max_tools=5, blacklist=BLACKLIST, rag_tools=RAG_TOOLS
)
print("SELECTED:", [(n, round(s, 3)) for n, s in ordered])
print("TOP 8 SCORES:")
for n, s in sorted(scores.items(), key=lambda kv: -kv[1])[:8]:
    print("   %.3f  %s" % (s, n))

if ordered and ordered[0][0] == 'web_search':
    r = tm.tools['web_search'].execute(q, ctx)
    ok = isinstance(r, dict) and r.get('success')
    print("web_search ->", "OK" if ok else "FAIL")
    if ok:
        print("LEN:", len(str(r.get('output', ''))))
        print("PREVIEW:", str(r.get('output', ''))[:400])