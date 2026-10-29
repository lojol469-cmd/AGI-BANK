import sys
sys.path.insert(0, r"D:\kibali-IA")
import app
from outils import ToolManager

tm = ToolManager()
q = "fais une recherche web sur le ballon d or de cette année"
ctx = {"has_pdfs": False, "vectordb_available": False, "web_enabled": True, "query": q}

BL = {"control_flutter_interface","server_memory","kibali_server_tools",
      "ultra_system_cleaner","ultra_lib_installer","gui_launcher_tool",
      "excel_com_tool","excel_organizer","cad_viewer_tool",
      "deep_research_agent","ultra_report_master"}
RAG = {"ultra_premium_rag","ultra_folder_rag","pdf_document_search",
       "rag_connector_tool","server_doc_search","ultra_aspirator"}

ordered, scores = app.select_tools_hybrid(
    q, tm, ctx, wants_web=True, wants_rag=False,
    max_tools=5, blacklist=BL, rag_tools=RAG
)
print("SANS PDF - SELECTED:", [(n, round(s,3)) for n, s in ordered])
assert ordered and ordered[0][0] == "web_search", "web_search doit etre 1er"
assert len(ordered) == 1, "seul web_search doit rester"
print("ASSERT OK - aucun RAG present")

# Validation
if ordered:
    r = tm.tools['web_search'].execute(q, ctx)
    out = str(r.get('output', ''))
    ok, reason = app._validate_tool_output('web_search', out, q)
    print("VALIDATION web_search:", ok, reason)