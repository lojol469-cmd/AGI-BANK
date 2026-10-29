
import requests
import json

RAG_BASE = "http://127.0.0.1:8765"
RAG_EXEC = f"{RAG_BASE}/execute_tool"

def rag_tools():
    try:
        r = requests.get(f"{RAG_BASE}/tool_schemas", timeout=10)
        return r.json().get("tools", [])
    except Exception as e:
        print(f"[SDK Erreur] rag_tools: {e}")
        return []

def rag_run(tool_name, **kwargs):
    try:
        payload = {"tool_name": tool_name, "args": kwargs}
        r = requests.post(RAG_EXEC, json=payload, timeout=120)
        return r.json()
    except Exception as e:
        print(f"[SDK Erreur] rag_run: {e}")
        return {"error": str(e)}

# --- CODE DE L'AGI ---

results = []
results.append(rag_run('full_security_audit'))
results.append(rag_run('security_list_users'))
results.append(rag_run('security_active_sessions'))
results.append(rag_run('security_stats'))
results.append(rag_run('plot_security_dashboard'))
print(results)