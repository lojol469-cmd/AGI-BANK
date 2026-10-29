# -*- coding: utf-8 -*-
"""
KIBALI ENGINE BRIDGE — Backend du frontend Flutter.
Gère le Mode Normal et le Mode Ultra.
"""
import sys
import os
import io
import re
import json
import time
import uuid
import asyncio
import traceback
from pathlib import Path
from datetime import datetime

# UTF-8 forcé
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_classic.agents import create_react_agent, AgentExecutor, Tool
except ImportError as e:
    print(f"[FATAL] Dépendances LangChain manquantes: {e}", flush=True)
    sys.exit(1)

try:
    from outils import ToolManager
except ImportError:
    print("[FATAL] Impossible d'importer ToolManager depuis outils.py", flush=True)
    sys.exit(1)


# ==========================================
# 0. 📁 CHEMINS CANONIQUES
# ==========================================
KIBALI_ROOT      = Path(r"D:\kibali-IA")
CPT_ROOT         = Path(r"D:\CPT_Analysis_Studio_PORTABLE")

APPS_DIR         = KIBALI_ROOT / "generated_apps"
OUTPUTS_DIR_K    = KIBALI_ROOT / "outputs"
OUTPUTS_DIR_CPT  = CPT_ROOT / "outputs"
UPLOADS_DIR      = CPT_ROOT / "images"
PROJECTS_DIR     = CPT_ROOT / "projects"
LOGS_DIR         = KIBALI_ROOT / "logs"
MEMORY_FILE      = KIBALI_ROOT / "kibali_rag_memory.json"
TOOLS_JSON_FILE  = KIBALI_ROOT / "outils" / "kibali_tools.json"


def ensure_dirs():
    for d in (APPS_DIR, OUTPUTS_DIR_K, OUTPUTS_DIR_CPT, UPLOADS_DIR, PROJECTS_DIR, LOGS_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[WARN] mkdir {d}: {e}", flush=True)


# ==========================================
# 1. 📡 ÉMETTEURS DE MARQUEURS
# ==========================================
def emit_tool_start(name, input_str):
    print(f"⚙️ [Outil] {name}", flush=True)
    preview = str(input_str).replace('\n', ' ')[:180]
    if preview: print(f"   → {preview}", flush=True)

def emit_tool_result(name, output, success=True):
    icon = "✅" if success else "❌"
    full = str(output)
    preview = full.replace('\n', ' ')[:280]
    print(f"   {icon} [{len(full)} chars] {preview}", flush=True)

def emit_final(text):
    print("💡 [Rapport]", flush=True)
    for line in str(text).split('\n'): print(line, flush=True)

def emit_log(text): print(str(text), flush=True)


# ==========================================
# 2. 🗄️ MÉMOIRE RAG (Conservée)
# ==========================================
class RagMemory:
    def __init__(self, path=MEMORY_FILE):
        self.path = Path(path)
        self.entries = []
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.entries = data.get('entries', [])
            except Exception: pass
            
    def add_exchange(self, user, assistant, tool_calls, mode="default"):
        if not assistant or not assistant.strip(): return None
        entry = {"id": f"ex_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}", "user": user, "assistant": assistant.strip(), "tool_calls": tool_calls}
        self.entries.append(entry)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"version": 2, "entries": self.entries[-100:]}, f, ensure_ascii=False, indent=2)
        except Exception: pass
        return entry["id"]


# ==========================================
# 3. 🎛️ MODE MANAGER (Normal vs Ultra)
# ==========================================
class ModeManager:
    def __init__(self):
        self.current = "normal"

    def set(self, m):
        m = (m or '').strip().lower()
        if m in ("normal", "default", ""):
            self.current = "normal"
            return True
        if m in ("ultra", "tbfiscal", "fiscal"):
            self.current = "ultra"
            return True
        return False

# ==========================================
# 4. 🔍 REGISTRE DYNAMIQUE
# ==========================================
CORE_TOOL_NAMES = {"datetime_tool", "web_search", "python_exec", "read_file", "write_file", "list_directory", "snapshot"}

# Outils exposés uniquement en mode ULTRA
ULTRA_EXTRA_TOOLS = {
    "search_doc", "report_dgi_article", "report_dgi_tva", "report_dgi_is",
    "table_tva_gen", "table_is_gen", "table_amortissement_gen", "python_verify", 
    "report_excel_audit", "rag_search", "fiscal_search"
}

def _stringify(r):
    if isinstance(r, dict):
        if r.get('success'): return str(r.get('output', r.get('content', '')))
        return f"ECHEC: {r.get('error')}"
    return str(r)

class ToolRegistry:
    def __init__(self, rag_memory):
        emit_log("🔍 Chargement du ToolManager...")
        self.tm = ToolManager()
        self.rag_memory = rag_memory
        self._lc_cache = {}
        emit_log(f"✅ {len(self.tm.tools)} outils locaux chargés.")

    def _build_lc_tool(self, name):
        if name in self._lc_cache: return self._lc_cache[name]
        obj = self.tm.tools.get(name)
        if obj is None or isinstance(obj, Exception): return None
        try:
            desc = getattr(obj, 'description', f"Outil {name}") or f"Outil {name}"
            def _run(query, _n=name, _o=obj): 
                ctx = {'has_pdfs': True, 'vectordb_available': True}
                return _stringify(_o.execute(query, ctx))
            t = Tool(name=name, func=_run, description=desc)
            self._lc_cache[name] = t
            return t
        except Exception:
            return None

    def get_exposed_tools(self, mode="normal"):
        exposed = []
        # 1. Outils de base (toujours exposés)
        for n in CORE_TOOL_NAMES:
            t = self._build_lc_tool(n)
            if t: exposed.append(t)
            
        # 2. search_doc (toujours exposé car c'est notre pépite)
        t_search = self._build_lc_tool("search_doc")
        if t_search: exposed.append(t_search)
            
        # 3. Outils Ultra (seulement si mode ultra)
        if mode == "ultra":
            for n in ULTRA_EXTRA_TOOLS:
                if n != "search_doc": # Déjà ajouté
                    t = self._build_lc_tool(n)
                    if t: exposed.append(t)
                    
        return exposed


# ==========================================
# 5. 🤖 AGENT FACTORY
# ==========================================
def create_agent(registry, mode):
    tools = registry.get_exposed_tools(mode=mode)
    emit_log(f"🧰 Agent Mode '{mode.upper()}' — {len(tools)} outils exposés.")

    llm = ChatOpenAI(
        model="qwen3-8b",
        base_url="http://127.0.0.1:8080/v1",
        api_key="local-no-key",
        temperature=0.1,
        max_tokens=3000,
        streaming=True,
    )

    # Le prompt s'adapte au mode
    if mode == "ultra":
        regle_rag = "- Tu es en mode ULTRA. Pour toute question fiscale, utilise `search_doc` ou les outils de calcul DGI."
    else:
        regle_rag = "- Tu es en mode NORMAL. Utilise `search_doc` pour chercher dans la base de lois (IRPP, IS, TVA)."

    template = f"""Tu es Kibali, un agent comptable et fiscaliste en RDC.
Date: {{date}}.

OUTILS DISPONIBLES:
{{tools}}

FORMAT STRICT :
Question: ...
Thought: ...
Action: nom parmi [{{tool_names}}]
Action Input: texte (la recherche)
Observation: ...
... (répète)
Thought: Je connais la réponse
Final Answer: réponse finale avec [source: ...]

RÈGLES CRITIQUES :
{regle_rag}
- L'outil `search_doc` interroge la base FAISS locale et te renvoie les articles de loi avec leur score. Base ta réponse dessus.
- Si aucun résultat n'est trouvé, utilise `web_search`.

Commence !
Question: {{input}}
Thought:{{agent_scratchpad}}"""

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, tools=tools, verbose=False, max_iterations=6,
        handle_parsing_errors="Format invalide. Action: nom_outil, Action Input: texte.",
        early_stopping_method="force", return_intermediate_steps=True,
    )


# ==========================================
# 6. EXÉCUTION ASYNCHRONE (Streaming)
# ==========================================
async def run_turn_async(agent, memory, user_input, date):
    emit_log("─" * 40)
    enriched = f"QUESTION UTILISATEUR:\n{user_input}"

    final_text = ""
    buffer = ""
    showing_final = False
    tool_calls_log = []
    pending = {}

    try:
        async for event in agent.astream_events({"input": enriched, "date": date}, version="v2"):
            kind = event.get("event", "")
            if kind == "on_chat_model_start":
                buffer = ""; showing_final = False
            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk is None: continue
                content = getattr(chunk, "content", "")
                if not content: continue
                if showing_final:
                    final_text += content
                else:
                    buffer += content
                    if "Final Answer:" in buffer:
                        showing_final = True
                        idx = buffer.index("Final Answer:") + len("Final Answer:")
                        rem = buffer[idx:].lstrip()
                        if rem: final_text += rem
                        buffer = ""
            elif kind == "on_tool_start":
                rid = event.get("run_id") or f"_t{time.time()}"
                name = event.get("name", "outil")
                raw = event["data"].get("input", "")
                if isinstance(raw, dict): raw = raw.get("input", str(raw))
                pending[rid] = {"name": name, "input": str(raw), "start": time.time()}
                emit_tool_start(name, raw)
            elif kind == "on_tool_end":
                rid = event.get("run_id") or f"_t{time.time()}"
                out = event["data"].get("output", "")
                info = pending.pop(rid, None)
                if info:
                    tool_calls_log.append({"name": info["name"], "input": info["input"], "output": str(out), "success": True})
                emit_tool_result(info["name"] if info else "outil", str(out), True)
            elif kind == "on_chain_end" and event.get("name") == "AgentExecutor":
                out = event["data"].get("output", {})
                if isinstance(out, dict): 
                    fallback = out.get("output", "")
                    if not final_text and fallback: final_text = fallback

        if final_text.strip():
            eid = memory.add_exchange(user_input, final_text.strip(), tool_calls_log)
            emit_log(f"🗄️ [mémorisé id={eid}]")
            emit_final(final_text.strip())
        else:
            emit_log("⚠️ Aucune réponse finale générée.")
    except Exception as e:
        emit_log(f"❌ Erreur de tour: {e}")
        emit_log(traceback.format_exc())

def run_turn(agent, memory, user_input, date):
    try:
        asyncio.run(run_turn_async(agent, memory, user_input, date))
    except Exception as e:
        emit_log(f"❌ Erreur asynchrone: {e}")


# ==========================================
# 7. BOUCLE PRINCIPALE
# ==========================================
def main():
    ensure_dirs()
    emit_log("=" * 60)
    emit_log("🤖 KIBALI ENGINE BRIDGE v2.0 (Normal / Ultra)")
    emit_log("=" * 60)

    memory = RagMemory(MEMORY_FILE)
    registry = ToolRegistry(memory)
    mode_manager = ModeManager()
    
    current_agent = create_agent(registry, mode=mode_manager.current)
    current_mode = mode_manager.current
    
    emit_log("✅ READY")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.rstrip('\n').rstrip('\r')
        if not line.strip(): continue
        
        if line.lower() in ('quit', 'exit'):
            emit_log("👋 Au revoir!")
            break
            
        # Gestion de la commande /mode
        if line.startswith("/mode"):
            parts = line.split()
            if len(parts) > 1:
                if mode_manager.set(parts[1]):
                    emit_log(f"✅ Mode changé : {mode_manager.current.upper()}")
                else:
                    emit_log(f"❌ Mode inconnu : {parts[1]}")
            else:
                emit_log(f"🎛️ Mode actuel : {mode_manager.current.upper()}")
            continue
            
        # Recharger l'agent si le mode a changé
        if mode_manager.current != current_mode:
            emit_log(f"🔄 Rechargement de l'agent en mode {mode_manager.current.upper()}...")
            current_agent = create_agent(registry, mode=mode_manager.current)
            current_mode = mode_manager.current

        date_str = datetime.now().strftime("%Y-%m-%d")
        try:
            run_turn(current_agent, memory, line, date_str)
        except Exception as e:
            emit_log(f"❌ Erreur: {e}")
        
        sys.stdout.flush()

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: emit_log("👋 Interrompu.")