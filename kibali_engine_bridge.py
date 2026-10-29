# -*- coding: utf-8 -*-
"""
KIBALI ENGINE BRIDGE v2.0 — Fusion CLI + Flutter
Protocole : stdin/stdout ligne par ligne. Émet des marqueurs pour les 4 vues.
Utilise l'API native llama-server (/v1/chat/completions) au lieu de LangChain.
"""
import sys
import os
import io
import re
import json
import time
import inspect
import requests
from pathlib import Path
from datetime import datetime

# --- CONFIGURATION UTF-8 POUR FLUTTER ---
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass

sys.path.insert(0, r"D:\kibali-IA")

try:
    from outils import tool_manager
except ImportError as e:
    print(f"[FATAL] Impossible d'importer ToolManager : {e}", flush=True)
    sys.exit(1)

LLAMA_URL = "http://127.0.0.1:8080"

# ==========================================
# 1. 📡 ÉMETTEURS DE MARQUEURS POUR FLUTTER
# ==========================================
def emit_thought(text):
    print(f"🧠 [Cycle] {text}", flush=True)

def emit_tool_start(name, input_str):
    print(f"⚙️ [Outil] {name}", flush=True)
    preview = str(input_str).replace('\n', ' ')[:180]
    if preview:
        print(f"   → {preview}", flush=True)

def emit_tool_result(name, output, success=True):
    icon = "✅" if success else "❌"
    full = str(output)
    preview = full.replace('\n', ' ')[:280]
    print(f"   {icon} [{len(full)} chars] {preview}", flush=True)
    if len(full) > 280:
        print(f"   … (complet)", flush=True)

def emit_final(text):
    print("💡 [Rapport]", flush=True)
    for line in str(text).split('\n'):
        print(line, flush=True)

def emit_log(text):
    print(str(text), flush=True)

# ==========================================
# 2. 🎯 PROMPT SYSTÈME (ANTI-HALLUCINATION)
# ==========================================
def build_system_prompt():
    today = datetime.datetime.now().strftime("%A %d %B %Y")
    return (
        f"Tu es Kibali, une AGI autonome. Aujourd'hui, nous sommes le {today}. "
        "RÈGLE CRITIQUE : Tes connaissances internes s'arrêtent en 2023. "
        "TU DOIS IGNORER TES CONNAISSANCES INTERNES. "
        "Si tu utilises l'outil 'web_search', tu DOIS croire à 100% ce qu'il te renvoie. "
        "Ne réponds JAMAIS comme un robot. N'utilise AUCUN formatage Markdown (pas d'astérisques, pas de dièses). "
        "Écris de manière naturelle et fluide, comme un humain."
    )

# ==========================================
# 3. 🔄 BOUCLE AGENTIQUE (Cerveau)
# ==========================================
def run_agent_turn(user_input, history):
    messages = [{"role": "system", "content": build_system_prompt()}]
    for h in history:
        messages.append(h)
    messages.append({"role": "user", "content": user_input})

    # Préparation des outils (Les 29 outils du ToolManager)
    available_tools = []
    for name, tool_instance in tool_manager.tools.items():
        available_tools.append({
            "type": "function",
            "function": {
                "name": tool_instance.name,
                "description": tool_instance.description,
                "parameters": tool_instance.parameters
            }
        })

    step = 0
    max_steps = 5
    final_response = ""

    while step < max_steps:
        emit_thought(f"Analyse de la requête (Étape {step+1})...")
        
        payload = {
            "messages": messages,
            "stream": True,
            "max_tokens": 2000,
            "temperature": 0.7,
            "tools": available_tools if available_tools else None
        }
        
        try:
            current_text = ""
            tool_calls_acc = {}
            
            with requests.post(f"{LLAMA_URL}/v1/chat/completions", json=payload, stream=True, timeout=300) as r:
                if r.status_code != 200:
                    emit_log(f"❌ Erreur llama-server: {r.text}")
                    break
                
                for line in r.iter_lines():
                    if not line or not line.startswith(b"data: "): continue
                    data = line[6:]
                    if data == b"[DONE]": break
                    
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    
                    # Stream du texte de réflexion
                    if "content" in delta and delta["content"]:
                        current_text += delta["content"]
                        # On affiche les réflexions en direct (sans casser le format Flutter)
                        # On n'affiche que si ce n'est pas une réponse finale en cours de génération
                        if "Final Answer:" not in current_text and not tool_calls_acc:
                            pass # Optionnel: on peut print(current_text) si on veut du vrai streaming
                    
                    # Capture des outils
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {"id": tc.get("id"), "name": "", "arguments": ""}
                            if "function" in tc:
                                if tc["function"].get("name"): tool_calls_acc[idx]["name"] += tc["function"]["name"]
                                if tc["function"].get("arguments"): tool_calls_acc[idx]["arguments"] += tc["function"]["arguments"]

            # SI AUCUN OUTIL N'EST APPELÉ -> C'EST LA RÉPONSE FINALE
            if not tool_calls_acc:
                final_response = current_text
                messages.append({"role": "assistant", "content": final_response})
                break
                
            # SINON -> EXÉCUTION DES OUTILS
            assistant_msg = {"role": "assistant", "content": current_text or None, "tool_calls": []}
            for idx, tc in tool_calls_acc.items():
                assistant_msg["tool_calls"].append({
                    "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]}
                })
            messages.append(assistant_msg)
            
            for idx, tc in tool_calls_acc.items():
                func_name = tc["name"]
                try: args = json.loads(tc["arguments"])
                except: args = {}
                
                emit_tool_start(func_name, args)
                
                result = ""
                tool = tool_manager.tools.get(func_name)
                if tool:
                    try:
                        # Introspection pour envoyer les bons arguments
                        sig = inspect.signature(tool.execute)
                        valid_args = {k: v for k, v in args.items() if k in sig.parameters}
                        if "query" in sig.parameters and "query" not in valid_args:
                            valid_args["query"] = user_input
                        
                        res = tool.execute(**valid_args)
                        result = json.dumps(res, ensure_ascii=False) if not isinstance(res, str) else res
                        emit_tool_result(func_name, result, success=True)
                    except Exception as e:
                        result = f"Erreur: {e}"
                        emit_tool_result(func_name, result, success=False)
                else:
                    result = f"Outil {func_name} introuvable."
                    emit_tool_result(func_name, result, success=False)
                    
                messages.append({"role": "tool", "tool_call_id": tc["id"], "name": func_name, "content": result})
            
            step += 1
            
        except Exception as e:
            emit_log(f"❌ Erreur réseau/API: {e}")
            break

    # Affichage final
    if final_response.strip():
        emit_final(final_response.strip())
        return final_response
    else:
        emit_log("⚠️ Aucune réponse finale générée.")
        return ""

# ==========================================
# 4. 🚀 BOUCLE PRINCIPALE (Écoute Flutter)
# ==========================================
def main():
    emit_log("=" * 60)
    emit_log("🤖 KIBALI ENGINE BRIDGE v2.0 (Fusion API + Flutter)")
    emit_log("=" * 60)
    emit_log("✅ READY")
    sys.stdout.flush()

    history = []

    for line in sys.stdin:
        line = line.rstrip('\n').rstrip('\r')
        if not line.strip(): continue
        if line.lower() in ('quit', 'exit'):
            emit_log("👋 Au revoir!")
            break
        
        # Affiche la question utilisateur dans Flutter
        print(f"👤 Vous > {line}", flush=True)
        
        try:
            response = run_agent_turn(line, history)
            if response:
                history.append({"role": "user", "content": line})
                history.append({"role": "assistant", "content": response})
        except Exception as e:
            emit_log(f"❌ Erreur: {e}")
        
        sys.stdout.flush()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        emit_log("👋 Interrompu.")