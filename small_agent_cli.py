# -*- coding: utf-8 -*-
import os
import sys
import json
import requests
import inspect
import time

# Ajout du dossier parent pour les imports
sys.path.insert(0, r"D:\kibali-IA")
from outils import tool_manager

LLAMA_URL = "http://127.0.0.1:8080"

def run_small_agent():
    print("="*60)
    print("🤖 KIBALI SMALL AGENT (Thinking + Streaming)")
    print("="*60)
    print("Tapez 'quit' pour quitter.\n")

    # 1. OUTILS ESSENTIELS
    essential_tools = ["execute_python", "web_search", "rag_bridge"]
    available_tools = []
    for name in essential_tools:
        if name in tool_manager.tools:
            tool = tool_manager.tools[name]
            available_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })

    # 2. PROMPT DIRECT
    messages = [{
        "role": "system", 
        "content": (
            "Tu es Kibali, une AGI. Pour créer une application, utilise l'outil 'execute_python' "
            "pour écrire le fichier avec open('app.py', 'w') et le lancer avec python. "
            "Ne mets JAMAIS de balises Markdown (```) dans le code que tu envoies à l'outil."
        )
    }]

    while True:
        user_input = input("\n👤 Vous : ")
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        
        while True:
            print("\n🤖 Kibali : ", end="", flush=True)
            
            payload = {
                "messages": messages,
                "stream": True,
                "max_tokens": 2500,
                "temperature": 0.7,
                "tools": available_tools,
                "chat_template_kwargs": {"enable_thinking": True}
            }
            
            current_text = ""
            tool_calls_acc = {}
            
            try:
                with requests.post(f"{LLAMA_URL}/v1/chat/completions", json=payload, stream=True, timeout=300) as r:
                    if r.status_code != 200:
                        print(f"\n❌ Erreur llama-server: {r.text}")
                        break
                        
                    for line in r.iter_lines():
                        if not line or not line.startswith(b"data: "): continue
                        data = line[6:]
                        if data == b"[DONE]": break
                        
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"]
                        
                        # --- STREAMING TOKEN PAR TOKEN ---
                        if "content" in delta and delta["content"]:
                            current_text += delta["content"]
                            print(delta["content"], end="", flush=True)
                        
                        # Capture des outils
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {"id": tc.get("id"), "name": "", "arguments": ""}
                                if "function" in tc:
                                    if tc["function"].get("name"): tool_calls_acc[idx]["name"] += tc["function"]["name"]
                                    if tc["function"].get("arguments"): tool_calls_acc[idx]["arguments"] += tc["function"]["arguments"]
            except Exception as e:
                print(f"\n❌ Erreur réseau: {e}")
                break

            print() # Saut de ligne

            if not tool_calls_acc:
                messages.append({"role": "assistant", "content": current_text})
                break

            # --- EXÉCUTION DE L'OUTIL ---
            assistant_msg = {"role": "assistant", "content": current_text or None, "tool_calls": []}
            for idx, tc in tool_calls_acc.items():
                assistant_msg["tool_calls"].append({
                    "id": tc["id"], "type": "function", 
                    "function": {"name": tc["name"], "arguments": tc["arguments"]}
                })
            messages.append(assistant_msg)

            for idx, tc in tool_calls_acc.items():
                func_name = tc["name"]
                print(f"\n🛠️ [Outil] Exécution de {func_name}...")
                
                try: args = json.loads(tc["arguments"])
                except: args = {}
                    
                if "query" in args and isinstance(args["query"], str):
                    args["query"] = args["query"].replace("```python", "").replace("```", "").strip()

                tool = tool_manager.tools.get(func_name)
                if tool:
                    try:
                        sig = inspect.signature(tool.execute)
                        valid_args = {k: v for k, v in args.items() if k in sig.parameters}
                        if "query" in sig.parameters and "query" not in valid_args:
                            valid_args["query"] = user_input
                            
                        res = tool.execute(**valid_args)
                        res_str = json.dumps(res, ensure_ascii=False) if not isinstance(res, str) else res
                        print(f"✅ [Résultat] {res_str[:200]}...\n")
                    except Exception as e:
                        res_str = f"Erreur: {e}"
                        print(f"❌ [Erreur] {res_str}\n")
                else:
                    res_str = "Outil introuvable."
                
                messages.append({"role": "tool", "tool_call_id": tc["id"], "name": func_name, "content": res_str})
            
            print("⏳ Reprise de la réflexion...\n")

if __name__ == "__main__":
    run_small_agent()