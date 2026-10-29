# -*- coding: utf-8 -*-
import os
import sys
import json
import requests
import datetime
import inspect
import time

# Ajout du dossier parent pour que l'import fonctionne
sys.path.insert(0, r"D:\kibali-IA")

from outils import tool_manager

LLAMA_URL = "http://127.0.0.1:8080"
MODEL_NAME = "Qwen3-8B-Q4_K_M.gguf" # Le nom du modèle chargé

def build_system_prompt():
    today = datetime.datetime.now().strftime("%A %d %B %Y")
    # LE PROMPT ULTIME POUR ÉCRASER LE BIAIS DU LLM
    return (
        f"Tu es Kibali, une AGI autonome. Aujourd'hui, nous sommes le {today}. "
        "RÈGLE CRITIQUE : Tes connaissances internes s'arrêtent en 2023. "
        "TU DOIS IGNORER TES CONNAISSANCES INTERNES. "
        "Si tu utilises l'outil 'web_search', tu DOIS croire à 100% ce qu'il te renvoie, "
        "même si cela contredit ce que tu penses savoir. "
        "Ne dis JAMAIS 'nous sommes en 2023'. La date du jour est celle donnée ci-dessus."
    )

def run_agent_cli():
    print("="*60)
    print("🤖 KIBALI AGI - MODE CLI (Terminal)")
    print(f"📅 Date système : {datetime.datetime.now().strftime('%Y-%m-%d')}")
    print("="*60)
    print("Tapez 'quit' ou 'exit' pour quitter.\n")

    # Préparation des outils
    available_tools = []
    for tool_name, tool_instance in tool_manager.tools.items():
        available_tools.append({
            "type": "function",
            "function": {
                "name": tool_instance.name,
                "description": tool_instance.description,
                "parameters": tool_instance.parameters
            }
        })
    
    print(f"🛠️ {len(available_tools)} outils chargés en mémoire.\n")

    messages = [{"role": "system", "content": build_system_prompt()}]

    while True:
        user_input = input("\n👤 Vous : ")
        if user_input.lower() in ["quit", "exit"]:
            print("Au revoir !")
            break
        
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        
        step = 0
        max_steps = 5
        
        while step < max_steps:
            print("⏳ L'AGI réfléchit...")
            
            payload = {
                "messages": messages,
                "stream": False, # On passe en False pour simplifier le CLI
                "max_tokens": 2000,
                "temperature": 0.7,
                "tools": available_tools if available_tools else None
            }
            
            try:
                r = requests.post(f"{LLAMA_URL}/v1/chat/completions", json=payload, timeout=300)
                if r.status_code != 200:
                    print(f"❌ Erreur llama-server: {r.text}")
                    break
                
                response = r.json()
                choice = response["choices"][0]
                msg = choice["message"]
                
                # Si l'IA appelle un outil
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func_name = tc["function"]["name"]
                        args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                        
                        print(f"🔧 [Appel Outil] {func_name} avec args: {args}")
                        
                        # On ajoute le message de l'assistant avec l'appel d'outil
                        messages.append({"role": "assistant", "tool_calls": [tc]})
                        
                        # Exécution de l'outil
                        tool = tool_manager.tools.get(func_name)
                        if tool:
                            try:
                                sig = inspect.signature(tool.execute)
                                valid_args = {k: v for k, v in args.items() if k in sig.parameters}
                                if "query" in sig.parameters and "query" not in valid_args:
                                    valid_args["query"] = user_input
                                
                                res = tool.execute(**valid_args)
                                res_str = json.dumps(res, ensure_ascii=False) if not isinstance(res, str) else res
                                print(f"✅ [Résultat Outil] {res_str[:300]}...\n")
                                
                                # On renvoie le résultat au LLM
                                messages.append({
                                    "role": "tool", 
                                    "tool_call_id": tc["id"], 
                                    "name": func_name, 
                                    "content": res_str
                                })
                            except Exception as e:
                                print(f"❌ Erreur outil: {e}")
                                messages.append({"role": "tool", "tool_call_id": tc["id"], "name": func_name, "content": f"Erreur: {e}"})
                        else:
                            print(f"❌ Outil introuvable: {func_name}")
                            messages.append({"role": "tool", "tool_call_id": tc["id"], "name": func_name, "content": "Outil introuvable."})
                    
                    step += 1
                    continue # On relance la boucle pour que le LLM analyse le résultat
                
                # Si pas d'appel d'outil, c'est la réponse finale
                else:
                    print(f"\n🤖 Kibali : {msg['content']}\n")
                    messages.append({"role": "assistant", "content": msg["content"]})
                    break
                    
            except Exception as e:
                print(f"❌ Erreur réseau: {e}")
                break

if __name__ == "__main__":
    run_agent_cli()