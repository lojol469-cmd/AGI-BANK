# -*- coding: utf-8 -*-
import os
import sys
import json
from openai import OpenAI

# Import de votre gestionnaire d'outils
try:
    from outils import tool_manager
except ImportError:
    print("❌ Erreur: Impossible d'importer 'outils'. Vérifiez le chemin.")
    sys.exit(1)

# Configuration
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-no-key")
MODEL = "qwen3-8b"

# ==========================================
# 1. VOIR CE QUE L'IA VOIT
# ==========================================
print("\n🔍 [1] EXTRACTION DES OUTILS POUR L'IA (Format OpenAI)")
openai_tools = []
available_tools = tool_manager.get_available_tools()

# On ajoute l'outil Python comme dans Streamlit
openai_tools.append({
    "type": "function",
    "function": {
        "name": "python_executor",
        "description": "Exécute du code Python.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
    }
})

for name, info in available_tools.items():
    if name == "python_executor": continue
    tool_instance = tool_manager.tools.get(name)
    params = getattr(tool_instance, 'parameters', {"type": "object", "properties": {}})
    
    openai_tools.append({
        "type": "function",
        "function": {
            "name": name,
            "description": info['description'],
            "parameters": params
        }
    })

# Afficher les outils que l'IA reçoit
print(f"✅ {len(openai_tools)} outils envoyés au modèle :")
for t in openai_tools:
    print(f"  - {t['function']['name']}")

# ==========================================
# 2. SIMULER LA RÉFLEXION DE L'AGENT
# ==========================================
user_message = "Cherche sur internet qui a gagné le ballon d'or 2026"
print(f"\n👤 [2] QUESTION DE L'UTILISATEUR : {user_message}")

system_prompt = """Tu es Kibali Ultra, un agent autonome.
Tu as l'OBLIGATION d'utiliser tes outils. N'invente JAMAIS de réponse disant que tu ne peux pas le faire."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_message}
]

print("\n🧠 [3] ENVOI AU MODÈLE (Qwen 8B)...")
try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=openai_tools,
        tool_choice="auto", # Laisse l'IA décider
        temperature=0.1
    )
    
    msg = response.choices[0].message
    
    # ==========================================
    # 3. VERDICT : L'IA A-T-ELLE APPELÉ L'OUTIL ?
    # ==========================================
    print("\n📊 [4] RÉSULTAT DE LA RÉFLEXION :")
    
    if msg.tool_calls:
        print("✅ SUCCÈS ! L'IA a décidé d'appeler un outil.")
        for tool_call in msg.tool_calls:
            print(f"  🛠️ Outil choisi : {tool_call.function.name}")
            print(f"  📦 Arguments     : {tool_call.function.arguments}")
            
            # Ici, Streamlit exécuterait l'outil
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            if func_name == "web_search":
                res = tool_manager.tools["web_search"].execute(user_message, args)
                print(f"  📤 Résultat de l'outil : {str(res)[:200]}...")
    else:
        print("❌ ÉCHEC : L'IA a préféré répondre avec du texte au lieu d'appeler l'outil.")
        print(f"  💬 Réponse de l'IA : {msg.content}")
        print("\n👉 CAUSE PROBABLE :")
        print("  1. La description de l'outil 'web_search' n'est pas assez incitative.")
        print("  2. Qwen 8B ne comprend pas le format JSON demandé.")
        print("  3. Le modèle local n'a pas été entraîné pour le Tool Calling.")

except Exception as e:
    print(f"\n❌ Erreur de connexion avec llama-server : {e}")