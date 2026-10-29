import atexit
import base64
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import requests
from fastapi.responses import FileResponse, HTMLResponse
from gradio.utils import abspath, get_upload_folder, is_in_or_equal

from params import PARAM_SPECS, inject_param_config, validate_params

# ============================================================
# CONFIGURATION (UI SEULEMENT)
# ============================================================
STATIC_DIR = Path(__file__).parent / "static"
KIBALI_LOGO_PATH = Path(r"D:\CPT_Analysis_Studio_PORTABLE\kibali_loops\tool\kibali-IA\kibali_logo.svg")
API_URL = "http://127.0.0.1:8000"

MODEL_REGISTRY = {
    "qwen3-8b": {"display_name": "Qwen3-8B — Texte Rapide"},
    "deepseek-14b": {"display_name": "DeepSeek-R1-14B — Raisonnement"},
    "qwen3-32b": {"display_name": "Qwen3-32B — Le Monstre"},
    "qwen2-vl-2b": {"display_name": "Qwen2-VL-2B — Vision"},
    "qwen3-asr-1.7b": {"display_name": "Qwen3-ASR-1.7B — Audio"},
}

ACTIVE_MODEL_KEY = os.getenv("DEFAULT_MODEL", "qwen3-8b")
CURRENT_UI_MODEL = ACTIVE_MODEL_KEY

# ============================================================
# APPEL DE L'API
# ============================================================
app = gr.Server()

@app.get("/api/health")
def health_check():
    try:
        r = requests.get(f"{API_URL}/health", timeout=1)
        if r.status_code == 200:
            data = r.json()
            return {"llm": True, "api": True, "active_model": data.get("active_model", "Inconnu")}
    except:
        pass
    return {"llm": False, "api": False}

@app.get("/api/set_model")
def set_model_route(model: str):
    global CURRENT_UI_MODEL
    if model in MODEL_REGISTRY:
        CURRENT_UI_MODEL = model
        try:
            requests.post(f"{API_URL}/set_model?model_key={model}", timeout=120)
        except Exception as e:
            print(f"Erreur changement modèle API: {e}")
        return {"status": "ok", "active": model}
    return {"status": "error"}

# --- CORRECTION ICI : On remet TOUS les arguments attendus par app.js ---
@app.api(name="chat")
def chat(
    text: str, files: list[str] | None = None, history: list[dict] | None = None,
    thinking: bool = False, model_key: str = "qwen3-8b",
    max_new_tokens: int = 2000, image_token_budget: int = 280, system_prompt: str = "",
    temperature: float = 1.0, top_p: float = 0.95, top_k: int = 64, repetition_penalty: float = 1.0,
    **kwargs
) -> Iterator[dict]:
    
    global CURRENT_UI_MODEL
    if CURRENT_UI_MODEL in MODEL_REGISTRY: model_key = CURRENT_UI_MODEL
    files = files or []
    history = history or []
    if not text.strip() and not files: raise gr.Error("Message vide.")

    hist_formatted = [{"role": h["role"], "content": h["text"]} for h in history]
    
    payload = {
        "message": text,
        "files": files,
        "history": hist_formatted,
        "model_key": model_key
    }
    
    # Variable pour conserver le texte affiché
    accumulated_content = ""
    
    try:
        with requests.post(f"{API_URL}/chat", json=payload, stream=True, timeout=300) as r:
            if r.status_code != 200:
                raise gr.Error(f"Erreur API ({r.status_code}): {r.text}")
            
            for line in r.iter_lines():
                if not line: continue
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]": break
                    
                    try:
                        data = json.loads(data_str)
                        
                        if data["type"] == "content":
                            # On met à jour le texte global
                            accumulated_content = data["text"]
                            yield {"reasoning": "", "content": accumulated_content}
                            
                        elif data["type"] == "tool_start":
                            # On garde accumulated_content pour ne pas effacer
                            yield {"reasoning": f"🛠️ Exécution de {data['name']}...", "content": accumulated_content}
                            
                        elif data["type"] == "status":
                            # On garde accumulated_content pour ne pas effacer
                            yield {"reasoning": f"ℹ️ {data['text']}", "content": accumulated_content}
                            
                        elif data["type"] == "error":
                            raise gr.Error(data["text"])
                            
                    except json.JSONDecodeError:
                        pass
                        
    except requests.RequestException as e:
        raise gr.Error(f"Impossible de contacter l'API sur le port 8000. Est-elle lancée ? ({e})")

# ============================================================
# INTERFACE & LOGO & HEALTH
# ============================================================
_NO_STORE = {"Cache-Control": "no-store"}

@app.get("/kibali_logo.svg")
def get_kibali_logo():
    if KIBALI_LOGO_PATH.exists():
        return FileResponse(KIBALI_LOGO_PATH, media_type="image/svg+xml")
    return HTMLResponse("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" rx="20" fill="#1e1e1e" stroke="#444" stroke-width="2"/><text x="50" y="75" font-size="70" font-weight="bold" fill="#4a9eff" text-anchor="middle" font-family="sans-serif">K</text></svg>""", media_type="image/svg+xml")

@app.get("/")
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    
    html = html.replace("Gemma 4 12B IT", "Kibali AGI")
    favicon_tag = '<link rel="icon" type="image/svg+xml" href="/kibali_logo.svg">'
    html = html.replace("</head>", f"{favicon_tag}\n</head>")
    
    options_html = "".join([f'<option value="{k}">{v["display_name"]}</option>' for k,v in MODEL_REGISTRY.items()])
    
    script_html = f"""
    <script>
    window.addEventListener('load', () => {{
        const div = document.createElement('div');
        div.id = 'kibali-model-widget';
        div.style.cssText = 'position:fixed; top:20px; right:20px; z-index:99999; background:rgba(30,30,30,0.95); padding:15px; border-radius:12px; border:2px solid #4a9eff; color:white; box-shadow: 0 8px 24px rgba(0,0,0,0.5); font-family:sans-serif; backdrop-filter: blur(10px); width: 250px;';
        div.innerHTML = `
            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <img src="/kibali_logo.svg" style="width:32px; height:32px; margin-right:10px; border-radius:6px;" />
                <label style="font-size:14px; font-weight:bold; color:#fff; letter-spacing: 0.5px;">Intelligence Engine:</label>
            </div>
            <select id="kibali-model-selector" style="width:100%; padding:8px; background:#2b2b2b; color:white; border:1px solid #555; border-radius:6px; font-size:13px; cursor:pointer; outline: none;">
                {options_html}
            </select>
            <div style="margin-top: 10px; display: flex; gap: 15px; justify-content: center; font-size: 11px;">
                <div style="display:flex; align-items:center;"><span id="dot-llm" style="width:10px; height:10px; border-radius:50%; background:gray; margin-right:5px;"></span> API</div>
                <div style="display:flex; align-items:center;"><span id="dot-active" style="color:#888; margin-left:5px;">Chargement...</span></div>
            </div>
        `;
        document.body.appendChild(div);
        
        const sel = document.getElementById('kibali-model-selector');
        sel.value = "{ACTIVE_MODEL_KEY}";
        sel.addEventListener('change', (e) => {{
            const dot = document.getElementById('dot-llm');
            if (dot) dot.style.background = '#ffaa00';
            
            fetch('/api/set_model?model=' + e.target.value)
                .then(r => r.json())
                .then(data => console.log('Engine switched:', data))
                .catch(err => console.error(err));
        }});

        // Sidebar Logo
        const sidebarLogoDiv = document.createElement('div');
        sidebarLogoDiv.style.cssText = 'padding: 20px 10px; display: flex; align-items: center; justify-content: center; border-bottom: 1px solid #333; margin-bottom: 20px;';
        sidebarLogoDiv.innerHTML = `
            <img src="/kibali_logo.svg" style="width:50px; height:50px; margin-right:15px; border-radius:8px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);" />
            <div>
                <div style="font-size:18px; font-weight:bold; color:#fff;">KIBALI</div>
                <div style="font-size:12px; color:#888;">Autonomous AGI</div>
            </div>
        `;
        const sidebar = document.querySelector('div[role="navigation"]') || document.querySelector('.sidebar');
        if (sidebar) {{
            sidebar.insertBefore(sidebarLogoDiv, sidebar.firstChild);
        }}

        // Health Check
        setInterval(async () => {{
            try {{
                const res = await fetch('/api/health');
                const data = await res.json();
                const dotLlm = document.getElementById('dot-llm');
                const dotActive = document.getElementById('dot-active');
                if (dotLlm) {{
                    if (data.api) {{
                        dotLlm.style.background = '#00ff00';
                        if (dotActive) dotActive.innerText = data.active_model;
                    }} else {{
                        dotLlm.style.background = '#ff0000';
                        if (dotActive) dotActive.innerText = 'Hors ligne';
                    }}
                }}
            }} catch (e) {{}}
        }}, 3000);
    }});
    </script>
    """
    html = html.replace('</body>', f'{script_html}</body>')
    return HTMLResponse(inject_param_config(html), headers=_NO_STORE)

@app.get("/app.js")
def app_js() -> FileResponse: return FileResponse(STATIC_DIR / "app.js", headers=_NO_STORE)
@app.get("/style.css")
def style_css() -> FileResponse: return FileResponse(STATIC_DIR / "style.css", headers=_NO_STORE)

if __name__ == "__main__":
    app.launch(allowed_paths=[str(STATIC_DIR)], max_file_size="20MB")