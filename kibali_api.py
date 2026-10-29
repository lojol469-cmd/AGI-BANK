# -*- coding: utf-8 -*-
import os
import sys
import json
import requests
import datetime
import inspect
import time
import subprocess
import base64
import sqlite3
import mimetypes
import shutil
from typing import List, Dict, Any, Generator, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Depends, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# CHEMINS & CHARGEMENT ENV (SÉCURITÉ CPT)
# ============================================================
KIBALI_ROOT   = r"D:\kibali-IA"
CPT_ROOT      = r"D:\CPT_Analysis_Studio_PORTABLE"
SECURITY_DIR  = os.path.join(CPT_ROOT, "security")

sys.path.insert(0, KIBALI_ROOT)
sys.path.insert(0, CPT_ROOT)
sys.path.insert(0, SECURITY_DIR)

# Charge .env pour MONGO_URI / CPT_SECRET_KEY / CPT_ACCESS_PIN
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(CPT_ROOT, ".env"), override=False)
except Exception:
    pass

# Déverrouille le dossier app/ au démarrage pour éviter les erreurs d'import
try:
    from folder_lock import unlock
    unlock()
except Exception:
    pass

from outils import tool_manager

# ============================================================
# CONFIGURATION & REGISTRE DES MODÈLES
# ============================================================
LLAMA_URL = "http://127.0.0.1:8080"
LLAMA_SERVER = r"C:\Users\BELIKANON\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"

ROOT_MODELS = r"D:\CPT_Analysis_Studio_PORTABLE\models"
ROOT_VL     = r"D:\kibali-IA\models\qwen2-vl-2b"
ROOT_ASR    = r"D:\CPT_Analysis_Studio_PORTABLE\models\qwen3-asr"

DB_PATH = r"D:\kibali-IA\kibali_history.db"
HPO_DIR = r"D:\CPT_Analysis_Studio_PORTABLE\HPO"

MODEL_REGISTRY = {
    "qwen3-8b":       {"path": f"{ROOT_MODELS}\\Qwen3-8B-Q4_K_M.gguf", "context": 32768, "flash_attn": True, "mmproj": None, "thinking": True},
    "deepseek-14b":   {"path": f"{ROOT_MODELS}\\DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf", "context": 32768, "flash_attn": True, "mmproj": None, "thinking": True},
    "qwen3-32b":      {"path": f"{ROOT_MODELS}\\Qwen3-32B-Q4_K_M.gguf", "context": 16384, "flash_attn": True, "mmproj": None, "thinking": True},
    "qwen2-vl-2b":    {"path": f"{ROOT_VL}\\Qwen2-VL-2B-Instruct-Q4_K_M.gguf", "context": 16384, "flash_attn": False, "mmproj": f"{ROOT_VL}\\mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", "thinking": False},
    "qwen3-asr-1.7b": {"path": f"{ROOT_ASR}\\qwen3-asr-1.7b-q4_k.gguf", "context": 8192, "flash_attn": True, "mmproj": None, "thinking": False}
}

CURRENT_LOADED_MODEL = None
_server_proc = None
_hpo_proc = None

REQUIRE_API_AUTH = os.environ.get("KIBALI_REQUIRE_AUTH", "0") == "1"

class ChatRequest(BaseModel):
    message: str
    files: List[Union[str, Dict]] = []
    chat_id: str = "default"
    model_key: str = "qwen3-8b"

# ============================================================
# INITIALISATION BASE DE DONNÉES CHAT (SQLITE)
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, title TEXT, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, role TEXT, content TEXT, reasoning TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ============================================================
# AUTH MANAGER (MongoDB) — SINGLETON
# ============================================================
_auth_manager = None
_auth_available = None

def get_auth_manager():
    global _auth_manager, _auth_available
    if _auth_manager is not None:
        return _auth_manager
    try:
        from auth_manager import AuthManager
        _auth_manager = AuthManager()
        _auth_available = True
        return _auth_manager
    except Exception as e:
        _auth_available = False
        raise HTTPException(status_code=503, detail=f"Sécurité indisponible : {str(e)[:120]}")

def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token manquant")
    token = authorization[7:].strip()
    am = get_auth_manager()
    username, err = am.verify_token(token)
    if err:
        raise HTTPException(status_code=401, detail=err)
    return username

# ============================================================
# GESTION DU SERVEUR LLAMA & HPO (ROBUSTE)
# ============================================================
def kill_port(port):
    if os.name == "nt":
        try:
            out = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True, text=True)
            for line in out.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 5 and parts[1].endswith(f':{port}') and parts[3] == 'LISTENING':
                    subprocess.run(["taskkill", "/F", "/PID", parts[-1]], capture_output=True)
        except: pass
    time.sleep(2)

def switch_model(model_key: str):
    global _server_proc, CURRENT_LOADED_MODEL
    if CURRENT_LOADED_MODEL == model_key: return
    spec = MODEL_REGISTRY.get(model_key)
    if not spec: raise Exception(f"Modèle {model_key} inconnu.")

    print(f"[API] 🔄 Basculement vers le modèle : {model_key}...")
    
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        try: _server_proc.wait(timeout=10)
        except: _server_proc.kill()
    _server_proc = None
    kill_port(8080)

    args = [LLAMA_SERVER, "-m", spec["path"], "--port", "8080", "-ngl", "99", "-c", str(spec["context"]), "--jinja"]
    if spec.get("flash_attn"): args += ["-fa", "on"]
    if spec.get("mmproj"): args += ["--mmproj", spec["mmproj"]]

    log_file = open("llama_server.log", "w", encoding="utf-8")
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    _server_proc = subprocess.Popen(args, stdout=log_file, stderr=subprocess.STDOUT, **kwargs)

    for _ in range(120):
        if _server_proc.poll() is not None:
            log_file.close()
            with open("llama_server.log", "r", encoding="utf-8") as f: err = f.read()[-1500:]
            raise Exception(f"Le modèle {model_key} a crashé au démarrage: {err}")
        try:
            if requests.get(f"{LLAMA_URL}/health", timeout=1).status_code == 200:
                CURRENT_LOADED_MODEL = model_key
                print(f"[API] ✅ Modèle {model_key} chargé avec succès.")
                return
        except: pass
        time.sleep(1)
    raise Exception("Timeout de démarrage du serveur LLM.")

def start_hpo():
    global _hpo_proc
    if _hpo_proc and _hpo_proc.poll() is None: return
    print("[API] 🚀 Lancement du module HPO (Streamlit)...", flush=True)
    _hpo_proc = subprocess.Popen(
        ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8501"],
        cwd=HPO_DIR,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    switch_model("qwen3-8b")
    start_hpo()
    yield
    if _hpo_proc: _hpo_proc.terminate()

app = FastAPI(title="Kibali AGI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ROUTES AUTH (Web / SPA)
# ============================================================
class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str
    otp: str

class VerifyOtpRequest(BaseModel):
    totp_secret: str
    otp: str

@app.get("/api/auth/status")
def auth_status():
    try:
        am = get_auth_manager()
        return {"ok": True, "has_users": am.has_any_user()}
    except HTTPException:
        return {"ok": False, "has_users": False, "error": "MongoDB indisponible"}
    except Exception as e:
        return {"ok": False, "has_users": False, "error": str(e)[:150]}

@app.post("/api/auth/register")
def auth_register(req: RegisterRequest):
    am = get_auth_manager()
    totp_secret, err = am.register_user(req.username, req.password)
    if err:
        raise HTTPException(status_code=400, detail=err)
    uri = am.get_totp_uri(req.username, totp_secret)
    return {"ok": True, "username": req.username.strip(), "totp_secret": totp_secret, "totp_uri": uri}

@app.post("/api/auth/verify-otp")
def auth_verify_otp(req: VerifyOtpRequest):
    am = get_auth_manager()
    if am.verify_totp_only(req.totp_secret, req.otp):
        return {"ok": True, "valid": True}
    return {"ok": True, "valid": False}

@app.post("/api/auth/login")
def auth_login(req: LoginRequest):
    am = get_auth_manager()
    token, err = am.authenticate(req.username, req.password, req.otp)
    if err:
        raise HTTPException(status_code=401, detail=err)
    remaining = am.remaining_seconds(token)
    user_doc = am.users.find_one({"username": req.username.strip()}, {"password": 0, "totp_secret": 0})
    return {
        "ok": True, "token": token, "username": req.username.strip(),
        "role": (user_doc or {}).get("role", "user"), "expires_in": remaining,
    }

@app.get("/api/auth/me")
def auth_me(username: str = Depends(get_current_user)):
    am = get_auth_manager()
    user = am.users.find_one({"username": username}, {"password": 0, "totp_secret": 0})
    if not user: raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    return {"ok": True, "username": user.get("username"), "role": user.get("role", "user")}

@app.post("/api/auth/logout")
def auth_logout(username: str = Depends(get_current_user), authorization: str = Header(None)):
    am = get_auth_manager()
    if authorization: am.revoke_token(authorization[7:].strip())
    return {"ok": True}

# ============================================================
# NOUVELLE ROUTE : UPLOAD DE FICHIERS (Images, Audio, Vidéo)
# ============================================================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        temp_dir = os.path.join(CPT_ROOT, "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        
        safe_filename = "".join([c for c in file.filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).rstrip()
        file_path = os.path.join(temp_dir, safe_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"ok": True, "path": file_path, "filename": safe_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# ROUTES HISTORIQUE CHAT
# ============================================================
@app.get("/api/chats")
def list_chats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM chats ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

@app.post("/api/chats")
def create_chat(title: str = "Nouveau Chat"):
    chat_id = str(int(time.time() * 1000))
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)", (chat_id, title, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return {"id": chat_id, "title": title}

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    c.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/api/chats/{chat_id}/messages")
def get_chat_messages(chat_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content, reasoning FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "reasoning": r[2]} for r in rows]

# ============================================================
# CHAT AGENT & MEMORY RAG (AUTO-SWAP VLM)
# ============================================================
def build_system_prompt(history_context="", has_image=False):
    today = datetime.datetime.now().strftime("%A %d %B %Y")
    
    # Si une image est présente, on donne des instructions strictes de vision
    if has_image:
        return (
            f"Tu es Kibali, une intelligence artificielle autonome dotée de vision. Aujourd'hui : {today}. "
            "L'utilisateur t'a envoyé une image. Analyse-la attentivement et réponds à sa question. "
            "Si l'utilisateur ne pose pas de question spécifique, fais une description détaillée de l'image. "
            "N'ÉCRIS JAMAIS de code Python sauf si l'utilisateur te le demande explicitement."
        )
    
    # Sinon, prompt standard pour le texte et la génération d'apps
    return (
        f"Tu es Kibali, une intelligence artificielle autonome. Aujourd'hui : {today}. "
        f"Voici le contexte de tes échanges passés : {history_context}\n"
        "Pour créer une application, utilise 'gui_launcher_tool' avec action='create'. "
        "Écris UNIQUEMENT du code Python brut. Pas de balises Markdown."
    )

@app.get("/health")
def health():
    return {"status": "ok", "active_model": CURRENT_LOADED_MODEL, "hpo": True, "auth_available": _auth_available}

def _maybe_auth_dep(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if not REQUIRE_API_AUTH: return None
    return get_current_user(authorization)

@app.post("/chat")
def chat_endpoint(req: ChatRequest, username: Optional[str] = Depends(_maybe_auth_dep)):
    def event_stream() -> Generator[str, None, None]:
        global CURRENT_LOADED_MODEL
        try:
            # --- 1. ANALYSE DES FICHIERS ET AUTO-SWAP ---
            user_content = [{"type": "text", "text": req.message}]
            has_image = False
            
            if req.files:
                for f in req.files:
                    file_path = None
                    if isinstance(f, dict):
                        file_path = f.get("path") or f.get("name")
                    elif isinstance(f, str):
                        file_path = f

                    if file_path:
                        # Cas 1 : Base64 direct depuis le navigateur JS
                        if file_path.startswith("data:image"):
                            has_image = True
                            user_content.append({"type": "image_url", "image_url": {"url": file_path}})
                        # Cas 2 : Fichier sur le disque (Gradio ou Upload)
                        elif os.path.exists(file_path):
                            mime_type, _ = mimetypes.guess_type(file_path)
                            if mime_type and mime_type.startswith('image'):
                                has_image = True
                                with open(file_path, "rb") as img_file:
                                    b64_data = base64.b64encode(img_file.read()).decode('utf-8')
                                user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}})
                            else:
                                user_content.append({"type": "text", "text": f"\n[Fichier joint: {os.path.basename(file_path)}]"})
                        else:
                            user_content.append({"type": "text", "text": f"\n[Fichier introuvable: {str(file_path)[:50]}]"})

            # AUTO-SWAP : Si une image est détectée et qu'on est sur un modèle texte, on bascule sur le VLM
            if has_image and not MODEL_REGISTRY[req.model_key].get("mmproj"):
                print("[API] 🖼️ Image détectée ! Basculement automatique vers le modèle Vision (qwen2-vl-2b).")
                req.model_key = "qwen2-vl-2b"

            # Si le texte est vide mais qu'il y a une image, on force un prompt par défaut
            if has_image and (not req.message or req.message.strip() == ""):
                req.message = "Décris cette image en détail."
                user_content[0]["text"] = req.message

            # --- 2. INITIALISATION DU CHAT ET BDD ---
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO messages (chat_id, role, content, reasoning, timestamp) VALUES (?, ?, ?, ?, ?)",
                      (req.chat_id, "user", req.message, "", datetime.datetime.now().isoformat()))
            c.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (req.chat_id,))
            if c.fetchone()[0] == 1:
                title = req.message[:30] + "..." if len(req.message) > 30 else req.message
                c.execute("UPDATE chats SET title = ? WHERE id = ?", (title, req.chat_id))
            conn.commit()
            conn.close()

            yield f"data: {json.dumps({'type': 'status', 'text': 'Démarrage...'})}\n\n"
            
            if CURRENT_LOADED_MODEL != req.model_key:
                switch_model(req.model_key)

            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT 6", (req.chat_id,))
            past_msgs = c.fetchall()[::-1]
            conn.close()

            history_context = "\n".join([f"{r}: {c}" for r, c in past_msgs[:-1]])
            
            # Appel du prompt système avec l'indicateur d'image
            messages = [{"role": "system", "content": build_system_prompt(history_context, has_image)}]
            
            for r, c in past_msgs[:-1]:
                messages.append({"role": r, "content": c})
            
            if len(user_content) > 1:
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": req.message})

            # --- 3. OUTILS ET BOUCLE AGENTIQUE ---
            available_tools = []
            # Qwen2-VL ne supporte pas bien les outils JSON, on les désactive pour lui
            if req.model_key != "qwen2-vl-2b":
                for name, tool_instance in tool_manager.tools.items():
                    available_tools.append({"type": "function", "function": {"name": tool_instance.name, "description": tool_instance.description, "parameters": tool_instance.parameters}})

            step = 0
            max_steps = 10
            accumulated_content = ""

            while step < max_steps:
                payload = {"messages": messages, "stream": True, "max_tokens": 2500, "temperature": 0.1}
                if available_tools: payload["tools"] = available_tools
                if MODEL_REGISTRY[req.model_key].get("thinking"): 
                    payload["chat_template_kwargs"] = {"enable_thinking": True}
                
                try:
                    with requests.post(f"{LLAMA_URL}/v1/chat/completions", json=payload, stream=True, timeout=300) as r:
                        if r.status_code != 200:
                            err_msg = r.text[:300]
                            print(f"[API] ❌ Erreur Llama Server ({r.status_code}): {err_msg}")
                            CURRENT_LOADED_MODEL = None
                            switch_model(req.model_key)
                            yield f"data: {json.dumps({'type': 'error', 'text': f'Erreur LLM ({r.status_code}). Vérifiez les logs.'})}\n\n"
                            break
                        
                        current_text = ""
                        tool_calls_acc = {}
                        
                        for line in r.iter_lines():
                            if not line or not line.startswith(b"data: "): continue
                            data = line[6:]
                            if data == b"[DONE]": break
                            
                            try:
                                chunk = json.loads(data)
                                delta = chunk["choices"][0]["delta"]
                                if "content" in delta and delta["content"]:
                                    current_text += delta["content"]
                                    accumulated_content += delta["content"]
                                    yield f"data: {json.dumps({'type': 'content', 'text': accumulated_content})}\n\n"
                                if "tool_calls" in delta:
                                    for tc in delta["tool_calls"]:
                                        idx = tc.get("index", 0)
                                        if idx not in tool_calls_acc: tool_calls_acc[idx] = {"id": tc.get("id"), "name": "", "arguments": ""}
                                        if "function" in tc:
                                            if tc["function"].get("name"): tool_calls_acc[idx]["name"] += tc["function"]["name"]
                                            if tc["function"].get("arguments"): tool_calls_acc[idx]["arguments"] += tc["function"]["arguments"]
                            except json.JSONDecodeError:
                                continue
                                
                except requests.RequestException as e:
                    print(f"[API] ❌ Connection Error: {e}")
                    CURRENT_LOADED_MODEL = None
                    switch_model(req.model_key)
                    yield f"data: {json.dumps({'type': 'error', 'text': 'Connexion au LLM perdue...'})}\n\n"
                    break

                if not tool_calls_acc:
                    messages.append({"role": "assistant", "content": current_text})
                    break

                # Traitement des outils
                assistant_msg = {"role": "assistant", "content": current_text or None, "tool_calls": []}
                for idx, tc in tool_calls_acc.items():
                    assistant_msg["tool_calls"].append({"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}})
                messages.append(assistant_msg)

                for idx, tc in tool_calls_acc.items():
                    func_name = tc["name"]
                    yield f"data: {json.dumps({'type': 'tool_start', 'name': func_name})}\n\n"
                    try: args = json.loads(tc["arguments"])
                    except: args = {}
                    tool = tool_manager.tools.get(func_name)
                    if tool:
                        try:
                            sig = inspect.signature(tool.execute)
                            valid_args = {k: v for k, v in args.items() if k in sig.parameters}
                            if "query" in sig.parameters and "query" not in valid_args: valid_args["query"] = req.message
                            res = tool.execute(**valid_args)
                            res_str = json.dumps(res, ensure_ascii=False) if not isinstance(res, str) else res
                        except Exception as e:
                            res_str = f"Erreur outil: {e}"
                    else:
                        res_str = "Outil introuvable."
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "name": func_name, "content": res_str})
                    yield f"data: {json.dumps({'type': 'tool_result', 'name': func_name, 'result': res_str[:200]})}\n\n"
                step += 1

            # Sauvegarde de la réponse finale
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO messages (chat_id, role, content, reasoning, timestamp) VALUES (?, ?, ?, ?, ?)", (req.chat_id, "assistant", accumulated_content, "", datetime.datetime.now().isoformat()))
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"[API] ❌ Exception inattendue: {e}")
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============================================================
# MAIN (DÉMARRAGE ULTRA-RAPIDE ET ROBUSTE)
# ============================================================
if __name__ == "__main__":
    print("[API] ✅ FastAPI Kibali Server v2.0 (AUTH TOTP ACTIVE) démarré.", flush=True)
    print("[API] 🔓 L'authentification est gérée par l'interface WEB (Gradio).", flush=True)
    
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)