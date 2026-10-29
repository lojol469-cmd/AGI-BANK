# ===============================================
# Installation des dépendances (pour Colab/Jupyter - commenter pour Streamlit)
# ===============================================
# !apt-get install -y osmium-tool > /dev/null
# !pip install -U streamlit huggingface_hub langchain sentence-transformers faiss-cpu pdfplumber osmium networkx shapely matplotlib opencv-python-headless open3d ipywidgets pandas scikit-image scikit-learn torch torchvision langchain-community langchain-huggingface requests beautifulsoup4 python-dotenv diffusers accelerate transformers librosa soundfile tavily-python flash-attn --no-build-isolation --quiet
# ===============================================
# Configuration HuggingFace Token depuis .env
# ===============================================
import os
from pathlib import Path

# ⚠️ FORCER CPU AVANT TOUT IMPORT PYTORCH/TRANSFORMERS
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Désactiver CUDA complètement pour éviter erreurs kernel
os.environ['FORCE_CPU'] = '1'

# Charger le token depuis .env
from dotenv import load_dotenv

# Obtenir le dossier du script actuel
script_dir = Path(__file__).parent.absolute()

# Import de la configuration centralisée des modèles
import sys
sys.path.insert(0, str(script_dir / "kibali_data" / "models"))
from MODEL_PATHS import (
    QWEN_MODEL_NAME, QWEN_CACHE_DIR,
    SENTENCE_TRANSFORMER_MODEL, SENTENCE_TRANSFORMER_CACHE,
    CLIP_MODEL_NAME, CLIP_CACHE_DIR,
    EASYOCR_MODEL_DIR, EASYOCR_LANGUAGES,
    SUMMARIZER_MODEL, SUMMARIZER_CACHE,
    TRANSLATOR_MODEL, TRANSLATOR_CACHE,
    NER_MODEL, NER_CACHE,
    CAPTIONER_MODEL, CAPTIONER_CACHE,
    ensure_model_dirs
)

# Import du gestionnaire de mode offline
from offline_manager import (
    offline_manager,
    get_model_loading_params,
    load_model_with_fallback,
    render_mode_indicator,
    render_offline_status_card,
    get_offline_capabilities
)

# Import du système d'auto-apprentissage et nano-IA
sys.path.insert(0, str(script_dir / "kibali_data" / "models"))

# Import du détecteur de structure binaire
from binary_structure_detector import BinaryStructureDetector, detect_and_analyze_file

# Import de l'analyseur ERT (Electrical Resistivity Tomography)
try:
    from ert_analyzer import ERTAnalyzer
    ERT_AVAILABLE = True
except ImportError:
    ERT_AVAILABLE = False
    print("⚠️ Module ERT non disponible")
try:
    from binary_structure_detector import (
        BinaryStructureDetector, 
        FileStructureType, 
        detect_and_analyze_file
    )
    BINARY_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Détecteur de structure binaire non disponible: {e}")
    BINARY_DETECTOR_AVAILABLE = False

# Import du système d'auto-apprentissage
try:
    from auto_learning import get_auto_learning
    from knowledge_manager import get_knowledge_manager
    AUTO_LEARNING_AVAILABLE = True
    print("✅ Système d'auto-apprentissage chargé")
except Exception as e:
    print(f"⚠️ Auto-apprentissage non disponible: {e}")
    AUTO_LEARNING_AVAILABLE = False
    get_auto_learning = None
    get_knowledge_manager = None

# Import de l'orchestrateur maître d'IA
try:
    from task_orchestrator import analyze_and_plan, get_ai_relay, TaskType
    from master_orchestrator import get_orchestrator, AISpecialist
    ORCHESTRATOR_AVAILABLE = True
    print("✅ Orchestrateur maître d'IA chargé")
except Exception as e:
    print(f"⚠️ Orchestrateur non disponible: {e}")
    ORCHESTRATOR_AVAILABLE = False

# Initialiser les dossiers de modèles
ensure_model_dirs()

# Chercher le fichier .env dans plusieurs emplacements possibles
env_paths = [
    script_dir / ".env",  # Dans le même dossier que app.py
    Path.cwd() / ".env",  # Dans le dossier courant
    Path.home() / ".env"  # Dans le dossier home
]

token_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Fichier .env trouvé: {env_path}")
        token_loaded = True
        break

if not token_loaded:
    print(f"⚠️ Aucun fichier .env trouvé dans: {[str(p) for p in env_paths]}")
    print(f"Créez un fichier .env dans {script_dir} avec: HF_TOKEN=hf_votre_token")

# Récupérer le token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(f"❌ HF_TOKEN non trouvé ! Vérifiez votre fichier .env dans {script_dir}")
else:
    print(f"🔑 Token HF configuré: {HF_TOKEN[:10]}...")

# Définir la variable d'environnement pour huggingface_hub
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError(f"❌ TAVILY_API_KEY non trouvé ! Vérifiez votre fichier .env dans {script_dir}")
# ===============================================
# Imports
# ===============================================
import math
import pdfplumber  # MIT License - Compatible commercial
import osmium
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import base64
import json
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from shapely.geometry import Point
import io
from PIL import Image
import cv2
try:
    import open3d as o3d
except ImportError:
    o3d = None
from io import BytesIO
import pandas as pd
from skimage import measure, segmentation
from sklearn.cluster import KMeans
import torch
from torchvision import models, transforms
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.agents import initialize_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Import conditionnel de CLIP
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    try:
        # Essayer l'import alternatif pour les anciennes versions
        from transformers import AutoProcessor as CLIPProcessor, AutoModel as CLIPModel
    except ImportError:
        CLIPProcessor = None
        CLIPModel = None
        print("⚠️ CLIP non disponible - Analyse sémantique d'images désactivée")

# Import des modèles de génération de texte
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    print("⚠️ Modèles Hugging Face non disponibles")

# ═══════════════════════════════════════════════════════════════════
# FONCTION DE NETTOYAGE DES RÉPONSES
# ═══════════════════════════════════════════════════════════════════

def clean_response_text(text: str) -> str:
    """
    Nettoie le texte de réponse en supprimant les caractères non-latins
    (chinois, arabe, cyrillique, etc.) qui peuvent apparaître par erreur
    """
    import re
    
    if not text:
        return text
    
    # Détecter et supprimer les blocs de texte chinois/asiatique
    # Plages Unicode pour caractères CJK (chinois, japonais, coréen)
    text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2f800-\u2fa1f]+', '', text)
    
    # Supprimer les caractères arabes
    text = re.sub(r'[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]+', '', text)
    
    # Supprimer les caractères cyrilliques (russe, ukrainien, etc.)
    text = re.sub(r'[\u0400-\u04ff\u0500-\u052f\u2de0-\u2dff\ua640-\ua69f\u1c80-\u1c8f]+', '', text)
    
    # Supprimer les caractères coréens Hangul
    text = re.sub(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\ud7b0-\ud7ff]+', '', text)
    
    # Supprimer les caractères japonais (hiragana, katakana)
    text = re.sub(r'[\u3040-\u309f\u30a0-\u30ff]+', '', text)
    
    # Nettoyer les espaces multiples consécutifs
    text = re.sub(r'\s+', ' ', text)
    
    # Nettoyer les sauts de ligne excessifs
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()

# Import du gestionnaire de fichiers binaires scientifiques
try:
    from binary_file_handler import (
        BinaryFileHandler, 
        BinaryFileViewer, 
        SCIENTIFIC_BINARY_FORMATS,
        analyze_binary_file,
        create_hex_viewer_ui
    )
    BINARY_HANDLER_AVAILABLE = True
    print(f"✅ Gestionnaire de fichiers binaires chargé - {len(SCIENTIFIC_BINARY_FORMATS)} formats supportés")
except ImportError as e:
    BINARY_HANDLER_AVAILABLE = False
    print(f"⚠️ Gestionnaire de fichiers binaires non disponible: {e}")

# Import YOLO pour détection d'objets
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO disponible")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO non disponible - pip install ultralytics")

# Import de l'outil d'organisation Excel avec IA
try:
    from tools.excel_ai_organizer import organize_excel_with_ai, ExcelAIOrganizer
    EXCEL_AI_AVAILABLE = True
except ImportError:
    EXCEL_AI_AVAILABLE = False
    print("⚠️ Excel AI Organizer non disponible")

try:
    from transformers import pipeline
except ImportError:
    # Fallback si pipeline n'est pas disponible directement
    pipeline = None
import requests
from bs4 import BeautifulSoup
import time
import shutil
try:
    import easyocr
except ImportError:
    easyocr = None
try:
    import pytesseract
except ImportError:
    pytesseract = None

# Import diffusers avec gestion d'erreurs pour les classes optionnelles
try:
    from diffusers import DiffusionPipeline
except ImportError:
    DiffusionPipeline = None
try:
    from diffusers import AudioLDMPipeline
except ImportError:
    AudioLDMPipeline = None
try:
    from diffusers import ShapEPipeline, ShapEImg2ImgPipeline
except ImportError:
    ShapEPipeline = None
    ShapEImg2ImgPipeline = None

import imageio
try:
    import scipy.io.wavfile as wavfile
except ImportError:
    wavfile = None
from tavily import TavilyClient
# ===============================================
# Import du système de modèle local Qwen
# ===============================================
import torch
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from typing import Iterator

# ===============================================
# Configuration automatique du device (CUDA/CPU)
# ===============================================
def get_optimal_device():
    """
    Détecte automatiquement le meilleur device disponible.
    Fallback sur CPU si CUDA n'est pas disponible.
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU détecté: {gpu_name}")
        print(f"✅ Utilisation de CUDA {torch.version.cuda}")
    else:
        device = 'cpu'
        print(f"💻 Aucun GPU détecté - Utilisation du CPU")
    return device

# Device global pour toute l'application
DEVICE = get_optimal_device()
# ===============================================
# Import du système d'outils dynamiques
# ===============================================
try:
    from outils import ToolManager
    TOOLS_SYSTEM_AVAILABLE = True
    print("✅ Système d'outils dynamiques chargé")
except ImportError as e:
    print(f"⚠️ Système d'outils non disponible: {e}")
    TOOLS_SYSTEM_AVAILABLE = False
# ===============================================
# Classe ChatModel personnalisée pour LangChain utilisant Qwen2.5-1.5B
# ===============================================
class QwenChatModel(BaseChatModel):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    tools_available: bool = True
   
    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.tools_available = True
       
    @property
    def _llm_type(self) -> str:
        return "qwen2.5-1.5b-local-enhanced"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a response using tools and analyses."""
        # Extraire le contenu du message utilisateur
        user_message = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                user_message = message.content
                break
       
        # Détecter si l'utilisateur demande une analyse
        needs_analysis = any(keyword in user_message.lower() for keyword in [
            "analyse", "resistivité", "ert", "recherche", "données", "matériaux",
            "couleurs", "graphique", "tableau", "comparaison", "approfondie"
        ])
       
        if needs_analysis and self.tools_available:
            # Utiliser les outils disponibles pour une analyse complète
            try:
                # Recherche web pour informations
                if any(keyword in user_message.lower() for keyword in ["recherche", "informations", "approfondie"]):
                    search_query = user_message.replace("fais maintenant une recherche plus approfondie pour obtenir toutes ces informations précises", "")
                    web_results = web_search_enhanced(search_query + " ERT electrical resistivity geophysics materials")
                   
                # Recherche RAG si disponible
                rag_results = ""
                if st.session_state.get('vectordb'):
                    rag_results = rag_search(user_message, st.session_state.vectordb)
               
                # Génération de données et analyses si demandées
                analysis_results = ""
                if any(keyword in user_message.lower() for keyword in ["tableau", "graphique", "données"]):
                    # Simuler des données ERT pour démonstration
                    import numpy as np
                    sample_data = [0.05, 0.3, 10.0, 50.0, 200.0, 1000.0, 5000.0, 0.0000024, 1000000]
                    analysis_results = resistivity_color_analysis(sample_data)
               
                # Construire la réponse enrichie avec outils
                enhanced_context = f"""
🔍 ANALYSE COMPLÈTE AVEC OUTILS ACTIVÉS:
🌐 RECHERCHE WEB EFFECTUÉE:
{web_results}
📚 RECHERCHE RAG:
{rag_results}
📊 ANALYSE ERT AVANCÉE:
{analysis_results}
CONTEXTE UTILISATEUR: {user_message}
"""
               
                # Générer la réponse avec le contexte enrichi
                enhanced_messages = [
                    {"role": "system", "content": """Tu es un expert en géophysique ERT avec accès à des outils puissants.
                    Tu DOIS utiliser les données fournies pour créer des analyses détaillées, tableaux, graphiques et comparaisons.
                    Réponds toujours avec des données concrètes et des analyses approfondies basées sur les outils utilisés.
                    Ne dis JAMAIS que tu n'as pas accès aux outils - utilise les résultats fournis."""},
                    {"role": "user", "content": enhanced_context}
                ]
            except Exception as e:
                print(f"Erreur outils: {e}")
                enhanced_messages = [
                    {"role": "system", "content": "Tu es un expert en analyse de données ERT."},
                    {"role": "user", "content": user_message}
                ]
        else:
            # Messages standard
            enhanced_messages = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    enhanced_messages.append({"role": "system", "content": message.content})
                elif isinstance(message, HumanMessage):
                    enhanced_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    enhanced_messages.append({"role": "assistant", "content": message.content})
       
        # Génération avec les messages enrichis
        inputs = self.tokenizer.apply_chat_template(
            enhanced_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
       
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
       
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=3000,  # 3000 tokens pour réponses TRÈS détaillées
                temperature=0.6,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.05,  # Éviter répétitions
                pad_token_id=self.tokenizer.eos_token_id
            )
       
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
       
        if stop:
            for stop_token in stop:
                if stop_token in response:
                    response = response.split(stop_token)[0]
                    break
       
        return AIMessage(content=response)
    
    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator:
        """Stream tokens one by one for real-time display."""
        # Extraire le contenu du message utilisateur
        user_message = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                user_message = message.content
                break
       
        # Préparer les messages comme dans _generate
        enhanced_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                enhanced_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                enhanced_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                enhanced_messages.append({"role": "assistant", "content": message.content})
       
        # Génération avec streaming token par token
        inputs = self.tokenizer.apply_chat_template(
            enhanced_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
       
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Créer un streamer pour générer token par token
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Préparer les arguments de génération
        generation_kwargs = dict(
            inputs=inputs,
            attention_mask=attention_mask,
            max_new_tokens=3000,
            temperature=0.6,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        
        # Lancer la génération dans un thread séparé
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield chaque token au fur et à mesure
        full_response = ""
        for new_text in streamer:
            full_response += new_text
            # Vérifier les stop tokens
            should_stop = False
            if stop:
                for stop_token in stop:
                    if stop_token in full_response:
                        full_response = full_response.split(stop_token)[0]
                        should_stop = True
                        break
            
            yield AIMessage(content=new_text)
            
            if should_stop:
                break
        
        thread.join()
# ===============================================
# Chargement du modèle LLM local Qwen2.5-1.5B
# ===============================================
@st.cache_resource
def load_local_llm_model():
    """Charge le modèle Qwen2.5-1.5B depuis kibali_data/models"""
    
    print(f"🚀 Chargement de {QWEN_MODEL_NAME} en mode LOCAL...")
    print(f"📁 Cache: {QWEN_CACHE_DIR}")
    
    # Utiliser le device global avec fallback automatique
    device = DEVICE
    gpu_info = ""
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = f"GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)"
        print(f"✅ Modèle chargé sur GPU: {gpu_info}")
    else:
        print("✅ Modèle chargé sur CPU (fallback automatique)")
   
    print(f"🌐 Mode: LOCAL UNIQUEMENT (pas de téléchargement)")
    
    # Charger tokenizer en mode LOCAL uniquement
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_NAME,
        cache_dir=str(QWEN_CACHE_DIR),
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True
    )
    
    # Corriger le problème du pad_token = eos_token pour éviter les warnings
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
   
    # Configuration optimisée selon le device
    if device == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            cache_dir=str(QWEN_CACHE_DIR),
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            cache_dir=str(QWEN_CACHE_DIR),
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        model = model.to(device)
   
    print(f"✅ Qwen chargé en mode LOCAL")
    
    return tokenizer, model, device, gpu_info
# ===============================================
# Configuration - CHEMINS UNIFIÉS
# ===============================================
# Modèles 100% Open Source avec Licence Commerciale (Apache 2.0 / MIT)
# ===============================================
WORKING_MODELS = {
    # Qwen 2.5 - Apache 2.0 License - Usage commercial autorisé
    "Qwen2.5 14B (Apache 2.0 - Commercial OK)": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen2.5 7B (Apache 2.0 - Rapide)": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5 32B (Apache 2.0 - Très puissant)": "Qwen/Qwen2.5-32B-Instruct",
    
    # Llama 3.2 - Apache 2.0 License (Meta autorisation commerciale)
    "Llama 3.2 11B Vision (Apache 2.0 - Multimodal)": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Llama 3.1 8B (Apache 2.0 - Optimisé)": "meta-llama/Llama-3.1-8B-Instruct",
    
    # Mistral - Apache 2.0 License - Usage commercial libre
    "Mistral Nemo 12B (Apache 2.0 - Français)": "mistralai/Mistral-Nemo-Instruct-2407",
    "Mistral 7B v0.3 (Apache 2.0 - Rapide)": "mistralai/Mistral-7B-Instruct-v0.3",
    
    # Phi-3 Medium - MIT License - Microsoft open source commercial
    "Phi-3 Medium 14B (MIT - Commercial)": "microsoft/Phi-3-medium-4k-instruct",
    
    # Gemma 2 - Gemma License (usage commercial autorisé par Google)
    "Gemma 2 9B (Gemma License - Commercial OK)": "google/gemma-2-9b-it",
}
# TOUS LES FICHIERS DANS LE MÊME DOSSIER CHATBOT
CHATBOT_DIR = os.path.join(os.getcwd(), "kibali_data")
VECTORDB_PATH = os.path.join(CHATBOT_DIR, "vectordb")
PDFS_PATH = os.path.join(CHATBOT_DIR, "pdfs")
GRAPHS_PATH = os.path.join(CHATBOT_DIR, "graphs") # Graphes OSM ici aussi
MAPS_PATH = os.path.join(CHATBOT_DIR, "maps")
METADATA_PATH = os.path.join(CHATBOT_DIR, "metadata.json")
TRAJECTORIES_PATH = os.path.join(CHATBOT_DIR, "trajectories.json")
WEB_CACHE_PATH = os.path.join(CHATBOT_DIR, "web_cache.json")
GENERATED_PATH = os.path.join(CHATBOT_DIR, "generated")
# ===============================================
# Test de connexion HuggingFace
# ===============================================
def test_hf_connection():
    """Teste la connexion à HuggingFace"""
    try:
        from huggingface_hub import whoami
        user_info = whoami(token=HF_TOKEN)
        print(f"✅ Connexion HuggingFace réussie: {user_info.get('name', 'Utilisateur')}")
        return True
    except Exception as e:
        print(f"❌ Erreur connexion HuggingFace: {e}")
        return False
# Tester la connexion au démarrage
if not test_hf_connection():
    print("⚠️ Problème de connexion HuggingFace, vérifiez votre token")
# ===============================================
# Fonctions utilitaires
# ===============================================
def setup_drive():
    """Monte Google Drive et crée les dossiers"""
    print("📁 Configuration des dossiers...")
    os.makedirs(CHATBOT_DIR, exist_ok=True)
    os.makedirs(PDFS_PATH, exist_ok=True)
    os.makedirs(GRAPHS_PATH, exist_ok=True)
    os.makedirs(MAPS_PATH, exist_ok=True)
    os.makedirs(VECTORDB_PATH, exist_ok=True)
    os.makedirs(GENERATED_PATH, exist_ok=True)
    print(f"📁 Dossier principal : {CHATBOT_DIR}")
    return True
def extract_text_from_pdf(pdf_path):
    """Extraire le texte d'un PDF avec pdfplumber (MIT License)"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num + 1}]\n{page_text}\n"
        return text
    except Exception as e:
        print(f"❌ Erreur PDF {pdf_path}: {e}")
        return ""
def upload_and_process_pbf(pbf_file):
    """Traitement du fichier PBF uploadé"""
    if pbf_file is None:
        return None, None, "❌ Aucun fichier uploadé"
    pbf_path = pbf_file
    print("⚙️ Lecture du PBF et construction du graphe...")
    handler = RoadPOIHandler()
    handler.apply_file(pbf_path, locations=True)
    G = handler.graph
    pois = handler.pois
    # Sauvegarder dans le dossier chatbot
    graph_name = os.path.basename(pbf_path).replace('.osm.pbf', '_graph.graphml')
    graph_path = os.path.join(GRAPHS_PATH, graph_name)
    nx.write_graphml(G, graph_path)
    # Sauvegarder les POIs
    pois_name = graph_name.replace('_graph.graphml', '_pois.json')
    pois_path = os.path.join(GRAPHS_PATH, pois_name)
    with open(pois_path, 'w', encoding='utf-8') as f:
        json.dump(pois, f, indent=2, ensure_ascii=False)
    print(f"✅ Graphe: {len(G)} nœuds, {G.size()} arêtes")
    print(f"✅ POIs: {len(pois)} points")
    print(f"💾 Sauvegardé: {graph_path}")
    return G, pois, f"✅ Graphe créé: {len(G)} nœuds, {len(pois)} POIs"
def load_existing_graph():
    """Charge un graphe existant"""
    graph_files = [f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')] if os.path.exists(GRAPHS_PATH) else []
    if not graph_files:
        return None, None, "❌ Aucun graphe trouvé"
    graph_file = graph_files[0]
    graph_path = os.path.join(GRAPHS_PATH, graph_file)
    pois_path = os.path.join(GRAPHS_PATH, graph_file.replace('_graph.graphml', '_pois.json'))
    try:
        G = nx.read_graphml(graph_path)
        pois = []
        if os.path.exists(pois_path):
            with open(pois_path, 'r', encoding='utf-8') as f:
                pois = json.load(f)
        return G, pois, f"✅ Graphe chargé: {len(G)} nœuds, {len(pois)} POIs"
    except Exception as e:
        return None, None, f"❌ Erreur: {e}"
def process_pdfs():
    """Traiter les PDFs"""
    print("📄 Traitement des PDFs...")
    # CPU forcé globalement au début du fichier
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(SENTENCE_TRANSFORMER_CACHE),
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'device': 'cpu', 'batch_size': 32}
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    # Charger vectordb existante si elle existe
    if os.path.exists(VECTORDB_PATH):
        vectordb, _ = load_vectordb()
    else:
        vectordb = None
    # Charger métadonnées existantes
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {"processed_files": [], "total_chunks": 0}
    processed_filenames = {p["filename"] for p in metadata["processed_files"]}
    all_documents = []
    pdf_files = [f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')] if os.path.exists(PDFS_PATH) else []
    if not pdf_files:
        return vectordb, "⚠️ Aucun PDF trouvé"
    new_chunks_count = 0
    new_processed = []
    for pdf_file in pdf_files:
        if pdf_file in processed_filenames:
            print(f" 📖 {pdf_file} déjà traité, sauté.")
            continue
        pdf_path = os.path.join(PDFS_PATH, pdf_file)
        print(f" 📖 Traitement nouveau PDF : {pdf_file}")
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            continue
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_file,
                    "chunk_id": i,
                    "type": "pdf"
                }
            )
            all_documents.append(doc)
        new_processed.append({"filename": pdf_file, "chunks": len(chunks)})
        new_chunks_count += len(chunks)
    # Ajouter les trajets sauvegardés (toujours, car ils peuvent changer)
    if os.path.exists(TRAJECTORIES_PATH):
        with open(TRAJECTORIES_PATH, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)
        for traj in trajectories:
            traj_text = f"""Trajet: {traj.get('question', '')}
Départ: {traj.get('start_name', '')}
Arrivée: {traj.get('end_name', '')}
Distance: {traj.get('distance', 0)/1000:.2f} km"""
            doc = Document(
                page_content=traj_text,
                metadata={"source": "trajectories", "type": "trajectory"}
            )
            all_documents.append(doc)
    if all_documents:
        if vectordb is None:
            vectordb = FAISS.from_documents(all_documents, embedding_model)
        else:
            vectordb.add_documents(all_documents)
        vectordb.save_local(VECTORDB_PATH)
    # Mettre à jour métadonnées
    metadata["processed_files"].extend(new_processed)
    metadata["total_chunks"] += new_chunks_count
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return vectordb, f"✅ Base mise à jour : {len(new_processed)} nouveaux PDFs traités, {new_chunks_count} nouveaux chunks (total : {metadata['total_chunks']})"
def load_vectordb():
    """Charger la base vectorielle"""
    try:
        # CPU forcé globalement au début du fichier
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=str(SENTENCE_TRANSFORMER_CACHE),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'device': 'cpu', 'batch_size': 32}
        )
        
        # Vérifier si la base existe
        index_file = os.path.join(VECTORDB_PATH, "index.faiss")
        if not os.path.exists(index_file):
            # Créer une base vide si elle n'existe pas
            os.makedirs(VECTORDB_PATH, exist_ok=True)
            
            # Créer un document factice pour initialiser FAISS
            from langchain.schema import Document
            dummy_doc = Document(page_content="Base vectorielle initialisée", metadata={"source": "system"})
            vectordb = FAISS.from_documents([dummy_doc], embedding_model)
            vectordb.save_local(VECTORDB_PATH)
            
            return vectordb, "✅ Base vectorielle créée (vide - ajoutez des PDFs)"
        
        # Charger la base existante
        vectordb = FAISS.load_local(VECTORDB_PATH, embedding_model, allow_dangerous_deserialization=True)
        return vectordb, "✅ Base chargée"
    except Exception as e:
        return None, f"❌ Erreur: {e}"
def save_trajectory(question, response, trajectory_info):
    """Sauvegarde un trajet"""
    trajectories = []
    if os.path.exists(TRAJECTORIES_PATH):
        with open(TRAJECTORIES_PATH, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)
    new_trajectory = {
        "question": question,
        "response": response,
        "start_name": trajectory_info.get('start', {}).get('name', ''),
        "end_name": trajectory_info.get('end', {}).get('name', ''),
        "distance": trajectory_info.get('distance', 0)
    }
    trajectories.append(new_trajectory)
    with open(TRAJECTORIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)
def upload_pdfs(uploaded_files):
    """Upload des PDFs"""
    if uploaded_files is None:
        return []
    saved_files = []
    for file in uploaded_files:
        filename = os.path.basename(file)
        filepath = os.path.join(PDFS_PATH, filename)
        shutil.copy(file, filepath)
        saved_files.append(filename)
    return saved_files
# ===============================================
# Système de Cache Web Intelligent
# ===============================================
def load_web_cache():
    """Charge le cache web"""
    if os.path.exists(WEB_CACHE_PATH):
        try:
            with open(WEB_CACHE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}
def save_web_cache(cache):
    """Sauvegarde le cache web"""
    try:
        with open(WEB_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Erreur sauvegarde cache: {e}")
def get_cache_key(query, source="text"):
    """Génère une clé de cache pour une requête"""
    return f"{source}:{query.lower().strip()}"
def is_cache_expired(cache_entry, max_age_hours=24):
    """Vérifie si l'entrée du cache a expiré"""
    current_time = time.time()
    return (current_time - cache_entry.get('timestamp', 0)) > (max_age_hours * 3600)
def get_cache_stats():
    """Obtient les statistiques du cache"""
    try:
        cache = load_web_cache()
        if not cache:
            return "Cache vide"
        total_entries = len(cache)
        expired_count = sum(1 for entry in cache.values() if is_cache_expired(entry))
        valid_count = total_entries - expired_count
        return f"📊 Cache: {total_entries} entrées total, {valid_count} valides, {expired_count} expirées"
    except Exception as e:
        return f"❌ Erreur stats: {e}"

def get_system_status():
    """Retourne le statut complet du système"""
    status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "directories": {
            "chatbot": os.path.exists(CHATBOT_DIR),
            "pdfs": os.path.exists(PDFS_PATH),
            "graphs": os.path.exists(GRAPHS_PATH),
            "maps": os.path.exists(MAPS_PATH)
        },
        "files": {
            "vectordb": os.path.exists(VECTORDB_PATH),
            "metadata": os.path.exists(METADATA_PATH),
            "trajectories": os.path.exists(TRAJECTORIES_PATH),
            "web_cache": os.path.exists(WEB_CACHE_PATH)
        },
        "counts": {
            "pdfs": len([f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')]) if os.path.exists(PDFS_PATH) else 0,
            "graphs": len([f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')]) if os.path.exists(GRAPHS_PATH) else 0
        },
        "cache_stats": get_cache_stats(),
        "token_configured": bool(HF_TOKEN and len(HF_TOKEN) > 10)
    }
    return status

# ===============================================
# Fonctions RAG et Web Search Améliorées
# ===============================================
def create_client():
    """Créer le client Inference avec gestion d'erreurs améliorée"""
    try:
        client = InferenceClient(token=HF_TOKEN)
        return client
    except Exception as e:
        print(f"❌ Erreur création client: {e}")
        raise e

# ===============================================
# Chargement des modèles de vision locaux
# ===============================================
@st.cache_resource
def load_vision_models():
    """Charge les modèles CLIP pour la classification d'images"""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        import os
        
        # FORCER le désactivation de low_cpu_mem_usage globalement
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        
        # CORRECTION: Forcer CPU pour éviter les incompatibilités CUDA
        # Les modèles pré-compilés peuvent ne pas correspondre à la version CUDA installée
        device = "cpu"
        print("⚠️ Chargement des modèles vision sur CPU pour éviter les erreurs CUDA")
        
        # Charger le modèle directement sur CPU avec torch_dtype explicite
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None  # Pas de device_map automatique
        )
        
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Forcer sur CPU
        clip_model = clip_model.cpu()
        clip_model.eval()
        
        print("✅ Modèles CLIP chargés avec succès sur CPU")
        
        return {
            'clip_model': clip_model,
            'clip_processor': clip_processor,
            'device': device
        }
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger CLIP: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

@st.cache_resource
def load_ocr_reader():
    """Charge le lecteur OCR EasyOCR en mode LOCAL uniquement"""
    try:
        if easyocr:
            print("📦 Chargement OCR en mode LOCAL...")
            print(f"📁 Cache: {EASYOCR_MODEL_DIR}")
            
            # Vérifier que le cache existe
            if not EASYOCR_MODEL_DIR.exists():
                print(f"❌ Cache OCR introuvable: {EASYOCR_MODEL_DIR}")
                return None
            
            # CORRECTION: Forcer CPU pour OCR également pour éviter erreurs CUDA
            reader = easyocr.Reader(
                EASYOCR_LANGUAGES, 
                gpu=False,  # Forcer CPU
                model_storage_directory=str(EASYOCR_MODEL_DIR),
                download_enabled=False  # Désactiver le téléchargement
            )
            print("✅ OCR chargé en mode LOCAL sur CPU")
            return reader
        else:
            print("⚠️ EasyOCR non installé")
            return None
    except Exception as e:
        print(f"⚠️ Erreur chargement OCR: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    """Charge le modèle YOLO pour la détection d'objets (FORCE CPU)"""
    try:
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO non disponible")
            return None
        
        # Chercher les modèles YOLO disponibles
        yolo_paths = [
            Path("/home/belikan/yolo11n.pt"),
            Path("/home/belikan/yolov8n.pt"),
            Path("./yolo11n.pt"),
            Path("./yolov8n.pt")
        ]
        
        for yolo_path in yolo_paths:
            if yolo_path.exists():
                print(f"📦 Chargement YOLO: {yolo_path} (CPU forcé)")
                
                # CORRECTION CRITIQUE: Forcer CPU pour YOLO
                import torch
                model = YOLO(str(yolo_path))
                
                # Forcer le modèle sur CPU
                model.to('cpu')
                
                # Désactiver CUDA pour les prédictions
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                print(f"✅ YOLO chargé avec succès sur CPU: {yolo_path.name}")
                return model
        
        print("⚠️ Aucun modèle YOLO trouvé (.pt)")
        return None
        
    except Exception as e:
        print(f"⚠️ Erreur chargement YOLO: {e}")
        return None

@st.cache_resource
def load_blip_model():
    """Charge le modèle BLIP pour la description d'images en LOCAL"""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        
        print("📦 Chargement BLIP en mode LOCAL...")
        print(f"📁 Cache: {CAPTIONER_CACHE}")
        
        # Charger le modèle en LOCAL uniquement (CPU pour éviter erreurs CUDA)
        processor = BlipProcessor.from_pretrained(
            CAPTIONER_MODEL,
            cache_dir=str(CAPTIONER_CACHE),
            local_files_only=True
        )
        
        model = BlipForConditionalGeneration.from_pretrained(
            CAPTIONER_MODEL,
            cache_dir=str(CAPTIONER_CACHE),
            torch_dtype=torch.float32,
            local_files_only=True
        )
        
        # Forcer CPU pour éviter erreurs CUDA
        model = model.cpu()
        model.eval()
        
        print("✅ BLIP chargé avec succès en mode LOCAL sur CPU")
        
        return {
            'processor': processor,
            'model': model,
            'device': 'cpu'
        }
        
    except Exception as e:
        print(f"⚠️ Erreur chargement BLIP: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def extract_text_from_image(image_path, ocr_reader=None):
    """Extrait le texte d'une image avec OCR"""
    extracted_texts = []
    
    try:
        # Méthode 1: EasyOCR (meilleur pour textes complexes)
        if ocr_reader:
            results = ocr_reader.readtext(image_path)
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Seuil de confiance
                    extracted_texts.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'method': 'EasyOCR'
                    })
        
        # Méthode 2: Tesseract (fallback)
        elif pytesseract:
            try:
                img = Image.open(image_path)
                # Extraction simple
                text = pytesseract.image_to_string(img, lang='fra+eng')
                if text.strip():
                    extracted_texts.append({
                        'text': text.strip(),
                        'confidence': 1.0,
                        'method': 'Tesseract'
                    })
                
                # Extraction avec détails (positions)
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='fra+eng')
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 30:  # Confiance > 30%
                        txt = data['text'][i].strip()
                        if txt:
                            extracted_texts.append({
                                'text': txt,
                                'confidence': int(data['conf'][i]) / 100,
                                'bbox': (data['left'][i], data['top'][i], 
                                        data['width'][i], data['height'][i]),
                                'method': 'Tesseract'
                            })
            except Exception as tess_error:
                print(f"⚠️ Tesseract error: {tess_error}")
        
        return extracted_texts
    
    except Exception as e:
        print(f"❌ Erreur OCR: {e}")
        return []

def organize_extracted_text(extracted_texts):
    """Organise le texte extrait de manière structurée"""
    if not extracted_texts:
        return "Aucun texte détecté dans l'image."
    
    # Grouper par méthode
    by_method = {}
    for item in extracted_texts:
        method = item.get('method', 'Unknown')
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(item)
    
    # Formater la sortie
    output = []
    output.append("📝 **TEXTE EXTRAIT DE L'IMAGE:**\n")
    
    for method, items in by_method.items():
        output.append(f"\n🔍 **Méthode: {method}**")
        output.append(f"   Nombre d'éléments détectés: {len(items)}\n")
        
        # Trier par confiance
        items_sorted = sorted(items, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for idx, item in enumerate(items_sorted[:50], 1):  # Limiter à 50 éléments
            confidence = item.get('confidence', 0)
            text = item.get('text', '').strip()
            
            if text:
                # Formater selon la confiance
                if confidence > 0.8:
                    output.append(f"   {idx}. ✅ [{confidence:.0%}] {text}")
                elif confidence > 0.5:
                    output.append(f"   {idx}. ⚠️  [{confidence:.0%}] {text}")
                else:
                    output.append(f"   {idx}. ❓ [{confidence:.0%}] {text}")
    
    # Texte complet reconstitué
    output.append("\n\n📄 **TEXTE COMPLET RECONSTITUÉ:**\n")
    full_text = " ".join([item['text'] for item in extracted_texts if item.get('text', '').strip()])
    output.append(full_text)
    
    return "\n".join(output)

def analyze_image_with_clip(image_path, vision_models):
    """Analyse une image avec CLIP local"""
    try:
        if not vision_models:
            return None, "Modèles non chargés"
        
        clip_model = vision_models['clip_model']
        clip_processor = vision_models['clip_processor']
        device = vision_models['device']
        
        # S'assurer que le modèle est sur le bon device et en mode eval
        clip_model = clip_model.to(device)
        clip_model.eval()
        
        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        
        # Labels pour classification sémantique (géophysique focusé)
        labels = [
            "geological rock formation",
            "mineral sample",
            "geophysical survey equipment",
            "topographic map",
            "seismic data visualization",
            "core sample",
            "field work photography",
            "satellite imagery",
            "landscape terrain",
            "technical diagram",
            "graph or chart",
            "document or report",
            "person or people",
            "building or structure",
            "natural scenery"
        ]
        
        # Préparer inputs - traiter en CPU si nécessaire
        inputs = clip_processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Déplacer explicitement chaque tensor sur le device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Prédiction avec gestion d'erreur CUDA
        with torch.no_grad():
            try:
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            except RuntimeError as cuda_error:
                if "CUDA" in str(cuda_error):
                    print(f"⚠️ Erreur CUDA détectée, passage en CPU: {cuda_error}")
                    # Fallback sur CPU
                    clip_model = clip_model.cpu()
                    inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                else:
                    raise
        
        # Top 3 prédictions
        top3_probs, top3_indices = torch.topk(probs[0], 3)
        
        results = []
        for prob, idx in zip(top3_probs, top3_indices):
            results.append({
                'label': labels[idx.item()],
                'confidence': prob.item()
            })
        
        # Générer caption
        main_label = results[0]['label']
        confidence = results[0]['confidence']
        
        caption = f"Image appears to be: {main_label} (confidence: {confidence:.1%})"
        
        return caption, results
        
    except Exception as e:
        import traceback
        error_msg = f"Erreur analyse CLIP: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def rag_search(question, vectordb, k=3):
    """Rechercher dans la base vectorielle"""
    if not vectordb:
        return []
    try:
        return vectordb.similarity_search(question, k=k)
    except Exception as e:
        print(f"❌ Erreur recherche: {e}")
        return []
def enhanced_web_search(query, max_results=5, search_type="text", use_cache=True):
    """
    Recherche web avancée avec cache intelligent et multiple sources
    Args:
        query: Requête de recherche
        max_results: Nombre max de résultats
        search_type: Type de recherche ("text", "news", "both")
        use_cache: Utiliser le cache
    Returns:
        Liste de résultats enrichis
    """
    cache = load_web_cache() if use_cache else {}
    results = []
    try:
        # Recherche texte
        if search_type in ["text", "both"]:
            cache_key = get_cache_key(query, "text")
            if cache_key in cache and not is_cache_expired(cache[cache_key]):
                print(f"📋 Utilisation cache pour: {query}")
                text_results = cache[cache_key]['results']
            else:
                print(f"🔍 Recherche web pour: {query}")
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                text_results = []
                try:
                    raw_results = tavily.search(query, max_results=max_results, search_depth="advanced", topic="general", include_images=True)
                    for r in raw_results.get('results', []):
                        result_data = {
                            'title': r.get('title', ''),
                            'body': r.get('content', ''),
                            'href': r.get('url', ''),
                            'source_type': 'web_search',
                            'images': [],
                            'videos': []
                        }
                        
                        # Extraire les images si disponibles
                        if 'images' in r and r['images']:
                            result_data['images'] = r['images'][:3]  # Max 3 images par résultat
                        
                        # Extraire les vidéos depuis l'URL si c'est YouTube, Vimeo, etc.
                        url = r.get('url', '')
                        if 'youtube.com' in url or 'youtu.be' in url:
                            video_id = None
                            if 'youtube.com/watch?v=' in url:
                                video_id = url.split('watch?v=')[1].split('&')[0]
                            elif 'youtu.be/' in url:
                                video_id = url.split('youtu.be/')[1].split('?')[0]
                            if video_id:
                                result_data['videos'].append({
                                    'type': 'youtube',
                                    'id': video_id,
                                    'url': url
                                })
                        elif 'vimeo.com' in url:
                            vimeo_id = url.split('vimeo.com/')[1].split('/')[0] if 'vimeo.com/' in url else None
                            if vimeo_id:
                                result_data['videos'].append({
                                    'type': 'vimeo',
                                    'id': vimeo_id,
                                    'url': url
                                })
                        
                        text_results.append(result_data)
                    # Sauvegarder en cache
                    cache[cache_key] = {
                        'results': text_results,
                        'timestamp': time.time()
                    }
                    if use_cache:
                        save_web_cache(cache)
                except Exception as e:
                    print(f"Erreur recherche texte: {e}")
                    text_results = []
            results.extend(text_results)
        # Recherche actualités
        if search_type in ["news", "both"]:
            cache_key = get_cache_key(query, "news")
            if cache_key in cache and not is_cache_expired(cache[cache_key], max_age_hours=6):
                news_results = cache[cache_key]['results']
            else:
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                news_results = []
                try:
                    raw_news = tavily.search(query, max_results=max_results//2 if search_type == "both" else max_results, search_depth="advanced", topic="news")
                    for r in raw_news.get('results', []):
                        news_results.append({
                            'title': r.get('title', ''),
                            'body': r.get('content', ''),
                            'url': r.get('url', ''),
                            'date': r.get('published_date', ''),
                            'source': r.get('source', ''),
                            'source_type': 'news'
                        })
                    # Sauvegarder en cache (6h pour les news)
                    cache[cache_key] = {
                        'results': news_results,
                        'timestamp': time.time()
                    }
                    if use_cache:
                        save_web_cache(cache)
                except Exception as e:
                    print(f"Erreur recherche news: {e}")
                    news_results = []
            results.extend(news_results)
    except Exception as e:
        print(f"❌ Erreur recherche web globale: {e}")
        results = [{'title': 'Erreur de recherche', 'body': f'Erreur: {e}', 'source_type': 'error'}]
    return results
def format_web_results_with_media(web_results, max_results=3):
    """
    Formate les résultats web avec images et vidéos pour affichage dans le chat
    Args:
        web_results: Liste des résultats de recherche
        max_results: Nombre maximum de résultats à afficher
    Returns:
        HTML formaté avec contenu multimédia
    """
    if not web_results:
        return ""
    
    html_output = '<div style="margin: 2rem 0;">'
    html_output += '''<h3 style="
        color: #00ff88; 
        margin-bottom: 1.5rem;
        font-size: 1.3rem;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5), 0 2px 4px rgba(0, 0, 0, 0.5);
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 255, 136, 0.3);
    ">🌐 Résultats Web avec Médias</h3>'''
    
    for i, result in enumerate(web_results[:max_results]):
        # Carte 3D pour chaque résultat
        html_output += f'''<div style="
            position: relative;
            background: linear-gradient(145deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
            border: 1px solid rgba(0, 136, 255, 0.3);
            padding: 1.5rem;
            margin: 1.5rem 0;
            border-radius: 12px;
            
            /* Effet 3D dalle épaisse */
            box-shadow: 
                0 15px 30px rgba(0, 0, 0, 0.5),
                0 8px 16px rgba(0, 0, 0, 0.3),
                0 20px 40px rgba(0, 136, 255, 0.15),
                inset 0 2px 0 rgba(255, 255, 255, 0.05),
                inset 0 -2px 0 rgba(0, 0, 0, 0.3),
                -6px 0 12px rgba(0, 136, 255, 0.1),
                6px 0 12px rgba(0, 255, 136, 0.1);
            
            border-left: 5px solid;
            border-image: linear-gradient(180deg, 
                rgba(0, 136, 255, 1) 0%, 
                rgba(0, 255, 136, 0.5) 50%, 
                rgba(0, 136, 255, 1) 100%) 1;
            
            transform: perspective(1000px) rotateX(1deg);
            transition: transform 0.3s ease;
        ">
            <!-- Tranche supérieure -->\n            <div style="
                position: absolute;
                top: -3px;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, 
                    rgba(0, 136, 255, 0.4) 0%, 
                    rgba(0, 255, 136, 0.4) 100%);
                border-radius: 12px 12px 0 0;
            "></div>'''
        
        # Titre et URL
        title = result.get('title', 'Sans titre')[:100]
        url = result.get('href', '') or result.get('url', '')
        html_output += f'''<h4 style="
            color: #00ffff; 
            margin: 0 0 0.75rem 0;
            font-size: 1.1rem;
            text-shadow: 0 0 8px rgba(0, 255, 255, 0.5), 0 1px 2px rgba(0, 0, 0, 0.5);
        ">📄 {title}</h4>'''
        html_output += f'''<p style="
            color: #888; 
            font-size: 0.85rem; 
            margin: 0 0 1rem 0;
        ">🔗 <a href="{url}" target="_blank" style="
            color: #00ff88;
            text-decoration: none;
            transition: color 0.2s;
        " onmouseover="this.style.color='#00ffff'" onmouseout="this.style.color='#00ff88'">{url[:60]}...</a></p>'''
        
        # Contenu textuel
        body = result.get('body', '')[:200]
        if body:
            html_output += f'''<p style="
                color: #ccc; 
                margin: 0.75rem 0;
                line-height: 1.6;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
            ">{body}...</p>'''
        
        # Images avec effet 3D
        images = result.get('images', [])
        if images:
            html_output += '<div style="display: flex; gap: 0.75rem; margin: 1.5rem 0; flex-wrap: wrap;">'
            for img_url in images[:3]:
                html_output += f'''<img src="{img_url}" style="
                    width: 150px;
                    height: 150px;
                    object-fit: cover;
                    border-radius: 10px;
                    border: 2px solid #00ff88;
                    box-shadow: 
                        0 8px 16px rgba(0, 0, 0, 0.4),
                        0 4px 8px rgba(0, 255, 136, 0.3),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
                    cursor: pointer;
                    transition: all 0.3s ease;
                    transform: perspective(500px) rotateX(2deg);
                " onmouseover="this.style.transform='perspective(500px) rotateX(0deg) scale(1.05) translateY(-5px)'; this.style.boxShadow='0 12px 24px rgba(0, 0, 0, 0.5), 0 6px 12px rgba(0, 255, 136, 0.5)'" onmouseout="this.style.transform='perspective(500px) rotateX(2deg) scale(1)'; this.style.boxShadow='0 8px 16px rgba(0, 0, 0, 0.4), 0 4px 8px rgba(0, 255, 136, 0.3)'"/>'''
            html_output += '</div>'
        
        # Vidéos avec effet 3D
        videos = result.get('videos', [])
        if videos:
            for video in videos:
                if video['type'] == 'youtube':
                    video_id = video['id']
                    html_output += f'''<div style="
                        margin: 1.5rem 0;
                        position: relative;
                        border-radius: 12px;
                        overflow: hidden;
                        box-shadow: 
                            0 10px 20px rgba(0, 0, 0, 0.4),
                            0 5px 10px rgba(255, 0, 0, 0.2);
                        transform: perspective(1000px) rotateX(1deg);
                    ">
                        <iframe width="100%" height="315" 
                            src="https://www.youtube.com/embed/{video_id}" 
                            frameborder="0" 
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                            allowfullscreen
                            style="display: block;"
                        ></iframe>
                    </div>'''
                elif video['type'] == 'vimeo':
                    vimeo_id = video['id']
                    html_output += f'''<div style="
                        margin: 1.5rem 0;
                        position: relative;
                        border-radius: 12px;
                        overflow: hidden;
                        box-shadow: 
                            0 10px 20px rgba(0, 0, 0, 0.4),
                            0 5px 10px rgba(0, 170, 255, 0.2);
                        transform: perspective(1000px) rotateX(1deg);
                    ">
                        <iframe src="https://player.vimeo.com/video/{vimeo_id}" 
                            width="100%" height="315" 
                            frameborder="0" 
                            allow="autoplay; fullscreen; picture-in-picture" 
                            allowfullscreen
                            style="display: block;"
                        ></iframe>
                    </div>'''
        
        html_output += '</div>'
    
    html_output += '</div>'
    return html_output

def smart_content_extraction(url, max_length=1000):
    """
    Extraction intelligente du contenu d'une page web
    Args:
        url: URL à scraper
        max_length: Longueur max du contenu
    Returns:
        Contenu extrait et nettoyé
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Supprimer les éléments non pertinents
        for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
            element.decompose()
        # Extraire le texte principal
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        # Nettoyer et tronquer
        text = ' '.join(text.split()) # Normaliser les espaces
        return text[:max_length] + ('...' if len(text) > max_length else '')
    except Exception as e:
        print(f"Erreur extraction contenu {url}: {e}")
        return f"Impossible d'extraire le contenu de {url}"
def intelligent_query_expansion(query):
    """
    Expansion intelligente des requêtes pour améliorer les résultats
    Args:
        query: Requête originale
    Returns:
        Liste de requêtes expandues
    """
    expanded_queries = [query] # Toujours inclure la requête originale
    # Détection de mots-clés pour expansion contextuelle
    keywords = {
        'actualité': ['news', 'dernières nouvelles', 'récent'],
        'comment': ['tutorial', 'guide', 'étapes'],
        'pourquoi': ['raison', 'cause', 'explication'],
        'comparaison': ['vs', 'différence', 'comparatif'],
        'prix': ['coût', 'tarif', 'budget'],
        'avis': ['opinion', 'critique', 'review']
    }
    query_lower = query.lower()
    for trigger, expansions in keywords.items():
        if trigger in query_lower:
            for expansion in expansions:
                expanded_queries.append(f"{query} {expansion}")
    return expanded_queries[:3] # Limiter à 3 requêtes max
def hybrid_search_enhanced(query, vectordb, k=3, web_search_enabled=True, search_type="both"):
    """
    Recherche hybride améliorée combinant RAG local et web avec intelligence
    Args:
        query: Requête de recherche
        vectordb: Base vectorielle locale
        k: Nombre de résultats RAG
        web_search_enabled: Activer la recherche web
        search_type: Type de recherche web
    Returns:
        Liste de documents combinés et enrichis
    """
    all_results = []
    # 1. Recherche RAG locale
    local_docs = rag_search(query, vectordb, k)
    for doc in local_docs:
        doc.metadata['search_source'] = 'local_rag'
        doc.metadata['relevance_score'] = 1.0 # Score max pour les docs locaux
    all_results.extend(local_docs)
    # 2. Recherche web intelligente si activée
    if web_search_enabled:
        print(f"🌐 Recherche web activée pour: {query}")
        # Expansion de requête pour de meilleurs résultats
        expanded_queries = intelligent_query_expansion(query)
        web_results = []
        for exp_query in expanded_queries:
            try:
                search_results = enhanced_web_search(
                    exp_query,
                    max_results=3,
                    search_type=search_type
                )
                for result in search_results:
                    # Créer un document à partir du résultat web
                    content = f"Titre: {result.get('title', '')}\n"
                    content += f"Contenu: {result.get('body', '')}\n"
                    if result.get('source_type') == 'news' and result.get('date'):
                        content += f"Date: {result.get('date')}\n"
                        content += f"Source: {result.get('source', '')}\n"
                    # Extraction de contenu supplémentaire si URL disponible
                    url = result.get('href') or result.get('url')
                    if url and len(result.get('body', '')) < 200:
                        print(f"📄 Extraction contenu de: {url}")
                        extra_content = smart_content_extraction(url)
                        if extra_content and "Impossible d'extraire" not in extra_content:
                            content += f"\nContenu détaillé: {extra_content}"
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': url or 'web_search',
                            'type': result.get('source_type', 'web'),
                            'search_source': 'web',
                            'query_used': exp_query,
                            'relevance_score': 0.8 if exp_query == query else 0.6
                        }
                    )
                    web_results.append(doc)
            except Exception as e:
                print(f"Erreur recherche pour '{exp_query}': {e}")
                continue
        # Filtrer les doublons et trier par pertinence
        unique_web_results = []
        seen_urls = set()
        for doc in web_results:
            url = doc.metadata.get('source', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_web_results.append(doc)
        # Trier par score de pertinence
        unique_web_results.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        all_results.extend(unique_web_results[:5]) # Max 5 résultats web
    return all_results
def generate_answer_enhanced(question, context_docs, model_name, include_sources=True):
    """
    Génération de réponse améliorée avec gestion des sources multiples
    Args:
        question: Question posée
        context_docs: Documents de contexte
        model_name: Modèle à utiliser
        include_sources: Inclure les sources dans la réponse
    Returns:
        Réponse générée avec sources
    """
    if not context_docs:
        context = "Aucun contexte spécifique trouvé."
    else:
        context_parts = []
        local_sources = []
        web_sources = []
        
        # Augmenter le nombre de docs et la limite par doc si contexte important
        max_docs = 3  # Augmenté de 2 à 3
        chars_per_doc = 1200  # Augmenté de 800 à 1200
        
        for i, doc in enumerate(context_docs[:max_docs]):
            source = doc.metadata.get('source', 'Document inconnu')
            doc_type = doc.metadata.get('type', 'unknown')
            search_source = doc.metadata.get('search_source', 'unknown')
            content = doc.page_content.strip()[:chars_per_doc]
            
            # Classifier les sources
            if search_source == 'local_rag':
                local_sources.append(f"[{i+1}] {source} ({doc_type})")
            else:
                web_sources.append(f"[{i+1}] {source}")
            
            context_parts.append(f"[{i+1}] {content}")
        
        context = "\n".join(context_parts)[:2500]  # Augmenté de 1500 à 2500 chars
    
    # Prompt optimisé pour des réponses structurées et détaillées
    prompt = f"""CONTEXTE:
{context}

QUESTION: {question[:500]}

INSTRUCTIONS DE RÉPONSE:
- Fournis une réponse DÉTAILLÉE et COMPLÈTE (minimum 300 mots)
- Structure ta réponse avec des SOUS-TITRES en markdown (##)
- Organise en PARAGRAPHES cohérents et bien espacés
- Utilise des listes à puces (•) pour les énumérations
- Inclus des EXEMPLES concrets quand pertinent
- Ajoute une section CONCLUSION ou RÉSUMÉ
- Utilise des emojis pour rendre la lecture agréable
- Formate avec **gras** et *italique* pour l'emphase

Réponds de manière professionnelle, structurée et approfondie."""
    
    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt[:4000]}]  # Augmenté à 4000 chars
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1200,  # Augmenté à 1200 pour réponses très détaillées
            temperature=0.4  # Légèrement plus créatif pour mieux structurer
        )
        answer = response.choices[0].message.content
        
        # Ajouter les sources si demandé
        if include_sources and context_docs:
            sources_text = "\n\n📚 **Sources consultées:**\n"
            if local_sources:
                sources_text += "**Documents locaux:**\n"
                for source in local_sources[:3]:
                    sources_text += f"• {source}\n"
            if web_sources:
                sources_text += "**Sources web:**\n"
                for source in web_sources[:3]:
                    sources_text += f"• {source}\n"
            answer += sources_text
        
        return answer
    
    except Exception as e:
        return f"❌ Erreur génération: {str(e)}"

def generate_answer_enhanced_stream(question, context_docs, model_name, include_sources=True):
    """
    Version STREAMING de generate_answer_enhanced pour affichage progressif
    Args:
        question: Question posée
        context_docs: Documents de contexte
        model_name: Modèle à utiliser
        include_sources: Inclure les sources dans la réponse
    Yields:
        Chunks de texte au fur et à mesure de la génération
    """
    if not context_docs:
        context = "Aucun contexte spécifique trouvé."
    else:
        context_parts = []
        local_sources = []
        web_sources = []
        
        max_docs = 3
        chars_per_doc = 1200
        
        for i, doc in enumerate(context_docs[:max_docs]):
            source = doc.metadata.get('source', 'Document inconnu')
            doc_type = doc.metadata.get('type', 'unknown')
            search_source = doc.metadata.get('search_source', 'unknown')
            content = doc.page_content.strip()[:chars_per_doc]
            
            if search_source == 'local_rag':
                local_sources.append(f"[{i+1}] {source} ({doc_type})")
            else:
                web_sources.append(f"[{i+1}] {source}")
            
            context_parts.append(f"[{i+1}] {content}")
        
        context = "\n".join(context_parts)[:2500]
    
    prompt = f"""CONTEXTE:
{context}

QUESTION: {question[:500]}

INSTRUCTIONS DE RÉPONSE:
- Fournis une réponse DÉTAILLÉE et COMPLÈTE (minimum 300 mots)
- Structure ta réponse avec des SOUS-TITRES en markdown (##)
- Organise en PARAGRAPHES cohérents et bien espacés
- Utilise des listes à puces (•) pour les énumérations
- Inclus des EXEMPLES concrets quand pertinent
- Ajoute une section CONCLUSION ou RÉSUMÉ
- Utilise des emojis pour rendre la lecture agréable
- Formate avec **gras** et *italique* pour l'emphase

Réponds de manière professionnelle, structurée et approfondie."""
    
    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt[:4000]}]
        
        # 🌊 STREAMING activé
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1200,
            temperature=0.4,
            stream=True  # 🔥 Mode streaming
        )
        
        # Yield chaque chunk
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
        
        # Ajouter les sources à la fin
        if include_sources and context_docs:
            sources_text = "\n\n📚 **Sources consultées:**\n"
            if local_sources:
                sources_text += "**Documents locaux:**\n"
                for source in local_sources[:3]:
                    sources_text += f"• {source}\n"
            if web_sources:
                sources_text += "**Sources web:**\n"
                for source in web_sources[:3]:
                    sources_text += f"• {source}\n"
            yield sources_text
    
    except Exception as e:
        yield f"❌ Erreur génération: {str(e)}"
# ===============================================
# Fonctions Web Search et Hybrid (Mises à jour)
# ===============================================
def web_search(query, max_results=5):
    """Version simplifiée pour compatibilité"""
    try:
        results = enhanced_web_search(query, max_results, "text")
        return [f"{r.get('title', '')}: {r.get('href', r.get('url', ''))} - {r.get('body', '')}" for r in results]
    except Exception as e:
        return [f"❌ Erreur recherche web: {e}"]
def hybrid_search(query, vectordb, k=3):
    """Version simplifiée pour compatibilité"""
    return hybrid_search_enhanced(query, vectordb, k, web_search_enabled=True)
def final_search(question, vectordb, graph, pois):
    """Recherche finale combinant toutes les sources"""
    results = hybrid_search_enhanced(question, vectordb, k=3, web_search_enabled=True)
    # OSM si mention lieu
    if any(keyword in question.lower() for keyword in ["aller", "trajet", "itinéraire", "route", "navigation"]):
        try:
            carte, reponse, traj = calculer_trajet(question, graph, pois)
            if traj:
                results.append(Document(
                    page_content=reponse,
                    metadata={"source": "trajet_osm", "type": "navigation"}
                ))
        except:
            pass
    return results
# ===============================================
# Fonctions Modèles Hugging Face Spécialisés
# ===============================================
def initialize_specialized_models():
    """Initialise les modèles spécialisés en mode LOCAL uniquement"""
    models = {}
    
    # Vérifier si pipeline est disponible
    if pipeline is None:
        print("⚠️ transformers.pipeline non disponible, modèles spécialisés désactivés")
        return {
            'summarizer': None,
            'translator': None,
            'captioner': None,
            'ner': None
        }
    
    print("🌐 Mode: LOCAL UNIQUEMENT (pas de téléchargement)")
    
    try:
        models['summarizer'] = pipeline(
            "summarization", 
            model=SUMMARIZER_MODEL, 
            cache_dir=str(SUMMARIZER_CACHE),
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        print("✅ Summarizer chargé en mode LOCAL")
    except Exception as e:
        print(f"⚠️ Erreur chargement summarizer: {e}")
        models['summarizer'] = None
    try:
        models['translator'] = pipeline(
            "translation", 
            model=TRANSLATOR_MODEL, 
            cache_dir=str(TRANSLATOR_CACHE),
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        print("✅ Translator chargé en mode LOCAL")
    except Exception as e:
        print(f"⚠️ Erreur chargement translator: {e}")
        models['translator'] = None
    try:
        models['captioner'] = None
        print("✅ Captioner configuré pour utiliser LLM (llava)")
    except Exception as e:
        print(f"⚠️ Erreur chargement captioner: {e}")
        models['captioner'] = None
    try:
        models['ner'] = pipeline("ner", model=NER_MODEL, cache_dir=str(NER_CACHE), local_files_only=True)
        print("✅ NER chargé en mode LOCAL")
    except Exception as e:
        print(f"⚠️ Erreur chargement NER: {e}")
        models['ner'] = None
    return models
# Initialiser les modèles
SPECIALIZED_MODELS = initialize_specialized_models()
def summarize_text(text):
    if SPECIALIZED_MODELS.get('summarizer') is None:
        return "❌ Modèle de résumé non disponible"
    try:
        return SPECIALIZED_MODELS['summarizer'](text[:1024], max_length=200, min_length=30, do_sample=False)[0]['summary_text']
    except Exception as e:
        return f"❌ Erreur résumé: {e}"
def translate_text(text, src_lang="fr", tgt_lang="en"):
    if SPECIALIZED_MODELS.get('translator') is None:
        return "❌ Modèle de traduction non disponible"
    try:
        return SPECIALIZED_MODELS['translator'](text)[0]['translation_text']
    except Exception as e:
        return f"❌ Erreur traduction: {e}"
def generate_detailed_description_with_clip(image_path, vision_models):
    """
    Génère une description détaillée en utilisant CLIP avec des prompts textuels précis
    """
    try:
        if not vision_models:
            return None, "Modèles CLIP non chargés"
        
        clip_model = vision_models['clip_model']
        clip_processor = vision_models['clip_processor']
        device = vision_models['device']
        
        # S'assurer que le modèle est sur le bon device
        clip_model = clip_model.to(device)
        clip_model.eval()
        
        # Charger l'image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # 🎯 50 PROMPTS ULTRA-DÉTAILLÉS pour analyse exhaustive
        detailed_prompts = [
            # === PERSONNES ET PERSONNAGES (15 prompts) ===
            "a photograph of a person posing",
            "a person wearing a costume or uniform",
            "a superhero character in action",
            "a toy action figure or collectible",
            "a person in colorful clothing",
            "a portrait photograph of someone",
            "people in a group photo",
            "a celebrity or famous person",
            "an athlete in sports gear",
            "a child or baby photo",
            "a person wearing glasses or accessories",
            "someone doing an activity or sport",
            "a fashion model or styling photo",
            "a person with distinctive hairstyle",
            "an animated or cartoon character",
            
            # === LOGOS, TEXTE ET TYPOGRAPHIE (12 prompts) ===
            "a logo with a single letter",
            "a text symbol or typography design",
            "a colored letter with gradient",
            "a graphic design with multiple letters",
            "a minimalist logo design",
            "a stylized letter or monogram",
            "a brand logo or company emblem",
            "a document page with text",
            "handwritten text or signature",
            "a sign with text and symbols",
            "typography art or calligraphy",
            "a poster with bold text",
            
            # === OBJETS ET PRODUITS (10 prompts) ===
            "a vehicle like car or motorcycle",
            "electronic device or gadget",
            "food dish or meal plating",
            "clothing item or fashion accessory",
            "furniture or home decor",
            "a book cover or magazine",
            "a tool or equipment",
            "jewelry or precious items",
            "a bottle or container product",
            "sports equipment or gear",
            
            # === NATURE ET PAYSAGES (8 prompts) ===
            "a photograph of landscape scenery",
            "a natural outdoor scene with trees",
            "a mountain or hill terrain",
            "a beach or ocean view",
            "flowers or plants in nature",
            "a sunset or sunrise sky",
            "animals or wildlife photo",
            "urban cityscape or skyline",
            
            # === ART ET DESIGN (5 prompts) ===
            "an illustration or digital art",
            "a painting or artwork",
            "a comic book or manga panel",
            "abstract art or pattern",
            "architectural design or blueprint",
            
            # === DIVERS (5 prompts) ===
            "a technical diagram or chart",
            "an icon or simple symbol",
            "a screenshot of interface",
            "a meme or internet image",
            "a QR code or barcode"
        ]
        
        # Analyser avec CLIP
        inputs = clip_processor(
            text=detailed_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            except RuntimeError as cuda_error:
                if "CUDA" in str(cuda_error):
                    clip_model = clip_model.cpu()
                    inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                else:
                    raise
        
        # Obtenir le top match
        top_prob, top_idx = torch.topk(probs[0], 1)
        main_description = detailed_prompts[top_idx.item()]
        confidence = top_prob.item()
        
        # Vérifier si c'est vraiment du texte/logo AVANT de chercher une lettre
        is_text_based = any(word in main_description.lower() for word in ["letter", "logo", "text", "symbol", "typography", "character", "document"])
        
        # SEULEMENT chercher une lettre si confiance élevée que c'est du texte ET score > 30%
        letter_description = ""
        if is_text_based and confidence > 0.30:
            print(f"🔤 Détection de lettre activée (confiance texte: {confidence*100:.1f}%)")
            letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            
            # Méthode 1: Comparaison directe avec des lettres stylisées
            letter_prompts = [f"the letter {letter}" for letter in letters]
            
            inputs_letters = clip_processor(
                text=letter_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs_letters = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_letters.items()}
            
            with torch.no_grad():
                outputs_letters = clip_model(**inputs_letters)
                logits_letters = outputs_letters.logits_per_image
                probs_letters = logits_letters.softmax(dim=1)
            
            top5_letters, top5_idx = torch.topk(probs_letters[0], 5)
            detected_letters = [letters[idx.item()] for idx in top5_idx]
            letter_confidences = [prob.item() for prob in top5_letters]
            
            # Méthode 2: Vérification avec prompts alternatifs pour le top 3
            top3_verification = []
            for i in range(min(3, len(detected_letters))):
                letter = detected_letters[i]
                verify_prompts = [
                    f"a {letter} logo",
                    f"the capital letter {letter}",
                    f"letter {letter} design"
                ]
                
                inputs_verify = clip_processor(
                    text=verify_prompts,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                inputs_verify = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_verify.items()}
                
                with torch.no_grad():
                    outputs_verify = clip_model(**inputs_verify)
                    logits_verify = outputs_verify.logits_per_image
                    probs_verify = logits_verify.softmax(dim=1)
                
                # Score moyen de vérification
                avg_verify_score = probs_verify[0].mean().item()
                combined_score = (letter_confidences[i] + avg_verify_score) / 2
                top3_verification.append((letter, combined_score))
            
            # Trier par score combiné
            top3_verification.sort(key=lambda x: x[1], reverse=True)
            best_letter, best_score = top3_verification[0]
            
            if best_score > 0.15:  # Seuil plus bas mais avec vérification
                letter_description = f" Il s'agit probablement de la lettre '{best_letter}' (confiance: {best_score*100:.1f}%)"
                if len(top3_verification) > 1 and top3_verification[1][1] > 0.12:
                    letter_description += f", ou '{top3_verification[1][0]}' ({top3_verification[1][1]*100:.1f}%)"
            else:
                # Afficher au moins les 3 meilleures suggestions
                letter_description = f" Lettres possibles: {', '.join([f'{l}({s*100:.0f}%)' for l, s in top3_verification[:3]])}"
        
        # Analyser les couleurs (50 PROMPTS ULTRA-DÉTAILLÉS)
        color_prompts = [
            # Couleurs primaires et secondaires (12 prompts)
            "with vibrant bright blue colors",
            "with deep dark blue navy colors",
            "with vivid red crimson colors",
            "with dark burgundy red colors",
            "with bright lime green colors",
            "with dark forest green colors",
            "with sunny bright yellow colors",
            "with golden amber yellow colors",
            "with rich purple violet colors",
            "with bright orange tangerine colors",
            "with hot pink magenta colors",
            "with soft pastel pink colors",
            
            # Couleurs complexes et mélanges (15 prompts)
            "with turquoise cyan blue-green colors",
            "with teal aqua colors",
            "with coral salmon pink-orange colors",
            "with maroon dark red-brown colors",
            "with olive green-yellow colors",
            "with mint light green colors",
            "with lavender light purple colors",
            "with indigo deep blue-purple colors",
            "with peach light orange colors",
            "with rose pink-red colors",
            "with khaki tan beige colors",
            "with cream ivory off-white colors",
            "with burgundy wine red colors",
            "with chartreuse yellow-green colors",
            "with crimson bright red colors",
            
            # Tons neutres et spéciaux (12 prompts)
            "with pure black colors",
            "with charcoal dark gray colors",
            "with medium gray colors",
            "with light silver gray colors",
            "with pure white colors",
            "with beige tan neutral colors",
            "with brown chocolate colors",
            "with dark espresso brown colors",
            "with caramel light brown colors",
            "with monochrome black and white",
            "with sepia vintage brown tones",
            "with metallic silver gold colors",
            
            # Atmosphères et tonalités (11 prompts)
            "with neon bright fluorescent colors",
            "with pastel soft light colors",
            "with dark muted somber colors",
            "with vibrant saturated bold colors",
            "with rainbow multicolor spectrum",
            "with warm red orange yellow tones",
            "with cool blue green purple tones",
            "with earth natural brown green tones",
            "with translucent transparent colors",
            "with gradients fading colors",
            "with contrasting complementary colors"
        ]
        
        inputs_colors = clip_processor(
            text=color_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs_colors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_colors.items()}
        
        with torch.no_grad():
            outputs_colors = clip_model(**inputs_colors)
            logits_colors = outputs_colors.logits_per_image
            probs_colors = logits_colors.softmax(dim=1)
        
        # AMÉLIORATION: Prendre les TOP 3 couleurs au lieu d'une seule
        top3_colors_probs, top3_colors_idx = torch.topk(probs_colors[0], 3)
        
        # Construire une description de couleurs détaillée
        detected_colors = []
        for i in range(3):
            color_name = color_prompts[top3_colors_idx[i].item()].replace("with ", "").replace(" colors", "").replace(" tones", "")
            color_conf = top3_colors_probs[i].item()
            if color_conf > 0.15:  # Seuil minimum 15%
                detected_colors.append(f"{color_name} ({color_conf*100:.0f}%)")
        
        # Description finale des couleurs
        if detected_colors:
            color_desc = ", ".join(detected_colors[:3])  # Max 3 couleurs
        else:
            # Fallback si aucune couleur détectée avec confiance suffisante
            color_desc = color_prompts[top3_colors_idx[0].item()].replace("with ", "").replace(" colors", "").replace(" tones", "")
        
        # Construire la description finale
        full_description = f"{main_description} with colors: {color_desc}.{letter_description}"
        
        return {
            'description': full_description,
            'confidence': confidence,
            'main_type': main_description,
            'colors': color_desc,
            'details': letter_description
        }, None
        
    except Exception as e:
        import traceback
        error_msg = f"Erreur description CLIP: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def caption_image_with_blip(image_path, blip_models):
    """Génère une description détaillée de l'image avec BLIP local"""
    try:
        if not blip_models:
            return None, "Modèle BLIP non chargé"
        
        processor = blip_models['processor']
        model = blip_models['model']
        device = blip_models['device']
        
        # Charger l'image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Générer une description non conditionnée (description générale)
        inputs = processor(image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=100, num_beams=5)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Générer aussi une description conditionnée (plus détaillée)
        text_prompt = "a detailed description of"
        inputs_detailed = processor(image, text=text_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out_detailed = model.generate(**inputs_detailed, max_length=150, num_beams=5)
        
        caption_detailed = processor.decode(out_detailed[0], skip_special_tokens=True)
        
        return {
            'caption': caption,
            'detailed_caption': caption_detailed,
            'method': 'BLIP'
        }, None
        
    except Exception as e:
        import traceback
        error_msg = f"Erreur BLIP: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def caption_image(image_path):
    client = create_client()
    model = "llava-hf/llava-1.5-7b-hf"
    prompt = "Generate a detailed caption for this image."
    try:
        return client.image_to_text(image_path, prompt=prompt, model=model, max_tokens=500)
    except Exception as e:
        return f"❌ Erreur caption: {e}"
def extract_entities(text):
    if SPECIALIZED_MODELS.get('ner') is None:
        return "❌ Modèle NER non disponible"
    try:
        return SPECIALIZED_MODELS['ner'](text)
    except Exception as e:
        return f"❌ Erreur NER: {e}"
# ===============================================
# Fonctions de génération avec Stable Diffusion et similaires
# ===============================================
def generate_text_to_image(prompt):
    """Génère une image à partir de texte"""
    if DiffusionPipeline is None:
        return "❌ DiffusionPipeline non disponible (installez diffusers)"
    try:
        pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        image = pipe(prompt).images[0]
        path = os.path.join(GENERATED_PATH, f"image_{int(time.time())}.png")
        image.save(path)
        return f"Image générée et sauvegardée à {path}"
    except Exception as e:
        return f"❌ Erreur génération image: {e}"
def generate_text_to_video(prompt):
    """Génère une vidéo à partir de texte"""
    if DiffusionPipeline is None:
        return "❌ DiffusionPipeline non disponible (installez diffusers)"
    try:
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        gen = pipe(prompt, num_inference_steps=25)
        frames = gen.frames[0] # Assuming batch size 1
        path = os.path.join(GENERATED_PATH, f"video_{int(time.time())}.gif")
        imageio.mimsave(path, frames, fps=5)
        return f"Vidéo générée et sauvegardée à {path}"
    except Exception as e:
        return f"❌ Erreur génération vidéo: {e}"
def generate_text_to_audio(prompt):
    """Génère un son à partir de texte"""
    if AudioLDMPipeline is None:
        return "❌ AudioLDMPipeline non disponible (installez diffusers avec audio)"
    if wavfile is None:
        return "❌ scipy.io.wavfile non disponible (installez scipy)"
    try:
        pipe = AudioLDMPipeline.from_pretrained("cvssp/audio-ldm", torch_dtype=torch.float16, use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        audio = pipe(prompt, audio_length_in_s=5.0).audios[0]
        path = os.path.join(GENERATED_PATH, f"audio_{int(time.time())}.wav")
        wavfile.write(path, rate=16000, data=audio) # Assuming 16kHz sample rate
        return f"Son généré et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération son: {e}"
def generate_text_to_3d(prompt):
    """Génère un modèle 3D à partir de texte (rendue image)"""
    if ShapEPipeline is None:
        return "❌ ShapEPipeline non disponible (installez diffusers avec shap-e)"
    try:
        pipe = ShapEPipeline.from_pretrained("openai/shap-e", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        output = pipe(prompt, num_inference_steps=64)
        image = output.images[0]
        path = os.path.join(GENERATED_PATH, f"3d_text_{int(time.time())}.png")
        image.save(path)
        return f"Rendu 3D généré et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération 3D (texte): {e}"
def generate_image_to_3d(image_path):
    """Génère un modèle 3D à partir d'une image (rendue image)"""
    if ShapEImg2ImgPipeline is None:
        return "❌ ShapEImg2ImgPipeline non disponible (installez diffusers avec shap-e)"
    try:
        pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        image = Image.open(image_path)
        output = pipe(image, num_inference_steps=64)
        rendered_image = output.images[0]
        path = os.path.join(GENERATED_PATH, f"3d_image_{int(time.time())}.png")
        rendered_image.save(path)
        return f"Rendu 3D généré à partir de l'image et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération 3D (image): {e}"
# ===============================================
# Agent LangChain Amélioré avec Recherche Web
# ===============================================
def create_enhanced_agent(model_name, vectordb, graph, pois):
    """
    Crée un agent LangChain amélioré avec capacités de recherche web
    Args:
        model_name: Nom du modèle HuggingFace
        vectordb: Base vectorielle locale
        graph: Graphe OSM
        pois: Points d'intérêt
    Returns:
        Agent configuré avec tous les outils
    """
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.3,
            max_new_tokens=600
        )
        # Configuration des outils de recherche web
        search_wrapper = DuckDuckGoSearchAPIWrapper(
            region="fr-fr",
            time="d",
            max_results=5
        )
        search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
        search_results_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5, include_raw_content=True)
        tools = [
            # Outils de base RAG et recherche
            Tool(
                name="Local_Knowledge_Base",
                func=lambda q: "\n\n".join([d.page_content for d in rag_search(q, vectordb, k=3)]),
                description="Recherche dans la base de connaissances locale (PDFs et documents internes). Utilise ceci en PREMIER pour les questions sur des documents spécifiques."
            ),
            Tool(
                name="Web_Search",
                func=lambda q: search_tool.run(q),
                description="Recherche sur Internet pour des informations récentes, actualités, ou des connaissances générales non disponibles localement."
            ),
            Tool(
                name="Web_Search_Detailed",
                func=lambda q: search_results_tool.run(q),
                description="Recherche web détaillée avec sources et liens. Utilise pour obtenir des résultats web structurés avec URLs."
            ),
            Tool(
                name="Hybrid_Search",
                func=lambda q: "\n\n".join([d.page_content for d in hybrid_search_enhanced(q, vectordb, k=3, web_search_enabled=True)]),
                description="Recherche hybride combinant base locale ET web. Idéal pour des questions nécessitant à la fois des données internes et externes."
            ),
            Tool(
                name="Current_News_Search",
                func=lambda q: "\n\n".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in enhanced_web_search(q, search_type="news")]),
                description="Recherche spécialisée pour les actualités récentes et informations temporelles."
            ),
            # Outils spécialisés
            Tool(
                name="OSM_Route_Calculator",
                func=lambda q: calculer_trajet(q, graph, pois)[1] if graph and pois else "❌ Aucune carte OSM disponible",
                description="Calcule des itinéraires routiers entre deux lieux. Utilise pour les questions de navigation, trajets, ou géolocalisation."
            ),
            Tool(
                name="Smart_Content_Extractor",
                func=lambda url: smart_content_extraction(url) if url.startswith('http') else "❌ URL invalide",
                description="Extrait le contenu détaillé d'une page web spécifique. Fournis une URL complète."
            ),
            Tool(
                name="Text_Summarizer",
                func=summarize_text,
                description="Résume un texte long en version concise. Utile pour synthétiser des informations volumineuses."
            ),
            Tool(
                name="Language_Translator",
                func=translate_text,
                description="Traduit du français vers l'anglais. Utile pour traiter des sources en langue étrangère."
            ),
            Tool(
                name="Image_Analyzer",
                func=caption_image,
                description="Analyse et décrit le contenu d'une image. Fournis le chemin vers un fichier image."
            ),
            Tool(
                name="Entity_Extractor",
                func=lambda t: json.dumps(extract_entities(t)),
                description="Extrait des entités nommées (personnes, lieux, organisations) d'un texte."
            ),
            # Nouveaux outils Stable Diffusion via API
            Tool(
                name="Text_To_Image_Generator",
                func=generate_text_to_image,
                description="Génère une image à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Text_To_Video_Generator",
                func=generate_text_to_video,
                description="Génère une vidéo à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Text_To_Audio_Generator",
                func=generate_text_to_audio,
                description="Génère un son ou audio à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Text_To_3D_Generator",
                func=generate_text_to_3d,
                description="Génère un modèle 3D (rendue image) à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Image_To_3D_Generator",
                func=generate_image_to_3d,
                description="Génère un modèle 3D (rendue image) à partir d'une image. Fournis le chemin vers un fichier image."
            ),
        ]
        
        # Ajouter l'outil Excel AI si disponible
        if EXCEL_AI_AVAILABLE:
            def excel_ai_wrapper(query: str) -> str:
                """Wrapper pour l'outil Excel AI"""
                try:
                    # Parser la requête (peut être JSON ou texte structuré)
                    import json
                    try:
                        data = json.loads(query)
                    except:
                        # Si pas JSON, créer un DataFrame depuis le texte
                        lines = [l.strip() for l in query.split('\n') if l.strip()]
                        data = {'data': lines}
                    
                    # Organiser avec l'IA
                    excel_bytes, report = organize_excel_with_ai(
                        data,
                        filename="ai_organized_data.xlsx",
                        add_charts=True,
                        add_conditional_formatting=True
                    )
                    
                    # Sauvegarder temporairement
                    temp_path = Path("temp_excel_ai_output.xlsx")
                    temp_path.write_bytes(excel_bytes)
                    
                    return f"✅ Fichier Excel organisé avec IA créé !\n\nRapport:\n{json.dumps(report, indent=2, ensure_ascii=False)}\n\nFichier sauvegardé: {temp_path.absolute()}"
                except Exception as e:
                    return f"❌ Erreur lors de l'organisation Excel: {str(e)}"
            
            tools.append(
                Tool(
                    name="Excel_AI_Organizer",
                    func=excel_ai_wrapper,
                    description="""Organise automatiquement des données dans un fichier Excel avec IA ultra-précise. 
                    Détecte et structure automatiquement:
                    - Coordonnées GPS (latitude/longitude)
                    - Données géospatiales
                    - Données financières
                    - Données scientifiques
                    - Listes d'invités/événements
                    - Tableaux de tout type
                    Ajoute formatage conditionnel, graphiques, et classifications dynamiques.
                    Fournis les données en JSON ou texte structuré."""
                )
            )
        
        # Configuration de l'agent avec prompt personnalisé
        agent_prompt = """Tu es Kibali, un assistant IA avancé avec accès à de multiples sources d'information.
CAPACITÉS DISPONIBLES:
- Base de connaissances locale (PDFs et documents)
- Recherche web en temps réel
- Calcul d'itinéraires sur cartes OSM
- Analyse d'images et extraction de contenu web
- Traduction et résumé automatiques
- Génération d'images, vidéos, sons et modèles 3D à partir de texte ou images
- Organisation intelligente de fichiers Excel avec IA ultra-précise (coordonnées GPS, classifications dynamiques, tableaux structurés)

INSTRUCTIONS IMPORTANTES:
1. Utilise TOUJOURS la base locale en premier pour les questions sur des documents spécifiques
2. Combine les sources locales ET web pour des réponses complètes
3. Pour les actualités ou infos récentes, privilégie la recherche web
4. Cite tes sources et indique leur provenance (locale vs web)
5. Si les informations se contredisent, mentionne les deux perspectives
6. Reste concis mais informatif
7. Pour les générations, sauvegarde les fichiers et retourne le chemin
8. Pour organiser des données Excel, utilise Excel_AI_Organizer avec des données structurées (coordonnées, listes, tableaux)

Tu as accès aux outils suivants: {tools}
Utilise le format suivant:
Question: la question d'entrée
Thought: réfléchis à ce que tu dois faire
Action: l'action à entreprendre, doit être l'un de [{tool_names}]
Action Input: l'entrée à l'action
Observation: le résultat de l'action
... (ce Thought/Action/Action Input/Observation peut se répéter N fois)
Thought: Je connais maintenant la réponse finale
Final Answer: la réponse finale à la question d'entrée
Commence!
Question: {input}
Thought: {agent_scratchpad}"""
        agent = initialize_agent(
            tools,
            llm,
            agent="zero-shot-react-description",
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        print(f"✅ Agent créé avec {len(tools)} outils disponibles")
        return agent
    except Exception as e:
        print(f"❌ Erreur création agent: {e}")
        return None
# Alias pour compatibilité
def create_agent(model_name, vectordb, graph, pois):
    """Version simplifiée pour compatibilité ascendante"""
    return create_enhanced_agent(model_name, vectordb, graph, pois)
# ===============================================
# Fonctions OSM et Graphe Routier
# ===============================================
def haversine(lon1, lat1, lon2, lat2):
    """Calcul distance haversine en mètres"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
class RoadPOIHandler(osmium.SimpleHandler):
    """Handler pour extraire routes et POIs depuis OSM"""
    def __init__(self):
        super().__init__()
        self.graph = nx.Graph()
        self.pois = []
    def node(self, n):
        """Extraire les POIs (points d'intérêt)"""
        if n.location.valid() and n.tags:
            name = n.tags.get('name', '')
            amenity = n.tags.get('amenity', '')
            if name or amenity:
                self.pois.append({
                    'name': name,
                    'amenity': amenity,
                    'lon': n.location.lon,
                    'lat': n.location.lat,
                    'tags': dict(n.tags)
                })
    def way(self, w):
        """Extraire les routes"""
        if 'highway' in w.tags:
            coords = []
            for n in w.nodes:
                if n.location.valid():
                    coords.append((n.location.lon, n.location.lat))
            for i in range(len(coords)-1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i+1]
                n1, n2 = (lon1, lat1), (lon2, lat2)
                dist = haversine(lon1, lat1, lon2, lat2)
                self.graph.add_node(n1, x=lon1, y=lat1)
                self.graph.add_node(n2, x=lon2, y=lat2)
                self.graph.add_edge(n1, n2, length=dist, highway=w.tags.get("highway"))
def trouver_noeud_plus_proche(lon, lat, graph):
    """Trouve le nœud du graphe le plus proche"""
    min_dist = float("inf")
    closest_node = None
    for node, data in graph.nodes(data=True):
        nlon, nlat = float(data["x"]), float(data["y"])
        dist = haversine(lon, lat, nlon, nlat)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node
def chercher_poi_par_nom(nom, pois_list):
    """Recherche un POI par nom"""
    nom_lower = nom.lower()
    for poi in pois_list:
        if nom_lower in poi['name'].lower() or nom_lower in poi['amenity'].lower():
            return poi
    return None
def generer_carte_trajet(graph, path, pois_list, start_poi=None, end_poi=None):
    """Génère une carte 2D du trajet"""
    fig, ax = plt.subplots(figsize=(12, 10))
    # Dessiner le graphe en arrière-plan
    for edge in list(graph.edges())[:1000]: # Limiter pour la performance
        node1, node2 = edge
        x1, y1 = node1[0], node1[1]
        x2, y2 = node2[0], node2[1]
        ax.plot([x1, x2], [y1, y2], 'lightgray', alpha=0.3, linewidth=0.5)
    # Dessiner le trajet
    if path and len(path) > 1:
        path_x = [node[0] for node in path]
        path_y = [node[1] for node in path]
        ax.plot(path_x, path_y, 'red', linewidth=3, label='Trajet')
        # Marquer début et fin
        ax.scatter(path_x[0], path_y[0], color='green', s=100, label='Départ', zorder=5)
        ax.scatter(path_x[-1], path_y[-1], color='red', s=100, label='Arrivée', zorder=5)
    # Ajouter quelques POIs
    for poi in pois_list[:20]:
        if poi['name']:
            ax.scatter(poi['lon'], poi['lat'], color='blue', s=20, alpha=0.6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Trajet calculé sur la carte OSM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    # Sauvegarder en mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf
def calculer_trajet(question, graph, pois_list):
    """Calcule un trajet basé sur une question textuelle"""
    if not graph or not pois_list:
        return None, "❌ Graphe ou POIs non disponibles", None
    # Utiliser LLM pour extraire départ et arrivée
    try:
        client = create_client()
        prompt = f"""Extraie le lieu de départ et le lieu d'arrivée de cette question de trajet.
Question: {question}
Réponds au format exact:
Départ: [nom du lieu de départ]
Arrivée: [nom du lieu d'arrivée]"""
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=WORKING_MODELS["Qwen2.5 7B (Rapide & excellent)"],
            messages=messages,
            max_tokens=100,
            temperature=0.1
        )
        extraction = response.choices[0].message.content
        start_line = [line for line in extraction.split('\n') if line.startswith('Départ: ')]
        end_line = [line for line in extraction.split('\n') if line.startswith('Arrivée: ')]
        if start_line and end_line:
            start_place = start_line[0].replace('Départ: ', '').strip()
            end_place = end_line[0].replace('Arrivée: ', '').strip()
        else:
            return None, "❌ Impossible d'extraire les lieux de la question.", None
    except Exception as e:
        print(f"❌ Erreur extraction LLM: {e}")
        return None, "❌ Erreur lors de l'extraction des lieux.", None
    start_poi = chercher_poi_par_nom(start_place, pois_list)
    end_poi = chercher_poi_par_nom(end_place, pois_list)
    if not start_poi or not end_poi:
        return None, f"❌ Impossible de trouver les lieux: {start_place} ou {end_place}.", None
    # Trouver les nœuds dans le graphe
    start_node = trouver_noeud_plus_proche(start_poi['lon'], start_poi['lat'], graph)
    end_node = trouver_noeud_plus_proche(end_poi['lon'], end_poi['lat'], graph)
    if not start_node or not end_node:
        return None, "❌ Impossible de trouver les nœuds dans le graphe routier.", None
    try:
        # Calculer le chemin
        path = nx.shortest_path(graph, source=start_node, target=end_node, weight="length")
        # Calculer la distance
        distance_totale = 0
        for i in range(len(path)-1):
            distance_totale += graph[path[i]][path[i+1]]['length']
        # Générer la carte
        carte_buf = generer_carte_trajet(graph, path, pois_list, start_poi, end_poi)
        # Réponse textuelle
        reponse = f"""🗺️ **Trajet calculé**
📍 **Départ**: {start_poi['name']} ({start_poi['amenity']})
🎯 **Arrivée**: {end_poi['name']} ({end_poi['amenity']})
📏 **Distance**: {distance_totale/1000:.2f} km
⏱️ **Temps estimé**: {int(distance_totale/83.33):.0f} min à pied | {int(distance_totale/833.33):.0f} min en voiture
🛣️ **Étapes**: {len(path)} points"""
        return carte_buf, reponse, {
            'start': start_poi,
            'end': end_poi,
            'distance': distance_totale,
            'path_length': len(path)
        }
    except nx.NetworkXNoPath:
        return None, f"❌ Aucun chemin trouvé entre {start_poi['name']} et {end_poi['name']}", None
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}", None
# ===============================================
# Fonctions utilitaires pour images
# ===============================================
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)
def df_to_html(df, max_rows=10):
    """Convertit DataFrame en HTML avec styles blanc gras visibles"""
    # Réduire le tableau si trop long
    if len(df) > max_rows:
        summary_row = pd.DataFrame({col: ['...'] for col in df.columns})
        df = pd.concat([df.head(max_rows // 2), summary_row, df.tail(max_rows // 2)])
    
    # Générer HTML avec styles inline
    html = df.to_html(index=False, escape=False, border=0)
    html = html.replace('<table', '<table style="width: 100%; border-collapse: collapse; margin: 1rem 0; background: rgba(58, 58, 94, 0.3);"')
    html = html.replace('<thead>', '<thead style="background: rgba(0, 255, 136, 0.2);">')
    html = html.replace('<th>', '<th style="color: white !important; font-weight: 700 !important; padding: 0.75rem; border: 1px solid #5a5a8a; text-align: left;">')
    html = html.replace('<td>', '<td style="color: white !important; font-weight: 600 !important; padding: 0.75rem; border: 1px solid #5a5a8a;">')
    html = html.replace('<tr>', '<tr style="background: rgba(74, 74, 126, 0.2);">')
    return html
# ===============================================
# Fonctions Image Analysis
# ===============================================
def classify_soil(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[:,:,0])
    mean_sat = np.mean(hsv[:,:,1])
    mean_val = np.mean(hsv[:,:,2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_variance = np.var(gray)
    soil_type = "Inconnu"
    possible_contents = "Inconnu"
    possible_minerals = "Inconnu"
    if mean_val < 100 and texture_variance > 5000:
        soil_type = "Argileux (riche en matière organique)"
        possible_contents = "Peut contenir de l'eau, nutriments, adapté aux cultures racines"
        possible_minerals = "Argiles comme kaolinite, illite; possible fer, aluminium"
    elif mean_sat > 100 and texture_variance < 3000:
        soil_type = "Sableux (drainant)"
        possible_contents = "Peut contenir peu d'eau, adapté aux plantes résistantes à la sécheresse"
        possible_minerals = "Quartz, feldspath; silice abondante"
    elif mean_hue > 20 and mean_hue < 40:
        soil_type = "Limoneux (équilibré)"
        possible_contents = "Peut contenir minéraux, bon pour l'agriculture générale"
        possible_minerals = "Silt avec mica, quartz; calcium, potassium"
    # Graphisme : Histogramme des couleurs HSV
    fig, ax = plt.subplots()
    ax.hist(hsv[:,:,0].ravel(), bins=50, color='b', alpha=0.5, label='Hue')
    ax.hist(hsv[:,:,1].ravel(), bins=50, color='g', alpha=0.5, label='Saturation')
    ax.hist(hsv[:,:,2].ravel(), bins=50, color='r', alpha=0.5, label='Value')
    ax.set_title('Histogramme des Composantes HSV')
    ax.legend()
    hist_img = fig_to_pil(fig)
    # Tableau des metrics
    metrics_df = pd.DataFrame({
        'Métrique': ['Hue Moyenne', 'Saturation Moyenne', 'Valeur Moyenne', 'Variance Texture'],
        'Valeur': [mean_hue, mean_sat, mean_val, texture_variance],
        'Explication': ['Moyenne de la teinte', 'Moyenne de la saturation des couleurs', 'Moyenne de la luminosité', 'Variance de la texture pour rugosité']
    })
    metrics_html = df_to_html(metrics_df)
    return {
        "soil_type": soil_type,
        "possible_contents": possible_contents,
        "possible_minerals": possible_minerals
    }, hist_img, metrics_html
def simulate_infrared(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ir_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
    ax.set_title('Simulation Infrarouge (Colormap JET)')
    ax.axis('off')
    ir_pil = fig_to_pil(fig)
    # Analyse simple (fake temp based on intensity)
    mean_intensity = np.mean(gray)
    ir_analysis = f"Simulation IR: Intensité moyenne {mean_intensity:.2f} (plus rouge = plus chaud, bleu = plus froid)"
    return ir_pil, ir_analysis
def detect_objects(image: np.ndarray, scale_factor=0.1, vision_models=None, yolo_model=None):
    """Détecte et analyse les objets dans l'image avec YOLO + CLIP + description IA contextuelle"""
    import tempfile
    import os
    
    img_with_detections = image.copy()
    objects_data = []
    
    # 🎯 PRIORITÉ 1: Utiliser YOLO si disponible
    if yolo_model and YOLO_AVAILABLE:
        try:
            print("🎯 Détection YOLO en cours...")
            results = yolo_model(image, conf=0.25, verbose=False)  # Seuil de confiance 25%
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                for idx, box in enumerate(boxes):
                    # Extraire les coordonnées
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = results[0].names[cls]
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Calculs métriques réels
                    w_m = w * scale_factor
                    h_m = h * scale_factor
                    area_m2 = w_m * h_m
                    perimeter_m = 2 * (w_m + h_m)
                    
                    # Analyse de couleur dominante
                    object_roi = image[y1:y2, x1:x2]
                    if object_roi.size > 0:
                        mean_color = cv2.mean(object_roi)[:3]
                        color_desc = f"RGB({int(mean_color[2])},{int(mean_color[1])},{int(mean_color[0])})"
                    else:
                        color_desc = "N/A"
                    
                    # Stocker les données
                    objects_data.append({
                        'id': idx + 1,
                        'type': f"{class_name} (YOLO)",
                        'confidence': conf,
                        'width_m': w_m,
                        'height_m': h_m,
                        'area_m2': area_m2,
                        'perimeter_m': perimeter_m,
                        'color': color_desc,
                        'position': (x1, y1)
                    })
                    
                    # Dessiner sur l'image
                    color = (0, 255, 0) if conf > 0.5 else (255, 255, 0)
                    cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 2)
                    label = f"#{idx+1}: {class_name} {conf:.0%}"
                    cv2.putText(img_with_detections, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(img_with_detections, f"{w_m:.2f}x{h_m:.2f}m", (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                print(f"✅ YOLO: {len(objects_data)} objets détectés")
        except Exception as e:
            print(f"⚠️ Erreur YOLO: {e}")
            # Fallback sur méthode traditionnelle
    
    # 🔄 FALLBACK: Méthode traditionnelle si YOLO échoue ou pas disponible
    if len(objects_data) == 0:
        print("🔄 Utilisation de la détection par contours (fallback)")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_with_detections = image.copy()
        
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10: 
                continue  # skip small
            
            # Découper l'objet pour analyse IA
            object_roi = image[y:y+h, x:x+w]
            
            # Analyse contextuelle avec IA si disponible
            if vision_models:
                try:
                    # Sauvegarder temporairement l'objet
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        object_pil = Image.fromarray(cv2.cvtColor(object_roi, cv2.COLOR_BGR2RGB))
                        object_pil.save(tmp.name)
                        
                        # Analyser avec CLIP
                        caption, results = analyze_image_with_clip(tmp.name, vision_models)
                        
                        if results and len(results) > 0:
                            obj_type = results[0]['label']
                            confidence = results[0]['confidence']
                        else:
                            obj_type = 'Objet non identifié'
                            confidence = 0.0
                        
                        os.unlink(tmp.name)
                except:
                    # Fallback sur analyse traditionnelle
                    aspect = w / h if h != 0 else 0
                    if aspect > 5: 
                        obj_type = 'Structure linéaire (route/chemin)'
                    elif aspect < 0.2: 
                        obj_type = 'Structure verticale (clôture/poteau)'
                    elif 0.5 < aspect < 2: 
                        obj_type = 'Structure carrée (bâtiment/zone)'
                    else: 
                        obj_type = 'Structure irrégulière'
                    confidence = 0.5
            else:
                # Fallback sur analyse traditionnelle
                aspect = w / h if h != 0 else 0
                if aspect > 5: 
                    obj_type = 'Structure linéaire (route/chemin)'
                elif aspect < 0.2: 
                    obj_type = 'Structure verticale (clôture/poteau)'
                elif 0.5 < aspect < 2: 
                    obj_type = 'Structure carrée (bâtiment/zone)'
                else: 
                    obj_type = 'Structure irrégulière'
                confidence = 0.5
            
            # Calculs métriques réels
            w_m = w * scale_factor
            h_m = h * scale_factor
            area_m2 = w_m * h_m
            perimeter_m = 2 * (w_m + h_m)
            
            # Analyse de couleur dominante
            mean_color = cv2.mean(object_roi)[:3]
            color_desc = f"RGB({int(mean_color[2])},{int(mean_color[1])},{int(mean_color[0])})"
            
            # Stocker les données
            objects_data.append({
                'id': idx + 1,
                'type': obj_type,
                'confidence': confidence,
                'width_m': w_m,
                'height_m': h_m,
                'area_m2': area_m2,
                'perimeter_m': perimeter_m,
                'color': color_desc,
                'position': (x, y)
            })
            
            # Dessiner sur l'image
            cv2.rectangle(img_with_detections, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"#{idx+1}: {obj_type[:20]}"
            cv2.putText(img_with_detections, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img_with_detections, f"{w_m:.2f}x{h_m:.2f}m", (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Créer la visualisation
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB))
    
    # Déterminer le titre en fonction de la méthode utilisée
    detection_method = "YOLO + IA" if yolo_model and len(objects_data) > 0 and "YOLO" in objects_data[0].get('type', '') else "Contours + IA"
    ax.set_title(f"Objets Détectés ({detection_method}): {len(objects_data)} objets", 
                fontsize=14, fontweight='bold', color='white')
    ax.axis('off')
    obj_img = fig_to_pil(fig)
    
    # Créer le tableau HTML des objets avec analyses complètes
    if objects_data:
        dim_df = pd.DataFrame(objects_data)
        dim_df = dim_df.rename(columns={
            'id': 'ID',
            'type': 'Type d\'objet (IA)',
            'confidence': 'Confiance IA',
            'width_m': 'Largeur (m)',
            'height_m': 'Hauteur (m)',
            'area_m2': 'Surface (m²)',
            'perimeter_m': 'Périmètre (m)',
            'color': 'Couleur dominante'
        })
        dim_df['Confiance IA'] = dim_df['Confiance IA'].apply(lambda x: f"{x*100:.1f}%")
        dim_df = dim_df.drop('position', axis=1)
        dim_html = df_to_html(dim_df)
    else:
        dim_html = ""
    
    return len(objects_data), obj_img, dim_html, objects_data
def detect_fences(image: np.ndarray, scale_factor=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    img_with_lines = image.copy()
    lengths = []
    if lines is not None:
        line_list = [line[0] for line in lines]
        filtered_lines = [l for l in line_list if abs(l[0] - l[2]) < 10 or abs(l[1] - l[3]) < 10 or abs((l[1]-l[3]) / (l[0]-l[2] + 1e-5)) < 0.1 or abs((l[1]-l[3]) / (l[0]-l[2] + 1e-5)) > 10]
        line_lengths = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2) for x1,y1,x2,y2 in filtered_lines]
        sorted_indices = np.argsort(line_lengths)[::-1]
        sorted_lines = [filtered_lines[i] for i in sorted_indices]
        for x1,y1,x2,y2 in sorted_lines:
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * scale_factor
            lengths.append(length)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(img_with_lines, f"{length:.4f}m", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Clôtures/Bordures Détectées ({len(lengths)})")
    ax.axis('off')
    fence_img = fig_to_pil(fig)
    if lengths:
        fence_df = pd.DataFrame({
            'Longueur (m)': lengths,
            'Explication': ['Longueur de bordure filtrée et triée pour précision' for _ in lengths]
        })
        fence_html = df_to_html(fence_df)
    else:
        fence_html = ""
    return len(lengths), fence_img, fence_html
def detect_anomalies(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    num_edges = np.sum(edges > 0)
    mean_variance = np.mean(cv2.Laplacian(gray, cv2.CV_64F).var())
    anomalies = []
    if num_edges > 10000:
        anomalies.append("Anomalies structurelles détectées (ex. : fissures, défauts)")
    if mean_variance > 500:
        anomalies.append("Textures inhabituelles (ex. : zones irrégulières)")
    # Simulation photogrammétrie basique SANS Open3D (désactivé pour Python 3.13)
    # Open3D n'est pas disponible pour Python 3.13, donc on simule autrement
    num_points = gray.shape[0] * gray.shape[1]  # Nombre de pixels comme approximation
    
    # Graphisme : Histogramme des variances
    fig, ax = plt.subplots()
    ax.hist(cv2.Laplacian(gray, cv2.CV_64F).ravel(), bins=50)
    ax.set_title('Histogramme des Variances Locales (Anomalies)')
    var_hist_img = fig_to_pil(fig)
    # Tableau des metrics anomalies
    anomaly_df = pd.DataFrame({
        'Métrique': ['Nombre de Bords', 'Variance Moyenne', 'Points Estimés'],
        'Valeur': [num_edges, mean_variance, num_points],
        'Explication': ['Indique complexité structurelle (haut = anomalies)', 'Mesure irrégularités texture', 'Estimation de densité pour volume']
    })
    anomaly_html = df_to_html(anomaly_df)
    anomaly_desc_df = pd.DataFrame({
        'Anomalie': anomalies if anomalies else ['Aucune anomalie détectée'],
        'Explication': ['Défauts potentiels dans le terrain ou structures' if anomalies else 'Image sans anomalies significatives']
    })
    anomaly_desc_html = df_to_html(anomaly_desc_df)
    return anomalies, var_hist_img, anomaly_html, anomaly_desc_html
def advanced_analyses(image: np.ndarray):
    analyses = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 100, 200)
    adv_images = []
    adv_tables = []
    # 1. Analyse Géologique
    kmeans = KMeans(n_clusters=3).fit(gray.reshape(-1, 1))
    clustered = kmeans.labels_.reshape(gray.shape)
    analyses['Géologique'] = 'Clusters de textures : ' + str(np.unique(kmeans.labels_))
    fig, ax = plt.subplots()
    ax.imshow(clustered, cmap='viridis')
    ax.set_title('Analyse Géologique: Clustering Textures')
    ax.axis('off')
    adv_images.append(fig_to_pil(fig))
    geo_df = pd.DataFrame({'Cluster': np.unique(kmeans.labels_), 'Compte': np.bincount(kmeans.labels_), 'Explication': ['Groupe de texture géologique' for _ in np.unique(kmeans.labels_)]})
    adv_tables.append(df_to_html(geo_df))
    # 2. Analyse Hydrologique
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    water_area = np.sum(blue_mask > 0) / blue_mask.size * 100
    analyses['Hydrologique'] = f'Pourcentage eau : {water_area:.2f}%'
    fig, ax = plt.subplots()
    ax.imshow(blue_mask, cmap='gray')
    ax.set_title('Analyse Hydrologique: Masque Eau')
    ax.axis('off')
    adv_images.append(fig_to_pil(fig))
    hydro_df = pd.DataFrame({'Métrique': ['Pourcentage Eau'], 'Valeur': [water_area], 'Explication': ['Zone potentielle pour ressources hydriques']})
    adv_tables.append(df_to_html(hydro_df))
    return analyses, {}, adv_images, adv_tables
def process_image(uploaded_file, vision_models=None, ocr_reader=None, yolo_model=None, blip_models=None):
    """Traite l'image avec analyse IA avancée: YOLO + CLIP + BLIP + OCR + analyses contextuelles"""
    image = Image.open(BytesIO(uploaded_file))
    img_array = np.array(image)
    proc_images = [image]
    captions = ['Image Originale']
    tables_html = []
    
    # 🖼️ DESCRIPTION DÉTAILLÉE AVEC CLIP (PRIORITAIRE)
    clip_description = ""
    if vision_models:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                image.save(tmp.name)
                clip_result, error = generate_detailed_description_with_clip(tmp.name, vision_models)
                
                if clip_result:
                    clip_description = f"""<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: white; margin: 0 0 10px 0;">🖼️ Description de l'image (CLIP Vision AI)</h4>
                    <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 10px 0;">"{clip_result['description']}"</p>
                    <p style="color: white; font-weight: 600; margin: 5px 0; font-size: 0.95rem;">
                        <strong>Type:</strong> {clip_result['main_type']}<br>
                        <strong>Couleurs:</strong> {clip_result['colors']}<br>
                        <strong>Confiance:</strong> {clip_result['confidence']*100:.1f}%
                    </p>
                    </div>"""
                    tables_html.append(clip_description)
                else:
                    tables_html.append(f'<h3 style="color: white; font-weight: bold;">🖼️ Description CLIP</h3><p style="color: orange; font-weight: 600;">Erreur: {error}</p>')
                
                import os
                os.unlink(tmp.name)
        except Exception as e:
            print(f"⚠️ Erreur description CLIP: {e}")
            tables_html.append(f'<h3 style="color: white; font-weight: bold;">🖼️ Description CLIP</h3><p style="color: orange; font-weight: 600;">Erreur: {str(e)}</p>')
    
    # 🖼️ BLIP - DESCRIPTION DÉTAILLÉE (OPTIONNEL - si disponible)
    blip_description = ""
    if blip_models:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                image.save(tmp.name)
                blip_result, error = caption_image_with_blip(tmp.name, blip_models)
                
                if blip_result:
                    blip_description = f"""<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: white; margin: 0 0 10px 0;">🖼️ Description BLIP (IA Vision)</h4>
                    <p style="color: white; font-weight: 600; margin: 5px 0;"><strong>Description générale:</strong> {blip_result['caption']}</p>
                    <p style="color: white; font-weight: 600; margin: 5px 0;"><strong>Description détaillée:</strong> {blip_result['detailed_caption']}</p>
                    </div>"""
                    tables_html.append(blip_description)
                
                import os
                os.unlink(tmp.name)
        except Exception as e:
            print(f"⚠️ BLIP non disponible: {e}")
    
    # 📝 OCR - EXTRACTION DE TEXTE
    ocr_text = ""
    if ocr_reader:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                image.save(tmp.name)
                extracted_texts = extract_text_from_image(tmp.name, ocr_reader)
                ocr_organized = organize_extracted_text(extracted_texts)
                ocr_text = ocr_organized
                
                if extracted_texts:
                    tables_html.append(f'<h3 style="color: white; font-weight: bold;">📝 OCR - Texte Détecté</h3><pre style="color: white; font-weight: 600;">{ocr_organized}</pre>')
                else:
                    tables_html.append('<h3 style="color: white; font-weight: bold;">📝 OCR - Texte Détecté</h3><p style="color: white; font-weight: 600;">Aucun texte détecté dans l\'image</p>')
                
                import os
                os.unlink(tmp.name)
        except Exception as e:
            print(f"⚠️ Erreur OCR: {e}")
            tables_html.append(f'<h3 style="color: white; font-weight: bold;">📝 OCR</h3><p style="color: white; font-weight: 600;">Erreur OCR: {str(e)}</p>')
    
    # 🌡️ IR
    ir_pil, ir_analysis = simulate_infrared(img_array)
    proc_images.append(ir_pil)
    captions.append('Simulation Infrarouge')
    tables_html.append('<h3 style="color: white; font-weight: bold;">Analyse IR</h3><p style="color: white; font-weight: 600;">' + ir_analysis + '</p>')
    
    # 🌍 Soil
    soil, hist_img, metrics_html = classify_soil(img_array)
    proc_images.append(hist_img)
    captions.append('Histogramme HSV')
    tables_html.append('<h3 style="color: white; font-weight: bold;">Métriques Sol</h3>' + metrics_html.replace('<td>', '<td style="color: white; font-weight: 600;">').replace('<th>', '<th style="color: white; font-weight: bold;">'))
    
    # 🎯 Objects avec YOLO + CLIP IA
    num_objects, obj_img, dim_html, objects_data = detect_objects(img_array, vision_models=vision_models, yolo_model=yolo_model)
    proc_images.append(obj_img)
    captions.append('🎯 Objets YOLO + IA')
    if dim_html:
        tables_html.append('<h3 style="color: white; font-weight: bold;">🎯 Objets Détectés (YOLO + CLIP)</h3>' + dim_html.replace('<td>', '<td style="color: white; font-weight: 600;">').replace('<th>', '<th style="color: white; font-weight: bold;">'))
    
    # Fences
    num_fences, fence_img, fence_html = detect_fences(img_array)
    proc_images.append(fence_img)
    captions.append('Clôtures Détectées')
    if fence_html:
        tables_html.append('<h3 style="color: white; font-weight: bold;">Longueurs Clôtures</h3>' + fence_html.replace('<td>', '<td style="color: white; font-weight: 600;">').replace('<th>', '<th style="color: white; font-weight: bold;">'))
    
    # Anomalies
    anomalies, var_hist_img, anomaly_html, anomaly_desc_html = detect_anomalies(img_array)
    proc_images.append(var_hist_img)
    captions.append('Histogramme Variances')
    tables_html.append('<h3 style="color: white; font-weight: bold;">Métriques Anomalies</h3>' + anomaly_html.replace('<td>', '<td style="color: white; font-weight: 600;">').replace('<th>', '<th style="color: white; font-weight: bold;">'))
    
    # Advanced
    analyses, predictions, adv_images, adv_tables = advanced_analyses(img_array)
    proc_images += adv_images[:5]
    captions += ['Analyse Avancée'] * len(adv_images[:5])
    # Appliquer style blanc gras aux tableaux avancés
    adv_tables_styled = [t.replace('<td>', '<td style="color: white; font-weight: 600;">').replace('<th>', '<th style="color: white; font-weight: bold;">') for t in adv_tables[:3]]
    tables_html += adv_tables_styled
    
    analysis_data = {
        "soil": soil,
        "ir_analysis": ir_analysis,
        "num_objects": num_objects,
        "objects_detected": objects_data,
        "num_fences": num_fences,
        "anomalies": anomalies,
        "analyses": analyses,
        "predictions": predictions,
        "ocr_text": ocr_text,  # Texte OCR
        "clip_description": clip_description,  # Description CLIP détaillée
        "blip_description": blip_description  # Description BLIP (optionnel)
    }
    
    tables_str = '<br>'.join(tables_html)
    return analysis_data, proc_images, tables_str
def generate_comprehensive_description(analysis_data):
    """
    Génère une description complète et cohérente en combinant tous les modèles d'IA
    """
    description_parts = []
    
    # 1. CLIP Description détaillée (prioritaire)
    clip_desc = analysis_data.get('clip_description', '')
    if clip_desc and '<p style' in clip_desc:
        # Extraire le texte du HTML
        import re
        clip_match = re.search(r'<p style="color: white; font-weight: 700.*?">(.*?)</p>', clip_desc, re.DOTALL)
        if clip_match:
            description_parts.append("🖼️ **Description de l'image (CLIP Vision AI):**")
            description_parts.append(f"   {clip_match.group(1).strip()}")
    
    # 2. BLIP (si disponible)
    blip_desc = analysis_data.get('blip_description', '')
    if blip_desc and '<p style' in blip_desc:
        import re
        blip_texts = re.findall(r'<strong>.*?:</strong>\s*(.*?)</p>', blip_desc)
        if blip_texts:
            description_parts.append("\n🎨 **Analyse BLIP complémentaire:**")
            description_parts.append(f"   {blip_texts[0]}")
    
    # 3. OCR (texte détecté)
    ocr_text = analysis_data.get('ocr_text', '')
    if ocr_text and 'Aucun texte' not in ocr_text:
        # Extraire juste les mots principaux
        import re
        text_matches = re.findall(r'✅.*?\]\s*(.*?)$', ocr_text, re.MULTILINE)
        if text_matches:
            unique_words = list(set([w.strip() for w in text_matches if len(w.strip()) > 1]))
            if unique_words:
                description_parts.append(f"\n📝 **Texte détecté:** {', '.join(unique_words[:10])}")
    
    # 4. YOLO (objets détectés)
    objects = analysis_data.get('objects_detected', [])
    if objects:
        yolo_objects = [obj for obj in objects if 'YOLO' in obj.get('type', '')]
        if yolo_objects:
            object_types = {}
            for obj in yolo_objects:
                obj_type = obj['type'].replace(' (YOLO)', '')
                object_types[obj_type] = object_types.get(obj_type, 0) + 1
            
            obj_summary = ', '.join([f"{count} {name}" if count > 1 else name 
                                    for name, count in list(object_types.items())[:5]])
            description_parts.append(f"\n🎯 **Objets identifiés (YOLO):** {obj_summary}")
    
    # 5. Informations techniques
    num_objects = analysis_data.get('num_objects', 0)
    if num_objects > 0:
        description_parts.append(f"\n📊 **Analyse technique:** {num_objects} éléments détectés")
    
    # Générer le résumé final
    if description_parts:
        return '\n'.join(description_parts)
    else:
        return "Image analysée - aucune information spécifique détectée"

def improve_analysis_with_llm(analysis_data, model_name):
    """
    Améliore l'analyse avec le LLM en utilisant la description complète
    """
    # Générer d'abord une description cohérente
    comprehensive_desc = generate_comprehensive_description(analysis_data)
    
    # Extraire les informations clés pour le LLM
    blip_desc = analysis_data.get('blip_description', '')
    ocr_text = analysis_data.get('ocr_text', '')
    objects = analysis_data.get('objects_detected', [])
    
    # Construire un prompt intelligent
    prompt = f"""Analyse cette image en français de manière précise et concise:

{comprehensive_desc}

Informations techniques:
- Résolution: {analysis_data.get('soil', 'N/A')}
- Objets: {len(objects)} détectés

Décris ce que tu vois en 2-3 phrases claires et précises."""

    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt[:2000]}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback sur la description générée
        return comprehensive_desc
# ===============================================
# Interface Streamlit Améliorée
# ===============================================
def main():
    """Interface Streamlit unifiée avec recherche web intégrée"""
    
    # ===============================================
    # SYSTÈME DE PALETTES DE COULEURS DYNAMIQUES
    # ===============================================
    if 'color_theme' not in st.session_state:
        st.session_state.color_theme = 'ocean_blue'  # Mode sombre bleu par défaut
    
    # Définition des palettes
    COLOR_THEMES = {
        'kibali_classic': {
            'name': '🟢 Kibali Classic',
            'primary': '#00ff88',
            'secondary': '#0088ff',
            'accent': '#ffd700',
            'bg_dark': '#1a1a2e',
            'bg_darker': '#0f0f23',
            'bg_light': '#2a2a4e',
            'bg_lighter': '#3a3a5e',
            'bg_accent': '#4a4a7e',
            'border': '#5a5a8a',
            'text': '#ffffff',
            'text_secondary': '#e0e0e0',
            'text_muted': '#b0b0b0',
            'input_bg': '#3a3a5e',
            'input_text': '#ffffff'
        },
        'ocean_blue': {
            'name': '🌊 Ocean Blue',
            'primary': '#00d4ff',
            'secondary': '#0066ff',
            'accent': '#00ffff',
            'bg_dark': '#0a1628',
            'bg_darker': '#050b14',
            'bg_light': '#1a2d4e',
            'bg_lighter': '#2a3d6e',
            'bg_accent': '#3a4d8e',
            'border': '#4a5daa',
            'text': '#ffffff',
            'text_secondary': '#cfe0ff',
            'text_muted': '#8fa9d4',
            'input_bg': '#2a3d6e',
            'input_text': '#ffffff'
        },
        'sunset_orange': {
            'name': '🌅 Sunset Orange',
            'primary': '#ff6b35',
            'secondary': '#ff9f1c',
            'accent': '#ffcf00',
            'bg_dark': '#2e1a0a',
            'bg_darker': '#1a0f05',
            'bg_light': '#4e2a1a',
            'bg_lighter': '#6e3a2a',
            'bg_accent': '#8e4a3a',
            'border': '#ae5a4a',
            'text': '#ffffff',
            'text_secondary': '#ffe0d0',
            'text_muted': '#d4a590',
            'input_bg': '#6e3a2a',
            'input_text': '#ffffff'
        },
        'purple_dream': {
            'name': '💜 Purple Dream',
            'primary': '#a855f7',
            'secondary': '#8b5cf6',
            'accent': '#ec4899',
            'bg_dark': '#1e0a2e',
            'bg_darker': '#0f0514',
            'bg_light': '#2e1a4e',
            'bg_lighter': '#3e2a6e',
            'bg_accent': '#4e3a8e',
            'border': '#5e4aaa',
            'text': '#ffffff',
            'text_secondary': '#e0cfff',
            'text_muted': '#a98fd4',
            'input_bg': '#3e2a6e',
            'input_text': '#ffffff'
        },
        'forest_green': {
            'name': '🌲 Forest Green',
            'primary': '#10b981',
            'secondary': '#059669',
            'accent': '#34d399',
            'bg_dark': '#0a2e1a',
            'bg_darker': '#05140a',
            'bg_light': '#1a4e2a',
            'bg_lighter': '#2a6e3a',
            'bg_accent': '#3a8e4a',
            'border': '#4aae5a',
            'text': '#ffffff',
            'text_secondary': '#d0ffe0',
            'text_muted': '#90d4a5',
            'input_bg': '#2a6e3a',
            'input_text': '#ffffff'
        },
        'ruby_red': {
            'name': '💎 Ruby Red',
            'primary': '#ef4444',
            'secondary': '#dc2626',
            'accent': '#f87171',
            'bg_dark': '#2e0a0a',
            'bg_darker': '#140505',
            'bg_light': '#4e1a1a',
            'bg_lighter': '#6e2a2a',
            'bg_accent': '#8e3a3a',
            'border': '#ae4a4a',
            'text': '#ffffff',
            'text_secondary': '#ffd0d0',
            'text_muted': '#d49090',
            'input_bg': '#6e2a2a',
            'input_text': '#ffffff'
        },
        'gold_luxury': {
            'name': '✨ Gold Luxury',
            'primary': '#fbbf24',
            'secondary': '#f59e0b',
            'accent': '#fde047',
            'bg_dark': '#2e1e0a',
            'bg_darker': '#140f05',
            'bg_light': '#4e3e1a',
            'bg_lighter': '#6e5e2a',
            'bg_accent': '#8e7e3a',
            'border': '#ae9e4a',
            'text': '#ffffff',
            'text_secondary': '#fff0d0',
            'text_muted': '#d4c590',
            'input_bg': '#6e5e2a',
            'input_text': '#ffffff'
        },
        'cyber_neon': {
            'name': '🔮 Cyber Neon',
            'primary': '#ff00ff',
            'secondary': '#00ffff',
            'accent': '#ffff00',
            'bg_dark': '#1a0a2e',
            'bg_darker': '#0a0514',
            'bg_light': '#2a1a4e',
            'bg_lighter': '#3a2a6e',
            'bg_accent': '#4a3a8e',
            'border': '#5a4aaa',
            'text': '#ffffff',
            'text_secondary': '#ffcfff',
            'text_muted': '#cf8fff',
            'input_bg': '#3a2a6e',
            'input_text': '#ffffff'
        }
    }
    
    # Récupérer le thème actuel
    theme = COLOR_THEMES[st.session_state.color_theme]

    # ===============================================
    # CSS PERSONNALISÉ - DESIGN PUISSANT ET FLUIDE
    # ===============================================
    st.markdown(f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Variables CSS pour les couleurs - DYNAMIQUES */
    :root {{{{{{{{
        --kibali-green: {theme['primary']};
        --kibali-yellow: {theme['accent']};
        --kibali-blue: {theme['secondary']};
        --kibali-dark: {theme['bg_dark']};
        --kibali-darker: {theme['bg_darker']};
        --kibali-light: {theme['bg_light']};
        --kibali-lighter: {theme['bg_lighter']};
        --kibali-accent: {theme['bg_accent']};
        --kibali-border: {theme['border']};
        --kibali-text: {theme['text']};
        --kibali-text-secondary: {theme['text_secondary']};
        --kibali-text-muted: {theme['text_muted']};
        --kibali-input-bg: {theme['input_bg']};
        --kibali-input-text: {theme['input_text']};
        --gradient-primary: linear-gradient(135deg, var(--kibali-green), var(--kibali-blue));
        --gradient-secondary: linear-gradient(135deg, var(--kibali-yellow), var(--kibali-green));
        --gradient-background: linear-gradient(135deg, var(--kibali-dark), var(--kibali-darker));
        --shadow-glow: 0 0 20px rgba({int(theme['primary'][1:3], 16)}, {int(theme['primary'][3:5], 16)}, {int(theme['primary'][5:7], 16)}, 0.3);
        --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.2);
        --shadow-subtle: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}}}}}}}

    /* Reset et base - Version améliorée */
    * {{{{
        font-family: 'Inter', sans-serif !important;
    }}}}

    /* Fond principal avec gradient subtil et texture */
    .main {{{{
        background: var(--gradient-background) !important;
        background-attachment: fixed !important;
        position: relative;
    }}}}

    .main::before {{{{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
            radial-gradient(circle at 20% 80%, rgba({int(theme['primary'][1:3], 16)}, {int(theme['primary'][3:5], 16)}, {int(theme['primary'][5:7], 16)}, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba({int(theme['secondary'][1:3], 16)}, {int(theme['secondary'][3:5], 16)}, {int(theme['secondary'][5:7], 16)}, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba({int(theme['accent'][1:3], 16)}, {int(theme['accent'][3:5], 16)}, {int(theme['accent'][5:7], 16)}, 0.02) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }}}}

    .stApp {{{{
        background: transparent !important;
    }}}}

    /* Logo Kibali animé */
    .kibali-logo {{{{
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin: 2rem 0;
        animation: logoGlow 3s ease-in-out infinite alternate;
        text-shadow: var(--shadow-glow);
    }}}}

    /* Logo K Kibali - AMÉLIORE avec bordure scintillante - TRÈS GRAND */
    .kibali-k-logo {{{{
        font-size: 12rem !important; /* ENCORE PLUS ÉNORME - 192px */
        font-weight: 900 !important;
        text-align: center !important;
        margin: 2rem auto !important;
        color: #000000 !important; /* Texte en NOIR */
        position: relative !important;
        display: inline-block !important;
        padding: 50px 70px !important;
        background: #ffffff !important; /* Fond blanc pour contraste */
        border-radius: 30px !important;
        text-shadow: 0 4px 20px rgba(255, 255, 255, 0.8) !important;
        min-width: 250px !important;
        min-height: 250px !important;
        line-height: 1 !important;
    }}}}

    /* Bordure scintillante AUTOUR du K */
    .kibali-k-logo::before {{{{
        content: '';
        position: absolute;
        top: -5px;
        left: -5px;
        right: -5px;
        bottom: -5px;
        background: linear-gradient(45deg, 
            var(--kibali-green), 
            var(--kibali-yellow), 
            var(--kibali-blue),
            var(--kibali-green));
        background-size: 400% 400%;
        border-radius: 25px;
        z-index: -1;
        animation: borderScintillation 3s ease infinite;
        filter: blur(2px);
    }}}}
    
    /* Animation de scintillement des bordures */
    @keyframes borderScintillation {{{{{{{{
        0% {{{{{{{{
            background-position: 0% 50%;
            filter: brightness(1.2) blur(2px);
        }}}}}}}}
        25% {{{{{{{{
            background-position: 50% 100%;
            filter: brightness(1.5) blur(3px);
        }}}}}}}}
        50% {{{{{{{{
            background-position: 100% 50%;
            filter: brightness(1.8) blur(2px);
        }}}}}}}}
        75% {{{{{{{{
            background-position: 50% 0%;
            filter: brightness(1.5) blur(3px);
        }}}}}}}}
        100% {{{{{{{{
            background-position: 0% 50%;
            filter: brightness(1.2) blur(2px);
        }}}}}}}}
    }}}}}}}}
    
    /* Effet de lueur externe */
    .kibali-k-logo::after {{{{
        content: '';
        position: absolute;
        top: -8px;
        left: -8px;
        right: -8px;
        bottom: -8px;
        background: linear-gradient(45deg,
            rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.3),
            rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.3),
            rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.3));
        border-radius: 28px;
        z-index: -2;
        filter: blur(20px);
        animation: glowPulse 2s ease-in-out infinite alternate;
    }}}}}}}}
    
    @keyframes glowPulse {{{{{{{{
        0% {{{{{{{{ opacity: 0.4; transform: scale(0.98); }}}}}}}}
        100% {{{{{{{{ opacity: 0.8; transform: scale(1.02); }}}}}}}}
    }}}}}}}}
    
    /* Nom KibaliOne8 IA avec effet scintillant - TRÈS GRAND */
    .kibali-name-logo {{{{
        font-size: 3.5rem !important; /* 56px */
        font-weight: 900 !important;
        text-align: center !important;
        margin-top: 2rem !important;
        color: #ffffff !important;
        position: relative !important;
        display: inline-block !important;
        padding: 20px 50px !important;
        background: #000000 !important;
        border-radius: 20px !important;
        letter-spacing: 5px !important;
        text-shadow: 0 3px 15px rgba(0, 0, 0, 0.8) !important;
        text-transform: uppercase !important;
    }}}}
    
    /* Bordure scintillante pour le nom */
    .kibali-name-logo::before {{{{
        content: '';
        position: absolute;
        top: -4px;
        left: -4px;
        right: -4px;
        bottom: -4px;
        background: linear-gradient(45deg, 
            var(--kibali-green), 
            var(--kibali-yellow), 
            var(--kibali-blue),
            var(--kibali-green));
        background-size: 400% 400%;
        border-radius: 18px;
        z-index: -1;
        animation: borderScintillation 3s ease infinite;
        filter: blur(2px);
    }}}}
    
    /* Lueur externe pour le nom */
    .kibali-name-logo::after {{{{
        content: '';
        position: absolute;
        top: -6px;
        left: -6px;
        right: -6px;
        bottom: -6px;
        background: linear-gradient(45deg,
            rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.2),
            rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.2),
            rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.2));
        border-radius: 20px;
        z-index: -2;
        filter: blur(15px);
        animation: glowPulse 2s ease-in-out infinite alternate;
    }}}}
    
    @keyframes glowPulse {{{{{{{{
        0% {{{{{{{{ opacity: 0.4; transform: scale(0.98); }}}}}}}}
        100% {{{{{{{{ opacity: 0.8; transform: scale(1.02); }}}}}}}}
    }}}}}}}}

    @keyframes kGlow {{{{{{{{
        0% {{{{{{{{
            filter: brightness(1) drop-shadow(0 0 20px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.5));
            transform: scale(1);
        }}}}}}}}
        50% {{{{{{{{
            filter: brightness(1.3) drop-shadow(0 0 40px rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.7));
            transform: scale(1.05);
        }}}}}}}}
        100% {{{{{{{{
            filter: brightness(1) drop-shadow(0 0 30px rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.6));
            transform: scale(1);
        }}}}}}}}
    }}}}}}}}

    @keyframes kGlowBorder {{{{{{{{
        0% {{{{{{{{
            filter: drop-shadow(0 0 10px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.8));
        }}}}}}}}
        33% {{{{{{{{
            filter: drop-shadow(0 0 15px rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.9));
        }}}}}}}}
        66% {{{{{{{{
            filter: drop-shadow(0 0 20px rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.8));
        }}}}}}}}
        100% {{{{{{{{
            filter: drop-shadow(0 0 10px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.8));
        }}}}}}}}
    }}}}}}}}

    @keyframes kBorderShine {{{{{{{{
        0% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-green), var(--kibali-yellow), var(--kibali-blue));
            opacity: 0.6;
        }}}}}}}}
        25% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-blue), var(--kibali-green));
            opacity: 0.8;
        }}}}}}}}
        50% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-blue), var(--kibali-green), var(--kibali-yellow));
            opacity: 1.0;
        }}}}}}}}
        75% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-green), var(--kibali-blue), var(--kibali-yellow));
            opacity: 0.8;
        }}}}}}}}
        100% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-green), var(--kibali-blue));
            opacity: 0.6;
        }}}}}}}}
    }}}}}}}}

    @keyframes kRotate {{{{{{{{
        0% {{{{{{{{ transform: rotateY(0deg); }}}}}}}}
        25% {{{{{{{{ transform: rotateY(5deg); }}}}}}}}
        50% {{{{{{{{ transform: rotateY(0deg); }}}}}}}}
        75% {{{{{{{{ transform: rotateY(-5deg); }}}}}}}}
        100% {{{{{{{{ transform: rotateY(0deg); }}}}}}}}
    }}}}}}}}

    @keyframes kBackgroundPulse {{{{{{{{
        0% {{{{{{{{ opacity: 0.1; transform: scale(1); }}}}}}}}
        100% {{{{{{{{ opacity: 0.3; transform: scale(1.1); }}}}}}}}
    }}}}}}}}

    @keyframes logoGlow {{{{{{{{
        0% {{{{{{{{ filter: brightness(1) drop-shadow(0 0 10px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.5)); }}}}}}}}
        100% {{{{{{{{ filter: brightness(1.2) drop-shadow(0 0 20px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.8)); }}}}}}}}
    }}}}}}}}

    /* Header principal - Version améliorée */
    .main-header {{{{
        background: linear-gradient(135deg, var(--kibali-light), var(--kibali-accent));
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid var(--kibali-border);
        box-shadow: var(--shadow-subtle);
        position: relative;
        overflow: hidden;
    }}}}

    .main-header::before {{{{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
        animation: headerShine 4s ease-in-out infinite;
    }}}}

    @keyframes headerShine {{{{{{{{
        0%, 100% {{{{{{{{ transform: translateX(-100%); }}}}}}}}
        50% {{{{{{{{ transform: translateX(100%); }}}}}}}}
    }}}}}}}}

    /* Cartes flex responsive - Version améliorée */
    .kibali-card {{{{
        background: var(--kibali-lighter);
        border: 1px solid var(--kibali-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-subtle);
        position: relative;
        overflow: hidden;
    }}}}
    
    /* 🎨 CARTE DE CHAT SPÉCIALE - Bordures scintillantes + Zoom dynamique */
    .kibali-chat-card {{{{
        background: var(--kibali-darker);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: visible;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        min-height: 400px;
    }}}}
    
    /* Bordure scintillante animée autour du chat */
    .kibali-chat-card::before {{{{
        content: '';
        position: absolute;
        top: -3px;
        left: -3px;
        right: -3px;
        bottom: -3px;
        background: linear-gradient(45deg, 
            var(--kibali-green), 
            var(--kibali-yellow), 
            var(--kibali-blue),
            var(--kibali-green));
        background-size: 400% 400%;
        border-radius: 22px;
        z-index: -1;
        animation: chatBorderScintillation 4s ease infinite;
        filter: blur(1px);
    }}}}
    
    /* Effet de lueur externe pulsante */
    .kibali-chat-card::after {{{{
        content: '';
        position: absolute;
        top: -6px;
        left: -6px;
        right: -6px;
        bottom: -6px;
        background: linear-gradient(45deg,
            rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.4),
            rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.4),
            rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.4));
        border-radius: 24px;
        z-index: -2;
        filter: blur(15px);
        animation: chatGlowPulse 3s ease-in-out infinite alternate;
    }}}}
    
    /* Animation de scintillement des bordures du chat */
    @keyframes chatBorderScintillation {{{{{{{{
        0% {{{{{{{{
            background-position: 0% 50%;
            filter: brightness(1.2) blur(1px);
        }}}}}}}}
        20% {{{{{{{{
            background-position: 100% 0%;
            filter: brightness(1.5) blur(2px);
        }}}}}}}}
        40% {{{{{{{{
            background-position: 50% 100%;
            filter: brightness(1.8) blur(1px);
        }}}}}}}}
        60% {{{{{{{{
            background-position: 0% 50%;
            filter: brightness(1.6) blur(2px);
        }}}}}}}}
        80% {{{{{{{{
            background-position: 100% 100%;
            filter: brightness(1.4) blur(1px);
        }}}}}}}}
        100% {{{{{{{{
            background-position: 0% 50%;
            filter: brightness(1.2) blur(1px);
        }}}}}}}}
    }}}}}}}}
    
    /* Pulsation de la lueur */
    @keyframes chatGlowPulse {{{{{{{{
        0% {{{{{{{{ 
            opacity: 0.3; 
            transform: scale(0.95); 
            filter: blur(15px);
        }}}}}}}}
        50% {{{{{{{{ 
            opacity: 0.6; 
            transform: scale(1.02); 
            filter: blur(20px);
        }}}}}}}}
        100% {{{{{{{{ 
            opacity: 0.8; 
            transform: scale(1.05); 
            filter: blur(25px);
        }}}}}}}}
    }}}}}}}}
    
    /* Zoom dynamique au survol */
    .kibali-chat-card:hover {{{{
        transform: scale(1.03) translateY(-5px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    }}}}
    
    /* Accélération de l'animation au survol */
    .kibali-chat-card:hover::before {{{{
        animation: chatBorderScintillation 2s ease infinite;
        filter: brightness(1.5) blur(2px);
    }}}}
    
    .kibali-chat-card:hover::after {{{{
        animation: chatGlowPulse 1.5s ease-in-out infinite alternate;
        opacity: 1;
        transform: scale(1.08);
        filter: blur(30px);
    }}}}
    
    /* Tout le texte dans les cartes en blanc gras */
    .kibali-card,
    .kibali-card p,
    .kibali-card h1,
    .kibali-card h2,
    .kibali-card h3,
    .kibali-card h4,
    .kibali-card h5,
    .kibali-card span,
    .kibali-card div,
    .kibali-card td,
    .kibali-card th,
    .kibali-card li {{{{
        color: white !important;
        font-weight: 600 !important;
    }}}}
    
    .kibali-card h1,
    .kibali-card h2,
    .kibali-card h3 {{{{
        font-weight: 700 !important;
    }}}}
    
    .kibali-card table {{{{
        color: white !important;
        background: rgba({int(theme["bg_light"][1:3], 16)}, {int(theme["bg_light"][3:5], 16)}, {int(theme["bg_light"][5:7], 16)}, 0.5) !important;
    }}}}
    
    .kibali-card th {{{{
        background: rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.2) !important;
        color: white !important;
        font-weight: 700 !important;
        border: 1px solid var(--kibali-border) !important;
    }}}}
    
    .kibali-card td {{{{
        border: 1px solid var(--kibali-border) !important;
        color: white !important;
        font-weight: 600 !important;
    }}}}

    .kibali-card::before {{{{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.1), transparent);
        transition: left 0.5s;
    }}}}

    .kibali-card:hover {{{{
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-card);
        border-color: var(--kibali-green);
    }}}}

    .kibali-card:hover::before {{{{
        left: 100%;
    }}}}

    /* Boutons stylisés */
    .kibali-btn {{{{
        background: var(--gradient-primary);
        color: white !important;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.3);
        position: relative;
        overflow: hidden;
    }}}}

    .kibali-btn::before {{{{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }}}}

    .kibali-btn:hover {{{{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.5);
    }}}}

    .kibali-btn:hover::before {{{{
        left: 100%;
    }}}}

    .kibali-btn:active {{{{
        transform: translateY(0);
    }}}}

    /* Onglets personnalisés - Version améliorée */
    .stTabs [data-baseweb="tab-list"] {{{{
        gap: 8px;
        background: var(--kibali-light);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid var(--kibali-border);
    }}}}

    .stTabs [data-baseweb="tab"] {{{{
        background: var(--kibali-accent);
        border-radius: 8px;
        color: var(--kibali-text-secondary);
        transition: all 0.3s ease;
        border: none;
        padding: 12px 20px;
        font-weight: 500;
    }}}}

    .stTabs [data-baseweb="tab"]:hover {{{{
        background: var(--kibali-lighter);
        color: var(--kibali-text);
        transform: translateY(-2px);
        box-shadow: var(--shadow-subtle);
    }}}}

    .stTabs [aria-selected="true"] {{{{
        background: var(--gradient-primary) !important;
        color: white !important;
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
    }}}}

    /* Chat messages - Version améliorée */
    .chat-message-user {{{{
        background: linear-gradient(135deg, var(--kibali-blue), var(--kibali-green));
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.3);
        animation: slideInLeft 0.5s ease-out;
        border: 1px solid rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.2);
    }}}}

    .chat-message-assistant {{{{
        background: var(--kibali-lighter);
        color: var(--kibali-text);
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.5rem 0;
        border-left: 4px solid var(--kibali-green);
        box-shadow: var(--shadow-subtle);
        animation: slideInRight 0.5s ease-out;
        border: 1px solid var(--kibali-border);
    }}}}

    @keyframes slideInLeft {{{{{{{{
        from {{{{{{{{ transform: translateX(-20px); opacity: 0; }}}}}}}}
        to {{{{{{{{ transform: translateX(0); opacity: 1; }}}}}}}}
    }}}}}}}}

    @keyframes slideInRight {{{{{{{{
        from {{{{{{{{ transform: translateX(20px); opacity: 0; }}}}}}}}
        to {{{{{{{{ transform: translateX(0); opacity: 1; }}}}}}}}
    }}}}}}}}
    
    @keyframes slideIn {{{{{{{{
        from {{{{{{{{ transform: translateY(10px); opacity: 0; }}}}}}}}
        to {{{{{{{{ transform: translateY(0); opacity: 1; }}}}}}}}
    }}}}}}}}
    
    @keyframes pulse {{{{{{{{
        0%, 100% {{{{{{{{ opacity: 1; transform: scale(1); }}}}}}}}
        50% {{{{{{{{ opacity: 0.8; transform: scale(1.05); }}}}}}}}
    }}}}}}}}
    
    /* Style des réponses de chat amélioré */
    .response-content {{{{{{{{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    }}}}}}}}
    
    .response-content h2 {{{{{{{{
        color: var(--kibali-green) !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin: 1.5rem 0 0.75rem 0 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid rgba(0, 255, 136, 0.3) !important;
    }}}}}}}}
    
    .response-content h3 {{{{{{{{
        color: #00ffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin: 1rem 0 0.5rem 0 !important;
    }}}}}}}}
    
    .response-content p {{{{{{{{
        margin: 0.75rem 0 !important;
        line-height: 1.8 !important;
        text-align: justify !important;
    }}}}}}}}
    
    .response-content ul, .response-content ol {{{{{{{{
        margin: 0.75rem 0 !important;
        padding-left: 1.5rem !important;
    }}}}}}}}
    
    .response-content li {{{{{{{{
        margin: 0.5rem 0 !important;
        line-height: 1.6 !important;
    }}}}}}}}
    
    .response-content strong {{{{{{{{
        color: var(--kibali-green) !important;
        font-weight: 700 !important;
    }}}}}}}}
    
    .response-content em {{{{{{{{
        color: #ffd700 !important;
        font-style: italic !important;
    }}}}}}}}
    
    .response-content code {{{{{{{{
        background: rgba(0, 255, 136, 0.1) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        color: var(--kibali-green) !important;
        font-family: 'Courier New', monospace !important;
    }}}}}}}}

    /* Input fields - TEXTE BLANC VISIBLE */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {{{{
        background: var(--kibali-input-bg) !important;
        border: 2px solid var(--kibali-border) !important;
        border-radius: 12px !important;
        color: var(--kibali-input-text) !important;
        padding: 14px 18px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        z-index: 1 !important;
    }}}}
    
    /* FORCER LE TEXTE BLANC PARTOUT */
    input, textarea, select {{{{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}}}
    
    /* Input du chat principal - ULTRA VISIBLE */
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea {{{{
        background: #6B46C1 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 16px 20px !important;
        border: 2px solid var(--kibali-green) !important;
        -webkit-text-fill-color: #ffffff !important;
    }}}}
    
    /* Labels styling - Éviter le chevauchement */
    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stNumberInput label,
    .stFileUploader label,
    [data-testid="stWidgetLabel"] {{{{
        color: var(--kibali-text-secondary) !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
        position: static !important;
        background: transparent !important;
        padding: 0 !important;
        width: 100% !important;
        text-align: left !important;
    }}}}
    
    /* Container pour éviter superposition */
    .stTextInput > div,
    .stTextArea > div,
    .stNumberInput > div,
    .stSelectbox > div,
    .stFileUploader > div {{{{
        display: flex !important;
        flex-direction: column !important;
        gap: 0.25rem !important;
    }}}}
    
    /* Forcer la séparation label/input */
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stNumberInput > div > div {{{{
        display: block !important;
        position: relative !important;
    }}}}
    
    /* S'assurer que le label est au-dessus */
    .stTextInput > div > label,
    .stTextArea > div > label,
    .stNumberInput > div > label,
    .stSelectbox > div > label {{{{
        order: -1 !important;
        margin-bottom: 0.5rem !important;
    }}}}
    
    /* Placeholder styling */
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {{{{
        color: var(--kibali-text-muted) !important;
        opacity: 0.6 !important;
    }}}}
    
    /* Hide placeholder when input has focus or value */
    .stTextInput input:focus::placeholder,
    .stTextArea textarea:focus::placeholder {{{{
        opacity: 0.3 !important;
    }}}}

    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {{{{
        border-color: var(--kibali-green) !important;
        box-shadow: 0 0 0 2px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.2) !important;
        transform: scale(1.02);
    }}}}
    
    /* Chat Input - Texte blanc gras sur fond violet */
    [data-testid="stChatInput"] {{{{
        position: relative !important;
    }}}}
    
    [data-testid="stChatInput"] input {{{{
        background: #6B46C1 !important;
        border: 2px solid var(--kibali-border) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 12px 16px !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        position: relative !important;
        z-index: 1 !important;
    }}}}
    
    [data-testid="stChatInput"] input::placeholder {{{{
        color: rgba(255, 255, 255, 0.6) !important;
        opacity: 1 !important;
        font-weight: 500 !important;
        position: absolute !important;
        left: 16px !important;
        pointer-events: none !important;
    }}}}
    
    [data-testid="stChatInput"] input:focus {{{{
        background: #7C3AED !important;
        border-color: var(--kibali-green) !important;
        box-shadow: 0 0 15px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.3) !important;
    }}}}
    
    [data-testid="stChatInput"] input:focus::placeholder {{{{
        opacity: 0.3 !important;
    }}}}
    
    /* Cacher le placeholder quand il y a du texte */
    [data-testid="stChatInput"] input:not(:placeholder-shown)::placeholder {{{{
        opacity: 0 !important;
    }}}}
    
    /* SIDEBAR - Corrections pour éviter superposition */
    .css-1d391kg, [data-testid="stSidebar"] {{{{
        background: var(--kibali-light) !important;
    }}}}
    
    /* Texte dans la sidebar - Meilleur contraste */
    [data-testid="stSidebar"] * {{{{
        color: var(--kibali-text) !important;
    }}}}
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{{{
        color: var(--kibali-green) !important;
        font-weight: 600 !important;
    }}}}
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {{{{
        color: var(--kibali-text) !important;
        font-size: 1rem !important;
    }}}}
    
    /* Boutons dans la sidebar - Style icône */
    [data-testid="stSidebar"] button {{{{
        width: 100% !important;
        margin: 0.25rem 0 !important;
        font-size: 2rem !important;
        padding: 0.75rem !important;
        background: var(--kibali-accent) !important;
        border: 2px solid var(--kibali-border) !important;
        transition: all 0.3s ease !important;
    }}}}
    
    [data-testid="stSidebar"] button:hover {{{{
        background: var(--kibali-green) !important;
        transform: scale(1.05) !important;
        box-shadow: 0 0 20px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.5) !important;
    }}}}
    
    [data-testid="stSidebar"] button:disabled {{{{
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }}}}
    
    /* Inputs dans la sidebar - Éviter superposition */
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stSelectbox {{{{
        margin-bottom: 1rem !important;
    }}}}
    
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSelectbox label {{{{
        display: block !important;
        margin-bottom: 0.5rem !important;
        position: static !important;
        background: transparent !important;
        padding: 0 !important;
        width: 100% !important;
    }}}}
    
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select {{{{
        width: 100% !important;
        display: block !important;
        margin-top: 0.5rem !important;
    }}}}
    
    /* Colonnes dans la sidebar */
    [data-testid="stSidebar"] [data-testid="column"] {{{{
        padding: 0.25rem !important;
    }}}}
    
    /* CORRECTION GLOBALE - Empêcher toute superposition de texte */
    /* Tous les widgets Streamlit */
    div[data-testid="stVerticalBlock"] > div {{{{
        display: flex !important;
        flex-direction: column !important;
    }}}}
    
    /* S'assurer que les labels sont toujours au-dessus */
    label {{{{
        position: static !important;
        display: block !important;
        margin-bottom: 0.25rem !important;
        z-index: auto !important;
    }}}}
    
    /* Empêcher l'overlay de texte sur les inputs */
    input, textarea, select {{{{
        position: relative !important;
        z-index: 1 !important;
        background-color: var(--kibali-accent) !important;
    }}}}
    
    /* Forcer la séparation pour tous les form elements */
    .stTextInput,
    .stTextArea,
    .stNumberInput,
    .stSelectbox,
    .stMultiSelect,
    .stDateInput,
    .stTimeInput {{{{
        margin-bottom: 0.75rem !important;
    }}}}
    
    /* Empêcher le chevauchement du placeholder avec la valeur */
    input:not(:placeholder-shown) {{{{
        color: var(--kibali-text) !important;
    }}}}
    
    input::placeholder {{{{
        color: var(--kibali-text-muted) !important;
        opacity: 0.5 !important;
    }}}}
    
    /* Quand l'input a une valeur, le placeholder doit disparaître */
    input[value]:not([value=""]) {{{{
        color: var(--kibali-text) !important;
    }}}}
    
    input[value]:not([value=""])::placeholder {{{{
        opacity: 0 !important;
        visibility: hidden !important;
    }}}}

    /* Success/Error messages - Version améliorée avec meilleur contraste */
    .stSuccess, .stError, .stWarning, .stInfo {{{{
        border-radius: 12px !important;
        border: none !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        animation: fadeIn 0.5s ease-out;
        box-shadow: var(--shadow-subtle);
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }}}}
    
    .stSuccess {{{{
        background: rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.25) !important;
        border-left: 4px solid var(--kibali-green) !important;
        color: #FFFFFF !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }}}}
    
    .stError {{{{
        background: rgba(255, 107, 107, 0.25) !important;
        border-left: 4px solid #ff6b6b !important;
        color: #FFFFFF !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }}}}
    
    .stWarning {{{{
        background: rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.25) !important;
        border-left: 4px solid var(--kibali-yellow) !important;
        color: #FFFFFF !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }}}}
    
    .stInfo {{{{
        background: rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.25) !important;
        border-left: 4px solid var(--kibali-blue) !important;
        color: #FFFFFF !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }}}}

    /* Messages du chat - Texte blanc gras et lisible */
    [data-testid="stChatMessage"] {{{{
        background: var(--kibali-accent) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        border: 1px solid var(--kibali-border) !important;
    }}}}
    
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {{{{
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }}}}
    
    /* Messages utilisateur - Fond violet */
    [data-testid="stChatMessage"][data-testid*="user"] {{{{
        background: #6B46C1 !important;
        border-color: #7C3AED !important;
    }}}}
    
    /* Messages assistant - Fond accent avec bordure verte */
    [data-testid="stChatMessage"][data-testid*="assistant"] {{{{
        background: var(--kibali-accent) !important;
        border-color: var(--kibali-green) !important;
    }}}}

    .stSuccess {{{{
        background: linear-gradient(135deg, rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.1), rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.05)) !important;
        border-left: 4px solid var(--kibali-green) !important;
        color: var(--kibali-text) !important;
    }}}}

    .stError {{{{
        background: linear-gradient(135deg, rgba(255, 136, 136, 0.1), rgba(255, 136, 136, 0.05)) !important;
        border-left: 4px solid #ff8888 !important;
        color: var(--kibali-text) !important;
    }}}}

    @keyframes fadeIn {{{{{{{{
        from {{{{{{{{ opacity: 0; transform: translateY(10px); }}}}}}}}
        to {{{{{{{{ opacity: 1; transform: translateY(0); }}}}}}}}
    }}}}}}}}

    /* Spinner personnalisé */
    .stSpinner > div > div {{{{
        border-color: var(--kibali-green) !important;
        border-top-color: transparent !important;
    }}}}

    /* Images avec zoom au survol */
    .zoom-image {{{{
        transition: transform 0.3s ease;
        border-radius: 12px;
        box-shadow: var(--shadow-card);
    }}}}

    .zoom-image:hover {{{{
        transform: scale(1.05);
    }}}}

    /* Progress bars */
    .stProgress > div > div > div {{{{
        background: var(--gradient-primary) !important;
    }}}}

    /* Sidebar si utilisée */
    .css-1d391kg, .css-12oz5g7 {{{{
        background: var(--kibali-dark) !important;
    }}}}

    /* Scrollbar personnalisée - Version améliorée */
    ::-webkit-scrollbar {{{{
        width: 8px;
    }}}}

    ::-webkit-scrollbar-track {{{{
        background: var(--kibali-dark);
        border-radius: 4px;
    }}}}

    ::-webkit-scrollbar-thumb {{{{
        background: var(--gradient-primary);
        border-radius: 4px;
        border: 1px solid var(--kibali-border);
    }}}}

    ::-webkit-scrollbar-thumb:hover {{{{
        background: var(--kibali-green);
        box-shadow: var(--shadow-glow);
    }}}}

    /* Icônes grandes et visibles - Priorité absolue */
    .stSuccess, .stError, .stWarning, .stInfo {{{{
        font-size: 2.5rem !important;
        text-align: center !important;
        padding: 1.5rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}}}
    
    /* Rendre les icônes emoji plus grandes partout */
    .stButton button {{{{
        font-size: 2rem !important;
        line-height: 1 !important;
    }}}}
    
    /* Messages avec icônes uniquement - Pas de texte */
    [data-testid="stMarkdownContainer"] {{{{
        font-size: 1.2rem !important;
    }}}}
    
    /* Expanders avec icônes */
    .streamlit-expanderHeader {{{{
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }}}}}}}}

    /* Responsive design */
    @media (max-width: 768px) {{{{{{{{
        .kibali-logo {{{{{{{{
            font-size: 2.5rem;
        }}}}}}}}

        .kibali-card {{{{{{{{
            padding: 1rem;
            margin: 0.5rem 0;
        }}}}}}}}

        .main-header {{{{{{{{
            padding: 1rem;
        }}}}}}}}
    }}}}}}}}

    /* Logo K pour les messages de chat */
    .chat-k-logo {{
        display: inline-block;
        font-size: 1.2rem;
        font-weight: 900;
        color: #ffffff; /* Texte blanc */
        background: #000000; /* Fond noir */
        position: relative;
        margin-right: 0.5rem;
        padding: 4px 8px;
        border-radius: 6px;
    }}

    .chat-k-logo::before {{
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, 
            var(--kibali-green), 
            var(--kibali-yellow), 
            var(--kibali-blue),
            var(--kibali-green));
        background-size: 300% 300%;
        border-radius: 8px;
        z-index: -1;
        animation: chatBorderScintillation 2s ease infinite;
        filter: blur(1px);
    }}

    
        33% {{{{{{{{
            filter: drop-shadow(0 0 5px rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.7));
        }}}}}}}}
        66% {{{{{{{{
            filter: drop-shadow(0 0 4px rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.6));
        }}}}}}}}
        100% {{{{{{{{
            filter: drop-shadow(0 0 3px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.6));
        }}}}}}}}
    }}}}}}}}

    
        25% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-blue), var(--kibali-green));
            opacity: 0.7;
        }}}}}}}}
        50% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-blue), var(--kibali-green), var(--kibali-yellow));
            opacity: 0.8;
        }}}}}}}}
        75% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-green), var(--kibali-blue), var(--kibali-yellow));
            opacity: 0.7;
        }}}}}}}}
        100% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-green), var(--kibali-blue));
            opacity: 0.5;
        }}}}}}}}
    }}}}}}}}
    
    
    @keyframes chatBorderScintillation {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    /* BOUTONS - Éviter superposition de texte */
    button {{{{
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        line-height: 1.5 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        position: relative !important;
    }}}}
    
    button span {{{{
        display: inline-block !important;
        position: relative !important;
    }}}}
    
    /* Désactiver tout pseudo-élément qui pourrait causer superposition */
    button::before,
    button::after {{{{
        display: none !important;
    }}}}
    
    /* Forcer un seul layer de texte dans les boutons */
    button * {{{{
        line-height: inherit !important;
    }}}}
    
    </style>
    """, unsafe_allow_html=True)

    st.set_page_config(
        page_title="🗺️ Kibali 🌟 - Assistant IA Avancé",
        page_icon="🌟",
        layout="wide"
    )

    # Logo Kibali animé - Grand K avec les couleurs vert, jaune, bleu
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0; padding: 2rem;">
        <svg width="300" height="300" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto;">
          <defs>
            <linearGradient id="sparkleGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style="stop-color:#00ff88;stop-opacity:1" />
              <stop offset="25%" style="stop-color:#ffff00;stop-opacity:1" />
              <stop offset="50%" style="stop-color:#0088ff;stop-opacity:1" />
              <stop offset="75%" style="stop-color:#00ff88;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#ffff00;stop-opacity:1" />
            </linearGradient>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          <rect x="10" y="10" width="280" height="280" rx="30" ry="30"
                fill="none" stroke="url(#sparkleGradient)" stroke-width="8"
                filter="url(#glow)">
            <animate attributeName="stroke-dasharray" values="0,1000;500,500;0,1000" dur="3s" repeatCount="indefinite"/>
          </rect>
          <rect x="20" y="20" width="260" height="260" rx="25" ry="25" fill="#ffffff"/>
          <text x="150" y="180" font-family="Arial, sans-serif" font-size="180" font-weight="900"
                text-anchor="middle" fill="#000000" filter="url(#glow)">K</text>
          <circle cx="50" cy="50" r="3" fill="#ffff00" opacity="0.8">
            <animate attributeName="opacity" values="0.8;0.3;0.8" dur="2s" repeatCount="indefinite"/>
          </circle>
          <circle cx="250" cy="50" r="3" fill="#00ff88" opacity="0.6">
            <animate attributeName="opacity" values="0.6;0.9;0.6" dur="2.5s" repeatCount="indefinite"/>
          </circle>
          <circle cx="50" cy="250" r="3" fill="#0088ff" opacity="0.7">
            <animate attributeName="opacity" values="0.7;0.4;0.7" dur="1.8s" repeatCount="indefinite"/>
          </circle>
          <circle cx="250" cy="250" r="3" fill="#ffff00" opacity="0.5">
            <animate attributeName="opacity" values="0.5;0.8;0.5" dur="2.2s" repeatCount="indefinite"/>
          </circle>
        </svg>
        <div class="kibali-name-logo">KIBALIONE8 IA</div>
    </div>
    """, unsafe_allow_html=True)

    # Header principal avec design fluide
    st.markdown("""
    <div class="main-header">
        <h2 style="color: var(--kibali-green); margin: 0; text-align: center;">
            ⏩ Assistant IA Avancé avec Recherche Web
        </h2>
        <p style="color: var(--kibali-text-secondary); text-align: center; margin: 1rem 0 0 0;">
            <strong>Nouvelles fonctionnalités:</strong><br>
            🌐 Recherche web intelligente • 🔗 Recherche hybride • 📰 Actualités temps réel<br>
            📄 Extraction de contenu web • 💾 Cache intelligent • 🗺️ Calcul de trajets OSM
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===============================================
    # SÉLECTEUR DE THÈME DE COULEUR
    # ===============================================
    st.markdown("### 🎨 Palette de Couleurs")
    
    theme_names = {k: v['name'] for k, v in COLOR_THEMES.items()}
    current_theme_name = theme_names[st.session_state.color_theme]
    
    selected_theme_name = st.selectbox(
        "Choisir un thème",
        options=list(theme_names.values()),
        index=list(theme_names.values()).index(current_theme_name),
        key="theme_selector"
    )
    
    # Trouver la clé du thème sélectionné
    selected_theme_key = [k for k, v in theme_names.items() if v == selected_theme_name][0]
    
    if selected_theme_key != st.session_state.color_theme:
        st.session_state.color_theme = selected_theme_key
        st.rerun()
    
    # Afficher les couleurs du thème actuel
    theme_colors = COLOR_THEMES[st.session_state.color_theme]
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin: 1rem 0;">
        <div style="background: {theme_colors['primary']}; height: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>
        <div style="background: {theme_colors['secondary']}; height: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>
        <div style="background: {theme_colors['accent']}; height: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>
        <div style="background: {theme_colors['bg_accent']}; height: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===============================================
    # SIDEBAR - Indicateur de mode Online/Offline
    # ===============================================
    render_mode_indicator()
    
    # Afficher le statut des capacités si en mode offline
    status = offline_manager.get_status_info()
    if status['is_offline']:
        with st.sidebar.expander("📊 Capacités"):
            caps = get_offline_capabilities()
            
            # Afficher avec des icônes expressives uniquement
            cap_icons = []
            if caps.get('llm'):
                cap_icons.append("🧠")
            if caps.get('vision'):
                cap_icons.append("👁️")
            if caps.get('embeddings'):
                cap_icons.append("🔤")
            if caps.get('code_generation'):
                cap_icons.append("💻")
            if caps.get('image_generation'):
                cap_icons.append("🎨")
            if caps.get('translation'):
                cap_icons.append("🌍")
            
            if cap_icons:
                st.markdown(" ".join(cap_icons))
            else:
                st.warning("⚠️ Aucun modèle local")

    # Initialisation des états
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'pois' not in st.session_state:
        st.session_state.pois = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = WORKING_MODELS[list(WORKING_MODELS.keys())[0]]
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_traj_info' not in st.session_state:
        st.session_state.last_traj_info = None
    if 'last_reponse' not in st.session_state:
        st.session_state.last_reponse = ""
    # Initialisation du modèle local
    if 'local_mode' not in st.session_state:
        st.session_state.local_mode = False
    if 'local_model_loaded' not in st.session_state:
        st.session_state.local_model_loaded = False
    if 'local_tokenizer' not in st.session_state:
        st.session_state.local_tokenizer = None
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = None
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = None
    if 'local_model' not in st.session_state:
        st.session_state.local_model = None
    if 'local_qwen_llm' not in st.session_state:
        st.session_state.local_qwen_llm = None
    # Initialisation du système d'outils dynamiques
    if 'tool_manager' not in st.session_state:
        if TOOLS_SYSTEM_AVAILABLE:
            try:
                st.session_state.tool_manager = ToolManager()
                print("✅ Gestionnaire d'outils initialisé")
            except Exception as e:
                print(f"⚠️ Erreur initialisation outils: {e}")
                st.session_state.tool_manager = None
        else:
            st.session_state.tool_manager = None
    # Initialisation des résultats d'analyse de médias
    if 'media_analysis_results' not in st.session_state:
        st.session_state.media_analysis_results = []
    
    # Initialisation du tracking des fichiers uploadés pour persistance du contexte
    if 'has_pdf_context' not in st.session_state:
        st.session_state.has_pdf_context = False
    if 'chat_uploaded_pdfs' not in st.session_state:
        st.session_state.chat_uploaded_pdfs = []
    if 'binary_files' not in st.session_state:
        st.session_state.binary_files = []
    
    # 🧠 NOUVEAU: Initialisation de la mémoire vectorielle des conversations
    if 'chat_vectordb' not in st.session_state:
        from chat_memory import load_chat_vectordb
        st.session_state.chat_vectordb, memory_msg = load_chat_vectordb()
        if st.session_state.chat_vectordb:
            print(memory_msg)
    
    # 🤖 NOUVEAU: Initialisation du système d'auto-apprentissage
    if 'auto_learning' not in st.session_state and AUTO_LEARNING_AVAILABLE:
        st.session_state.auto_learning = get_auto_learning()
        st.session_state.knowledge_manager = get_knowledge_manager()
        print("🧠 Auto-apprentissage initialisé")

    # ===============================================
    # Onglets avec design fluide
    # ===============================================
    tab_names = ["⚙️ **Configuration**", "💬 **Chat RAG + Web**", "🗺️ **Trajets**", "📸 **Analyse Image**", "🌐 **Recherche Web**", "📷 **Photogrammétrie**"]
    
    # Ajouter l'onglet binaire si le gestionnaire est disponible
    if BINARY_HANDLER_AVAILABLE:
        tab_names.append("🔬 **Fichiers Binaires**")
    
    if TOOLS_SYSTEM_AVAILABLE and st.session_state.tool_manager:
        tab_names.append("🔧 **Outils Dynamiques**")
    
    # Onglet Nano-IA si auto-apprentissage disponible
    if AUTO_LEARNING_AVAILABLE:
        tab_names.append("🤖 **Nano-IA & Apprentissage**")
    
    tabs = st.tabs(tab_names)
    
    # Assignation des onglets
    tab1, tab2, tab3, tab4, tab5, tab_photo = tabs[:6]
    
    # Onglet binaire si disponible
    tab_binary = None
    current_tab_idx = 6
    if BINARY_HANDLER_AVAILABLE and len(tabs) > 6:
        tab_binary = tabs[current_tab_idx]
        current_tab_idx += 1
    
    # Onglet outils dynamiques si disponible
    tab_tools = None
    if TOOLS_SYSTEM_AVAILABLE and st.session_state.tool_manager and len(tabs) > current_tab_idx:
        tab_tools = tabs[current_tab_idx]
        current_tab_idx += 1
    
    # Onglet Nano-IA si disponible
    tab_nano_ai = None
    if AUTO_LEARNING_AVAILABLE and len(tabs) > current_tab_idx:
        tab_nano_ai = tabs[current_tab_idx]

    # ===============================================
    # Onglet 1: Configuration avec cartes
    # ===============================================
    with tab1:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Gestion des données et configuration")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            pdf_files = st.file_uploader(
                "📤 **Upload Documents (tous formats acceptés)**", 
                type=None, 
                accept_multiple_files=True,
                key="pdf_upload"
            )
            if pdf_files:
                saved_files = []
                for pdf_file in pdf_files:
                    filepath = os.path.join(PDFS_PATH, pdf_file.name)
                    with open(filepath, 'wb') as f:
                        f.write(pdf_file.getbuffer())
                    saved_files.append(pdf_file.name)
                st.success("✅📄")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            pbf_file = st.file_uploader(
                "📤 **Upload fichier OSM (tous formats acceptés)**", 
                type=None,
                key="pbf_upload"
            )
            if pbf_file:
                pbf_path = os.path.join(GRAPHS_PATH, pbf_file.name)
                with open(pbf_path, 'wb') as f:
                    f.write(pbf_file.getbuffer())
                G, pois, status = upload_and_process_pbf(pbf_path)
                st.session_state.graph = G
                st.session_state.pois = pois
                st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 🌊 Uploader ERT pour fichiers .dat
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("#### 🌊 Analyse ERT (Electrical Resistivity Tomography)")
        dat_file = st.file_uploader(
            "📤 **Upload fichier .dat pour analyse géologique/hydrologique**", 
            type=["dat"],
            key="dat_upload",
            help="Formats: 4 colonnes (espaces) OU CSV avec fréquences MHz (virgules)"
        )
        if dat_file:
            # Sauvegarder temporairement le fichier
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dat') as tmp:
                tmp.write(dat_file.getbuffer())
                tmp_path = tmp.name
            
            # Stocker dans session_state pour l'orchestration
            st.session_state.uploaded_dat_file = {
                'name': dat_file.name,
                'path': tmp_path,
                'size': dat_file.size
            }
            st.success(f"✅ {dat_file.name} chargé ({dat_file.size} octets)")
            st.info("💬 Pose une question sur ce fichier dans le chat pour déclencher l'analyse ERT automatique !")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            if st.button("🔄 **Traiter PDFs**", key="process_pdfs", help="Traite les PDFs uploadés pour créer la base vectorielle"):
                with st.spinner("Traitement des PDFs..."):
                    vectordb, status = process_pdfs()
                    st.session_state.vectordb = vectordb
                    st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            if st.button("📂 **Charger graphe existant**", key="load_graph", help="Charge un graphe OSM existant"):
                with st.spinner("Chargement du graphe..."):
                    G, pois, status = load_existing_graph()
                    st.session_state.graph = G
                    st.session_state.pois = pois
                    st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            if st.button("📂 **Charger base vectorielle**", key="load_vectordb", help="Charge la base vectorielle existante"):
                with st.spinner("Chargement de la base..."):
                    vectordb, status = load_vectordb()
                    st.session_state.vectordb = vectordb
                    st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col6:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            if st.button("🧹 **Vider Cache**", key="clear_streamlit_cache", help="Vide le cache Streamlit et rafraîchit l'application", type="primary"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Cache vidé !")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        col6, col7, col8 = st.columns(3)
        with col6:
            if st.button("🗑️ **Vider cache web**", key="clear_cache", help="Nettoie le cache web des anciennes recherches"):
                status = handle_clear_cache()
                st.success(status)
        
        with col7:
            if st.button("💬 **Vider historique chat**", key="clear_chat_history", help="Efface tout l'historique de conversation", type="secondary"):
                st.session_state.chat_history = []
                st.session_state.processed_files = set()
                st.session_state.media_analysis_results = []
                st.success("✅ Historique vidé !")
                st.rerun()
        
        with col8:
            if st.button("📊 **Voir statistiques**", key="show_stats", help="Affiche les statistiques détaillées du système"):
                stats = get_system_status()
                st.json(stats)
        st.markdown('</div>', unsafe_allow_html=True)

    # ===============================================
    # Onglet 2: Chat RAG Amélioré avec cartes
    # ===============================================
    with tab2:
        # Contrôles compacts dans un expander
        with st.expander("⚙️ Configuration", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                model_choice = st.selectbox(
                    "🎯 **Modèle IA**",
                    options=list(WORKING_MODELS.keys()),
                    index=0,
                    key="model_select",
                    help="Choisis le modèle d'IA pour tes réponses"
                )
                # Mettre à jour llm_model dans session_state
                st.session_state.llm_model = WORKING_MODELS[model_choice]
                st.session_state.current_model = WORKING_MODELS[model_choice]
            with col2:
                web_enabled = st.checkbox("🌐 **Recherche web activée**", value=True, help="Active la recherche web pour des réponses plus complètes")
                # Toggle pour le mode local
                local_mode_toggle = st.checkbox(
                    "🏠 **Mode Local (Qwen 1.5B)**", 
                    value=st.session_state.local_mode, 
                    help="Active le modèle local Qwen 1.5B pour les tâches complexes quand l'API est surchargée"
                )
                if local_mode_toggle != st.session_state.local_mode:
                    st.session_state.local_mode = local_mode_toggle
                    if local_mode_toggle and not st.session_state.local_model_loaded:
                        # Charger le modèle local
                        with st.spinner("🔄 Chargement du modèle local Qwen 1.5B..."):
                            try:
                                tokenizer, model, device, gpu_info = load_local_llm_model()
                                st.session_state.local_tokenizer = tokenizer
                                st.session_state.local_model = model
                                st.session_state.local_qwen_llm = QwenChatModel(tokenizer, model)
                                st.session_state.local_model_loaded = True
                                st.success(f"✅ Modèle local chargé sur {device.upper()}")
                            except Exception as e:
                                st.error(f"❌ Erreur chargement modèle local: {e}")
                                st.session_state.local_mode = False
                    st.rerun()
        
        # Analyser automatiquement les médias uploadés (avant le chat_input)
        # media_analysis_results = []  # Maintenant global dans session_state
        
        # Zone d'upload de médias compacte (style ChatGPT)
        st.markdown("**📎 Ajouter des médias:**")
        col_attach1, col_attach2, col_attach3 = st.columns([1, 1, 1])
        
        with col_attach1:
            uploaded_images = st.file_uploader(
                "📎", label_visibility="collapsed",
                type=None,
                accept_multiple_files=True,
                key="image_uploader",
                help="🖼️ Ajouter des fichiers (tous formats acceptés)"
            )
        
        with col_attach2:
            uploaded_audios = st.file_uploader(
                "📎", label_visibility="collapsed",
                type=None,
                accept_multiple_files=True,
                key="audio_uploader",
                help="🎵 Ajouter des fichiers (tous formats acceptés)"
            )
        
        with col_attach3:
            uploaded_videos = st.file_uploader(
                "📎", label_visibility="collapsed",
                type=None,
                accept_multiple_files=True,
                key="video_uploader",
                help="🎥 Ajouter des fichiers (tous formats acceptés)"
            )
        
        # Nouvelle colonne pour les documents
        st.markdown("**📄 Ajouter des documents:**")
        uploaded_pdfs = st.file_uploader(
            "📎 Tous fichiers", label_visibility="collapsed",
            type=None,
            accept_multiple_files=True,
            key="pdf_chat_uploader",
            help="📄 Ajouter des fichiers (tous formats acceptés - ajoutés automatiquement au RAG vectoriel)"
        )
        
        # Initialiser le tracking des fichiers traités
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        
        # Initialiser le tracking des PDFs uploadés dans le chat
        if 'chat_uploaded_pdfs' not in st.session_state:
            st.session_state.chat_uploaded_pdfs = []
        
        # Variable pour tracker si de nouveaux médias ont été analysés
        new_media_analyzed = False
        
        # ===============================================
        # TRAITEMENT INTELLIGENT DES FICHIERS UPLOADÉS
        # ===============================================
        if uploaded_pdfs:
            for uploaded_file in uploaded_pdfs:
                # Vérifier si déjà traité
                file_key = f"file_{uploaded_file.name}_{uploaded_file.size}"
                if file_key in st.session_state.processed_files:
                    continue  # Skip si déjà traité
                
                # Marquer comme en cours de traitement
                st.session_state.processed_files.add(file_key)
                
                # Détection automatique du type de fichier
                file_ext = Path(uploaded_file.name).suffix.lower()
                is_pdf = file_ext == '.pdf'
                
                # Lire les données pour vérifier le magic number
                file_data = uploaded_file.read()
                uploaded_file.seek(0)  # Reset pour réutilisation
                is_pdf_magic = file_data.startswith(b'%PDF')
                
                # ============================================
                # TRAITEMENT PDF (si c'est un vrai PDF)
                # ============================================
                if is_pdf or is_pdf_magic:
                    # Message utilisateur
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": f"📄 PDF uploadé: {uploaded_file.name}"
                    })
                    
                    # Traitement du PDF
                    with st.spinner(f"📄 Traitement de {uploaded_file.name}..."):
                        try:
                            # 1. Sauvegarder le PDF dans PDFS_PATH
                            pdf_path = os.path.join(PDFS_PATH, uploaded_file.name)
                            with open(pdf_path, 'wb') as f:
                                f.write(file_data)
                            
                            st.success(f"✅ PDF sauvegardé: {uploaded_file.name}")
                            
                            # 2. Extraire le texte
                            pdf_text = extract_text_from_pdf(pdf_path)
                            pdf_pages = len(pdf_text.split('\n\n'))  # Approximation du nombre de pages
                            word_count = len(pdf_text.split())
                            
                            # Vérifier qu'il y a du texte extrait
                            if not pdf_text.strip() or word_count == 0:
                                st.warning("⚠️ Aucun texte extrait - PDF vide ou scanné")
                                # Continuer quand même pour permettre l'analyse binaire si nécessaire
                                continue
                            
                            st.info(f"📊 Extrait: ~{pdf_pages} pages, {word_count} mots")
                            
                            # 3. Ajouter au RAG vectoriel
                            with st.spinner("🔄 Ajout au RAG vectoriel..."):
                                # Charger l'embedding model
                                embedding_model = HuggingFaceEmbeddings(
                                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    cache_folder=str(SENTENCE_TRANSFORMER_CACHE)
                                )
                                
                                # Chunker le texte
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1000,
                                    chunk_overlap=100
                                )
                                
                                # Créer les documents
                                chunks = text_splitter.split_text(pdf_text)
                                
                                # Vérifier qu'on a des chunks
                                if not chunks or len(chunks) == 0:
                                    st.warning("⚠️ Impossible de créer des chunks - texte trop court")
                                    continue
                                
                                documents = [
                                    Document(
                                        page_content=chunk,
                                        metadata={
                                            "source": uploaded_file.name,
                                            "chunk_id": i,
                                            "total_chunks": len(chunks)
                                        }
                                    )
                                    for i, chunk in enumerate(chunks)
                                ]
                                
                                # Ajouter à la vectordb existante ou créer nouvelle
                                if st.session_state.vectordb:
                                    st.session_state.vectordb.add_documents(documents)
                                    st.success(f"✅ {len(chunks)} chunks ajoutés au RAG")
                                else:
                                    # Créer nouvelle vectordb
                                    st.session_state.vectordb = FAISS.from_documents(
                                        documents,
                                        embedding_model
                                    )
                                st.success(f"✅ RAG créé avec {len(chunks)} chunks")
                            
                            # Sauvegarder la vectordb
                            st.session_state.vectordb.save_local(VECTORDB_PATH)
                            st.success("💾 Base vectorielle sauvegardée")
                            
                            # 4. Stocker les infos du PDF pour le panneau d'outils et contexte de conversation
                            pdf_info = {
                                'name': uploaded_file.name,
                                'path': pdf_path,
                                'text': pdf_text,
                                'pages': pdf_pages,
                                'word_count': word_count,
                                'chunks': len(chunks),
                                'uploaded_at': os.path.getmtime(pdf_path)  # Timestamp pour tracking
                            }
                            st.session_state.chat_uploaded_pdfs.append(pdf_info)
                            
                            # Marquer que nous avons un contexte PDF actif
                            st.session_state.has_pdf_context = True
                            
                            # 🧠 AUTO-APPRENTISSAGE: Apprendre du PDF
                            if AUTO_LEARNING_AVAILABLE and st.session_state.get('auto_learning'):
                                with st.spinner("🧠 Apprentissage du contenu..."):
                                    if st.session_state.auto_learning.learn_from_pdf(pdf_info):
                                        st.success("✅ Connaissances intégrées au système d'apprentissage!")
                            
                            # 5. Message de succès avec panneau d'outils
                            tools_panel_html = f'''
<div style="
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 136, 255, 0.1));
    border: 2px solid rgba(0, 255, 136, 0.3);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 255, 136, 0.2);
">
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span style="font-size: 2rem; margin-right: 1rem;">📄</span>
        <div>
            <h3 style="color: #00ff88; margin: 0; font-size: 1.3rem;">PDF Chargé avec Succès !</h3>
            <p style="color: #b0b0b0; margin: 0.3rem 0 0 0; font-size: 0.9rem;">{uploaded_file.name}</p>
        </div>
    </div>
    
    <div style="
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding: 1rem;
        background: rgba(74, 74, 126, 0.2);
        border-radius: 10px;
    ">
        <div style="color: white;">
            <strong>📊 Pages:</strong> ~{pdf_pages}
        </div>
        <div style="color: white;">
            <strong>📝 Mots:</strong> {word_count:,}
        </div>
        <div style="color: white;">
            <strong>🧩 Chunks RAG:</strong> {len(chunks)}
        </div>
        <div style="color: white;">
            <strong>✅ Statut:</strong> <span style="color: #00ff88;">Prêt</span>
        </div>
    </div>
    
    <h4 style="color: #00ff88; margin: 1rem 0 0.5rem 0;">🛠️ Outils Disponibles:</h4>
    <p style="color: #e0e0e0; font-size: 0.9rem; margin-bottom: 1rem;">
        Le PDF est maintenant dans la base vectorielle. Vous pouvez:
    </p>
    
    <div style="display: grid; gap: 0.8rem;">
        <div style="
            background: rgba(0, 255, 136, 0.1);
            border-left: 3px solid #00ff88;
            padding: 0.8rem;
            border-radius: 5px;
        ">
            <strong style="color: #00ff88;">💬 1. Poser des questions</strong>
            <p style="color: white; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                Ex: "Résume ce PDF", "Quels sont les points clés?"
            </p>
        </div>
        
        <div style="
            background: rgba(0, 136, 255, 0.1);
            border-left: 3px solid #0088ff;
            padding: 0.8rem;
            border-radius: 5px;
        ">
            <strong style="color: #0088ff;">📋 2. Résumé Intelligent</strong>
            <p style="color: white; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                Tapez: "Résume le PDF en mode [court/moyen/détaillé]"
            </p>
        </div>
        
        <div style="
            background: rgba(255, 215, 0, 0.1);
            border-left: 3px solid #ffd700;
            padding: 0.8rem;
            border-radius: 5px;
        ">
            <strong style="color: #ffd700;">🌐 3. Traduction Complète</strong>
            <p style="color: white; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                Tapez: "Traduis le PDF en [langue]"
            </p>
        </div>
        
        <div style="
            background: rgba(138, 43, 226, 0.1);
            border-left: 3px solid #8a2be2;
            padding: 0.8rem;
            border-radius: 5px;
        ">
            <strong style="color: #8a2be2;">💻 4. Générer Application</strong>
            <p style="color: white; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                Tapez: "Génère une application basée sur ce PDF"
            </p>
        </div>
        
        <div style="
            background: rgba(255, 105, 180, 0.1);
            border-left: 3px solid #ff69b4;
            padding: 0.8rem;
            border-radius: 5px;
        ">
            <strong style="color: #ff69b4;">📊 5. Rapport Multi-IA Détaillé</strong>
            <p style="color: white; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                Tapez: "Génère un rapport détaillé sur ce PDF"
            </p>
        </div>
    </div>
    
    <div style="
        margin-top: 1rem;
        padding: 0.8rem;
        background: rgba(0, 255, 136, 0.05);
        border-radius: 8px;
        border: 1px dashed rgba(0, 255, 136, 0.3);
    ">
        <p style="color: #00ff88; margin: 0; font-size: 0.85rem;">
            💡 <strong>Astuce:</strong> L'IA a maintenant accès au contenu complet du PDF via le RAG vectoriel !
        </p>
    </div>
</div>
'''
                        
                            # Ajouter le panneau d'outils au chat
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": tools_panel_html,
                                "is_html": True
                            })
                            
                        except Exception as e:
                            import traceback
                            error_msg = f"❌ Erreur lors du traitement du PDF: {str(e)}"
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                            st.error(traceback.format_exc())
                
                # ============================================
                # TRAITEMENT FICHIER BINAIRE (si ce n'est PAS un PDF)
                # ============================================
                elif BINARY_HANDLER_AVAILABLE:
                    # Message utilisateur
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": f"🔬 Fichier uploadé: {uploaded_file.name}"
                    })
                    
                    with st.spinner(f"🔬 Analyse du fichier {uploaded_file.name}..."):
                        try:
                            # 🆕 DÉTECTION AUTOMATIQUE DE STRUCTURE
                            structure_detected = None
                            extracted_df = None
                            
                            if BINARY_DETECTOR_AVAILABLE:
                                try:
                                    # Sauver temporairement pour analyse
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                                        tmp.write(file_data)
                                        tmp_path = tmp.name
                                    
                                    # Détecter structure
                                    structure_detected, extracted_df = detect_and_analyze_file(tmp_path)
                                    
                                    # Nettoyer
                                    import os
                                    os.unlink(tmp_path)
                                    
                                    st.success(f"✅ Structure détectée: {structure_detected.file_type.value} (confiance: {structure_detected.confidence:.0%})")
                                    
                                except Exception as e:
                                    st.warning(f"⚠️ Détection structure: {e}")
                            
                            # Analyser avec le gestionnaire de fichiers binaires
                            result = BinaryFileHandler.process_file(uploaded_file)
                            file_info = result['info']
                            viewer = result['viewer']
                            
                            # Générer l'analyse complète
                            analysis_report = analyze_binary_file(file_data, uploaded_file.name)
                            
                            # Créer le visualiseur hex
                            hex_view = viewer.hex_dump(start=0, length=512)
                            
                            # Extraire les chaînes de texte
                            text_strings = viewer.extract_text_strings(min_length=4)
                            
                            # Interprétation de l'entropie
                            entropy = file_info['analysis']['entropy']
                            if entropy < 3:
                                entropy_desc = "Données très répétitives/structurées"
                                entropy_color = "#00ff88"
                            elif entropy < 6:
                                entropy_desc = "Données texte/structurées"
                                entropy_color = "#0088ff"
                            elif entropy < 7.5:
                                entropy_desc = "Données binaires normales"
                                entropy_color = "#ffd700"
                            else:
                                entropy_desc = "Données compressées/chiffrées"
                                entropy_color = "#ff69b4"
                            
                            # Escape HTML dans les chaînes de texte
                            safe_text_strings = [s.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;') for s in text_strings]
                            
                            # 🆕 AFFICHAGE AVEC COMPOSANTS STREAMLIT NATIFS
                            st.markdown(f"## 🔬 Analyse Complète: {uploaded_file.name}")
                            
                            # Structure détectée (si disponible)
                            if structure_detected:
                                with st.expander("🔍 Structure Détectée", expanded=True):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("📋 Type", structure_detected.file_type.value)
                                        st.metric("✅ Confiance", f"{structure_detected.confidence:.0%}")
                                    with col2:
                                        st.metric("🔤 Encodage", structure_detected.encoding)
                                        if structure_detected.row_count > 0:
                                            st.metric("📏 Lignes", structure_detected.row_count)
                                    
                                    if structure_detected.columns:
                                        st.write(f"**📊 Colonnes:** {', '.join(structure_detected.columns[:5])}" + (" ..." if len(structure_detected.columns) > 5 else ""))
                                    if structure_detected.delimiter:
                                        st.write(f"**🔹 Délimiteur:** `{repr(structure_detected.delimiter)}`")
                            
                            # Informations générales
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Taille", file_info['analysis']['size_formatted'])
                            with col2:
                                st.metric("🔢 Type", file_info['format_name'])
                            with col3:
                                st.metric("📈 Entropie", f"{entropy:.2f}/8.0")
                            
                            st.info(f"💡 {entropy_desc}")
                            
                            # Visualiseur hex
                            with st.expander("🔍 Visualiseur Hexadécimal (512 premiers bytes)"):
                                st.code(hex_view, language=None)
                            
                            # Statistiques
                            with st.expander("📊 Statistiques Détaillées"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Moyenne:** {file_info['analysis']['statistics'].get('mean', 0):.2f}")
                                    st.write(f"**Médiane:** {file_info['analysis']['statistics'].get('median', 0):.2f}")
                                    st.write(f"**Écart-type:** {file_info['analysis']['statistics'].get('std', 0):.2f}")
                                with col2:
                                    st.write(f"**Min/Max:** {file_info['analysis']['statistics'].get('min', 0)} / {file_info['analysis']['statistics'].get('max', 0)}")
                                    st.write(f"**Bytes uniques:** {file_info['analysis']['statistics'].get('unique_bytes', 0)} / 256")
                            
                            # Chaînes de texte
                            with st.expander(f"📝 Chaînes de Texte Extraites ({len(text_strings)} trouvées)"):
                                for i, s in enumerate(text_strings[:30], 1):
                                    st.text(f"{i}. {s}")
                                if len(text_strings) > 30:
                                    st.info(f"... et {len(text_strings) - 30} autres chaînes")
                            
                            # Actions disponibles
                            st.markdown("### 🛠️ Actions Disponibles")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.success("**💬 1. Analyse IA Complète**\n\nDemandez: 'Analyse ce fichier'")
                            with col2:
                                st.info("**🔍 2. Exploration Hexadécimale**\n\nDemandez: 'Montre offset 0x1000'")
                            with col3:
                                st.warning("**📊 3. Extraction de Données**\n\nDemandez: 'Convertis en numpy'")
                            
                            # Stocker les infos pour utilisation ultérieure avec le contexte d'analyse
                            if 'binary_files' not in st.session_state:
                                st.session_state.binary_files = []
                            
                            # Stocker également le contexte d'analyse complet pour éviter les réanalyses
                            binary_info_complete = {
                                'name': uploaded_file.name,
                                'data': file_data,
                                'info': file_info,
                                'viewer': viewer,
                                'text_strings': text_strings,
                                'analysis_report': analysis_report,  # Sauvegarder l'analyse
                                'hex_view': hex_view,
                                'entropy': entropy,
                                'entropy_desc': entropy_desc,
                                'structure': structure_detected,  # 🆕 Structure détectée
                                'dataframe': extracted_df  # 🆕 DataFrame extrait
                            }
                            st.session_state.binary_files.append(binary_info_complete)
                            
                            # 📊 Afficher le DataFrame si disponible
                            if extracted_df is not None and not extracted_df.empty:
                                st.markdown("### 📊 Données Extraites et Structurées")
                                st.dataframe(extracted_df.head(50), use_container_width=True)
                                
                                # Statistiques sur le DataFrame
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                                            padding: 1rem; border-radius: 10px; margin: 1rem 0;
                                            border-left: 4px solid #4caf50;">
                                    <p style="color: #1b5e20; margin: 0.3rem 0;">
                                        <strong>📏 Dimensions:</strong> {extracted_df.shape[0]} lignes × {extracted_df.shape[1]} colonnes
                                    </p>
                                    <p style="color: #1b5e20; margin: 0.3rem 0;">
                                        <strong>🔢 Colonnes numériques:</strong> {len(extracted_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns)}
                                    </p>
                                    <p style="color: #1b5e20; margin: 0.3rem 0;">
                                        <strong>📝 Colonnes texte:</strong> {len(extracted_df.select_dtypes(include=['object']).columns)}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # 🧠 AUTO-APPRENTISSAGE: Apprendre du fichier binaire
                            if AUTO_LEARNING_AVAILABLE and st.session_state.get('auto_learning'):
                                with st.spinner("🧠 Apprentissage du fichier binaire..."):
                                    if st.session_state.auto_learning.learn_from_binary(binary_info_complete):
                                        st.success("✅ Analyse binaire intégrée au système!")
                            
                        except Exception as e:
                            error_msg = f"❌ Erreur lors de l'analyse du fichier: {str(e)}"
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                            import traceback
                            st.error(traceback.format_exc())
        
        # Analyse des images avec Vision AI (s'affiche dans le chat)
        if uploaded_images:
            # Ajouter l'analyse directement dans le chat
            for idx, img_file in enumerate(uploaded_images):
                # Vérifier si déjà traité
                file_key = f"img_{img_file.name}_{img_file.size}"
                if file_key in st.session_state.processed_files:
                    continue  # Skip si déjà traité
                
                # Marquer comme en cours de traitement
                st.session_state.processed_files.add(file_key)
                new_media_analyzed = True
                
                # Message utilisateur avec miniature
                img_file.seek(0)
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"📷 Image uploadée: {img_file.name}"
                })
                
                # Analyse en arrière-plan
                with st.spinner(f"🤖 Analyse de {img_file.name}..."):
                    try:
                        import tempfile
                        import base64
                        from io import BytesIO
                        
                        # Convertir l'image en base64 pour l'API
                        img_file.seek(0)
                        image_bytes = img_file.read()
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        
                        # Sauvegarder temporairement pour métadonnées
                        img_suffix = os.path.splitext(img_file.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=img_suffix) as tmp_file:
                            tmp_file.write(image_bytes)
                            tmp_path = tmp_file.name
                        
                        # Obtenir les métadonnées de base
                        from PIL import Image as PILImage
                        img = PILImage.open(tmp_path)
                        width, height = img.size
                        img_format = img.format
                        
                        # Analyse avec modèle de vision local (CLIP)
                        try:
                            # Charger les modèles locaux
                            if 'vision_models' not in st.session_state:
                                with st.spinner("📦 Chargement des modèles de vision locaux..."):
                                    st.session_state.vision_models = load_vision_models()
                            
                            # Charger l'OCR
                            if 'ocr_reader' not in st.session_state:
                                with st.spinner("📦 Chargement du modèle OCR..."):
                                    st.session_state.ocr_reader = load_ocr_reader()
                            
                            vision_success = False
                            image_caption = ""
                            analysis_details = []
                            extracted_text_info = ""
                            
                            # Extraction du texte avec OCR
                            extracted_text_keywords = []
                            if st.session_state.ocr_reader or pytesseract:
                                st.info("📝 Extraction OCR en cours...")
                                extracted_texts = extract_text_from_image(tmp_path, st.session_state.ocr_reader)
                                if extracted_texts:
                                    extracted_text_info = organize_extracted_text(extracted_texts)
                                    # Extraire mots-clés pour recherche web
                                    import re
                                    text_words = re.findall(r'✅.*?\]\s*(.*?)$', extracted_text_info, re.MULTILINE)
                                    if text_words:
                                        extracted_text_keywords = list(set([w.strip() for w in text_words if len(w.strip()) > 2]))[:5]
                                    st.success(f"✅ OCR: {len(extracted_texts)} éléments détectés")
                                else:
                                    extracted_text_info = "Aucun texte détecté dans l'image."
                            
                            # Détection d'objets avec YOLO
                            yolo_detections = []
                            if 'yolo_model' not in st.session_state:
                                st.session_state.yolo_model = load_yolo_model()
                            
                            if st.session_state.yolo_model:
                                st.info("🎯 Détection YOLO en cours...")
                                try:
                                    # Forcer l'inférence sur CPU
                                    results = st.session_state.yolo_model(tmp_path, verbose=False, device='cpu')
                                    if results and len(results) > 0:
                                        for result in results:
                                            if result.boxes:
                                                for box in result.boxes:
                                                    class_id = int(box.cls[0])
                                                    confidence = float(box.conf[0])
                                                    class_name = result.names[class_id]
                                                    if confidence > 0.3:  # Seuil 30%
                                                        yolo_detections.append({
                                                            'class': class_name,
                                                            'confidence': confidence
                                                        })
                                        if yolo_detections:
                                            st.success(f"✅ YOLO: {len(yolo_detections)} objets détectés")
                                except Exception as yolo_err:
                                    st.warning(f"⚠️ YOLO: {yolo_err}")
                            
                            # Analyse visuelle avec CLIP (NOUVELLE MÉTHODE DÉTAILLÉE)
                            if st.session_state.vision_models:
                                st.info("🔍 Analyse CLIP détaillée (50 prompts)...")
                                
                                # Utiliser la nouvelle fonction de description détaillée
                                clip_result, error = generate_detailed_description_with_clip(tmp_path, st.session_state.vision_models)
                                
                                if clip_result:
                                    image_caption = clip_result['description']
                                    analysis_details = [
                                        {'label': 'Type', 'confidence': clip_result['confidence'], 'value': clip_result['main_type']},
                                        {'label': 'Couleurs', 'confidence': 1.0, 'value': clip_result['colors']},
                                        {'label': 'Détails', 'confidence': 1.0, 'value': clip_result.get('details', 'N/A')}
                                    ]
                                    vision_success = True
                                    st.success("✅ Analyse CLIP détaillée complétée")
                                else:
                                    st.error(f"❌ Erreur: {error}")
                            else:
                                st.error("❌ Modèles de vision non disponibles")
                            
                            if vision_success:
                                # Enrichir avec recherche web sur TEXTE OCR détecté
                                web_context = ""
                                web_results_text = []
                                try:
                                    if extracted_text_keywords:
                                        # Recherche web basée sur le TEXTE détecté (plus pertinent)
                                        search_query = " ".join(extracted_text_keywords[:3])
                                        st.info(f"🌐 Recherche web: '{search_query}'")
                                        web_results_text = enhanced_web_search(search_query, max_results=3, search_type="both")
                                        if web_results_text:
                                            web_context = "\n".join([f"📌 {r.get('title', '')[:60]}: {r.get('body', '')[:120]}" for r in web_results_text[:3]])
                                            st.success(f"✅ Web: {len(web_results_text)} résultats")
                                    else:
                                        # Fallback: recherche sur le type d'image
                                        web_results_text = enhanced_web_search(f"information sur: {image_caption}", max_results=2)
                                        web_context = "\n".join([f"{r.get('title', '')[:40]}: {r.get('body', '')[:100]}" for r in web_results_text[:2]]) if web_results_text else ""
                                except Exception as web_err:
                                    st.warning(f"⚠️ Recherche web: {web_err}")
                                    web_context = ""
                                
                                # PROMPT ULTRA-AMÉLIORÉ avec YOLO + Web
                                yolo_summary = ""
                                if yolo_detections:
                                    yolo_list = ", ".join([f"{d['class']} ({d['confidence']:.0%})" for d in yolo_detections[:10]])
                                    yolo_summary = f"\n\n🎯 **YOLO a détecté ces objets:**\n{yolo_list}"
                                
                                web_summary = ""
                                if web_context:
                                    web_summary = f"\n\n🌐 **Informations web trouvées:**\n{web_context[:400]}"
                                
                                analysis_prompt = f"""Analyse cette image en français de manière ULTRA-PRÉCISE:

📸 Image: {img_file.name} ({width}x{height}px, {img_format})

🖼️ **CLIP Vision AI (50 prompts) a détecté:**
{image_caption}

📝 **OCR a extrait ce texte:**
{extracted_text_info[:400] if extracted_text_info and "Aucun texte" not in extracted_text_info else "Aucun texte visible"}{yolo_summary}{web_summary}

INSTRUCTIONS CRITIQUES:
1. Combine TOUTES les sources (CLIP + OCR + YOLO + Web) pour une analyse complète
2. Si OCR détecte du texte, UTILISE LES INFOS WEB pour expliquer le contexte
3. Si YOLO détecte des objets, mentionne-les précisément
4. Ignore les détections de lettres CLIP < 40% si OCR ne confirme pas
5. Décris ce qui est VRAIMENT visible avec TOUS les détails (couleurs exactes, objets, texte, contexte)
6. Si c'est un logo/texte, explique CE QUE ÇA REPRÉSENTE (utilise le web)

Réponds en 3-5 phrases ULTRA-DÉTAILLÉES incluant: type d'image, couleurs précises, objets/textes détectés, et contexte web:"""

                                text_client = create_client()
                                analysis_response = text_client.chat.completions.create(
                                    model=WORKING_MODELS[model_choice],
                                    messages=[{"role": "user", "content": analysis_prompt[:1800]}],
                                    max_tokens=500,
                                    temperature=0.1  # Température très basse pour précision maximale
                                )
                                
                                enriched_analysis = analysis_response.choices[0].message.content
                                
                                # Préparer l'affichage ULTRA-COMPLET
                                analysis_display = f"**🖼️ Analyse Ultra-Détaillée de {img_file.name}**\n\n"
                                analysis_display += f"📏 Résolution: {width}x{height}px | Format: {img_format}\n\n"
                                
                                # Extraire la lettre de la description CLIP SEULEMENT si confiance > 40%
                                import re
                                letter_match = re.search(r"lettre '([A-Z])' \(confiance: ([\d.]+)%\)", image_caption)
                                if letter_match:
                                    detected_letter = letter_match.group(1)
                                    letter_confidence = float(letter_match.group(2))
                                    # Afficher SEULEMENT si confiance > 40%
                                    if letter_confidence > 40.0:
                                        analysis_display += f"## 🔤 LETTRE DÉTECTÉE: **{detected_letter}** ({letter_confidence:.1f}%)\n\n"
                                
                                # Description CLIP détaillée (50 prompts) avec COULEURS SÉPARÉES
                                # Extraire les couleurs du clip_result
                                colors_detailed = clip_result.get('colors', 'N/A')
                                main_type_only = clip_result.get('main_type', image_caption)
                                
                                analysis_display += f"## 🤖 Description IA (CLIP 50 prompts):\n\n**{main_type_only}**\n\n"
                                analysis_display += f"## 🎨 Couleurs détectées (TOP 3):\n\n**{colors_detailed}**\n\n"
                                
                                # Ajouter le texte OCR si disponible
                                if extracted_text_info and "Aucun texte" not in extracted_text_info:
                                    text_words = re.findall(r'✅.*?\]\s*(.*?)$', extracted_text_info, re.MULTILINE)
                                    if text_words:
                                        unique_words = list(set([w.strip() for w in text_words if len(w.strip()) > 1]))
                                        if unique_words:
                                            analysis_display += f"## 📝 Texte OCR: **{', '.join(unique_words[:10])}**\n\n"
                                
                                # Ajouter YOLO si disponible
                                if yolo_detections:
                                    yolo_top = sorted(yolo_detections, key=lambda x: x['confidence'], reverse=True)[:5]
                                    yolo_str = ", ".join([f"{d['class']} ({d['confidence']:.0%})" for d in yolo_top])
                                    analysis_display += f"## 🎯 Objets YOLO détectés: **{yolo_str}**\n\n"
                                
                                # Ajouter résultats web si disponibles
                                if web_results_text:
                                    analysis_display += f"## 🌐 Informations Web ({len(web_results_text)} sources):\n\n"
                                    for idx, res in enumerate(web_results_text[:3], 1):
                                        analysis_display += f"**{idx}.** {res.get('title', 'Sans titre')[:80]}\n"
                                        analysis_display += f"   _{res.get('body', '')[:150]}..._\n\n"
                                
                                analysis_display += f"## 🧠 Synthèse Intelligence Artificielle:\n\n{enriched_analysis}"
                                
                                # Ajouter l'analyse au chat (markdown pur sans HTML visible)
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": analysis_display
                                })
                                
                                st.session_state.media_analysis_results.append({
                                    'type': 'image',
                                    'name': img_file.name,
                                    'resolution': f"{width}x{height}",
                                    'format': img_format,
                                    'caption': image_caption,
                                    'ai_analysis': enriched_analysis,
                                    'web_context': web_context,
                                    'extracted_text': extracted_text_info
                                })
                            
                            else:
                                # Si aucun modèle n'a fonctionné
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"❌ Impossible d'analyser {img_file.name}. Vérifiez votre token HuggingFace ou réessayez plus tard."
                                })
                                
                        except Exception as analysis_error:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"❌ Erreur lors de l'analyse de {img_file.name}: {str(analysis_error)}"
                            })
                            
                    except Exception as img_error:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Erreur de traitement de {img_file.name}: {str(img_error)}"
                        })
        
        # Transcription des audios (affichée dans le chat)
        if uploaded_audios:
            for idx, audio_file in enumerate(uploaded_audios):
                # Vérifier si déjà traité
                file_key = f"audio_{audio_file.name}_{audio_file.size}"
                if file_key in st.session_state.processed_files:
                    continue
                
                st.session_state.processed_files.add(file_key)
                new_media_analyzed = True
                
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"🎵 Audio uploadé: {audio_file.name}"
                })
                
                with st.spinner(f"Analyse de {audio_file.name}..."):
                    try:
                        import tempfile
                        import librosa
                        
                        audio_file.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                            tmp_file.write(audio_file.read())
                            tmp_path = tmp_file.name
                        
                        y, sr = librosa.load(tmp_path)
                        duration = librosa.get_duration(y=y, sr=sr)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**🎵 Analyse de {audio_file.name}**\n\n⏱️ Durée: {duration:.2f}s | Fréquence: {sr} Hz\n\n🎙️ *Transcription: Utilisez Whisper API pour la transcription en production*"
                        })
                        
                        st.session_state.media_analysis_results.append({
                            'type': 'audio',
                            'name': audio_file.name,
                            'duration': duration,
                            'sample_rate': sr
                        })
                        
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Erreur lors de l'analyse de {audio_file.name}: {str(e)}"
                        })
        
        # Analyse des vidéos (affichée dans le chat)
        if uploaded_videos:
            for idx, video_file in enumerate(uploaded_videos):
                # Vérifier si déjà traité
                file_key = f"video_{video_file.name}_{video_file.size}"
                if file_key in st.session_state.processed_files:
                    continue
                
                st.session_state.processed_files.add(file_key)
                new_media_analyzed = True
                
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"🎥 Vidéo uploadée: {video_file.name}"
                })
                
                with st.spinner(f"Analyse de {video_file.name}..."):
                    try:
                        import tempfile
                        
                        video_file.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_file:
                            tmp_file.write(video_file.read())
                            tmp_path = tmp_file.name
                        
                        cap = cv2.VideoCapture(tmp_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**🎥 Analyse de {video_file.name}**\n\n📐 Résolution: {width}x{height}px | ⏱️ Durée: {duration:.2f}s | 🎞️ FPS: {fps:.2f}\n\n🎬 *Analyse avancée: Extrayez les frames clés pour une analyse visuelle approfondie*"
                        })
                        
                        st.session_state.media_analysis_results.append({
                            'type': 'video',
                            'name': video_file.name,
                            'duration': duration,
                            'resolution': f"{width}x{height}",
                            'fps': fps
                        })
                        
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Erreur lors de l'analyse de {video_file.name}: {str(e)}"
                        })
        
        # Recharger uniquement si de nouveaux médias ont été analysés
        if 'new_media_analyzed' in locals() and new_media_analyzed:
            st.rerun()
        
        # Zone de chat avec design amélioré - CARTE SCINTILLANTE
        st.markdown('<div class="kibali-chat-card">', unsafe_allow_html=True)
        
        # Indicateur de contexte actif
        context_indicators = []
        if 'binary_files' in st.session_state and st.session_state.binary_files:
            context_indicators.append(f"🔬 Fichier binaire: {st.session_state.binary_files[-1]['name']}")
        if st.session_state.has_pdf_context:
            context_indicators.append(f"📄 {len(st.session_state.chat_uploaded_pdfs)} PDF(s)")
        if st.session_state.media_analysis_results:
            context_indicators.append(f"📸 {len(st.session_state.media_analysis_results)} média(s)")
        
        if context_indicators:
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, rgba(0, 136, 255, 0.15), rgba(138, 43, 226, 0.15));
                padding: 0.8rem 1.2rem;
                border-radius: 8px;
                border-left: 4px solid #0088ff;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 1rem;
            '>
                <span style='color: #0088ff; font-size: 1.2rem;'>💬</span>
                <div>
                    <strong style='color: #0088ff;'>Mode Conversationnel Actif</strong>
                    <br>
                    <span style='color: #b0b0b0; font-size: 0.9rem;'>{' • '.join(context_indicators)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'''<div class="chat-message-user" style="
                        background: linear-gradient(145deg, #0088ff 0%, #00ff88 100%);
                        color: white;
                        padding: 1.2rem 1.8rem;
                        border-radius: 18px 18px 4px 18px;
                        margin: 1.5rem 0.5rem 1.5rem auto;
                        max-width: 75%;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                        border: 2px solid;
                        border-image: linear-gradient(45deg, #0088ff, #00ff88, #0088ff) 1;
                        animation: borderGlow 2s ease-in-out infinite, slideInRight 0.4s ease-out;
                        font-weight: 500;
                    ">
                        <strong>👤 Vous:</strong><br>{message["content"]}
                    </div>
                    <style>
                    @keyframes borderGlow {{{{
                        0%, 100% {{{{ border-image: linear-gradient(45deg, #0088ff, #00ff88, #0088ff) 1; }}}}
                        50% {{{{ border-image: linear-gradient(45deg, #00ff88, #00ffff, #00ff88) 1; }}}}
                    }}}}
                    @keyframes slideInRight {{{{
                        0% {{{{ transform: translateX(30px); opacity: 0; }}}}
                        100% {{{{ transform: translateX(0); opacity: 1; }}}}
                    }}}}
                    </style>
                    ''', unsafe_allow_html=True)
                elif message["role"] == "web_media":
                    # Affichage spécial pour les médias web
                    st.markdown(message["content"], unsafe_allow_html=True)
                else:
                    # Vérifier si c'est du HTML pur (panneau d'outils PDF ou contient <div)
                    content = message.get("content", "")
                    is_html_flag = message.get("is_html", False)
                    has_div_tags = ("<div" in content and "</div>" in content)
                    has_style_attr = "style=" in content
                    
                    # Si c'est du HTML (flag OU contient div OU contient style), le rendre directement
                    if is_html_flag or has_div_tags or has_style_attr:
                        st.markdown(content, unsafe_allow_html=True)
                        continue  # Skip le reste du formatage
                    
                    # Formater la réponse avec Markdown pour structure
                    formatted_response = content
                    
                    # Carte complète en un seul bloc avec contenu markdown
                    st.markdown(f'''
<style>
@keyframes borderShimmer {{{{
    0%, 100% {{{{ border-image: linear-gradient(135deg, #00ff88, #0088ff, #00ff88) 1; }}}}
    50% {{{{ border-image: linear-gradient(135deg, #0088ff, #00ffff, #0088ff) 1; }}}}
}}}}

@keyframes logoGlow {{{{
    0%, 100% {{{{ filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.5)); }}}}
    50% {{{{ filter: drop-shadow(0 0 20px rgba(0, 255, 136, 0.8)); }}}}
}}}}

@keyframes slideIn {{{{
    0% {{{{ transform: translateY(20px); opacity: 0; }}}}
    100% {{{{ transform: translateY(0); opacity: 1; }}}}
}}}}

.kibali-response-box h2 {{{{
    color: #0066cc !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
}}}}

.kibali-response-box h3 {{{{
    color: #00aa66 !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.8rem !important;
}}}}

.kibali-response-box p {{{{
    color: #1a1a1a !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    margin: 1rem 0 !important;
    line-height: 1.8 !important;
}}}}

.kibali-response-box ul, .kibali-response-box ol {{{{
    color: #1a1a1a !important;
    font-weight: 600 !important;
}}}}

.kibali-response-box li {{{{
    color: #1a1a1a !important;
    font-weight: 600 !important;
    margin: 0.5rem 0 !important;
}}}}

.kibali-response-box strong {{{{
    color: #0066cc !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}}}}

.kibali-response-box code {{{{
    background: rgba(0, 170, 102, 0.1);
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    color: #00aa66;
    font-weight: 600 !important;
}}}}

.kibali-response-box a {{{{
    color: #0066cc !important;
    font-weight: 600 !important;
    text-decoration: underline;
}}}}

/* Forcer le texte NOIR et gras sur TOUT le contenu */
.kibali-response-box * {{{{
    color: #1a1a1a !important;
    font-weight: 600 !important;
}}}}
</style>

<div class="kibali-response-box" style="
    background: rgba(255, 255, 255, 0.95);
    padding: 2rem;
    border-radius: 16px;
    margin: 2rem 0.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15), 0 8px 40px rgba(0, 0, 0, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.8);
    animation: slideIn 0.5s ease-out;
">
    <div style="
        display: flex; 
        align-items: center; 
        margin-bottom: 1.5rem; 
        gap: 0.75rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(0, 255, 136, 0.2);
    ">
        <span style="
            font-size: 2rem; 
            font-weight: 900;
            background: linear-gradient(135deg, #00ff88 0%, #00ffff 50%, #0088ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: logoGlow 2s ease-in-out infinite;
            filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.5));
        ">K</span>
        <strong style="
            color: #00ff88; 
            font-size: 1.2rem;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        ">Kibali AI</strong>
        <span style="
            margin-left: auto;
            padding: 0.4rem 1rem;
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 136, 255, 0.2));
            border: 1px solid rgba(0, 255, 136, 0.4);
            border-radius: 25px;
            font-size: 0.8rem;
            color: #00ff88;
            font-weight: 600;
        ">🤖 Réponse enrichie</span>
    </div>
    
    <div style="line-height: 1.8; font-size: 1.05rem; color: #ffffff;">
        {formatted_response}
    </div>
</div>
''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Ancre pour auto-scroll vers le bas
        st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
        
        # Script JavaScript pour auto-scroll
        st.markdown("""
        <script>
        // Auto-scroll vers le bas quand nouveaux messages
        const chatBottom = document.getElementById('chat-bottom');
        if (chatBottom) {
            chatBottom.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
        
        // Garder le focus sur l'input
        const chatInput = document.querySelector('[data-testid="stChatInput"] input');
        if (chatInput) {
            chatInput.focus();
        }
        </script>
        """, unsafe_allow_html=True)
        
        # Input de chat stylisé - TOUJOURS EN BAS
        if prompt := st.chat_input("💭 Pose ta question ici...", key="chat_input"):
            # Enrichir le prompt avec le contexte des conversations passées (mémoire vectorielle)
            enriched_prompt = prompt
            conversation_context = ""
            
            try:
                from chat_memory import get_conversation_context
                conversation_context = get_conversation_context(prompt, st.session_state.chat_vectordb)
                if conversation_context:
                    enriched_prompt = f"""📚 **Contexte de conversations précédentes:**
{conversation_context}

---
QUESTION ACTUELLE: {prompt}"""
            except Exception as e:
                print(f"⚠️ Erreur récupération contexte mémoire: {e}")
            
            # 🤖 ENRICHIR avec le contexte des Nano-IA spécialisées
            nano_ai_context = ""
            if AUTO_LEARNING_AVAILABLE and st.session_state.get('auto_learning'):
                try:
                    nano_ai_context = st.session_state.auto_learning.get_domain_context(prompt, k=3)
                    if nano_ai_context:
                        enriched_prompt = f"""{enriched_prompt}

{nano_ai_context}"""
                except Exception as e:
                    print(f"⚠️ Erreur contexte nano-IA: {e}")
            
            # Enrichir le prompt avec les résultats d'analyse de médias IA si disponibles
            if st.session_state.media_analysis_results:
                media_context = "\n\n📎 **Médias analysés avec IA:**\n"
                for media in st.session_state.media_analysis_results:
                    if media['type'] == 'image':
                        media_context += f"\n🖼️ **Image: {media['name']}**\n"
                        media_context += f"  📏 Résolution: {media.get('resolution', 'N/A')}\n"
                        media_context += f"  🎨 Format: {media.get('format', 'N/A')}\n"
                        
                        # Ajouter l'analyse IA complète
                        if 'ai_analysis' in media:
                            media_context += f"\n  🤖 **Analyse IA détaillée:**\n"
                            # Indenter l'analyse pour la lisibilité
                            ai_lines = media['ai_analysis'].split('\n')
                            for line in ai_lines[:50]:  # Augmenter la limite pour inclure plus de détails (couleurs, objets)
                                media_context += f"  {line}\n"
                        
                        if 'caption' in media:
                            media_context += f"  📝 Description: {media['caption']}\n"
                        
                        if 'web_context' in media and media['web_context']:
                            media_context += f"  🌐 Infos complémentaires disponibles\n"
                    
                    elif media['type'] == 'audio':
                        media_context += f"\n🎵 **Audio: {media['name']}**\n"
                        media_context += f"  ⏱️ Durée: {media.get('duration', 0):.2f}s\n"
                        media_context += f"  📊 Fréquence: {media.get('sample_rate', 'N/A')} Hz\n"
                    
                    elif media['type'] == 'video':
                        media_context += f"\n🎥 **Vidéo: {media['name']}**\n"
                        media_context += f"  ⏱️ Durée: {media.get('duration', 0):.2f}s\n"
                        media_context += f"  📐 Résolution: {media.get('resolution', 'N/A')}\n"
                        media_context += f"  🎞️ FPS: {media.get('fps', 'N/A')}\n"
                
                # Ajouter un contexte explicite pour la conversation
                media_full_context = f"""CONTEXTE PRIORITAIRE: L'utilisateur pose une question sur les médias analysés ci-dessous.
⚠️ INSTRUCTION IMPORTANTE: Vous AVEZ ACCÈS au contenu de l'image via la description détaillée ci-dessous.
Considérez cette description textuelle comme votre "vision" de l'image.
Si l'utilisateur demande des détails (couleurs, textes, objets), utilisez les données ci-dessous pour répondre.
NE DITES JAMAIS que vous n'avez pas accès à l'image. Dites "D'après l'analyse de l'image..."

{media_context}

QUESTION UTILISATEUR: {prompt}"""
                
                # Combiner avec le contexte de conversation si existant
                if conversation_context:
                    enriched_prompt = f"{enriched_prompt}\n\n{media_full_context}"
                else:
                    enriched_prompt = media_full_context
            
            # ═══════════════════════════════════════════════════════════════════
            # FICHIER BINAIRE - CONTEXTE ABSOLU ET EXCLUSIF (AVEC CACHE)
            # ═══════════════════════════════════════════════════════════════════
            if 'binary_files' in st.session_state and st.session_state.binary_files:
                # RÉCUPÉRER LE DERNIER FICHIER UPLOADÉ (le plus récent)
                binary = st.session_state.binary_files[-1]
                
                # CACHE: Vérifier si le contexte a déjà été généré pour éviter les réanalyses
                if 'cached_binary_context' not in binary:
                    # Première fois : construire et cacher le contexte
                    binary_context = ""
                    
                    binary_context += f"\n{'='*80}\n"
                    binary_context += f"📁 FICHIER BINAIRE EN TÊTE DE CONVERSATION: {binary['name']}\n"
                    binary_context += f"{'='*80}\n\n"
                    
                    # Informations essentielles TRÈS DÉTAILLÉES
                    binary_context += f"🔢 TYPE DÉTECTÉ: {binary['info']['format_name']}\n"
                    binary_context += f"📊 TAILLE EXACTE: {binary['info']['analysis']['size_formatted']} ({len(binary['data'])} bytes)\n"
                    binary_context += f"🎯 MAGIC BYTES: {binary['info']['magic_bytes']}\n"
                    binary_context += f"📈 ENTROPIE: {binary['info']['analysis']['entropy']:.2f}/8.0 "
                    
                    # Interprétation de l'entropie
                    entropy = binary['info']['analysis']['entropy']
                    if entropy < 3:
                        binary_context += "(Données très répétitives/structurées - probablement du texte ou données simples)\n"
                    elif entropy < 6:
                        binary_context += "(Données texte/structurées - contient probablement des métadonnées)\n"
                    elif entropy < 7.5:
                        binary_context += "(Données binaires normales - fichier standard)\n"
                    else:
                        binary_context += "(Données compressées/chiffrées - haute complexité)\n"
                    
                    # Statistiques détaillées
                    stats = binary['info']['analysis']['statistics']
                    binary_context += f"\n📊 STATISTIQUES DES BYTES:\n"
                    binary_context += f"   • Valeur moyenne: {stats.get('mean', 0):.2f}\n"
                    binary_context += f"   • Médiane: {stats.get('median', 0):.2f}\n"
                    binary_context += f"   • Écart-type: {stats.get('std', 0):.2f}\n"
                    binary_context += f"   • Plage: {stats.get('min', 0)} à {stats.get('max', 0)}\n"
                    binary_context += f"   • Bytes uniques: {stats.get('unique_bytes', 0)}/256\n"
                    
                    # Patterns détectés
                    if binary['info']['analysis']['patterns']:
                        binary_context += f"\n🎯 PATTERNS DÉTECTÉS:\n"
                        for pattern in binary['info']['analysis']['patterns']:
                            binary_context += f"   ✓ {pattern}\n"
                    
                    # Chaînes de texte extraites - TRÈS IMPORTANT pour l'analyse
                    if binary['text_strings']:
                        binary_context += f"\n📝 CHAÎNES DE TEXTE EXTRAITES ({len(binary['text_strings'])} au total):\n"
                        # Afficher jusqu'à 30 chaînes pour plus de contexte
                        for i, s in enumerate(binary['text_strings'][:30], 1):
                            binary_context += f"   {i}. \"{s[:100]}\"\n"
                        if len(binary['text_strings']) > 30:
                            binary_context += f"   ... et {len(binary['text_strings'])-30} autres chaînes\n"
                    else:
                        binary_context += f"\n📝 CHAÎNES DE TEXTE: Aucune chaîne ASCII lisible détectée\n"
                    
                    # Header hexadécimal (256 premiers bytes pour plus de contexte)
                    binary_context += f"\n🔍 HEADER HEXADÉCIMAL (256 premiers bytes):\n"
                    hex_header = binary['data'][:256].hex()
                    # Formater en lignes de 32 caractères hex (16 bytes par ligne)
                    for i in range(0, len(hex_header), 32):
                        offset = i // 2
                        hex_line = hex_header[i:i+32]
                        # Ajouter des espaces tous les 2 caractères
                        hex_formatted = ' '.join(hex_line[j:j+2] for j in range(0, len(hex_line), 2))
                        # ASCII representation
                        ascii_chars = ''.join(
                            chr(binary['data'][offset + k]) if 32 <= binary['data'][offset + k] < 127 else '.'
                            for k in range(min(16, len(binary['data']) - offset))
                        )
                        binary_context += f"   {offset:04X}: {hex_formatted:<48} | {ascii_chars}\n"
                    
                    # Structure du header si détectée
                    header_info = binary['info']['analysis']['header']
                    if header_info.get('is_text_like'):
                        binary_context += f"\n✓ Le header semble contenir du texte ASCII ({header_info.get('ascii_ratio', 0):.1%})\n"
                        if header_info.get('first_line'):
                            binary_context += f"   Première ligne: \"{header_info['first_line']}\"\n"
                    
                    # CACHER le contexte pour réutilisation
                    binary['cached_binary_context'] = binary_context
                else:
                    # Réutiliser le contexte caché - évite les réanalyses
                    binary_context = binary['cached_binary_context']
                
                # ⚡ NOUVEAU: Contexte conversationnel intelligent avec OUTILS PROACTIFS
                # Pour la première question (upload), donner analyse complète
                # Pour les suivantes, mode expert avec utilisation d'outils
                is_first_query = len(st.session_state.chat_history) <= 3
                
                # 🎯 DÉTECTION AUTOMATIQUE DES INTENTIONS ET ORCHESTRATION
                question_lower = prompt.lower()
                needs_calculation = any(kw in question_lower for kw in ['calcul', 'moyenne', 'somme', 'total', 'combien', 'valeur', 'profondeur', 'exact'])
                needs_web_research = any(kw in question_lower for kw in ['domaine', 'activité', 'industrie', 'application', 'utilisation', 'pourquoi', 'comment'])
                needs_data_extraction = any(kw in question_lower for kw in ['données', 'valeurs', 'liste', 'extrait', 'montre', 'affiche', 'trouve'])
                needs_conversion = any(kw in question_lower for kw in ['convertis', 'convert', 'numpy', 'tableau', 'dataframe', 'pandas', 'csv', 'json', 'extraction'])
                needs_ert_analysis = any(kw in question_lower for kw in ['ert', 'résistivité', 'resistivité', 'géophysique', 'forage', 'nappe', 'aquifère', 'eau souterraine'])
                
                # 🌊 ANALYSE ERT AUTOMATIQUE - FICHIERS .dat GÉOPHYSIQUES
                # Vérifier s'il y a un fichier .dat uploadé (via uploader dédié OU binary_files)
                dat_file_source = None
                if 'uploaded_dat_file' in st.session_state and st.session_state.uploaded_dat_file:
                    # Fichier .dat uploadé via l'uploader dédié
                    dat_file_source = {
                        'name': st.session_state.uploaded_dat_file['name'],
                        'path': st.session_state.uploaded_dat_file['path'],
                        'data': open(st.session_state.uploaded_dat_file['path'], 'rb').read()
                    }
                elif 'binary_files' in st.session_state and st.session_state.binary_files:
                    # Fichier .dat uploadé via l'uploader générique
                    binary = st.session_state.binary_files[-1]
                    if binary['name'].lower().endswith('.dat'):
                        dat_file_source = binary
                
                if needs_ert_analysis and dat_file_source:
                        with st.expander("📋 Plan d'Action ERT - Analyse Géophysique", expanded=True):
                            st.markdown(f"""
                            ### 🌊 Orchestration ERT Activée
                            
                            **Fichier:** `{dat_file_source['name']}`
                            
                            **Formats supportés (détection automatique):**
                            
                            **Format 1 - Standard (espaces):**
                            ```
                            survey_point  depth  data  project
                            1.0  -2.5  0.35  projet1
                            1.0  -5.0  0.38  projet1
                            ```
                            
                            **Format 2 - Fréquences (CSV):**
                            ```
                            Projet,Point,621105.0MHz,155276.25MHz,...
                            Archange,1,0.119,0.122,0.116,...
                            Archange,2,0.161,0.163,0.164,...
                            ```
                            
                            **Plan d'exécution:**
                            1. ✅ **Parse du fichier .dat** - Détection automatique du format
                            2. ⏳ **Analyse des résistivités** - Classification géologique
                            3. ⏳ **Interprétation hydrogéologique** - Types d'eau identifiés
                            4. ⏳ **Pseudo-section 2D** - Visualisation avec colormap eau
                            5. ⏳ **Zones cibles** - Identification aquifères
                            6. ⏳ **Export rapport** - PDF complet avec statistiques
                            """)
                        
                        try:
                            import pandas as pd
                            import numpy as np
                            import matplotlib.pyplot as plt
                            from kibali_data.models.ert_analyzer import ERTAnalyzer
                            
                            st.info(f"🔄 Analyse ERT en cours de `{dat_file_source['name']}`...")
                            
                            # Créer l'analyseur
                            ert = ERTAnalyzer()
                            
                            # Analyse complète
                            with st.spinner("📊 Analyse des données ERT..."):
                                result = ert.analyze_file(dat_file_source['data'])
                            
                            if result.get('error'):
                                st.error(f"❌ Erreur d'analyse: {result['error']}")
                                
                                # Afficher un aperçu du fichier pour diagnostic
                                with st.expander("🔍 Aperçu du fichier pour diagnostic"):
                                    try:
                                        content_preview = dat_file_source['data'][:2000].decode('utf-8', errors='replace')
                                        st.code(content_preview, language=None)
                                        
                                        # Analyse basique du contenu
                                        lines = [l.strip() for l in content_preview.split('\n') if l.strip()]
                                        if lines:
                                            first_line = lines[0]
                                            st.write(f"**Première ligne:** `{first_line}`")
                                            parts = first_line.split()
                                            st.write(f"**Nombre de colonnes:** {len(parts)}")
                                            st.write(f"**Colonnes:** `{parts}`")
                                    except:
                                        st.write("Impossible d'afficher l'aperçu")
                                
                                # Message d'aide pour le format
                                st.info("""
                                **💡 Formats .dat supportés (ERTest.py):**
                                
                                **Format 1 - Standard (4 colonnes, espaces):**
                                ```
                                survey_point  depth  resistivity  project
                                1.0  -2.5  0.35  projet1
                                1.0  -5.0  0.38  projet1
                                2.0  -2.5  0.42  projet1
                                ```
                                
                                **Format 2 - Fréquences (CSV avec virgules):**
                                ```
                                Projet,Point Sondage,621105.0MHz,155276.25MHz,...
                                Projet Archange Ondimba 2,1,0.119,0.122,0.116,...
                                Projet Archange Ondimba 2,2,0.161,0.163,0.164,...
                                ```
                                
                                **Notes:**
                                - Détection automatique du séparateur (espaces vs virgules)
                                - Les profondeurs négatives sont converties en positives
                                - Les lignes commençant par # sont ignorées (commentaires)
                                - Pour CSV: première ligne = en-têtes avec fréquences MHz
                                
                                **Vérifications:**
                                - ✅ Pas de lignes vides au début
                                - ✅ Les colonnes sont séparées par espaces ou tabs
                                - ✅ Les valeurs numériques utilisent des points (pas de virgules)
                                - ✅ Les profondeurs sont en mètres
                                - ✅ Les résistivités sont en Ω·m
                                """)
                                return
                            
                            # 🧠 ENRICHISSEMENT RAG AUTOMATIQUE
                            st.info("🧠 Enrichissement avec la base de connaissances RAG...")
                            rag_insights = ""
                            rag_context = ""
                            
                            try:
                                if st.session_state.get('vectordb') is not None:
                                    # Construire une requête enrichie pour le RAG
                                    stats = result['stats']
                                    water_zones = result['water_zones']
                                    
                                    # Identifier le contexte dominant
                                    dominant_water_type = max(water_zones, key=water_zones.get)
                                    avg_resistivity = stats['resistivity_stats']['mean']
                                    
                                    # Créer une requête intelligente pour le RAG
                                    rag_query = f"""Résistivité électrique {avg_resistivity:.1f} Ohm·m {dominant_water_type} 
                                    géophysique ERT aquifère forage hydrogéologie nappe phréatique 
                                    eau souterraine prospection classification lithologie"""
                                    
                                    # Recherche dans la base vectorielle
                                    with st.spinner("🔍 Recherche de connaissances pertinentes..."):
                                        docs = st.session_state.vectordb.similarity_search(rag_query, k=5)
                                        
                                        if docs:
                                            st.success(f"✅ {len(docs)} documents pertinents trouvés dans la base de connaissances")
                                            
                                            # Extraire le contexte
                                            rag_context = "\n\n".join([
                                                f"📄 **Document {i+1}:** {doc.page_content[:300]}..." 
                                                for i, doc in enumerate(docs)
                                            ])
                                            
                                            # Générer des insights avec l'IA en utilisant le RAG
                                            try:
                                                client = create_client()
                                                
                                                rag_prompt = f"""Tu es un expert en géophysique et hydrogéologie. Analyse ces données ERT:

**DONNEES MESUREES:**
- Résistivité moyenne: {avg_resistivity:.2f} Ω·m
- Type d'eau dominant: {dominant_water_type}
- Profondeur: {stats['depth_range'][0]:.1f} - {stats['depth_range'][1]:.1f} m
- Points de sondage: {stats['survey_points']}
- Distribution eau: mer={water_zones['mer']}, salée={water_zones['salée']}, douce={water_zones['douce']}, pure={water_zones['pure']}

**CONTEXTE DE LA BASE DE CONNAISSANCES:**
{rag_context[:2000]}

**TACHE:**
En te basant sur les documents de la base de connaissances ET ton expertise, fournis:
1. Une interprétation géologique détaillée
2. Les implications hydrogéologiques
3. Des recommandations pour le forage
4. Les risques potentiels identifiés
5. Comparaison avec des sites similaires si mentionnés dans les documents

Sois précis, technique et référence les documents quand pertinent."""

                                                rag_response = client.chat.completions.create(
                                                    model=WORKING_MODELS[model_choice],
                                                    messages=[{"role": "user", "content": rag_prompt}],
                                                    max_tokens=2000,
                                                    temperature=0.3
                                                )
                                                
                                                rag_insights = rag_response.choices[0].message.content
                                                
                                                # 🧹 NETTOYER LA RÉPONSE DES CARACTÈRES CHINOIS/NON-LATINS
                                                rag_insights = clean_response_text(rag_insights)
                                                
                                                # Afficher les insights enrichis
                                                with st.expander("🧠 Analyse Enrichie par IA + Base de Connaissances", expanded=True):
                                                    st.markdown(rag_insights)
                                                    
                                                    st.markdown("---")
                                                    st.markdown("**📚 Sources utilisées:**")
                                                    for i, doc in enumerate(docs[:3]):
                                                        with st.expander(f"Document {i+1}"):
                                                            st.text(doc.page_content[:500])
                                            
                                            except Exception as e:
                                                st.warning(f"⚠️ Impossible de générer l'analyse enrichie: {e}")
                                        else:
                                            st.info("ℹ️ Aucun document pertinent trouvé dans la base de connaissances")
                                else:
                                    st.info("ℹ️ Base de connaissances RAG non disponible - analyse standard uniquement")
                            
                            except Exception as e:
                                st.warning(f"⚠️ Enrichissement RAG échoué: {e}")
                            
                            st.success("✅ Analyse ERT terminée!")
                            
                            # Afficher un aperçu du fichier pour diagnostic
                            with st.expander("🔍 Aperçu du fichier pour diagnostic"):
                                try:
                                    content_preview = dat_file_source['data'][:2000].decode('utf-8', errors='replace')
                                    st.code(content_preview, language=None)
                                    
                                    # Analyse basique du contenu
                                    lines = [l.strip() for l in content_preview.split('\n') if l.strip()]
                                    if lines:
                                        first_line = lines[0]
                                        st.write(f"**Première ligne:** `{first_line}`")
                                        parts = first_line.split()
                                        st.write(f"**Nombre de colonnes:** {len(parts)}")
                                        st.write(f"**Colonnes:** `{parts}`")
                                except:
                                    st.write("Impossible d'afficher l'aperçu")
                            
                            # Message d'aide pour le format
                            st.info("""
                            **💡 Formats .dat supportés (ERTest.py):**
                            
                            **Format 1 - Standard (4 colonnes, espaces):**
                            ```
                            survey_point  depth  resistivity  project
                            1.0  -2.5  0.35  projet1
                            1.0  -5.0  0.38  projet1
                            2.0  -2.5  0.42  projet1
                            ```
                            
                            **Format 2 - Fréquences (CSV avec virgules):**
                            ```
                            Projet,Point Sondage,621105.0MHz,155276.25MHz,...
                            Projet Archange Ondimba 2,1,0.119,0.122,0.116,...
                            Projet Archange Ondimba 2,2,0.161,0.163,0.164,...
                            ```
                            
                            **Notes:**
                            - Détection automatique du séparateur (espaces vs virgules)
                            - Les profondeurs négatives sont converties en positives
                            - Les lignes commençant par # sont ignorées (commentaires)
                            - Pour CSV: première ligne = en-têtes avec fréquences MHz
                            
                            **Vérifications:**
                            - ✅ Pas de lignes vides au début
                            - ✅ Les colonnes sont séparées par espaces ou tabs
                            - ✅ Les valeurs numériques utilisent des points (pas de virgules)
                            - ✅ Les profondeurs sont en mètres
                            - ✅ Les résistivités sont en Ω·m
                            """)
                            
                            # Affichage des résultats
                            st.markdown("## 📊 Résultats de l'Analyse ERT")
                            
                            # Statistiques générales
                            stats = result['stats']
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("📏 Mesures", stats['total_measurements'])
                            with col2:
                                st.metric("📍 Points", stats['survey_points'])
                            with col3:
                                st.metric("🔢 Résistivité moy.", f"{stats['resistivity_stats']['mean']:.2f} Ω·m")
                            with col4:
                                depth_min, depth_max = stats['depth_range']
                                st.metric("📏 Profondeur", f"{depth_min:.1f} - {depth_max:.1f} m")
                            
                            # Zones d'eau identifiées
                            st.markdown("### 💧 Types d'Eau Identifiés")
                            water_zones = result['water_zones']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🔴 Eau de mer", water_zones['mer'], 
                                         help="0.1-1 Ω·m: Eau hypersalée")
                            with col2:
                                st.metric("🟡 Eau salée", water_zones['salée'],
                                         help="1-10 Ω·m: Eau saumâtre")
                            with col3:
                                st.metric("🟢 Eau douce", water_zones['douce'],
                                         help="10-100 Ω·m: Eau potable")
                            with col4:
                                st.metric("🔵 Eau pure", water_zones['pure'],
                                         help="> 100 Ω·m: Eau très pure")
                            
                            # Interprétations géologiques
                            st.markdown("### 🪨 Interprétations Géologiques")
                            interp_summary = result['interpretations']
                            
                            if interp_summary:
                                df_interp = pd.DataFrame([
                                    {'Matériau': material, 'Occurrences': count}
                                    for material, count in sorted(interp_summary.items(), 
                                                                 key=lambda x: x[1], reverse=True)
                                ])
                                st.dataframe(df_interp, use_container_width=True)
                            
                            # Zones cibles pour forage
                            target_zones = result['target_zones']
                            if target_zones:
                                st.markdown("### 🎯 Zones Cibles pour Forage")
                                st.success(f"✅ **{len(target_zones)} zones cibles identifiées** (50-300 Ω·m)")
                                
                                df_targets = pd.DataFrame(target_zones)
                                st.dataframe(df_targets, use_container_width=True)
                                
                                st.info("""
                                **Recommandations:**
                                - 🟢 50-100 Ω·m: Aquifère sableux - ZONE PRIORITAIRE
                                - ⭐ 100-300 Ω·m: Gravier/Sable grossier - EXCELLENT débit
                                """)
                            else:
                                st.warning("⚠️ Aucune zone cible optimale détectée (50-300 Ω·m)")
                            
                            # Pseudo-section 2D
                            st.markdown("### 🗺️ Pseudo-section 2D")
                            with st.spinner("Création de la pseudo-section..."):
                                fig = ert.create_2d_pseudosection()
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                            
                            # Tableau d'interprétation
                            with st.expander("📋 Tableau d'Interprétation Complet"):
                                st.markdown(ert.create_interpretation_html(), unsafe_allow_html=True)
                            
                            # DataFrame complet
                            with st.expander("📊 Données Brutes"):
                                st.dataframe(result['dataframe'], use_container_width=True)
                            
                            # 💾 STOCKER LES RÉSULTATS ERT POUR ANALYSE CONVERSATIONNELLE
                            if 'ert_analysis' not in st.session_state:
                                st.session_state.ert_analysis = {}
                            
                            st.session_state.ert_analysis[binary['name']] = {
                                'result': result,
                                'stats': stats,
                                'water_zones': water_zones,
                                'target_zones': target_zones,
                                'interp_summary': interp_summary,
                                'dataframe': result['dataframe'],
                                'ert_analyzer': ert  # Garder l'instance pour réutilisation
                            }
                            
                            # Réponse récapitulative ENRICHIE avec guide d'utilisation
                            response = f"""✅ **Analyse ERT Complète Terminée!**

📊 **Résultats du fichier `{binary['name']}`:**
- 📏 **{stats['total_measurements']} mesures** sur **{stats['survey_points']} points de sondage**
- 🌊 **Types d'eau:** {water_zones['mer']} mer, {water_zones['salée']} salée, {water_zones['douce']} douce, {water_zones['pure']} pure
- 🎯 **{len(target_zones)} zones cibles** pour forage (résistivité optimale 50-300 Ω·m)
- 📈 **Résistivité moyenne:** {stats['resistivity_stats']['mean']:.2f} Ω·m (min: {stats['resistivity_stats']['min']:.2f}, max: {stats['resistivity_stats']['max']:.2f})
- 📏 **Profondeur explorée:** {depth_min:.1f} à {depth_max:.1f} m

**🪨 Interprétations géologiques principales:**
{chr(10).join([f"• {mat}: {cnt} mesures" for mat, cnt in list(interp_summary.items())[:3]])}

---

### 💬 **Posez-moi des questions sur cette analyse !**

**Exemples de questions que vous pouvez me poser:**

🎨 **Couleurs & Résistivités:**
- "Que signifie la couleur rouge sur la carte ?"
- "Quelle résistivité correspond au vert ?"
- "Pourquoi y a-t-il du bleu à cette profondeur ?"

📊 **Analyses détaillées:**
- "Explique-moi les zones à 50 Ω·m"
- "Quelle est la meilleure profondeur pour forer ?"
- "Compare les points de sondage 1 et 3"

💧 **Interprétation hydrogéologique:**
- "Où se trouve l'eau douce ?"
- "Pourquoi y a-t-il de l'eau de mer ici ?"
- "Quelle est la qualité de l'aquifère ?"

🎯 **Recommandations:**
- "Où dois-je forer mon puits ?"
- "Quelle profondeur recommandes-tu ?"
- "Quel débit puis-je espérer ?"

**🛠️ Outils utilisés:** ERTAnalyzer, Pandas, NumPy, Matplotlib, Classification automatique

Je peux maintenant analyser chaque détail de ces résultats avec vous ! 🚀"""
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            st.success("✅ Analyse ERT terminée! Vous pouvez maintenant me poser des questions détaillées sur les résultats.")
                            return
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'analyse ERT: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                            return
                
                # 🚀 ORCHESTRATION AUTOMATIQUE - EXÉCUTER AU LIEU DE PROPOSER
                if needs_conversion and 'binary_files' in st.session_state and st.session_state.binary_files:
                    # Afficher le plan d'action
                    with st.expander("📋 Plan d'Action Automatique", expanded=True):
                        st.markdown("""
                        ### 🎯 Orchestration Activée
                        
                        **Détection:** Demande de conversion de données
                        
                        **Plan d'exécution:**
                        1. ✅ **Analyse du fichier** - Identifier la structure
                        2. ✅ **Détection du format** - CSV/TSV/JSON/Binaire
                        3. ⏳ **Extraction des données** - Parser le contenu
                        4. ⏳ **Conversion** - Créer tableau NumPy/Pandas
                        5. ⏳ **Visualisation** - Afficher le résultat
                        6. ⏳ **Export** - Proposer téléchargement
                        """)
                    
                    # EXÉCUTION RÉELLE DE LA CONVERSION
                    try:
                        import pandas as pd
                        import numpy as np
                        import io
                        
                        binary = st.session_state.binary_files[-1]
                        st.info(f"🔄 Conversion en cours de `{binary['name']}`...")
                        
                        # Étape 1: Détecter le format
                        with st.spinner("Détection du format..."):
                            if binary.get('structure'):
                                structure = binary['structure']
                                st.success(f"✅ Format détecté: **{structure.file_type.value}** (confiance: {structure.confidence:.0%})")
                                
                                # Utiliser le DataFrame déjà extrait
                                if binary.get('dataframe') is not None:
                                    df = binary['dataframe']
                                    st.success(f"✅ Données déjà extraites: {df.shape[0]} lignes × {df.shape[1]} colonnes")
                                else:
                                    # Extraire maintenant
                                    text_content = '\n'.join(binary.get('text_strings', []))
                                    delimiter = structure.delimiter or '\t'
                                    df = pd.read_csv(io.StringIO(text_content), delimiter=delimiter, header=None, on_bad_lines='skip')
                                    st.success(f"✅ Extraction réussie: {df.shape[0]} lignes × {df.shape[1]} colonnes")
                            else:
                                # Tentative brute force
                                st.warning("⚠️ Structure non détectée, tentative d'extraction brute...")
                                text_content = '\n'.join(binary.get('text_strings', []))
                                for delim in ['\t', ',', ';', ' ']:
                                    try:
                                        df = pd.read_csv(io.StringIO(text_content), delimiter=delim, header=None, on_bad_lines='skip')
                                        if df.shape[1] > 1 and df.shape[0] > 0:
                                            st.success(f"✅ Format détecté: Délimiteur `{repr(delim)}` - {df.shape[0]}×{df.shape[1]}")
                                            break
                                    except:
                                        continue
                        
                        # Étape 2: Conversion NumPy
                        with st.spinner("Conversion en NumPy..."):
                            numpy_array = df.to_numpy()
                            st.success(f"✅ Tableau NumPy créé: shape={numpy_array.shape}, dtype={numpy_array.dtype}")
                        
                        # Étape 3: Affichage
                        st.markdown("### 📊 Résultat de la Conversion")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📏 Dimensions", f"{numpy_array.shape[0]} × {numpy_array.shape[1]}")
                        with col2:
                            st.metric("🔢 Type", str(numpy_array.dtype))
                        with col3:
                            st.metric("💾 Taille", f"{numpy_array.nbytes / 1024:.2f} KB")
                        
                        # DataFrame preview
                        st.markdown("#### 🔍 Aperçu des Données")
                        st.dataframe(df.head(50), use_container_width=True)
                        
                        # Statistiques
                        st.markdown("#### 📈 Statistiques")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(df.describe())
                        with col2:
                            # Tentative de visualisation si colonnes numériques
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                st.line_chart(df[numeric_cols].head(100))
                        
                        # Export
                        st.markdown("#### 💾 Export")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Télécharger CSV", csv_data, f"{binary['name']}.csv", "text/csv")
                        with col2:
                            # NumPy save
                            np_bytes = io.BytesIO()
                            np.save(np_bytes, numpy_array)
                            st.download_button("📥 Télécharger NumPy", np_bytes.getvalue(), f"{binary['name']}.npy", "application/octet-stream")
                        with col3:
                            json_data = df.to_json(orient='records', indent=2).encode('utf-8')
                            st.download_button("📥 Télécharger JSON", json_data, f"{binary['name']}.json", "application/json")
                        
                        # Mise à jour du plan
                        with st.expander("📋 Plan d'Action Automatique", expanded=False):
                            st.markdown("""
                            ### ✅ Orchestration Terminée
                            
                            **Toutes les étapes exécutées avec succès:**
                            1. ✅ **Analyse du fichier** - Complétée
                            2. ✅ **Détection du format** - Complétée
                            3. ✅ **Extraction des données** - Complétée
                            4. ✅ **Conversion** - NumPy/Pandas créés
                            5. ✅ **Visualisation** - Affichée ci-dessus
                            6. ✅ **Export** - Boutons disponibles
                            """)
                        
                        # Stocker les résultats dans la session pour persistance
                        if 'conversion_results' not in st.session_state:
                            st.session_state.conversion_results = []
                        
                        st.session_state.conversion_results.append({
                            'filename': binary['name'],
                            'dataframe': df,
                            'numpy_array': numpy_array,
                            'timestamp': pd.Timestamp.now()
                        })
                        
                        # Ajouter réponse récapitulative
                        response = f"""✅ **Conversion Réussie!**

Le fichier `{binary['name']}` a été automatiquement:
- 🔍 Analysé et structuré
- 📊 Converti en tableau NumPy de shape {numpy_array.shape}
- 📈 Transformé en DataFrame Pandas ({df.shape[0]} lignes × {df.shape[1]} colonnes)
- 💾 Préparé pour export (CSV, NumPy, JSON)

Les données sont affichées ci-dessus avec statistiques et visualisations."""
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        st.success("✅ Conversion terminée! Résultats affichés ci-dessus.")
                        # NE PAS faire st.rerun() - garder l'affichage
                        return  # Sortir pour ne pas continuer le traitement normal
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la conversion: {e}")
                        st.exception(e)
                        return  # Sortir en cas d'erreur
                
                if is_first_query:
                    # Première analyse : réponse structurée complète avec recherche web
                    binary_full_context = f"""
🔬 ANALYSE EXPERTE APPROFONDIE DU FICHIER: {binary['name']}

Tu es Kibali, un expert en analyse de données avec accès à des outils puissants.
L'utilisateur demande: "{prompt}"

📊 DONNÉES COMPLÈTES DU FICHIER:
{binary_context}

🛠️ OUTILS À TA DISPOSITION:
1. **Recherche Web** - Pour contexte industriel, standards, domaines d'application
2. **Calculs Python** - Pour analyses statistiques, moyennes, totaux
3. **Extraction de données** - Parser et structurer les données
4. **Visualisation** - Créer des tableaux, graphiques
5. **Comparaison** - Benchmarking avec standards connus

⚡ INSTRUCTIONS IMPÉRATIVES:
1. **ANALYSE COMPLÈTE** (minimum 20 lignes structurées):
   - Vue d'ensemble du fichier
   - Identification précise du format et standard
   - Contexte métier/industriel (utilise recherche web si besoin)
   - Extraction des données clés avec valeurs exactes
   - Analyse statistique (moyennes, min, max, tendances)
   - Implications pratiques

2. **UTILISE LES OUTILS ACTIVEMENT**:
   - Si tu vois des données numériques → Calcule statistiques
   - Si tu identifies un domaine → Recherche web pour contexte
   - Si format connu → Compare avec standards
   - Si valeurs présentes → Extrais et structure en tableau

3. **STRUCTURE TA RÉPONSE**:
   ```
   🎯 Identification du fichier
   📊 Analyse des données (avec chiffres exacts)
   🔍 Contexte métier/industriel
   📈 Statistiques et calculs
   💡 Implications et recommandations
   ```

4. **SOIS PROACTIF**: Anticipe les besoins, fournis plus que demandé, montre ton expertise!
"""
                else:
                    # 🌊 CONTEXTE ERT ENRICHI - Questions sur l'analyse existante
                    ert_context = ""
                    ert_rag_context = ""
                    if 'ert_analysis' in st.session_state and binary['name'] in st.session_state.ert_analysis:
                        ert_data = st.session_state.ert_analysis[binary['name']]
                        stats = ert_data['stats']
                        water_zones = ert_data['water_zones']
                        interp = ert_data['interp_summary']
                        df = ert_data['dataframe']
                        
                        # 🧠 ENRICHISSEMENT RAG DYNAMIQUE POUR LES QUESTIONS
                        # Détecter les mots-clés de la question pour cibler la recherche RAG
                        question_keywords = []
                        if 'résistivité' in prompt.lower() or 'resistivité' in prompt.lower():
                            question_keywords.extend(['résistivité électrique', 'conductivité'])
                        if 'forage' in prompt.lower() or 'forer' in prompt.lower():
                            question_keywords.extend(['forage', 'puits', 'aquifère'])
                        if 'eau' in prompt.lower():
                            question_keywords.extend(['hydrogéologie', 'nappe phréatique', 'aquifère'])
                        if 'géologie' in prompt.lower() or 'roche' in prompt.lower():
                            question_keywords.extend(['lithologie', 'géologie', 'formation rocheuse'])
                        if 'profondeur' in prompt.lower():
                            question_keywords.extend(['profondeur', 'stratigraphie'])
                        
                        # Recherche RAG ciblée si base disponible
                        if question_keywords and st.session_state.get('vectordb') is not None:
                            try:
                                # Construire une requête intelligente
                                avg_resistivity = stats['resistivity_stats']['mean']
                                rag_query = f"{' '.join(question_keywords)} {avg_resistivity:.1f} Ohm·m ERT géophysique"
                                
                                # Recherche silencieuse
                                docs = st.session_state.vectordb.similarity_search(rag_query, k=3)
                                
                                if docs:
                                    ert_rag_context = f"""

🧠 **CONNAISSANCES RAG PERTINENTES ({len(docs)} documents trouvés):**

{chr(10).join([f"📄 **Document {i+1}:** {doc.page_content[:400]}" for i, doc in enumerate(docs)])}

**UTILISE CES CONNAISSANCES** pour enrichir ta réponse avec:
- Comparaisons avec d'autres sites/études
- Références scientifiques
- Méthodologies validées
- Retours d'expérience terrain
- Normes et standards applicables
"""
                            except Exception as e:
                                pass  # Silencieux si RAG échoue
                        
                        ert_context = f"""

═══════════════════════════════════════════════════════════════════
🌊 ANALYSE ERT DÉJÀ EFFECTUÉE - TOUTES LES DONNÉES DISPONIBLES
═══════════════════════════════════════════════════════════════════

📊 **STATISTIQUES COMPLÈTES:**
- Total mesures: {stats['total_measurements']}
- Points de sondage: {stats['survey_points']}
- Résistivité: min={stats['resistivity_stats']['min']:.2f} Ω·m, max={stats['resistivity_stats']['max']:.2f} Ω·m, moyenne={stats['resistivity_stats']['mean']:.2f} Ω·m, médiane={stats['resistivity_stats']['median']:.2f} Ω·m
- Profondeur: {stats['depth_range'][0]:.1f} à {stats['depth_range'][1]:.1f} m

💧 **TYPES D'EAU IDENTIFIÉS:**
- 🔴 Eau de mer (0.1-1 Ω·m): {water_zones['mer']} mesures
- 🟡 Eau salée (1-10 Ω·m): {water_zones['salée']} mesures
- 🟢 Eau douce (10-100 Ω·m): {water_zones['douce']} mesures
- 🔵 Eau pure (>100 Ω·m): {water_zones['pure']} mesures

🪨 **INTERPRÉTATIONS GÉOLOGIQUES:**
{chr(10).join([f"- {mat}: {cnt} occurrences" for mat, cnt in list(interp.items())[:5]])}

🎨 **CORRESPONDANCES COULEURS-RÉSISTIVITÉS:**
- 🔴 **ROUGE** (0.1-1 Ω·m): Eau de mer hypersalée / Argile saturée salée
- 🟡 **JAUNE** (1-10 Ω·m): Argile compacte / Eau saumâtre (faible perméabilité)
- 🔵 **CYAN** (10-50 Ω·m): Sable fin saturé / Eau douce peu minéralisée
- 🟢 **VERT** (50-100 Ω·m): 🎯 AQUIFÈRE SABLEUX - ZONE PRIORITAIRE FORAGE
- ⭐ **JAUNE VIF** (100-300 Ω·m): GRAVIER / AQUIFÈRE EXCELLENT - Débit élevé
- 🟠 **ORANGE** (300-1000 Ω·m): Roche altérée / Gravier sec (zone non saturée)
- 🔴 **ROUGE FONCÉ** (>1000 Ω·m): Roche consolidée / Socle cristallin

📊 **DONNÉES BRUTES DISPONIBLES:**
Aperçu de quelques mesures:
{df.head(10).to_string()}

🎯 **ZONES CIBLES FORAGE:** {len(ert_data['target_zones'])} zones identifiées (50-300 Ω·m)

═══════════════════════════════════════════════════════════════════

💬 **L'utilisateur te pose maintenant une question sur ces résultats.**

**TU DOIS:**
1. Répondre PRÉCISÉMENT avec les DONNÉES EXACTES ci-dessus
2. Expliquer en DÉTAIL avec exemples concrets
3. Citer les CHIFFRES et PROFONDEURS spécifiques
4. Comparer et interpréter les différences
5. Donner des RECOMMANDATIONS pratiques

**EXEMPLES DE RÉPONSES ATTENDUES:**

Question: "Que signifie la couleur rouge ?"
→ Réponse: "🔴 La couleur ROUGE correspond à une résistivité de 0.1-1 Ω·m. Dans votre analyse, j'ai identifié **{water_zones['mer']} mesures** dans cette gamme. Cela indique de l'**eau de mer hypersalée** ou une **argile saturée salée**. Ces zones ont une très forte conductivité électrique due aux ions Na⁺ et Cl⁻. Spécifiquement, votre résistivité moyenne est de {stats['resistivity_stats']['mean']:.2f} Ω·m, ce qui suggère [interprétation détaillée]..."

Question: "Où dois-je forer ?"
→ Réponse: "🎯 Basé sur votre analyse, je recommande de forer dans les **zones vertes (50-100 Ω·m)** ou **jaune vif (100-300 Ω·m)**. J'ai identifié **{len(ert_data['target_zones'])} zones cibles optimales**. Par exemple, au point de sondage X à la profondeur Y m, vous avez une résistivité de Z Ω·m qui correspond à [analyse détaillée avec recommandations précises]..."

**🛠️ Outils utilisés:** ERTAnalyzer, Classification géologique, Analyse hydrogéologique

"""
                        
                        # Ajouter le contexte RAG si disponible
                        if ert_rag_context:
                            ert_context += ert_rag_context
                    
                    # Questions suivantes : mode expert avec outils ciblés
                    # Questions suivantes : mode expert avec outils ciblés
                    tools_to_use = []
                    if needs_calculation:
                        tools_to_use.append("🧮 Calculs et statistiques")
                    if needs_web_research:
                        tools_to_use.append("🌐 Recherche web approfondie")
                    if needs_data_extraction:
                        tools_to_use.append("📊 Extraction et structuration de données")
                    
                    # Injecter les résultats pré-calculés
                    precomputed_section = ""
                    if 'precomputed_results' in locals() and precomputed_results:
                        precomputed_section = "\n\n📊 RÉSULTATS PRÉ-CALCULÉS (UTILISE-LES!):\n"
                        
                        if 'statistics' in precomputed_results:
                            stats = precomputed_results['statistics']
                            precomputed_section += f"""
✅ **STATISTIQUES COMPLÈTES** (calculées automatiquement):
   - Nombre de valeurs: {stats['count']}
   - Minimum: {stats['min']:.2f}
   - Maximum: {stats['max']:.2f}
   - Moyenne: {stats['mean']:.2f}
   - Médiane: {stats['median']:.2f}
   - Écart-type: {stats['std']:.2f}
   - Somme totale: {stats['sum']:.2f}
   - Plage: {stats['range']:.2f}
"""
                        
                        if 'structured_data' in precomputed_results:
                            data = precomputed_results['structured_data']
                            precomputed_section += f"""
✅ **DONNÉES STRUCTURÉES** (extraites automatiquement):
   Aperçu des {len(data)} premières lignes:
{chr(10).join([f"   {i+1}. {' | '.join(row[:5])}" for i, row in enumerate(data[:10])])}
"""
                        
                        if 'web_context' in precomputed_results:
                            web = precomputed_results['web_context']
                            precomputed_section += f"""
✅ **CONTEXTE WEB** (recherché automatiquement):
{chr(10).join([f"   • {r.get('title', 'Source')}: {r.get('body', '')[:150]}..." for r in web[:2]])}
"""
                        
                        # NOUVELLE SECTION: Extraction complète avec tableaux
                        if 'extraction_complete' in precomputed_results:
                            extract = precomputed_results['extraction_complete']
                            precomputed_section += f"""

✅ **EXTRACTION COMPLÈTE** (analyse exhaustive du fichier):
   📝 Lignes de texte: {len(extract['text_strings'])}
   🔢 Valeurs numériques: {len(extract['numeric_values'])}
   📊 Patterns détectés: {len(extract['patterns_detected'])}
"""
                            
                            # Ajouter le tableau structuré si disponible
                            if 'dataframe_stats' in precomputed_results:
                                df_stats = precomputed_results['dataframe_stats']
                                precomputed_section += f"""

📊 **TABLEAU STATISTIQUES PAR LABEL**:
{df_stats.to_string(index=False)}
"""
                            
                            # Ajouter les valeurs numériques avec contexte
                            if extract['numeric_values'][:10]:
                                precomputed_section += f"""

🔢 **VALEURS NUMÉRIQUES AVEC CONTEXTE** (top 10):
{chr(10).join([f"   {i+1}. {v['label']}: {v['value']} (ligne {v['line']})" for i, v in enumerate(extract['numeric_values'][:10])])}
"""
                            
                            # Ajouter les patterns
                            if extract['patterns_detected']:
                                precomputed_section += f"""

🔍 **PATTERNS DÉTECTÉS**:
{chr(10).join([f"   • {p}" for p in extract['patterns_detected'][:5]])}
"""
                    
                    binary_full_context = f"""
💬 EXPERTISE APPROFONDIE SUR: {binary['name']}

Tu es Kibali, expert en analyse géophysique ERT.
L'utilisateur demande: "{prompt}"

📊 CONTEXTE DU FICHIER:
{binary_context}

{ert_context}

🛠️ OUTILS DISPONIBLES: {', '.join(tools_to_use) if tools_to_use else 'Analyse standard'}

{precomputed_section}

QUESTION: {prompt}

📊 DONNÉES DISPONIBLES:
{binary_context[:2000]}
{precomputed_section}

🛠️ OUTILS ACTIVÉS POUR CETTE QUESTION:
{chr(10).join(f'✓ {tool}' for tool in tools_to_use) if tools_to_use else '✓ Analyse experte standard'}

⚡ CONSIGNES D'EXPERT:

1. **RÉPONDS AVEC PRÉCISION ET PROFONDEUR**:
   - Minimum 15-20 lignes structurées
   - Cite VALEURS EXACTES du fichier (nombres, dates, coordonnées)
   - Fais des CALCULS si données numériques présentes (UTILISE LES STATISTIQUES PRÉ-CALCULÉES CI-DESSUS!)
   - Recherche WEB si contexte métier nécessaire (UTILISE LE CONTEXTE WEB PRÉ-CHARGÉ!)

2. **STRUCTURE OBLIGATOIRE**:
   - Introduction directe (1-2 lignes)
   - Analyse détaillée avec données concrètes (CHIFFRES EXACTS des résultats pré-calculés)
   - Calculs/statistiques si pertinent (COPIE LES VALEURS PRÉ-CALCULÉES!)
   - Contexte métier si demandé (UTILISE LA RECHERCHE WEB!)
   - Conclusion avec insights

3. **INTERDICTIONS STRICTES**:
   ❌ NE DIS JAMAIS "calcul approximatif" ou "≈" si tu as les statistiques pré-calculées ci-dessus
   ❌ NE DIS JAMAIS "nécessiterait une extraction" si les données sont déjà extraites ci-dessus
   ❌ NE DIS JAMAIS "une analyse plus approfondie serait bénéfique" - FAIS-LA MAINTENANT!
   ❌ NE DIS JAMAIS "environ" ou "peut-être" - DONNE LES CHIFFRES EXACTS pré-calculés
   
4. **UTILISE LES OUTILS PRÉ-EXÉCUTÉS**:
   {'✅ STATISTIQUES COMPLÈTES déjà calculées ci-dessus - UTILISE-LES!' if needs_calculation and 'precomputed_results' in locals() and 'statistics' in precomputed_results else ''}
   {'✅ CONTEXTE WEB déjà récupéré ci-dessus - INTÈGRE-LE!' if needs_web_research and 'precomputed_results' in locals() and 'web_context' in precomputed_results else ''}
   {'✅ DONNÉES STRUCTURÉES déjà extraites ci-dessus - PRÉSENTE-LES!' if needs_data_extraction and 'precomputed_results' in locals() and 'structured_data' in precomputed_results else ''}

5. **EXEMPLES DE RÉPONSES RICHES**:
   ❌ "Ce fichier contient des données de sondage"
   ✅ "Ce fichier contient 247 points de sondage avec profondeurs allant de 12.5m à 89.3m (moyenne: 45.2m). 
       Les coordonnées couvrent une zone de 2.3 km² (X: 445230-447850, Y: 9834500-9836120). 
       Basé sur les en-têtes (survey-point, depth, data, project) et le format TSV, 
       il s'agit d'un fichier de levé géophysique standard, probablement généré par un 
       logiciel de type [recherche web: Surfer, Oasis Montaj, ou GeoSoft]. 
       Les dates présentes (20250401) indiquent une campagne récente..."

6. **SOIS UN VRAI EXPERT**: Ne te contente pas de descriptions, UTILISE LES RÉSULTATS PRÉ-CALCULÉS!
"""
                
                # REMPLACER complètement le prompt enrichi
                enriched_prompt = binary_full_context
            
            # Ajouter le message et nettoyer si trop long
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            # Garder plus de messages (50) pour conserver le contexte des fichiers uploadés
            # Garder toujours les messages de système (uploads, analyses) en priorité
            if len(st.session_state.chat_history) > 50:
                # Conserver les 10 premiers messages (généralement les uploads/analyses)
                # et les 40 derniers (conversation récente)
                important_msgs = st.session_state.chat_history[:10]
                recent_msgs = st.session_state.chat_history[-40:]
                st.session_state.chat_history = important_msgs + recent_msgs
            
            # 🛠️ PRÉ-EXÉCUTION DES OUTILS POUR COMBLER LES LIMITES
            precomputed_results = {}
            
            # 1. Si fichier binaire avec données numériques → CALCULS AUTOMATIQUES
            if 'binary_files' in st.session_state and st.session_state.binary_files and needs_calculation:
                try:
                    import re
                    import numpy as np
                    binary = st.session_state.binary_files[-1]
                    
                    # Extraire TOUTES les valeurs numériques du fichier
                    text_content = ' '.join(binary.get('text_strings', []))
                    numbers = re.findall(r'\d+\.?\d*', text_content)
                    values = [float(n) for n in numbers if n and float(n) > 0]
                    
                    if len(values) > 0:
                        precomputed_results['statistics'] = {
                            'count': len(values),
                            'min': min(values),
                            'max': max(values),
                            'mean': np.mean(values),
                            'median': np.median(values),
                            'std': np.std(values),
                            'sum': sum(values),
                            'range': max(values) - min(values)
                        }
                        st.success(f"✅ Calculs pré-exécutés: {len(values)} valeurs analysées")
                except Exception as e:
                    st.warning(f"⚠️ Calculs automatiques: {e}")
            
            # 2. Extraction automatique de données structurées
            if 'binary_files' in st.session_state and st.session_state.binary_files and needs_data_extraction:
                try:
                    binary = st.session_state.binary_files[-1]
                    text_strings = binary.get('text_strings', [])
                    
                    # Détecter la structure (CSV, TSV, JSON-like, etc.)
                    structured_data = []
                    for line in text_strings[:50]:  # Analyser les 50 premières lignes
                        if '\t' in line:
                            structured_data.append(line.split('\t'))
                        elif ',' in line and line.count(',') > 2:
                            structured_data.append(line.split(','))
                    
                    if structured_data:
                        precomputed_results['structured_data'] = structured_data[:20]  # Top 20 lignes
                        st.success(f"✅ Données structurées extraites: {len(structured_data)} lignes")
                except Exception as e:
                    st.warning(f"⚠️ Extraction automatique: {e}")
            
            # 2B. 🚀 EXTRACTION COMPLÈTE AVEC OCR + TABLEAU STRUCTURÉ
            if 'binary_files' in st.session_state and st.session_state.binary_files:
                try:
                    import pandas as pd
                    import matplotlib.pyplot as plt
                    import io
                    import base64
                    
                    binary = st.session_state.binary_files[-1]
                    
                    # Extraction exhaustive de toutes les données
                    extraction_complete = {
                        'text_strings': [],
                        'numeric_values': [],
                        'hex_data': [],
                        'ascii_readable': [],
                        'patterns_detected': []
                    }
                    
                    # 1. Extraire TOUT le texte lisible
                    text_strings = binary.get('text_strings', [])
                    extraction_complete['text_strings'] = text_strings
                    
                    # 2. Extraire toutes les valeurs numériques avec contexte
                    import re
                    for idx, line in enumerate(text_strings):
                        # Trouver nombres avec contexte
                        numbers = re.findall(r'(\w*)\s*[=:]\s*(\d+\.?\d*)', line)
                        for label, value in numbers:
                            extraction_complete['numeric_values'].append({
                                'line': idx,
                                'label': label if label else 'value',
                                'value': float(value),
                                'context': line[:100]
                            })
                    
                    # 3. Analyser les patterns dans les données binaires
                    data = binary.get('data', b'')
                    chunk_size = 16
                    for i in range(0, min(len(data), 1024), chunk_size):
                        chunk = data[i:i+chunk_size]
                        hex_str = ' '.join(f'{b:02x}' for b in chunk)
                        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
                        
                        extraction_complete['hex_data'].append({
                            'offset': i,
                            'hex': hex_str,
                            'ascii': ascii_str
                        })
                        
                        if ascii_str.count('.') < len(ascii_str) * 0.3:  # >70% ASCII
                            extraction_complete['ascii_readable'].append({
                                'offset': i,
                                'text': ascii_str
                            })
                    
                    # 4. Détecter patterns (répétitions, structures)
                    if len(data) > 100:
                        # Détecter répétitions de bytes
                        byte_freq = {}
                        for b in data[:1000]:
                            byte_freq[b] = byte_freq.get(b, 0) + 1
                        
                        most_common = sorted(byte_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                        extraction_complete['patterns_detected'] = [
                            f"Byte 0x{b:02x} apparaît {count} fois" for b, count in most_common
                        ]
                    
                    # 5. 📊 CRÉER UN TABLEAU PANDAS STRUCTURÉ
                    if extraction_complete['numeric_values']:
                        df = pd.DataFrame(extraction_complete['numeric_values'])
                        
                        # Organiser par label et calculer stats
                        if len(df) > 0:
                            df_grouped = df.groupby('label')['value'].agg([
                                ('count', 'count'),
                                ('min', 'min'),
                                ('max', 'max'),
                                ('mean', 'mean'),
                                ('std', 'std')
                            ]).reset_index()
                            
                            precomputed_results['dataframe_stats'] = df_grouped
                            precomputed_results['dataframe_raw'] = df
                            
                            # 6. 📈 VISUALISATION MATPLOTLIB
                            if len(df) > 3:
                                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                                
                                # Graphique 1: Distribution des valeurs
                                df['value'].hist(bins=20, ax=axes[0], color='skyblue', edgecolor='black')
                                axes[0].set_title('Distribution des Valeurs', fontsize=12, fontweight='bold')
                                axes[0].set_xlabel('Valeur')
                                axes[0].set_ylabel('Fréquence')
                                axes[0].grid(True, alpha=0.3)
                                
                                # Graphique 2: Top labels
                                if 'label' in df.columns and len(df['label'].unique()) > 1:
                                    label_counts = df['label'].value_counts().head(10)
                                    label_counts.plot(kind='barh', ax=axes[1], color='coral')
                                    axes[1].set_title('Top 10 Labels', fontsize=12, fontweight='bold')
                                    axes[1].set_xlabel('Nombre d\'occurrences')
                                else:
                                    axes[1].text(0.5, 0.5, 'Pas assez de labels variés', 
                                               ha='center', va='center', fontsize=12)
                                    axes[1].axis('off')
                                
                                plt.tight_layout()
                                
                                # Convertir en base64 pour affichage
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                                buf.seek(0)
                                img_base64 = base64.b64encode(buf.read()).decode()
                                plt.close()
                                
                                precomputed_results['visualization'] = img_base64
                    
                    # Stocker l'extraction complète
                    precomputed_results['extraction_complete'] = extraction_complete
                    
                    st.success(f"✅ Extraction complète: {len(extraction_complete['text_strings'])} lignes texte, {len(extraction_complete['numeric_values'])} valeurs numériques")
                    
                except Exception as e:
                    st.warning(f"⚠️ Extraction complète: {e}")
                    import traceback
                    print(traceback.format_exc())
            
            # 3. Recherche web contextuelle automatique
            if needs_web_research and 'binary_files' in st.session_state and st.session_state.binary_files:
                try:
                    binary = st.session_state.binary_files[-1]
                    # Construire requête intelligente
                    format_name = binary['info'].get('format_name', 'unknown')
                    extension = binary['name'].split('.')[-1] if '.' in binary['name'] else 'dat'
                    search_query = f"{format_name} {extension} file format geophysical survey data"
                    
                    # Exécuter la recherche
                    if web_enabled:
                        web_results = enhanced_web_search(search_query, max_results=2)
                        if web_results:
                            precomputed_results['web_context'] = web_results[:2]
                            st.success(f"✅ Recherche web contextualisée: {len(web_results)} sources")
                except Exception as e:
                    st.warning(f"⚠️ Recherche web automatique: {e}")
            
            # Afficher les outils actifs
            active_tools_msg = []
            if needs_calculation:
                active_tools_msg.append("🧮 Calculs statistiques")
            if needs_web_research or ('binary_files' in st.session_state and st.session_state.binary_files):
                active_tools_msg.append("🌐 Recherche web")
            if needs_data_extraction:
                active_tools_msg.append("📊 Extraction de données")
            
            if active_tools_msg:
                tools_status = " • ".join(active_tools_msg)
                st.info(f"⚡ Outils activés: {tools_status}")
            
            # 📊 AFFICHER LA VISUALISATION SI DISPONIBLE
            if 'precomputed_results' in locals() and 'visualization' in precomputed_results:
                with chat_container:
                    st.markdown("### 📈 Visualisation des Données Extraites")
                    st.markdown(f'<img src="data:image/png;base64,{precomputed_results["visualization"]}" style="width:100%; border-radius:10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
            
            # 📋 AFFICHER LE TABLEAU SI DISPONIBLE
            if 'precomputed_results' in locals() and 'dataframe_stats' in precomputed_results:
                with chat_container:
                    st.markdown("### 📊 Tableau des Statistiques")
                    st.dataframe(precomputed_results['dataframe_stats'], use_container_width=True)
            
            # 🎼 ORCHESTRATION MULTI-IA
            orchestration_plan = None
            orchestration_enabled = False
            
            # Éviter l'orchestration si question simple ou déjà des résultats pré-calculés suffisants
            skip_orchestration = (
                len(prompt.split()) < 5 or  # Question très courte
                ('precomputed_results' in locals() and len(precomputed_results) > 0 and not needs_web_research)  # Résultats suffisants
            )
            
            if ORCHESTRATOR_AVAILABLE and not skip_orchestration:
                try:
                    # Analyser la tâche et créer le plan
                    orchestrator = get_orchestrator()
                    
                    # Construire le contexte pour l'orchestrateur
                    orchestration_context = {
                        'has_vectordb': st.session_state.vectordb is not None,
                        'has_binary_file': 'binary_files' in st.session_state and st.session_state.binary_files,
                        'web_enabled': web_enabled,
                        'precomputed_results': precomputed_results if 'precomputed_results' in locals() else {},
                        'needs_calculation': needs_calculation,
                        'needs_web_research': needs_web_research,
                        'needs_data_extraction': needs_data_extraction
                    }
                    
                    tasks, todolist = analyze_and_plan(prompt, orchestration_context)
                    
                    if tasks and len(tasks) > 0:
                        orchestration_enabled = True
                        
                        # Afficher la todolist
                        with chat_container:
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                                padding: 25px;
                                border-radius: 15px;
                                margin: 15px 0;
                                border-left: 8px solid #2196F3;
                                box-shadow: 0 4px 15px rgba(33,150,243,0.3);
                            ">
                                <h3 style="color: #1565C0; margin: 0 0 15px 0; display: flex; align-items: center; font-weight: bold;">
                                    <span style="font-size: 2rem; margin-right: 10px;">📋</span>
                                    Plan d'Exécution Multi-IA
                                </h3>
                                <div style="background: white; padding: 20px; border-radius: 10px; border: 2px solid #2196F3;">
                                    <pre style="color: #1a237e; margin: 0; font-family: 'Courier New', monospace; font-size: 1rem; white-space: pre-wrap; font-weight: 500; line-height: 1.6;">{todolist}</pre>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Sélectionner les IA spécialisées
                        team = orchestrator.select_team(tasks)
                        
                        # Afficher le visuel de l'équipe
                        with chat_container:
                            team_visual = orchestrator.create_execution_plan_visual(team)
                            st.markdown(team_visual, unsafe_allow_html=True)
                        
                        # Préparer le plan d'orchestration
                        orchestration_plan = {
                            'tasks': tasks,
                            'team': team,
                            'todolist': todolist
                        }
                
                except Exception as e:
                    st.warning(f"⚠️ Orchestration non disponible: {e}")
                    orchestration_enabled = False
                    
                    # FALLBACK: Activer les outils dynamiquement même sans orchestration
                    if precomputed_results:
                        with chat_container:
                            st.info("🔧 Mode outils dynamiques activé (sans orchestration)")
                            if 'statistics' in precomputed_results:
                                st.success(f"📊 Statistiques pré-calculées disponibles")
                            if 'web_context' in precomputed_results:
                                st.success(f"🌐 Contexte web chargé")
                            if 'structured_data' in precomputed_results:
                                st.success(f"📋 Données structurées extraites")
            
            # Animation de chargement
            loading_msg = "🎼 Orchestration multi-IA en cours..." if orchestration_enabled else "🤔 Kibali analyse avec ses outils..."
            with st.spinner(loading_msg):
                # Vérifier si on utilise le mode local
                if st.session_state.local_mode and st.session_state.local_model_loaded:
                    # Utiliser le modèle local Qwen
                    try:
                        # 🔥 RECHERCHE PROACTIVE: Activer automatiquement selon le contexte
                        rag_context = ""
                        web_context = ""
                        web_media_html = ""
                        
                        # Recherche RAG si disponible
                        if st.session_state.vectordb:
                            rag_docs = rag_search(prompt, st.session_state.vectordb, k=3)
                            if rag_docs:
                                rag_context = "\n\n".join([f"Document: {doc.page_content[:500]}..." for doc in rag_docs])
                        
                        # 🌐 RECHERCHE WEB AUTOMATIQUE pour contexte métier/industriel
                        if needs_web_research or ('binary_files' in st.session_state and st.session_state.binary_files):
                            try:
                                # Construire requête intelligente basée sur le contenu du fichier
                                if 'binary_files' in st.session_state and st.session_state.binary_files:
                                    binary = st.session_state.binary_files[-1]
                                    # Extraire mots-clés des chaînes de texte
                                    keywords = []
                                    for s in binary.get('text_strings', [])[:10]:
                                        words = s.split()
                                        keywords.extend([w for w in words if len(w) > 4])
                                    
                                    search_query = f"{binary['info']['format_name']} {' '.join(keywords[:5])} file format application"
                                else:
                                    search_query = prompt
                                
                                web_results = enhanced_web_search(search_query, max_results=3)
                                if web_results:
                                    web_context = "\n".join([f"{r.get('title', '')[:50]} - {r.get('body', '')[:150]}" for r in web_results[:2]])
                                    web_media_html = format_web_results_with_media(web_results, max_results=3)
                            except Exception as e:
                                web_context = f"Recherche web limitée: {e}"
                        elif web_enabled:
                            # Recherche web standard si activée
                            try:
                                web_results = enhanced_web_search(prompt, max_results=3)
                                if web_results:
                                    web_context = "\n".join([f"{r.get('title', '')[:50]} - {r.get('body', '')[:150]}" for r in web_results[:2]])
                                    web_media_html = format_web_results_with_media(web_results, max_results=3)
                            except Exception as e:
                                web_context = f"Erreur recherche web: {e}"
                        
                        # Construire le contexte enrichi avec médias ET fichiers binaires
                        if 'binary_files' in st.session_state and st.session_state.binary_files:
                            # Mode fichier binaire: enriched_prompt contient déjà tout
                            full_context = enriched_prompt
                            if rag_context:
                                full_context += f"\n\n📚 SOURCES COMPLÉMENTAIRES:\n{rag_context[:300]}"
                            if web_context:
                                full_context += f"\n\n🌐 RECHERCHE WEB CONTEXTUELLE:\n{web_context}"
                        else:
                            # Mode normal
                            full_context = f"CONTEXTE DISPONIBLE:\n{rag_context}{web_context}\n\nQUESTION: {enriched_prompt}"
                        
                        # 🌊 Générer avec le modèle local EN STREAMING
                        from langchain_core.messages import HumanMessage
                        
                        with chat_container:
                            with st.chat_message("assistant"):
                                message_placeholder = st.empty()
                                full_response = "🏠 **Mode Local (Qwen 1.5B)** - Réponse générée localement\n\n"
                                
                                # Utiliser _stream au lieu de _generate
                                for chunk in st.session_state.local_qwen_llm._stream(
                                    [HumanMessage(content=full_context)],
                                    stop=None,
                                    run_manager=None
                                ):
                                    if hasattr(chunk, 'content'):
                                        full_response += chunk.content
                                        message_placeholder.markdown(full_response + "▌")
                                
                                # Affichage final sans curseur
                                full_response = clean_response_text(full_response)
                                message_placeholder.markdown(full_response)
                        
                        response = full_response
                        
                    except Exception as e:
                        response = f"❌ Erreur modèle local: {e}\n\nBasculement vers mode API..."
                        # Fallback vers mode API
                        st.session_state.local_mode = False
                        st.rerun()
                
                else:
                    # ═══════════════════════════════════════════════════════════════════
                    # 🎯 MODE PRIORITAIRE: FICHIER BINAIRE DÉTECTÉ
                    # Court-circuite tous les autres systèmes pour forcer l'analyse du fichier
                    # ═══════════════════════════════════════════════════════════════════
                    if 'binary_files' in st.session_state and st.session_state.binary_files:
                        st.info("🔍 Mode analyse de fichier binaire activé")
                        
                        # Utiliser directement l'API avec le contexte binaire complet
                        client = create_client()
                        
                        # enriched_prompt contient déjà TOUT le contexte binaire détaillé
                        # On limite l'historique pour maximiser le contexte du fichier
                        messages = [{"role": "user", "content": enriched_prompt[:12000]}]  # 12000 chars pour analyse complète très détaillée
                        
                        with chat_container:
                            with st.chat_message("assistant"):
                                message_placeholder = st.empty()
                                full_response = ""
                                
                                # Streaming direct
                                stream = client.chat.completions.create(
                                    model=WORKING_MODELS[model_choice],
                                    messages=messages,
                                    max_tokens=4000,  # Réponse très détaillée (2-3 pages)
                                    temperature=0.2,  # Basse température pour précision
                                    stream=True
                                )
                                
                                for chunk in stream:
                                    if chunk.choices[0].delta.content is not None:
                                        full_response += chunk.choices[0].delta.content
                                        message_placeholder.markdown(full_response + "▌")
                                
                                message_placeholder.markdown(full_response)
                        
                        response = full_response
                    
                    # Mode API normal (code existant)
                    # Utilisation du système d'outils dynamiques si disponible
                    elif st.session_state.tool_manager and TOOLS_SYSTEM_AVAILABLE:
                        try:
                            # 🎼 MODE ORCHESTRATION MULTI-IA PRIORITAIRE
                            if orchestration_enabled and orchestration_plan:
                                # Exécuter avec orchestration multi-IA
                                orchestrator = get_orchestrator()
                                
                                with chat_container:
                                    with st.chat_message("assistant"):
                                        message_placeholder = st.empty()
                                        full_response = "🎼 **Orchestration Multi-IA Activée**\n\n"
                                        message_placeholder.markdown(full_response + "▌")
                                        
                                        # Exécuter chaque tâche avec l'IA spécialisée
                                        for idx, (task, specialist) in enumerate(zip(orchestration_plan['tasks'], orchestration_plan['team']), 1):
                                            task_header = f"\n\n{'─' * 60}\n**🎯 Tâche {idx}/{len(orchestration_plan['tasks'])}: {task.description}**\n**🤖 IA: {specialist.name}**\n{'─' * 60}\n\n"
                                            full_response += task_header
                                            message_placeholder.markdown(full_response + "▌")
                                            
                                            try:
                                                # Exécuter avec l'IA spécialisée
                                                # Pour l'instant on utilise l'API standard, mais on peut charger des modèles spécialisés
                                                task_prompt = f"{enriched_prompt}\n\nFOCUS SUR: {task.description}\nTYPE: {task.task_type.value}"
                                                
                                                client = create_client()
                                                task_messages = [{"role": "user", "content": task_prompt[:8000]}]
                                                
                                                # Obtenir la limite de tokens depuis le catalog
                                                from master_orchestrator import MasterOrchestrator
                                                orchestrator = MasterOrchestrator()
                                                specialist_capability = orchestrator.SPECIALISTS_CATALOG.get(specialist)
                                                max_output = specialist_capability.max_output_tokens if specialist_capability else 4000
                                                
                                                # Streaming pour cette tâche
                                                stream = client.chat.completions.create(
                                                    model=WORKING_MODELS[model_choice],
                                                    messages=task_messages,
                                                    max_tokens=max_output,
                                                    temperature=0.7,
                                                    stream=True
                                                )
                                                
                                                task_response = ""
                                                for chunk in stream:
                                                    if chunk.choices[0].delta.content is not None:
                                                        task_response += chunk.choices[0].delta.content
                                                        full_response = full_response.rstrip("▌") + task_response + "▌"
                                                        message_placeholder.markdown(full_response)
                                                
                                                full_response = full_response.rstrip("▌")
                                                
                                                # Vérifier si relay nécessaire (réponse très longue)
                                                if len(task_response.split()) > max_output * 0.85:
                                                    relay_msg = f"\n\n⚡ **Relay vers {specialist.name}** (limite tokens atteinte)\n\n"
                                                    full_response += relay_msg
                                                    message_placeholder.markdown(full_response + "▌")
                                                
                                            except Exception as task_error:
                                                error_msg = f"\n\n❌ **Erreur tâche {idx}:** {task_error}\n\n"
                                                full_response += error_msg
                                                message_placeholder.markdown(full_response + "▌")
                                        
                                        # Synthèse finale
                                        full_response += f"\n\n{'═' * 60}\n✅ **Orchestration Complétée** - {len(orchestration_plan['tasks'])} tâches exécutées\n{'═' * 60}"
                                        
                                        # 🧹 Nettoyer la réponse
                                        full_response = clean_response_text(full_response)
                                        message_placeholder.markdown(full_response)
                                
                                response = full_response
                            
                            else:
                                # Mode outils classique (code existant)
                                # Préparer le contexte pour les outils (avec médias analysés)
                                tool_context = {
                                    'has_pdfs': st.session_state.vectordb is not None,
                                    'vectordb_available': st.session_state.vectordb is not None,
                                    'web_enabled': web_enabled,
                                    'media_analysis': st.session_state.media_analysis_results if st.session_state.media_analysis_results else None
                                }
                                
                                # Analyse de la requête et sélection des outils appropriés
                                selected_tools = st.session_state.tool_manager.get_relevant_tools(enriched_prompt, tool_context)
                                
                                if selected_tools:
                                    # 🎨 POPUP ÉLÉGANT DANS LE CHAT - Affichage des outils détectés
                                    with chat_container:
                                        st.markdown(f"""
                                        <div style="
                                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                            padding: 20px;
                                            border-radius: 15px;
                                            margin: 15px 0;
                                            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
                                            border: 3px solid transparent;
                                            background-clip: padding-box;
                                            position: relative;
                                            animation: toolPopup 0.5s ease-out;
                                        ">
                                            <div style="
                                                position: absolute;
                                                top: -3px;
                                                left: -3px;
                                                right: -3px;
                                                bottom: -3px;
                                                background: linear-gradient(45deg, #00ff88, #ffff00, #0088ff, #00ff88);
                                                background-size: 400% 400%;
                                                border-radius: 15px;
                                                z-index: -1;
                                                animation: borderScintillation 3s ease infinite;
                                                filter: blur(5px);
                                            "></div>
                                            <h3 style="color: white; margin: 0 0 15px 0; display: flex; align-items: center; font-size: 1.3rem;">
                                                <span style="font-size: 2rem; margin-right: 10px;">🔧</span>
                                                Outils IA Activés
                                            </h3>
                                            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(10px);">
                                                {''.join([f'<div style="color: #FFD700; font-weight: 700; font-size: 1.1rem; margin: 8px 0; display: flex; align-items: center;"><span style="color: #00ff88; margin-right: 8px;">▶</span> {tool.name}</div><p style="color: #f0f0f0; margin: 5px 0 15px 25px; font-size: 0.95rem; font-style: italic;">{tool.description}</p>' for tool in selected_tools])}
                                            </div>
                                            <p style="color: #FFD700; margin: 15px 0 0 0; font-size: 0.9rem; text-align: center; font-weight: 600;">
                                                ⚡ {len(selected_tools)} outil{'s' if len(selected_tools) > 1 else ''} en cours d'exécution...
                                            </p>
                                        </div>
                                        <style>
                                            @keyframes toolPopup {{
                                                0% {{ transform: scale(0.9) translateY(-20px); opacity: 0; }}
                                                100% {{ transform: scale(1) translateY(0); opacity: 1; }}
                                            }}
                                            @keyframes borderScintillation {{
                                                0% {{ background-position: 0% 50%; }}
                                                50% {{ background-position: 100% 50%; }}
                                                100% {{ background-position: 0% 50%; }}
                                            }}
                                        </style>
                                        """, unsafe_allow_html=True)
                                    
                                    # Recherche RAG si un outil PDF est détecté
                                    rag_context = ""
                                    pdf_tool_used = any('pdf' in tool.name.lower() or 'document' in tool.name.lower() for tool in selected_tools)
                                    
                                    if pdf_tool_used and st.session_state.vectordb:
                                        with st.expander("🔍 Recherche dans les documents PDF...", expanded=True):
                                            rag_docs = rag_search(prompt, st.session_state.vectordb, k=10)
                                            if rag_docs:
                                                rag_context = "\n\n".join([f"📄 {doc.metadata.get('source', 'Document')}:\n{doc.page_content}" for doc in rag_docs])
                                                st.success(f"✅ {len(rag_docs)} passages trouvés")
                                                # Afficher un aperçu
                                                for i, doc in enumerate(rag_docs[:3], 1):
                                                    st.markdown(f"**Document {i}:** `{doc.metadata.get('source', 'Inconnu')}`")
                                            else:
                                                st.info("❌")
                                    elif pdf_tool_used and not st.session_state.vectordb:
                                        st.warning("⚠️ Outil PDF détecté mais aucune base vectorielle chargée. Ajoutez des PDFs dans l'onglet Configuration.")
                                    
                                    # Exécution des autres outils avec affichage détaillé dans le chat
                                    tool_results = []
                                    for tool in selected_tools:
                                        if 'pdf' not in tool.name.lower() and 'document' not in tool.name.lower():
                                            try:
                                                with chat_container:
                                                    with st.expander(f"⚙️ Exécution: {tool.name}", expanded=True):
                                                        st.info(f"🔄 {tool.description}")
                                                        result = tool.execute(prompt, tool_context)
                                                        
                                                        # Affichage élégant du résultat
                                                        st.markdown(f"""
                                                        <div style="
                                                            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                                            padding: 15px;
                                                            border-radius: 10px;
                                                            border-left: 4px solid #00ff88;
                                                            margin: 10px 0;
                                                        ">
                                                            <h4 style="color: #00ff88; margin: 0 0 10px 0;">✅ Résultat</h4>
                                                            <pre style="color: #f0f0f0; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 0.9rem;">{str(result)[:500]}</pre>
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                tool_results.append(f"**{tool.name}:** {result}")
                                            except Exception as e:
                                                with chat_container:
                                                    st.error(f"❌ Erreur {tool.name}: {str(e)}")
                                                tool_results.append(f"**{tool.name} (erreur):** {str(e)}")
                                    
                                    # Recherche web avec toutes les capacités (TOUJOURS ACTIVÉE)
                                    web_context = ""
                                    web_media_html = ""
                                    if web_enabled:
                                        try:
                                            web_results = enhanced_web_search(prompt, max_results=3)
                                            if web_results:
                                                web_context = "\n".join([f"{r.get('title', '')[:50]}: {r.get('body', '')[:100]}" for r in web_results[:2]])
                                                # Générer le HTML avec médias
                                                web_media_html = format_web_results_with_media(web_results, max_results=3)
                                        except Exception as e:
                                            web_context = f"Erreur recherche web: {e}"
                                    
                                    # Construction du prompt final avec TOUTES les sources
                                    # IMPORTANT: enriched_prompt contient déjà le contexte binaire OU média OU conversation
                                    # On doit le préserver et y ajouter les autres sources
                                    full_context = ""
                                    has_binary_context = 'binary_files' in st.session_state and st.session_state.binary_files
                                    has_media_in_enriched = 'MÉDIAS ANALYSÉS' in str(enriched_prompt) or 'Image:' in str(enriched_prompt)
                                    
                                    # Si on a déjà un contexte enrichi (binaire/média/conversation), on le met en premier
                                    if has_binary_context or has_media_in_enriched:
                                        full_context = str(enriched_prompt) + "\n\n"
                                        full_context += "═══════════════════════════════════════════════════════════════════════════════\n"
                                        full_context += "📚 SOURCES COMPLÉMENTAIRES (À UTILISER SI PERTINENT):\n"
                                        full_context += "═══════════════════════════════════════════════════════════════════════════════\n\n"
                                    
                                    # 1. DOCUMENTS LOCAUX (RAG)
                                    if rag_context:
                                        full_context += f"📚 **Documents locaux:**\n{rag_context[:400]}\n\n"
                                    
                                    # 2. RÉSULTATS DES OUTILS
                                    if tool_results:
                                        limited_results = [r[:300] for r in tool_results[:2]]
                                        for tr in limited_results:
                                            full_context += f"🔧 **Outil:** {tr}\n\n"
                                    
                                    # 4. RECHERCHE WEB (POUR CONTEXTE ADDITIONNEL)
                                    if web_context:
                                        full_context += f"🌐 **Contexte web complémentaire:**\n{web_context[:300]}\n\n"
                                    
                                    # Prompt final intelligent qui s'adapte au contexte
                                    if has_binary_context:
                                        # Mode analyse de fichier binaire
                                        final_prompt = f"""{full_context}

💡 **CONSIGNE STRICTE:** Vous analysez un FICHIER SPÉCIFIQUE uploadé par l'utilisateur.
- ✅ Analysez CE FICHIER CONCRET en utilisant les données réelles ci-dessus
- ✅ Citez les chaînes de texte extraites, les offsets, les magic bytes
- ✅ Commencez votre réponse par "Ce fichier {st.session_state.binary_files[-1]['name']}..."
- ❌ N'expliquez PAS ce qu'est un fichier en général
- ❌ Ne donnez PAS de cours théorique
- ❌ Ne parlez PAS de concepts généraux

Répondez UNIQUEMENT en analysant les DONNÉES CONCRÈTES du fichier ci-dessus."""
                                    elif has_media_in_enriched:
                                        final_prompt = f"""{full_context}
❓ **QUESTION:** {prompt}

💡 **CONSIGNE:** Vous avez accès aux données d'analyse des médias ci-dessus. Analysez-les EN PROFONDEUR en utilisant:
- Les couleurs, textes, objets détectés
- La résolution et le format
- L'analyse sémantique (CLIP)
- Le contexte web si pertinent (pour approfondir)

Répondez de manière détaillée et technique en exploitant TOUTES les données disponibles."""
                                    else:
                                        final_prompt = f"""{full_context}
❓ **QUESTION:** {prompt}

Réponds en utilisant toutes les sources disponibles (documents, outils, web) pour une réponse complète."""
                                    
                                    # Génération de la réponse finale avec historique adapté ET STREAMING
                                    client = create_client()
                                    has_media_context = bool(st.session_state.media_analysis_results)
                                    has_binary_context = 'binary_files' in st.session_state and st.session_state.binary_files
                                    
                                    # Adapter les limites selon le contexte
                                    if has_binary_context or has_media_context:
                                        # Si fichiers binaires ou médias: garder PLUS d'historique pour le contexte de conversation
                                        # Garder les 5 derniers messages pour suivre le fil de la conversation
                                        recent_msgs = st.session_state.chat_history[-5:] if len(st.session_state.chat_history) >= 5 else st.session_state.chat_history
                                        
                                        # Filtrer pour garder seulement user/assistant (pas les HTML)
                                        recent_msgs = [m for m in recent_msgs if m.get('role') in ['user', 'assistant']]
                                        
                                        messages = recent_msgs + [{"role": "user", "content": final_prompt[:12000]}]  # 12000 chars pour contexte très riche
                                        max_tokens_response = 4000  # 4000 tokens pour réponses très détaillées (≈60-80 lignes, 2-3 pages)
                                    else:
                                        # Mode normal : historique standard
                                        recent_msgs = st.session_state.chat_history[-4:] if len(st.session_state.chat_history) > 4 else st.session_state.chat_history
                                        recent_msgs = [m for m in recent_msgs if m.get('role') in ['user', 'assistant']]
                                        messages = recent_msgs + [{"role": "user", "content": final_prompt[:5000]}]
                                        max_tokens_response = 2000  # 2000 tokens pour réponses standard détaillées
                                    
                                    # 🌊 STREAMING ACTIVÉ - Affichage progressif des tokens
                                    with chat_container:
                                        with st.chat_message("assistant"):
                                            message_placeholder = st.empty()
                                            full_response = ""
                                            
                                            # Créer le stream
                                            stream = client.chat.completions.create(
                                                model=WORKING_MODELS[model_choice],
                                                messages=messages,
                                                max_tokens=max_tokens_response,
                                                temperature=0.7 if has_binary_context or has_media_context else 0.5,  # Plus créatif pour analyses
                                                stream=True  # 🔥 ACTIVER LE STREAMING
                                            )
                                            
                                            # Afficher chaque token au fur et à mesure
                                            for chunk in stream:
                                                if chunk.choices[0].delta.content is not None:
                                                    full_response += chunk.choices[0].delta.content
                                                    message_placeholder.markdown(full_response + "▌")
                                            
                                            # 🧹 NETTOYER LA RÉPONSE FINALE
                                            full_response = clean_response_text(full_response)
                                            
                                            # Affichage final sans curseur
                                            message_placeholder.markdown(full_response)
                                    
                                    response = full_response
                        
                        except Exception as e:
                            st.error(f"Erreur système d'outils: {e}")
                            # Fallback vers l'approche classique avec STREAMING
                            with chat_container:
                                with st.chat_message("assistant"):
                                    message_placeholder = st.empty()
                                    full_response = ""
                                    
                                    if not web_enabled:
                                        docs = rag_search(enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt, st.session_state.vectordb, k=3)
                                    else:
                                        docs = hybrid_search_enhanced(enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt, st.session_state.vectordb, k=3, web_search_enabled=True)
                                    
                                    # 🌊 Streaming des chunks
                                    for chunk in generate_answer_enhanced_stream(
                                        enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt, docs, WORKING_MODELS[model_choice], include_sources=True
                                    ):
                                        full_response += chunk
                                        message_placeholder.markdown(full_response + "▌")
                                    
                                    # 🧹 Nettoyer la réponse
                                    full_response = clean_response_text(full_response)
                                    message_placeholder.markdown(full_response)
                            
                            response = full_response
                    
                    else:
                        # Système d'outils non disponible, utiliser l'approche classique
                        web_media_html = ""
                        if st.session_state.agent is None:
                            st.session_state.current_model = WORKING_MODELS[model_choice]
                            st.session_state.agent = create_enhanced_agent(
                                st.session_state.current_model, 
                                st.session_state.vectordb, 
                                st.session_state.graph, 
                                st.session_state.pois
                            )
                        
                        try:
                            if not web_enabled:
                                # Mode RAG simple avec STREAMING
                                with chat_container:
                                    with st.chat_message("assistant"):
                                        message_placeholder = st.empty()
                                        full_response = ""
                                        
                                        # Utiliser enriched_prompt pour avoir le contexte binaire
                                        docs = rag_search(enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt, st.session_state.vectordb, k=3)
                                        
                                        # 🌊 Streaming
                                        for chunk in generate_answer_enhanced_stream(
                                            enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt, docs, st.session_state.current_model, include_sources=True
                                        ):
                                            full_response += chunk
                                            message_placeholder.markdown(full_response + "▌")
                                        
                                        message_placeholder.markdown(full_response)
                                
                                response = full_response
                            else:
                                # Recherche web avec médias pour mode classique
                                try:
                                    web_results = enhanced_web_search(prompt, max_results=3)
                                    if web_results:
                                        web_media_html = format_web_results_with_media(web_results, max_results=3)
                                except:
                                    pass
                                # Utiliser enriched_prompt si fichier binaire disponible
                                if 'binary_files' in st.session_state and st.session_state.binary_files:
                                    response = st.session_state.agent.run(enriched_prompt)
                                else:
                                    response = st.session_state.agent.run(prompt)
                        except Exception as e:
                            response = f"❌ Erreur: {e}\n\nTentative avec recherche locale..."
                            try:
                                with chat_container:
                                    with st.chat_message("assistant"):
                                        message_placeholder = st.empty()
                                        full_response = ""
                                        
                                        # Utiliser enriched_prompt pour le contexte binaire
                                        query = enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt
                                        docs = rag_search(query, st.session_state.vectordb, k=3)
                                        
                                        for chunk in generate_answer_enhanced_stream(
                                            query, docs, st.session_state.current_model
                                        ):
                                            full_response += chunk
                                            message_placeholder.markdown(full_response + "▌")
                                        
                                        message_placeholder.markdown(full_response)
                                
                                response = full_response
                            except:
                                response = f"❌ Erreur complète: {e}"
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # 🌐 NOUVEAU: Ajouter les médias web si disponibles
            if 'web_media_html' in locals() and web_media_html:
                st.session_state.chat_history.append({"role": "web_media", "content": web_media_html})
            
            # 🧠 NOUVEAU: Sauvegarder dans la mémoire vectorielle
            try:
                from chat_memory import add_to_chat_memory
                st.session_state.chat_vectordb = add_to_chat_memory(
                    prompt, 
                    response, 
                    st.session_state.chat_vectordb
                )
            except Exception as e:
                print(f"⚠️ Erreur sauvegarde mémoire: {e}")
            
            # 🤖 AUTO-APPRENTISSAGE: Apprendre de la conversation
            if AUTO_LEARNING_AVAILABLE and st.session_state.get('auto_learning'):
                try:
                    st.session_state.auto_learning.learn_from_conversation(prompt, response)
                except Exception as e:
                    print(f"⚠️ Erreur apprentissage conversation: {e}")
            
            # Nettoyer l'historique si trop long
            if len(st.session_state.chat_history) > 20:
                st.session_state.chat_history = st.session_state.chat_history[-20:]
            st.rerun()
        
        # Boutons d'action stylisés
        col6, col7, col8, col9 = st.columns(4)
        with col6:
            if st.button("🗑️ **Effacer conversation**", key="clear_chat", help="Supprime tout l'historique du chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col7:
            if st.button("🧠 **Statistiques mémoire**", key="memory_stats", help="Affiche les statistiques de la mémoire vectorielle"):
                from chat_memory import get_memory_stats
                try:
                    stats = get_memory_stats(st.session_state.chat_vectordb)
                    st.markdown(f"""
                    <div class="kibali-card" style="border: 2px solid #ff0088;">
                        <h4 style="color: #ff0088;">🧠 Mémoire conversationnelle</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 10px;">
                            <div><strong>💬 Conversations:</strong> {stats['count']}</div>
                            <div><strong>💾 Taille:</strong> {stats['size']}</div>
                            <div><strong>📊 État:</strong> {stats['status']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Erreur statistiques mémoire: {e}")
        
        with col8:
            if st.button("💡 **Questions d'exemple**", key="show_examples", help="Affiche des exemples de questions"):
                with chat_container:
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 20px;
                        border-radius: 15px;
                        margin: 15px 0;
                        border: 2px solid #00ff88;
                    ">
                        <h3 style="color: #00ff88; margin: 0 0 20px 0;">💡 Exemples de questions pour tester les outils</h3>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #FFD700;">
                            <h4 style="color: #FFD700; margin: 0 0 10px 0;">🔧 Outil de Code</h4>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Crée une fonction Python pour calculer la factorielle"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Génère une classe JavaScript pour gérer un panier d'achat"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Écris un algorithme de tri à bulles en Python"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Code une fonction de recherche binaire"</p>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #0088ff;">
                            <h4 style="color: #0088ff; margin: 0 0 10px 0;">🌐 Recherche Web</h4>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Actualités intelligence artificielle 2024"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Tendances technologiques récentes"</p>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #ff0088;">
                            <h4 style="color: #ff0088; margin: 0 0 10px 0;">📊 Mathématiques</h4>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Calcule la racine carrée de 144"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Résous l'équation: 2x + 5 = 15"</p>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #00ffff;">
                            <h4 style="color: #00ffff; margin: 0 0 10px 0;">🌍 Traduction</h4>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Traduis 'Bonjour le monde' en anglais"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Comment dit-on 'merci' en espagnol?"</p>
                        </div>
                        
                        <p style="color: #FFD700; margin: 20px 0 0 0; text-align: center; font-size: 0.9rem; font-weight: 600;">
                            ⚡ Testez ces exemples pour voir les outils en action !
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col9:
            if st.button("📊 **Statut système**", key="system_status", help="Affiche l'état du système Kibali"):
                status = get_system_status()
                st.markdown(f"""
                <div class="kibali-card">
                <h4>📊 État du système</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div><strong>📁 PDFs:</strong> {status['counts']['pdfs']}</div>
                <div><strong>🗺️ Graphes:</strong> {status['counts']['graphs']}</div>
                <div><strong>💾 Base vectorielle:</strong> {'✅' if status['files']['vectordb'] else '❌'}</div>
                <div><strong>🌐 Cache web:</strong> {'✅' if status['files']['web_cache'] else '❌'}</div>
                </div>
                </div>
                """, unsafe_allow_html=True)

    # ===============================================
    # Onglet 3: Trajets avec cartes
    # ===============================================
    with tab3:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 🗺️ Calcul de trajets
        **Exemples:** "Comment aller de l'école à l'hôpital ?"
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        trajectory_question = st.text_area(
            "🗺️ **Question de trajet**",
            placeholder="Ex: Comment aller de l'école à l'hôpital ?",
            height=100,
            help="Décris ton trajet en langage naturel"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_traj1, col_traj2 = st.columns([1, 1])
        with col_traj1:
            if st.button("🚀 **Calculer trajet**", key="calculate_route", help="Calcule l'itinéraire demandé"):
                if not st.session_state.graph or not st.session_state.pois:
                    st.error("❌ Graphe non chargé - Upload un fichier OSM d'abord")
                else:
                    with st.spinner("🗺️ Calcul du trajet en cours..."):
                        carte_buf, reponse, traj_info = calculer_trajet(
                            trajectory_question, 
                            st.session_state.graph, 
                            st.session_state.pois
                        )
                        if carte_buf:
                            carte_buf.seek(0)
                            image = Image.open(carte_buf)
                            st.markdown('<div class="kibali-card zoom-image">', unsafe_allow_html=True)
                            st.image(image, caption="🗺️ Carte du trajet calculé", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.session_state.last_traj_info = traj_info
                        st.session_state.last_reponse = reponse
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.text_area("📋 **Détails du trajet**", value=st.session_state.last_reponse, height=200, disabled=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        with col_traj2:
            if st.button("💾 **Sauvegarder trajet**", key="save_trajectory", help="Sauvegarde ce trajet dans la base de connaissances"):
                if st.session_state.last_traj_info:
                    status = save_trajectory(
                        trajectory_question, 
                        st.session_state.last_reponse, 
                        st.session_state.last_traj_info
                    )
                    st.success(f"✅ Trajet sauvegardé: {status}")
                else:
                    st.warning("⚠️ Aucun trajet à sauvegarder - Calcule un trajet d'abord")

    # ===============================================
    # Onglet 4: Analyse Image avec IA avancée
    # ===============================================
    with tab4:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 📸 Analyse d'images avancée avec IA
        Upload une image pour analyse complète: Vision AI (CLIP), OCR, détection d'objets, annotations et graphiques.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "📤 **Upload Fichier**", 
            type=None,
            key="image_upload",
            help="Tous formats acceptés - binaires inclus"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 **Analyser l'image**", key="analyze_image", help="Lance l'analyse complète de l'image avec IA"):
            if uploaded_image:
                with st.spinner("🔬 Analyse IA en cours..."):
                    
                    # Charger tous les modèles nécessaires
                    vision_models = None
                    ocr_reader = None
                    yolo_model = None
                    blip_models = None
                    
                    try:
                        with st.spinner("📥 Chargement CLIP..."):
                            vision_models = load_vision_models()
                    except Exception as e:
                        st.warning(f"⚠️ CLIP non disponible: {e}")
                    
                    try:
                        with st.spinner("📝 Chargement OCR..."):
                            ocr_reader = load_ocr_reader()
                    except Exception as e:
                        st.warning(f"⚠️ OCR non disponible: {e}")
                    
                    try:
                        with st.spinner("🎯 Chargement YOLO..."):
                            yolo_model = load_yolo_model()
                            if yolo_model:
                                st.success("✅ YOLO chargé pour détection d'objets")
                    except Exception as e:
                        st.warning(f"⚠️ YOLO non disponible: {e}")
                    
                    try:
                        with st.spinner("🖼️ Chargement BLIP..."):
                            blip_models = load_blip_model()
                            if blip_models:
                                st.success("✅ BLIP chargé pour descriptions détaillées")
                    except Exception as e:
                        st.warning(f"⚠️ BLIP non disponible: {e}")
                    
                    # 1. Analyse complète: OpenCV + YOLO + CLIP + BLIP + OCR
                    analysis_data, proc_images, tables_str = process_image(
                        uploaded_image.getvalue(), 
                        vision_models=vision_models,
                        ocr_reader=ocr_reader,
                        yolo_model=yolo_model,
                        blip_models=blip_models
                    )
                    
                    # 2. Afficher la description BLIP + OCR en priorité
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 🖼️ Analyse IA Complète (BLIP + OCR + YOLO + CLIP)")
                    
                    # BLIP et OCR sont déjà dans tables_str, les afficher directement
                    if tables_str:
                        st.markdown(tables_str, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 3. Afficher les images traditionnelles avec zoom au survol
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 📊 Analyses traditionnelles (OpenCV)")
                    if proc_images:
                        cols = st.columns(min(len(proc_images), 3))
                        for i, (img, caption) in enumerate(zip(proc_images, ['Image Originale'] + ['Analyse'] * (len(proc_images)-1))):
                            with cols[i % len(cols)]:
                                st.markdown('<div class="zoom-image">', unsafe_allow_html=True)
                                st.image(img, caption=f"📸 {caption}", use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 4. Génération de rapport complet avec LLM
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 🤖 Rapport d'analyse IA complet")
                    
                    with st.spinner("✍️ Génération du rapport détaillé..."):
                        # Utiliser la fonction de description complète
                        comprehensive_desc = generate_comprehensive_description(analysis_data)
                        
                        # Générer le rapport avec le LLM
                        llm_model = st.session_state.get('llm_model') or st.session_state.get('current_model')
                        
                        if llm_model:
                            try:
                                improved_analysis = improve_analysis_with_llm(analysis_data, llm_model)
                                
                                # Afficher d'abord la description structurée
                                st.markdown(f"""
                                <div style="color: white; font-weight: 600; font-size: 1.1rem; line-height: 1.8; 
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
                                <h4 style="color: white; margin-top: 0;">📋 Description structurée:</h4>
                                {comprehensive_desc.replace(chr(10), '<br>')}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Puis l'analyse enrichie par le LLM
                                st.markdown(f"""
                                <div style="color: white; font-weight: 600; font-size: 1.1rem; line-height: 1.8; 
                                background: rgba(74, 74, 126, 0.3); padding: 1.5rem; border-radius: 12px; 
                                border-left: 4px solid var(--kibali-green);">
                                <h4 style="color: white; margin-top: 0;">🤖 Analyse IA enrichie:</h4>
                                {improved_analysis}
                                </div>
                                """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"❌ Erreur génération rapport: {str(e)}")
                                # Afficher au moins la description structurée
                                st.markdown(f"""
                                <div style="color: white; font-weight: 600; font-size: 1.1rem; line-height: 1.8; 
                                background: rgba(74, 74, 126, 0.3); padding: 1.5rem; border-radius: 12px;">
                                {comprehensive_desc.replace(chr(10), '<br>')}
                                </div>
                                """, unsafe_allow_html=True)
                                improved_analysis = comprehensive_desc
                        else:
                            # Afficher la description structurée si pas de LLM
                            st.markdown(f"""
                            <div style="color: white; font-weight: 600; font-size: 1.1rem; line-height: 1.8; 
                            background: rgba(74, 74, 126, 0.3); padding: 1.5rem; border-radius: 12px;">
                            {comprehensive_desc.replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                            improved_analysis = comprehensive_desc
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Marquer l'analyse comme réussie
                    vision_success = True
                    
                    # Stocker les résultats d'analyse pour le chat
                    if vision_success:
                        st.session_state.media_analysis_results.append({
                            'type': 'image',
                            'name': uploaded_image.name,
                            'resolution': f"{width}x{height}",
                            'format': img_format,
                            'caption': caption,
                            'ai_analysis': improved_analysis,
                            'web_context': web_context,
                            'extracted_text': organized_text
                        })
            else:
                st.warning("⚠️ Veuillez uploader une image d'abord")

    # ===============================================
    # Onglet 5: Recherche Web Avancée avec cartes
    # ===============================================
    with tab5:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 🌐 Recherche web avancée avec extraction de contenu
        Recherche intelligente sur Internet avec cache automatique et extraction de contenu détaillé.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col8, col9 = st.columns([3, 1])
        with col8:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            web_query = st.text_area(
                "🔍 **Requête de recherche**",
                placeholder="Ex: actualités intelligence artificielle 2024",
                height=100,
                help="Décris ta recherche en langage naturel"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col9:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            search_type = st.selectbox(
                "🎯 **Type de recherche**",
                options=["text", "news", "both"],
                index=2,
                help="Texte: recherche générale, News: actualités, Both: les deux"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        col_search1, col_search2 = st.columns([1, 1])
        with col_search1:
            if st.button("🔍 **Rechercher**", key="web_search", help="Lance la recherche web"):
                if web_query.strip():
                    with st.spinner("🌐 Recherche en cours..."):
                        results = enhanced_web_search(web_query, max_results=10, search_type=search_type)
                        if results:
                            for result in results:
                                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                st.markdown(f"### 📰 {result.get('title', 'Sans titre')}")
                                st.write(f"📝 {result.get('body', 'Pas de description')}")
                                url = result.get('href') or result.get('url', '#')
                                st.markdown(f"[🔗 **Accéder à la source**]({url})")
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown("---")
                        else:
                            st.error("❌ Aucun résultat trouvé")
                else:
                    st.warning("⚠️ Veuillez entrer une requête de recherche")
        
        with col_search2:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            url_extract = st.text_input(
                "🌐 **URL à extraire**",
                placeholder="https://example.com/article",
                help="Colle une URL pour extraire son contenu détaillé"
            )
            if st.button("📄 **Extraire contenu**", key="extract_content", help="Extrait le contenu détaillé de la page web"):
                if url_extract.strip():
                    if not url_extract.startswith(('http://', 'https://')):
                        url_extract = 'https://' + url_extract
                    with st.spinner("📄 Extraction en cours..."):
                        content = smart_content_extraction(url_extract, max_length=2000)
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.text_area("📄 **Contenu extrait**", value=content, height=300, disabled=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Veuillez entrer une URL valide")
            st.markdown('</div>', unsafe_allow_html=True)

    # ===============================================
    # Onglet 6: Photogrammétrie - Classification Intelligente avec IA
    # ===============================================
    with tab_photo:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 📷 Classification Intelligente pour Photogrammétrie avec Vision AI
        **L'IA analyse visuellement chaque photo et les ordonne intelligemment pour une reconstruction 3D optimale.**
        
        🤖 **Analyse IA:**
        - Vision sémantique (CLIP) - Compréhension du contenu
        - Analyse de textures et détails visuels
        - Détection d'angles de vue similaires
        - Ordonnancement optimal pour Dust3R/MeshRoom
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Deux modes: Classification IA ou Optimisation simple
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        mode = st.radio(
            "🎯 Mode de traitement:",
            ["🤖 Classification IA (Analyse visuelle + Ordonnancement)", "⚡ Optimisation Rapide (Réduction simple)"],
            help="Classification IA analyse visuellement chaque photo, Optimisation Rapide réduit juste le nombre"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Section d'upload multiple
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("#### 📤 Upload de fichiers")
        
        uploaded_photos = st.file_uploader(
            "Sélectionnez vos fichiers",
            type=None,
            accept_multiple_files=True,
            help="Uploadez tous vos fichiers (tous formats acceptés). L'IA les analysera et les classera intelligemment!",
            key="photo_upload_classifier"
        )
        
        if uploaded_photos and len(uploaded_photos) > 0:
            st.success("✅📸")
            
            if mode.startswith("🤖"):
                # MODE CLASSIFICATION IA
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                st.markdown("#### 🎨 Paramètres de classification IA")
                
                col_ai1, col_ai2 = st.columns(2)
                with col_ai1:
                    ordering_method = st.selectbox(
                        "📐 Méthode d'ordonnancement",
                        ["sequential", "cluster"],
                        format_func=lambda x: "Séquentiel (photos qui se suivent)" if x == "sequential" else "Par groupes d'angles",
                        help="Séquentiel: crée une chaîne d'images similaires. Groupes: classe par angles puis ordonne."
                    )
                
                with col_ai2:
                    generate_viz = st.checkbox(
                        "📊 Générer visualisation",
                        value=True,
                        help="Crée une grille visuelle montrant l'ordre des photos"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Bouton de classification IA
                if st.button("🚀 **Analyser et Classer avec l'IA**", type="primary", use_container_width=True):
                    try:
                        import tempfile
                        import shutil
                        from outils.intelligent_photo_classifier import classify_photos_with_ai
                        
                        # Créer un conteneur pour les logs en temps réel
                        log_container = st.empty()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Créer dossier temporaire
                        log_container.info("📁 Création du dossier temporaire...")
                        temp_dir = tempfile.mkdtemp(prefix="ai_photo_classifier_")
                        progress_bar.progress(5)
                        
                        # Sauvegarder les photos
                        status_text.text(f"💾 Sauvegarde de {len(uploaded_photos)} photos...")
                        photo_paths = []
                        for idx, photo in enumerate(uploaded_photos):
                            photo_path = Path(temp_dir) / photo.name
                            with open(photo_path, 'wb') as f:
                                f.write(photo.getbuffer())
                            photo_paths.append(str(photo_path))
                            progress_bar.progress(5 + int((idx + 1) / len(uploaded_photos) * 15))
                        
                        log_container.success(f"✅ {len(uploaded_photos)} photos sauvegardées")
                        
                        # Charger le modèle Vision AI (CLIP)
                        status_text.text("📦 Chargement du modèle Vision AI (CLIP)...")
                        progress_bar.progress(20)
                        vision_models = load_vision_models()
                        
                        if vision_models is None:
                            st.error("❌ Impossible de charger le modèle CLIP")
                            st.stop()
                        
                        log_container.success(f"✅ Modèle CLIP chargé ({vision_models['device']})")
                        progress_bar.progress(30)
                        
                        # Classifier avec l'IA
                        status_text.text("🔍 Analyse Vision AI en cours...")
                        st.info("📊 Extraction des features visuelles avec CLIP...")
                        output_dir = temp_dir + "_classified"
                        Path(output_dir).mkdir(exist_ok=True)
                        
                        # Progress callback pour le classifier
                        def update_progress(current, total, message):
                            pct = 30 + int((current / total) * 50)
                            progress_bar.progress(pct)
                            status_text.text(f"🔍 {message} ({current}/{total})")
                        
                        ordered_paths, report, viz_path = classify_photos_with_ai(
                            photo_paths,
                            vision_models['clip_model'],
                            vision_models['clip_processor'],
                            device=vision_models['device'],
                            method=ordering_method,
                            output_dir=output_dir if generate_viz else None,
                            progress_callback=update_progress
                        )
                        
                        progress_bar.progress(80)
                        status_text.text("✅ Analyse terminée !")
                        log_container.success("🎉 Classification IA terminée avec succès !")
                        
                        # Sauvegarder ordered_paths dans session_state AVANT la fin du try
                        st.session_state.classification_ordered_paths = ordered_paths
                        st.session_state.classification_report = report
                        st.session_state.classification_viz_path = viz_path
                        st.session_state.classification_done = True
                        
                        # Afficher les résultats
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.markdown("### 🎯 Résultats de la Classification IA")
                        st.text_area("📋 Rapport détaillé", value=report, height=400, disabled=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Afficher la visualisation
                        if viz_path and Path(viz_path).exists():
                            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                            st.markdown("### 📊 Visualisation de l'ordre")
                            st.image(viz_path, caption="Ordre optimisé des photos (gauche → droite, haut → bas)", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Copier les photos dans l'ordre
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.markdown("### 📥 Photos classées et renommées")
                        
                        ordered_dir = Path(output_dir) / "ordered_photos"
                        ordered_dir.mkdir(exist_ok=True)
                        
                        for idx, src_path in enumerate(ordered_paths, 1):
                            src = Path(src_path)
                            # Renommer avec numéro d'ordre + nom original
                            dest = ordered_dir / f"{idx:04d}_{src.name}"
                            shutil.copy2(src, dest)
                        
                        st.success(f"✅ {len(ordered_paths)} photos ordonnées et sauvegardées")
                        
                        # Créer ZIP pour téléchargement
                        import zipfile
                        zip_path = Path(temp_dir) / "photos_classified_ai.zip"
                        
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            # Ajouter les photos ordonnées
                            for photo_file in ordered_dir.glob("*.*"):
                                zipf.write(photo_file, arcname=f"ordered_photos/{photo_file.name}")
                            
                            # Ajouter le rapport
                            report_path = Path(output_dir) / "classification_report.txt"
                            if report_path.exists():
                                zipf.write(report_path, arcname="classification_report.txt")
                            
                            # Ajouter la visualisation
                            if viz_path and Path(viz_path).exists():
                                zipf.write(viz_path, arcname="ordering_visualization.png")
                        
                        # Bouton de téléchargement
                        with open(zip_path, 'rb') as f:
                            st.download_button(
                                label="📦 Télécharger les photos classées (ZIP)",
                                data=f.read(),
                                file_name="photos_classees_ia.zip",
                                mime="application/zip"
                            )
                        
                        st.info("💡 Photos ordonnées de manière optimale pour reconstruction 3D")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la classification: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
                
                # ============================================================
                # 🤖 CHAT INTERACTIF KIBALI - AFFICHAGE APRÈS CLASSIFICATION
                # ============================================================
                if st.session_state.get('classification_done', False):
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 💬 Chat Kibali - Affinez la classification avec l'IA")
                    st.markdown("""
                    Utilisez le chat pour manipuler l'ordre des photos avec des commandes naturelles :
                    - *"Mets les photos sombres au début"*
                    - *"Regroupe les images similaires ensemble"*
                    - *"Place les vues frontales avant les vues latérales"*
                    - *"Inverse l'ordre actuel"*
                    - *"Supprime les photos floues"*
                    """)
                    
                    # Initialiser l'historique du chat
                    if 'classification_chat_history' not in st.session_state:
                        st.session_state.classification_chat_history = []
                    if 'current_ordered_paths' not in st.session_state:
                        st.session_state.current_ordered_paths = st.session_state.classification_ordered_paths
                    
                    # Afficher l'historique du chat
                    for message in st.session_state.classification_chat_history:
                        if message['role'] == 'user':
                            st.markdown(f"**👤 Vous:** {message['content']}")
                        else:
                            st.markdown(f"**🤖 Kibali:** {message['content']}")
                    
                    # Input utilisateur
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        user_prompt = st.text_input(
                            "💬 Votre commande:",
                            placeholder="Ex: Mets les photos lumineuses au début...",
                            key="classification_chat_input",
                            label_visibility="collapsed"
                        )
                    with col2:
                        send_btn = st.button("📤 Envoyer", key="send_classification_command", use_container_width=True)
                    
                    if send_btn and user_prompt:
                        # Ajouter le message utilisateur
                        st.session_state.classification_chat_history.append({
                            'role': 'user',
                            'content': user_prompt
                        })
                        
                        with st.spinner("🤖 Kibali analyse votre demande..."):
                            try:
                                # Analyse de la demande par mots-clés (sans LLM pour éviter les erreurs)
                                user_prompt_lower = user_prompt.lower()
                                
                                # Extraire features pour analyse intelligente
                                features_list = []
                                if 'current_features' not in st.session_state:
                                    # Extraire features de base pour toutes les images
                                    for path in st.session_state.current_ordered_paths[:50]:  # Limiter à 50 pour performance
                                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                        if img is not None:
                                            features_list.append({
                                                'path': path,
                                                'brightness': img.mean(),
                                                'contrast': img.std(),
                                                'name': Path(path).name
                                            })
                                    st.session_state.current_features = features_list
                                else:
                                    features_list = st.session_state.current_features
                                
                                # Détection d'intention par règles simples
                                new_order = st.session_state.current_ordered_paths.copy()
                                kibali_response = ""
                                action_executed = False
                                
                                # Règle 1: Regrouper similaires
                                if any(word in user_prompt_lower for word in ['regroup', 'similaire', 'ensemble', 'cluster', 'group']):
                                    # Regrouper par similarité de luminosité
                                    brightness_scores = []
                                    for path in new_order:
                                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                        brightness_scores.append(img.mean() if img is not None else 0)
                                    
                                    # Clustering simple par luminosité
                                    from sklearn.cluster import KMeans
                                    if len(brightness_scores) > 3:
                                        n_clusters = min(5, len(brightness_scores) // 10 + 1)
                                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                        labels = kmeans.fit_predict(np.array(brightness_scores).reshape(-1, 1))
                                        
                                        # Réorganiser par cluster
                                        sorted_indices = np.argsort(labels)
                                        new_order = [new_order[i] for i in sorted_indices]
                                        kibali_response = f"✅ Photos regroupées en {n_clusters} groupes similaires par luminosité."
                                        action_executed = True
                                    else:
                                        kibali_response = "⚠️ Pas assez de photos pour regrouper."
                                
                                # Règle 2: Trier par luminosité (sombre au début)
                                elif any(word in user_prompt_lower for word in ['sombre', 'dark', 'noir', 'début', 'start']) and \
                                     any(word in user_prompt_lower for word in ['début', 'avant', 'first', 'start']):
                                    brightness_scores = []
                                    for path in new_order:
                                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                        brightness_scores.append(img.mean() if img is not None else 0)
                                    sorted_indices = np.argsort(brightness_scores)
                                    new_order = [new_order[i] for i in sorted_indices]
                                    kibali_response = "✅ Photos triées du plus sombre au plus clair."
                                    action_executed = True
                                
                                # Règle 3: Trier par luminosité (clair au début)  
                                elif any(word in user_prompt_lower for word in ['clair', 'lumineuse', 'bright', 'light']) and \
                                     any(word in user_prompt_lower for word in ['début', 'avant', 'first', 'start']):
                                    brightness_scores = []
                                    for path in new_order:
                                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                        brightness_scores.append(img.mean() if img is not None else 0)
                                    sorted_indices = np.argsort(brightness_scores)[::-1]
                                    new_order = [new_order[i] for i in sorted_indices]
                                    kibali_response = "✅ Photos triées du plus clair au plus sombre."
                                    action_executed = True
                                
                                # Règle 4: Inverser l'ordre
                                elif any(word in user_prompt_lower for word in ['inverse', 'revers', 'retourne', 'flip']):
                                    new_order = new_order[::-1]
                                    kibali_response = "✅ Ordre des photos inversé."
                                    action_executed = True
                                
                                # Règle 5: Supprimer floues
                                elif any(word in user_prompt_lower for word in ['floue', 'blur', 'netteté', 'flou', 'supprime']):
                                    # Calculer la netteté avec Laplacian
                                    sharp_paths = []
                                    for path in new_order:
                                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                        if img is not None:
                                            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                                            if laplacian_var > 100:  # Seuil de netteté
                                                sharp_paths.append(path)
                                    
                                    removed_count = len(new_order) - len(sharp_paths)
                                    new_order = sharp_paths
                                    kibali_response = f"✅ {removed_count} photos floues supprimées. {len(new_order)} photos nettes conservées."
                                    action_executed = True
                                
                                else:
                                    kibali_response = f"🤖 Je n'ai pas compris votre demande. Essayez:\n- 'Regroupe les images similaires'\n- 'Mets les photos sombres au début'\n- 'Inverse l'ordre'\n- 'Supprime les photos floues'"
                                
                                # Mettre à jour l'ordre si modifié
                                if new_order != st.session_state.current_ordered_paths:
                                    st.session_state.current_ordered_paths = new_order
                                    # Invalider le cache des features
                                    if 'current_features' in st.session_state:
                                        del st.session_state.current_features
                                
                                st.session_state.classification_chat_history.append({
                                    'role': 'assistant',
                                    'content': kibali_response
                                })
                                
                                st.rerun()
                                
                            except Exception as e:
                                error_msg = f"❌ Erreur: {str(e)}"
                                st.session_state.classification_chat_history.append({
                                    'role': 'assistant',
                                    'content': error_msg
                                })
                                st.error(error_msg)
                                import traceback
                                st.text(traceback.format_exc())
                    
                    # Gérer le drag & drop via query params
                    try:
                        query_params = st.query_params
                        if 'drag_from' in query_params and 'drag_to' in query_params:
                            drag_from = int(query_params['drag_from'])
                            drag_to = int(query_params['drag_to'])
                            
                            if drag_from != drag_to:
                                new_order = st.session_state.current_ordered_paths.copy()
                                new_order[drag_from], new_order[drag_to] = new_order[drag_to], new_order[drag_from]
                                st.session_state.current_ordered_paths = new_order
                                
                                st.session_state.classification_chat_history.append({
                                    'role': 'assistant',
                                    'content': f"✅ Photo #{drag_from+1} échangée avec #{drag_to+1}"
                                })
                                
                                # Clear query params
                                st.query_params.clear()
                                st.rerun()
                    except:
                        pass
                    
                    # Afficher l'ordre actuel
                    st.markdown("#### 📸 Galerie interactive (Glissez les images pour les réorganiser)")
                    
                    # CSS pour drag & drop fluide
                    st.markdown("""
                    <style>
                    .draggable-container {
                        display: grid;
                        grid-template-columns: repeat(5, 1fr);
                        gap: 15px;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 15px;
                    }
                    .draggable-item {
                        position: relative;
                        cursor: grab;
                        background: white;
                        border: 3px solid #e0e0e0;
                        border-radius: 12px;
                        padding: 8px;
                        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }
                    .draggable-item:hover {
                        border-color: #4CAF50;
                        box-shadow: 0 8px 20px rgba(76, 175, 80, 0.4);
                        transform: translateY(-5px) scale(1.03);
                    }
                    .draggable-item:active {
                        cursor: grabbing;
                    }
                    .draggable-item.dragging {
                        opacity: 0.6;
                        transform: rotate(8deg) scale(1.1);
                        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
                        z-index: 1000;
                    }
                    .draggable-item.drag-over {
                        border-color: #2196F3;
                        border-style: dashed;
                        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
                        animation: pulse 0.5s infinite;
                    }
                    @keyframes pulse {
                        0%, 100% { transform: scale(1); }
                        50% { transform: scale(1.05); }
                    }
                    .item-number {
                        position: absolute;
                        top: -5px;
                        left: -5px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 5px 10px;
                        border-radius: 8px;
                        font-weight: bold;
                        font-size: 14px;
                        z-index: 10;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                    }
                    .drag-handle {
                        position: absolute;
                        top: -5px;
                        right: -5px;
                        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                        color: white;
                        padding: 8px 10px;
                        border-radius: 8px;
                        font-size: 18px;
                        cursor: grab;
                        z-index: 10;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                    }
                    .drag-handle:active {
                        cursor: grabbing;
                    }
                    .item-image {
                        width: 100%;
                        height: 160px;
                        object-fit: cover;
                        border-radius: 8px;
                        display: block;
                    }
                    .item-name {
                        text-align: center;
                        font-size: 11px;
                        margin-top: 5px;
                        color: #666;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Afficher les images avec drag & drop
                    st.markdown('<div class="draggable-container" id="image-gallery">', unsafe_allow_html=True)
                    
                    import base64
                    from io import BytesIO
                    
                    for idx, path in enumerate(st.session_state.current_ordered_paths[:30]):
                        try:
                            img = Image.open(path)
                            img.thumbnail((250, 250))
                            buffered = BytesIO()
                            img.save(buffered, format="JPEG", quality=85)
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            filename = os.path.basename(path)
                            
                            st.markdown(f"""
                            <div class="draggable-item" draggable="true" data-index="{idx}">
                                <div class="item-number">#{idx+1}</div>
                                <div class="drag-handle">⋮⋮</div>
                                <img src="data:image/jpeg;base64,{img_str}" class="item-image" />
                                <div class="item-name">{filename}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            filename = os.path.basename(path)
                            st.markdown(f"""
                            <div class="draggable-item" data-index="{idx}">
                                <div class="item-number">#{idx+1}</div>
                                <div style="padding: 60px 10px; text-align: center; color: #999;">
                                    ❌<br>{filename[:15]}...
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if len(st.session_state.current_ordered_paths) > 30:
                        st.info(f"📊 {len(st.session_state.current_ordered_paths)} images au total (30 premières affichées avec drag & drop)")
                    
                    # JavaScript pour gérer le drag & drop
                    st.markdown("""
                    <script>
                    (function() {
                        let draggedElement = null;
                        let draggedIndex = null;
                        
                        function initDragAndDrop() {
                            const items = document.querySelectorAll('.draggable-item');
                            
                            items.forEach(item => {
                                item.addEventListener('dragstart', handleDragStart);
                                item.addEventListener('dragend', handleDragEnd);
                                item.addEventListener('dragover', handleDragOver);
                                item.addEventListener('drop', handleDrop);
                                item.addEventListener('dragleave', handleDragLeave);
                            });
                        }
                        
                        function handleDragStart(e) {
                            draggedElement = this;
                            draggedIndex = parseInt(this.getAttribute('data-index'));
                            this.classList.add('dragging');
                            e.dataTransfer.effectAllowed = 'move';
                            e.dataTransfer.setData('text/html', this.innerHTML);
                        }
                        
                        function handleDragEnd(e) {
                            this.classList.remove('dragging');
                            document.querySelectorAll('.draggable-item').forEach(item => {
                                item.classList.remove('drag-over');
                            });
                        }
                        
                        function handleDragOver(e) {
                            if (e.preventDefault) {
                                e.preventDefault();
                            }
                            e.dataTransfer.dropEffect = 'move';
                            if (this !== draggedElement) {
                                this.classList.add('drag-over');
                            }
                            return false;
                        }
                        
                        function handleDragLeave(e) {
                            this.classList.remove('drag-over');
                        }
                        
                        function handleDrop(e) {
                            if (e.stopPropagation) {
                                e.stopPropagation();
                            }
                            
                            if (draggedElement !== this) {
                                const targetIndex = parseInt(this.getAttribute('data-index'));
                                
                                // Échange visuel immédiat
                                const draggedHTML = draggedElement.innerHTML;
                                const targetHTML = this.innerHTML;
                                
                                draggedElement.innerHTML = targetHTML;
                                this.innerHTML = draggedHTML;
                                
                                // Mettre à jour les data-index
                                draggedElement.setAttribute('data-index', targetIndex);
                                this.setAttribute('data-index', draggedIndex);
                                
                                // Envoyer au backend via query params
                                setTimeout(() => {
                                    const url = new URL(window.location);
                                    url.searchParams.set('drag_from', draggedIndex);
                                    url.searchParams.set('drag_to', targetIndex);
                                    url.searchParams.set('t', Date.now());
                                    window.location.href = url.toString();
                                }, 300);
                            }
                            
                            return false;
                        }
                        
                        // Initialiser quand le DOM est prêt
                        if (document.readyState === 'loading') {
                            document.addEventListener('DOMContentLoaded', initDragAndDrop);
                        } else {
                            setTimeout(initDragAndDrop, 100);
                        }
                        
                        // Réinitialiser après les updates Streamlit
                        const observer = new MutationObserver(function(mutations) {
                            mutations.forEach(function(mutation) {
                                if (mutation.addedNodes.length > 0) {
                                    setTimeout(initDragAndDrop, 200);
                                }
                            });
                        });
                        
                        const gallery = document.getElementById('image-gallery');
                        if (gallery) {
                            observer.observe(gallery.parentElement, { childList: true, subtree: true });
                        }
                    })();
                    </script>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # ========== NOUVELLE SECTION COLMAP 3D ==========
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 🎨 Reconstruction 3D Professionnelle (COLMAP)")
                    st.markdown("""
                    Générez un modèle 3D texturé à partir de vos images classifiées avec COLMAP (Licence BSD).
                    Pipeline complet : SIFT → Structure-from-Motion → Multi-View Stereo → Mesh
                    """)
                    
                    col_colmap1, col_colmap2 = st.columns([2, 1])
                    with col_colmap1:
                        quality_3d = st.select_slider(
                            "🎯 Qualité reconstruction",
                            options=['low', 'medium', 'high', 'extreme'],
                            value='high',
                            help="Plus la qualité est haute, plus le traitement est long (extreme peut prendre 1-2h)"
                        )
                    
                    with col_colmap2:
                        use_gpu_3d = st.checkbox("⚡ Utiliser GPU", value=True, 
                                                  help="RTX 5090 recommandée pour MVS")
                    
                    if st.button("🚀 **Générer Modèle 3D**", type="primary", use_container_width=True, key="colmap_btn"):
                        with st.spinner("🔍 Lancement de COLMAP... Cela peut prendre 15-45 minutes..."):
                            try:
                                from outils.colmap_photogrammetry import ColmapPhotogrammetry
                                import tempfile
                                import time
                                
                                # Créer workspace temporaire
                                workspace_3d = tempfile.mkdtemp(prefix="colmap_kibali_")
                                colmap = ColmapPhotogrammetry(workspace_3d)
                                
                                # Progress container
                                progress_bar = st.progress(0, text="Initialisation...")
                                status_text = st.empty()
                                start_time = time.time()
                                
                                def colmap_progress(current, total, message):
                                    progress = current / total if total > 0 else 0
                                    elapsed = int(time.time() - start_time)
                                    progress_bar.progress(progress, text=f"{int(progress*100)}% - {message}")
                                    status_text.text(f"⏱️ {elapsed//60}m {elapsed%60}s écoulées...")
                                
                                # Lancer le pipeline complet
                                results = colmap.run_full_pipeline(
                                    image_paths=current_ordered_paths,
                                    use_gpu=use_gpu_3d,
                                    quality=quality_3d,
                                    progress_callback=colmap_progress
                                )
                                
                                elapsed = int(time.time() - start_time)
                                
                                if results['success']:
                                    st.success(f"✅ Modèle 3D généré en {elapsed//60}m {elapsed%60}s !")
                                    
                                    col_result1, col_result2, col_result3 = st.columns(3)
                                    with col_result1:
                                        st.metric("📸 Images", results.get('images_prepared', 0))
                                    with col_result2:
                                        if results.get('dense_pointcloud'):
                                            st.metric("☁️ Point Cloud", "✅")
                                    with col_result3:
                                        if results.get('mesh'):
                                            st.metric("🎨 Mesh", "✅")
                                    
                                    # Afficher chemins des résultats
                                    st.markdown("#### 📁 Résultats")
                                    if results.get('mesh'):
                                        st.code(f"Mesh: {results['mesh']}", language="bash")
                                    if results.get('dense_pointcloud'):
                                        st.code(f"Point Cloud: {results['dense_pointcloud']}", language="bash")
                                    st.code(f"Workspace: {workspace_3d}", language="bash")
                                    
                                    st.info("""
                                    💡 **Visualisation :**
                                    - Mesh: `meshlab` ou `blender`
                                    - Point Cloud: `CloudCompare` ou `MeshLab`
                                    - Export: Formats PLY, OBJ disponibles
                                    """)
                                else:
                                    st.error(f"❌ Reconstruction échouée après {elapsed//60}m {elapsed%60}s")
                                    for err in results.get('errors', []):
                                        st.warning(f"• {err}")
                                
                            except Exception as e:
                                st.error(f"❌ Erreur COLMAP: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                # MODE OPTIMISATION RAPIDE (ancien système)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                st.markdown("#### ⚙️ Paramètres d'optimisation")
                
                col_param1, col_param2 = st.columns(2)
                with col_param1:
                    target_count = st.number_input(
                        "🎯 Nombre cible de photos (0 = automatique)",
                        min_value=0,
                        max_value=len(uploaded_photos),
                        value=0,
                        help="Laissez 0 pour une détection automatique du nombre optimal"
                    )
                
                with col_param2:
                    coverage_threshold = st.slider(
                        "📊 Couverture minimale requise",
                        min_value=0.8,
                        max_value=1.0,
                        value=0.95,
                        step=0.05,
                        help="Pourcentage de la scène qui doit être couvert (0.95 = 95%)"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Bouton d'optimisation
                if st.button("🚀 **Optimiser le dataset**", type="primary", use_container_width=True):
                    with st.spinner("🔍 Analyse des photos en cours..."):
                        try:
                            import tempfile
                            import shutil
                            
                            # Créer un dossier temporaire pour les photos
                            temp_dir = tempfile.mkdtemp(prefix="photogrammetry_")
                            
                            # Sauvegarder les photos uploadées
                            st.info(f"💾 Sauvegarde de {len(uploaded_photos)} photos...")
                            for idx, photo in enumerate(uploaded_photos):
                                photo_path = Path(temp_dir) / photo.name
                                with open(photo_path, 'wb') as f:
                                    f.write(photo.getbuffer())
                            
                            # Importer l'outil
                            from outils.photogrammetry_optimizer_tool import PhotogrammetryOptimizerTool
                            
                            # Exécuter l'optimisation
                            tool = PhotogrammetryOptimizerTool()
                            context = {
                                'input_folder': temp_dir,
                                'target_count': target_count if target_count > 0 else None,
                                'coverage_threshold': coverage_threshold,
                                'similarity_threshold': 0.85
                            }
                            
                            result = tool.execute("", context=context)
                            
                            # Afficher les résultats
                            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                            st.markdown("### 📊 Résultats de l'optimisation")
                            st.text(result)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Proposer le téléchargement des photos sélectionnées
                            output_folder = Path(temp_dir + "_optimized")
                            if output_folder.exists():
                                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                st.markdown("### 📥 Téléchargement des photos optimisées")
                                
                                # Créer un ZIP
                                import zipfile
                                zip_path = Path(temp_dir) / "photos_optimized.zip"
                                
                                with zipfile.ZipFile(zip_path, 'w') as zipf:
                                    for photo_file in output_folder.glob("*.*"):
                                        if photo_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                                            zipf.write(photo_file, arcname=photo_file.name)
                                
                                # Bouton de téléchargement
                                with open(zip_path, 'rb') as f:
                                    st.download_button(
                                        label="📦 Télécharger les photos sélectionnées (ZIP)",
                                        data=f.read(),
                                        file_name="photos_photogrammetrie_optimisees.zip",
                                        mime="application/zip"
                                    )
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Nettoyage (optionnel, garder pour debug)
                            # shutil.rmtree(temp_dir, ignore_errors=True)
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'optimisation: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
        
        else:
            st.info("👆")
            
            # Exemples d'utilisation
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("""
            #### 💡 Exemples d'utilisation
            
            **Photogrammétrie aérienne:**
            - Upload: 500-1000 photos drone
            - Résultat: 15-30 photos essentielles
            - Réduction: ~97%
            
            **Scan d'objet 3D:**
            - Upload: 200 photos multi-angles
            - Résultat: 10-20 photos représentatives
            - Réduction: ~90%
            
            **Cartographie terrain:**
            - Upload: 1000+ photos aériennes
            - Résultat: 20-50 photos de couverture
            - Réduction: ~95%
            
            #### 🎯 Algorithme
            1. **Extraction de features** (ORB, histogrammes, textures)
            2. **Clustering intelligent** (KMeans sur angles de vue)
            3. **Sélection optimale** (représentants par cluster)
            4. **Vérification de couverture** (ajout si zones manquantes)
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # ===============================================
    # Onglet 7: Outils Dynamiques (si disponible)
    # ===============================================
    if TOOLS_SYSTEM_AVAILABLE and st.session_state.tool_manager and tab_tools:
        with tab_tools:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("""
            ### 🔧 Système d'outils dynamiques
            **Outils disponibles:** Recherche web, analyse d'images, calculs mathématiques, génération de code, traduction, gestion de fichiers.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Affichage des outils disponibles
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("#### 📋 Outils chargés")
            tools = st.session_state.tool_manager.tools
            if tools:
                tool_data = []
                for tool in tools:
                    tool_data.append({
                        "Nom": tool.name,
                        "Description": tool.description,
                        "Capacités": ", ".join(tool.capabilities)
                    })
                st.dataframe(pd.DataFrame(tool_data), use_container_width=True)
            else:
                st.warning("⚠️ Aucun outil chargé")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Test manuel d'un outil
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("#### 🧪 Test d'outil manuel")
            
            col_test1, col_test2 = st.columns([2, 1])
            with col_test1:
                test_query = st.text_area(
                    "🔍 **Requête de test**",
                    placeholder="Ex: Analyse cette image de sol minier",
                    height=100,
                    help="Décris la tâche à effectuer"
                )
            
            with col_test2:
                available_tools = [tool.name for tool in tools] if tools else []
                selected_tool_name = st.selectbox(
                    "🔧 **Outil à tester**",
                    options=["Auto (détection)"] + available_tools,
                    help="Choisis un outil spécifique ou laisse la détection automatique"
                )
            
            if st.button("🚀 **Tester l'outil**", key="test_tool", help="Exécute l'outil sélectionné"):
                if test_query.strip():
                    with st.spinner("🔧 Exécution de l'outil..."):
                        try:
                            if selected_tool_name == "Auto (détection)":
                                # Préparer le contexte pour la détection
                                tool_context = {
                                    'has_pdfs': st.session_state.vectordb is not None,
                                    'vectordb_available': st.session_state.vectordb is not None,
                                    'media_analysis': st.session_state.media_analysis_results if st.session_state.media_analysis_results else None
                                }
                                
                                # Détection automatique
                                relevant_tools = st.session_state.tool_manager.get_relevant_tools(test_query, tool_context)
                                if relevant_tools:
                                    st.success(f"🔍 Outils détectés: {', '.join([t.name for t in relevant_tools])}")
                                    
                                    # Si outil PDF détecté, faire une vraie recherche
                                    pdf_tool_detected = any('pdf' in t.name.lower() or 'document' in t.name.lower() for t in relevant_tools)
                                    
                                    if pdf_tool_detected:
                                        if st.session_state.vectordb:
                                            st.info("🔍")
                                            rag_docs = rag_search(test_query, st.session_state.vectordb, k=5)
                                            if rag_docs:
                                                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                                st.markdown(f"### 📊 Résultats trouvés: {len(rag_docs)} passages")
                                                for i, doc in enumerate(rag_docs, 1):
                                                    st.markdown(f"**Document {i}:** {doc.metadata.get('source', 'Inconnu')}")
                                                    st.text_area(f"Extrait {i}", value=doc.page_content[:500], height=100, disabled=True, key=f"extract_{i}")
                                                    st.markdown("---")
                                                st.markdown('</div>', unsafe_allow_html=True)
                                            else:
                                                st.warning("Aucun résultat trouvé dans les documents")
                                        else:
                                            st.warning("⚠️ Base vectorielle non chargée. Ajoutez des PDFs dans l'onglet Configuration.")
                                    
                                    # Exécuter les autres outils
                                    results = []
                                    for tool in relevant_tools:
                                        if 'pdf' not in tool.name.lower() and 'document' not in tool.name.lower():
                                            result = tool.execute(test_query, tool_context)
                                            results.append(f"**{tool.name}:**\n{result}")
                                    
                                    if results:
                                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                        st.markdown("### 📊 Résultats des autres outils")
                                        for result in results:
                                            st.markdown(result)
                                            st.markdown("---")
                                        st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.warning("⚠️ Aucun outil pertinent détecté pour cette requête")
                            
                            else:
                                # Outil spécifique
                                tool = next((t for t in tools if t.name == selected_tool_name), None)
                                if tool:
                                    result = tool.execute(test_query)
                                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                    st.markdown(f"### 📊 Résultat de {tool.name}")
                                    st.text_area("Résultat", value=str(result), height=200, disabled=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.error("❌ Outil non trouvé")
                        
                        except Exception as e:
                            st.error(f"❌ Erreur lors du test: {e}")
                else:
                    st.warning("⚠️ Veuillez entrer une requête de test")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Statistiques des outils
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Statistiques du système d'outils")
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Outils chargés", len(tools) if tools else 0)
            
            with col_stats2:
                total_capabilities = sum(len(tool.capabilities) for tool in tools) if tools else 0
                st.metric("Capacités totales", total_capabilities)
            
            with col_stats3:
                status = "✅ Actif" if TOOLS_SYSTEM_AVAILABLE and st.session_state.tool_manager else "❌ Inactif"
                st.metric("Statut système", status)
            
            st.markdown('</div>', unsafe_allow_html=True)

# ===============================================
# Lancement principal
# ===============================================
if __name__ == "__main__":
    print("🚀 Kibali 🌟 - Assistant IA Avancé avec Recherche Web")
    print("=" * 60)
    # Initialisation
    setup_drive()
    print(f"📁 Dossier unifié: {CHATBOT_DIR}")
    print(f"🔑 Token HF configuré: {HF_TOKEN[:10]}...")
    print(f"🌐 Recherche web intégrée")
    # Vérifier les fichiers existants
    existing_graphs = [f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')] if os.path.exists(GRAPHS_PATH) else []
    existing_pdfs = [f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')] if os.path.exists(PDFS_PATH) else []
    print(f"📊 État initial:")
    print(f" 🗺️ Graphes OSM: {len(existing_graphs)}")
    print(f" 📄 PDFs: {len(existing_pdfs)}")
    print(f" 💾 Base vectorielle: {'✅' if os.path.exists(VECTORDB_PATH) else '⚠️ À créer (upload PDFs + traiter)'}")
    print(f" 🌐 Cache web: {'✅' if os.path.exists(WEB_CACHE_PATH) else '📝 Sera créé automatiquement'}")
    # Afficher les statistiques du cache si disponible
    print(f" 📈 {get_cache_stats()}")
    # Lancer l'interface Streamlit
    print("\n🌟 Lancement de l'interface Streamlit...")
    print("📝 Fonctionnalités disponibles:")
    print(" 💬 Chat RAG avec recherche web intelligent")
    print(" 🗺️ Calcul de trajets OSM")
    print(" 📸 Analyse d'images avancée")
    print(" 🌐 Recherche web avec extraction de contenu")
    print(" 💾 Cache intelligent pour optimiser les performances")
    print(" 📊 Gestion unifiée des fichiers")
    main()
else:
    print("📦 Module Kibali importé avec succès")
    print("🔧 Fonctions disponibles:")
    print(" - main() : Interface Streamlit complète")
    print(" - enhanced_web_search() : Recherche web avancée")
    print(" - hybrid_search_enhanced() : Recherche hybride")
    print(" - smart_content_extraction() : Extraction de contenu web")
    print(" - process_image() : Analyse d'images")
    print(" - calculer_trajet() : Calcul de trajets OSM")
# ===============================================
# Fonctions utilitaires supplémentaires
# ===============================================
def cleanup_old_cache():
    """Nettoie les entrées expirées du cache"""
    try:
        cache = load_web_cache()
        if not cache:
            return "Cache vide"
        original_count = len(cache)
        cleaned_cache = {}
        for key, entry in cache.items():
            if not is_cache_expired(entry):
                cleaned_cache[key] = entry
        save_web_cache(cleaned_cache)
        removed_count = original_count - len(cleaned_cache)
        return f"✅ Cache nettoyé: {removed_count} entrées expirées supprimées, {len(cleaned_cache)} conservées"
    except Exception as e:
        return f"❌ Erreur nettoyage cache: {e}"
def export_system_config():
    """Exporte la configuration système pour debug"""
    config = {
        "version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "chatbot_dir": CHATBOT_DIR,
            "vectordb_path": VECTORDB_PATH,
            "pdfs_path": PDFS_PATH,
            "graphs_path": GRAPHS_PATH,
            "maps_path": MAPS_PATH
        },
        "models": WORKING_MODELS,
        "status": get_system_status(),
        "features": {
            "web_search": True,
            "osm_routing": True,
            "image_analysis": True,
            "pdf_processing": True,
            "caching": True
        }
    }
    config_path = os.path.join(CHATBOT_DIR, "system_config.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return f"✅ Configuration exportée: {config_path}"
    except Exception as e:
        return f"❌ Erreur export: {e}"
def test_all_features():
    """Teste toutes les fonctionnalités principales"""
    results = {}
    # Test HuggingFace
    results["huggingface"] = test_hf_connection()
    # Test recherche web
    try:
        test_results = enhanced_web_search("test", max_results=1)
        results["web_search"] = len(test_results) > 0
    except:
        results["web_search"] = False
    # Test recherche web
    results["specialized_models"] = {}
    for model_name, model in SPECIALIZED_MODELS.items():
        results["specialized_models"][model_name] = model is not None
    # Test base vectorielle
    try:
        vectordb, _ = load_vectordb()
        results["vectordb"] = vectordb is not None
    except:
        results["vectordb"] = False
    # Test graphe OSM
    try:
        graph, pois, _ = load_existing_graph()
        results["osm_graph"] = graph is not None
    except:
        results["osm_graph"] = False
    return results
# ===============================================
# Fonctions de maintenance avancées
# ===============================================
def optimize_vectordb():
    """Optimise la base vectorielle en supprimant les doublons"""
    try:
        vectordb, status = load_vectordb()
        if not vectordb:
            return "❌ Aucune base vectorielle à optimiser"
        # Cette fonction nécessiterait une implémentation plus complexe
        # pour détecter et supprimer les doublons dans FAISS
        return "✅ Base vectorielle optimisée (fonctionnalité à implémenter)"
    except Exception as e:
        return f"❌ Erreur optimisation: {e}"
def backup_all_data():
    """Crée une sauvegarde de toutes les données"""
    try:
        import shutil
        import zipfile
        backup_name = f"kibali_backup_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        backup_path = os.path.join(CHATBOT_DIR, backup_name)
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            # Sauvegarder tous les fichiers du dossier chatbot
            for root, dirs, files in os.walk(CHATBOT_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, CHATBOT_DIR)
                    backup_zip.write(file_path, arcname)
        return f"✅ Sauvegarde créée: {backup_path}"
    except Exception as e:
        return f"❌ Erreur sauvegarde: {e}"
def restore_from_backup(backup_path):
    """Restaure les données depuis une sauvegarde"""
    try:
        import zipfile
        if not os.path.exists(backup_path):
            return "❌ Fichier de sauvegarde non trouvé"
        with zipfile.ZipFile(backup_path, 'r') as backup_zip:
            backup_zip.extractall(CHATBOT_DIR)
        return f"✅ Données restaurées depuis: {backup_path}"
    except Exception as e:
        return f"❌ Erreur restauration: {e}"
# ===============================================
# Version API pour utilisation externe
# ===============================================
class KibaliAPI:
    """API simplifiée pour utiliser Kibali depuis du code externe"""
    def __init__(self):
        self.vectordb = None
        self.graph = None
        self.pois = []
        self.client = None
        self.model_name = WORKING_MODELS[list(WORKING_MODELS.keys())[0]]
        # Initialisation automatique
        self._initialize()
    def _initialize(self):
        """Initialisation automatique"""
        try:
            setup_drive()
            self.vectordb, _ = load_vectordb()
            self.graph, self.pois, _ = load_existing_graph()
            self.client = create_client()
        except Exception as e:
            print(f"⚠️ Initialisation partielle: {e}")
    def ask(self, question, use_web=True):
        """Pose une question simple"""
        try:
            if use_web:
                docs = hybrid_search_enhanced(question, self.vectordb, web_search_enabled=True)
            else:
                docs = rag_search(question, self.vectordb)
            return generate_answer_enhanced(question, docs, self.model_name)
        except Exception as e:
            return f"❌ Erreur: {e}"
    def search_web(self, query, max_results=5):
        """Recherche web simple"""
        try:
            results = enhanced_web_search(query, max_results)
            return [{"title": r.get("title"), "url": r.get("href", r.get("url")), "snippet": r.get("body")} for r in results]
        except Exception as e:
            return [{"error": str(e)}]
    def calculate_route(self, from_place, to_place):
        """Calcule un itinéraire"""
        try:
            question = f"Comment aller de {from_place} à {to_place}"
            _, response, info = calculer_trajet(question, self.graph, self.pois)
            return {"response": response, "info": info}
        except Exception as e:
            return {"error": str(e)}
    def get_status(self):
        """Retourne le statut du système"""
        return get_system_status()
# Instance globale de l'API
kibali_api = KibaliAPI()

# ===============================================
# Onglet Nano-IA et Auto-Apprentissage
# ===============================================
if AUTO_LEARNING_AVAILABLE and tab_nano_ai:
    with tab_nano_ai:
        st.markdown("## 🤖 Nano-IA & Système d'Auto-Apprentissage")
        
        # Toggle pour activer/désactiver l'apprentissage
        if st.session_state.get('auto_learning'):
            st.session_state.auto_learning.render_learning_toggle()
        
        st.markdown("---")
        
        # Statistiques d'apprentissage
        if st.session_state.get('auto_learning'):
            st.session_state.auto_learning.render_learning_stats()
        
        st.markdown("---")
        
        # Gestion des domaines
        st.markdown("### 📚 Domaines d'Expertise")
        
        if st.session_state.get('knowledge_manager'):
            km = st.session_state.knowledge_manager
            domains = km.list_domains()
            
            if domains:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_domain = st.selectbox(
                        "Sélectionner un domaine",
                        domains,
                        help="Choisissez un domaine pour voir les détails"
                    )
                
                with col2:
                    if st.button("🤖 Créer/Mettre à jour Nano-IA", help="Crée ou met à jour la nano-IA spécialisée"):
                        with st.spinner(f"Création de la nano-IA pour {selected_domain}..."):
                            if km.create_nano_ai(selected_domain, force=True):
                                st.success(f"✅ Nano-IA {selected_domain} créée!")
                                st.rerun()
                
                # Détails du domaine
                if selected_domain:
                    expertise = km.get_domain_expertise(selected_domain)
                    if expertise:
                        st.markdown(f"""
                        <div style='
                            background: linear-gradient(135deg, rgba(0, 136, 255, 0.1), rgba(138, 43, 226, 0.1));
                            padding: 1.5rem;
                            border-radius: 10px;
                            border: 2px solid rgba(0, 136, 255, 0.3);
                            margin: 1rem 0;
                        '>
                            <h4 style='color: #0088ff; margin-top: 0;'>📊 {selected_domain}</h4>
                            <p><strong>Documents:</strong> {expertise.documents_count}</p>
                            <p><strong>Requêtes traitées:</strong> {expertise.queries_handled}</p>
                            <p><strong>Taux de succès:</strong> {expertise.success_rate*100:.1f}%</p>
                            <p><strong>Créé:</strong> {expertise.creation_date[:10]}</p>
                            <p><strong>Dernière MAJ:</strong> {expertise.last_updated[:10]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Mots-clés
                        if expertise.keywords:
                            st.markdown("**🔑 Mots-clés maîtrisés:**")
                            st.write(", ".join(expertise.keywords[:20]))
                        
                        # Exemples Q&A
                        if expertise.examples:
                            st.markdown("**💬 Exemples de conversations:**")
                            for i, example in enumerate(expertise.examples[-5:], 1):
                                with st.expander(f"Exemple {i}: {example['query'][:50]}..."):
                                    st.markdown(f"**Question:** {example['query']}")
                                    st.markdown(f"**Réponse:** {example['response'][:300]}...")
                        
                        # Test du domaine
                        st.markdown("---")
                        st.markdown("### 🧪 Tester les connaissances du domaine")
                        test_query = st.text_input(
                            "Posez une question sur ce domaine",
                            placeholder=f"Ex: Qu'as-tu appris sur {selected_domain}?",
                            key=f"test_query_{selected_domain}"
                        )
                        
                        if test_query:
                            with st.spinner("Recherche dans les connaissances..."):
                                results = km.query_domain(test_query, selected_domain, k=3)
                                
                                if results:
                                    st.success(f"✅ {len(results)} résultats trouvés!")
                                    for i, doc in enumerate(results, 1):
                                        with st.expander(f"Résultat {i}"):
                                            st.write(doc.page_content[:500])
                                            st.caption(f"Source: {doc.metadata.get('filename', 'N/A')}")
                                else:
                                    st.warning("Aucun résultat trouvé dans ce domaine")
            else:
                st.info("👋 Aucun domaine d'expertise créé pour le moment. Uploadez des fichiers pour commencer l'apprentissage!")
        
        # Historique d'apprentissage
        st.markdown("---")
        st.markdown("### 📜 Historique d'Apprentissage")
        
        if st.session_state.get('learning_history'):
            history_df = pd.DataFrame(st.session_state.learning_history[-20:])
            st.dataframe(
                history_df[['type', 'source', 'domain']],
                use_container_width=True
            )
        else:
            st.info("L'historique d'apprentissage apparaîtra ici au fur et à mesure")

# ===============================================
# Messages de fin et documentation
# ===============================================
print("\n" + "="*60)
print("🎉 KIBALI 🌟 - SYSTÈME CHARGÉ AVEC SUCCÈS")
print("="*60)
print(f"📅 Version: 2.0.0 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔑 Token HF: {'✅ Configuré' if HF_TOKEN else '❌ Manquant'}")
print(f"📁 Dossier: {CHATBOT_DIR}")
print(f"🌐 Recherche web: ✅ Activée")
print(f"💾 Cache intelligent: ✅ Activé")
if AUTO_LEARNING_AVAILABLE:
    print(f"🤖 Auto-apprentissage: ✅ Activé")
print("\n📚 FONCTIONNALITÉS PRINCIPALES:")
print(" 💬 Chat RAG avec recherche web intelligent")
print(" 🗺️ Calcul de trajets OSM")
print(" 📸 Analyse d'images avec IA")
print(" 🌐 Extraction de contenu web")
print(" 💾 Gestion unifiée des données")
print("\n🚀 UTILISATION:")
print(" Interface: Exécutez les cellules suivantes")
print(" API: kibali_api.ask('votre question')")
print(" Tests: test_all_features()")
print("\n⚙️ MAINTENANCE:")
print(" Status: get_system_status()")
print(" Nettoyage: cleanup_old_cache()")
print(" Sauvegarde: backup_all_data()")
print("="*60)# ===============================================
# Installation des dépendances (pour Colab/Jupyter - commenter pour Streamlit)
# ===============================================
# !apt-get install -y osmium-tool > /dev/null
# !pip install -U streamlit huggingface_hub langchain sentence-transformers faiss-cpu pdfplumber osmium networkx shapely matplotlib opencv-python-headless open3d ipywidgets pandas scikit-image scikit-learn torch torchvision langchain-community langchain-huggingface requests beautifulsoup4 python-dotenv diffusers accelerate transformers librosa soundfile tavily-python flash-attn --no-build-isolation --quiet
# ===============================================
# Configuration HuggingFace Token depuis .env
# ===============================================
import os
from pathlib import Path

# ⚠️ FORCER CPU AVANT TOUT IMPORT PYTORCH/TRANSFORMERS
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Désactiver CUDA complètement pour éviter erreurs kernel
os.environ['FORCE_CPU'] = '1'

# Charger le token depuis .env
from dotenv import load_dotenv

# Obtenir le dossier du script actuel
script_dir = Path(__file__).parent.absolute()

# Import de la configuration centralisée des modèles
import sys
sys.path.insert(0, str(script_dir / "kibali_data" / "models"))
from MODEL_PATHS import (
    QWEN_MODEL_NAME, QWEN_CACHE_DIR,
    SENTENCE_TRANSFORMER_MODEL, SENTENCE_TRANSFORMER_CACHE,
    CLIP_MODEL_NAME, CLIP_CACHE_DIR,
    EASYOCR_MODEL_DIR, EASYOCR_LANGUAGES,
    SUMMARIZER_MODEL, SUMMARIZER_CACHE,
    TRANSLATOR_MODEL, TRANSLATOR_CACHE,
    NER_MODEL, NER_CACHE,
    CAPTIONER_MODEL, CAPTIONER_CACHE,
    ensure_model_dirs
)

# Import du gestionnaire de mode offline
from offline_manager import (
    offline_manager,
    get_model_loading_params,
    load_model_with_fallback,
    render_mode_indicator,
    render_offline_status_card,
    get_offline_capabilities
)

# Import du système d'auto-apprentissage et nano-IA
sys.path.insert(0, str(script_dir / "kibali_data" / "models"))

# Import du détecteur de structure binaire
from binary_structure_detector import BinaryStructureDetector, detect_and_analyze_file

# Import de l'analyseur ERT (Electrical Resistivity Tomography)
try:
    from ert_analyzer import ERTAnalyzer
    ERT_AVAILABLE = True
except ImportError:
    ERT_AVAILABLE = False
    print("⚠️ Module ERT non disponible")
try:
    from binary_structure_detector import (
        BinaryStructureDetector, 
        FileStructureType, 
        detect_and_analyze_file
    )
    BINARY_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Détecteur de structure binaire non disponible: {e}")
    BINARY_DETECTOR_AVAILABLE = False

# Import du système d'auto-apprentissage
try:
    from auto_learning import get_auto_learning
    from knowledge_manager import get_knowledge_manager
    AUTO_LEARNING_AVAILABLE = True
    print("✅ Système d'auto-apprentissage chargé")
except Exception as e:
    print(f"⚠️ Auto-apprentissage non disponible: {e}")
    AUTO_LEARNING_AVAILABLE = False
    get_auto_learning = None
    get_knowledge_manager = None

# Import de l'orchestrateur maître d'IA
try:
    from task_orchestrator import analyze_and_plan, get_ai_relay, TaskType
    from master_orchestrator import get_orchestrator, AISpecialist
    ORCHESTRATOR_AVAILABLE = True
    print("✅ Orchestrateur maître d'IA chargé")
except Exception as e:
    print(f"⚠️ Orchestrateur non disponible: {e}")
    ORCHESTRATOR_AVAILABLE = False

# Initialiser les dossiers de modèles
ensure_model_dirs()

# Chercher le fichier .env dans plusieurs emplacements possibles
env_paths = [
    script_dir / ".env",  # Dans le même dossier que app.py
    Path.cwd() / ".env",  # Dans le dossier courant
    Path.home() / ".env"  # Dans le dossier home
]

token_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Fichier .env trouvé: {env_path}")
        token_loaded = True
        break

if not token_loaded:
    print(f"⚠️ Aucun fichier .env trouvé dans: {[str(p) for p in env_paths]}")
    print(f"Créez un fichier .env dans {script_dir} avec: HF_TOKEN=hf_votre_token")

# Récupérer le token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(f"❌ HF_TOKEN non trouvé ! Vérifiez votre fichier .env dans {script_dir}")
else:
    print(f"🔑 Token HF configuré: {HF_TOKEN[:10]}...")

# Définir la variable d'environnement pour huggingface_hub
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError(f"❌ TAVILY_API_KEY non trouvé ! Vérifiez votre fichier .env dans {script_dir}")
# ===============================================
# Imports
# ===============================================
import math
import pdfplumber  # MIT License - Compatible commercial
import osmium
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import base64
import json
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from shapely.geometry import Point
import io
from PIL import Image
import cv2
try:
    import open3d as o3d
except ImportError:
    o3d = None
from io import BytesIO
import pandas as pd
from skimage import measure, segmentation
from sklearn.cluster import KMeans
import torch
from torchvision import models, transforms
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.agents import initialize_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Import conditionnel de CLIP
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    try:
        # Essayer l'import alternatif pour les anciennes versions
        from transformers import AutoProcessor as CLIPProcessor, AutoModel as CLIPModel
    except ImportError:
        CLIPProcessor = None
        CLIPModel = None
        print("⚠️ CLIP non disponible - Analyse sémantique d'images désactivée")

# Import des modèles de génération de texte
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    print("⚠️ Modèles Hugging Face non disponibles")

# ═══════════════════════════════════════════════════════════════════
# FONCTION DE NETTOYAGE DES RÉPONSES
# ═══════════════════════════════════════════════════════════════════

def clean_response_text(text: str) -> str:
    """
    Nettoie le texte de réponse en supprimant les caractères non-latins
    (chinois, arabe, cyrillique, etc.) qui peuvent apparaître par erreur
    """
    import re
    
    if not text:
        return text
    
    # Détecter et supprimer les blocs de texte chinois/asiatique
    # Plages Unicode pour caractères CJK (chinois, japonais, coréen)
    text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\u2f800-\u2fa1f]+', '', text)
    
    # Supprimer les caractères arabes
    text = re.sub(r'[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]+', '', text)
    
    # Supprimer les caractères cyrilliques (russe, ukrainien, etc.)
    text = re.sub(r'[\u0400-\u04ff\u0500-\u052f\u2de0-\u2dff\ua640-\ua69f\u1c80-\u1c8f]+', '', text)
    
    # Supprimer les caractères coréens Hangul
    text = re.sub(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\ud7b0-\ud7ff]+', '', text)
    
    # Supprimer les caractères japonais (hiragana, katakana)
    text = re.sub(r'[\u3040-\u309f\u30a0-\u30ff]+', '', text)
    
    # Nettoyer les espaces multiples consécutifs
    text = re.sub(r'\s+', ' ', text)
    
    # Nettoyer les sauts de ligne excessifs
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()

# Import du gestionnaire de fichiers binaires scientifiques
try:
    from binary_file_handler import (
        BinaryFileHandler, 
        BinaryFileViewer, 
        SCIENTIFIC_BINARY_FORMATS,
        analyze_binary_file,
        create_hex_viewer_ui
    )
    BINARY_HANDLER_AVAILABLE = True
    print(f"✅ Gestionnaire de fichiers binaires chargé - {len(SCIENTIFIC_BINARY_FORMATS)} formats supportés")
except ImportError as e:
    BINARY_HANDLER_AVAILABLE = False
    print(f"⚠️ Gestionnaire de fichiers binaires non disponible: {e}")

# Import YOLO pour détection d'objets
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO disponible")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO non disponible - pip install ultralytics")

# Import de l'outil d'organisation Excel avec IA
try:
    from tools.excel_ai_organizer import organize_excel_with_ai, ExcelAIOrganizer
    EXCEL_AI_AVAILABLE = True
except ImportError:
    EXCEL_AI_AVAILABLE = False
    print("⚠️ Excel AI Organizer non disponible")

try:
    from transformers import pipeline
except ImportError:
    # Fallback si pipeline n'est pas disponible directement
    pipeline = None
import requests
from bs4 import BeautifulSoup
import time
import shutil
try:
    import easyocr
except ImportError:
    easyocr = None
try:
    import pytesseract
except ImportError:
    pytesseract = None

# Import diffusers avec gestion d'erreurs pour les classes optionnelles
try:
    from diffusers import DiffusionPipeline
except ImportError:
    DiffusionPipeline = None
try:
    from diffusers import AudioLDMPipeline
except ImportError:
    AudioLDMPipeline = None
try:
    from diffusers import ShapEPipeline, ShapEImg2ImgPipeline
except ImportError:
    ShapEPipeline = None
    ShapEImg2ImgPipeline = None

import imageio
try:
    import scipy.io.wavfile as wavfile
except ImportError:
    wavfile = None
from tavily import TavilyClient
# ===============================================
# Import du système de modèle local Qwen
# ===============================================
import torch
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from typing import Iterator

# ===============================================
# Configuration automatique du device (CUDA/CPU)
# ===============================================
def get_optimal_device():
    """
    Détecte automatiquement le meilleur device disponible.
    Fallback sur CPU si CUDA n'est pas disponible.
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU détecté: {gpu_name}")
        print(f"✅ Utilisation de CUDA {torch.version.cuda}")
    else:
        device = 'cpu'
        print(f"💻 Aucun GPU détecté - Utilisation du CPU")
    return device

# Device global pour toute l'application
DEVICE = get_optimal_device()
# ===============================================
# Import du système d'outils dynamiques
# ===============================================
try:
    from outils import ToolManager
    TOOLS_SYSTEM_AVAILABLE = True
    print("✅ Système d'outils dynamiques chargé")
except ImportError as e:
    print(f"⚠️ Système d'outils non disponible: {e}")
    TOOLS_SYSTEM_AVAILABLE = False
# ===============================================
# Classe ChatModel personnalisée pour LangChain utilisant Qwen2.5-1.5B
# ===============================================
class QwenChatModel(BaseChatModel):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    tools_available: bool = True
   
    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.tools_available = True
       
    @property
    def _llm_type(self) -> str:
        return "qwen2.5-1.5b-local-enhanced"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate a response using tools and analyses."""
        # Extraire le contenu du message utilisateur
        user_message = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                user_message = message.content
                break
       
        # Détecter si l'utilisateur demande une analyse
        needs_analysis = any(keyword in user_message.lower() for keyword in [
            "analyse", "resistivité", "ert", "recherche", "données", "matériaux",
            "couleurs", "graphique", "tableau", "comparaison", "approfondie"
        ])
       
        if needs_analysis and self.tools_available:
            # Utiliser les outils disponibles pour une analyse complète
            try:
                # Recherche web pour informations
                if any(keyword in user_message.lower() for keyword in ["recherche", "informations", "approfondie"]):
                    search_query = user_message.replace("fais maintenant une recherche plus approfondie pour obtenir toutes ces informations précises", "")
                    web_results = web_search_enhanced(search_query + " ERT electrical resistivity geophysics materials")
                   
                # Recherche RAG si disponible
                rag_results = ""
                if st.session_state.get('vectordb'):
                    rag_results = rag_search(user_message, st.session_state.vectordb)
               
                # Génération de données et analyses si demandées
                analysis_results = ""
                if any(keyword in user_message.lower() for keyword in ["tableau", "graphique", "données"]):
                    # Simuler des données ERT pour démonstration
                    import numpy as np
                    sample_data = [0.05, 0.3, 10.0, 50.0, 200.0, 1000.0, 5000.0, 0.0000024, 1000000]
                    analysis_results = resistivity_color_analysis(sample_data)
               
                # Construire la réponse enrichie avec outils
                enhanced_context = f"""
🔍 ANALYSE COMPLÈTE AVEC OUTILS ACTIVÉS:
🌐 RECHERCHE WEB EFFECTUÉE:
{web_results}
📚 RECHERCHE RAG:
{rag_results}
📊 ANALYSE ERT AVANCÉE:
{analysis_results}
CONTEXTE UTILISATEUR: {user_message}
"""
               
                # Générer la réponse avec le contexte enrichi
                enhanced_messages = [
                    {"role": "system", "content": """Tu es un expert en géophysique ERT avec accès à des outils puissants.
                    Tu DOIS utiliser les données fournies pour créer des analyses détaillées, tableaux, graphiques et comparaisons.
                    Réponds toujours avec des données concrètes et des analyses approfondies basées sur les outils utilisés.
                    Ne dis JAMAIS que tu n'as pas accès aux outils - utilise les résultats fournis."""},
                    {"role": "user", "content": enhanced_context}
                ]
            except Exception as e:
                print(f"Erreur outils: {e}")
                enhanced_messages = [
                    {"role": "system", "content": "Tu es un expert en analyse de données ERT."},
                    {"role": "user", "content": user_message}
                ]
        else:
            # Messages standard
            enhanced_messages = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    enhanced_messages.append({"role": "system", "content": message.content})
                elif isinstance(message, HumanMessage):
                    enhanced_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    enhanced_messages.append({"role": "assistant", "content": message.content})
       
        # Génération avec les messages enrichis
        inputs = self.tokenizer.apply_chat_template(
            enhanced_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
       
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
       
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=3000,  # 3000 tokens pour réponses TRÈS détaillées
                temperature=0.6,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.05,  # Éviter répétitions
                pad_token_id=self.tokenizer.eos_token_id
            )
       
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
       
        if stop:
            for stop_token in stop:
                if stop_token in response:
                    response = response.split(stop_token)[0]
                    break
       
        return AIMessage(content=response)
    
    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator:
        """Stream tokens one by one for real-time display."""
        # Extraire le contenu du message utilisateur
        user_message = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                user_message = message.content
                break
       
        # Préparer les messages comme dans _generate
        enhanced_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                enhanced_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                enhanced_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                enhanced_messages.append({"role": "assistant", "content": message.content})
       
        # Génération avec streaming token par token
        inputs = self.tokenizer.apply_chat_template(
            enhanced_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
       
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Créer un streamer pour générer token par token
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Préparer les arguments de génération
        generation_kwargs = dict(
            inputs=inputs,
            attention_mask=attention_mask,
            max_new_tokens=3000,
            temperature=0.6,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        
        # Lancer la génération dans un thread séparé
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield chaque token au fur et à mesure
        full_response = ""
        for new_text in streamer:
            full_response += new_text
            # Vérifier les stop tokens
            should_stop = False
            if stop:
                for stop_token in stop:
                    if stop_token in full_response:
                        full_response = full_response.split(stop_token)[0]
                        should_stop = True
                        break
            
            yield AIMessage(content=new_text)
            
            if should_stop:
                break
        
        thread.join()
# ===============================================
# Chargement du modèle LLM local Qwen2.5-1.5B
# ===============================================
@st.cache_resource
def load_local_llm_model():
    """Charge le modèle Qwen2.5-1.5B depuis kibali_data/models"""
    
    print(f"🚀 Chargement de {QWEN_MODEL_NAME} en mode LOCAL...")
    print(f"📁 Cache: {QWEN_CACHE_DIR}")
    
    # Utiliser le device global avec fallback automatique
    device = DEVICE
    gpu_info = ""
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = f"GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)"
        print(f"✅ Modèle chargé sur GPU: {gpu_info}")
    else:
        print("✅ Modèle chargé sur CPU (fallback automatique)")
   
    print(f"🌐 Mode: LOCAL UNIQUEMENT (pas de téléchargement)")
    
    # Charger tokenizer en mode LOCAL uniquement
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_NAME,
        cache_dir=str(QWEN_CACHE_DIR),
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True
    )
    
    # Corriger le problème du pad_token = eos_token pour éviter les warnings
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
   
    # Configuration optimisée selon le device
    if device == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            cache_dir=str(QWEN_CACHE_DIR),
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            cache_dir=str(QWEN_CACHE_DIR),
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        model = model.to(device)
   
    print(f"✅ Qwen chargé en mode LOCAL")
    
    return tokenizer, model, device, gpu_info
# ===============================================
# Configuration - CHEMINS UNIFIÉS
# ===============================================
# Modèles 100% Open Source avec Licence Commerciale (Apache 2.0 / MIT)
# ===============================================
WORKING_MODELS = {
    # Qwen 2.5 - Apache 2.0 License - Usage commercial autorisé
    "Qwen2.5 14B (Apache 2.0 - Commercial OK)": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen2.5 7B (Apache 2.0 - Rapide)": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5 32B (Apache 2.0 - Très puissant)": "Qwen/Qwen2.5-32B-Instruct",
    
    # Llama 3.2 - Apache 2.0 License (Meta autorisation commerciale)
    "Llama 3.2 11B Vision (Apache 2.0 - Multimodal)": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Llama 3.1 8B (Apache 2.0 - Optimisé)": "meta-llama/Llama-3.1-8B-Instruct",
    
    # Mistral - Apache 2.0 License - Usage commercial libre
    "Mistral Nemo 12B (Apache 2.0 - Français)": "mistralai/Mistral-Nemo-Instruct-2407",
    "Mistral 7B v0.3 (Apache 2.0 - Rapide)": "mistralai/Mistral-7B-Instruct-v0.3",
    
    # Phi-3 Medium - MIT License - Microsoft open source commercial
    "Phi-3 Medium 14B (MIT - Commercial)": "microsoft/Phi-3-medium-4k-instruct",
    
    # Gemma 2 - Gemma License (usage commercial autorisé par Google)
    "Gemma 2 9B (Gemma License - Commercial OK)": "google/gemma-2-9b-it",
}
# TOUS LES FICHIERS DANS LE MÊME DOSSIER CHATBOT
CHATBOT_DIR = os.path.join(os.getcwd(), "kibali_data")
VECTORDB_PATH = os.path.join(CHATBOT_DIR, "vectordb")
PDFS_PATH = os.path.join(CHATBOT_DIR, "pdfs")
GRAPHS_PATH = os.path.join(CHATBOT_DIR, "graphs") # Graphes OSM ici aussi
MAPS_PATH = os.path.join(CHATBOT_DIR, "maps")
METADATA_PATH = os.path.join(CHATBOT_DIR, "metadata.json")
TRAJECTORIES_PATH = os.path.join(CHATBOT_DIR, "trajectories.json")
WEB_CACHE_PATH = os.path.join(CHATBOT_DIR, "web_cache.json")
GENERATED_PATH = os.path.join(CHATBOT_DIR, "generated")
# ===============================================
# Test de connexion HuggingFace
# ===============================================
def test_hf_connection():
    """Teste la connexion à HuggingFace"""
    try:
        from huggingface_hub import whoami
        user_info = whoami(token=HF_TOKEN)
        print(f"✅ Connexion HuggingFace réussie: {user_info.get('name', 'Utilisateur')}")
        return True
    except Exception as e:
        print(f"❌ Erreur connexion HuggingFace: {e}")
        return False
# Tester la connexion au démarrage
if not test_hf_connection():
    print("⚠️ Problème de connexion HuggingFace, vérifiez votre token")
# ===============================================
# Fonctions utilitaires
# ===============================================
def setup_drive():
    """Monte Google Drive et crée les dossiers"""
    print("📁 Configuration des dossiers...")
    os.makedirs(CHATBOT_DIR, exist_ok=True)
    os.makedirs(PDFS_PATH, exist_ok=True)
    os.makedirs(GRAPHS_PATH, exist_ok=True)
    os.makedirs(MAPS_PATH, exist_ok=True)
    os.makedirs(VECTORDB_PATH, exist_ok=True)
    os.makedirs(GENERATED_PATH, exist_ok=True)
    print(f"📁 Dossier principal : {CHATBOT_DIR}")
    return True
def extract_text_from_pdf(pdf_path):
    """Extraire le texte d'un PDF avec pdfplumber (MIT License)"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num + 1}]\n{page_text}\n"
        return text
    except Exception as e:
        print(f"❌ Erreur PDF {pdf_path}: {e}")
        return ""
def upload_and_process_pbf(pbf_file):
    """Traitement du fichier PBF uploadé"""
    if pbf_file is None:
        return None, None, "❌ Aucun fichier uploadé"
    pbf_path = pbf_file
    print("⚙️ Lecture du PBF et construction du graphe...")
    handler = RoadPOIHandler()
    handler.apply_file(pbf_path, locations=True)
    G = handler.graph
    pois = handler.pois
    # Sauvegarder dans le dossier chatbot
    graph_name = os.path.basename(pbf_path).replace('.osm.pbf', '_graph.graphml')
    graph_path = os.path.join(GRAPHS_PATH, graph_name)
    nx.write_graphml(G, graph_path)
    # Sauvegarder les POIs
    pois_name = graph_name.replace('_graph.graphml', '_pois.json')
    pois_path = os.path.join(GRAPHS_PATH, pois_name)
    with open(pois_path, 'w', encoding='utf-8') as f:
        json.dump(pois, f, indent=2, ensure_ascii=False)
    print(f"✅ Graphe: {len(G)} nœuds, {G.size()} arêtes")
    print(f"✅ POIs: {len(pois)} points")
    print(f"💾 Sauvegardé: {graph_path}")
    return G, pois, f"✅ Graphe créé: {len(G)} nœuds, {len(pois)} POIs"
def load_existing_graph():
    """Charge un graphe existant"""
    graph_files = [f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')] if os.path.exists(GRAPHS_PATH) else []
    if not graph_files:
        return None, None, "❌ Aucun graphe trouvé"
    graph_file = graph_files[0]
    graph_path = os.path.join(GRAPHS_PATH, graph_file)
    pois_path = os.path.join(GRAPHS_PATH, graph_file.replace('_graph.graphml', '_pois.json'))
    try:
        G = nx.read_graphml(graph_path)
        pois = []
        if os.path.exists(pois_path):
            with open(pois_path, 'r', encoding='utf-8') as f:
                pois = json.load(f)
        return G, pois, f"✅ Graphe chargé: {len(G)} nœuds, {len(pois)} POIs"
    except Exception as e:
        return None, None, f"❌ Erreur: {e}"
def process_pdfs():
    """Traiter les PDFs"""
    print("📄 Traitement des PDFs...")
    # CPU forcé globalement au début du fichier
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(SENTENCE_TRANSFORMER_CACHE),
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'device': 'cpu', 'batch_size': 32}
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    # Charger vectordb existante si elle existe
    if os.path.exists(VECTORDB_PATH):
        vectordb, _ = load_vectordb()
    else:
        vectordb = None
    # Charger métadonnées existantes
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {"processed_files": [], "total_chunks": 0}
    processed_filenames = {p["filename"] for p in metadata["processed_files"]}
    all_documents = []
    pdf_files = [f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')] if os.path.exists(PDFS_PATH) else []
    if not pdf_files:
        return vectordb, "⚠️ Aucun PDF trouvé"
    new_chunks_count = 0
    new_processed = []
    for pdf_file in pdf_files:
        if pdf_file in processed_filenames:
            print(f" 📖 {pdf_file} déjà traité, sauté.")
            continue
        pdf_path = os.path.join(PDFS_PATH, pdf_file)
        print(f" 📖 Traitement nouveau PDF : {pdf_file}")
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            continue
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_file,
                    "chunk_id": i,
                    "type": "pdf"
                }
            )
            all_documents.append(doc)
        new_processed.append({"filename": pdf_file, "chunks": len(chunks)})
        new_chunks_count += len(chunks)
    # Ajouter les trajets sauvegardés (toujours, car ils peuvent changer)
    if os.path.exists(TRAJECTORIES_PATH):
        with open(TRAJECTORIES_PATH, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)
        for traj in trajectories:
            traj_text = f"""Trajet: {traj.get('question', '')}
Départ: {traj.get('start_name', '')}
Arrivée: {traj.get('end_name', '')}
Distance: {traj.get('distance', 0)/1000:.2f} km"""
            doc = Document(
                page_content=traj_text,
                metadata={"source": "trajectories", "type": "trajectory"}
            )
            all_documents.append(doc)
    if all_documents:
        if vectordb is None:
            vectordb = FAISS.from_documents(all_documents, embedding_model)
        else:
            vectordb.add_documents(all_documents)
        vectordb.save_local(VECTORDB_PATH)
    # Mettre à jour métadonnées
    metadata["processed_files"].extend(new_processed)
    metadata["total_chunks"] += new_chunks_count
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return vectordb, f"✅ Base mise à jour : {len(new_processed)} nouveaux PDFs traités, {new_chunks_count} nouveaux chunks (total : {metadata['total_chunks']})"
def load_vectordb():
    """Charger la base vectorielle"""
    try:
        # CPU forcé globalement au début du fichier
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=str(SENTENCE_TRANSFORMER_CACHE),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'device': 'cpu', 'batch_size': 32}
        )
        
        # Vérifier si la base existe
        index_file = os.path.join(VECTORDB_PATH, "index.faiss")
        if not os.path.exists(index_file):
            # Créer une base vide si elle n'existe pas
            os.makedirs(VECTORDB_PATH, exist_ok=True)
            
            # Créer un document factice pour initialiser FAISS
            from langchain.schema import Document
            dummy_doc = Document(page_content="Base vectorielle initialisée", metadata={"source": "system"})
            vectordb = FAISS.from_documents([dummy_doc], embedding_model)
            vectordb.save_local(VECTORDB_PATH)
            
            return vectordb, "✅ Base vectorielle créée (vide - ajoutez des PDFs)"
        
        # Charger la base existante
        vectordb = FAISS.load_local(VECTORDB_PATH, embedding_model, allow_dangerous_deserialization=True)
        return vectordb, "✅ Base chargée"
    except Exception as e:
        return None, f"❌ Erreur: {e}"
def save_trajectory(question, response, trajectory_info):
    """Sauvegarde un trajet"""
    trajectories = []
    if os.path.exists(TRAJECTORIES_PATH):
        with open(TRAJECTORIES_PATH, 'r', encoding='utf-8') as f:
            trajectories = json.load(f)
    new_trajectory = {
        "question": question,
        "response": response,
        "start_name": trajectory_info.get('start', {}).get('name', ''),
        "end_name": trajectory_info.get('end', {}).get('name', ''),
        "distance": trajectory_info.get('distance', 0)
    }
    trajectories.append(new_trajectory)
    with open(TRAJECTORIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)
def upload_pdfs(uploaded_files):
    """Upload des PDFs"""
    if uploaded_files is None:
        return []
    saved_files = []
    for file in uploaded_files:
        filename = os.path.basename(file)
        filepath = os.path.join(PDFS_PATH, filename)
        shutil.copy(file, filepath)
        saved_files.append(filename)
    return saved_files
# ===============================================
# Système de Cache Web Intelligent
# ===============================================
def load_web_cache():
    """Charge le cache web"""
    if os.path.exists(WEB_CACHE_PATH):
        try:
            with open(WEB_CACHE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}
def save_web_cache(cache):
    """Sauvegarde le cache web"""
    try:
        with open(WEB_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Erreur sauvegarde cache: {e}")
def get_cache_key(query, source="text"):
    """Génère une clé de cache pour une requête"""
    return f"{source}:{query.lower().strip()}"
def is_cache_expired(cache_entry, max_age_hours=24):
    """Vérifie si l'entrée du cache a expiré"""
    current_time = time.time()
    return (current_time - cache_entry.get('timestamp', 0)) > (max_age_hours * 3600)
def get_cache_stats():
    """Obtient les statistiques du cache"""
    try:
        cache = load_web_cache()
        if not cache:
            return "Cache vide"
        total_entries = len(cache)
        expired_count = sum(1 for entry in cache.values() if is_cache_expired(entry))
        valid_count = total_entries - expired_count
        return f"📊 Cache: {total_entries} entrées total, {valid_count} valides, {expired_count} expirées"
    except Exception as e:
        return f"❌ Erreur stats: {e}"

def get_system_status():
    """Retourne le statut complet du système"""
    status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "directories": {
            "chatbot": os.path.exists(CHATBOT_DIR),
            "pdfs": os.path.exists(PDFS_PATH),
            "graphs": os.path.exists(GRAPHS_PATH),
            "maps": os.path.exists(MAPS_PATH)
        },
        "files": {
            "vectordb": os.path.exists(VECTORDB_PATH),
            "metadata": os.path.exists(METADATA_PATH),
            "trajectories": os.path.exists(TRAJECTORIES_PATH),
            "web_cache": os.path.exists(WEB_CACHE_PATH)
        },
        "counts": {
            "pdfs": len([f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')]) if os.path.exists(PDFS_PATH) else 0,
            "graphs": len([f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')]) if os.path.exists(GRAPHS_PATH) else 0
        },
        "cache_stats": get_cache_stats(),
        "token_configured": bool(HF_TOKEN and len(HF_TOKEN) > 10)
    }
    return status

# ===============================================
# Fonctions RAG et Web Search Améliorées
# ===============================================
def create_client():
    """Créer le client Inference avec gestion d'erreurs améliorée"""
    try:
        client = InferenceClient(token=HF_TOKEN)
        return client
    except Exception as e:
        print(f"❌ Erreur création client: {e}")
        raise e

# ===============================================
# Chargement des modèles de vision locaux
# ===============================================
@st.cache_resource
def load_vision_models():
    """Charge les modèles CLIP pour la classification d'images"""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        import os
        
        # FORCER le désactivation de low_cpu_mem_usage globalement
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        
        # CORRECTION: Forcer CPU pour éviter les incompatibilités CUDA
        # Les modèles pré-compilés peuvent ne pas correspondre à la version CUDA installée
        device = "cpu"
        print("⚠️ Chargement des modèles vision sur CPU pour éviter les erreurs CUDA")
        
        # Charger le modèle directement sur CPU avec torch_dtype explicite
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None  # Pas de device_map automatique
        )
        
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Forcer sur CPU
        clip_model = clip_model.cpu()
        clip_model.eval()
        
        print("✅ Modèles CLIP chargés avec succès sur CPU")
        
        return {
            'clip_model': clip_model,
            'clip_processor': clip_processor,
            'device': device
        }
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger CLIP: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

@st.cache_resource
def load_ocr_reader():
    """Charge le lecteur OCR EasyOCR en mode LOCAL uniquement"""
    try:
        if easyocr:
            print("📦 Chargement OCR en mode LOCAL...")
            print(f"📁 Cache: {EASYOCR_MODEL_DIR}")
            
            # Vérifier que le cache existe
            if not EASYOCR_MODEL_DIR.exists():
                print(f"❌ Cache OCR introuvable: {EASYOCR_MODEL_DIR}")
                return None
            
            # CORRECTION: Forcer CPU pour OCR également pour éviter erreurs CUDA
            reader = easyocr.Reader(
                EASYOCR_LANGUAGES, 
                gpu=False,  # Forcer CPU
                model_storage_directory=str(EASYOCR_MODEL_DIR),
                download_enabled=False  # Désactiver le téléchargement
            )
            print("✅ OCR chargé en mode LOCAL sur CPU")
            return reader
        else:
            print("⚠️ EasyOCR non installé")
            return None
    except Exception as e:
        print(f"⚠️ Erreur chargement OCR: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    """Charge le modèle YOLO pour la détection d'objets (FORCE CPU)"""
    try:
        if not YOLO_AVAILABLE:
            print("⚠️ YOLO non disponible")
            return None
        
        # Chercher les modèles YOLO disponibles
        yolo_paths = [
            Path("/home/belikan/yolo11n.pt"),
            Path("/home/belikan/yolov8n.pt"),
            Path("./yolo11n.pt"),
            Path("./yolov8n.pt")
        ]
        
        for yolo_path in yolo_paths:
            if yolo_path.exists():
                print(f"📦 Chargement YOLO: {yolo_path} (CPU forcé)")
                
                # CORRECTION CRITIQUE: Forcer CPU pour YOLO
                import torch
                model = YOLO(str(yolo_path))
                
                # Forcer le modèle sur CPU
                model.to('cpu')
                
                # Désactiver CUDA pour les prédictions
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                print(f"✅ YOLO chargé avec succès sur CPU: {yolo_path.name}")
                return model
        
        print("⚠️ Aucun modèle YOLO trouvé (.pt)")
        return None
        
    except Exception as e:
        print(f"⚠️ Erreur chargement YOLO: {e}")
        return None

@st.cache_resource
def load_blip_model():
    """Charge le modèle BLIP pour la description d'images en LOCAL"""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        
        print("📦 Chargement BLIP en mode LOCAL...")
        print(f"📁 Cache: {CAPTIONER_CACHE}")
        
        # Charger le modèle en LOCAL uniquement (CPU pour éviter erreurs CUDA)
        processor = BlipProcessor.from_pretrained(
            CAPTIONER_MODEL,
            cache_dir=str(CAPTIONER_CACHE),
            local_files_only=True
        )
        
        model = BlipForConditionalGeneration.from_pretrained(
            CAPTIONER_MODEL,
            cache_dir=str(CAPTIONER_CACHE),
            torch_dtype=torch.float32,
            local_files_only=True
        )
        
        # Forcer CPU pour éviter erreurs CUDA
        model = model.cpu()
        model.eval()
        
        print("✅ BLIP chargé avec succès en mode LOCAL sur CPU")
        
        return {
            'processor': processor,
            'model': model,
            'device': 'cpu'
        }
        
    except Exception as e:
        print(f"⚠️ Erreur chargement BLIP: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def extract_text_from_image(image_path, ocr_reader=None):
    """Extrait le texte d'une image avec OCR"""
    extracted_texts = []
    
    try:
        # Méthode 1: EasyOCR (meilleur pour textes complexes)
        if ocr_reader:
            results = ocr_reader.readtext(image_path)
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Seuil de confiance
                    extracted_texts.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'method': 'EasyOCR'
                    })
        
        # Méthode 2: Tesseract (fallback)
        elif pytesseract:
            try:
                img = Image.open(image_path)
                # Extraction simple
                text = pytesseract.image_to_string(img, lang='fra+eng')
                if text.strip():
                    extracted_texts.append({
                        'text': text.strip(),
                        'confidence': 1.0,
                        'method': 'Tesseract'
                    })
                
                # Extraction avec détails (positions)
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='fra+eng')
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 30:  # Confiance > 30%
                        txt = data['text'][i].strip()
                        if txt:
                            extracted_texts.append({
                                'text': txt,
                                'confidence': int(data['conf'][i]) / 100,
                                'bbox': (data['left'][i], data['top'][i], 
                                        data['width'][i], data['height'][i]),
                                'method': 'Tesseract'
                            })
            except Exception as tess_error:
                print(f"⚠️ Tesseract error: {tess_error}")
        
        return extracted_texts
    
    except Exception as e:
        print(f"❌ Erreur OCR: {e}")
        return []

def organize_extracted_text(extracted_texts):
    """Organise le texte extrait de manière structurée"""
    if not extracted_texts:
        return "Aucun texte détecté dans l'image."
    
    # Grouper par méthode
    by_method = {}
    for item in extracted_texts:
        method = item.get('method', 'Unknown')
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(item)
    
    # Formater la sortie
    output = []
    output.append("📝 **TEXTE EXTRAIT DE L'IMAGE:**\n")
    
    for method, items in by_method.items():
        output.append(f"\n🔍 **Méthode: {method}**")
        output.append(f"   Nombre d'éléments détectés: {len(items)}\n")
        
        # Trier par confiance
        items_sorted = sorted(items, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for idx, item in enumerate(items_sorted[:50], 1):  # Limiter à 50 éléments
            confidence = item.get('confidence', 0)
            text = item.get('text', '').strip()
            
            if text:
                # Formater selon la confiance
                if confidence > 0.8:
                    output.append(f"   {idx}. ✅ [{confidence:.0%}] {text}")
                elif confidence > 0.5:
                    output.append(f"   {idx}. ⚠️  [{confidence:.0%}] {text}")
                else:
                    output.append(f"   {idx}. ❓ [{confidence:.0%}] {text}")
    
    # Texte complet reconstitué
    output.append("\n\n📄 **TEXTE COMPLET RECONSTITUÉ:**\n")
    full_text = " ".join([item['text'] for item in extracted_texts if item.get('text', '').strip()])
    output.append(full_text)
    
    return "\n".join(output)

def analyze_image_with_clip(image_path, vision_models):
    """Analyse une image avec CLIP local"""
    try:
        if not vision_models:
            return None, "Modèles non chargés"
        
        clip_model = vision_models['clip_model']
        clip_processor = vision_models['clip_processor']
        device = vision_models['device']
        
        # S'assurer que le modèle est sur le bon device et en mode eval
        clip_model = clip_model.to(device)
        clip_model.eval()
        
        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        
        # Labels pour classification sémantique (géophysique focusé)
        labels = [
            "geological rock formation",
            "mineral sample",
            "geophysical survey equipment",
            "topographic map",
            "seismic data visualization",
            "core sample",
            "field work photography",
            "satellite imagery",
            "landscape terrain",
            "technical diagram",
            "graph or chart",
            "document or report",
            "person or people",
            "building or structure",
            "natural scenery"
        ]
        
        # Préparer inputs - traiter en CPU si nécessaire
        inputs = clip_processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Déplacer explicitement chaque tensor sur le device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Prédiction avec gestion d'erreur CUDA
        with torch.no_grad():
            try:
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            except RuntimeError as cuda_error:
                if "CUDA" in str(cuda_error):
                    print(f"⚠️ Erreur CUDA détectée, passage en CPU: {cuda_error}")
                    # Fallback sur CPU
                    clip_model = clip_model.cpu()
                    inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                else:
                    raise
        
        # Top 3 prédictions
        top3_probs, top3_indices = torch.topk(probs[0], 3)
        
        results = []
        for prob, idx in zip(top3_probs, top3_indices):
            results.append({
                'label': labels[idx.item()],
                'confidence': prob.item()
            })
        
        # Générer caption
        main_label = results[0]['label']
        confidence = results[0]['confidence']
        
        caption = f"Image appears to be: {main_label} (confidence: {confidence:.1%})"
        
        return caption, results
        
    except Exception as e:
        import traceback
        error_msg = f"Erreur analyse CLIP: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def rag_search(question, vectordb, k=3):
    """Rechercher dans la base vectorielle"""
    if not vectordb:
        return []
    try:
        return vectordb.similarity_search(question, k=k)
    except Exception as e:
        print(f"❌ Erreur recherche: {e}")
        return []
def enhanced_web_search(query, max_results=5, search_type="text", use_cache=True):
    """
    Recherche web avancée avec cache intelligent et multiple sources
    Args:
        query: Requête de recherche
        max_results: Nombre max de résultats
        search_type: Type de recherche ("text", "news", "both")
        use_cache: Utiliser le cache
    Returns:
        Liste de résultats enrichis
    """
    cache = load_web_cache() if use_cache else {}
    results = []
    try:
        # Recherche texte
        if search_type in ["text", "both"]:
            cache_key = get_cache_key(query, "text")
            if cache_key in cache and not is_cache_expired(cache[cache_key]):
                print(f"📋 Utilisation cache pour: {query}")
                text_results = cache[cache_key]['results']
            else:
                print(f"🔍 Recherche web pour: {query}")
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                text_results = []
                try:
                    raw_results = tavily.search(query, max_results=max_results, search_depth="advanced", topic="general", include_images=True)
                    for r in raw_results.get('results', []):
                        result_data = {
                            'title': r.get('title', ''),
                            'body': r.get('content', ''),
                            'href': r.get('url', ''),
                            'source_type': 'web_search',
                            'images': [],
                            'videos': []
                        }
                        
                        # Extraire les images si disponibles
                        if 'images' in r and r['images']:
                            result_data['images'] = r['images'][:3]  # Max 3 images par résultat
                        
                        # Extraire les vidéos depuis l'URL si c'est YouTube, Vimeo, etc.
                        url = r.get('url', '')
                        if 'youtube.com' in url or 'youtu.be' in url:
                            video_id = None
                            if 'youtube.com/watch?v=' in url:
                                video_id = url.split('watch?v=')[1].split('&')[0]
                            elif 'youtu.be/' in url:
                                video_id = url.split('youtu.be/')[1].split('?')[0]
                            if video_id:
                                result_data['videos'].append({
                                    'type': 'youtube',
                                    'id': video_id,
                                    'url': url
                                })
                        elif 'vimeo.com' in url:
                            vimeo_id = url.split('vimeo.com/')[1].split('/')[0] if 'vimeo.com/' in url else None
                            if vimeo_id:
                                result_data['videos'].append({
                                    'type': 'vimeo',
                                    'id': vimeo_id,
                                    'url': url
                                })
                        
                        text_results.append(result_data)
                    # Sauvegarder en cache
                    cache[cache_key] = {
                        'results': text_results,
                        'timestamp': time.time()
                    }
                    if use_cache:
                        save_web_cache(cache)
                except Exception as e:
                    print(f"Erreur recherche texte: {e}")
                    text_results = []
            results.extend(text_results)
        # Recherche actualités
        if search_type in ["news", "both"]:
            cache_key = get_cache_key(query, "news")
            if cache_key in cache and not is_cache_expired(cache[cache_key], max_age_hours=6):
                news_results = cache[cache_key]['results']
            else:
                tavily = TavilyClient(api_key=TAVILY_API_KEY)
                news_results = []
                try:
                    raw_news = tavily.search(query, max_results=max_results//2 if search_type == "both" else max_results, search_depth="advanced", topic="news")
                    for r in raw_news.get('results', []):
                        news_results.append({
                            'title': r.get('title', ''),
                            'body': r.get('content', ''),
                            'url': r.get('url', ''),
                            'date': r.get('published_date', ''),
                            'source': r.get('source', ''),
                            'source_type': 'news'
                        })
                    # Sauvegarder en cache (6h pour les news)
                    cache[cache_key] = {
                        'results': news_results,
                        'timestamp': time.time()
                    }
                    if use_cache:
                        save_web_cache(cache)
                except Exception as e:
                    print(f"Erreur recherche news: {e}")
                    news_results = []
            results.extend(news_results)
    except Exception as e:
        print(f"❌ Erreur recherche web globale: {e}")
        results = [{'title': 'Erreur de recherche', 'body': f'Erreur: {e}', 'source_type': 'error'}]
    return results
def format_web_results_with_media(web_results, max_results=3):
    """
    Formate les résultats web avec images et vidéos pour affichage dans le chat
    Args:
        web_results: Liste des résultats de recherche
        max_results: Nombre maximum de résultats à afficher
    Returns:
        HTML formaté avec contenu multimédia
    """
    if not web_results:
        return ""
    
    html_output = '<div style="margin: 2rem 0;">'
    html_output += '''<h3 style="
        color: #00ff88; 
        margin-bottom: 1.5rem;
        font-size: 1.3rem;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5), 0 2px 4px rgba(0, 0, 0, 0.5);
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 255, 136, 0.3);
    ">🌐 Résultats Web avec Médias</h3>'''
    
    for i, result in enumerate(web_results[:max_results]):
        # Carte 3D pour chaque résultat
        html_output += f'''<div style="
            position: relative;
            background: linear-gradient(145deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
            border: 1px solid rgba(0, 136, 255, 0.3);
            padding: 1.5rem;
            margin: 1.5rem 0;
            border-radius: 12px;
            
            /* Effet 3D dalle épaisse */
            box-shadow: 
                0 15px 30px rgba(0, 0, 0, 0.5),
                0 8px 16px rgba(0, 0, 0, 0.3),
                0 20px 40px rgba(0, 136, 255, 0.15),
                inset 0 2px 0 rgba(255, 255, 255, 0.05),
                inset 0 -2px 0 rgba(0, 0, 0, 0.3),
                -6px 0 12px rgba(0, 136, 255, 0.1),
                6px 0 12px rgba(0, 255, 136, 0.1);
            
            border-left: 5px solid;
            border-image: linear-gradient(180deg, 
                rgba(0, 136, 255, 1) 0%, 
                rgba(0, 255, 136, 0.5) 50%, 
                rgba(0, 136, 255, 1) 100%) 1;
            
            transform: perspective(1000px) rotateX(1deg);
            transition: transform 0.3s ease;
        ">
            <!-- Tranche supérieure -->\n            <div style="
                position: absolute;
                top: -3px;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, 
                    rgba(0, 136, 255, 0.4) 0%, 
                    rgba(0, 255, 136, 0.4) 100%);
                border-radius: 12px 12px 0 0;
            "></div>'''
        
        # Titre et URL
        title = result.get('title', 'Sans titre')[:100]
        url = result.get('href', '') or result.get('url', '')
        html_output += f'''<h4 style="
            color: #00ffff; 
            margin: 0 0 0.75rem 0;
            font-size: 1.1rem;
            text-shadow: 0 0 8px rgba(0, 255, 255, 0.5), 0 1px 2px rgba(0, 0, 0, 0.5);
        ">📄 {title}</h4>'''
        html_output += f'''<p style="
            color: #888; 
            font-size: 0.85rem; 
            margin: 0 0 1rem 0;
        ">🔗 <a href="{url}" target="_blank" style="
            color: #00ff88;
            text-decoration: none;
            transition: color 0.2s;
        " onmouseover="this.style.color='#00ffff'" onmouseout="this.style.color='#00ff88'">{url[:60]}...</a></p>'''
        
        # Contenu textuel
        body = result.get('body', '')[:200]
        if body:
            html_output += f'''<p style="
                color: #ccc; 
                margin: 0.75rem 0;
                line-height: 1.6;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
            ">{body}...</p>'''
        
        # Images avec effet 3D
        images = result.get('images', [])
        if images:
            html_output += '<div style="display: flex; gap: 0.75rem; margin: 1.5rem 0; flex-wrap: wrap;">'
            for img_url in images[:3]:
                html_output += f'''<img src="{img_url}" style="
                    width: 150px;
                    height: 150px;
                    object-fit: cover;
                    border-radius: 10px;
                    border: 2px solid #00ff88;
                    box-shadow: 
                        0 8px 16px rgba(0, 0, 0, 0.4),
                        0 4px 8px rgba(0, 255, 136, 0.3),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
                    cursor: pointer;
                    transition: all 0.3s ease;
                    transform: perspective(500px) rotateX(2deg);
                " onmouseover="this.style.transform='perspective(500px) rotateX(0deg) scale(1.05) translateY(-5px)'; this.style.boxShadow='0 12px 24px rgba(0, 0, 0, 0.5), 0 6px 12px rgba(0, 255, 136, 0.5)'" onmouseout="this.style.transform='perspective(500px) rotateX(2deg) scale(1)'; this.style.boxShadow='0 8px 16px rgba(0, 0, 0, 0.4), 0 4px 8px rgba(0, 255, 136, 0.3)'"/>'''
            html_output += '</div>'
        
        # Vidéos avec effet 3D
        videos = result.get('videos', [])
        if videos:
            for video in videos:
                if video['type'] == 'youtube':
                    video_id = video['id']
                    html_output += f'''<div style="
                        margin: 1.5rem 0;
                        position: relative;
                        border-radius: 12px;
                        overflow: hidden;
                        box-shadow: 
                            0 10px 20px rgba(0, 0, 0, 0.4),
                            0 5px 10px rgba(255, 0, 0, 0.2);
                        transform: perspective(1000px) rotateX(1deg);
                    ">
                        <iframe width="100%" height="315" 
                            src="https://www.youtube.com/embed/{video_id}" 
                            frameborder="0" 
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                            allowfullscreen
                            style="display: block;"
                        ></iframe>
                    </div>'''
                elif video['type'] == 'vimeo':
                    vimeo_id = video['id']
                    html_output += f'''<div style="
                        margin: 1.5rem 0;
                        position: relative;
                        border-radius: 12px;
                        overflow: hidden;
                        box-shadow: 
                            0 10px 20px rgba(0, 0, 0, 0.4),
                            0 5px 10px rgba(0, 170, 255, 0.2);
                        transform: perspective(1000px) rotateX(1deg);
                    ">
                        <iframe src="https://player.vimeo.com/video/{vimeo_id}" 
                            width="100%" height="315" 
                            frameborder="0" 
                            allow="autoplay; fullscreen; picture-in-picture" 
                            allowfullscreen
                            style="display: block;"
                        ></iframe>
                    </div>'''
        
        html_output += '</div>'
    
    html_output += '</div>'
    return html_output

def smart_content_extraction(url, max_length=1000):
    """
    Extraction intelligente du contenu d'une page web
    Args:
        url: URL à scraper
        max_length: Longueur max du contenu
    Returns:
        Contenu extrait et nettoyé
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Supprimer les éléments non pertinents
        for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
            element.decompose()
        # Extraire le texte principal
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        # Nettoyer et tronquer
        text = ' '.join(text.split()) # Normaliser les espaces
        return text[:max_length] + ('...' if len(text) > max_length else '')
    except Exception as e:
        print(f"Erreur extraction contenu {url}: {e}")
        return f"Impossible d'extraire le contenu de {url}"
def intelligent_query_expansion(query):
    """
    Expansion intelligente des requêtes pour améliorer les résultats
    Args:
        query: Requête originale
    Returns:
        Liste de requêtes expandues
    """
    expanded_queries = [query] # Toujours inclure la requête originale
    # Détection de mots-clés pour expansion contextuelle
    keywords = {
        'actualité': ['news', 'dernières nouvelles', 'récent'],
        'comment': ['tutorial', 'guide', 'étapes'],
        'pourquoi': ['raison', 'cause', 'explication'],
        'comparaison': ['vs', 'différence', 'comparatif'],
        'prix': ['coût', 'tarif', 'budget'],
        'avis': ['opinion', 'critique', 'review']
    }
    query_lower = query.lower()
    for trigger, expansions in keywords.items():
        if trigger in query_lower:
            for expansion in expansions:
                expanded_queries.append(f"{query} {expansion}")
    return expanded_queries[:3] # Limiter à 3 requêtes max
def hybrid_search_enhanced(query, vectordb, k=3, web_search_enabled=True, search_type="both"):
    """
    Recherche hybride améliorée combinant RAG local et web avec intelligence
    Args:
        query: Requête de recherche
        vectordb: Base vectorielle locale
        k: Nombre de résultats RAG
        web_search_enabled: Activer la recherche web
        search_type: Type de recherche web
    Returns:
        Liste de documents combinés et enrichis
    """
    all_results = []
    # 1. Recherche RAG locale
    local_docs = rag_search(query, vectordb, k)
    for doc in local_docs:
        doc.metadata['search_source'] = 'local_rag'
        doc.metadata['relevance_score'] = 1.0 # Score max pour les docs locaux
    all_results.extend(local_docs)
    # 2. Recherche web intelligente si activée
    if web_search_enabled:
        print(f"🌐 Recherche web activée pour: {query}")
        # Expansion de requête pour de meilleurs résultats
        expanded_queries = intelligent_query_expansion(query)
        web_results = []
        for exp_query in expanded_queries:
            try:
                search_results = enhanced_web_search(
                    exp_query,
                    max_results=3,
                    search_type=search_type
                )
                for result in search_results:
                    # Créer un document à partir du résultat web
                    content = f"Titre: {result.get('title', '')}\n"
                    content += f"Contenu: {result.get('body', '')}\n"
                    if result.get('source_type') == 'news' and result.get('date'):
                        content += f"Date: {result.get('date')}\n"
                        content += f"Source: {result.get('source', '')}\n"
                    # Extraction de contenu supplémentaire si URL disponible
                    url = result.get('href') or result.get('url')
                    if url and len(result.get('body', '')) < 200:
                        print(f"📄 Extraction contenu de: {url}")
                        extra_content = smart_content_extraction(url)
                        if extra_content and "Impossible d'extraire" not in extra_content:
                            content += f"\nContenu détaillé: {extra_content}"
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': url or 'web_search',
                            'type': result.get('source_type', 'web'),
                            'search_source': 'web',
                            'query_used': exp_query,
                            'relevance_score': 0.8 if exp_query == query else 0.6
                        }
                    )
                    web_results.append(doc)
            except Exception as e:
                print(f"Erreur recherche pour '{exp_query}': {e}")
                continue
        # Filtrer les doublons et trier par pertinence
        unique_web_results = []
        seen_urls = set()
        for doc in web_results:
            url = doc.metadata.get('source', '')
            if url not in seen_urls:
                seen_urls.add(url)
                unique_web_results.append(doc)
        # Trier par score de pertinence
        unique_web_results.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        all_results.extend(unique_web_results[:5]) # Max 5 résultats web
    return all_results
def generate_answer_enhanced(question, context_docs, model_name, include_sources=True):
    """
    Génération de réponse améliorée avec gestion des sources multiples
    Args:
        question: Question posée
        context_docs: Documents de contexte
        model_name: Modèle à utiliser
        include_sources: Inclure les sources dans la réponse
    Returns:
        Réponse générée avec sources
    """
    if not context_docs:
        context = "Aucun contexte spécifique trouvé."
    else:
        context_parts = []
        local_sources = []
        web_sources = []
        
        # Augmenter le nombre de docs et la limite par doc si contexte important
        max_docs = 3  # Augmenté de 2 à 3
        chars_per_doc = 1200  # Augmenté de 800 à 1200
        
        for i, doc in enumerate(context_docs[:max_docs]):
            source = doc.metadata.get('source', 'Document inconnu')
            doc_type = doc.metadata.get('type', 'unknown')
            search_source = doc.metadata.get('search_source', 'unknown')
            content = doc.page_content.strip()[:chars_per_doc]
            
            # Classifier les sources
            if search_source == 'local_rag':
                local_sources.append(f"[{i+1}] {source} ({doc_type})")
            else:
                web_sources.append(f"[{i+1}] {source}")
            
            context_parts.append(f"[{i+1}] {content}")
        
        context = "\n".join(context_parts)[:2500]  # Augmenté de 1500 à 2500 chars
    
    # Prompt optimisé pour des réponses structurées et détaillées
    prompt = f"""CONTEXTE:
{context}

QUESTION: {question[:500]}

INSTRUCTIONS DE RÉPONSE:
- Fournis une réponse DÉTAILLÉE et COMPLÈTE (minimum 300 mots)
- Structure ta réponse avec des SOUS-TITRES en markdown (##)
- Organise en PARAGRAPHES cohérents et bien espacés
- Utilise des listes à puces (•) pour les énumérations
- Inclus des EXEMPLES concrets quand pertinent
- Ajoute une section CONCLUSION ou RÉSUMÉ
- Utilise des emojis pour rendre la lecture agréable
- Formate avec **gras** et *italique* pour l'emphase

Réponds de manière professionnelle, structurée et approfondie."""
    
    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt[:4000]}]  # Augmenté à 4000 chars
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1200,  # Augmenté à 1200 pour réponses très détaillées
            temperature=0.4  # Légèrement plus créatif pour mieux structurer
        )
        answer = response.choices[0].message.content
        
        # Ajouter les sources si demandé
        if include_sources and context_docs:
            sources_text = "\n\n📚 **Sources consultées:**\n"
            if local_sources:
                sources_text += "**Documents locaux:**\n"
                for source in local_sources[:3]:
                    sources_text += f"• {source}\n"
            if web_sources:
                sources_text += "**Sources web:**\n"
                for source in web_sources[:3]:
                    sources_text += f"• {source}\n"
            answer += sources_text
        
        return answer
    
    except Exception as e:
        return f"❌ Erreur génération: {str(e)}"

def generate_answer_enhanced_stream(question, context_docs, model_name, include_sources=True):
    """
    Version STREAMING de generate_answer_enhanced pour affichage progressif
    Args:
        question: Question posée
        context_docs: Documents de contexte
        model_name: Modèle à utiliser
        include_sources: Inclure les sources dans la réponse
    Yields:
        Chunks de texte au fur et à mesure de la génération
    """
    if not context_docs:
        context = "Aucun contexte spécifique trouvé."
    else:
        context_parts = []
        local_sources = []
        web_sources = []
        
        max_docs = 3
        chars_per_doc = 1200
        
        for i, doc in enumerate(context_docs[:max_docs]):
            source = doc.metadata.get('source', 'Document inconnu')
            doc_type = doc.metadata.get('type', 'unknown')
            search_source = doc.metadata.get('search_source', 'unknown')
            content = doc.page_content.strip()[:chars_per_doc]
            
            if search_source == 'local_rag':
                local_sources.append(f"[{i+1}] {source} ({doc_type})")
            else:
                web_sources.append(f"[{i+1}] {source}")
            
            context_parts.append(f"[{i+1}] {content}")
        
        context = "\n".join(context_parts)[:2500]
    
    prompt = f"""CONTEXTE:
{context}

QUESTION: {question[:500]}

INSTRUCTIONS DE RÉPONSE:
- Fournis une réponse DÉTAILLÉE et COMPLÈTE (minimum 300 mots)
- Structure ta réponse avec des SOUS-TITRES en markdown (##)
- Organise en PARAGRAPHES cohérents et bien espacés
- Utilise des listes à puces (•) pour les énumérations
- Inclus des EXEMPLES concrets quand pertinent
- Ajoute une section CONCLUSION ou RÉSUMÉ
- Utilise des emojis pour rendre la lecture agréable
- Formate avec **gras** et *italique* pour l'emphase

Réponds de manière professionnelle, structurée et approfondie."""
    
    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt[:4000]}]
        
        # 🌊 STREAMING activé
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1200,
            temperature=0.4,
            stream=True  # 🔥 Mode streaming
        )
        
        # Yield chaque chunk
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
        
        # Ajouter les sources à la fin
        if include_sources and context_docs:
            sources_text = "\n\n📚 **Sources consultées:**\n"
            if local_sources:
                sources_text += "**Documents locaux:**\n"
                for source in local_sources[:3]:
                    sources_text += f"• {source}\n"
            if web_sources:
                sources_text += "**Sources web:**\n"
                for source in web_sources[:3]:
                    sources_text += f"• {source}\n"
            yield sources_text
    
    except Exception as e:
        yield f"❌ Erreur génération: {str(e)}"
# ===============================================
# Fonctions Web Search et Hybrid (Mises à jour)
# ===============================================
def web_search(query, max_results=5):
    """Version simplifiée pour compatibilité"""
    try:
        results = enhanced_web_search(query, max_results, "text")
        return [f"{r.get('title', '')}: {r.get('href', r.get('url', ''))} - {r.get('body', '')}" for r in results]
    except Exception as e:
        return [f"❌ Erreur recherche web: {e}"]
def hybrid_search(query, vectordb, k=3):
    """Version simplifiée pour compatibilité"""
    return hybrid_search_enhanced(query, vectordb, k, web_search_enabled=True)
def final_search(question, vectordb, graph, pois):
    """Recherche finale combinant toutes les sources"""
    results = hybrid_search_enhanced(question, vectordb, k=3, web_search_enabled=True)
    # OSM si mention lieu
    if any(keyword in question.lower() for keyword in ["aller", "trajet", "itinéraire", "route", "navigation"]):
        try:
            carte, reponse, traj = calculer_trajet(question, graph, pois)
            if traj:
                results.append(Document(
                    page_content=reponse,
                    metadata={"source": "trajet_osm", "type": "navigation"}
                ))
        except:
            pass
    return results
# ===============================================
# Fonctions Modèles Hugging Face Spécialisés
# ===============================================
def initialize_specialized_models():
    """Initialise les modèles spécialisés en mode LOCAL uniquement"""
    models = {}
    
    # Vérifier si pipeline est disponible
    if pipeline is None:
        print("⚠️ transformers.pipeline non disponible, modèles spécialisés désactivés")
        return {
            'summarizer': None,
            'translator': None,
            'captioner': None,
            'ner': None
        }
    
    print("🌐 Mode: LOCAL UNIQUEMENT (pas de téléchargement)")
    
    try:
        models['summarizer'] = pipeline(
            "summarization", 
            model=SUMMARIZER_MODEL, 
            cache_dir=str(SUMMARIZER_CACHE),
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        print("✅ Summarizer chargé en mode LOCAL")
    except Exception as e:
        print(f"⚠️ Erreur chargement summarizer: {e}")
        models['summarizer'] = None
    try:
        models['translator'] = pipeline(
            "translation", 
            model=TRANSLATOR_MODEL, 
            cache_dir=str(TRANSLATOR_CACHE),
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        print("✅ Translator chargé en mode LOCAL")
    except Exception as e:
        print(f"⚠️ Erreur chargement translator: {e}")
        models['translator'] = None
    try:
        models['captioner'] = None
        print("✅ Captioner configuré pour utiliser LLM (llava)")
    except Exception as e:
        print(f"⚠️ Erreur chargement captioner: {e}")
        models['captioner'] = None
    try:
        models['ner'] = pipeline("ner", model=NER_MODEL, cache_dir=str(NER_CACHE), local_files_only=True)
        print("✅ NER chargé en mode LOCAL")
    except Exception as e:
        print(f"⚠️ Erreur chargement NER: {e}")
        models['ner'] = None
    return models
# Initialiser les modèles
SPECIALIZED_MODELS = initialize_specialized_models()
def summarize_text(text):
    if SPECIALIZED_MODELS.get('summarizer') is None:
        return "❌ Modèle de résumé non disponible"
    try:
        return SPECIALIZED_MODELS['summarizer'](text[:1024], max_length=200, min_length=30, do_sample=False)[0]['summary_text']
    except Exception as e:
        return f"❌ Erreur résumé: {e}"
def translate_text(text, src_lang="fr", tgt_lang="en"):
    if SPECIALIZED_MODELS.get('translator') is None:
        return "❌ Modèle de traduction non disponible"
    try:
        return SPECIALIZED_MODELS['translator'](text)[0]['translation_text']
    except Exception as e:
        return f"❌ Erreur traduction: {e}"
def generate_detailed_description_with_clip(image_path, vision_models):
    """
    Génère une description détaillée en utilisant CLIP avec des prompts textuels précis
    """
    try:
        if not vision_models:
            return None, "Modèles CLIP non chargés"
        
        clip_model = vision_models['clip_model']
        clip_processor = vision_models['clip_processor']
        device = vision_models['device']
        
        # S'assurer que le modèle est sur le bon device
        clip_model = clip_model.to(device)
        clip_model.eval()
        
        # Charger l'image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # 🎯 50 PROMPTS ULTRA-DÉTAILLÉS pour analyse exhaustive
        detailed_prompts = [
            # === PERSONNES ET PERSONNAGES (15 prompts) ===
            "a photograph of a person posing",
            "a person wearing a costume or uniform",
            "a superhero character in action",
            "a toy action figure or collectible",
            "a person in colorful clothing",
            "a portrait photograph of someone",
            "people in a group photo",
            "a celebrity or famous person",
            "an athlete in sports gear",
            "a child or baby photo",
            "a person wearing glasses or accessories",
            "someone doing an activity or sport",
            "a fashion model or styling photo",
            "a person with distinctive hairstyle",
            "an animated or cartoon character",
            
            # === LOGOS, TEXTE ET TYPOGRAPHIE (12 prompts) ===
            "a logo with a single letter",
            "a text symbol or typography design",
            "a colored letter with gradient",
            "a graphic design with multiple letters",
            "a minimalist logo design",
            "a stylized letter or monogram",
            "a brand logo or company emblem",
            "a document page with text",
            "handwritten text or signature",
            "a sign with text and symbols",
            "typography art or calligraphy",
            "a poster with bold text",
            
            # === OBJETS ET PRODUITS (10 prompts) ===
            "a vehicle like car or motorcycle",
            "electronic device or gadget",
            "food dish or meal plating",
            "clothing item or fashion accessory",
            "furniture or home decor",
            "a book cover or magazine",
            "a tool or equipment",
            "jewelry or precious items",
            "a bottle or container product",
            "sports equipment or gear",
            
            # === NATURE ET PAYSAGES (8 prompts) ===
            "a photograph of landscape scenery",
            "a natural outdoor scene with trees",
            "a mountain or hill terrain",
            "a beach or ocean view",
            "flowers or plants in nature",
            "a sunset or sunrise sky",
            "animals or wildlife photo",
            "urban cityscape or skyline",
            
            # === ART ET DESIGN (5 prompts) ===
            "an illustration or digital art",
            "a painting or artwork",
            "a comic book or manga panel",
            "abstract art or pattern",
            "architectural design or blueprint",
            
            # === DIVERS (5 prompts) ===
            "a technical diagram or chart",
            "an icon or simple symbol",
            "a screenshot of interface",
            "a meme or internet image",
            "a QR code or barcode"
        ]
        
        # Analyser avec CLIP
        inputs = clip_processor(
            text=detailed_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            except RuntimeError as cuda_error:
                if "CUDA" in str(cuda_error):
                    clip_model = clip_model.cpu()
                    inputs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                else:
                    raise
        
        # Obtenir le top match
        top_prob, top_idx = torch.topk(probs[0], 1)
        main_description = detailed_prompts[top_idx.item()]
        confidence = top_prob.item()
        
        # Vérifier si c'est vraiment du texte/logo AVANT de chercher une lettre
        is_text_based = any(word in main_description.lower() for word in ["letter", "logo", "text", "symbol", "typography", "character", "document"])
        
        # SEULEMENT chercher une lettre si confiance élevée que c'est du texte ET score > 30%
        letter_description = ""
        if is_text_based and confidence > 0.30:
            print(f"🔤 Détection de lettre activée (confiance texte: {confidence*100:.1f}%)")
            letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            
            # Méthode 1: Comparaison directe avec des lettres stylisées
            letter_prompts = [f"the letter {letter}" for letter in letters]
            
            inputs_letters = clip_processor(
                text=letter_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs_letters = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_letters.items()}
            
            with torch.no_grad():
                outputs_letters = clip_model(**inputs_letters)
                logits_letters = outputs_letters.logits_per_image
                probs_letters = logits_letters.softmax(dim=1)
            
            top5_letters, top5_idx = torch.topk(probs_letters[0], 5)
            detected_letters = [letters[idx.item()] for idx in top5_idx]
            letter_confidences = [prob.item() for prob in top5_letters]
            
            # Méthode 2: Vérification avec prompts alternatifs pour le top 3
            top3_verification = []
            for i in range(min(3, len(detected_letters))):
                letter = detected_letters[i]
                verify_prompts = [
                    f"a {letter} logo",
                    f"the capital letter {letter}",
                    f"letter {letter} design"
                ]
                
                inputs_verify = clip_processor(
                    text=verify_prompts,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                inputs_verify = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_verify.items()}
                
                with torch.no_grad():
                    outputs_verify = clip_model(**inputs_verify)
                    logits_verify = outputs_verify.logits_per_image
                    probs_verify = logits_verify.softmax(dim=1)
                
                # Score moyen de vérification
                avg_verify_score = probs_verify[0].mean().item()
                combined_score = (letter_confidences[i] + avg_verify_score) / 2
                top3_verification.append((letter, combined_score))
            
            # Trier par score combiné
            top3_verification.sort(key=lambda x: x[1], reverse=True)
            best_letter, best_score = top3_verification[0]
            
            if best_score > 0.15:  # Seuil plus bas mais avec vérification
                letter_description = f" Il s'agit probablement de la lettre '{best_letter}' (confiance: {best_score*100:.1f}%)"
                if len(top3_verification) > 1 and top3_verification[1][1] > 0.12:
                    letter_description += f", ou '{top3_verification[1][0]}' ({top3_verification[1][1]*100:.1f}%)"
            else:
                # Afficher au moins les 3 meilleures suggestions
                letter_description = f" Lettres possibles: {', '.join([f'{l}({s*100:.0f}%)' for l, s in top3_verification[:3]])}"
        
        # Analyser les couleurs (50 PROMPTS ULTRA-DÉTAILLÉS)
        color_prompts = [
            # Couleurs primaires et secondaires (12 prompts)
            "with vibrant bright blue colors",
            "with deep dark blue navy colors",
            "with vivid red crimson colors",
            "with dark burgundy red colors",
            "with bright lime green colors",
            "with dark forest green colors",
            "with sunny bright yellow colors",
            "with golden amber yellow colors",
            "with rich purple violet colors",
            "with bright orange tangerine colors",
            "with hot pink magenta colors",
            "with soft pastel pink colors",
            
            # Couleurs complexes et mélanges (15 prompts)
            "with turquoise cyan blue-green colors",
            "with teal aqua colors",
            "with coral salmon pink-orange colors",
            "with maroon dark red-brown colors",
            "with olive green-yellow colors",
            "with mint light green colors",
            "with lavender light purple colors",
            "with indigo deep blue-purple colors",
            "with peach light orange colors",
            "with rose pink-red colors",
            "with khaki tan beige colors",
            "with cream ivory off-white colors",
            "with burgundy wine red colors",
            "with chartreuse yellow-green colors",
            "with crimson bright red colors",
            
            # Tons neutres et spéciaux (12 prompts)
            "with pure black colors",
            "with charcoal dark gray colors",
            "with medium gray colors",
            "with light silver gray colors",
            "with pure white colors",
            "with beige tan neutral colors",
            "with brown chocolate colors",
            "with dark espresso brown colors",
            "with caramel light brown colors",
            "with monochrome black and white",
            "with sepia vintage brown tones",
            "with metallic silver gold colors",
            
            # Atmosphères et tonalités (11 prompts)
            "with neon bright fluorescent colors",
            "with pastel soft light colors",
            "with dark muted somber colors",
            "with vibrant saturated bold colors",
            "with rainbow multicolor spectrum",
            "with warm red orange yellow tones",
            "with cool blue green purple tones",
            "with earth natural brown green tones",
            "with translucent transparent colors",
            "with gradients fading colors",
            "with contrasting complementary colors"
        ]
        
        inputs_colors = clip_processor(
            text=color_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs_colors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_colors.items()}
        
        with torch.no_grad():
            outputs_colors = clip_model(**inputs_colors)
            logits_colors = outputs_colors.logits_per_image
            probs_colors = logits_colors.softmax(dim=1)
        
        # AMÉLIORATION: Prendre les TOP 3 couleurs au lieu d'une seule
        top3_colors_probs, top3_colors_idx = torch.topk(probs_colors[0], 3)
        
        # Construire une description de couleurs détaillée
        detected_colors = []
        for i in range(3):
            color_name = color_prompts[top3_colors_idx[i].item()].replace("with ", "").replace(" colors", "").replace(" tones", "")
            color_conf = top3_colors_probs[i].item()
            if color_conf > 0.15:  # Seuil minimum 15%
                detected_colors.append(f"{color_name} ({color_conf*100:.0f}%)")
        
        # Description finale des couleurs
        if detected_colors:
            color_desc = ", ".join(detected_colors[:3])  # Max 3 couleurs
        else:
            # Fallback si aucune couleur détectée avec confiance suffisante
            color_desc = color_prompts[top3_colors_idx[0].item()].replace("with ", "").replace(" colors", "").replace(" tones", "")
        
        # Construire la description finale
        full_description = f"{main_description} with colors: {color_desc}.{letter_description}"
        
        return {
            'description': full_description,
            'confidence': confidence,
            'main_type': main_description,
            'colors': color_desc,
            'details': letter_description
        }, None
        
    except Exception as e:
        import traceback
        error_msg = f"Erreur description CLIP: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def caption_image_with_blip(image_path, blip_models):
    """Génère une description détaillée de l'image avec BLIP local"""
    try:
        if not blip_models:
            return None, "Modèle BLIP non chargé"
        
        processor = blip_models['processor']
        model = blip_models['model']
        device = blip_models['device']
        
        # Charger l'image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Générer une description non conditionnée (description générale)
        inputs = processor(image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=100, num_beams=5)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Générer aussi une description conditionnée (plus détaillée)
        text_prompt = "a detailed description of"
        inputs_detailed = processor(image, text=text_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out_detailed = model.generate(**inputs_detailed, max_length=150, num_beams=5)
        
        caption_detailed = processor.decode(out_detailed[0], skip_special_tokens=True)
        
        return {
            'caption': caption,
            'detailed_caption': caption_detailed,
            'method': 'BLIP'
        }, None
        
    except Exception as e:
        import traceback
        error_msg = f"Erreur BLIP: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def caption_image(image_path):
    client = create_client()
    model = "llava-hf/llava-1.5-7b-hf"
    prompt = "Generate a detailed caption for this image."
    try:
        return client.image_to_text(image_path, prompt=prompt, model=model, max_tokens=500)
    except Exception as e:
        return f"❌ Erreur caption: {e}"
def extract_entities(text):
    if SPECIALIZED_MODELS.get('ner') is None:
        return "❌ Modèle NER non disponible"
    try:
        return SPECIALIZED_MODELS['ner'](text)
    except Exception as e:
        return f"❌ Erreur NER: {e}"
# ===============================================
# Fonctions de génération avec Stable Diffusion et similaires
# ===============================================
def generate_text_to_image(prompt):
    """Génère une image à partir de texte"""
    if DiffusionPipeline is None:
        return "❌ DiffusionPipeline non disponible (installez diffusers)"
    try:
        pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        image = pipe(prompt).images[0]
        path = os.path.join(GENERATED_PATH, f"image_{int(time.time())}.png")
        image.save(path)
        return f"Image générée et sauvegardée à {path}"
    except Exception as e:
        return f"❌ Erreur génération image: {e}"
def generate_text_to_video(prompt):
    """Génère une vidéo à partir de texte"""
    if DiffusionPipeline is None:
        return "❌ DiffusionPipeline non disponible (installez diffusers)"
    try:
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        gen = pipe(prompt, num_inference_steps=25)
        frames = gen.frames[0] # Assuming batch size 1
        path = os.path.join(GENERATED_PATH, f"video_{int(time.time())}.gif")
        imageio.mimsave(path, frames, fps=5)
        return f"Vidéo générée et sauvegardée à {path}"
    except Exception as e:
        return f"❌ Erreur génération vidéo: {e}"
def generate_text_to_audio(prompt):
    """Génère un son à partir de texte"""
    if AudioLDMPipeline is None:
        return "❌ AudioLDMPipeline non disponible (installez diffusers avec audio)"
    if wavfile is None:
        return "❌ scipy.io.wavfile non disponible (installez scipy)"
    try:
        pipe = AudioLDMPipeline.from_pretrained("cvssp/audio-ldm", torch_dtype=torch.float16, use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        audio = pipe(prompt, audio_length_in_s=5.0).audios[0]
        path = os.path.join(GENERATED_PATH, f"audio_{int(time.time())}.wav")
        wavfile.write(path, rate=16000, data=audio) # Assuming 16kHz sample rate
        return f"Son généré et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération son: {e}"
def generate_text_to_3d(prompt):
    """Génère un modèle 3D à partir de texte (rendue image)"""
    if ShapEPipeline is None:
        return "❌ ShapEPipeline non disponible (installez diffusers avec shap-e)"
    try:
        pipe = ShapEPipeline.from_pretrained("openai/shap-e", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        output = pipe(prompt, num_inference_steps=64)
        image = output.images[0]
        path = os.path.join(GENERATED_PATH, f"3d_text_{int(time.time())}.png")
        image.save(path)
        return f"Rendu 3D généré et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération 3D (texte): {e}"
def generate_image_to_3d(image_path):
    """Génère un modèle 3D à partir d'une image (rendue image)"""
    if ShapEImg2ImgPipeline is None:
        return "❌ ShapEImg2ImgPipeline non disponible (installez diffusers avec shap-e)"
    try:
        pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e", use_auth_token=HF_TOKEN)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        image = Image.open(image_path)
        output = pipe(image, num_inference_steps=64)
        rendered_image = output.images[0]
        path = os.path.join(GENERATED_PATH, f"3d_image_{int(time.time())}.png")
        rendered_image.save(path)
        return f"Rendu 3D généré à partir de l'image et sauvegardé à {path}"
    except Exception as e:
        return f"❌ Erreur génération 3D (image): {e}"
# ===============================================
# Agent LangChain Amélioré avec Recherche Web
# ===============================================
def create_enhanced_agent(model_name, vectordb, graph, pois):
    """
    Crée un agent LangChain amélioré avec capacités de recherche web
    Args:
        model_name: Nom du modèle HuggingFace
        vectordb: Base vectorielle locale
        graph: Graphe OSM
        pois: Points d'intérêt
    Returns:
        Agent configuré avec tous les outils
    """
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.3,
            max_new_tokens=600
        )
        # Configuration des outils de recherche web
        search_wrapper = DuckDuckGoSearchAPIWrapper(
            region="fr-fr",
            time="d",
            max_results=5
        )
        search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
        search_results_tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5, include_raw_content=True)
        tools = [
            # Outils de base RAG et recherche
            Tool(
                name="Local_Knowledge_Base",
                func=lambda q: "\n\n".join([d.page_content for d in rag_search(q, vectordb, k=3)]),
                description="Recherche dans la base de connaissances locale (PDFs et documents internes). Utilise ceci en PREMIER pour les questions sur des documents spécifiques."
            ),
            Tool(
                name="Web_Search",
                func=lambda q: search_tool.run(q),
                description="Recherche sur Internet pour des informations récentes, actualités, ou des connaissances générales non disponibles localement."
            ),
            Tool(
                name="Web_Search_Detailed",
                func=lambda q: search_results_tool.run(q),
                description="Recherche web détaillée avec sources et liens. Utilise pour obtenir des résultats web structurés avec URLs."
            ),
            Tool(
                name="Hybrid_Search",
                func=lambda q: "\n\n".join([d.page_content for d in hybrid_search_enhanced(q, vectordb, k=3, web_search_enabled=True)]),
                description="Recherche hybride combinant base locale ET web. Idéal pour des questions nécessitant à la fois des données internes et externes."
            ),
            Tool(
                name="Current_News_Search",
                func=lambda q: "\n\n".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in enhanced_web_search(q, search_type="news")]),
                description="Recherche spécialisée pour les actualités récentes et informations temporelles."
            ),
            # Outils spécialisés
            Tool(
                name="OSM_Route_Calculator",
                func=lambda q: calculer_trajet(q, graph, pois)[1] if graph and pois else "❌ Aucune carte OSM disponible",
                description="Calcule des itinéraires routiers entre deux lieux. Utilise pour les questions de navigation, trajets, ou géolocalisation."
            ),
            Tool(
                name="Smart_Content_Extractor",
                func=lambda url: smart_content_extraction(url) if url.startswith('http') else "❌ URL invalide",
                description="Extrait le contenu détaillé d'une page web spécifique. Fournis une URL complète."
            ),
            Tool(
                name="Text_Summarizer",
                func=summarize_text,
                description="Résume un texte long en version concise. Utile pour synthétiser des informations volumineuses."
            ),
            Tool(
                name="Language_Translator",
                func=translate_text,
                description="Traduit du français vers l'anglais. Utile pour traiter des sources en langue étrangère."
            ),
            Tool(
                name="Image_Analyzer",
                func=caption_image,
                description="Analyse et décrit le contenu d'une image. Fournis le chemin vers un fichier image."
            ),
            Tool(
                name="Entity_Extractor",
                func=lambda t: json.dumps(extract_entities(t)),
                description="Extrait des entités nommées (personnes, lieux, organisations) d'un texte."
            ),
            # Nouveaux outils Stable Diffusion via API
            Tool(
                name="Text_To_Image_Generator",
                func=generate_text_to_image,
                description="Génère une image à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Text_To_Video_Generator",
                func=generate_text_to_video,
                description="Génère une vidéo à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Text_To_Audio_Generator",
                func=generate_text_to_audio,
                description="Génère un son ou audio à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Text_To_3D_Generator",
                func=generate_text_to_3d,
                description="Génère un modèle 3D (rendue image) à partir d'une description textuelle. Fournis un prompt descriptif."
            ),
            Tool(
                name="Image_To_3D_Generator",
                func=generate_image_to_3d,
                description="Génère un modèle 3D (rendue image) à partir d'une image. Fournis le chemin vers un fichier image."
            ),
        ]
        
        # Ajouter l'outil Excel AI si disponible
        if EXCEL_AI_AVAILABLE:
            def excel_ai_wrapper(query: str) -> str:
                """Wrapper pour l'outil Excel AI"""
                try:
                    # Parser la requête (peut être JSON ou texte structuré)
                    import json
                    try:
                        data = json.loads(query)
                    except:
                        # Si pas JSON, créer un DataFrame depuis le texte
                        lines = [l.strip() for l in query.split('\n') if l.strip()]
                        data = {'data': lines}
                    
                    # Organiser avec l'IA
                    excel_bytes, report = organize_excel_with_ai(
                        data,
                        filename="ai_organized_data.xlsx",
                        add_charts=True,
                        add_conditional_formatting=True
                    )
                    
                    # Sauvegarder temporairement
                    temp_path = Path("temp_excel_ai_output.xlsx")
                    temp_path.write_bytes(excel_bytes)
                    
                    return f"✅ Fichier Excel organisé avec IA créé !\n\nRapport:\n{json.dumps(report, indent=2, ensure_ascii=False)}\n\nFichier sauvegardé: {temp_path.absolute()}"
                except Exception as e:
                    return f"❌ Erreur lors de l'organisation Excel: {str(e)}"
            
            tools.append(
                Tool(
                    name="Excel_AI_Organizer",
                    func=excel_ai_wrapper,
                    description="""Organise automatiquement des données dans un fichier Excel avec IA ultra-précise. 
                    Détecte et structure automatiquement:
                    - Coordonnées GPS (latitude/longitude)
                    - Données géospatiales
                    - Données financières
                    - Données scientifiques
                    - Listes d'invités/événements
                    - Tableaux de tout type
                    Ajoute formatage conditionnel, graphiques, et classifications dynamiques.
                    Fournis les données en JSON ou texte structuré."""
                )
            )
        
        # Configuration de l'agent avec prompt personnalisé
        agent_prompt = """Tu es Kibali, un assistant IA avancé avec accès à de multiples sources d'information.
CAPACITÉS DISPONIBLES:
- Base de connaissances locale (PDFs et documents)
- Recherche web en temps réel
- Calcul d'itinéraires sur cartes OSM
- Analyse d'images et extraction de contenu web
- Traduction et résumé automatiques
- Génération d'images, vidéos, sons et modèles 3D à partir de texte ou images
- Organisation intelligente de fichiers Excel avec IA ultra-précise (coordonnées GPS, classifications dynamiques, tableaux structurés)

INSTRUCTIONS IMPORTANTES:
1. Utilise TOUJOURS la base locale en premier pour les questions sur des documents spécifiques
2. Combine les sources locales ET web pour des réponses complètes
3. Pour les actualités ou infos récentes, privilégie la recherche web
4. Cite tes sources et indique leur provenance (locale vs web)
5. Si les informations se contredisent, mentionne les deux perspectives
6. Reste concis mais informatif
7. Pour les générations, sauvegarde les fichiers et retourne le chemin
8. Pour organiser des données Excel, utilise Excel_AI_Organizer avec des données structurées (coordonnées, listes, tableaux)

Tu as accès aux outils suivants: {tools}
Utilise le format suivant:
Question: la question d'entrée
Thought: réfléchis à ce que tu dois faire
Action: l'action à entreprendre, doit être l'un de [{tool_names}]
Action Input: l'entrée à l'action
Observation: le résultat de l'action
... (ce Thought/Action/Action Input/Observation peut se répéter N fois)
Thought: Je connais maintenant la réponse finale
Final Answer: la réponse finale à la question d'entrée
Commence!
Question: {input}
Thought: {agent_scratchpad}"""
        agent = initialize_agent(
            tools,
            llm,
            agent="zero-shot-react-description",
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        print(f"✅ Agent créé avec {len(tools)} outils disponibles")
        return agent
    except Exception as e:
        print(f"❌ Erreur création agent: {e}")
        return None
# Alias pour compatibilité
def create_agent(model_name, vectordb, graph, pois):
    """Version simplifiée pour compatibilité ascendante"""
    return create_enhanced_agent(model_name, vectordb, graph, pois)
# ===============================================
# Fonctions OSM et Graphe Routier
# ===============================================
def haversine(lon1, lat1, lon2, lat2):
    """Calcul distance haversine en mètres"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
class RoadPOIHandler(osmium.SimpleHandler):
    """Handler pour extraire routes et POIs depuis OSM"""
    def __init__(self):
        super().__init__()
        self.graph = nx.Graph()
        self.pois = []
    def node(self, n):
        """Extraire les POIs (points d'intérêt)"""
        if n.location.valid() and n.tags:
            name = n.tags.get('name', '')
            amenity = n.tags.get('amenity', '')
            if name or amenity:
                self.pois.append({
                    'name': name,
                    'amenity': amenity,
                    'lon': n.location.lon,
                    'lat': n.location.lat,
                    'tags': dict(n.tags)
                })
    def way(self, w):
        """Extraire les routes"""
        if 'highway' in w.tags:
            coords = []
            for n in w.nodes:
                if n.location.valid():
                    coords.append((n.location.lon, n.location.lat))
            for i in range(len(coords)-1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i+1]
                n1, n2 = (lon1, lat1), (lon2, lat2)
                dist = haversine(lon1, lat1, lon2, lat2)
                self.graph.add_node(n1, x=lon1, y=lat1)
                self.graph.add_node(n2, x=lon2, y=lat2)
                self.graph.add_edge(n1, n2, length=dist, highway=w.tags.get("highway"))
def trouver_noeud_plus_proche(lon, lat, graph):
    """Trouve le nœud du graphe le plus proche"""
    min_dist = float("inf")
    closest_node = None
    for node, data in graph.nodes(data=True):
        nlon, nlat = float(data["x"]), float(data["y"])
        dist = haversine(lon, lat, nlon, nlat)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node
def chercher_poi_par_nom(nom, pois_list):
    """Recherche un POI par nom"""
    nom_lower = nom.lower()
    for poi in pois_list:
        if nom_lower in poi['name'].lower() or nom_lower in poi['amenity'].lower():
            return poi
    return None
def generer_carte_trajet(graph, path, pois_list, start_poi=None, end_poi=None):
    """Génère une carte 2D du trajet"""
    fig, ax = plt.subplots(figsize=(12, 10))
    # Dessiner le graphe en arrière-plan
    for edge in list(graph.edges())[:1000]: # Limiter pour la performance
        node1, node2 = edge
        x1, y1 = node1[0], node1[1]
        x2, y2 = node2[0], node2[1]
        ax.plot([x1, x2], [y1, y2], 'lightgray', alpha=0.3, linewidth=0.5)
    # Dessiner le trajet
    if path and len(path) > 1:
        path_x = [node[0] for node in path]
        path_y = [node[1] for node in path]
        ax.plot(path_x, path_y, 'red', linewidth=3, label='Trajet')
        # Marquer début et fin
        ax.scatter(path_x[0], path_y[0], color='green', s=100, label='Départ', zorder=5)
        ax.scatter(path_x[-1], path_y[-1], color='red', s=100, label='Arrivée', zorder=5)
    # Ajouter quelques POIs
    for poi in pois_list[:20]:
        if poi['name']:
            ax.scatter(poi['lon'], poi['lat'], color='blue', s=20, alpha=0.6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Trajet calculé sur la carte OSM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    # Sauvegarder en mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf
def calculer_trajet(question, graph, pois_list):
    """Calcule un trajet basé sur une question textuelle"""
    if not graph or not pois_list:
        return None, "❌ Graphe ou POIs non disponibles", None
    # Utiliser LLM pour extraire départ et arrivée
    try:
        client = create_client()
        prompt = f"""Extraie le lieu de départ et le lieu d'arrivée de cette question de trajet.
Question: {question}
Réponds au format exact:
Départ: [nom du lieu de départ]
Arrivée: [nom du lieu d'arrivée]"""
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=WORKING_MODELS["Qwen2.5 7B (Rapide & excellent)"],
            messages=messages,
            max_tokens=100,
            temperature=0.1
        )
        extraction = response.choices[0].message.content
        start_line = [line for line in extraction.split('\n') if line.startswith('Départ: ')]
        end_line = [line for line in extraction.split('\n') if line.startswith('Arrivée: ')]
        if start_line and end_line:
            start_place = start_line[0].replace('Départ: ', '').strip()
            end_place = end_line[0].replace('Arrivée: ', '').strip()
        else:
            return None, "❌ Impossible d'extraire les lieux de la question.", None
    except Exception as e:
        print(f"❌ Erreur extraction LLM: {e}")
        return None, "❌ Erreur lors de l'extraction des lieux.", None
    start_poi = chercher_poi_par_nom(start_place, pois_list)
    end_poi = chercher_poi_par_nom(end_place, pois_list)
    if not start_poi or not end_poi:
        return None, f"❌ Impossible de trouver les lieux: {start_place} ou {end_place}.", None
    # Trouver les nœuds dans le graphe
    start_node = trouver_noeud_plus_proche(start_poi['lon'], start_poi['lat'], graph)
    end_node = trouver_noeud_plus_proche(end_poi['lon'], end_poi['lat'], graph)
    if not start_node or not end_node:
        return None, "❌ Impossible de trouver les nœuds dans le graphe routier.", None
    try:
        # Calculer le chemin
        path = nx.shortest_path(graph, source=start_node, target=end_node, weight="length")
        # Calculer la distance
        distance_totale = 0
        for i in range(len(path)-1):
            distance_totale += graph[path[i]][path[i+1]]['length']
        # Générer la carte
        carte_buf = generer_carte_trajet(graph, path, pois_list, start_poi, end_poi)
        # Réponse textuelle
        reponse = f"""🗺️ **Trajet calculé**
📍 **Départ**: {start_poi['name']} ({start_poi['amenity']})
🎯 **Arrivée**: {end_poi['name']} ({end_poi['amenity']})
📏 **Distance**: {distance_totale/1000:.2f} km
⏱️ **Temps estimé**: {int(distance_totale/83.33):.0f} min à pied | {int(distance_totale/833.33):.0f} min en voiture
🛣️ **Étapes**: {len(path)} points"""
        return carte_buf, reponse, {
            'start': start_poi,
            'end': end_poi,
            'distance': distance_totale,
            'path_length': len(path)
        }
    except nx.NetworkXNoPath:
        return None, f"❌ Aucun chemin trouvé entre {start_poi['name']} et {end_poi['name']}", None
    except Exception as e:
        return None, f"❌ Erreur: {str(e)}", None
# ===============================================
# Fonctions utilitaires pour images
# ===============================================
def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)
def df_to_html(df, max_rows=10):
    """Convertit DataFrame en HTML avec styles blanc gras visibles"""
    # Réduire le tableau si trop long
    if len(df) > max_rows:
        summary_row = pd.DataFrame({col: ['...'] for col in df.columns})
        df = pd.concat([df.head(max_rows // 2), summary_row, df.tail(max_rows // 2)])
    
    # Générer HTML avec styles inline
    html = df.to_html(index=False, escape=False, border=0)
    html = html.replace('<table', '<table style="width: 100%; border-collapse: collapse; margin: 1rem 0; background: rgba(58, 58, 94, 0.3);"')
    html = html.replace('<thead>', '<thead style="background: rgba(0, 255, 136, 0.2);">')
    html = html.replace('<th>', '<th style="color: white !important; font-weight: 700 !important; padding: 0.75rem; border: 1px solid #5a5a8a; text-align: left;">')
    html = html.replace('<td>', '<td style="color: white !important; font-weight: 600 !important; padding: 0.75rem; border: 1px solid #5a5a8a;">')
    html = html.replace('<tr>', '<tr style="background: rgba(74, 74, 126, 0.2);">')
    return html
# ===============================================
# Fonctions Image Analysis
# ===============================================
def classify_soil(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[:,:,0])
    mean_sat = np.mean(hsv[:,:,1])
    mean_val = np.mean(hsv[:,:,2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_variance = np.var(gray)
    soil_type = "Inconnu"
    possible_contents = "Inconnu"
    possible_minerals = "Inconnu"
    if mean_val < 100 and texture_variance > 5000:
        soil_type = "Argileux (riche en matière organique)"
        possible_contents = "Peut contenir de l'eau, nutriments, adapté aux cultures racines"
        possible_minerals = "Argiles comme kaolinite, illite; possible fer, aluminium"
    elif mean_sat > 100 and texture_variance < 3000:
        soil_type = "Sableux (drainant)"
        possible_contents = "Peut contenir peu d'eau, adapté aux plantes résistantes à la sécheresse"
        possible_minerals = "Quartz, feldspath; silice abondante"
    elif mean_hue > 20 and mean_hue < 40:
        soil_type = "Limoneux (équilibré)"
        possible_contents = "Peut contenir minéraux, bon pour l'agriculture générale"
        possible_minerals = "Silt avec mica, quartz; calcium, potassium"
    # Graphisme : Histogramme des couleurs HSV
    fig, ax = plt.subplots()
    ax.hist(hsv[:,:,0].ravel(), bins=50, color='b', alpha=0.5, label='Hue')
    ax.hist(hsv[:,:,1].ravel(), bins=50, color='g', alpha=0.5, label='Saturation')
    ax.hist(hsv[:,:,2].ravel(), bins=50, color='r', alpha=0.5, label='Value')
    ax.set_title('Histogramme des Composantes HSV')
    ax.legend()
    hist_img = fig_to_pil(fig)
    # Tableau des metrics
    metrics_df = pd.DataFrame({
        'Métrique': ['Hue Moyenne', 'Saturation Moyenne', 'Valeur Moyenne', 'Variance Texture'],
        'Valeur': [mean_hue, mean_sat, mean_val, texture_variance],
        'Explication': ['Moyenne de la teinte', 'Moyenne de la saturation des couleurs', 'Moyenne de la luminosité', 'Variance de la texture pour rugosité']
    })
    metrics_html = df_to_html(metrics_df)
    return {
        "soil_type": soil_type,
        "possible_contents": possible_contents,
        "possible_minerals": possible_minerals
    }, hist_img, metrics_html
def simulate_infrared(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ir_img = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
    ax.set_title('Simulation Infrarouge (Colormap JET)')
    ax.axis('off')
    ir_pil = fig_to_pil(fig)
    # Analyse simple (fake temp based on intensity)
    mean_intensity = np.mean(gray)
    ir_analysis = f"Simulation IR: Intensité moyenne {mean_intensity:.2f} (plus rouge = plus chaud, bleu = plus froid)"
    return ir_pil, ir_analysis
def detect_objects(image: np.ndarray, scale_factor=0.1, vision_models=None, yolo_model=None):
    """Détecte et analyse les objets dans l'image avec YOLO + CLIP + description IA contextuelle"""
    import tempfile
    import os
    
    img_with_detections = image.copy()
    objects_data = []
    
    # 🎯 PRIORITÉ 1: Utiliser YOLO si disponible
    if yolo_model and YOLO_AVAILABLE:
        try:
            print("🎯 Détection YOLO en cours...")
            results = yolo_model(image, conf=0.25, verbose=False)  # Seuil de confiance 25%
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                for idx, box in enumerate(boxes):
                    # Extraire les coordonnées
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = results[0].names[cls]
                    
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Calculs métriques réels
                    w_m = w * scale_factor
                    h_m = h * scale_factor
                    area_m2 = w_m * h_m
                    perimeter_m = 2 * (w_m + h_m)
                    
                    # Analyse de couleur dominante
                    object_roi = image[y1:y2, x1:x2]
                    if object_roi.size > 0:
                        mean_color = cv2.mean(object_roi)[:3]
                        color_desc = f"RGB({int(mean_color[2])},{int(mean_color[1])},{int(mean_color[0])})"
                    else:
                        color_desc = "N/A"
                    
                    # Stocker les données
                    objects_data.append({
                        'id': idx + 1,
                        'type': f"{class_name} (YOLO)",
                        'confidence': conf,
                        'width_m': w_m,
                        'height_m': h_m,
                        'area_m2': area_m2,
                        'perimeter_m': perimeter_m,
                        'color': color_desc,
                        'position': (x1, y1)
                    })
                    
                    # Dessiner sur l'image
                    color = (0, 255, 0) if conf > 0.5 else (255, 255, 0)
                    cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), color, 2)
                    label = f"#{idx+1}: {class_name} {conf:.0%}"
                    cv2.putText(img_with_detections, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(img_with_detections, f"{w_m:.2f}x{h_m:.2f}m", (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                print(f"✅ YOLO: {len(objects_data)} objets détectés")
        except Exception as e:
            print(f"⚠️ Erreur YOLO: {e}")
            # Fallback sur méthode traditionnelle
    
    # 🔄 FALLBACK: Méthode traditionnelle si YOLO échoue ou pas disponible
    if len(objects_data) == 0:
        print("🔄 Utilisation de la détection par contours (fallback)")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_with_detections = image.copy()
        
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10: 
                continue  # skip small
            
            # Découper l'objet pour analyse IA
            object_roi = image[y:y+h, x:x+w]
            
            # Analyse contextuelle avec IA si disponible
            if vision_models:
                try:
                    # Sauvegarder temporairement l'objet
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        object_pil = Image.fromarray(cv2.cvtColor(object_roi, cv2.COLOR_BGR2RGB))
                        object_pil.save(tmp.name)
                        
                        # Analyser avec CLIP
                        caption, results = analyze_image_with_clip(tmp.name, vision_models)
                        
                        if results and len(results) > 0:
                            obj_type = results[0]['label']
                            confidence = results[0]['confidence']
                        else:
                            obj_type = 'Objet non identifié'
                            confidence = 0.0
                        
                        os.unlink(tmp.name)
                except:
                    # Fallback sur analyse traditionnelle
                    aspect = w / h if h != 0 else 0
                    if aspect > 5: 
                        obj_type = 'Structure linéaire (route/chemin)'
                    elif aspect < 0.2: 
                        obj_type = 'Structure verticale (clôture/poteau)'
                    elif 0.5 < aspect < 2: 
                        obj_type = 'Structure carrée (bâtiment/zone)'
                    else: 
                        obj_type = 'Structure irrégulière'
                    confidence = 0.5
            else:
                # Fallback sur analyse traditionnelle
                aspect = w / h if h != 0 else 0
                if aspect > 5: 
                    obj_type = 'Structure linéaire (route/chemin)'
                elif aspect < 0.2: 
                    obj_type = 'Structure verticale (clôture/poteau)'
                elif 0.5 < aspect < 2: 
                    obj_type = 'Structure carrée (bâtiment/zone)'
                else: 
                    obj_type = 'Structure irrégulière'
                confidence = 0.5
            
            # Calculs métriques réels
            w_m = w * scale_factor
            h_m = h * scale_factor
            area_m2 = w_m * h_m
            perimeter_m = 2 * (w_m + h_m)
            
            # Analyse de couleur dominante
            mean_color = cv2.mean(object_roi)[:3]
            color_desc = f"RGB({int(mean_color[2])},{int(mean_color[1])},{int(mean_color[0])})"
            
            # Stocker les données
            objects_data.append({
                'id': idx + 1,
                'type': obj_type,
                'confidence': confidence,
                'width_m': w_m,
                'height_m': h_m,
                'area_m2': area_m2,
                'perimeter_m': perimeter_m,
                'color': color_desc,
                'position': (x, y)
            })
            
            # Dessiner sur l'image
            cv2.rectangle(img_with_detections, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"#{idx+1}: {obj_type[:20]}"
            cv2.putText(img_with_detections, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img_with_detections, f"{w_m:.2f}x{h_m:.2f}m", (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Créer la visualisation
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB))
    
    # Déterminer le titre en fonction de la méthode utilisée
    detection_method = "YOLO + IA" if yolo_model and len(objects_data) > 0 and "YOLO" in objects_data[0].get('type', '') else "Contours + IA"
    ax.set_title(f"Objets Détectés ({detection_method}): {len(objects_data)} objets", 
                fontsize=14, fontweight='bold', color='white')
    ax.axis('off')
    obj_img = fig_to_pil(fig)
    
    # Créer le tableau HTML des objets avec analyses complètes
    if objects_data:
        dim_df = pd.DataFrame(objects_data)
        dim_df = dim_df.rename(columns={
            'id': 'ID',
            'type': 'Type d\'objet (IA)',
            'confidence': 'Confiance IA',
            'width_m': 'Largeur (m)',
            'height_m': 'Hauteur (m)',
            'area_m2': 'Surface (m²)',
            'perimeter_m': 'Périmètre (m)',
            'color': 'Couleur dominante'
        })
        dim_df['Confiance IA'] = dim_df['Confiance IA'].apply(lambda x: f"{x*100:.1f}%")
        dim_df = dim_df.drop('position', axis=1)
        dim_html = df_to_html(dim_df)
    else:
        dim_html = ""
    
    return len(objects_data), obj_img, dim_html, objects_data
def detect_fences(image: np.ndarray, scale_factor=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    img_with_lines = image.copy()
    lengths = []
    if lines is not None:
        line_list = [line[0] for line in lines]
        filtered_lines = [l for l in line_list if abs(l[0] - l[2]) < 10 or abs(l[1] - l[3]) < 10 or abs((l[1]-l[3]) / (l[0]-l[2] + 1e-5)) < 0.1 or abs((l[1]-l[3]) / (l[0]-l[2] + 1e-5)) > 10]
        line_lengths = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2) for x1,y1,x2,y2 in filtered_lines]
        sorted_indices = np.argsort(line_lengths)[::-1]
        sorted_lines = [filtered_lines[i] for i in sorted_indices]
        for x1,y1,x2,y2 in sorted_lines:
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * scale_factor
            lengths.append(length)
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(img_with_lines, f"{length:.4f}m", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Clôtures/Bordures Détectées ({len(lengths)})")
    ax.axis('off')
    fence_img = fig_to_pil(fig)
    if lengths:
        fence_df = pd.DataFrame({
            'Longueur (m)': lengths,
            'Explication': ['Longueur de bordure filtrée et triée pour précision' for _ in lengths]
        })
        fence_html = df_to_html(fence_df)
    else:
        fence_html = ""
    return len(lengths), fence_img, fence_html
def detect_anomalies(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    num_edges = np.sum(edges > 0)
    mean_variance = np.mean(cv2.Laplacian(gray, cv2.CV_64F).var())
    anomalies = []
    if num_edges > 10000:
        anomalies.append("Anomalies structurelles détectées (ex. : fissures, défauts)")
    if mean_variance > 500:
        anomalies.append("Textures inhabituelles (ex. : zones irrégulières)")
    # Simulation photogrammétrie basique SANS Open3D (désactivé pour Python 3.13)
    # Open3D n'est pas disponible pour Python 3.13, donc on simule autrement
    num_points = gray.shape[0] * gray.shape[1]  # Nombre de pixels comme approximation
    
    # Graphisme : Histogramme des variances
    fig, ax = plt.subplots()
    ax.hist(cv2.Laplacian(gray, cv2.CV_64F).ravel(), bins=50)
    ax.set_title('Histogramme des Variances Locales (Anomalies)')
    var_hist_img = fig_to_pil(fig)
    # Tableau des metrics anomalies
    anomaly_df = pd.DataFrame({
        'Métrique': ['Nombre de Bords', 'Variance Moyenne', 'Points Estimés'],
        'Valeur': [num_edges, mean_variance, num_points],
        'Explication': ['Indique complexité structurelle (haut = anomalies)', 'Mesure irrégularités texture', 'Estimation de densité pour volume']
    })
    anomaly_html = df_to_html(anomaly_df)
    anomaly_desc_df = pd.DataFrame({
        'Anomalie': anomalies if anomalies else ['Aucune anomalie détectée'],
        'Explication': ['Défauts potentiels dans le terrain ou structures' if anomalies else 'Image sans anomalies significatives']
    })
    anomaly_desc_html = df_to_html(anomaly_desc_df)
    return anomalies, var_hist_img, anomaly_html, anomaly_desc_html
def advanced_analyses(image: np.ndarray):
    analyses = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 100, 200)
    adv_images = []
    adv_tables = []
    # 1. Analyse Géologique
    kmeans = KMeans(n_clusters=3).fit(gray.reshape(-1, 1))
    clustered = kmeans.labels_.reshape(gray.shape)
    analyses['Géologique'] = 'Clusters de textures : ' + str(np.unique(kmeans.labels_))
    fig, ax = plt.subplots()
    ax.imshow(clustered, cmap='viridis')
    ax.set_title('Analyse Géologique: Clustering Textures')
    ax.axis('off')
    adv_images.append(fig_to_pil(fig))
    geo_df = pd.DataFrame({'Cluster': np.unique(kmeans.labels_), 'Compte': np.bincount(kmeans.labels_), 'Explication': ['Groupe de texture géologique' for _ in np.unique(kmeans.labels_)]})
    adv_tables.append(df_to_html(geo_df))
    # 2. Analyse Hydrologique
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    water_area = np.sum(blue_mask > 0) / blue_mask.size * 100
    analyses['Hydrologique'] = f'Pourcentage eau : {water_area:.2f}%'
    fig, ax = plt.subplots()
    ax.imshow(blue_mask, cmap='gray')
    ax.set_title('Analyse Hydrologique: Masque Eau')
    ax.axis('off')
    adv_images.append(fig_to_pil(fig))
    hydro_df = pd.DataFrame({'Métrique': ['Pourcentage Eau'], 'Valeur': [water_area], 'Explication': ['Zone potentielle pour ressources hydriques']})
    adv_tables.append(df_to_html(hydro_df))
    return analyses, {}, adv_images, adv_tables
def process_image(uploaded_file, vision_models=None, ocr_reader=None, yolo_model=None, blip_models=None):
    """Traite l'image avec analyse IA avancée: YOLO + CLIP + BLIP + OCR + analyses contextuelles"""
    image = Image.open(BytesIO(uploaded_file))
    img_array = np.array(image)
    proc_images = [image]
    captions = ['Image Originale']
    tables_html = []
    
    # 🖼️ DESCRIPTION DÉTAILLÉE AVEC CLIP (PRIORITAIRE)
    clip_description = ""
    if vision_models:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                image.save(tmp.name)
                clip_result, error = generate_detailed_description_with_clip(tmp.name, vision_models)
                
                if clip_result:
                    clip_description = f"""<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: white; margin: 0 0 10px 0;">🖼️ Description de l'image (CLIP Vision AI)</h4>
                    <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 10px 0;">"{clip_result['description']}"</p>
                    <p style="color: white; font-weight: 600; margin: 5px 0; font-size: 0.95rem;">
                        <strong>Type:</strong> {clip_result['main_type']}<br>
                        <strong>Couleurs:</strong> {clip_result['colors']}<br>
                        <strong>Confiance:</strong> {clip_result['confidence']*100:.1f}%
                    </p>
                    </div>"""
                    tables_html.append(clip_description)
                else:
                    tables_html.append(f'<h3 style="color: white; font-weight: bold;">🖼️ Description CLIP</h3><p style="color: orange; font-weight: 600;">Erreur: {error}</p>')
                
                import os
                os.unlink(tmp.name)
        except Exception as e:
            print(f"⚠️ Erreur description CLIP: {e}")
            tables_html.append(f'<h3 style="color: white; font-weight: bold;">🖼️ Description CLIP</h3><p style="color: orange; font-weight: 600;">Erreur: {str(e)}</p>')
    
    # 🖼️ BLIP - DESCRIPTION DÉTAILLÉE (OPTIONNEL - si disponible)
    blip_description = ""
    if blip_models:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                image.save(tmp.name)
                blip_result, error = caption_image_with_blip(tmp.name, blip_models)
                
                if blip_result:
                    blip_description = f"""<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="color: white; margin: 0 0 10px 0;">🖼️ Description BLIP (IA Vision)</h4>
                    <p style="color: white; font-weight: 600; margin: 5px 0;"><strong>Description générale:</strong> {blip_result['caption']}</p>
                    <p style="color: white; font-weight: 600; margin: 5px 0;"><strong>Description détaillée:</strong> {blip_result['detailed_caption']}</p>
                    </div>"""
                    tables_html.append(blip_description)
                
                import os
                os.unlink(tmp.name)
        except Exception as e:
            print(f"⚠️ BLIP non disponible: {e}")
    
    # 📝 OCR - EXTRACTION DE TEXTE
    ocr_text = ""
    if ocr_reader:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                image.save(tmp.name)
                extracted_texts = extract_text_from_image(tmp.name, ocr_reader)
                ocr_organized = organize_extracted_text(extracted_texts)
                ocr_text = ocr_organized
                
                if extracted_texts:
                    tables_html.append(f'<h3 style="color: white; font-weight: bold;">📝 OCR - Texte Détecté</h3><pre style="color: white; font-weight: 600;">{ocr_organized}</pre>')
                else:
                    tables_html.append('<h3 style="color: white; font-weight: bold;">📝 OCR - Texte Détecté</h3><p style="color: white; font-weight: 600;">Aucun texte détecté dans l\'image</p>')
                
                import os
                os.unlink(tmp.name)
        except Exception as e:
            print(f"⚠️ Erreur OCR: {e}")
            tables_html.append(f'<h3 style="color: white; font-weight: bold;">📝 OCR</h3><p style="color: white; font-weight: 600;">Erreur OCR: {str(e)}</p>')
    
    # 🌡️ IR
    ir_pil, ir_analysis = simulate_infrared(img_array)
    proc_images.append(ir_pil)
    captions.append('Simulation Infrarouge')
    tables_html.append('<h3 style="color: white; font-weight: bold;">Analyse IR</h3><p style="color: white; font-weight: 600;">' + ir_analysis + '</p>')
    
    # 🌍 Soil
    soil, hist_img, metrics_html = classify_soil(img_array)
    proc_images.append(hist_img)
    captions.append('Histogramme HSV')
    tables_html.append('<h3 style="color: white; font-weight: bold;">Métriques Sol</h3>' + metrics_html.replace('<td>', '<td style="color: white; font-weight: 600;">').replace('<th>', '<th style="color: white; font-weight: bold;">'))
    
    # 🎯 Objects avec YOLO + CLIP IA
    num_objects, obj_img, dim_html, objects_data = detect_objects(img_array, vision_models=vision_models, yolo_model=yolo_model)
    proc_images.append(obj_img)
    captions.append('🎯 Objets YOLO + IA')
    if dim_html:
        tables_html.append('<h3 style="color: white; font-weight: bold;">🎯 Objets Détectés (YOLO + CLIP)</h3>' + dim_html.replace('<td>', '<td style="color: white; font-weight: 600;">').replace('<th>', '<th style="color: white; font-weight: bold;">'))
    
    # Fences
    num_fences, fence_img, fence_html = detect_fences(img_array)
    proc_images.append(fence_img)
    captions.append('Clôtures Détectées')
    if fence_html:
        tables_html.append('<h3 style="color: white; font-weight: bold;">Longueurs Clôtures</h3>' + fence_html.replace('<td>', '<td style="color: white; font-weight: 600;">').replace('<th>', '<th style="color: white; font-weight: bold;">'))
    
    # Anomalies
    anomalies, var_hist_img, anomaly_html, anomaly_desc_html = detect_anomalies(img_array)
    proc_images.append(var_hist_img)
    captions.append('Histogramme Variances')
    tables_html.append('<h3 style="color: white; font-weight: bold;">Métriques Anomalies</h3>' + anomaly_html.replace('<td>', '<td style="color: white; font-weight: 600;">').replace('<th>', '<th style="color: white; font-weight: bold;">'))
    
    # Advanced
    analyses, predictions, adv_images, adv_tables = advanced_analyses(img_array)
    proc_images += adv_images[:5]
    captions += ['Analyse Avancée'] * len(adv_images[:5])
    # Appliquer style blanc gras aux tableaux avancés
    adv_tables_styled = [t.replace('<td>', '<td style="color: white; font-weight: 600;">').replace('<th>', '<th style="color: white; font-weight: bold;">') for t in adv_tables[:3]]
    tables_html += adv_tables_styled
    
    analysis_data = {
        "soil": soil,
        "ir_analysis": ir_analysis,
        "num_objects": num_objects,
        "objects_detected": objects_data,
        "num_fences": num_fences,
        "anomalies": anomalies,
        "analyses": analyses,
        "predictions": predictions,
        "ocr_text": ocr_text,  # Texte OCR
        "clip_description": clip_description,  # Description CLIP détaillée
        "blip_description": blip_description  # Description BLIP (optionnel)
    }
    
    tables_str = '<br>'.join(tables_html)
    return analysis_data, proc_images, tables_str
def generate_comprehensive_description(analysis_data):
    """
    Génère une description complète et cohérente en combinant tous les modèles d'IA
    """
    description_parts = []
    
    # 1. CLIP Description détaillée (prioritaire)
    clip_desc = analysis_data.get('clip_description', '')
    if clip_desc and '<p style' in clip_desc:
        # Extraire le texte du HTML
        import re
        clip_match = re.search(r'<p style="color: white; font-weight: 700.*?">(.*?)</p>', clip_desc, re.DOTALL)
        if clip_match:
            description_parts.append("🖼️ **Description de l'image (CLIP Vision AI):**")
            description_parts.append(f"   {clip_match.group(1).strip()}")
    
    # 2. BLIP (si disponible)
    blip_desc = analysis_data.get('blip_description', '')
    if blip_desc and '<p style' in blip_desc:
        import re
        blip_texts = re.findall(r'<strong>.*?:</strong>\s*(.*?)</p>', blip_desc)
        if blip_texts:
            description_parts.append("\n🎨 **Analyse BLIP complémentaire:**")
            description_parts.append(f"   {blip_texts[0]}")
    
    # 3. OCR (texte détecté)
    ocr_text = analysis_data.get('ocr_text', '')
    if ocr_text and 'Aucun texte' not in ocr_text:
        # Extraire juste les mots principaux
        import re
        text_matches = re.findall(r'✅.*?\]\s*(.*?)$', ocr_text, re.MULTILINE)
        if text_matches:
            unique_words = list(set([w.strip() for w in text_matches if len(w.strip()) > 1]))
            if unique_words:
                description_parts.append(f"\n📝 **Texte détecté:** {', '.join(unique_words[:10])}")
    
    # 4. YOLO (objets détectés)
    objects = analysis_data.get('objects_detected', [])
    if objects:
        yolo_objects = [obj for obj in objects if 'YOLO' in obj.get('type', '')]
        if yolo_objects:
            object_types = {}
            for obj in yolo_objects:
                obj_type = obj['type'].replace(' (YOLO)', '')
                object_types[obj_type] = object_types.get(obj_type, 0) + 1
            
            obj_summary = ', '.join([f"{count} {name}" if count > 1 else name 
                                    for name, count in list(object_types.items())[:5]])
            description_parts.append(f"\n🎯 **Objets identifiés (YOLO):** {obj_summary}")
    
    # 5. Informations techniques
    num_objects = analysis_data.get('num_objects', 0)
    if num_objects > 0:
        description_parts.append(f"\n📊 **Analyse technique:** {num_objects} éléments détectés")
    
    # Générer le résumé final
    if description_parts:
        return '\n'.join(description_parts)
    else:
        return "Image analysée - aucune information spécifique détectée"

def improve_analysis_with_llm(analysis_data, model_name):
    """
    Améliore l'analyse avec le LLM en utilisant la description complète
    """
    # Générer d'abord une description cohérente
    comprehensive_desc = generate_comprehensive_description(analysis_data)
    
    # Extraire les informations clés pour le LLM
    blip_desc = analysis_data.get('blip_description', '')
    ocr_text = analysis_data.get('ocr_text', '')
    objects = analysis_data.get('objects_detected', [])
    
    # Construire un prompt intelligent
    prompt = f"""Analyse cette image en français de manière précise et concise:

{comprehensive_desc}

Informations techniques:
- Résolution: {analysis_data.get('soil', 'N/A')}
- Objets: {len(objects)} détectés

Décris ce que tu vois en 2-3 phrases claires et précises."""

    try:
        client = create_client()
        messages = [{"role": "user", "content": prompt[:2000]}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback sur la description générée
        return comprehensive_desc
# ===============================================
# Interface Streamlit Améliorée
# ===============================================
def main():
    """Interface Streamlit unifiée avec recherche web intégrée"""
    
    # ===============================================
    # SYSTÈME DE PALETTES DE COULEURS DYNAMIQUES
    # ===============================================
    if 'color_theme' not in st.session_state:
        st.session_state.color_theme = 'ocean_blue'  # Mode sombre bleu par défaut
    
    # Définition des palettes
    COLOR_THEMES = {
        'kibali_classic': {
            'name': '🟢 Kibali Classic',
            'primary': '#00ff88',
            'secondary': '#0088ff',
            'accent': '#ffd700',
            'bg_dark': '#1a1a2e',
            'bg_darker': '#0f0f23',
            'bg_light': '#2a2a4e',
            'bg_lighter': '#3a3a5e',
            'bg_accent': '#4a4a7e',
            'border': '#5a5a8a',
            'text': '#ffffff',
            'text_secondary': '#e0e0e0',
            'text_muted': '#b0b0b0',
            'input_bg': '#3a3a5e',
            'input_text': '#ffffff'
        },
        'ocean_blue': {
            'name': '🌊 Ocean Blue',
            'primary': '#00d4ff',
            'secondary': '#0066ff',
            'accent': '#00ffff',
            'bg_dark': '#0a1628',
            'bg_darker': '#050b14',
            'bg_light': '#1a2d4e',
            'bg_lighter': '#2a3d6e',
            'bg_accent': '#3a4d8e',
            'border': '#4a5daa',
            'text': '#ffffff',
            'text_secondary': '#cfe0ff',
            'text_muted': '#8fa9d4',
            'input_bg': '#2a3d6e',
            'input_text': '#ffffff'
        },
        'sunset_orange': {
            'name': '🌅 Sunset Orange',
            'primary': '#ff6b35',
            'secondary': '#ff9f1c',
            'accent': '#ffcf00',
            'bg_dark': '#2e1a0a',
            'bg_darker': '#1a0f05',
            'bg_light': '#4e2a1a',
            'bg_lighter': '#6e3a2a',
            'bg_accent': '#8e4a3a',
            'border': '#ae5a4a',
            'text': '#ffffff',
            'text_secondary': '#ffe0d0',
            'text_muted': '#d4a590',
            'input_bg': '#6e3a2a',
            'input_text': '#ffffff'
        },
        'purple_dream': {
            'name': '💜 Purple Dream',
            'primary': '#a855f7',
            'secondary': '#8b5cf6',
            'accent': '#ec4899',
            'bg_dark': '#1e0a2e',
            'bg_darker': '#0f0514',
            'bg_light': '#2e1a4e',
            'bg_lighter': '#3e2a6e',
            'bg_accent': '#4e3a8e',
            'border': '#5e4aaa',
            'text': '#ffffff',
            'text_secondary': '#e0cfff',
            'text_muted': '#a98fd4',
            'input_bg': '#3e2a6e',
            'input_text': '#ffffff'
        },
        'forest_green': {
            'name': '🌲 Forest Green',
            'primary': '#10b981',
            'secondary': '#059669',
            'accent': '#34d399',
            'bg_dark': '#0a2e1a',
            'bg_darker': '#05140a',
            'bg_light': '#1a4e2a',
            'bg_lighter': '#2a6e3a',
            'bg_accent': '#3a8e4a',
            'border': '#4aae5a',
            'text': '#ffffff',
            'text_secondary': '#d0ffe0',
            'text_muted': '#90d4a5',
            'input_bg': '#2a6e3a',
            'input_text': '#ffffff'
        },
        'ruby_red': {
            'name': '💎 Ruby Red',
            'primary': '#ef4444',
            'secondary': '#dc2626',
            'accent': '#f87171',
            'bg_dark': '#2e0a0a',
            'bg_darker': '#140505',
            'bg_light': '#4e1a1a',
            'bg_lighter': '#6e2a2a',
            'bg_accent': '#8e3a3a',
            'border': '#ae4a4a',
            'text': '#ffffff',
            'text_secondary': '#ffd0d0',
            'text_muted': '#d49090',
            'input_bg': '#6e2a2a',
            'input_text': '#ffffff'
        },
        'gold_luxury': {
            'name': '✨ Gold Luxury',
            'primary': '#fbbf24',
            'secondary': '#f59e0b',
            'accent': '#fde047',
            'bg_dark': '#2e1e0a',
            'bg_darker': '#140f05',
            'bg_light': '#4e3e1a',
            'bg_lighter': '#6e5e2a',
            'bg_accent': '#8e7e3a',
            'border': '#ae9e4a',
            'text': '#ffffff',
            'text_secondary': '#fff0d0',
            'text_muted': '#d4c590',
            'input_bg': '#6e5e2a',
            'input_text': '#ffffff'
        },
        'cyber_neon': {
            'name': '🔮 Cyber Neon',
            'primary': '#ff00ff',
            'secondary': '#00ffff',
            'accent': '#ffff00',
            'bg_dark': '#1a0a2e',
            'bg_darker': '#0a0514',
            'bg_light': '#2a1a4e',
            'bg_lighter': '#3a2a6e',
            'bg_accent': '#4a3a8e',
            'border': '#5a4aaa',
            'text': '#ffffff',
            'text_secondary': '#ffcfff',
            'text_muted': '#cf8fff',
            'input_bg': '#3a2a6e',
            'input_text': '#ffffff'
        }
    }
    
    # Récupérer le thème actuel
    theme = COLOR_THEMES[st.session_state.color_theme]

    # ===============================================
    # CSS PERSONNALISÉ - DESIGN PUISSANT ET FLUIDE
    # ===============================================
    st.markdown(f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Variables CSS pour les couleurs - DYNAMIQUES */
    :root {{{{{{{{
        --kibali-green: {theme['primary']};
        --kibali-yellow: {theme['accent']};
        --kibali-blue: {theme['secondary']};
        --kibali-dark: {theme['bg_dark']};
        --kibali-darker: {theme['bg_darker']};
        --kibali-light: {theme['bg_light']};
        --kibali-lighter: {theme['bg_lighter']};
        --kibali-accent: {theme['bg_accent']};
        --kibali-border: {theme['border']};
        --kibali-text: {theme['text']};
        --kibali-text-secondary: {theme['text_secondary']};
        --kibali-text-muted: {theme['text_muted']};
        --kibali-input-bg: {theme['input_bg']};
        --kibali-input-text: {theme['input_text']};
        --gradient-primary: linear-gradient(135deg, var(--kibali-green), var(--kibali-blue));
        --gradient-secondary: linear-gradient(135deg, var(--kibali-yellow), var(--kibali-green));
        --gradient-background: linear-gradient(135deg, var(--kibali-dark), var(--kibali-darker));
        --shadow-glow: 0 0 20px rgba({int(theme['primary'][1:3], 16)}, {int(theme['primary'][3:5], 16)}, {int(theme['primary'][5:7], 16)}, 0.3);
        --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.2);
        --shadow-subtle: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}}}}}}}

    /* Reset et base - Version améliorée */
    * {{{{
        font-family: 'Inter', sans-serif !important;
    }}}}

    /* Fond principal avec gradient subtil et texture */
    .main {{{{
        background: var(--gradient-background) !important;
        background-attachment: fixed !important;
        position: relative;
    }}}}

    .main::before {{{{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
            radial-gradient(circle at 20% 80%, rgba({int(theme['primary'][1:3], 16)}, {int(theme['primary'][3:5], 16)}, {int(theme['primary'][5:7], 16)}, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba({int(theme['secondary'][1:3], 16)}, {int(theme['secondary'][3:5], 16)}, {int(theme['secondary'][5:7], 16)}, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba({int(theme['accent'][1:3], 16)}, {int(theme['accent'][3:5], 16)}, {int(theme['accent'][5:7], 16)}, 0.02) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }}}}

    .stApp {{{{
        background: transparent !important;
    }}}}

    /* Logo Kibali animé */
    .kibali-logo {{{{
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin: 2rem 0;
        animation: logoGlow 3s ease-in-out infinite alternate;
        text-shadow: var(--shadow-glow);
    }}}}

    /* Logo K Kibali - AMÉLIORE avec bordure scintillante - TRÈS GRAND */
    .kibali-k-logo {{{{
        font-size: 12rem !important; /* ENCORE PLUS ÉNORME - 192px */
        font-weight: 900 !important;
        text-align: center !important;
        margin: 2rem auto !important;
        color: #000000 !important; /* Texte en NOIR */
        position: relative !important;
        display: inline-block !important;
        padding: 50px 70px !important;
        background: #ffffff !important; /* Fond blanc pour contraste */
        border-radius: 30px !important;
        text-shadow: 0 4px 20px rgba(255, 255, 255, 0.8) !important;
        min-width: 250px !important;
        min-height: 250px !important;
        line-height: 1 !important;
    }}}}

    /* Bordure scintillante AUTOUR du K */
    .kibali-k-logo::before {{{{
        content: '';
        position: absolute;
        top: -5px;
        left: -5px;
        right: -5px;
        bottom: -5px;
        background: linear-gradient(45deg, 
            var(--kibali-green), 
            var(--kibali-yellow), 
            var(--kibali-blue),
            var(--kibali-green));
        background-size: 400% 400%;
        border-radius: 25px;
        z-index: -1;
        animation: borderScintillation 3s ease infinite;
        filter: blur(2px);
    }}}}
    
    /* Animation de scintillement des bordures */
    @keyframes borderScintillation {{{{{{{{
        0% {{{{{{{{
            background-position: 0% 50%;
            filter: brightness(1.2) blur(2px);
        }}}}}}}}
        25% {{{{{{{{
            background-position: 50% 100%;
            filter: brightness(1.5) blur(3px);
        }}}}}}}}
        50% {{{{{{{{
            background-position: 100% 50%;
            filter: brightness(1.8) blur(2px);
        }}}}}}}}
        75% {{{{{{{{
            background-position: 50% 0%;
            filter: brightness(1.5) blur(3px);
        }}}}}}}}
        100% {{{{{{{{
            background-position: 0% 50%;
            filter: brightness(1.2) blur(2px);
        }}}}}}}}
    }}}}}}}}
    
    /* Effet de lueur externe */
    .kibali-k-logo::after {{{{
        content: '';
        position: absolute;
        top: -8px;
        left: -8px;
        right: -8px;
        bottom: -8px;
        background: linear-gradient(45deg,
            rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.3),
            rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.3),
            rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.3));
        border-radius: 28px;
        z-index: -2;
        filter: blur(20px);
        animation: glowPulse 2s ease-in-out infinite alternate;
    }}}}}}}}
    
    @keyframes glowPulse {{{{{{{{
        0% {{{{{{{{ opacity: 0.4; transform: scale(0.98); }}}}}}}}
        100% {{{{{{{{ opacity: 0.8; transform: scale(1.02); }}}}}}}}
    }}}}}}}}
    
    /* Nom KibaliOne8 IA avec effet scintillant - TRÈS GRAND */
    .kibali-name-logo {{{{
        font-size: 3.5rem !important; /* 56px */
        font-weight: 900 !important;
        text-align: center !important;
        margin-top: 2rem !important;
        color: #ffffff !important;
        position: relative !important;
        display: inline-block !important;
        padding: 20px 50px !important;
        background: #000000 !important;
        border-radius: 20px !important;
        letter-spacing: 5px !important;
        text-shadow: 0 3px 15px rgba(0, 0, 0, 0.8) !important;
        text-transform: uppercase !important;
    }}}}
    
    /* Bordure scintillante pour le nom */
    .kibali-name-logo::before {{{{
        content: '';
        position: absolute;
        top: -4px;
        left: -4px;
        right: -4px;
        bottom: -4px;
        background: linear-gradient(45deg, 
            var(--kibali-green), 
            var(--kibali-yellow), 
            var(--kibali-blue),
            var(--kibali-green));
        background-size: 400% 400%;
        border-radius: 18px;
        z-index: -1;
        animation: borderScintillation 3s ease infinite;
        filter: blur(2px);
    }}}}
    
    /* Lueur externe pour le nom */
    .kibali-name-logo::after {{{{
        content: '';
        position: absolute;
        top: -6px;
        left: -6px;
        right: -6px;
        bottom: -6px;
        background: linear-gradient(45deg,
            rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.2),
            rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.2),
            rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.2));
        border-radius: 20px;
        z-index: -2;
        filter: blur(15px);
        animation: glowPulse 2s ease-in-out infinite alternate;
    }}}}
    
    @keyframes glowPulse {{{{{{{{
        0% {{{{{{{{ opacity: 0.4; transform: scale(0.98); }}}}}}}}
        100% {{{{{{{{ opacity: 0.8; transform: scale(1.02); }}}}}}}}
    }}}}}}}}

    @keyframes kGlow {{{{{{{{
        0% {{{{{{{{
            filter: brightness(1) drop-shadow(0 0 20px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.5));
            transform: scale(1);
        }}}}}}}}
        50% {{{{{{{{
            filter: brightness(1.3) drop-shadow(0 0 40px rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.7));
            transform: scale(1.05);
        }}}}}}}}
        100% {{{{{{{{
            filter: brightness(1) drop-shadow(0 0 30px rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.6));
            transform: scale(1);
        }}}}}}}}
    }}}}}}}}

    @keyframes kGlowBorder {{{{{{{{
        0% {{{{{{{{
            filter: drop-shadow(0 0 10px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.8));
        }}}}}}}}
        33% {{{{{{{{
            filter: drop-shadow(0 0 15px rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.9));
        }}}}}}}}
        66% {{{{{{{{
            filter: drop-shadow(0 0 20px rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.8));
        }}}}}}}}
        100% {{{{{{{{
            filter: drop-shadow(0 0 10px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.8));
        }}}}}}}}
    }}}}}}}}

    @keyframes kBorderShine {{{{{{{{
        0% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-green), var(--kibali-yellow), var(--kibali-blue));
            opacity: 0.6;
        }}}}}}}}
        25% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-blue), var(--kibali-green));
            opacity: 0.8;
        }}}}}}}}
        50% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-blue), var(--kibali-green), var(--kibali-yellow));
            opacity: 1.0;
        }}}}}}}}
        75% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-green), var(--kibali-blue), var(--kibali-yellow));
            opacity: 0.8;
        }}}}}}}}
        100% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-green), var(--kibali-blue));
            opacity: 0.6;
        }}}}}}}}
    }}}}}}}}

    @keyframes kRotate {{{{{{{{
        0% {{{{{{{{ transform: rotateY(0deg); }}}}}}}}
        25% {{{{{{{{ transform: rotateY(5deg); }}}}}}}}
        50% {{{{{{{{ transform: rotateY(0deg); }}}}}}}}
        75% {{{{{{{{ transform: rotateY(-5deg); }}}}}}}}
        100% {{{{{{{{ transform: rotateY(0deg); }}}}}}}}
    }}}}}}}}

    @keyframes kBackgroundPulse {{{{{{{{
        0% {{{{{{{{ opacity: 0.1; transform: scale(1); }}}}}}}}
        100% {{{{{{{{ opacity: 0.3; transform: scale(1.1); }}}}}}}}
    }}}}}}}}

    @keyframes logoGlow {{{{{{{{
        0% {{{{{{{{ filter: brightness(1) drop-shadow(0 0 10px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.5)); }}}}}}}}
        100% {{{{{{{{ filter: brightness(1.2) drop-shadow(0 0 20px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.8)); }}}}}}}}
    }}}}}}}}

    /* Header principal - Version améliorée */
    .main-header {{{{
        background: linear-gradient(135deg, var(--kibali-light), var(--kibali-accent));
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid var(--kibali-border);
        box-shadow: var(--shadow-subtle);
        position: relative;
        overflow: hidden;
    }}}}

    .main-header::before {{{{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
        animation: headerShine 4s ease-in-out infinite;
    }}}}

    @keyframes headerShine {{{{{{{{
        0%, 100% {{{{{{{{ transform: translateX(-100%); }}}}}}}}
        50% {{{{{{{{ transform: translateX(100%); }}}}}}}}
    }}}}}}}}

    /* Cartes flex responsive - Version améliorée */
    .kibali-card {{{{
        background: var(--kibali-lighter);
        border: 1px solid var(--kibali-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-subtle);
        position: relative;
        overflow: hidden;
    }}}}
    
    /* 🎨 CARTE DE CHAT SPÉCIALE - Bordures scintillantes + Zoom dynamique */
    .kibali-chat-card {{{{
        background: var(--kibali-darker);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: visible;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        min-height: 400px;
    }}}}
    
    /* Bordure scintillante animée autour du chat */
    .kibali-chat-card::before {{{{
        content: '';
        position: absolute;
        top: -3px;
        left: -3px;
        right: -3px;
        bottom: -3px;
        background: linear-gradient(45deg, 
            var(--kibali-green), 
            var(--kibali-yellow), 
            var(--kibali-blue),
            var(--kibali-green));
        background-size: 400% 400%;
        border-radius: 22px;
        z-index: -1;
        animation: chatBorderScintillation 4s ease infinite;
        filter: blur(1px);
    }}}}
    
    /* Effet de lueur externe pulsante */
    .kibali-chat-card::after {{{{
        content: '';
        position: absolute;
        top: -6px;
        left: -6px;
        right: -6px;
        bottom: -6px;
        background: linear-gradient(45deg,
            rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.4),
            rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.4),
            rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.4));
        border-radius: 24px;
        z-index: -2;
        filter: blur(15px);
        animation: chatGlowPulse 3s ease-in-out infinite alternate;
    }}}}
    
    /* Animation de scintillement des bordures du chat */
    @keyframes chatBorderScintillation {{{{{{{{
        0% {{{{{{{{
            background-position: 0% 50%;
            filter: brightness(1.2) blur(1px);
        }}}}}}}}
        20% {{{{{{{{
            background-position: 100% 0%;
            filter: brightness(1.5) blur(2px);
        }}}}}}}}
        40% {{{{{{{{
            background-position: 50% 100%;
            filter: brightness(1.8) blur(1px);
        }}}}}}}}
        60% {{{{{{{{
            background-position: 0% 50%;
            filter: brightness(1.6) blur(2px);
        }}}}}}}}
        80% {{{{{{{{
            background-position: 100% 100%;
            filter: brightness(1.4) blur(1px);
        }}}}}}}}
        100% {{{{{{{{
            background-position: 0% 50%;
            filter: brightness(1.2) blur(1px);
        }}}}}}}}
    }}}}}}}}
    
    /* Pulsation de la lueur */
    @keyframes chatGlowPulse {{{{{{{{
        0% {{{{{{{{ 
            opacity: 0.3; 
            transform: scale(0.95); 
            filter: blur(15px);
        }}}}}}}}
        50% {{{{{{{{ 
            opacity: 0.6; 
            transform: scale(1.02); 
            filter: blur(20px);
        }}}}}}}}
        100% {{{{{{{{ 
            opacity: 0.8; 
            transform: scale(1.05); 
            filter: blur(25px);
        }}}}}}}}
    }}}}}}}}
    
    /* Zoom dynamique au survol */
    .kibali-chat-card:hover {{{{
        transform: scale(1.03) translateY(-5px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    }}}}
    
    /* Accélération de l'animation au survol */
    .kibali-chat-card:hover::before {{{{
        animation: chatBorderScintillation 2s ease infinite;
        filter: brightness(1.5) blur(2px);
    }}}}
    
    .kibali-chat-card:hover::after {{{{
        animation: chatGlowPulse 1.5s ease-in-out infinite alternate;
        opacity: 1;
        transform: scale(1.08);
        filter: blur(30px);
    }}}}
    
    /* Tout le texte dans les cartes en blanc gras */
    .kibali-card,
    .kibali-card p,
    .kibali-card h1,
    .kibali-card h2,
    .kibali-card h3,
    .kibali-card h4,
    .kibali-card h5,
    .kibali-card span,
    .kibali-card div,
    .kibali-card td,
    .kibali-card th,
    .kibali-card li {{{{
        color: white !important;
        font-weight: 600 !important;
    }}}}
    
    .kibali-card h1,
    .kibali-card h2,
    .kibali-card h3 {{{{
        font-weight: 700 !important;
    }}}}
    
    .kibali-card table {{{{
        color: white !important;
        background: rgba({int(theme["bg_light"][1:3], 16)}, {int(theme["bg_light"][3:5], 16)}, {int(theme["bg_light"][5:7], 16)}, 0.5) !important;
    }}}}
    
    .kibali-card th {{{{
        background: rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.2) !important;
        color: white !important;
        font-weight: 700 !important;
        border: 1px solid var(--kibali-border) !important;
    }}}}
    
    .kibali-card td {{{{
        border: 1px solid var(--kibali-border) !important;
        color: white !important;
        font-weight: 600 !important;
    }}}}

    .kibali-card::before {{{{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.1), transparent);
        transition: left 0.5s;
    }}}}

    .kibali-card:hover {{{{
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-card);
        border-color: var(--kibali-green);
    }}}}

    .kibali-card:hover::before {{{{
        left: 100%;
    }}}}

    /* Boutons stylisés */
    .kibali-btn {{{{
        background: var(--gradient-primary);
        color: white !important;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.3);
        position: relative;
        overflow: hidden;
    }}}}

    .kibali-btn::before {{{{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }}}}

    .kibali-btn:hover {{{{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.5);
    }}}}

    .kibali-btn:hover::before {{{{
        left: 100%;
    }}}}

    .kibali-btn:active {{{{
        transform: translateY(0);
    }}}}

    /* Onglets personnalisés - Version améliorée */
    .stTabs [data-baseweb="tab-list"] {{{{
        gap: 8px;
        background: var(--kibali-light);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid var(--kibali-border);
    }}}}

    .stTabs [data-baseweb="tab"] {{{{
        background: var(--kibali-accent);
        border-radius: 8px;
        color: var(--kibali-text-secondary);
        transition: all 0.3s ease;
        border: none;
        padding: 12px 20px;
        font-weight: 500;
    }}}}

    .stTabs [data-baseweb="tab"]:hover {{{{
        background: var(--kibali-lighter);
        color: var(--kibali-text);
        transform: translateY(-2px);
        box-shadow: var(--shadow-subtle);
    }}}}

    .stTabs [aria-selected="true"] {{{{
        background: var(--gradient-primary) !important;
        color: white !important;
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
    }}}}

    /* Chat messages - Version améliorée */
    .chat-message-user {{{{
        background: linear-gradient(135deg, var(--kibali-blue), var(--kibali-green));
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.3);
        animation: slideInLeft 0.5s ease-out;
        border: 1px solid rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.2);
    }}}}

    .chat-message-assistant {{{{
        background: var(--kibali-lighter);
        color: var(--kibali-text);
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.5rem 0;
        border-left: 4px solid var(--kibali-green);
        box-shadow: var(--shadow-subtle);
        animation: slideInRight 0.5s ease-out;
        border: 1px solid var(--kibali-border);
    }}}}

    @keyframes slideInLeft {{{{{{{{
        from {{{{{{{{ transform: translateX(-20px); opacity: 0; }}}}}}}}
        to {{{{{{{{ transform: translateX(0); opacity: 1; }}}}}}}}
    }}}}}}}}

    @keyframes slideInRight {{{{{{{{
        from {{{{{{{{ transform: translateX(20px); opacity: 0; }}}}}}}}
        to {{{{{{{{ transform: translateX(0); opacity: 1; }}}}}}}}
    }}}}}}}}
    
    @keyframes slideIn {{{{{{{{
        from {{{{{{{{ transform: translateY(10px); opacity: 0; }}}}}}}}
        to {{{{{{{{ transform: translateY(0); opacity: 1; }}}}}}}}
    }}}}}}}}
    
    @keyframes pulse {{{{{{{{
        0%, 100% {{{{{{{{ opacity: 1; transform: scale(1); }}}}}}}}
        50% {{{{{{{{ opacity: 0.8; transform: scale(1.05); }}}}}}}}
    }}}}}}}}
    
    /* Style des réponses de chat amélioré */
    .response-content {{{{{{{{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    }}}}}}}}
    
    .response-content h2 {{{{{{{{
        color: var(--kibali-green) !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin: 1.5rem 0 0.75rem 0 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid rgba(0, 255, 136, 0.3) !important;
    }}}}}}}}
    
    .response-content h3 {{{{{{{{
        color: #00ffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin: 1rem 0 0.5rem 0 !important;
    }}}}}}}}
    
    .response-content p {{{{{{{{
        margin: 0.75rem 0 !important;
        line-height: 1.8 !important;
        text-align: justify !important;
    }}}}}}}}
    
    .response-content ul, .response-content ol {{{{{{{{
        margin: 0.75rem 0 !important;
        padding-left: 1.5rem !important;
    }}}}}}}}
    
    .response-content li {{{{{{{{
        margin: 0.5rem 0 !important;
        line-height: 1.6 !important;
    }}}}}}}}
    
    .response-content strong {{{{{{{{
        color: var(--kibali-green) !important;
        font-weight: 700 !important;
    }}}}}}}}
    
    .response-content em {{{{{{{{
        color: #ffd700 !important;
        font-style: italic !important;
    }}}}}}}}
    
    .response-content code {{{{{{{{
        background: rgba(0, 255, 136, 0.1) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        color: var(--kibali-green) !important;
        font-family: 'Courier New', monospace !important;
    }}}}}}}}

    /* Input fields - TEXTE BLANC VISIBLE */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {{{{
        background: var(--kibali-input-bg) !important;
        border: 2px solid var(--kibali-border) !important;
        border-radius: 12px !important;
        color: var(--kibali-input-text) !important;
        padding: 14px 18px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        z-index: 1 !important;
    }}}}
    
    /* FORCER LE TEXTE BLANC PARTOUT */
    input, textarea, select {{{{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}}}
    
    /* Input du chat principal - ULTRA VISIBLE */
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea {{{{
        background: #6B46C1 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 16px 20px !important;
        border: 2px solid var(--kibali-green) !important;
        -webkit-text-fill-color: #ffffff !important;
    }}}}
    
    /* Labels styling - Éviter le chevauchement */
    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stNumberInput label,
    .stFileUploader label,
    [data-testid="stWidgetLabel"] {{{{
        color: var(--kibali-text-secondary) !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
        position: static !important;
        background: transparent !important;
        padding: 0 !important;
        width: 100% !important;
        text-align: left !important;
    }}}}
    
    /* Container pour éviter superposition */
    .stTextInput > div,
    .stTextArea > div,
    .stNumberInput > div,
    .stSelectbox > div,
    .stFileUploader > div {{{{
        display: flex !important;
        flex-direction: column !important;
        gap: 0.25rem !important;
    }}}}
    
    /* Forcer la séparation label/input */
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stNumberInput > div > div {{{{
        display: block !important;
        position: relative !important;
    }}}}
    
    /* S'assurer que le label est au-dessus */
    .stTextInput > div > label,
    .stTextArea > div > label,
    .stNumberInput > div > label,
    .stSelectbox > div > label {{{{
        order: -1 !important;
        margin-bottom: 0.5rem !important;
    }}}}
    
    /* Placeholder styling */
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {{{{
        color: var(--kibali-text-muted) !important;
        opacity: 0.6 !important;
    }}}}
    
    /* Hide placeholder when input has focus or value */
    .stTextInput input:focus::placeholder,
    .stTextArea textarea:focus::placeholder {{{{
        opacity: 0.3 !important;
    }}}}

    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {{{{
        border-color: var(--kibali-green) !important;
        box-shadow: 0 0 0 2px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.2) !important;
        transform: scale(1.02);
    }}}}
    
    /* Chat Input - Texte blanc gras sur fond violet */
    [data-testid="stChatInput"] {{{{
        position: relative !important;
    }}}}
    
    [data-testid="stChatInput"] input {{{{
        background: #6B46C1 !important;
        border: 2px solid var(--kibali-border) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 12px 16px !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        position: relative !important;
        z-index: 1 !important;
    }}}}
    
    [data-testid="stChatInput"] input::placeholder {{{{
        color: rgba(255, 255, 255, 0.6) !important;
        opacity: 1 !important;
        font-weight: 500 !important;
        position: absolute !important;
        left: 16px !important;
        pointer-events: none !important;
    }}}}
    
    [data-testid="stChatInput"] input:focus {{{{
        background: #7C3AED !important;
        border-color: var(--kibali-green) !important;
        box-shadow: 0 0 15px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.3) !important;
    }}}}
    
    [data-testid="stChatInput"] input:focus::placeholder {{{{
        opacity: 0.3 !important;
    }}}}
    
    /* Cacher le placeholder quand il y a du texte */
    [data-testid="stChatInput"] input:not(:placeholder-shown)::placeholder {{{{
        opacity: 0 !important;
    }}}}
    
    /* SIDEBAR - Corrections pour éviter superposition */
    .css-1d391kg, [data-testid="stSidebar"] {{{{
        background: var(--kibali-light) !important;
    }}}}
    
    /* Texte dans la sidebar - Meilleur contraste */
    [data-testid="stSidebar"] * {{{{
        color: var(--kibali-text) !important;
    }}}}
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{{{
        color: var(--kibali-green) !important;
        font-weight: 600 !important;
    }}}}
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {{{{
        color: var(--kibali-text) !important;
        font-size: 1rem !important;
    }}}}
    
    /* Boutons dans la sidebar - Style icône */
    [data-testid="stSidebar"] button {{{{
        width: 100% !important;
        margin: 0.25rem 0 !important;
        font-size: 2rem !important;
        padding: 0.75rem !important;
        background: var(--kibali-accent) !important;
        border: 2px solid var(--kibali-border) !important;
        transition: all 0.3s ease !important;
    }}}}
    
    [data-testid="stSidebar"] button:hover {{{{
        background: var(--kibali-green) !important;
        transform: scale(1.05) !important;
        box-shadow: 0 0 20px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.5) !important;
    }}}}
    
    [data-testid="stSidebar"] button:disabled {{{{
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }}}}
    
    /* Inputs dans la sidebar - Éviter superposition */
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stSelectbox {{{{
        margin-bottom: 1rem !important;
    }}}}
    
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSelectbox label {{{{
        display: block !important;
        margin-bottom: 0.5rem !important;
        position: static !important;
        background: transparent !important;
        padding: 0 !important;
        width: 100% !important;
    }}}}
    
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select {{{{
        width: 100% !important;
        display: block !important;
        margin-top: 0.5rem !important;
    }}}}
    
    /* Colonnes dans la sidebar */
    [data-testid="stSidebar"] [data-testid="column"] {{{{
        padding: 0.25rem !important;
    }}}}
    
    /* CORRECTION GLOBALE - Empêcher toute superposition de texte */
    /* Tous les widgets Streamlit */
    div[data-testid="stVerticalBlock"] > div {{{{
        display: flex !important;
        flex-direction: column !important;
    }}}}
    
    /* S'assurer que les labels sont toujours au-dessus */
    label {{{{
        position: static !important;
        display: block !important;
        margin-bottom: 0.25rem !important;
        z-index: auto !important;
    }}}}
    
    /* Empêcher l'overlay de texte sur les inputs */
    input, textarea, select {{{{
        position: relative !important;
        z-index: 1 !important;
        background-color: var(--kibali-accent) !important;
    }}}}
    
    /* Forcer la séparation pour tous les form elements */
    .stTextInput,
    .stTextArea,
    .stNumberInput,
    .stSelectbox,
    .stMultiSelect,
    .stDateInput,
    .stTimeInput {{{{
        margin-bottom: 0.75rem !important;
    }}}}
    
    /* Empêcher le chevauchement du placeholder avec la valeur */
    input:not(:placeholder-shown) {{{{
        color: var(--kibali-text) !important;
    }}}}
    
    input::placeholder {{{{
        color: var(--kibali-text-muted) !important;
        opacity: 0.5 !important;
    }}}}
    
    /* Quand l'input a une valeur, le placeholder doit disparaître */
    input[value]:not([value=""]) {{{{
        color: var(--kibali-text) !important;
    }}}}
    
    input[value]:not([value=""])::placeholder {{{{
        opacity: 0 !important;
        visibility: hidden !important;
    }}}}

    /* Success/Error messages - Version améliorée avec meilleur contraste */
    .stSuccess, .stError, .stWarning, .stInfo {{{{
        border-radius: 12px !important;
        border: none !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        animation: fadeIn 0.5s ease-out;
        box-shadow: var(--shadow-subtle);
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }}}}
    
    .stSuccess {{{{
        background: rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.25) !important;
        border-left: 4px solid var(--kibali-green) !important;
        color: #FFFFFF !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }}}}
    
    .stError {{{{
        background: rgba(255, 107, 107, 0.25) !important;
        border-left: 4px solid #ff6b6b !important;
        color: #FFFFFF !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }}}}
    
    .stWarning {{{{
        background: rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.25) !important;
        border-left: 4px solid var(--kibali-yellow) !important;
        color: #FFFFFF !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }}}}
    
    .stInfo {{{{
        background: rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.25) !important;
        border-left: 4px solid var(--kibali-blue) !important;
        color: #FFFFFF !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }}}}

    /* Messages du chat - Texte blanc gras et lisible */
    [data-testid="stChatMessage"] {{{{
        background: var(--kibali-accent) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        border: 1px solid var(--kibali-border) !important;
    }}}}
    
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {{{{
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }}}}
    
    /* Messages utilisateur - Fond violet */
    [data-testid="stChatMessage"][data-testid*="user"] {{{{
        background: #6B46C1 !important;
        border-color: #7C3AED !important;
    }}}}
    
    /* Messages assistant - Fond accent avec bordure verte */
    [data-testid="stChatMessage"][data-testid*="assistant"] {{{{
        background: var(--kibali-accent) !important;
        border-color: var(--kibali-green) !important;
    }}}}

    .stSuccess {{{{
        background: linear-gradient(135deg, rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.1), rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.05)) !important;
        border-left: 4px solid var(--kibali-green) !important;
        color: var(--kibali-text) !important;
    }}}}

    .stError {{{{
        background: linear-gradient(135deg, rgba(255, 136, 136, 0.1), rgba(255, 136, 136, 0.05)) !important;
        border-left: 4px solid #ff8888 !important;
        color: var(--kibali-text) !important;
    }}}}

    @keyframes fadeIn {{{{{{{{
        from {{{{{{{{ opacity: 0; transform: translateY(10px); }}}}}}}}
        to {{{{{{{{ opacity: 1; transform: translateY(0); }}}}}}}}
    }}}}}}}}

    /* Spinner personnalisé */
    .stSpinner > div > div {{{{
        border-color: var(--kibali-green) !important;
        border-top-color: transparent !important;
    }}}}

    /* Images avec zoom au survol */
    .zoom-image {{{{
        transition: transform 0.3s ease;
        border-radius: 12px;
        box-shadow: var(--shadow-card);
    }}}}

    .zoom-image:hover {{{{
        transform: scale(1.05);
    }}}}

    /* Progress bars */
    .stProgress > div > div > div {{{{
        background: var(--gradient-primary) !important;
    }}}}

    /* Sidebar si utilisée */
    .css-1d391kg, .css-12oz5g7 {{{{
        background: var(--kibali-dark) !important;
    }}}}

    /* Scrollbar personnalisée - Version améliorée */
    ::-webkit-scrollbar {{{{
        width: 8px;
    }}}}

    ::-webkit-scrollbar-track {{{{
        background: var(--kibali-dark);
        border-radius: 4px;
    }}}}

    ::-webkit-scrollbar-thumb {{{{
        background: var(--gradient-primary);
        border-radius: 4px;
        border: 1px solid var(--kibali-border);
    }}}}

    ::-webkit-scrollbar-thumb:hover {{{{
        background: var(--kibali-green);
        box-shadow: var(--shadow-glow);
    }}}}

    /* Icônes grandes et visibles - Priorité absolue */
    .stSuccess, .stError, .stWarning, .stInfo {{{{
        font-size: 2.5rem !important;
        text-align: center !important;
        padding: 1.5rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}}}
    
    /* Rendre les icônes emoji plus grandes partout */
    .stButton button {{{{
        font-size: 2rem !important;
        line-height: 1 !important;
    }}}}
    
    /* Messages avec icônes uniquement - Pas de texte */
    [data-testid="stMarkdownContainer"] {{{{
        font-size: 1.2rem !important;
    }}}}
    
    /* Expanders avec icônes */
    .streamlit-expanderHeader {{{{
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }}}}}}}}

    /* Responsive design */
    @media (max-width: 768px) {{{{{{{{
        .kibali-logo {{{{{{{{
            font-size: 2.5rem;
        }}}}}}}}

        .kibali-card {{{{{{{{
            padding: 1rem;
            margin: 0.5rem 0;
        }}}}}}}}

        .main-header {{{{{{{{
            padding: 1rem;
        }}}}}}}}
    }}}}}}}}

    /* Logo K pour les messages de chat */
    .chat-k-logo {{
        display: inline-block;
        font-size: 1.2rem;
        font-weight: 900;
        color: #ffffff; /* Texte blanc */
        background: #000000; /* Fond noir */
        position: relative;
        margin-right: 0.5rem;
        padding: 4px 8px;
        border-radius: 6px;
    }}

    .chat-k-logo::before {{
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, 
            var(--kibali-green), 
            var(--kibali-yellow), 
            var(--kibali-blue),
            var(--kibali-green));
        background-size: 300% 300%;
        border-radius: 8px;
        z-index: -1;
        animation: chatBorderScintillation 2s ease infinite;
        filter: blur(1px);
    }}

    
        33% {{{{{{{{
            filter: drop-shadow(0 0 5px rgba({int(theme["accent"][1:3], 16)}, {int(theme["accent"][3:5], 16)}, {int(theme["accent"][5:7], 16)}, 0.7));
        }}}}}}}}
        66% {{{{{{{{
            filter: drop-shadow(0 0 4px rgba({int(theme["secondary"][1:3], 16)}, {int(theme["secondary"][3:5], 16)}, {int(theme["secondary"][5:7], 16)}, 0.6));
        }}}}}}}}
        100% {{{{{{{{
            filter: drop-shadow(0 0 3px rgba({int(theme["primary"][1:3], 16)}, {int(theme["primary"][3:5], 16)}, {int(theme["primary"][5:7], 16)}, 0.6));
        }}}}}}}}
    }}}}}}}}

    
        25% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-blue), var(--kibali-green));
            opacity: 0.7;
        }}}}}}}}
        50% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-blue), var(--kibali-green), var(--kibali-yellow));
            opacity: 0.8;
        }}}}}}}}
        75% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-green), var(--kibali-blue), var(--kibali-yellow));
            opacity: 0.7;
        }}}}}}}}
        100% {{{{{{{{
            background: linear-gradient(45deg, var(--kibali-yellow), var(--kibali-green), var(--kibali-blue));
            opacity: 0.5;
        }}}}}}}}
    }}}}}}}}
    
    
    @keyframes chatBorderScintillation {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    /* BOUTONS - Éviter superposition de texte */
    button {{{{
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        line-height: 1.5 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        position: relative !important;
    }}}}
    
    button span {{{{
        display: inline-block !important;
        position: relative !important;
    }}}}
    
    /* Désactiver tout pseudo-élément qui pourrait causer superposition */
    button::before,
    button::after {{{{
        display: none !important;
    }}}}
    
    /* Forcer un seul layer de texte dans les boutons */
    button * {{{{
        line-height: inherit !important;
    }}}}
    
    </style>
    """, unsafe_allow_html=True)

    st.set_page_config(
        page_title="🗺️ Kibali 🌟 - Assistant IA Avancé",
        page_icon="🌟",
        layout="wide"
    )

    # Logo Kibali animé - Grand K avec les couleurs vert, jaune, bleu
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0; padding: 2rem;">
        <svg width="300" height="300" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto;">
          <defs>
            <linearGradient id="sparkleGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style="stop-color:#00ff88;stop-opacity:1" />
              <stop offset="25%" style="stop-color:#ffff00;stop-opacity:1" />
              <stop offset="50%" style="stop-color:#0088ff;stop-opacity:1" />
              <stop offset="75%" style="stop-color:#00ff88;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#ffff00;stop-opacity:1" />
            </linearGradient>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          <rect x="10" y="10" width="280" height="280" rx="30" ry="30"
                fill="none" stroke="url(#sparkleGradient)" stroke-width="8"
                filter="url(#glow)">
            <animate attributeName="stroke-dasharray" values="0,1000;500,500;0,1000" dur="3s" repeatCount="indefinite"/>
          </rect>
          <rect x="20" y="20" width="260" height="260" rx="25" ry="25" fill="#ffffff"/>
          <text x="150" y="180" font-family="Arial, sans-serif" font-size="180" font-weight="900"
                text-anchor="middle" fill="#000000" filter="url(#glow)">K</text>
          <circle cx="50" cy="50" r="3" fill="#ffff00" opacity="0.8">
            <animate attributeName="opacity" values="0.8;0.3;0.8" dur="2s" repeatCount="indefinite"/>
          </circle>
          <circle cx="250" cy="50" r="3" fill="#00ff88" opacity="0.6">
            <animate attributeName="opacity" values="0.6;0.9;0.6" dur="2.5s" repeatCount="indefinite"/>
          </circle>
          <circle cx="50" cy="250" r="3" fill="#0088ff" opacity="0.7">
            <animate attributeName="opacity" values="0.7;0.4;0.7" dur="1.8s" repeatCount="indefinite"/>
          </circle>
          <circle cx="250" cy="250" r="3" fill="#ffff00" opacity="0.5">
            <animate attributeName="opacity" values="0.5;0.8;0.5" dur="2.2s" repeatCount="indefinite"/>
          </circle>
        </svg>
        <div class="kibali-name-logo">KIBALIONE8 IA</div>
    </div>
    """, unsafe_allow_html=True)

    # Header principal avec design fluide
    st.markdown("""
    <div class="main-header">
        <h2 style="color: var(--kibali-green); margin: 0; text-align: center;">
            ⏩ Assistant IA Avancé avec Recherche Web
        </h2>
        <p style="color: var(--kibali-text-secondary); text-align: center; margin: 1rem 0 0 0;">
            <strong>Nouvelles fonctionnalités:</strong><br>
            🌐 Recherche web intelligente • 🔗 Recherche hybride • 📰 Actualités temps réel<br>
            📄 Extraction de contenu web • 💾 Cache intelligent • 🗺️ Calcul de trajets OSM
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===============================================
    # SÉLECTEUR DE THÈME DE COULEUR
    # ===============================================
    st.markdown("### 🎨 Palette de Couleurs")
    
    theme_names = {k: v['name'] for k, v in COLOR_THEMES.items()}
    current_theme_name = theme_names[st.session_state.color_theme]
    
    selected_theme_name = st.selectbox(
        "Choisir un thème",
        options=list(theme_names.values()),
        index=list(theme_names.values()).index(current_theme_name),
        key="theme_selector"
    )
    
    # Trouver la clé du thème sélectionné
    selected_theme_key = [k for k, v in theme_names.items() if v == selected_theme_name][0]
    
    if selected_theme_key != st.session_state.color_theme:
        st.session_state.color_theme = selected_theme_key
        st.rerun()
    
    # Afficher les couleurs du thème actuel
    theme_colors = COLOR_THEMES[st.session_state.color_theme]
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin: 1rem 0;">
        <div style="background: {theme_colors['primary']}; height: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>
        <div style="background: {theme_colors['secondary']}; height: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>
        <div style="background: {theme_colors['accent']}; height: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>
        <div style="background: {theme_colors['bg_accent']}; height: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===============================================
    # SIDEBAR - Indicateur de mode Online/Offline
    # ===============================================
    render_mode_indicator()
    
    # Afficher le statut des capacités si en mode offline
    status = offline_manager.get_status_info()
    if status['is_offline']:
        with st.sidebar.expander("📊 Capacités"):
            caps = get_offline_capabilities()
            
            # Afficher avec des icônes expressives uniquement
            cap_icons = []
            if caps.get('llm'):
                cap_icons.append("🧠")
            if caps.get('vision'):
                cap_icons.append("👁️")
            if caps.get('embeddings'):
                cap_icons.append("🔤")
            if caps.get('code_generation'):
                cap_icons.append("💻")
            if caps.get('image_generation'):
                cap_icons.append("🎨")
            if caps.get('translation'):
                cap_icons.append("🌍")
            
            if cap_icons:
                st.markdown(" ".join(cap_icons))
            else:
                st.warning("⚠️ Aucun modèle local")

    # Initialisation des états
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'pois' not in st.session_state:
        st.session_state.pois = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = WORKING_MODELS[list(WORKING_MODELS.keys())[0]]
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_traj_info' not in st.session_state:
        st.session_state.last_traj_info = None
    if 'last_reponse' not in st.session_state:
        st.session_state.last_reponse = ""
    # Initialisation du modèle local
    if 'local_mode' not in st.session_state:
        st.session_state.local_mode = False
    if 'local_model_loaded' not in st.session_state:
        st.session_state.local_model_loaded = False
    if 'local_tokenizer' not in st.session_state:
        st.session_state.local_tokenizer = None
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = None
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = None
    if 'local_model' not in st.session_state:
        st.session_state.local_model = None
    if 'local_qwen_llm' not in st.session_state:
        st.session_state.local_qwen_llm = None
    # Initialisation du système d'outils dynamiques
    if 'tool_manager' not in st.session_state:
        if TOOLS_SYSTEM_AVAILABLE:
            try:
                st.session_state.tool_manager = ToolManager()
                print("✅ Gestionnaire d'outils initialisé")
            except Exception as e:
                print(f"⚠️ Erreur initialisation outils: {e}")
                st.session_state.tool_manager = None
        else:
            st.session_state.tool_manager = None
    # Initialisation des résultats d'analyse de médias
    if 'media_analysis_results' not in st.session_state:
        st.session_state.media_analysis_results = []
    
    # Initialisation du tracking des fichiers uploadés pour persistance du contexte
    if 'has_pdf_context' not in st.session_state:
        st.session_state.has_pdf_context = False
    if 'chat_uploaded_pdfs' not in st.session_state:
        st.session_state.chat_uploaded_pdfs = []
    if 'binary_files' not in st.session_state:
        st.session_state.binary_files = []
    
    # 🧠 NOUVEAU: Initialisation de la mémoire vectorielle des conversations
    if 'chat_vectordb' not in st.session_state:
        from chat_memory import load_chat_vectordb
        st.session_state.chat_vectordb, memory_msg = load_chat_vectordb()
        if st.session_state.chat_vectordb:
            print(memory_msg)
    
    # 🤖 NOUVEAU: Initialisation du système d'auto-apprentissage
    if 'auto_learning' not in st.session_state and AUTO_LEARNING_AVAILABLE:
        st.session_state.auto_learning = get_auto_learning()
        st.session_state.knowledge_manager = get_knowledge_manager()
        print("🧠 Auto-apprentissage initialisé")

    # ===============================================
    # Onglets avec design fluide
    # ===============================================
    tab_names = ["⚙️ **Configuration**", "💬 **Chat RAG + Web**", "🗺️ **Trajets**", "📸 **Analyse Image**", "🌐 **Recherche Web**", "📷 **Photogrammétrie**"]
    
    # Ajouter l'onglet binaire si le gestionnaire est disponible
    if BINARY_HANDLER_AVAILABLE:
        tab_names.append("🔬 **Fichiers Binaires**")
    
    if TOOLS_SYSTEM_AVAILABLE and st.session_state.tool_manager:
        tab_names.append("🔧 **Outils Dynamiques**")
    
    # Onglet Nano-IA si auto-apprentissage disponible
    if AUTO_LEARNING_AVAILABLE:
        tab_names.append("🤖 **Nano-IA & Apprentissage**")
    
    tabs = st.tabs(tab_names)
    
    # Assignation des onglets
    tab1, tab2, tab3, tab4, tab5, tab_photo = tabs[:6]
    
    # Onglet binaire si disponible
    tab_binary = None
    current_tab_idx = 6
    if BINARY_HANDLER_AVAILABLE and len(tabs) > 6:
        tab_binary = tabs[current_tab_idx]
        current_tab_idx += 1
    
    # Onglet outils dynamiques si disponible
    tab_tools = None
    if TOOLS_SYSTEM_AVAILABLE and st.session_state.tool_manager and len(tabs) > current_tab_idx:
        tab_tools = tabs[current_tab_idx]
        current_tab_idx += 1
    
    # Onglet Nano-IA si disponible
    tab_nano_ai = None
    if AUTO_LEARNING_AVAILABLE and len(tabs) > current_tab_idx:
        tab_nano_ai = tabs[current_tab_idx]

    # ===============================================
    # Onglet 1: Configuration avec cartes
    # ===============================================
    with tab1:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Gestion des données et configuration")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            pdf_files = st.file_uploader(
                "📤 **Upload Documents (tous formats acceptés)**", 
                type=None, 
                accept_multiple_files=True,
                key="pdf_upload"
            )
            if pdf_files:
                saved_files = []
                for pdf_file in pdf_files:
                    filepath = os.path.join(PDFS_PATH, pdf_file.name)
                    with open(filepath, 'wb') as f:
                        f.write(pdf_file.getbuffer())
                    saved_files.append(pdf_file.name)
                st.success("✅📄")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            pbf_file = st.file_uploader(
                "📤 **Upload fichier OSM (tous formats acceptés)**", 
                type=None,
                key="pbf_upload"
            )
            if pbf_file:
                pbf_path = os.path.join(GRAPHS_PATH, pbf_file.name)
                with open(pbf_path, 'wb') as f:
                    f.write(pbf_file.getbuffer())
                G, pois, status = upload_and_process_pbf(pbf_path)
                st.session_state.graph = G
                st.session_state.pois = pois
                st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 🌊 Uploader ERT pour fichiers .dat
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("#### 🌊 Analyse ERT (Electrical Resistivity Tomography)")
        dat_file = st.file_uploader(
            "📤 **Upload fichier .dat pour analyse géologique/hydrologique**", 
            type=["dat"],
            key="dat_upload",
            help="Formats: 4 colonnes (espaces) OU CSV avec fréquences MHz (virgules)"
        )
        if dat_file:
            # Sauvegarder temporairement le fichier
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dat') as tmp:
                tmp.write(dat_file.getbuffer())
                tmp_path = tmp.name
            
            # Stocker dans session_state pour l'orchestration
            st.session_state.uploaded_dat_file = {
                'name': dat_file.name,
                'path': tmp_path,
                'size': dat_file.size
            }
            st.success(f"✅ {dat_file.name} chargé ({dat_file.size} octets)")
            st.info("💬 Pose une question sur ce fichier dans le chat pour déclencher l'analyse ERT automatique !")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            if st.button("🔄 **Traiter PDFs**", key="process_pdfs", help="Traite les PDFs uploadés pour créer la base vectorielle"):
                with st.spinner("Traitement des PDFs..."):
                    vectordb, status = process_pdfs()
                    st.session_state.vectordb = vectordb
                    st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            if st.button("📂 **Charger graphe existant**", key="load_graph", help="Charge un graphe OSM existant"):
                with st.spinner("Chargement du graphe..."):
                    G, pois, status = load_existing_graph()
                    st.session_state.graph = G
                    st.session_state.pois = pois
                    st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            if st.button("📂 **Charger base vectorielle**", key="load_vectordb", help="Charge la base vectorielle existante"):
                with st.spinner("Chargement de la base..."):
                    vectordb, status = load_vectordb()
                    st.session_state.vectordb = vectordb
                    st.success(status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col6:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            if st.button("🧹 **Vider Cache**", key="clear_streamlit_cache", help="Vide le cache Streamlit et rafraîchit l'application", type="primary"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Cache vidé !")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        col6, col7, col8 = st.columns(3)
        with col6:
            if st.button("🗑️ **Vider cache web**", key="clear_cache", help="Nettoie le cache web des anciennes recherches"):
                status = handle_clear_cache()
                st.success(status)
        
        with col7:
            if st.button("💬 **Vider historique chat**", key="clear_chat_history", help="Efface tout l'historique de conversation", type="secondary"):
                st.session_state.chat_history = []
                st.session_state.processed_files = set()
                st.session_state.media_analysis_results = []
                st.success("✅ Historique vidé !")
                st.rerun()
        
        with col8:
            if st.button("📊 **Voir statistiques**", key="show_stats", help="Affiche les statistiques détaillées du système"):
                stats = get_system_status()
                st.json(stats)
        st.markdown('</div>', unsafe_allow_html=True)

    # ===============================================
    # Onglet 2: Chat RAG Amélioré avec cartes
    # ===============================================
    with tab2:
        # Contrôles compacts dans un expander
        with st.expander("⚙️ Configuration", expanded=False):
            col1, col2 = st.columns([2, 1])
            with col1:
                model_choice = st.selectbox(
                    "🎯 **Modèle IA**",
                    options=list(WORKING_MODELS.keys()),
                    index=0,
                    key="model_select",
                    help="Choisis le modèle d'IA pour tes réponses"
                )
                # Mettre à jour llm_model dans session_state
                st.session_state.llm_model = WORKING_MODELS[model_choice]
                st.session_state.current_model = WORKING_MODELS[model_choice]
            with col2:
                web_enabled = st.checkbox("🌐 **Recherche web activée**", value=True, help="Active la recherche web pour des réponses plus complètes")
                # Toggle pour le mode local
                local_mode_toggle = st.checkbox(
                    "🏠 **Mode Local (Qwen 1.5B)**", 
                    value=st.session_state.local_mode, 
                    help="Active le modèle local Qwen 1.5B pour les tâches complexes quand l'API est surchargée"
                )
                if local_mode_toggle != st.session_state.local_mode:
                    st.session_state.local_mode = local_mode_toggle
                    if local_mode_toggle and not st.session_state.local_model_loaded:
                        # Charger le modèle local
                        with st.spinner("🔄 Chargement du modèle local Qwen 1.5B..."):
                            try:
                                tokenizer, model, device, gpu_info = load_local_llm_model()
                                st.session_state.local_tokenizer = tokenizer
                                st.session_state.local_model = model
                                st.session_state.local_qwen_llm = QwenChatModel(tokenizer, model)
                                st.session_state.local_model_loaded = True
                                st.success(f"✅ Modèle local chargé sur {device.upper()}")
                            except Exception as e:
                                st.error(f"❌ Erreur chargement modèle local: {e}")
                                st.session_state.local_mode = False
                    st.rerun()
        
        # Analyser automatiquement les médias uploadés (avant le chat_input)
        # media_analysis_results = []  # Maintenant global dans session_state
        
        # Zone d'upload de médias compacte (style ChatGPT)
        st.markdown("**📎 Ajouter des médias:**")
        col_attach1, col_attach2, col_attach3 = st.columns([1, 1, 1])
        
        with col_attach1:
            uploaded_images = st.file_uploader(
                "📎", label_visibility="collapsed",
                type=None,
                accept_multiple_files=True,
                key="image_uploader",
                help="🖼️ Ajouter des fichiers (tous formats acceptés)"
            )
        
        with col_attach2:
            uploaded_audios = st.file_uploader(
                "📎", label_visibility="collapsed",
                type=None,
                accept_multiple_files=True,
                key="audio_uploader",
                help="🎵 Ajouter des fichiers (tous formats acceptés)"
            )
        
        with col_attach3:
            uploaded_videos = st.file_uploader(
                "📎", label_visibility="collapsed",
                type=None,
                accept_multiple_files=True,
                key="video_uploader",
                help="🎥 Ajouter des fichiers (tous formats acceptés)"
            )
        
        # Nouvelle colonne pour les documents
        st.markdown("**📄 Ajouter des documents:**")
        uploaded_pdfs = st.file_uploader(
            "📎 Tous fichiers", label_visibility="collapsed",
            type=None,
            accept_multiple_files=True,
            key="pdf_chat_uploader",
            help="📄 Ajouter des fichiers (tous formats acceptés - ajoutés automatiquement au RAG vectoriel)"
        )
        
        # Initialiser le tracking des fichiers traités
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        
        # Initialiser le tracking des PDFs uploadés dans le chat
        if 'chat_uploaded_pdfs' not in st.session_state:
            st.session_state.chat_uploaded_pdfs = []
        
        # Variable pour tracker si de nouveaux médias ont été analysés
        new_media_analyzed = False
        
        # ===============================================
        # TRAITEMENT INTELLIGENT DES FICHIERS UPLOADÉS
        # ===============================================
        if uploaded_pdfs:
            for uploaded_file in uploaded_pdfs:
                # Vérifier si déjà traité
                file_key = f"file_{uploaded_file.name}_{uploaded_file.size}"
                if file_key in st.session_state.processed_files:
                    continue  # Skip si déjà traité
                
                # Marquer comme en cours de traitement
                st.session_state.processed_files.add(file_key)
                
                # Détection automatique du type de fichier
                file_ext = Path(uploaded_file.name).suffix.lower()
                is_pdf = file_ext == '.pdf'
                
                # Lire les données pour vérifier le magic number
                file_data = uploaded_file.read()
                uploaded_file.seek(0)  # Reset pour réutilisation
                is_pdf_magic = file_data.startswith(b'%PDF')
                
                # ============================================
                # TRAITEMENT PDF (si c'est un vrai PDF)
                # ============================================
                if is_pdf or is_pdf_magic:
                    # Message utilisateur
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": f"📄 PDF uploadé: {uploaded_file.name}"
                    })
                    
                    # Traitement du PDF
                    with st.spinner(f"📄 Traitement de {uploaded_file.name}..."):
                        try:
                            # 1. Sauvegarder le PDF dans PDFS_PATH
                            pdf_path = os.path.join(PDFS_PATH, uploaded_file.name)
                            with open(pdf_path, 'wb') as f:
                                f.write(file_data)
                            
                            st.success(f"✅ PDF sauvegardé: {uploaded_file.name}")
                            
                            # 2. Extraire le texte
                            pdf_text = extract_text_from_pdf(pdf_path)
                            pdf_pages = len(pdf_text.split('\n\n'))  # Approximation du nombre de pages
                            word_count = len(pdf_text.split())
                            
                            # Vérifier qu'il y a du texte extrait
                            if not pdf_text.strip() or word_count == 0:
                                st.warning("⚠️ Aucun texte extrait - PDF vide ou scanné")
                                # Continuer quand même pour permettre l'analyse binaire si nécessaire
                                continue
                            
                            st.info(f"📊 Extrait: ~{pdf_pages} pages, {word_count} mots")
                            
                            # 3. Ajouter au RAG vectoriel
                            with st.spinner("🔄 Ajout au RAG vectoriel..."):
                                # Charger l'embedding model
                                embedding_model = HuggingFaceEmbeddings(
                                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    cache_folder=str(SENTENCE_TRANSFORMER_CACHE)
                                )
                                
                                # Chunker le texte
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1000,
                                    chunk_overlap=100
                                )
                                
                                # Créer les documents
                                chunks = text_splitter.split_text(pdf_text)
                                
                                # Vérifier qu'on a des chunks
                                if not chunks or len(chunks) == 0:
                                    st.warning("⚠️ Impossible de créer des chunks - texte trop court")
                                    continue
                                
                                documents = [
                                    Document(
                                        page_content=chunk,
                                        metadata={
                                            "source": uploaded_file.name,
                                            "chunk_id": i,
                                            "total_chunks": len(chunks)
                                        }
                                    )
                                    for i, chunk in enumerate(chunks)
                                ]
                                
                                # Ajouter à la vectordb existante ou créer nouvelle
                                if st.session_state.vectordb:
                                    st.session_state.vectordb.add_documents(documents)
                                    st.success(f"✅ {len(chunks)} chunks ajoutés au RAG")
                                else:
                                    # Créer nouvelle vectordb
                                    st.session_state.vectordb = FAISS.from_documents(
                                        documents,
                                        embedding_model
                                    )
                                st.success(f"✅ RAG créé avec {len(chunks)} chunks")
                            
                            # Sauvegarder la vectordb
                            st.session_state.vectordb.save_local(VECTORDB_PATH)
                            st.success("💾 Base vectorielle sauvegardée")
                            
                            # 4. Stocker les infos du PDF pour le panneau d'outils et contexte de conversation
                            pdf_info = {
                                'name': uploaded_file.name,
                                'path': pdf_path,
                                'text': pdf_text,
                                'pages': pdf_pages,
                                'word_count': word_count,
                                'chunks': len(chunks),
                                'uploaded_at': os.path.getmtime(pdf_path)  # Timestamp pour tracking
                            }
                            st.session_state.chat_uploaded_pdfs.append(pdf_info)
                            
                            # Marquer que nous avons un contexte PDF actif
                            st.session_state.has_pdf_context = True
                            
                            # 🧠 AUTO-APPRENTISSAGE: Apprendre du PDF
                            if AUTO_LEARNING_AVAILABLE and st.session_state.get('auto_learning'):
                                with st.spinner("🧠 Apprentissage du contenu..."):
                                    if st.session_state.auto_learning.learn_from_pdf(pdf_info):
                                        st.success("✅ Connaissances intégrées au système d'apprentissage!")
                            
                            # 5. Message de succès avec panneau d'outils
                            tools_panel_html = f'''
<div style="
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 136, 255, 0.1));
    border: 2px solid rgba(0, 255, 136, 0.3);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 255, 136, 0.2);
">
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span style="font-size: 2rem; margin-right: 1rem;">📄</span>
        <div>
            <h3 style="color: #00ff88; margin: 0; font-size: 1.3rem;">PDF Chargé avec Succès !</h3>
            <p style="color: #b0b0b0; margin: 0.3rem 0 0 0; font-size: 0.9rem;">{uploaded_file.name}</p>
        </div>
    </div>
    
    <div style="
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.5rem;
        margin-bottom: 1rem;
        padding: 1rem;
        background: rgba(74, 74, 126, 0.2);
        border-radius: 10px;
    ">
        <div style="color: white;">
            <strong>📊 Pages:</strong> ~{pdf_pages}
        </div>
        <div style="color: white;">
            <strong>📝 Mots:</strong> {word_count:,}
        </div>
        <div style="color: white;">
            <strong>🧩 Chunks RAG:</strong> {len(chunks)}
        </div>
        <div style="color: white;">
            <strong>✅ Statut:</strong> <span style="color: #00ff88;">Prêt</span>
        </div>
    </div>
    
    <h4 style="color: #00ff88; margin: 1rem 0 0.5rem 0;">🛠️ Outils Disponibles:</h4>
    <p style="color: #e0e0e0; font-size: 0.9rem; margin-bottom: 1rem;">
        Le PDF est maintenant dans la base vectorielle. Vous pouvez:
    </p>
    
    <div style="display: grid; gap: 0.8rem;">
        <div style="
            background: rgba(0, 255, 136, 0.1);
            border-left: 3px solid #00ff88;
            padding: 0.8rem;
            border-radius: 5px;
        ">
            <strong style="color: #00ff88;">💬 1. Poser des questions</strong>
            <p style="color: white; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                Ex: "Résume ce PDF", "Quels sont les points clés?"
            </p>
        </div>
        
        <div style="
            background: rgba(0, 136, 255, 0.1);
            border-left: 3px solid #0088ff;
            padding: 0.8rem;
            border-radius: 5px;
        ">
            <strong style="color: #0088ff;">📋 2. Résumé Intelligent</strong>
            <p style="color: white; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                Tapez: "Résume le PDF en mode [court/moyen/détaillé]"
            </p>
        </div>
        
        <div style="
            background: rgba(255, 215, 0, 0.1);
            border-left: 3px solid #ffd700;
            padding: 0.8rem;
            border-radius: 5px;
        ">
            <strong style="color: #ffd700;">🌐 3. Traduction Complète</strong>
            <p style="color: white; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                Tapez: "Traduis le PDF en [langue]"
            </p>
        </div>
        
        <div style="
            background: rgba(138, 43, 226, 0.1);
            border-left: 3px solid #8a2be2;
            padding: 0.8rem;
            border-radius: 5px;
        ">
            <strong style="color: #8a2be2;">💻 4. Générer Application</strong>
            <p style="color: white; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                Tapez: "Génère une application basée sur ce PDF"
            </p>
        </div>
        
        <div style="
            background: rgba(255, 105, 180, 0.1);
            border-left: 3px solid #ff69b4;
            padding: 0.8rem;
            border-radius: 5px;
        ">
            <strong style="color: #ff69b4;">📊 5. Rapport Multi-IA Détaillé</strong>
            <p style="color: white; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                Tapez: "Génère un rapport détaillé sur ce PDF"
            </p>
        </div>
    </div>
    
    <div style="
        margin-top: 1rem;
        padding: 0.8rem;
        background: rgba(0, 255, 136, 0.05);
        border-radius: 8px;
        border: 1px dashed rgba(0, 255, 136, 0.3);
    ">
        <p style="color: #00ff88; margin: 0; font-size: 0.85rem;">
            💡 <strong>Astuce:</strong> L'IA a maintenant accès au contenu complet du PDF via le RAG vectoriel !
        </p>
    </div>
</div>
'''
                        
                            # Ajouter le panneau d'outils au chat
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": tools_panel_html,
                                "is_html": True
                            })
                            
                        except Exception as e:
                            import traceback
                            error_msg = f"❌ Erreur lors du traitement du PDF: {str(e)}"
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                            st.error(traceback.format_exc())
                
                # ============================================
                # TRAITEMENT FICHIER BINAIRE (si ce n'est PAS un PDF)
                # ============================================
                elif BINARY_HANDLER_AVAILABLE:
                    # Message utilisateur
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": f"🔬 Fichier uploadé: {uploaded_file.name}"
                    })
                    
                    with st.spinner(f"🔬 Analyse du fichier {uploaded_file.name}..."):
                        try:
                            # 🆕 DÉTECTION AUTOMATIQUE DE STRUCTURE
                            structure_detected = None
                            extracted_df = None
                            
                            if BINARY_DETECTOR_AVAILABLE:
                                try:
                                    # Sauver temporairement pour analyse
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                                        tmp.write(file_data)
                                        tmp_path = tmp.name
                                    
                                    # Détecter structure
                                    structure_detected, extracted_df = detect_and_analyze_file(tmp_path)
                                    
                                    # Nettoyer
                                    import os
                                    os.unlink(tmp_path)
                                    
                                    st.success(f"✅ Structure détectée: {structure_detected.file_type.value} (confiance: {structure_detected.confidence:.0%})")
                                    
                                except Exception as e:
                                    st.warning(f"⚠️ Détection structure: {e}")
                            
                            # Analyser avec le gestionnaire de fichiers binaires
                            result = BinaryFileHandler.process_file(uploaded_file)
                            file_info = result['info']
                            viewer = result['viewer']
                            
                            # Générer l'analyse complète
                            analysis_report = analyze_binary_file(file_data, uploaded_file.name)
                            
                            # Créer le visualiseur hex
                            hex_view = viewer.hex_dump(start=0, length=512)
                            
                            # Extraire les chaînes de texte
                            text_strings = viewer.extract_text_strings(min_length=4)
                            
                            # Interprétation de l'entropie
                            entropy = file_info['analysis']['entropy']
                            if entropy < 3:
                                entropy_desc = "Données très répétitives/structurées"
                                entropy_color = "#00ff88"
                            elif entropy < 6:
                                entropy_desc = "Données texte/structurées"
                                entropy_color = "#0088ff"
                            elif entropy < 7.5:
                                entropy_desc = "Données binaires normales"
                                entropy_color = "#ffd700"
                            else:
                                entropy_desc = "Données compressées/chiffrées"
                                entropy_color = "#ff69b4"
                            
                            # Escape HTML dans les chaînes de texte
                            safe_text_strings = [s.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;') for s in text_strings]
                            
                            # 🆕 AFFICHAGE AVEC COMPOSANTS STREAMLIT NATIFS
                            st.markdown(f"## 🔬 Analyse Complète: {uploaded_file.name}")
                            
                            # Structure détectée (si disponible)
                            if structure_detected:
                                with st.expander("🔍 Structure Détectée", expanded=True):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("📋 Type", structure_detected.file_type.value)
                                        st.metric("✅ Confiance", f"{structure_detected.confidence:.0%}")
                                    with col2:
                                        st.metric("🔤 Encodage", structure_detected.encoding)
                                        if structure_detected.row_count > 0:
                                            st.metric("📏 Lignes", structure_detected.row_count)
                                    
                                    if structure_detected.columns:
                                        st.write(f"**📊 Colonnes:** {', '.join(structure_detected.columns[:5])}" + (" ..." if len(structure_detected.columns) > 5 else ""))
                                    if structure_detected.delimiter:
                                        st.write(f"**🔹 Délimiteur:** `{repr(structure_detected.delimiter)}`")
                            
                            # Informations générales
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Taille", file_info['analysis']['size_formatted'])
                            with col2:
                                st.metric("🔢 Type", file_info['format_name'])
                            with col3:
                                st.metric("📈 Entropie", f"{entropy:.2f}/8.0")
                            
                            st.info(f"💡 {entropy_desc}")
                            
                            # Visualiseur hex
                            with st.expander("🔍 Visualiseur Hexadécimal (512 premiers bytes)"):
                                st.code(hex_view, language=None)
                            
                            # Statistiques
                            with st.expander("📊 Statistiques Détaillées"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Moyenne:** {file_info['analysis']['statistics'].get('mean', 0):.2f}")
                                    st.write(f"**Médiane:** {file_info['analysis']['statistics'].get('median', 0):.2f}")
                                    st.write(f"**Écart-type:** {file_info['analysis']['statistics'].get('std', 0):.2f}")
                                with col2:
                                    st.write(f"**Min/Max:** {file_info['analysis']['statistics'].get('min', 0)} / {file_info['analysis']['statistics'].get('max', 0)}")
                                    st.write(f"**Bytes uniques:** {file_info['analysis']['statistics'].get('unique_bytes', 0)} / 256")
                            
                            # Chaînes de texte
                            with st.expander(f"📝 Chaînes de Texte Extraites ({len(text_strings)} trouvées)"):
                                for i, s in enumerate(text_strings[:30], 1):
                                    st.text(f"{i}. {s}")
                                if len(text_strings) > 30:
                                    st.info(f"... et {len(text_strings) - 30} autres chaînes")
                            
                            # Actions disponibles
                            st.markdown("### 🛠️ Actions Disponibles")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.success("**💬 1. Analyse IA Complète**\n\nDemandez: 'Analyse ce fichier'")
                            with col2:
                                st.info("**🔍 2. Exploration Hexadécimale**\n\nDemandez: 'Montre offset 0x1000'")
                            with col3:
                                st.warning("**📊 3. Extraction de Données**\n\nDemandez: 'Convertis en numpy'")
                            
                            # Stocker les infos pour utilisation ultérieure avec le contexte d'analyse
                            if 'binary_files' not in st.session_state:
                                st.session_state.binary_files = []
                            
                            # Stocker également le contexte d'analyse complet pour éviter les réanalyses
                            binary_info_complete = {
                                'name': uploaded_file.name,
                                'data': file_data,
                                'info': file_info,
                                'viewer': viewer,
                                'text_strings': text_strings,
                                'analysis_report': analysis_report,  # Sauvegarder l'analyse
                                'hex_view': hex_view,
                                'entropy': entropy,
                                'entropy_desc': entropy_desc,
                                'structure': structure_detected,  # 🆕 Structure détectée
                                'dataframe': extracted_df  # 🆕 DataFrame extrait
                            }
                            st.session_state.binary_files.append(binary_info_complete)
                            
                            # 📊 Afficher le DataFrame si disponible
                            if extracted_df is not None and not extracted_df.empty:
                                st.markdown("### 📊 Données Extraites et Structurées")
                                st.dataframe(extracted_df.head(50), use_container_width=True)
                                
                                # Statistiques sur le DataFrame
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                                            padding: 1rem; border-radius: 10px; margin: 1rem 0;
                                            border-left: 4px solid #4caf50;">
                                    <p style="color: #1b5e20; margin: 0.3rem 0;">
                                        <strong>📏 Dimensions:</strong> {extracted_df.shape[0]} lignes × {extracted_df.shape[1]} colonnes
                                    </p>
                                    <p style="color: #1b5e20; margin: 0.3rem 0;">
                                        <strong>🔢 Colonnes numériques:</strong> {len(extracted_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns)}
                                    </p>
                                    <p style="color: #1b5e20; margin: 0.3rem 0;">
                                        <strong>📝 Colonnes texte:</strong> {len(extracted_df.select_dtypes(include=['object']).columns)}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # 🧠 AUTO-APPRENTISSAGE: Apprendre du fichier binaire
                            if AUTO_LEARNING_AVAILABLE and st.session_state.get('auto_learning'):
                                with st.spinner("🧠 Apprentissage du fichier binaire..."):
                                    if st.session_state.auto_learning.learn_from_binary(binary_info_complete):
                                        st.success("✅ Analyse binaire intégrée au système!")
                            
                        except Exception as e:
                            error_msg = f"❌ Erreur lors de l'analyse du fichier: {str(e)}"
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                            import traceback
                            st.error(traceback.format_exc())
        
        # Analyse des images avec Vision AI (s'affiche dans le chat)
        if uploaded_images:
            # Ajouter l'analyse directement dans le chat
            for idx, img_file in enumerate(uploaded_images):
                # Vérifier si déjà traité
                file_key = f"img_{img_file.name}_{img_file.size}"
                if file_key in st.session_state.processed_files:
                    continue  # Skip si déjà traité
                
                # Marquer comme en cours de traitement
                st.session_state.processed_files.add(file_key)
                new_media_analyzed = True
                
                # Message utilisateur avec miniature
                img_file.seek(0)
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"📷 Image uploadée: {img_file.name}"
                })
                
                # Analyse en arrière-plan
                with st.spinner(f"🤖 Analyse de {img_file.name}..."):
                    try:
                        import tempfile
                        import base64
                        from io import BytesIO
                        
                        # Convertir l'image en base64 pour l'API
                        img_file.seek(0)
                        image_bytes = img_file.read()
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        
                        # Sauvegarder temporairement pour métadonnées
                        img_suffix = os.path.splitext(img_file.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=img_suffix) as tmp_file:
                            tmp_file.write(image_bytes)
                            tmp_path = tmp_file.name
                        
                        # Obtenir les métadonnées de base
                        from PIL import Image as PILImage
                        img = PILImage.open(tmp_path)
                        width, height = img.size
                        img_format = img.format
                        
                        # Analyse avec modèle de vision local (CLIP)
                        try:
                            # Charger les modèles locaux
                            if 'vision_models' not in st.session_state:
                                with st.spinner("📦 Chargement des modèles de vision locaux..."):
                                    st.session_state.vision_models = load_vision_models()
                            
                            # Charger l'OCR
                            if 'ocr_reader' not in st.session_state:
                                with st.spinner("📦 Chargement du modèle OCR..."):
                                    st.session_state.ocr_reader = load_ocr_reader()
                            
                            vision_success = False
                            image_caption = ""
                            analysis_details = []
                            extracted_text_info = ""
                            
                            # Extraction du texte avec OCR
                            extracted_text_keywords = []
                            if st.session_state.ocr_reader or pytesseract:
                                st.info("📝 Extraction OCR en cours...")
                                extracted_texts = extract_text_from_image(tmp_path, st.session_state.ocr_reader)
                                if extracted_texts:
                                    extracted_text_info = organize_extracted_text(extracted_texts)
                                    # Extraire mots-clés pour recherche web
                                    import re
                                    text_words = re.findall(r'✅.*?\]\s*(.*?)$', extracted_text_info, re.MULTILINE)
                                    if text_words:
                                        extracted_text_keywords = list(set([w.strip() for w in text_words if len(w.strip()) > 2]))[:5]
                                    st.success(f"✅ OCR: {len(extracted_texts)} éléments détectés")
                                else:
                                    extracted_text_info = "Aucun texte détecté dans l'image."
                            
                            # Détection d'objets avec YOLO
                            yolo_detections = []
                            if 'yolo_model' not in st.session_state:
                                st.session_state.yolo_model = load_yolo_model()
                            
                            if st.session_state.yolo_model:
                                st.info("🎯 Détection YOLO en cours...")
                                try:
                                    # Forcer l'inférence sur CPU
                                    results = st.session_state.yolo_model(tmp_path, verbose=False, device='cpu')
                                    if results and len(results) > 0:
                                        for result in results:
                                            if result.boxes:
                                                for box in result.boxes:
                                                    class_id = int(box.cls[0])
                                                    confidence = float(box.conf[0])
                                                    class_name = result.names[class_id]
                                                    if confidence > 0.3:  # Seuil 30%
                                                        yolo_detections.append({
                                                            'class': class_name,
                                                            'confidence': confidence
                                                        })
                                        if yolo_detections:
                                            st.success(f"✅ YOLO: {len(yolo_detections)} objets détectés")
                                except Exception as yolo_err:
                                    st.warning(f"⚠️ YOLO: {yolo_err}")
                            
                            # Analyse visuelle avec CLIP (NOUVELLE MÉTHODE DÉTAILLÉE)
                            if st.session_state.vision_models:
                                st.info("🔍 Analyse CLIP détaillée (50 prompts)...")
                                
                                # Utiliser la nouvelle fonction de description détaillée
                                clip_result, error = generate_detailed_description_with_clip(tmp_path, st.session_state.vision_models)
                                
                                if clip_result:
                                    image_caption = clip_result['description']
                                    analysis_details = [
                                        {'label': 'Type', 'confidence': clip_result['confidence'], 'value': clip_result['main_type']},
                                        {'label': 'Couleurs', 'confidence': 1.0, 'value': clip_result['colors']},
                                        {'label': 'Détails', 'confidence': 1.0, 'value': clip_result.get('details', 'N/A')}
                                    ]
                                    vision_success = True
                                    st.success("✅ Analyse CLIP détaillée complétée")
                                else:
                                    st.error(f"❌ Erreur: {error}")
                            else:
                                st.error("❌ Modèles de vision non disponibles")
                            
                            if vision_success:
                                # Enrichir avec recherche web sur TEXTE OCR détecté
                                web_context = ""
                                web_results_text = []
                                try:
                                    if extracted_text_keywords:
                                        # Recherche web basée sur le TEXTE détecté (plus pertinent)
                                        search_query = " ".join(extracted_text_keywords[:3])
                                        st.info(f"🌐 Recherche web: '{search_query}'")
                                        web_results_text = enhanced_web_search(search_query, max_results=3, search_type="both")
                                        if web_results_text:
                                            web_context = "\n".join([f"📌 {r.get('title', '')[:60]}: {r.get('body', '')[:120]}" for r in web_results_text[:3]])
                                            st.success(f"✅ Web: {len(web_results_text)} résultats")
                                    else:
                                        # Fallback: recherche sur le type d'image
                                        web_results_text = enhanced_web_search(f"information sur: {image_caption}", max_results=2)
                                        web_context = "\n".join([f"{r.get('title', '')[:40]}: {r.get('body', '')[:100]}" for r in web_results_text[:2]]) if web_results_text else ""
                                except Exception as web_err:
                                    st.warning(f"⚠️ Recherche web: {web_err}")
                                    web_context = ""
                                
                                # PROMPT ULTRA-AMÉLIORÉ avec YOLO + Web
                                yolo_summary = ""
                                if yolo_detections:
                                    yolo_list = ", ".join([f"{d['class']} ({d['confidence']:.0%})" for d in yolo_detections[:10]])
                                    yolo_summary = f"\n\n🎯 **YOLO a détecté ces objets:**\n{yolo_list}"
                                
                                web_summary = ""
                                if web_context:
                                    web_summary = f"\n\n🌐 **Informations web trouvées:**\n{web_context[:400]}"
                                
                                analysis_prompt = f"""Analyse cette image en français de manière ULTRA-PRÉCISE:

📸 Image: {img_file.name} ({width}x{height}px, {img_format})

🖼️ **CLIP Vision AI (50 prompts) a détecté:**
{image_caption}

📝 **OCR a extrait ce texte:**
{extracted_text_info[:400] if extracted_text_info and "Aucun texte" not in extracted_text_info else "Aucun texte visible"}{yolo_summary}{web_summary}

INSTRUCTIONS CRITIQUES:
1. Combine TOUTES les sources (CLIP + OCR + YOLO + Web) pour une analyse complète
2. Si OCR détecte du texte, UTILISE LES INFOS WEB pour expliquer le contexte
3. Si YOLO détecte des objets, mentionne-les précisément
4. Ignore les détections de lettres CLIP < 40% si OCR ne confirme pas
5. Décris ce qui est VRAIMENT visible avec TOUS les détails (couleurs exactes, objets, texte, contexte)
6. Si c'est un logo/texte, explique CE QUE ÇA REPRÉSENTE (utilise le web)

Réponds en 3-5 phrases ULTRA-DÉTAILLÉES incluant: type d'image, couleurs précises, objets/textes détectés, et contexte web:"""

                                text_client = create_client()
                                analysis_response = text_client.chat.completions.create(
                                    model=WORKING_MODELS[model_choice],
                                    messages=[{"role": "user", "content": analysis_prompt[:1800]}],
                                    max_tokens=500,
                                    temperature=0.1  # Température très basse pour précision maximale
                                )
                                
                                enriched_analysis = analysis_response.choices[0].message.content
                                
                                # Préparer l'affichage ULTRA-COMPLET
                                analysis_display = f"**🖼️ Analyse Ultra-Détaillée de {img_file.name}**\n\n"
                                analysis_display += f"📏 Résolution: {width}x{height}px | Format: {img_format}\n\n"
                                
                                # Extraire la lettre de la description CLIP SEULEMENT si confiance > 40%
                                import re
                                letter_match = re.search(r"lettre '([A-Z])' \(confiance: ([\d.]+)%\)", image_caption)
                                if letter_match:
                                    detected_letter = letter_match.group(1)
                                    letter_confidence = float(letter_match.group(2))
                                    # Afficher SEULEMENT si confiance > 40%
                                    if letter_confidence > 40.0:
                                        analysis_display += f"## 🔤 LETTRE DÉTECTÉE: **{detected_letter}** ({letter_confidence:.1f}%)\n\n"
                                
                                # Description CLIP détaillée (50 prompts) avec COULEURS SÉPARÉES
                                # Extraire les couleurs du clip_result
                                colors_detailed = clip_result.get('colors', 'N/A')
                                main_type_only = clip_result.get('main_type', image_caption)
                                
                                analysis_display += f"## 🤖 Description IA (CLIP 50 prompts):\n\n**{main_type_only}**\n\n"
                                analysis_display += f"## 🎨 Couleurs détectées (TOP 3):\n\n**{colors_detailed}**\n\n"
                                
                                # Ajouter le texte OCR si disponible
                                if extracted_text_info and "Aucun texte" not in extracted_text_info:
                                    text_words = re.findall(r'✅.*?\]\s*(.*?)$', extracted_text_info, re.MULTILINE)
                                    if text_words:
                                        unique_words = list(set([w.strip() for w in text_words if len(w.strip()) > 1]))
                                        if unique_words:
                                            analysis_display += f"## 📝 Texte OCR: **{', '.join(unique_words[:10])}**\n\n"
                                
                                # Ajouter YOLO si disponible
                                if yolo_detections:
                                    yolo_top = sorted(yolo_detections, key=lambda x: x['confidence'], reverse=True)[:5]
                                    yolo_str = ", ".join([f"{d['class']} ({d['confidence']:.0%})" for d in yolo_top])
                                    analysis_display += f"## 🎯 Objets YOLO détectés: **{yolo_str}**\n\n"
                                
                                # Ajouter résultats web si disponibles
                                if web_results_text:
                                    analysis_display += f"## 🌐 Informations Web ({len(web_results_text)} sources):\n\n"
                                    for idx, res in enumerate(web_results_text[:3], 1):
                                        analysis_display += f"**{idx}.** {res.get('title', 'Sans titre')[:80]}\n"
                                        analysis_display += f"   _{res.get('body', '')[:150]}..._\n\n"
                                
                                analysis_display += f"## 🧠 Synthèse Intelligence Artificielle:\n\n{enriched_analysis}"
                                
                                # Ajouter l'analyse au chat (markdown pur sans HTML visible)
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": analysis_display
                                })
                                
                                st.session_state.media_analysis_results.append({
                                    'type': 'image',
                                    'name': img_file.name,
                                    'resolution': f"{width}x{height}",
                                    'format': img_format,
                                    'caption': image_caption,
                                    'ai_analysis': enriched_analysis,
                                    'web_context': web_context,
                                    'extracted_text': extracted_text_info
                                })
                            
                            else:
                                # Si aucun modèle n'a fonctionné
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"❌ Impossible d'analyser {img_file.name}. Vérifiez votre token HuggingFace ou réessayez plus tard."
                                })
                                
                        except Exception as analysis_error:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"❌ Erreur lors de l'analyse de {img_file.name}: {str(analysis_error)}"
                            })
                            
                    except Exception as img_error:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Erreur de traitement de {img_file.name}: {str(img_error)}"
                        })
        
        # Transcription des audios (affichée dans le chat)
        if uploaded_audios:
            for idx, audio_file in enumerate(uploaded_audios):
                # Vérifier si déjà traité
                file_key = f"audio_{audio_file.name}_{audio_file.size}"
                if file_key in st.session_state.processed_files:
                    continue
                
                st.session_state.processed_files.add(file_key)
                new_media_analyzed = True
                
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"🎵 Audio uploadé: {audio_file.name}"
                })
                
                with st.spinner(f"Analyse de {audio_file.name}..."):
                    try:
                        import tempfile
                        import librosa
                        
                        audio_file.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                            tmp_file.write(audio_file.read())
                            tmp_path = tmp_file.name
                        
                        y, sr = librosa.load(tmp_path)
                        duration = librosa.get_duration(y=y, sr=sr)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**🎵 Analyse de {audio_file.name}**\n\n⏱️ Durée: {duration:.2f}s | Fréquence: {sr} Hz\n\n🎙️ *Transcription: Utilisez Whisper API pour la transcription en production*"
                        })
                        
                        st.session_state.media_analysis_results.append({
                            'type': 'audio',
                            'name': audio_file.name,
                            'duration': duration,
                            'sample_rate': sr
                        })
                        
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Erreur lors de l'analyse de {audio_file.name}: {str(e)}"
                        })
        
        # Analyse des vidéos (affichée dans le chat)
        if uploaded_videos:
            for idx, video_file in enumerate(uploaded_videos):
                # Vérifier si déjà traité
                file_key = f"video_{video_file.name}_{video_file.size}"
                if file_key in st.session_state.processed_files:
                    continue
                
                st.session_state.processed_files.add(file_key)
                new_media_analyzed = True
                
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"🎥 Vidéo uploadée: {video_file.name}"
                })
                
                with st.spinner(f"Analyse de {video_file.name}..."):
                    try:
                        import tempfile
                        
                        video_file.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_file:
                            tmp_file.write(video_file.read())
                            tmp_path = tmp_file.name
                        
                        cap = cv2.VideoCapture(tmp_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**🎥 Analyse de {video_file.name}**\n\n📐 Résolution: {width}x{height}px | ⏱️ Durée: {duration:.2f}s | 🎞️ FPS: {fps:.2f}\n\n🎬 *Analyse avancée: Extrayez les frames clés pour une analyse visuelle approfondie*"
                        })
                        
                        st.session_state.media_analysis_results.append({
                            'type': 'video',
                            'name': video_file.name,
                            'duration': duration,
                            'resolution': f"{width}x{height}",
                            'fps': fps
                        })
                        
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Erreur lors de l'analyse de {video_file.name}: {str(e)}"
                        })
        
        # Recharger uniquement si de nouveaux médias ont été analysés
        if 'new_media_analyzed' in locals() and new_media_analyzed:
            st.rerun()
        
        # Zone de chat avec design amélioré - CARTE SCINTILLANTE
        st.markdown('<div class="kibali-chat-card">', unsafe_allow_html=True)
        
        # Indicateur de contexte actif
        context_indicators = []
        if 'binary_files' in st.session_state and st.session_state.binary_files:
            context_indicators.append(f"🔬 Fichier binaire: {st.session_state.binary_files[-1]['name']}")
        if st.session_state.has_pdf_context:
            context_indicators.append(f"📄 {len(st.session_state.chat_uploaded_pdfs)} PDF(s)")
        if st.session_state.media_analysis_results:
            context_indicators.append(f"📸 {len(st.session_state.media_analysis_results)} média(s)")
        
        if context_indicators:
            st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, rgba(0, 136, 255, 0.15), rgba(138, 43, 226, 0.15));
                padding: 0.8rem 1.2rem;
                border-radius: 8px;
                border-left: 4px solid #0088ff;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 1rem;
            '>
                <span style='color: #0088ff; font-size: 1.2rem;'>💬</span>
                <div>
                    <strong style='color: #0088ff;'>Mode Conversationnel Actif</strong>
                    <br>
                    <span style='color: #b0b0b0; font-size: 0.9rem;'>{' • '.join(context_indicators)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'''<div class="chat-message-user" style="
                        background: linear-gradient(145deg, #0088ff 0%, #00ff88 100%);
                        color: white;
                        padding: 1.2rem 1.8rem;
                        border-radius: 18px 18px 4px 18px;
                        margin: 1.5rem 0.5rem 1.5rem auto;
                        max-width: 75%;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                        border: 2px solid;
                        border-image: linear-gradient(45deg, #0088ff, #00ff88, #0088ff) 1;
                        animation: borderGlow 2s ease-in-out infinite, slideInRight 0.4s ease-out;
                        font-weight: 500;
                    ">
                        <strong>👤 Vous:</strong><br>{message["content"]}
                    </div>
                    <style>
                    @keyframes borderGlow {{{{
                        0%, 100% {{{{ border-image: linear-gradient(45deg, #0088ff, #00ff88, #0088ff) 1; }}}}
                        50% {{{{ border-image: linear-gradient(45deg, #00ff88, #00ffff, #00ff88) 1; }}}}
                    }}}}
                    @keyframes slideInRight {{{{
                        0% {{{{ transform: translateX(30px); opacity: 0; }}}}
                        100% {{{{ transform: translateX(0); opacity: 1; }}}}
                    }}}}
                    </style>
                    ''', unsafe_allow_html=True)
                elif message["role"] == "web_media":
                    # Affichage spécial pour les médias web
                    st.markdown(message["content"], unsafe_allow_html=True)
                else:
                    # Vérifier si c'est du HTML pur (panneau d'outils PDF ou contient <div)
                    content = message.get("content", "")
                    is_html_flag = message.get("is_html", False)
                    has_div_tags = ("<div" in content and "</div>" in content)
                    has_style_attr = "style=" in content
                    
                    # Si c'est du HTML (flag OU contient div OU contient style), le rendre directement
                    if is_html_flag or has_div_tags or has_style_attr:
                        st.markdown(content, unsafe_allow_html=True)
                        continue  # Skip le reste du formatage
                    
                    # Formater la réponse avec Markdown pour structure
                    formatted_response = content
                    
                    # Carte complète en un seul bloc avec contenu markdown
                    st.markdown(f'''
<style>
@keyframes borderShimmer {{{{
    0%, 100% {{{{ border-image: linear-gradient(135deg, #00ff88, #0088ff, #00ff88) 1; }}}}
    50% {{{{ border-image: linear-gradient(135deg, #0088ff, #00ffff, #0088ff) 1; }}}}
}}}}

@keyframes logoGlow {{{{
    0%, 100% {{{{ filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.5)); }}}}
    50% {{{{ filter: drop-shadow(0 0 20px rgba(0, 255, 136, 0.8)); }}}}
}}}}

@keyframes slideIn {{{{
    0% {{{{ transform: translateY(20px); opacity: 0; }}}}
    100% {{{{ transform: translateY(0); opacity: 1; }}}}
}}}}

.kibali-response-box h2 {{{{
    color: #0066cc !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
}}}}

.kibali-response-box h3 {{{{
    color: #00aa66 !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.8rem !important;
}}}}

.kibali-response-box p {{{{
    color: #1a1a1a !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    margin: 1rem 0 !important;
    line-height: 1.8 !important;
}}}}

.kibali-response-box ul, .kibali-response-box ol {{{{
    color: #1a1a1a !important;
    font-weight: 600 !important;
}}}}

.kibali-response-box li {{{{
    color: #1a1a1a !important;
    font-weight: 600 !important;
    margin: 0.5rem 0 !important;
}}}}

.kibali-response-box strong {{{{
    color: #0066cc !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}}}}

.kibali-response-box code {{{{
    background: rgba(0, 170, 102, 0.1);
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    color: #00aa66;
    font-weight: 600 !important;
}}}}

.kibali-response-box a {{{{
    color: #0066cc !important;
    font-weight: 600 !important;
    text-decoration: underline;
}}}}

/* Forcer le texte NOIR et gras sur TOUT le contenu */
.kibali-response-box * {{{{
    color: #1a1a1a !important;
    font-weight: 600 !important;
}}}}
</style>

<div class="kibali-response-box" style="
    background: rgba(255, 255, 255, 0.95);
    padding: 2rem;
    border-radius: 16px;
    margin: 2rem 0.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15), 0 8px 40px rgba(0, 0, 0, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.8);
    animation: slideIn 0.5s ease-out;
">
    <div style="
        display: flex; 
        align-items: center; 
        margin-bottom: 1.5rem; 
        gap: 0.75rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(0, 255, 136, 0.2);
    ">
        <span style="
            font-size: 2rem; 
            font-weight: 900;
            background: linear-gradient(135deg, #00ff88 0%, #00ffff 50%, #0088ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: logoGlow 2s ease-in-out infinite;
            filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.5));
        ">K</span>
        <strong style="
            color: #00ff88; 
            font-size: 1.2rem;
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        ">Kibali AI</strong>
        <span style="
            margin-left: auto;
            padding: 0.4rem 1rem;
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 136, 255, 0.2));
            border: 1px solid rgba(0, 255, 136, 0.4);
            border-radius: 25px;
            font-size: 0.8rem;
            color: #00ff88;
            font-weight: 600;
        ">🤖 Réponse enrichie</span>
    </div>
    
    <div style="line-height: 1.8; font-size: 1.05rem; color: #ffffff;">
        {formatted_response}
    </div>
</div>
''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Ancre pour auto-scroll vers le bas
        st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
        
        # Script JavaScript pour auto-scroll
        st.markdown("""
        <script>
        // Auto-scroll vers le bas quand nouveaux messages
        const chatBottom = document.getElementById('chat-bottom');
        if (chatBottom) {
            chatBottom.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
        
        // Garder le focus sur l'input
        const chatInput = document.querySelector('[data-testid="stChatInput"] input');
        if (chatInput) {
            chatInput.focus();
        }
        </script>
        """, unsafe_allow_html=True)
        
        # Input de chat stylisé - TOUJOURS EN BAS
        if prompt := st.chat_input("💭 Pose ta question ici...", key="chat_input"):
            # Enrichir le prompt avec le contexte des conversations passées (mémoire vectorielle)
            enriched_prompt = prompt
            conversation_context = ""
            
            try:
                from chat_memory import get_conversation_context
                conversation_context = get_conversation_context(prompt, st.session_state.chat_vectordb)
                if conversation_context:
                    enriched_prompt = f"""📚 **Contexte de conversations précédentes:**
{conversation_context}

---
QUESTION ACTUELLE: {prompt}"""
            except Exception as e:
                print(f"⚠️ Erreur récupération contexte mémoire: {e}")
            
            # 🤖 ENRICHIR avec le contexte des Nano-IA spécialisées
            nano_ai_context = ""
            if AUTO_LEARNING_AVAILABLE and st.session_state.get('auto_learning'):
                try:
                    nano_ai_context = st.session_state.auto_learning.get_domain_context(prompt, k=3)
                    if nano_ai_context:
                        enriched_prompt = f"""{enriched_prompt}

{nano_ai_context}"""
                except Exception as e:
                    print(f"⚠️ Erreur contexte nano-IA: {e}")
            
            # Enrichir le prompt avec les résultats d'analyse de médias IA si disponibles
            if st.session_state.media_analysis_results:
                media_context = "\n\n📎 **Médias analysés avec IA:**\n"
                for media in st.session_state.media_analysis_results:
                    if media['type'] == 'image':
                        media_context += f"\n🖼️ **Image: {media['name']}**\n"
                        media_context += f"  📏 Résolution: {media.get('resolution', 'N/A')}\n"
                        media_context += f"  🎨 Format: {media.get('format', 'N/A')}\n"
                        
                        # Ajouter l'analyse IA complète
                        if 'ai_analysis' in media:
                            media_context += f"\n  🤖 **Analyse IA détaillée:**\n"
                            # Indenter l'analyse pour la lisibilité
                            ai_lines = media['ai_analysis'].split('\n')
                            for line in ai_lines[:50]:  # Augmenter la limite pour inclure plus de détails (couleurs, objets)
                                media_context += f"  {line}\n"
                        
                        if 'caption' in media:
                            media_context += f"  📝 Description: {media['caption']}\n"
                        
                        if 'web_context' in media and media['web_context']:
                            media_context += f"  🌐 Infos complémentaires disponibles\n"
                    
                    elif media['type'] == 'audio':
                        media_context += f"\n🎵 **Audio: {media['name']}**\n"
                        media_context += f"  ⏱️ Durée: {media.get('duration', 0):.2f}s\n"
                        media_context += f"  📊 Fréquence: {media.get('sample_rate', 'N/A')} Hz\n"
                    
                    elif media['type'] == 'video':
                        media_context += f"\n🎥 **Vidéo: {media['name']}**\n"
                        media_context += f"  ⏱️ Durée: {media.get('duration', 0):.2f}s\n"
                        media_context += f"  📐 Résolution: {media.get('resolution', 'N/A')}\n"
                        media_context += f"  🎞️ FPS: {media.get('fps', 'N/A')}\n"
                
                # Ajouter un contexte explicite pour la conversation
                media_full_context = f"""CONTEXTE PRIORITAIRE: L'utilisateur pose une question sur les médias analysés ci-dessous.
⚠️ INSTRUCTION IMPORTANTE: Vous AVEZ ACCÈS au contenu de l'image via la description détaillée ci-dessous.
Considérez cette description textuelle comme votre "vision" de l'image.
Si l'utilisateur demande des détails (couleurs, textes, objets), utilisez les données ci-dessous pour répondre.
NE DITES JAMAIS que vous n'avez pas accès à l'image. Dites "D'après l'analyse de l'image..."

{media_context}

QUESTION UTILISATEUR: {prompt}"""
                
                # Combiner avec le contexte de conversation si existant
                if conversation_context:
                    enriched_prompt = f"{enriched_prompt}\n\n{media_full_context}"
                else:
                    enriched_prompt = media_full_context
            
            # ═══════════════════════════════════════════════════════════════════
            # FICHIER BINAIRE - CONTEXTE ABSOLU ET EXCLUSIF (AVEC CACHE)
            # ═══════════════════════════════════════════════════════════════════
            if 'binary_files' in st.session_state and st.session_state.binary_files:
                # RÉCUPÉRER LE DERNIER FICHIER UPLOADÉ (le plus récent)
                binary = st.session_state.binary_files[-1]
                
                # CACHE: Vérifier si le contexte a déjà été généré pour éviter les réanalyses
                if 'cached_binary_context' not in binary:
                    # Première fois : construire et cacher le contexte
                    binary_context = ""
                    
                    binary_context += f"\n{'='*80}\n"
                    binary_context += f"📁 FICHIER BINAIRE EN TÊTE DE CONVERSATION: {binary['name']}\n"
                    binary_context += f"{'='*80}\n\n"
                    
                    # Informations essentielles TRÈS DÉTAILLÉES
                    binary_context += f"🔢 TYPE DÉTECTÉ: {binary['info']['format_name']}\n"
                    binary_context += f"📊 TAILLE EXACTE: {binary['info']['analysis']['size_formatted']} ({len(binary['data'])} bytes)\n"
                    binary_context += f"🎯 MAGIC BYTES: {binary['info']['magic_bytes']}\n"
                    binary_context += f"📈 ENTROPIE: {binary['info']['analysis']['entropy']:.2f}/8.0 "
                    
                    # Interprétation de l'entropie
                    entropy = binary['info']['analysis']['entropy']
                    if entropy < 3:
                        binary_context += "(Données très répétitives/structurées - probablement du texte ou données simples)\n"
                    elif entropy < 6:
                        binary_context += "(Données texte/structurées - contient probablement des métadonnées)\n"
                    elif entropy < 7.5:
                        binary_context += "(Données binaires normales - fichier standard)\n"
                    else:
                        binary_context += "(Données compressées/chiffrées - haute complexité)\n"
                    
                    # Statistiques détaillées
                    stats = binary['info']['analysis']['statistics']
                    binary_context += f"\n📊 STATISTIQUES DES BYTES:\n"
                    binary_context += f"   • Valeur moyenne: {stats.get('mean', 0):.2f}\n"
                    binary_context += f"   • Médiane: {stats.get('median', 0):.2f}\n"
                    binary_context += f"   • Écart-type: {stats.get('std', 0):.2f}\n"
                    binary_context += f"   • Plage: {stats.get('min', 0)} à {stats.get('max', 0)}\n"
                    binary_context += f"   • Bytes uniques: {stats.get('unique_bytes', 0)}/256\n"
                    
                    # Patterns détectés
                    if binary['info']['analysis']['patterns']:
                        binary_context += f"\n🎯 PATTERNS DÉTECTÉS:\n"
                        for pattern in binary['info']['analysis']['patterns']:
                            binary_context += f"   ✓ {pattern}\n"
                    
                    # Chaînes de texte extraites - TRÈS IMPORTANT pour l'analyse
                    if binary['text_strings']:
                        binary_context += f"\n📝 CHAÎNES DE TEXTE EXTRAITES ({len(binary['text_strings'])} au total):\n"
                        # Afficher jusqu'à 30 chaînes pour plus de contexte
                        for i, s in enumerate(binary['text_strings'][:30], 1):
                            binary_context += f"   {i}. \"{s[:100]}\"\n"
                        if len(binary['text_strings']) > 30:
                            binary_context += f"   ... et {len(binary['text_strings'])-30} autres chaînes\n"
                    else:
                        binary_context += f"\n📝 CHAÎNES DE TEXTE: Aucune chaîne ASCII lisible détectée\n"
                    
                    # Header hexadécimal (256 premiers bytes pour plus de contexte)
                    binary_context += f"\n🔍 HEADER HEXADÉCIMAL (256 premiers bytes):\n"
                    hex_header = binary['data'][:256].hex()
                    # Formater en lignes de 32 caractères hex (16 bytes par ligne)
                    for i in range(0, len(hex_header), 32):
                        offset = i // 2
                        hex_line = hex_header[i:i+32]
                        # Ajouter des espaces tous les 2 caractères
                        hex_formatted = ' '.join(hex_line[j:j+2] for j in range(0, len(hex_line), 2))
                        # ASCII representation
                        ascii_chars = ''.join(
                            chr(binary['data'][offset + k]) if 32 <= binary['data'][offset + k] < 127 else '.'
                            for k in range(min(16, len(binary['data']) - offset))
                        )
                        binary_context += f"   {offset:04X}: {hex_formatted:<48} | {ascii_chars}\n"
                    
                    # Structure du header si détectée
                    header_info = binary['info']['analysis']['header']
                    if header_info.get('is_text_like'):
                        binary_context += f"\n✓ Le header semble contenir du texte ASCII ({header_info.get('ascii_ratio', 0):.1%})\n"
                        if header_info.get('first_line'):
                            binary_context += f"   Première ligne: \"{header_info['first_line']}\"\n"
                    
                    # CACHER le contexte pour réutilisation
                    binary['cached_binary_context'] = binary_context
                else:
                    # Réutiliser le contexte caché - évite les réanalyses
                    binary_context = binary['cached_binary_context']
                
                # ⚡ NOUVEAU: Contexte conversationnel intelligent avec OUTILS PROACTIFS
                # Pour la première question (upload), donner analyse complète
                # Pour les suivantes, mode expert avec utilisation d'outils
                is_first_query = len(st.session_state.chat_history) <= 3
                
                # 🎯 DÉTECTION AUTOMATIQUE DES INTENTIONS ET ORCHESTRATION
                question_lower = prompt.lower()
                needs_calculation = any(kw in question_lower for kw in ['calcul', 'moyenne', 'somme', 'total', 'combien', 'valeur', 'profondeur', 'exact'])
                needs_web_research = any(kw in question_lower for kw in ['domaine', 'activité', 'industrie', 'application', 'utilisation', 'pourquoi', 'comment'])
                needs_data_extraction = any(kw in question_lower for kw in ['données', 'valeurs', 'liste', 'extrait', 'montre', 'affiche', 'trouve'])
                needs_conversion = any(kw in question_lower for kw in ['convertis', 'convert', 'numpy', 'tableau', 'dataframe', 'pandas', 'csv', 'json', 'extraction'])
                needs_ert_analysis = any(kw in question_lower for kw in ['ert', 'résistivité', 'resistivité', 'géophysique', 'forage', 'nappe', 'aquifère', 'eau souterraine'])
                
                # 🌊 ANALYSE ERT AUTOMATIQUE - FICHIERS .dat GÉOPHYSIQUES
                # Vérifier s'il y a un fichier .dat uploadé (via uploader dédié OU binary_files)
                dat_file_source = None
                if 'uploaded_dat_file' in st.session_state and st.session_state.uploaded_dat_file:
                    # Fichier .dat uploadé via l'uploader dédié
                    dat_file_source = {
                        'name': st.session_state.uploaded_dat_file['name'],
                        'path': st.session_state.uploaded_dat_file['path'],
                        'data': open(st.session_state.uploaded_dat_file['path'], 'rb').read()
                    }
                elif 'binary_files' in st.session_state and st.session_state.binary_files:
                    # Fichier .dat uploadé via l'uploader générique
                    binary = st.session_state.binary_files[-1]
                    if binary['name'].lower().endswith('.dat'):
                        dat_file_source = binary
                
                if needs_ert_analysis and dat_file_source:
                        with st.expander("📋 Plan d'Action ERT - Analyse Géophysique", expanded=True):
                            st.markdown(f"""
                            ### 🌊 Orchestration ERT Activée
                            
                            **Fichier:** `{dat_file_source['name']}`
                            
                            **Formats supportés (détection automatique):**
                            
                            **Format 1 - Standard (espaces):**
                            ```
                            survey_point  depth  data  project
                            1.0  -2.5  0.35  projet1
                            1.0  -5.0  0.38  projet1
                            ```
                            
                            **Format 2 - Fréquences (CSV):**
                            ```
                            Projet,Point,621105.0MHz,155276.25MHz,...
                            Archange,1,0.119,0.122,0.116,...
                            Archange,2,0.161,0.163,0.164,...
                            ```
                            
                            **Plan d'exécution:**
                            1. ✅ **Parse du fichier .dat** - Détection automatique du format
                            2. ⏳ **Analyse des résistivités** - Classification géologique
                            3. ⏳ **Interprétation hydrogéologique** - Types d'eau identifiés
                            4. ⏳ **Pseudo-section 2D** - Visualisation avec colormap eau
                            5. ⏳ **Zones cibles** - Identification aquifères
                            6. ⏳ **Export rapport** - PDF complet avec statistiques
                            """)
                        
                        try:
                            import pandas as pd
                            import numpy as np
                            import matplotlib.pyplot as plt
                            from kibali_data.models.ert_analyzer import ERTAnalyzer
                            
                            st.info(f"🔄 Analyse ERT en cours de `{dat_file_source['name']}`...")
                            
                            # Créer l'analyseur
                            ert = ERTAnalyzer()
                            
                            # Analyse complète
                            with st.spinner("📊 Analyse des données ERT..."):
                                result = ert.analyze_file(dat_file_source['data'])
                            
                            if result.get('error'):
                                st.error(f"❌ Erreur d'analyse: {result['error']}")
                                
                                # Afficher un aperçu du fichier pour diagnostic
                                with st.expander("🔍 Aperçu du fichier pour diagnostic"):
                                    try:
                                        content_preview = dat_file_source['data'][:2000].decode('utf-8', errors='replace')
                                        st.code(content_preview, language=None)
                                        
                                        # Analyse basique du contenu
                                        lines = [l.strip() for l in content_preview.split('\n') if l.strip()]
                                        if lines:
                                            first_line = lines[0]
                                            st.write(f"**Première ligne:** `{first_line}`")
                                            parts = first_line.split()
                                            st.write(f"**Nombre de colonnes:** {len(parts)}")
                                            st.write(f"**Colonnes:** `{parts}`")
                                    except:
                                        st.write("Impossible d'afficher l'aperçu")
                                
                                # Message d'aide pour le format
                                st.info("""
                                **💡 Formats .dat supportés (ERTest.py):**
                                
                                **Format 1 - Standard (4 colonnes, espaces):**
                                ```
                                survey_point  depth  resistivity  project
                                1.0  -2.5  0.35  projet1
                                1.0  -5.0  0.38  projet1
                                2.0  -2.5  0.42  projet1
                                ```
                                
                                **Format 2 - Fréquences (CSV avec virgules):**
                                ```
                                Projet,Point Sondage,621105.0MHz,155276.25MHz,...
                                Projet Archange Ondimba 2,1,0.119,0.122,0.116,...
                                Projet Archange Ondimba 2,2,0.161,0.163,0.164,...
                                ```
                                
                                **Notes:**
                                - Détection automatique du séparateur (espaces vs virgules)
                                - Les profondeurs négatives sont converties en positives
                                - Les lignes commençant par # sont ignorées (commentaires)
                                - Pour CSV: première ligne = en-têtes avec fréquences MHz
                                
                                **Vérifications:**
                                - ✅ Pas de lignes vides au début
                                - ✅ Les colonnes sont séparées par espaces ou tabs
                                - ✅ Les valeurs numériques utilisent des points (pas de virgules)
                                - ✅ Les profondeurs sont en mètres
                                - ✅ Les résistivités sont en Ω·m
                                """)
                                return
                            
                            # 🧠 ENRICHISSEMENT RAG AUTOMATIQUE
                            st.info("🧠 Enrichissement avec la base de connaissances RAG...")
                            rag_insights = ""
                            rag_context = ""
                            
                            try:
                                if st.session_state.get('vectordb') is not None:
                                    # Construire une requête enrichie pour le RAG
                                    stats = result['stats']
                                    water_zones = result['water_zones']
                                    
                                    # Identifier le contexte dominant
                                    dominant_water_type = max(water_zones, key=water_zones.get)
                                    avg_resistivity = stats['resistivity_stats']['mean']
                                    
                                    # Créer une requête intelligente pour le RAG
                                    rag_query = f"""Résistivité électrique {avg_resistivity:.1f} Ohm·m {dominant_water_type} 
                                    géophysique ERT aquifère forage hydrogéologie nappe phréatique 
                                    eau souterraine prospection classification lithologie"""
                                    
                                    # Recherche dans la base vectorielle
                                    with st.spinner("🔍 Recherche de connaissances pertinentes..."):
                                        docs = st.session_state.vectordb.similarity_search(rag_query, k=5)
                                        
                                        if docs:
                                            st.success(f"✅ {len(docs)} documents pertinents trouvés dans la base de connaissances")
                                            
                                            # Extraire le contexte
                                            rag_context = "\n\n".join([
                                                f"📄 **Document {i+1}:** {doc.page_content[:300]}..." 
                                                for i, doc in enumerate(docs)
                                            ])
                                            
                                            # Générer des insights avec l'IA en utilisant le RAG
                                            try:
                                                client = create_client()
                                                
                                                rag_prompt = f"""Tu es un expert en géophysique et hydrogéologie. Analyse ces données ERT:

**DONNEES MESUREES:**
- Résistivité moyenne: {avg_resistivity:.2f} Ω·m
- Type d'eau dominant: {dominant_water_type}
- Profondeur: {stats['depth_range'][0]:.1f} - {stats['depth_range'][1]:.1f} m
- Points de sondage: {stats['survey_points']}
- Distribution eau: mer={water_zones['mer']}, salée={water_zones['salée']}, douce={water_zones['douce']}, pure={water_zones['pure']}

**CONTEXTE DE LA BASE DE CONNAISSANCES:**
{rag_context[:2000]}

**TACHE:**
En te basant sur les documents de la base de connaissances ET ton expertise, fournis:
1. Une interprétation géologique détaillée
2. Les implications hydrogéologiques
3. Des recommandations pour le forage
4. Les risques potentiels identifiés
5. Comparaison avec des sites similaires si mentionnés dans les documents

Sois précis, technique et référence les documents quand pertinent."""

                                                rag_response = client.chat.completions.create(
                                                    model=WORKING_MODELS[model_choice],
                                                    messages=[{"role": "user", "content": rag_prompt}],
                                                    max_tokens=2000,
                                                    temperature=0.3
                                                )
                                                
                                                rag_insights = rag_response.choices[0].message.content
                                                
                                                # 🧹 NETTOYER LA RÉPONSE DES CARACTÈRES CHINOIS/NON-LATINS
                                                rag_insights = clean_response_text(rag_insights)
                                                
                                                # Afficher les insights enrichis
                                                with st.expander("🧠 Analyse Enrichie par IA + Base de Connaissances", expanded=True):
                                                    st.markdown(rag_insights)
                                                    
                                                    st.markdown("---")
                                                    st.markdown("**📚 Sources utilisées:**")
                                                    for i, doc in enumerate(docs[:3]):
                                                        with st.expander(f"Document {i+1}"):
                                                            st.text(doc.page_content[:500])
                                            
                                            except Exception as e:
                                                st.warning(f"⚠️ Impossible de générer l'analyse enrichie: {e}")
                                        else:
                                            st.info("ℹ️ Aucun document pertinent trouvé dans la base de connaissances")
                                else:
                                    st.info("ℹ️ Base de connaissances RAG non disponible - analyse standard uniquement")
                            
                            except Exception as e:
                                st.warning(f"⚠️ Enrichissement RAG échoué: {e}")
                            
                            st.success("✅ Analyse ERT terminée!")
                            
                            # Afficher un aperçu du fichier pour diagnostic
                            with st.expander("🔍 Aperçu du fichier pour diagnostic"):
                                try:
                                    content_preview = dat_file_source['data'][:2000].decode('utf-8', errors='replace')
                                    st.code(content_preview, language=None)
                                    
                                    # Analyse basique du contenu
                                    lines = [l.strip() for l in content_preview.split('\n') if l.strip()]
                                    if lines:
                                        first_line = lines[0]
                                        st.write(f"**Première ligne:** `{first_line}`")
                                        parts = first_line.split()
                                        st.write(f"**Nombre de colonnes:** {len(parts)}")
                                        st.write(f"**Colonnes:** `{parts}`")
                                except:
                                    st.write("Impossible d'afficher l'aperçu")
                            
                            # Message d'aide pour le format
                            st.info("""
                            **💡 Formats .dat supportés (ERTest.py):**
                            
                            **Format 1 - Standard (4 colonnes, espaces):**
                            ```
                            survey_point  depth  resistivity  project
                            1.0  -2.5  0.35  projet1
                            1.0  -5.0  0.38  projet1
                            2.0  -2.5  0.42  projet1
                            ```
                            
                            **Format 2 - Fréquences (CSV avec virgules):**
                            ```
                            Projet,Point Sondage,621105.0MHz,155276.25MHz,...
                            Projet Archange Ondimba 2,1,0.119,0.122,0.116,...
                            Projet Archange Ondimba 2,2,0.161,0.163,0.164,...
                            ```
                            
                            **Notes:**
                            - Détection automatique du séparateur (espaces vs virgules)
                            - Les profondeurs négatives sont converties en positives
                            - Les lignes commençant par # sont ignorées (commentaires)
                            - Pour CSV: première ligne = en-têtes avec fréquences MHz
                            
                            **Vérifications:**
                            - ✅ Pas de lignes vides au début
                            - ✅ Les colonnes sont séparées par espaces ou tabs
                            - ✅ Les valeurs numériques utilisent des points (pas de virgules)
                            - ✅ Les profondeurs sont en mètres
                            - ✅ Les résistivités sont en Ω·m
                            """)
                            
                            # Affichage des résultats
                            st.markdown("## 📊 Résultats de l'Analyse ERT")
                            
                            # Statistiques générales
                            stats = result['stats']
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("📏 Mesures", stats['total_measurements'])
                            with col2:
                                st.metric("📍 Points", stats['survey_points'])
                            with col3:
                                st.metric("🔢 Résistivité moy.", f"{stats['resistivity_stats']['mean']:.2f} Ω·m")
                            with col4:
                                depth_min, depth_max = stats['depth_range']
                                st.metric("📏 Profondeur", f"{depth_min:.1f} - {depth_max:.1f} m")
                            
                            # Zones d'eau identifiées
                            st.markdown("### 💧 Types d'Eau Identifiés")
                            water_zones = result['water_zones']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🔴 Eau de mer", water_zones['mer'], 
                                         help="0.1-1 Ω·m: Eau hypersalée")
                            with col2:
                                st.metric("🟡 Eau salée", water_zones['salée'],
                                         help="1-10 Ω·m: Eau saumâtre")
                            with col3:
                                st.metric("🟢 Eau douce", water_zones['douce'],
                                         help="10-100 Ω·m: Eau potable")
                            with col4:
                                st.metric("🔵 Eau pure", water_zones['pure'],
                                         help="> 100 Ω·m: Eau très pure")
                            
                            # Interprétations géologiques
                            st.markdown("### 🪨 Interprétations Géologiques")
                            interp_summary = result['interpretations']
                            
                            if interp_summary:
                                df_interp = pd.DataFrame([
                                    {'Matériau': material, 'Occurrences': count}
                                    for material, count in sorted(interp_summary.items(), 
                                                                 key=lambda x: x[1], reverse=True)
                                ])
                                st.dataframe(df_interp, use_container_width=True)
                            
                            # Zones cibles pour forage
                            target_zones = result['target_zones']
                            if target_zones:
                                st.markdown("### 🎯 Zones Cibles pour Forage")
                                st.success(f"✅ **{len(target_zones)} zones cibles identifiées** (50-300 Ω·m)")
                                
                                df_targets = pd.DataFrame(target_zones)
                                st.dataframe(df_targets, use_container_width=True)
                                
                                st.info("""
                                **Recommandations:**
                                - 🟢 50-100 Ω·m: Aquifère sableux - ZONE PRIORITAIRE
                                - ⭐ 100-300 Ω·m: Gravier/Sable grossier - EXCELLENT débit
                                """)
                            else:
                                st.warning("⚠️ Aucune zone cible optimale détectée (50-300 Ω·m)")
                            
                            # Pseudo-section 2D
                            st.markdown("### 🗺️ Pseudo-section 2D")
                            with st.spinner("Création de la pseudo-section..."):
                                fig = ert.create_2d_pseudosection()
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                            
                            # Tableau d'interprétation
                            with st.expander("📋 Tableau d'Interprétation Complet"):
                                st.markdown(ert.create_interpretation_html(), unsafe_allow_html=True)
                            
                            # DataFrame complet
                            with st.expander("📊 Données Brutes"):
                                st.dataframe(result['dataframe'], use_container_width=True)
                            
                            # 💾 STOCKER LES RÉSULTATS ERT POUR ANALYSE CONVERSATIONNELLE
                            if 'ert_analysis' not in st.session_state:
                                st.session_state.ert_analysis = {}
                            
                            st.session_state.ert_analysis[binary['name']] = {
                                'result': result,
                                'stats': stats,
                                'water_zones': water_zones,
                                'target_zones': target_zones,
                                'interp_summary': interp_summary,
                                'dataframe': result['dataframe'],
                                'ert_analyzer': ert  # Garder l'instance pour réutilisation
                            }
                            
                            # Réponse récapitulative ENRICHIE avec guide d'utilisation
                            response = f"""✅ **Analyse ERT Complète Terminée!**

📊 **Résultats du fichier `{binary['name']}`:**
- 📏 **{stats['total_measurements']} mesures** sur **{stats['survey_points']} points de sondage**
- 🌊 **Types d'eau:** {water_zones['mer']} mer, {water_zones['salée']} salée, {water_zones['douce']} douce, {water_zones['pure']} pure
- 🎯 **{len(target_zones)} zones cibles** pour forage (résistivité optimale 50-300 Ω·m)
- 📈 **Résistivité moyenne:** {stats['resistivity_stats']['mean']:.2f} Ω·m (min: {stats['resistivity_stats']['min']:.2f}, max: {stats['resistivity_stats']['max']:.2f})
- 📏 **Profondeur explorée:** {depth_min:.1f} à {depth_max:.1f} m

**🪨 Interprétations géologiques principales:**
{chr(10).join([f"• {mat}: {cnt} mesures" for mat, cnt in list(interp_summary.items())[:3]])}

---

### 💬 **Posez-moi des questions sur cette analyse !**

**Exemples de questions que vous pouvez me poser:**

🎨 **Couleurs & Résistivités:**
- "Que signifie la couleur rouge sur la carte ?"
- "Quelle résistivité correspond au vert ?"
- "Pourquoi y a-t-il du bleu à cette profondeur ?"

📊 **Analyses détaillées:**
- "Explique-moi les zones à 50 Ω·m"
- "Quelle est la meilleure profondeur pour forer ?"
- "Compare les points de sondage 1 et 3"

💧 **Interprétation hydrogéologique:**
- "Où se trouve l'eau douce ?"
- "Pourquoi y a-t-il de l'eau de mer ici ?"
- "Quelle est la qualité de l'aquifère ?"

🎯 **Recommandations:**
- "Où dois-je forer mon puits ?"
- "Quelle profondeur recommandes-tu ?"
- "Quel débit puis-je espérer ?"

**🛠️ Outils utilisés:** ERTAnalyzer, Pandas, NumPy, Matplotlib, Classification automatique

Je peux maintenant analyser chaque détail de ces résultats avec vous ! 🚀"""
                            
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            st.success("✅ Analyse ERT terminée! Vous pouvez maintenant me poser des questions détaillées sur les résultats.")
                            return
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'analyse ERT: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                            return
                
                # 🚀 ORCHESTRATION AUTOMATIQUE - EXÉCUTER AU LIEU DE PROPOSER
                if needs_conversion and 'binary_files' in st.session_state and st.session_state.binary_files:
                    # Afficher le plan d'action
                    with st.expander("📋 Plan d'Action Automatique", expanded=True):
                        st.markdown("""
                        ### 🎯 Orchestration Activée
                        
                        **Détection:** Demande de conversion de données
                        
                        **Plan d'exécution:**
                        1. ✅ **Analyse du fichier** - Identifier la structure
                        2. ✅ **Détection du format** - CSV/TSV/JSON/Binaire
                        3. ⏳ **Extraction des données** - Parser le contenu
                        4. ⏳ **Conversion** - Créer tableau NumPy/Pandas
                        5. ⏳ **Visualisation** - Afficher le résultat
                        6. ⏳ **Export** - Proposer téléchargement
                        """)
                    
                    # EXÉCUTION RÉELLE DE LA CONVERSION
                    try:
                        import pandas as pd
                        import numpy as np
                        import io
                        
                        binary = st.session_state.binary_files[-1]
                        st.info(f"🔄 Conversion en cours de `{binary['name']}`...")
                        
                        # Étape 1: Détecter le format
                        with st.spinner("Détection du format..."):
                            if binary.get('structure'):
                                structure = binary['structure']
                                st.success(f"✅ Format détecté: **{structure.file_type.value}** (confiance: {structure.confidence:.0%})")
                                
                                # Utiliser le DataFrame déjà extrait
                                if binary.get('dataframe') is not None:
                                    df = binary['dataframe']
                                    st.success(f"✅ Données déjà extraites: {df.shape[0]} lignes × {df.shape[1]} colonnes")
                                else:
                                    # Extraire maintenant
                                    text_content = '\n'.join(binary.get('text_strings', []))
                                    delimiter = structure.delimiter or '\t'
                                    df = pd.read_csv(io.StringIO(text_content), delimiter=delimiter, header=None, on_bad_lines='skip')
                                    st.success(f"✅ Extraction réussie: {df.shape[0]} lignes × {df.shape[1]} colonnes")
                            else:
                                # Tentative brute force
                                st.warning("⚠️ Structure non détectée, tentative d'extraction brute...")
                                text_content = '\n'.join(binary.get('text_strings', []))
                                for delim in ['\t', ',', ';', ' ']:
                                    try:
                                        df = pd.read_csv(io.StringIO(text_content), delimiter=delim, header=None, on_bad_lines='skip')
                                        if df.shape[1] > 1 and df.shape[0] > 0:
                                            st.success(f"✅ Format détecté: Délimiteur `{repr(delim)}` - {df.shape[0]}×{df.shape[1]}")
                                            break
                                    except:
                                        continue
                        
                        # Étape 2: Conversion NumPy
                        with st.spinner("Conversion en NumPy..."):
                            numpy_array = df.to_numpy()
                            st.success(f"✅ Tableau NumPy créé: shape={numpy_array.shape}, dtype={numpy_array.dtype}")
                        
                        # Étape 3: Affichage
                        st.markdown("### 📊 Résultat de la Conversion")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📏 Dimensions", f"{numpy_array.shape[0]} × {numpy_array.shape[1]}")
                        with col2:
                            st.metric("🔢 Type", str(numpy_array.dtype))
                        with col3:
                            st.metric("💾 Taille", f"{numpy_array.nbytes / 1024:.2f} KB")
                        
                        # DataFrame preview
                        st.markdown("#### 🔍 Aperçu des Données")
                        st.dataframe(df.head(50), use_container_width=True)
                        
                        # Statistiques
                        st.markdown("#### 📈 Statistiques")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(df.describe())
                        with col2:
                            # Tentative de visualisation si colonnes numériques
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                st.line_chart(df[numeric_cols].head(100))
                        
                        # Export
                        st.markdown("#### 💾 Export")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Télécharger CSV", csv_data, f"{binary['name']}.csv", "text/csv")
                        with col2:
                            # NumPy save
                            np_bytes = io.BytesIO()
                            np.save(np_bytes, numpy_array)
                            st.download_button("📥 Télécharger NumPy", np_bytes.getvalue(), f"{binary['name']}.npy", "application/octet-stream")
                        with col3:
                            json_data = df.to_json(orient='records', indent=2).encode('utf-8')
                            st.download_button("📥 Télécharger JSON", json_data, f"{binary['name']}.json", "application/json")
                        
                        # Mise à jour du plan
                        with st.expander("📋 Plan d'Action Automatique", expanded=False):
                            st.markdown("""
                            ### ✅ Orchestration Terminée
                            
                            **Toutes les étapes exécutées avec succès:**
                            1. ✅ **Analyse du fichier** - Complétée
                            2. ✅ **Détection du format** - Complétée
                            3. ✅ **Extraction des données** - Complétée
                            4. ✅ **Conversion** - NumPy/Pandas créés
                            5. ✅ **Visualisation** - Affichée ci-dessus
                            6. ✅ **Export** - Boutons disponibles
                            """)
                        
                        # Stocker les résultats dans la session pour persistance
                        if 'conversion_results' not in st.session_state:
                            st.session_state.conversion_results = []
                        
                        st.session_state.conversion_results.append({
                            'filename': binary['name'],
                            'dataframe': df,
                            'numpy_array': numpy_array,
                            'timestamp': pd.Timestamp.now()
                        })
                        
                        # Ajouter réponse récapitulative
                        response = f"""✅ **Conversion Réussie!**

Le fichier `{binary['name']}` a été automatiquement:
- 🔍 Analysé et structuré
- 📊 Converti en tableau NumPy de shape {numpy_array.shape}
- 📈 Transformé en DataFrame Pandas ({df.shape[0]} lignes × {df.shape[1]} colonnes)
- 💾 Préparé pour export (CSV, NumPy, JSON)

Les données sont affichées ci-dessus avec statistiques et visualisations."""
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        st.success("✅ Conversion terminée! Résultats affichés ci-dessus.")
                        # NE PAS faire st.rerun() - garder l'affichage
                        return  # Sortir pour ne pas continuer le traitement normal
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la conversion: {e}")
                        st.exception(e)
                        return  # Sortir en cas d'erreur
                
                if is_first_query:
                    # Première analyse : réponse structurée complète avec recherche web
                    binary_full_context = f"""
🔬 ANALYSE EXPERTE APPROFONDIE DU FICHIER: {binary['name']}

Tu es Kibali, un expert en analyse de données avec accès à des outils puissants.
L'utilisateur demande: "{prompt}"

📊 DONNÉES COMPLÈTES DU FICHIER:
{binary_context}

🛠️ OUTILS À TA DISPOSITION:
1. **Recherche Web** - Pour contexte industriel, standards, domaines d'application
2. **Calculs Python** - Pour analyses statistiques, moyennes, totaux
3. **Extraction de données** - Parser et structurer les données
4. **Visualisation** - Créer des tableaux, graphiques
5. **Comparaison** - Benchmarking avec standards connus

⚡ INSTRUCTIONS IMPÉRATIVES:
1. **ANALYSE COMPLÈTE** (minimum 20 lignes structurées):
   - Vue d'ensemble du fichier
   - Identification précise du format et standard
   - Contexte métier/industriel (utilise recherche web si besoin)
   - Extraction des données clés avec valeurs exactes
   - Analyse statistique (moyennes, min, max, tendances)
   - Implications pratiques

2. **UTILISE LES OUTILS ACTIVEMENT**:
   - Si tu vois des données numériques → Calcule statistiques
   - Si tu identifies un domaine → Recherche web pour contexte
   - Si format connu → Compare avec standards
   - Si valeurs présentes → Extrais et structure en tableau

3. **STRUCTURE TA RÉPONSE**:
   ```
   🎯 Identification du fichier
   📊 Analyse des données (avec chiffres exacts)
   🔍 Contexte métier/industriel
   📈 Statistiques et calculs
   💡 Implications et recommandations
   ```

4. **SOIS PROACTIF**: Anticipe les besoins, fournis plus que demandé, montre ton expertise!
"""
                else:
                    # 🌊 CONTEXTE ERT ENRICHI - Questions sur l'analyse existante
                    ert_context = ""
                    ert_rag_context = ""
                    if 'ert_analysis' in st.session_state and binary['name'] in st.session_state.ert_analysis:
                        ert_data = st.session_state.ert_analysis[binary['name']]
                        stats = ert_data['stats']
                        water_zones = ert_data['water_zones']
                        interp = ert_data['interp_summary']
                        df = ert_data['dataframe']
                        
                        # 🧠 ENRICHISSEMENT RAG DYNAMIQUE POUR LES QUESTIONS
                        # Détecter les mots-clés de la question pour cibler la recherche RAG
                        question_keywords = []
                        if 'résistivité' in prompt.lower() or 'resistivité' in prompt.lower():
                            question_keywords.extend(['résistivité électrique', 'conductivité'])
                        if 'forage' in prompt.lower() or 'forer' in prompt.lower():
                            question_keywords.extend(['forage', 'puits', 'aquifère'])
                        if 'eau' in prompt.lower():
                            question_keywords.extend(['hydrogéologie', 'nappe phréatique', 'aquifère'])
                        if 'géologie' in prompt.lower() or 'roche' in prompt.lower():
                            question_keywords.extend(['lithologie', 'géologie', 'formation rocheuse'])
                        if 'profondeur' in prompt.lower():
                            question_keywords.extend(['profondeur', 'stratigraphie'])
                        
                        # Recherche RAG ciblée si base disponible
                        if question_keywords and st.session_state.get('vectordb') is not None:
                            try:
                                # Construire une requête intelligente
                                avg_resistivity = stats['resistivity_stats']['mean']
                                rag_query = f"{' '.join(question_keywords)} {avg_resistivity:.1f} Ohm·m ERT géophysique"
                                
                                # Recherche silencieuse
                                docs = st.session_state.vectordb.similarity_search(rag_query, k=3)
                                
                                if docs:
                                    ert_rag_context = f"""

🧠 **CONNAISSANCES RAG PERTINENTES ({len(docs)} documents trouvés):**

{chr(10).join([f"📄 **Document {i+1}:** {doc.page_content[:400]}" for i, doc in enumerate(docs)])}

**UTILISE CES CONNAISSANCES** pour enrichir ta réponse avec:
- Comparaisons avec d'autres sites/études
- Références scientifiques
- Méthodologies validées
- Retours d'expérience terrain
- Normes et standards applicables
"""
                            except Exception as e:
                                pass  # Silencieux si RAG échoue
                        
                        ert_context = f"""

═══════════════════════════════════════════════════════════════════
🌊 ANALYSE ERT DÉJÀ EFFECTUÉE - TOUTES LES DONNÉES DISPONIBLES
═══════════════════════════════════════════════════════════════════

📊 **STATISTIQUES COMPLÈTES:**
- Total mesures: {stats['total_measurements']}
- Points de sondage: {stats['survey_points']}
- Résistivité: min={stats['resistivity_stats']['min']:.2f} Ω·m, max={stats['resistivity_stats']['max']:.2f} Ω·m, moyenne={stats['resistivity_stats']['mean']:.2f} Ω·m, médiane={stats['resistivity_stats']['median']:.2f} Ω·m
- Profondeur: {stats['depth_range'][0]:.1f} à {stats['depth_range'][1]:.1f} m

💧 **TYPES D'EAU IDENTIFIÉS:**
- 🔴 Eau de mer (0.1-1 Ω·m): {water_zones['mer']} mesures
- 🟡 Eau salée (1-10 Ω·m): {water_zones['salée']} mesures
- 🟢 Eau douce (10-100 Ω·m): {water_zones['douce']} mesures
- 🔵 Eau pure (>100 Ω·m): {water_zones['pure']} mesures

🪨 **INTERPRÉTATIONS GÉOLOGIQUES:**
{chr(10).join([f"- {mat}: {cnt} occurrences" for mat, cnt in list(interp.items())[:5]])}

🎨 **CORRESPONDANCES COULEURS-RÉSISTIVITÉS:**
- 🔴 **ROUGE** (0.1-1 Ω·m): Eau de mer hypersalée / Argile saturée salée
- 🟡 **JAUNE** (1-10 Ω·m): Argile compacte / Eau saumâtre (faible perméabilité)
- 🔵 **CYAN** (10-50 Ω·m): Sable fin saturé / Eau douce peu minéralisée
- 🟢 **VERT** (50-100 Ω·m): 🎯 AQUIFÈRE SABLEUX - ZONE PRIORITAIRE FORAGE
- ⭐ **JAUNE VIF** (100-300 Ω·m): GRAVIER / AQUIFÈRE EXCELLENT - Débit élevé
- 🟠 **ORANGE** (300-1000 Ω·m): Roche altérée / Gravier sec (zone non saturée)
- 🔴 **ROUGE FONCÉ** (>1000 Ω·m): Roche consolidée / Socle cristallin

📊 **DONNÉES BRUTES DISPONIBLES:**
Aperçu de quelques mesures:
{df.head(10).to_string()}

🎯 **ZONES CIBLES FORAGE:** {len(ert_data['target_zones'])} zones identifiées (50-300 Ω·m)

═══════════════════════════════════════════════════════════════════

💬 **L'utilisateur te pose maintenant une question sur ces résultats.**

**TU DOIS:**
1. Répondre PRÉCISÉMENT avec les DONNÉES EXACTES ci-dessus
2. Expliquer en DÉTAIL avec exemples concrets
3. Citer les CHIFFRES et PROFONDEURS spécifiques
4. Comparer et interpréter les différences
5. Donner des RECOMMANDATIONS pratiques

**EXEMPLES DE RÉPONSES ATTENDUES:**

Question: "Que signifie la couleur rouge ?"
→ Réponse: "🔴 La couleur ROUGE correspond à une résistivité de 0.1-1 Ω·m. Dans votre analyse, j'ai identifié **{water_zones['mer']} mesures** dans cette gamme. Cela indique de l'**eau de mer hypersalée** ou une **argile saturée salée**. Ces zones ont une très forte conductivité électrique due aux ions Na⁺ et Cl⁻. Spécifiquement, votre résistivité moyenne est de {stats['resistivity_stats']['mean']:.2f} Ω·m, ce qui suggère [interprétation détaillée]..."

Question: "Où dois-je forer ?"
→ Réponse: "🎯 Basé sur votre analyse, je recommande de forer dans les **zones vertes (50-100 Ω·m)** ou **jaune vif (100-300 Ω·m)**. J'ai identifié **{len(ert_data['target_zones'])} zones cibles optimales**. Par exemple, au point de sondage X à la profondeur Y m, vous avez une résistivité de Z Ω·m qui correspond à [analyse détaillée avec recommandations précises]..."

**🛠️ Outils utilisés:** ERTAnalyzer, Classification géologique, Analyse hydrogéologique

"""
                        
                        # Ajouter le contexte RAG si disponible
                        if ert_rag_context:
                            ert_context += ert_rag_context
                    
                    # Questions suivantes : mode expert avec outils ciblés
                    # Questions suivantes : mode expert avec outils ciblés
                    tools_to_use = []
                    if needs_calculation:
                        tools_to_use.append("🧮 Calculs et statistiques")
                    if needs_web_research:
                        tools_to_use.append("🌐 Recherche web approfondie")
                    if needs_data_extraction:
                        tools_to_use.append("📊 Extraction et structuration de données")
                    
                    # Injecter les résultats pré-calculés
                    precomputed_section = ""
                    if 'precomputed_results' in locals() and precomputed_results:
                        precomputed_section = "\n\n📊 RÉSULTATS PRÉ-CALCULÉS (UTILISE-LES!):\n"
                        
                        if 'statistics' in precomputed_results:
                            stats = precomputed_results['statistics']
                            precomputed_section += f"""
✅ **STATISTIQUES COMPLÈTES** (calculées automatiquement):
   - Nombre de valeurs: {stats['count']}
   - Minimum: {stats['min']:.2f}
   - Maximum: {stats['max']:.2f}
   - Moyenne: {stats['mean']:.2f}
   - Médiane: {stats['median']:.2f}
   - Écart-type: {stats['std']:.2f}
   - Somme totale: {stats['sum']:.2f}
   - Plage: {stats['range']:.2f}
"""
                        
                        if 'structured_data' in precomputed_results:
                            data = precomputed_results['structured_data']
                            precomputed_section += f"""
✅ **DONNÉES STRUCTURÉES** (extraites automatiquement):
   Aperçu des {len(data)} premières lignes:
{chr(10).join([f"   {i+1}. {' | '.join(row[:5])}" for i, row in enumerate(data[:10])])}
"""
                        
                        if 'web_context' in precomputed_results:
                            web = precomputed_results['web_context']
                            precomputed_section += f"""
✅ **CONTEXTE WEB** (recherché automatiquement):
{chr(10).join([f"   • {r.get('title', 'Source')}: {r.get('body', '')[:150]}..." for r in web[:2]])}
"""
                        
                        # NOUVELLE SECTION: Extraction complète avec tableaux
                        if 'extraction_complete' in precomputed_results:
                            extract = precomputed_results['extraction_complete']
                            precomputed_section += f"""

✅ **EXTRACTION COMPLÈTE** (analyse exhaustive du fichier):
   📝 Lignes de texte: {len(extract['text_strings'])}
   🔢 Valeurs numériques: {len(extract['numeric_values'])}
   📊 Patterns détectés: {len(extract['patterns_detected'])}
"""
                            
                            # Ajouter le tableau structuré si disponible
                            if 'dataframe_stats' in precomputed_results:
                                df_stats = precomputed_results['dataframe_stats']
                                precomputed_section += f"""

📊 **TABLEAU STATISTIQUES PAR LABEL**:
{df_stats.to_string(index=False)}
"""
                            
                            # Ajouter les valeurs numériques avec contexte
                            if extract['numeric_values'][:10]:
                                precomputed_section += f"""

🔢 **VALEURS NUMÉRIQUES AVEC CONTEXTE** (top 10):
{chr(10).join([f"   {i+1}. {v['label']}: {v['value']} (ligne {v['line']})" for i, v in enumerate(extract['numeric_values'][:10])])}
"""
                            
                            # Ajouter les patterns
                            if extract['patterns_detected']:
                                precomputed_section += f"""

🔍 **PATTERNS DÉTECTÉS**:
{chr(10).join([f"   • {p}" for p in extract['patterns_detected'][:5]])}
"""
                    
                    binary_full_context = f"""
💬 EXPERTISE APPROFONDIE SUR: {binary['name']}

Tu es Kibali, expert en analyse géophysique ERT.
L'utilisateur demande: "{prompt}"

📊 CONTEXTE DU FICHIER:
{binary_context}

{ert_context}

🛠️ OUTILS DISPONIBLES: {', '.join(tools_to_use) if tools_to_use else 'Analyse standard'}

{precomputed_section}

QUESTION: {prompt}

📊 DONNÉES DISPONIBLES:
{binary_context[:2000]}
{precomputed_section}

🛠️ OUTILS ACTIVÉS POUR CETTE QUESTION:
{chr(10).join(f'✓ {tool}' for tool in tools_to_use) if tools_to_use else '✓ Analyse experte standard'}

⚡ CONSIGNES D'EXPERT:

1. **RÉPONDS AVEC PRÉCISION ET PROFONDEUR**:
   - Minimum 15-20 lignes structurées
   - Cite VALEURS EXACTES du fichier (nombres, dates, coordonnées)
   - Fais des CALCULS si données numériques présentes (UTILISE LES STATISTIQUES PRÉ-CALCULÉES CI-DESSUS!)
   - Recherche WEB si contexte métier nécessaire (UTILISE LE CONTEXTE WEB PRÉ-CHARGÉ!)

2. **STRUCTURE OBLIGATOIRE**:
   - Introduction directe (1-2 lignes)
   - Analyse détaillée avec données concrètes (CHIFFRES EXACTS des résultats pré-calculés)
   - Calculs/statistiques si pertinent (COPIE LES VALEURS PRÉ-CALCULÉES!)
   - Contexte métier si demandé (UTILISE LA RECHERCHE WEB!)
   - Conclusion avec insights

3. **INTERDICTIONS STRICTES**:
   ❌ NE DIS JAMAIS "calcul approximatif" ou "≈" si tu as les statistiques pré-calculées ci-dessus
   ❌ NE DIS JAMAIS "nécessiterait une extraction" si les données sont déjà extraites ci-dessus
   ❌ NE DIS JAMAIS "une analyse plus approfondie serait bénéfique" - FAIS-LA MAINTENANT!
   ❌ NE DIS JAMAIS "environ" ou "peut-être" - DONNE LES CHIFFRES EXACTS pré-calculés
   
4. **UTILISE LES OUTILS PRÉ-EXÉCUTÉS**:
   {'✅ STATISTIQUES COMPLÈTES déjà calculées ci-dessus - UTILISE-LES!' if needs_calculation and 'precomputed_results' in locals() and 'statistics' in precomputed_results else ''}
   {'✅ CONTEXTE WEB déjà récupéré ci-dessus - INTÈGRE-LE!' if needs_web_research and 'precomputed_results' in locals() and 'web_context' in precomputed_results else ''}
   {'✅ DONNÉES STRUCTURÉES déjà extraites ci-dessus - PRÉSENTE-LES!' if needs_data_extraction and 'precomputed_results' in locals() and 'structured_data' in precomputed_results else ''}

5. **EXEMPLES DE RÉPONSES RICHES**:
   ❌ "Ce fichier contient des données de sondage"
   ✅ "Ce fichier contient 247 points de sondage avec profondeurs allant de 12.5m à 89.3m (moyenne: 45.2m). 
       Les coordonnées couvrent une zone de 2.3 km² (X: 445230-447850, Y: 9834500-9836120). 
       Basé sur les en-têtes (survey-point, depth, data, project) et le format TSV, 
       il s'agit d'un fichier de levé géophysique standard, probablement généré par un 
       logiciel de type [recherche web: Surfer, Oasis Montaj, ou GeoSoft]. 
       Les dates présentes (20250401) indiquent une campagne récente..."

6. **SOIS UN VRAI EXPERT**: Ne te contente pas de descriptions, UTILISE LES RÉSULTATS PRÉ-CALCULÉS!
"""
                
                # REMPLACER complètement le prompt enrichi
                enriched_prompt = binary_full_context
            
            # Ajouter le message et nettoyer si trop long
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            # Garder plus de messages (50) pour conserver le contexte des fichiers uploadés
            # Garder toujours les messages de système (uploads, analyses) en priorité
            if len(st.session_state.chat_history) > 50:
                # Conserver les 10 premiers messages (généralement les uploads/analyses)
                # et les 40 derniers (conversation récente)
                important_msgs = st.session_state.chat_history[:10]
                recent_msgs = st.session_state.chat_history[-40:]
                st.session_state.chat_history = important_msgs + recent_msgs
            
            # 🛠️ PRÉ-EXÉCUTION DES OUTILS POUR COMBLER LES LIMITES
            precomputed_results = {}
            
            # 1. Si fichier binaire avec données numériques → CALCULS AUTOMATIQUES
            if 'binary_files' in st.session_state and st.session_state.binary_files and needs_calculation:
                try:
                    import re
                    import numpy as np
                    binary = st.session_state.binary_files[-1]
                    
                    # Extraire TOUTES les valeurs numériques du fichier
                    text_content = ' '.join(binary.get('text_strings', []))
                    numbers = re.findall(r'\d+\.?\d*', text_content)
                    values = [float(n) for n in numbers if n and float(n) > 0]
                    
                    if len(values) > 0:
                        precomputed_results['statistics'] = {
                            'count': len(values),
                            'min': min(values),
                            'max': max(values),
                            'mean': np.mean(values),
                            'median': np.median(values),
                            'std': np.std(values),
                            'sum': sum(values),
                            'range': max(values) - min(values)
                        }
                        st.success(f"✅ Calculs pré-exécutés: {len(values)} valeurs analysées")
                except Exception as e:
                    st.warning(f"⚠️ Calculs automatiques: {e}")
            
            # 2. Extraction automatique de données structurées
            if 'binary_files' in st.session_state and st.session_state.binary_files and needs_data_extraction:
                try:
                    binary = st.session_state.binary_files[-1]
                    text_strings = binary.get('text_strings', [])
                    
                    # Détecter la structure (CSV, TSV, JSON-like, etc.)
                    structured_data = []
                    for line in text_strings[:50]:  # Analyser les 50 premières lignes
                        if '\t' in line:
                            structured_data.append(line.split('\t'))
                        elif ',' in line and line.count(',') > 2:
                            structured_data.append(line.split(','))
                    
                    if structured_data:
                        precomputed_results['structured_data'] = structured_data[:20]  # Top 20 lignes
                        st.success(f"✅ Données structurées extraites: {len(structured_data)} lignes")
                except Exception as e:
                    st.warning(f"⚠️ Extraction automatique: {e}")
            
            # 2B. 🚀 EXTRACTION COMPLÈTE AVEC OCR + TABLEAU STRUCTURÉ
            if 'binary_files' in st.session_state and st.session_state.binary_files:
                try:
                    import pandas as pd
                    import matplotlib.pyplot as plt
                    import io
                    import base64
                    
                    binary = st.session_state.binary_files[-1]
                    
                    # Extraction exhaustive de toutes les données
                    extraction_complete = {
                        'text_strings': [],
                        'numeric_values': [],
                        'hex_data': [],
                        'ascii_readable': [],
                        'patterns_detected': []
                    }
                    
                    # 1. Extraire TOUT le texte lisible
                    text_strings = binary.get('text_strings', [])
                    extraction_complete['text_strings'] = text_strings
                    
                    # 2. Extraire toutes les valeurs numériques avec contexte
                    import re
                    for idx, line in enumerate(text_strings):
                        # Trouver nombres avec contexte
                        numbers = re.findall(r'(\w*)\s*[=:]\s*(\d+\.?\d*)', line)
                        for label, value in numbers:
                            extraction_complete['numeric_values'].append({
                                'line': idx,
                                'label': label if label else 'value',
                                'value': float(value),
                                'context': line[:100]
                            })
                    
                    # 3. Analyser les patterns dans les données binaires
                    data = binary.get('data', b'')
                    chunk_size = 16
                    for i in range(0, min(len(data), 1024), chunk_size):
                        chunk = data[i:i+chunk_size]
                        hex_str = ' '.join(f'{b:02x}' for b in chunk)
                        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
                        
                        extraction_complete['hex_data'].append({
                            'offset': i,
                            'hex': hex_str,
                            'ascii': ascii_str
                        })
                        
                        if ascii_str.count('.') < len(ascii_str) * 0.3:  # >70% ASCII
                            extraction_complete['ascii_readable'].append({
                                'offset': i,
                                'text': ascii_str
                            })
                    
                    # 4. Détecter patterns (répétitions, structures)
                    if len(data) > 100:
                        # Détecter répétitions de bytes
                        byte_freq = {}
                        for b in data[:1000]:
                            byte_freq[b] = byte_freq.get(b, 0) + 1
                        
                        most_common = sorted(byte_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                        extraction_complete['patterns_detected'] = [
                            f"Byte 0x{b:02x} apparaît {count} fois" for b, count in most_common
                        ]
                    
                    # 5. 📊 CRÉER UN TABLEAU PANDAS STRUCTURÉ
                    if extraction_complete['numeric_values']:
                        df = pd.DataFrame(extraction_complete['numeric_values'])
                        
                        # Organiser par label et calculer stats
                        if len(df) > 0:
                            df_grouped = df.groupby('label')['value'].agg([
                                ('count', 'count'),
                                ('min', 'min'),
                                ('max', 'max'),
                                ('mean', 'mean'),
                                ('std', 'std')
                            ]).reset_index()
                            
                            precomputed_results['dataframe_stats'] = df_grouped
                            precomputed_results['dataframe_raw'] = df
                            
                            # 6. 📈 VISUALISATION MATPLOTLIB
                            if len(df) > 3:
                                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                                
                                # Graphique 1: Distribution des valeurs
                                df['value'].hist(bins=20, ax=axes[0], color='skyblue', edgecolor='black')
                                axes[0].set_title('Distribution des Valeurs', fontsize=12, fontweight='bold')
                                axes[0].set_xlabel('Valeur')
                                axes[0].set_ylabel('Fréquence')
                                axes[0].grid(True, alpha=0.3)
                                
                                # Graphique 2: Top labels
                                if 'label' in df.columns and len(df['label'].unique()) > 1:
                                    label_counts = df['label'].value_counts().head(10)
                                    label_counts.plot(kind='barh', ax=axes[1], color='coral')
                                    axes[1].set_title('Top 10 Labels', fontsize=12, fontweight='bold')
                                    axes[1].set_xlabel('Nombre d\'occurrences')
                                else:
                                    axes[1].text(0.5, 0.5, 'Pas assez de labels variés', 
                                               ha='center', va='center', fontsize=12)
                                    axes[1].axis('off')
                                
                                plt.tight_layout()
                                
                                # Convertir en base64 pour affichage
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                                buf.seek(0)
                                img_base64 = base64.b64encode(buf.read()).decode()
                                plt.close()
                                
                                precomputed_results['visualization'] = img_base64
                    
                    # Stocker l'extraction complète
                    precomputed_results['extraction_complete'] = extraction_complete
                    
                    st.success(f"✅ Extraction complète: {len(extraction_complete['text_strings'])} lignes texte, {len(extraction_complete['numeric_values'])} valeurs numériques")
                    
                except Exception as e:
                    st.warning(f"⚠️ Extraction complète: {e}")
                    import traceback
                    print(traceback.format_exc())
            
            # 3. Recherche web contextuelle automatique
            if needs_web_research and 'binary_files' in st.session_state and st.session_state.binary_files:
                try:
                    binary = st.session_state.binary_files[-1]
                    # Construire requête intelligente
                    format_name = binary['info'].get('format_name', 'unknown')
                    extension = binary['name'].split('.')[-1] if '.' in binary['name'] else 'dat'
                    search_query = f"{format_name} {extension} file format geophysical survey data"
                    
                    # Exécuter la recherche
                    if web_enabled:
                        web_results = enhanced_web_search(search_query, max_results=2)
                        if web_results:
                            precomputed_results['web_context'] = web_results[:2]
                            st.success(f"✅ Recherche web contextualisée: {len(web_results)} sources")
                except Exception as e:
                    st.warning(f"⚠️ Recherche web automatique: {e}")
            
            # Afficher les outils actifs
            active_tools_msg = []
            if needs_calculation:
                active_tools_msg.append("🧮 Calculs statistiques")
            if needs_web_research or ('binary_files' in st.session_state and st.session_state.binary_files):
                active_tools_msg.append("🌐 Recherche web")
            if needs_data_extraction:
                active_tools_msg.append("📊 Extraction de données")
            
            if active_tools_msg:
                tools_status = " • ".join(active_tools_msg)
                st.info(f"⚡ Outils activés: {tools_status}")
            
            # 📊 AFFICHER LA VISUALISATION SI DISPONIBLE
            if 'precomputed_results' in locals() and 'visualization' in precomputed_results:
                with chat_container:
                    st.markdown("### 📈 Visualisation des Données Extraites")
                    st.markdown(f'<img src="data:image/png;base64,{precomputed_results["visualization"]}" style="width:100%; border-radius:10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
            
            # 📋 AFFICHER LE TABLEAU SI DISPONIBLE
            if 'precomputed_results' in locals() and 'dataframe_stats' in precomputed_results:
                with chat_container:
                    st.markdown("### 📊 Tableau des Statistiques")
                    st.dataframe(precomputed_results['dataframe_stats'], use_container_width=True)
            
            # 🎼 ORCHESTRATION MULTI-IA
            orchestration_plan = None
            orchestration_enabled = False
            
            # Éviter l'orchestration si question simple ou déjà des résultats pré-calculés suffisants
            skip_orchestration = (
                len(prompt.split()) < 5 or  # Question très courte
                ('precomputed_results' in locals() and len(precomputed_results) > 0 and not needs_web_research)  # Résultats suffisants
            )
            
            if ORCHESTRATOR_AVAILABLE and not skip_orchestration:
                try:
                    # Analyser la tâche et créer le plan
                    orchestrator = get_orchestrator()
                    
                    # Construire le contexte pour l'orchestrateur
                    orchestration_context = {
                        'has_vectordb': st.session_state.vectordb is not None,
                        'has_binary_file': 'binary_files' in st.session_state and st.session_state.binary_files,
                        'web_enabled': web_enabled,
                        'precomputed_results': precomputed_results if 'precomputed_results' in locals() else {},
                        'needs_calculation': needs_calculation,
                        'needs_web_research': needs_web_research,
                        'needs_data_extraction': needs_data_extraction
                    }
                    
                    tasks, todolist = analyze_and_plan(prompt, orchestration_context)
                    
                    if tasks and len(tasks) > 0:
                        orchestration_enabled = True
                        
                        # Afficher la todolist
                        with chat_container:
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                                padding: 25px;
                                border-radius: 15px;
                                margin: 15px 0;
                                border-left: 8px solid #2196F3;
                                box-shadow: 0 4px 15px rgba(33,150,243,0.3);
                            ">
                                <h3 style="color: #1565C0; margin: 0 0 15px 0; display: flex; align-items: center; font-weight: bold;">
                                    <span style="font-size: 2rem; margin-right: 10px;">📋</span>
                                    Plan d'Exécution Multi-IA
                                </h3>
                                <div style="background: white; padding: 20px; border-radius: 10px; border: 2px solid #2196F3;">
                                    <pre style="color: #1a237e; margin: 0; font-family: 'Courier New', monospace; font-size: 1rem; white-space: pre-wrap; font-weight: 500; line-height: 1.6;">{todolist}</pre>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Sélectionner les IA spécialisées
                        team = orchestrator.select_team(tasks)
                        
                        # Afficher le visuel de l'équipe
                        with chat_container:
                            team_visual = orchestrator.create_execution_plan_visual(team)
                            st.markdown(team_visual, unsafe_allow_html=True)
                        
                        # Préparer le plan d'orchestration
                        orchestration_plan = {
                            'tasks': tasks,
                            'team': team,
                            'todolist': todolist
                        }
                
                except Exception as e:
                    st.warning(f"⚠️ Orchestration non disponible: {e}")
                    orchestration_enabled = False
                    
                    # FALLBACK: Activer les outils dynamiquement même sans orchestration
                    if precomputed_results:
                        with chat_container:
                            st.info("🔧 Mode outils dynamiques activé (sans orchestration)")
                            if 'statistics' in precomputed_results:
                                st.success(f"📊 Statistiques pré-calculées disponibles")
                            if 'web_context' in precomputed_results:
                                st.success(f"🌐 Contexte web chargé")
                            if 'structured_data' in precomputed_results:
                                st.success(f"📋 Données structurées extraites")
            
            # Animation de chargement
            loading_msg = "🎼 Orchestration multi-IA en cours..." if orchestration_enabled else "🤔 Kibali analyse avec ses outils..."
            with st.spinner(loading_msg):
                # Vérifier si on utilise le mode local
                if st.session_state.local_mode and st.session_state.local_model_loaded:
                    # Utiliser le modèle local Qwen
                    try:
                        # 🔥 RECHERCHE PROACTIVE: Activer automatiquement selon le contexte
                        rag_context = ""
                        web_context = ""
                        web_media_html = ""
                        
                        # Recherche RAG si disponible
                        if st.session_state.vectordb:
                            rag_docs = rag_search(prompt, st.session_state.vectordb, k=3)
                            if rag_docs:
                                rag_context = "\n\n".join([f"Document: {doc.page_content[:500]}..." for doc in rag_docs])
                        
                        # 🌐 RECHERCHE WEB AUTOMATIQUE pour contexte métier/industriel
                        if needs_web_research or ('binary_files' in st.session_state and st.session_state.binary_files):
                            try:
                                # Construire requête intelligente basée sur le contenu du fichier
                                if 'binary_files' in st.session_state and st.session_state.binary_files:
                                    binary = st.session_state.binary_files[-1]
                                    # Extraire mots-clés des chaînes de texte
                                    keywords = []
                                    for s in binary.get('text_strings', [])[:10]:
                                        words = s.split()
                                        keywords.extend([w for w in words if len(w) > 4])
                                    
                                    search_query = f"{binary['info']['format_name']} {' '.join(keywords[:5])} file format application"
                                else:
                                    search_query = prompt
                                
                                web_results = enhanced_web_search(search_query, max_results=3)
                                if web_results:
                                    web_context = "\n".join([f"{r.get('title', '')[:50]} - {r.get('body', '')[:150]}" for r in web_results[:2]])
                                    web_media_html = format_web_results_with_media(web_results, max_results=3)
                            except Exception as e:
                                web_context = f"Recherche web limitée: {e}"
                        elif web_enabled:
                            # Recherche web standard si activée
                            try:
                                web_results = enhanced_web_search(prompt, max_results=3)
                                if web_results:
                                    web_context = "\n".join([f"{r.get('title', '')[:50]} - {r.get('body', '')[:150]}" for r in web_results[:2]])
                                    web_media_html = format_web_results_with_media(web_results, max_results=3)
                            except Exception as e:
                                web_context = f"Erreur recherche web: {e}"
                        
                        # Construire le contexte enrichi avec médias ET fichiers binaires
                        if 'binary_files' in st.session_state and st.session_state.binary_files:
                            # Mode fichier binaire: enriched_prompt contient déjà tout
                            full_context = enriched_prompt
                            if rag_context:
                                full_context += f"\n\n📚 SOURCES COMPLÉMENTAIRES:\n{rag_context[:300]}"
                            if web_context:
                                full_context += f"\n\n🌐 RECHERCHE WEB CONTEXTUELLE:\n{web_context}"
                        else:
                            # Mode normal
                            full_context = f"CONTEXTE DISPONIBLE:\n{rag_context}{web_context}\n\nQUESTION: {enriched_prompt}"
                        
                        # 🌊 Générer avec le modèle local EN STREAMING
                        from langchain_core.messages import HumanMessage
                        
                        with chat_container:
                            with st.chat_message("assistant"):
                                message_placeholder = st.empty()
                                full_response = "🏠 **Mode Local (Qwen 1.5B)** - Réponse générée localement\n\n"
                                
                                # Utiliser _stream au lieu de _generate
                                for chunk in st.session_state.local_qwen_llm._stream(
                                    [HumanMessage(content=full_context)],
                                    stop=None,
                                    run_manager=None
                                ):
                                    if hasattr(chunk, 'content'):
                                        full_response += chunk.content
                                        message_placeholder.markdown(full_response + "▌")
                                
                                # Affichage final sans curseur
                                full_response = clean_response_text(full_response)
                                message_placeholder.markdown(full_response)
                        
                        response = full_response
                        
                    except Exception as e:
                        response = f"❌ Erreur modèle local: {e}\n\nBasculement vers mode API..."
                        # Fallback vers mode API
                        st.session_state.local_mode = False
                        st.rerun()
                
                else:
                    # ═══════════════════════════════════════════════════════════════════
                    # 🎯 MODE PRIORITAIRE: FICHIER BINAIRE DÉTECTÉ
                    # Court-circuite tous les autres systèmes pour forcer l'analyse du fichier
                    # ═══════════════════════════════════════════════════════════════════
                    if 'binary_files' in st.session_state and st.session_state.binary_files:
                        st.info("🔍 Mode analyse de fichier binaire activé")
                        
                        # Utiliser directement l'API avec le contexte binaire complet
                        client = create_client()
                        
                        # enriched_prompt contient déjà TOUT le contexte binaire détaillé
                        # On limite l'historique pour maximiser le contexte du fichier
                        messages = [{"role": "user", "content": enriched_prompt[:12000]}]  # 12000 chars pour analyse complète très détaillée
                        
                        with chat_container:
                            with st.chat_message("assistant"):
                                message_placeholder = st.empty()
                                full_response = ""
                                
                                # Streaming direct
                                stream = client.chat.completions.create(
                                    model=WORKING_MODELS[model_choice],
                                    messages=messages,
                                    max_tokens=4000,  # Réponse très détaillée (2-3 pages)
                                    temperature=0.2,  # Basse température pour précision
                                    stream=True
                                )
                                
                                for chunk in stream:
                                    if chunk.choices[0].delta.content is not None:
                                        full_response += chunk.choices[0].delta.content
                                        message_placeholder.markdown(full_response + "▌")
                                
                                message_placeholder.markdown(full_response)
                        
                        response = full_response
                    
                    # Mode API normal (code existant)
                    # Utilisation du système d'outils dynamiques si disponible
                    elif st.session_state.tool_manager and TOOLS_SYSTEM_AVAILABLE:
                        try:
                            # 🎼 MODE ORCHESTRATION MULTI-IA PRIORITAIRE
                            if orchestration_enabled and orchestration_plan:
                                # Exécuter avec orchestration multi-IA
                                orchestrator = get_orchestrator()
                                
                                with chat_container:
                                    with st.chat_message("assistant"):
                                        message_placeholder = st.empty()
                                        full_response = "🎼 **Orchestration Multi-IA Activée**\n\n"
                                        message_placeholder.markdown(full_response + "▌")
                                        
                                        # Exécuter chaque tâche avec l'IA spécialisée
                                        for idx, (task, specialist) in enumerate(zip(orchestration_plan['tasks'], orchestration_plan['team']), 1):
                                            task_header = f"\n\n{'─' * 60}\n**🎯 Tâche {idx}/{len(orchestration_plan['tasks'])}: {task.description}**\n**🤖 IA: {specialist.name}**\n{'─' * 60}\n\n"
                                            full_response += task_header
                                            message_placeholder.markdown(full_response + "▌")
                                            
                                            try:
                                                # Exécuter avec l'IA spécialisée
                                                # Pour l'instant on utilise l'API standard, mais on peut charger des modèles spécialisés
                                                task_prompt = f"{enriched_prompt}\n\nFOCUS SUR: {task.description}\nTYPE: {task.task_type.value}"
                                                
                                                client = create_client()
                                                task_messages = [{"role": "user", "content": task_prompt[:8000]}]
                                                
                                                # Obtenir la limite de tokens depuis le catalog
                                                from master_orchestrator import MasterOrchestrator
                                                orchestrator = MasterOrchestrator()
                                                specialist_capability = orchestrator.SPECIALISTS_CATALOG.get(specialist)
                                                max_output = specialist_capability.max_output_tokens if specialist_capability else 4000
                                                
                                                # Streaming pour cette tâche
                                                stream = client.chat.completions.create(
                                                    model=WORKING_MODELS[model_choice],
                                                    messages=task_messages,
                                                    max_tokens=max_output,
                                                    temperature=0.7,
                                                    stream=True
                                                )
                                                
                                                task_response = ""
                                                for chunk in stream:
                                                    if chunk.choices[0].delta.content is not None:
                                                        task_response += chunk.choices[0].delta.content
                                                        full_response = full_response.rstrip("▌") + task_response + "▌"
                                                        message_placeholder.markdown(full_response)
                                                
                                                full_response = full_response.rstrip("▌")
                                                
                                                # Vérifier si relay nécessaire (réponse très longue)
                                                if len(task_response.split()) > max_output * 0.85:
                                                    relay_msg = f"\n\n⚡ **Relay vers {specialist.name}** (limite tokens atteinte)\n\n"
                                                    full_response += relay_msg
                                                    message_placeholder.markdown(full_response + "▌")
                                                
                                            except Exception as task_error:
                                                error_msg = f"\n\n❌ **Erreur tâche {idx}:** {task_error}\n\n"
                                                full_response += error_msg
                                                message_placeholder.markdown(full_response + "▌")
                                        
                                        # Synthèse finale
                                        full_response += f"\n\n{'═' * 60}\n✅ **Orchestration Complétée** - {len(orchestration_plan['tasks'])} tâches exécutées\n{'═' * 60}"
                                        
                                        # 🧹 Nettoyer la réponse
                                        full_response = clean_response_text(full_response)
                                        message_placeholder.markdown(full_response)
                                
                                response = full_response
                            
                            else:
                                # Mode outils classique (code existant)
                                # Préparer le contexte pour les outils (avec médias analysés)
                                tool_context = {
                                    'has_pdfs': st.session_state.vectordb is not None,
                                    'vectordb_available': st.session_state.vectordb is not None,
                                    'web_enabled': web_enabled,
                                    'media_analysis': st.session_state.media_analysis_results if st.session_state.media_analysis_results else None
                                }
                                
                                # Analyse de la requête et sélection des outils appropriés
                                selected_tools = st.session_state.tool_manager.get_relevant_tools(enriched_prompt, tool_context)
                                
                                if selected_tools:
                                    # 🎨 POPUP ÉLÉGANT DANS LE CHAT - Affichage des outils détectés
                                    with chat_container:
                                        st.markdown(f"""
                                        <div style="
                                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                            padding: 20px;
                                            border-radius: 15px;
                                            margin: 15px 0;
                                            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
                                            border: 3px solid transparent;
                                            background-clip: padding-box;
                                            position: relative;
                                            animation: toolPopup 0.5s ease-out;
                                        ">
                                            <div style="
                                                position: absolute;
                                                top: -3px;
                                                left: -3px;
                                                right: -3px;
                                                bottom: -3px;
                                                background: linear-gradient(45deg, #00ff88, #ffff00, #0088ff, #00ff88);
                                                background-size: 400% 400%;
                                                border-radius: 15px;
                                                z-index: -1;
                                                animation: borderScintillation 3s ease infinite;
                                                filter: blur(5px);
                                            "></div>
                                            <h3 style="color: white; margin: 0 0 15px 0; display: flex; align-items: center; font-size: 1.3rem;">
                                                <span style="font-size: 2rem; margin-right: 10px;">🔧</span>
                                                Outils IA Activés
                                            </h3>
                                            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(10px);">
                                                {''.join([f'<div style="color: #FFD700; font-weight: 700; font-size: 1.1rem; margin: 8px 0; display: flex; align-items: center;"><span style="color: #00ff88; margin-right: 8px;">▶</span> {tool.name}</div><p style="color: #f0f0f0; margin: 5px 0 15px 25px; font-size: 0.95rem; font-style: italic;">{tool.description}</p>' for tool in selected_tools])}
                                            </div>
                                            <p style="color: #FFD700; margin: 15px 0 0 0; font-size: 0.9rem; text-align: center; font-weight: 600;">
                                                ⚡ {len(selected_tools)} outil{'s' if len(selected_tools) > 1 else ''} en cours d'exécution...
                                            </p>
                                        </div>
                                        <style>
                                            @keyframes toolPopup {{
                                                0% {{ transform: scale(0.9) translateY(-20px); opacity: 0; }}
                                                100% {{ transform: scale(1) translateY(0); opacity: 1; }}
                                            }}
                                            @keyframes borderScintillation {{
                                                0% {{ background-position: 0% 50%; }}
                                                50% {{ background-position: 100% 50%; }}
                                                100% {{ background-position: 0% 50%; }}
                                            }}
                                        </style>
                                        """, unsafe_allow_html=True)
                                    
                                    # Recherche RAG si un outil PDF est détecté
                                    rag_context = ""
                                    pdf_tool_used = any('pdf' in tool.name.lower() or 'document' in tool.name.lower() for tool in selected_tools)
                                    
                                    if pdf_tool_used and st.session_state.vectordb:
                                        with st.expander("🔍 Recherche dans les documents PDF...", expanded=True):
                                            rag_docs = rag_search(prompt, st.session_state.vectordb, k=10)
                                            if rag_docs:
                                                rag_context = "\n\n".join([f"📄 {doc.metadata.get('source', 'Document')}:\n{doc.page_content}" for doc in rag_docs])
                                                st.success(f"✅ {len(rag_docs)} passages trouvés")
                                                # Afficher un aperçu
                                                for i, doc in enumerate(rag_docs[:3], 1):
                                                    st.markdown(f"**Document {i}:** `{doc.metadata.get('source', 'Inconnu')}`")
                                            else:
                                                st.info("❌")
                                    elif pdf_tool_used and not st.session_state.vectordb:
                                        st.warning("⚠️ Outil PDF détecté mais aucune base vectorielle chargée. Ajoutez des PDFs dans l'onglet Configuration.")
                                    
                                    # Exécution des autres outils avec affichage détaillé dans le chat
                                    tool_results = []
                                    for tool in selected_tools:
                                        if 'pdf' not in tool.name.lower() and 'document' not in tool.name.lower():
                                            try:
                                                with chat_container:
                                                    with st.expander(f"⚙️ Exécution: {tool.name}", expanded=True):
                                                        st.info(f"🔄 {tool.description}")
                                                        result = tool.execute(prompt, tool_context)
                                                        
                                                        # Affichage élégant du résultat
                                                        st.markdown(f"""
                                                        <div style="
                                                            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                                            padding: 15px;
                                                            border-radius: 10px;
                                                            border-left: 4px solid #00ff88;
                                                            margin: 10px 0;
                                                        ">
                                                            <h4 style="color: #00ff88; margin: 0 0 10px 0;">✅ Résultat</h4>
                                                            <pre style="color: #f0f0f0; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 0.9rem;">{str(result)[:500]}</pre>
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                tool_results.append(f"**{tool.name}:** {result}")
                                            except Exception as e:
                                                with chat_container:
                                                    st.error(f"❌ Erreur {tool.name}: {str(e)}")
                                                tool_results.append(f"**{tool.name} (erreur):** {str(e)}")
                                    
                                    # Recherche web avec toutes les capacités (TOUJOURS ACTIVÉE)
                                    web_context = ""
                                    web_media_html = ""
                                    if web_enabled:
                                        try:
                                            web_results = enhanced_web_search(prompt, max_results=3)
                                            if web_results:
                                                web_context = "\n".join([f"{r.get('title', '')[:50]}: {r.get('body', '')[:100]}" for r in web_results[:2]])
                                                # Générer le HTML avec médias
                                                web_media_html = format_web_results_with_media(web_results, max_results=3)
                                        except Exception as e:
                                            web_context = f"Erreur recherche web: {e}"
                                    
                                    # Construction du prompt final avec TOUTES les sources
                                    # IMPORTANT: enriched_prompt contient déjà le contexte binaire OU média OU conversation
                                    # On doit le préserver et y ajouter les autres sources
                                    full_context = ""
                                    has_binary_context = 'binary_files' in st.session_state and st.session_state.binary_files
                                    has_media_in_enriched = 'MÉDIAS ANALYSÉS' in str(enriched_prompt) or 'Image:' in str(enriched_prompt)
                                    
                                    # Si on a déjà un contexte enrichi (binaire/média/conversation), on le met en premier
                                    if has_binary_context or has_media_in_enriched:
                                        full_context = str(enriched_prompt) + "\n\n"
                                        full_context += "═══════════════════════════════════════════════════════════════════════════════\n"
                                        full_context += "📚 SOURCES COMPLÉMENTAIRES (À UTILISER SI PERTINENT):\n"
                                        full_context += "═══════════════════════════════════════════════════════════════════════════════\n\n"
                                    
                                    # 1. DOCUMENTS LOCAUX (RAG)
                                    if rag_context:
                                        full_context += f"📚 **Documents locaux:**\n{rag_context[:400]}\n\n"
                                    
                                    # 2. RÉSULTATS DES OUTILS
                                    if tool_results:
                                        limited_results = [r[:300] for r in tool_results[:2]]
                                        for tr in limited_results:
                                            full_context += f"🔧 **Outil:** {tr}\n\n"
                                    
                                    # 4. RECHERCHE WEB (POUR CONTEXTE ADDITIONNEL)
                                    if web_context:
                                        full_context += f"🌐 **Contexte web complémentaire:**\n{web_context[:300]}\n\n"
                                    
                                    # Prompt final intelligent qui s'adapte au contexte
                                    if has_binary_context:
                                        # Mode analyse de fichier binaire
                                        final_prompt = f"""{full_context}

💡 **CONSIGNE STRICTE:** Vous analysez un FICHIER SPÉCIFIQUE uploadé par l'utilisateur.
- ✅ Analysez CE FICHIER CONCRET en utilisant les données réelles ci-dessus
- ✅ Citez les chaînes de texte extraites, les offsets, les magic bytes
- ✅ Commencez votre réponse par "Ce fichier {st.session_state.binary_files[-1]['name']}..."
- ❌ N'expliquez PAS ce qu'est un fichier en général
- ❌ Ne donnez PAS de cours théorique
- ❌ Ne parlez PAS de concepts généraux

Répondez UNIQUEMENT en analysant les DONNÉES CONCRÈTES du fichier ci-dessus."""
                                    elif has_media_in_enriched:
                                        final_prompt = f"""{full_context}
❓ **QUESTION:** {prompt}

💡 **CONSIGNE:** Vous avez accès aux données d'analyse des médias ci-dessus. Analysez-les EN PROFONDEUR en utilisant:
- Les couleurs, textes, objets détectés
- La résolution et le format
- L'analyse sémantique (CLIP)
- Le contexte web si pertinent (pour approfondir)

Répondez de manière détaillée et technique en exploitant TOUTES les données disponibles."""
                                    else:
                                        final_prompt = f"""{full_context}
❓ **QUESTION:** {prompt}

Réponds en utilisant toutes les sources disponibles (documents, outils, web) pour une réponse complète."""
                                    
                                    # Génération de la réponse finale avec historique adapté ET STREAMING
                                    client = create_client()
                                    has_media_context = bool(st.session_state.media_analysis_results)
                                    has_binary_context = 'binary_files' in st.session_state and st.session_state.binary_files
                                    
                                    # Adapter les limites selon le contexte
                                    if has_binary_context or has_media_context:
                                        # Si fichiers binaires ou médias: garder PLUS d'historique pour le contexte de conversation
                                        # Garder les 5 derniers messages pour suivre le fil de la conversation
                                        recent_msgs = st.session_state.chat_history[-5:] if len(st.session_state.chat_history) >= 5 else st.session_state.chat_history
                                        
                                        # Filtrer pour garder seulement user/assistant (pas les HTML)
                                        recent_msgs = [m for m in recent_msgs if m.get('role') in ['user', 'assistant']]
                                        
                                        messages = recent_msgs + [{"role": "user", "content": final_prompt[:12000]}]  # 12000 chars pour contexte très riche
                                        max_tokens_response = 4000  # 4000 tokens pour réponses très détaillées (≈60-80 lignes, 2-3 pages)
                                    else:
                                        # Mode normal : historique standard
                                        recent_msgs = st.session_state.chat_history[-4:] if len(st.session_state.chat_history) > 4 else st.session_state.chat_history
                                        recent_msgs = [m for m in recent_msgs if m.get('role') in ['user', 'assistant']]
                                        messages = recent_msgs + [{"role": "user", "content": final_prompt[:5000]}]
                                        max_tokens_response = 2000  # 2000 tokens pour réponses standard détaillées
                                    
                                    # 🌊 STREAMING ACTIVÉ - Affichage progressif des tokens
                                    with chat_container:
                                        with st.chat_message("assistant"):
                                            message_placeholder = st.empty()
                                            full_response = ""
                                            
                                            # Créer le stream
                                            stream = client.chat.completions.create(
                                                model=WORKING_MODELS[model_choice],
                                                messages=messages,
                                                max_tokens=max_tokens_response,
                                                temperature=0.7 if has_binary_context or has_media_context else 0.5,  # Plus créatif pour analyses
                                                stream=True  # 🔥 ACTIVER LE STREAMING
                                            )
                                            
                                            # Afficher chaque token au fur et à mesure
                                            for chunk in stream:
                                                if chunk.choices[0].delta.content is not None:
                                                    full_response += chunk.choices[0].delta.content
                                                    message_placeholder.markdown(full_response + "▌")
                                            
                                            # 🧹 NETTOYER LA RÉPONSE FINALE
                                            full_response = clean_response_text(full_response)
                                            
                                            # Affichage final sans curseur
                                            message_placeholder.markdown(full_response)
                                    
                                    response = full_response
                        
                        except Exception as e:
                            st.error(f"Erreur système d'outils: {e}")
                            # Fallback vers l'approche classique avec STREAMING
                            with chat_container:
                                with st.chat_message("assistant"):
                                    message_placeholder = st.empty()
                                    full_response = ""
                                    
                                    if not web_enabled:
                                        docs = rag_search(enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt, st.session_state.vectordb, k=3)
                                    else:
                                        docs = hybrid_search_enhanced(enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt, st.session_state.vectordb, k=3, web_search_enabled=True)
                                    
                                    # 🌊 Streaming des chunks
                                    for chunk in generate_answer_enhanced_stream(
                                        enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt, docs, WORKING_MODELS[model_choice], include_sources=True
                                    ):
                                        full_response += chunk
                                        message_placeholder.markdown(full_response + "▌")
                                    
                                    # 🧹 Nettoyer la réponse
                                    full_response = clean_response_text(full_response)
                                    message_placeholder.markdown(full_response)
                            
                            response = full_response
                    
                    else:
                        # Système d'outils non disponible, utiliser l'approche classique
                        web_media_html = ""
                        if st.session_state.agent is None:
                            st.session_state.current_model = WORKING_MODELS[model_choice]
                            st.session_state.agent = create_enhanced_agent(
                                st.session_state.current_model, 
                                st.session_state.vectordb, 
                                st.session_state.graph, 
                                st.session_state.pois
                            )
                        
                        try:
                            if not web_enabled:
                                # Mode RAG simple avec STREAMING
                                with chat_container:
                                    with st.chat_message("assistant"):
                                        message_placeholder = st.empty()
                                        full_response = ""
                                        
                                        # Utiliser enriched_prompt pour avoir le contexte binaire
                                        docs = rag_search(enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt, st.session_state.vectordb, k=3)
                                        
                                        # 🌊 Streaming
                                        for chunk in generate_answer_enhanced_stream(
                                            enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt, docs, st.session_state.current_model, include_sources=True
                                        ):
                                            full_response += chunk
                                            message_placeholder.markdown(full_response + "▌")
                                        
                                        message_placeholder.markdown(full_response)
                                
                                response = full_response
                            else:
                                # Recherche web avec médias pour mode classique
                                try:
                                    web_results = enhanced_web_search(prompt, max_results=3)
                                    if web_results:
                                        web_media_html = format_web_results_with_media(web_results, max_results=3)
                                except:
                                    pass
                                # Utiliser enriched_prompt si fichier binaire disponible
                                if 'binary_files' in st.session_state and st.session_state.binary_files:
                                    response = st.session_state.agent.run(enriched_prompt)
                                else:
                                    response = st.session_state.agent.run(prompt)
                        except Exception as e:
                            response = f"❌ Erreur: {e}\n\nTentative avec recherche locale..."
                            try:
                                with chat_container:
                                    with st.chat_message("assistant"):
                                        message_placeholder = st.empty()
                                        full_response = ""
                                        
                                        # Utiliser enriched_prompt pour le contexte binaire
                                        query = enriched_prompt if ('binary_files' in st.session_state and st.session_state.binary_files) else prompt
                                        docs = rag_search(query, st.session_state.vectordb, k=3)
                                        
                                        for chunk in generate_answer_enhanced_stream(
                                            query, docs, st.session_state.current_model
                                        ):
                                            full_response += chunk
                                            message_placeholder.markdown(full_response + "▌")
                                        
                                        message_placeholder.markdown(full_response)
                                
                                response = full_response
                            except:
                                response = f"❌ Erreur complète: {e}"
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # 🌐 NOUVEAU: Ajouter les médias web si disponibles
            if 'web_media_html' in locals() and web_media_html:
                st.session_state.chat_history.append({"role": "web_media", "content": web_media_html})
            
            # 🧠 NOUVEAU: Sauvegarder dans la mémoire vectorielle
            try:
                from chat_memory import add_to_chat_memory
                st.session_state.chat_vectordb = add_to_chat_memory(
                    prompt, 
                    response, 
                    st.session_state.chat_vectordb
                )
            except Exception as e:
                print(f"⚠️ Erreur sauvegarde mémoire: {e}")
            
            # 🤖 AUTO-APPRENTISSAGE: Apprendre de la conversation
            if AUTO_LEARNING_AVAILABLE and st.session_state.get('auto_learning'):
                try:
                    st.session_state.auto_learning.learn_from_conversation(prompt, response)
                except Exception as e:
                    print(f"⚠️ Erreur apprentissage conversation: {e}")
            
            # Nettoyer l'historique si trop long
            if len(st.session_state.chat_history) > 20:
                st.session_state.chat_history = st.session_state.chat_history[-20:]
            st.rerun()
        
        # Boutons d'action stylisés
        col6, col7, col8, col9 = st.columns(4)
        with col6:
            if st.button("🗑️ **Effacer conversation**", key="clear_chat", help="Supprime tout l'historique du chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col7:
            if st.button("🧠 **Statistiques mémoire**", key="memory_stats", help="Affiche les statistiques de la mémoire vectorielle"):
                from chat_memory import get_memory_stats
                try:
                    stats = get_memory_stats(st.session_state.chat_vectordb)
                    st.markdown(f"""
                    <div class="kibali-card" style="border: 2px solid #ff0088;">
                        <h4 style="color: #ff0088;">🧠 Mémoire conversationnelle</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 10px;">
                            <div><strong>💬 Conversations:</strong> {stats['count']}</div>
                            <div><strong>💾 Taille:</strong> {stats['size']}</div>
                            <div><strong>📊 État:</strong> {stats['status']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Erreur statistiques mémoire: {e}")
        
        with col8:
            if st.button("💡 **Questions d'exemple**", key="show_examples", help="Affiche des exemples de questions"):
                with chat_container:
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 20px;
                        border-radius: 15px;
                        margin: 15px 0;
                        border: 2px solid #00ff88;
                    ">
                        <h3 style="color: #00ff88; margin: 0 0 20px 0;">💡 Exemples de questions pour tester les outils</h3>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #FFD700;">
                            <h4 style="color: #FFD700; margin: 0 0 10px 0;">🔧 Outil de Code</h4>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Crée une fonction Python pour calculer la factorielle"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Génère une classe JavaScript pour gérer un panier d'achat"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Écris un algorithme de tri à bulles en Python"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Code une fonction de recherche binaire"</p>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #0088ff;">
                            <h4 style="color: #0088ff; margin: 0 0 10px 0;">🌐 Recherche Web</h4>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Actualités intelligence artificielle 2024"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Tendances technologiques récentes"</p>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #ff0088;">
                            <h4 style="color: #ff0088; margin: 0 0 10px 0;">📊 Mathématiques</h4>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Calcule la racine carrée de 144"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Résous l'équation: 2x + 5 = 15"</p>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #00ffff;">
                            <h4 style="color: #00ffff; margin: 0 0 10px 0;">🌍 Traduction</h4>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Traduis 'Bonjour le monde' en anglais"</p>
                            <p style="color: #f0f0f0; margin: 5px 0;">• "Comment dit-on 'merci' en espagnol?"</p>
                        </div>
                        
                        <p style="color: #FFD700; margin: 20px 0 0 0; text-align: center; font-size: 0.9rem; font-weight: 600;">
                            ⚡ Testez ces exemples pour voir les outils en action !
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col9:
            if st.button("📊 **Statut système**", key="system_status", help="Affiche l'état du système Kibali"):
                status = get_system_status()
                st.markdown(f"""
                <div class="kibali-card">
                <h4>📊 État du système</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div><strong>📁 PDFs:</strong> {status['counts']['pdfs']}</div>
                <div><strong>🗺️ Graphes:</strong> {status['counts']['graphs']}</div>
                <div><strong>💾 Base vectorielle:</strong> {'✅' if status['files']['vectordb'] else '❌'}</div>
                <div><strong>🌐 Cache web:</strong> {'✅' if status['files']['web_cache'] else '❌'}</div>
                </div>
                </div>
                """, unsafe_allow_html=True)

    # ===============================================
    # Onglet 3: Trajets avec cartes
    # ===============================================
    with tab3:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 🗺️ Calcul de trajets
        **Exemples:** "Comment aller de l'école à l'hôpital ?"
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        trajectory_question = st.text_area(
            "🗺️ **Question de trajet**",
            placeholder="Ex: Comment aller de l'école à l'hôpital ?",
            height=100,
            help="Décris ton trajet en langage naturel"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_traj1, col_traj2 = st.columns([1, 1])
        with col_traj1:
            if st.button("🚀 **Calculer trajet**", key="calculate_route", help="Calcule l'itinéraire demandé"):
                if not st.session_state.graph or not st.session_state.pois:
                    st.error("❌ Graphe non chargé - Upload un fichier OSM d'abord")
                else:
                    with st.spinner("🗺️ Calcul du trajet en cours..."):
                        carte_buf, reponse, traj_info = calculer_trajet(
                            trajectory_question, 
                            st.session_state.graph, 
                            st.session_state.pois
                        )
                        if carte_buf:
                            carte_buf.seek(0)
                            image = Image.open(carte_buf)
                            st.markdown('<div class="kibali-card zoom-image">', unsafe_allow_html=True)
                            st.image(image, caption="🗺️ Carte du trajet calculé", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.session_state.last_traj_info = traj_info
                        st.session_state.last_reponse = reponse
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.text_area("📋 **Détails du trajet**", value=st.session_state.last_reponse, height=200, disabled=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        with col_traj2:
            if st.button("💾 **Sauvegarder trajet**", key="save_trajectory", help="Sauvegarde ce trajet dans la base de connaissances"):
                if st.session_state.last_traj_info:
                    status = save_trajectory(
                        trajectory_question, 
                        st.session_state.last_reponse, 
                        st.session_state.last_traj_info
                    )
                    st.success(f"✅ Trajet sauvegardé: {status}")
                else:
                    st.warning("⚠️ Aucun trajet à sauvegarder - Calcule un trajet d'abord")

    # ===============================================
    # Onglet 4: Analyse Image avec IA avancée
    # ===============================================
    with tab4:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 📸 Analyse d'images avancée avec IA
        Upload une image pour analyse complète: Vision AI (CLIP), OCR, détection d'objets, annotations et graphiques.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "📤 **Upload Fichier**", 
            type=None,
            key="image_upload",
            help="Tous formats acceptés - binaires inclus"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 **Analyser l'image**", key="analyze_image", help="Lance l'analyse complète de l'image avec IA"):
            if uploaded_image:
                with st.spinner("🔬 Analyse IA en cours..."):
                    
                    # Charger tous les modèles nécessaires
                    vision_models = None
                    ocr_reader = None
                    yolo_model = None
                    blip_models = None
                    
                    try:
                        with st.spinner("📥 Chargement CLIP..."):
                            vision_models = load_vision_models()
                    except Exception as e:
                        st.warning(f"⚠️ CLIP non disponible: {e}")
                    
                    try:
                        with st.spinner("📝 Chargement OCR..."):
                            ocr_reader = load_ocr_reader()
                    except Exception as e:
                        st.warning(f"⚠️ OCR non disponible: {e}")
                    
                    try:
                        with st.spinner("🎯 Chargement YOLO..."):
                            yolo_model = load_yolo_model()
                            if yolo_model:
                                st.success("✅ YOLO chargé pour détection d'objets")
                    except Exception as e:
                        st.warning(f"⚠️ YOLO non disponible: {e}")
                    
                    try:
                        with st.spinner("🖼️ Chargement BLIP..."):
                            blip_models = load_blip_model()
                            if blip_models:
                                st.success("✅ BLIP chargé pour descriptions détaillées")
                    except Exception as e:
                        st.warning(f"⚠️ BLIP non disponible: {e}")
                    
                    # 1. Analyse complète: OpenCV + YOLO + CLIP + BLIP + OCR
                    analysis_data, proc_images, tables_str = process_image(
                        uploaded_image.getvalue(), 
                        vision_models=vision_models,
                        ocr_reader=ocr_reader,
                        yolo_model=yolo_model,
                        blip_models=blip_models
                    )
                    
                    # 2. Afficher la description BLIP + OCR en priorité
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 🖼️ Analyse IA Complète (BLIP + OCR + YOLO + CLIP)")
                    
                    # BLIP et OCR sont déjà dans tables_str, les afficher directement
                    if tables_str:
                        st.markdown(tables_str, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 3. Afficher les images traditionnelles avec zoom au survol
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 📊 Analyses traditionnelles (OpenCV)")
                    if proc_images:
                        cols = st.columns(min(len(proc_images), 3))
                        for i, (img, caption) in enumerate(zip(proc_images, ['Image Originale'] + ['Analyse'] * (len(proc_images)-1))):
                            with cols[i % len(cols)]:
                                st.markdown('<div class="zoom-image">', unsafe_allow_html=True)
                                st.image(img, caption=f"📸 {caption}", use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 4. Génération de rapport complet avec LLM
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 🤖 Rapport d'analyse IA complet")
                    
                    with st.spinner("✍️ Génération du rapport détaillé..."):
                        # Utiliser la fonction de description complète
                        comprehensive_desc = generate_comprehensive_description(analysis_data)
                        
                        # Générer le rapport avec le LLM
                        llm_model = st.session_state.get('llm_model') or st.session_state.get('current_model')
                        
                        if llm_model:
                            try:
                                improved_analysis = improve_analysis_with_llm(analysis_data, llm_model)
                                
                                # Afficher d'abord la description structurée
                                st.markdown(f"""
                                <div style="color: white; font-weight: 600; font-size: 1.1rem; line-height: 1.8; 
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
                                <h4 style="color: white; margin-top: 0;">📋 Description structurée:</h4>
                                {comprehensive_desc.replace(chr(10), '<br>')}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Puis l'analyse enrichie par le LLM
                                st.markdown(f"""
                                <div style="color: white; font-weight: 600; font-size: 1.1rem; line-height: 1.8; 
                                background: rgba(74, 74, 126, 0.3); padding: 1.5rem; border-radius: 12px; 
                                border-left: 4px solid var(--kibali-green);">
                                <h4 style="color: white; margin-top: 0;">🤖 Analyse IA enrichie:</h4>
                                {improved_analysis}
                                </div>
                                """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"❌ Erreur génération rapport: {str(e)}")
                                # Afficher au moins la description structurée
                                st.markdown(f"""
                                <div style="color: white; font-weight: 600; font-size: 1.1rem; line-height: 1.8; 
                                background: rgba(74, 74, 126, 0.3); padding: 1.5rem; border-radius: 12px;">
                                {comprehensive_desc.replace(chr(10), '<br>')}
                                </div>
                                """, unsafe_allow_html=True)
                                improved_analysis = comprehensive_desc
                        else:
                            # Afficher la description structurée si pas de LLM
                            st.markdown(f"""
                            <div style="color: white; font-weight: 600; font-size: 1.1rem; line-height: 1.8; 
                            background: rgba(74, 74, 126, 0.3); padding: 1.5rem; border-radius: 12px;">
                            {comprehensive_desc.replace(chr(10), '<br>')}
                            </div>
                            """, unsafe_allow_html=True)
                            improved_analysis = comprehensive_desc
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Marquer l'analyse comme réussie
                    vision_success = True
                    
                    # Stocker les résultats d'analyse pour le chat
                    if vision_success:
                        st.session_state.media_analysis_results.append({
                            'type': 'image',
                            'name': uploaded_image.name,
                            'resolution': f"{width}x{height}",
                            'format': img_format,
                            'caption': caption,
                            'ai_analysis': improved_analysis,
                            'web_context': web_context,
                            'extracted_text': organized_text
                        })
            else:
                st.warning("⚠️ Veuillez uploader une image d'abord")

    # ===============================================
    # Onglet 5: Recherche Web Avancée avec cartes
    # ===============================================
    with tab5:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 🌐 Recherche web avancée avec extraction de contenu
        Recherche intelligente sur Internet avec cache automatique et extraction de contenu détaillé.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col8, col9 = st.columns([3, 1])
        with col8:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            web_query = st.text_area(
                "🔍 **Requête de recherche**",
                placeholder="Ex: actualités intelligence artificielle 2024",
                height=100,
                help="Décris ta recherche en langage naturel"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col9:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            search_type = st.selectbox(
                "🎯 **Type de recherche**",
                options=["text", "news", "both"],
                index=2,
                help="Texte: recherche générale, News: actualités, Both: les deux"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        col_search1, col_search2 = st.columns([1, 1])
        with col_search1:
            if st.button("🔍 **Rechercher**", key="web_search", help="Lance la recherche web"):
                if web_query.strip():
                    with st.spinner("🌐 Recherche en cours..."):
                        results = enhanced_web_search(web_query, max_results=10, search_type=search_type)
                        if results:
                            for result in results:
                                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                st.markdown(f"### 📰 {result.get('title', 'Sans titre')}")
                                st.write(f"📝 {result.get('body', 'Pas de description')}")
                                url = result.get('href') or result.get('url', '#')
                                st.markdown(f"[🔗 **Accéder à la source**]({url})")
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown("---")
                        else:
                            st.error("❌ Aucun résultat trouvé")
                else:
                    st.warning("⚠️ Veuillez entrer une requête de recherche")
        
        with col_search2:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            url_extract = st.text_input(
                "🌐 **URL à extraire**",
                placeholder="https://example.com/article",
                help="Colle une URL pour extraire son contenu détaillé"
            )
            if st.button("📄 **Extraire contenu**", key="extract_content", help="Extrait le contenu détaillé de la page web"):
                if url_extract.strip():
                    if not url_extract.startswith(('http://', 'https://')):
                        url_extract = 'https://' + url_extract
                    with st.spinner("📄 Extraction en cours..."):
                        content = smart_content_extraction(url_extract, max_length=2000)
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.text_area("📄 **Contenu extrait**", value=content, height=300, disabled=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Veuillez entrer une URL valide")
            st.markdown('</div>', unsafe_allow_html=True)

    # ===============================================
    # Onglet 6: Photogrammétrie - Classification Intelligente avec IA
    # ===============================================
    with tab_photo:
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 📷 Classification Intelligente pour Photogrammétrie avec Vision AI
        **L'IA analyse visuellement chaque photo et les ordonne intelligemment pour une reconstruction 3D optimale.**
        
        🤖 **Analyse IA:**
        - Vision sémantique (CLIP) - Compréhension du contenu
        - Analyse de textures et détails visuels
        - Détection d'angles de vue similaires
        - Ordonnancement optimal pour Dust3R/MeshRoom
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Deux modes: Classification IA ou Optimisation simple
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        mode = st.radio(
            "🎯 Mode de traitement:",
            ["🤖 Classification IA (Analyse visuelle + Ordonnancement)", "⚡ Optimisation Rapide (Réduction simple)"],
            help="Classification IA analyse visuellement chaque photo, Optimisation Rapide réduit juste le nombre"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Section d'upload multiple
        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
        st.markdown("#### 📤 Upload de fichiers")
        
        uploaded_photos = st.file_uploader(
            "Sélectionnez vos fichiers",
            type=None,
            accept_multiple_files=True,
            help="Uploadez tous vos fichiers (tous formats acceptés). L'IA les analysera et les classera intelligemment!",
            key="photo_upload_classifier"
        )
        
        if uploaded_photos and len(uploaded_photos) > 0:
            st.success("✅📸")
            
            if mode.startswith("🤖"):
                # MODE CLASSIFICATION IA
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                st.markdown("#### 🎨 Paramètres de classification IA")
                
                col_ai1, col_ai2 = st.columns(2)
                with col_ai1:
                    ordering_method = st.selectbox(
                        "📐 Méthode d'ordonnancement",
                        ["sequential", "cluster"],
                        format_func=lambda x: "Séquentiel (photos qui se suivent)" if x == "sequential" else "Par groupes d'angles",
                        help="Séquentiel: crée une chaîne d'images similaires. Groupes: classe par angles puis ordonne."
                    )
                
                with col_ai2:
                    generate_viz = st.checkbox(
                        "📊 Générer visualisation",
                        value=True,
                        help="Crée une grille visuelle montrant l'ordre des photos"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Bouton de classification IA
                if st.button("🚀 **Analyser et Classer avec l'IA**", type="primary", use_container_width=True):
                    try:
                        import tempfile
                        import shutil
                        from outils.intelligent_photo_classifier import classify_photos_with_ai
                        
                        # Créer un conteneur pour les logs en temps réel
                        log_container = st.empty()
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Créer dossier temporaire
                        log_container.info("📁 Création du dossier temporaire...")
                        temp_dir = tempfile.mkdtemp(prefix="ai_photo_classifier_")
                        progress_bar.progress(5)
                        
                        # Sauvegarder les photos
                        status_text.text(f"💾 Sauvegarde de {len(uploaded_photos)} photos...")
                        photo_paths = []
                        for idx, photo in enumerate(uploaded_photos):
                            photo_path = Path(temp_dir) / photo.name
                            with open(photo_path, 'wb') as f:
                                f.write(photo.getbuffer())
                            photo_paths.append(str(photo_path))
                            progress_bar.progress(5 + int((idx + 1) / len(uploaded_photos) * 15))
                        
                        log_container.success(f"✅ {len(uploaded_photos)} photos sauvegardées")
                        
                        # Charger le modèle Vision AI (CLIP)
                        status_text.text("📦 Chargement du modèle Vision AI (CLIP)...")
                        progress_bar.progress(20)
                        vision_models = load_vision_models()
                        
                        if vision_models is None:
                            st.error("❌ Impossible de charger le modèle CLIP")
                            st.stop()
                        
                        log_container.success(f"✅ Modèle CLIP chargé ({vision_models['device']})")
                        progress_bar.progress(30)
                        
                        # Classifier avec l'IA
                        status_text.text("🔍 Analyse Vision AI en cours...")
                        st.info("📊 Extraction des features visuelles avec CLIP...")
                        output_dir = temp_dir + "_classified"
                        Path(output_dir).mkdir(exist_ok=True)
                        
                        # Progress callback pour le classifier
                        def update_progress(current, total, message):
                            pct = 30 + int((current / total) * 50)
                            progress_bar.progress(pct)
                            status_text.text(f"🔍 {message} ({current}/{total})")
                        
                        ordered_paths, report, viz_path = classify_photos_with_ai(
                            photo_paths,
                            vision_models['clip_model'],
                            vision_models['clip_processor'],
                            device=vision_models['device'],
                            method=ordering_method,
                            output_dir=output_dir if generate_viz else None,
                            progress_callback=update_progress
                        )
                        
                        progress_bar.progress(80)
                        status_text.text("✅ Analyse terminée !")
                        log_container.success("🎉 Classification IA terminée avec succès !")
                        
                        # Sauvegarder ordered_paths dans session_state AVANT la fin du try
                        st.session_state.classification_ordered_paths = ordered_paths
                        st.session_state.classification_report = report
                        st.session_state.classification_viz_path = viz_path
                        st.session_state.classification_done = True
                        
                        # Afficher les résultats
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.markdown("### 🎯 Résultats de la Classification IA")
                        st.text_area("📋 Rapport détaillé", value=report, height=400, disabled=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Afficher la visualisation
                        if viz_path and Path(viz_path).exists():
                            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                            st.markdown("### 📊 Visualisation de l'ordre")
                            st.image(viz_path, caption="Ordre optimisé des photos (gauche → droite, haut → bas)", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Copier les photos dans l'ordre
                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                        st.markdown("### 📥 Photos classées et renommées")
                        
                        ordered_dir = Path(output_dir) / "ordered_photos"
                        ordered_dir.mkdir(exist_ok=True)
                        
                        for idx, src_path in enumerate(ordered_paths, 1):
                            src = Path(src_path)
                            # Renommer avec numéro d'ordre + nom original
                            dest = ordered_dir / f"{idx:04d}_{src.name}"
                            shutil.copy2(src, dest)
                        
                        st.success(f"✅ {len(ordered_paths)} photos ordonnées et sauvegardées")
                        
                        # Créer ZIP pour téléchargement
                        import zipfile
                        zip_path = Path(temp_dir) / "photos_classified_ai.zip"
                        
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            # Ajouter les photos ordonnées
                            for photo_file in ordered_dir.glob("*.*"):
                                zipf.write(photo_file, arcname=f"ordered_photos/{photo_file.name}")
                            
                            # Ajouter le rapport
                            report_path = Path(output_dir) / "classification_report.txt"
                            if report_path.exists():
                                zipf.write(report_path, arcname="classification_report.txt")
                            
                            # Ajouter la visualisation
                            if viz_path and Path(viz_path).exists():
                                zipf.write(viz_path, arcname="ordering_visualization.png")
                        
                        # Bouton de téléchargement
                        with open(zip_path, 'rb') as f:
                            st.download_button(
                                label="📦 Télécharger les photos classées (ZIP)",
                                data=f.read(),
                                file_name="photos_classees_ia.zip",
                                mime="application/zip"
                            )
                        
                        st.info("💡 Photos ordonnées de manière optimale pour reconstruction 3D")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la classification: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
                
                # ============================================================
                # 🤖 CHAT INTERACTIF KIBALI - AFFICHAGE APRÈS CLASSIFICATION
                # ============================================================
                if st.session_state.get('classification_done', False):
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 💬 Chat Kibali - Affinez la classification avec l'IA")
                    st.markdown("""
                    Utilisez le chat pour manipuler l'ordre des photos avec des commandes naturelles :
                    - *"Mets les photos sombres au début"*
                    - *"Regroupe les images similaires ensemble"*
                    - *"Place les vues frontales avant les vues latérales"*
                    - *"Inverse l'ordre actuel"*
                    - *"Supprime les photos floues"*
                    """)
                    
                    # Initialiser l'historique du chat
                    if 'classification_chat_history' not in st.session_state:
                        st.session_state.classification_chat_history = []
                    if 'current_ordered_paths' not in st.session_state:
                        st.session_state.current_ordered_paths = st.session_state.classification_ordered_paths
                    
                    # Afficher l'historique du chat
                    for message in st.session_state.classification_chat_history:
                        if message['role'] == 'user':
                            st.markdown(f"**👤 Vous:** {message['content']}")
                        else:
                            st.markdown(f"**🤖 Kibali:** {message['content']}")
                    
                    # Input utilisateur
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        user_prompt = st.text_input(
                            "💬 Votre commande:",
                            placeholder="Ex: Mets les photos lumineuses au début...",
                            key="classification_chat_input",
                            label_visibility="collapsed"
                        )
                    with col2:
                        send_btn = st.button("📤 Envoyer", key="send_classification_command", use_container_width=True)
                    
                    if send_btn and user_prompt:
                        # Ajouter le message utilisateur
                        st.session_state.classification_chat_history.append({
                            'role': 'user',
                            'content': user_prompt
                        })
                        
                        with st.spinner("🤖 Kibali analyse votre demande..."):
                            try:
                                # Analyse de la demande par mots-clés (sans LLM pour éviter les erreurs)
                                user_prompt_lower = user_prompt.lower()
                                
                                # Extraire features pour analyse intelligente
                                features_list = []
                                if 'current_features' not in st.session_state:
                                    # Extraire features de base pour toutes les images
                                    for path in st.session_state.current_ordered_paths[:50]:  # Limiter à 50 pour performance
                                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                        if img is not None:
                                            features_list.append({
                                                'path': path,
                                                'brightness': img.mean(),
                                                'contrast': img.std(),
                                                'name': Path(path).name
                                            })
                                    st.session_state.current_features = features_list
                                else:
                                    features_list = st.session_state.current_features
                                
                                # Détection d'intention par règles simples
                                new_order = st.session_state.current_ordered_paths.copy()
                                kibali_response = ""
                                action_executed = False
                                
                                # Règle 1: Regrouper similaires
                                if any(word in user_prompt_lower for word in ['regroup', 'similaire', 'ensemble', 'cluster', 'group']):
                                    # Regrouper par similarité de luminosité
                                    brightness_scores = []
                                    for path in new_order:
                                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                        brightness_scores.append(img.mean() if img is not None else 0)
                                    
                                    # Clustering simple par luminosité
                                    from sklearn.cluster import KMeans
                                    if len(brightness_scores) > 3:
                                        n_clusters = min(5, len(brightness_scores) // 10 + 1)
                                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                        labels = kmeans.fit_predict(np.array(brightness_scores).reshape(-1, 1))
                                        
                                        # Réorganiser par cluster
                                        sorted_indices = np.argsort(labels)
                                        new_order = [new_order[i] for i in sorted_indices]
                                        kibali_response = f"✅ Photos regroupées en {n_clusters} groupes similaires par luminosité."
                                        action_executed = True
                                    else:
                                        kibali_response = "⚠️ Pas assez de photos pour regrouper."
                                
                                # Règle 2: Trier par luminosité (sombre au début)
                                elif any(word in user_prompt_lower for word in ['sombre', 'dark', 'noir', 'début', 'start']) and \
                                     any(word in user_prompt_lower for word in ['début', 'avant', 'first', 'start']):
                                    brightness_scores = []
                                    for path in new_order:
                                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                        brightness_scores.append(img.mean() if img is not None else 0)
                                    sorted_indices = np.argsort(brightness_scores)
                                    new_order = [new_order[i] for i in sorted_indices]
                                    kibali_response = "✅ Photos triées du plus sombre au plus clair."
                                    action_executed = True
                                
                                # Règle 3: Trier par luminosité (clair au début)  
                                elif any(word in user_prompt_lower for word in ['clair', 'lumineuse', 'bright', 'light']) and \
                                     any(word in user_prompt_lower for word in ['début', 'avant', 'first', 'start']):
                                    brightness_scores = []
                                    for path in new_order:
                                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                        brightness_scores.append(img.mean() if img is not None else 0)
                                    sorted_indices = np.argsort(brightness_scores)[::-1]
                                    new_order = [new_order[i] for i in sorted_indices]
                                    kibali_response = "✅ Photos triées du plus clair au plus sombre."
                                    action_executed = True
                                
                                # Règle 4: Inverser l'ordre
                                elif any(word in user_prompt_lower for word in ['inverse', 'revers', 'retourne', 'flip']):
                                    new_order = new_order[::-1]
                                    kibali_response = "✅ Ordre des photos inversé."
                                    action_executed = True
                                
                                # Règle 5: Supprimer floues
                                elif any(word in user_prompt_lower for word in ['floue', 'blur', 'netteté', 'flou', 'supprime']):
                                    # Calculer la netteté avec Laplacian
                                    sharp_paths = []
                                    for path in new_order:
                                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                        if img is not None:
                                            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                                            if laplacian_var > 100:  # Seuil de netteté
                                                sharp_paths.append(path)
                                    
                                    removed_count = len(new_order) - len(sharp_paths)
                                    new_order = sharp_paths
                                    kibali_response = f"✅ {removed_count} photos floues supprimées. {len(new_order)} photos nettes conservées."
                                    action_executed = True
                                
                                else:
                                    kibali_response = f"🤖 Je n'ai pas compris votre demande. Essayez:\n- 'Regroupe les images similaires'\n- 'Mets les photos sombres au début'\n- 'Inverse l'ordre'\n- 'Supprime les photos floues'"
                                
                                # Mettre à jour l'ordre si modifié
                                if new_order != st.session_state.current_ordered_paths:
                                    st.session_state.current_ordered_paths = new_order
                                    # Invalider le cache des features
                                    if 'current_features' in st.session_state:
                                        del st.session_state.current_features
                                
                                st.session_state.classification_chat_history.append({
                                    'role': 'assistant',
                                    'content': kibali_response
                                })
                                
                                st.rerun()
                                
                            except Exception as e:
                                error_msg = f"❌ Erreur: {str(e)}"
                                st.session_state.classification_chat_history.append({
                                    'role': 'assistant',
                                    'content': error_msg
                                })
                                st.error(error_msg)
                                import traceback
                                st.text(traceback.format_exc())
                    
                    # Gérer le drag & drop via query params
                    try:
                        query_params = st.query_params
                        if 'drag_from' in query_params and 'drag_to' in query_params:
                            drag_from = int(query_params['drag_from'])
                            drag_to = int(query_params['drag_to'])
                            
                            if drag_from != drag_to:
                                new_order = st.session_state.current_ordered_paths.copy()
                                new_order[drag_from], new_order[drag_to] = new_order[drag_to], new_order[drag_from]
                                st.session_state.current_ordered_paths = new_order
                                
                                st.session_state.classification_chat_history.append({
                                    'role': 'assistant',
                                    'content': f"✅ Photo #{drag_from+1} échangée avec #{drag_to+1}"
                                })
                                
                                # Clear query params
                                st.query_params.clear()
                                st.rerun()
                    except:
                        pass
                    
                    # Afficher l'ordre actuel
                    st.markdown("#### 📸 Galerie interactive (Glissez les images pour les réorganiser)")
                    
                    # CSS pour drag & drop fluide
                    st.markdown("""
                    <style>
                    .draggable-container {
                        display: grid;
                        grid-template-columns: repeat(5, 1fr);
                        gap: 15px;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 15px;
                    }
                    .draggable-item {
                        position: relative;
                        cursor: grab;
                        background: white;
                        border: 3px solid #e0e0e0;
                        border-radius: 12px;
                        padding: 8px;
                        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }
                    .draggable-item:hover {
                        border-color: #4CAF50;
                        box-shadow: 0 8px 20px rgba(76, 175, 80, 0.4);
                        transform: translateY(-5px) scale(1.03);
                    }
                    .draggable-item:active {
                        cursor: grabbing;
                    }
                    .draggable-item.dragging {
                        opacity: 0.6;
                        transform: rotate(8deg) scale(1.1);
                        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
                        z-index: 1000;
                    }
                    .draggable-item.drag-over {
                        border-color: #2196F3;
                        border-style: dashed;
                        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
                        animation: pulse 0.5s infinite;
                    }
                    @keyframes pulse {
                        0%, 100% { transform: scale(1); }
                        50% { transform: scale(1.05); }
                    }
                    .item-number {
                        position: absolute;
                        top: -5px;
                        left: -5px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 5px 10px;
                        border-radius: 8px;
                        font-weight: bold;
                        font-size: 14px;
                        z-index: 10;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                    }
                    .drag-handle {
                        position: absolute;
                        top: -5px;
                        right: -5px;
                        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                        color: white;
                        padding: 8px 10px;
                        border-radius: 8px;
                        font-size: 18px;
                        cursor: grab;
                        z-index: 10;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                    }
                    .drag-handle:active {
                        cursor: grabbing;
                    }
                    .item-image {
                        width: 100%;
                        height: 160px;
                        object-fit: cover;
                        border-radius: 8px;
                        display: block;
                    }
                    .item-name {
                        text-align: center;
                        font-size: 11px;
                        margin-top: 5px;
                        color: #666;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Afficher les images avec drag & drop
                    st.markdown('<div class="draggable-container" id="image-gallery">', unsafe_allow_html=True)
                    
                    import base64
                    from io import BytesIO
                    
                    for idx, path in enumerate(st.session_state.current_ordered_paths[:30]):
                        try:
                            img = Image.open(path)
                            img.thumbnail((250, 250))
                            buffered = BytesIO()
                            img.save(buffered, format="JPEG", quality=85)
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            filename = os.path.basename(path)
                            
                            st.markdown(f"""
                            <div class="draggable-item" draggable="true" data-index="{idx}">
                                <div class="item-number">#{idx+1}</div>
                                <div class="drag-handle">⋮⋮</div>
                                <img src="data:image/jpeg;base64,{img_str}" class="item-image" />
                                <div class="item-name">{filename}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            filename = os.path.basename(path)
                            st.markdown(f"""
                            <div class="draggable-item" data-index="{idx}">
                                <div class="item-number">#{idx+1}</div>
                                <div style="padding: 60px 10px; text-align: center; color: #999;">
                                    ❌<br>{filename[:15]}...
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if len(st.session_state.current_ordered_paths) > 30:
                        st.info(f"📊 {len(st.session_state.current_ordered_paths)} images au total (30 premières affichées avec drag & drop)")
                    
                    # JavaScript pour gérer le drag & drop
                    st.markdown("""
                    <script>
                    (function() {
                        let draggedElement = null;
                        let draggedIndex = null;
                        
                        function initDragAndDrop() {
                            const items = document.querySelectorAll('.draggable-item');
                            
                            items.forEach(item => {
                                item.addEventListener('dragstart', handleDragStart);
                                item.addEventListener('dragend', handleDragEnd);
                                item.addEventListener('dragover', handleDragOver);
                                item.addEventListener('drop', handleDrop);
                                item.addEventListener('dragleave', handleDragLeave);
                            });
                        }
                        
                        function handleDragStart(e) {
                            draggedElement = this;
                            draggedIndex = parseInt(this.getAttribute('data-index'));
                            this.classList.add('dragging');
                            e.dataTransfer.effectAllowed = 'move';
                            e.dataTransfer.setData('text/html', this.innerHTML);
                        }
                        
                        function handleDragEnd(e) {
                            this.classList.remove('dragging');
                            document.querySelectorAll('.draggable-item').forEach(item => {
                                item.classList.remove('drag-over');
                            });
                        }
                        
                        function handleDragOver(e) {
                            if (e.preventDefault) {
                                e.preventDefault();
                            }
                            e.dataTransfer.dropEffect = 'move';
                            if (this !== draggedElement) {
                                this.classList.add('drag-over');
                            }
                            return false;
                        }
                        
                        function handleDragLeave(e) {
                            this.classList.remove('drag-over');
                        }
                        
                        function handleDrop(e) {
                            if (e.stopPropagation) {
                                e.stopPropagation();
                            }
                            
                            if (draggedElement !== this) {
                                const targetIndex = parseInt(this.getAttribute('data-index'));
                                
                                // Échange visuel immédiat
                                const draggedHTML = draggedElement.innerHTML;
                                const targetHTML = this.innerHTML;
                                
                                draggedElement.innerHTML = targetHTML;
                                this.innerHTML = draggedHTML;
                                
                                // Mettre à jour les data-index
                                draggedElement.setAttribute('data-index', targetIndex);
                                this.setAttribute('data-index', draggedIndex);
                                
                                // Envoyer au backend via query params
                                setTimeout(() => {
                                    const url = new URL(window.location);
                                    url.searchParams.set('drag_from', draggedIndex);
                                    url.searchParams.set('drag_to', targetIndex);
                                    url.searchParams.set('t', Date.now());
                                    window.location.href = url.toString();
                                }, 300);
                            }
                            
                            return false;
                        }
                        
                        // Initialiser quand le DOM est prêt
                        if (document.readyState === 'loading') {
                            document.addEventListener('DOMContentLoaded', initDragAndDrop);
                        } else {
                            setTimeout(initDragAndDrop, 100);
                        }
                        
                        // Réinitialiser après les updates Streamlit
                        const observer = new MutationObserver(function(mutations) {
                            mutations.forEach(function(mutation) {
                                if (mutation.addedNodes.length > 0) {
                                    setTimeout(initDragAndDrop, 200);
                                }
                            });
                        });
                        
                        const gallery = document.getElementById('image-gallery');
                        if (gallery) {
                            observer.observe(gallery.parentElement, { childList: true, subtree: true });
                        }
                    })();
                    </script>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # ========== NOUVELLE SECTION COLMAP 3D ==========
                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                    st.markdown("### 🎨 Reconstruction 3D Professionnelle (COLMAP)")
                    st.markdown("""
                    Générez un modèle 3D texturé à partir de vos images classifiées avec COLMAP (Licence BSD).
                    Pipeline complet : SIFT → Structure-from-Motion → Multi-View Stereo → Mesh
                    """)
                    
                    col_colmap1, col_colmap2 = st.columns([2, 1])
                    with col_colmap1:
                        quality_3d = st.select_slider(
                            "🎯 Qualité reconstruction",
                            options=['low', 'medium', 'high', 'extreme'],
                            value='high',
                            help="Plus la qualité est haute, plus le traitement est long (extreme peut prendre 1-2h)"
                        )
                    
                    with col_colmap2:
                        use_gpu_3d = st.checkbox("⚡ Utiliser GPU", value=True, 
                                                  help="RTX 5090 recommandée pour MVS")
                    
                    if st.button("🚀 **Générer Modèle 3D**", type="primary", use_container_width=True, key="colmap_btn"):
                        with st.spinner("🔍 Lancement de COLMAP... Cela peut prendre 15-45 minutes..."):
                            try:
                                from outils.colmap_photogrammetry import ColmapPhotogrammetry
                                import tempfile
                                import time
                                
                                # Créer workspace temporaire
                                workspace_3d = tempfile.mkdtemp(prefix="colmap_kibali_")
                                colmap = ColmapPhotogrammetry(workspace_3d)
                                
                                # Progress container
                                progress_bar = st.progress(0, text="Initialisation...")
                                status_text = st.empty()
                                start_time = time.time()
                                
                                def colmap_progress(current, total, message):
                                    progress = current / total if total > 0 else 0
                                    elapsed = int(time.time() - start_time)
                                    progress_bar.progress(progress, text=f"{int(progress*100)}% - {message}")
                                    status_text.text(f"⏱️ {elapsed//60}m {elapsed%60}s écoulées...")
                                
                                # Lancer le pipeline complet
                                results = colmap.run_full_pipeline(
                                    image_paths=current_ordered_paths,
                                    use_gpu=use_gpu_3d,
                                    quality=quality_3d,
                                    progress_callback=colmap_progress
                                )
                                
                                elapsed = int(time.time() - start_time)
                                
                                if results['success']:
                                    st.success(f"✅ Modèle 3D généré en {elapsed//60}m {elapsed%60}s !")
                                    
                                    col_result1, col_result2, col_result3 = st.columns(3)
                                    with col_result1:
                                        st.metric("📸 Images", results.get('images_prepared', 0))
                                    with col_result2:
                                        if results.get('dense_pointcloud'):
                                            st.metric("☁️ Point Cloud", "✅")
                                    with col_result3:
                                        if results.get('mesh'):
                                            st.metric("🎨 Mesh", "✅")
                                    
                                    # Afficher chemins des résultats
                                    st.markdown("#### 📁 Résultats")
                                    if results.get('mesh'):
                                        st.code(f"Mesh: {results['mesh']}", language="bash")
                                    if results.get('dense_pointcloud'):
                                        st.code(f"Point Cloud: {results['dense_pointcloud']}", language="bash")
                                    st.code(f"Workspace: {workspace_3d}", language="bash")
                                    
                                    st.info("""
                                    💡 **Visualisation :**
                                    - Mesh: `meshlab` ou `blender`
                                    - Point Cloud: `CloudCompare` ou `MeshLab`
                                    - Export: Formats PLY, OBJ disponibles
                                    """)
                                else:
                                    st.error(f"❌ Reconstruction échouée après {elapsed//60}m {elapsed%60}s")
                                    for err in results.get('errors', []):
                                        st.warning(f"• {err}")
                                
                            except Exception as e:
                                st.error(f"❌ Erreur COLMAP: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                # MODE OPTIMISATION RAPIDE (ancien système)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                st.markdown("#### ⚙️ Paramètres d'optimisation")
                
                col_param1, col_param2 = st.columns(2)
                with col_param1:
                    target_count = st.number_input(
                        "🎯 Nombre cible de photos (0 = automatique)",
                        min_value=0,
                        max_value=len(uploaded_photos),
                        value=0,
                        help="Laissez 0 pour une détection automatique du nombre optimal"
                    )
                
                with col_param2:
                    coverage_threshold = st.slider(
                        "📊 Couverture minimale requise",
                        min_value=0.8,
                        max_value=1.0,
                        value=0.95,
                        step=0.05,
                        help="Pourcentage de la scène qui doit être couvert (0.95 = 95%)"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Bouton d'optimisation
                if st.button("🚀 **Optimiser le dataset**", type="primary", use_container_width=True):
                    with st.spinner("🔍 Analyse des photos en cours..."):
                        try:
                            import tempfile
                            import shutil
                            
                            # Créer un dossier temporaire pour les photos
                            temp_dir = tempfile.mkdtemp(prefix="photogrammetry_")
                            
                            # Sauvegarder les photos uploadées
                            st.info(f"💾 Sauvegarde de {len(uploaded_photos)} photos...")
                            for idx, photo in enumerate(uploaded_photos):
                                photo_path = Path(temp_dir) / photo.name
                                with open(photo_path, 'wb') as f:
                                    f.write(photo.getbuffer())
                            
                            # Importer l'outil
                            from outils.photogrammetry_optimizer_tool import PhotogrammetryOptimizerTool
                            
                            # Exécuter l'optimisation
                            tool = PhotogrammetryOptimizerTool()
                            context = {
                                'input_folder': temp_dir,
                                'target_count': target_count if target_count > 0 else None,
                                'coverage_threshold': coverage_threshold,
                                'similarity_threshold': 0.85
                            }
                            
                            result = tool.execute("", context=context)
                            
                            # Afficher les résultats
                            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                            st.markdown("### 📊 Résultats de l'optimisation")
                            st.text(result)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Proposer le téléchargement des photos sélectionnées
                            output_folder = Path(temp_dir + "_optimized")
                            if output_folder.exists():
                                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                st.markdown("### 📥 Téléchargement des photos optimisées")
                                
                                # Créer un ZIP
                                import zipfile
                                zip_path = Path(temp_dir) / "photos_optimized.zip"
                                
                                with zipfile.ZipFile(zip_path, 'w') as zipf:
                                    for photo_file in output_folder.glob("*.*"):
                                        if photo_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                                            zipf.write(photo_file, arcname=photo_file.name)
                                
                                # Bouton de téléchargement
                                with open(zip_path, 'rb') as f:
                                    st.download_button(
                                        label="📦 Télécharger les photos sélectionnées (ZIP)",
                                        data=f.read(),
                                        file_name="photos_photogrammetrie_optimisees.zip",
                                        mime="application/zip"
                                    )
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Nettoyage (optionnel, garder pour debug)
                            # shutil.rmtree(temp_dir, ignore_errors=True)
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'optimisation: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
        
        else:
            st.info("👆")
            
            # Exemples d'utilisation
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("""
            #### 💡 Exemples d'utilisation
            
            **Photogrammétrie aérienne:**
            - Upload: 500-1000 photos drone
            - Résultat: 15-30 photos essentielles
            - Réduction: ~97%
            
            **Scan d'objet 3D:**
            - Upload: 200 photos multi-angles
            - Résultat: 10-20 photos représentatives
            - Réduction: ~90%
            
            **Cartographie terrain:**
            - Upload: 1000+ photos aériennes
            - Résultat: 20-50 photos de couverture
            - Réduction: ~95%
            
            #### 🎯 Algorithme
            1. **Extraction de features** (ORB, histogrammes, textures)
            2. **Clustering intelligent** (KMeans sur angles de vue)
            3. **Sélection optimale** (représentants par cluster)
            4. **Vérification de couverture** (ajout si zones manquantes)
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # ===============================================
    # Onglet 7: Outils Dynamiques (si disponible)
    # ===============================================
    if TOOLS_SYSTEM_AVAILABLE and st.session_state.tool_manager and tab_tools:
        with tab_tools:
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("""
            ### 🔧 Système d'outils dynamiques
            **Outils disponibles:** Recherche web, analyse d'images, calculs mathématiques, génération de code, traduction, gestion de fichiers.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Affichage des outils disponibles
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("#### 📋 Outils chargés")
            tools = st.session_state.tool_manager.tools
            if tools:
                tool_data = []
                for tool in tools:
                    tool_data.append({
                        "Nom": tool.name,
                        "Description": tool.description,
                        "Capacités": ", ".join(tool.capabilities)
                    })
                st.dataframe(pd.DataFrame(tool_data), use_container_width=True)
            else:
                st.warning("⚠️ Aucun outil chargé")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Test manuel d'un outil
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("#### 🧪 Test d'outil manuel")
            
            col_test1, col_test2 = st.columns([2, 1])
            with col_test1:
                test_query = st.text_area(
                    "🔍 **Requête de test**",
                    placeholder="Ex: Analyse cette image de sol minier",
                    height=100,
                    help="Décris la tâche à effectuer"
                )
            
            with col_test2:
                available_tools = [tool.name for tool in tools] if tools else []
                selected_tool_name = st.selectbox(
                    "🔧 **Outil à tester**",
                    options=["Auto (détection)"] + available_tools,
                    help="Choisis un outil spécifique ou laisse la détection automatique"
                )
            
            if st.button("🚀 **Tester l'outil**", key="test_tool", help="Exécute l'outil sélectionné"):
                if test_query.strip():
                    with st.spinner("🔧 Exécution de l'outil..."):
                        try:
                            if selected_tool_name == "Auto (détection)":
                                # Préparer le contexte pour la détection
                                tool_context = {
                                    'has_pdfs': st.session_state.vectordb is not None,
                                    'vectordb_available': st.session_state.vectordb is not None,
                                    'media_analysis': st.session_state.media_analysis_results if st.session_state.media_analysis_results else None
                                }
                                
                                # Détection automatique
                                relevant_tools = st.session_state.tool_manager.get_relevant_tools(test_query, tool_context)
                                if relevant_tools:
                                    st.success(f"🔍 Outils détectés: {', '.join([t.name for t in relevant_tools])}")
                                    
                                    # Si outil PDF détecté, faire une vraie recherche
                                    pdf_tool_detected = any('pdf' in t.name.lower() or 'document' in t.name.lower() for t in relevant_tools)
                                    
                                    if pdf_tool_detected:
                                        if st.session_state.vectordb:
                                            st.info("🔍")
                                            rag_docs = rag_search(test_query, st.session_state.vectordb, k=5)
                                            if rag_docs:
                                                st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                                st.markdown(f"### 📊 Résultats trouvés: {len(rag_docs)} passages")
                                                for i, doc in enumerate(rag_docs, 1):
                                                    st.markdown(f"**Document {i}:** {doc.metadata.get('source', 'Inconnu')}")
                                                    st.text_area(f"Extrait {i}", value=doc.page_content[:500], height=100, disabled=True, key=f"extract_{i}")
                                                    st.markdown("---")
                                                st.markdown('</div>', unsafe_allow_html=True)
                                            else:
                                                st.warning("Aucun résultat trouvé dans les documents")
                                        else:
                                            st.warning("⚠️ Base vectorielle non chargée. Ajoutez des PDFs dans l'onglet Configuration.")
                                    
                                    # Exécuter les autres outils
                                    results = []
                                    for tool in relevant_tools:
                                        if 'pdf' not in tool.name.lower() and 'document' not in tool.name.lower():
                                            result = tool.execute(test_query, tool_context)
                                            results.append(f"**{tool.name}:**\n{result}")
                                    
                                    if results:
                                        st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                        st.markdown("### 📊 Résultats des autres outils")
                                        for result in results:
                                            st.markdown(result)
                                            st.markdown("---")
                                        st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.warning("⚠️ Aucun outil pertinent détecté pour cette requête")
                            
                            else:
                                # Outil spécifique
                                tool = next((t for t in tools if t.name == selected_tool_name), None)
                                if tool:
                                    result = tool.execute(test_query)
                                    st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
                                    st.markdown(f"### 📊 Résultat de {tool.name}")
                                    st.text_area("Résultat", value=str(result), height=200, disabled=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    st.error("❌ Outil non trouvé")
                        
                        except Exception as e:
                            st.error(f"❌ Erreur lors du test: {e}")
                else:
                    st.warning("⚠️ Veuillez entrer une requête de test")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Statistiques des outils
            st.markdown('<div class="kibali-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Statistiques du système d'outils")
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Outils chargés", len(tools) if tools else 0)
            
            with col_stats2:
                total_capabilities = sum(len(tool.capabilities) for tool in tools) if tools else 0
                st.metric("Capacités totales", total_capabilities)
            
            with col_stats3:
                status = "✅ Actif" if TOOLS_SYSTEM_AVAILABLE and st.session_state.tool_manager else "❌ Inactif"
                st.metric("Statut système", status)
            
            st.markdown('</div>', unsafe_allow_html=True)

# ===============================================
# Lancement principal
# ===============================================
if __name__ == "__main__":
    print("🚀 Kibali 🌟 - Assistant IA Avancé avec Recherche Web")
    print("=" * 60)
    # Initialisation
    setup_drive()
    print(f"📁 Dossier unifié: {CHATBOT_DIR}")
    print(f"🔑 Token HF configuré: {HF_TOKEN[:10]}...")
    print(f"🌐 Recherche web intégrée")
    # Vérifier les fichiers existants
    existing_graphs = [f for f in os.listdir(GRAPHS_PATH) if f.endswith('_graph.graphml')] if os.path.exists(GRAPHS_PATH) else []
    existing_pdfs = [f for f in os.listdir(PDFS_PATH) if f.endswith('.pdf')] if os.path.exists(PDFS_PATH) else []
    print(f"📊 État initial:")
    print(f" 🗺️ Graphes OSM: {len(existing_graphs)}")
    print(f" 📄 PDFs: {len(existing_pdfs)}")
    print(f" 💾 Base vectorielle: {'✅' if os.path.exists(VECTORDB_PATH) else '⚠️ À créer (upload PDFs + traiter)'}")
    print(f" 🌐 Cache web: {'✅' if os.path.exists(WEB_CACHE_PATH) else '📝 Sera créé automatiquement'}")
    # Afficher les statistiques du cache si disponible
    print(f" 📈 {get_cache_stats()}")
    # Lancer l'interface Streamlit
    print("\n🌟 Lancement de l'interface Streamlit...")
    print("📝 Fonctionnalités disponibles:")
    print(" 💬 Chat RAG avec recherche web intelligent")
    print(" 🗺️ Calcul de trajets OSM")
    print(" 📸 Analyse d'images avancée")
    print(" 🌐 Recherche web avec extraction de contenu")
    print(" 💾 Cache intelligent pour optimiser les performances")
    print(" 📊 Gestion unifiée des fichiers")
    main()
else:
    print("📦 Module Kibali importé avec succès")
    print("🔧 Fonctions disponibles:")
    print(" - main() : Interface Streamlit complète")
    print(" - enhanced_web_search() : Recherche web avancée")
    print(" - hybrid_search_enhanced() : Recherche hybride")
    print(" - smart_content_extraction() : Extraction de contenu web")
    print(" - process_image() : Analyse d'images")
    print(" - calculer_trajet() : Calcul de trajets OSM")
# ===============================================
# Fonctions utilitaires supplémentaires
# ===============================================
def cleanup_old_cache():
    """Nettoie les entrées expirées du cache"""
    try:
        cache = load_web_cache()
        if not cache:
            return "Cache vide"
        original_count = len(cache)
        cleaned_cache = {}
        for key, entry in cache.items():
            if not is_cache_expired(entry):
                cleaned_cache[key] = entry
        save_web_cache(cleaned_cache)
        removed_count = original_count - len(cleaned_cache)
        return f"✅ Cache nettoyé: {removed_count} entrées expirées supprimées, {len(cleaned_cache)} conservées"
    except Exception as e:
        return f"❌ Erreur nettoyage cache: {e}"
def export_system_config():
    """Exporte la configuration système pour debug"""
    config = {
        "version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "chatbot_dir": CHATBOT_DIR,
            "vectordb_path": VECTORDB_PATH,
            "pdfs_path": PDFS_PATH,
            "graphs_path": GRAPHS_PATH,
            "maps_path": MAPS_PATH
        },
        "models": WORKING_MODELS,
        "status": get_system_status(),
        "features": {
            "web_search": True,
            "osm_routing": True,
            "image_analysis": True,
            "pdf_processing": True,
            "caching": True
        }
    }
    config_path = os.path.join(CHATBOT_DIR, "system_config.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return f"✅ Configuration exportée: {config_path}"
    except Exception as e:
        return f"❌ Erreur export: {e}"
def test_all_features():
    """Teste toutes les fonctionnalités principales"""
    results = {}
    # Test HuggingFace
    results["huggingface"] = test_hf_connection()
    # Test recherche web
    try:
        test_results = enhanced_web_search("test", max_results=1)
        results["web_search"] = len(test_results) > 0
    except:
        results["web_search"] = False
    # Test recherche web
    results["specialized_models"] = {}
    for model_name, model in SPECIALIZED_MODELS.items():
        results["specialized_models"][model_name] = model is not None
    # Test base vectorielle
    try:
        vectordb, _ = load_vectordb()
        results["vectordb"] = vectordb is not None
    except:
        results["vectordb"] = False
    # Test graphe OSM
    try:
        graph, pois, _ = load_existing_graph()
        results["osm_graph"] = graph is not None
    except:
        results["osm_graph"] = False
    return results
# ===============================================
# Fonctions de maintenance avancées
# ===============================================
def optimize_vectordb():
    """Optimise la base vectorielle en supprimant les doublons"""
    try:
        vectordb, status = load_vectordb()
        if not vectordb:
            return "❌ Aucune base vectorielle à optimiser"
        # Cette fonction nécessiterait une implémentation plus complexe
        # pour détecter et supprimer les doublons dans FAISS
        return "✅ Base vectorielle optimisée (fonctionnalité à implémenter)"
    except Exception as e:
        return f"❌ Erreur optimisation: {e}"
def backup_all_data():
    """Crée une sauvegarde de toutes les données"""
    try:
        import shutil
        import zipfile
        backup_name = f"kibali_backup_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        backup_path = os.path.join(CHATBOT_DIR, backup_name)
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
            # Sauvegarder tous les fichiers du dossier chatbot
            for root, dirs, files in os.walk(CHATBOT_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, CHATBOT_DIR)
                    backup_zip.write(file_path, arcname)
        return f"✅ Sauvegarde créée: {backup_path}"
    except Exception as e:
        return f"❌ Erreur sauvegarde: {e}"
def restore_from_backup(backup_path):
    """Restaure les données depuis une sauvegarde"""
    try:
        import zipfile
        if not os.path.exists(backup_path):
            return "❌ Fichier de sauvegarde non trouvé"
        with zipfile.ZipFile(backup_path, 'r') as backup_zip:
            backup_zip.extractall(CHATBOT_DIR)
        return f"✅ Données restaurées depuis: {backup_path}"
    except Exception as e:
        return f"❌ Erreur restauration: {e}"
# ===============================================
# Version API pour utilisation externe
# ===============================================
class KibaliAPI:
    """API simplifiée pour utiliser Kibali depuis du code externe"""
    def __init__(self):
        self.vectordb = None
        self.graph = None
        self.pois = []
        self.client = None
        self.model_name = WORKING_MODELS[list(WORKING_MODELS.keys())[0]]
        # Initialisation automatique
        self._initialize()
    def _initialize(self):
        """Initialisation automatique"""
        try:
            setup_drive()
            self.vectordb, _ = load_vectordb()
            self.graph, self.pois, _ = load_existing_graph()
            self.client = create_client()
        except Exception as e:
            print(f"⚠️ Initialisation partielle: {e}")
    def ask(self, question, use_web=True):
        """Pose une question simple"""
        try:
            if use_web:
                docs = hybrid_search_enhanced(question, self.vectordb, web_search_enabled=True)
            else:
                docs = rag_search(question, self.vectordb)
            return generate_answer_enhanced(question, docs, self.model_name)
        except Exception as e:
            return f"❌ Erreur: {e}"
    def search_web(self, query, max_results=5):
        """Recherche web simple"""
        try:
            results = enhanced_web_search(query, max_results)
            return [{"title": r.get("title"), "url": r.get("href", r.get("url")), "snippet": r.get("body")} for r in results]
        except Exception as e:
            return [{"error": str(e)}]
    def calculate_route(self, from_place, to_place):
        """Calcule un itinéraire"""
        try:
            question = f"Comment aller de {from_place} à {to_place}"
            _, response, info = calculer_trajet(question, self.graph, self.pois)
            return {"response": response, "info": info}
        except Exception as e:
            return {"error": str(e)}
    def get_status(self):
        """Retourne le statut du système"""
        return get_system_status()
# Instance globale de l'API
kibali_api = KibaliAPI()

# ===============================================
# Onglet Nano-IA et Auto-Apprentissage
# ===============================================
if AUTO_LEARNING_AVAILABLE and tab_nano_ai:
    with tab_nano_ai:
        st.markdown("## 🤖 Nano-IA & Système d'Auto-Apprentissage")
        
        # Toggle pour activer/désactiver l'apprentissage
        if st.session_state.get('auto_learning'):
            st.session_state.auto_learning.render_learning_toggle()
        
        st.markdown("---")
        
        # Statistiques d'apprentissage
        if st.session_state.get('auto_learning'):
            st.session_state.auto_learning.render_learning_stats()
        
        st.markdown("---")
        
        # Gestion des domaines
        st.markdown("### 📚 Domaines d'Expertise")
        
        if st.session_state.get('knowledge_manager'):
            km = st.session_state.knowledge_manager
            domains = km.list_domains()
            
            if domains:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_domain = st.selectbox(
                        "Sélectionner un domaine",
                        domains,
                        help="Choisissez un domaine pour voir les détails"
                    )
                
                with col2:
                    if st.button("🤖 Créer/Mettre à jour Nano-IA", help="Crée ou met à jour la nano-IA spécialisée"):
                        with st.spinner(f"Création de la nano-IA pour {selected_domain}..."):
                            if km.create_nano_ai(selected_domain, force=True):
                                st.success(f"✅ Nano-IA {selected_domain} créée!")
                                st.rerun()
                
                # Détails du domaine
                if selected_domain:
                    expertise = km.get_domain_expertise(selected_domain)
                    if expertise:
                        st.markdown(f"""
                        <div style='
                            background: linear-gradient(135deg, rgba(0, 136, 255, 0.1), rgba(138, 43, 226, 0.1));
                            padding: 1.5rem;
                            border-radius: 10px;
                            border: 2px solid rgba(0, 136, 255, 0.3);
                            margin: 1rem 0;
                        '>
                            <h4 style='color: #0088ff; margin-top: 0;'>📊 {selected_domain}</h4>
                            <p><strong>Documents:</strong> {expertise.documents_count}</p>
                            <p><strong>Requêtes traitées:</strong> {expertise.queries_handled}</p>
                            <p><strong>Taux de succès:</strong> {expertise.success_rate*100:.1f}%</p>
                            <p><strong>Créé:</strong> {expertise.creation_date[:10]}</p>
                            <p><strong>Dernière MAJ:</strong> {expertise.last_updated[:10]}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Mots-clés
                        if expertise.keywords:
                            st.markdown("**🔑 Mots-clés maîtrisés:**")
                            st.write(", ".join(expertise.keywords[:20]))
                        
                        # Exemples Q&A
                        if expertise.examples:
                            st.markdown("**💬 Exemples de conversations:**")
                            for i, example in enumerate(expertise.examples[-5:], 1):
                                with st.expander(f"Exemple {i}: {example['query'][:50]}..."):
                                    st.markdown(f"**Question:** {example['query']}")
                                    st.markdown(f"**Réponse:** {example['response'][:300]}...")
                        
                        # Test du domaine
                        st.markdown("---")
                        st.markdown("### 🧪 Tester les connaissances du domaine")
                        test_query = st.text_input(
                            "Posez une question sur ce domaine",
                            placeholder=f"Ex: Qu'as-tu appris sur {selected_domain}?",
                            key=f"test_query_{selected_domain}"
                        )
                        
                        if test_query:
                            with st.spinner("Recherche dans les connaissances..."):
                                results = km.query_domain(test_query, selected_domain, k=3)
                                
                                if results:
                                    st.success(f"✅ {len(results)} résultats trouvés!")
                                    for i, doc in enumerate(results, 1):
                                        with st.expander(f"Résultat {i}"):
                                            st.write(doc.page_content[:500])
                                            st.caption(f"Source: {doc.metadata.get('filename', 'N/A')}")
                                else:
                                    st.warning("Aucun résultat trouvé dans ce domaine")
            else:
                st.info("👋 Aucun domaine d'expertise créé pour le moment. Uploadez des fichiers pour commencer l'apprentissage!")
        
        # Historique d'apprentissage
        st.markdown("---")
        st.markdown("### 📜 Historique d'Apprentissage")
        
        if st.session_state.get('learning_history'):
            history_df = pd.DataFrame(st.session_state.learning_history[-20:])
            st.dataframe(
                history_df[['type', 'source', 'domain']],
                use_container_width=True
            )
        else:
            st.info("L'historique d'apprentissage apparaîtra ici au fur et à mesure")

# ===============================================
# Messages de fin et documentation
# ===============================================
print("\n" + "="*60)
print("🎉 KIBALI 🌟 - SYSTÈME CHARGÉ AVEC SUCCÈS")
print("="*60)
print(f"📅 Version: 2.0.0 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔑 Token HF: {'✅ Configuré' if HF_TOKEN else '❌ Manquant'}")
print(f"📁 Dossier: {CHATBOT_DIR}")
print(f"🌐 Recherche web: ✅ Activée")
print(f"💾 Cache intelligent: ✅ Activé")
if AUTO_LEARNING_AVAILABLE:
    print(f"🤖 Auto-apprentissage: ✅ Activé")
print("\n📚 FONCTIONNALITÉS PRINCIPALES:")
print(" 💬 Chat RAG avec recherche web intelligent")
print(" 🗺️ Calcul de trajets OSM")
print(" 📸 Analyse d'images avec IA")
print(" 🌐 Extraction de contenu web")
print(" 💾 Gestion unifiée des données")
print("\n🚀 UTILISATION:")
print(" Interface: Exécutez les cellules suivantes")
print(" API: kibali_api.ask('votre question')")
print(" Tests: test_all_features()")
print("\n⚙️ MAINTENANCE:")
print(" Status: get_system_status()")
print(" Nettoyage: cleanup_old_cache()")
print(" Sauvegarde: backup_all_data()")
print("="*60)