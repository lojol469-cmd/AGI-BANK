# -*- coding: utf-8 -*-
"""
Module Outils - Système d'outils dynamiques pour Kibali
"""
import os
import sys
import importlib
import inspect
import glob
import subprocess
import json
import uuid
import datetime
from typing import Dict, List, Any
from abc import ABC, abstractmethod

# ==========================================
# SÉCURITÉ POUR LES IMPORTS RELATIFS
# ==========================================
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ==========================================
# MOTEUR D'EXÉCUTION SÉCURISÉ (Env Portable)
# ==========================================
def run_in_portable(code: str, timeout: int = 60) -> dict:
    """
    Exécute du code Python dans l'environnement portable isolé.
    Injecte automatiquement la date du jour.
    Commande : D:\\kibali-IA\\env_portable\\python.exe
    Dossier : D:\\kibali-IA\\workspace
    """
    python_exe = r"D:\kibali-IA\env_portable\python.exe"
    workspace = r"D:\kibali-IA\workspace"
    os.makedirs(workspace, exist_ok=True)
    
    script_path = os.path.join(workspace, f"task_{uuid.uuid4().hex[:8]}.py")
    
    # INJECTION DE LA DATE DANS TOUS LES SCRIPTS
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    prelude = f'CURRENT_DATE = "{today_str}"\n'
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(prelude + code)
        
    try:
        result = subprocess.run(
            [python_exe, script_path], 
            capture_output=True, text=True, timeout=timeout, cwd=workspace
        )
        
        if result.returncode != 0:
            return {"success": False, "output": f"Erreur d'exécution (Python Portable) :\n{result.stderr}"}
            
        output_lines = result.stdout.strip().split('\n')
        if not output_lines or not output_lines[-1]:
            return {"success": False, "output": "Aucune sortie générée par le script."}
            
        # Récupération de la dernière ligne qui doit être le JSON
        return json.loads(output_lines[-1])
        
    except subprocess.TimeoutExpired:
        return {"success": False, "output": f"Le script a dépassé le temps imparti ({timeout}s)."}
    except json.JSONDecodeError:
        return {"success": False, "output": f"Le script n'a pas renvoyé de JSON valide. Sortie : {result.stdout[-500:]}"}
    except Exception as e:
        return {"success": False, "output": f"Erreur système : {str(e)}"}

# ==========================================
# CLASSE DE BASE POUR TOUS LES OUTILS
# ==========================================
class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "La demande de l'utilisateur"}
            },
            "required": ["query"]
        }

    @abstractmethod
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        pass

    @abstractmethod
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        pass

# ==========================================
# CLASSE POUR LES APPLICATIONS DYNAMIQUES
# ==========================================
class GeneratedAppTool(BaseTool):
    def __init__(self, app_name: str, app_path: str, exe_path: str, description: str):
        self._app_name = app_name
        self._app_path = app_path
        self._exe_path = exe_path
        self._description = description

    @property
    def name(self) -> str:
        return f"launch_{self._app_name}"

    @property
    def description(self) -> str:
        return f"Permet d'ouvrir l'application : {self._description}"

    @property
    def capabilities(self) -> list:
        return ["launch_app", "open_app", "gui", "pyqt6"]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def can_handle(self, query: str, context: dict = None) -> float:
        q = query.lower().replace("_", " ")
        app_name_clean = self._app_name.lower().replace("_", " ")
        if app_name_clean in q and any(kw in q for kw in ["lance", "ouvre", "démarre", "exécute", "start", "affiche", "app"]):
            return 0.95
        return 0.0

    def execute(self, query: str, context: dict = None) -> dict:
        try:
            subprocess.Popen([self._exe_path, self._app_path], shell=False)
            return {"success": True, "output": f"✅ L'application '{self._app_name}' a été lancée."}
        except Exception as e:
            return {"success": False, "error": f"Erreur lors du lancement : {str(e)}"}

# ==========================================
# GESTIONNAIRE D'OUTILS (Singleton)
# ==========================================
class ToolManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolManager, cls).__new__(cls)
            cls._instance.tools: Dict[str, BaseTool] = {}
            cls._instance._load_tools()
            cls._instance._load_generated_apps()
        return cls._instance

    def _load_tools(self):
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        print("\n🔍 [ToolManager] Scan du dossier des outils...")

        for filename in os.listdir(tools_dir):
            if filename.endswith('.py') and filename != '__init__.py' and not filename.startswith('.'):
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f'outils.{module_name}')
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, BaseTool) and obj is not BaseTool:
                            if obj.__module__ == module.__name__:
                                try:
                                    tool_instance = obj()
                                    self.tools[tool_instance.name] = tool_instance
                                    print(f"  ✅ Outil chargé : {tool_instance.name}")
                                except Exception as inst_err:
                                    print(f"  ⚠️ Impossible d'instancier {name} : {inst_err}")
                except Exception as e:
                    print(f"  ❌ Erreur chargement {module_name}.py : {e}")

    def _load_generated_apps(self):
        apps_dir = r"D:\kibali-IA\generated_apps"
        if not os.path.exists(apps_dir):
            try: os.makedirs(apps_dir, exist_ok=True)
            except: return

        pyw_exe = r"D:\kibali-IA\env_portable\pythonw.exe"
        if not os.path.exists(pyw_exe):
            pyw_exe = r"D:\CPT_Analysis_Studio_PORTABLE\environment\pythonw.exe"
        if not os.path.exists(pyw_exe):
             pyw_exe = "pythonw" 

        app_files = glob.glob(os.path.join(apps_dir, "*.py"))
        if app_files:
            print(f"\n📱 [ToolManager] Scan des applications générées ({len(app_files)})...")
            
        for app_path in app_files:
            app_name = os.path.basename(app_path).replace('.py', '')
            desc = f"Lance l'application '{app_name}'."
            try:
                with open(app_path, 'r', encoding='utf-8') as f:
                    for _ in range(10):
                        line = f.readline()
                        if not line: break
                        if "# APP_DESCRIPTION:" in line:
                            desc = line.split(":", 1)[1].strip()
                            break
            except: pass

            tool_instance = GeneratedAppTool(app_name, app_path, pyw_exe, desc)
            self.tools[tool_instance.name] = tool_instance
            print(f"  ✅ App dynamique chargée : {tool_instance.name}")

    def get_relevant_tools(self, query: str, context: Dict[str, Any] = None, max_tools: int = 3) -> List[BaseTool]:
        tool_scores = []
        for tool in self.tools.values():
            try:
                score = tool.can_handle(query, context)
                if score > 0.0:
                    tool_scores.append((tool, score))
            except Exception:
                pass
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, score in tool_scores[:max_tools]]

    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                'description': tool.description,
                'capabilities': tool.capabilities,
                'parameters': tool.parameters
            }
            for name, tool in self.tools.items()
        }

# ==========================================
# INSTANCIATION UNIQUE
# ==========================================
tool_manager = ToolManager()

# ==========================================
# EXPORTS ISOLÉS
# ==========================================
try: from .python_executor_tool import PythonExecutorTool
except ImportError: PythonExecutorTool = None

try: from .pdf_generator_tool import generate_massive_pdf, MassivePDFGenerator
except ImportError: generate_massive_pdf = None; MassivePDFGenerator = None

try:
    from .excel_ai_organizer import ExcelAIOrganizer, excel_organizer_tool
    EXCEL_AVAILABLE = True
except ImportError:
    ExcelAIOrganizer = None
    excel_organizer_tool = None
    EXCEL_AVAILABLE = False

try: from .web_search_tool import WebSearchTool
except ImportError: WebSearchTool = None

try: from .server_tools_bridge import ServerToolsBridge
except ImportError: ServerToolsBridge = None

try: from .ultra_premium_rag import UltraPremiumRAG
except ImportError: UltraPremiumRAG = None

try: from .ultra_aspirator import UltraAspirator
except ImportError: UltraAspirator = None

try: from .ultra_system_cleaner import UltraSystemCleaner
except ImportError: UltraSystemCleaner = None

try: from .ultra_folder_rag import UltraFolderRAG
except ImportError: UltraFolderRAG = None

try: from .ultra_local_auditor import UltraLocalAuditor
except ImportError: UltraLocalAuditor = None

try: from .ultra_report_master import UltraReportMaster
except ImportError: UltraReportMaster = None

try: from .ultra_lib_installer import UltraLibInstaller
except ImportError: UltraLibInstaller = None

try: from .server_memory import ServerMemory
except ImportError: ServerMemory = None

try: from .local_rag_coder import LocalRagCoder
except ImportError:
    try: from .local_rag_coder import LocalRAGCoder as LocalRagCoder 
    except ImportError: LocalRagCoder = None

__all__ = [
    'BaseTool', 'ToolManager', 'tool_manager', 'GeneratedAppTool',
    'run_in_portable',
    'generate_massive_pdf', 'MassivePDFGenerator', 'ExcelAIOrganizer',
    'excel_organizer_tool', 'EXCEL_AVAILABLE', 'WebSearchTool',
    'ServerToolsBridge', 'UltraPremiumRAG', 'UltraAspirator',
    'UltraSystemCleaner', 'UltraFolderRAG', 'UltraLocalAuditor',
    'UltraReportMaster', 'UltraLibInstaller', 'ServerMemory', 'LocalRagCoder'
]