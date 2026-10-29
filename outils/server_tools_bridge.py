import requests
import json

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class ServerToolsBridge(BaseTool):
    @property
    def name(self) -> str:
        return "kibali_server_tools"

    @property
    def description(self) -> str:
        return "Passerelle intelligente vers les 384 outils avancés de Kibali (entreprise, sécurité, excel, rag)."

    @property
    def capabilities(self) -> list:
        return ["enterprise", "security", "excel", "rag", "server", "native", "advanced"]

    def can_handle(self, query: str, context: dict = None) -> float:
        keywords = ["entreprise", "sécurité", "comptabilité", "fiscal", "excel", "audit", "serveur", "outils", "système", "rag"]
        if any(kw in query.lower() for kw in keywords):
            return 0.95
        return 0.5

    def execute(self, query: str, context: dict = None) -> dict:
        base_url = "http://127.0.0.1:8765"
        
        try:
            # 1. Chercher le meilleur outil
            search_res = requests.post(f"{base_url}/search_tools", json={"query": query, "top_k": 1}, timeout=10)
            
            if search_res.status_code != 200:
                return {"success": False, "error": f"Recherche échouée: {search_res.text}"}
            
            search_data = search_res.json()
            tool_name = None
            
            # Gérer le format exact {"tools": ["nom_de_l_outil"]}
            if "tools" in search_data and isinstance(search_data["tools"], list) and len(search_data["tools"]) > 0:
                tool_name = search_data["tools"][0]
            
            if not tool_name:
                return {"success": False, "error": "Aucun outil correspondant trouvé sur le serveur."}

            # 2. Récupérer le schéma exact de l'outil pour ses paramètres
            schema_res = requests.get(f"{base_url}/tool_schema/{tool_name}", timeout=5)
            args = {}
            
            if schema_res.status_code == 200:
                schema = schema_res.json()
                # Extraire les propriétés attendues par l'outil
                props = schema.get("parameters", {}).get("properties", {}) or schema.get("properties", {})
                for prop_name in props.keys():
                    args[prop_name] = query
            else:
                # Fallback standard au cas où
                args = {"query": query, "text": query, "input": query, "target": query}

            # 3. Exécuter l'outil avec les arguments parfaits
            exec_payload = {
                "tool_name": tool_name,
                "args": args
            }
            
            exec_res = requests.post(f"{base_url}/execute_tool", json=exec_payload, timeout=120)
            
            if exec_res.status_code == 200:
                # On renvoie le résultat brut du serveur à l'IA
                return {"success": True, "output": f"✅ L'outil '{tool_name}' a répondu :\n{json.dumps(exec_res.json(), indent=2, ensure_ascii=False)}"}
            else:
                return {"success": False, "error": f"Erreur d'exécution ({exec_res.status_code}): {exec_res.text}"}

        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Le serveur Kibali (port 8765) est éteint."}
        except Exception as e:
            return {"success": False, "error": f"Erreur du pont : {str(e)}"}