# -*- coding: utf-8 -*-
import requests
import json

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class RagConnectorTool(BaseTool):
    @property
    def name(self) -> str:
        return "rag_connector_tool"

    @property
    def description(self) -> str:
        return "Connecte l'IA au serveur RAG local (port 8765) pour lister et exécuter les 384 outils métiers (Fiscalité, Excel, Code, etc.). Utilise cet outil pour faire des calculs complexes ou récupérer des articles de lois."

    @property
    def capabilities(self) -> list:
        return ["rag", "outils", "fiscalité", "calculs", "lois", "excel", "métier", "serveur"]

    def can_handle(self, query: str, context: dict = None) -> float:
        query_lower = query.lower()
        score = 0.0
        keywords = ["rag", "outil", "fiscalité", "calcul", "loi", "excel", "métier"]
        for kw in keywords:
            if kw in query_lower:
                score += 0.3
        return min(score, 1.0)

    def execute(self, query: str, context: dict = None) -> dict:
        base_url = "http://127.0.0.1:8765"
        
        # Si l'IA demande de lister les outils
        if "LIST_TOOLS" in query:
            try:
                r = requests.get(f"{base_url}/tool_schemas", timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    # On extrait juste les noms et descriptions pour ne pas saturer le prompt
                    tools_summary = []
                    for t in data:
                        tools_summary.append(f"- {t.get('name')} : {t.get('description', 'Pas de description')[:100]}...")
                    
                    output = f"✅ {len(tools_summary)} outils trouvés sur le serveur RAG.\n\n"
                    output += "Voici une sélection :\n" + "\n".join(tools_summary[:50])
                    return {"success": True, "output": output}
                else:
                    return {"success": False, "error": f"Erreur serveur RAG : {r.status_code}"}
            except Exception as e:
                return {"success": False, "error": f"Impossible de contacter le serveur RAG : {e}"}
                
        # Si l'IA demande d'exécuter un outil spécifique (Format : EXECUTE|nom_outil|{json_params})
        elif "EXECUTE|" in query:
            try:
                parts = query.split("|", 2)
                tool_name = parts[1].strip()
                params_str = parts[2].strip() if len(parts) > 2 else "{}"
                
                try:
                    params = json.loads(params_str)
                except:
                    params = {"input": params_str}
                
                payload = {"name": tool_name, "args": params, "kwargs": params}
                
                # On tente les endpoints standards du serveur RAG
                for endpoint in ["/call_tool", "/execute", "/tools/execute"]:
                    try:
                        r = requests.post(f"{base_url}{endpoint}", json=payload, timeout=30)
                        if r.status_code == 200:
                            return {"success": True, "output": str(r.json())}
                    except:
                        continue
                        
                return {"success": False, "error": f"Impossible d'exécuter l'outil {tool_name}. Vérifie l'endpoint du serveur."}
                
            except Exception as e:
                return {"success": False, "error": f"Erreur de format d'exécution : {e}"}
                
        return {"success": False, "error": "Action non reconnue. Utilise 'LIST_TOOLS' ou 'EXECUTE|nom_outil|{params}'."}