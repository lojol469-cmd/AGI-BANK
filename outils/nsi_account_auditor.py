# -*- coding: utf-8 -*-
import os
import sys
import glob
from typing import Dict, Any, List

# Securite imports relatifs (charte KIBALI)
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from . import BaseTool
from .nsia_auditor_modules.extractor import NSIAExtractor
from .nsia_auditor_modules.hybrid_search import NSIAHybridSearch
from .nsia_auditor_modules.report_generator import NSIAReportGenerator


class NSIAAccountAuditorTool(BaseTool):
    """
    Outil KIBALI : Audit des comptes NSIA (pipeline 4 modules).
    Entree : fichiers Excel/CSV/JSON/PDF + requete naturelle.
    Sortie : alertes + rapports PDF/Excel/Markdown signes KIBALI AGI EXPERT.
    """
    
    @property
    def name(self) -> str:
        return "audit_comptes_nsia"
    
    @property
    def description(self) -> str:
        return (
            "Audit des comptes bancaires NSIA. "
            "Prend en entree un fichier Excel/CSV/JSON/PDF et une requete naturelle. "
            "Detecte les decouverts, prets excessifs, taux d'usure. "
            "Croise avec les textes BCEAO. "
            "Genere rapports PDF/Excel signes KIBALI AGI EXPERT. "
            "Actions : 'audit' (file_path, query)."
        )
    
    @property
    def capabilities(self) -> List[str]:
        return [
            "audit_bancaire", "decouvert", "analyse_comptes", 
            "nsia", "conformite_bceao", "rapport_pdf", "rapport_excel"
        ]
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["audit"],
                    "description": "L'action a effectuer (audit)."
                },
                "file_path": {
                    "type": "string",
                    "description": "Chemin absolu vers le fichier a auditer (Excel/CSV/JSON/PDF)."
                },
                "query": {
                    "type": "string",
                    "description": "Requete naturelle (ex: 'comptes en decouvert', 'prets > 200000', 'comptes a risque')."
                }
            },
            "required": ["action", "file_path", "query"]
        }
    
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        q = query.lower()
        keywords = [
            "audit", "compte", "nsia", "decouvert", "banque", 
            "conformite", "bceao", "rapport", "analyse", "pret"
        ]
        if any(kw in q for kw in keywords):
            return 0.95
        return 0.0
    
    def _find_supporting_files(self, primary_file: str) -> Dict[str, Any]:
        """
        Cherche les fichiers complementaires (customers, accounts, transactions)
        dans le meme dossier que le fichier principal.
        """
        base_dir = os.path.dirname(primary_file)
        extracted = {}
        extractor = NSIAExtractor()
        
        # Mapping des noms de fichiers attendus
        patterns = {
            "customer_profiles": ["customer_profiles*.csv", "customers*.csv", "clients*.csv", "*.xlsx"],
            "bank_accounts": ["bank_accounts*.csv", "accounts*.csv", "comptes*.csv"],
            "account_transactions": ["account_transactions*.csv", "transactions*.csv"]
        }
        
        for key, pats in patterns.items():
            for pat in pats:
                matches = glob.glob(os.path.join(base_dir, pat))
                if matches:
                    res = extractor.extract(matches[0])
                    if res["success"]:
                        first_sheet = list(res["data"].keys())[0]
                        extracted[key] = res["data"][first_sheet]
                        print(f"[ORCHESTRATEUR] {key} charge : {matches[0]} ({res['total_rows']} lignes)")
                        break
        
        # Si rien n'est trouve, on charge le fichier principal seul
        if not extracted:
            res = extractor.extract(primary_file)
            if res["success"]:
                first_sheet = list(res["data"].keys())[0]
                extracted["bank_accounts"] = res["data"][first_sheet]
        
        return extracted
    
    def execute(self, action: str, file_path: str = "", query: str = "", **kwargs) -> Dict[str, Any]:
        try:
            if action != "audit":
                return {"success": False, "output": f"Action inconnue : {action}"}
            
            if not file_path or not os.path.exists(file_path):
                return {"success": False, "output": f"Fichier introuvable : {file_path}"}
            
            if not query:
                query = "Tous les comptes"
            
            # ETAPE 1 : Extraction (Module 1)
            print(f"\n[ORCHESTRATEUR] Etape 1/4 : Extraction des donnees...")
            data = self._find_supporting_files(file_path)
            
            if not data:
                return {"success": False, "output": "Aucune donnee exploitable dans le fichier."}
            
            # ETAPE 2 : Recherche hybride (Module 3)
            print(f"[ORCHESTRATEUR] Etape 2/4 : Recherche hybride...")
            engine = NSIAHybridSearch()
            engine.load_data(data)
            results = engine.search(query, limit=1000)
            print(f"[ORCHESTRATEUR] {results['count']} comptes trouves via {results['engine']}")
            
            # ETAPE 3 : Generation du rapport (Module 4)
            print(f"[ORCHESTRATEUR] Etape 3/4 : Generation du rapport...")
            generator = NSIAReportGenerator()
            report = generator.generate_full_report(query, results)
            
            # ETAPE 4 : Synthese pour l'IA
            print(f"[ORCHESTRATEUR] Etape 4/4 : Synthese.")
            
            synthesis = (
                f"# AUDIT NSIA TERMINE\n\n"
                f"**Requete** : {query}\n"
                f"**Moteur** : {results['engine']}\n"
                f"**Comptes analyses** : {results['count']}\n"
                f"**Alertes detectees** : {report['alertes_count']}\n\n"
                f"## Rapports generes\n"
                f"- PDF : `{report['pdf']}`\n"
                f"- Excel : `{report['excel']}`\n"
                f"- Markdown : `{report['markdown']}`\n\n"
                f"## Top 5 alertes\n"
            )
            
            for a in report['alerts'][:5]:
                synthesis += f"- [{a['niveau']}] {a['type']} - Compte {str(a['account_id'])[:8]}... - {a['legal_ref']}\n"
            
            synthesis += "\n**Signe : KIBALI AGI EXPERT**"
            
            return {
                "success": True,
                "output": synthesis,
                "data": {
                    "alerts_count": report['alertes_count'],
                    "accounts_found": results['count'],
                    "reports": {
                        "pdf": report['pdf'],
                        "excel": report['excel'],
                        "markdown": report['markdown']
                    }
                }
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "output": f"Erreur orchestrateur : {str(e)}\n{traceback.format_exc()}"
            }