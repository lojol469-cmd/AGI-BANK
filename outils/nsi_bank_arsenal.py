# -*- coding: utf-8 -*-
import os
import sys
from typing import Dict, Any

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from . import BaseTool
from .nsia_auditor_modules.extractor import NSIAExtractor

# Import de tous les modules
from .nsia_auditor_modules.accounting_engine import NSIAAccountingEngine
from .nsia_auditor_modules.financial_statements import NSIAFinancialStatements
from .nsia_auditor_modules.tax_engine import NSIATaxEngine
from .nsia_auditor_modules.payroll_engine import NSIAPayrollEngine
from .nsia_auditor_modules.billing_engine import NSIABillingEngine
from .nsia_auditor_modules.inventory_engine import NSIAInventoryEngine
from .nsia_auditor_modules.customs_engine import NSIACustomsEngine
from .nsia_auditor_modules.fraud_detector import NSIAFraudDetector
from .nsia_auditor_modules.budget_engine import NSIABudgetEngine
from .nsia_auditor_modules.fixed_assets_engine import NSIAFixedAssetsEngine
from .nsia_auditor_modules.crm_engine import NSIACRMEngine
from .nsia_auditor_modules.compliance_engine import NSIAComplianceEngine

class NSIBankArsenalTool(BaseTool):
    """
    Outil Master : Accès à l'arsenal complet des 20 modules de gestion bancaire KIBALI.
    """
    
    @property
    def name(self) -> str:
        return "nsi_bank_arsenal"
    
    @property
    def description(self) -> str:
        return (
            "Arsenal bancaire complet (20 modules). "
            "Permet d'executer : 'compta', 'bilan', 'fiscalite', 'paie', 'facture', 'stock', "
            "'douane', 'fraude', 'budget', 'immobilisation', 'crm', 'conformite'. "
            "Utilise l'action 'execute_module' avec le nom du module et les parametres."
        )
    
    @property
    def capabilities(self) -> list:
        return [
            "comptabilite", "syscohada", "fiscalite", "paie", "facturation", 
            "stocks", "douane", "sydonia", "fraude", "budget", "crm", "audit"
        ]
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["execute_module"]},
                "module_name": {
                    "type": "string", 
                    "enum": ["compta", "bilan", "fiscalite", "paie", "facture", "stock", "douane", "fraude", "budget", "immobilisation", "crm", "conformite"]
                },
                "params": {"type": "object", "description": "Parametres specifiques au module (ex: salaire, items, etc.)"}
            },
            "required": ["action", "module_name"]
        }
    
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        q = query.lower()
        keywords = ["compta", "bilan", "tva", "is", "paie", "salaire", "facture", "stock", "douane", "fraude", "budget", "amortissement", "crm", "conformite"]
        if any(kw in q for kw in keywords):
            return 0.95
        return 0.0
    
    def execute(self, action: str, module_name: str = "", params: Dict = None, **kwargs) -> Dict[str, Any]:
        if action != "execute_module":
            return {"success": False, "output": "Action inconnue."}
            
        params = params or {}
        base_dir = r"D:\kibali-IA\outputs\datasets\nsia"
        
        try:
            # --- MODULE 8/9 : COMPTA / BILAN ---
            if module_name in ["compta", "bilan"]:
                extractor = NSIAExtractor()
                res = extractor.extract(os.path.join(base_dir, "bank_accounts.csv"))
                accounts = list(res["data"].values())[0]
                
                if module_name == "compta":
                    engine = NSIAAccountingEngine()
                    engine.generate_journal(limit=1000)
                    return {"success": True, "output": engine.generate_full_report()["output"]}
                else:
                    engine = NSIAFinancialStatements(accounts)
                    return engine.generate_report()
            
            # --- MODULE 11 : FISCALITE ---
            elif module_name == "fiscalite":
                engine = NSIATaxEngine()
                ca = params.get("ca", 10000000)
                achats = params.get("achats", 5000000)
                resultat = params.get("resultat", 2000000)
                return engine.generate_report(ca, achats, resultat)
                
            # --- MODULE 12 : PAIE ---
            elif module_name == "paie":
                engine = NSIAPayrollEngine()
                salaire = params.get("salaire", 500000)
                return engine.generate_report(salaire)
                
            # --- MODULE 13 : FACTURE ---
            elif module_name == "facture":
                engine = NSIABillingEngine()
                items = params.get("items", [{"description": "Service", "quantity": 1, "unit_price": 100000}])
                inv = engine.create_invoice(params.get("client", "Client X"), items)
                return {"success": True, "output": engine.generate_markdown(inv)}
                
            # --- MODULE 14 : STOCKS ---
            elif module_name == "stock":
                engine = NSIAInventoryEngine()
                engine.add_product("P1", "Produit Test", 10, 1000)
                return engine.generate_report()
                
            # --- MODULE 15 : DOUANE ---
            elif module_name == "douane":
                engine = NSIACustomsEngine()
                data = engine.calculate_landed_cost(
                    params.get("fob", 1000000), params.get("freight", 100000), 
                    params.get("insurance", 20000), params.get("category_rate", 0.2)
                )
                return {"success": True, "output": engine.generate_report(data)}
                
            # --- MODULE 16 : FRAUDE ---
            elif module_name == "fraude":
                extractor = NSIAExtractor()
                res = extractor.extract(os.path.join(base_dir, "account_transactions.csv"))
                tx = list(res["data"].values())[0][:50000]
                engine = NSIAFraudDetector()
                return engine.analyze_transactions(tx, amount_threshold=500000)
                
            # --- MODULE 17 : BUDGET ---
            elif module_name == "budget":
                engine = NSIABudgetEngine()
                # CORRECTION ICI : Plus de "10M"
                forecast = engine.forecast_revenue([10000000, 12000000], 0.05)
                variance = engine.analyze_variance({"Marketing": 2500000}, {"Marketing": 2000000})
                return {"success": True, "output": engine.generate_report(forecast, variance)}
                
            # --- MODULE 18 : IMMOBILISATION ---
            elif module_name == "immobilisation":
                engine = NSIAFixedAssetsEngine()
                data = engine.calculate_linear("Machine", params.get("cost", 5000000), 500000, 5)
                return {"success": True, "output": engine.generate_report(data)}
                
            # --- MODULE 19 : CRM ---
            elif module_name == "crm":
                extractor = NSIAExtractor()
                res_c = extractor.extract(os.path.join(base_dir, "customer_profiles.csv"))
                res_a = extractor.extract(os.path.join(base_dir, "bank_accounts.csv"))
                engine = NSIACRMEngine()
                data = engine.segment_clients(list(res_a["data"].values())[0], list(res_c["data"].values())[0])
                return {"success": True, "output": engine.generate_report(data)}
                
            # --- MODULE 20 : CONFORMITE ---
            elif module_name == "conformite":
                extractor = NSIAExtractor()
                res_c = extractor.extract(os.path.join(base_dir, "customer_profiles.csv"))
                res_a = extractor.extract(os.path.join(base_dir, "bank_accounts.csv"))
                engine = NSIAComplianceEngine()
                data = engine.run_audit(list(res_a["data"].values())[0], list(res_c["data"].values())[0])
                return {"success": True, "output": engine.generate_report(data)}
                
            else:
                return {"success": False, "output": f"Module '{module_name}' non reconnu."}
                
        except Exception as e:
            import traceback
            return {"success": False, "output": f"Erreur : {str(e)}\n{traceback.format_exc()}"}