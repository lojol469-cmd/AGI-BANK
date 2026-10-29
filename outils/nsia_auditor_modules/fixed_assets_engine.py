# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from datetime import datetime

def safe_float(val) -> float:
    try:
        return float(val)
    except:
        return 0.0

class NSIAFixedAssetsEngine:
    """
    Module 18 : Gestion des Immobilisations & Amortissements (SYSCOHADA).
    Calcule les plans d'amortissement (linéaire et dégressif).
    """
    
    def __init__(self):
        pass

    def calculate_linear(self, asset_name: str, cost: float, salvage_value: float, useful_life: int) -> Dict[str, Any]:
        """
        Amortissement Linéaire : (Coût - Valeur Résiduelle) / Durée de vie
        """
        cost = safe_float(cost)
        salvage = safe_float(salvage_value)
        base_amortissable = cost - salvage
        
        if useful_life <= 0:
            return {"success": False, "error": "La durée de vie doit être > 0."}
            
        annual_depreciation = base_amortissable / useful_life
        
        schedule = []
        net_book_value = cost
        
        for year in range(1, useful_life + 1):
            net_book_value -= annual_depreciation
            schedule.append({
                "year": year,
                "depreciation": round(annual_depreciation, 2),
                "cumulative": round(annual_depreciation * year, 2),
                "net_book_value": round(max(net_book_value, salvage), 2)
            })
            
        return {
            "success": True,
            "asset_name": asset_name,
            "method": "Linéaire",
            "cost": cost,
            "useful_life": useful_life,
            "annual_depreciation": round(annual_depreciation, 2),
            "schedule": schedule
        }

    def calculate_declining(self, asset_name: str, cost: float, salvage_value: float, useful_life: int, multiplier: float = 2.0) -> Dict[str, Any]:
        """
        Amortissement Dégressif (Accéléré).
        Taux = (1 / Durée) * Coefficient (ex: 2.0 pour un coefficient de 2).
        """
        cost = safe_float(cost)
        salvage = safe_float(salvage_value)
        
        if useful_life <= 0:
            return {"success": False, "error": "La durée de vie doit être > 0."}
            
        rate = (1.0 / useful_life) * multiplier
        schedule = []
        net_book_value = cost
        
        for year in range(1, useful_life + 1):
            # L'amortissement dégressif s'applique sur la valeur nette comptable (VNC)
            depreciation = net_book_value * rate
            
            # On ne peut pas descendre en dessous de la valeur résiduelle
            if net_book_value - depreciation < salvage:
                depreciation = net_book_value - salvage
                
            net_book_value -= depreciation
            
            schedule.append({
                "year": year,
                "depreciation": round(depreciation, 2),
                "cumulative": round(cost - net_book_value, 2),
                "net_book_value": round(net_book_value, 2)
            })
            
            if net_book_value <= salvage:
                break
                
        return {
            "success": True,
            "asset_name": asset_name,
            "method": "Dégressif",
            "cost": cost,
            "useful_life": useful_life,
            "rate": rate,
            "schedule": schedule
        }

    def generate_report(self, asset_data: Dict[str, Any]) -> str:
        if not asset_data.get("success"):
            return "Erreur de calcul."
            
        lines = []
        lines.append(f"# PLAN D'AMORTISSEMENT : {asset_data['asset_name']}")
        lines.append(f"**Généré le** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Méthode** : {asset_data['method']}")
        lines.append(f"**Valeur d'origine** : {asset_data['cost']:,.2f} FCFA")
        lines.append(f"**Durée de vie** : {asset_data['useful_life']} ans")
        if 'annual_depreciation' in asset_data:
            lines.append(f"**Annuité constante** : {asset_data['annual_depreciation']:,.2f} FCFA")
        if 'rate' in asset_data:
            lines.append(f"**Taux d'amortissement** : {asset_data['rate']*100:.2f}%")
        lines.append("")
        
        lines.append("## Tableau d'Amortissement")
        lines.append("| Année | Dotation | Cumul | Valeur Nette Comptable |")
        lines.append("|---|---|---|---|")
        for row in asset_data['schedule']:
            lines.append(f"| {row['year']} | {row['depreciation']:,.2f} | {row['cumulative']:,.2f} | {row['net_book_value']:,.2f} |")
        lines.append("")
        
        lines.append("---")
        lines.append("**Écriture comptable annuelle** :")
        lines.append("- Débit : 681 (Dotations aux amortissements)")
        lines.append("- Crédit : 281 (Amortissements des immobilisations)")
        lines.append("")
        lines.append("**Signé : KIBALI AGI EXPERT - Comptabilité des Immobilisations**")
        
        return "\n".join(lines)