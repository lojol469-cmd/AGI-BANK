# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from datetime import datetime

class NSIABudgetEngine:
    """
    Module 17 : Prévisionnel & Budget.
    Permet de créer des budgets, de projeter les revenus (Forecasting)
    et d'analyser les écarts (Contrôle de Gestion).
    """
    
    def __init__(self):
        pass

    def forecast_revenue(self, historical_data: List[float], growth_rate: float, periods: int = 3) -> Dict[str, Any]:
        if not historical_data:
            return {"success": False, "error": "Aucune donnée historique fournie."}
            
        average_hist = sum(historical_data) / len(historical_data)
        projections = []
        
        current_base = average_hist
        for i in range(periods):
            current_base = current_base * (1 + growth_rate)
            projections.append(round(current_base, 2))
            
        return {
            "success": True,
            "moyenne_historique": round(average_hist, 2),
            "taux_croissance": growth_rate,
            "projections": projections
        }

    def analyze_variance(self, actuals: Dict[str, float], budgets: Dict[str, float], is_expense: bool = True) -> Dict[str, Any]:
        """
        Compare les réalisations avec les budgets.
        is_expense=True : on analyse des charges (dépenser moins = favorable).
        is_expense=False : on analyse des revenus (gagner plus = favorable).
        """
        results = []
        total_actual = 0
        total_budget = 0
        
        for category in budgets.keys():
            b = budgets.get(category, 0)
            a = actuals.get(category, 0)
            
            total_budget += b
            total_actual += a
            
            variance = a - b
            variance_pct = (variance / b * 100) if b != 0 else 0
            
            # Logique corrigée selon le type (Charge ou Revenu)
            if variance == 0:
                status = "NEUTRE"
            elif is_expense:
                status = "FAVORABLE" if variance < 0 else "DEFAVORABLE"
            else:
                status = "FAVORABLE" if variance > 0 else "DEFAVORABLE"
            
            results.append({
                "category": category,
                "budget": b,
                "actual": a,
                "variance": variance,
                "variance_pct": variance_pct,
                "status": status
            })
            
        return {
            "success": True,
            "total_budget": total_budget,
            "total_actual": total_actual,
            "total_variance": total_actual - total_budget,
            "details": results
        }

    def generate_report(self, forecast_data: Dict[str, Any], variance_data: Dict[str, Any], is_expense: bool = True) -> str:
        lines = []
        lines.append("# RAPPORT PRÉVISIONNEL & BUDGET")
        lines.append(f"**Généré le** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        # --- SECTION 1 : FORECASTING ---
        if forecast_data.get("success"):
            lines.append("## 1. Projection des Revenus")
            lines.append(f"- Moyenne historique : {forecast_data['moyenne_historique']:,.2f} FCFA")
            lines.append(f"- Taux de croissance appliqué : {forecast_data['taux_croissance']*100:.1f}%")
            lines.append("")
            lines.append("| Période | Revenu Projeté |")
            lines.append("|---|---|")
            for i, val in enumerate(forecast_data['projections'], 1):
                lines.append(f"| Mois +{i} | {val:,.2f} FCFA |")
            lines.append("")
            
        # --- SECTION 2 : ANALYSE DES ÉCARTS ---
        if variance_data.get("success"):
            lines.append("## 2. Analyse des Écarts Budgétaires")
            lines.append(f"- Budget Total : {variance_data['total_budget']:,.2f} FCFA")
            lines.append(f"- Réalisé Total : {variance_data['total_actual']:,.2f} FCFA")
            
            var_tot = variance_data['total_variance']
            var_tot_pct = (var_tot / variance_data['total_budget'] * 100) if variance_data['total_budget'] else 0
            
            # Logique globale
            if var_tot == 0:
                emoji = "⚪"
            elif is_expense:
                emoji = "🟢" if var_tot < 0 else "🔴"
            else:
                emoji = "🟢" if var_tot > 0 else "🔴"
                
            lines.append(f"- **Écart Global** : {emoji} {var_tot:+,.2f} FCFA ({var_tot_pct:+.1f}%)")
            lines.append("")
            
            lines.append("### Détail par Catégorie")
            lines.append("| Catégorie | Budget | Réalisé | Écart | % | Statut |")
            lines.append("|---|---|---|---|---|---|")
            for d in variance_data['details']:
                emoji_det = "🟢" if d['status'] == "FAVORABLE" else "🔴" if d['status'] == "DEFAVORABLE" else "⚪"
                lines.append(
                    f"| {d['category']} | {d['budget']:,.0f} | {d['actual']:,.0f} | "
                    f"{d['variance']:+,.0f} | {d['variance_pct']:+.1f}% | {emoji_det} {d['status']} |"
                )
            lines.append("")
            
        lines.append("---")
        lines.append("**Signé : KIBALI AGI EXPERT - Contrôle de Gestion**")
        
        return "\n".join(lines)