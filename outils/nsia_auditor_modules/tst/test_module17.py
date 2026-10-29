# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.budget_engine import NSIABudgetEngine

def main():
    print("="*70)
    print("=== TEST MODULE 17 : PRÉVISIONNEL & BUDGET ===")
    print("="*70)
    
    engine = NSIABudgetEngine()
    
    # 1. Projection : On a fait 10M, 12M et 11M ces 3 derniers mois. On projette avec une croissance de 5%.
    historique = [10000000, 12000000, 11000000]
    forecast = engine.forecast_revenue(historique, growth_rate=0.05, periods=3)
    
    # 2. Analyse des écarts (Budget vs Réalisé)
    budgets = {
        "Salaires": 5000000,
        "Loyer & Charges": 1500000,
        "Marketing": 2000000,
        "Achats IT": 3000000
    }
    realises = {
        "Salaires": 5000000,
        "Loyer & Charges": 1450000, # On a économisé 50k
        "Marketing": 2500000,       # On a dépassé de 500k
        "Achats IT": 2800000        # On a économisé 200k
    }
    variance = engine.analyze_variance(realises, budgets)
    
    # 3. Génération du rapport
    report = engine.generate_report(forecast, variance)
    print(report)

if __name__ == "__main__":
    main()