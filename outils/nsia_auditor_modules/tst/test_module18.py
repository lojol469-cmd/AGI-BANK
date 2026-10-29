# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.fixed_assets_engine import NSIAFixedAssetsEngine

def main():
    print("="*70)
    print("=== TEST MODULE 18 : IMMOBILISATIONS & AMORTISSEMENTS ===")
    print("="*70)
    
    engine = NSIAFixedAssetsEngine()
    
    # Achat d'un Serveur Informatique
    # Valeur : 5 000 000 FCFA, Valeur résiduelle : 500 000 FCFA, Durée : 5 ans.
    
    # 1. Test Linéaire
    print("--- MÉTHODE LINÉAIRE ---")
    linear = engine.calculate_linear("Serveur Informatique", cost=5000000, salvage_value=500000, useful_life=5)
    print(engine.generate_report(linear))
    
    print("\n" + "="*70)
    
    # 2. Test Dégressif (Coefficient 2.0)
    print("--- MÉTHODE DÉGRESSIVE (Accélérée) ---")
    declining = engine.calculate_declining("Serveur Informatique", cost=5000000, salvage_value=500000, useful_life=5, multiplier=2.0)
    print(engine.generate_report(declining))

if __name__ == "__main__":
    main()