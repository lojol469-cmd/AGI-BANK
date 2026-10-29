# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.tax_engine import NSIATaxEngine

def main():
    print("="*70)
    print("=== TEST MODULE 11 : MOTEUR FISCAL (TVA, IS) ===")
    print("="*70)
    
    engine = NSIATaxEngine()
    
    # On reprend les vrais chiffres du Module 9 !
    CA = 11564054.52       # Produits
    ACHATS = 6642574.18    # Charges
    RESULTAT = 4921480.34  # Résultat Net
    
    report = engine.generate_report(ca=CA, achats=ACHATS, resultat=RESULTAT)
    print(report["output"])
    
    print("\n" + "="*70)
    print("=== TEST CALCUL IRPP (Simulation Salaire 800 000 FCFA) ===")
    print("="*70)
    
    irpp = engine.calculate_irpp(800000)
    print(f"Salaire Brut : {irpp['salaire_brut']:,.2f}")
    for d in irpp['details']:
        print(f"  Tranche {d['tranche']} à {d['taux']} => Impôt: {d['impot']:,.2f}")
    print(f"Total IRPP : {irpp['irpp_total']:,.2f}")
    print(f"Salaire Net : {irpp['salaire_net']:,.2f}")

if __name__ == "__main__":
    main()