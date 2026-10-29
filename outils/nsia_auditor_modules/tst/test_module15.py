# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.customs_engine import NSIACustomsEngine

def main():
    print("="*70)
    print("=== TEST MODULE 15 : DOUANES & SYDONIA ===")
    print("="*70)
    
    engine = NSIACustomsEngine()
    
    # Exemple : Importation de matériel électronique de Chine au Sénégal
    # FOB (Valeur marchandise) : 10 000 000 FCFA
    # Fret (Transport) : 1 500 000 FCFA
    # Assurance : 200 000 FCFA
    # Catégorie douanière : 20% (Taux standard pour électronique)
    
    data = engine.calculate_landed_cost(
        fob=10000000, 
        freight=1500000, 
        insurance=200000, 
        category_rate=0.20
    )
    
    report = engine.generate_report(data)
    print(report)

if __name__ == "__main__":
    main()