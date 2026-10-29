# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.inventory_engine import NSIAInventoryEngine

def main():
    print("="*70)
    print("=== TEST MODULE 14 : STOCKS & INVENTAIRE (CUMP) ===")
    print("="*70)
    
    engine = NSIAInventoryEngine()
    
    # 1. Ajout de produits
    engine.add_product("ORD-001", "Ordinateur Portable Pro", initial_qty=50, unit_cost=500000)
    engine.add_product("SOU-002", "Souris Ergonomique", initial_qty=150, unit_cost=15000)
    engine.add_product("ECR-003", "Ecran 27 pouces", initial_qty=5, unit_cost=120000) # Stock bas
    
    # 2. Mouvements de stock
    # Entrée : On achète 10 PC de plus à 520 000 (le prix a augmenté)
    engine.record_movement("ORD-001", "IN", qty=10, unit_cost=520000)
    
    # Sortie : On vend 30 PC
    engine.record_movement("ORD-001", "OUT", qty=30)
    
    # Sortie : On vend 145 souris
    engine.record_movement("SOU-002", "OUT", qty=145)
    
    # 3. Génération du rapport
    report = engine.generate_report()
    print(report["output"])

if __name__ == "__main__":
    main()