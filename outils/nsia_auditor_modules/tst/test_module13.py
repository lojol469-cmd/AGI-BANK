# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.billing_engine import NSIABillingEngine

def main():
    print("="*70)
    print("=== TEST MODULE 13 : FACTURATION & DEVIS ===")
    print("="*70)
    
    engine = NSIABillingEngine()
    
    # Exemple : Vente de matériel informatique à un client
    items = [
        {"description": "Ordinateur Portable Pro", "quantity": 5, "unit_price": 650000},
        {"description": "Licence Antivirus (1 an)", "quantity": 5, "unit_price": 25000},
        {"description": "Installation & Configuration", "quantity": 1, "unit_price": 150000}
    ]
    
    # Génération d'une facture avec 5% de remise commerciale
    invoice = engine.create_invoice(
        client_name="Société Générale Sénégal",
        items=items,
        discount_pct=5.0,
        is_quote=False
    )
    
    if invoice["success"]:
        print(engine.generate_markdown(invoice))
    else:
        print(f"Erreur : {invoice['error']}")

if __name__ == "__main__":
    main()