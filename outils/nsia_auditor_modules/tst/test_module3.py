# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.extractor import NSIAExtractor
from nsia_auditor_modules.hybrid_search import NSIAHybridSearch

def main():
    print("=== TEST MODULE 3 : HYBRID SEARCH ===")
    
    base_dir = r"D:\kibali-IA\outputs\datasets\nsia"
    extractor = NSIAExtractor()
    
    data = {}
    for f in ["bank_accounts.csv", "customer_profiles.csv"]:
        res = extractor.extract(os.path.join(base_dir, f))
        if res["success"]:
            first_sheet = list(res["data"].keys())[0]
            data[f.replace(".csv", "")] = res["data"][first_sheet]
    
    # Mode sans RAG (rapide, utilise le fallback)
    engine = NSIAHybridSearch(rag_instance=None)
    engine.load_data(data)
    
    queries = [
        "Trouve les comptes avec un solde negatif",
        "Comptes avec un pret superieur a 200000",
        "Tous les comptes avec solde positif",
        "Comptes a risque",
        "Bons clients",
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        result = engine.search(q, limit=3)
        print(f"Requete : {q}")
        print(f"Moteur  : {result['engine']}")
        print(f"Resultats : {result['count']} trouve(s)")
        for r in result['results'][:3]:
            aid = str(r.get('account_id', 'N/A'))[:12]
            solde = r.get('balance', 'N/A')
            pret = r.get('loan_amount', 0)
            print(f"  -> Compte {aid}... | Solde: {solde} | Pret: {pret}")

if __name__ == "__main__":
    main()