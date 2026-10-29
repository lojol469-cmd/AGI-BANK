# -*- coding: utf-8 -*-
import os
import sys
import time

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.extractor import NSIAExtractor
from nsia_auditor_modules.graph_rag import NSIAGraphRAG

def main():
    print("=== TEST MODULE 2 : GRAPH RAG & FAISS ===")
    base_dir = r"D:\kibali-IA\outputs\datasets\nsia"
    
    extractor = NSIAExtractor()
    data = {}
    
    for f in ["customer_profiles.csv", "bank_accounts.csv", "account_transactions.csv"]:
        res = extractor.extract(os.path.join(base_dir, f))
        if res["success"]:
            first_sheet = list(res["data"].keys())[0]
            data[f.replace(".csv", "")] = res["data"][first_sheet]
            
    rag = NSIAGraphRAG()
    start = time.time()
    rag.build(data)
    print(f"Temps de construction : {time.time() - start:.2f}s")
    
    print("\n--- TESTS DE RECHERCHE ---")
    queries = [
        "Trouve les comptes avec un solde negatif",
        "Comptes avec un pret important",
        "Comptes a risque"
    ]
    
    for q in queries:
        results = rag.search(q, k=2)
        print(f"\nRequete : '{q}'")
        for r in results:
            print(f"  -> Compte {r['account_id']} | Solde: {r['balance']} | Statut: {r['status']}")

if __name__ == "__main__":
    main()