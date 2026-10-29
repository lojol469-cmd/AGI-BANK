# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.extractor import NSIAExtractor
from nsia_auditor_modules.hybrid_search import NSIAHybridSearch
from nsia_auditor_modules.report_generator import NSIAReportGenerator

def main():
    print("=== TEST MODULE 4 : ALERTES & RAPPORTS ===")
    
    base_dir = r"D:\kibali-IA\outputs\datasets\nsia"
    extractor = NSIAExtractor()
    
    data = {}
    for f in ["bank_accounts.csv", "customer_profiles.csv"]:
        res = extractor.extract(os.path.join(base_dir, f))
        if res["success"]:
            first_sheet = list(res["data"].keys())[0]
            data[f.replace(".csv", "")] = res["data"][first_sheet]
    
    # 1. Recherche (Module 3)
    engine = NSIAHybridSearch()
    engine.load_data(data)
    
    query = "Comptes avec un pret superieur a 200000"
    results = engine.search(query, limit=500)
    print(f"\nRequete : {query}")
    print(f"Moteur : {results['engine']}")
    print(f"Resultats : {results['count']}")
    
    # 2. Generation du rapport (Module 4)
    print("\n--- GENERATION DU RAPPORT ---")
    generator = NSIAReportGenerator()
    report = generator.generate_full_report(query, results)
    
    print(f"\n[RESULTATS]")
    print(f"  Markdown : {report['markdown']}")
    print(f"  PDF      : {report['pdf']}")
    print(f"  Excel    : {report['excel']}")
    print(f"  Alertes  : {report['alertes_count']}")
    
    if report['alerts']:
        print(f"\n  Premiere alerte : {report['alerts'][0]['type']} ({report['alerts'][0]['niveau']})")
        print(f"  Reference legale : {report['alerts'][0]['legal_ref']}")

if __name__ == "__main__":
    main()