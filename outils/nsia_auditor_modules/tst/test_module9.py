# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.extractor import NSIAExtractor
from nsia_auditor_modules.financial_statements import NSIAFinancialStatements

def main():
    print("="*70)
    print("=== TEST MODULE 9 : ETATS FINANCIERS BANCAIRES ===")
    print("="*70)
    
    base_dir = r"D:\kibali-IA\outputs\datasets\nsia"
    extractor = NSIAExtractor()
    
    # Charger le portefeuille
    res = extractor.extract(os.path.join(base_dir, "bank_accounts.csv"))
    accounts = res["data"]["default"] if "default" in res["data"] else list(res["data"].values())[0]
    
    # Générer les états financiers
    statements = NSIAFinancialStatements(accounts)
    report = statements.generate_report()
    
    print(report["output"])

if __name__ == "__main__":
    main()