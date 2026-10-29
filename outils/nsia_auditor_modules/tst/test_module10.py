# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.extractor import NSIAExtractor
from nsia_auditor_modules.bank_reconciliation import NSIABankReconciliation

def main():
    print("="*70)
    print("=== TEST MODULE 10 : RAPPROCHEMENT BANCAIRE ===")
    print("="*70)
    
    base_dir = r"D:\kibali-IA\outputs\datasets\nsia"
    extractor = NSIAExtractor()
    
    res = extractor.extract(os.path.join(base_dir, "account_transactions.csv"))
    transactions = list(res["data"].values())[0][:1000] # On prend 1000 transactions pour le test
    
    engine = NSIABankReconciliation(transactions)
    report = engine.reconcile()
    
    print(report["output"])

if __name__ == "__main__":
    main()