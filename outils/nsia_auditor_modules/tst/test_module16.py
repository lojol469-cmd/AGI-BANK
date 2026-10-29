# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.extractor import NSIAExtractor
from nsia_auditor_modules.fraud_detector import NSIAFraudDetector

def main():
    print("="*70)
    print("=== TEST MODULE 16 : DÉTECTION DE FRAUDE (LBC/FT) ===")
    print("="*70)
    
    base_dir = r"D:\kibali-IA\outputs\datasets\nsia"
    extractor = NSIAExtractor()
    
    # On charge un échantillon de 50 000 transactions
    res = extractor.extract(os.path.join(base_dir, "account_transactions.csv"))
    all_tx = list(res["data"].values())[0]
    sample_tx = all_tx[:50000]
    
    print(f"Chargement de {len(sample_tx):,} transactions pour analyse...")
    
    detector = NSIAFraudDetector()
    report = detector.analyze_transactions(sample_tx, amount_threshold=300000)
    
    print(report["output"])

if __name__ == "__main__":
    main()