# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.extractor import NSIAExtractor
from nsia_auditor_modules.crm_engine import NSIACRMEngine

def main():
    print("="*70)
    print("=== TEST MODULE 19 : CRM & SEGMENTATION CLIENT ===")
    print("="*70)
    
    base_dir = r"D:\kibali-IA\outputs\datasets\nsia"
    extractor = NSIAExtractor()
    
    # On charge les clients et les comptes
    res_c = extractor.extract(os.path.join(base_dir, "customer_profiles.csv"))
    customers = list(res_c["data"].values())[0]
    
    res_a = extractor.extract(os.path.join(base_dir, "bank_accounts.csv"))
    accounts = list(res_a["data"].values())[0]
    
    engine = NSIACRMEngine()
    data = engine.segment_clients(accounts, customers)
    
    report = engine.generate_report(data)
    print(report)

if __name__ == "__main__":
    main()