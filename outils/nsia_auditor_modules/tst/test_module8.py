# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.accounting_engine import NSIAAccountingEngine

def main():
    print("="*70)
    print("=== TEST MODULE 8 : COMPTABILITE SYSCOHADA ===")
    print("="*70)
    
    engine = NSIAAccountingEngine()
    report = engine.generate_full_report()
    
    print(report["output"])

if __name__ == "__main__":
    main()