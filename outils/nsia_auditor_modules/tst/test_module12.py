# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.payroll_engine import NSIAPayrollEngine

def main():
    print("="*70)
    print("=== TEST MODULE 12 : MOTEUR DE PAIE (SÉNÉGAL) ===")
    print("="*70)
    
    engine = NSIAPayrollEngine()
    
    # Simulation d'un cadre avec 1 000 000 FCFA de salaire de base et 100 000 de primes
    report = engine.generate_report(salaire_brut=1000000, primes=100000)
    print(report["output"])

if __name__ == "__main__":
    main()