# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.agency_benchmark import NSIAAgencyBenchmark

def main():
    print("="*70)
    print("=== TEST MODULE 7 : BENCHMARK INTER-AGENCES ===")
    print("="*70)
    
    engine = NSIAAgencyBenchmark()
    result = engine.generate_report()
    
    if result["success"]:
        print(result["output"])
    else:
        print(f"ECHEC : {result['output']}")

if __name__ == "__main__":
    main()