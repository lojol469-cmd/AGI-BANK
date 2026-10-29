# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
_parent = os.path.dirname(_outils_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from outils.nsi_bank_arsenal import NSIBankArsenalTool

def main():
    print("="*70)
    print("=== TEST FINAL : ARSENAL KIBALI BANQUE (20 MODULES) ===")
    print("="*70)
    
    tool = NSIBankArsenalTool()
    
    # 1. Test de routage
    print(f"Score de routage ('calcule la TVA') : {tool.can_handle('calcule la TVA')}")
    
    # 2. Test d'un module (Paie)
    print("\n--- TEST MODULE PAIE ---")
    result = tool.execute(action="execute_module", module_name="paie", params={"salaire": 750000})
    print(result["output"][:400] + "\n... [tronque]")
    
    # 3. Test d'un module (Conformité)
    print("\n--- TEST MODULE CONFORMITE ---")
    result2 = tool.execute(action="execute_module", module_name="conformite")
    print(result2["output"][:400] + "\n... [tronque]")

if __name__ == "__main__":
    main()