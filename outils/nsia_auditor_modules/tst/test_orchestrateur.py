# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
_parent = os.path.dirname(_outils_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Import via le package outils (simule le ToolManager)
from outils.nsi_account_auditor import NSIAAccountAuditorTool

def main():
    print("=== TEST ORCHESTRATEUR NSIA (simulation KIBALI) ===\n")
    
    tool = NSIAAccountAuditorTool()
    
    print(f"Outil : {tool.name}")
    print(f"Capabilities : {tool.capabilities}")
    print(f"Score can_handle('audit NSIA') : {tool.can_handle('audit NSIA')}")
    print(f"Score can_handle('bonjour') : {tool.can_handle('bonjour')}")
    print()
    
    # Simulation d'un appel KIBALI
    result = tool.execute(
        action="audit",
        file_path=r"D:\kibali-IA\outputs\datasets\nsia\bank_accounts.csv",
        query="Comptes avec un pret superieur a 200000"
    )
    
    print("\n" + "="*70)
    print("RESULTAT FINAL")
    print("="*70)
    if result["success"]:
        print(result["output"])
    else:
        print("ECHEC :", result["output"])

if __name__ == "__main__":
    main()