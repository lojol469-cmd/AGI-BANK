# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.account_deep_dive import NSIAAccountDeepDive

def main():
    print("="*70)
    print("=== TEST MODULE 6 : DEEP DIVE COMPTE + BILAN SYSCOHADA ===")
    print("="*70)
    
    engine = NSIAAccountDeepDive()
    engine.load()
    
    # Test 1 : compte specifique (on prend le premier avec un pret)
    import pandas as pd
    df = engine.accounts_df
    sample = df[df['loan_amount'].fillna(0) > 400000].head(1)
    
    if not sample.empty:
        acc_id = sample.iloc[0]['account_id']
        print(f"\n[TEST 1] Analyse du compte {str(acc_id)[:12]}...")
        result = engine.analyze_account(account_id=acc_id)
        print(result['output'])
    
    # Test 2 : compte en decouvert simule (on cherche un solde bas)
    low_balance = df.nsmallest(1, 'balance')
    acc_id2 = low_balance.iloc[0]['account_id']
    print(f"\n{'='*70}")
    print(f"[TEST 2] Analyse du compte le plus bas : {str(acc_id2)[:12]}...")
    result2 = engine.analyze_account(account_id=acc_id2)
    print(result2['output'][:800] + "\n... [tronque]")

if __name__ == "__main__":
    main()