# -*- coding: utf-8 -*-
import os
import sys
import time

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.extractor import NSIAExtractor

def main():
    print("=== TEST MODULE 1 : DONNEES REELLES KAGGLE ===")
    
    base_dir = r"D:\kibali-IA\outputs\datasets\nsia"
    fichiers = [
        "transaction_codes.csv",     # Le plus petit (dictionnaire)
        "customer_profiles.csv",
        "bank_accounts.csv",
        "account_transactions.csv",  # Le monstre de 300 Mo
    ]
    
    extractor = NSIAExtractor()
    
    for fichier in fichiers:
        chemin = os.path.join(base_dir, fichier)
        if not os.path.exists(chemin):
            print(f"ATTENTION - Fichier manquant : {chemin}")
            continue
        
        taille_mb = os.path.getsize(chemin) / (1024 * 1024)
        print(f"\n>>> Test de {fichier} ({taille_mb:.2f} Mo)...")
        
        start = time.time()
        try:
            result = extractor.extract(chemin)
        except MemoryError:
            print(f"   MEMORY ERROR : le fichier est trop gros pour la RAM disponible.")
            continue
        except Exception as e:
            print(f"   ERREUR INATTENDUE : {e}")
            continue
        duree = time.time() - start
        
        if result["success"]:
            print(f"   OK - Extrait en {duree:.2f}s | {result['total_rows']} lignes")
            
            first_sheet = list(result['data'].keys())[0]
            preview = result['data'][first_sheet][:1]
            if preview:
                print(f"   Colonnes : {list(preview[0].keys())}")
        else:
            print(f"   ECHEC : {result['error']}")

if __name__ == "__main__":
    main()
