# -*- coding: utf-8 -*-
import os
import sys
import time
import pandas as pd
import random
import json

# Ajout du chemin pour importer le module
sys.path.insert(0, r"D:\kibali-IA\outils")
from nsia_auditor_modules.extractor import NSIAExtractor

def generate_mock_excel(file_path: str, num_rows: int = 10000):
    """Génère un fichier Excel de test avec 10 000 comptes NSIA"""
    print(f"⏳ Génération du fichier de test ({num_rows} lignes)...")
    
    data = []
    for i in range(num_rows):
        data.append({
            "ID_Client": f"NSIA-{1000+i}",
            "Nom": f"Client_{i}",
            "Solde": round(random.uniform(-500000, 2000000), 2),
            "Decouvert_Autorise": round(random.uniform(0, 100000), 2),
            "Date_Ouverture": f"202{random.randint(0,3)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "Agence": f"Agence_{random.randint(1,10)}"
        })
    
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    print(f"✅ Fichier généré : {file_path}")

def main():
    print("=== TEST DU MODULE 1 : EXTRACTEUR NSIA ===")
    
    test_file = r"D:\kibali-IA\outputs\test_nsia_10000.xlsx"
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    # 1. Générer le fichier de test
    generate_mock_excel(test_file, 10000)
    
    # 2. Tester l'extraction
    extractor = NSIAExtractor()
    print("\n🔍 Lancement de l'extraction...")
    start_time = time.time()
    
    result = extractor.extract(test_file)
    
    end_time = time.time()
    
    # 3. Afficher les résultats
    if result["success"]:
        print(f"\n✅ SUCCÈS en {end_time - start_time:.2f} secondes !")
        print(f"📊 Format détecté : {result['format']}")
        print(f"📈 Total de lignes extraites : {result['total_rows']}")
        
        # Aperçu des 2 premières lignes
        first_sheet = list(result['data'].keys())[0]
        preview = result['data'][first_sheet][:2]
        print(f"\n👀 Aperçu des données ({first_sheet}) :")
        print(json.dumps(preview, indent=4, ensure_ascii=False))
        
        if result['total_rows'] == 10000:
            print("\n🏆 TEST RÉUSSI : Les 10 000 comptes ont été extraits et structurés sans perte.")
        else:
            print("\n⚠️ ATTENTION : Le nombre de lignes extraites ne correspond pas.")
    else:
        print(f"\n❌ ÉCHEC : {result['error']}")

if __name__ == "__main__":
    main()