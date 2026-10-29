# -*- coding: utf-8 -*-
import os

base = r"D:\kibali-IA\outils\nsia_auditor_modules"

readme1 = """# MODULE 1 : Extracteur NSIA (extractor.py)

## Role
Ingestion et normalisation de fichiers bancaires massifs (Excel, CSV, JSON, PDF, TXT) pour le pipeline NSIA.

## Capacites
- Lecture multi-onglets Excel (.xlsx, .xls)
- Detection automatique du separateur CSV (, ; tab)
- Fallback d encodage (utf-8 -> latin-1) pour les exports bancaires legacy
- Extraction de texte brut depuis PDF (releves scannes)
- Normalisation JSON universelle

## Resultats des tests CLI (preuve d execution)

### Test 1 : Synthetique (10 000 comptes)
| Metrique | Valeur |
|---|---|
| Lignes | 10 000 |
| Temps | 0.55 s |
| Format | .xlsx |

### Test 2 : Donnees reelles Kaggle (NSIA-like)
| Fichier | Lignes | Temps |
|---|---|---|
| transaction_codes.csv | 15 | 0.00 s |
| customer_profiles.csv | 1 000 | 0.01 s |
| bank_accounts.csv | 3 553 | 0.01 s |
| account_transactions.csv | 2 377 169 | 13.15 s |

**Debit moyen : ~180 000 lignes/seconde** [OK]
**Aucun MemoryError** [OK]

## Contrat de sortie
{
  success: true,
  format: excel|csv|json|pdf|txt,
  total_rows: 2377169,
  data: {Sheet1: [{col1: val1}, ...]},
  metadata: {sheets: [Sheet1]}
}

## Statut
[CERTIFIE] - Teste sur 2,37M de lignes reelles
"""

readme2 = """# MODULE 2 : Graph RAG & FAISS (graph_rag.py)

## Role
Construction d un graphe de relations (Client -> Compte -> Transaction) et indexation vectorielle (BGE-M3 + FAISS) pour la recherche semantique bancaire.

## Architecture
1. Graphe NetworkX : modelisation deterministe des relations
   - Noeud CLIENT_<id> lie a ACCOUNT_<id>
   - Noeud ACCOUNT_<id> lie a TX_<id>
2. Index FAISS : vecteurs denses de dimension 1024 (BGE-M3)
3. Serveur d embedding : Port 8081 (Kibali natif)

## Test CLI
Voir tst/test_module2.py

## API
rag = NSIAGraphRAG()
rag.build(extracted_data)
results = rag.search(query)

## Statut
[EN COURS DE TEST]
"""

with open(os.path.join(base, "extractor_README.md"), "w", encoding="utf-8") as f:
    f.write(readme1)

with open(os.path.join(base, "graph_rag_README.md"), "w", encoding="utf-8") as f:
    f.write(readme2)

print("OK - extractor_README.md cree")
print("OK - graph_rag_README.md cree")
