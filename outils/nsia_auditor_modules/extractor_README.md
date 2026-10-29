# MODULE 1 : Extracteur NSIA (extractor.py)

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
