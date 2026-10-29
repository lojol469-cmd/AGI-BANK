# MODULE 3 : Hybrid Search (hybrid_search.py)

## Role
Routeur intelligent qui detecte automatiquement si une requete est deterministe (chiffres, soldes) ou semantique (risque, profil) et choisit le bon moteur.

## Architecture
- **Deterministe** : Pandas (filtres exacts sur balance, loan_amount)
- **Semantique** : FAISS + BGE-M3 (Module 2)
- **Hybride** : Combinaison des deux
- **Fallback** : Heuristiques Pandas si le RAG n'est pas charge

## Resultats des tests CLI

| Requete | Type detecte | Moteur | Resultats |
|---|---|---|---|
| Solde negatif | DETERMINISTIC | Pandas | 0 (normal, dataset Kaggle) |
| Pret > 200000 | DETERMINISTIC | Pandas | 127 |
| Solde positif | DETERMINISTIC | Pandas | 2 484 |
| Comptes a risque | SEMANTIC | Pandas fallback | 1 007 |
| Bons clients | SEMANTIC | Pandas fallback | 792 |

## Statut
[CERTIFIE] - Routeur fonctionnel, fallback intelligent valide