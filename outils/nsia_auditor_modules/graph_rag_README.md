# MODULE 2 : Graph RAG & FAISS (graph_rag.py)

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
