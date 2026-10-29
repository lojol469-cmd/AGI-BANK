# -*- coding: utf-8 -*-
import os
import json
import requests
import numpy as np
import networkx as nx
import faiss
from typing import List, Dict, Any

class NSIAGraphRAG:
    """Module 2 : Graph RAG & FAISS pour NSIA."""
    
    def __init__(self, embedding_url="http://127.0.0.1:8081/v1/embeddings"):
        self.embedding_url = embedding_url
        self.graph = nx.DiGraph()
        self.index = None
        self.account_ids = []
        self.dimension = 1024

    def _get_embedding(self, text: str) -> List[float]:
        try:
            payload = {"input": [text], "model": "bge-m3"}
            response = requests.post(self.embedding_url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"Warning Embedding : {e}")
            return [0.0] * self.dimension

    def build(self, extracted_data: Dict[str, Any]):
        print("Construction du Graphe de relations...")
        customers = extracted_data.get("customer_profiles", [])
        accounts = extracted_data.get("bank_accounts", [])
        transactions = extracted_data.get("account_transactions", [])
        
        for c in customers:
            self.graph.add_node(f"CLIENT_{c['customer_id']}", type="client", data=c)
            
        account_texts = []
        for a in accounts:
            node_id = f"ACCOUNT_{a['account_id']}"
            self.graph.add_node(node_id, type="account", data=a)
            self.graph.add_edge(f"CLIENT_{a['customer_id']}", node_id, relation="POSSEDE")
            text_desc = (
                f"Compte {a['account_id']} du client {a['customer_id']}. "
                f"Type: {a['account_type']}. Statut: {a['account_status']}. "
                f"Solde: {a['balance']}. Pret: {a.get('loan_amount', 0)}."
            )
            account_texts.append((node_id, text_desc))
            
        print("Ajout des transactions (limite 50000 pour test)...")
        for t in transactions[:50000]:
            t_id = f"TX_{t['transaction_id']}"
            self.graph.add_node(t_id, type="transaction", data=t)
            self.graph.add_edge(f"ACCOUNT_{t['account_id']}", t_id, relation="A_EFFECTUE")
            
        print(f"Graphe construit : {self.graph.number_of_nodes()} noeuds, {self.graph.number_of_edges()} liens.")
        
        print(f"Vectorisation de {len(account_texts)} comptes via BGE-M3...")
        vectors = []
        for node_id, text in account_texts:
            vec = self._get_embedding(text)
            vectors.append(vec)
            self.account_ids.append(node_id)
            
        if vectors:
            vectors_np = np.array(vectors).astype('float32')
            self.dimension = vectors_np.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(vectors_np)
            print(f"Index FAISS pret ({self.index.ntotal} vecteurs de dimension {self.dimension}).")
        else:
            print("Aucun vecteur genere.")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.index:
            return []
        print(f"Recherche semantique pour : '{query}'")
        query_vec = np.array([self._get_embedding(query)]).astype('float32')
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.account_ids):
                node_id = self.account_ids[idx]
                account_data = self.graph.nodes[node_id]['data']
                results.append({
                    "score": float(distances[0][i]),
                    "account_id": account_data['account_id'],
                    "balance": account_data['balance'],
                    "status": account_data['account_status'],
                    "node_id": node_id
                })
        return results