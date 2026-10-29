# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from typing import List, Dict, Any, Tuple

class NSIAHybridSearch:
    """
    Module 3 : Recherche Hybride (Déterministe + Sémantique).
    Route automatiquement les requêtes vers le bon moteur.
    """
    
    def __init__(self, rag_instance=None):
        self.rag = rag_instance
        self.accounts_df = None
        self.customers_df = None
        
    def load_data(self, extracted_data: Dict[str, Any]):
        self.accounts_df = pd.DataFrame(extracted_data.get("bank_accounts", []))
        self.customers_df = pd.DataFrame(extracted_data.get("customer_profiles", []))
        print(f"Donnees chargees : {len(self.accounts_df)} comptes, {len(self.customers_df)} clients")
        
    def _detect_query_type(self, query: str) -> str:
        q = query.lower()
        deterministic_keywords = [
            "solde", "negatif", "positif", "superieur", "inferieur", 
            "plus grand", "plus petit", "entre", "decouvert", "pret",
            "loan", "balance", "montant", ">", "<", "=", ">=", "<=",
            "depasse"
        ]
        semantic_keywords = [
            "risque", "suspect", "anormal", "bon client", "profil",
            "comportement", "tendance", "typique", "atypique"
        ]
        has_det = any(kw in q for kw in deterministic_keywords)
        has_sem = any(kw in q for kw in semantic_keywords)
        
        if has_det and has_sem:
            return "hybrid"
        elif has_det:
            return "deterministic"
        else:
            return "semantic"
    
    def _parse_threshold(self, query: str, default: float = 0.0) -> Tuple[str, float]:
        numbers = re.findall(r'\d+(?:[.,]\d+)?', query.replace(" ", ""))
        threshold = float(numbers[0].replace(",", ".")) if numbers else default
        if "superieur" in query.lower() or "plus grand" in query.lower() or ">" in query:
            op = ">"
        elif "inferieur" in query.lower() or "plus petit" in query.lower() or "<" in query:
            op = "<"
        else:
            op = ">"
        return op, threshold
    
    def deterministic_search(self, query: str) -> List[Dict[str, Any]]:
        if self.accounts_df is None or self.accounts_df.empty:
            return []
        q = query.lower()
        df = self.accounts_df.copy()
        
        if "decouvert" in q or ("solde" in q and "negatif" in q):
            return df[df['balance'] < 0].to_dict(orient='records')
        
        if "solde" in q and "positif" in q:
            return df[df['balance'] > 0].to_dict(orient='records')
        
        if "pret" in q or "loan" in q:
            op, threshold = self._parse_threshold(query, default=100000)
            if 'loan_amount' in df.columns:
                if op == ">":
                    return df[df['loan_amount'].fillna(0) > threshold].to_dict(orient='records')
                else:
                    return df[df['loan_amount'].fillna(0) < threshold].to_dict(orient='records')
        
        if "solde" in q:
            op, threshold = self._parse_threshold(query, default=0)
            if op == ">":
                return df[df['balance'] > threshold].to_dict(orient='records')
            else:
                return df[df['balance'] < threshold].to_dict(orient='records')
        
        return df.to_dict(orient='records')
    
    def simple_text_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Fallback intelligent quand le RAG n'est pas charge."""
        if self.accounts_df is None or self.accounts_df.empty:
            return []
        
        q = query.lower()
        df = self.accounts_df.copy()
        
        # Heuristique : "risque" = comptes avec pret (exposition)
        if "risque" in q:
            if 'loan_amount' in df.columns:
                return df[df['loan_amount'].fillna(0) > 0].to_dict(orient='records')
        
        # Heuristique : "bon client" = solde eleve et pas de pret
        if "bon" in q and "client" in q:
            if 'balance' in df.columns and 'loan_amount' in df.columns:
                seuil = df['balance'].quantile(0.75)
                return df[(df['balance'] > seuil) & (df['loan_amount'].fillna(0) == 0)].to_dict(orient='records')
        
        # Fallback generique : recherche textuelle sur colonnes string
        words = [w for w in q.split() if len(w) > 3 and w not in ["compte", "client", "trouve", "avec", "dans", "pour"]]
        mask = pd.Series([False] * len(df))
        for col in df.columns:
            if df[col].dtype == 'object':
                for word in words:
                    try:
                        mask |= df[col].astype(str).str.lower().str.contains(word, na=False)
                    except:
                        pass
        
        filtered = df[mask]
        if len(filtered) == 0:
            return df.head(10).to_dict(orient='records')
        return filtered.to_dict(orient='records')
    
    def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        query_type = self._detect_query_type(query)
        print(f"Type de requete detecte : [{query_type.upper()}]")
        
        results = []
        engine = "Inconnu"  # FIX : initialisation systematique
        
        if query_type == "deterministic":
            results = self.deterministic_search(query)
            engine = "Pandas (deterministe)"
            
        elif query_type == "semantic":
            if self.rag is not None:
                try:
                    results = self.rag.search(query, k=limit)
                    engine = "FAISS (semantique)"
                except Exception as e:
                    print(f"Erreur RAG : {e}")
                    results = self.simple_text_fallback(query)
                    engine = "Pandas fallback (RAG en erreur)"
            else:
                results = self.simple_text_fallback(query)
                engine = "Pandas fallback (RAG non charge)"
                
        elif query_type == "hybrid":
            det_results = self.deterministic_search(query)
            if self.rag is not None:
                results = det_results
                engine = "Hybride (Pandas + FAISS)"
            else:
                results = det_results
                engine = "Hybride (Pandas seul)"
        
        return {
            "query": query,
            "type": query_type,
            "engine": engine,
            "count": len(results),
            "results": results[:limit]
        }