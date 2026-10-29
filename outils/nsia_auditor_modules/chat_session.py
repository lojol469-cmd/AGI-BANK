# -*- coding: utf-8 -*-
import os
import re
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd

from .extractor import NSIAExtractor
from .hybrid_search import NSIAHybridSearch
from .report_generator import NSIAReportGenerator


class NSIAChatSession:
    """
    Module 5 : Session de chat temps reel sur un fichier NSIA.
    - Charge les donnees UNE SEULE FOIS en memoire
    - Maintient l'historique conversationnel
    - Repond a 10+ questions successives
    - Se souvient des resultats precedents ("et parmi eux...")
    """
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or r"D:\kibali-IA\outputs\datasets\nsia"
        self.data = {}
        self.history = []
        self.last_results = None
        self.last_query = None
        self.engine = NSIAHybridSearch()
        self.generator = NSIAReportGenerator()
        self._loaded = False
        self.session_start = datetime.now()
    
    def load(self):
        """Charge les donnees en memoire une seule fois."""
        if self._loaded:
            return
        
        print(f"[SESSION] Chargement des donnees depuis {self.data_dir}...")
        extractor = NSIAExtractor()
        files = {
            "customer_profiles": "customer_profiles.csv",
            "bank_accounts": "bank_accounts.csv",
            "account_transactions": "account_transactions.csv"
        }
        
        for key, fname in files.items():
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                res = extractor.extract(path)
                if res["success"]:
                    first_sheet = list(res["data"].keys())[0]
                    self.data[key] = res["data"][first_sheet]
                    print(f"[SESSION] {key} : {res['total_rows']} lignes")
        
        self.engine.load_data(self.data)
        self._loaded = True
        print(f"[SESSION] Prete ({datetime.now().strftime('%H:%M:%S')})")
    
    # ---------- STATISTIQUES ----------
    def _handle_stats(self, question: str) -> Dict[str, Any]:
        q = question.lower()
        df = pd.DataFrame(self.data.get("bank_accounts", []))
        if df.empty:
            return None
        
        # "Combien de comptes"
        if "combien" in q and ("compte" in q or "total" in q):
            return {
                "success": True,
                "answer": f"**{len(df)} comptes** au total dans le portefeuille.",
                "type": "stats"
            }
        
        # "Combien de clients"
        if "combien" in q and "client" in q:
            clients = self.data.get("customer_profiles", [])
            return {
                "success": True,
                "answer": f"**{len(clients)} clients** dans le dataset.",
                "type": "stats"
            }
        
        # "Moyenne des soldes"
        if "moyenne" in q and "solde" in q:
            mean = df['balance'].mean()
            median = df['balance'].median()
            return {
                "success": True,
                "answer": f"**Solde moyen** : {mean:,.2f}\n**Solde median** : {median:,.2f}",
                "type": "stats"
            }
        
        # "Total des prets"
        if ("total" in q or "somme" in q) and "pret" in q:
            total = df['loan_amount'].fillna(0).sum()
            count = (df['loan_amount'].fillna(0) > 0).sum()
            return {
                "success": True,
                "answer": f"**Total des prets** : {total:,.2f}\n**Nombre de comptes avec pret** : {count}",
                "type": "stats"
            }
        
        # "Plus gros prets" / "top 5"
        if ("plus gros" in q or "plus grand" in q or "top" in q) and "pret" in q:
            top = df.nlargest(5, 'loan_amount')[['account_id', 'loan_amount', 'balance']]
            lines = ["**Top 5 des plus gros prets :**"]
            for _, row in top.iterrows():
                lines.append(
                    f"- Compte `{str(row['account_id'])[:8]}...` | Pret: {row['loan_amount']:,.2f} | Solde: {row['balance']:,.2f}"
                )
            return {"success": True, "answer": "\n".join(lines), "type": "stats"}
        
        # "Maximum de solde"
        if ("maximum" in q or "plus eleve" in q) and "solde" in q:
            top = df.nlargest(3, 'balance')[['account_id', 'balance']]
            lines = ["**Top 3 des soldes :**"]
            for _, row in top.iterrows():
                lines.append(f"- Compte `{str(row['account_id'])[:8]}...` | Solde: {row['balance']:,.2f}")
            return {"success": True, "answer": "\n".join(lines), "type": "stats"}
        
        return None
    
    # ---------- GESTION DES PRONOMS ("et parmi eux") ----------
    def _resolve_context(self, question: str) -> str:
        q = question.lower().strip()
        # Pronoms de reprise
        reprise = ["et parmi eux", "et parmi ces", "lesquels", "combien parmi", 
                   "dans ce lot", "dans ces comptes", "sur ces comptes"]
        
        for p in reprise:
            if q.startswith(p) or p in q:
                return "context"
        return "normal"
    
    # ---------- POINT D'ENTREE ----------
    def ask(self, question: str, generate_report: bool = False) -> Dict[str, Any]:
        """Pose une question, garde la memoire, retourne une reponse."""
        self.load()
        
        # 1. Statistiques rapides
        stats = self._handle_stats(question)
        if stats:
            self._add_history(question, stats["answer"], "stats")
            return stats
        
        # 2. Detection de contexte
        ctx = self._resolve_context(question)
        context_note = ""
        if ctx == "context" and self.last_results:
            context_note = f"\n*[Contexte : reprise des {self.last_results['count']} resultats precedents]*\n"
        
        # 3. Recherche hybride
        results = self.engine.search(question, limit=50)
        
        # Si contexte et pas de nouveaux resultats, on repart des precedents
        if ctx == "context" and self.last_results and results['count'] == 0:
            results = self.last_results
        
        self.last_results = results
        self.last_query = question
        
        # 4. Formatage
        answer = context_note
        answer += f"**Requete** : `{question}`\n"
        answer += f"**Moteur** : {results['engine']}\n"
        answer += f"**Resultats** : {results['count']} compte(s)\n\n"
        
        if results['results']:
            answer += "**Apercu des 5 premiers** :\n"
            for r in results['results'][:5]:
                aid = str(r.get('account_id', 'N/A'))[:12]
                solde = r.get('balance', 0) or 0
                pret = r.get('loan_amount', 0) or 0
                answer += f"- `{aid}...` | Solde: {solde:,.2f} | Pret: {pret:,.2f}\n"
        
        # 5. Rapport optionnel
        if generate_report and results['count'] > 0:
            report = self.generator.generate_full_report(question, results)
            answer += f"\n**Rapport PDF** : `{report['pdf']}`\n"
            answer += f"**Alertes** : {report['alertes_count']}\n"
            self._add_history(question, answer, "report", report)
            return {
                "success": True,
                "answer": answer,
                "type": "report",
                "count": results['count'],
                "report": report
            }
        
        self._add_history(question, answer, results['type'])
        return {
            "success": True,
            "answer": answer,
            "type": results['type'],
            "count": results['count'],
            "results": results['results'][:10]
        }
    
    def _add_history(self, question: str, answer: str, type_: str, extra: Dict = None):
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "question": question,
            "answer": answer,
            "type": type_
        }
        if extra:
            entry["extra"] = extra
        self.history.append(entry)
    
    def get_history(self) -> List[Dict]:
        return self.history
    
    def summary(self) -> str:
        """Resume de la session."""
        duration = (datetime.now() - self.session_start).total_seconds()
        lines = [
            f"# Session de chat NSIA",
            f"- Debut : {self.session_start.strftime('%H:%M:%S')}",
            f"- Duree : {duration:.1f}s",
            f"- Questions posees : {len(self.history)}",
            ""
        ]
        for i, h in enumerate(self.history, 1):
            lines.append(f"{i}. [{h['timestamp']}] ({h['type']}) {h['question']}")
        return "\n".join(lines)