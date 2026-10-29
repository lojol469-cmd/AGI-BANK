# -*- coding: utf-8 -*-
import os
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd

from .extractor import NSIAExtractor
from .report_generator import LEGAL_KB


class NSIAAccountDeepDive:
    """
    Module 6 : Analyse temps reel d'un compte + Bilan SYSCOHADA + Challenge RAG.
    """
    
    def __init__(self, data_dir: str = None, rag_bridge=None):
        self.data_dir = data_dir or r"D:\kibali-IA\outputs\datasets\nsia"
        self.rag_bridge = rag_bridge  # Pour challenger avec la doc comptable
        self.accounts_df = None
        self.customers_df = None
        self.transactions_df = None
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        extractor = NSIAExtractor()
        for key, fname in {
            "accounts": "bank_accounts.csv",
            "customers": "customer_profiles.csv",
            "transactions": "account_transactions.csv"
        }.items():
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                res = extractor.extract(path)
                if res["success"]:
                    sheet = list(res["data"].keys())[0]
                    df = pd.DataFrame(res["data"][sheet])
                    if key == "accounts": self.accounts_df = df
                    elif key == "customers": self.customers_df = df
                    else: self.transactions_df = df
        self._loaded = True
        print(f"[DEEP_DIVE] Charge : {len(self.accounts_df)} comptes, {len(self.customers_df)} clients, {len(self.transactions_df)} transactions")
    
    # ---------- FICHE IDENTITE ----------
    def _get_identity(self, account: Dict) -> Dict[str, Any]:
        cid = account.get("customer_id")
        client = {}
        if self.customers_df is not None:
            match = self.customers_df[self.customers_df['customer_id'] == cid]
            if not match.empty:
                client = match.iloc[0].to_dict()
        
        return {
            "account_id": account.get("account_id"),
            "customer_id": cid,
            "client_nom": client.get("city", "N/A"),
            "client_ville": client.get("city", "N/A"),
            "account_type": account.get("account_type"),
            "account_status": account.get("account_status"),
            "creation_date": account.get("creation_date")
        }
    
    # ---------- BILAN SYSCOHADA (Perspective Banque) ----------
    def _build_balance_sheet(self, account: Dict) -> Dict[str, Any]:
        """
        Perspective BANQUE :
        - ACTIF = ce que le client doit a la banque (prêt accordé)
        - PASSIF = ce que la banque doit au client (solde créditeur)
        """
        balance = float(account.get("balance", 0) or 0)
        loan = float(account.get("loan_amount", 0) or 0)
        
        # ACTIF (créances de la banque)
        actif = {
            "credits_clients": loan if loan > 0 else 0,
            "tresorerie": balance if balance > 0 else 0
        }
        
        # PASSIF (dettes de la banque)
        passif = {
            "depots_clients": balance if balance > 0 else 0,
            "decouverts": abs(balance) if balance < 0 else 0,
            "fonds_propres": 0  # simplifié
        }
        
        total_actif = sum(actif.values())
        total_passif = sum(passif.values())
        equilibre = abs(total_actif - total_passif) < 0.01
        
        return {
            "actif": actif,
            "passif": passif,
            "total_actif": total_actif,
            "total_passif": total_passif,
            "equilibre": equilibre
        }
    
    # ---------- RATIOS ----------
    def _compute_ratios(self, account: Dict) -> Dict[str, float]:
        balance = float(account.get("balance", 0) or 0)
        loan = float(account.get("loan_amount", 0) or 0)
        rate = float(account.get("interest_rate", 0) or 0)
        term = float(account.get("term_months", 0) or 0)
        
        # Taux d'endettement : prêt / (prêt + solde)
        endettement = (loan / (loan + max(balance, 0))) * 100 if (loan + max(balance, 0)) > 0 else 0
        
        # Capacité de remboursement (approximation)
        mensualite = (loan / term) if term > 0 else 0
        capacite = (balance / mensualite) if mensualite > 0 else float('inf')
        
        # Ratio taux (usure si > 24%)
        taux_usure = rate > 0.24
        
        return {
            "taux_endettement_pct": round(endettement, 2),
            "mensualite_estimee": round(mensualite, 2),
            "mois_de_couverture": round(capacite, 1) if capacite != float('inf') else "Illimite",
            "taux_usure_detecte": taux_usure
        }
    
    # ---------- SCORE QUALITE ----------
    def _compute_score(self, account: Dict, ratios: Dict) -> Dict[str, Any]:
        score = 100
        details = []
        
        # Pénalités
        if ratios["taux_endettement_pct"] > 80:
            score -= 30
            details.append("-30 : Endettement > 80%")
        elif ratios["taux_endettement_pct"] > 50:
            score -= 15
            details.append("-15 : Endettement > 50%")
        
        if ratios["taux_usure_detecte"]:
            score -= 40
            details.append("-40 : Taux d'usure")
        
        balance = float(account.get("balance", 0) or 0)
        if balance < 0:
            score -= 20
            details.append("-20 : Solde negatif")
        
        status = str(account.get("account_status", "")).lower()
        if "closed" in status:
            score -= 10
            details.append("-10 : Compte ferme")
        
        # Note
        if score >= 90: note = "AAA"
        elif score >= 75: note = "AA"
        elif score >= 60: note = "A"
        elif score >= 45: note = "BBB"
        elif score >= 30: note = "BB"
        elif score >= 15: note = "B"
        else: note = "CCC (RISQUE)"
        
        return {"score": max(score, 0), "note": note, "details": details}
    
    # ---------- CHALLENGE RAG ----------
    def _challenge_with_rag(self, account: Dict, ratios: Dict) -> List[Dict]:
        """Croise avec la doc comptable via RAG (si dispo), sinon utilise KB embarquee."""
        challenges = []
        
        # Challenge 1 : Taux d'usure
        if ratios["taux_usure_detecte"]:
            challenges.append({
                "probleme": "Taux d'usure detecte",
                "reference": LEGAL_KB["taux_usure"]["reference"],
                "regle": LEGAL_KB["taux_usure"]["regle"],
                "action": "Requalification juridique immediate",
                "source": "KB embarquee"
            })
        
        # Challenge 2 : Endettement excessif
        if ratios["taux_endettement_pct"] > 80:
            challenges.append({
                "probleme": "Endettement > 80%",
                "reference": "Ratio prudentiel Bale III",
                "regle": "Un taux d'endettement > 80% exige une reevaluation.",
                "action": "Revue obligatoire par le comite de credit",
                "source": "KB embarquee"
            })
        
        # Challenge 3 : RAG doc si dispo
        if self.rag_bridge is not None:
            try:
                q = f"Regles SYSCOHADA pour compte avec solde {account.get('balance')} et pret {account.get('loan_amount')}"
                if hasattr(self.rag_bridge, 'search'):
                    rag_results = self.rag_bridge.search(q)
                    if rag_results:
                        challenges.append({
                            "probleme": "Verification reglementaire SYSCOHADA",
                            "reference": "RAG doc Kibali",
                            "regle": str(rag_results)[:300],
                            "action": "Verifier conformite",
                            "source": "RAG doc"
                        })
            except Exception as e:
                print(f"[DEEP_DIVE] RAG indisponible : {e}")
        
        return challenges
    
    # ---------- ANALYSE PRINCIPALE ----------
    def analyze_account(self, account_id: str = None, customer_id: str = None) -> Dict[str, Any]:
        """Analyse complete d'un compte (drill-down temps reel)."""
        self.load()
        
        if self.accounts_df is None or self.accounts_df.empty:
            return {"success": False, "output": "Aucune donnee comptes chargee."}
        
        # Recherche du compte
        df = self.accounts_df
        if account_id:
            match = df[df['account_id'].astype(str).str.startswith(str(account_id))]
        elif customer_id:
            match = df[df['customer_id'].astype(str).str.startswith(str(customer_id))]
        else:
            return {"success": False, "output": "Fournir account_id ou customer_id."}
        
        if match.empty:
            return {"success": False, "output": f"Compte introuvable : {account_id or customer_id}"}
        
        account = match.iloc[0].to_dict()
        
        # Analyser chaque dimension
        identity = self._get_identity(account)
        bilan = self._build_balance_sheet(account)
        ratios = self._compute_ratios(account)
        score = self._compute_score(account, ratios)
        challenges = self._challenge_with_rag(account, ratios)
        
        # Construire le rapport
        report = self._format_report(identity, bilan, ratios, score, challenges)
        
        return {
            "success": True,
            "output": report,
            "data": {
                "identity": identity,
                "bilan": bilan,
                "ratios": ratios,
                "score": score,
                "challenges": challenges
            }
        }
    
    def _format_report(self, identity, bilan, ratios, score, challenges) -> str:
        lines = []
        lines.append(f"# FICHE COMPTE : {str(identity['account_id'])[:8]}...")
        lines.append(f"**Client** : {identity['customer_id']}")
        lines.append(f"**Type** : {identity['account_type']} | **Statut** : {identity['account_status']}")
        lines.append(f"**Ouverture** : {identity['creation_date']}")
        lines.append("")
        
        lines.append("## BILAN COMPTABLE (Perspective BANQUE)")
        lines.append("### ACTIF (Créances)")
        for k, v in bilan['actif'].items():
            lines.append(f"- {k} : **{v:,.2f}**")
        lines.append(f"**Total Actif : {bilan['total_actif']:,.2f}**")
        lines.append("")
        lines.append("### PASSIF (Dettes)")
        for k, v in bilan['passif'].items():
            lines.append(f"- {k} : **{v:,.2f}**")
        lines.append(f"**Total Passif : {bilan['total_passif']:,.2f}**")
        lines.append("")
        lines.append(f"**Équilibre** : {'✅ OUI' if bilan['equilibre'] else '⚠️ NON (écart de ' + str(round(abs(bilan['total_actif']-bilan['total_passif']),2)) + ')'}")
        lines.append("")
        
        lines.append("## RATIOS FINANCIERS")
        lines.append(f"- Taux d'endettement : **{ratios['taux_endettement_pct']}%**")
        lines.append(f"- Mensualité estimée : **{ratios['mensualite_estimee']:,.2f}**")
        lines.append(f"- Mois de couverture : **{ratios['mois_de_couverture']}**")
        lines.append(f"- Taux d'usure : {'🚨 DETECTE' if ratios['taux_usure_detecte'] else '✅ Non'}")
        lines.append("")
        
        lines.append(f"## SCORE QUALITE : **{score['note']}** ({score['score']}/100)")
        for d in score['details']:
            lines.append(f"  {d}")
        lines.append("")
        
        if challenges:
            lines.append("## CHALLENGES (Croisement Légal)")
            for c in challenges:
                lines.append(f"### ⚠️ {c['probleme']}")
                lines.append(f"- **Référence** : {c['reference']}")
                lines.append(f"- **Règle** : {c['regle']}")
                lines.append(f"- **Action** : {c['action']}")
                lines.append("")
        
        lines.append("---")
        lines.append("**Signé : KIBALI AGI EXPERT**")
        return "\n".join(lines)