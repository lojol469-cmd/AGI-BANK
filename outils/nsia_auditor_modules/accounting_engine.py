# -*- coding: utf-8 -*-
import os
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd

from .extractor import NSIAExtractor


# Plan comptable SYSCOHADA révisé (extrait)
SYSCOHADA_ACCOUNTS = {
    # Classe 1 : Ressources durables
    "101": "Capital social",
    "106": "Réserves",
    "12": "Résultat net",
    "16": "Emprunts et dettes assimilées",
    "17": "Dettes de crédit-bail",
    # Classe 2 : Actif immobilisé
    "20": "Charges immobilisées",
    "21": "Immobilisations incorporelles",
    "22": "Terrains",
    "23": "Bâtiments",
    "24": "Matériel",
    # Classe 3 : Stocks
    "31": "Marchandises",
    "32": "Matières premières",
    "33": "Autres approvisionnements",
    # Classe 4 : Tiers
    "401": "Fournisseurs",
    "411": "Clients",
    "421": "Personnel",
    "431": "Sécurité sociale",
    "441": "État",
    "445": "TVA",
    # Classe 5 : Trésorerie
    "512": "Banque",
    "514": "Chèques postaux",
    "53": "Caisse",
    "56": "Crédits de trésorerie",
    # Classe 6 : Charges
    "601": "Achats de marchandises",
    "605": "Achats de matières",
    "62": "Services extérieurs",
    "63": "Impôts et taxes",
    "64": "Charges de personnel",
    "66": "Charges financières",
    # Classe 7 : Produits
    "701": "Ventes de marchandises",
    "706": "Prestations de services",
    "75": "Autres produits",
    "77": "Revenus financiers"
}


class NSIAAccountingEngine:
    """
    Module 8 : Moteur d'écritures comptables SYSCOHADA.
    - Génère le Journal, le Grand Livre, la Balance
    - Vérifie l'équilibre Débit = Crédit
    - Détecte les anomalies comptables
    """
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or r"D:\kibali-IA\outputs\datasets\nsia"
        self.transactions_df = None
        self.accounts_df = None
        self._loaded = False
        self.ecritures = []
    
    def load(self):
        if self._loaded:
            return
        extractor = NSIAExtractor()
        for key, fname in {
            "transactions": "account_transactions.csv",
            "accounts": "bank_accounts.csv"
        }.items():
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                res = extractor.extract(path)
                if res["success"]:
                    sheet = list(res["data"].keys())[0]
                    df = pd.DataFrame(res["data"][sheet])
                    if key == "transactions": self.transactions_df = df
                    else: self.accounts_df = df
        self._loaded = True
        print(f"[COMPTA] Charge : {len(self.transactions_df)} transactions, {len(self.accounts_df)} comptes")
    
    # ---------- MAPPING TRANSACTION -> ECRITURE COMPTABLE ----------
    def _map_transaction_to_ecriture(self, tx: Dict, account: Dict) -> List[Dict]:
        """
        Transforme une transaction bancaire en écritures comptables SYSCOHADA.
        Retourne une liste (car une transaction = 2 lignes minimum : débit + crédit).
        """
        amount = float(tx.get("amount", 0) or 0)
        if amount == 0:
            return []
        
        tx_type = str(tx.get("transaction_type", "")).lower()
        tx_code = str(tx.get("transaction_code", ""))
        date = str(tx.get("created_time", ""))[:10]
        account_id = str(account.get("account_id", ""))[:8]
        
        # Règles de mapping SYSCOHADA
        ecritures = []
        
        if "credit" in tx_type or amount > 0:
            # Crédit client = augmentation du compte bancaire
            ecritures.append({
                "date": date,
                "compte": "512",  # Banque
                "libelle": f"Encaissement client {account_id}",
                "debit": amount,
                "credit": 0,
                "piece": tx.get("transaction_id", "")[:8],
                "ref_transaction": tx.get("transaction_id")
            })
            ecritures.append({
                "date": date,
                "compte": "411",  # Clients
                "libelle": f"Règlement client {account_id}",
                "debit": 0,
                "credit": amount,
                "piece": tx.get("transaction_id", "")[:8],
                "ref_transaction": tx.get("transaction_id")
            })
        else:
            # Débit client = paiement / retrait
            ecritures.append({
                "date": date,
                "compte": "411",  # Clients
                "libelle": f"Paiement client {account_id}",
                "debit": abs(amount),
                "credit": 0,
                "piece": tx.get("transaction_id", "")[:8],
                "ref_transaction": tx.get("transaction_id")
            })
            ecritures.append({
                "date": date,
                "compte": "512",  # Banque
                "libelle": f"Décaissement banque {account_id}",
                "debit": 0,
                "credit": abs(amount),
                "piece": tx.get("transaction_id", "")[:8],
                "ref_transaction": tx.get("transaction_id")
            })
        
        return ecritures
    
    # ---------- GENERATION DU JOURNAL ----------
    def generate_journal(self, limit: int = 1000) -> List[Dict]:
        """Génère les écritures comptables (journal)."""
        self.load()
        
        if self.transactions_df is None or self.transactions_df.empty:
            return []
        
        # Mapping account_id -> account data
        accounts_map = {}
        if self.accounts_df is not None:
            for _, acc in self.accounts_df.iterrows():
                accounts_map[str(acc['account_id'])] = acc.to_dict()
        
        self.ecritures = []
        for _, tx in self.transactions_df.head(limit).iterrows():
            tx_dict = tx.to_dict()
            acc_id = str(tx_dict.get("account_id", ""))
            account = accounts_map.get(acc_id, {})
            ecritures = self._map_transaction_to_ecriture(tx_dict, account)
            self.ecritures.extend(ecritures)
        
        return self.ecritures
    
    # ---------- GRAND LIVRE ----------
    def generate_grand_livre(self) -> Dict[str, List[Dict]]:
        """Regroupe les écritures par compte comptable."""
        if not self.ecritures:
            self.generate_journal()
        
        grand_livre = {}
        for e in self.ecritures:
            compte = e["compte"]
            if compte not in grand_livre:
                grand_livre[compte] = []
            grand_livre[compte].append(e)
        
        return grand_livre
    
    # ---------- BALANCE ----------
    def generate_balance(self) -> List[Dict]:
        """Calcule la balance comptable (totaux débit/crédit par compte)."""
        grand_livre = self.generate_grand_livre()
        
        balance = []
        for compte, ecritures in grand_livre.items():
            total_debit = sum(e["debit"] for e in ecritures)
            total_credit = sum(e["credit"] for e in ecritures)
            solde = total_debit - total_credit
            
            balance.append({
                "compte": compte,
                "intitule": SYSCOHADA_ACCOUNTS.get(compte, "Compte inconnu"),
                "total_debit": round(total_debit, 2),
                "total_credit": round(total_credit, 2),
                "solde_debiteur": round(solde, 2) if solde > 0 else 0,
                "solde_crediteur": round(-solde, 2) if solde < 0 else 0
            })
        
        return sorted(balance, key=lambda x: x["compte"])
    
    # ---------- VERIFICATION EQUILIBRE ----------
    def verify_equilibre(self) -> Dict[str, Any]:
        """Vérifie que Total Débit = Total Crédit."""
        if not self.ecritures:
            self.generate_journal()
        
        total_debit = sum(e["debit"] for e in self.ecritures)
        total_credit = sum(e["credit"] for e in self.ecritures)
        ecart = abs(total_debit - total_credit)
        
        return {
            "total_debit": round(total_debit, 2),
            "total_credit": round(total_credit, 2),
            "ecart": round(ecart, 2),
            "equilibre": ecart < 0.01,
            "nb_ecritures": len(self.ecritures)
        }
    
    # ---------- DETECTION ANOMALIES ----------
    def detect_anomalies(self) -> List[Dict]:
        """Détecte les anomalies comptables (comptes inconnus, montants suspects)."""
        anomalies = []
        
        for e in self.ecritures:
            # Compte non référencé
            if e["compte"] not in SYSCOHADA_ACCOUNTS:
                anomalies.append({
                    "type": "COMPTE_INCONNU",
                    "piece": e["piece"],
                    "compte": e["compte"],
                    "niveau": "MOYEN"
                })
            
            # Montant très élevé (> 1M)
            if e["debit"] > 1000000 or e["credit"] > 1000000:
                anomalies.append({
                    "type": "MONTANT_ELEVE",
                    "piece": e["piece"],
                    "montant": max(e["debit"], e["credit"]),
                    "niveau": "ELEVE"
                })
        
        return anomalies[:50]  # Limiter
    
    # ---------- RAPPORT COMPLET ----------
    def generate_full_report(self) -> Dict[str, Any]:
        """Rapport comptable complet signé KIBALI."""
        self.generate_journal(limit=500)
        equilibre = self.verify_equilibre()
        balance = self.generate_balance()
        anomalies = self.detect_anomalies()
        
        lines = ["# RAPPORT COMPTABLE SYSCOHADA", ""]
        lines.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Écritures générées** : {equilibre['nb_ecritures']}")
        lines.append(f"**Total Débit** : {equilibre['total_debit']:,.2f}")
        lines.append(f"**Total Crédit** : {equilibre['total_credit']:,.2f}")
        lines.append(f"**Écart** : {equilibre['ecart']:,.2f}")
        lines.append(f"**Équilibre** : {'✅ OUI' if equilibre['equilibre'] else '⚠️ NON'}")
        lines.append("")
        
        lines.append("## Balance comptable")
        lines.append("| Compte | Intitulé | Débit | Crédit | Solde |")
        lines.append("|---|---|---|---|---|")
        for b in balance:
            solde = b['solde_debiteur'] if b['solde_debiteur'] > 0 else -b['solde_crediteur']
            lines.append(
                f"| {b['compte']} | {b['intitule']} | "
                f"{b['total_debit']:,.0f} | {b['total_credit']:,.0f} | "
                f"{solde:,.0f} |"
            )
        lines.append("")
        
        if anomalies:
            lines.append(f"## Anomalies ({len(anomalies)})")
            for a in anomalies[:10]:
                lines.append(f"- **{a['type']}** : pièce {a.get('piece', '?')} ({a.get('compte', a.get('montant', '?'))})")
            lines.append("")
        
        lines.append("---")
        lines.append("**Signé : KIBALI AGI EXPERT - Module Comptabilité SYSCOHADA**")
        
        return {
            "success": True,
            "output": "\n".join(lines),
            "data": {
                "equilibre": equilibre,
                "balance": balance,
                "anomalies": anomalies,
                "ecritures_count": len(self.ecritures)
            }
        }