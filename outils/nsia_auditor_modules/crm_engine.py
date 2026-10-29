# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from datetime import datetime
import math

def safe_float(val) -> float:
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except:
        return 0.0

class NSIACRMEngine:
    """
    Module 19 : CRM & Relation Client.
    Segmente les clients et évalue le risque de départ (Churn).
    """
    
    def __init__(self):
        pass

    def segment_clients(self, accounts: List[Dict], customers: List[Dict]) -> Dict[str, Any]:
        """
        Analyse les comptes et retourne les segments CRM.
        """
        # Dictionnaire pour regrouper par client
        client_data = {}
        
        # 1. Initialiser avec les infos clients
        for c in customers:
            cid = c.get("customer_id")
            client_data[cid] = {
                "customer_id": cid,
                "city": c.get("city", "Inconnue"),
                "total_balance": 0.0,
                "total_loan": 0.0,
                "nb_accounts": 0,
                "statuses": []
            }

        # 2. Agréger les comptes par client
        for a in accounts:
            cid = a.get("customer_id")
            if cid not in client_data:
                continue
                
            client_data[cid]["total_balance"] += safe_float(a.get("balance"))
            client_data[cid]["total_loan"] += safe_float(a.get("loan_amount"))
            client_data[cid]["nb_accounts"] += 1
            client_data[cid]["statuses"].append(str(a.get("account_status", "")).lower())

        # 3. Segmentation
        segments = {
            "VIP": [],
            "PREMIUM": [],
            "STANDARD": [],
            "CHURN_RISK": [],
            "SURENDETTE": []
        }

        for cid, data in client_data.items():
            balance = data["total_balance"]
            loan = data["total_loan"]
            statuses = data["statuses"]
            
            # Règles de segmentation (Heuristiques)
            if any("inactive" in s or "closed" in s for s in statuses) and balance < 100000:
                segments["CHURN_RISK"].append(data)
            elif loan > 1000000 and balance < 100000:
                segments["SURENDETTE"].append(data)
            elif balance > 500000 and loan == 0:
                segments["VIP"].append(data)
            elif balance > 200000:
                segments["PREMIUM"].append(data)
            else:
                segments["STANDARD"].append(data)

        return {
            "success": True,
            "total_clients": len(client_data),
            "segments": segments
        }

    def generate_report(self, data: Dict[str, Any]) -> str:
        if not data.get("success"):
            return "Erreur d'analyse."
            
        lines = []
        lines.append("# RAPPORT CRM & SEGMENTATION CLIENT")
        lines.append(f"**Généré le** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Clients analysés** : {data['total_clients']:,}")
        lines.append("")
        
        seg = data["segments"]
        lines.append("## Synthèse des Segments")
        lines.append("| Segment | Nombre | Description |")
        lines.append("|---|---|---|")
        lines.append(f"| 👑 VIP | {len(seg['VIP'])} | Solde > 500k, Aucun prêt |")
        lines.append(f"| 💎 PREMIUM | {len(seg['PREMIUM'])} | Solde > 200k |")
        lines.append(f"| 👤 STANDARD | {len(seg['STANDARD'])} | Solde < 200k |")
        lines.append(f"| ⚠️ CHURN RISK | {len(seg['CHURN_RISK'])} | Comptes inactifs/faibles soldes |")
        lines.append(f"| 🚨 SURENDETTÉ | {len(seg['SURENDETTE'])} | Prêt > 1M et Solde < 100k |")
        lines.append("")
        
        # Actions recommandées
        lines.append("## Plan d'Action Marketing")
        if len(seg['VIP']) > 0:
            lines.append("### 👑 Clients VIP")
            lines.append("- **Action** : Leur proposer des produits d'investissement (Assurance-Vie, Bourse) et des cartes haut de gamme.")
            lines.append("- **Échantillon** :")
            for c in seg['VIP'][:3]:
                lines.append(f"  - Client `{c['customer_id'][:8]}` (Ville: {c['city']}) - Solde: {c['total_balance']:,.0f} FCFA")
            lines.append("")
            
        if len(seg['CHURN_RISK']) > 0:
            lines.append("### ⚠️ Clients à Risque de Départ (Churn)")
            lines.append("- **Action** : Campagne de réactivation, appels de courtoisie, offres promotionnelles personnalisées.")
            lines.append("- **Échantillon** :")
            for c in seg['CHURN_RISK'][:3]:
                lines.append(f"  - Client `{c['customer_id'][:8]}` (Ville: {c['city']}) - Solde: {c['total_balance']:,.0f} FCFA")
            lines.append("")

        if len(seg['SURENDETTE']) > 0:
            lines.append("### 🚨 Clients Surendettés")
            lines.append("- **Action** : Transférer au département Risques. Proposer un rééchelonnement de la dette.")
            lines.append("- **Échantillon** :")
            for c in seg['SURENDETTE'][:3]:
                lines.append(f"  - Client `{c['customer_id'][:8]}` (Ville: {c['city']}) - Prêt: {c['total_loan']:,.0f} FCFA")
            lines.append("")
            
        lines.append("---")
        lines.append("**Signé : KIBALI AGI EXPERT - Direction Marketing & CRM**")
        
        return "\n".join(lines)