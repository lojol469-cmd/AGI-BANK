# -*- coding: utf-8 -*-
import math
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

def safe_float(val) -> float:
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except:
        return 0.0

class NSIAFraudDetector:
    """
    Module 16 : Audit Trail & Détection de Fraude.
    Analyse les transactions pour détecter des anomalies comportementales.
    """
    
    def __init__(self):
        self.alerts = []

    def analyze_transactions(self, transactions: List[Dict], amount_threshold: float = 500000) -> Dict[str, Any]:
        """
        Analyse un lot de transactions et détecte les fraudes.
        """
        if not transactions:
            return {"success": False, "error": "Aucune transaction à analyser."}
            
        df = pd.DataFrame(transactions)
        self.alerts = []
        
        # Sécurité : conversion des types
        df['amount'] = df['amount'].apply(safe_float)
        df['created_time'] = pd.to_datetime(df['created_time'], errors='coerce')
        
        # --- REGLE 1 : MONTANTS SUSPECTS (Seuil élevé) ---
        high_amounts = df[df['amount'] > amount_threshold]
        for _, row in high_amounts.iterrows():
            self.alerts.append({
                "type": "MONTANT_ELEVE",
                "severity": "ELEVEE",
                "transaction_id": row.get('transaction_id', 'N/A'),
                "account_id": row.get('account_id', 'N/A'),
                "amount": row['amount'],
                "reason": f"Montant supérieur au seuil de {amount_threshold:,.0f} FCFA"
            })
            
        # --- REGLE 2 : TRANSACTIONS NOCTURNES (Entre 00h et 05h) ---
        # On s'assure que les dates sont valides
        valid_dates = df.dropna(subset=['created_time'])
        night_tx = valid_dates[valid_dates['created_time'].dt.hour.isin([0, 1, 2, 3, 4, 5])]
        
        # On limite aux 50 plus grosses transactions nocturnes pour ne pas surcharger le rapport
        night_tx = night_tx.nlargest(50, 'amount')
        for _, row in night_tx.iterrows():
            self.alerts.append({
                "type": "TRANSACTION_NOCTURNE",
                "severity": "MOYENNE",
                "transaction_id": row.get('transaction_id', 'N/A'),
                "account_id": row.get('account_id', 'N/A'),
                "amount": row['amount'],
                "reason": f"Transaction effectuée à {row['created_time'].strftime('%H:%M')} (Heures indues)"
            })
            
        # --- REGLE 3 : MONTANTS RONDS (Ex: 1 000 000, 500 000) ---
        # Souvent utilisé pour le blanchiment ou les pots-de-vin
        round_tx = df[(df['amount'] >= 100000) & (df['amount'] % 100000 == 0)]
        for _, row in round_tx.head(20).iterrows():
            self.alerts.append({
                "type": "MONTANT_ROND",
                "severity": "FAIBLE",
                "transaction_id": row.get('transaction_id', 'N/A'),
                "account_id": row.get('account_id', 'N/A'),
                "amount": row['amount'],
                "reason": "Montant exact (multiple de 100 000)"
            })
            
        # --- REGLE 4 : VELOCITE (Plus de 5 transactions le même jour sur le même compte) ---
        if 'created_time' in valid_dates.columns:
            valid_dates['date_only'] = valid_dates['created_time'].dt.date
            velocity = valid_dates.groupby(['account_id', 'date_only']).size().reset_index(name='count')
            high_velocity = velocity[velocity['count'] > 5]
            
            for _, row in high_velocity.head(20).iterrows():
                self.alerts.append({
                    "type": "VELOCITE_ANORMALE",
                    "severity": "ELEVEE",
                    "transaction_id": "MULTIPLE",
                    "account_id": row['account_id'],
                    "amount": 0,
                    "reason": f"{row['count']} transactions effectuées le {row['date_only']}"
                })

        return self.generate_report(len(transactions))
        
    def generate_report(self, total_analyzed: int) -> Dict[str, Any]:
        lines = []
        lines.append("# RAPPORT D'AUDIT ET DÉTECTION DE FRAUDE")
        lines.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Transactions analysées** : {total_analyzed:,}")
        lines.append(f"**Alertes détectées** : {len(self.alerts)}")
        lines.append("")
        
        if not self.alerts:
            lines.append("✅ **Aucune anomalie détectée.** Le trafic est conforme.")
            return {"success": True, "output": "\n".join(lines), "data": []}
            
        # Grouper par sévérité
        sev_haute = [a for a in self.alerts if a['severity'] == 'ELEVEE']
        sev_moy = [a for a in self.alerts if a['severity'] == 'MOYENNE']
        
        lines.append("## 🚨 Alertes de Haute Sévérité")
        lines.append("| Type | Compte | Montant | Motif |")
        lines.append("|---|---|---|---|")
        for a in sev_haute[:15]:
            lines.append(f"| {a['type']} | {str(a['account_id'])[:8]} | {a['amount']:,.0f} | {a['reason']} |")
        lines.append("")
        
        if sev_moy:
            lines.append("## ⚠️ Alertes de Sévérité Moyenne")
            lines.append("| Type | Compte | Montant | Motif |")
            lines.append("|---|---|---|---|")
            for a in sev_moy[:10]:
                lines.append(f"| {a['type']} | {str(a['account_id'])[:8]} | {a['amount']:,.0f} | {a['reason']} |")
            lines.append("")
            
        lines.append("---")
        lines.append("**Signé : KIBALI AGI EXPERT - Conformité & LBC/FT**")
        
        return {"success": True, "output": "\n".join(lines), "data": self.alerts}