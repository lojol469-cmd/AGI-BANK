# -*- coding: utf-8 -*-
import math
from typing import Dict, Any, List
from datetime import datetime

def safe_float(val) -> float:
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except:
        return 0.0

class NSIAComplianceEngine:
    """
    Module 20 : Audit Interne & Conformité BCEAO.
    Vérifie les règles KYC, LBC/FT et les ratios prudentiels.
    """
    
    def __init__(self):
        pass

    def run_audit(self, accounts: List[Dict], customers: List[Dict]) -> Dict[str, Any]:
        anomalies = []
        conformite_kyc = 0
        total_clients = len(customers)
        
        # 1. Audit KYC (Know Your Customer)
        # On vérifie si les champs obligatoires sont remplis
        for c in customers:
            cid = c.get("customer_id")
            national_id = c.get("national_id")
            birth_date = c.get("birth_date")
            
            if not national_id or str(national_id).strip() == "" or str(national_id).lower() == "nan":
                anomalies.append({
                    "type": "KYC_MANQUANT",
                    "severity": "CRITIQUE",
                    "client": cid,
                    "detail": "Numéro d'identification national manquant.",
                    "reference": "Directive BCEAO n°02/2015 (LBC/FT)"
                })
            elif not birth_date or str(birth_date).lower() == "nan":
                anomalies.append({
                    "type": "KYC_MANQUANT",
                    "severity": "MOYENNE",
                    "client": cid,
                    "detail": "Date de naissance manquante.",
                    "reference": "Directive BCEAO n°02/2015"
                })
            else:
                conformite_kyc += 1

        # 2. Audit des Prêts (Risque de Crédit)
        for a in accounts:
            aid = a.get("account_id")
            loan = safe_float(a.get("loan_amount"))
            status = str(a.get("account_status", "")).lower()
            balance = safe_float(a.get("balance"))
            
            if loan > 0 and "inactive" in status:
                anomalies.append({
                    "type": "PRET_COMPTE_INACTIF",
                    "severity": "CRITIQUE",
                    "client": aid,
                    "detail": f"Prêt de {loan:,.0f} FCFA sur un compte inactif.",
                    "reference": "Normes prudentielles Bale II/III"
                })
            
            if balance < 0 and abs(balance) > 500000:
                anomalies.append({
                    "type": "DECOUVERT_NON_AUTORISE",
                    "severity": "ELEVEE",
                    "client": aid,
                    "detail": f"Découvert non autorisé de {abs(balance):,.0f} FCFA.",
                    "reference": "Instruction BCEAO n°008-2023"
                })

        # 3. Calcul du Score de Conformité (Base 100)
        # On pénalise selon la gravité
        score = 100
        for a in anomalies:
            if a["severity"] == "CRITIQUE": score -= 5
            elif a["severity"] == "ELEVEE": score -= 2
            elif a["severity"] == "MOYENNE": score -= 1
            
        score = max(0, score) # Éviter le négatif

        return {
            "success": True,
            "total_clients": total_clients,
            "conformite_kyc_pct": (conformite_kyc / total_clients * 100) if total_clients > 0 else 0,
            "score_global": score,
            "anomalies": anomalies,
            "nb_anomalies": len(anomalies)
        }

    def generate_report(self, data: Dict[str, Any]) -> str:
        lines = []
        lines.append("# RAPPORT D'AUDIT INTERNE & CONFORMITÉ BCEAO")
        lines.append(f"**Généré le** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        # --- SCORE GLOBAL ---
        score = data['score_global']
        if score >= 90: mention = "🟢 EXCELLENT"
        elif score >= 75: mention = "🟡 SATISFAISANT"
        elif score >= 50: mention = "🟠 A AMÉLIORER"
        else: mention = "🔴 CRITIQUE"
        
        lines.append("## 1. Synthèse de la Conformité")
        lines.append(f"### **Score Global : {score}/100** ({mention})")
        lines.append(f"- Conformité KYC : **{data['conformite_kyc_pct']:.1f}%** des clients")
        lines.append(f"- Anomalies détectées : **{data['nb_anomalies']}**")
        lines.append("")
        
        # --- DETAIL DES ANOMALIES ---
        if data['anomalies']:
            lines.append("## 2. Détail des Anomalies")
            lines.append("| Sévérité | Type | Client/Compte | Détail | Référence Légale |")
            lines.append("|---|---|---|---|---|")
            
            # Trier par sévérité
            sorted_anomalies = sorted(data['anomalies'], key=lambda x: (
                0 if x['severity'] == 'CRITIQUE' else 1 if x['severity'] == 'ELEVEE' else 2
            ))
            
            for a in sorted_anomalies[:20]: # Limiter à 20 pour l'affichage
                emoji = "🚨" if a['severity'] == 'CRITIQUE' else "⚠️" if a['severity'] == 'ELEVEE' else "🔸"
                lines.append(f"| {emoji} {a['severity']} | {a['type']} | {str(a['client'])[:8]} | {a['detail']} | {a['reference']} |")
            
            if len(data['anomalies']) > 20:
                lines.append(f"| ... | ... | ... | *et {len(data['anomalies']) - 20} autres anomalies* | ... |")
                
            lines.append("")
            lines.append("## 3. Recommandations de l'Auditeur")
            lines.append("1. **Régularisation KYC** : Geler les opérations des clients sans pièce d'identité jusqu'à régularisation.")
            lines.append("2. **Provisionnement** : Constituer des provisions pour les prêts sur comptes inactifs (risque de perte).")
            lines.append("3. **Recouvrement** : Lancer une procédure de recouvrement pour les découverts non autorisés > 500k.")
        else:
            lines.append("✅ **Aucune anomalie détectée. La banque est en parfaite conformité.**")
            
        lines.append("")
        lines.append("---")
        lines.append("**Signé : KIBALI AGI EXPERT - Inspecteur Général (Audit Interne)**")
        
        return "\n".join(lines)