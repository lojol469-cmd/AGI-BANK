# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# Legal knowledge base (BCEAO / UEMOA) - enrichissable
LEGAL_KB = {
    "decouvert": {
        "reference": "Instruction BCEAO n°008-2023 (KYC & surveillance)",
        "regle": "Tout decouvert non autorise doit etre regularise sous 30 jours.",
        "risque": "ELEVE"
    },
    "taux_usure": {
        "reference": "Loi UEMOA n°2010-10 (taux d'usure)",
        "regle": "Taux d'interet superieur a 24% constitue une usure.",
        "risque": "CRITIQUE"
    },
    "kyc": {
        "reference": "Directive BCEAO n°02/2015 (LBC/FT)",
        "regle": "Identification client obligatoire avant toute operation.",
        "risque": "CRITIQUE"
    },
    "credit": {
        "reference": "Circulaire BCEAO n°008-2019 (octroi de credit)",
        "regle": "Verification de la capacite de remboursement obligatoire.",
        "risque": "MOYEN"
    }
}

class NSIAReportGenerator:
    """Module 4 : Generation d'alertes et de rapports PDF/Excel signes KIBALI."""
    
    def __init__(self, output_dir: str = r"D:\kibali-IA\outputs\reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.alerts = []
    
    # ---------- DETECTION D'ALERTES ----------
    def detect_alerts(self, accounts: List[Dict], thresholds: Dict = None) -> List[Dict]:
        """Detecte les alertes sur les comptes."""
        if thresholds is None:
            thresholds = {"pret_max": 500000, "solde_min": 0, "taux_max": 0.24}
        
        self.alerts = []
        
        for acc in accounts:
            aid = acc.get("account_id", "N/A")
            balance = acc.get("balance", 0) or 0
            loan = acc.get("loan_amount", 0) or 0
            rate = acc.get("interest_rate", 0) or 0
            status = acc.get("account_status", "")
            
            # ALERTE 1 : Decouvert non autorise
            if balance < thresholds["solde_min"]:
                self.alerts.append({
                    "niveau": "CRITIQUE",
                    "type": "DECOUVERT",
                    "account_id": aid,
                    "valeur": balance,
                    "legal_ref": LEGAL_KB["decouvert"]["reference"],
                    "regle": LEGAL_KB["decouvert"]["regle"],
                    "action": "Contacter le client sous 48h"
                })
            
            # ALERTE 2 : Pret excessif
            if loan > thresholds["pret_max"]:
                self.alerts.append({
                    "niveau": "ELEVE",
                    "type": "PRET_EXCESSIF",
                    "account_id": aid,
                    "valeur": loan,
                    "legal_ref": LEGAL_KB["credit"]["reference"],
                    "regle": LEGAL_KB["credit"]["regle"],
                    "action": "Reevaluation de la capacite de remboursement"
                })
            
            # ALERTE 3 : Taux d'usure
            if rate > thresholds["taux_max"]:
                self.alerts.append({
                    "niveau": "CRITIQUE",
                    "type": "TAUX_USURE",
                    "account_id": aid,
                    "valeur": rate,
                    "legal_ref": LEGAL_KB["taux_usure"]["reference"],
                    "regle": LEGAL_KB["taux_usure"]["regle"],
                    "action": "Requalification juridique immediate"
                })
            
            # ALERTE 4 : Compte ferme avec solde
            if status and "closed" in str(status).lower() and balance != 0:
                self.alerts.append({
                    "niveau": "MOYEN",
                    "type": "COMPTE_FERME_ACTIF",
                    "account_id": aid,
                    "valeur": balance,
                    "legal_ref": "Procedure interne NSIA",
                    "regle": "Un compte ferme ne doit pas avoir de solde.",
                    "action": "Verification manuelle"
                })
        
        print(f"[ALERTES] {len(self.alerts)} alertes detectees sur {len(accounts)} comptes.")
        return self.alerts
    
    # ---------- RAPPORT TEXTE (Markdown) ----------
    def generate_markdown(self, query: str, results: Dict, alerts: List[Dict]) -> str:
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        md = []
        md.append("# RAPPORT D'AUDIT NSIA")
        md.append(f"**Genere le** : {today}")
        md.append(f"**Requete** : {query}")
        md.append(f"**Moteur** : {results.get('engine', 'N/A')}")
        md.append("")
        
        md.append("## 1. Synthese")
        md.append(f"- Comptes analyses : **{results.get('count', 0)}**")
        md.append(f"- Alertes detectees : **{len(alerts)}**")
        md.append("")
        
        if alerts:
            md.append("## 2. Alertes & Croisement Legal")
            md.append("| Niveau | Type | Compte | Valeur | Reference legale | Action |")
            md.append("|---|---|---|---|---|---|")
            for a in alerts[:20]:
                md.append(f"| {a['niveau']} | {a['type']} | {a['account_id'][:8]}... | {a['valeur']} | {a['legal_ref']} | {a['action']} |")
            md.append("")
        
        md.append("## 3. Echantillon des comptes")
        md.append("| Compte | Solde | Pret | Statut |")
        md.append("|---|---|---|---|")
        for r in results.get("results", [])[:10]:
            md.append(f"| {str(r.get('account_id',''))[:8]}... | {r.get('balance')} | {r.get('loan_amount', 0)} | {r.get('account_status', '')} |")
        md.append("")
        
        md.append("---")
        md.append("**Signe : KIBALI AGI EXPERT**")
        md.append(f"*Rapport genere automatiquement - {today}*")
        return "\n".join(md)
    
    # ---------- RAPPORT PDF ----------
    def generate_pdf(self, query: str, results: Dict, alerts: List[Dict], filename: str = None) -> str:
        try:
            from fpdf import FPDF
        except ImportError:
            print("[PDF] fpdf2 non installe - pip install fpdf2")
            return ""
        
        if not filename:
            filename = f"rapport_nsia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 10, "RAPPORT D'AUDIT NSIA", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Genere le {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "1. Synthese", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Requete : {query[:80]}", ln=True)
        pdf.cell(0, 6, f"Moteur : {results.get('engine', 'N/A')}", ln=True)
        pdf.cell(0, 6, f"Comptes analyses : {results.get('count', 0)}", ln=True)
        pdf.cell(0, 6, f"Alertes : {len(alerts)}", ln=True)
        pdf.ln(5)
        
        if alerts:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "2. Alertes", ln=True)
            pdf.set_font("Helvetica", "", 9)
            for a in alerts[:15]:
                txt = f"[{a['niveau']}] {a['type']} - Compte {str(a['account_id'])[:8]}... - Ref: {a['legal_ref']}"
                pdf.multi_cell(0, 5, txt)
                pdf.ln(1)
            pdf.ln(3)
        
        pdf.set_y(-30)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Signe : KIBALI AGI EXPERT", ln=True, align="R")
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, "Document automatique - verification humaine recommandee", ln=True, align="R")
        
        pdf.output(filepath)
        print(f"[PDF] Rapport genere : {filepath}")
        return filepath
    
    # ---------- RAPPORT EXCEL ----------
    def generate_excel(self, results: Dict, alerts: List[Dict], filename: str = None) -> str:
        try:
            import pandas as pd
        except ImportError:
            return ""
        
        if not filename:
            filename = f"rapport_nsia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Onglet 1 : Resultats
            df_results = pd.DataFrame(results.get("results", []))
            if not df_results.empty:
                df_results.to_excel(writer, sheet_name="Resultats", index=False)
            
            # Onglet 2 : Alertes
            df_alerts = pd.DataFrame(alerts)
            if not df_alerts.empty:
                df_alerts.to_excel(writer, sheet_name="Alertes", index=False)
            
            # Onglet 3 : Meta
            meta = pd.DataFrame([{
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Requete": results.get("query", ""),
                "Moteur": results.get("engine", ""),
                "Nb resultats": results.get("count", 0),
                "Nb alertes": len(alerts),
                "Signature": "KIBALI AGI EXPERT"
            }])
            meta.to_excel(writer, sheet_name="Meta", index=False)
        
        print(f"[EXCEL] Rapport genere : {filepath}")
        return filepath
    
    # ---------- PIPELINE COMPLET ----------
    def generate_full_report(self, query: str, results: Dict) -> Dict[str, str]:
        """Genere PDF + Excel + Markdown en un seul appel."""
        accounts = results.get("results", [])
        alerts = self.detect_alerts(accounts)
        
        md = self.generate_markdown(query, results, alerts)
        md_path = os.path.join(self.output_dir, f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        
        pdf_path = self.generate_pdf(query, results, alerts)
        excel_path = self.generate_excel(results, alerts)
        
        return {
            "markdown": md_path,
            "pdf": pdf_path,
            "excel": excel_path,
            "alertes_count": len(alerts),
            "alerts": alerts
        }