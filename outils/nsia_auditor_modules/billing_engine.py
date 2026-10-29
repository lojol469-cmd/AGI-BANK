# -*- coding: utf-8 -*-
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any

def safe_float(val) -> float:
    try:
        return float(val)
    except:
        return 0.0

class NSIABillingEngine:
    """
    Module 13 : Facturation & Devis.
    Gère la création de factures, calculs de TVA, remises et échéances.
    """
    
    TAUX_TVA = 0.18 # 18% UEMOA

    def __init__(self, company_name: str = "NSIA BANQUE"):
        self.company_name = company_name

    def create_invoice(self, client_name: str, items: List[Dict], discount_pct: float = 0.0, is_quote: bool = False) -> Dict[str, Any]:
        """
        Crée une facture ou un devis.
        items: liste de dicts {"description": str, "quantity": float, "unit_price": float}
        """
        if not items:
            return {"success": False, "error": "Aucun article dans la facture."}

        # 1. Calcul des lignes
        lines = []
        subtotal = 0.0
        
        for item in items:
            qty = safe_float(item.get("quantity", 1))
            price = safe_float(item.get("unit_price", 0))
            total_line = qty * price
            subtotal += total_line
            lines.append({
                "description": item.get("description", "Article"),
                "quantity": qty,
                "unit_price": price,
                "total": total_line
            })

        # 2. Remise
        discount_amount = subtotal * (safe_float(discount_pct) / 100.0)
        subtotal_after_discount = subtotal - discount_amount

        # 3. TVA
        tva_amount = subtotal_after_discount * self.TAUX_TVA

        # 4. Total TTC
        total_ttc = subtotal_after_discount + tva_amount

        # 5. Métadonnées
        doc_type = "DEVIS" if is_quote else "FACTURE"
        doc_number = f"{doc_type[:3]}-{datetime.now().strftime('%Y%m')}-{str(uuid.uuid4())[:4].upper()}"
        date_emission = datetime.now()
        date_echeance = date_emission + timedelta(days=30) # 30 jours

        return {
            "success": True,
            "doc_type": doc_type,
            "doc_number": doc_number,
            "client": client_name,
            "date_emission": date_emission.strftime("%Y-%m-%d"),
            "date_echeance": date_echeance.strftime("%Y-%m-%d"),
            "lines": lines,
            "subtotal": subtotal,
            "discount_pct": safe_float(discount_pct),
            "discount_amount": discount_amount,
            "subtotal_after_discount": subtotal_after_discount,
            "tva_amount": tva_amount,
            "total_ttc": total_ttc,
            "is_quote": is_quote
        }

    def generate_markdown(self, invoice_data: Dict[str, Any]) -> str:
        if not invoice_data.get("success"):
            return "Erreur lors de la génération."

        lines = []
        lines.append(f"# {invoice_data['doc_type']} : {invoice_data['doc_number']}")
        lines.append(f"**Émetteur** : {self.company_name}")
        lines.append(f"**Client** : {invoice_data['client']}")
        lines.append(f"**Date d'émission** : {invoice_data['date_emission']}")
        
        if not invoice_data['is_quote']:
            lines.append(f"**Date d'échéance** : {invoice_data['date_echeance']}")
        
        lines.append("")
        lines.append("## Détail des prestations")
        lines.append("| Description | Qté | Prix Unitaire | Total |")
        lines.append("|---|---|---|---|")
        for l in invoice_data['lines']:
            lines.append(f"| {l['description']} | {l['quantity']} | {l['unit_price']:,.2f} | {l['total']:,.2f} |")
        lines.append("")
        
        lines.append("## Récapitulatif Financier")
        lines.append(f"- **Sous-total** : {invoice_data['subtotal']:,.2f} FCFA")
        if invoice_data['discount_amount'] > 0:
            lines.append(f"- **Remise ({invoice_data['discount_pct']}%)** : -{invoice_data['discount_amount']:,.2f} FCFA")
            lines.append(f"- **Net Commercial** : {invoice_data['subtotal_after_discount']:,.2f} FCFA")
        
        lines.append(f"- **TVA (18%)** : {invoice_data['tva_amount']:,.2f} FCFA")
        lines.append(f"### **TOTAL TTC : {invoice_data['total_ttc']:,.2f} FCFA**")
        lines.append("")
        
        if invoice_data['is_quote']:
            lines.append("*Ceci est un devis. Validité 30 jours.*")
        else:
            lines.append("*Paiement à 30 jours. Passé ce délai, des pénalités de retard s'appliquent.*")
            
        lines.append("")
        lines.append("---")
        lines.append(f"**{self.company_name} - Signé : KIBALI AGI EXPERT**")
        
        return "\n".join(lines)