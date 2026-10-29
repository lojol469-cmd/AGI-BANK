# -*- coding: utf-8 -*-
from typing import Dict, Any
from datetime import datetime

def safe_float(val) -> float:
    try:
        return float(val)
    except:
        return 0.0

class NSIACustomsEngine:
    """
    Module 15 : Moteur Douanier (UEMOA / SYDONIA).
    Calcule les taxes à l'importation (DD, TVA, PCS, PC, RIF).
    """
    
    # Taux standards UEMOA
    TAUX_PCS = 0.01     # Prélèvement Communautaire de Solidarité (1%)
    TAUX_PC = 0.005     # Prélèvement Communautaire UEMOA (0.5%)
    TAUX_RIF = 0.0075   # Redevance Informatique (0.75%)
    TAUX_TVA = 0.18     # TVA (18%)

    def __init__(self):
        pass

    def calculate_landed_cost(self, fob: float, freight: float, insurance: float, category_rate: float) -> Dict[str, Any]:
        """
        Calcule le coût de revient complet d'une importation.
        fob: Valeur de la marchandise
        freight: Fret international
        insurance: Assurance
        category_rate: Taux de droit de douane (ex: 0.05, 0.10, 0.20, 0.35)
        """
        fob = safe_float(fob)
        freight = safe_float(freight)
        insurance = safe_float(insurance)
        dd_rate = safe_float(category_rate)

        # 1. Valeur CAF (Coût, Assurance, Fret) = Base taxable
        caf = fob + freight + insurance
        
        # 2. Droits de Douane (DD)
        dd = caf * dd_rate
        
        # 3. Prélèvements communautaires
        pcs = caf * self.TAUX_PCS
        pc = caf * self.TAUX_PC
        rif = caf * self.TAUX_RIF
        
        # 4. Base de la TVA (CAF + DD + PCS + PC + RIF)
        base_tva = caf + dd + pcs + pc + rif
        
        # 5. TVA
        tva = base_tva * self.TAUX_TVA
        
        # 6. Total des taxes
        total_taxes = dd + pcs + pc + rif + tva
        
        # 7. Coût total de l'importation
        cout_total = caf + total_taxes

        return {
            "fob": fob,
            "freight": freight,
            "insurance": insurance,
            "caf": caf,
            "dd": dd,
            "pcs": pcs,
            "pc": pc,
            "rif": rif,
            "tva": tva,
            "total_taxes": total_taxes,
            "cout_total": cout_total,
            "taux_dd": dd_rate
        }

    def generate_report(self, data: Dict[str, Any]) -> str:
        lines = []
        lines.append("# DÉCLARATION EN DOUANE ESTIMÉE (SYDONIA)")
        lines.append(f"**Généré le** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        lines.append("## 1. Valeur en Douane (Base CAF)")
        lines.append("| Élément | Montant (FCFA) |")
        lines.append("|---|---|")
        lines.append(f"| Valeur FOB | {data['fob']:,.2f} |")
        lines.append(f"| Fret International | {data['freight']:,.2f} |")
        lines.append(f"| Assurance | {data['insurance']:,.2f} |")
        lines.append(f"| **Valeur CAF Totale** | **{data['caf']:,.2f}** |")
        lines.append("")
        
        lines.append("## 2. Droits et Taxes exigibles")
        lines.append("| Taxe | Base de calcul | Taux | Montant (FCFA) |")
        lines.append("|---|---|---|---|")
        lines.append(f"| Droits de Douane (DD) | CAF | {data['taux_dd']*100:.0f}% | {data['dd']:,.2f} |")
        lines.append(f"| Prélèv. Communautaire (PCS) | CAF | 1% | {data['pcs']:,.2f} |")
        lines.append(f"| Prélèv. UEMOA (PC) | CAF | 0.5% | {data['pc']:,.2f} |")
        lines.append(f"| Redevance Info (RIF) | CAF | 0.75% | {data['rif']:,.2f} |")
        lines.append(f"| TVA à l'importation | CAF + Taxes | 18% | {data['tva']:,.2f} |")
        lines.append(f"| **TOTAL TAXES** | | | **{data['total_taxes']:,.2f}** |")
        lines.append("")
        
        lines.append("## 3. Coût Total de Revient")
        lines.append(f"### **{data['cout_total']:,.2f} FCFA**")
        lines.append("")
        
        lines.append("---")
        lines.append("**Signé : KIBALI AGI EXPERT - Déclarant en Douane**")
        
        return "\n".join(lines)