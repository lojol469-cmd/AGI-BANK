# -*- coding: utf-8 -*-
import math
from typing import Dict, Any
from datetime import datetime

def safe_float(val) -> float:
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except:
        return 0.0

class NSIATaxEngine:
    """
    Module 11 : Moteur de calcul fiscal (TVA, IS, IRPP).
    Basé sur les règles de l'UEMOA / Sénégal (CGI).
    """
    
    # Taux standards UEMOA
    TAUX_TVA = 0.18           # 18%
    TAUX_IS = 0.25            # 25%
    
    # Tranches IRPP (Simulation simplifiée pour le Sénégal)
    TRANCHES_IRPP = [
        (0, 50000, 0.00),       # 0 à 50k : 0%
        (50000, 150000, 0.20),  # 50k à 150k : 20%
        (150000, 300000, 0.30), # 150k à 300k : 30%
        (300000, 500000, 0.35), # 300k à 500k : 35%
        (500000, float('inf'), 0.40) # > 500k : 40%
    ]

    def __init__(self):
        self.rapport_lignes = []
    
    # ---------- CALCUL TVA ----------
    def calculate_tva(self, chiffre_affaires: float, achats: float) -> Dict[str, Any]:
        """
        TVA Collectée = CA * 18%
        TVA Déductible = Achats * 18%
        TVA à Payer = Collectée - Déductible
        """
        ca = safe_float(chiffre_affaires)
        achats = safe_float(achats)
        
        tva_collectee = ca * self.TAUX_TVA
        tva_deductible = achats * self.TAUX_TVA
        tva_a_payer = tva_collectee - tva_deductible
        
        return {
            "chiffre_affaires": ca,
            "tva_collectee": tva_collectee,
            "achats": achats,
            "tva_deductible": tva_deductible,
            "tva_a_payer": max(0, tva_a_payer),
            "credit_tva": abs(min(0, tva_a_payer))
        }
    
    # ---------- CALCUL IS (Impôt sur les Sociétés) ----------
    def calculate_is(self, resultat_net: float, ca: float) -> Dict[str, Any]:
        """
        IS = Résultat Net * 25%
        Minimum perçu = 0.5% du CA (si bénéfice faible ou perte)
        """
        resultat = safe_float(resultat_net)
        ca = safe_float(ca)
        
        if resultat > 0:
            is_calcule = resultat * self.TAUX_IS
        else:
            is_calcule = 0
            
        is_minimum = ca * 0.005 # 0.5% du CA
        is_final = max(is_calcule, is_minimum)
        
        return {
            "resultat_net": resultat,
            "is_theorique": is_calcule,
            "is_minimum": is_minimum,
            "is_a_payer": is_final,
            "regime": "Bénéfice" if is_calcule > is_minimum else "Minimum fiscal"
        }
    
    # ---------- CALCUL IRPP ----------
    def calculate_irpp(self, salaire_brut: float) -> Dict[str, Any]:
        """
        Calcule l'IRPP selon un barème progressif.
        """
        salaire = safe_float(salaire_brut)
        reste = salaire
        impot_total = 0.0
        details = []
        
        for min_val, max_val, taux in self.TRANCHES_IRPP:
            if reste <= 0:
                break
                
            if salaire > min_val:
                # Montant imposable dans cette tranche
                montant_tranche = min(salaire, max_val) - min_val
                if montant_tranche > 0:
                    impot_tranche = montant_tranche * taux
                    impot_total += impot_tranche
                    details.append({
                        "tranche": f"{min_val:,.0f} - {max_val:,.0f}" if max_val != float('inf') else f"> {min_val:,.0f}",
                        "taux": f"{taux*100:.0f}%",
                        "impot": impot_tranche
                    })
        
        salaire_net = salaire - impot_total
        
        return {
            "salaire_brut": salaire,
            "irpp_total": impot_total,
            "salaire_net": salaire_net,
            "details": details
        }
    
    # ---------- RAPPORT GLOBAL ----------
    def generate_report(self, ca: float, achats: float, resultat: float) -> Dict[str, Any]:
        tva = self.calculate_tva(ca, achats)
        is_ = self.calculate_is(resultat, ca)
        
        lines = []
        lines.append("# DÉCLARATION FISCALE ESTIMÉE (UEMOA)")
        lines.append(f"**Généré le** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        lines.append("## 1. TVA (Taux 18%)")
        lines.append("| Élément | Base | Taux | Montant |")
        lines.append("|---|---|---|---|")
        lines.append(f"| TVA Collectée | {tva['chiffre_affaires']:,.2f} | 18% | {tva['tva_collectee']:,.2f} |")
        lines.append(f"| TVA Déductible | {tva['achats']:,.2f} | 18% | {tva['tva_deductible']:,.2f} |")
        if tva['tva_a_payer'] > 0:
            lines.append(f"| **TVA à Payer** | | | **{tva['tva_a_payer']:,.2f}** |")
        else:
            lines.append(f"| **Crédit de TVA** | | | **{tva['credit_tva']:,.2f}** |")
        lines.append("")
        
        lines.append("## 2. Impôt sur les Sociétés (IS)")
        lines.append("| Élément | Montant |")
        lines.append("|---|---|")
        lines.append(f"| Résultat Net | {is_['resultat_net']:,.2f} |")
        lines.append(f"| IS Théorique (25%) | {is_['is_theorique']:,.2f} |")
        lines.append(f"| IS Minimum (0.5% CA) | {is_['is_minimum']:,.2f} |")
        lines.append(f"| **IS à Payer** | **{is_['is_a_payer']:,.2f}** |")
        lines.append(f"| *Régime appliqué* | *{is_['regime']}* |")
        lines.append("")
        
        lines.append("---")
        lines.append("**Références légales** :")
        lines.append("- TVA : Article 351 du CGI Sénégalais (Taux 18%)")
        lines.append("- IS : Article 8 du CGI (Taux 25%, minimum 0.5% du CA)")
        lines.append("")
        lines.append("**Signé : KIBALI AGI EXPERT - Fiscaliste**")
        
        return {"success": True, "output": "\n".join(lines)}