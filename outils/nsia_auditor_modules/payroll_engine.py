# -*- coding: utf-8 -*-
from typing import Dict, Any
from datetime import datetime

def safe_float(val) -> float:
    try:
        return float(val)
    except:
        return 0.0

class NSIAPayrollEngine:
    """
    Module 12 : Moteur de Paie (Sénégal / UEMOA).
    Gère les cotisations sociales (IPRES, CSS) et l'impôt (IRPP).
    """
    
    # Taux de cotisations (Simulation Sénégal)
    TAUX_IPRES_SALARIE = 0.056  # 5.6%
    TAUX_IPRES_PATRONAL = 0.084 # 8.4%
    TAUX_CSS_SALARIE = 0.03     # 3% (Maladie, etc.)
    TAUX_CSS_PATRONAL = 0.08    # 8% (Prestations familiales, accidents)
    
    # Barème IRPP (Mensuel)
    TRANCHES_IRPP = [
        (0, 50000, 0.00),
        (50000, 150000, 0.20),
        (150000, 300000, 0.30),
        (300000, 500000, 0.35),
        (500000, float('inf'), 0.40)
    ]

    def __init__(self):
        pass
    
    def _calculate_irpp(self, base_imposable: float) -> float:
        impot_total = 0.0
        for min_val, max_val, taux in self.TRANCHES_IRPP:
            if base_imposable > min_val:
                montant_tranche = min(base_imposable, max_val) - min_val
                impot_total += montant_tranche * taux
        return impot_total

    def calculate_payslip(self, salaire_brut: float, primes: float = 0.0) -> Dict[str, Any]:
        """
        Calcule un bulletin de paie complet avec charges salariales et patronales.
        """
        brut = safe_float(salaire_brut)
        primes = safe_float(primes)
        
        # 1. Salaire Brut Total
        salaire_brut_total = brut + primes
        
        # 2. Cotisations Salariales (Déduites du salaire)
        ipres_salarie = salaire_brut_total * self.TAUX_IPRES_SALARIE
        css_salarie = salaire_brut_total * self.TAUX_CSS_SALARIE
        
        # 3. Calcul de l'IRPP
        # Base imposable = Brut - Cotisations sociales
        base_imposable = salaire_brut_total - ipres_salarie - css_salarie
        irpp = self._calculate_irpp(base_imposable)
        
        # 4. Salaire Net à payer
        total_retenues = ipres_salarie + css_salarie + irpp
        salaire_net = salaire_brut_total - total_retenues
        
        # 5. Charges Patronales (Coût pour l'entreprise)
        ipres_patronal = salaire_brut_total * self.TAUX_IPRES_PATRONAL
        css_patronal = salaire_brut_total * self.TAUX_CSS_PATRONAL
        cout_total_employeur = salaire_brut_total + ipres_patronal + css_patronal
        
        return {
            "salaire_brut": salaire_brut_total,
            "ipres_salarie": ipres_salarie,
            "css_salarie": css_salarie,
            "irpp": irpp,
            "salaire_net": salaire_net,
            "total_retenues": total_retenues,
            "ipres_patronal": ipres_patronal,
            "css_patronal": css_patronal,
            "cout_total_employeur": cout_total_employeur
        }

    def generate_report(self, salaire_brut: float, primes: float = 0.0) -> Dict[str, Any]:
        data = self.calculate_payslip(salaire_brut, primes)
        
        lines = []
        lines.append("# BULLETIN DE PAIE")
        lines.append(f"**Généré le** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        lines.append("## 1. Éléments du Salaire")
        lines.append("| Libellé | Base | Montant |")
        lines.append("|---|---|---|")
        lines.append(f"| Salaire de base | | {salaire_brut:,.2f} |")
        lines.append(f"| Primes | | {primes:,.2f} |")
        lines.append(f"| **Salaire Brut** | | **{data['salaire_brut']:,.2f}** |")
        lines.append("")
        
        lines.append("## 2. Retenues Salariales")
        lines.append("| Libellé | Taux | Montant |")
        lines.append("|---|---|---|")
        lines.append(f"| IPRES (Salarié) | 5.60% | {data['ipres_salarie']:,.2f} |")
        lines.append(f"| CSS (Salarié) | 3.00% | {data['css_salarie']:,.2f} |")
        lines.append(f"| IRPP | Progressif | {data['irpp']:,.2f} |")
        lines.append(f"| **Total Retenues** | | **{data['total_retenues']:,.2f}** |")
        lines.append("")
        
        lines.append("## 3. Net à Payer")
        lines.append(f"### **{data['salaire_net']:,.2f} FCFA**")
        lines.append("")
        
        lines.append("## 4. Charges Patronales (Information)")
        lines.append("| Libellé | Taux | Montant |")
        lines.append("|---|---|---|")
        lines.append(f"| IPRES (Patronal) | 8.40% | {data['ipres_patronal']:,.2f} |")
        lines.append(f"| CSS (Patronal) | 8.00% | {data['css_patronal']:,.2f} |")
        lines.append(f"| **Coût Total Employeur** | | **{data['cout_total_employeur']:,.2f}** |")
        lines.append("")
        
        lines.append("---")
        lines.append("**Signé : KIBALI AGI EXPERT - Gestionnaire de Paie**")
        
        return {"success": True, "output": "\n".join(lines), "data": data}