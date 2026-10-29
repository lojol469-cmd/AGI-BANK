# -*- coding: utf-8 -*-
import os
from typing import Dict, Any, List
import pandas as pd

from .extractor import NSIAExtractor


class NSIAAgencyBenchmark:
    """
    Module 7 : Benchmark inter-agences + detection d'anomalies.
    Compare les agences NSIA entre elles et identifie les outliers.
    """
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or r"D:\kibali-IA\outputs\datasets\nsia"
        self.accounts_df = None
        self.customers_df = None
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        extractor = NSIAExtractor()
        for key, fname in {
            "accounts": "bank_accounts.csv",
            "customers": "customer_profiles.csv"
        }.items():
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                res = extractor.extract(path)
                if res["success"]:
                    sheet = list(res["data"].keys())[0]
                    df = pd.DataFrame(res["data"][sheet])
                    if key == "accounts": self.accounts_df = df
                    else: self.customers_df = df
        self._loaded = True
    
    def _enrich_with_agency(self) -> pd.DataFrame:
        """
        Joint les comptes avec les clients pour recuperer la ville (proxy agence).
        """
        df = self.accounts_df.copy()
        if self.customers_df is not None and 'city' in self.customers_df.columns:
            merged = df.merge(
                self.customers_df[['customer_id', 'city']],
                on='customer_id',
                how='left'
            )
            merged['agency'] = merged['city'].fillna('Inconnue')
        else:
            merged['agency'] = 'Inconnue'
        return merged
    
    def compute_agency_stats(self) -> pd.DataFrame:
        """Statistiques par agence."""
        df = self._enrich_with_agency()
        
        stats = df.groupby('agency').agg(
            nb_comptes=('account_id', 'count'),
            solde_total=('balance', 'sum'),
            solde_moyen=('balance', 'mean'),
            pret_total=('loan_amount', 'sum'),
            pret_moyen=('loan_amount', 'mean'),
            taux_moyen=('interest_rate', 'mean'),
            nb_prets=('loan_amount', lambda x: (x.fillna(0) > 0).sum()),
            nb_decouverts=('balance', lambda x: (x < 0).sum()),
            nb_inactifs=('account_status', lambda x: (x.astype(str).str.lower().str.contains('inactive')).sum())
        ).reset_index()
        
        # Ratios derives
        stats['ratio_endettement'] = (stats['pret_total'] / (stats['pret_total'] + stats['solde_total'].clip(lower=0))) * 100
        stats['pct_decouverts'] = (stats['nb_decouverts'] / stats['nb_comptes']) * 100
        stats['pct_inactifs'] = (stats['nb_inactifs'] / stats['nb_comptes']) * 100
        
        return stats.sort_values('solde_total', ascending=False)
    
    def detect_outliers(self) -> List[Dict[str, Any]]:
        """Detecte les agences hors norme (anomalies)."""
        stats = self.compute_agency_stats()
        if stats.empty:
            return []
        
        outliers = []
        
        # Seuils (percentiles 10 et 90)
        seuil_solde_bas = stats['solde_moyen'].quantile(0.10)
        seuil_solde_haut = stats['solde_moyen'].quantile(0.90)
        seuil_endettement = stats['ratio_endettement'].quantile(0.90)
        seuil_inactifs = stats['pct_inactifs'].quantile(0.90)
        
        for _, row in stats.iterrows():
            agency = row['agency']
            
            if row['solde_moyen'] < seuil_solde_bas:
                outliers.append({
                    "agence": agency,
                    "type": "SOLDE_MOYEN_BAS",
                    "valeur": round(row['solde_moyen'], 2),
                    "seuil": round(seuil_solde_bas, 2),
                    "niveau": "MOYEN"
                })
            
            if row['ratio_endettement'] > seuil_endettement:
                outliers.append({
                    "agence": agency,
                    "type": "ENDETTEMENT_ELEVE",
                    "valeur": round(row['ratio_endettement'], 2),
                    "seuil": round(seuil_endettement, 2),
                    "niveau": "ELEVE"
                })
            
            if row['pct_inactifs'] > seuil_inactifs:
                outliers.append({
                    "agence": agency,
                    "type": "TROP_INACTIFS",
                    "valeur": round(row['pct_inactifs'], 2),
                    "seuil": round(seuil_inactifs, 2),
                    "niveau": "MOYEN"
                })
        
        return outliers
    
    def generate_report(self) -> Dict[str, Any]:
        """Genere le rapport de benchmark complet."""
        self.load()
        stats = self.compute_agency_stats()
        outliers = self.detect_outliers()
        
        if stats.empty:
            return {"success": False, "output": "Aucune agence detectee (donnees manquantes)."}
        
        lines = ["# BENCHMARK INTER-AGENCES NSIA", ""]
        lines.append(f"**Nombre d'agences** : {len(stats)}")
        lines.append(f"**Total comptes** : {stats['nb_comptes'].sum():,}")
        lines.append(f"**Total soldes** : {stats['solde_total'].sum():,.2f}")
        lines.append(f"**Total prets** : {stats['pret_total'].sum():,.2f}")
        lines.append("")
        
        lines.append("## Classement des agences (par solde total)")
        lines.append("| Rang | Agence | Comptes | Solde Total | Solde Moyen | Pret Total | Endettement % | Inactifs % |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, (_, row) in enumerate(stats.iterrows(), 1):
            lines.append(
                f"| {i} | {row['agency']} | {int(row['nb_comptes'])} | "
                f"{row['solde_total']:,.0f} | {row['solde_moyen']:,.0f} | "
                f"{row['pret_total']:,.0f} | {row['ratio_endettement']:.1f} | "
                f"{row['pct_inactifs']:.1f} |"
            )
        lines.append("")
        
        if outliers:
            lines.append(f"## Anomalies detectees ({len(outliers)})")
            lines.append("| Agence | Type | Valeur | Seuil | Niveau |")
            lines.append("|---|---|---|---|---|")
            for o in outliers:
                lines.append(
                    f"| {o['agence']} | {o['type']} | {o['valeur']} | "
                    f"{o['seuil']} | {o['niveau']} |"
                )
            lines.append("")
        
        # Top 3 et Bottom 3
        top3 = stats.head(3)
        bottom3 = stats.tail(3)
        lines.append("## Top 3 (meilleures agences)")
        for _, row in top3.iterrows():
            lines.append(f"- **{row['agency']}** : {row['solde_total']:,.0f} (endettement {row['ratio_endettement']:.1f}%)")
        lines.append("")
        lines.append("## Bottom 3 (agences a surveiller)")
        for _, row in bottom3.iterrows():
            lines.append(f"- **{row['agency']}** : {row['solde_total']:,.0f} (inactifs {row['pct_inactifs']:.1f}%)")
        lines.append("")
        
        lines.append("---")
        lines.append("**Signe : KIBALI AGI EXPERT**")
        
        return {
            "success": True,
            "output": "\n".join(lines),
            "data": {
                "stats": stats.to_dict(orient='records'),
                "outliers": outliers,
                "nb_agences": len(stats)
            }
        }