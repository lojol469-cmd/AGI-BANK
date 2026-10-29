# -*- coding: utf-8 -*-
from typing import List, Dict, Any
from datetime import datetime

class NSIAInventoryEngine:
    """
    Module 14 : Gestion des Stocks & Inventaire.
    Gère les entrées/sorties, valorisation (CUMP) et alertes de rupture.
    """
    
    def __init__(self):
        # Base de données en mémoire (simulée)
        self.products = {}

    def add_product(self, sku: str, name: str, initial_qty: float, unit_cost: float) -> Dict[str, Any]:
        """Ajoute un produit au catalogue."""
        if sku in self.products:
            return {"success": False, "error": f"Le produit {sku} existe déjà."}
            
        self.products[sku] = {
            "sku": sku,
            "name": name,
            "qty": initial_qty,
            "unit_cost": unit_cost,
            "total_value": initial_qty * unit_cost,
            "movements": [{"type": "INIT", "qty": initial_qty, "date": datetime.now().strftime("%Y-%m-%d")}]
        }
        return {"success": True, "message": f"Produit {name} ajouté."}

    def record_movement(self, sku: str, movement_type: str, qty: float, unit_cost: float = None) -> Dict[str, Any]:
        """
        Enregistre une entrée ou sortie de stock et recalcule le CUMP (Coût Unitaire Moyen Pondéré).
        movement_type: 'IN' (Entrée) ou 'OUT' (Sortie)
        """
        if sku not in self.products:
            return {"success": False, "error": "Produit introuvable."}

        product = self.products[sku]
        qty = float(qty)
        
        if movement_type == "OUT" and qty > product["qty"]:
            return {"success": False, "error": f"Stock insuffisant pour {sku}. Disponible: {product['qty']}"}

        if movement_type == "IN":
            if unit_cost is None:
                return {"success": False, "error": "Le coût unitaire est requis pour une entrée."}
            # Recalcul du CUMP
            total_value = product["total_value"] + (qty * unit_cost)
            new_qty = product["qty"] + qty
            product["unit_cost"] = total_value / new_qty if new_qty > 0 else 0
            product["qty"] = new_qty
            product["total_value"] = total_value
            
        elif movement_type == "OUT":
            # Sortie valorisée au CUMP
            product["qty"] -= qty
            product["total_value"] = product["qty"] * product["unit_cost"]

        # Historique
        product["movements"].append({
            "type": movement_type,
            "qty": qty,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "unit_cost": product["unit_cost"]
        })

        return {"success": True, "product": product}

    def check_alerts(self, threshold: float = 10.0) -> List[Dict]:
        """Détecte les produits en rupture ou sous le seuil d'alerte."""
        alerts = []
        for sku, p in self.products.items():
            if p["qty"] <= 0:
                alerts.append({"sku": sku, "name": p["name"], "status": "RUPTURE", "qty": p["qty"]})
            elif p["qty"] <= threshold:
                alerts.append({"sku": sku, "name": p["name"], "status": "ALERTE", "qty": p["qty"]})
        return alerts

    def generate_report(self) -> Dict[str, Any]:
        lines = []
        lines.append("# RAPPORT D'INVENTAIRE")
        lines.append(f"**Généré le** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        total_value = sum(p["total_value"] for p in self.products.values())
        lines.append(f"**Valeur totale du stock** : {total_value:,.2f} FCFA")
        lines.append("")
        
        lines.append("## État des Stocks (Méthode CUMP)")
        lines.append("| SKU | Produit | Quantité | Coût Unitaire | Valeur Totale |")
        lines.append("|---|---|---|---|---|")
        for sku, p in self.products.items():
            lines.append(f"| {sku} | {p['name']} | {p['qty']:,.2f} | {p['unit_cost']:,.2f} | {p['total_value']:,.2f} |")
        lines.append("")
        
        alerts = self.check_alerts()
        if alerts:
            lines.append("## ⚠️ Alertes de Stock")
            lines.append("| SKU | Produit | Statut | Quantité |")
            lines.append("|---|---|---|---|")
            for a in alerts:
                emoji = "🚨" if a["status"] == "RUPTURE" else "⚠️"
                lines.append(f"| {a['sku']} | {a['name']} | {emoji} {a['status']} | {a['qty']} |")
        else:
            lines.append("## ✅ Aucune alerte de stock.")
            
        lines.append("")
        lines.append("---")
        lines.append("**Signé : KIBALI AGI EXPERT - Supply Chain**")
        
        return {"success": True, "output": "\n".join(lines)}