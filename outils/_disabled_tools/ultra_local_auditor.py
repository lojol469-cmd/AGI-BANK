import os
import json
import pandas as pd
import pdfplumber
import re

try:
    from . import BaseTool
except ImportError:
    from outils import BaseTool

class UltraLocalAuditor(BaseTool):
    @property
    def name(self) -> str:
        return "ultra_local_auditor"

    @property
    def description(self) -> str:
        return (
            "🔒 OUTIL PREMIUM ULTRA (Audit Local).\n"
            "Permet de scanner un dossier local (Excel, PDF, CSV) et d'extraire les données financières propres.\n\n"
            "⚠️ MÉTHODE 1 (Recommandée) : Envoie un JSON.\n"
            "{\"path\": \"C:\\\\chemin\\\\vers\\\\dossier\"}\n\n"
            "🚨 MÉTHODE 2 (Si tu n'arrives pas à faire le JSON) : Copie-colle EXACTEMENT ce script Python et exécute-le avec l'outil `python_executor` :\n"
            "```python\n"
            "import os, pandas as pd, pdfplumber\n"
            "path = r'C:\\\\Users\\\\BELIKANON\\\\Desktop\\\\AUDIT VAALCO GABON 2005 & 2006'\n"
            "for root, _, files in os.walk(path):\n"
            "    for f in files:\n"
            "        fp = os.path.join(root, f)\n"
            "        if f.endswith(('.xls', '.xlsx', '.csv')):\n"
            "            try:\n"
            "                df = pd.read_excel(fp) if not f.endswith('.csv') else pd.read_csv(fp)\n"
            "                print(f'--- Fichier: {f} ---')\n"
            "                print(df.head(10).to_string())\n"
            "            except Exception as e: print(f'Erreur {f}: {e}')\n"
            "```\n"
            "Fais ça si tu bloques, ça marchera à 100%."
        )

    @property
    def capabilities(self) -> list:
        return ["audit", "dossier_local", "excel", "pdf", "premium"]

    def can_handle(self, query: str, context: dict = None) -> float:
        if any(kw in query.lower() for kw in ["dossier", "audit", "scan local", "fichier excel", "analyse dossier"]):
            return 0.99
        return 0.0

    def execute(self, query: str, context: dict = None) -> dict:
        print(f"💎 [Ultra Local Auditor] Requête reçue : {query}")
        
        # 🚀 Si l'IA a utilisé la Méthode 2 (Code Python)
        if "import os, pandas as pd" in query or "pdfplumber" in query and "for root" in query:
            try:
                import io
                import contextlib
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(query)
                return {"success": True, "output": output.getvalue()}
            except Exception as e:
                return {"success": False, "error": f"Erreur script Python : {e}"}

        # 📦 Si l'IA a utilisé la Méthode 1 (JSON)
        clean_input = query.replace("```json", "").replace("```", "").strip()
        try:
            payload = json.loads(clean_input)
        except json.JSONDecodeError:
            return {"success": False, "error": "❌ JSON invalide. Utilise la Méthode 2 (le script Python)."}

        path = payload.get("path")

        if not path or not os.path.exists(path):
            return {"success": False, "error": f"❌ Le chemin '{path}' n'existe pas."}

        print(f"   📂 Scan du dossier : {path}...")
        results = []

        for root, _, files in os.walk(path):
            for f in files:
                file_path = os.path.join(root, f)
                ext = f.lower().split('.')[-1]
                
                # 📄 Lire les PDFs
                if ext == 'pdf':
                    try:
                        with pdfplumber.open(file_path) as pdf:
                            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                            if text.strip():
                                results.append(f"📄 **Fichier PDF : {f}**\nExtrait : {text[:1000]}...\n")
                    except: pass
                
                # 📊 Lire les Excels/CSV
                elif ext in ['xlsx', 'xls', 'csv']:
                    try:
                        if ext == 'csv':
                            df = pd.read_csv(file_path, nrows=50)
                        else:
                            df = pd.read_excel(file_path, nrows=50)
                        
                        # 🧹 NETTOYAGE DES DONNÉES (On transforme le texte en nombres)
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                cleaned_series = df[col].astype(str).str.replace(',', '', regex=False)
                                cleaned_series = cleaned_series.str.replace(' ', '', regex=False)
                                cleaned_series = cleaned_series.str.replace('USD', '', regex=False)
                                cleaned_series = cleaned_series.str.replace('FCFA', '', regex=False)
                                try:
                                    df[col] = pd.to_numeric(cleaned_series, errors='ignore')
                                except: pass

                        summary = f"📊 **Fichier Excel/CSV : {f}**\n"
                        summary += f"Colonnes : {', '.join(df.columns.astype(str))}\n"
                        summary += f"Aperçu (3 lignes) :\n{df.head(3).to_string()}\n"
                        results.append(summary)
                    except Exception as e:
                        results.append(f"⚠️ Impossible de lire {f} : {e}\n")

        if not results:
            return {"success": False, "error": "❌ Aucun fichier lisible (PDF/Excel) trouvé."}

        output_text = "✅ **Résultat de l'audit du dossier :**\n\n" + "\n---\n".join(results)
        return {"success": True, "output": output_text}