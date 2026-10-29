# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
import pdfplumber
from typing import Dict, Any, List

class NSIAExtractor:
    """
    Module 1 : Ingestion et normalisation des données.
    Gère les fichiers massifs (Excel, CSV, JSON, PDF, TXT).
    """
    
    def __init__(self):
        self.supported_extensions = ['.csv', '.xlsx', '.xls', '.json', '.txt', '.pdf']

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extrait les données de n'importe quel fichier supporté et les normalise.
        """
        if not os.path.exists(file_path):
            return {"success": False, "error": f"Fichier introuvable : {file_path}"}

        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.supported_extensions:
            return {"success": False, "error": f"Format non supporté : {ext}"}

        try:
            if ext in ['.xlsx', '.xls']:
                return self._extract_excel(file_path)
            elif ext == '.csv':
                return self._extract_csv(file_path)
            elif ext == '.json':
                return self._extract_json(file_path)
            elif ext == '.pdf':
                return self._extract_pdf(file_path)
            elif ext == '.txt':
                return self._extract_txt(file_path)
        except Exception as e:
            return {"success": False, "error": f"Erreur lors de l'extraction : {str(e)}"}

    def _extract_excel(self, file_path: str) -> Dict[str, Any]:
        # On lit tout le classeur, peu importe le nombre d'onglets
        xls = pd.ExcelFile(file_path)
        all_sheets = {}
        total_rows = 0
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            # Nettoyage basique : on supprime les lignes vides
            df = df.dropna(how='all')
            all_sheets[sheet_name] = df.to_dict(orient='records')
            total_rows += len(df)
            
        return {
            "success": True, 
            "format": "excel", 
            "total_rows": total_rows,
            "data": all_sheets,
            "metadata": {"sheets": xls.sheet_names}
        }

    def _extract_csv(self, file_path: str) -> Dict[str, Any]:
        # Détection automatique du séparateur (virgule, point-virgule, tabulation)
        try:
            df = pd.read_csv(file_path, sep=None, engine='python')
        except:
            df = pd.read_csv(file_path, encoding='latin-1', sep=None, engine='python')
            
        df = df.dropna(how='all')
        return {
            "success": True,
            "format": "csv",
            "total_rows": len(df),
            "data": {"default": df.to_dict(orient='records')},
            "metadata": {"columns": list(df.columns)}
        }

    def _extract_json(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Normalisation si c'est une liste ou un dict
        if isinstance(data, list):
            df = pd.DataFrame(data)
            return {"success": True, "format": "json", "total_rows": len(df), "data": {"default": df.to_dict(orient='records')}}
        return {"success": True, "format": "json", "total_rows": 1, "data": {"default": data}}

    def _extract_pdf(self, file_path: str) -> Dict[str, Any]:
        # Extraction du texte brut pour le RAG (les relevés de compte scannés)
        text_content = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
        
        full_text = "\n".join(text_content)
        return {
            "success": True,
            "format": "pdf",
            "total_rows": len(text_content), # Nombre de pages
            "data": {"raw_text": full_text},
            "metadata": {"pages": len(text_content)}
        }

    def _extract_txt(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return {"success": True, "format": "txt", "total_rows": 1, "data": {"raw_text": content}}