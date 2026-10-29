# -*- coding: utf-8 -*-
"""
Module de détection et d'interface pour l'analyse de fichiers binaires.
Fait le pont entre l'application Streamlit (app.py) et le gestionnaire principal (binary_file_handler.py).
"""

import os
from enum import Enum
from pathlib import Path
from typing import Union, Dict, Any, Optional

# Importation des fonctions robustes que tu as développées
from binary_file_handler import BinaryFileHandler, BinaryFileViewer, analyze_binary_file

# Type d'entrée accepté : Chemin (str/Path), Bytes bruts, ou fichier Streamlit (UploadedFile)
FileSource = Union[str, Path, bytes, Any]

class FileStructureType(Enum):
    """Énumération des types de structure de fichier détectables."""
    TEXT = "text"
    BINARY = "binary"
    MIXED = "mixed"
    COMPRESSED = "compressed"
    IMAGE = "image"
    DOCUMENT = "document"
    UNKNOWN = "unknown"

class BinaryStructureDetector:
    """
    Détecteur de structure de fichiers binaires.
    Accepte un chemin de fichier, des bytes bruts ou un fichier uploadé via Streamlit.
    """
    
    def __init__(self, file_source: FileSource):
        self.file_source = file_source

    def _extract_data(self) -> tuple[bytes, str]:
        """
        Extrait les données brutes et le nom du fichier depuis la source fournie.
        """
        # Cas 1 : Fichier uploadé via Streamlit (UploadedFile)
        if hasattr(self.file_source, 'read') and hasattr(self.file_source, 'name'):
            data = self.file_source.read()
            self.file_source.seek(0)  # Remet le curseur au début pour les lectures futures
            return data, self.file_source.name
            
        # Cas 2 : Chemin de fichier (str ou pathlib.Path)
        elif isinstance(self.file_source, (str, Path)):
            path = Path(self.file_source)
            if path.exists() and path.is_file():
                with open(path, 'rb') as f:
                    return f.read(), path.name
            else:
                raise FileNotFoundError(f"Le fichier '{path}' n'existe pas.")
                
        # Cas 3 : Bytes bruts
        elif isinstance(self.file_source, bytes):
            return self.file_source, "unknown_binary.bin"
            
        # Cas 4 : Type non supporté
        else:
            raise TypeError(f"Type de source non supporté : {type(self.file_source)}. Attendu: str, Path, bytes ou UploadedFile.")

    def analyze(self) -> Dict[str, Any]:
        """
        Analyse le fichier et retourne un dictionnaire contenant les métadonnées.
        """
        try:
            data, filename = self._extract_data()
            info = BinaryFileHandler.identify_format(filename, data)
            
            # On ajoute un champ structure_type basé sur l'analyse
            if info.get('analysis', {}).get('header', {}).get('is_text_like'):
                info['structure_type'] = FileStructureType.TEXT.value
            elif info.get('extension') in ['.png', '.jpg', '.jpeg', '.gif', '.tiff']:
                info['structure_type'] = FileStructureType.IMAGE.value
            elif info.get('extension') in ['.pdf', '.docx', '.xlsx']:
                info['structure_type'] = FileStructureType.DOCUMENT.value
            elif info.get('analysis', {}).get('entropy', 0) > 7.5:
                info['structure_type'] = FileStructureType.COMPRESSED.value
            else:
                info['structure_type'] = FileStructureType.BINARY.value
                
            return info
        except Exception as e:
            return {"error": str(e), "structure_type": FileStructureType.UNKNOWN.value}

    def detect(self) -> Dict[str, Any]:
        """Alias de analyze() pour la compatibilité."""
        return self.analyze()


def detect_and_analyze_file(file_source: FileSource) -> str:
    """
    Fonction principale appelée par app.py.
    Génère et retourne le rapport d'analyse complet au format Markdown.
    """
    try:
        # 1. On extrait les données via notre classe détecteur
        detector = BinaryStructureDetector(file_source)
        data, filename = detector._extract_data()
        
        # 2. On génère le rapport Markdown complet en utilisant ta fonction robuste
        report = analyze_binary_file(data, filename)
        return report
        
    except FileNotFoundError as e:
        return f"❌ **Erreur de lecture** : {e}"
    except TypeError as e:
        return f"❌ **Erreur de type** : {e}"
    except Exception as e:
        return f"❌ **Erreur inattendue lors de l'analyse** : {e}\n\n*Vérifiez que le fichier est valide et non corrompu.*"