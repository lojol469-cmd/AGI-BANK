"""
Outil d'Analyse d'Images
Analyse et décrit le contenu des images avec IA
"""

import re
import os
import time
from typing import Dict, List, Any, Optional
from .__init__ import BaseTool
import cv2
import numpy as np
from PIL import Image
import io

class ImageAnalysisTool(BaseTool):
    """Outil spécialisé dans l'analyse d'images"""

    @property
    def name(self) -> str:
        return "image_analysis"

    @property
    def description(self) -> str:
        return "Analyse intelligente d'images avec description et classification"

    @property
    def capabilities(self) -> List[str]:
        return [
            "analyse d'image",
            "description d'image",
            "détection d'objets",
            "classification d'images",
            "analyse de couleurs",
            "détection de texte",
            "analyse de composition"
        ]

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        """Détermine si la requête concerne l'analyse d'images"""
        image_keywords = [
            "image", "photo", "photo", "visuel", "graphique",
            "analyser", "décrire", "voir", "regarder",
            "couleur", "objet", "forme", "contour",
            "texte dans l'image", "lire l'image"
        ]

        query_lower = query.lower()

        # Mots-clés liés aux images
        score = 0.0
        for keyword in image_keywords:
            if keyword in query_lower:
                score += 0.4

        # Contexte avec image
        if context and 'image_path' in context:
            score += 0.8

        # Questions sur l'apparence visuelle
        visual_questions = ["à quoi ressemble", "quelle couleur", "qu'est-ce qu'on voit"]
        for question in visual_questions:
            if question in query_lower:
                score += 0.5

        return min(score, 1.0)

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyse une image"""
        try:
            image_path = None

            # Chercher le chemin de l'image dans le contexte
            if context and 'image_path' in context:
                image_path = context['image_path']
            elif context and 'uploaded_file' in context:
                # Sauvegarder temporairement le fichier uploadé
                uploaded_file = context['uploaded_file']
                image_path = f"/tmp/temp_image_{hash(query)}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

            if not image_path:
                return {
                    'error': 'Aucune image fournie pour l\'analyse',
                    'query': query
                }

            # Analyser l'image
            analysis = self._analyze_image(image_path, query)

            return {
                'query': query,
                'image_path': image_path,
                'analysis': analysis,
                'timestamp': time.time()
            }

        except Exception as e:
            return {
                'error': f"Erreur lors de l'analyse d'image: {str(e)}",
                'query': query
            }

    def _analyze_image(self, image_path: str, query: str) -> Dict[str, Any]:
        """Analyse détaillée de l'image"""
        try:
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Impossible de charger l\'image'}

            pil_image = Image.open(image_path)

            # Analyse de base
            height, width = image.shape[:2]
            analysis = {
                'dimensions': f"{width}x{height}",
                'format': pil_image.format,
                'mode': pil_image.mode,
                'size_bytes': os.path.getsize(image_path) if os.path.exists(image_path) else 0
            }

            # Analyse des couleurs
            analysis['color_analysis'] = self._analyze_colors(image)

            # Détection d'objets simples
            analysis['object_detection'] = self._detect_objects(image)

            # Analyse de la composition
            analysis['composition'] = self._analyze_composition(image)

            # Description basée sur la requête
            analysis['description'] = self._generate_description(query, analysis)

            return analysis

        except Exception as e:
            return {'error': f'Erreur analyse: {str(e)}'}

    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyse la palette de couleurs"""
        # Convertir en HSV pour une meilleure analyse
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Couleur dominante
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)

        # K-means pour trouver les couleurs dominantes
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_

        # Convertir en hex
        hex_colors = []
        for color in colors:
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(color[2]), int(color[1]), int(color[0])
            )
            hex_colors.append(hex_color)

        return {
            'dominant_colors': hex_colors,
            'brightness': float(np.mean(hsv[:, :, 2])),
            'saturation': float(np.mean(hsv[:, :, 1]))
        }

    def _detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """Détection simple d'objets"""
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Détection de contours
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Analyse des contours
        objects = []
        for i, contour in enumerate(contours[:10]):  # Limiter à 10 objets
            area = cv2.contourArea(contour)
            if area > 100:  # Filtrer les petits objets
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Classification simple basée sur le ratio
                if aspect_ratio > 2:
                    obj_type = "objet allongé"
                elif aspect_ratio < 0.5:
                    obj_type = "objet vertical"
                else:
                    obj_type = "objet compact"

                objects.append({
                    'id': i,
                    'position': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'type': obj_type
                })

        return {
            'total_objects': len(objects),
            'objects': objects
        }

    def _analyze_composition(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyse de la composition de l'image"""
        height, width = image.shape[:2]

        # Analyse de la luminosité
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness_variation = np.std(gray)

        # Analyse des contrastes
        contrast = np.max(gray) - np.min(gray)

        # Détection de lignes (règle des tiers approximative)
        lines = cv2.HoughLinesP(
            cv2.Canny(gray, 50, 150),
            1, np.pi/180, 100, minLineLength=100, maxLineGap=10
        )

        return {
            'brightness_variation': float(brightness_variation),
            'contrast': float(contrast),
            'aspect_ratio': width / height if height > 0 else 0,
            'lines_detected': len(lines) if lines is not None else 0
        }

    def _generate_description(self, query: str, analysis: Dict[str, Any]) -> str:
        """Génère une description basée sur la requête et l'analyse"""
        description = f"Image de dimensions {analysis.get('dimensions', 'inconnues')}."

        # Couleurs
        if 'color_analysis' in analysis:
            colors = analysis['color_analysis'].get('dominant_colors', [])
            if colors:
                description += f" Couleurs dominantes: {', '.join(colors[:3])}."

        # Objets
        if 'object_detection' in analysis:
            objects = analysis['object_detection'].get('objects', [])
            if objects:
                description += f" {len(objects)} objet(s) détecté(s)."

        # Composition
        if 'composition' in analysis:
            comp = analysis['composition']
            if comp.get('brightness_variation', 0) > 50:
                description += " Image avec beaucoup de contraste et de détails."
            else:
                description += " Image plutôt uniforme."

        return description