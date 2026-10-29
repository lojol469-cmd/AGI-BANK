"""
Classificateur intelligent de photos pour photogrammétrie
Utilise Vision AI (CLIP) pour analyser et ordonner les photos par similarité visuelle
"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from PIL import Image

class IntelligentPhotoClassifier:
    """
    Classe pour classifier intelligemment les photos de photogrammétrie
    en analysant les textures, détails visuels et angles de vue avec l'IA
    """
    
    def __init__(self, clip_model, clip_processor, device='cuda'):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device
        
        # Prompts puissants pour analyse d'angles et positions
        self.view_angle_prompts = [
            "front view facing forward directly",
            "slightly rotated to the left side view",
            "slightly rotated to the right side view",
            "left side profile view",
            "right side profile view",
            "three quarter view from front left",
            "three quarter view from front right",
            "three quarter view from back left",
            "three quarter view from back right",
            "back view from behind",
            "top down overhead view from above",
            "low angle view from below looking up",
            "diagonal angle tilted perspective",
            "close up detailed macro view",
            "wide angle distant full view"
        ]
        
        # Prompts pour caractéristiques visuelles
        self.visual_quality_prompts = [
            "sharp focused clear detailed image",
            "bright well lit illuminated scene",
            "dark shadowy dim lighting",
            "high contrast dramatic lighting",
            "soft diffused even lighting",
            "colorful vibrant saturated colors",
            "monochrome muted desaturated colors",
            "smooth flat surface texture",
            "rough textured detailed surface",
            "reflective shiny glossy surface",
            "matte non reflective surface"
        ]
        
        # Prompts pour distance et cadrage
        self.distance_framing_prompts = [
            "extreme close up macro shot",
            "close up tight framing",
            "medium shot balanced framing",
            "wide shot full view",
            "extreme wide panoramic view",
            "centered composition symmetrical",
            "off center asymmetrical composition",
            "high detail intricate texture",
            "simple clean minimal detail",
            "complex busy scene with many elements"
        ]
        
    def extract_visual_features(self, image_path: str) -> Dict:
        """
        Extrait les features visuelles d'une image avec CLIP + analyse détaillée multi-prompts
        
        Returns:
            Dict avec:
            - clip_embedding: Vecteur CLIP (512 dim)
            - clip_text_similarities: Scores de similarité avec les prompts textuels
            - view_angle_scores: Scores pour chaque angle de vue
            - visual_quality_scores: Scores pour caractéristiques visuelles
            - distance_framing_scores: Scores pour distance/cadrage
            - texture_features: Analyse de texture
            - color_histogram: Distribution des couleurs
            - edge_density: Densité de contours
            - brightness: Luminosité moyenne
            - contrast: Contraste
        """
        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        img_cv = cv2.imread(str(image_path))
        
        # 1. CLIP Embedding (compréhension sémantique de base)
        with torch.no_grad():
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            clip_features = self.clip_model.get_image_features(**inputs)
            clip_embedding = clip_features.cpu().numpy().flatten()
        
        # 2. CLIP avec prompts textuels pour analyse d'angle de vue
        view_angle_scores = {}
        with torch.no_grad():
            text_inputs = self.clip_processor(
                text=self.view_angle_prompts,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            text_features = self.clip_model.get_text_features(**text_inputs)
            
            # Calcul similarité image-texte pour chaque prompt
            image_features_norm = clip_features / clip_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features_norm @ text_features_norm.T).squeeze(0)
            
            for idx, prompt in enumerate(self.view_angle_prompts):
                view_angle_scores[prompt] = similarity[idx].cpu().item()
        
        # 3. CLIP avec prompts pour qualité visuelle
        visual_quality_scores = {}
        with torch.no_grad():
            text_inputs = self.clip_processor(
                text=self.visual_quality_prompts,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            text_features = self.clip_model.get_text_features(**text_inputs)
            
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features_norm @ text_features_norm.T).squeeze(0)
            
            for idx, prompt in enumerate(self.visual_quality_prompts):
                visual_quality_scores[prompt] = similarity[idx].cpu().item()
        
        # 4. CLIP avec prompts pour distance/cadrage
        distance_framing_scores = {}
        with torch.no_grad():
            text_inputs = self.clip_processor(
                text=self.distance_framing_prompts,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            text_features = self.clip_model.get_text_features(**text_inputs)
            
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features_norm @ text_features_norm.T).squeeze(0)
            
            for idx, prompt in enumerate(self.distance_framing_prompts):
                distance_framing_scores[prompt] = similarity[idx].cpu().item()
        
        # 2. Analyse de texture (gradients)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        texture_features = {
            'mean': gradient_magnitude.mean(),
            'std': gradient_magnitude.std(),
            'max': gradient_magnitude.max(),
            'p25': np.percentile(gradient_magnitude, 25),
            'p75': np.percentile(gradient_magnitude, 75)
        }
        
        # 3. Histogramme couleur (distribution RGB)
        color_histogram = []
        for i in range(3):
            hist = cv2.calcHist([img_cv], [i], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-5)
            color_histogram.extend(hist.tolist())
        
        # 4. Densité de contours (Canny)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (edges.shape[0] * edges.shape[1])
        
        # 5. Luminosité et contraste
        brightness = gray.mean()
        contrast = gray.std()
        
        # 6. Features ORB (points clés pour matching)
        orb = cv2.ORB_create(nfeatures=100)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        orb_features = {
            'num_keypoints': len(keypoints),
            'keypoint_response': np.mean([kp.response for kp in keypoints]) if keypoints else 0.0
        }
        
        return {
            'clip_embedding': clip_embedding,
            'view_angle_scores': view_angle_scores,
            'visual_quality_scores': visual_quality_scores,
            'distance_framing_scores': distance_framing_scores,
            'texture': texture_features,
            'color_histogram': np.array(color_histogram),
            'edge_density': edge_density,
            'brightness': brightness,
            'contrast': contrast,
            'orb': orb_features,
            'image_shape': img_cv.shape
        }
    
    def compute_visual_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Calcule la similarité visuelle entre deux images avec prompts CLIP multi-critères
        
        Returns:
            Score de similarité (0.0 à 1.0)
            - 1.0 = très similaires (angles proches)
            - 0.0 = très différentes
        """
        # 1. Similarité CLIP embedding (sémantique générale) - poids 30%
        clip_sim = np.dot(features1['clip_embedding'], features2['clip_embedding'])
        clip_sim = (clip_sim + 1) / 2  # Normaliser de [-1,1] à [0,1]
        
        # 2. Similarité d'angle de vue (prompts) - poids 25%
        # Comparer les top 3 angles les plus probables
        view1_sorted = sorted(features1['view_angle_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
        view2_sorted = sorted(features2['view_angle_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
        
        view_angle_sim = 0
        for v1_prompt, v1_score in view1_sorted:
            if v1_prompt in dict(view2_sorted):
                view_angle_sim += v1_score * dict(view2_sorted)[v1_prompt]
        view_angle_sim = view_angle_sim / 3.0  # Normaliser
        
        # 3. Similarité de qualité visuelle (lighting, focus, etc.) - poids 15%
        quality_scores1 = np.array(list(features1['visual_quality_scores'].values()))
        quality_scores2 = np.array(list(features2['visual_quality_scores'].values()))
        quality_sim = 1.0 - np.mean(np.abs(quality_scores1 - quality_scores2))
        
        # 4. Similarité de distance/cadrage - poids 10%
        distance_scores1 = np.array(list(features1['distance_framing_scores'].values()))
        distance_scores2 = np.array(list(features2['distance_framing_scores'].values()))
        distance_sim = 1.0 - np.mean(np.abs(distance_scores1 - distance_scores2))
        
        # 5. Similarité de texture - poids 10%
        texture_diff = 0
        for key in ['mean', 'std', 'p25', 'p75']:
            t1 = features1['texture'][key]
            t2 = features2['texture'][key]
            texture_diff += abs(t1 - t2) / (max(t1, t2) + 1e-5)
        texture_sim = 1.0 - (texture_diff / 4.0)
        
        # 6. Similarité couleur (histogramme) - poids 5%
        hist1 = features1['color_histogram']
        hist2 = features2['color_histogram']
        color_sim = 1.0 - np.sum(np.abs(hist1 - hist2)) / 2.0
        
        # 7. Similarité de luminosité/contraste - poids 3%
        brightness_diff = abs(features1['brightness'] - features2['brightness']) / 255.0
        contrast_diff = abs(features1['contrast'] - features2['contrast']) / 100.0
        lighting_sim = 1.0 - (brightness_diff + contrast_diff) / 2.0
        
        # 8. Similarité de contours - poids 2%
        edge_diff = abs(features1['edge_density'] - features2['edge_density'])
        edge_sim = 1.0 - edge_diff
        
        # Score final pondéré avec focus sur angles de vue
        final_similarity = (
            0.30 * clip_sim +              # Sémantique générale
            0.25 * view_angle_sim +        # ANGLE DE VUE (priorité!)
            0.15 * quality_sim +           # Qualité visuelle
            0.10 * distance_sim +          # Distance/cadrage
            0.10 * texture_sim +           # Texture
            0.05 * color_sim +             # Couleur
            0.03 * lighting_sim +          # Luminosité
            0.02 * edge_sim                # Contours
        )
        
        return np.clip(final_similarity, 0.0, 1.0)
    
    def classify_and_order_photos(
        self, 
        photo_paths: List[str],
        method: str = 'sequential',
        progress_callback=None
    ) -> Tuple[List[str], Dict]:
        """
        Classifie et ordonne les photos pour reconstruction 3D optimale
        
        Args:
            photo_paths: Liste des chemins d'images
            method: 'sequential' (ordre séquentiel) ou 'cluster' (par groupes)
            progress_callback: Fonction optionnelle (current, total, message)
        
        Returns:
            - Liste ordonnée des chemins
            - Dictionnaire avec statistiques et analyse
        """
        print(f"🔍 Analyse de {len(photo_paths)} photos avec Vision AI...")
        
        # 1. Extraire features pour toutes les images
        features_list = []
        for idx, path in enumerate(photo_paths):
            if progress_callback:
                progress_callback(idx + 1, len(photo_paths), f"Analyse Vision AI")
            print(f"   📸 Analyse {idx+1}/{len(photo_paths)}: {Path(path).name}")
            features = self.extract_visual_features(path)
            features['path'] = path
            features['index'] = idx
            features_list.append(features)
        
        print(f"✅ Features extraites pour {len(features_list)} images")
        
        # 2. Calculer matrice de similarité
        n = len(features_list)
        similarity_matrix = np.zeros((n, n))
        
        print("🔗 Calcul des similarités entre images...")
        for i in range(n):
            for j in range(i+1, n):
                sim = self.compute_visual_similarity(features_list[i], features_list[j])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
        
        print("✅ Matrice de similarité calculée")
        
        # 3. Ordonner selon la méthode choisie
        if method == 'sequential':
            ordered_indices = self._order_sequential(similarity_matrix, features_list)
            order_type = "Séquentiel (photos qui se suivent)"
        else:
            ordered_indices = self._order_by_clusters(similarity_matrix, features_list)
            order_type = "Par groupes d'angles similaires"
        
        # 4. Créer la liste ordonnée
        ordered_paths = [features_list[idx]['path'] for idx in ordered_indices]
        
        # 5. Calculer statistiques
        stats = self._compute_statistics(similarity_matrix, ordered_indices, features_list)
        stats['order_type'] = order_type
        stats['total_photos'] = len(photo_paths)
        
        return ordered_paths, stats
    
    def _order_sequential(self, similarity_matrix: np.ndarray, features_list: List[Dict]) -> List[int]:
        """
        Ordonne les photos de manière séquentielle (chaîne de similarité)
        Algorithme: Nearest Neighbor Chain
        """
        n = len(features_list)
        visited = [False] * n
        ordered = []
        
        # Commencer par l'image avec la luminosité moyenne (ni trop sombre, ni trop claire)
        brightnesses = [f['brightness'] for f in features_list]
        median_brightness = np.median(brightnesses)
        current = np.argmin([abs(b - median_brightness) for b in brightnesses])
        
        ordered.append(current)
        visited[current] = True
        
        # Construire la chaîne
        for _ in range(n - 1):
            # Trouver l'image non visitée la plus similaire
            similarities = similarity_matrix[current].copy()
            similarities[visited] = -1  # Exclure les visitées
            
            next_idx = np.argmax(similarities)
            ordered.append(next_idx)
            visited[next_idx] = True
            current = next_idx
        
        return ordered
    
    def _order_by_clusters(self, similarity_matrix: np.ndarray, features_list: List[Dict]) -> List[int]:
        """
        Ordonne les photos par groupes d'angles similaires
        """
        from sklearn.cluster import AgglomerativeClustering
        
        n = len(features_list)
        
        # Convertir similarité en distance
        distance_matrix = 1.0 - similarity_matrix
        
        # Clustering hiérarchique
        n_clusters = max(3, min(10, n // 5))  # Entre 3 et 10 clusters
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # Ordonner par cluster, puis par similarité dans chaque cluster
        ordered = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i in range(n) if labels[i] == cluster_id]
            
            # Trier par luminosité dans le cluster (progression naturelle)
            cluster_indices.sort(key=lambda i: features_list[i]['brightness'])
            ordered.extend(cluster_indices)
        
        return ordered
    
    def _compute_statistics(
        self, 
        similarity_matrix: np.ndarray, 
        ordered_indices: List[int],
        features_list: List[Dict]
    ) -> Dict:
        """Calcule des statistiques sur l'ordonnancement"""
        
        # Similarité moyenne entre images consécutives
        consecutive_similarities = []
        for i in range(len(ordered_indices) - 1):
            idx1 = ordered_indices[i]
            idx2 = ordered_indices[i + 1]
            consecutive_similarities.append(similarity_matrix[idx1][idx2])
        
        avg_consecutive_sim = np.mean(consecutive_similarities) if consecutive_similarities else 0
        
        # Diversité de la couverture (variance des embeddings CLIP)
        embeddings = np.array([features_list[i]['clip_embedding'] for i in ordered_indices])
        coverage_variance = np.var(embeddings)
        
        # Distribution de luminosité
        brightnesses = [features_list[i]['brightness'] for i in ordered_indices]
        
        return {
            'avg_consecutive_similarity': float(avg_consecutive_sim),
            'min_consecutive_similarity': float(min(consecutive_similarities)) if consecutive_similarities else 0,
            'max_consecutive_similarity': float(max(consecutive_similarities)) if consecutive_similarities else 0,
            'coverage_variance': float(coverage_variance),
            'brightness_range': (float(min(brightnesses)), float(max(brightnesses))),
            'brightness_progression': 'smooth' if np.std(np.diff(brightnesses)) < 20 else 'varied'
        }
    
    def visualize_ordering(
        self, 
        photo_paths: List[str], 
        ordered_paths: List[str],
        output_dir: str
    ):
        """
        Crée une visualisation de l'ordonnancement avec miniatures
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        n = len(ordered_paths)
        cols = min(10, n)
        rows = (n + cols - 1) // cols
        
        fig = plt.figure(figsize=(cols * 2, rows * 2))
        gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.1)
        
        for idx, path in enumerate(ordered_paths):
            row = idx // cols
            col = idx % cols
            
            ax = fig.add_subplot(gs[row, col])
            
            # Charger et afficher miniature
            img = Image.open(path)
            img.thumbnail((200, 200))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"#{idx+1}", fontsize=8)
        
        plt.suptitle(f"Ordre des {n} photos pour reconstruction 3D", fontsize=16, fontweight='bold')
        
        output_path = Path(output_dir) / "photo_ordering_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_report(
        self, 
        ordered_paths: List[str], 
        stats: Dict,
        output_dir: str
    ) -> str:
        """Génère un rapport détaillé de la classification"""
        
        report = []
        report.append("=" * 70)
        report.append("📷 RAPPORT DE CLASSIFICATION INTELLIGENTE - PHOTOGRAMMÉTRIE")
        report.append("=" * 70)
        report.append("")
        
        report.append(f"📊 **Statistiques globales:**")
        report.append(f"   Nombre total de photos: {stats['total_photos']}")
        report.append(f"   Méthode d'ordonnancement: {stats['order_type']}")
        report.append("")
        
        report.append(f"🔗 **Cohérence de l'ordre:**")
        report.append(f"   Similarité moyenne (photos consécutives): {stats['avg_consecutive_similarity']*100:.1f}%")
        report.append(f"   Similarité minimale: {stats['min_consecutive_similarity']*100:.1f}%")
        report.append(f"   Similarité maximale: {stats['max_consecutive_similarity']*100:.1f}%")
        report.append("")
        
        report.append(f"🎨 **Analyse de couverture:**")
        report.append(f"   Variance de couverture: {stats['coverage_variance']:.4f}")
        report.append(f"   Plage de luminosité: {stats['brightness_range'][0]:.1f} - {stats['brightness_range'][1]:.1f}")
        report.append(f"   Progression: {stats['brightness_progression']}")
        report.append("")
        
        report.append(f"✅ **Qualité de l'ordonnancement:**")
        if stats['avg_consecutive_similarity'] > 0.75:
            report.append(f"   ⭐⭐⭐⭐⭐ EXCELLENT - Photos très bien ordonnées")
        elif stats['avg_consecutive_similarity'] > 0.60:
            report.append(f"   ⭐⭐⭐⭐ BON - Ordre cohérent pour reconstruction")
        elif stats['avg_consecutive_similarity'] > 0.45:
            report.append(f"   ⭐⭐⭐ ACCEPTABLE - Peut nécessiter ajustements")
        else:
            report.append(f"   ⭐⭐ MOYEN - Photos très différentes")
        
        report.append("")
        report.append("=" * 70)
        report.append("")
        report.append(f"📋 **Ordre des photos (optimisé pour Dust3R/reconstruction 3D):**")
        report.append("")
        
        for idx, path in enumerate(ordered_paths, 1):
            filename = Path(path).name
            report.append(f"   {idx:3d}. {filename}")
        
        report.append("")
        report.append("=" * 70)
        report.append("💡 **Recommandations:**")
        report.append("   - Utilisez cet ordre pour Dust3R, MeshRoom ou autres outils 3D")
        report.append("   - Les photos sont classées par similarité d'angles de vue")
        report.append("   - La progression est optimisée pour une reconstruction fluide")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        # Sauvegarder le rapport
        report_path = Path(output_dir) / "classification_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text


def classify_photos_with_ai(
    photo_paths: List[str],
    clip_model,
    clip_processor,
    device: str = 'cuda',
    method: str = 'sequential',
    output_dir: str = None,
    progress_callback=None
) -> Tuple[List[str], str, str]:
    """
    Fonction principale pour classifier des photos avec l'IA
    
    Args:
        progress_callback: Fonction (current, total, message) pour progression
    
    Returns:
        - ordered_paths: Liste ordonnée des chemins
        - report: Rapport textuel
        - visualization_path: Chemin de la visualisation
    """
    classifier = IntelligentPhotoClassifier(clip_model, clip_processor, device)
    
    # Classifier et ordonner
    ordered_paths, stats = classifier.classify_and_order_photos(
        photo_paths, 
        method=method,
        progress_callback=progress_callback
    )
    
    # Générer rapport
    if output_dir:
        report = classifier.generate_report(ordered_paths, stats, output_dir)
        
        # Créer visualisation
        try:
            viz_path = classifier.visualize_ordering(photo_paths, ordered_paths, output_dir)
        except Exception as e:
            print(f"⚠️ Erreur visualisation: {e}")
            viz_path = None
    else:
        report = f"Photos ordonnées: {len(ordered_paths)} images"
        viz_path = None
    
    return ordered_paths, report, viz_path
