"""
COLMAP Photogrammetry Integration
==================================
Licence: BSD 3-Clause (Usage commercial autorisé)
Pipeline professionnel pour reconstruction 3D

Fonctionnalités:
- Structure-from-Motion (SfM)
- Multi-View Stereo (MVS)
- Dense reconstruction
- Mesh generation
- Texture mapping
"""

import os
import subprocess
import shutil
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_headless_env():
    """Créer environnement pour mode headless (sans GUI)"""
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    return env


class ColmapPhotogrammetry:
    """
    Interface Python pour COLMAP (BSD 3-Clause License - Commercial OK)
    
    Pipeline complet de photogrammétrie:
    1. Feature extraction (SIFT)
    2. Feature matching
    3. Sparse reconstruction (SfM)
    4. Image undistortion
    5. Dense reconstruction (MVS)
    6. Meshing
    7. Texturing
    """
    
    def __init__(self, workspace_path: str):
        """
        Initialise le workspace COLMAP
        
        Args:
            workspace_path: Chemin du dossier de travail
        """
        self.workspace = Path(workspace_path)
        self.images_path = self.workspace / "images"
        self.database_path = self.workspace / "database.db"
        self.sparse_path = self.workspace / "sparse"
        self.dense_path = self.workspace / "dense"
        
        # Créer la structure
        for path in [self.images_path, self.sparse_path, self.dense_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Workspace COLMAP initialisé: {self.workspace}")
    
    @staticmethod
    def check_installation() -> Tuple[bool, str]:
        """Vérifie si COLMAP est installé"""
        try:
            result = subprocess.run(
                ['colmap', '--help'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, "✅ COLMAP installé et fonctionnel"
            else:
                return False, "❌ COLMAP trouvé mais non fonctionnel"
        except FileNotFoundError:
            return False, "❌ COLMAP non installé. Installer avec: sudo apt install colmap"
        except Exception as e:
            return False, f"❌ Erreur: {str(e)}"
    
    def prepare_images(self, image_paths: List[str], progress_callback=None) -> int:
        """
        Copie les images dans le workspace COLMAP
        
        Args:
            image_paths: Liste des chemins d'images
            progress_callback: Fonction de callback pour progression
            
        Returns:
            Nombre d'images copiées
        """
        logger.info(f"📸 Préparation de {len(image_paths)} images...")
        
        for idx, img_path in enumerate(image_paths):
            src = Path(img_path)
            dst = self.images_path / src.name
            shutil.copy2(src, dst)
            
            if progress_callback:
                progress_callback(idx + 1, len(image_paths), f"Copie: {src.name}")
        
        logger.info(f"✅ {len(image_paths)} images préparées")
        return len(image_paths)
    
    def feature_extraction(
        self,
        use_gpu: bool = True,
        sift_options: Optional[Dict] = None,
        progress_callback=None
    ) -> bool:
        """
        Extraction des features SIFT
        
        Args:
            use_gpu: Utiliser GPU si disponible
            sift_options: Options SIFT personnalisées
            progress_callback: Callback pour progression
            
        Returns:
            True si succès
        """
        logger.info("🔍 Extraction des features SIFT...")
        
        if progress_callback:
            progress_callback(0, 100, "Extraction features SIFT...")
        
        # Force headless mode for COLMAP
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        
        cmd = [
            'colmap', 'feature_extractor',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_path),
            '--ImageReader.single_camera', '1',  # Une seule caméra
            '--ImageReader.camera_model', 'OPENCV',  # Modèle OpenCV
        ]
        
        # En mode headless/WSL, désactiver GPU pour SIFT (OpenGL issues)
        if use_gpu:
            # Vérifier si on peut vraiment utiliser GPU
            import platform
            if 'microsoft' in platform.uname().release.lower():
                logger.warning("⚠️  WSL détecté: extraction SIFT sur CPU au lieu de GPU")
                use_gpu = False
        
        if use_gpu:
            cmd.extend(['--SiftExtraction.use_gpu', '1'])
        else:
            cmd.extend(['--SiftExtraction.use_gpu', '0'])
        
        if sift_options:
            for key, value in sift_options.items():
                cmd.extend([f'--SiftExtraction.{key}', str(value)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            logger.info("✅ Features extraites")
            if progress_callback:
                progress_callback(100, 100, "Features extraites ✅")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur extraction: {e.stderr}")
            return False
    
    def feature_matching(
        self,
        use_gpu: bool = True,
        matching_type: str = 'exhaustive',
        progress_callback=None
    ) -> bool:
        """
        Matching des features entre images
        
        Args:
            use_gpu: Utiliser GPU
            matching_type: 'exhaustive', 'sequential', 'spatial'
            progress_callback: Callback progression
            
        Returns:
            True si succès
        """
        logger.info(f"🔗 Matching des features ({matching_type})...")
        
        if progress_callback:
            progress_callback(0, 100, "Matching des features...")
        
        if matching_type == 'exhaustive':
            cmd = [
                'colmap', 'exhaustive_matcher',
                '--database_path', str(self.database_path),
            ]
        elif matching_type == 'sequential':
            cmd = [
                'colmap', 'sequential_matcher',
                '--database_path', str(self.database_path),
            ]
        elif matching_type == 'spatial':
            cmd = [
                'colmap', 'spatial_matcher',
                '--database_path', str(self.database_path),
            ]
        else:
            raise ValueError(f"Type de matching invalide: {matching_type}")
        
        # En mode headless/WSL, désactiver GPU pour matching
        import platform
        if use_gpu and 'microsoft' in platform.uname().release.lower():
            logger.warning("⚠️  WSL: matching SIFT sur CPU")
            use_gpu = False
        
        if use_gpu:
            cmd.extend(['--SiftMatching.use_gpu', '1'])
        else:
            cmd.extend(['--SiftMatching.use_gpu', '0'])
        
        env = get_headless_env()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            logger.info("✅ Features matchées")
            if progress_callback:
                progress_callback(100, 100, "Features matchées ✅")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur matching: {e.stderr}")
            return False
    
    def sparse_reconstruction(self, progress_callback=None) -> bool:
        """
        Reconstruction sparse (Structure-from-Motion)
        
        Returns:
            True si succès
        """
        logger.info("🎯 Reconstruction sparse (SfM)...")
        
        if progress_callback:
            progress_callback(0, 100, "Reconstruction sparse...")
        
        cmd = [
            'colmap', 'mapper',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_path),
            '--output_path', str(self.sparse_path),
        ]
        
        env = get_headless_env()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            logger.info("✅ Reconstruction sparse terminée")
            if progress_callback:
                progress_callback(100, 100, "Reconstruction sparse ✅")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur reconstruction: {e.stderr}")
            return False
    
    def image_undistortion(self, progress_callback=None) -> bool:
        """
        Undistortion des images pour dense reconstruction
        
        Returns:
            True si succès
        """
        logger.info("📐 Undistortion des images...")
        
        if progress_callback:
            progress_callback(0, 100, "Undistortion des images...")
        
        input_path = self.sparse_path / "0"
        if not input_path.exists():
            logger.error("❌ Pas de reconstruction sparse trouvée")
            return False
        
        cmd = [
            'colmap', 'image_undistorter',
            '--image_path', str(self.images_path),
            '--input_path', str(input_path),
            '--output_path', str(self.dense_path),
            '--output_type', 'COLMAP',
        ]
        
        env = get_headless_env()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            logger.info("✅ Images undistorted")
            if progress_callback:
                progress_callback(100, 100, "Images undistorted ✅")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur undistortion: {e.stderr}")
            return False
    
    def dense_reconstruction(
        self,
        quality: str = 'high',
        progress_callback=None
    ) -> bool:
        """
        Dense reconstruction (Multi-View Stereo)
        
        Args:
            quality: 'low', 'medium', 'high', 'extreme'
            progress_callback: Callback progression
            
        Returns:
            True si succès
        """
        logger.info(f"🌟 Dense reconstruction (qualité: {quality})...")
        
        if progress_callback:
            progress_callback(0, 100, "Dense reconstruction (peut prendre du temps)...")
        
        # Paramètres de qualité
        quality_params = {
            'low': {'--PatchMatchStereo.window_radius': '3'},
            'medium': {'--PatchMatchStereo.window_radius': '5'},
            'high': {'--PatchMatchStereo.window_radius': '7'},
            'extreme': {'--PatchMatchStereo.window_radius': '9'},
        }
        
        cmd = [
            'colmap', 'patch_match_stereo',
            '--workspace_path', str(self.dense_path),
            '--workspace_format', 'COLMAP',
        ]
        
        # Ajouter paramètres de qualité
        for key, value in quality_params.get(quality, quality_params['high']).items():
            cmd.extend([key, value])
        
        env = get_headless_env()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            logger.info("✅ Dense reconstruction complète")
            if progress_callback:
                progress_callback(100, 100, "Dense reconstruction ✅")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur dense reconstruction: {e.stderr}")
            return False
    
    def stereo_fusion(self, progress_callback=None) -> bool:
        """
        Fusion des cartes de profondeur en nuage de points dense
        
        Returns:
            True si succès
        """
        logger.info("🔗 Fusion des cartes de profondeur...")
        
        if progress_callback:
            progress_callback(0, 100, "Fusion stereo...")
        
        cmd = [
            'colmap', 'stereo_fusion',
            '--workspace_path', str(self.dense_path),
            '--workspace_format', 'COLMAP',
            '--input_type', 'geometric',
            '--output_path', str(self.dense_path / 'fused.ply'),
        ]
        
        env = get_headless_env()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            logger.info("✅ Fusion complète")
            if progress_callback:
                progress_callback(100, 100, "Fusion stereo ✅")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur fusion: {e.stderr}")
            return False
    
    def poisson_meshing(
        self,
        trim: float = 7,
        progress_callback=None
    ) -> Optional[str]:
        """
        Génération du mesh avec Poisson reconstruction
        
        Args:
            trim: Paramètre de trim (0-14, plus haut = plus agressif)
            progress_callback: Callback progression
            
        Returns:
            Chemin du mesh généré ou None
        """
        logger.info("🎨 Génération du mesh (Poisson)...")
        
        if progress_callback:
            progress_callback(0, 100, "Génération mesh...")
        
        input_ply = self.dense_path / 'fused.ply'
        output_ply = self.dense_path / 'meshed-poisson.ply'
        
        if not input_ply.exists():
            logger.error("❌ Nuage de points fusé non trouvé")
            return None
        
        cmd = [
            'colmap', 'poisson_mesher',
            '--input_path', str(input_ply),
            '--output_path', str(output_ply),
            '--PoissonMeshing.trim', str(trim),
        ]
        
        env = get_headless_env()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            logger.info(f"✅ Mesh généré: {output_ply}")
            if progress_callback:
                progress_callback(100, 100, "Mesh généré ✅")
            return str(output_ply)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur meshing: {e.stderr}")
            return None
    
    def run_full_pipeline(
        self,
        image_paths: List[str],
        use_gpu: bool = True,
        quality: str = 'high',
        progress_callback=None
    ) -> Dict:
        """
        Exécute le pipeline complet de photogrammétrie
        
        Args:
            image_paths: Liste des chemins d'images
            use_gpu: Utiliser GPU
            quality: Qualité dense reconstruction
            progress_callback: Callback progression globale
            
        Returns:
            Dictionnaire avec les résultats
        """
        results = {
            'success': False,
            'images_prepared': 0,
            'sparse_model': None,
            'dense_pointcloud': None,
            'mesh': None,
            'errors': []
        }
        
        # Étape 1: Préparation images
        try:
            results['images_prepared'] = self.prepare_images(image_paths, progress_callback)
        except Exception as e:
            results['errors'].append(f"Préparation images: {str(e)}")
            return results
        
        # Étape 2: Feature extraction
        if not self.feature_extraction(use_gpu, progress_callback=progress_callback):
            results['errors'].append("Feature extraction échouée")
            return results
        
        # Étape 3: Feature matching
        if not self.feature_matching(use_gpu, progress_callback=progress_callback):
            results['errors'].append("Feature matching échoué")
            return results
        
        # Étape 4: Sparse reconstruction
        if not self.sparse_reconstruction(progress_callback):
            results['errors'].append("Reconstruction sparse échouée")
            return results
        
        results['sparse_model'] = str(self.sparse_path / "0")
        
        # Étape 5: Undistortion
        if not self.image_undistortion(progress_callback):
            results['errors'].append("Undistortion échouée")
            return results
        
        # Étape 6: Dense reconstruction
        if not self.dense_reconstruction(quality, progress_callback):
            results['errors'].append("Dense reconstruction échouée")
            return results
        
        # Étape 7: Fusion
        if not self.stereo_fusion(progress_callback):
            results['errors'].append("Fusion échouée")
            return results
        
        results['dense_pointcloud'] = str(self.dense_path / 'fused.ply')
        
        # Étape 8: Meshing
        mesh_path = self.poisson_meshing(progress_callback=progress_callback)
        if mesh_path:
            results['mesh'] = mesh_path
        else:
            results['errors'].append("Meshing échoué")
        
        results['success'] = True
        return results
    
    def get_reconstruction_stats(self) -> Dict:
        """
        Obtient les statistiques de la reconstruction
        
        Returns:
            Dictionnaire avec les stats
        """
        stats = {
            'num_images': 0,
            'num_points': 0,
            'num_cameras': 0,
            'workspace': str(self.workspace)
        }
        
        # Compter les images
        if self.images_path.exists():
            stats['num_images'] = len(list(self.images_path.glob('*.jpg'))) + \
                                   len(list(self.images_path.glob('*.png')))
        
        # TODO: Parser les fichiers COLMAP pour obtenir plus de stats
        
        return stats


# Exemple d'utilisation
if __name__ == "__main__":
    # Vérifier installation
    installed, message = ColmapPhotogrammetry.check_installation()
    print(message)
    
    if installed:
        # Créer workspace
        colmap = ColmapPhotogrammetry("/tmp/colmap_test")
        
        # Exemple pipeline
        # images = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        # results = colmap.run_full_pipeline(images, use_gpu=True, quality='high')
        # print(results)
