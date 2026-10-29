"""
Structure-from-Motion Léger (<1GB)
===================================
Alternative à COLMAP pour WSL - Python pur avec OpenCV

Basé sur: alyssaq/3Dreconstruction (MIT License)
Adapté pour: Reconstruction 3D multi-images drone
"""

import numpy as np
import cv2
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightweightSfM:
    """
    Structure-from-Motion léger compatible WSL
    Utilise SIFT + Essential Matrix + Triangulation
    """
    
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.images_path = self.workspace / "images"
        self.output_path = self.workspace / "output"
        self.images_path.mkdir(exist_ok=True)
        self.output_path.mkdir(exist_ok=True)
        
        logger.info(f"✅ Workspace initialisé: {workspace_path}")
    
    def cart2hom(self, arr: np.ndarray) -> np.ndarray:
        """Convertir coordonnées cartésiennes en homogènes"""
        if arr.ndim == 1:
            return np.hstack([arr, 1])
        return np.vstack([arr, np.ones(arr.shape[1])])
    
    def hom2cart(self, arr: np.ndarray) -> np.ndarray:
        """Convertir homogènes en cartésiennes"""
        num_rows = len(arr)
        if num_rows == 1 or arr.ndim == 1:
            return arr
        return arr[:num_rows - 1] / arr[num_rows - 1]
    
    def find_features(self, img1: np.ndarray, img2: np.ndarray, max_features: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Détecter et matcher les features SIFT entre 2 images
        
        Returns:
            points1, points2: Coordonnées des points matchés (2xN)
        """
        # SIFT detector
        sift = cv2.SIFT_create(nfeatures=max_features)
        
        # Détecter keypoints
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return np.array([]).reshape(2, 0), np.array([]).reshape(2, 0)
        
        # Matcher FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Ratio test de Lowe
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        
        if len(good) < 8:
            logger.warning(f"Pas assez de matches: {len(good)}")
            return np.array([]).reshape(2, 0), np.array([]).reshape(2, 0)
        
        # Extraire coordonnées
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        
        # RANSAC pour éliminer outliers
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if mask is None:
            return src_pts.T, dst_pts.T
        
        mask = mask.ravel()
        pts1 = src_pts[mask == 1]
        pts2 = dst_pts[mask == 1]
        
        return pts1.T, pts2.T
    
    def compute_essential_matrix(self, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Calculer matrice essentielle E
        pts1, pts2: points 2D homogènes (3xN)
        K: matrice intrinsèque caméra
        """
        # Normaliser
        pts1n = np.dot(np.linalg.inv(K), pts1)
        pts2n = np.dot(np.linalg.inv(K), pts2)
        
        # 8-point algorithm
        E, mask = cv2.findEssentialMat(pts1n[:2].T, pts2n[:2].T, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=0.001)
        
        return E
    
    def decompose_essential_matrix(self, E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Décomposer E en 4 configurations possibles [R|t]
        """
        U, S, Vt = np.linalg.svd(E)
        
        # Assurer det(R) = +1
        if np.linalg.det(np.dot(U, Vt)) < 0:
            Vt = -Vt
        
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        
        # 4 solutions possibles
        R1 = np.dot(U, np.dot(W, Vt))
        R2 = np.dot(U, np.dot(W.T, Vt))
        t1 = U[:, 2]
        t2 = -U[:, 2]
        
        return [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]
    
    def triangulate_points(self, pts1: np.ndarray, pts2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """
        Triangulation linéaire pour retrouver points 3D
        pts1, pts2: 2D points (3xN homogènes)
        P1, P2: Camera matrices (3x4)
        Returns: 4xN points 3D homogènes
        """
        n = pts1.shape[1]
        points3d = np.ones((4, n))
        
        for i in range(n):
            A = np.array([
                pts1[0, i] * P1[2, :] - P1[0, :],
                pts1[1, i] * P1[2, :] - P1[1, :],
                pts2[0, i] * P2[2, :] - P2[0, :],
                pts2[1, i] * P2[2, :] - P2[1, :]
            ])
            
            _, _, V = np.linalg.svd(A)
            X = V[-1]
            points3d[:, i] = X / X[3]
        
        return points3d
    
    def estimate_intrinsic(self, img_width: int, img_height: int, focal_ratio: float = 1.2) -> np.ndarray:
        """
        Estimer matrice intrinsèque K depuis dimensions image
        focal_ratio: ratio focal/sensor (typique: 1.0-1.5)
        """
        focal = max(img_width, img_height) * focal_ratio
        K = np.array([
            [focal, 0, img_width / 2],
            [0, focal, img_height / 2],
            [0, 0, 1]
        ])
        return K
    
    def reconstruct_from_pair(self, img1_path: str, img2_path: str, progress_callback=None) -> Dict:
        """
        Reconstruction 3D depuis une paire d'images
        """
        logger.info(f"📸 Reconstruction: {Path(img1_path).name} <-> {Path(img2_path).name}")
        
        # Charger images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return {'success': False, 'error': 'Cannot load images'}
        
        h, w = img1.shape[:2]
        
        # Estimer intrinsèque
        K = self.estimate_intrinsic(w, h)
        
        if progress_callback:
            progress_callback(10, 100, "Détection features SIFT...")
        
        # Feature matching
        pts1, pts2 = self.find_features(img1, img2)
        
        if pts1.shape[1] < 8:
            return {'success': False, 'error': 'Not enough matches', 'matches': pts1.shape[1]}
        
        logger.info(f"✅ {pts1.shape[1]} matches trouvés")
        
        if progress_callback:
            progress_callback(40, 100, f"{pts1.shape[1]} matches...")
        
        # Convertir en homogènes
        pts1_hom = self.cart2hom(pts1)
        pts2_hom = self.cart2hom(pts2)
        
        # Essential matrix
        E = self.compute_essential_matrix(pts1_hom, pts2_hom, K)
        
        if progress_callback:
            progress_callback(60, 100, "Essential matrix...")
        
        # Décomposer E
        configs = self.decompose_essential_matrix(E)
        
        # Tester les 4 configurations
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # [I|0]
        
        best_config = None
        max_positive = 0
        
        for R, t in configs:
            P2 = np.hstack([R, t.reshape(3, 1)])
            
            # Triangulate
            pts1n = np.dot(np.linalg.inv(K), pts1_hom)
            pts2n = np.dot(np.linalg.inv(K), pts2_hom)
            
            points3d = self.triangulate_points(pts1n, pts2n, P1, P2)
            
            # Compter points devant les 2 caméras
            positive = np.sum((points3d[2, :] > 0))
            
            if positive > max_positive:
                max_positive = positive
                best_config = (R, t, points3d)
        
        if progress_callback:
            progress_callback(90, 100, "Triangulation...")
        
        R, t, points3d = best_config
        
        # Export
        output_ply = self.output_path / "pointcloud.ply"
        self.export_ply(points3d, output_ply)
        
        if progress_callback:
            progress_callback(100, 100, "Terminé ✅")
        
        return {
            'success': True,
            'num_points': points3d.shape[1],
            'num_matches': pts1.shape[1],
            'pointcloud': str(output_ply),
            'intrinsic': K.tolist(),
            'rotation': R.tolist(),
            'translation': t.tolist()
        }
    
    def export_ply(self, points3d: np.ndarray, output_path: Path):
        """
        Exporter nuage de points au format PLY
        points3d: 4xN homogènes
        """
        points = self.hom2cart(points3d)
        n = points.shape[1]
        
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            for i in range(n):
                f.write(f"{points[0,i]} {points[1,i]} {points[2,i]}\n")
        
        logger.info(f"✅ PLY exporté: {output_path} ({n} points)")
    
    def run_reconstruction(self, image_paths: List[str], progress_callback=None) -> Dict:
        """
        Pipeline reconstruction multi-images
        Pour MVP: utilise première paire
        """
        if len(image_paths) < 2:
            return {'success': False, 'error': 'Need at least 2 images'}
        
        # Sélectionner paire avec meilleur overlap (simple: prendre 2 images consécutives)
        img1 = image_paths[0]
        img2 = image_paths[1]
        
        logger.info(f"🚀 Reconstruction SfM sur {len(image_paths)} images")
        logger.info(f"   Paire sélectionnée: {Path(img1).name} + {Path(img2).name}")
        
        result = self.reconstruct_from_pair(img1, img2, progress_callback)
        
        if result['success']:
            result['total_images'] = len(image_paths)
            result['images_used'] = 2
        
        return result


# Test simple
if __name__ == "__main__":
    import sys
    
    sfm = LightweightSfM("/tmp/sfm_test")
    
    if len(sys.argv) > 2:
        result = sfm.run_reconstruction([sys.argv[1], sys.argv[2]])
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python3 sfm_lightweight.py <img1> <img2>")
        print("\nTest avec images de démonstration:")
        print("  python3 sfm_lightweight.py /path/to/img1.jpg /path/to/img2.jpg")
