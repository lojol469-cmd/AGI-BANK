import os
import torch
import gc
import argparse
from diffusers import ZImagePipeline

# ==============================================================================
# CONFIGURATION DES CHEMINS
# ==============================================================================
# Adapte ces chemins si nécessaire.
# On suppose que les fichiers sont dans D:\kibali-IA\models\comfyui_image\
MODEL_DIR = r"D:\kibali-IA\models\comfyui_image" 

# Noms des fichiers tels que tu les as copiés
UNET_PATH = os.path.join(MODEL_DIR, "z_image_turbo_bf16.safetensors")
VAE_PATH = os.path.join(MODEL_DIR, "ae.safetensors")
CLIP_PATH = os.path.join(MODEL_DIR, "qwen_3_4b.safetensors")

# ==============================================================================
# GESTION DU PIPELINE (CHARGEMENT / DÉCHARGEMENT)
# ==============================================================================
def load_pipeline():
    """Charge le pipeline Z-Image-Turbo en mémoire (VRAM)."""
    print("[Pipeline] Chargement du modèle Z-Image-Turbo...")
    
    # Vérification de l'existence des fichiers
    for path in [UNET_PATH, VAE_PATH, CLIP_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier modèle est introuvable : {path}")

    # Chargement du pipeline à partir des fichiers locaux
    # On utilise torch.bfloat16 pour des performances optimales sur la RTX 5090
    pipe = ZImagePipeline.from_pretrained(
        MODEL_DIR, # Le pipeline cherche les sous-dossiers (unet, vae, text_encoder)
        torch_dtype=torch.bfloat16,
        local_files_only=True, # On force l'utilisation des fichiers locaux
    )
    
    # Envoi du pipeline sur le GPU (cuda)
    pipe.to("cuda")
    
    print("[Pipeline] Modèle chargé avec succès sur le GPU.")
    return pipe

def unload_pipeline(pipe):
    """Décharge le pipeline de la VRAM pour libérer la mémoire."""
    print("[Pipeline] Déchargement du modèle...")
    del pipe
    gc.collect() # Force le ramasse-miettes de Python
    torch.cuda.empty_cache() # Vide le cache de la VRAM
    print("[Pipeline] VRAM libérée.")

# ==============================================================================
# GÉNÉRATION D'IMAGE
# ==============================================================================
def generate_image(prompt: str, output_path: str = "output.png", steps: int = 9, seed: int = 42):
    """Fonction principale de génération."""
    pipe = None
    try:
        # 1. Chargement
        pipe = load_pipeline()
        
        # 2. Paramètres de génération
        print(f"[Génération] Prompt : {prompt}")
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # 3. Inférence (l'appel au modèle)
        # Z-Image-Turbo nécessite un guidance_scale de 0.0 et ~9 étapes
        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        
        # 4. Sauvegarde
        image.save(output_path)
        print(f"[Succès] Image sauvegardée ici : {output_path}")
        
    except Exception as e:
        print(f"[Erreur] Une erreur est survenue : {e}")
    finally:
        # 5. Déchargement (toujours exécuté, même en cas d'erreur)
        if pipe is not None:
            unload_pipeline(pipe)

# ==============================================================================
# INTERFACE EN LIGNE DE COMMANDE (CLI)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur d'images Z-Image-Turbo en CLI.")
    parser.add_argument("--prompt", type=str, required=True, help="Le prompt pour générer l'image.")
    parser.add_argument("--output", type=str, default="z_image_output.png", help="Le chemin de sauvegarde de l'image.")
    parser.add_argument("--steps", type=int, default=9, help="Nombre d'étapes de débruitage (9 par défaut pour Turbo).")
    parser.add_argument("--seed", type=int, default=42, help="La graine aléatoire pour la reproductibilité.")
    
    args = parser.parse_args()
    
    generate_image(
        prompt=args.prompt,
        output_path=args.output,
        steps=args.steps,
        seed=args.seed
    )