from huggingface_hub import snapshot_download

print("📥 Téléchargement du modèle de résumé en cours...")
snapshot_download(
    repo_id="CATIE-AQ/LMF2_350M_french_summary",
    local_dir=r"D:\kibali-IA\models\lmf2_350m_summary",
    local_dir_use_symlinks=False,
)
print("✅ Modèle téléchargé avec succès dans : D:\\kibali-IA\\models\\lmf2_350m_summary")