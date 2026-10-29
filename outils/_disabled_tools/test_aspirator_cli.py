import os
import json

def test_aspirator():
    path = r"D:\kibali-IA\outils"
    query = "rag" # On cherche le mot-clé "rag"
    
    print(f"🚀 Test d'aspiration du dossier : {path}")
    if not os.path.exists(path):
        print("❌ Le dossier n'existe pas.")
        return

    files_found = []
    results = []

    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(('.py', '.js', '.html', '.txt')):
                file_path = os.path.join(root, f)
                files_found.append(file_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        for i, line in enumerate(lines):
                            if query in line.lower():
                                results.append(f"📄 {f} (Ligne {i+1}):\n{line.strip()}")
                except Exception:
                    pass

    print(f"\n✅ {len(files_found)} fichiers trouvés.")
    print(f"🔍 Résultats pour '{query}' ({len(results)} trouvés) :\n")
    
    for r in results[:5]: # On affiche les 5 premiers
        print(r + "\n")

if __name__ == "__main__":
    test_aspirator()