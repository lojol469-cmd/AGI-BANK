import pathlib
import datetime
import shutil

target = pathlib.Path(r"D:\kibali-IA\outils\gui_launcher_tool.py")
content = target.read_text(encoding="utf-8")

# On change la description pour inclure l'ouverture de fichiers
old_desc = '''        return "Lance des applications graphiques (PyQt6, Tkinter) en arrière-plan sans bloquer l'interface. L'IA peut créer une application de bureau et l'exécuter directement sur la machine de l'utilisateur."'''
new_desc = '''        return "Lance des applications, ouvre des fichiers, des dossiers ou des images sur l'écran Windows. Utilise du code Python avec 'os.startfile()' pour ouvrir un fichier, ou lance une application PyQt6/Tkinter."'''

if old_desc in content:
    content = content.replace(old_desc, new_desc)
    print("✅ Description de l'outil GUI mise à jour.")

# On change aussi les capacités
old_caps = '''        return ["interface", "pyqt6", "pyqt", "fenêtre", "application de bureau", "gui", "tkinter", "lancer"]'''
new_caps = '''        return ["interface", "pyqt6", "pyqt", "fenêtre", "application", "gui", "tkinter", "lancer", "ouvrir", "afficher", "image", "dossier"]'''

if old_caps in content:
    content = content.replace(old_caps, new_caps)
    print("✅ Capacités de l'outil GUI mises à jour.")

# Backup
bak = str(target) + ".bak_gui_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy2(target, bak)
target.write_text(content, encoding="utf-8")
print(f"[OK] Outil GUI prêt. Backup: {bak}")