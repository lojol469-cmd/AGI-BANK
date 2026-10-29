C'est une excellente idée. Le `ToolManager` est le cœur de ton AGI, et si un seul outil ne respecte pas le "contrat", tout le système de chargement plante (comme on l'a vu avec `RagBridgeTool`). 

Voici le **README officiel** pour ton dossier `outils/`. Tu peux le sauvegarder sous le nom `README_TOOLS.md` dans `D:\kibali-IA\outils\`. Il servira de référence absolue pour toi, mais aussi pour l'AGI si elle décide un jour de coder ses propres outils.

***

# 📘 KIBALI AGI - Guide de Création d'Outils (ToolManager)

Ce document décrit **exactement** le contrat que chaque fichier `.py` doit respecter pour être chargé automatiquement par le `ToolManager` et être utilisable par l'Intelligence Artificielle.

## 1. Structure du Dossier
Tous les outils doivent être placés dans le dossier :
`D:\kibali-IA\outils\`

Le `ToolManager` scanne ce dossier au démarrage. Chaque fichier `.py` est inspecté.

## 2. Le Contrat du `BaseTool` (Obligatoire)

Pour être valide, une classe doit **impérativement** :
1. Importer la classe de base : `from . import BaseTool`
2. Hériter de `BaseTool` : `class MonOutil(BaseTool):`
3. Implémenter les **3 propriétés** et les **2 méthodes** abstraites (sinon Python lèvera l'erreur `Can't instantiate abstract class`).

### A. Les Propriétés (Ce que l'IA voit)

*   **`name`** (`str`) : Le nom unique de l'outil (ex: `"execute_python"`, `"rag_bridge"`). Utilisé par l'IA pour appeler l'outil.
*   **`description`** (`str`) : **LA PLUS IMPORTANTE**. C'est le texte que l'IA lit pour comprendre *quand* et *comment* utiliser cet outil. Sois précis.
*   **`capabilities`** (`List[str]`) : Une liste de mots-clés (ex: `["fichier", "lecture", "system"]`). Utilisée par le `ToolManager` pour filtrer les outils pertinents.
*   **`parameters`** (`Dict`) : Le schéma JSON (format OpenAI) décrivant les arguments attendus par la fonction. Si l'outil n'a pas besoin d'arguments, retourne `{"type": "object", "properties": {}}`.

### B. Les Méthodes (Ce que l'IA fait)

*   **`can_handle(self, query: str, context: dict = None) -> float`** :
    *   Reçoit la question de l'utilisateur.
    *   Doit retourner un **score entre 0.0 et 1.0**.
    *   Si l'outil est très pertinent pour la question, retourne `0.9` ou `0.95`. Sinon, `0.0` ou `0.1`.
*   **`execute(self, **kwargs) -> dict`** :
    *   C'est la logique réelle de l'outil.
    *   Reçoit les arguments définis dans `parameters` (ex: `query="..."`, ou `action="search"`).
    *   **Doit TOUJOURS retourner un dictionnaire** contenant au minimum une clé `"success"` (booléen) et `"output"` (ce que l'IA va lire, sous forme de texte ou de JSON).

---

## 3. Le Template Parfait (Copier-Coller)

Voici le squelette exact à utiliser pour créer un nouvel outil :

```python
# -*- coding: utf-8 -*-
import os
import sys
from typing import Dict, Any, List

# 1. Sécurité pour les imports relatifs (NE PAS TOUCHER)
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# 2. Import de la classe de base
from . import BaseTool

# 3. Déclaration de la classe
class MonSuperOutil(BaseTool):
    
    # --- IDENTITÉ DE L'OUTIL ---
    
    @property
    def name(self) -> str:
        return "mon_super_outil"  # Doit être unique

    @property
    def description(self) -> str:
        return """Description claire de ce que fait l'outil.
        Explique à l'IA QUAND l'utiliser et COMMENT passer les arguments.
        Ex: 'Permet de lire un fichier. action="read" (args: path).'"""

    @property
    def capabilities(self) -> List[str]:
        return ["lecture", "fichier", "texte"]

    @property
    def parameters(self) -> Dict[str, Any]:
        # Schéma JSON des arguments
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string", 
                    "enum": ["read", "write"],
                    "description": "L'action à effectuer."
                },
                "file_path": {
                    "type": "string", 
                    "description": "Le chemin du fichier."
                }
            },
            "required": ["action"] # Arguments obligatoires
        }

    # --- LOGIQUE D'EXÉCUTION ---

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        q = query.lower()
        # Si l'utilisateur parle de fichier ou de lecture, on donne un score élevé
        if any(k in q for k in ["fichier", "lire", "écrire", "document"]):
            return 0.9
        return 0.0

    def execute(self, action: str, file_path: str = "", **kwargs) -> Dict[str, Any]:
        # Le ToolManager injecte les arguments du JSON ici.
        # Le **kwargs permet d'absorber les arguments en trop sans crasher.
        
        try:
            if action == "read":
                if not os.path.exists(file_path):
                    return {"success": False, "output": f"❌ Fichier introuvable : {file_path}"}
                
                with open(file_path, "r", encoding="utf-8") as f:
                    contenu = f.read()
                return {"success": True, "output": f"Contenu du fichier :\n{contenu}"}
                
            elif action == "write":
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(kwargs.get("content", "vide"))
                return {"success": True, "output": f"✅ Fichier écrit : {file_path}"}
                
            else:
                return {"success": False, "output": f"Action inconnue : {action}"}
                
        except Exception as e:
            # Toujours capturer les erreurs pour ne pas crasher l'AGI
            return {"success": False, "output": f"Erreur lors de l'exécution : {str(e)}"}
```

---

## 4. Les 3 Règles d'Or (À ne jamais enfreindre)

1. **Ne jamais oublier `capabilities` et `can_handle`** : C'est ce qui a fait planter `RagBridgeTool` la dernière fois. Python refuse de créer un outil si ces méthodes manquent.
2. **Toujours retourner un dictionnaire** : La méthode `execute` doit renvoyer un format comme `{"success": True, "output": "résultat"}`. Si vous renvoyez une chaîne de caractères brute, l'interface Gradio va afficher une erreur de type.
3. **Gérer les erreurs avec `try/except`** : Si votre code plante (ex: division par zéro), l'erreur doit être capturée et renvoyée dans `"output"` pour que l'IA puisse lire l'erreur et décider de réessayer autrement. Ne laissez jamais Python générer une exception non gérée.

---

## 5. Comment l'IA utilise vos outils

1. Au démarrage, le `ToolManager` lit `description` et `parameters` de tous les outils.
2. L'utilisateur pose une question.
3. L'AGI (Qwen/DeepSeek) choisit un outil et génère un JSON avec les arguments.
4. Le `ToolManager` appelle la fonction `execute(**arguments_json)`.
5. Le résultat (le dictionnaire `{"success": ..., "output": ...}`) est renvoyé à l'AGI.
6. L'AGI lit le contenu de `"output"` et formule sa réponse finale.

**C'est aussi simple que ça.** Avec ce README, tu as la bible pour étendre ton AGI à l'infini sans jamais casser le système.