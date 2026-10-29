"""
Outil de Génération de Code
Génère du code dans différents langages de programmation
"""

import re
import time
from typing import Dict, List, Any, Optional
from .__init__ import BaseTool

class CodeGenerationTool(BaseTool):
    """Outil spécialisé dans la génération de code"""

    @property
    def name(self) -> str:
        return "code_generator"

    @property
    def description(self) -> str:
        return "Générateur de code intelligent pour plusieurs langages"

    @property
    def capabilities(self) -> List[str]:
        return [
            "génération de code Python",
            "génération de code JavaScript",
            "génération de code Java",
            "génération de code C++",
            "génération de code SQL",
            "génération de code HTML/CSS",
            "optimisation de code",
            "debugging de code",
            "documentation de code",
            "tests unitaires"
        ]

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        """Détermine si la requête concerne la génération de code"""
        code_keywords = [
            "code", "programme", "fonction", "classe", "méthode",
            "script", "algorithme", "coder", "développer",
            "python", "javascript", "java", "c++", "sql", "html", "css",
            "bug", "erreur", "debug", "optimiser", "améliorer",
            "test", "unittest", "documentation"
        ]

        programming_languages = [
            "python", "javascript", "java", "c++", "c#", "php", "ruby",
            "go", "rust", "swift", "kotlin", "typescript", "sql",
            "html", "css", "bash", "shell"
        ]

        query_lower = query.lower()

        # Mots-clés de programmation
        score = 0.0
        for keyword in code_keywords:
            if keyword in query_lower:
                score += 0.3

        # Langages de programmation mentionnés
        for lang in programming_languages:
            if lang in query_lower:
                score += 0.4

        # Patterns de code
        code_patterns = [
            r'fonction\s+\w+',  # Définition de fonction
            r'class\s+\w+',     # Définition de classe
            r'def\s+\w+',       # Fonction Python
            r'function\s+\w+',  # Fonction JS
            r'public\s+class',  # Classe Java
            r'#include',        # C++
            r'import\s+\w+',    # Import
            r'print\(|console\.log\(',  # Instructions d'affichage
        ]

        for pattern in code_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.5

        # Contexte de développement
        if context and any(key in context for key in ['language', 'framework', 'project_type']):
            score += 0.3

        return min(score, 1.0)

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Génère ou optimise du code"""
        try:
            # Identifier le type de tâche
            task_type = self._identify_task_type(query)

            # Détecter le langage
            language = self._detect_language(query, context)

            if task_type == "generate_function":
                result = self._generate_function(query, language)
            elif task_type == "generate_class":
                result = self._generate_class(query, language)
            elif task_type == "generate_algorithm":
                result = self._generate_algorithm(query, language)
            elif task_type == "optimize_code":
                result = self._optimize_code(query, language)
            elif task_type == "debug_code":
                result = self._debug_code(query, language)
            elif task_type == "generate_tests":
                result = self._generate_tests(query, language)
            else:
                result = self._generate_general_code(query, language)

            return {
                'query': query,
                'task_type': task_type,
                'language': language,
                'generated_code': result,
                'timestamp': time.time()
            }

        except Exception as e:
            return {
                'error': f"Erreur lors de la génération de code: {str(e)}",
                'query': query
            }

    def _identify_task_type(self, query: str) -> str:
        """Identifie le type de tâche demandée"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["fonction", "function", "def"]):
            return "generate_function"
        elif any(word in query_lower for word in ["classe", "class"]):
            return "generate_class"
        elif any(word in query_lower for word in ["algorithme", "algorithm", "trier", "chercher"]):
            return "generate_algorithm"
        elif any(word in query_lower for word in ["optimiser", "optimize", "améliorer", "improve"]):
            return "optimize_code"
        elif any(word in query_lower for word in ["bug", "erreur", "debug", "corriger"]):
            return "debug_code"
        elif any(word in query_lower for word in ["test", "unittest", "testing"]):
            return "generate_tests"
        else:
            return "generate_general"

    def _detect_language(self, query: str, context: Dict[str, Any] = None) -> str:
        """Détecte le langage de programmation demandé"""
        # Contexte explicite
        if context and 'language' in context:
            return context['language'].lower()

        # Détection dans la requête
        languages = {
            'python': ['python', 'py', 'django', 'flask', 'pandas', 'numpy'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue', 'angular'],
            'java': ['java', 'spring', 'maven', 'gradle'],
            'cpp': ['c++', 'cpp', 'qt', 'opencv'],
            'csharp': ['c#', 'csharp', '.net', 'unity'],
            'php': ['php', 'laravel', 'symfony'],
            'sql': ['sql', 'mysql', 'postgresql', 'sqlite'],
            'html': ['html', 'css', 'bootstrap', 'tailwind'],
            'bash': ['bash', 'shell', 'script', 'linux']
        }

        query_lower = query.lower()

        for lang, keywords in languages.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return lang

        # Langage par défaut
        return 'python'

    def _generate_function(self, query: str, language: str) -> Dict[str, Any]:
        """Génère une fonction"""
        # Extraire le nom de la fonction et ses paramètres
        func_name = self._extract_function_name(query)
        params = self._extract_parameters(query)

        if language == 'python':
            code = self._generate_python_function(func_name, params, query)
        elif language == 'javascript':
            code = self._generate_js_function(func_name, params, query)
        elif language == 'java':
            code = self._generate_java_function(func_name, params, query)
        else:
            code = self._generate_python_function(func_name, params, query)

        return {
            'function_name': func_name,
            'parameters': params,
            'code': code,
            'language': language
        }

    def _generate_class(self, query: str, language: str) -> Dict[str, Any]:
        """Génère une classe"""
        class_name = self._extract_class_name(query)

        if language == 'python':
            code = self._generate_python_class(class_name, query)
        elif language == 'javascript':
            code = self._generate_js_class(class_name, query)
        elif language == 'java':
            code = self._generate_java_class(class_name, query)
        else:
            code = self._generate_python_class(class_name, query)

        return {
            'class_name': class_name,
            'code': code,
            'language': language
        }

    def _generate_algorithm(self, query: str, language: str) -> Dict[str, Any]:
        """Génère un algorithme"""
        algorithm_type = self._identify_algorithm_type(query)

        if algorithm_type == "sort":
            return self._generate_sort_algorithm(language)
        elif algorithm_type == "search":
            return self._generate_search_algorithm(language)
        elif algorithm_type == "fibonacci":
            return self._generate_fibonacci_algorithm(language)
        else:
            return self._generate_generic_algorithm(query, language)

    def _generate_python_function(self, name: str, params: List[str], query: str) -> str:
        """Génère une fonction Python"""
        params_str = ', '.join(params) if params else ''

        # Logique basée sur la requête
        if 'addition' in query.lower() or 'somme' in query.lower():
            body = f'    return {" + ".join(params) if params else "0"}'
        elif 'multiplication' in query.lower():
            body = f'    return {" * ".join(params) if params else "1"}'
        elif 'factorielle' in query.lower():
            body = '''    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)'''
        else:
            body = '    # TODO: Implémenter la logique\n    pass'

        return f'''def {name}({params_str}):
{body}'''

    def _generate_js_function(self, name: str, params: List[str], query: str) -> str:
        """Génère une fonction JavaScript"""
        params_str = ', '.join(params) if params else ''

        if 'addition' in query.lower() or 'somme' in query.lower():
            body = f'    return {(" + ".join(params)) if params else "0"};'
        else:
            body = '    // TODO: Implémenter la logique\n    return null;'

        return f'''function {name}({params_str}) {{
{body}
}}'''

    def _generate_java_function(self, name: str, params: List[str], query: str) -> str:
        """Génère une méthode Java"""
        params_str = ', '.join([f'int {p}' for p in params]) if params else ''

        if 'addition' in query.lower() or 'somme' in query.lower():
            body = f'        return {" + ".join(params)};'
        else:
            body = '        // TODO: Implémenter la logique\n        return 0;'

        return f'''public static int {name}({params_str}) {{
{body}
}}'''

    def _generate_python_class(self, name: str, query: str) -> str:
        """Génère une classe Python"""
        return f'''class {name}:
    """Classe {name} générée automatiquement"""

    def __init__(self):
        """Constructeur"""
        pass

    def method_example(self):
        """Méthode d'exemple"""
        return "Hello from {name}"'''

    def _generate_sort_algorithm(self, language: str) -> Dict[str, Any]:
        """Génère un algorithme de tri"""
        if language == 'python':
            code = '''def bubble_sort(arr):
    """Tri à bulles"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr'''
        else:
            code = "# Algorithme de tri non implémenté pour ce langage"

        return {
            'algorithm': 'bubble_sort',
            'code': code,
            'language': language
        }

    def _extract_function_name(self, query: str) -> str:
        """Extrait le nom de la fonction de la requête"""
        # Patterns pour extraire le nom de fonction
        patterns = [
            r'fonction\s+(\w+)',
            r'function\s+(\w+)',
            r'def\s+(\w+)',
            r'créer\s+(\w+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)

        return 'ma_fonction'

    def _extract_parameters(self, query: str) -> List[str]:
        """Extrait les paramètres de la requête"""
        # Chercher les paramètres mentionnés
        param_patterns = [
            r'paramètre[s]?\s*:\s*([^.]+)',
            r'avec\s+([^.]+)',
            r'(\w+(?:\s*,\s*\w+)*)\s+en\s+paramètre'
        ]

        for pattern in param_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params_text = match.group(1)
                # Extraire les noms de paramètres
                params = re.findall(r'\b\w+\b', params_text)
                return [p for p in params if len(p) > 1][:5]  # Max 5 paramètres

        # Paramètres par défaut
        return ['param1', 'param2']

    def _extract_class_name(self, query: str) -> str:
        """Extrait le nom de la classe"""
        patterns = [
            r'classe\s+(\w+)',
            r'class\s+(\w+)',
            r'créer\s+la\s+classe\s+(\w+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)

        return 'MaClasse'

    def _identify_algorithm_type(self, query: str) -> str:
        """Identifie le type d'algorithme demandé"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['trier', 'sort', 'tri']):
            return 'sort'
        elif any(word in query_lower for word in ['chercher', 'search', 'recherche']):
            return 'search'
        elif 'fibonacci' in query_lower:
            return 'fibonacci'
        else:
            return 'generic'

    def _optimize_code(self, query: str, language: str) -> Dict[str, Any]:
        """Optimise du code existant"""
        # Extraire le code de la requête
        code_match = re.search(r'```(?:\w+)?\n(.*?)\n```', query, re.DOTALL)
        if code_match:
            original_code = code_match.group(1)
        else:
            original_code = "Code non trouvé dans la requête"

        # Suggestions d'optimisation génériques
        suggestions = [
            "Utiliser des comprehensions de liste au lieu des boucles",
            "Éviter les variables globales",
            "Utiliser des générateurs pour les grandes séquences",
            "Optimiser les requêtes de base de données",
            "Utiliser la mise en cache quand approprié"
        ]

        return {
            'original_code': original_code,
            'optimizations': suggestions,
            'language': language
        }

    def _debug_code(self, query: str, language: str) -> Dict[str, Any]:
        """Aide au debugging"""
        # Analyser les erreurs communes mentionnées
        error_patterns = {
            'indentation': ['indentation', 'indent'],
            'syntax': ['syntax', 'erreur de syntaxe'],
            'name': ['name', 'not defined'],
            'type': ['type', ' incompatible'],
            'index': ['index', 'out of range']
        }

        detected_errors = []
        for error_type, patterns in error_patterns.items():
            for pattern in patterns:
                if pattern in query.lower():
                    detected_errors.append(error_type)

        suggestions = []
        for error in detected_errors:
            if error == 'indentation':
                suggestions.append("Vérifier l'indentation (espaces vs tabulations)")
            elif error == 'syntax':
                suggestions.append("Vérifier la syntaxe: parenthèses, guillemets, virgules")
            elif error == 'name':
                suggestions.append("Vérifier que toutes les variables sont définies")
            elif error == 'type':
                suggestions.append("Vérifier les types de données utilisés")
            elif error == 'index':
                suggestions.append("Vérifier les indices des tableaux/listes")

        return {
            'detected_errors': detected_errors,
            'debugging_suggestions': suggestions,
            'language': language
        }

    def _generate_tests(self, query: str, language: str) -> Dict[str, Any]:
        """Génère des tests unitaires"""
        # Extraire le nom de la fonction/classe à tester
        test_target = self._extract_test_target(query)

        if language == 'python':
            test_code = self._generate_python_tests(test_target)
        else:
            test_code = f"# Tests pour {test_target} - Langage {language} non supporté"

        return {
            'test_target': test_target,
            'test_code': test_code,
            'language': language
        }

    def _extract_test_target(self, query: str) -> str:
        """Extrait la cible des tests"""
        patterns = [
            r'test(?:er)?\s+(?:la\s+)?(?:fonction\s+)?(\w+)',
            r'tests?\s+pour\s+(\w+)',
            r'(\w+)\s+tests?'
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)

        return 'ma_fonction'

    def _generate_python_tests(self, target: str) -> str:
        """Génère des tests Python avec unittest"""
        return f'''import unittest

class Test{target.title()}(unittest.TestCase):
    """Tests unitaires pour {target}"""

    def setUp(self):
        """Configuration des tests"""
        pass

    def test_{target}_basic(self):
        """Test basique"""
        result = {target}(1, 2)
        self.assertEqual(result, 3)

    def test_{target}_edge_cases(self):
        """Test des cas limites"""
        # TODO: Ajouter des tests pour les cas limites
        pass

    def tearDown(self):
        """Nettoyage après les tests"""
        pass

if __name__ == '__main__':
    unittest.main()'''

    def _generate_general_code(self, query: str, language: str) -> Dict[str, Any]:
        """Génère du code général"""
        return {
            'code': f'# Code généré pour {language}\n# Requête: {query}\n\n# TODO: Implémenter',
            'language': language,
            'note': 'Code générique - veuillez préciser la tâche'
        }