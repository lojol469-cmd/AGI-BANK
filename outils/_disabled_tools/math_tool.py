"""
Outil de Calcul Mathématique
Effectue des calculs mathématiques avancés et résout des équations
"""

import re
import math
import time
import sympy as sp
from typing import Dict, List, Any, Optional
from .__init__ import BaseTool

class MathTool(BaseTool):
    """Outil spécialisé dans les calculs mathématiques"""

    @property
    def name(self) -> str:
        return "math_calculator"

    @property
    def description(self) -> str:
        return "Calculateur mathématique avancé avec résolution d'équations"

    @property
    def capabilities(self) -> List[str]:
        return [
            "calculs arithmétiques",
            "algèbre",
            "résolution d'équations",
            "calculus",
            "géométrie",
            "statistiques",
            "trigonométrie",
            "logarithmes",
            "dérivées",
            "intégrales"
        ]

    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        """Détermine si la requête nécessite des calculs mathématiques"""
        math_keywords = [
            "calculer", "calcule", "résoudre", "équation",
            "intégrale", "dérivée", "limite", "factorielle",
            "racine", "puissance", "logarithme", "exponentielle",
            "sinus", "cosinus", "tangente", "trigonométrie",
            "algèbre", "géométrie", "statistiques", "probabilités",
            "+", "-", "*", "/", "=", "x²", "π", "√"
        ]

        math_patterns = [
            r'\d+[\+\-\*\/]\d+',  # Opérations arithmétiques
            r'x\s*[\+\-\*\/]\s*\d+',  # Équations simples
            r'\d+\s*\^\s*\d+',  # Puissances
            r'sqrt\(.+\)',  # Racines carrées
            r'sin\(|cos\(|tan\(',  # Fonctions trigonométriques
            r'∫|∂|∑|∏'  # Symboles mathématiques
        ]

        query_lower = query.lower()

        # Mots-clés mathématiques
        score = 0.0
        for keyword in math_keywords:
            if keyword in query_lower:
                score += 0.3

        # Patterns mathématiques
        for pattern in math_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.4

        # Questions mathématiques explicites
        math_questions = ["combien", "quel est le résultat", "résous", "calcule"]
        for question in math_questions:
            if question in query_lower:
                score += 0.3

        return min(score, 1.0)

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Effectue les calculs mathématiques demandés"""
        try:
            # Analyser la requête pour identifier le type de calcul
            calculation_type = self._identify_calculation_type(query)

            if calculation_type == "arithmetic":
                result = self._perform_arithmetic(query)
            elif calculation_type == "equation":
                result = self._solve_equation(query)
            elif calculation_type == "symbolic":
                result = self._perform_symbolic_math(query)
            elif calculation_type == "geometry":
                result = self._perform_geometry(query)
            else:
                result = self._perform_general_math(query)

            return {
                'query': query,
                'calculation_type': calculation_type,
                'result': result,
                'timestamp': time.time()
            }

        except Exception as e:
            return {
                'error': f"Erreur lors du calcul: {str(e)}",
                'query': query
            }

    def _identify_calculation_type(self, query: str) -> str:
        """Identifie le type de calcul demandé"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["équation", "résoudre", "="]):
            return "equation"
        elif any(word in query_lower for word in ["intégrale", "dérivée", "limite"]):
            return "symbolic"
        elif any(word in query_lower for word in ["aire", "volume", "périmètre", "angle"]):
            return "geometry"
        elif re.search(r'\d+[\+\-\*\/]\d+', query):
            return "arithmetic"
        else:
            return "general"

    def _perform_arithmetic(self, query: str) -> Dict[str, Any]:
        """Effectue des calculs arithmétiques simples"""
        # Extraire l'expression mathématique
        expression = self._extract_math_expression(query)

        if not expression:
            return {'error': 'Aucune expression mathématique trouvée'}

        try:
            # Évaluer l'expression de manière sécurisée
            result = self._safe_eval(expression)

            return {
                'expression': expression,
                'result': result,
                'type': 'arithmetic'
            }
        except Exception as e:
            return {'error': f'Erreur calcul: {str(e)}'}

    def _solve_equation(self, query: str) -> Dict[str, Any]:
        """Résout des équations"""
        try:
            # Utiliser sympy pour résoudre les équations
            x = sp.Symbol('x')

            # Extraire l'équation (simplifié)
            equation_str = self._extract_equation(query)
            if not equation_str:
                return {'error': 'Équation non reconnue'}

            # Parser et résoudre
            equation = sp.Eq(sp.sympify(equation_str.split('=')[0]),
                           sp.sympify(equation_str.split('=')[1]))

            solutions = sp.solve(equation, x)

            return {
                'equation': equation_str,
                'solutions': [str(sol) for sol in solutions],
                'type': 'equation'
            }
        except Exception as e:
            return {'error': f'Erreur résolution équation: {str(e)}'}

    def _perform_symbolic_math(self, query: str) -> Dict[str, Any]:
        """Effectue des calculs symboliques (dérivées, intégrales, etc.)"""
        try:
            x = sp.Symbol('x')

            if "dérivée" in query.lower() or "derive" in query.lower():
                # Extraire la fonction
                func_match = re.search(r'dérivée\s+de\s+(.+)', query, re.IGNORECASE)
                if func_match:
                    func_str = func_match.group(1).strip()
                    func = sp.sympify(func_str)
                    derivative = sp.diff(func, x)

                    return {
                        'operation': 'derivative',
                        'function': func_str,
                        'result': str(derivative),
                        'type': 'symbolic'
                    }

            elif "intégrale" in query.lower():
                # Calcul d'intégrale
                func_match = re.search(r'intégrale\s+de\s+(.+)', query, re.IGNORECASE)
                if func_match:
                    func_str = func_match.group(1).strip()
                    func = sp.sympify(func_str)
                    integral = sp.integrate(func, x)

                    return {
                        'operation': 'integral',
                        'function': func_str,
                        'result': str(integral),
                        'type': 'symbolic'
                    }

            return {'error': 'Opération symbolique non reconnue'}

        except Exception as e:
            return {'error': f'Erreur calcul symbolique: {str(e)}'}

    def _perform_geometry(self, query: str) -> Dict[str, Any]:
        """Effectue des calculs géométriques"""
        query_lower = query.lower()

        try:
            if "cercle" in query_lower:
                return self._calculate_circle(query)
            elif "rectangle" in query_lower or "carré" in query_lower:
                return self._calculate_rectangle(query)
            elif "triangle" in query_lower:
                return self._calculate_triangle(query)
            else:
                return {'error': 'Forme géométrique non reconnue'}

        except Exception as e:
            return {'error': f'Erreur calcul géométrique: {str(e)}'}

    def _perform_general_math(self, query: str) -> Dict[str, Any]:
        """Calculs mathématiques généraux"""
        # Tenter différents types de calculs
        results = []

        # Essayer l'arithmétique
        arith_result = self._perform_arithmetic(query)
        if 'result' in arith_result:
            results.append(arith_result)

        # Essayer les fonctions mathématiques communes
        func_results = self._calculate_common_functions(query)
        if func_results:
            results.extend(func_results)

        if results:
            return {
                'results': results,
                'type': 'general'
            }
        else:
            return {'error': 'Type de calcul non reconnu'}

    def _extract_math_expression(self, query: str) -> str:
        """Extrait l'expression mathématique de la requête"""
        # Patterns pour extraire les expressions
        patterns = [
            r'(\d+(?:\.\d+)?[\+\-\*\/]\d+(?:\.\d+)?(?:[\+\-\*\/]\d+(?:\.\d+)?)*)',  # Opérations
            r'(\w+\([^)]+\))',  # Fonctions comme sin(30)
            r'(\d+(?:\.\d+)?(?:\^\d+(?:\.\d+)?)+)'  # Puissances
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)

        return ""

    def _extract_equation(self, query: str) -> str:
        """Extrait l'équation de la requête"""
        # Chercher les patterns d'équations
        equation_patterns = [
            r'(\w+)\s*=\s*(.+)',  # x = 5
            r'(.+)\s*=\s*(.+)'    # expressions = expressions
        ]

        for pattern in equation_patterns:
            match = re.search(pattern, query)
            if match:
                return f"{match.group(1)}={match.group(2)}"

        return ""

    def _safe_eval(self, expression: str) -> float:
        """Évalue une expression mathématique de manière sécurisée"""
        # Dictionnaire des fonctions autorisées
        safe_dict = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
            'pi': math.pi, 'e': math.e, 'abs': abs,
            'pow': pow, 'round': round
        }

        # Remplacer ^ par **
        expression = expression.replace('^', '**')

        # Évaluer avec restrictions
        return eval(expression, {"__builtins__": {}}, safe_dict)

    def _calculate_common_functions(self, query: str) -> List[Dict[str, Any]]:
        """Calcule les fonctions mathématiques communes"""
        results = []

        # Factorielle
        fact_match = re.search(r'(\d+)!', query)
        if fact_match:
            n = int(fact_match.group(1))
            results.append({
                'operation': 'factorial',
                'input': n,
                'result': math.factorial(n)
            })

        # Fonctions trigonométriques
        trig_patterns = [
            (r'sin\((\d+(?:\.\d+)?)\)', 'sin', math.sin),
            (r'cos\((\d+(?:\.\d+)?)\)', 'cos', math.cos),
            (r'tan\((\d+(?:\.\d+)?)\)', 'tan', math.tan)
        ]

        for pattern, name, func in trig_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                angle_deg = float(match.group(1))
                angle_rad = math.radians(angle_deg)
                results.append({
                    'operation': name,
                    'input_degrees': angle_deg,
                    'input_radians': angle_rad,
                    'result': func(angle_rad)
                })

        return results

    def _calculate_circle(self, query: str) -> Dict[str, Any]:
        """Calculs pour les cercles"""
        # Extraire le rayon
        radius_match = re.search(r'rayon\s*(?:de\s*)?(\d+(?:\.\d+)?)', query, re.IGNORECASE)
        if radius_match:
            radius = float(radius_match.group(1))

            return {
                'shape': 'circle',
                'radius': radius,
                'area': math.pi * radius ** 2,
                'circumference': 2 * math.pi * radius,
                'type': 'geometry'
            }

        return {'error': 'Rayon non spécifié pour le cercle'}

    def _calculate_rectangle(self, query: str) -> Dict[str, Any]:
        """Calculs pour les rectangles"""
        # Extraire longueur et largeur
        dimensions = re.findall(r'(\d+(?:\.\d+)?)', query)
        if len(dimensions) >= 2:
            length = float(dimensions[0])
            width = float(dimensions[1])

            return {
                'shape': 'rectangle',
                'length': length,
                'width': width,
                'area': length * width,
                'perimeter': 2 * (length + width),
                'type': 'geometry'
            }

        return {'error': 'Dimensions non spécifiées pour le rectangle'}

    def _calculate_triangle(self, query: str) -> Dict[str, Any]:
        """Calculs pour les triangles"""
        # Extraire les côtés
        sides = re.findall(r'(\d+(?:\.\d+)?)', query)
        if len(sides) >= 3:
            a, b, c = float(sides[0]), float(sides[1]), float(sides[2])

            # Calculer l'aire avec la formule de Heron
            s = (a + b + c) / 2
            area = math.sqrt(s * (s - a) * (s - b) * (s - c))

            return {
                'shape': 'triangle',
                'sides': [a, b, c],
                'perimeter': a + b + c,
                'area': area,
                'type': 'geometry'
            }

        return {'error': 'Côtés non spécifiés pour le triangle'}