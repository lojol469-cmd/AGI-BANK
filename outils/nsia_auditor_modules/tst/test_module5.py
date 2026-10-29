# -*- coding: utf-8 -*-
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_outils_dir = os.path.dirname(os.path.dirname(_current_dir))
if _outils_dir not in sys.path:
    sys.path.insert(0, _outils_dir)

from nsia_auditor_modules.chat_session import NSIAChatSession

def main():
    print("="*70)
    print("=== TEST MODULE 5 : CHAT TEMPS REEL (10 QUESTIONS) ===")
    print("="*70)
    
    session = NSIAChatSession()
    
    # 10 questions variees : stats, deterministes, semantiques, rapport
    questions = [
        ("Combien de comptes au total ?", False),
        ("Combien de clients dans le dataset ?", False),
        ("Quelle est la moyenne des soldes ?", False),
        ("Trouve les comptes avec un pret superieur a 200000", False),
        ("Parmi eux, lesquels sont a risque ?", False),  # test contexte
        ("Montre-moi les bons clients", False),
        ("Liste les 5 plus gros prets", False),
        ("Y a-t-il des comptes avec solde negatif ?", False),
        ("Total des prets sur le portefeuille", False),
        ("Genere un rapport PDF pour les comptes a risque", True),
    ]
    
    for i, (q, gen_rep) in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"[Q{i}/10] {q}")
        print('='*70)
        
        result = session.ask(q, generate_report=gen_rep)
        
        # Affichage tronque pour clarte
        answer = result['answer']
        if len(answer) > 500:
            answer = answer[:500] + "\n... [tronque]"
        print(answer)
    
    # Resume final
    print(f"\n{'='*70}")
    print("=== RESUME DE LA SESSION ===")
    print('='*70)
    print(session.summary())
    
    print(f"\n[SESSION] {len(session.get_history())} echanges memorises.")
    print(f"[SESSION] Toutes les questions ont ete traitees avec succes.")

if __name__ == "__main__":
    main()