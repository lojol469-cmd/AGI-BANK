# APP_NAME: paie_calculator
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
from PyQt6.QtCore import Qt

class PaieCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculateur de Paie")
        self.setGeometry(100, 100, 300, 250)
        self.initUI()

    def initUI(self):
        self.nom_label = QLabel("Nom:", self)
        self.nom_input = QLineEdit(self)
        self.salaire_brut_label = QLabel("Salaire Brut:", self)
        self.salaire_brut_input = QLineEdit(self)
        self.primes_label = QLabel("Primes:", self)
        self.primes_input = QLineEdit(self)
        self.mois_label = QLabel("Mois:", self)
        self.mois_input = QLineEdit(self)
        self.calculer_button = QPushButton("Calculer", self)
        self.resultat_label = QLabel("", self)

        layout = QVBoxLayout()
        layout.addWidget(self.nom_label)
        layout.addWidget(self.nom_input)
        layout.addWidget(self.salaire_brut_label)
        layout.addWidget(self.salaire_brut_input)
        layout.addWidget(self.primes_label)
        layout.addWidget(self.primes_input)
        layout.addWidget(self.mois_label)
        layout.addWidget(self.mois_input)
        layout.addWidget(self.calculer_button)
        layout.addWidget(self.resultat_label)
        self.setLayout(layout)

        self.calculer_button.clicked.connect(self.calculer_paie)

    def calculer_paie(self):
        try:
            nom = self.nom_input.text()
            salaire_brut = float(self.salaire_brut_input.text())
            primes = float(self.primes_input.text())
            mois = self.mois_input.text()

            salaire_total = salaire_brut + primes
            cnss = salaire_total * 0.05
            irpp = salaire_total * 0.15
            net = salaire_total - cnss - irpp

            resultat = f"Nom: {nom}\nSalaire Brut: {salaire_brut}\nPrimes: {primes}\nSalaire Total: {salaire_total}\nCNSS (5%): {cnss}\nIRPP (15%): {irpp}\nNet à Payer: {net}"
            self.resultat_label.setText(resultat)
        except ValueError:
            QMessageBox.critical(self, "Erreur", "Veuillez entrer des valeurs numériques valides.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PaieCalculator()
    window.show()
    sys.exit(app.exec())