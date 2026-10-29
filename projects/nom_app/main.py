import sys
from PyQt6 import QtWidgets, QtCore

class TVACalculator(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculateur de TVA")
        self.setGeometry(100, 100, 300, 200)

        self.label_ht = QtWidgets.QLabel("Montant HT :")
        self.input_ht = QtWidgets.QLineEdit()
        self.button_calculer = QtWidgets.QPushButton("Calculer")
        self.label_ttc = QtWidgets.QLabel("Montant TTC :")
        self.output_ttc = QtWidgets.QLabel("")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label_ht)
        layout.addWidget(self.input_ht)
        layout.addWidget(self.button_calculer)
        layout.addWidget(self.label_ttc)
        layout.addWidget(self.output_ttc)

        self.button_calculer.clicked.connect(self.calculer_tva)
        self.setLayout(layout)

    def calculer_tva(self):
        try:
            montant_ht = float(self.input_ht.text())
            tva = montant_ht * 0.18
            montant_ttc = montant_ht + tva
            self.output_ttc.setText(f"{montant_ttc:.2f}")
        except ValueError:
            self.output_ttc.setText("Veuillez entrer un nombre valide")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TVACalculator()
    window.show()
    sys.exit(app.exec())