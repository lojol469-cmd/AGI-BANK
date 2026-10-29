import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout

class VRAMCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculatrice VRAM")
        self.setGeometry(100, 100, 300, 200)
        self.initUI()
    
    def initUI(self):
        self.label = QLabel("Résultat VRAM:")
        self.input1 = QLineEdit("0")
        self.input2 = QLineEdit("0")
        self.button = QPushButton("Calculer VRAM")
        self.button.clicked.connect(self.calculate)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.input1)
        layout.addWidget(self.input2)
        layout.addWidget(self.button)
        self.setLayout(layout)
    
    def calculate(self):
        try:
            val1 = float(self.input1.text())
            val2 = float(self.input2.text())
            result = val1 + val2  # Exemple de calcul simple
            self.label.setText(f"Résultat VRAM: {result} Mo")
        except ValueError:
            self.label.setText("Veuillez entrer des valeurs numériques")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    calc = VRAMCalculator()
    calc.show()
    sys.exit(app.exec())