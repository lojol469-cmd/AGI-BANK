import sys
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Kibali ULTIME")

label = QLabel("Kibali ULTIME")
label.setStyleSheet("background-color: blue; color: white; padding: 10px; font-size: 24px; border: 2px solid white; border-radius: 10px;")

window.setLayout(label)
window.show()

sys.exit(app.exec())