import sys
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle('Bonjour')
window.resize(200, 100)
label = QLabel('Bonjour', window)
label.move(50, 30)
window.show()
app.exec()