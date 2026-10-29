# APP_NAME: authenticator_app
import sys
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
from qrcode import make as qrcode_make
from io import BytesIO
from PIL import Image
import base64
import random
import string

class AuthenticatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Authenticator App')
        self.setGeometry(100, 100, 400, 500)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText('Username')
        self.layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText('Password')
        self.layout.addWidget(self.password_input)

        self.register_button = QPushButton('Register')
        self.register_button.clicked.connect(self.register)
        self.layout.addWidget(self.register_button)

        self.login_button = QPushButton('Login')
        self.login_button.clicked.connect(self.login)
        self.layout.addWidget(self.login_button)

        self.qr_code_button = QPushButton('Generate QR Code')
        self.qr_code_button.clicked.connect(self.generate_qr_code)
        self.layout.addWidget(self.qr_code_button)

        self.setLayout(self.layout)

    def register(self):
        username = self.username_input.text()
        password = self.password_input.text()
        if not username or not password:
            QMessageBox.critical(self, 'Error', 'Please enter both username and password')
            return
        # Simulate registration
        QMessageBox.information(self, 'Registration', f'Registered: {username}')

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        if not username or not password:
            QMessageBox.critical(self, 'Error', 'Please enter both username and password')
            return
        # Simulate login
        QMessageBox.information(self, 'Login', f'Logged in as: {username}')

    def generate_qr_code(self):
        # Simulate a secret key for the QR code
        secret_key = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        # Generate QR code
        qr = qrcode_make(secret_key)
        buffer = BytesIO()
        qr.save(buffer, format='PNG')
        img_data = buffer.getvalue()
        # Convert to base64
        base64_img = base64.b64encode(img_data).decode('utf-8')
        # Display QR code
        qr_label = QLabel()
        qr_label.setText(f'<img src="data:image/png;base64,{base64_img}">')
        qr_label.setOpenExternalLinks(True)
        self.layout.addWidget(qr_label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AuthenticatorApp()
    window.show()
    sys.exit(app.exec())