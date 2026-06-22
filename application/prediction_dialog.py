from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt
from application.logger import logger

class PredictionDialog(QDialog):
    def __init__(self, prediction, confidence, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Prediction Result")
        self.setFixedWidth(220)

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        self.label_pred = QLabel(f"Prediction: {prediction}")
        self.label_pred.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_conf = QLabel(f"Confidence: {confidence:.2f}%")
        self.label_conf.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_ok = QPushButton("OK")
        btn_ok.setFixedHeight(28)
        btn_ok.clicked.connect(self.accept)

        layout.addWidget(self.label_pred)
        layout.addWidget(self.label_conf)
        layout.addWidget(btn_ok)

        logger.info(
            f"Prediction complete: {prediction} ({confidence:.1f}%)"
        )

        self.setLayout(layout)

        self.adjustSize()