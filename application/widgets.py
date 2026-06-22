from PyQt6.QtWidgets import QLabel, QWidget, QPushButton, QHBoxLayout
from PyQt6.QtCore import pyqtSignal, Qt

class ClickableLabel(QLabel):
    clicked = pyqtSignal(int)
    rightClicked = pyqtSignal(int, object)

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.selected = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)

        elif event.button() == Qt.MouseButton.RightButton:
            self.rightClicked.emit(self.index, event.globalPosition())

    def enterEvent(self, event):
        if not hasattr(self, "selected") or not self.selected:
            self.setStyleSheet("background-color: rgba(200,200,200,50);")

    def leaveEvent(self, event):
        if not hasattr(self, "selected") or not self.selected:
            self.setStyleSheet("")
    
class PeakTagWidget(QWidget):
    remove_requested = pyqtSignal(int)  # emits peak_id

    def __init__(self, peak_id, coords, parent=None):
        super().__init__(parent)

        self.peak_id = peak_id
        self.coords = coords

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        # Label text (coords-based naming)
        x, y = coords
        self.label = QLabel(f"Peak ({x}, {y})")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Optional tooltip (nice UX)
        self.setToolTip(f"X: {x}\nY: {y}")

        # Remove button
        self.remove_button = QPushButton("✕")
        self.remove_button.setFixedSize(16, 16)
        self.remove_button.setCursor(Qt.CursorShape.PointingHandCursor)

        # Styling (clean tag look)
        self.setStyleSheet("""
            PeakTagWidget {
                background-color: rgba(0, 120, 215, 40);
                border: 1px solid rgba(0, 120, 215, 120);
                border-radius: 10px;
            }
            QPushButton {
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                color: red;
            }
        """)

        # Connect remove action
        self.remove_button.clicked.connect(self.emit_remove)

        # Add to layout
        layout.addWidget(self.label)
        layout.addWidget(self.remove_button)

        self.setLayout(layout)

    def emit_remove(self):
        self.remove_requested.emit(self.peak_id)