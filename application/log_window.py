from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit
from application.logger import log_emitter


class LogWindow(QWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.log_output = QPlainTextEdit()

        self.log_output.setReadOnly(True)

        self.log_output.setMaximumBlockCount(1000)

        self.log_output.setStyleSheet("""
            QPlainTextEdit {
                background-color: #111111;
                color: #dddddd;
                font-family: Consolas;
                font-size: 12px;
                border: none;
            }
        """)

        self.log_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        layout.addWidget(self.log_output)

        log_emitter.log_signal.connect(self.append_log)

    def append_log(self, message):

        self.log_output.appendPlainText(message)

        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())