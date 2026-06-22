import sys
from application.main_window import MainWindow
from application.logger import logger
from PyQt6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)

    logger.info("Application starting")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())