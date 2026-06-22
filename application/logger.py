import logging
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger("QuantPhase")

class LogEmitter(QObject):
    log_signal = pyqtSignal(str)

log_emitter = LogEmitter()

class QtLogHandler(logging.Handler):

    def emit(self, record):

        message = self.format(record)

        log_emitter.log_signal.emit(message)

# Prevent duplicate handlers
if not logger.handlers:

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    )

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    qt_handler = QtLogHandler()
    qt_handler.setFormatter(formatter)

    logger.addHandler(qt_handler)

    logger.propagate = False