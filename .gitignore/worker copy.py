from PyQt6.QtCore import QThread, pyqtSignal, QObject

class ReconstructionWorker(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, obj_image, ref_images, peaks):
        super().__init__()
        self.obj_image = obj_image
        self.ref_images = ref_images
        self.peaks = peaks

    def run(self):
        try:
            from backend.pipeline import generate_reconstructions

            reconstructions = {}
            progress_count = 0
            total = len(self.ref_images) * len(self.peaks)

            def update_progress(_, __):
                nonlocal progress_count
                progress_count += 1
                percent = int((progress_count / total) * 100)
                self.progress.emit(percent)

            for peak in self.peaks:
                peak_id = peak["id"]
                x, y = peak["coords"]

                recon = generate_reconstructions(
                    self.obj_image,
                    self.ref_images,
                    y,
                    x,
                    progress_callback=update_progress
                )

                reconstructions[peak_id] = recon

            self.finished.emit(reconstructions)

        except Exception as e:
            self.error.emit(str(e))

class AnalysisWorker(QObject):
    finished = pyqtSignal(object, object, str)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, phase):
        super().__init__()
        self.phase = phase

    def run(self):
        try:
            from backend.pipeline import analyze_reconstruction

            class ProgressWrapper:
                def __init__(self, signal):
                    self.signal = signal

                def emit(self, val, msg):
                    self.signal.emit(val, msg)

            progress_callback = ProgressWrapper(self.progress)

            segmented_image, table_data, summary = analyze_reconstruction(
                self.phase,
                progress_callback
            )

            self.finished.emit(segmented_image, table_data, summary)

        except Exception as e:
            self.error.emit(str(e))

class StartupWorker(QObject):

    finished = pyqtSignal()
    log = pyqtSignal(str)

    def run(self):

        try:
            # self.log.emit("Loading Cellpose model...")

            from backend.pipeline import get_model
            get_model()

            # self.log.emit("Cellpose ready")

            # self.log.emit("Loading prediction models...")

            from backend.pipeline import preload_prediction_models
            preload_prediction_models()

            # self.log.emit("Prediction backend ready")

            self.log.emit("Startup initialization complete")

            self.finished.emit()

        except Exception as e:
            self.log.emit(f"Startup error: {str(e)}")