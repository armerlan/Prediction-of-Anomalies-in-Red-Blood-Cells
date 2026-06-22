from PyQt6.QtCore import QThread, pyqtSignal, QObject

class ReconstructionWorker(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, obj_image, ref_images, peaks):
        super().__init__()
        self.obj_image = obj_image
        self.ref_images = ref_images
        self.peaks = peaks

        self._is_cancelled = False

    def stop(self):
        self._is_cancelled = True

    def run(self):
        try:
            from backend.pipeline import generate_reconstructions

            reconstructions = {}
            progress_count = 0
            total = len(self.ref_images) * len(self.peaks)

            def update_progress(_, __):

                if self._is_cancelled:
                    return False

                nonlocal progress_count
                progress_count += 1
                percent = int((progress_count / total) * 100)
                self.progress.emit(percent)

            for peak in self.peaks:

                if self._is_cancelled:
                    self.cancelled.emit()
                    return

                peak_id = peak["id"]
                x, y = peak["coords"]

                recon = generate_reconstructions(
                    self.obj_image,
                    self.ref_images,
                    y,
                    x,
                    progress_callback=update_progress
                )

                if self._is_cancelled:
                    self.cancelled.emit()
                    return
                
                if recon is None:
                    return

                reconstructions[peak_id] = recon

            self.finished.emit(reconstructions)

        except Exception as e:
            self.error.emit(str(e))

class AnalysisWorker(QObject):

    finished = pyqtSignal(object, object, object)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)

    def __init__(self, phase):
        super().__init__()

        self.phase = phase
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):

        try:
            from backend.pipeline import analyze_reconstruction

            class ProgressWrapper:

                def __init__(self, worker, signal):
                    self.worker = worker
                    self.signal = signal

                def emit(self, val, msg):

                    if self.worker.cancel_requested:
                        raise InterruptedError("Analysis cancelled")

                    self.signal.emit(val, msg)

                def is_cancelled(self):
                    return self.worker.cancel_requested

            progress_callback = ProgressWrapper(
                self,
                self.progress
            )

            segmented_image, table_data, summary = analyze_reconstruction(
                self.phase,
                progress_callback
            )

            if self.cancel_requested:
                return

            self.finished.emit(
                segmented_image,
                table_data,
                summary
            )

        except InterruptedError:
            return

        except Exception as e:

            import traceback
            from application.logger import logger

            logger.exception(
                f"Analysis worker crashed: {e}"
            )

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