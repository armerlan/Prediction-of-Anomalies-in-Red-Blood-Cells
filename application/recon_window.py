import os
import tempfile
from PyQt6.QtWidgets import(
    QApplication, QWidget, QPushButton, QHBoxLayout, QLineEdit,
    QLabel, QVBoxLayout, QFileDialog, QScrollArea, QGridLayout, QProgressDialog,
    QMenu, QSizePolicy, QTabWidget
) 
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QPixmap, QImage, QDesktopServices, QIntValidator

from application.widgets import ClickableLabel, PeakTagWidget
from application.peak_selector_dialog import PeakSelectorDialog
from application.worker import ReconstructionWorker
from application.logger import logger

class ReconWindow(QWidget):

    analyze_requested = pyqtSignal(object)

    def __init__(self, parent = None):
        super().__init__(parent)

        self.obj_image = None
        self.ref_images = None
        self.peaks = []              # list of peak dicts
        self.next_peak_id = 1        # unique ID generator
        self.reconstructions = {}    # per-peak storage
        self.current_peak_id = None  # dropdown selection
        self.thread = None
        self.worker = None
        self.progress = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Object Image Row
        obj_layout = QHBoxLayout()

        self.obj_label = QLabel("Select Object Image:")

        self.obj_path_display = QLineEdit()
        self.obj_path_display.setReadOnly(True)

        self.obj_browse_button = QPushButton("Browse")
        self.obj_browse_button.clicked.connect(self.load_obj_image)

        obj_layout.addWidget(self.obj_label)
        obj_layout.addWidget(self.obj_path_display)
        obj_layout.addWidget(self.obj_browse_button)

        self.layout.addLayout(obj_layout)

        # Peak Selection Row
        peak_layout = QHBoxLayout()

        self.peak_label = QLabel("Peak Coordinates:")

        # Container widget for tags
        self.tags_container = QWidget()
        self.peak_tags_layout = QHBoxLayout(self.tags_container)
        self.peak_tags_layout.setContentsMargins(0, 0, 0, 0)
        self.peak_tags_layout.setSpacing(6)
        
        # Scroll area
        self.tags_scroll = QScrollArea()
        self.tags_scroll.setWidgetResizable(True)
        self.tags_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tags_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tags_scroll.setWidget(self.tags_container)
        target_height = self.obj_path_display.sizeHint().height()

        self.tags_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
        self.tags_scroll.setFixedHeight(40)

        # Empty Peak Section
        self.no_peak_label = QLabel(" No Peak Selected")
        self.no_peak_label.setStyleSheet("color: gray; font-style: italic;")
        self.no_peak_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.peak_tags_layout.addWidget(self.no_peak_label)

        # Make tags area expand
        self.tags_container.setSizePolicy(
            self.tags_container.sizePolicy().horizontalPolicy(),
            self.tags_container.sizePolicy().verticalPolicy()
        )

        # Add Peak button
        self.add_peak_button = QPushButton("Add New Peak")
        self.add_peak_button.setEnabled(False)
        self.add_peak_button.clicked.connect(self.add_peak)

        peak_layout.addWidget(self.peak_label)
        peak_layout.addWidget(self.tags_scroll)
        peak_layout.addWidget(self.add_peak_button)

        self.tags_scroll.setFixedHeight(target_height)
        self.add_peak_button.setFixedHeight(target_height)

        self.layout.addLayout(peak_layout)

        # Reference Images Row
        ref_layout = QHBoxLayout()

        self.ref_label = QLabel("Select Reference Images:")

        self.ref_path_display = QLineEdit()
        self.ref_path_display.setReadOnly(True)

        self.ref_browse_button = QPushButton("Browse")
        self.ref_browse_button.clicked.connect(self.load_ref_image)

        ref_layout.addWidget(self.ref_label)
        ref_layout.addWidget(self.ref_path_display)
        ref_layout.addWidget(self.ref_browse_button)

        self.layout.addLayout(ref_layout)

        #Remove focus
        self.obj_path_display.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.ref_path_display.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        #Same box width
        label_width = max(
            self.obj_label.sizeHint().width(),
            self.ref_label.sizeHint().width(),
            self.peak_label.sizeHint().width()
        )

        self.obj_label.setFixedWidth(label_width)
        self.ref_label.setFixedWidth(label_width)
        self.peak_label.setFixedWidth(label_width)
        self.ref_browse_button.setFixedWidth(self.add_peak_button.sizeHint().width())
        self.obj_browse_button.setFixedWidth(self.add_peak_button.sizeHint().width())

        #reconstructions
        self.generate_button = QPushButton("Generate Reconstructions")
        self.generate_button.setEnabled(
            self.obj_image is not None and
            self.ref_images is not None
        )
        self.generate_button.clicked.connect(self.generate_reconstruction)
        self.layout.addWidget(self.generate_button)

        # Results Container
        self.results_container = QVBoxLayout()
        self.layout.addLayout(self.results_container)
        self.placeholder_label = QLabel(
            "No reconstructions yet.\nClick 'Generate Reconstructions'"
        )
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet("color: gray; font-size: 14px;")

        self.results_container.addWidget(self.placeholder_label)

        # Bottom Analyze Bar
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.run_analysis)

        # Optional styling (makes disabled state more obvious)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                height: 18px;
                font-size: 12.5px;
            }
        """)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()  # pushes button to right
        bottom_layout.addWidget(self.analyze_button)

        self.layout.addLayout(bottom_layout)

    def load_obj_image(self):
        import cv2

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Object Image",
            "",
            "Images (*.png *.jpg *.bmp *.tif *.tiff)"
        )

        if file_path:
            self.reset_workspace()
            self.obj_path = file_path

            # Read image
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            self.obj_path_display.setText(os.path.basename(file_path))

            if image is None:
                logger.error("Failed to load object image")
                return

            # Store numpy image
            self.obj_image = image
            logger.info(f"Loaded object image: {os.path.basename(file_path)}")
            self.add_peak_button.setEnabled(True)

            self.update_generate_button_state()

    def add_peak(self):
        from backend.pipeline import compute_fft_magnitude

        if self.obj_image is None:
            return

        mag = compute_fft_magnitude(self.obj_image)

        dialog = PeakSelectorDialog(mag, self)
        if dialog.exec():
            peak = dialog.selected_peak

            if peak is None:
                return

            x, y = peak

            peak_id = self.next_peak_id
            self.next_peak_id += 1

            peak_data = {
                "id": peak_id,
                "coords": (x, y)
            }

            self.peaks.append(peak_data)
            logger.info(f"Added peak at ({x}, {y})")

            tag = PeakTagWidget(peak_id, (x, y))
            tag.remove_requested.connect(self.remove_peak)

            self.peak_tags_layout.addWidget(tag)
            if self.no_peak_label.isVisible():
                self.no_peak_label.hide()

            self.update_generate_button_state()

    def remove_peak(self, peak_id):

        # Remove from data
        self.peaks = [p for p in self.peaks if p["id"] != peak_id]

        if len(self.peaks) == 0:
            self.no_peak_label.show()

        # Remove widget from layout
        for i in reversed(range(self.peak_tags_layout.count())):
            widget = self.peak_tags_layout.itemAt(i).widget()

            if widget and hasattr(widget, "peak_id") and widget.peak_id == peak_id:
                widget.setParent(None)
                widget.deleteLater()
        
        logger.info(f"Removed peak ID {peak_id}")

        # Reset state
        self.reconstructions.clear()
        self.selected_index = None
        self.analyze_button.setEnabled(False)

        self.update_generate_button_state()

    def load_ref_image(self):
        import cv2
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Reference Images",
            "",
            "Images (*.png *.jpg *.bmp *.tif *.tiff)"
        )

        if files:
            # Store paths
            self.ref_paths = files

            # Load all images into list
            self.ref_images = []

            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.ref_images.append(img)

            self.ref_path_display.setText(f"{len(files)} files selected")
            logger.info(f"Loaded {len(self.ref_images)} reference images")
            self.update_generate_button_state()
    
    def show_context_menu(self, peak_id, index, position):

        menu = QMenu()

        open_action = menu.addAction("Open")
        save_action = menu.addAction("Save As")

        action = menu.exec(position.toPoint())

        if action == open_action:
            self.open_large_view(peak_id, index)

        elif action == save_action:
            self.save_image(peak_id, index)


    def generate_reconstruction(self):

        if self.obj_image is None or self.ref_images is None or len(self.peaks) == 0:
            return
        
        if hasattr(self, "thread") and self.thread is not None:
            if self.thread.isRunning():
                return

        self.progress = QProgressDialog(
            "Generating reconstructions...",
            "Cancel",
            0,
            100,
            self
        )

        self.progress.setWindowTitle("Reconstruction")
        self.progress.setMinimumDuration(0)
        self.progress.setAutoClose(False)
        self.progress.setAutoReset(False)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.show()

        # Create thread + worker
        self.thread = QThread()
        self.worker = ReconstructionWorker(self.obj_image, self.ref_images, self.peaks)

        self.worker.moveToThread(self.thread)
        logger.info(
            f"Starting reconstruction with "
            f"{len(self.peaks)} peaks and "
            f"{len(self.ref_images)} references"
        )

        self.progress.canceled.connect(self.cancel_reconstruction)
        self.worker.cancelled.connect(self.on_reconstruction_cancelled)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)

        self.worker.finished.connect(self.on_reconstruction_done)
        self.worker.error.connect(self.on_reconstruction_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)

        self.worker.cancelled.connect(self.thread.quit)
        self.worker.cancelled.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.cleanup_reconstruction_thread)

        self.thread.start()

    def cancel_reconstruction(self):

        logger.info("Stopping reconstruction generation...")

        if self.worker is not None:
            self.worker.stop()

        self.progress.setLabelText("Stopping reconstruction...")
        self.progress.setCancelButton(None)

    def on_reconstruction_cancelled(self):

        logger.info("Reconstruction generation cancelled")

        if self.progress is not None:
            self.progress.close()

    def cleanup_reconstruction_thread(self):

        if self.thread is not None:
            self.thread.deleteLater()

        self.thread = None
        self.worker = None

    def on_reconstruction_done(self, reconstructions):
        self.progress.close()

        self.reconstructions = reconstructions

        if self.peaks:
            self.current_peak_id = self.peaks[0]["id"]

        logger.info("Reconstruction generation complete")

        self.build_tabs()


    def on_reconstruction_error(self, message):
        self.progress.close()
        logger.error(f"Reconstruction failed: {message}")

    def build_tabs(self):

        # Clear previous content
        for i in reversed(range(self.results_container.count())):
            widget = self.results_container.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.labels = {}

        for peak in self.peaks:
            peak_id = peak["id"]
            x, y = peak["coords"]
            self.labels[peak_id] = []

            tab = QWidget()
            tab_layout = QVBoxLayout(tab)

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)

            scroll_widget = QWidget()
            grid = QGridLayout(scroll_widget)

            scroll_area.setWidget(scroll_widget)
            tab_layout.addWidget(scroll_area)

            recon_list = self.reconstructions[peak_id]

            for i, recon in enumerate(recon_list):

                img = recon["phase_display"]

                h, w, ch = img.shape
                bytes_per_line = ch * w

                q_img = QImage(
                    img.data,
                    w,
                    h,
                    bytes_per_line,
                    QImage.Format.Format_BGR888
                )
                pixmap = QPixmap.fromImage(q_img)

                label = ClickableLabel(i)
                label.setStyleSheet("padding: 5px;")

                # IMPORTANT: bind peak_id with index
                label.clicked.connect(lambda idx, pid=peak_id: self.select_reconstruction(pid, idx))
                label.rightClicked.connect(
                    lambda idx, pos, pid=peak_id: self.show_context_menu(pid, idx, pos)
                )

                label.setPixmap(
                    pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
                )
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                self.labels[peak_id].append(label)

                row = i // 4
                col = i % 4

                grid.addWidget(label, row, col)

            scroll_area.setWidget(scroll_widget)
            tab_layout.addWidget(scroll_area)
            
            self.tabs.addTab(tab, f"Peak ({x}, {y})")

        self.results_container.addWidget(self.tabs)

    def clear_reconstruction_ui(self):

        # Remove all widgets from results container
        for i in reversed(range(self.results_container.count())):

            item = self.results_container.itemAt(i)
            widget = item.widget()

            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        # Re-add placeholder
        self.placeholder_label = QLabel(
            "No reconstructions yet.\nClick 'Generate Reconstructions'"
        )
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet(
            "color: gray; font-size: 14px;"
        )

        self.results_container.addWidget(self.placeholder_label)

    def reset_workspace(self):

        # ---- Clear reconstruction data ----
        self.reconstructions.clear()

        # ---- Clear selection state ----
        self.selected_index = None
        self.selected_phase = None
        self.current_peak_id = None

        # ---- Disable analysis ----
        self.analyze_button.setEnabled(False)

        # ---- Remove peak widgets ----
        for i in reversed(range(self.peak_tags_layout.count())):

            item = self.peak_tags_layout.itemAt(i)
            widget = item.widget()

            if widget is not None and widget != self.no_peak_label:
                widget.setParent(None)
                widget.deleteLater()

        # ---- Reset peak state ----
        self.peaks.clear()
        self.next_peak_id = 1

        # ---- Show placeholder peak label ----
        self.no_peak_label.show()

        # ---- Clear reconstruction UI ----
        self.clear_reconstruction_ui()

        # ---- Clear labels cache ----
        self.labels = {}

        # ---- Clear references ----
        self.ref_images = None
        self.ref_paths = []

        # ---- Clear ref display ----
        self.ref_path_display.clear()

        # ---- Disable buttons ----
        self.generate_button.setEnabled(False)

        logger.info("Workspace reset")
    
    def display_reconstructions(self):

        self.selected_index = None
        self.analyze_button.setEnabled(False)

        # Clear old grid
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        # Display images
        for i, recon in enumerate(self.reconstructions):

            img = recon["phase_display"]

            h, w, ch = img.shape
            bytes_per_line = ch * w

            q_img = QImage(
                img.data,
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_BGR888
            )
            pixmap = QPixmap.fromImage(q_img)

            label = ClickableLabel(i)
            label.setStyleSheet("padding: 5px;")
            label.clicked.connect(self.select_reconstruction)
            label.rightClicked.connect(self.show_context_menu)
            label.setPixmap(
                pixmap.scaled(170, 170, Qt.AspectRatioMode.KeepAspectRatio)
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            row = i // 4
            col = i % 4

            self.grid_layout.addWidget(label, row, col)
    
    def update_generate_button_state(self):
        has_obj = hasattr(self, "obj_image")
        has_ref = self.ref_images is not None and len(self.ref_images) > 0
        has_peaks = len(self.peaks) > 0

        self.generate_button.setEnabled(has_obj and has_ref and has_peaks)
    
    def select_reconstruction(self, peak_id, index):

        self.current_peak_id = peak_id
        self.selected_index = index

        self.selected_phase = self.reconstructions[peak_id][index]["phase"]

        main_window = self.window()

        can_enable = True

        if hasattr(main_window, "thread") and main_window.thread is not None:
            try:
                if main_window.thread.isRunning():
                    can_enable = False
            except RuntimeError:
                pass

        if hasattr(main_window, "analysis_stopping"):
            if main_window.analysis_stopping:
                can_enable = False

        self.analyze_button.setEnabled(can_enable)

        self.highlight_selection(peak_id, index)

    def highlight_selection(self, peak_id, index):

        for i, label in enumerate(self.labels[peak_id]):

            if i == index:
                label.selected = True
                label.setStyleSheet("""
                    background-color: rgba(0, 120, 215, 80);
                    border: 2px solid rgba(0, 120, 215, 180);
                    border-radius: 8px;
                """)
            else:
                label.selected = False
                label.setStyleSheet("""
                    background-color: transparent;
                    border: none;
                """)
    
    def open_large_view(self, peak_id, index):

        import cv2
        from PIL import Image

        img = self.reconstructions[peak_id][index]["phase_display"]

        if img is None:
            return

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        )

        temp_path = temp_file.name
        temp_file.close()

        # Convert OpenCV BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img_rgb)

        # Save with 300 DPI metadata
        pil_img.save(temp_path, dpi=(300, 300))

        # Open with system viewer
        QDesktopServices.openUrl(QUrl.fromLocalFile(temp_path))
    
    def save_image(self, peak_id, index):
        import cv2

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif)"
        )

        if file_path:
            img = self.reconstructions[peak_id][index]["phase_display"]
            from PIL import Image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img.save(file_path, dpi=(300, 300))

    def on_tab_changed(self, index):
        self.selected_index = None
        self.analyze_button.setEnabled(False)

    def run_analysis(self):
        if self.selected_index is None or self.current_peak_id is None:
            return

        phase = self.reconstructions[self.current_peak_id][self.selected_index]["phase"]
        # logger.info("Starting reconstruction analysis")
        self.analyze_requested.emit(phase)