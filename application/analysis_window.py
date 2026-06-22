from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QMenu, QFileDialog, QSplitter, QSizePolicy, QAbstractItemView,
    QTextEdit, QComboBox, QDoubleSpinBox, QSlider, QFormLayout
)
from PyQt6.QtCore import Qt, QUrl, QTimer
from PyQt6.QtGui import QPixmap, QImage, QDesktopServices, QGuiApplication
import os
import tempfile
from application.logger import logger
from backend.pipeline import get_prediction_threshold

class AnalysisWindow(QWidget):
    def __init__(self, segmented_image=None, table_data = None, stats_data=None):
        super().__init__()

        # self.setWindowTitle("Analysis Results")
        # self.resize(900, 600)

        # ===== LEFT SIDE =====
        # left_layout = QVBoxLayout()
        bottom_layout = QVBoxLayout()

        # 🔹 Segmented Image (Top Left)
        self.image_label = QLabel("Segmented Image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_image_menu)

        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored
        )

        self.image_label.setMinimumSize(100, 100)

        self.pixmap = None

        if segmented_image is not None:
            if len(segmented_image.shape) == 2:
                # Grayscale
                h, w = segmented_image.shape
                q_img = QImage(segmented_image.data, w, h, w, QImage.Format.Format_Grayscale8)

            else:
                # Color
                h, w, ch = segmented_image.shape
                bytes_per_line = ch * w
                q_img = QImage(segmented_image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)

            self.pixmap = QPixmap.fromImage(q_img)

            self.image_label.setPixmap(
                self.pixmap
            )

        # Summary (Bottom Left)

        summary_layout = QHBoxLayout()
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Feature", "Value"])

        self.stats_data = stats_data
        if self.stats_data:
            self.populate_stats_table("Mean")

        self.stat_selector = QComboBox()
        self.stat_selector.addItems([
            "Mean",
            "Median",
            "Std Dev",
            "Min",
            "Max"
        ])

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        default_threshold = get_prediction_threshold()

        self.threshold_spin.setValue(default_threshold)

        controls_layout = QFormLayout()

        controls_layout.addRow(
            "Statistic:",
            self.stat_selector
        )

        controls_layout.addRow(
            "Threshold:",
            self.threshold_spin
        )

        self.show_stats_button = QPushButton("Show Results")
        controls_layout.addRow(self.show_stats_button)

        self.show_stats_button.clicked.connect(
            self.update_statistics_display
        )

        self.predict_button = QPushButton("Predict")
        controls_layout.addRow(self.predict_button)

        self.predict_button.clicked.connect(
            self.run_prediction
        )

        self.predict_button.setEnabled(
            table_data is not None and len(table_data) > 0
        )

        summary_layout.addWidget(self.stats_table, 4)
        summary_layout.addLayout(controls_layout, 2)

        # self.summary_label = QTextEdit()
        # self.summary_label.setReadOnly(True)
        # self.summary_label.setHtml(summary_text)
        # self.summary_label.setMinimumHeight(120)

        # self.predict_button = QPushButton("Predict")
        # self.predict_button.setFixedHeight(25)
        # self.predict_button.setEnabled(table_data is not None)
        
        # bottom_layout.addWidget(self.summary_label)
        # bottom_layout.addWidget(self.predict_button)

        bottom_widget = QWidget()
        bottom_widget.setLayout(summary_layout)

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.addWidget(self.image_label)
        left_splitter.addWidget(bottom_widget)

        left_splitter.setSizes([300, 300])  # [image, summary]
        left_splitter.setStretchFactor(0, 2)  # image
        left_splitter.setStretchFactor(1, 2)  # summary

        # ===== RIGHT SIDE =====
        self.table = QTableWidget()

        if table_data is not None:
            self.populate_table(table_data)
        
        self.table_data = table_data

        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_splitter)
        splitter.addWidget(self.table)

        splitter.setSizes([380, 520])
        splitter.setCollapsible(0, False)
        self.table.setMinimumWidth(400)

        splitter.setStretchFactor(0, 1)  # left
        splitter.setStretchFactor(1, 3)  # right

        splitter.setStyleSheet("""
        QSplitter::handle {
            background-color: #aaaaaa;
            width: 2px;
        }
        QSplitter::handle:hover {
            background-color: #666666;
        }
        """)

        left_splitter.setStyleSheet("""
        QSplitter::handle {
            background-color: #aaaaaa;
            height: 2px;
        }
        QSplitter::handle:hover {
            background-color: #666666;
        }
        """)

        # ===== MAIN LAYOUT =====
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)

        QTimer.singleShot(0, self.update_image_display)

        self.setLayout(main_layout)

    def update_statistics_display(self):

        stat_name = self.stat_selector.currentText()

        self.populate_stats_table(stat_name)

    def populate_table(self, data):
        """
        data = list of dicts
        Example:
        [
            {"Area": 120, "Perimeter": 50},
            {"Area": 130, "Perimeter": 55}
        ]
        """

        if not data:
            return

        headers = list(data[0].keys())

        self.table.setColumnCount(len(headers))
        self.table.setRowCount(len(data))
        self.table.setHorizontalHeaderLabels(headers)

        for row_idx, row_data in enumerate(data):
            for col_idx, key in enumerate(headers):
                item = QTableWidgetItem(str(row_data[key]))
                self.table.setItem(row_idx, col_idx, item)

        logger.info(
            f"Analysis table populated with {len(data)} rows"
        )

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def populate_stats_table(self, stat_name):

        if not self.stats_data:
            return

        stats = self.stats_data["statistics"].get(stat_name, {})

        self.stats_table.setRowCount(len(stats))

        for row, (feature, value) in enumerate(stats.items()):

            feature_item = QTableWidgetItem(str(feature))
            value_item = QTableWidgetItem(str(value))

            self.stats_table.setItem(row, 0, feature_item)
            self.stats_table.setItem(row, 1, value_item)

        self.stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.stats_table.verticalHeader().setVisible(False)

    def show_image_menu(self, position):
        if not hasattr(self, "pixmap"):
            return

        menu = QMenu()

        open_action = menu.addAction("Open")
        save_action = menu.addAction("Save As")

        action = menu.exec(self.image_label.mapToGlobal(position))

        if action == open_action:
            self.open_image()

        elif action == save_action:
            self.save_image()

    def open_image(self):

        from PIL import Image
        from PIL.ImageQt import fromqimage

        if self.pixmap is None or self.pixmap.isNull():
            return

        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "segmented_image.png")

        # Convert QPixmap -> QImage
        qimage = self.pixmap.toImage()

        # Convert QImage -> PIL Image
        pil_img = fromqimage(qimage)

        # Save with 300 DPI
        pil_img.save(temp_path, dpi=(300, 300))

        # Open with default viewer
        QDesktopServices.openUrl(QUrl.fromLocalFile(temp_path))
    
    def save_image(self):

        from PIL import Image
        from PIL.ImageQt import fromqimage

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;TIFF Files (*.tif)"
        )

        if file_path:

            # Convert QPixmap -> QImage
            qimage = self.pixmap.toImage()

            # Convert QImage -> PIL Image
            pil_img = fromqimage(qimage)

            # Save with 300 DPI metadata
            pil_img.save(file_path, dpi=(300, 300))

            logger.info(
                f"Segmented image saved: "
                f"{os.path.basename(file_path)}"
            )

    def update_image_display(self):
        if self.pixmap is None:
            return

        label_size = self.image_label.size()

        scaled_pixmap = self.pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_image_display()
        super().resizeEvent(event)

    def copy_selection(self):
        selected_ranges = self.table.selectedRanges()

        if not selected_ranges:
            return

        text = ""

        for r in selected_ranges:
            for row in range(r.topRow(), r.bottomRow() + 1):
                row_data = []
                for col in range(r.leftColumn(), r.rightColumn() + 1):
                    item = self.table.item(row, col)
                    row_data.append(item.text() if item else "")
                text += "\t".join(row_data) + "\n"

        QGuiApplication.clipboard().setText(text)
        logger.info("Copied selected table data to clipboard")

    def run_prediction(self):

        if not self.table_data:
            return

        from backend.pipeline import predict_from_table
        from application.prediction_dialog import PredictionDialog

        threshold = self.threshold_spin.value()

        prediction, confidence = predict_from_table(
            self.table_data,
            threshold=threshold
        )

        dialog = PredictionDialog(
            prediction,
            confidence,
            self
        )

        dialog.exec()

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_C:
            self.copy_selection()
        else:
            super().keyPressEvent(event)