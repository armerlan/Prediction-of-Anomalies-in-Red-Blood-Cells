import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, 
    QPushButton, QLineEdit, QHBoxLayout, 
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsLineItem, QGraphicsItemGroup)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QIntValidator
from PyQt6.QtCore import Qt

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )

        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self._panning = False
        self._pan_start = None

        # smoother appearance
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

    def wheelEvent(self, event):
        zoom_factor = 1.15

        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1 / zoom_factor, 1 / zoom_factor)

    def mousePressEvent(self, event):

        # Middle mouse = pan
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):

        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()

            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )

            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):

        # Send hover coordinates to parent dialog
        if self.parent() is not None:

            scene_pos = self.mapToScene(event.pos())

            self.parent().update_coordinates(scene_pos)

        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()

            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )

            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        
        # # Auto-scroll near edges
        # margin = 40
        # speed = 15

        # pos = event.pos()

        # if pos.x() > self.viewport().width() - margin:
        #     self.horizontalScrollBar().setValue(
        #         self.horizontalScrollBar().value() + speed
        #     )

        # elif pos.x() < margin:
        #     self.horizontalScrollBar().setValue(
        #         self.horizontalScrollBar().value() - speed
        #     )

        # if pos.y() > self.viewport().height() - margin:
        #     self.verticalScrollBar().setValue(
        #         self.verticalScrollBar().value() + speed
        #     )

        # elif pos.y() < margin:
        #     self.verticalScrollBar().setValue(
        #         self.verticalScrollBar().value() - speed
        #     )

        super().mouseMoveEvent(event)

class DraggableCrosshair(QGraphicsItemGroup):

    def __init__(self, x, y, size=12):
        super().__init__()

        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(2)

        h_line = QGraphicsLineItem(
            x - size,
            y,
            x + size,
            y
        )

        v_line = QGraphicsLineItem(
            x,
            y - size,
            x,
            y + size
        )

        h_line.setPen(pen)
        v_line.setPen(pen)

        self.addToGroup(h_line)
        self.addToGroup(v_line)

        self.setFlags(
            QGraphicsItemGroup.GraphicsItemFlag.ItemIsMovable
        )

class PeakSelectorDialog(QDialog):
    def __init__(self, fft_magnitude, parent=None):
        super().__init__(parent)

        self.selected_peak = None
        self.setMouseTracking(True)

        self.setWindowTitle("Select Peak")
        self.fft_magnitude = fft_magnitude

        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- Convert numpy array to displayable image ---
        mag = fft_magnitude

        # Normalize to 0–255
        mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
        mag_uint8 = (mag_norm * 255).astype(np.uint8)

        import cv2

        mag_color = cv2.applyColorMap(
            mag_uint8,
            cv2.COLORMAP_OCEAN
        )

        height, width, ch = mag_color.shape
        bytes_per_line = ch * width

        qimg = QImage(
            mag_color.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_BGR888
        )

        self.original_pixmap = QPixmap.fromImage(qimg)

        # Scale for display
        display_size = 500

        self.display_pixmap = self.original_pixmap.scaled(
            display_size,
            display_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # --- Display ---
        self.scene = QGraphicsScene()
        self.marker_item = None

        self.pixmap_item = QGraphicsPixmapItem(self.display_pixmap)
        self.scene.addItem(self.pixmap_item)

        self.graphics_view = ZoomableGraphicsView(self)
        self.graphics_view.setScene(self.scene)

        layout.addWidget(self.graphics_view)

        self.graphics_view.mousePressEvent = self.handle_graphics_click

        # --- coord label ---
        self.coord_label = QLabel("x: -, y: -")
        layout.addWidget(self.coord_label)

        # --- Peak Text Box ---
        input_layout = QHBoxLayout()

        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("x")
        self.x_input.setFixedWidth(80)

        self.y_input = QLineEdit()
        self.y_input.setPlaceholderText("y")
        self.y_input.setFixedWidth(80)

        # Validator based on image size
        h, w = self.fft_magnitude.shape
        validator = QIntValidator(0, max(w, h))
        self.x_input.setValidator(validator)
        self.y_input.setValidator(validator)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_manual_input)

        input_layout.addWidget(QLabel("x:"))
        input_layout.addWidget(self.x_input)
        input_layout.addWidget(QLabel("y:"))
        input_layout.addWidget(self.y_input)
        input_layout.addWidget(self.apply_button)

        layout.addLayout(input_layout)

        # --- Confirm Button ---
        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_selection)

        layout.addWidget(self.confirm_button)

    def handle_graphics_click(self, event):

        if event.button() == Qt.MouseButton.LeftButton:

            scene_pos = self.graphics_view.mapToScene(event.pos())

            x_display = int(scene_pos.x())
            y_display = int(scene_pos.y())

            orig_height, orig_width = self.fft_magnitude.shape

            x_real = int(
                x_display * (orig_width / self.display_pixmap.width())
            )

            y_real = int(
                y_display * (orig_height / self.display_pixmap.height())
            )

            self.selected_peak = (x_real, y_real)

            self.x_input.setText(str(x_real))
            self.y_input.setText(str(y_real))

            self.coord_label.setText(
                f"Selected Peak → x: {x_real}, y: {y_real}"
            )

            self.draw_marker(x_display, y_display)

        ZoomableGraphicsView.mousePressEvent(
            self.graphics_view,
            event
        )

    def update_coordinates(self, pos):

        x_display = int(pos.x())
        y_display = int(pos.y())

        orig_height, orig_width = self.fft_magnitude.shape

        x_real = int(
            x_display * (orig_width / self.display_pixmap.width())
        )

        y_real = int(
            y_display * (orig_height / self.display_pixmap.height())
        )

        self.coord_label.setText(
            f"x: {x_real}, y: {y_real}"
        )
    
    def confirm_selection(self):
        if self.selected_peak is not None:
            self.accept()
        else:
            self.coord_label.setText("Please select a peak first!")

    def apply_manual_input(self):
        if not self.x_input.text() or not self.y_input.text():
            return

        x_real = int(self.x_input.text())
        y_real = int(self.y_input.text())

        self.selected_peak = (x_real, y_real)

        self.coord_label.setText(f"Selected Peak → x: {x_real}, y: {y_real}")

        # Convert back to display coordinates
        pixmap_width = self.display_pixmap.width()
        pixmap_height = self.display_pixmap.height()

        orig_height, orig_width = self.fft_magnitude.shape

        x_display = int(x_real * (pixmap_width / orig_width))
        y_display = int(y_real * (pixmap_height / orig_height))

        self.draw_marker(x_display, y_display)
    
    def draw_marker(self, x_display, y_display):

        if self.marker_item is not None:
            self.scene.removeItem(self.marker_item)

        self.marker_item = DraggableCrosshair(
            x_display,
            y_display
        )

        self.scene.addItem(self.marker_item)