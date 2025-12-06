#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Any, List

from PyQt6 import QtCore
from PyQt6.QtCore import (
    Qt, QSize, QPoint, QRect, pyqtSignal
)
from PyQt6.QtGui import (
    QPixmap, QPainter, QColor, QCursor, QPen, QPaintEvent, QMouseEvent
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QDoubleSpinBox, QAbstractSpinBox, QSplitter, QScrollArea, QSizePolicy
)

# Imports from core
from core.common import MODE_VIEW, MODE_CROP, pil_to_qpixmap, Image, ImageOps

# --- 1. Image Viewer (For Editor) ---


class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundRole(self.backgroundRole())
        self.setAutoFillBackground(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Ignored)
        self.setScaledContents(False)

        self.editor_mode = MODE_VIEW
        self.crop_start_point: Optional[QPoint] = None
        self.crop_end_point: Optional[QPoint] = None
        # Original (full-resolution) pixmap
        self._base_pixmap: Optional[QPixmap] = None
        # Currently displayed (scaled) pixmap
        self.current_qpixmap: Optional[QPixmap] = None
        self.editor_ref: Optional[Any] = None

        # Crop state
        self.crop_mode_type: str = "rect"  # "rect" or "free"
        self.crop_rect: Optional[QRect] = None
        self.freeform_pts: List[QPoint] = []
        self._drag_idx: Optional[int] = None
        self._handle_px: int = 6
        self._handle_radius: int = 8

    def set_crop_mode_type(self, mode: str) -> None:
        if mode not in ("rect", "free"):
            return
        self.crop_mode_type = mode
        if self.editor_mode == MODE_CROP:
            self.set_editor_mode(MODE_CROP)
        else:
            self.update()

    def set_pixmap(self, qpixmap: QPixmap):
        """Store the base pixmap and display a scaled copy that fits the widget."""
        self._base_pixmap = qpixmap
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        """Scale the base pixmap to the current widget size while keeping aspect ratio."""
        if not self._base_pixmap or self._base_pixmap.isNull():
            self.current_qpixmap = None
            self.setPixmap(QPixmap())
            # Allow the layout to control the size completely
            self.setMinimumSize(QSize(50, 50))
            return

        avail_size = self.size()
        if avail_size.width() <= 0 or avail_size.height() <= 0:
            # Fallback to the pixmap's own size
            scaled = self._base_pixmap
        else:
            scaled = self._base_pixmap.scaled(
                avail_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.current_qpixmap = scaled
        self.setPixmap(scaled)
        # Do not force a larger minimum size than the layout - keeps it responsive
        self.setMinimumSize(QSize(50, 50))
        self.update()

    def resizeEvent(self, event):
        """On resize, re-scale the base pixmap to fit the new size."""
        super().resizeEvent(event)
        if self._base_pixmap is not None:
            self._update_scaled_pixmap()

    def clear_pixmap(self, text: str = "No Image Loaded"):
        self._base_pixmap = None
        self.current_qpixmap = None
        self.setPixmap(QPixmap())
        self.setMinimumSize(QSize(100, 100))
        self.setText(text)
        self.update()

    def _current_rect(self) -> Optional[QRect]:
        if not (self.crop_start_point and self.crop_end_point):
            return None
        x1 = min(self.crop_start_point.x(), self.crop_end_point.x())
        y1 = min(self.crop_start_point.y(), self.crop_end_point.y())
        x2 = max(self.crop_start_point.x(), self.crop_end_point.x())
        y2 = max(self.crop_start_point.y(), self.crop_end_point.y())
        return QRect(QPoint(x1, y1), QPoint(x2, y2))

    def set_editor_mode(self, mode: int):
        self.editor_mode = mode
        if self.editor_mode == MODE_VIEW:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self.crop_start_point = None
            self.crop_end_point = None
            self.crop_rect = None
            self.freeform_pts.clear()
            self._drag_idx = None
        elif self.editor_mode == MODE_CROP:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            img_rect = self.get_image_display_rect()
            if img_rect.isEmpty():
                return

            if self.crop_mode_type == "rect":
                side = min(50, img_rect.width(), img_rect.height())
                cx = img_rect.center().x()
                cy = img_rect.center().y()
                tl = QPoint(cx - side // 2, cy - side // 2)
                br = QPoint(cx + side // 2, cy + side // 2)

                # Clamp
                tl.setX(max(img_rect.left(), min(tl.x(), img_rect.right())))
                tl.setY(max(img_rect.top(), min(tl.y(), img_rect.bottom())))
                br.setX(max(img_rect.left(), min(br.x(), img_rect.right())))
                br.setY(max(img_rect.top(), min(br.y(), img_rect.bottom())))

                self.crop_start_point = tl
                self.crop_end_point = br
                self._sync_pts_from_rect()
            else:
                self.freeform_pts.clear()
                self.crop_rect = None
                self._drag_idx = None

        self.update()

    def set_editor_ref(self, ref: Any):
        self.editor_ref = ref

    def _sync_pts_from_rect(self):
        rect = self._current_rect()
        if rect is None:
            self.freeform_pts.clear()
            self.crop_rect = None
            return
        self.crop_rect = rect
        self.freeform_pts = [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomRight(),
            rect.bottomLeft(),
        ]

    def _sync_rect_from_pts(self):
        if not self.freeform_pts:
            self.crop_rect = None
            return
        xs = [p.x() for p in self.freeform_pts]
        ys = [p.y() for p in self.freeform_pts]
        tl = QPoint(min(xs), min(ys))
        br = QPoint(max(xs), max(ys))
        self.crop_rect = QRect(tl, br)

    def get_image_display_rect(self) -> QRect:
        if not self.current_qpixmap:
            return QRect(0, 0, 0, 0)
        w_img = self.current_qpixmap.width()
        h_img = self.current_qpixmap.height()
        w_can = self.width()
        h_can = self.height()
        x_offset = (w_can - w_img) // 2
        y_offset = (h_can - h_img) // 2
        return QRect(x_offset, y_offset, w_img, h_img)

    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        if self.editor_mode != MODE_CROP:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        img_rect = self.get_image_display_rect()
        if img_rect.isEmpty():
            return

        crop_mode = getattr(self, "crop_mode_type", "rect")
        if crop_mode == "rect":
            self._paint_rect_mode(painter, img_rect)
        else:
            self._paint_freeform_mode(painter, img_rect)

    def _paint_rect_mode(self, painter: QPainter, img_rect: QRect):
        if not (self.crop_start_point and self.crop_end_point):
            return

        rect = self._current_rect()
        if not rect:
            return

        # Darken outside region
        painter.save()
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())

        # Clear inside
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_Clear)
        painter.drawRect(rect)
        painter.restore()

        # Draw border
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.drawRect(rect)

        # Draw drag handles
        handle_color = QColor(255, 255, 255)
        handle_pen = QPen(QColor(0, 0, 0), 1)
        painter.setPen(handle_pen)
        painter.setBrush(handle_color)

        for p in self._rect_handles(rect):
            painter.drawEllipse(p, self._handle_radius, self._handle_radius)

    def _rect_handles(self, rect: QRect) -> List[QPoint]:
        cx = rect.center().x()
        cy = rect.center().y()
        return [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomRight(),
            rect.bottomLeft(),
            QPoint(cx, rect.top()),
            QPoint(rect.right(), cy),
            QPoint(cx, rect.bottom()),
            QPoint(rect.left(), cy),
        ]

    def _paint_freeform_mode(self, painter: QPainter, img_rect: QRect):
        if len(self.freeform_pts) < 3:
            # Just dim fullscreen
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(self.rect())
            return

        # Darken outside
        painter.save()
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())

        # Clear inside the polygon
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_Clear)
        painter.drawPolygon(*self.freeform_pts)
        painter.restore()

        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(*self.freeform_pts)

        # Draw handles
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        for p in self.freeform_pts:
            painter.drawEllipse(p, self._handle_radius, self._handle_radius)

    def _hit_test_handles(self, pos: QPoint) -> int:
        # Return index of handle in freeform_pts if near, else -1
        for i, p in enumerate(self.freeform_pts):
            if (p - pos).manhattanLength() <= self._handle_px:
                return i
        return -1

    def mousePressEvent(self, event: QMouseEvent):
        if self.editor_mode != MODE_CROP:
            return super().mousePressEvent(event)

        if event.button() == Qt.MouseButton.LeftButton:
            img_rect = self.get_image_display_rect()
            if img_rect.isEmpty():
                return

            p = event.pos()
            if not img_rect.contains(p):
                return

            crop_mode = getattr(self, "crop_mode_type", "rect")
            if crop_mode == "rect":
                # Check handles first
                rect = self._current_rect()
                if rect is None:
                    return
                handles = self._rect_handles(rect)
                for i, h in enumerate(handles):
                    if (h - p).manhattanLength() <= self._handle_px:
                        self._drag_idx = i
                        return
                # Otherwise start drawing new rect
                self.crop_start_point = p
                self.crop_end_point = p
            else:
                # Freeform - either start new polygon or drag existing handle
                idx = self._hit_test_handles(p)
                if idx >= 0:
                    self._drag_idx = idx
                else:
                    self.freeform_pts.append(p)
                    self._sync_rect_from_pts()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.editor_mode != MODE_CROP:
            return super().mouseMoveEvent(event)

        p = event.pos()
        img_rect = self.get_image_display_rect()
        if img_rect.isEmpty() or not img_rect.contains(p):
            return

        crop_mode = getattr(self, "crop_mode_type", "rect")
        if crop_mode == "rect":
            if self._drag_idx is None:
                if self.crop_start_point:
                    self.crop_end_point = p
                    self._sync_pts_from_rect()
                    self.update()
                return

            # Dragging an existing handle
            rect = self._current_rect()
            if rect is None:
                return

            # Map handle index to corners/edges
            # 0: TL, 1: TR, 2: BR, 3: BL, 4: top, 5: right, 6: bottom, 7: left
            new_left = rect.left()
            new_top = rect.top()
            new_right = rect.right()
            new_bottom = rect.bottom()

            if self._drag_idx in (0, 3, 7):
                new_left = p.x()
            if self._drag_idx in (1, 2, 5):
                new_right = p.x()
            if self._drag_idx in (0, 1, 4):
                new_top = p.y()
            if self._drag_idx in (2, 3, 6):
                new_bottom = p.y()

            # Normalize
            if new_left > new_right:
                new_left, new_right = new_right, new_left
            if new_top > new_bottom:
                new_top, new_bottom = new_bottom, new_top

            # Clamp to image rect
            new_left = max(img_rect.left(), min(new_left, img_rect.right()))
            new_right = max(img_rect.left(), min(new_right, img_rect.right()))
            new_top = max(img_rect.top(), min(new_top, img_rect.bottom()))
            new_bottom = max(img_rect.top(), min(
                new_bottom, img_rect.bottom()))

            self.crop_start_point = QPoint(new_left, new_top)
            self.crop_end_point = QPoint(new_right, new_bottom)
            self._sync_pts_from_rect()
            self.update()
            return

        if self._drag_idx is not None and 0 <= self._drag_idx < len(self.freeform_pts):
            self.freeform_pts[self._drag_idx] = p
            self._sync_rect_from_pts()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.editor_mode != MODE_CROP:
            return super().mouseReleaseEvent(event)

        self._drag_idx = None

    # --- 2. Dual Image Viewer (For Comparisons) ---


class DualImageViewer(QWidget):
    left_clicked = pyqtSignal()
    right_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QSplitter(Qt.Orientation.Horizontal, self)

        self.left_viewer = ImageViewer()
        self.right_viewer = ImageViewer()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self.left_viewer)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(self.right_viewer)

        layout.addWidget(left_scroll)
        layout.addWidget(right_scroll)

        self.left_viewer.mousePressEvent = self._wrap_click(
            self.left_viewer.mousePressEvent, self.left_clicked.emit
        )
        self.right_viewer.mousePressEvent = self._wrap_click(
            self.right_viewer.mousePressEvent, self.right_clicked.emit
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(layout)

    def _wrap_click(self, original, signal_emit):
        def handler(event):
            if event.button() == Qt.MouseButton.LeftButton:
                signal_emit()
            return original(event)

        return handler

    def set_left_pixmap(self, pm: QPixmap):
        self.left_viewer.set_pixmap(pm)

    def set_right_pixmap(self, pm: QPixmap):
        self.right_viewer.set_pixmap(pm)

    def clear(self):
        self.left_viewer.clear_pixmap()
        self.right_viewer.clear_pixmap()

    # --- 3. Slider + DoubleSpinBox helper ---


class SliderSpinBox(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, minimum: float, maximum: float, initial: float = 0.0, step: float = 0.1, parent=None):
        super().__init__(parent)

        # 1. Label
        self.label = QLabel(label)
        self.label.setMinimumWidth(80)

        # 2. Slider (Int)
        self.slider = QSlider(Qt.Orientation.Horizontal)

        self._minimum = minimum
        self._maximum = maximum
        self._step = step if step > 0 else 0.1

        # We map float range [min, max] to int range [0, num_steps]
        steps = int((self._maximum - self._minimum) / self._step)
        self.slider.setMinimum(0)
        self.slider.setMaximum(steps)

        # 3. SpinBox (Float)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(self._step)
        self.spin.setDecimals(2)
        self.spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin.setFixedWidth(60)

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)

        # Connections
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)

        # Initialize
        self.set_value(initial)

    def _on_slider_changed(self, v: int):
        # int -> float
        val = self._minimum + (v * self._step)
        # Avoid slight float errors
        val = round(val, 3)

        self.spin.blockSignals(True)
        self.spin.setValue(val)
        self.spin.blockSignals(False)
        self.valueChanged.emit(val)

    def _on_spin_changed(self, val: float):
        # float -> int
        rel = val - self._minimum
        slider_v = int(round(rel / self._step))

        self.slider.blockSignals(True)
        self.slider.setValue(slider_v)
        self.slider.blockSignals(False)
        self.valueChanged.emit(val)

    def setValue(self, value: float):
        self.spin.setValue(value)

    def set_value(self, value: float):
        """Alias for compatibility with photos_tab.py calls."""
        self.setValue(value)

    def value(self) -> float:
        return self.spin.value()
