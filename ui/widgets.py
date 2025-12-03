#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Any, List

from PyQt6 import QtCore
from PyQt6.QtCore import (
    Qt,
    QSize,
    QPoint,
    QRect,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QPixmap,
    QPainter,
    QColor,
    QCursor,
    QPen,
    QPaintEvent,
    QMouseEvent,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QPushButton, QLabel, QSlider, QTabWidget, QGridLayout,
    QDoubleSpinBox, QAbstractSpinBox, QSplitter, QSizePolicy, QScrollArea,
    QMessageBox, QFileDialog, QGroupBox, QLineEdit, QStackedWidget,
    QRadioButton, QButtonGroup, QFrame
)

from core.common import MODE_VIEW, MODE_CROP


class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundRole(
            self.backgroundRole()
        )  # no-op but keeps compatible call
        self.setAutoFillBackground(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self.setScaledContents(False)

        self.editor_mode = MODE_VIEW
        self.crop_start_point: Optional[QPoint] = None
        self.crop_end_point: Optional[QPoint] = None
        self.current_qpixmap: Optional[QPixmap] = None
        self.editor_ref: Optional[Any] = None

        self.crop_mode_type: str = "rect"
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
        self.current_qpixmap = qpixmap
        self.setPixmap(qpixmap)
        self.setMinimumSize(qpixmap.size())
        self.update()

    def clear_pixmap(self, text: str = "No Image Loaded"):
        self.current_qpixmap = None
        self.setPixmap(QPixmap())
        self.setMinimumSize(QSize(100, 100))
        self.setText(text)
        self.update()

    def _current_rect(self) -> Optional[QRect]:
        if not (self.crop_start_point and self.crop_end_point):
            return None
        return QRect(self.crop_start_point, self.crop_end_point).normalized()

    def _sync_pts_from_rect(self):
        r = self._current_rect()
        if not r:
            return
        tl = QPoint(r.left(), r.top())
        tr = QPoint(r.right(), r.top())
        br = QPoint(r.right(), r.bottom())
        bl = QPoint(r.left(), r.bottom())
        self.freeform_pts = [tl, tr, br, bl]

    def set_editor_mode(self, mode: int):
        self.editor_mode = mode
        if mode == MODE_VIEW:
            self.crop_start_point = None
            self.crop_end_point = None
            self.freeform_pts = []
            self._drag_idx = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        else:
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
                tl.setX(max(img_rect.left(), min(tl.x(), img_rect.right())))
                tl.setY(max(img_rect.top(), min(tl.y(), img_rect.bottom())))
                br.setX(max(img_rect.left(), min(br.x(), img_rect.right())))
                br.setY(max(img_rect.top(), min(br.y(), img_rect.bottom())))
                self.crop_start_point = tl
                self.crop_end_point = br

                tr = QPoint(br.x(), tl.y())
                bl = QPoint(tl.x(), br.y())
                self.freeform_pts = [tl, tr, br, bl]
                if self.editor_ref:
                    self.editor_ref.apply_crop_button.setEnabled(True)
            elif self.crop_mode_type == "free" and self.current_qpixmap:
                inset = max(10, min(img_rect.width(), img_rect.height()) // 10)
                tl = QPoint(img_rect.left() + inset, img_rect.top() + inset)
                tr = QPoint(img_rect.right() - inset, img_rect.top() + inset)
                br = QPoint(img_rect.right() - inset, img_rect.bottom() - inset)
                bl = QPoint(img_rect.left() + inset, img_rect.bottom() - inset)
                self.freeform_pts = [tl, tr, br, bl]
        self.update()

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
            r = self._current_rect()
            if not r:
                return
            r = r & img_rect
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
            painter.drawRect(r)
            painter.setBrush(QColor(255, 0, 0))
            rr = self._handle_radius
            for pt in [
                QPoint(r.left(), r.top()),
                QPoint(r.right(), r.top()),
                QPoint(r.right(), r.bottom()),
                QPoint(r.left(), r.bottom()),
            ]:
                painter.drawEllipse(
                    QtCore.QRectF(pt.x() - rr, pt.y() - rr, 2 * rr, 2 * rr)
                )
            return

        if len(self.freeform_pts) == 4:
            painter.save()
            painter.setBrush(QColor(0, 0, 0, 90))
            painter.setPen(Qt.PenStyle.NoPen)
            outer = QtCore.QPainterPath()
            outer.addRect(QtCore.QRectF(self.rect()))
            polyf = QtCore.QPolygonF(
                [QtCore.QPointF(p) for p in self.freeform_pts]
            )
            inner = QtCore.QPainterPath()
            inner.addPolygon(polyf)
            inner.closeSubpath()
            painter.drawPath(outer.subtracted(inner))
            painter.restore()

            painter.setPen(QPen(QColor(255, 0, 0), 2))
            for i in range(4):
                a = self.freeform_pts[i]
                b = self.freeform_pts[(i + 1) % 4]
                painter.drawLine(a, b)

            painter.setBrush(QColor(255, 0, 0))
            rr = self._handle_radius
            for pt in self.freeform_pts:
                painter.drawEllipse(
                    QtCore.QRectF(pt.x() - rr, pt.y() - rr, 2 * rr, 2 * rr)
                )

    def _clamp_to_img_rect(self, p: QPoint) -> QPoint:
        r = self.get_image_display_rect()
        x = max(r.left(), min(p.x(), r.right()))
        y = max(r.top(), min(p.y(), r.bottom()))
        return QPoint(int(x), int(y))

    def _hit_handle(self, pos: QPoint) -> Optional[int]:
        if self.crop_mode_type == "rect":
            r = self._current_rect()
            if not r:
                return None
            pts = [
                QPoint(r.left(), r.top()),
                QPoint(r.right(), r.top()),
                QPoint(r.right(), r.bottom()),
                QPoint(r.left(), r.bottom()),
            ]
        else:
            pts = list(self.freeform_pts)

        radius = self._handle_radius * 1.6
        for i, pt in enumerate(pts):
            if (
                abs(pos.x() - pt.x()) <= radius
                and abs(pos.y() - pt.y()) <= radius
            ):
                return i
        return None

    def mousePressEvent(self, event: QMouseEvent):
        img_rect = self.get_image_display_rect()
        if (
            self.editor_mode != MODE_CROP
            or not self.current_qpixmap
            or not img_rect.contains(event.position().toPoint())
        ):
            return

        p = self._clamp_to_img_rect(event.position().toPoint())
        if self.crop_mode_type == "rect":
            hit = self._hit_handle(p)
            if hit is not None:
                self._drag_idx = hit
            if self.editor_ref:
                self.editor_ref.apply_crop_button.setEnabled(True)
        else:
            if len(self.freeform_pts) == 4:
                hit = self._hit_handle(p)
                if hit is not None:
                    self._drag_idx = hit

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.editor_mode != MODE_CROP:
            return
        p = self._clamp_to_img_rect(event.position().toPoint())

        if self.crop_mode_type == "rect":
            if self._drag_idx is None:
                return
            r = self._current_rect()
            if not r:
                return
            tl = QPoint(r.left(), r.top())
            tr = QPoint(r.right(), r.top())
            br = QPoint(r.right(), r.bottom())
            bl = QPoint(r.left(), r.bottom())
            if self._drag_idx == 0:
                tl = p
            elif self._drag_idx == 1:
                tr = QPoint(p.x(), p.y())
            elif self._drag_idx == 2:
                br = p
            elif self._drag_idx == 3:
                bl = QPoint(p.x(), p.y())

            new_left = min(tl.x(), bl.x())
            new_right = max(tr.x(), br.x())
            new_top = min(tl.y(), tr.y())
            new_bottom = max(bl.y(), br.y())
            self.crop_start_point = QPoint(new_left, new_top)
            self.crop_end_point = QPoint(new_right, new_bottom)
            self._sync_pts_from_rect()
            self.update()
            return

        if self._drag_idx is not None and 0 <= self._drag_idx < len(
            self.freeform_pts
        ):
            self.freeform_pts[self._drag_idx] = p
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.editor_mode != MODE_CROP:
            return
        if self.crop_mode_type == "rect":
            if self.editor_ref:
                r = self._current_rect()
                ok = bool(r and r.width() > 10 and r.height() > 10)
                self.editor_ref.apply_crop_button.setEnabled(ok)
            self._drag_idx = None
            return
        if self.editor_ref and len(self.freeform_pts) == 4:
            xs = [p.x() for p in self.freeform_pts]
            ys = [p.y() for p in self.freeform_pts]
            area_ok = (max(xs) - min(xs) >= 10) and (
                max(ys) - min(ys) >= 10
            )
            self.editor_ref.apply_crop_button.setEnabled(bool(area_ok))
        self._drag_idx = None


class DualImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split)

        left = QWidget()
        vl = QVBoxLayout(left)
        vl.setContentsMargins(2, 2, 2, 2)
        self.left_title = QLabel("Original")
        self.left_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_view = ImageViewer()
        self.left_view.set_editor_mode(MODE_VIEW)
        vl.addWidget(self.left_title)
        lv_scroll = QScrollArea()
        lv_scroll.setWidgetResizable(True)
        lv_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv_scroll.setWidget(self.left_view)
        self.left_scroll = lv_scroll
        vl.addWidget(lv_scroll)
        split.addWidget(left)

        right = QWidget()
        vr = QVBoxLayout(right)
        vr.setContentsMargins(2, 2, 2, 2)
        self.right_title = QLabel("Mirrored")
        self.right_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_view = ImageViewer()
        self.right_view.set_editor_mode(MODE_VIEW)
        vr.addWidget(self.right_title)
        rv_scroll = QScrollArea()
        rv_scroll.setWidgetResizable(True)
        rv_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rv_scroll.setWidget(self.right_view)
        self.right_scroll = rv_scroll
        vr.addWidget(rv_scroll)
        split.addWidget(right)

        split.setSizes([1, 1])

    def set_pixmaps(
        self,
        left: Optional[QPixmap],
        right: Optional[QPixmap],
    ):
        if left is None:
            self.left_view.clear_pixmap("No Image")
        else:
            self.left_view.set_pixmap(left)
        if right is None:
            self.right_view.clear_pixmap("No Image")
        else:
            self.right_view.set_pixmap(right)


class SliderSpinBox(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        default_val: float,
        step: float,
        parent=None,
    ):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setFixedWidth(140)
        layout.addWidget(self.label)

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setValue(default_val)
        self.spinbox.setDecimals(2)
        self.spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinbox.setFixedWidth(64)
        layout.addWidget(self.spinbox)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setTracking(True)
        self.slider.setRange(int(min_val / step), int(max_val / step))
        self.slider.setValue(int(default_val / step))
        self.slider.setTickInterval(
            max(1, int((max_val - min_val) / step / 10))
        )
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._slider_to_spinbox)
        self.spinbox.valueChanged.connect(self._spinbox_to_slider)
        self.spinbox.valueChanged.connect(self.valueChanged)

    def _slider_to_spinbox(self, val: int):
        new_val = val * self.step
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(new_val)
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(new_val)

    def _spinbox_to_slider(self, val: float):
        new_val = int(round(val / self.step))
        self.slider.blockSignals(True)
        self.slider.setValue(new_val)
        self.slider.blockSignals(False)

    def set_value(self, val: float):
        self.spinbox.setValue(val)

    def get_value(self) -> float:
        return self.spinbox.value()

    def setEnabled(self, enabled: bool):
        self.slider.setEnabled(enabled)
        self.spinbox.setEnabled(enabled)
        self.label.setEnabled(enabled)
