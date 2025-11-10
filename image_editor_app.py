#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import math
import threading
from pathlib import Path
from typing import Optional, Any, Tuple, List
from datetime import datetime

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QPushButton, QLabel, QSlider, QTabWidget, QGridLayout,
    QDoubleSpinBox, QAbstractSpinBox, QSplitter, QSizePolicy, QScrollArea,
    QMessageBox, QFileDialog, QGroupBox, QLineEdit, QStackedWidget
)
import PyQt6.QtGui as QtGui
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QCursor, QAction,
    QPalette, QResizeEvent, QPen, QPaintEvent, QMouseEvent, QKeySequence
)
from PyQt6.QtCore import (
    Qt, QSize, QPoint, QRect, pyqtSignal, QDir, QTimer,
    QThreadPool, QRunnable, QObject
)

# PIL
from PIL import Image, ImageOps, ImageEnhance

# Optional deps
_CV2 = None
_NP = None
try:
    import cv2
    _CV2 = cv2
    import numpy as np
    _NP = np
except Exception:
    _CV2 = None
    _NP = None

# Optional HEIF/AVIF support
_HEIF_PLUGIN = False
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
    _HEIF_PLUGIN = True
except Exception:
    _HEIF_PLUGIN = False

# Optional EXIF writer for JPEG/TIFF
_PIEXIF = None
try:
    import piexif  # type: ignore
    _PIEXIF = piexif
except Exception:
    _PIEXIF = None

# Optional send-to-trash (Recycle Bin / Trash)
_S2T = None
try:
    from send2trash import send2trash  # type: ignore
    _S2T = send2trash
except Exception:
    _S2T = None

# Optional MediaInfo for .mov date taken
_MEDIAINFO = None
try:
    from pymediainfo import MediaInfo  # type: ignore
    _MEDIAINFO = MediaInfo
except Exception:
    _MEDIAINFO = None

# Formats
EXT_TO_FMT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP",
    ".gif": "GIF", ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP",
    ".heic": "HEIF", ".heif": "HEIF", ".heics": "HEIF", ".heifs": "HEIF",
    ".hif": "HEIF", ".avif": "AVIF"
}
HEIF_LIKE_EXTS = {".heic", ".heif", ".heics", ".heifs", ".hif", ".avif"}
SUPPORTED_IMAGE_EXTS = set(EXT_TO_FMT.keys())
SUPPORTED_LIVE_EXTS = {".mov"}

MODE_VIEW = 0
MODE_CROP = 1

def _fmt_ts_local(ts: float) -> str:
    try:
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"

def _sanitize_exif_datetime(s: Optional[str]) -> Optional[str]:
    # EXIF usually "YYYY:MM:DD HH:MM:SS"
    if not s:
        return None
    try:
        s2 = s.strip().replace("/", ":")
        if len(s2) >= 19 and s2[4] == ":" and s2[7] == ":":
            dt = datetime.strptime(s2[:19], "%Y:%m:%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        # Some cameras write "YYYY-MM-DD HH:MM:SS"
        try:
            dt = datetime.strptime(s2[:19], "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return s
    except Exception:
        return s

def pil_to_qimage(pil_image: Image.Image) -> QImage:
    """Convert PIL Image to QImage (Format_RGBA8888) and deep-copy buffer."""
    if pil_image.mode not in ("RGB", "RGBA", "L"):
        pil_image = pil_image.convert("RGBA")
    elif pil_image.mode == "RGB":
        r, g, b = pil_image.split()
        pil_image = Image.merge(
            "RGBA", (r, g, b, Image.new("L", pil_image.size, 255))
        )
    data = pil_image.tobytes("raw", "RGBA")
    qimg = QImage(
        data,
        pil_image.width,
        pil_image.height,
        4 * pil_image.width,
        QImage.Format.Format_RGBA8888,
    ).copy()
    return qimg

def pil_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    return QPixmap.fromImage(pil_to_qimage(pil_image))

# ---------------------- Viewer Widgets ----------------------

class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundRole(QPalette.ColorRole.Dark)
        self.setAutoFillBackground(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setScaledContents(False)

        self.editor_mode = MODE_VIEW
        self.crop_start_point: Optional[QPoint] = None
        self.crop_end_point: Optional[QPoint] = None
        self.current_qpixmap: Optional[QPixmap] = None
        self.editor_ref: Optional[Any] = None

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

    def set_editor_mode(self, mode: int):
        self.editor_mode = mode
        if mode == MODE_VIEW:
            self.crop_start_point = None
            self.crop_end_point = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
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
        if self.editor_mode == MODE_CROP and self.crop_start_point and self.crop_end_point:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))

            img_rect = self.get_image_display_rect()
            if img_rect.isEmpty():
                return

            x1 = max(self.crop_start_point.x(), img_rect.left())
            y1 = max(self.crop_start_point.y(), img_rect.top())
            x2 = min(self.crop_end_point.x(), img_rect.right())
            y2 = min(self.crop_end_point.y(), img_rect.bottom())

            crop_rect = QRect(QPoint(x1, y1), QPoint(x2, y2)).normalized()
            painter.drawRect(crop_rect)

    def mousePressEvent(self, event: QMouseEvent):
        img_rect = self.get_image_display_rect()
        if self.editor_mode == MODE_CROP and self.current_qpixmap and img_rect.contains(event.position().toPoint()):
            x = max(img_rect.left(), min(event.position().x(), img_rect.right()))
            y = max(img_rect.top(), min(event.position().y(), img_rect.bottom()))
            self.crop_start_point = QPoint(int(x), int(y))
            self.crop_end_point = QPoint(int(x), int(y))
            if self.editor_ref:
                self.editor_ref.apply_crop_button.setEnabled(False)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.editor_mode == MODE_CROP and self.crop_start_point:
            img_rect = self.get_image_display_rect()
            if img_rect.isEmpty():
                return
            x = max(img_rect.left(), min(event.position().x(), img_rect.right()))
            y = max(img_rect.top(), min(event.position().y(), img_rect.bottom()))
            self.crop_end_point = QPoint(int(x), int(y))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.editor_mode == MODE_CROP and self.crop_start_point and self.crop_end_point and self.editor_ref:
            rect = QRect(self.crop_start_point, self.crop_end_point).normalized()
            if rect.width() > 10 and rect.height() > 10:
                self.editor_ref.apply_crop_button.setEnabled(True)
            else:
                self.crop_start_point = None
                self.crop_end_point = None
                self.editor_ref.apply_crop_button.setEnabled(False)
                self.update()

class DualImageViewer(QWidget):
    """Two side-by-side labeled viewers for mirror mode."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split)

        # Left (Original)
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

        # Right (Mirrored)
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

    def set_pixmaps(self, left: Optional[QPixmap], right: Optional[QPixmap]):
        if left is None:
            self.left_view.clear_pixmap("No Image")
        else:
            self.left_view.set_pixmap(left)
        if right is None:
            self.right_view.clear_pixmap("No Image")
        else:
            self.right_view.set_pixmap(right)

# ---------------------- SliderSpinBox ----------------------

class SliderSpinBox(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float, default_val: float, step: float, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setFixedWidth(120)
        layout.addWidget(self.label)

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setValue(default_val)
        self.spinbox.setDecimals(2)
        self.spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spinbox.setFixedWidth(58)
        layout.addWidget(self.spinbox)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(min_val / step), int(max_val / step))
        self.slider.setValue(int(default_val / step))
        self.slider.setTickInterval(max(1, int((max_val - min_val) / step / 10)))
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTracking(True)
        layout.addWidget(self.slider)

        # connections
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

# ---------------------- Preview Worker ----------------------

class _PreviewSignals(QObject):
    # job_id, main_qimage, mirror_qimage_or_none
    done = pyqtSignal(int, QImage, object)

class _PreviewJob(QRunnable):
    def __init__(
        self,
        job_id: int,
        base_image: Image.Image,
        rotation_degrees: int,
        factors: dict,
        single_target_size: Tuple[int, int],
        do_mirror: bool,
        interactive: bool = False,  # FIX: plumb interactive flag
    ):
        super().__init__()
        self.job_id = job_id
        self.image = base_image
        self.rotation_degrees = rotation_degrees
        self.factors = factors
        self.target_size = single_target_size
        self.do_mirror = do_mirror
        self.interactive = interactive
        self.signals = _PreviewSignals()

    def run(self):
        img = self.image
        try:
            rotated = img.rotate(self.rotation_degrees, expand=True, resample=Image.Resampling.NEAREST)
        except Exception:
            rotated = img.rotate(self.rotation_degrees, expand=True)

        target_w, target_h = self.target_size
        if self.interactive:
            target_w = min(target_w, 960)
            target_h = min(target_h, 960)
        if target_w <= 0 or target_h <= 0:
            target_w, target_h = max(1, rotated.size[0]), max(1, rotated.size[1])

        rw, rh = rotated.size
        ratio = min(target_w / float(rw), target_h / float(rh))
        new_size = (max(1, int(rw * ratio)), max(1, int(rh * ratio)))
        preview = rotated.resize(new_size, Image.Resampling.BILINEAR)

        final_preview = self._apply_edits(preview, self.factors)

        main_qimage = pil_to_qimage(final_preview)

        mirror_qimage = None
        if self.do_mirror:
            try:
                flip_const = Image.Transpose.FLIP_LEFT_RIGHT
            except AttributeError:
                flip_const = Image.FLIP_LEFT_RIGHT
            mirror_preview = final_preview.transpose(flip_const)
            mirror_qimage = pil_to_qimage(mirror_preview)

        self.signals.done.emit(self.job_id, main_qimage, mirror_qimage)

    def _apply_edits(self, pil_image: Image.Image, f: dict) -> Image.Image:
        # Ensure RGB working image
        img = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
        working_img = img.copy()

        # --- Global tone (PIL enhancers first) ---
        # Exposure (approx in stops)
        exposure_adj = 1.0 + (f["exposure"] / 2.0)
        working_img = ImageEnhance.Brightness(working_img).enhance(exposure_adj)

        # Brightness
        brightness_adj = 1.0 + (f["brightness"] / 100.0)
        working_img = ImageEnhance.Brightness(working_img).enhance(brightness_adj)

        # Contrast
        contrast_adj = 1.0 + (f["contrast"] / 100.0)
        working_img = ImageEnhance.Contrast(working_img).enhance(contrast_adj)

        # --- Midtones/highlights/shadows in float space ---
        if _NP is not None:
            img_np = _NP.array(working_img).astype(_NP.float32) / 255.0

            # Highlights: gamma on upper half
            if f["highlights"] != 0:
                gamma_h = 1.0 - (f["highlights"] / 200.0)
                img_np = _NP.where(img_np > 0.5, _NP.power(img_np, gamma_h), img_np)

            # Shadows: gamma on lower half
            if f["shadows"] != 0:
                gamma_s = 1.0 + (f["shadows"] / 200.0)
                img_np = _NP.where(img_np < 0.5, _NP.power(img_np, gamma_s), img_np)

            # Blackpoint: lift or lower baseline
            if f["blackpoint"] != 0:
                offset = f["blackpoint"] / 500.0
                img_np = _NP.clip(img_np + offset, 0.0, 1.0)

            working_img = Image.fromarray((_NP.clip(img_np, 0.0, 1.0) * 255).astype(_NP.uint8))

        # --- Brilliance (local contrast on L channel, positive = boost, negative = compress) ---
        # Works while dragging, using lighter params for responsiveness.
        if _CV2 is not None and _NP is not None:
            brill = float(f["brilliance"])
            if abs(brill) > 0.01:
                img_rgb = _NP.array(working_img)
                lab = _CV2.cvtColor(img_rgb, _CV2.COLOR_RGB2LAB)
                L, a, b = _CV2.split(lab)

                if brill >= 0:
                    # Positive brilliance: CLAHE on L
                    # lighter settings while dragging
                    tiles = 4 if self.interactive else 8
                    clip = 1.0 + (brill / 100.0) * (4.0 if self.interactive else 8.0)  # 1..~9
                    clip = max(1.01, min(clip, 9.0))
                    clahe = _CV2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
                    L2 = clahe.apply(L)
                else:
                    # Negative brilliance: compress micro-contrast (blend with blurred L)
                    k = int(round(3 + (abs(brill) / 100.0) * (6 if self.interactive else 12)))
                    if k % 2 == 0:
                        k += 1
                    L_blur = _CV2.GaussianBlur(L, (k, k), 0)
                    alpha = (0.4 if self.interactive else 0.8) * (abs(brill) / 100.0)  # fraction of blur to mix in
                    alpha = max(0.0, min(alpha, 0.9))
                    L2 = _CV2.addWeighted(L, 1.0 - alpha, L_blur, alpha, 0)

                lab2 = _CV2.merge((L2, a, b))
                rgb2 = _CV2.cvtColor(lab2, _CV2.COLOR_LAB2RGB)
                working_img = Image.fromarray(rgb2)

        # --- Color (after brilliance) ---
        # Saturation (global)
        sat_adj = 1.0 + (f["saturation"] / 100.0)
        working_img = ImageEnhance.Color(working_img).enhance(sat_adj)

        if _NP is not None:
            img_rgb_np = _NP.array(working_img).astype(_NP.float32)

            # Vibrance (boost low-sat more than high-sat)
            if f["vibrance"] != 0:
                hsv = _NP.array(working_img.convert("HSV")).astype(_NP.float32)
                v_adj = f["vibrance"] / 100.0
                saturation = hsv[:, :, 1] / 255.0
                mask = 1.0 - saturation
                hsv[:, :, 1] = _NP.clip(hsv[:, :, 1] + (hsv[:, :, 1] * v_adj * mask), 0, 255)
                working_img = Image.fromarray(hsv.astype(_NP.uint8), "HSV").convert("RGB")
                img_rgb_np = _NP.array(working_img).astype(_NP.float32)

            # Warmth (R/B gain) and Tint (G gain)
            if f["warmth"] != 0:
                r_gain = 1.0 + (f["warmth"] / 150.0)
                b_gain = 1.0 - (f["warmth"] / 150.0)
                img_rgb_np[:, :, 0] = _NP.clip(img_rgb_np[:, :, 0] * r_gain, 0, 255)
                img_rgb_np[:, :, 2] = _NP.clip(img_rgb_np[:, :, 2] * b_gain, 0, 255)
            if f["tint"] != 0:
                g_gain = 1.0 + (f["tint"] / 150.0)
                img_rgb_np[:, :, 1] = _NP.clip(img_rgb_np[:, :, 1] * g_gain, 0, 255)

            working_img = Image.fromarray(img_rgb_np.astype(_NP.uint8))

        # --- Noise Reduction BEFORE sharpness ---
        if _CV2 is not None and _NP is not None and f["noise_reduction"] > 0.0:
            nr = float(f["noise_reduction"])
            img_bgr = _CV2.cvtColor(_NP.array(working_img), _CV2.COLOR_RGB2BGR)
            if self.interactive:
                # Cheaper real-time approximation
                d = 5
                sigma = 5 + int(nr * 5)  # gentle, scales with slider
                den = _CV2.bilateralFilter(img_bgr, d, sigma, sigma)
            else:
                # Full-strength denoise
                h = 3 + int(nr * 7)      # ~3..73 across slider range 0..10
                den = _CV2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)
            working_img = Image.fromarray(_CV2.cvtColor(den, _CV2.COLOR_BGR2RGB))

        # --- Sharpness AFTER denoise (less haloing) ---
        sharp_adj = 1.0 + ((f["sharpness"] / 100.0) * (0.5 if self.interactive else 1.0))
        working_img = ImageEnhance.Sharpness(working_img).enhance(sharp_adj)

        # --- Vignette (vectorized) ---
        if f["vignette"] > 0.1 and _NP is not None:
            image = working_img
            if image.mode != "RGB":
                image = image.convert("RGB")
            w, h = image.size
            y_coords, x_coords = _NP.indices((h, w))
            cx, cy = w / 2.0, h / 2.0
            radius = min(w, h) / 2.0
            dist = _NP.hypot(x_coords - cx, y_coords - cy) / radius
            falloff = _NP.power(dist, (1.0 + (f["vignette"] * 0.5)))
            mask = _NP.clip(1.0 - (falloff * 0.6), 0.0, 1.0)
            image_np = _NP.array(image).astype(_NP.float32) / 255.0
            final_np = _NP.clip(image_np * mask[:, :, _NP.newaxis], 0.0, 1.0)
            working_img = Image.fromarray((final_np * 255).astype(_NP.uint8))

        return working_img

# ---------------------- Main App ----------------------

class ImageEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced PyQt6 Image Editor + Live Photo Tool")
        self.setGeometry(100, 100, 1300, 850)

        # shared state
        self.current_folder: Optional[Path] = None

        # photos state
        self.image_files: List[str] = []
        self.current_image_index: int = -1
        self.original_image_path: Optional[str] = None
        self.original_image_pil: Optional[Image.Image] = None
        self.working_image_pil: Optional[Image.Image] = None
        self._orig_format: Optional[str] = None
        self._orig_exif: Optional[bytes] = None
        self.crop_area: Optional[Tuple[int, int, int, int]] = None

        # edits state
        self.rotation_degrees = 0
        self.brightness_factor = 0.0
        self.contrast_factor = 0.0
        self.exposure_factor = 0.0
        self.highlights_factor = 0.0
        self.shadows_factor = 0.0
        self.blackpoint_factor = 0.0
        self.saturation_factor = 0.0
        self.vibrance_factor = 0.0
        self.warmth_factor = 0.0
        self.tint_factor = 0.0
        self.sharpness_factor = 0.0
        self.noise_reduction_factor = 0.0
        self.vignette_factor = 0.0
        self.brilliance_factor = 0.0

        # live tab state
        self.mov_files: List[str] = []
        self.current_mov_index: int = -1
        self.cap = None
        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._advance_video_frame)
        self.video_fps = 30.0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_playing = False

        # mirror state
        self._mirror_on = False

        # sliders + preview source
        self._all_sliders = []          # type: list[QSlider]
        self._preview_source = None     # type: Optional[Image.Image]

        # preview rendering infra (dedicated single-worker pool)
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(1)
        self._preview_job_id = 0
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(16)  # ~60 Hz debounce
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._start_preview_job)
        self._preview_lock = threading.Lock()

        self._init_ui()
        self.set_controls_state(False)

        # Global Delete shortcut -> move selected file to trash
        self.delete_action = QAction("Delete", self)
        self.delete_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        self.delete_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.delete_action.triggered.connect(self._on_delete_key)
        self.addAction(self.delete_action)

        if _CV2 is None or _NP is None:
            QMessageBox.warning(
                self,
                "Missing Dependencies",
                "Video (.mov) or advanced edits require opencv-python and numpy.\n\n"
                "pip install opencv-python numpy",
            )

    # ---------- UI ----------

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Photos tab
        self.photos_tab = QWidget()
        self.tabs.addTab(self.photos_tab, "Photos")
        self._build_photos_tab(self.photos_tab)

        # Live tab
        self.live_tab = self._build_live_tab()
        self.tabs.addTab(self.live_tab, "Live (.mov)")

        # when switching tabs, ensure timers/preview behave
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _build_photos_tab(self, tab: QWidget):
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split)

        # left: sidebar
        left = QWidget()
        lv = QVBoxLayout(left)

        self.photo_load_btn = QPushButton("Load Folder...")
        self.photo_load_btn.clicked.connect(self._browse_and_load_folder)
        lv.addWidget(self.photo_load_btn)

        self.photo_list = QListWidget()
        self.photo_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.photo_list.currentItemChanged.connect(self._on_photo_select)
        lv.addWidget(self.photo_list)
        split.addWidget(left)

        # right: viewer + editor
        right = QWidget()
        rv = QVBoxLayout(right)

        viewer_editor_split = QSplitter(Qt.Orientation.Vertical)
        rv.addWidget(viewer_editor_split)

        # viewer stack: single or dual for mirror mode
        self.viewer_stack = QStackedWidget()
        # single viewer
        self.single_viewer = ImageViewer()
        self.single_viewer.editor_ref = self
        self.single_scroll = QScrollArea()
        self.single_scroll.setWidgetResizable(True)
        self.single_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.single_scroll.setWidget(self.single_viewer)
        single_holder = QWidget()
        svl = QVBoxLayout(single_holder)
        svl.setContentsMargins(0, 0, 0, 0)
        svl.addWidget(self.single_scroll)
        self.viewer_stack.addWidget(single_holder)

        # dual viewer (mirror display)
        self.dual_viewer = DualImageViewer()
        self.viewer_stack.addWidget(self.dual_viewer)

        viewer_editor_split.addWidget(self.viewer_stack)

        # editor controls
        self.editor_widget = QTabWidget()
        viewer_editor_split.addWidget(self.editor_widget)
        viewer_editor_split.setSizes([700, 300])

        # tab1 transform/save + metadata
        t1 = QWidget()
        t1l = QVBoxLayout(t1)
        t1l.setAlignment(Qt.AlignmentFlag.AlignTop)

        rotate_group = QGroupBox("Rotation")
        rgl = QHBoxLayout(rotate_group)
        self.rotate_left_button = QPushButton("Rotate Left (90)")
        self.rotate_left_button.clicked.connect(lambda: self.rotate_image(-90))
        self.rotate_right_button = QPushButton("Rotate Right (90)")
        self.rotate_right_button.clicked.connect(lambda: self.rotate_image(90))
        rgl.addWidget(self.rotate_left_button)
        rgl.addWidget(self.rotate_right_button)
        t1l.addWidget(rotate_group)

        crop_group = QGroupBox("Crop / Mirror")
        cgl = QHBoxLayout(crop_group)
        self.crop_mode_button = QPushButton("Enable Crop Mode")
        self.crop_mode_button.setCheckable(True)
        self.crop_mode_button.toggled.connect(self._toggle_crop_mode)
        self.apply_crop_button = QPushButton("Apply Crop")
        self.apply_crop_button.setEnabled(False)
        self.apply_crop_button.clicked.connect(self.apply_crop)

        self.mirror_mode_button = QPushButton("Mirror View")
        self.mirror_mode_button.setCheckable(True)
        self.mirror_mode_button.toggled.connect(self._toggle_mirror_mode)

        cgl.addWidget(self.crop_mode_button)
        cgl.addWidget(self.apply_crop_button)
        cgl.addWidget(self.mirror_mode_button)
        t1l.addWidget(crop_group)

        save_group = QGroupBox("Save & Reset")
        sgl = QHBoxLayout(save_group)
        self.save_button = QPushButton("Save (Overwrite)")
        self.save_button.clicked.connect(self.save_image)
        self.save_as_button = QPushButton("Save As...")
        self.save_as_button.clicked.connect(self.save_image_as)
        self.reset_button = QPushButton("Reset All")
        self.reset_button.clicked.connect(self.reset_edits)
        sgl.addWidget(self.save_button)
        sgl.addWidget(self.save_as_button)
        sgl.addWidget(self.reset_button)
        t1l.addWidget(save_group)

        # metadata group
        meta_group = QGroupBox("Metadata")
        mgl = QGridLayout(meta_group)
        self.meta_taken_lbl = QLabel("Taken: -")
        self.meta_modified_lbl = QLabel("Modified: -")
        self.meta_note_edit = QLineEdit()
        self.meta_note_edit.setPlaceholderText("Custom note (JPEG/TIFF via EXIF ImageDescription/UserComment)")
        self.meta_write_btn = QPushButton("Write Note to File")
        self.meta_write_btn.clicked.connect(self._write_custom_note_to_file)
        # enable only when piexif and format supported
        if _PIEXIF is None:
            self.meta_note_edit.setEnabled(False)
            self.meta_write_btn.setEnabled(False)
            self.meta_write_btn.setToolTip("piexif not installed. pip install piexif")
        mgl.addWidget(self.meta_taken_lbl, 0, 0, 1, 2)
        mgl.addWidget(self.meta_modified_lbl, 1, 0, 1, 2)
        mgl.addWidget(self.meta_note_edit, 2, 0, 1, 1)
        mgl.addWidget(self.meta_write_btn, 2, 1, 1, 1)
        t1l.addWidget(meta_group)

        self.editor_widget.addTab(t1, "Transform / Save")

        # tab2 tone/light
        t2 = QWidget()
        t2l = QVBoxLayout(t2)
        t2l.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.exposure_slider = self._create_slider(t2l, "Exposure (Stops)", -2.0, 2.0, 0.01, "exposure_factor")
        self.brilliance_slider = self._create_slider(t2l, "Brilliance", -100.0, 100.0, 1.0, "brilliance_factor")
        self.brightness_slider = self._create_slider(t2l, "Brightness", -100.0, 100.0, 1.0, "brightness_factor")
        self.contrast_slider = self._create_slider(t2l, "Contrast", -100.0, 100.0, 1.0, "contrast_factor")
        self.highlights_slider = self._create_slider(t2l, "Highlights", -100.0, 100.0, 1.0, "highlights_factor")
        self.shadows_slider = self._create_slider(t2l, "Shadows", -100.0, 100.0, 1.0, "shadows_factor")
        self.blackpoint_slider = self._create_slider(t2l, "Blackpoint", -100.0, 100.0, 1.0, "blackpoint_factor")

        self.editor_widget.addTab(t2, "Tone / Light")

        # tab3 color/effects
        t3 = QWidget()
        t3l = QVBoxLayout(t3)
        t3l.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.saturation_slider = self._create_slider(t3l, "Saturation", -100.0, 100.0, 1.0, "saturation_factor")
        self.vibrance_slider = self._create_slider(t3l, "Vibrance", -100.0, 100.0, 1.0, "vibrance_factor")
        self.warmth_slider = self._create_slider(t3l, "Warmth (Temp)", -100.0, 100.0, 1.0, "warmth_factor")
        self.tint_slider = self._create_slider(t3l, "Tint (G/M)", -100.0, 100.0, 1.0, "tint_factor")
        self.sharpness_slider = self._create_slider(t3l, "Sharpness", -100.0, 100.0, 1.0, "sharpness_factor")
        self.noise_reduction_slider = self._create_slider(t3l, "Noise Reduction", 0.0, 10.0, 0.1, "noise_reduction_factor")
        self.vignette_slider = self._create_slider(t3l, "Vignette", 0.0, 10.0, 0.01, "vignette_factor")

        self.editor_widget.addTab(t3, "Color / Effects")

        split.addWidget(right)
        split.setSizes([260, 1040])

        # initial viewer text
        self.single_viewer.clear_pixmap()

    def _build_live_tab(self) -> QWidget:
        """Creates the Live (.mov) tab with file chooser and a separate frames chooser beneath it."""
        tab = QWidget()
        root = QHBoxLayout(tab)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # Left panel: file chooser + frames chooser stacked vertically (half/half)
        left_panel = QWidget()
        left_v = QVBoxLayout(left_panel)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(6)

        self.live_load_btn = QPushButton("Load Folder...")
        self.live_load_btn.clicked.connect(self._browse_and_load_folder)
        left_v.addWidget(self.live_load_btn)

        vstack = QSplitter(Qt.Orientation.Vertical)

        self.live_list = QListWidget()
        self.live_list.setMinimumWidth(200)
        self.live_list.setMaximumWidth(420)
        self.live_list.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.live_list.itemSelectionChanged.connect(lambda: self._on_live_select(self.live_list.currentItem(), None))
        vstack.addWidget(self.live_list)

        self.frames_list = QListWidget()
        self.frames_list.setMinimumWidth(200)
        self.frames_list.setMaximumWidth(420)
        self.frames_list.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.frames_list.itemSelectionChanged.connect(
            lambda: self._on_frame_selected(self.frames_list.currentItem(), None)
        )
        vstack.addWidget(self.frames_list)

        # split half/half
        vstack.setSizes([1, 1])
        left_v.addWidget(vstack)
        splitter.addWidget(left_panel)

        # Right: big player + compact controls
        right_panel = QWidget()
        right_v = QVBoxLayout(right_panel)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(6)

        self.player_label = QLabel("No .mov loaded")
        self.player_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.player_label.setMinimumSize(320, 240)
        self.player_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.player_label.setStyleSheet("background:#000; color:#aaa;")
        right_v.addWidget(self.player_label, 1)

        controls_row = QWidget()
        controls_h = QHBoxLayout(controls_row)
        controls_h.setContentsMargins(0, 0, 0, 0)
        controls_h.setSpacing(8)

        self.play_button = QPushButton("Play")
        self.play_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.play_button.setMaximumWidth(110)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_play_pause)

        self.save_frame_button = QPushButton("Save Selected Frame")
        self.save_frame_button.setEnabled(False)
        self.save_frame_button.clicked.connect(self._save_selected_frame)

        controls_h.addWidget(self.play_button)
        controls_h.addWidget(self.save_frame_button)
        controls_h.addStretch(1)
        right_v.addWidget(controls_row, 0)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1200])

        return tab

    # ---------- Folder load ----------

    def _browse_and_load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select a folder", QDir.homePath()
        )
        if not folder_path:
            return
        self.load_directory(Path(folder_path))

    def _image_date_taken(self, path: Path) -> Optional[str]:
        # EXIF DateTimeOriginal (0x9003) or DateTime (0x0132)
        try:
            if path.suffix.lower() in HEIF_LIKE_EXTS and not _HEIF_PLUGIN:
                return None
            with Image.open(str(path)) as im:
                exif = getattr(im, "getexif", None)
                if not exif:
                    return None
                ex = exif()
                if not ex:
                    return None
                dto = ex.get(36867) or ex.get(306)  # DateTimeOriginal or DateTime
                return _sanitize_exif_datetime(dto)
        except Exception:
            return None

    def _mov_date_taken(self, path: Path) -> Optional[str]:
        # Best-effort via pymediainfo if available; otherwise None
        if _MEDIAINFO is None:
            return None
        try:
            info = _MEDIAINFO.parse(str(path))
            for track in info.tracks:
                if track.track_type == "General":
                    cand = getattr(track, "encoded_date", None) or getattr(track, "tagged_date", None)
                    if not cand:
                        continue
                    s = str(cand)
                    s = s.replace("UTC ", "").strip()
                    try:
                        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
                        return dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        return s
            return None
        except Exception:
            return None

    def _format_list_item_with_meta(self, p: Path, date_taken: Optional[str]) -> str:
        date_mod = _fmt_ts_local(os.path.getmtime(p))
        taken = date_taken if date_taken else "-"
        return f"{p.name} | taken: {taken} | modified: {date_mod}"

    def load_directory(self, folder: Path):
        """Load both photos and .mov lists in one go, from either tab."""
        try:
            if not folder.is_dir():
                QMessageBox.warning(self, "Folder", f"Folder not found: {folder}")
                return

            self.current_folder = folder

            image_paths: List[Path] = []
            mov_paths: List[Path] = []
            for p in folder.iterdir():
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTS:
                    image_paths.append(p)
                elif ext in SUPPORTED_LIVE_EXTS:
                    mov_paths.append(p)

            image_paths.sort(key=lambda x: str(x).lower())
            mov_paths.sort(key=lambda x: str(x).lower())

            # populate photos tab with metadata
            self.image_files = [str(p) for p in image_paths]
            self.photo_list.clear()
            for p in image_paths:
                date_taken = self._image_date_taken(p)
                item_text = self._format_list_item_with_meta(p, date_taken)
                self.photo_list.addItem(item_text)
                self.photo_list.item(self.photo_list.count() - 1).setToolTip(str(p))

            if self.image_files:
                self._load_image_by_index(0)
                self.photo_list.setCurrentRow(0)
                self.set_controls_state(True)
            else:
                self._clear_photos_state()
                self.set_controls_state(False)
                self.single_viewer.clear_pixmap("No Images Found")

            # populate live tab with metadata
            self._stop_video()
            self.mov_files = [str(p) for p in mov_paths]
            self.live_list.clear()
            self.frames_list.clear()
            if self.mov_files:
                for p in mov_paths:
                    taken = self._mov_date_taken(p)
                    item_text = self._format_list_item_with_meta(p, taken)
                    self.live_list.addItem(item_text)
                    self.live_list.item(self.live_list.count() - 1).setToolTip(str(p))

                # auto-open the first .mov
                self.current_mov_index = 0
                self._open_video(self.mov_files[0])
                self.play_button.setEnabled(True)
            else:
                self._clear_live_state()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not process folder: {e}")

    # ---------- Photos: list and load ----------

    def _on_photo_select(self, current, previous):
        if not self.image_files or not current:
            return
        index = self.photo_list.row(current)
        if index != self.current_image_index:
            self._load_image_by_index(index)

    def _load_image_by_index(self, index: int):
        if not (0 <= index < len(self.image_files)):
            return
        self.current_image_index = index
        self._load_image(self.image_files[index])

    def _ensure_heif_plugin_for_path(self, path: str, when: str) -> bool:
        ext = Path(path).suffix.lower()
        if ext in HEIF_LIKE_EXTS and not _HEIF_PLUGIN:
            QMessageBox.critical(
                self,
                "HEIF/AVIF support missing",
                f"You attempted to {when} a {ext} file but pillow-heif is not installed.\n\n"
                f"pip install pillow-heif\n"
                f"(On some systems, also install libheif.)",
            )
            return False
        return True

    def _infer_format_from_path(self, path: str) -> Optional[str]:
        return EXT_TO_FMT.get(Path(path).suffix.lower())

    def _update_meta_labels_for_image(self, path: Path):
        taken = self._image_date_taken(path) or "-"
        modified = _fmt_ts_local(os.path.getmtime(path))
        self.meta_taken_lbl.setText(f"Taken: {taken}")
        self.meta_modified_lbl.setText(f"Modified: {modified}")

        fmt = (self._orig_format or self._infer_format_from_path(str(path)) or "").upper()
        writable = (_PIEXIF is not None and fmt in ("JPEG", "TIFF"))
        self.meta_note_edit.setEnabled(writable)
        self.meta_write_btn.setEnabled(writable)
        if not writable:
            if _PIEXIF is None:
                self.meta_write_btn.setToolTip("piexif not installed. pip install piexif")
            else:
                self.meta_write_btn.setToolTip("Custom note supported only for JPEG/TIFF via EXIF")

    def _load_image(self, filepath: str):
        # clear state
        self.original_image_path = None
        self.original_image_pil = None
        self.working_image_pil = None
        self._orig_format = None
        self._orig_exif = None
        self.crop_area = None
        self._reset_controls(reset_rotation=True, reset_sliders=True)
        self.single_viewer.set_editor_mode(MODE_VIEW)
        self.crop_mode_button.setChecked(False)
        self.mirror_mode_button.setChecked(False)  # default back to single view
        self._mirror_on = False

        if not os.path.exists(filepath):
            return

        if not self._ensure_heif_plugin_for_path(filepath, "open"):
            self.set_controls_state(False)
            self.single_viewer.clear_pixmap()
            return

        try:
            with Image.open(filepath) as im:
                self._orig_format = (im.format or "") or self._infer_format_from_path(filepath)
                self._orig_exif = im.info.get("exif")
                im = ImageOps.exif_transpose(im)
                self.original_image_pil = im.copy()

            self.original_image_path = filepath
            self.working_image_pil = self.original_image_pil.copy()
            self._preview_source = self._make_preview_source(self.working_image_pil)  # ensure preview source set
            self.set_controls_state(True)
            self._update_meta_labels_for_image(Path(filepath))
            self._schedule_preview()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not process image: {e}")
            self.original_image_path = None
            self.single_viewer.setText(f"Error loading image: {e}")
            self.set_controls_state(False)

    def _clear_photos_state(self):
        self.original_image_path = None
        self.original_image_pil = None
        self.working_image_pil = None
        self._orig_format = None
        self._orig_exif = None
        self.crop_area = None
        self.current_image_index = -1
        self.photo_list.clear()
        self.single_viewer.clear_pixmap()
        self.single_viewer.set_editor_mode(MODE_VIEW)
        self.crop_mode_button.setChecked(False)
        self.mirror_mode_button.setChecked(False)
        self._mirror_on = False
        self._reset_controls(reset_rotation=True, reset_sliders=True)
        self.meta_taken_lbl.setText("Taken: -")
        self.meta_modified_lbl.setText("Modified: -")

    # ---------- Live: list and playback ----------

    def _on_live_select(self, current, previous):
        if not self.mov_files or not current:
            return
        idx = self.live_list.row(current)
        if idx != self.current_mov_index:
            self._select_live_by_index(idx)

    def _select_live_by_index(self, idx: int):
        if not (0 <= idx < len(self.mov_files)):
            return
        self.current_mov_index = idx
        self._open_video(self.mov_files[idx])

    def _open_video(self, path: str):
        self._stop_video()
        self.frames_list.clear()
        self.save_frame_button.setEnabled(False)
        self.player_label.setText("Loading...")
        self.player_label.setPixmap(QPixmap())

        if _CV2 is None:
            QMessageBox.warning(self, "Live", "opencv-python is required for .mov playback.")
            return
        try:
            cap = _CV2.VideoCapture(path)
            if not cap.isOpened():
                raise IOError("Failed to open video file")
            self.cap = cap
            self.video_fps = cap.get(_CV2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(cap.get(_CV2.CAP_PROP_FRAME_COUNT) or 0)
            self.current_frame_idx = 0

            # populate frame list
            for i in range(self.total_frames):
                self.frames_list.addItem(f"Frame {i:04d}")

            self.play_button.setEnabled(True)
            self.play_button.setText("Play")
            self.is_playing = False

            # display first frame
            self._seek_and_show_frame(0)
            if self.total_frames > 0:
                self.frames_list.setCurrentRow(0)
                self.save_frame_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Live", f"Could not process video: {e}")
            self._clear_live_state()

    def _toggle_play_pause(self):
        if not self.cap:
            return
        if self.is_playing:
            self._stop_video()
        else:
            self._start_video()

    def _start_video(self):
        if not self.cap:
            return
        self.is_playing = True
        self.play_button.setText("Pause")
        interval = max(10, int(1000 / max(1.0, min(60.0, self.video_fps))))
        self.live_timer.start(interval)

    def _stop_video(self):
        self.is_playing = False
        self.live_timer.stop()
        if hasattr(self, "play_button") and self.play_button:
            self.play_button.setText("Play")

    def _advance_video_frame(self):
        if not self.cap:
            return
        ok, frame_bgr = self.cap.read()
        if not ok:
            # loop
            self.cap.set(_CV2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = self.cap.read()
            if not ok:
                return
            self.current_frame_idx = 0
        else:
            self.current_frame_idx = int(self.cap.get(_CV2.CAP_PROP_POS_FRAMES)) - 1

        self._show_frame_on_label(frame_bgr, self.player_label)

        if 0 <= self.current_frame_idx < self.frames_list.count():
            self.frames_list.blockSignals(True)
            self.frames_list.setCurrentRow(self.current_frame_idx)
            self.frames_list.blockSignals(False)

    def _on_frame_selected(self, current, previous):
        if not current or self.cap is None:
            self.save_frame_button.setEnabled(False)
            return
        self.save_frame_button.setEnabled(True)
        idx = self.frames_list.row(current)
        self._seek_and_show_frame(idx)

    def _seek_and_show_frame(self, idx: int):
        if not self.cap:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(_CV2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = self.cap.read()
        if not ok:
            return
        self.current_frame_idx = idx
        self._show_frame_on_label(frame_bgr, self.player_label)

    def _show_frame_on_label(self, frame_bgr, label: QLabel):
        try:
            frame_rgb = _CV2.cvtColor(frame_bgr, _CV2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg)
            target = label.size()
            if target.width() > 0 and target.height() > 0:
                pm = pm.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(pm)
            label.setText("")
        except Exception as e:
            label.setText(f"Frame error: {e}")

    def _save_selected_frame(self):
        if self.cap is None:
            return
        current = self.frames_list.currentRow()
        if current < 0:
            QMessageBox.warning(self, "Save Frame", "Select a frame first.")
            return
        if not (0 <= self.current_mov_index < len(self.mov_files)):
            return

        mov_path = Path(self.mov_files[self.current_mov_index])
        save_name = f"{mov_path.stem}_img_{current:04d}.jpeg"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Frame As", str(mov_path.parent / save_name),
            "JPEG (*.jpeg *.jpg);;PNG (*.png);;All Files (*.*)"
        )
        if not save_path:
            return

        try:
            self.cap.set(_CV2.CAP_PROP_POS_FRAMES, current)
            ok, frame_bgr = self.cap.read()
            if not ok:
                QMessageBox.warning(self, "Save Frame", "Could not read frame.")
                return
            frame_rgb = _CV2.cvtColor(frame_bgr, _CV2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            if Path(save_path).suffix.lower() in (".jpg", ".jpeg"):
                img.save(save_path, "JPEG", quality=90)
            else:
                img.save(save_path)
            QMessageBox.information(self, "Save Frame", f"Saved:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Frame", f"Error: {e}")

    def _clear_live_state(self):
        self._stop_video()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.mov_files = []
        self.current_mov_index = -1
        if hasattr(self, "live_list"):
            self.live_list.clear()
        if hasattr(self, "frames_list"):
            self.frames_list.clear()
        self.player_label.setText("No .mov loaded")
        self.player_label.setPixmap(QPixmap())
        self.play_button.setEnabled(False)
        self.save_frame_button.setEnabled(False)

    # ---------- Photos: controls ----------
    def _make_preview_source(self, img: Image.Image, max_side: int = 2048) -> Image.Image:
        w, h = img.size
        m = max(w, h)
        if m <= max_side:
            return img.convert("RGB") if img.mode != "RGB" else img
        ratio = max_side / float(m)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        return img.convert("RGB").resize(new_size, Image.Resampling.BILINEAR)

    def _create_slider(self, parent_layout: QVBoxLayout, label: str, min_val: float, max_val: float, step: float, attr_name: str) -> 'SliderSpinBox':
        default_val = getattr(self, attr_name)
        s = SliderSpinBox(label, min_val, max_val, default_val, step)
        self._all_sliders.append(s.slider)
        s.valueChanged.connect(lambda v: self._on_slider_change(v, attr_name))
        setattr(self, attr_name.replace("_factor", "_slider"), s)
        parent_layout.addWidget(s)
        return s

    def set_controls_state(self, enabled: bool):
        self.rotate_left_button.setEnabled(enabled)
        self.rotate_right_button.setEnabled(enabled)
        self.crop_mode_button.setEnabled(enabled)
        self.mirror_mode_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.save_as_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)

        for attr_name in [
            "exposure", "brilliance", "brightness", "contrast",
            "highlights", "shadows", "blackpoint",
            "saturation", "vibrance", "warmth", "tint",
            "sharpness", "noise_reduction", "vignette"
        ]:
            slider = getattr(self, f"{attr_name}_slider", None)
            if slider:
                slider.setEnabled(enabled)

    def _reset_controls(self, reset_rotation=True, reset_sliders=True):
        if reset_rotation:
            self.rotation_degrees = 0
        if reset_sliders:
            for name in [
                "brightness", "contrast", "exposure", "highlights", "shadows",
                "blackpoint", "saturation", "vibrance", "warmth", "tint",
                "sharpness", "noise_reduction", "vignette", "brilliance"
            ]:
                setattr(self, f"{name}_factor", 0.0)
                slider = getattr(self, f"{name}_slider", None)
                if slider:
                    slider.set_value(0.0)

    def reset_edits(self):
        if not self.original_image_pil:
            return
        self.working_image_pil = self.original_image_pil.copy()
        self._preview_source = self._make_preview_source(self.working_image_pil)
        self.crop_area = None
        self._reset_controls(reset_rotation=True, reset_sliders=True)
        self._schedule_preview()

    def rotate_image(self, degrees: int):
        if not self.working_image_pil:
            return
        self.rotation_degrees = (self.rotation_degrees + degrees) % 360
        self._schedule_preview()

    def _toggle_crop_mode(self, checked: bool):
        if not self.original_image_path:
            self.crop_mode_button.setChecked(False)
            return
        if checked:
            self.single_viewer.set_editor_mode(MODE_CROP)
            self.crop_mode_button.setText("Exit Crop Mode")
        else:
            self.single_viewer.set_editor_mode(MODE_VIEW)
            self.crop_mode_button.setText("Enable Crop Mode")
            self.apply_crop_button.setEnabled(False)

    def _toggle_mirror_mode(self, checked: bool):
        self._mirror_on = bool(checked)
        if checked:
            self.viewer_stack.setCurrentIndex(1)  # dual
        else:
            self.viewer_stack.setCurrentIndex(0)  # single
        self._schedule_preview()

    def apply_crop(self):
        start_point = self.single_viewer.crop_start_point
        end_point = self.single_viewer.crop_end_point

        if not start_point or not end_point or not self.original_image_pil or not self.single_viewer.current_qpixmap:
            QMessageBox.warning(self, "Crop", "No valid crop area selected.")
            return

        crop_rect_can = QRect(start_point, end_point).normalized()
        if crop_rect_can.width() < 10 or crop_rect_can.height() < 10:
            QMessageBox.warning(self, "Crop", "Selection is too small.")
            return

        qpixmap_size = self.single_viewer.current_qpixmap.size()
        w_disp, h_disp = qpixmap_size.width(), qpixmap_size.height()
        img_rect_can = self.single_viewer.get_image_display_rect()

        x1_img = crop_rect_can.left() - img_rect_can.left()
        y1_img = crop_rect_can.top() - img_rect_can.top()
        x2_img = crop_rect_can.right() - img_rect_can.left()
        y2_img = crop_rect_can.bottom() - img_rect_can.top()

        x1_img = max(0, min(x1_img, w_disp))
        y1_img = max(0, min(y1_img, h_disp))
        x2_img = max(0, min(x2_img, w_disp))
        y2_img = max(0, min(y2_img, h_disp))

        temp_rotated_size = self.working_image_pil.rotate(self.rotation_degrees, expand=True).size
        w_rot, h_rot = temp_rotated_size
        ratio_w = w_rot / w_disp if w_disp else 1.0
        ratio_h = h_rot / h_disp if h_disp else 1.0

        crop_box_rotated = (
            int(x1_img * ratio_w),
            int(y1_img * ratio_h),
            int(x2_img * ratio_w),
            int(y2_img * ratio_h),
        )

        try:
            temp_rotated_image = self.working_image_pil.rotate(self.rotation_degrees, expand=True)
            cropped_image = temp_rotated_image.crop(crop_box_rotated)
            self.working_image_pil = cropped_image
            self._preview_source = self._make_preview_source(self.working_image_pil)
            self.rotation_degrees = 0
            self.crop_mode_button.setChecked(False)
            self.single_viewer.set_editor_mode(MODE_VIEW)
            self.apply_crop_button.setEnabled(False)
            self._schedule_preview()
        except Exception as e:
            QMessageBox.critical(self, "Crop", f"Could not apply crop: {e}")

    def save_image(self) -> bool:
        if not self.original_image_path:
            QMessageBox.warning(self, "Save", "No image loaded.")
            return False
        reply = QMessageBox.question(
            self, "Confirm Overwrite",
            f"Overwrite original file?\n{self.original_image_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False
        ok = self._perform_save(self.original_image_path)
        if ok:
            QMessageBox.information(self, "Save", f"Saved:\n{self.original_image_path}")
            self._load_image(self.original_image_path)  # refresh as new original
        return ok

    def save_image_as(self) -> bool:
        if not self.original_image_path:
            QMessageBox.warning(self, "Save As", "No image loaded.")
            return False
        default_ext = Path(self.original_image_path).suffix
        filetypes = "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif *.tiff);;WEBP (*.webp);;All Files (*.*)"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image As",
            str(Path(self.original_image_path).parent / f"{Path(self.original_image_path).stem}_edited{default_ext}"),
            filetypes
        )
        if not save_path:
            return False
        ok = self._perform_save(save_path)
        if ok:
            QMessageBox.information(self, "Save", f"Saved:\n{save_path}")
        return ok

    def _perform_save(self, save_path: str) -> bool:
        try:
            if not self.working_image_pil:
                return False
            rotated_image = self.working_image_pil.rotate(
                self.rotation_degrees, expand=True, resample=Image.Resampling.LANCZOS
            )
            final_image = self._apply_edits_full(rotated_image)

            # If mirror view is on, save the flipped result (requested behavior)
            if self._mirror_on:
                try:
                    flip_const = Image.Transpose.FLIP_LEFT_RIGHT
                except AttributeError:
                    flip_const = Image.FLIP_LEFT_RIGHT
                final_image = final_image.transpose(flip_const)

            p = Path(save_path)
            if not self._ensure_heif_plugin_for_path(save_path, "save"):
                return False

            save_fmt = (self._orig_format or self._infer_format_from_path(save_path) or "PNG").upper()
            if save_fmt == "JPEG" and final_image.mode not in ("RGB", "L"):
                final_image = final_image.convert("RGB")

            save_kwargs = {"format": save_fmt}
            if self._orig_exif and save_fmt in ("JPEG", "TIFF", "WEBP", "HEIF"):
                save_kwargs["exif"] = self._orig_exif
            if save_fmt in ("JPEG", "WEBP", "HEIF", "AVIF"):
                save_kwargs.setdefault("quality", 90)

            if Path(save_path).exists() and save_path == self.original_image_path:
                tmp_path = p.with_name(f".tmp_{p.name}")
                final_image.save(str(tmp_path), **save_kwargs)
                os.replace(str(tmp_path), save_path)
            else:
                final_image.save(str(save_path), **save_kwargs)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save", f"Could not save image: {e}")
            try:
                if "tmp_path" in locals() and Path(tmp_path).exists():
                    Path(tmp_path).unlink()
            except Exception:
                pass
            return False

    def _apply_edits_full(self, pil_image: Image.Image) -> Image.Image:
        f = {
            "exposure": self.exposure_factor,
            "brightness": self.brightness_factor,
            "contrast": self.contrast_factor,
            "highlights": self.highlights_factor,
            "shadows": self.shadows_factor,
            "blackpoint": self.blackpoint_factor,
            "saturation": self.saturation_factor,
            "vibrance": self.vibrance_factor,
            "warmth": self.warmth_factor,
            "tint": self.tint_factor,
            "sharpness": self.sharpness_factor,
            "noise_reduction": self.noise_reduction_factor,
            "vignette": self.vignette_factor,
            "brilliance": self.brilliance_factor,
        }
        job = _PreviewJob(0, pil_image, 0, f, pil_image.size, False, interactive=False)
        return job._apply_edits(pil_image, f)

    # ---------- Photos: live preview while dragging ----------

    def _on_slider_change(self, value: float, attr_name: str):
        setattr(self, attr_name, value)
        self._schedule_preview()

    def _schedule_preview(self):
        self._preview_timer.start()

    def _start_preview_job(self):
        if not self.working_image_pil:
            return
        do_mirror = self.viewer_stack.currentIndex() == 1
        if do_mirror:
            avail = self.dual_viewer.left_scroll.viewport().size()
        else:
            avail = self.single_scroll.viewport().size()
        cap_w = min(avail.width(), 1280)
        cap_h = min(avail.height(), 1280)
        target_size = (max(1, cap_w), max(1, cap_h))
        interactive = any(sl.isSliderDown() for sl in self._all_sliders)

        with self._preview_lock:
            self._preview_job_id += 1
            jid = self._preview_job_id

        f = {
            "exposure": self.exposure_factor,
            "brightness": self.brightness_factor,
            "contrast": self.contrast_factor,
            "highlights": self.highlights_factor,
            "shadows": self.shadows_factor,
            "blackpoint": self.blackpoint_factor,
            "saturation": self.saturation_factor,
            "vibrance": self.vibrance_factor,
            "warmth": self.warmth_factor,
            "tint": self.tint_factor,
            "sharpness": self.sharpness_factor,
            "noise_reduction": self.noise_reduction_factor,
            "vignette": self.vignette_factor,
            "brilliance": self.brilliance_factor,
        }
        job = _PreviewJob(
            job_id=jid,
            base_image=(self._preview_source or self.working_image_pil),
            rotation_degrees=self.rotation_degrees,
            factors=f,
            single_target_size=target_size,
            do_mirror=do_mirror,
            interactive=interactive,
        )
        job.signals.done.connect(self._on_preview_ready)
        self.threadpool.start(job, 1)

    def _on_preview_ready(self, job_id: int, main_img: QImage, mirror_img_obj: object):
        if job_id != self._preview_job_id:
            return
        pm = QPixmap.fromImage(main_img)
        mirror_pm = QPixmap.fromImage(mirror_img_obj) if isinstance(mirror_img_obj, QImage) else None

        if self.viewer_stack.currentIndex() == 0:
            self.single_viewer.set_pixmap(pm)
        else:
            self.dual_viewer.set_pixmaps(pm, mirror_pm)

    # ---------- Metadata writing ----------

    def _write_custom_note_to_file(self):
        if not self.original_image_path:
            QMessageBox.warning(self, "Metadata", "No image loaded.")
            return
        if _PIEXIF is None:
            QMessageBox.warning(self, "Metadata", "piexif not installed.")
            return
        note = self.meta_note_edit.text().strip()
        if not note:
            QMessageBox.warning(self, "Metadata", "Please enter a note.")
            return

        p = Path(self.original_image_path)
        fmt = (self._orig_format or self._infer_format_from_path(str(p)) or "").upper()
        if fmt not in ("JPEG", "TIFF"):
            QMessageBox.warning(self, "Metadata", "Custom note writing is supported only for JPEG/TIFF.")
            return

        try:
            if p.exists() and p.stat().st_size > 0:
                try:
                    exif_dict = _PIEXIF.load(str(p))
                except Exception:
                    exif_dict = {"0th": {}, "Exif": {}, "1st": {}, "thumbnail": None}
            else:
                exif_dict = {"0th": {}, "Exif": {}, "1st": {}, "thumbnail": None}

            exif_dict.setdefault("0th", {})
            exif_dict.setdefault("Exif", {})
            exif_dict["0th"][_PIEXIF.ImageIFD.ImageDescription] = note.encode("utf-8", "ignore")
            exif_dict["Exif"][_PIEXIF.ExifIFD.UserComment] = b"ASCII\0\0\0" + note.encode("ascii", "ignore")

            exif_bytes = _PIEXIF.dump(exif_dict)
            _PIEXIF.insert(exif_bytes, str(p))
            QMessageBox.information(self, "Metadata", "Custom note written to file metadata.")
        except Exception as e:
            QMessageBox.critical(self, "Metadata", f"Failed to write note: {e}")

    # ---------- Delete to trash ----------

    def _on_delete_key(self):
        # Decide by active tab and selection
        tab_text = self.tabs.tabText(self.tabs.currentIndex())
        if tab_text == "Photos":
            idx = self.photo_list.currentRow()
            if idx >= 0:
                self._delete_image_by_index(idx)
            else:
                QMessageBox.information(self, "Delete", "Select an image to delete.")
        elif tab_text == "Live (.mov)":
            idx = self.live_list.currentRow()
            if idx >= 0:
                self._delete_mov_by_index(idx)
            else:
                QMessageBox.information(self, "Delete", "Select a .mov to delete.")

    def _require_send2trash(self) -> bool:
        if _S2T is None:
            QMessageBox.critical(
                self,
                "Delete",
                "send2trash is not installed. Install it with:\n\npip install send2trash"
            )
            return False
        return True

    def _confirm_delete(self, path: Path) -> bool:
        reply = QMessageBox.question(
            self, "Move to Trash",
            f"Move to Trash?\n{path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        return reply == QMessageBox.StandardButton.Yes

    def _delete_image_by_index(self, idx: int):
        if not (0 <= idx < len(self.image_files)):
            return
        p = Path(self.image_files[idx])
        if not self._require_send2trash():
            return
        if not self._confirm_delete(p):
            return

        # If the currently loaded image is this file, clear preview state
        if self.original_image_path and Path(self.original_image_path) == p:
            self.original_image_path = None
            self.original_image_pil = None
            self.working_image_pil = None
            self._preview_source = None

        try:
            _S2T(str(p))
        except Exception as e:
            QMessageBox.critical(self, "Delete", f"Failed to move to Trash:\n{e}")
            return

        # Update list and selection
        self.image_files.pop(idx)
        self.photo_list.takeItem(idx)

        if self.image_files:
            new_idx = min(idx, len(self.image_files) - 1)
            self._load_image_by_index(new_idx)
            self.photo_list.setCurrentRow(new_idx)
        else:
            self._clear_photos_state()
            self.single_viewer.clear_pixmap("No Images Found")
            self.set_controls_state(False)

    def _delete_mov_by_index(self, idx: int):
        if not (0 <= idx < len(self.mov_files)):
            return
        p = Path(self.mov_files[idx])
        if not self._require_send2trash():
            return
        if not self._confirm_delete(p):
            return

        # If this video is currently open, stop and release
        if self.current_mov_index == idx:
            self._stop_video()
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.frames_list.clear()
            self.player_label.setText("No .mov loaded")
            self.player_label.setPixmap(QPixmap())

        try:
            _S2T(str(p))
        except Exception as e:
            QMessageBox.critical(self, "Delete", f"Failed to move to Trash:\n{e}")
            return

        # Update list and selection
        self.mov_files.pop(idx)
        self.live_list.takeItem(idx)

        if self.mov_files:
            new_idx = min(idx, len(self.mov_files) - 1)
            self.current_mov_index = -1
            self.live_list.setCurrentRow(new_idx)
            self._select_live_by_index(new_idx)
        else:
            self._clear_live_state()

    # ---------- Tab change ----------

    def _on_tab_changed(self, idx: int):
        if self.tabs.tabText(idx) != "Live (.mov)":
            self._stop_video()
        if self.tabs.tabText(idx) == "Photos":
            self._schedule_preview()

# ------------------------------- main -------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = ImageEditorApp()
    editor.show()
    sys.exit(app.exec())
