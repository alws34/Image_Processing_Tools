#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import math
import threading
from pathlib import Path
from typing import Optional, Any, Tuple, List, Dict
from datetime import datetime

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QPushButton, QLabel, QSlider, QTabWidget, QGridLayout,
    QDoubleSpinBox, QAbstractSpinBox, QSplitter, QSizePolicy, QScrollArea,
    QMessageBox, QFileDialog, QGroupBox, QLineEdit, QStackedWidget,
    QRadioButton, QButtonGroup, QFrame
)
import PyQt6.QtGui as QtGui
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QCursor, QAction,
    QPalette, QResizeEvent, QPen, QPaintEvent, QMouseEvent, QKeySequence
)
from PyQt6.QtCore import (
    Qt, QSize, QPoint, QRect, pyqtSignal, QDir, QTimer,
    QThreadPool, QRunnable, QObject, QEvent
)
from PyQt6.QtWidgets import QAbstractItemView, QAbstractScrollArea

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

FILL_KEEP = "keep"        # keep borders (letterbox)
FILL_AUTOCROP = "crop"    # auto-crop valid area
FILL_STRETCH = "stretch"  # stretch valid area to fill canvas (requested)
LEFT_PANEL_WIDTH = 320 # px
# ---------------------- Utility ----------------------

def _fmt_ts_local(ts: float) -> str:
    try:
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"

def _sanitize_exif_datetime(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        s2 = s.strip().replace("/", ":")
        if len(s2) >= 19 and s2[4] == ":" and s2[7] == ":":
            dt = datetime.strptime(s2[:19], "%Y:%m:%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        try:
            dt = datetime.strptime(s2[:19], "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return s
    except Exception:
        return s

def pil_to_qimage(pil_image: Image.Image) -> QImage:
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

# ---------------------- Geometry: perspective straighten ----------------------

def _compose_homography_from_euler(w: int, h: int, rx_deg: float, ry_deg: float, rz_deg: float):
    if _NP is None:
        return None
    cx, cy = w / 2.0, h / 2.0
    f = max(w, h)
    rx = _NP.deg2rad(rx_deg)
    ry = _NP.deg2rad(ry_deg)
    rz = _NP.deg2rad(rz_deg)

    Rx = _NP.array([[1, 0, 0],
                    [0, math.cos(rx), -math.sin(rx)],
                    [0, math.sin(rx),  math.cos(rx)]], dtype=_NP.float32)
    Ry = _NP.array([[ math.cos(ry), 0, math.sin(ry)],
                    [0,             1, 0],
                    [-math.sin(ry), 0, math.cos(ry)]], dtype=_NP.float32)
    Rz = _NP.array([[math.cos(rz), -math.sin(rz), 0],
                    [math.sin(rz),  math.cos(rz), 0],
                    [0,             0,            1]], dtype=_NP.float32)

    R = Rz @ (Ry @ Rx)

    K = _NP.array([[f, 0, cx],
                   [0, f, cy],
                   [0, 0, 1]], dtype=_NP.float32)
    Kinv = _NP.linalg.inv(K)
    H = K @ R @ Kinv
    return H.astype(_NP.float32)

def _auto_crop_bounds_from_mask(mask: "np.ndarray") -> Optional[Tuple[int,int,int,int]]:
    if _CV2 is None or _NP is None:
        return None
    nz = _CV2.findNonZero(mask)
    if nz is None:
        return None
    x, y, w, h = _CV2.boundingRect(nz)
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)

def _apply_geometry_perspective(
    pil_img: Image.Image,
    rx: float, ry: float, rz: float,
    coarse_roll_deg: float,
    fill_mode: str,
    preview_fast: bool = False,
) -> Image.Image:
    """
    preview_fast:
      - True: optimize for speed. Downscale first, warp on the smaller grid, then scale for display.
      - False: full-quality path used for final preview/save.
    """
    total_rz = rz + coarse_roll_deg

    # If OpenCV/numpy missing, fall back to rotate+resize
    if _CV2 is None or _NP is None:
        base = pil_img.rotate(total_rz, expand=True, resample=Image.Resampling.BICUBIC)
        base = base.resize(pil_img.size, Image.Resampling.LANCZOS)
        return base

    # Convert once
    base_rgb = pil_img.convert("RGB")
    img_np = _NP.array(base_rgb)
    h, w = img_np.shape[:2]

    # Fast path: downscale before warp to reduce compute while sliders move
    if preview_fast:
        # Choose a smaller working size that preserves aspect but keeps warp cheap
        target_long = 720  # was 480; 720 still smooth but sharper during drag
        long_side = max(w, h)
        if long_side > target_long:
            scale = target_long / float(long_side)
            fast_w, fast_h = max(1, int(w * scale)), max(1, int(h * scale))
            img_np_fast = _CV2.resize(img_np, (fast_w, fast_h), interpolation=_CV2.INTER_AREA)
            w_fast, h_fast = fast_w, fast_h
        else:
            img_np_fast = img_np
            w_fast, h_fast = w, h

        H = _compose_homography_from_euler(w_fast, h_fast, rx, ry, total_rz)
        if H is None:
            return Image.fromarray(img_np_fast)

        if fill_mode == FILL_KEEP:
            warped = _CV2.warpPerspective(
                img_np_fast, H, (w_fast, h_fast),
                flags=_CV2.INTER_LINEAR,
                borderMode=_CV2.BORDER_REPLICATE
            )
            return Image.fromarray(warped)

        ones = _NP.full((h_fast, w_fast), 255, dtype=_NP.uint8)
        mask = _CV2.warpPerspective(
            ones, H, (w_fast, h_fast),
            flags=_CV2.INTER_NEAREST,
            borderMode=_CV2.BORDER_CONSTANT,
            borderValue=0
        )
        warped = _CV2.warpPerspective(
            img_np_fast, H, (w_fast, h_fast),
            flags=_CV2.INTER_LINEAR,
            borderMode=_CV2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        bounds = _auto_crop_bounds_from_mask(mask)
        if not bounds:
            return Image.fromarray(warped)

        x1, y1, x2, y2 = bounds
        x1 = max(0, min(x1, w_fast - 1))
        y1 = max(0, min(y1, h_fast - 1))
        x2 = max(x1 + 1, min(x2, w_fast))
        y2 = max(y1 + 1, min(y2, h_fast))
        roi = warped[y1:y2, x1:x2, :]

        if fill_mode == FILL_AUTOCROP:
            return Image.fromarray(roi)
        if fill_mode == FILL_STRETCH:
            stretched = _CV2.resize(roi, (w_fast, h_fast), interpolation=_CV2.INTER_CUBIC)
            return Image.fromarray(stretched)

        return Image.fromarray(warped)

    # Full-quality path
    H = _compose_homography_from_euler(w, h, rx, ry, total_rz)
    if H is None:
        return base_rgb

    if fill_mode == FILL_KEEP:
        warped = _CV2.warpPerspective(
            img_np, H, (w, h),
            flags=_CV2.INTER_LINEAR,
            borderMode=_CV2.BORDER_REPLICATE
        )
        return Image.fromarray(warped)

    ones = _NP.full((h, w), 255, dtype=_NP.uint8)
    mask = _CV2.warpPerspective(
        ones, H, (w, h),
        flags=_CV2.INTER_NEAREST,
        borderMode=_CV2.BORDER_CONSTANT,
        borderValue=0
    )
    warped = _CV2.warpPerspective(
        img_np, H, (w, h),
        flags=_CV2.INTER_LINEAR,
        borderMode=_CV2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    bounds = _auto_crop_bounds_from_mask(mask)
    if not bounds:
        return Image.fromarray(warped)

    x1, y1, x2, y2 = bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    roi = warped[y1:y2, x1:x2, :]

    if fill_mode == FILL_AUTOCROP:
        return Image.fromarray(roi)

    if fill_mode == FILL_STRETCH:
        stretched = _CV2.resize(roi, (w, h), interpolation=_CV2.INTER_CUBIC)
        return Image.fromarray(stretched)

    return Image.fromarray(warped)

# ---------------------- Filters ----------------------

IPHONE_FILTERS = [
    "None",
    "Vivid", "Vivid Warm", "Vivid Cool",
    "Dramatic", "Dramatic Warm", "Dramatic Cool",
    "Mono", "Silvertone", "Noir",
]

CAMSCANNER_FILTERS = [
    "Original",
    "Magic Color", "Magic Pro",
    "Grayscale", "B&W",
    "No Shadow",
    "Soft Tone",
    "OCV Black",
]

ALL_FILTERS = IPHONE_FILTERS + ["—"] + CAMSCANNER_FILTERS

def _blend(a: Image.Image, b: Image.Image, alpha: float) -> Image.Image:
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0:
        return a
    if alpha >= 1:
        return b
    return Image.blend(a.convert("RGB"), b.convert("RGB"), alpha)

def _apply_filter_iphone(img: Image.Image, name: str) -> Image.Image:
    im = img.convert("RGB")

    def adj_contrast(x, pct):
        return ImageEnhance.Contrast(x).enhance(1.0 + pct)

    def adj_brightness(x, pct):
        return ImageEnhance.Brightness(x).enhance(1.0 + pct)

    def adj_color(x, pct):
        return ImageEnhance.Color(x).enhance(1.0 + pct)

    if name == "Vivid":
        im = adj_contrast(im, 0.15)
        im = adj_color(im, 0.25)
    elif name == "Vivid Warm":
        im = _apply_filter_iphone(im, "Vivid")
        if _NP is not None:
            npimg = _NP.array(im).astype(_NP.float32)
            npimg[:, :, 0] = _NP.clip(npimg[:, :, 0] * 1.06, 0, 255)
            npimg[:, :, 2] = _NP.clip(npimg[:, :, 2] * 0.94, 0, 255)
            im = Image.fromarray(npimg.astype(_NP.uint8))
    elif name == "Vivid Cool":
        im = _apply_filter_iphone(im, "Vivid")
        if _NP is not None:
            npimg = _NP.array(im).astype(_NP.float32)
            npimg[:, :, 2] = _NP.clip(npimg[:, :, 2] * 1.06, 0, 255)
            npimg[:, :, 0] = _NP.clip(npimg[:, :, 0] * 0.94, 0, 255)
            im = Image.fromarray(npimg.astype(_NP.uint8))
    elif name == "Dramatic":
        im = ImageEnhance.Contrast(im).enhance(1.2)
        if _NP is not None:
            npimg = _NP.array(im).astype(_NP.float32) / 255.0
            npimg = _NP.where(npimg > 0.7, npimg**1.25, npimg)
            npimg = _NP.where(npimg < 0.3, _NP.sqrt(npimg), npimg)
            im = Image.fromarray((_NP.clip(npimg, 0, 1) * 255).astype(_NP.uint8))
    elif name == "Dramatic Warm":
        im = _apply_filter_iphone(im, "Dramatic")
        if _NP is not None:
            npimg = _NP.array(im).astype(_NP.float32)
            npimg[:, :, 0] = _NP.clip(npimg[:, :, 0] * 1.06, 0, 255)
            im = Image.fromarray(npimg.astype(_NP.uint8))
    elif name == "Dramatic Cool":
        im = _apply_filter_iphone(im, "Dramatic")
        if _NP is not None:
            npimg = _NP.array(im).astype(_NP.float32)
            npimg[:, :, 2] = _NP.clip(npimg[:, :, 2] * 1.06, 0, 255)
            im = Image.fromarray(npimg.astype(_NP.uint8))
    elif name == "Mono":
        im = im.convert("L").convert("RGB")
    elif name == "Silvertone":
        im = ImageEnhance.Color(im).enhance(0.0)
        im = ImageEnhance.Contrast(im).enhance(1.15)
        im = ImageEnhance.Brightness(im).enhance(0.95)
    elif name == "Noir":
        im = ImageEnhance.Color(im).enhance(0.0)
        im = ImageEnhance.Contrast(im).enhance(1.35)
    return im

def _apply_filter_camscanner(img: Image.Image, name: str) -> Image.Image:
    im = img.convert("RGB")
    if name == "Original":
        return im
    if _CV2 is None or _NP is None:
        if name == "Grayscale":
            return im.convert("L").convert("RGB")
        if name == "B&W":
            g = im.convert("L")
            return g.point(lambda v: 255 if v > 140 else 0).convert("RGB")
        if name == "Soft Tone":
            return ImageEnhance.Sharpness(im).enhance(0.9)
        if name in ("Magic Color", "Magic Pro", "No Shadow", "OCV Black"):
            return ImageEnhance.Contrast(im).enhance(1.1)
        return im

    npimg = _NP.array(im)
    if name in ("Magic Color", "Magic Pro"):
        b, g, r = _CV2.split(_CV2.cvtColor(npimg, _CV2.COLOR_RGB2BGR))
        avg_b, avg_g, avg_r = b.mean(), g.mean(), r.mean()
        kb = (avg_g + avg_r) / (2 * avg_b + 1e-6)
        kg = (avg_b + avg_r) / (2 * avg_g + 1e-6)
        kr = (avg_b + avg_g) / (2 * avg_r + 1e-6)
        b = _NP.clip(b * kb, 0, 255).astype(_NP.uint8)
        g = _NP.clip(g * kg, 0, 255).astype(_NP.uint8)
        r = _NP.clip(r * kr, 0, 255).astype(_NP.uint8)
        wb = _CV2.cvtColor(_CV2.merge([b, g, r]), _CV2.COLOR_BGR2RGB)

        lab = _CV2.cvtColor(wb, _CV2.COLOR_RGB2LAB)
        L, A, B = _CV2.split(lab)
        clip = 3.0 if name == "Magic Color" else 4.0
        clahe = _CV2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        L2 = clahe.apply(L)
        lab2 = _CV2.merge((L2, A, B))
        rgb = _CV2.cvtColor(lab2, _CV2.COLOR_LAB2RGB)

        blur = _CV2.GaussianBlur(rgb, (0, 0), 1.0 if name == "Magic Color" else 1.4)
        sharpen = _CV2.addWeighted(rgb, 1.25, blur, -0.25, 0)
        return Image.fromarray(sharpen)

    if name == "Grayscale":
        g = _CV2.cvtColor(npimg, _CV2.COLOR_RGB2GRAY)
        return Image.fromarray(_CV2.cvtColor(g, _CV2.COLOR_GRAY2RGB))

    if name == "B&W":
        g = _CV2.cvtColor(npimg, _CV2.COLOR_RGB2GRAY)
        bw = _CV2.adaptiveThreshold(g, 255, _CV2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    _CV2.THRESH_BINARY, 31, 10)
        return Image.fromarray(_CV2.cvtColor(bw, _CV2.COLOR_GRAY2RGB))

    if name == "No Shadow":
        g = _CV2.cvtColor(npimg, _CV2.COLOR_RGB2GRAY)
        bg = _CV2.medianBlur(g, 31)
        bg = _NP.where(bg < 1, 1, bg)
        norm = (g.astype(_NP.float32) / bg.astype(_NP.float32)) * 128.0
        norm = _NP.clip(norm, 0, 255).astype(_NP.uint8)
        norm = _CV2.equalizeHist(norm)
        return Image.fromarray(_CV2.cvtColor(norm, _CV2.COLOR_GRAY2RGB))

    if name == "Soft Tone":
        blur = _CV2.GaussianBlur(npimg, (0, 0), 3.0)
        soft = _CV2.addWeighted(npimg, 0.75, blur, 0.25, 0)
        return Image.fromarray(soft)

    if name == "OCV Black":
        g = _CV2.cvtColor(npimg, _CV2.COLOR_RGB2GRAY)
        _, th = _CV2.threshold(g, 0, 255, _CV2.THRESH_BINARY + _CV2.THRESH_OTSU)
        if th.mean() < 127:
            th = _CV2.bitwise_not(th)
        return Image.fromarray(_CV2.cvtColor(th, _CV2.COLOR_GRAY2RGB))

    return im

def _apply_filter_pipeline(img: Image.Image, preset: str, strength_0_to_1: float) -> Image.Image:
    if not preset or preset in ("None", "—", "Original"):
        return img
    if preset in IPHONE_FILTERS:
        out = _apply_filter_iphone(img, preset)
        return _blend(img, out, strength_0_to_1)
    if preset in CAMSCANNER_FILTERS:
        out = _apply_filter_camscanner(img, preset)
        return _blend(img, out, strength_0_to_1)
    return img

# ---------------------- Viewer Widgets ----------------------

class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundRole(QPalette.ColorRole.Dark)
        self.setAutoFillBackground(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setScaledContents(False)

        # Explicit init for attributes used elsewhere
        self.editor_mode = MODE_VIEW
        self.crop_start_point: Optional[QPoint] = None
        self.crop_end_point: Optional[QPoint] = None
        self.current_qpixmap: Optional[QPixmap] = None
        self.editor_ref: Optional[Any] = None

        # Freeform crop state
        self.crop_mode_type: str = "rect"
        self.freeform_pts: List[QPoint] = []
        self._drag_idx: Optional[int] = None
        self._handle_px: int = 6  # handle half-size in px
        self._handle_radius: int = 8  # circle handle radius (px)

    def set_crop_mode_type(self, mode: str) -> None:
        if mode not in ("rect", "free"):
            return
        self.crop_mode_type = mode
        # If already in crop mode, reinitialize default handles so the user sees feedback immediately
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
        tl = QPoint(r.left(),  r.top())
        tr = QPoint(r.right(), r.top())
        br = QPoint(r.right(), r.bottom())
        bl = QPoint(r.left(),  r.bottom())
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
                # centered 50x50 rubberband (clamped to image rect)
                side = min(50, img_rect.width(), img_rect.height())
                cx = img_rect.center().x()
                cy = img_rect.center().y()
                tl = QPoint(cx - side // 2, cy - side // 2)
                br = QPoint(cx + side // 2, cy + side // 2)
                # clamp into image rect
                tl.setX(max(img_rect.left(),  min(tl.x(), img_rect.right())))
                tl.setY(max(img_rect.top(),   min(tl.y(), img_rect.bottom())))
                br.setX(max(img_rect.left(),  min(br.x(), img_rect.right())))
                br.setY(max(img_rect.top(),   min(br.y(), img_rect.bottom())))
                self.crop_start_point = tl
                self.crop_end_point = br
                # keep freeform_pts in sync for unified hit-testing
                tr = QPoint(br.x(), tl.y()); bl = QPoint(tl.x(), br.y())
                self.freeform_pts = [tl, tr, br, bl]
                if self.editor_ref:
                    self.editor_ref.apply_crop_button.setEnabled(True)
            elif self.crop_mode_type == "free" and self.current_qpixmap:
                inset = max(10, min(img_rect.width(), img_rect.height()) // 10)
                tl = QPoint(img_rect.left() + inset,  img_rect.top() + inset)
                tr = QPoint(img_rect.right() - inset, img_rect.top() + inset)
                br = QPoint(img_rect.right() - inset, img_rect.bottom() - inset)
                bl = QPoint(img_rect.left() + inset,  img_rect.bottom() - inset)
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

        from PyQt6 import QtCore
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
            # clamp to image area
            r = r & img_rect
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
            painter.drawRect(r)
            # draw circular handles at corners
            painter.setBrush(QColor(255, 0, 0))
            rr = self._handle_radius
            for pt in [QPoint(r.left(), r.top()),
                       QPoint(r.right(), r.top()),
                       QPoint(r.right(), r.bottom()),
                       QPoint(r.left(), r.bottom())]:
                painter.drawEllipse(QtCore.QRectF(pt.x() - rr, pt.y() - rr, 2*rr, 2*rr))
            return

        # Freeform poly overlay + circular corners
        if len(self.freeform_pts) == 4:
            painter.save()
            painter.setBrush(QColor(0, 0, 0, 90))
            painter.setPen(Qt.PenStyle.NoPen)
            outer = QtGui.QPainterPath()
            outer.addRect(QtCore.QRectF(self.rect()))
            polyf = QtGui.QPolygonF([QtCore.QPointF(p) for p in self.freeform_pts])
            inner = QtGui.QPainterPath()
            inner.addPolygon(polyf); inner.closeSubpath()
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
                painter.drawEllipse(QtCore.QRectF(pt.x() - rr, pt.y() - rr, 2*rr, 2*rr))

    def _clamp_to_img_rect(self, p: QPoint) -> QPoint:
        r = self.get_image_display_rect()
        x = max(r.left(), min(p.x(), r.right()))
        y = max(r.top(),  min(p.y(), r.bottom()))
        return QPoint(int(x), int(y))

    def _hit_handle(self, pos: QPoint) -> Optional[int]:
        pts: List[QPoint]
        if self.crop_mode_type == "rect":
            r = self._current_rect()
            if not r:
                return None
            pts = [QPoint(r.left(), r.top()),
                   QPoint(r.right(), r.top()),
                   QPoint(r.right(), r.bottom()),
                   QPoint(r.left(), r.bottom())]
        else:
            pts = list(self.freeform_pts)

        radius = self._handle_radius * 1.6
        for i, pt in enumerate(pts):
            if abs(pos.x() - pt.x()) <= radius and abs(pos.y() - pt.y()) <= radius:
                return i
        return None


    def mousePressEvent(self, event: QMouseEvent):
        img_rect = self.get_image_display_rect()
        if self.editor_mode != MODE_CROP or not self.current_qpixmap or not img_rect.contains(event.position().toPoint()):
            return

        p = self._clamp_to_img_rect(event.position().toPoint())
        if self.crop_mode_type == "rect":
            hit = self._hit_handle(p)
            if hit is not None:
                self._drag_idx = hit
            else:
                # no handle hit; do nothing (or add move-whole-rect if desired)
                pass
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
            # indices: 0=tl,1=tr,2=br,3=bl
            tl = QPoint(r.left(),  r.top())
            tr = QPoint(r.right(), r.top())
            br = QPoint(r.right(), r.bottom())
            bl = QPoint(r.left(),  r.bottom())
            if self._drag_idx == 0:  # tl
                tl = p
            if self._drag_idx == 1:  # tr
                tr = QPoint(p.x(), p.y())
            elif self._drag_idx == 2:  # br
                br = p
            elif self._drag_idx == 3:  # bl
                bl = QPoint(p.x(), p.y())
            # rebuild normalized rect from opposite corners
            new_left   = min(tl.x(), bl.x())
            new_right  = max(tr.x(), br.x())
            new_top    = min(tl.y(), tr.y())
            new_bottom = max(bl.y(), br.y())
            self.crop_start_point = QPoint(new_left, new_top)
            self.crop_end_point   = QPoint(new_right, new_bottom)
            self._sync_pts_from_rect()
            self.update()
            return

        # freeform
        if self._drag_idx is not None and 0 <= self._drag_idx < len(self.freeform_pts):
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
        # freeform
        if self.editor_ref and len(self.freeform_pts) == 4:
            xs = [p.x() for p in self.freeform_pts]
            ys = [p.y() for p in self.freeform_pts]
            area_ok = (max(xs) - min(xs) >= 10) and (max(ys) - min(ys) >= 10)
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
        self.slider.setTracking(True)  # live updates; we debounce at ~30Hz
        self.slider.setRange(int(min_val / step), int(max_val / step))
        self.slider.setValue(int(default_val / step))
        self.slider.setTickInterval(max(1, int((max_val - min_val) / step / 10)))
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

# ---------------------- Async folder scan ----------------------

class DirScanSignals(QObject):
    started = pyqtSignal(int, str)
    found_image = pyqtSignal(int, str, str)  # job_id, path, taken
    found_mov = pyqtSignal(int, str, str)    # job_id, path, taken
    finished = pyqtSignal(int, list, list)   # job_id, images, movs
    error = pyqtSignal(int, str)

class DirScanJob(QRunnable):
    def __init__(self, job_id: int, folder: Path, owner):
        super().__init__()
        self.job_id = job_id
        self.folder = folder
        self.signals = DirScanSignals()
        self.owner = owner  # to reuse helpers safely (EXIF read)

    def run(self):
        try:
            self.signals.started.emit(self.job_id, str(self.folder))
            if not self.folder.is_dir():
                self.signals.error.emit(self.job_id, f"Not a directory: {self.folder}")
                return

            images: List[str] = []
            movs: List[str] = []

            for p in sorted(self.folder.iterdir(), key=lambda x: str(x).lower()):
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTS:
                    taken = self.owner._image_date_taken(p)
                    images.append(str(p))
                    self.signals.found_image.emit(self.job_id, str(p), taken or "-")
                elif ext in SUPPORTED_LIVE_EXTS:
                    taken = self.owner._mov_date_taken(p)
                    movs.append(str(p))
                    self.signals.found_mov.emit(self.job_id, str(p), taken or "-")

            self.signals.finished.emit(self.job_id, images, movs)
        except Exception as e:
            self.signals.error.emit(self.job_id, f"{e}")

# ---------------------- Preview Worker ----------------------

class _PreviewSignals(QObject):
    done = pyqtSignal(int, QImage, object)

class _PreviewJob(QRunnable):
    def __init__(
        self,
        job_id: int,
        base_image: Image.Image,
        coarse_rotation_degrees: int,
        factors: dict,
        single_target_size: Tuple[int, int],
        do_mirror: bool,
        interactive: bool,
        geom_rx: float,
        geom_ry: float,
        geom_rz: float,
        preset_name: str,
        preset_strength: float,
        fill_mode: str,
        fast_geometry_preview: bool = False
    ):
        super().__init__()
        self.job_id = job_id
        self.image = base_image
        self.coarse_rotation_degrees = coarse_rotation_degrees
        self.factors = factors
        self.target_size = single_target_size
        self.do_mirror = do_mirror
        self.interactive = interactive
        self.geom_rx = geom_rx
        self.geom_ry = geom_ry
        self.geom_rz = geom_rz
        self.preset_name = preset_name
        self.preset_strength = preset_strength
        self.fill_mode = fill_mode
        self.signals = _PreviewSignals()
        self.fast_geometry_preview = fast_geometry_preview

    def run(self):
        img = self.image

        # 1) geometry first
        geo_img = _apply_geometry_perspective(
            img,
            self.geom_rx, self.geom_ry, self.geom_rz,
            self.coarse_rotation_degrees,
            self.fill_mode,
            preview_fast=self.fast_geometry_preview,
        )
        # fast geometry path while sliders move
        if self.fast_geometry_preview:
            target_w, target_h = self.target_size
            target_w = min(target_w, 480)
            target_h = min(target_h, 480)
            if target_w <= 0 or target_h <= 0:
                target_w, target_h = max(1, geo_img.size[0]), max(1, geo_img.size[1])
            gw, gh = geo_img.size
            ratio = min(target_w / float(gw), target_h / float(gh))
            new_size = (max(1, int(gw * ratio)), max(1, int(gh * ratio)))
            preview = geo_img.resize(new_size, Image.Resampling.BILINEAR)

            main_qimage = pil_to_qimage(preview)
            mirror_qimage = None
            if self.do_mirror:
                try:
                    flip_const = Image.Transpose.FLIP_LEFT_RIGHT
                except AttributeError:
                    flip_const = Image.FLIP_LEFT_RIGHT
                mirror_preview = preview.transpose(flip_const)
                mirror_qimage = pil_to_qimage(mirror_preview)

            self.signals.done.emit(self.job_id, main_qimage, mirror_qimage)
            return

        # 2) normal preview
        target_w, target_h = self.target_size
        if self.interactive:
            target_w = min(target_w, 720)
            target_h = min(target_h, 720)
        if target_w <= 0 or target_h <= 0:
            target_w, target_h = max(1, geo_img.size[0]), max(1, geo_img.size[1])

        gw, gh = geo_img.size
        ratio = min(target_w / float(gw), target_h / float(gh))
        new_size = (max(1, int(gw * ratio)), max(1, int(gh * ratio)))
        preview = geo_img.resize(new_size, Image.Resampling.BILINEAR)

        # 3) filters before manual sliders
        if self.preset_name and self.preset_name not in ("None", "—", "Original"):
            preview = _apply_filter_pipeline(preview, self.preset_name, max(0.0, min(1.0, self.preset_strength)))

        # 4) manual edits
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
        img = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
        working_img = img.copy()

        exposure_adj = 1.0 + (f["exposure"] / 2.0)
        working_img = ImageEnhance.Brightness(working_img).enhance(exposure_adj)

        brightness_adj = 1.0 + (f["brightness"] / 100.0)
        working_img = ImageEnhance.Brightness(working_img).enhance(brightness_adj)

        contrast_adj = 1.0 + (f["contrast"] / 100.0)
        working_img = ImageEnhance.Contrast(working_img).enhance(contrast_adj)

        if _NP is not None:
            img_np = _NP.array(working_img).astype(_NP.float32) / 255.0
            if f["highlights"] != 0:
                gamma_h = 1.0 - (f["highlights"] / 200.0)
                img_np = _NP.where(img_np > 0.5, _NP.power(img_np, gamma_h), img_np)
            if f["shadows"] != 0:
                gamma_s = 1.0 + (f["shadows"] / 200.0)
                img_np = _NP.where(img_np < 0.5, _NP.power(img_np, gamma_s), img_np)
            if f["blackpoint"] != 0:
                offset = f["blackpoint"] / 500.0
                img_np = _NP.clip(img_np + offset, 0.0, 1.0)
            working_img = Image.fromarray((_NP.clip(img_np, 0.0, 1.0) * 255).astype(_NP.uint8))

        if _CV2 is not None and _NP is not None:
            brill = float(f["brilliance"])
            if abs(brill) > 0.01:
                img_rgb = _NP.array(working_img)
                lab = _CV2.cvtColor(img_rgb, _CV2.COLOR_RGB2LAB)
                L, a, b = _CV2.split(lab)
                if brill >= 0:
                    tiles = 4 if self.interactive else 8
                    clip = 1.0 + (brill / 100.0) * (4.0 if self.interactive else 8.0)
                    clip = max(1.01, min(clip, 9.0))
                    clahe = _CV2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
                    L2 = clahe.apply(L)
                else:
                    k = int(round(3 + (abs(brill) / 100.0) * (6 if self.interactive else 12)))
                    if k % 2 == 0:
                        k += 1
                    L_blur = _CV2.GaussianBlur(L, (k, k), 0)
                    alpha = (0.4 if self.interactive else 0.8) * (abs(brill) / 100.0)
                    alpha = max(0.0, min(alpha, 0.9))
                    L2 = _CV2.addWeighted(L, 1.0 - alpha, L_blur, alpha, 0)
                lab2 = _CV2.merge((L2, a, b))
                rgb2 = _CV2.cvtColor(lab2, _CV2.COLOR_LAB2RGB)
                working_img = Image.fromarray(rgb2)

        sat_adj = 1.0 + (f["saturation"] / 100.0)
        working_img = ImageEnhance.Color(working_img).enhance(sat_adj)

        if _NP is not None:
            img_rgb_np = _NP.array(working_img).astype(_NP.float32)
            if f["vibrance"] != 0:
                hsv = _NP.array(working_img.convert("HSV")).astype(_NP.float32)
                v_adj = f["vibrance"] / 100.0
                saturation = hsv[:, :, 1] / 255.0
                mask = 1.0 - saturation
                hsv[:, :, 1] = _NP.clip(hsv[:, :, 1] + (hsv[:, :, 1] * v_adj * mask), 0, 255)
                working_img = Image.fromarray(hsv.astype(_NP.uint8), "HSV").convert("RGB")
                img_rgb_np = _NP.array(working_img).astype(_NP.float32)
            if f["warmth"] != 0:
                r_gain = 1.0 + (f["warmth"] / 150.0)
                b_gain = 1.0 - (f["warmth"] / 150.0)
                img_rgb_np[:, :, 0] = _NP.clip(img_rgb_np[:, :, 0] * r_gain, 0, 255)
                img_rgb_np[:, :, 2] = _NP.clip(img_rgb_np[:, :, 2] * b_gain, 0, 255)
            if f["tint"] != 0:
                g_gain = 1.0 + (f["tint"] / 150.0)
                img_rgb_np[:, :, 1] = _NP.clip(img_rgb_np[:, :, 1] * g_gain, 0, 255)
            working_img = Image.fromarray(img_rgb_np.astype(_NP.uint8))

        if _CV2 is not None and _NP is not None and f["noise_reduction"] > 0.0:
            nr = float(f["noise_reduction"])
            img_bgr = _CV2.cvtColor(_NP.array(working_img), _CV2.COLOR_RGB2BGR)
            if self.interactive:
                d = 5
                sigma = 5 + int(nr * 5)
                den = _CV2.bilateralFilter(img_bgr, d, sigma, sigma)
            else:
                h = 3 + int(nr * 7)
                den = _CV2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)
            working_img = Image.fromarray(_CV2.cvtColor(den, _CV2.COLOR_BGR2RGB))

        sharp_adj = 1.0 + ((f["sharpness"] / 100.0) * (0.5 if self.interactive else 1.0))
        working_img = ImageEnhance.Sharpness(working_img).enhance(sharp_adj)

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
        self.setGeometry(100, 100, 1300, 870)
        self.setAcceptDrops(True)

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

        # edits state (manual sliders)
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

        # geometry straighten (pitch/yaw/roll) + fill mode
        self.geom_rx_deg = 0.0
        self.geom_ry_deg = 0.0
        self.geom_rz_deg = 0.0
        self.fill_mode = FILL_STRETCH
        self._warned_perspective = False
        self._last_changed_name: str = ""

        # filter preset
        self.filter_name = "None"
        self.filter_strength = 1.0  # 0..1

        # live tab state
        self.mov_files: List[str] = []
               # index of selected mov in live tab
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
        self._all_sliders: List[QSlider] = []
        self._preview_source = None  # type: Optional[Image.Image]

        # thread pools and timers
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(2)  # 1 for preview, 1 for scan
        self._preview_job_id = 0
        self._scan_job_id = 0

        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(20) 
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._start_preview_job)
        self._preview_lock = threading.Lock()

        self._init_ui()
        self.set_controls_state(False)

        # Delete shortcut
        self.delete_action = QAction("Delete", self)
        self.delete_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        self.delete_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.delete_action.triggered.connect(self._on_delete_key)
        self.addAction(self.delete_action)

        if _CV2 is None or _NP is None:
            QMessageBox.warning(
                self,
                "Optional Dependencies",
                "Perspective straighten (X/Y) and some filters require opencv-python and numpy.\n\n"
                "pip install opencv-python numpy"
            )

    def _on_geom_auto_crop(self):
        if not self.working_image_pil:
            QMessageBox.warning(self, "Geometry", "No image loaded.")
            return
        # Apply geometry with auto-crop and bake result into working image
        try:
            baked = _apply_geometry_perspective(
                self.working_image_pil,
                self.geom_rx_deg, self.geom_ry_deg, self.geom_rz_deg,
                self.rotation_degrees,
                FILL_AUTOCROP,
                preview_fast=False,
            )
            self.working_image_pil = baked
            self._preview_source = self._make_preview_source(self.working_image_pil)
            # Reset geometry after commit
            self.rotation_degrees = 0
            self.geom_rx_deg = self.geom_ry_deg = self.geom_rz_deg = 0.0
            for n in ("geom_rx", "geom_ry", "geom_rz"):
                s = getattr(self, f"{n}_slider", None)
                if s:
                    s.set_value(0.0)
            self._schedule_preview()
        except Exception as e:
            QMessageBox.critical(self, "Geometry", f"Auto crop failed: {e}")

    def _on_geom_commit(self, mode: str):
        if not self.working_image_pil:
            QMessageBox.warning(self, "Geometry", "No image loaded.")
            return
        try:
            baked = _apply_geometry_perspective(
                self.working_image_pil,
                self.geom_rx_deg, self.geom_ry_deg, self.geom_rz_deg,
                self.rotation_degrees,
                mode,
                preview_fast=False,
            )
            self.working_image_pil = baked
            self._preview_source = self._make_preview_source(self.working_image_pil)
            # Reset geometry after commit
            self.rotation_degrees = 0
            self.geom_rx_deg = self.geom_ry_deg = self.geom_rz_deg = 0.0
            for n in ("geom_rx", "geom_ry", "geom_rz"):
                s = getattr(self, f"{n}_slider", None)
                if s:
                    s.set_value(0.0)
            self._schedule_preview()
        except Exception as e:
            QMessageBox.critical(self, "Geometry", f"Commit failed: {e}")

    # ---------- UI ----------

    # def _on_photos_splitter_moved(self, pos: int, index: int):
    #     # Prevent re-entrancy jitter
    #     if getattr(self, "_in_splitter_handler", False):
    #         return
    #     self._in_splitter_handler = True
        
    #     # Define a minimum size for the left panel (e.g., 20 pixels)
    #     # This prevents the auto-collapse to 0.
    #     MIN_LEFT_SIZE = 20 
        
    #     try:
    #         sizes = self._photos_splitter.sizes()
    #         if not sizes or len(sizes) < 2:
    #             return
                
    #         left, right = sizes[0], sizes[1]
    #         total = max(1, left + right)

    #         # --- REPLACING THE ORIGINAL COLLAPSE LOGIC ---
    #         # The original code: 
    #         # if left <= self._left_auto_collapse_px and left != 0:
    #         #     self._photos_splitter.setSizes([0, total]) 
    #         #     return
            
    #         # 1. Prevent collapse to 0 by forcing a minimum size
    #         if left < MIN_LEFT_SIZE and left != 0:
    #             left = MIN_LEFT_SIZE
    #             right = max(1, total - left)
    #             self._photos_splitter.setSizes([left, right])
    #             return 
                
    #         # 2. Original RESTORE logic (only runs if left == 0)
    #         # This only needs to run if something else is still collapsing it, 
    #         # or if the user is dragging from an initial collapsed state.
    #         if left == 0 and pos > self._left_auto_collapse_px:
    #             restore = self._last_split_sizes if hasattr(self, "_last_split_sizes") else [280, max(1, total - 280)]
    #             # Ensure it sums to total
    #             scale = total / max(1, sum(restore))
    #             restore_scaled = [max(1, int(r * scale)) for r in restore]
    #             # Fix rounding drift
    #             drift = total - sum(restore_scaled)
    #             restore_scaled[0] += drift
    #             self._photos_splitter.setSizes(restore_scaled)
    #             return

    #     finally:
    #         self._in_splitter_handler = False
    

    def _on_crop_mode_changed(self, mode: str):
        self.crop_mode_type = mode
        self.single_viewer.set_crop_mode_type(mode)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Top quick-load row
        quick_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Paste folder path here or drop a folder anywhere in the window")
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self._open_path_from_edit)
        self.photo_load_btn = QPushButton("Browse...")
        self.photo_load_btn.clicked.connect(self._browse_and_load_folder)
        quick_row.addWidget(self.path_edit, 1)
        quick_row.addWidget(self.open_btn)
        quick_row.addWidget(self.photo_load_btn)
        main_layout.addLayout(quick_row)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Photos tab
        self.photos_tab = QWidget()
        self.tabs.addTab(self.photos_tab, "Photos")
        self._build_photos_tab(self.photos_tab)

        # Live tab
        self.live_tab = self._build_live_tab()
        self.tabs.addTab(self.live_tab, "Live (.mov)")

        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _build_photos_tab(self, tab: QWidget):
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split)

        self._in_splitter_handler = False
        self._last_split_sizes = [280, 1020]       # remembered non-collapsed sizes

        # left: sidebar
        left = QWidget()
        lv = QVBoxLayout(left)

        self.scan_status_lbl = QLabel("No folder loaded")
        self.scan_status_lbl.setStyleSheet("color:#999;")
        lv.addWidget(self.scan_status_lbl)

        self.photo_list = QListWidget()
        self.photo_list.currentItemChanged.connect(self._on_photo_select)
        left.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        left.setFixedWidth(LEFT_PANEL_WIDTH)  # <- locks the sidebar width

        self.photo_list.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.photo_list.setMinimumWidth(LEFT_PANEL_WIDTH)
        self.photo_list.setMaximumWidth(LEFT_PANEL_WIDTH)  # <- list never widens

        # Do not grow/shrink with content; always scroll instead
        self.photo_list.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        self.photo_list.setWordWrap(False)
        self.photo_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.photo_list.setUniformItemSizes(True)

        # Always show scrollbars; allow horizontal scrolling for long lines
        self.photo_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.photo_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.photo_list.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        lv.addWidget(self.photo_list)
        split.addWidget(left)

        # right: viewer + editor
        right = QWidget()
        rv = QVBoxLayout(right)

        viewer_editor_split = QSplitter(Qt.Orientation.Vertical)
        rv.addWidget(viewer_editor_split)

        # viewer stack
        self.viewer_stack = QStackedWidget()
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

        self.dual_viewer = DualImageViewer()
        self.viewer_stack.addWidget(self.dual_viewer)

        viewer_editor_split.addWidget(self.viewer_stack)

        # editor controls tabs
        self.editor_widget = QTabWidget()
        viewer_editor_split.addWidget(self.editor_widget)
        viewer_editor_split.setSizes([700, 340])

        # Transform / Save
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

        # Tone / Light
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

        # Color / Effects
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

        # Geometry / Straighten
        tg = QWidget()
        tgl = QVBoxLayout(tg)
        tgl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.geom_rx_slider = self._create_slider(tgl, "Pitch (X) deg", -30.0, 30.0, 0.1, "geom_rx_deg")
        self.geom_ry_slider = self._create_slider(tgl, "Yaw (Y) deg", -30.0, 30.0, 0.1, "geom_ry_deg")
        self.geom_rz_slider = self._create_slider(tgl, "Roll (Z) deg", -45.0, 45.0, 0.1, "geom_rz_deg")
        self.editor_widget.addTab(tg, "Geometry / Straighten")
        self._geom_layout = tgl
        
        # Geometry fill mode
        fill_group = QGroupBox("Geometry Fill Mode")
        fgl = QHBoxLayout(fill_group)
        self.fill_keep_rb = QRadioButton("Keep borders")
        self.fill_crop_rb = QRadioButton("Auto crop")
        self.fill_stretch_rb = QRadioButton("Stretch to fill")
        self.fill_group = QButtonGroup(self)
        self.fill_group.addButton(self.fill_keep_rb)
        self.fill_group.addButton(self.fill_crop_rb)
        self.fill_group.addButton(self.fill_stretch_rb)
        self.fill_stretch_rb.setChecked(True)
        fgl.addWidget(self.fill_keep_rb)
        fgl.addWidget(self.fill_crop_rb)
        fgl.addWidget(self.fill_stretch_rb)
        self.fill_keep_rb.toggled.connect(lambda _: self._on_fill_mode_changed())
        self.fill_crop_rb.toggled.connect(lambda _: self._on_fill_mode_changed())
        self.fill_stretch_rb.toggled.connect(lambda _: self._on_fill_mode_changed())

        # add to Geometry tab layout
        self._geom_layout.addWidget(fill_group)


        # Geometry Crop
        geom_crop_group = QGroupBox("Geometry Crop")
        gcl = QHBoxLayout(geom_crop_group)

        self.geom_auto_crop_btn = QPushButton("Auto crop (valid area)")
        self.geom_auto_crop_btn.setToolTip("Apply perspective and crop to the valid area (mask bounding box).")
        self.geom_auto_crop_btn.clicked.connect(self._on_geom_auto_crop)

        self.geom_commit_keep_btn = QPushButton("Commit geometry (keep borders)")
        self.geom_commit_keep_btn.setToolTip("Bake current pitch/yaw/roll into the image, keeping replicated borders.")
        self.geom_commit_keep_btn.clicked.connect(lambda: self._on_geom_commit(FILL_KEEP))

        self.geom_commit_stretch_btn = QPushButton("Commit geometry (stretch to fill)")
        self.geom_commit_stretch_btn.setToolTip("Bake current pitch/yaw/roll and stretch the valid area back to the canvas.")
        self.geom_commit_stretch_btn.clicked.connect(lambda: self._on_geom_commit(FILL_STRETCH))

        gcl.addWidget(self.geom_auto_crop_btn)
        gcl.addWidget(self.geom_commit_keep_btn)
        gcl.addWidget(self.geom_commit_stretch_btn)
        self._geom_layout.addWidget(geom_crop_group)


        # Crop mode radios
        self.crop_rect_rb = QRadioButton("Rect")
        self.crop_free_rb = QRadioButton("Freeform (4 corners)")
        self.crop_mode_group = QButtonGroup(self)
        self.crop_mode_group.addButton(self.crop_rect_rb)
        self.crop_mode_group.addButton(self.crop_free_rb)
        self.crop_rect_rb.setChecked(True)
        self.crop_rect_rb.toggled.connect(lambda checked: self._on_crop_mode_changed("rect" if checked else "free"))
        cgl.addWidget(self.crop_rect_rb)
        cgl.addWidget(self.crop_free_rb)

        # default
        self.crop_mode_type = "rect"

        # Filters tab (radio group)
        tf = QWidget()
        tfl = QVBoxLayout(tf)
        tfl.setAlignment(Qt.AlignmentFlag.AlignTop)

        rb_container = QWidget()
        rb_layout = QGridLayout(rb_container)
        rb_layout.setContentsMargins(0, 0, 0, 0)
        rb_layout.setHorizontalSpacing(12)
        rb_layout.setVerticalSpacing(4)

        self.filter_buttons: Dict[str, QRadioButton] = {}
        self.filter_group_rb = QButtonGroup(self)
        self.filter_group_rb.setExclusive(True)

        def add_filter_rb(name: str, row: int, col: int):
            rb = QRadioButton(name)
            if name == "None":
                rb.setChecked(True)
            self.filter_buttons[name] = rb
            self.filter_group_rb.addButton(rb)
            rb_layout.addWidget(rb, row, col)
            rb.toggled.connect(lambda checked, n=name: self._on_filter_radio(n, checked))

        all_names = [n for n in ALL_FILTERS if n != "—"]
        cols = 3
        for i, name in enumerate(all_names):
            r = i // cols
            c = i % cols
            add_filter_rb(name, r, c)

        # Strength
        self.filter_strength_slider = SliderSpinBox("Intensity", 0.0, 1.0, 1.0, 0.01)

        rb_scroll = QScrollArea()
        rb_scroll.setWidgetResizable(True)
        rb_scroll.setWidget(rb_container)

        tfl.addWidget(rb_scroll, 1)
        tfl.addWidget(self.filter_strength_slider, 0)
        self.filter_strength_slider.valueChanged.connect(self._on_filter_strength)
        self.editor_widget.addTab(tf, "Filters")

        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setCollapsible(0, True)
        left.setMinimumWidth(0)
        self._photos_splitter = split
        self._photos_splitter.setCollapsible(0, False)
        self._left_auto_collapse_px = 200
        # split.splitterMoved.connect(self._on_photos_splitter_moved)
        # split.setSizes([280, 1020])

        self.single_viewer.clear_pixmap()

    def _build_live_tab(self) -> QWidget:
        tab = QWidget()
        root = QHBoxLayout(tab)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        left_panel = QWidget()
        left_v = QVBoxLayout(left_panel)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(6)

        self.live_list = QListWidget()
        self.live_list.setMinimumWidth(200)
        self.live_list.setMaximumWidth(420)
        self.live_list.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.live_list.itemSelectionChanged.connect(lambda: self._on_live_select(self.live_list.currentItem(), None))
        left_v.addWidget(self.live_list)

        self.frames_list = QListWidget()
        self.frames_list.setMinimumWidth(200)
        self.frames_list.setMaximumWidth(420)
        self.frames_list.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.frames_list.itemSelectionChanged.connect(
            lambda: self._on_frame_selected(self.frames_list.currentItem(), None)
        )
        left_v.addWidget(self.frames_list)

        splitter.addWidget(left_panel)

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

    # ---------- Drag and drop for folders ----------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.exists() and p.is_dir():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.exists() and p.is_dir():
                self.path_edit.setText(str(p))
                self.load_directory_async(p)
                break

    # ---------- Folder load, video, list handlers ----------

    def _open_path_from_edit(self):
        p = self.path_edit.text().strip()
        if not p:
            return
        folder = Path(p)
        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(self, "Folder", f"Folder not found:\n{folder}")
            return
        self.load_directory_async(folder)

    def _browse_and_load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select a folder", QDir.homePath(),
            options=QFileDialog.Option.ShowDirsOnly
        )
        if not folder_path:
            return
        self.path_edit.setText(folder_path)
        self.load_directory_async(Path(folder_path))

    def _image_date_taken(self, path: Path) -> Optional[str]:
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
                dto = ex.get(36867) or ex.get(306)
                return _sanitize_exif_datetime(dto)
        except Exception:
            return None

    def _mov_date_taken(self, path: Path) -> Optional[str]:
        if _MEDIAINFO is None:
            return None
        try:
            info = _MEDIAINFO.parse(str(path))
            for track in info.tracks:
                if track.track_type == "General":
                    cand = getattr(track, "encoded_date", None) or getattr(track, "tagged_date", None)
                    if not cand:
                        continue
                    s = str(cand).replace("UTC ", "").strip()
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

    def load_directory_async(self, folder: Path):
        self._scan_job_id += 1
        jid = self._scan_job_id

        self.scan_status_lbl.setText(f"Scanning {folder} ...")
        self.photo_list.clear()
        self.live_list.clear()
        self.frames_list.clear()
        self.image_files = []
        self.mov_files = []
        self._clear_photos_state(reset_lists_only=True)

        job = DirScanJob(jid, folder, self)
        job.signals.started.connect(self._on_scan_started)
        job.signals.found_image.connect(self._on_scan_found_image)
        job.signals.found_mov.connect(self._on_scan_found_mov)
        job.signals.finished.connect(self._on_scan_finished)
        job.signals.error.connect(self._on_scan_error)
        self.threadpool.start(job, 0)

    def _on_scan_started(self, job_id: int, folder: str):
        if job_id != self._scan_job_id:
            return
        self.current_folder = Path(folder)

    def _on_scan_found_image(self, job_id: int, path: str, taken: str):
        if job_id != self._scan_job_id:
            return
        p = Path(path)
        self.image_files.append(path)
        item_text = self._format_list_item_with_meta(p, taken)
        self.photo_list.addItem(item_text)
        self.photo_list.item(self.photo_list.count() - 1).setToolTip(str(p))
        if self.current_image_index < 0 and len(self.image_files) == 1:
            self._load_image_by_index(0)
            self.photo_list.setCurrentRow(0)
            self.set_controls_state(True)

    def _on_scan_found_mov(self, job_id: int, path: str, taken: str):
        if job_id != self._scan_job_id:
            return
        p = Path(path)
        self.mov_files.append(path)

    def _on_scan_finished(self, job_id: int, images: List[str], movs: List[str]):
        if job_id != self._scan_job_id:
            return
        self.scan_status_lbl.setText(
            f"Loaded {len(images)} images, {len(movs)} videos from {self.current_folder}"
        )
        self.live_list.clear()
        if self.mov_files:
            for s in self.mov_files:
                p = Path(s)
                taken = self._mov_date_taken(p)
                item_text = self._format_list_item_with_meta(p, taken)
                self.live_list.addItem(item_text)
                self.live_list.item(self.live_list.count() - 1).setToolTip(str(p))

    def _on_scan_error(self, job_id: int, msg: str):
        if job_id != self._scan_job_id:
            return
        QMessageBox.critical(self, "Scan Error", msg)
        self.scan_status_lbl.setText("Scan error")

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
        self.original_image_path = None
        self.original_image_pil = None
        self.working_image_pil = None
        self._orig_format = None
        self._orig_exif = None
        self.crop_area = None
        self._reset_controls(reset_rotation=True, reset_sliders=True)
        self.single_viewer.set_editor_mode(MODE_VIEW)
        self.crop_mode_button.setChecked(False)
        self.mirror_mode_button.setChecked(False)
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
            self._preview_source = self._make_preview_source(self.working_image_pil)
            self.set_controls_state(True)
            self._update_meta_labels_for_image(Path(filepath))
            self._schedule_preview()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not process image: {e}")
            self.original_image_path = None
            self.single_viewer.setText(f"Error loading image: {e}")
            self.set_controls_state(False)

    def _clear_photos_state(self, reset_lists_only: bool = False):
        if not reset_lists_only:
            self.original_image_path = None
            self.original_image_pil = None
            self.working_image_pil = None
            self._orig_format = None
            self._orig_exif = None
            self.crop_area = None
            self.current_image_index = -1
            self.single_viewer.clear_pixmap()
            self.single_viewer.set_editor_mode(MODE_VIEW)
            self.crop_mode_button.setChecked(False)
            self.mirror_mode_button.setChecked(False)
            self._mirror_on = False
            self._reset_controls(reset_rotation=True, reset_sliders=True)
            self.meta_taken_lbl.setText("Taken: -")
            self.meta_modified_lbl.setText("Modified: -")

    # ---------- Live (.mov) ----------

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

            for i in range(self.total_frames):
                self.frames_list.addItem(f"Frame {i:04d}")

            self.play_button.setEnabled(True)
            self.play_button.setText("Play")
            self.is_playing = False

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
        idx = self.frames_list.currentRow()
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
        s.valueChanged.connect(lambda v, name=attr_name: self._on_slider_change(v, name))
        s.slider.sliderReleased.connect(self._schedule_preview)
        setattr(self, attr_name.replace("_factor", "_slider").replace("_deg", "_slider"), s)
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
        self.fill_keep_rb.setEnabled(enabled)
        self.fill_crop_rb.setEnabled(enabled)
        self.fill_stretch_rb.setEnabled(enabled)

        slider_groups = [
            "exposure", "brilliance", "brightness", "contrast",
            "highlights", "shadows", "blackpoint",
            "saturation", "vibrance", "warmth", "tint",
            "sharpness", "noise_reduction", "vignette",
        ]
        for attr_name in slider_groups:
            slider = getattr(self, f"{attr_name}_slider", None)
            if slider:
                slider.setEnabled(enabled)

        for name in ("geom_rx", "geom_ry", "geom_rz"):
            slider = getattr(self, f"{name}_slider", None)
            if slider:
                if name in ("geom_rx", "geom_ry") and (_CV2 is None or _NP is None):
                    slider.setEnabled(False)
                else:
                    slider.setEnabled(enabled)

        for rb in self.filter_buttons.values():
            rb.setEnabled(enabled)
        if hasattr(self, "filter_strength_slider"):
            self.filter_strength_slider.setEnabled(enabled)

    def _reset_controls(self, reset_rotation=True, reset_sliders=True):
        if reset_rotation:
            self.rotation_degrees = 0
            self.geom_rx_deg = 0.0
            self.geom_ry_deg = 0.0
            self.geom_rz_deg = 0.0
            for n in ("geom_rx", "geom_ry", "geom_rz"):
                s = getattr(self, f"{n}_slider", None)
                if s:
                    s.set_value(0.0)

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
            self.filter_name = "None"
            self.filter_strength = 1.0
            if "None" in self.filter_buttons:
                self.filter_buttons["None"].setChecked(True)
            if hasattr(self, "filter_strength_slider"):
                self.filter_strength_slider.set_value(1.0)

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
        self.viewer_stack.setCurrentIndex(1 if checked else 0)
        self._schedule_preview()

    def _on_fill_mode_changed(self):
        if self.fill_keep_rb.isChecked():
            self.fill_mode = FILL_KEEP
        elif self.fill_crop_rb.isChecked():
            self.fill_mode = FILL_AUTOCROP
        else:
            self.fill_mode = FILL_STRETCH
        self._schedule_preview()

    def apply_crop(self):
        if not self.original_image_path or not self.single_viewer.current_qpixmap:
            QMessageBox.warning(self, "Crop", "No valid crop area selected.")
            return

        # FREEFORM
        if self.crop_mode_type == "free":
            if _CV2 is None or _NP is None:
                QMessageBox.warning(self, "Crop", "Freeform crop requires opencv-python and numpy.")
                return
            if len(self.single_viewer.freeform_pts) != 4:
                QMessageBox.warning(self, "Crop", "Please position all four corners.")
                return

            img_rect_can = self.single_viewer.get_image_display_rect()
            qpixmap_size = self.single_viewer.current_qpixmap.size()
            w_disp, h_disp = qpixmap_size.width(), qpixmap_size.height()

            temp_rotated = self.working_image_pil.rotate(self.rotation_degrees, expand=True)
            w_rot, h_rot = temp_rotated.size
            ratio_w = w_rot / float(w_disp) if w_disp else 1.0
            ratio_h = h_rot / float(h_disp) if h_disp else 1.0

            pts = self.single_viewer.freeform_pts
            pts_img = []
            for pt in pts:
                x_img = (pt.x() - img_rect_can.left())
                y_img = (pt.y() - img_rect_can.top())
                x_img = max(0, min(x_img, w_disp))
                y_img = max(0, min(y_img, h_disp))
                pts_img.append([x_img * ratio_w, y_img * ratio_h])

            pts_np = _NP.array(pts_img, dtype=_NP.float32)
            s = pts_np.sum(axis=1)
            diff = _NP.diff(pts_np, axis=1).ravel()
            tl = pts_np[_NP.argmin(s)]
            br = pts_np[_NP.argmax(s)]
            tr = pts_np[_NP.argmin(diff)]
            bl = pts_np[_NP.argmax(diff)]
            src = _NP.array([tl, tr, br, bl], dtype=_NP.float32)

            def _dist(a, b): return float(_NP.linalg.norm(a - b))
            widthA = _dist(br, bl)
            widthB = _dist(tr, tl)
            maxW = int(round(max(widthA, widthB)))
            heightA = _dist(tr, br)
            heightB = _dist(tl, bl)
            maxH = int(round(max(heightA, heightB)))
            maxW = max(1, maxW)
            maxH = max(1, maxH)

            dst = _NP.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=_NP.float32)

            img_np = _NP.array(temp_rotated.convert("RGB"))
            M = _CV2.getPerspectiveTransform(src, dst)
            warped = _CV2.warpPerspective(img_np, M, (maxW, maxH), flags=_CV2.INTER_CUBIC)
            cropped_image = Image.fromarray(warped)

            self.working_image_pil = cropped_image
            self._preview_source = self._make_preview_source(self.working_image_pil)
            self.rotation_degrees = 0
            self.geom_rx_deg = 0.0
            self.geom_ry_deg = 0.0
            self.geom_rz_deg = 0.0
            self.crop_mode_button.setChecked(False)
            self.single_viewer.set_editor_mode(MODE_VIEW)
            self.apply_crop_button.setEnabled(False)
            self._schedule_preview()
            return

        # RECTANGULAR
        if self.single_viewer.crop_start_point and self.single_viewer.crop_end_point:
            img_rect_can = self.single_viewer.get_image_display_rect()
            qpixmap_size = self.single_viewer.current_qpixmap.size()
            w_disp, h_disp = qpixmap_size.width(), qpixmap_size.height()

            temp_rotated = self.working_image_pil.rotate(self.rotation_degrees, expand=True)
            w_rot, h_rot = temp_rotated.size
            ratio_w = w_rot / float(w_disp) if w_disp else 1.0
            ratio_h = h_rot / float(h_disp) if h_disp else 1.0

            x1 = max(self.single_viewer.crop_start_point.x(), img_rect_can.left())
            y1 = max(self.single_viewer.crop_start_point.y(), img_rect_can.top())
            x2 = min(self.single_viewer.crop_end_point.x(),   img_rect_can.right())
            y2 = min(self.single_viewer.crop_end_point.y(),   img_rect_can.bottom())
            rect = QRect(QPoint(x1, y1), QPoint(x2, y2)).normalized()

            # convert from canvas coords to rotated image coords
            ix1 = int((rect.left()   - img_rect_can.left()) * ratio_w)
            iy1 = int((rect.top()    - img_rect_can.top())  * ratio_h)
            ix2 = int((rect.right()  - img_rect_can.left()) * ratio_w)
            iy2 = int((rect.bottom() - img_rect_can.top())  * ratio_h)

            ix1 = max(0, min(ix1, w_rot - 1))
            iy1 = max(0, min(iy1, h_rot - 1))
            ix2 = max(ix1 + 1, min(ix2, w_rot))
            iy2 = max(iy1 + 1, min(iy2, h_rot))

            try:
                cropped_image = temp_rotated.crop((ix1, iy1, ix2, iy2))
            except Exception as e:
                QMessageBox.critical(self, "Crop", f"Crop failed: {e}")
                return

            self.working_image_pil = cropped_image
            self._preview_source = self._make_preview_source(self.working_image_pil)
            self.rotation_degrees = 0
            self.geom_rx_deg = 0.0
            self.geom_ry_deg = 0.0
            self.geom_rz_deg = 0.0
            self.crop_mode_button.setChecked(False)
            self.single_viewer.set_editor_mode(MODE_VIEW)
            self.apply_crop_button.setEnabled(False)
            self._schedule_preview()
            return

        QMessageBox.warning(self, "Crop", "No valid crop area selected.")

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
            self._load_image(self.original_image_path)
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

            # 1) geometry at full resolution
            geo_full = _apply_geometry_perspective(
                self.working_image_pil, self.geom_rx_deg, self.geom_ry_deg, self.geom_rz_deg,
                self.rotation_degrees, self.fill_mode
            )

            # 2) filter at full res
            if self.filter_name and self.filter_name not in ("None", "—", "Original"):
                geo_full = _apply_filter_pipeline(geo_full, self.filter_name, max(0.0, min(1.0, self.filter_strength)))

            # 3) manual edits
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
            job = _PreviewJob(0, geo_full, 0, f, geo_full.size, False, False,
                              0.0, 0.0, 0.0, "None", 0.0, self.fill_mode)
            final_image = job._apply_edits(geo_full, f)

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

    # ---------- Filter handlers (radio) ----------

    def _on_filter_radio(self, name: str, checked: bool):
        if not checked:
            return
        self.filter_name = name
        self._schedule_preview()

    def _on_filter_strength(self, v: float):
        self.filter_strength = max(0.0, min(1.0, v))
        self._schedule_preview()

    # ---------- Live preview ----------

    def _on_slider_change(self, value: float, attr_name: str):
        self._last_changed_name = attr_name
        setattr(self, attr_name, value)
        if attr_name in ("geom_rx_deg", "geom_ry_deg") and (_CV2 is None or _NP is None) and not self._warned_perspective:
            self._warned_perspective = True
            QMessageBox.warning(self, "Perspective", "Pitch/Yaw straighten needs opencv-python + numpy. Only Roll (Z) will apply.")
        self._schedule_preview()

    def _schedule_preview(self):
        self._preview_timer.start()

    def _start_preview_job(self):
        if not self.working_image_pil:
            return
        do_mirror = self.viewer_stack.currentIndex() == 1
        avail = self.dual_viewer.left_scroll.viewport().size() if do_mirror else self.single_scroll.viewport().size()
        cap_w = min(avail.width(), 1280)
        cap_h = min(avail.height(), 1280)
        target_size = (max(1, cap_w), max(1, cap_h))

        interactive = any(sl.isSliderDown() for sl in self._all_sliders)
        geom_sliders_down = any(
            getattr(self, n + "_slider", None) and getattr(self, n + "_slider").slider.isSliderDown()
            for n in ("geom_rx", "geom_ry", "geom_rz")
        )
        fast_geom = interactive and geom_sliders_down


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
            coarse_rotation_degrees=self.rotation_degrees,
            factors=f,
            single_target_size=target_size,
            do_mirror=do_mirror,
            interactive=interactive,
            geom_rx=self.geom_rx_deg,
            geom_ry=self.geom_ry_deg,
            geom_rz=self.geom_rz_deg,
            preset_name=self.filter_name,
            preset_strength=self.filter_strength,
            fill_mode=self.fill_mode,
            fast_geometry_preview=fast_geom,
        )
        job.signals.done.connect(self._on_preview_ready)
        self.threadpool.start(job, 1)

    def _on_preview_ready(self, job_id: int, main_img: QImage, mirror_img_obj: object):
        if job_id != self._preview_job_id:
            return
        main_pm = QPixmap.fromImage(main_img)
        mirror_pm = QPixmap.fromImage(mirror_img_obj) if isinstance(mirror_img_obj, QImage) else None
        if self.viewer_stack.currentIndex() == 0:
            self.single_viewer.set_pixmap(main_pm)
        else:
            self.dual_viewer.set_pixmaps(main_pm, mirror_pm)


    # ---------- Metadata ----------

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

    # ---------- Delete ----------

    def _on_delete_key(self):
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

        self.mov_files.pop(idx)
        self.live_list.takeItem(idx)

        if self.mov_files:
            new_idx = min(idx, len(self.mov_files) - 1)
            self.current_mov_index = -1
            self.live_list.setCurrentRow(new_idx)
            self._select_live_by_index(new_idx)
        else:
            self._clear_live_state()

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
