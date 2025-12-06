#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from datetime import datetime
from typing import Optional

# PIL Imports (Critical for filters/editing)
from PIL import Image, ImageOps, ImageEnhance

# PyQt
from PyQt6.QtGui import QImage, QPixmap

# --- Optional Dependencies ---
_CV2 = None
_NP = None
try:
    import cv2
    _CV2 = cv2
    import numpy as np
    _NP = np
except ImportError:
    _CV2 = None
    _NP = None

_HEIF_PLUGIN = False
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIF_PLUGIN = True
except ImportError:
    pass

_PIEXIF = None
try:
    import piexif
    _PIEXIF = piexif
except ImportError:
    pass

_MEDIAINFO = None
try:
    from pymediainfo import MediaInfo
    _MEDIAINFO = MediaInfo
except ImportError:
    pass

# --- Constants ---
EXT_TO_FMT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP",
    ".gif": "GIF", ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP",
    ".heic": "HEIF", ".heif": "HEIF", ".heics": "HEIF", ".heifs": "HEIF",
    ".hif": "HEIF", ".avif": "AVIF",
}
HEIF_LIKE_EXTS = {".heic", ".heif", ".heics", ".heifs", ".hif", ".avif"}
SUPPORTED_IMAGE_EXTS = set(EXT_TO_FMT.keys())
SUPPORTED_LIVE_EXTS = {".mov", ".mp4"}

MODE_VIEW = 0
MODE_CROP = 1

FILL_KEEP = "keep"
FILL_AUTOCROP = "crop"
FILL_STRETCH = "stretch"

LEFT_PANEL_WIDTH = 320

# --- Utils ---


def pil_to_qimage(pil_image: Image.Image) -> QImage:
    """Converts a PIL Image to a PyQt6 QImage."""
    if pil_image.mode == "RGB":
        r, g, b = pil_image.split()
        pil_image = Image.merge(
            "RGBA", (r, g, b, Image.new("L", pil_image.size, 255)))
    elif pil_image.mode == "L":
        pil_image = pil_image.convert("RGBA")
    elif pil_image.mode != "RGBA":
        pil_image = pil_image.convert("RGBA")

    data = pil_image.tobytes("raw", "RGBA")
    qimg = QImage(
        data,
        pil_image.width,
        pil_image.height,
        QImage.Format.Format_RGBA8888
    ).copy()
    return qimg


def pil_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    """Converts a PIL Image directly to a QPixmap."""
    return QPixmap.fromImage(pil_to_qimage(pil_image))


def _fmt_ts_local(ts: float) -> str:
    """Formats a timestamp into a readable string."""
    try:
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def _sanitize_exif_datetime(s: Optional[str]) -> Optional[str]:
    """Cleans up EXIF date strings."""
    if not s:
        return None
    try:
        # Standardize "YYYY:MM:DD" to "YYYY-MM-DD" if needed
        s2 = s.strip().replace("/", ":")
        if len(s2) >= 19 and s2[4] == ":" and s2[7] == ":":
            dt = datetime.strptime(s2[:19], "%Y:%m:%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        # Try direct parse if it's already dashes
        dt = datetime.strptime(s2[:19], "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return s
