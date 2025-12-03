#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

# PyQt
from PyQt6.QtGui import QImage, QPixmap

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

_HEIF_PLUGIN = False
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
    _HEIF_PLUGIN = True
except Exception:
    _HEIF_PLUGIN = False

_PIEXIF = None
try:
    import piexif  # type: ignore
    _PIEXIF = piexif
except Exception:
    _PIEXIF = None

_S2T = None
try:
    from send2trash import send2trash  # type: ignore
    _S2T = send2trash
except Exception:
    _S2T = None

_MEDIAINFO = None
try:
    from pymediainfo import MediaInfo  # type: ignore
    _MEDIAINFO = MediaInfo
except Exception:
    _MEDIAINFO = None

# Formats / constants
EXT_TO_FMT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP",
    ".gif": "GIF", ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP",
    ".heic": "HEIF", ".heif": "HEIF", ".heics": "HEIF", ".heifs": "HEIF",
    ".hif": "HEIF", ".avif": "AVIF",
}
HEIF_LIKE_EXTS = {".heic", ".heif", ".heics", ".heifs", ".hif", ".avif"}
SUPPORTED_IMAGE_EXTS = set(EXT_TO_FMT.keys())
SUPPORTED_LIVE_EXTS = {".mov"}

MODE_VIEW = 0
MODE_CROP = 1

FILL_KEEP = "keep"
FILL_AUTOCROP = "crop"
FILL_STRETCH = "stretch"

LEFT_PANEL_WIDTH = 320  # px


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
