#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import hashlib
from pathlib import Path
from typing import List

from PyQt6.QtCore import QRunnable, pyqtSignal, QObject
from PyQt6.QtGui import QImage

# --- Core Imports ---
from core.common import (
    Image, ImageEnhance, _CV2, _NP, _MEDIAINFO,
    SUPPORTED_IMAGE_EXTS, SUPPORTED_LIVE_EXTS,
    pil_to_qimage, _sanitize_exif_datetime
)
from core.geometry import _apply_geometry_perspective
from core.filters import _apply_filter_pipeline

# Core Logic
from core.duplicates_logic import compute_image_hash, compute_video_hash, DuplicateRecord

# --- Signals ---


class WorkerSignals(QObject):
    started = pyqtSignal(int, str)
    finished = pyqtSignal(int, list, list)  # id, images, movs
    progress = pyqtSignal(int)
    error = pyqtSignal(int, str)


class PreviewSignals(QObject):
    done = pyqtSignal(int, object, object)  # id, main_qimage, mirror_qimage


class DupScanSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, str)  # groups, message
    error = pyqtSignal(str)


class ThumbnailSignals(QObject):
    loaded = pyqtSignal(str, object)  # path, QImage


# --- 1. Directory Scanner Job (OPTIMIZED) ---


class DirScanJob(QRunnable):
    def __init__(self, job_id: int, folder: Path):
        super().__init__()
        self.job_id = job_id
        self.folder = folder
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.started.emit(self.job_id, str(self.folder))
            images = []
            movs = []

            # Fast walk: Case-insensitive sort, but NO metadata extraction here.
            # Metadata should be loaded lazily by the UI when needed.
            for p in sorted(self.folder.rglob("*"), key=lambda x: str(x).lower()):
                if not p.is_file():
                    continue
                if p.name.startswith('.'):
                    continue

                # Check for "deleted" folder in path parts
                parts_lower = [part.lower() for part in p.parts]
                if "deleted" in parts_lower:
                    continue

                ext = p.suffix.lower()

                if ext in SUPPORTED_IMAGE_EXTS:
                    images.append(str(p))

                elif ext in SUPPORTED_LIVE_EXTS:
                    movs.append(str(p))

            self.signals.finished.emit(self.job_id, images, movs)
        except Exception as e:
            self.signals.error.emit(self.job_id, str(e))

# --- 2. Image Editor Preview Job ---


class PreviewJob(QRunnable):
    def __init__(self, job_id, base_image, rotation, factors, target_size, do_mirror,
                 geom_rx, geom_ry, geom_rz, filter_name, filter_str, fill_mode,
                 fast_geometry_preview=True):
        super().__init__()
        self.job_id = job_id
        self.image = base_image
        self.rotation = rotation
        self.factors = factors
        self.target_size = target_size
        self.do_mirror = do_mirror
        self.geom_rx = geom_rx
        self.geom_ry = geom_ry
        self.geom_rz = geom_rz
        self.filter_name = filter_name
        self.filter_str = filter_str
        self.fill_mode = fill_mode
        self.signals = PreviewSignals()

    def run(self):
        if self.image is None:
            return

        # 1. Geometry
        geo_img = _apply_geometry_perspective(
            self.image, self.geom_rx, self.geom_ry, self.geom_rz,
            self.rotation, self.fill_mode, preview_fast=True
        )

        # 2. Resize
        t_w = max(1, min(self.target_size[0], 1280))
        t_h = max(1, min(self.target_size[1], 1280))

        gw, gh = geo_img.size
        if gw > 0 and gh > 0:
            ratio = min(t_w / float(gw), t_h / float(gh))
            new_size = (max(1, int(gw * ratio)), max(1, int(gh * ratio)))
            preview = geo_img.resize(new_size, Image.Resampling.BILINEAR)
        else:
            preview = geo_img

        # 3. Filters
        if self.filter_name and self.filter_name not in ("None", "—", "Original"):
            preview = _apply_filter_pipeline(
                preview, self.filter_name, self.filter_str)

        # 4. Color/Tone Edits
        final_preview = self._apply_edits(preview, self.factors)

        # 5. Output
        main_q = pil_to_qimage(final_preview)
        mirror_q = None
        if self.do_mirror:
            try:
                flip = Image.Transpose.FLIP_LEFT_RIGHT
            except:
                flip = Image.FLIP_LEFT_RIGHT
            mirror_q = pil_to_qimage(final_preview.transpose(flip))

        self.signals.done.emit(self.job_id, main_q, mirror_q)

    def _apply_edits(self, img, f):
        img = img.convert("RGB")

        if f["exposure"] != 0:
            img = ImageEnhance.Brightness(
                img).enhance(1.0 + f["exposure"] / 2.0)
        if f["brightness"] != 0:
            img = ImageEnhance.Brightness(img).enhance(
                1.0 + f["brightness"] / 100.0)
        if f["contrast"] != 0:
            img = ImageEnhance.Contrast(img).enhance(
                1.0 + f["contrast"] / 100.0)
        if f["saturation"] != 0:
            img = ImageEnhance.Color(img).enhance(
                1.0 + f["saturation"] / 100.0)
        if f["sharpness"] != 0:
            img = ImageEnhance.Sharpness(img).enhance(
                1.0 + f["sharpness"] / 100.0)

        if _NP is not None:
            arr = _NP.array(img).astype(float)
            if f["warmth"] != 0:
                r_gain = 1.0 + (f["warmth"] / 150.0)
                b_gain = 1.0 - (f["warmth"] / 150.0)
                arr[:, :, 0] *= r_gain
                arr[:, :, 2] *= b_gain

            arr = _NP.clip(arr, 0, 255).astype(_NP.uint8)
            img = Image.fromarray(arr)

        return img

# --- 3. Duplicate Scanning Worker ---


class ScanDuplicatesWorker(QRunnable):
    def __init__(self, paths, mode="Images", exact=False):
        super().__init__()
        self.paths = paths
        self.mode = mode
        self.exact = exact
        self.signals = DupScanSignals()
        self.is_killed = False

    def run(self):
        try:
            total = len(self.paths)
            if total < 2:
                self.signals.finished.emit([], "Not enough files.")
                return

            # 1. Quick grouping by size
            size_map = {}
            for i, p in enumerate(self.paths):
                if self.is_killed:
                    return
                if i % 100 == 0:
                    self.signals.progress.emit(int(i/total * 20))
                try:
                    s = os.path.getsize(p)
                    size_map.setdefault(s, []).append(p)
                except:
                    pass

            candidates = [g for g in size_map.values() if len(g) > 1]
            final_groups = []

            processed = 0
            total_cand = sum(len(g) for g in candidates)
            if total_cand == 0:
                self.signals.finished.emit(
                    [], "No duplicates found (size check).")
                return

            for group in candidates:
                if self.is_killed:
                    return
                hashes = {}

                for p in group:
                    if self.is_killed:
                        return
                    processed += 1
                    if processed % 10 == 0:
                        self.signals.progress.emit(
                            20 + int(processed/max(1, total_cand) * 80))

                    k = None
                    if self.exact:
                        k = self._get_md5(p)
                    elif self.mode == "Images":
                        k = compute_image_hash(p)
                    else:
                        k = compute_video_hash(p)  # returns tuple

                    if k:
                        hashes.setdefault(k, []).append(p)

                for dupes in hashes.values():
                    if len(dupes) > 1:
                        final_groups.append([DuplicateRecord(x)
                                            for x in dupes])

            self.signals.finished.emit(
                final_groups, f"Found {len(final_groups)} groups.")

        except Exception as e:
            self.signals.error.emit(str(e))

    def _get_md5(self, path):
        try:
            h = hashlib.md5()
            with open(path, "rb") as f:
                chunk = f.read(65536)
                h.update(chunk)
            return h.hexdigest()
        except:
            return None

    def kill(self):
        self.is_killed = True

# --- 4. Worker Aliases (For Compatibility) ---


class FastDuplicateWorker(ScanDuplicatesWorker):
    def __init__(self, paths):
        super().__init__(paths, mode="Images", exact=False)


class ExactDuplicateWorker(ScanDuplicatesWorker):
    def __init__(self, paths):
        super().__init__(paths, mode="Images", exact=True)


class VideoDuplicateWorker(ScanDuplicatesWorker):
    def __init__(self, paths):
        super().__init__(paths, mode="Videos", exact=False)


# --- 5. Thumbnail Loader Job ---


class ThumbnailLoaderJob(QRunnable):
    """
    Background job to load thumbnails for duplicates to prevent UI lag.
    """

    def __init__(self, path: str, size: int = 160):
        super().__init__()
        self.path = path
        self.size = size
        self.signals = ThumbnailSignals()

    def run(self):
        qimg = None
        ext = os.path.splitext(self.path)[1].lower()

        # 1. Try Video Thumbnail
        if ext in SUPPORTED_LIVE_EXTS and _CV2 is not None:
            try:
                cap = _CV2.VideoCapture(self.path)
                if cap.isOpened():
                    cap.grab()  # Grab first frame
                    ret, frame = cap.retrieve()
                    cap.release()
                    if ret and frame is not None:
                        frame = _CV2.cvtColor(frame, _CV2.COLOR_BGR2RGB)
                        h, w, ch = frame.shape
                        qimg = QImage(frame.data, w, h, ch * w,
                                      QImage.Format.Format_RGB888).copy()
            except:
                pass

        # 2. Image Thumbnail (Fallback or Primary)
        if qimg is None:
            try:
                with Image.open(self.path) as img:
                    img.thumbnail((self.size, self.size))
                    qimg = pil_to_qimage(img)
            except:
                pass

        # Emit result (even if None, so UI knows we failed)
        self.signals.loaded.emit(self.path, qimg)
