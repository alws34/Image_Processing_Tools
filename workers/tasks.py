#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, pyqtSlot

# Core imports
from core.common import (
    Image, ImageEnhance, _CV2, _NP, _MEDIAINFO,
    SUPPORTED_IMAGE_EXTS, SUPPORTED_LIVE_EXTS,
    pil_to_qimage,
)
from core.geometry import _apply_geometry_perspective
from core.filters import _apply_filter_pipeline
from core.duplicates import DuplicateRecord


# --- 1. Directory Scanner ---
class DirScanSignals(QObject):
    started = pyqtSignal(int, str)
    found_image = pyqtSignal(int, str, str)
    found_mov = pyqtSignal(int, str, str)
    finished = pyqtSignal(int, list, list)
    error = pyqtSignal(int, str)

class DirScanJob(QRunnable):
    def __init__(self, job_id: int, folder: Path, date_extractor_func):
        super().__init__()
        self.job_id = job_id
        self.folder = folder
        self.signals = DirScanSignals()
        self.get_date = date_extractor_func

    def run(self):
        try:
            self.signals.started.emit(self.job_id, str(self.folder))
            if not self.folder.is_dir():
                self.signals.error.emit(self.job_id, f"Not a directory: {self.folder}")
                return

            images: List[str] = []
            movs: List[str] = []

            # Case-insensitive sort for consistency
            for p in sorted(self.folder.rglob("*"), key=lambda x: str(x).lower()):
                if not p.is_file():
                    continue
                
                # --- FILTERING LOGIC ---
                # 1. Skip hidden files (starts with .)
                if p.name.startswith('.'):
                    continue

                # 2. Skip any file located inside a folder named "deleted"
                # We check the path parts. e.g. /photos/deleted/image.jpg -> 'deleted' is in parts
                path_parts_lower = set(part.lower() for part in p.parts)
                if "deleted" in path_parts_lower:
                    continue
                # -----------------------

                ext = p.suffix.lower()

                if ext in SUPPORTED_IMAGE_EXTS:
                    taken = self.get_date(p, is_video=False)
                    images.append(str(p))
                    self.signals.found_image.emit(self.job_id, str(p), taken or "-")

                elif ext in SUPPORTED_LIVE_EXTS:
                    taken = self.get_date(p, is_video=True)
                    movs.append(str(p))
                    self.signals.found_mov.emit(self.job_id, str(p), taken or "-")

            self.signals.finished.emit(self.job_id, images, movs)
        except Exception as e:
            self.signals.error.emit(self.job_id, str(e))


# --- 2. Preview Job ---
class PreviewSignals(QObject):
    done = pyqtSignal(int, object, object)

class PreviewJob(QRunnable):
    def __init__(self, job_id, base_image, coarse_rotation_degrees, factors, single_target_size, do_mirror, interactive, geom_rx, geom_ry, geom_rz, preset_name, preset_strength, fill_mode, fast_geometry_preview=False):
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
        self.fast_geometry_preview = fast_geometry_preview
        self.signals = PreviewSignals()

    def run(self):
        if self.image is None: return

        # Geometry
        geo_img = _apply_geometry_perspective(
            self.image, self.geom_rx, self.geom_ry, self.geom_rz,
            self.coarse_rotation_degrees, self.fill_mode,
            preview_fast=self.fast_geometry_preview,
        )

        # Resize
        if self.fast_geometry_preview:
            t_w, t_h = min(self.target_size[0], 480), min(self.target_size[1], 480)
        else:
            t_w, t_h = self.target_size
            if self.interactive: t_w, t_h = min(t_w, 720), min(t_h, 720)

        if t_w <= 0 or t_h <= 0: t_w, t_h = geo_img.size
        
        gw, gh = geo_img.size
        ratio = min(t_w / float(gw), t_h / float(gh))
        new_size = (max(1, int(gw * ratio)), max(1, int(gh * ratio)))
        preview = geo_img.resize(new_size, Image.Resampling.BILINEAR)

        # Filters
        if self.preset_name and self.preset_name not in ("None", "—", "Original"):
            preview = _apply_filter_pipeline(preview, self.preset_name, max(0.0, min(1.0, self.preset_strength)))

        # Color Edits
        final_preview = self._apply_edits(preview, self.factors)

        # Convert
        main_qimage = pil_to_qimage(final_preview)
        mirror_qimage = None
        if self.do_mirror:
            try: flip_const = Image.Transpose.FLIP_LEFT_RIGHT
            except AttributeError: flip_const = Image.FLIP_LEFT_RIGHT
            mirror_preview = final_preview.transpose(flip_const)
            mirror_qimage = pil_to_qimage(mirror_preview)

        self.signals.done.emit(self.job_id, main_qimage, mirror_qimage)

    def _apply_edits(self, pil_image, f):
        img = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
        working_img = img.copy()

        if f["exposure"] != 0:
            working_img = ImageEnhance.Brightness(working_img).enhance(1.0 + f["exposure"] / 2.0)
        if f["brightness"] != 0:
            working_img = ImageEnhance.Brightness(working_img).enhance(1.0 + f["brightness"] / 100.0)
        if f["contrast"] != 0:
            working_img = ImageEnhance.Contrast(working_img).enhance(1.0 + f["contrast"] / 100.0)
        
        # Numpy/CV2 enhancements if available
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

        if _CV2 is not None and _NP is not None and abs(f["brilliance"]) > 0.01:
            brill = float(f["brilliance"])
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
                if k % 2 == 0: k += 1
                L_blur = _CV2.GaussianBlur(L, (k, k), 0)
                alpha = (0.4 if self.interactive else 0.8) * (abs(brill) / 100.0)
                alpha = max(0.0, min(alpha, 0.9))
                L2 = _CV2.addWeighted(L, 1.0 - alpha, L_blur, alpha, 0)
            lab2 = _CV2.merge((L2, a, b))
            working_img = Image.fromarray(_CV2.cvtColor(lab2, _CV2.COLOR_LAB2RGB))

        if f["saturation"] != 0:
            working_img = ImageEnhance.Color(working_img).enhance(1.0 + f["saturation"] / 100.0)

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

        if f["sharpness"] != 0:
            sharp_adj = 1.0 + ((f["sharpness"] / 100.0) * (0.5 if self.interactive else 1.0))
            working_img = ImageEnhance.Sharpness(working_img).enhance(sharp_adj)

        if f["vignette"] > 0.1 and _NP is not None:
            w, h = working_img.size
            y_coords, x_coords = _NP.indices((h, w))
            cx, cy = w / 2.0, h / 2.0
            radius = min(w, h) / 2.0
            dist = _NP.hypot(x_coords - cx, y_coords - cy) / radius
            falloff = _NP.power(dist, (1.0 + (f["vignette"] * 0.5)))
            mask = _NP.clip(1.0 - (falloff * 0.6), 0.0, 1.0)
            image_np = _NP.array(working_img).astype(_NP.float32) / 255.0
            final_np = _NP.clip(image_np * mask[:, :, _NP.newaxis], 0.0, 1.0)
            working_img = Image.fromarray((final_np * 255).astype(_NP.uint8))

        return working_img


# --- 3. Fast Duplicate Scanner (Visual dHash) ---
class DupScanSignals(QObject):
    finished = pyqtSignal(list, str)
    progress = pyqtSignal(int)


class FastDuplicateWorker(QRunnable):
    """
    Fast visual duplicate detector for images.

    Speed optimizations vs previous version:
    - Uses NEAREST resize to 9x8 (good enough for dHash).
    - Computes an integer hash directly instead of building a bit-string.
    - Optional min_size_bytes filter to skip tiny files/icons.
    """
    def __init__(self, image_paths, min_size_bytes: int = 0):
        super().__init__()
        self.paths = list(image_paths)
        self.min_size_bytes = int(min_size_bytes)
        self.signals = DupScanSignals()
        self.is_killed = False

    def _compute_dhash_int(self, img: Image.Image) -> int:
        """
        Compute a simple dHash as a single integer.

        - Convert to grayscale
        - Resize to 9x8
        - Compare each pixel to its neighbor on the right
        - Pack bits into an int
        """
        # 1. Grayscale + cheap resize
        img = img.convert("L").resize((9, 8), Image.Resampling.NEAREST)
        pixels = list(img.getdata())

        # 9x8 => 8 * 8 = 64 comparisons
        # Pack bits into an int
        val = 0
        bit_index = 0
        width = 9

        for y in range(8):
            row_offset = y * width
            for x in range(width - 1):
                left = pixels[row_offset + x]
                right = pixels[row_offset + x + 1]

                bit = 1 if left > right else 0
                val = (val << 1) | bit
                bit_index += 1

        return val

    def run(self):
        hashes = {}
        count = 0
        total = len(self.paths) if self.paths else 1

        for path in self.paths:
            if self.is_killed:
                return

            try:
                # Optional: skip very small files (icons / thumbnails)
                if self.min_size_bytes > 0:
                    try:
                        if os.path.getsize(path) < self.min_size_bytes:
                            count += 1
                            if count % 10 == 0:
                                self.signals.progress.emit(int((count / total) * 100))
                            continue
                    except OSError:
                        # If size check fails just continue to next file
                        count += 1
                        if count % 10 == 0:
                            self.signals.progress.emit(int((count / total) * 100))
                        continue

                with Image.open(path) as img:
                    img_hash = self._compute_dhash_int(img)

                # Use the int directly as dict key
                if img_hash not in hashes:
                    hashes[img_hash] = []
                hashes[img_hash].append(path)

            except Exception:
                # Ignore unreadable or unsupported files
                pass

            count += 1
            if count % 10 == 0:
                self.signals.progress.emit(int((count / total) * 100))

        groups = []
        for paths in hashes.values():
            if len(paths) > 1:
                groups.append([DuplicateRecord(p, 0) for p in paths])

        self.signals.finished.emit(groups, f"Found {len(groups)} visual duplicate groups.")

    def kill(self):
        self.is_killed = True


# --- 4. Exact Duplicate Worker (Byte-for-byte) ---
class ExactDuplicateWorker(QRunnable):
    """
    Finds byte-identical files regardless of visual content (images or videos).
    1. Group by file size.
    2. Group by partial hash (first 4KB).
    3. Group by full hash (only for remaining collisions).
    """
    def __init__(self, paths: List[str]):
        super().__init__()
        self.paths = paths
        self.signals = DupScanSignals() # Reuse signals
        self.is_killed = False

    def run(self):
        try:
            total_files = len(self.paths)
            if total_files < 2:
                self.signals.finished.emit([], "Not enough files.")
                return

            # 1. Group by Size
            size_map = {}
            for i, p in enumerate(self.paths):
                if self.is_killed: return
                try:
                    s = os.path.getsize(p)
                    if s not in size_map: size_map[s] = []
                    size_map[s].append(p)
                except OSError: pass
                
                if i % 100 == 0:
                    self.signals.progress.emit(int(i / total_files * 30))

            candidates = [g for g in size_map.values() if len(g) > 1]
            final_groups = []
            
            total_candidates = sum(len(g) for g in candidates)
            processed = 0

            # 2. Hash Check
            for group in candidates:
                if self.is_killed: return
                
                # Check partial hash first
                partial_map = {}
                for p in group:
                    if self.is_killed: return
                    ph = self._get_hash(p, first_chunk_only=True)
                    if ph:
                        if ph not in partial_map: partial_map[ph] = []
                        partial_map[ph].append(p)
                    
                    processed += 1
                    if processed % 10 == 0:
                        pct = 30 + int(processed / max(1, total_candidates) * 30)
                        self.signals.progress.emit(pct)

                # Check full hash for partial collisions
                for sub_group in partial_map.values():
                    if len(sub_group) < 2: continue
                    
                    full_map = {}
                    for p in sub_group:
                        if self.is_killed: return
                        fh = self._get_hash(p, first_chunk_only=False)
                        if fh:
                            if fh not in full_map: full_map[fh] = []
                            full_map[fh].append(p)
                    
                    for duplicates in full_map.values():
                        if len(duplicates) > 1:
                            recs = [DuplicateRecord(path, 0) for path in duplicates]
                            final_groups.append(recs)

            self.signals.progress.emit(100)
            msg = f"Found {len(final_groups)} exact duplicate groups."
            self.signals.finished.emit(final_groups, msg)

        except Exception as e:
            self.signals.finished.emit([], f"Error: {e}")

    def _get_hash(self, path, first_chunk_only=False):
        try:
            hash_md5 = hashlib.md5()
            with open(path, "rb") as f:
                if first_chunk_only:
                    chunk = f.read(4096) # Read 4KB header
                    hash_md5.update(chunk)
                else:
                    for chunk in iter(lambda: f.read(65536), b""): # Read 64KB chunks
                        hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except OSError:
            return None

    def kill(self): self.is_killed = True


# --- 5. Video Duplicate Worker (Optimized) ---
class VideoDuplicateWorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, str)

class VideoDuplicateWorker(QRunnable):
    def __init__(self, video_paths: List[str]):
        super().__init__()
        self.video_paths = list(video_paths)
        self.signals = VideoDuplicateWorkerSignals()
        self.is_killed = False

    def run(self) -> None:
        try:
            total = len(self.video_paths)
            if total < 2:
                self.signals.finished.emit([], "Not enough videos.")
                return

            # 1. Group by Size
            size_map = {}
            for idx, p in enumerate(self.video_paths):
                if self.is_killed: return
                try:
                    sz = os.path.getsize(p)
                    if sz not in size_map: size_map[sz] = []
                    size_map[sz].append(p)
                except: pass
                if idx % 50 == 0: self.signals.progress.emit(int(idx/total*20))

            candidates = [g for g in size_map.values() if len(g) > 1]
            final_groups = []
            
            total_cand = sum(len(g) for g in candidates)
            processed = 0

            # 2. Visual Hash on candidates
            for group in candidates:
                if self.is_killed: return
                hash_map = {}
                for p in group:
                    if self.is_killed: return
                    
                    vh, dur = self._get_fingerprint(p)
                    if vh:
                        # Duration + Hash key
                        k = (int(dur), vh)
                        if k not in hash_map: hash_map[k] = []
                        hash_map[k].append(p)

                    processed += 1
                    if processed % 5 == 0:
                         self.signals.progress.emit(20 + int(processed/max(1,total_cand)*80))

                for dupes in hash_map.values():
                    if len(dupes) > 1:
                        final_groups.append([DuplicateRecord(x, 0) for x in dupes])

            self.signals.finished.emit(final_groups, f"Found {len(final_groups)} video groups.")
        except Exception as e:
            self.signals.finished.emit([], str(e))

    def _get_fingerprint(self, path):
        if _CV2 is None: return None, 0
        try:
            cap = _CV2.VideoCapture(path)
            if not cap.isOpened(): return None, 0
            frames = int(cap.get(_CV2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(_CV2.CAP_PROP_FPS)
            dur = frames/fps if fps > 0 else 0
            
            # Middle frame
            cap.set(_CV2.CAP_PROP_POS_FRAMES, frames // 2)
            ret, frame = cap.read()
            cap.release()
            if not ret: return None, 0
            
            gray = _CV2.cvtColor(frame, _CV2.COLOR_BGR2GRAY)
            small = _CV2.resize(gray, (9, 8))
            pixels = list(small.flatten())
            # Simple dHash
            diff = [1 if pixels[i] > pixels[i+1] else 0 for i in range(len(pixels)-1)]
            # Convert list of bits to hex string
            val = 0
            for b in diff: val = (val << 1) | b
            return hex(val), dur
        except: return None, 0

    def kill(self): self.is_killed = True