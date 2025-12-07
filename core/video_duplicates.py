#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video duplicate detection utilities.

Efficiently finds duplicate / near-identical videos by sampling a few keyframes
from each file and computing a perceptual hash (pHash) for those frames.

Public API:

    groups = VideoDuplicateFinder.find_video_duplicates(
        video_paths,
        progress_cb=None,
    )

where:
    - video_paths: iterable of paths to video files (.mov, .mp4, etc.)
    - progress_cb: optional callback taking an int percentage [0..100]
    - return value: List[List[DuplicateRecord]]
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Dict

import cv2
import numpy as np
import os
from .duplicates import DuplicateRecord
from PyQt6.QtCore import  QRunnable, pyqtSlot, QObject, pyqtSignal
from .common import (
    Image, ImageOps, ImageEnhance, _CV2, _NP, _HEIF_PLUGIN, _PIEXIF, _MEDIAINFO,)
class VideoDuplicateWorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, str)



class VideoDuplicateWorker(QRunnable):
    """
    Highly Optimized Video Duplicate Finder.
    
    Strategy:
    1. Filter by File Size: Videos with different file sizes cannot be duplicates.
    2. Filter by Duration: Same size? Check duration.
    3. Filter by Visual Hash: Extract 1 frame from the MIDDLE of the video,
       resize to 9x8 grayscale, and compare bits (dHash).
    
    This avoids reading the full file or decoding every frame.
    """
    def __init__(self, video_paths: List[str]):
        super().__init__()
        self.video_paths = list(video_paths)
        self.signals = VideoDuplicateWorkerSignals()
        self.is_killed = False

    @pyqtSlot()
    def run(self) -> None:
        try:
            total_files = len(self.video_paths)
            if total_files < 2:
                self.signals.finished.emit([], "Not enough videos to scan.")
                return

            # --- Step 1: Group by File Size (Instant) ---
            size_map = {}
            for idx, path in enumerate(self.video_paths):
                if self.is_killed: return
                try:
                    sz = os.path.getsize(path)
                    if sz not in size_map: size_map[sz] = []
                    size_map[sz].append(path)
                except OSError: pass
                
                # Emit progress for the first pass (0-10%)
                if idx % 10 == 0:
                    pct = int((idx / total_files) * 10)
                    self.signals.progress.emit(pct)

            # Filter groups with > 1 item
            candidates = [paths for paths in size_map.values() if len(paths) > 1]
            
            # --- Step 2 & 3: Duration & Visual Hash ---
            # We only process files that already share a file size
            final_groups = []
            
            total_candidates = sum(len(g) for g in candidates)
            processed_count = 0

            for group in candidates:
                if self.is_killed: return
                
                # Sub-group by duration
                duration_map = {}
                
                for path in group:
                    if self.is_killed: return
                    
                    # 10% to 100% progress range
                    processed_count += 1
                    current_progress = 10 + int((processed_count / max(1, total_candidates)) * 90)
                    if processed_count % 5 == 0:
                        self.signals.progress.emit(current_progress)

                    # Get Visual Hash + Duration
                    # We bundle them to avoid opening the file twice
                    vid_hash, duration = self._get_video_fingerprint(path)
                    
                    if vid_hash is None: continue

                    # Create a composite key: (Duration, VisualHash)
                    # We round duration to nearest second to handle minor encoding diffs
                    key = (round(duration), vid_hash)
                    
                    if key not in duration_map: duration_map[key] = []
                    duration_map[key].append(path)

                # Collect duplicates from this size bucket
                for key, paths in duration_map.items():
                    if len(paths) > 1:
                        # Convert to DuplicateRecord
                        group_records = [DuplicateRecord(p, 0) for p in paths]
                        final_groups.append(group_records)

            msg = f"Found {len(final_groups)} duplicate video groups." if final_groups else "No duplicate videos found."
            self.signals.finished.emit(final_groups, msg)

        except Exception as e:
            self.signals.finished.emit([], f"Error scanning videos: {str(e)}")

    def _get_video_fingerprint(self, path: str) -> Tuple[Optional[str], float]:
        """
        Returns (dHash_string, duration_seconds).
        """
        if _CV2 is None: return None, 0.0
        try:
            cap = _CV2.VideoCapture(path)
            if not cap.isOpened(): return None, 0.0

            frame_count = int(cap.get(_CV2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(_CV2.CAP_PROP_FPS)
            duration = 0.0
            if fps > 0: duration = frame_count / fps

            # Seek to middle frame to avoid black intro/outro
            target = frame_count // 2
            cap.set(_CV2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None: return None, 0.0

            # dHash Logic
            # 1. Grayscale
            gray = _CV2.cvtColor(frame, _CV2.COLOR_BGR2GRAY)
            # 2. Resize to 9x8
            small = _CV2.resize(gray, (9, 8))
            # 3. Compare adjacent pixels
            # 0 if P[i] < P[i+1], else 1
            diff = small[:, 1:] > small[:, :-1]
            # 4. Convert to hex string
            # Flatten 8x8 boolean array -> 64 bits -> integer -> hex
            decimal_value = 0
            for i, val in enumerate(diff.flatten()):
                if val: decimal_value += 2**i
            
            hex_hash = hex(decimal_value)[2:]
            return hex_hash, duration

        except Exception:
            return None, 0.0

    def kill(self):
        self.is_killed = True
