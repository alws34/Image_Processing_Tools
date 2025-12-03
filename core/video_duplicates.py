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

from .duplicates import DuplicateRecord


class VideoDuplicateFinder:
    """
    Efficiently find duplicate videos by sparsely sampling keyframes and
    clustering videos with identical signatures.

    For each video we:
        - Grab up to 3 frames: start, middle, and end.
        - Compute a 64-bit pHash for each frame.
        - Use the (h_start, h_mid, h_end) tuple as the signature.
        - Group videos whose full signature matches exactly.

    This is very fast: only ~3 frame decodes per video.
    """

    @staticmethod
    def _compute_frame_phash(frame: np.ndarray) -> Optional[int]:
        """
        Compute an image pHash from a BGR frame (OpenCV format).

        Returns:
            64-bit integer or None if the frame is invalid.
        """
        if frame is None or frame.size == 0:
            return None

        try:
            # Convert BGR -> gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize to a small, fixed size; 32x32 is typical for pHash
            small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

            # Convert to float32 for DCT
            small_f = small.astype("float32")

            # 2D DCT
            dct = cv2.dct(small_f)

            # Take the top-left 8x8 block
            dct_low = dct[:8, :8]

            # Use median (excluding DC) as threshold
            flat = dct_low.flatten()
            dc = flat[0]
            rest = flat[1:]
            thresh = np.median(rest)

            bits = dct_low > thresh

            # Pack into 64-bit integer
            h = 0
            for bit in bits.flatten():
                h = (h << 1) | int(bool(bit))

            return int(h)
        except Exception:
            return None

    @staticmethod
    def _get_signature_for_video(path: Path) -> Optional[Tuple[int, int, int]]:
        """
        Compute the (start, middle, end) pHash signature for a video.

        Returns:
            Tuple of three ints, or None if we cannot compute at least
            one valid frame hash.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                # Some containers do not expose frame count reliably.
                # Fallback: probe a few frames sequentially.
                indices = [0, 15, 30]
            else:
                # Frame indices for start, middle, near end
                indices = sorted(
                    set(
                        [
                            0,
                            max(0, frame_count // 2),
                            max(0, frame_count - 1),
                        ]
                    )
                )

            hashes: List[int] = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    hashes.append(-1)
                    continue
                h = VideoDuplicateFinder._compute_frame_phash(frame)
                hashes.append(h if h is not None else -1)

            # If all failed, consider the video unusable for hashing
            if all(h == -1 for h in hashes):
                return None

            # Ensure we always have exactly 3 entries
            while len(hashes) < 3:
                hashes.append(-1)

            return (hashes[0], hashes[1], hashes[2])
        finally:
            cap.release()

    @staticmethod
    def find_video_duplicates(
        video_paths: Iterable[str],
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> List[List[DuplicateRecord]]:
        """
        Scan the given video files and return groups of duplicates.

        Args:
            video_paths: iterable of filesystem paths to video files.
            progress_cb: optional function receiving an int percentage [0..100].

        Returns:
            A list of groups; each group is a list of DuplicateRecord objects.
            Only groups with more than one file are returned.
        """
        paths = [str(p) for p in video_paths]
        total = len(paths)
        if total == 0:
            return []

        # Map from 3-tuple signature -> list[DuplicateRecord]
        clusters: Dict[Tuple[int, int, int], List[DuplicateRecord]] = {}

        for idx, p in enumerate(paths):
            sig = VideoDuplicateFinder._get_signature_for_video(Path(p))
            if sig is None:
                # Skip unreadable videos
                if progress_cb is not None:
                    pct = int(((idx + 1) / total) * 100)
                    progress_cb(pct)
                continue

            clusters.setdefault(sig, []).append(
                DuplicateRecord(path=p, phash=sig[1] if sig[1] != -1 else sig[0])
            )

            if progress_cb is not None:
                pct = int(((idx + 1) / total) * 100)
                progress_cb(pct)

        # Keep only signatures that have more than one file
        groups: List[List[DuplicateRecord]] = [
            recs for recs in clusters.values() if len(recs) > 1
        ]
        return groups
