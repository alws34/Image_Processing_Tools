#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Callable

from common import Image, HEIF_LIKE_EXTS, _HEIF_PLUGIN


@dataclass
class DuplicateRecord:
    path: str
    phash: int


class DuplicateFinder:
    @staticmethod
    def compute_phash(path: str) -> Optional[int]:
        """
        Compute a simple 64-bit average hash for the image at `path`.
        Returns None on failure (I/O error, unsupported format, etc.).
        """
        suffix = Path(path).suffix.lower()
        # Respect HEIF plugin availability
        if suffix in HEIF_LIKE_EXTS and not _HEIF_PLUGIN:
            return None

        try:
            with Image.open(path) as im:
                im = im.convert("L")  # grayscale
                im = im.resize((8, 8))  # 64 pixels
                pixels = list(im.getdata())
        except Exception:
            return None

        if not pixels:
            return None

        avg = sum(pixels) / float(len(pixels))
        bits = 0
        for p in pixels:
            bits = (bits << 1) | (1 if p > avg else 0)

        return bits

    @staticmethod
    def find_duplicates(
        image_paths: List[str],
        min_group_size: int = 2,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[DuplicateRecord]]:
        """
        Group images by identical pHash.
        Returns a list of groups, each group is a list[DuplicateRecord].
        """
        hash_map: Dict[int, List[DuplicateRecord]] = {}
        total = len(image_paths)

        for idx, path in enumerate(image_paths):
            h = DuplicateFinder.compute_phash(path)
            if h is None:
                continue
            hash_map.setdefault(h, []).append(DuplicateRecord(path, h))

            # Optional progress callback every 50 images
            if progress_cb and (idx + 1) % 50 == 0:
                progress_cb(idx + 1, total)

        groups = [grp for grp in hash_map.values() if len(grp) >= min_group_size]
        return groups
