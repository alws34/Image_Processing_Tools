#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple

from core.common import (
    Image,
    _CV2,
    _NP,
    FILL_KEEP,
    FILL_AUTOCROP,
    FILL_STRETCH,
)


def _compose_homography_from_euler(
    w: int, h: int, rx_deg: float, ry_deg: float, rz_deg: float
):
    if _NP is None:
        return None
    cx, cy = w / 2.0, h / 2.0
    f = max(w, h)
    rx = _NP.deg2rad(rx_deg)
    ry = _NP.deg2rad(ry_deg)
    rz = _NP.deg2rad(rz_deg)

    Rx = _NP.array(
        [
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)],
        ],
        dtype=_NP.float32,
    )
    Ry = _NP.array(
        [
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)],
        ],
        dtype=_NP.float32,
    )
    Rz = _NP.array(
        [
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1],
        ],
        dtype=_NP.float32,
    )

    R = Rz @ (Ry @ Rx)

    K = _NP.array(
        [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ],
        dtype=_NP.float32,
    )
    Kinv = _NP.linalg.inv(K)
    H = K @ R @ Kinv
    return H.astype(_NP.float32)


def _auto_crop_bounds_from_mask(mask: "object") -> Optional[Tuple[int, int, int, int]]:
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
    rx: float,
    ry: float,
    rz: float,
    coarse_roll_deg: float,
    fill_mode: str,
    preview_fast: bool = False,
) -> Image.Image:
    """
    preview_fast:
      - True: downscale first and warp on a smaller grid (for interactive preview).
      - False: full-quality path.
    """
    total_rz = rz + coarse_roll_deg

    if _CV2 is None or _NP is None:
        base = pil_img.rotate(
            total_rz,
            expand=True,
            resample=Image.Resampling.BICUBIC,
        )
        base = base.resize(pil_img.size, Image.Resampling.LANCZOS)
        return base

    base_rgb = pil_img.convert("RGB")
    img_np = _NP.array(base_rgb)
    h, w = img_np.shape[:2]

    # Fast path for slider-dragging
    if preview_fast:
        target_long = 720
        long_side = max(w, h)
        if long_side > target_long:
            scale = target_long / float(long_side)
            fast_w = max(1, int(w * scale))
            fast_h = max(1, int(h * scale))
            img_np_fast = _CV2.resize(
                img_np, (fast_w, fast_h), interpolation=_CV2.INTER_AREA
            )
            w_fast, h_fast = fast_w, fast_h
        else:
            img_np_fast = img_np
            w_fast, h_fast = w, h

        H = _compose_homography_from_euler(w_fast, h_fast, rx, ry, total_rz)
        if H is None:
            return Image.fromarray(img_np_fast)

        if fill_mode == FILL_KEEP:
            warped = _CV2.warpPerspective(
                img_np_fast,
                H,
                (w_fast, h_fast),
                flags=_CV2.INTER_LINEAR,
                borderMode=_CV2.BORDER_REPLICATE,
            )
            return Image.fromarray(warped)

        ones = _NP.full((h_fast, w_fast), 255, dtype=_NP.uint8)
        mask = _CV2.warpPerspective(
            ones,
            H,
            (w_fast, h_fast),
            flags=_CV2.INTER_NEAREST,
            borderMode=_CV2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped = _CV2.warpPerspective(
            img_np_fast,
            H,
            (w_fast, h_fast),
            flags=_CV2.INTER_LINEAR,
            borderMode=_CV2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
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
            stretched = _CV2.resize(
                roi, (w_fast, h_fast), interpolation=_CV2.INTER_CUBIC
            )
            return Image.fromarray(stretched)
        return Image.fromarray(warped)

    # Full-quality path
    H = _compose_homography_from_euler(w, h, rx, ry, total_rz)
    if H is None:
        return base_rgb

    if fill_mode == FILL_KEEP:
        warped = _CV2.warpPerspective(
            img_np,
            H,
            (w, h),
            flags=_CV2.INTER_LINEAR,
            borderMode=_CV2.BORDER_REPLICATE,
        )
        return Image.fromarray(warped)

    ones = _NP.full((h, w), 255, dtype=_NP.uint8)
    mask = _CV2.warpPerspective(
        ones,
        H,
        (w, h),
        flags=_CV2.INTER_NEAREST,
        borderMode=_CV2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped = _CV2.warpPerspective(
        img_np,
        H,
        (w, h),
        flags=_CV2.INTER_LINEAR,
        borderMode=_CV2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
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
