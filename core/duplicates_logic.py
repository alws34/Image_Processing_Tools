# core/duplicates_logic.py
import os
from dataclasses import dataclass
from core.common import Image, _CV2, _HEIF_PLUGIN, HEIF_LIKE_EXTS


@dataclass
class DuplicateRecord:
    path: str
    group_id: int = 0


def compute_image_hash(path: str) -> int:
    """Returns a simple visual 64-bit integer hash."""
    try:
        with Image.open(path) as im:
            # Resize to 9x8 for dHash (difference hash)
            im = im.convert("L").resize((9, 8), Image.Resampling.NEAREST)
            pixels = list(im.getdata())
            # Compare pixel i with pixel i+1
            diff = 0
            for i in range(8):
                for j in range(8):
                    if pixels[i*9 + j] > pixels[i*9 + j + 1]:
                        diff |= 1 << (i*8 + j)
            return diff
    except:
        return 0


def compute_video_hash(path: str) -> tuple:
    """Returns (duration_int, visual_hash_int)."""
    if _CV2 is None:
        return (0, 0)
    try:
        cap = _CV2.VideoCapture(path)
        if not cap.isOpened():
            return (0, 0)

        frames = int(cap.get(_CV2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(_CV2.CAP_PROP_FPS)
        dur = int(frames / fps) if fps > 0 else 0

        # Hash middle frame
        cap.set(_CV2.CAP_PROP_POS_FRAMES, frames // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return (dur, 0)

        gray = _CV2.cvtColor(frame, _CV2.COLOR_BGR2GRAY)
        small = _CV2.resize(gray, (9, 8))
        diff = 0
        flat = small.flatten()
        # Simple dHash logic on numpy array
        for i in range(64):
            if flat[i] > flat[i+1]:  # Note: simplified index logic
                diff |= 1 << i

        return (dur, diff)
    except:
        return (0, 0)
