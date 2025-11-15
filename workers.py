#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Dict, Tuple

from PyQt6.QtCore import QObject, pyqtSignal, QRunnable

from common import (
    Image,
    ImageEnhance,
    _CV2,
    _NP,
    _MEDIAINFO,
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_LIVE_EXTS,
    _sanitize_exif_datetime,
    pil_to_qimage,
)
from geometry import _apply_geometry_perspective
from filters import _apply_filter_pipeline


class DirScanSignals(QObject):
    started = pyqtSignal(int, str)
    found_image = pyqtSignal(int, str, str)
    found_mov = pyqtSignal(int, str, str)
    finished = pyqtSignal(int, list, list)
    error = pyqtSignal(int, str)


class DirScanJob(QRunnable):
    def __init__(self, job_id: int, folder: Path, owner):
        super().__init__()
        self.job_id = job_id
        self.folder = folder
        self.signals = DirScanSignals()
        self.owner = owner  # ImageEditorApp (for _image_date_taken / _mov_date_taken)

    def run(self):
        try:
            self.signals.started.emit(self.job_id, str(self.folder))
            if not self.folder.is_dir():
                self.signals.error.emit(
                    self.job_id, f"Not a directory: {self.folder}"
                )
                return

            images: List[str] = []
            movs: List[str] = []

            for p in sorted(
                self.folder.iterdir(),
                key=lambda x: str(x).lower(),
            ):
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTS:
                    taken = self.owner._image_date_taken(p)
                    images.append(str(p))
                    self.signals.found_image.emit(
                        self.job_id, str(p), taken or "-"
                    )
                elif ext in SUPPORTED_LIVE_EXTS:
                    taken = self.owner._mov_date_taken(p)
                    movs.append(str(p))
                    self.signals.found_mov.emit(
                        self.job_id, str(p), taken or "-"
                    )

            self.signals.finished.emit(self.job_id, images, movs)
        except Exception as e:
            self.signals.error.emit(self.job_id, f"{e}")


class _PreviewSignals(QObject):
    done = pyqtSignal(int, object, object)  # job_id, main_qimage, mirror_qimage


class _PreviewJob(QRunnable):
    def __init__(
        self,
        job_id: int,
        base_image: Image.Image,
        coarse_rotation_degrees: int,
        factors: Dict[str, float],
        single_target_size: Tuple[int, int],
        do_mirror: bool,
        interactive: bool,
        geom_rx: float,
        geom_ry: float,
        geom_rz: float,
        preset_name: str,
        preset_strength: float,
        fill_mode: str,
        fast_geometry_preview: bool = False,
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
        self.fast_geometry_preview = fast_geometry_preview
        self.signals = _PreviewSignals()

    def run(self):
        img = self.image

        geo_img = _apply_geometry_perspective(
            img,
            self.geom_rx,
            self.geom_ry,
            self.geom_rz,
            self.coarse_rotation_degrees,
            self.fill_mode,
            preview_fast=self.fast_geometry_preview,
        )

        if self.fast_geometry_preview:
            target_w, target_h = self.target_size
            target_w = min(target_w, 480)
            target_h = min(target_h, 480)
            if target_w <= 0 or target_h <= 0:
                target_w = max(1, geo_img.size[0])
                target_h = max(1, geo_img.size[1])
            gw, gh = geo_img.size
            ratio = min(target_w / float(gw), target_h / float(gh))
            new_size = (
                max(1, int(gw * ratio)),
                max(1, int(gh * ratio)),
            )
            preview = geo_img.resize(
                new_size, Image.Resampling.BILINEAR
            )

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

        target_w, target_h = self.target_size
        if self.interactive:
            target_w = min(target_w, 720)
            target_h = min(target_h, 720)
        if target_w <= 0 or target_h <= 0:
            target_w = max(1, geo_img.size[0])
            target_h = max(1, geo_img.size[1])

        gw, gh = geo_img.size
        ratio = min(target_w / float(gw), target_h / float(gh))
        new_size = (
            max(1, int(gw * ratio)),
            max(1, int(gh * ratio)),
        )
        preview = geo_img.resize(new_size, Image.Resampling.BILINEAR)

        if (
            self.preset_name
            and self.preset_name not in ("None", "—", "Original")
        ):
            preview = _apply_filter_pipeline(
                preview,
                self.preset_name,
                max(0.0, min(1.0, self.preset_strength)),
            )

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

    def _apply_edits(
        self,
        pil_image: Image.Image,
        f: Dict[str, float],
    ) -> Image.Image:
        img = (
            pil_image.convert("RGB")
            if pil_image.mode != "RGB"
            else pil_image
        )
        working_img = img.copy()

        exposure_adj = 1.0 + (f["exposure"] / 2.0)
        working_img = ImageEnhance.Brightness(working_img).enhance(
            exposure_adj
        )

        brightness_adj = 1.0 + (f["brightness"] / 100.0)
        working_img = ImageEnhance.Brightness(working_img).enhance(
            brightness_adj
        )

        contrast_adj = 1.0 + (f["contrast"] / 100.0)
        working_img = ImageEnhance.Contrast(working_img).enhance(
            contrast_adj
        )

        if _NP is not None:
            img_np = _NP.array(working_img).astype(_NP.float32) / 255.0
            if f["highlights"] != 0:
                gamma_h = 1.0 - (f["highlights"] / 200.0)
                img_np = _NP.where(
                    img_np > 0.5,
                    _NP.power(img_np, gamma_h),
                    img_np,
                )
            if f["shadows"] != 0:
                gamma_s = 1.0 + (f["shadows"] / 200.0)
                img_np = _NP.where(
                    img_np < 0.5,
                    _NP.power(img_np, gamma_s),
                    img_np,
                )
            if f["blackpoint"] != 0:
                offset = f["blackpoint"] / 500.0
                img_np = _NP.clip(img_np + offset, 0.0, 1.0)
            working_img = Image.fromarray(
                (_NP.clip(img_np, 0.0, 1.0) * 255).astype(_NP.uint8)
            )

        if _CV2 is not None and _NP is not None:
            brill = float(f["brilliance"])
            if abs(brill) > 0.01:
                img_rgb = _NP.array(working_img)
                lab = _CV2.cvtColor(img_rgb, _CV2.COLOR_RGB2LAB)
                L, a, b = _CV2.split(lab)
                if brill >= 0:
                    tiles = 4 if self.interactive else 8
                    clip = 1.0 + (brill / 100.0) * (
                        4.0 if self.interactive else 8.0
                    )
                    clip = max(1.01, min(clip, 9.0))
                    clahe = _CV2.createCLAHE(
                        clipLimit=clip, tileGridSize=(tiles, tiles)
                    )
                    L2 = clahe.apply(L)
                else:
                    k = int(
                        round(
                            3
                            + (abs(brill) / 100.0)
                            * (6 if self.interactive else 12)
                        )
                    )
                    if k % 2 == 0:
                        k += 1
                    L_blur = _CV2.GaussianBlur(L, (k, k), 0)
                    alpha = (0.4 if self.interactive else 0.8) * (
                        abs(brill) / 100.0
                    )
                    alpha = max(0.0, min(alpha, 0.9))
                    L2 = _CV2.addWeighted(
                        L,
                        1.0 - alpha,
                        L_blur,
                        alpha,
                        0,
                    )
                lab2 = _CV2.merge((L2, a, b))
                rgb2 = _CV2.cvtColor(lab2, _CV2.COLOR_LAB2RGB)
                working_img = Image.fromarray(rgb2)

        sat_adj = 1.0 + (f["saturation"] / 100.0)
        working_img = ImageEnhance.Color(working_img).enhance(sat_adj)

        if _NP is not None:
            img_rgb_np = _NP.array(working_img).astype(_NP.float32)
            if f["vibrance"] != 0:
                hsv = _NP.array(
                    working_img.convert("HSV")
                ).astype(_NP.float32)
                v_adj = f["vibrance"] / 100.0
                saturation = hsv[:, :, 1] / 255.0
                mask = 1.0 - saturation
                hsv[:, :, 1] = _NP.clip(
                    hsv[:, :, 1] + (hsv[:, :, 1] * v_adj * mask),
                    0,
                    255,
                )
                working_img = Image.fromarray(
                    hsv.astype(_NP.uint8),
                    "HSV",
                ).convert("RGB")
                img_rgb_np = _NP.array(working_img).astype(_NP.float32)
            if f["warmth"] != 0:
                r_gain = 1.0 + (f["warmth"] / 150.0)
                b_gain = 1.0 - (f["warmth"] / 150.0)
                img_rgb_np[:, :, 0] = _NP.clip(
                    img_rgb_np[:, :, 0] * r_gain, 0, 255
                )
                img_rgb_np[:, :, 2] = _NP.clip(
                    img_rgb_np[:, :, 2] * b_gain, 0, 255
                )
            if f["tint"] != 0:
                g_gain = 1.0 + (f["tint"] / 150.0)
                img_rgb_np[:, :, 1] = _NP.clip(
                    img_rgb_np[:, :, 1] * g_gain, 0, 255
                )
            working_img = Image.fromarray(img_rgb_np.astype(_NP.uint8))

        if (
            _CV2 is not None
            and _NP is not None
            and f["noise_reduction"] > 0.0
        ):
            nr = float(f["noise_reduction"])
            img_bgr = _CV2.cvtColor(
                _NP.array(working_img), _CV2.COLOR_RGB2BGR
            )
            if self.interactive:
                d = 5
                sigma = 5 + int(nr * 5)
                den = _CV2.bilateralFilter(img_bgr, d, sigma, sigma)
            else:
                h = 3 + int(nr * 7)
                den = _CV2.fastNlMeansDenoisingColored(
                    img_bgr, None, h, h, 7, 21
                )
            working_img = Image.fromarray(
                _CV2.cvtColor(den, _CV2.COLOR_BGR2RGB)
            )

        sharp_adj = 1.0 + (
            (f["sharpness"] / 100.0)
            * (0.5 if self.interactive else 1.0)
        )
        working_img = ImageEnhance.Sharpness(working_img).enhance(
            sharp_adj
        )

        if f["vignette"] > 0.1 and _NP is not None:
            image = working_img
            if image.mode != "RGB":
                image = image.convert("RGB")
            w, h = image.size
            y_coords, x_coords = _NP.indices((h, w))
            cx, cy = w / 2.0, h / 2.0
            radius = min(w, h) / 2.0
            dist = _NP.hypot(
                x_coords - cx,
                y_coords - cy,
            ) / radius
            falloff = _NP.power(
                dist,
                (1.0 + (f["vignette"] * 0.5)),
            )
            mask = _NP.clip(1.0 - (falloff * 0.6), 0.0, 1.0)
            image_np = _NP.array(image).astype(_NP.float32) / 255.0
            final_np = _NP.clip(
                image_np * mask[:, :, _NP.newaxis],
                0.0,
                1.0,
            )
            working_img = Image.fromarray(
                (final_np * 255).astype(_NP.uint8)
            )

        return working_img
