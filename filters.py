#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from common import Image, ImageEnhance, _CV2, _NP


IPHONE_FILTERS: List[str] = [
    "None",
    "Vivid",
    "Vivid Warm",
    "Vivid Cool",
    "Dramatic",
    "Dramatic Warm",
    "Dramatic Cool",
    "Mono",
    "Silvertone",
    "Noir",
]

CAMSCANNER_FILTERS: List[str] = [
    "Original",
    "Magic Color",
    "Magic Pro",
    "Grayscale",
    "B&W",
    "No Shadow",
    "Soft Tone",
    "OCV Black",
]

ALL_FILTERS = IPHONE_FILTERS + ["—"] + CAMSCANNER_FILTERS


def _blend(a: Image.Image, b: Image.Image, alpha: float) -> Image.Image:
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0:
        return a
    if alpha >= 1:
        return b
    return Image.blend(a.convert("RGB"), b.convert("RGB"), alpha)


def _apply_filter_iphone(img: Image.Image, name: str) -> Image.Image:
    im = img.convert("RGB")

    def adj_contrast(x, pct):
        return ImageEnhance.Contrast(x).enhance(1.0 + pct)

    def adj_brightness(x, pct):
        return ImageEnhance.Brightness(x).enhance(1.0 + pct)

    def adj_color(x, pct):
        return ImageEnhance.Color(x).enhance(1.0 + pct)

    if name == "Vivid":
        im = adj_contrast(im, 0.15)
        im = adj_color(im, 0.25)
    elif name == "Vivid Warm":
        im = _apply_filter_iphone(im, "Vivid")
        if _NP is not None:
            npimg = _NP.array(im).astype(_NP.float32)
            npimg[:, :, 0] = _NP.clip(npimg[:, :, 0] * 1.06, 0, 255)
            npimg[:, :, 2] = _NP.clip(npimg[:, :, 2] * 0.94, 0, 255)
            im = Image.fromarray(npimg.astype(_NP.uint8))
    elif name == "Vivid Cool":
        im = _apply_filter_iphone(im, "Vivid")
        if _NP is not None:
            npimg = _NP.array(im).astype(_NP.float32)
            npimg[:, :, 2] = _NP.clip(npimg[:, :, 2] * 1.06, 0, 255)
            npimg[:, :, 0] = _NP.clip(npimg[:, :, 0] * 0.94, 0, 255)
            im = Image.fromarray(npimg.astype(_NP.uint8))
    elif name == "Dramatic":
        im = ImageEnhance.Contrast(im).enhance(1.2)
        if _NP is not None:
            npimg = _NP.array(im).astype(_NP.float32) / 255.0
            npimg = _NP.where(npimg > 0.7, npimg ** 1.25, npimg)
            npimg = _NP.where(npimg < 0.3, _NP.sqrt(npimg), npimg)
            im = Image.fromarray((_NP.clip(npimg, 0, 1) * 255).astype(_NP.uint8))
    elif name == "Dramatic Warm":
        im = _apply_filter_iphone(im, "Dramatic")
        if _NP is not None:
            npimg = _NP.array(im).astype(_NP.float32)
            npimg[:, :, 0] = _NP.clip(npimg[:, :, 0] * 1.06, 0, 255)
            im = Image.fromarray(npimg.astype(_NP.uint8))
    elif name == "Dramatic Cool":
        im = _apply_filter_iphone(im, "Dramatic")
        if _NP is not None:
            npimg = _NP.array(im).astype(_NP.float32)
            npimg[:, :, 2] = _NP.clip(npimg[:, :, 2] * 1.06, 0, 255)
            im = Image.fromarray(npimg.astype(_NP.uint8))
    elif name == "Mono":
        im = im.convert("L").convert("RGB")
    elif name == "Silvertone":
        im = ImageEnhance.Color(im).enhance(0.0)
        im = ImageEnhance.Contrast(im).enhance(1.15)
        im = ImageEnhance.Brightness(im).enhance(0.95)
    elif name == "Noir":
        im = ImageEnhance.Color(im).enhance(0.0)
        im = ImageEnhance.Contrast(im).enhance(1.35)
    return im


def _apply_filter_camscanner(img: Image.Image, name: str) -> Image.Image:
    im = img.convert("RGB")
    if name == "Original":
        return im

    if _CV2 is None or _NP is None:
        if name == "Grayscale":
            return im.convert("L").convert("RGB")
        if name == "B&W":
            g = im.convert("L")
            return g.point(lambda v: 255 if v > 140 else 0).convert("RGB")
        if name == "Soft Tone":
            return ImageEnhance.Sharpness(im).enhance(0.9)
        if name in ("Magic Color", "Magic Pro", "No Shadow", "OCV Black"):
            return ImageEnhance.Contrast(im).enhance(1.1)
        return im

    npimg = _NP.array(im)

    if name in ("Magic Color", "Magic Pro"):
        b, g, r = _CV2.split(_CV2.cvtColor(npimg, _CV2.COLOR_RGB2BGR))
        avg_b, avg_g, avg_r = b.mean(), g.mean(), r.mean()
        kb = (avg_g + avg_r) / (2 * avg_b + 1e-6)
        kg = (avg_b + avg_r) / (2 * avg_g + 1e-6)
        kr = (avg_b + avg_g) / (2 * avg_r + 1e-6)
        b = _NP.clip(b * kb, 0, 255).astype(_NP.uint8)
        g = _NP.clip(g * kg, 0, 255).astype(_NP.uint8)
        r = _NP.clip(r * kr, 0, 255).astype(_NP.uint8)
        wb = _CV2.cvtColor(_CV2.merge([b, g, r]), _CV2.COLOR_BGR2RGB)

        lab = _CV2.cvtColor(wb, _CV2.COLOR_RGB2LAB)
        L, A, B = _CV2.split(lab)
        clip = 3.0 if name == "Magic Color" else 4.0
        clahe = _CV2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        L2 = clahe.apply(L)
        lab2 = _CV2.merge((L2, A, B))
        rgb = _CV2.cvtColor(lab2, _CV2.COLOR_LAB2RGB)

        blur = _CV2.GaussianBlur(rgb, (0, 0), 1.0 if name == "Magic Color" else 1.4)
        sharpen = _CV2.addWeighted(rgb, 1.25, blur, -0.25, 0)
        return Image.fromarray(sharpen)

    if name == "Grayscale":
        g = _CV2.cvtColor(npimg, _CV2.COLOR_RGB2GRAY)
        return Image.fromarray(_CV2.cvtColor(g, _CV2.COLOR_GRAY2RGB))

    if name == "B&W":
        g = _CV2.cvtColor(npimg, _CV2.COLOR_RGB2GRAY)
        bw = _CV2.adaptiveThreshold(
            g,
            255,
            _CV2.ADAPTIVE_THRESH_GAUSSIAN_C,
            _CV2.THRESH_BINARY,
            31,
            10,
        )
        return Image.fromarray(_CV2.cvtColor(bw, _CV2.COLOR_GRAY2RGB))

    if name == "No Shadow":
        g = _CV2.cvtColor(npimg, _CV2.COLOR_RGB2GRAY)
        bg = _CV2.medianBlur(g, 31)
        bg = _NP.where(bg < 1, 1, bg)
        norm = (g.astype(_NP.float32) / bg.astype(_NP.float32)) * 128.0
        norm = _NP.clip(norm, 0, 255).astype(_NP.uint8)
        norm = _CV2.equalizeHist(norm)
        return Image.fromarray(_CV2.cvtColor(norm, _CV2.COLOR_GRAY2RGB))

    if name == "Soft Tone":
        blur = _CV2.GaussianBlur(npimg, (0, 0), 3.0)
        soft = _CV2.addWeighted(npimg, 0.75, blur, 0.25, 0)
        return Image.fromarray(soft)

    if name == "OCV Black":
        g = _CV2.cvtColor(npimg, _CV2.COLOR_RGB2GRAY)
        _, th = _CV2.threshold(
            g, 0, 255, _CV2.THRESH_BINARY + _CV2.THRESH_OTSU
        )
        if th.mean() < 127:
            th = _CV2.bitwise_not(th)
        return Image.fromarray(_CV2.cvtColor(th, _CV2.COLOR_GRAY2RGB))

    return im


def _apply_filter_pipeline(
    img: Image.Image,
    preset: str,
    strength_0_to_1: float,
) -> Image.Image:
    if not preset or preset in ("None", "—", "Original"):
        return img
    if preset in IPHONE_FILTERS:
        out = _apply_filter_iphone(img, preset)
        return _blend(img, out, strength_0_to_1)
    if preset in CAMSCANNER_FILTERS:
        out = _apply_filter_camscanner(img, preset)
        return _blend(img, out, strength_0_to_1)
    return img
