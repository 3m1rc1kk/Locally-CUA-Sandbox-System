# vision.py
from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from src.config import IMAGE_MIME, cfg

if TYPE_CHECKING:
    from src.sandbox import Sandbox


def image_to_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = IMAGE_MIME.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def resize_keep_aspect(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img
    if w >= h:
        new_w = max_dim
        new_h = int(h * max_dim / w)
    else:
        new_h = max_dim
        new_w = int(w * max_dim / h)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def capture_screen(sandbox, save_path: str) -> Image.Image:
    """Capture screenshot for LLM: resized to MAX_DIM and saved to disk."""
    img = sandbox.screenshot().convert("RGB")
    img = resize_keep_aspect(img, cfg.MAX_DIM)
    img.save(save_path)
    return img


def capture_screen_raw(sandbox) -> Image.Image:
    """For the GUI: return raw image without touching resolution."""
    return sandbox.screenshot().convert("RGB")


def draw_preview(img: Image.Image, x: float, y: float, out_path: str, r: int = 10) -> None:
    cp = img.copy().convert("RGB")
    w, h = cp.size
    px = int(max(0.0, min(1.0, x)) * max(0, w - 1))
    py = int(max(0.0, min(1.0, y)) * max(0, h - 1))
    d = ImageDraw.Draw(cp)
    d.ellipse((px - r, py - r, px + r, py + r), fill="red", outline="white", width=2)
    cp.save(out_path)
    print(f"[PREVIEW] {out_path} (x={x:.4f}, y={y:.4f})")
