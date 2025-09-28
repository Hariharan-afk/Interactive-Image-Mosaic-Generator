# File: mosaic_core.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.filters import sobel_h, sobel_v
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# ------------------------------- I/O utils ---------------------------------


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as uint8 HxWx3 RGB."""
    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image not found: {path}") from e
    except Exception as e:
        raise ValueError(f"Failed to load image: {path}") from e
    return np.asarray(img, dtype=np.uint8)


def save_image(arr: np.ndarray, path: str | Path) -> None:
    """Save an RGB array to disk as PNG/JPEG (format inferred from extension)."""
    try:
        Image.fromarray(arr.astype(np.uint8)).save(path)
    except Exception as e:
        raise ValueError(f"Failed to save image: {path}") from e


# ----------------------- Resize / crop / quantize ---------------------------


def resize_and_crop_to_grid(img: np.ndarray, cell: int, grid: Tuple[int, int]) -> np.ndarray:
    """
    Resize (preserving aspect) and center-crop so final size == (grid_y*cell, grid_x*cell).
    Ensures the result tiles cleanly into the given grid.
    """
    gy, gx = grid
    target_h, target_w = gy * cell, gx * cell

    pil = Image.fromarray(img)
    scale = max(target_w / pil.width, target_h / pil.height)
    new_w, new_h = int(np.ceil(pil.width * scale)), int(np.ceil(pil.height * scale))
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    pil = pil.crop((left, top, left + target_w, top + target_h))
    return np.asarray(pil, dtype=np.uint8)


def quantize_colors_pillow(img: np.ndarray, k: int = 16) -> np.ndarray:
    """Palette reduction with Pillow to k colors (k<=0 returns input)."""
    if k <= 0:
        return img
    q = Image.fromarray(img).quantize(colors=k, method=Image.Quantize.FASTOCTREE).convert("RGB")
    return np.asarray(q, dtype=np.uint8)


# ------------------------------ Grid overlay -------------------------------


def draw_grid(img: np.ndarray, cell: int, color: tuple[int, int, int] = (255, 255, 255),
              alpha: float = 0.35) -> np.ndarray:
    """Overlay faint grid lines every `cell` pixels."""
    out = img.astype(np.float32).copy()
    H, W = out.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    mask[::cell, :] = True
    mask[:, ::cell] = True
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------- Feature extraction ---------------------------


def _gradients(gray_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gx, gy, magnitude) Sobel derivatives for a float32 grayscale image."""
    gy = sobel_h(gray_f32)
    gx = sobel_v(gray_f32)
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def cell_lab_means(img: np.ndarray, cell: int) -> np.ndarray:
    """Per-cell Lab mean, shape (gy, gx, 3)."""
    H, W = img.shape[:2]
    gy, gx = H // cell, W // cell
    lab = rgb2lab(img / 255.0)
    blocks = lab.reshape(gy, cell, gx, cell, 3).swapaxes(1, 2)  # (gy, gx, cell, cell, 3)
    mu = blocks.mean(axis=(2, 3))
    return mu.astype(np.float32)


def cell_texture_scalar(img: np.ndarray, cell: int) -> np.ndarray:
    """Per-cell mean Sobel magnitude over L channel, shape (gy, gx)."""
    H, W = img.shape[:2]
    gy, gx = H // cell, W // cell
    L = rgb2lab(img / 255.0)[..., 0].astype(np.float32)
    _, _, mag = _gradients(L)
    blocks = mag.reshape(gy, cell, gx, cell).swapaxes(1, 2)  # (gy, gx, cell, cell)
    t = blocks.mean(axis=(2, 3))
    return t.astype(np.float32)


# ----------------------------- Reconstruction ------------------------------


def reconstruct_mosaic(tile_idx: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    """Assemble mosaic using `tiles` indexed by `tile_idx` (gy,gx)."""
    gy, gx = tile_idx.shape
    s = tiles.shape[1]
    out = np.zeros((gy * s, gx * s, 3), dtype=np.uint8)
    for iy in range(gy):
        for ix in range(gx):
            out[iy * s:(iy + 1) * s, ix * s:(ix + 1) * s] = tiles[tile_idx[iy, ix]]
    return out


# --------------------------------- Metrics ---------------------------------


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error over RGB."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def psnr_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR over RGB."""
    return float(psnr(a, b, data_range=255))


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM over RGB (sliding window)."""
    return float(ssim(a, b, channel_axis=2, data_range=255))


def ssim_blockwise(proc: np.ndarray, mosaic: np.ndarray, cell: int) -> float:
    """
    SSIM computed per cell (window ~ cell size) and averaged.
    Falls back to grayscale if RGB windowing fails (e.g., tiny tiles).
    """
    H, W = proc.shape[:2]
    gy, gx = H // cell, W // cell
    scores: list[float] = []

    # choose an odd window size <= cell
    k = min(cell, 11)
    if k % 2 == 0:
        k -= 1
    k = max(3, k)

    for iy in range(gy):
        for ix in range(gx):
            a = proc[iy * cell:(iy + 1) * cell, ix * cell:(ix + 1) * cell]
            b = mosaic[iy * cell:(iy + 1) * cell, ix * cell:(ix + 1) * cell]
            try:
                s = ssim(a, b, channel_axis=2, data_range=255, win_size=k)
            except Exception:
                ag = a.mean(axis=2)
                bg = b.mean(axis=2)
                kk = min(k, min(ag.shape))
                if kk % 2 == 0:
                    kk -= 1
                kk = max(3, kk)
                s = ssim(ag, bg, data_range=255, win_size=kk)
            scores.append(float(s))
    return float(np.mean(scores)) if scores else 0.0


def ssim_per_channel(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """SSIM for R/G/B channels separately, plus their mean."""
    rs = float(ssim(a[..., 0], b[..., 0], data_range=255))
    gs = float(ssim(a[..., 1], b[..., 1], data_range=255))
    bs = float(ssim(a[..., 2], b[..., 2], data_range=255))
    return rs, gs, bs, float((rs + gs + bs) / 3.0)


def ssim_global_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Single-window SSIM on grayscale (whole image)."""
    ag = a.mean(axis=2)
    bg = b.mean(axis=2)
    win = min(ag.shape)
    if win % 2 == 0:
        win -= 1
    win = max(3, win)
    return float(ssim(ag, bg, data_range=255, win_size=win))


def ssim_multiscale_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Multi-scale grayscale SSIM (1.0, 0.5, 0.25 scales)."""
    def to_gray(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=2).astype(np.float32)

    ag, bg = to_gray(a), to_gray(b)
    scores: list[float] = []

    for scale in (1.0, 0.5, 0.25):
        if scale == 1.0:
            Ag, Bg = ag, bg
        else:
            new_h = max(2, int(round(ag.shape[0] * scale)))
            new_w = max(2, int(round(ag.shape[1] * scale)))
            Ag = np.array(Image.fromarray(ag).resize((new_w, new_h), Image.Resampling.BICUBIC))
            Bg = np.array(Image.fromarray(bg).resize((new_w, new_h), Image.Resampling.BICUBIC))

        win = min(Ag.shape)
        if win % 2 == 0:
            win -= 1
        win = max(3, win)
        scores.append(float(ssim(Ag, Bg, data_range=255, win_size=win)))

    return float(np.mean(scores)) if scores else 0.0


# ------------------------ Custom mapping (color+texture) --------------------


def texture_aware_mapping(
    block_avgs: np.ndarray,
    block_textures: np.ndarray,
    tile_means: np.ndarray,
    tile_textures: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Combine color (Lab means) and texture (Sobel magnitude) distances.
    Returns int indices shaped (gy, gx).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    gy, gx, _ = block_avgs.shape
    Bc = block_avgs.reshape(-1, 3)            # (B, 3)
    Bt = block_textures.reshape(-1)           # (B,)

    color_d = ((Bc[:, None, :] - tile_means[None, :, :]) ** 2).sum(axis=2)      # (B, T)
    text_d = (Bt[:, None] - tile_textures[None, :]) ** 2                        # (B, T)

    # normalize to [0,1] to balance terms
    color_d = color_d / (color_d.max() + 1e-8)
    text_d = text_d / (text_d.max() + 1e-8)

    combined = alpha * color_d + (1.0 - alpha) * text_d
    idx = combined.argmin(axis=1).reshape(gy, gx)
    return idx.astype(np.int32)
