# File: app.py
from __future__ import annotations

import os
import time
import tempfile
import uuid
from typing import Tuple

import gradio as gr
import numpy as np
from PIL import Image

from mosaic_core import (
    resize_and_crop_to_grid,
    quantize_colors_pillow,
    draw_grid,
    cell_lab_means,
    cell_texture_scalar,
    reconstruct_mosaic,
    mse,
    psnr_rgb,
    ssim_rgb,
    ssim_blockwise,
    ssim_per_channel,
    ssim_global_gray,
    ssim_multiscale_gray,
    texture_aware_mapping,
)
from tile_cache import ensure_caches, load_tile_cache, build_tile_cache
from tile_cache import digest_tile_dir as ensure_digest


# ------------------------------- Config -------------------------------------

CACHE_DIR = "cache"
PRESET_GRID = ["16×16", "32×32", "64×64", "128×128"]
PRESET_TILE = ["4×4", "8×8", "16×16", "32×32"]
GRID_MAP = {"16×16": (16, 16), "32×32": (32, 32), "64×64": (64, 64), "128×128": (128, 128)}
TILE_MAP = {"4×4": 4, "8×8": 8, "16×16": 16, "32×32": 32}
METHODS = ["Only Color based", "Custom (Color+Texture)"]
DISPLAY_H = 512
SAMPLE_DIR = "samples"  # put sample images here (e.g., samples/beach.jpg)


# ------------------------------- Helpers ------------------------------------

def _error_md(msg: str) -> str:
    return f"**❌ Error:** {msg}"


def _resolve_cache(tile_dir: str, tile_size: int) -> dict:
    """Load cache for a tile_size; if missing or incompatible, build it."""
    key = {
        "tile_dir": os.path.abspath(tile_dir),
        "tile_size": int(tile_size),
        "angles": [0, 90, 180, 270],
        "flips": True,
        "digest": ensure_digest(tile_dir),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = load_tile_cache(CACHE_DIR, key)
    if cache is None:
        cache = build_tile_cache(tile_dir, tile_size, CACHE_DIR)
    return cache


# ------------------------------- Core fn ------------------------------------

def build_mosaic(
    img: np.ndarray | str,
    tile_dir: str,
    grid_label: str,
    tile_label: str,
    method: str,
    quant_k_label: str,
    alpha: float,
    _status_box: gr.Textbox,  # kept for signature alignment
):
    """Main pipeline used by the UI. Returns: processed preview, mosaic, metrics md, timing md, download path."""
    try:
        # normalize image
        if isinstance(img, str):
            img = np.asarray(Image.open(img).convert("RGB"), dtype=np.uint8)
        elif isinstance(img, np.ndarray):
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("Input array must be HxWx3 RGB.")
        else:
            raise ValueError("Unsupported image input; expected filepath or RGB array.")

        rows, cols = GRID_MAP[grid_label]
        cell = TILE_MAP[tile_label]
        quant_k = int(quant_k_label)
        alpha = float(alpha)

        t_all0 = time.perf_counter()

        # cache
        t0 = time.perf_counter()
        cache = _resolve_cache(tile_dir, cell)
        tiles = cache["tiles"]
        tile_means = cache["tile_means_lab_rot"]
        tile_textures = cache["tile_texture_rot"]
        t_cache = time.perf_counter() - t0

        # preprocess
        t1 = time.perf_counter()
        proc = resize_and_crop_to_grid(img, cell=cell, grid=(rows, cols))
        if quant_k > 0:
            proc = quantize_colors_pillow(proc, k=quant_k)
        proc_vis = draw_grid(proc, cell)
        t_prep = time.perf_counter() - t1

        # features
        t2 = time.perf_counter()
        mu_lab = cell_lab_means(proc, cell)
        tex = cell_texture_scalar(proc, cell)
        t_feat = time.perf_counter() - t2

        # matching
        t3 = time.perf_counter()
        if method == "Only Color based" or alpha >= 0.999:
            # z-scored L2 in Lab-mean space
            B = mu_lab.reshape(-1, 3)
            T = tile_means
            mu = T.mean(axis=0, keepdims=True)
            sd = T.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-6, 1.0, sd)
            Cn = (B - mu) / sd
            Tn = (T - mu) / sd
            d2 = ((Cn ** 2).sum(axis=1, keepdims=True)
                  + (Tn ** 2).sum(axis=1, keepdims=True).T
                  - 2.0 * (Cn @ Tn.T))
            idx = np.argmin(d2, axis=1).reshape(rows, cols)
        else:
            idx = texture_aware_mapping(mu_lab, tex, tile_means, tile_textures, alpha=alpha)
        t_match = time.perf_counter() - t3

        # reconstruct
        t4 = time.perf_counter()
        mosaic = reconstruct_mosaic(idx, tiles)
        t_recon = time.perf_counter() - t4

        # metrics
        t5 = time.perf_counter()
        m_mse = mse(proc, mosaic)
        m_psnr = psnr_rgb(proc, mosaic)
        m_ssim_pix = ssim_rgb(proc, mosaic)
        m_ssim_blk = ssim_blockwise(proc, mosaic, cell)
        r_ssim, g_ssim, b_ssim, m_ssim_mean = ssim_per_channel(proc, mosaic)
        m_ssim_global = ssim_global_gray(proc, mosaic)
        m_ssim_ms = ssim_multiscale_gray(proc, mosaic)
        t_metrics = time.perf_counter() - t5

        # save (download)
        t6 = time.perf_counter()
        tmp_path = os.path.join(tempfile.gettempdir(), f"mosaic_{uuid.uuid4().hex}.png")
        Image.fromarray(mosaic).save(tmp_path)
        t_save = time.perf_counter() - t6

        t_total = time.perf_counter() - t_all0

        metrics_md = (
            f"**MSE:** {m_mse:.2f}  \n"
            f"**PSNR:** {m_psnr:.2f} dB  \n"
            f"**SSIM (pixel):** {m_ssim_pix:.4f}  \n"
            f"**SSIM (block):** {m_ssim_blk:.4f}  \n"
            f"**SSIM (R,G,B):** {r_ssim:.4f}, {g_ssim:.4f}, {b_ssim:.4f} **(mean {m_ssim_mean:.4f})**  \n"
            f"**SSIM (global gray):** {m_ssim_global:.4f}  \n"
            f"**SSIM (multi-scale gray):** {m_ssim_ms:.4f}"
        )

        timing_md = (
            f"- **Cache:** {t_cache:.3f} s  \n"
            f"- **Resize / Quantize / Grid:** {t_prep:.3f} s  \n"
            f"- **Cell features:** {t_feat:.3f} s  \n"
            f"- **Matching:** {t_match:.3f} s  \n"
            f"- **Reconstruction:** {t_recon:.3f} s  \n"
            f"- **Metrics:** {t_metrics:.3f} s  \n"
            f"- **Save PNG:** {t_save:.3f} s  \n\n"
            f"**TOTAL:** {t_total:.3f} s"
        )

        return proc_vis, mosaic, metrics_md, timing_md, tmp_path

    except FileNotFoundError as e:
        return None, None, _error_md(str(e)), "", ""
    except ValueError as e:
        return None, None, _error_md(str(e)), "", ""
    except Exception as e:
        # last-resort guard; keep UI responsive
        return None, None, _error_md("Unexpected failure. Check logs."), "", ""


# ------------------------------- UI ----------------------------------------

def ui():
    with gr.Blocks(title="🎨 Interactive Image Mosaic Generator") as demo:
        gr.Markdown(
            """
# 🎨 Interactive Image Mosaic Generator

Create stunning image mosaics using advanced tile-matching algorithms!

**Instructions:**
1. Upload an image or select a sample
2. Adjust parameters (tile size, grid dimensions, color/texture balance)
3. Click **Build Mosaic** to generate your mosaic
4. View quality metrics to assess the result

**Tips:**
- Lower Alpha = More texture-focused matching  
- Higher Alpha = More color-focused matching  
- Larger grids = More detail but longer processing
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(label="Input Image", type="filepath", height=DISPLAY_H)
                tile_dir = gr.Textbox(
                    label="Tile Directory",
                    value="Tiles",
                    placeholder="path to folder with tile images",
                )
                grid = gr.Radio(PRESET_GRID, value="64×64", label="Grid Size", interactive=True)
                tile = gr.Radio(PRESET_TILE, value="16×16", label="Tile Size", interactive=True)
                method = gr.Radio(METHODS, value="Custom (Color+Texture)", label="Method", interactive=True)
                quant = gr.Radio(choices=["0", "8", "16", "32"], value="0", label="Quantization (k)", interactive=True)
                alpha_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="α (color weight for Custom)")

                # Image-only examples: click to prefill input image
                example_paths: list[str] = []
                if os.path.isdir(SAMPLE_DIR):
                    for name in sorted(os.listdir(SAMPLE_DIR)):
                        if os.path.splitext(name.lower())[1] in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                            example_paths.append(os.path.join(SAMPLE_DIR, name))

                if example_paths:
                    gr.Examples(
                        examples=example_paths,
                        inputs=[img],
                        label="Examples (click to prefill)",
                        run_on_click=False,
                        examples_per_page=12,
                    )

                run = gr.Button("Build Mosaic", variant="primary")
                download = gr.DownloadButton(label="Download Mosaic")
                status = gr.Textbox(label="Status / Cache", value="", interactive=False, lines=6)

            with gr.Column(scale=1):
                out_proc = gr.Image(label="Processed (grid overlay)", height=DISPLAY_H)
                out_mosaic = gr.Image(label="Mosaic", height=DISPLAY_H)
                out_metrics = gr.Markdown()
                out_timing = gr.Markdown()

        # Build/verify caches on app load
        def on_load(tile_dir_path: str):
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                msgs = ensure_caches(tile_dir_path or "Tiles", sizes=[4, 8, 16, 32], cache_dir=CACHE_DIR)
                return "\n".join(msgs)
            except Exception as e:
                return f"❌ Cache init failed: {e}"

        demo.load(on_load, inputs=[tile_dir], outputs=[status])

        run.click(
            fn=build_mosaic,
            inputs=[img, tile_dir, grid, tile, method, quant, alpha_slider, status],
            outputs=[out_proc, out_mosaic, out_metrics, out_timing, download],
        )

    return demo


if __name__ == "__main__":
    ui().launch()
