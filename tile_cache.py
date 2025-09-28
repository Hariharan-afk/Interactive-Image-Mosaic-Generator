# File: tile_cache.py
from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.filters import sobel_h, sobel_v

# Always include these transforms when building the cache
ANGLES_DEFAULT = (0, 90, 180, 270)
FLIPS_DEFAULT = (False, True)


# --------------------------------- Helpers ---------------------------------


def _iter_image_files(tile_dir: str) -> list[str]:
    """All image file paths under tile_dir (sorted)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths: list[str] = []
    for root, _, files in os.walk(tile_dir):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths


def digest_tile_dir(tile_dir: str) -> str:
    """Hash filenames + sizes + mtimes (used for cache invalidation)."""
    h = hashlib.sha256()
    for p in _iter_image_files(tile_dir):
        try:
            st = os.stat(p)
        except FileNotFoundError:
            continue
        h.update(p.encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()


def _apply_transform(pil: Image.Image, angle: int, hflip: bool, vflip: bool) -> Image.Image:
    im = pil.rotate(angle, expand=True)
    if hflip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if vflip:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    return im


# ------------------------------ Feature funcs ------------------------------


def _lab_mean(arr: np.ndarray) -> np.ndarray:
    """Mean Lab over an RGB tile."""
    lab = rgb2lab(arr / 255.0)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)


def _texture_scalar(arr: np.ndarray) -> float:
    """Mean Sobel magnitude over L channel for an RGB tile."""
    L = rgb2lab(arr / 255.0)[..., 0].astype(np.float32)
    gy = sobel_h(L)
    gx = sobel_v(L)
    mag = np.hypot(gx, gy)
    return float(mag.mean())


# ------------------------------ Cache build --------------------------------


def cache_filename(key: dict) -> str:
    """Filename uses tile size and folder digest (first 16 chars)."""
    d16 = key["digest"][:16]
    return f"tiles_s{key['tile_size']:02d}_allrotflips_{d16}.npz"


def build_tile_cache(tile_dir: str, tile_size: int, cache_dir: str) -> dict:
    """
    Build cache for a tile size with all rotations & flips, save to cache_dir, return dict:
      - tiles (T,s,s,3) uint8
      - tile_means_lab_rot (T,3) float32
      - tile_texture_rot (T,) float32
      - transform_ids (T,3) int16  (angle,hflip,vflip)
      - file_index (T,) int32      (source image index)
      - key (dict), tile_size (int)
    """
    if not os.path.isdir(tile_dir):
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

    paths = _iter_image_files(tile_dir)
    if not paths:
        raise ValueError(f"No images found in {tile_dir}")

    variants_pixels: list[np.ndarray] = []
    means_lab: list[np.ndarray] = []
    textures: list[float] = []
    transform_ids: list[tuple[int, int, int]] = []
    file_index: list[int] = []

    valid_sources = 0
    for idx, p in enumerate(paths):
        try:
            base = Image.open(p).convert("RGB")
            valid_sources += 1
        except Exception:
            # Skip unreadable/corrupt images
            continue

        for ang in ANGLES_DEFAULT:
            for hf in FLIPS_DEFAULT:
                for vf in FLIPS_DEFAULT:
                    pil = _apply_transform(base, ang, hf, vf).resize(
                        (tile_size, tile_size), Image.Resampling.LANCZOS
                    )
                    arr = np.asarray(pil, dtype=np.uint8)
                    variants_pixels.append(arr)
                    means_lab.append(_lab_mean(arr))
                    textures.append(_texture_scalar(arr))
                    transform_ids.append((ang, int(hf), int(vf)))
                    file_index.append(idx)

    if not variants_pixels or valid_sources == 0:
        raise ValueError(f"No valid tile images could be processed in {tile_dir}")

    tiles = np.stack(variants_pixels, axis=0).astype(np.uint8)
    tile_means_lab_rot = np.stack(means_lab, axis=0).astype(np.float32)
    tile_texture_rot = np.asarray(textures, dtype=np.float32)
    transform_ids = np.asarray(transform_ids, dtype=np.int16)
    file_index = np.asarray(file_index, dtype=np.int32)

    key = {
        "tile_dir": os.path.abspath(tile_dir),
        "tile_size": int(tile_size),
        "angles": list(ANGLES_DEFAULT),
        "flips": True,
        "digest": digest_tile_dir(tile_dir),
    }

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, cache_filename(key))
    np.savez_compressed(
        path,
        key_json=json.dumps(key).encode(),
        tiles=tiles,
        tile_means_lab_rot=tile_means_lab_rot,
        tile_texture_rot=tile_texture_rot,
        transform_ids=transform_ids,
        file_index=file_index,
        tile_size=int(tile_size),
    )

    return {
        "key": key,
        "tiles": tiles,
        "tile_means_lab_rot": tile_means_lab_rot,
        "tile_texture_rot": tile_texture_rot,
        "transform_ids": transform_ids,
        "file_index": file_index,
        "tile_size": int(tile_size),
    }


def load_tile_cache(cache_dir: str, key_expected: dict) -> dict | None:
    """Load cache from disk if present and compatible; else return None (signals rebuild)."""
    path = os.path.join(cache_dir, cache_filename(key_expected))
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        key = json.loads(bytes(z["key_json"]).decode())
        cache = {
            "key": key,
            "tiles": z["tiles"].astype(np.uint8),
            "tile_means_lab_rot": z["tile_means_lab_rot"].astype(np.float32),
            "tile_texture_rot": z["tile_texture_rot"].astype(np.float32),
            "transform_ids": z["transform_ids"].astype(np.int16),
            "file_index": z["file_index"].astype(np.int32),
            "tile_size": int(z["tile_size"]),
        }
    except Exception:
        return None  # corrupted or incompatible; let caller rebuild

    for k in ["tile_dir", "tile_size", "digest"]:
        if str(cache["key"].get(k)) != str(key_expected.get(k)):
            return None
    return cache


def ensure_caches(tile_dir: str, sizes: Iterable[int], cache_dir: str) -> list[str]:
    """Ensure caches for given tile sizes; build missing. Returns log lines."""
    logs: list[str] = []
    digest = digest_tile_dir(tile_dir)
    for s in sizes:
        key = {
            "tile_dir": os.path.abspath(tile_dir),
            "tile_size": int(s),
            "angles": list(ANGLES_DEFAULT),
            "flips": True,
            "digest": digest,
        }
        found = load_tile_cache(cache_dir, key)
        if found is None:
            logs.append(f"Building cache for tile size {s} x {s} …")
            try:
                build_tile_cache(tile_dir, s, cache_dir)
                logs.append(f"Built cache for tile size {s} x {s} ✅")
            except Exception as e:
                logs.append(f"❌ Failed to build cache for {s} x {s}: {e}")
        else:
            logs.append(f"Cache for tile size {s} x {s} ready ✅")
    return logs
