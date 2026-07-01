"""Local preprocessing — build each backbone's prep cache on this machine.

Every refactored modal_train_*.py does an idempotent `ensure_prep()` remotely:
if a scene's tiles already exist under /datasets/<name>/prep/<tag>/ it skips
them. This module produces byte-layout-compatible caches locally (same dir
names, file names, npz keys, deterministic seed-42 holdout split, min-point
thresholds and strides), so after `modal volume put` the remote prep is a
no-op and the GPU container goes straight to training.

The big win is OctFormer's per-point PCA normal estimation, which otherwise
runs on the GPU container's CPU at GPU-instance prices. Here it's vectorized
(same math: k-NN, covariance, smallest eigenvector).

IMPORTANT: this mirrors the canonical-dataset branches of the scripts. If a
script's tiling logic changes, change it here too — a mismatch is harmless
(the remote ensure_prep just re-tiles what's missing) but wastes the upload.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- loaders
# These mirror the scripts' load_canonical() functions exactly.

def _load_warm(npz_path):
    """xyz, intensity (0..1; rgb-luminance or 0.5 fallback), ret_num, lab."""
    z = np.load(npz_path)
    xyz = z["xyz"].astype(np.float32)
    if "intensity" in z:
        intensity = z["intensity"].astype(np.float32)
    elif "rgb" in z:
        rgb = z["rgb"].astype(np.float32)
        intensity = ((0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2])
                     / 255.0).astype(np.float32)
    else:
        intensity = np.full(len(xyz), 0.5, np.float32)
    ret_num = z["return_number"].astype(np.float32) if "return_number" in z \
        else np.zeros(len(xyz), np.float32)
    lab = z["label"].astype(np.int32) if "label" in z \
        else np.full(len(xyz), -1, np.int32)
    return xyz, intensity, ret_num, lab


def _load_cold(npz_path):
    """xyz, rgb (intensity-gray or 128 fallback), lab."""
    z = np.load(npz_path)
    xyz = z["xyz"].astype(np.float32)
    if "rgb" in z:
        rgb = z["rgb"].astype(np.float32)
    elif "intensity" in z:
        rgb = np.repeat((z["intensity"].astype(np.float32) * 255.0)[:, None], 3, axis=1)
    else:
        rgb = np.full((len(xyz), 3), 128.0, dtype=np.float32)
    lab = z["label"].astype(np.int32) if "label" in z \
        else np.full(len(xyz), -1, np.int32)
    return xyz, rgb, lab


def estimate_normals(xyz, k=12):
    """Vectorized version of the scripts' estimate_normals_naive: k-NN local
    PCA, smallest eigenvector. Same math, ~100x faster than the python loop."""
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)
    idx = idx[:, 1:]
    P = xyz[idx] - xyz[:, None, :]            # (N, k, 3)
    cov = np.einsum("nki,nkj->nij", P, P)     # (N, 3, 3) = P^T P per point
    _, V = np.linalg.eigh(cov)
    return V[:, :, 0].astype(np.float32)      # smallest-eigenvalue eigenvector


# ---------------------------------------------------------------- splits

def _scenes(staged_root: Path):
    """The three materialized whole-scene folders, read verbatim (val = selection
    holdout, test = final report). The dataset stage owns the split; prep does
    NOT re-derive val."""
    train = sorted(glob.glob(str(staged_root / "train" / "*.npz")))
    val   = sorted(glob.glob(str(staged_root / "val" / "*.npz")))
    test  = sorted(glob.glob(str(staged_root / "test" / "*.npz")))
    if not train:
        raise FileNotFoundError(f"No canonical scenes under {staged_root}/train - "
                                f"convert the dataset first.")
    return train, val, test


# ---------------------------------------------------------------- tilers

def _tile_warm(name, src, out_dir, chunk, stride, min_pts, normals, max_tile_pts, say):
    """Tile loop matching the warm scripts' tile_and_save (intensity layout)."""
    os.makedirs(out_dir, exist_ok=True)
    xyz, intensity, ret_num, lab = _load_warm(src)
    mins, maxs = xyz[:, :2].min(0), xyz[:, :2].max(0)
    n_tiles = 0
    for x0 in np.arange(mins[0], maxs[0], stride):
        for y0 in np.arange(mins[1], maxs[1], stride):
            m = ((xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + chunk) &
                 (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + chunk))
            n = int(m.sum())
            if n < min_pts:
                continue
            pts, itn, rn, lb = xyz[m], intensity[m], ret_num[m], lab[m]
            out = {"xyz": pts.astype(np.float32),
                   "intensity": itn.astype(np.float32),
                   "lab": lb.astype(np.int32)}
            if normals:
                if n > max_tile_pts:
                    keep = np.random.choice(n, max_tile_pts, replace=False)
                    pts, itn, lb = pts[keep], itn[keep], lb[keep]
                    out = {"xyz": pts.astype(np.float32),
                           "intensity": itn.astype(np.float32),
                           "lab": lb.astype(np.int32)}
                out["nrm"] = estimate_normals(pts, k=12)
            else:
                out["ret_num"] = rn.astype(np.float32)
            np.savez_compressed(f"{out_dir}/{name}_x{int(x0)}_y{int(y0)}.npz", **out)
            n_tiles += 1
    say(f"    {name}: {n_tiles} tiles")
    return n_tiles


def _tile_cold(name, src, out_dir, chunk, stride, min_pts, normals, max_tile_pts, say):
    """Tile loop matching the cold scripts' tilers (rgb layout)."""
    os.makedirs(out_dir, exist_ok=True)
    xyz, rgb, lab = _load_cold(src)
    mins, maxs = xyz[:, :2].min(0), xyz[:, :2].max(0)
    n_tiles = 0
    for x0 in np.arange(mins[0], maxs[0], stride):
        for y0 in np.arange(mins[1], maxs[1], stride):
            m = ((xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + chunk) &
                 (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + chunk))
            n = int(m.sum())
            if n < min_pts:
                continue
            pts, col, lb = xyz[m], rgb[m], lab[m]
            out = {"xyz": pts.astype(np.float32),
                   "rgb": col.astype(np.uint8),
                   "lab": lb.astype(np.int32)}
            if normals:
                if n > max_tile_pts:
                    keep = np.random.choice(n, max_tile_pts, replace=False)
                    pts, col, lb = pts[keep], col[keep], lb[keep]
                    out = {"xyz": pts.astype(np.float32),
                           "rgb": col.astype(np.uint8),
                           "lab": lb.astype(np.int32)}
                out["nrm"] = estimate_normals(pts, k=12)
            np.savez_compressed(f"{out_dir}/{name}_x{int(x0)}_y{int(y0)}.npz", **out)
            n_tiles += 1
    say(f"    {name}: {n_tiles} tiles")
    return n_tiles


def _grid_subsample_scene(name, src, out_dir, grid, warm, say):
    """Whole-scene grid subsample matching the RandLA-Net scripts' ensure_prep."""
    os.makedirs(out_dir, exist_ok=True)
    if warm:
        xyz, intensity, ret_num, lab = _load_warm(src)
        keys = np.floor(xyz / grid).astype(np.int64)
        _, uniq = np.unique(keys, axis=0, return_index=True)
        np.savez_compressed(f"{out_dir}/{name}.npz",
                            xyz=xyz[uniq].astype(np.float32),
                            intensity=intensity[uniq].astype(np.float32),
                            ret_num=ret_num[uniq].astype(np.float32),
                            lab=lab[uniq].astype(np.int32))
        n = len(uniq)
    else:
        xyz, rgb, lab = _load_cold(src)
        keys = np.floor(xyz / grid).astype(np.int64)
        _, uniq = np.unique(keys, axis=0, return_index=True)
        np.savez_compressed(f"{out_dir}/{name}.npz",
                            xyz=xyz[uniq].astype(np.float32),
                            rgb=rgb[uniq].astype(np.uint8),
                            lab=lab[uniq].astype(np.int32))
        n = len(uniq)
    say(f"    {name}: {len(xyz):,} -> {n:,} pts")


# ---------------------------------------------------------------- per-backbone

def prep_tag(backbone_key: str, params: dict) -> str:
    """The prep/<tag> dir name the script will look for — must match PREP_DIR."""
    chunk = int(params.get("chunk-xy", 50))
    if backbone_key == "ptv3_warm":
        return f"ptv3_warm_chunk{chunk}"
    if backbone_key == "ptv3":
        return f"ptv3_cold_chunk{chunk}"
    if backbone_key == "randlanet_warm":
        return f"randlanet_warm_grid{int(round(params['sub-grid'] * 100))}"
    if backbone_key == "randlanet":
        return f"randlanet_cold_grid{int(round(params['sub-grid'] * 100))}"
    if backbone_key == "octformer_warm":
        return f"octformer_warm_chunk{int(params.get('chunk-xy', 25))}"
    if backbone_key == "octformer":
        return f"octformer_cold_chunk{int(params.get('chunk-xy', 25))}"
    if backbone_key == "kpconvx_warm":
        return f"kpconvx_warm_chunk{int(params.get('chunk-xy', 30))}"
    raise ValueError(f"Local prep not supported for backbone '{backbone_key}'")


def supports_local_prep(backbone_key: str) -> bool:
    try:
        prep_tag(backbone_key, {"chunk-xy": 50, "sub-grid": 0.3})
        return True
    except ValueError:
        return False


def prep_dataset(backbone_key: str, staged_root: str | Path, params: dict,
                 progress=None) -> Path:
    """Build the prep cache for one backbone under <staged_root>/prep/<tag>/.

    Returns the local prep dir; upload it to terminal-datasets at
    /<dataset_name>/prep/<tag> and the script's remote ensure_prep becomes a
    no-op. Idempotent locally too (skips scenes whose outputs already exist).
    """
    say = progress or (lambda s: None)
    staged_root = Path(staged_root)
    tag = prep_tag(backbone_key, params)
    prep_dir = staged_root / "prep" / tag
    train_paths, val_paths, test_paths = _scenes(staged_root)

    warm = backbone_key.endswith("_warm")
    # The dataset stage already materialized train/val/test; read all three folders
    # verbatim (val = selection holdout, test = final report) — no val re-derivation.
    split_items = [("train", [(Path(p).stem, p) for p in train_paths]),
                   ("val",   [(Path(p).stem, p) for p in val_paths]),
                   ("test",  [(Path(p).stem, p) for p in test_paths])]

    if backbone_key in ("randlanet_warm", "randlanet"):
        grid = float(params["sub-grid"])
        for split, items in split_items:
            say(f"  [{split}] {len(items)} scenes -> grid {grid} m")
            out_dir = prep_dir / split
            for name, src in items:
                if (out_dir / f"{name}.npz").exists():
                    continue
                _grid_subsample_scene(name, src, str(out_dir), grid, warm, say)
        return prep_dir

    # Tiled backbones — strides/min-points per script.
    if backbone_key in ("ptv3_warm", "ptv3"):
        chunk = float(params.get("chunk-xy", 50.0))
        min_pts, normals, max_tile = 2048, False, 0
        stride_train = chunk / 2.0
    elif backbone_key in ("octformer_warm", "octformer"):
        chunk = float(params.get("chunk-xy", 25.0))
        min_pts, normals, max_tile = 2048, True, 120_000
        stride_train = chunk / 2.0
    elif backbone_key == "kpconvx_warm":
        chunk = float(params.get("chunk-xy", 30.0))
        min_pts, normals, max_tile = 1024, False, 0
        stride_train = chunk * 2.0 / 3.0
    else:
        raise ValueError(f"Local prep not supported for backbone '{backbone_key}'")

    tiler = _tile_warm if warm else _tile_cold
    # modal_train_ptv3.py (cold) now tiles the TEST split at chunk/2 too, so the
    # final eval can vote per-voxel over the up-to-4 overlapping tiles. Keep this
    # in lockstep — otherwise the uploaded cache has non-overlapping test tiles
    # and the remote `already_tiled` check silently disables overlap voting.
    stride_test = stride_train if backbone_key == "ptv3" else chunk
    for split, items in split_items:
        stride = stride_train if split == "train" else stride_test
        say(f"  [{split}] {len(items)} scenes -> {chunk:.0f} m tiles, stride {stride:.0f}"
            + (", computing normals" if normals else ""))
        out_dir = prep_dir / split
        for name, src in items:
            if glob.glob(str(out_dir / f"{name}_x*.npz")):
                continue
            tiler(name, src, str(out_dir), chunk, stride, min_pts, normals,
                  max_tile, say)
    return prep_dir
