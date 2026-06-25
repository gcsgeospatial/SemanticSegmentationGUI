"""Density analysis + per-backbone parameter recommendations.

Grid heuristic (same as utonia_pdal's _suggest_grid): mean 2D point spacing =
sqrt(bbox_area / n); a voxel grid of ~3x the spacing keeps a few points per
cell. Clamped into each backbone's sane band.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .backbones import BACKBONES
from .readers import read_points

MAX_FILES_PER_SPLIT = 5
MAX_POINTS_PER_FILE = 2_000_000


def scan_folder(files: list[Path]) -> dict:
    """Quick local stats over a sample of scenes (pre-conversion 'Analyze' button)."""
    n_total, area_total, max_pts = 0, 0.0, 0
    has_rgb = has_intensity = True
    for path in files[:MAX_FILES_PER_SPLIT]:
        cloud = read_points(path)
        xyz = cloud.xyz
        if len(xyz) > MAX_POINTS_PER_FILE:
            xyz = xyz[:: len(xyz) // MAX_POINTS_PER_FILE + 1]
        bbox = cloud.xyz[:, :2].max(0) - cloud.xyz[:, :2].min(0)
        area = max(float(bbox[0] * bbox[1]), 1.0)
        n_total += cloud.n
        area_total += area
        max_pts = max(max_pts, cloud.n)
        has_rgb &= cloud.rgb is not None
        has_intensity &= cloud.intensity is not None
    density = n_total / max(area_total, 1.0)
    return {
        "files_scanned": min(len(files), MAX_FILES_PER_SPLIT),
        "total_points_scanned": n_total,
        "mean_pts_per_m2": density,
        "mean_spacing_m": (area_total / max(n_total, 1)) ** 0.5,
        "max_scene_points": max_pts,
        "has_rgb": has_rgb,
        "has_intensity": has_intensity,
    }


def recommend(meta: dict) -> dict:
    """Per-backbone parameter recommendations from dataset_meta-shaped stats."""
    stats = meta.get("stats", meta)
    spacing = float(stats.get("mean_spacing_m", 0.3))
    density = float(stats.get("mean_pts_per_m2", 10.0))

    # tile size targeting ~250k raw points per tile, in [25, 100] m, rounded to 5
    chunk = math.sqrt(250_000 / max(density, 0.01))
    chunk = min(max(5 * round(chunk / 5), 25), 100)

    recs: dict[str, dict] = {}
    for key, b in BACKBONES.items():
        rec: dict[str, float] = {"chunk_xy": float(chunk)}
        if b.grid_kind == "octree_depth":
            grid = max(b.grid_mult * spacing, 0.01)
            depth = math.ceil(math.log2(max(chunk / grid, 2.0)))
            rec["octree_depth"] = int(min(max(depth, b.grid_clamp[0]), b.grid_clamp[1]))
            est_cells = (chunk / grid) ** 2 * 0.25
        else:
            lo, hi = b.grid_clamp
            grid = round(min(max(b.grid_mult * spacing, lo), hi), 2)
            rec["grid"] = grid
            est_cells = (chunk / grid) ** 2 * 0.25  # ~25% of 2D cells occupied

        batch_default = next((p.default for p in b.params if p.flag == "batch"), 4)
        rec["batch"] = int(max(1, batch_default // 2)) if est_cells > 120_000 else int(batch_default)
        recs[key] = rec
    return recs


def warnings_for(meta: dict) -> list[str]:
    """Human-readable cautions shown next to the recommendations."""
    warns = []
    if not meta.get("has_rgb", False) and not meta.get("has_intensity", False):
        warns.append("Dataset has neither RGB nor intensity — warm-started encoders "
                     "expecting color/intensity inputs will run with degraded features.")
    # Dedupe by class index: a combined class can appear once per source value in
    # older metas (same index, each carrying the full count), which would inflate
    # the total and warn about the class twice. Keep one entry per index.
    seen: dict = {}
    for c in meta.get("classes", []):
        seen.setdefault(c.get("index", c.get("name")), c)
    classes = list(seen.values())
    total = sum(int(c.get("train_count", 0)) for c in classes) or 1
    for c in classes:
        share = int(c.get("train_count", 0)) / total
        if 0 < share < 0.005:
            warns.append(f"Class '{c.get('name')}' is only {share * 100:.2f}% of training "
                         f"points — consider rare-class oversampling / focal loss.")
    return warns
