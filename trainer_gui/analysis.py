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
        warns.append("Dataset has neither RGB nor intensity — models expecting "
                     "color/intensity inputs will run with degraded features.")
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


# ---- density-generalization advice (Datasets page "Advanced" panel) ----------

def dg_recommend(train_density: float, infer_density: float) -> dict:
    """Suggest density-generalization settings from the dataset's training density
    and a target inference density. Heuristic; the user can override every field.

    The occupancy lens (o = rho*g^2): training canonicalizes density to the model
    grid, so the gap that hurts is inference at a DIFFERENT density. You can always
    thin a denser cloud DOWN (the grid does it for free) but never invent points
    for a sparser one — so the sparse direction needs train-time tolerance (density
    aug + optional log-d_k channel), while the dense direction is handled by the
    grid plus the label-free inference patches (AdaBN, TTA).
    """
    train_density = max(float(train_density), 1e-6)
    infer_density = max(float(infer_density), 1e-6)
    ratio = train_density / infer_density          # >1 => inference is SPARSER
    gap = max(ratio, 1.0 / ratio)                  # fold factor, >=1

    rec = {"density_aug": False, "coarsen_max": 2.5, "p_native": 0.5,
           "logdk": False, "logdk_k": 8, "adabn": False, "tta": 0}

    if gap < 1.2:
        rec["rationale"] = (f"Train {train_density:.1f} vs infer {infer_density:.1f} pts/m² are "
                            f"within {(gap - 1) * 100:.0f}% — no density adaptation needed.")
        return rec

    rec["adabn"] = True                            # label-free, cheap insurance either way
    rec["tta"] = 3 if gap > 1.5 else 2
    if ratio > 1.0:                                # inference SPARSER -> the hard direction
        rec["density_aug"] = True
        rec["coarsen_max"] = round(min(max(math.sqrt(ratio), 1.5), 4.0), 2)
        rec["logdk"] = gap > 2.5
        rec["rationale"] = (
            f"Inference ({infer_density:.1f}) is {gap:.1f}x SPARSER than training "
            f"({train_density:.1f} pts/m^2) — the hard direction. Train with density aug "
            f"(coarsen x{rec['coarsen_max']}) to reach the sparse end"
            + ("; add the log-d_k channel for the large gap" if rec["logdk"] else "")
            + ". AdaBN + TTA patch the rest at inference (no retrain).")
    else:                                          # inference DENSER -> the easy direction
        rec["rationale"] = (
            f"Inference ({infer_density:.1f}) is {gap:.1f}x DENSER than training "
            f"({train_density:.1f} pts/m^2) — the easy direction: the grid subsample "
            f"canonicalizes it down for free, so AdaBN + TTA suffice (no train aug).")
    return rec


def dg_config_to_env(cfg: dict) -> dict:
    """Per-dataset TRAIN-time DG config -> DG_* env vars the training scripts read.
    Emits only the toggles that are ON (+ their values); empty/unset = baseline.

    Train-time only: density aug + the logdk channel (logdk changes the input width,
    so it's baked into the weights and recorded in run.json). The label-free inference
    patches (AdaBN, TTA) are set per-run on the Inference page, not here."""
    if not cfg:
        return {}
    env: dict[str, str] = {}
    if cfg.get("density_aug"):
        env["DG_DENSITY_AUG"] = "1"
        env["DG_COARSEN_MAX"] = str(cfg.get("coarsen_max", 2.5))
        env["DG_P_NATIVE"] = str(cfg.get("p_native", 0.5))
    if cfg.get("logdk"):
        env["DG_LOGDK_FEAT"] = "1"
        env["DG_LOGDK_K"] = str(int(cfg.get("logdk_k", 8)))
    return env


# Script defaults for the loss / class-balance knobs (same in all 6 scripts); the
# panel only emits an env var when the user departs from these, so a baseline run
# stays env-free and reproducible from the script constants alone.
LOSS_DEFAULTS = {"focal": False, "focal_gamma": 2.0, "class_weighting": True,
                 "weight_beta": 0.5, "rare_oversample": True}


def loss_config_to_env(cfg: dict) -> dict:
    """Per-run loss / class-balance config -> LOSS_*/RARE_* env vars the training
    scripts read (mirrors the DG env pattern). Emits only values that differ from
    the script defaults; the run's choices are recorded in run_config.json's loss
    block. focal_gamma is emitted only when focal is on (otherwise it's a no-op)."""
    env: dict[str, str] = {}
    if not cfg:
        return env
    b = lambda v: "1" if v else "0"
    if cfg.get("focal", False) != LOSS_DEFAULTS["focal"]:
        env["LOSS_FOCAL"] = b(cfg.get("focal"))
    if cfg.get("focal") and float(cfg.get("focal_gamma", 2.0)) != LOSS_DEFAULTS["focal_gamma"]:
        env["LOSS_FOCAL_GAMMA"] = str(float(cfg["focal_gamma"]))
    if cfg.get("class_weighting", True) != LOSS_DEFAULTS["class_weighting"]:
        env["LOSS_CLASS_WEIGHTING"] = b(cfg.get("class_weighting"))
    if float(cfg.get("weight_beta", 0.5)) != LOSS_DEFAULTS["weight_beta"]:
        env["LOSS_WEIGHT_BETA"] = str(float(cfg["weight_beta"]))
    if cfg.get("rare_oversample", True) != LOSS_DEFAULTS["rare_oversample"]:
        env["RARE_OVERSAMPLE"] = b(cfg.get("rare_oversample"))
    return env
