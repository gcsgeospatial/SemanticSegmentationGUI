"""Density analysis + per-backbone parameter recommendations.

Built for aerial data from 0.5 pts/m2 (legacy national scans) to 1000 pts/m2
(UAV). One law drives the grid: occupancy o = rho * g^2 (scripts/DENSITY_DG.md).
Each backbone's grid tracks mean point spacing (grid_mult, so o ~ grid_mult^2
sits just above 1) - the finest grid that stays in the density-invariant band.
Tiles follow the grid in receptive-field units; PTv3's batch comes from a
per-forward voxel budget via the occupied-cell law (1 - e^-o), while the
pinned-sample backbones keep their proven default batches - so GPU load stays
level across three decades of density instead of exploding at either end.
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


# --- recommendation targets (aerial data, 0.5-1000 pts/m2) --------------------
RL_SAMPLE_AREA = 3600.0     # RandLA target sample footprint (m^2), ~34 m radius
RL_N_LIM = (8192, 45056)    # floor: 32 pts survive the 256x pyramid; cap: the
                            # script's fixed VAL_BATCH=12 eval flush stays at the
                            # proven SemKITTI-recipe level (8 GB class)
PTV3_CROP = 80_000          # script constant: bigger TRAIN tiles collapse to 15 m
                            # discs (local_train_ptv3.py to_ptv3_batch's train
                            # crop) while eval keeps the whole tile; crop
                            # triggers at rho > 80k/chunk^2
PTV3_CROP_RHO = 113.0       # 80k/(pi*15^2): densest cloud PTv3 training ever sees
PTV3_RAW_TARGET = 40_000    # raw pts/tile the chunk aims for: 50% of the crop so
                            # 5 m rounding / local variation never straddles it
                            # (holds until the 30 m floor binds at ~44 pts/m2)
PTV3_VOX_BUDGET = 60_000    # train voxels per forward. The script runs fp32 with
                            # standard (non-flash) SerializedAttention, which
                            # retains ~0.2 MB/voxel of attention probs for
                            # backward: ~12 GB attention + ~3 GB activations/
                            # optimizer at this cap = ~15 GB peak; forest tiles
                            # dedup less than the VERT estimate (up to ~19 GB),
                            # hence PTv3's 24 GB min-VRAM floor in backbones.py.
                            # Raise only if the script gains AMP/flash.
VERT = 1.3                  # occupied 3D cells per occupied 2D cell (walls,
                            # canopy) for a typical mixed scene; the vegetation
                            # worst case (canopy columns barely dedup at o < 3,
                            # ~0.8x raw survives) is absorbed by the 24 GB floor,
                            # not by inflating every estimate


def recommend(meta: dict) -> dict:
    """Per-backbone parameter recommendations from dataset_meta-shaped stats,
    tuned for aerial densities 0.5-1000 pts/m2 (constants above)."""
    stats = meta.get("stats", meta)
    # floor 0.25: below it even the 2.0 m grid cap can't hold o >= 1 and a 200 m
    # tile can't reach PTv3's 2048-pt admission floor - warnings_for() covers it
    density = max(float(stats.get("mean_pts_per_m2", 10.0)), 0.25)
    spacing = float(stats.get("mean_spacing_m", 0.0)) or density ** -0.5

    recs: dict[str, dict] = {}
    for key, b in BACKBONES.items():
        lo, hi = b.grid_clamp
        batch_hi = next((int(p.default) for p in b.params if p.flag == "batch"), 4)

        if not b.has_chunk:
            # RandLA samples fixed-N kNN spheres, no tiles. N ~ half the raw
            # points on the target footprint (snapped to the pyramid-friendly
            # 4096); grid is the LARGER of the invariance grid and the footprint
            # grid sqrt(area/N), so a sample always covers ~RL_SAMPLE_AREA m^2
            # and never sinks below o=1. Post-subsample density law:
            # rho_sub = (1 - e^-o) / g^2 - the density the net actually sees.
            # batch stays the script default: 6 * N <= the proven 6x45056 recipe
            # at every N the clamp allows.
            n = int(4096 * round(min(max(RL_SAMPLE_AREA * density / 2.0,
                                         RL_N_LIM[0]), RL_N_LIM[1]) / 4096))
            grid = round(min(max(b.grid_mult * spacing,
                                 (RL_SAMPLE_AREA / n) ** 0.5, lo), hi), 2)
            recs[key] = {"grid": grid, "num_points": n, "batch": batch_hi}
            continue

        grid = round(min(max(b.grid_mult * spacing, lo), hi), 2)
        o = density * grid * grid                  # >= grid_mult^2 by construction
        if key.startswith("ptv3"):
            # tile targets PTV3_RAW_TARGET raw pts to stay clear of the train
            # crop. The 30 m context floor binds from ~45 pts/m2 (the crop first
            # triggers at ~89, i.e. 80k/30^2); 30 m also bounds the EVAL tile on
            # the 24 GB class. Batch sinks to 1 at UAV density: fp32 non-flash
            # attention leaves no headroom there, and PTv3 normalizes per-sample
            # (LayerNorm; stem BN still sees 30k+ voxels per tile).
            chunk = min(max((PTV3_RAW_TARGET / density) ** 0.5, 30.0), 200.0)
            o_t = min(density, PTV3_CROP_RHO) * grid * grid   # crop-capped train occupancy
            tile = (min(chunk * chunk * density, PTV3_CROP)
                    * (1.0 - math.exp(-o_t)) / o_t * VERT)    # voxels the GPU sees
            batch = int(min(max(PTV3_VOX_BUDGET // int(tile), 1), batch_hi))
        else:
            # KPConvX: the conv-radius ladder tops out near 50*g; below 40 m
            # tiles lose the building/tree context the deep layers are sized for
            # (at the 0.4 m grid floor the top radius is ~16 m - past ~150 pts/m2
            # PTv3 is the detail backbone, KPConvX the context one). The pack is
            # <= ~52k pts at any grid in the band, so the script's default batch
            # (x2 via its grad-accum) always fits the 24 GB class.
            chunk = min(max(50.0 * grid, 40.0), 200.0)
            batch = batch_hi
        recs[key] = {"chunk_xy": 5.0 * round(chunk / 5.0), "grid": grid,
                     "batch": batch}
    return recs


def warnings_for(meta: dict) -> list[str]:
    """Human-readable cautions shown next to the recommendations."""
    warns = []
    if not meta.get("has_rgb", False) and not meta.get("has_intensity", False):
        warns.append("Dataset has neither RGB nor intensity - models expecting "
                     "color/intensity inputs will run with degraded features.")
    stats = meta.get("stats", meta)
    density = float(stats.get("mean_pts_per_m2", 0) or 0)
    if density > 0:
        side = (2048.0 / density) ** 0.5   # PTv3 prep skips tiles under 2048 raw pts
        if side > 40:
            warns.append(f"Sparse data ({density:.2g} pts/m²): tiles under ~{5 * math.ceil(side / 5):.0f} m "
                         "hold <2048 points and PTv3 prep silently skips them - keep "
                         "the tile size at or above the recommendation.")
        rl = meta.get("recommendations", {}).get("randlanet", {})
        ext = stats.get("mean_scene_extent_m")
        if rl.get("num_points") and ext:
            # a sample pads iff the whole SUBSAMPLED scene holds fewer points
            # than it draws (rho_sub = (1 - e^-o)/g^2), not iff the scene is
            # narrow - corridor scenes are fine as long as they hold the points
            g = float(rl["grid"])
            n_scene = (float(ext[0]) * float(ext[1])
                       * (1.0 - math.exp(-density * g * g)) / (g * g))
            if n_scene < rl["num_points"]:
                warns.append(f"Scenes hold only ~{n_scene:,.0f} points after the {g:.2f} m "
                             f"subsample but a RandLA sample draws {rl['num_points']:,} - "
                             "samples will pad with duplicated points; prefer larger "
                             "scenes or a tiled backbone.")
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
                         f"points - consider rare-class oversampling / focal loss.")
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
                            f"within {(gap - 1) * 100:.0f}% - no density adaptation needed.")
        return rec

    rec["adabn"] = True                            # label-free, cheap insurance either way
    rec["tta"] = 4 if gap > 8.0 else (3 if gap > 1.5 else 2)
    if ratio > 1.0:                                # inference SPARSER -> the hard direction
        # coarsen_max = 1/(g0*sqrt(rho_min)) (density.py), with g0 the grid the
        # model actually trains at. recommend()'s clamps pin g0 coarse on dense
        # data, so a raw sqrt(ratio) badly overshoots there (train@1000: KPConvX
        # at g=0.4 needs only ~1.8x for infer@2, not the cap). Size to the
        # finest backbone grid (the binding case); the rationale names the range.
        needs = sorted(1.0 / (r["grid"] * math.sqrt(infer_density))
                       for r in recommend({"stats": {"mean_pts_per_m2": train_density}}).values())
        need = needs[-1]
        if need <= 1.0:
            # the training grids already cap density at or below the inference
            # target - the voxel subsample canonicalizes the gap away; aug would
            # only dilute native-density accuracy
            rec["rationale"] = (
                f"Training ({train_density:.1f} pts/m^2) uses grids that already cap "
                f"density at or below the inference target ({infer_density:.1f}) - the "
                "voxel grid canonicalizes the gap for free; AdaBN + TTA are enough.")
            return rec
        rec["density_aug"] = True
        rec["coarsen_max"] = round(min(max(need, 1.5), 6.0), 2)
        rec["p_native"] = 0.35 if rec["coarsen_max"] > 3.0 else 0.5   # big gap: more mass on the coarse end
        rec["logdk"] = gap > 2.5
        span = (f" (finest backbone; coarser ones need down to x{max(needs[0], 1.0):.1f})"
                if needs[-1] > 1.5 * max(needs[0], 1.0) else "")
        rec["rationale"] = (
            f"Inference ({infer_density:.1f}) is {gap:.1f}x SPARSER than training "
            f"({train_density:.1f} pts/m^2) - the hard direction. Train with density aug "
            f"(coarsen x{rec['coarsen_max']}{span}) to reach the sparse end"
            + ("; add the log-d_k channel for the large gap" if rec["logdk"] else "")
            + (f"; the needed x{need:.1f} exceeds the aug range - prefer retraining "
               "at the density recommend() picks for the inference data"
               if need > 6.0 else "")
            + ". AdaBN + TTA patch the rest at inference (no retrain).")
    else:                                          # inference DENSER -> the easy direction
        rec["rationale"] = (
            f"Inference ({infer_density:.1f}) is {gap:.1f}x DENSER than training "
            f"({train_density:.1f} pts/m^2) - the easy direction: the grid subsample "
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
    the script defaults; the run's choices are recorded in run.json's loss
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


# ---- prediction vs ground truth (Inference page "Compare to ground truth") ----
# Stats only — both files must carry an EXPLICIT per-point classification (a
# prediction npz's 'classification'/'pred', a dataset npz's 'label', a LAS/LAZ
# classification, or any label column readers.read_points exposes). Decoding
# classes back out of palette RGB is gone with the viewer.

_CLASS_KEYS = ("classification", "pred", "label")


def _npz_class(z) -> np.ndarray | None:
    """An npz's per-point class array: exported prediction ('classification'),
    raw prediction ('pred'), or dataset/GT ('label'). -1 stays -1 (ignore)."""
    for k in _CLASS_KEYS:
        if k in z:
            return np.asarray(z[k], np.int64).reshape(-1)
    return None


def _read_classes(path: Path) -> np.ndarray:
    """Per-point class indices from a file with an explicit classification."""
    if path.suffix.lower() == ".npz":
        cls = _npz_class(np.load(str(path), allow_pickle=False))
        if cls is None:
            raise ValueError(f"{path.name}: npz has no "
                             f"{'/'.join(_CLASS_KEYS)} array to compare")
        return cls
    fields = read_points(path).fields
    for k in fields:
        if k.lower() in _CLASS_KEYS or k.lower() in ("class", "scalar_label"):
            return np.asarray(fields[k], np.int64).reshape(-1)
    raise ValueError(f"{path.name}: no classification/label field to compare - "
                     f"use files that carry explicit per-point classes")


def prediction_metrics(pred_path, gt_path) -> dict:
    """Overall accuracy + mIoU + per-class IoU of a prediction cloud against
    ground truth, scored on points that carry a GT label. mIoU averages only the
    classes present in GT or prediction (absent classes don't drag it to zero)."""
    pred_path, gt_path = Path(pred_path), Path(gt_path)
    pred = _read_classes(pred_path)
    gt = _read_classes(gt_path)
    scene = pred_path.stem
    for suffix in ("_pred", "_gt"):
        scene = scene.replace(suffix, "")
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    has = gt >= 0
    labeled = int(has.sum())
    acc = float((pred[has] == gt[has]).sum()) / max(labeled, 1)
    present = sorted({int(c) for c in np.unique(pred[has])} | {int(c) for c in np.unique(gt[has])})
    present = [c for c in present if c >= 0]   # drop the unlabeled (-1) class from mIoU
    ious = {}
    for c in present:
        inter = int(((pred == c) & (gt == c) & has).sum())
        union = int((((pred == c) | (gt == c)) & has).sum())
        ious[c] = inter / union if union else 0.0
    miou = float(np.mean(list(ious.values()))) if ious else 0.0
    return {"scene": scene, "accuracy": acc, "miou": miou,
            "labeled": labeled, "per_class_iou": ious}
