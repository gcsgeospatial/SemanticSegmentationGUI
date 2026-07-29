"""Density read-out + prediction metrics + train-time env-var config."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .readers import read_points

MAX_FILES_PER_SPLIT = 5


def scan_folder(files: list[Path]) -> dict:
    """Quick local stats over a sample of scenes (pre-conversion 'Analyze' button)."""
    n_total, area_total, max_pts = 0, 0.0, 0
    has_rgb = has_intensity = True
    for path in files[:MAX_FILES_PER_SPLIT]:
        cloud = read_points(path)
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


def dg_config_to_env(cfg: dict) -> dict:
    """Train-time DG config -> DG_* env vars; only ON toggles emit (empty = baseline).
    AdaBN/TTA are inference-page settings, not here."""
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


# script defaults for the loss knobs (same in all trainers); a baseline run stays env-free
LOSS_DEFAULTS = {"focal": False, "focal_gamma": 2.0, "class_weighting": True,
                 "weight_beta": 0.5, "rare_oversample": True}


def loss_config_to_env(cfg: dict) -> dict:
    """Loss config -> LOSS_*/RARE_* env vars; only departures from LOSS_DEFAULTS emit."""
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


_CLASS_KEYS = ("classification", "pred", "label")


def _npz_class(z) -> np.ndarray | None:
    """An npz's per-point class array; -1 stays -1 (ignore)."""
    for k in _CLASS_KEYS:
        if k in z:
            return np.asarray(z[k], np.int64).reshape(-1)
    return None


def read_classes(path: Path) -> np.ndarray:
    """Per-point class indices from a file with an explicit classification."""
    path = Path(path)
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


def prediction_metrics(pred: np.ndarray, gt: np.ndarray,
                       gt_map: dict[int, int] | None = None) -> dict:
    """Accuracy / mIoU / macro-F1 + per-class IoU-precision-recall-F1.

    gt_map remaps truth classes into the prediction's class space; truth
    classes absent from the map are excluded from scoring entirely, and the
    macro averages run only over classes present in the (mapped) truth - so
    predicting classes the truth doesn't contain is never penalized."""
    pred = np.asarray(pred, np.int64).reshape(-1)
    gt = np.asarray(gt, np.int64).reshape(-1)
    if gt_map is not None:
        mapped = np.full_like(gt, -1)
        for src, dst in gt_map.items():
            mapped[gt == int(src)] = int(dst)
        gt = mapped
    has = gt >= 0
    evaluated = int(has.sum())
    p, g = pred[has], gt[has]
    acc = float((p == g).sum()) / max(evaluated, 1)
    per = {}
    for c in (int(c) for c in np.unique(g)):
        tp = int(((p == c) & (g == c)).sum())
        fp = int(((p == c) & (g != c)).sum())
        fn = int(((p != c) & (g == c)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per[c] = {"iou": tp / max(tp + fp + fn, 1),
                  "precision": prec, "recall": rec,
                  "f1": 2 * prec * rec / max(prec + rec, 1e-12),
                  "support": tp + fn}
    mean = lambda k: float(np.mean([v[k] for v in per.values()])) if per else 0.0
    return {"accuracy": acc, "miou": mean("iou"), "macro_f1": mean("f1"),
            "evaluated": evaluated, "ignored": int(len(gt) - evaluated),
            "per_class": per}
