"""Convert user folders into the canonical dataset the refactored scripts consume.

Canonical layout (staged locally, then `modal volume put terminal-datasets ...`):
  <staging>/<name>/
    dataset_meta.json
    train/<scene>.npz     # xyz f32, label i32 (-1 = ignore), [rgb u8, intensity f32 0..1,
    val/<scene>.npz       #   return_number f32]
Inference jobs use the same npz minus `label`, under scenes/.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .readers import SUPPORTED_EXTS, Cloud, read_points


@dataclass
class LabelSpec:
    """Where ground-truth labels come from.

    kind="field": a named field in the cloud file itself (LAS dim, PLY prop,
        "column N" of an ASCII/npy file).
    kind="file": a companion ASCII file, one label per point — the IEEE layout.
        The companion path = truth_dir / scene_name with src_suffix replaced by
        dst_suffix (e.g. "_PC3.txt" -> "_CLS.txt").
    """
    kind: str                      # "field" | "file"
    field: str = ""                # for kind="field"
    truth_dir: str = ""            # for kind="file"
    src_suffix: str = "_PC3.txt"
    dst_suffix: str = "_CLS.txt"


def discover_scenes(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def read_labels(path: Path, cloud: Cloud, spec: LabelSpec) -> np.ndarray:
    """Raw (source-valued) labels for one scene, validated against point count."""
    if spec.kind == "field":
        if spec.field not in cloud.fields:
            raise ValueError(f"{path.name}: label field '{spec.field}' not found "
                             f"(have: {sorted(cloud.fields)})")
        raw = np.asarray(cloud.fields[spec.field])
    else:
        name = path.name
        if spec.src_suffix and name.endswith(spec.src_suffix):
            truth_name = name[: -len(spec.src_suffix)] + spec.dst_suffix
        else:
            truth_name = path.stem + spec.dst_suffix
        truth_path = Path(spec.truth_dir) / truth_name
        if not truth_path.exists():
            raise FileNotFoundError(f"{path.name}: companion label file not found: {truth_path}")
        raw = np.loadtxt(str(truth_path), dtype=np.float64).reshape(-1)
    if len(raw) != cloud.n:
        raise ValueError(f"{path.name}: {cloud.n} points vs {len(raw)} labels")
    return np.round(raw).astype(np.int64)


def scan_label_values(files: list[Path], spec: LabelSpec, max_files: int = 8) -> dict[int, int]:
    """value -> count across a sample of scenes (for the class table UI)."""
    counts: dict[int, int] = {}
    for path in files[:max_files]:
        cloud = read_points(path)
        raw = read_labels(path, cloud, spec)
        vals, cnts = np.unique(raw, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            counts[int(v)] = counts.get(int(v), 0) + int(c)
    return dict(sorted(counts.items()))


def sanitize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_")
    if not s:
        raise ValueError("Dataset name is empty after sanitizing")
    return s.lower()


def convert_scene(path: Path, spec: LabelSpec | None, value_to_index: dict[int, int],
                  out_path: Path, intensity_norm: str = "max") -> dict:
    """Convert one source file to a canonical npz; returns per-scene stats.

    intensity_norm: "max" -> i/max in [0,1] (canonical-dataset default);
    "p95" -> clip(i/p95, 0, 2), matching how the IEEE training scripts normalize
    raw intensity, so inference on those weights sees the same feature scale.
    """
    cloud = read_points(path)
    out: dict[str, np.ndarray] = {"xyz": cloud.xyz.astype(np.float32)}

    class_counts: dict[int, int] = {}
    if spec is not None:
        raw = read_labels(path, cloud, spec)
        label = np.full(cloud.n, -1, dtype=np.int32)
        for src_val, idx in value_to_index.items():
            label[raw == src_val] = idx
        out["label"] = label
        vals, cnts = np.unique(label[label >= 0], return_counts=True)
        class_counts = {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}

    if cloud.rgb is not None:
        out["rgb"] = cloud.rgb
    raw_imax = None
    if cloud.intensity is not None:
        if intensity_norm == "p95":
            denom = max(float(np.percentile(cloud.intensity, 95)), 1.0)
            out["intensity"] = np.clip(cloud.intensity / denom, 0.0, 2.0).astype(np.float32)
            raw_imax = denom
        else:
            raw_imax = max(float(cloud.intensity.max()), 1.0)
            out["intensity"] = (cloud.intensity / raw_imax).astype(np.float32)
    if cloud.return_number is not None:
        out["return_number"] = cloud.return_number.astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    bbox = cloud.xyz[:, :2].max(0) - cloud.xyz[:, :2].min(0)
    area = max(float(bbox[0] * bbox[1]), 1.0)
    return {
        "n_points": cloud.n,
        "area_m2": area,
        "extent_m": [float(bbox[0]), float(bbox[1])],
        "intensity_raw_max": raw_imax,
        "class_counts": class_counts,
        "has_rgb": cloud.rgb is not None,
        "has_intensity": cloud.intensity is not None,
        "has_return_number": cloud.return_number is not None,
    }


def convert_dataset(name: str, train_dir: str, val_dir: str, spec: LabelSpec,
                    classes: list[dict], ignore_values: list[int],
                    staging_root: Path, progress=None) -> Path:
    """Convert both splits, write dataset_meta.json; returns the staged dataset dir.

    classes: [{"index", "source_value", "name"}] — built by the Datasets page.
    progress: optional callable(str) for live status lines.
    """
    from . import analysis

    name = sanitize_name(name)
    out_root = staging_root / name
    value_to_index = {int(c["source_value"]): int(c["index"]) for c in classes}

    say = progress or (lambda s: None)
    splits, scene_stats = {}, {"train": [], "val": []}
    for split, src_dir in (("train", train_dir), ("val", val_dir)):
        files = discover_scenes(src_dir)
        if not files:
            raise FileNotFoundError(f"No supported point-cloud files in {src_dir}")
        say(f"[{split}] {len(files)} scenes")
        scenes = []
        for path in files:
            out_path = out_root / split / (path.stem + ".npz")
            say(f"  converting {path.name} ...")
            st = convert_scene(path, spec, value_to_index, out_path)
            st["scene"] = out_path.name
            scenes.append(out_path.name)
            scene_stats[split].append(st)
        splits[split] = {"scenes": scenes,
                         "total_points": sum(s["n_points"] for s in scene_stats[split])}

    # aggregate stats + per-class counts over the real (full) conversion pass
    all_stats = scene_stats["train"] + scene_stats["val"]
    total_pts = sum(s["n_points"] for s in all_stats)
    total_area = sum(s["area_m2"] for s in all_stats)
    density = total_pts / max(total_area, 1.0)
    spacing = (total_area / max(total_pts, 1)) ** 0.5
    for c in classes:
        c["train_count"] = sum(s["class_counts"].get(int(c["index"]), 0)
                               for s in scene_stats["train"])
        c["val_count"] = sum(s["class_counts"].get(int(c["index"]), 0)
                             for s in scene_stats["val"])

    stats = {
        "mean_pts_per_m2": density,
        "mean_spacing_m": spacing,
        "mean_scene_extent_m": [
            float(np.mean([s["extent_m"][0] for s in all_stats])),
            float(np.mean([s["extent_m"][1] for s in all_stats])),
        ],
        "max_scene_points": max(s["n_points"] for s in all_stats),
        "intensity_raw_max": {s["scene"]: s["intensity_raw_max"] for s in all_stats
                              if s["intensity_raw_max"] is not None},
    }
    meta = {
        "schema_version": 1,
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "train_dir": str(train_dir), "val_dir": str(val_dir),
            "label_kind": spec.kind, "label_field": spec.field,
            "truth_dir": spec.truth_dir, "ignore_values": [int(v) for v in ignore_values],
        },
        "classes": classes,
        "num_classes": len(classes),
        "class_names": [c["name"] for c in sorted(classes, key=lambda c: int(c["index"]))],
        "has_rgb": all(s["has_rgb"] for s in all_stats),
        "has_intensity": all(s["has_intensity"] for s in all_stats),
        "has_return_number": all(s["has_return_number"] for s in all_stats),
        "splits": splits,
        "stats": stats,
    }
    meta["recommendations"] = analysis.recommend(meta)

    with open(out_root / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    say(f"staged dataset -> {out_root}")
    return out_root


def convert_infer_job(job_id: str, input_dir: str, staging_root: Path, progress=None,
                      intensity_norm: str = "p95") -> Path:
    """Label-less conversion for inference-only jobs -> <staging>/_infer/<job_id>/.

    Defaults to p95 intensity normalization to match the IEEE training scripts
    (the weights you run inference with). ponytail: assumes IEEE-style weights;
    pass intensity_norm="max" for models trained on a canonical --dataset.
    """
    say = progress or (lambda s: None)
    out_root = staging_root / "_infer" / job_id
    files = discover_scenes(input_dir)
    if not files:
        raise FileNotFoundError(f"No supported point-cloud files in {input_dir}")
    scenes, stats = [], []
    for path in files:
        out_path = out_root / "scenes" / (path.stem + ".npz")
        say(f"  converting {path.name} ...")
        st = convert_scene(path, None, {}, out_path, intensity_norm=intensity_norm)
        st["scene"] = out_path.name
        st["source_file"] = str(path)
        scenes.append(out_path.name)
        stats.append(st)
    meta = {
        "schema_version": 1,
        "job_id": job_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "scenes": scenes,
        "sources": {s["scene"]: s["source_file"] for s in stats},
    }
    with open(out_root / "job_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    say(f"staged inference job -> {out_root}")
    return out_root
