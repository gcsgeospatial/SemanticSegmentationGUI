"""Convert user folders/files into the canonical dataset the scripts consume.

Canonical layout (staged locally, then `modal volume put terminal-datasets ...`):
  <staging>/<name>/
    dataset_meta.json
    train/<scene>.npz     # xyz f32, label i32 (-1 = ignore), [rgb u8, intensity f32 0..1,
    val/<scene>.npz       #   return_number f32]
Inference jobs use the same npz minus `label`, under scenes/.

The converter is format- and layout-agnostic. The minimal case is: point at one
file (or one folder) that carries a classification field; everything else (a
held-out val folder, companion label files, custom intensity normalization) is
opt-in. When no explicit val split is given the data is split automatically:
several files -> hold out whole scenes; a single file -> cut it into a coarse
grid and hold out whole tiles (so train/val don't share neighbouring points).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .readers import SUPPORTED_EXTS, Cloud, read_points

MIN_TILE_PTS = 256   # drop grid tiles smaller than this when tile-splitting


@dataclass
class LabelSpec:
    """Where ground-truth labels come from.

    kind="field": a named field in the cloud file itself (LAS dim, PLY prop,
        "column N" of an ASCII/npy file). This is the general/default case.
    kind="file": a companion ASCII file, one label per point — the IEEE layout.
        The companion path = truth_dir / scene_name with src_suffix replaced by
        dst_suffix (e.g. "_PC3.txt" -> "_CLS.txt").
    """
    kind: str = "field"            # "field" | "file"
    field: str = ""                # for kind="field"
    truth_dir: str = ""            # for kind="file"
    src_suffix: str = "_PC3.txt"
    dst_suffix: str = "_CLS.txt"


@dataclass
class SplitConfig:
    """How to derive train/val when the user hasn't pre-split into two folders.

    strategy="auto"     1 file -> "tile", many files -> "scene".
    strategy="scene"    hold out whole files (needs >= 2 files).
    strategy="tile"     cut each file into a grid, hold out whole tiles.
    strategy="provided" caller passes an explicit val set (set via val_inputs).
    """
    strategy: str = "auto"
    val_ratio: float = 0.2
    seed: int = 42
    tile_m: float = 0.0            # 0 = auto (≈ a 5×5 grid over the scene extent)


@dataclass
class SceneSpec:
    """One output scene: a source file, optionally cropped to an XY box."""
    src: Path
    name: str
    bounds: tuple | None = None    # (x0, y0, x1, y1) or None for the whole file


def discover_scenes(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if folder.is_file():
        return [folder] if folder.suffix.lower() in SUPPORTED_EXTS else []
    top = sorted(p for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if top:
        return top
    # A converted/tiled dataset has no clouds at the top level — its scenes live
    # under train/ and val/ (the canonical npz layout). Look there so HAG, Scan
    # and Analyze accept a tiled dataset folder, not just a flat folder of clouds.
    sub = [p for split in ("train", "val") if (folder / split).is_dir()
           for p in sorted((folder / split).iterdir())
           if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sub


def expand_inputs(inputs) -> list[Path]:
    """Flatten a mix of file and folder paths into a sorted list of cloud files."""
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    out: list[Path] = []
    for p in inputs:
        out.extend(discover_scenes(p))
    return sorted(set(out))


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


# --------------------------------------------------------------------- splitting

def resolve_strategy(strategy: str, n_files: int) -> str:
    if strategy in ("scene", "tile", "provided"):
        return strategy
    return "tile" if n_files == 1 else "scene"      # "auto"


def _ratio_split(items: list, val_ratio: float, seed: int) -> tuple[list, list]:
    """Shuffle `items` and split off a val fraction; both sides get >= 1."""
    n = len(items)
    if n < 2:
        raise ValueError("scene-split needs >= 2 files; use tile-split or add a val folder")
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = min(max(1, round(val_ratio * n)), n - 1)
    val_pos = set(idx[:n_val].tolist())
    train = [items[i] for i in range(n) if i not in val_pos]
    val = [items[i] for i in range(n) if i in val_pos]
    return train, val


def density_tile_m(xyz: np.ndarray) -> float:
    """Density-based tile size (m): target ~250k points per tile, clamped to
    [25, 100] m and rounded to 5 — the same heuristic analysis.recommend uses,
    so the converter's default tiling matches the per-backbone recommendation."""
    ext = xyz[:, :2].max(0) - xyz[:, :2].min(0)
    area = max(float(ext[0] * ext[1]), 1.0)
    density = len(xyz) / area
    t = float(np.sqrt(250_000 / max(density, 0.01)))
    return float(min(max(5 * round(t / 5), 25), 100))


def grid_cells(xyz: np.ndarray, tile_m: float) -> list[tuple]:
    """Coarse XY grid over a cloud -> [(name, (x0,y0,x1,y1)), ...]. tile_m<=0
    auto-sizes from point density (~250k pts/tile)."""
    mn = xyz[:, :2].min(0)
    mx = xyz[:, :2].max(0)
    ext = mx - mn
    if tile_m <= 0:
        tile_m = density_tile_m(xyz)
    cells = []
    xs = np.arange(mn[0], mx[0] + tile_m, tile_m)[:-1] if ext[0] > 0 else [mn[0]]
    ys = np.arange(mn[1], mx[1] + tile_m, tile_m)[:-1] if ext[1] > 0 else [mn[1]]
    for i, x0 in enumerate(xs):
        for j, y0 in enumerate(ys):
            cells.append((f"r{i}_c{j}", (float(x0), float(y0),
                                         float(x0) + tile_m, float(y0) + tile_m)))
    return cells


def _crop(cloud: Cloud, raw: np.ndarray | None, bounds: tuple):
    x0, y0, x1, y1 = bounds
    m = ((cloud.xyz[:, 0] >= x0) & (cloud.xyz[:, 0] < x1) &
         (cloud.xyz[:, 1] >= y0) & (cloud.xyz[:, 1] < y1))
    sub = Cloud(
        xyz=cloud.xyz[m],
        rgb=cloud.rgb[m] if cloud.rgb is not None else None,
        intensity=cloud.intensity[m] if cloud.intensity is not None else None,
        return_number=cloud.return_number[m] if cloud.return_number is not None else None,
        fields={k: v[m] for k, v in cloud.fields.items()},
    )
    return sub, (raw[m] if raw is not None else None)


# --------------------------------------------------------------------- conversion

def _hag_from_cloud(cloud: Cloud) -> np.ndarray | None:
    """Return a source HeightAboveGround/HAG field when one exists and aligns 1:1.

    `pretrain.add_hag()` writes LAS/LAZ with a `HeightAboveGround` extra dim;
    readers.py exposes that as a numeric field. Preserve it into the canonical
    npz so *_hag trainers get the real HAG feature instead of their z-min proxy.
    """
    for name, arr in cloud.fields.items():
        key = name.lower().replace("_", "")
        if key in ("heightaboveground", "hag"):
            h = np.asarray(arr, dtype=np.float32).reshape(-1)
            if len(h) == cloud.n:
                return h
    return None


def _convert_one(cloud: Cloud, raw: np.ndarray | None, value_to_index: dict[int, int],
                 out_path: Path, intensity_norm: str = "max",
                 compute_hag: bool = False) -> dict:
    """Write one (already-read, already-cropped) cloud to a canonical npz.

    compute_hag (inference only): also store a per-point real HeightAboveGround
    ("hag", SMRF->hag_nn) aligned to xyz, for running *_hag weights trained on real
    HAG. Silently skipped (the infer loaders fall back to a z-min proxy) when PDAL
    can't produce an aligned result."""
    out: dict[str, np.ndarray] = {"xyz": cloud.xyz.astype(np.float32)}

    class_counts: dict[int, int] = {}
    if raw is not None:
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
    source_hag = _hag_from_cloud(cloud)
    if source_hag is not None:
        out["hag"] = source_hag.astype(np.float32)
    if compute_hag and "hag" not in out:
        from . import pretrain
        h = pretrain.hag_for_cloud(cloud)
        if h is not None and len(h) == cloud.n:
            out["hag"] = h.astype(np.float32)

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
        "has_hag": "hag" in out,
    }


def convert_scene(path: Path, spec: LabelSpec | None, value_to_index: dict[int, int],
                  out_path: Path, intensity_norm: str = "max",
                  compute_hag: bool = False) -> dict:
    """Read one source file and convert the whole cloud (no cropping)."""
    cloud = read_points(path)
    raw = read_labels(path, cloud, spec) if spec is not None else None
    return _convert_one(cloud, raw, value_to_index, out_path, intensity_norm,
                        compute_hag=compute_hag)


def _plan_and_convert(train_files: list[Path], val_files: list[Path] | None,
                      strategy: str, split: SplitConfig, spec: LabelSpec | None,
                      value_to_index: dict[int, int], out_root: Path,
                      intensity_norm: str, say) -> dict:
    """Drive conversion per strategy; returns {"train": [stats], "val": [stats]}.
    Each big file is read at most once."""
    stats = {"train": [], "val": []}

    def emit(split_name: str, cloud: Cloud, raw, scene_name: str):
        out_path = out_root / split_name / f"{scene_name}.npz"
        st = _convert_one(cloud, raw, value_to_index, out_path, intensity_norm)
        st["scene"] = out_path.name
        stats[split_name].append(st)

    if strategy == "provided":
        for split_name, files in (("train", train_files), ("val", val_files or [])):
            say(f"[{split_name}] {len(files)} scenes")
            for f in files:
                say(f"  converting {f.name} ...")
                cloud = read_points(f)
                raw = read_labels(f, cloud, spec) if spec is not None else None
                emit(split_name, cloud, raw, f.stem)
        return stats

    if strategy == "scene":
        tr, va = _ratio_split(train_files, split.val_ratio, split.seed)
        say(f"scene-split: {len(tr)} train / {len(va)} val scenes")
        for split_name, files in (("train", tr), ("val", va)):
            for f in files:
                say(f"  converting {f.name} ...")
                cloud = read_points(f)
                raw = read_labels(f, cloud, spec) if spec is not None else None
                emit(split_name, cloud, raw, f.stem)
        return stats

    # strategy == "tile": grid each file, hold out whole tiles.
    for f in train_files:
        say(f"  tiling {f.name} ...")
        cloud = read_points(f)
        raw = read_labels(f, cloud, spec) if spec is not None else None
        kept = []
        for cname, bounds in grid_cells(cloud.xyz, split.tile_m):
            sub, sub_raw = _crop(cloud, raw, bounds)
            if sub.n >= MIN_TILE_PTS:
                kept.append((cname, sub, sub_raw))
        if len(kept) < 2:
            raise ValueError(
                f"{f.name}: tile-split produced {len(kept)} usable tile(s) "
                f"(need >= 2). Use a smaller tile size, add more data, or supply "
                f"a separate validation folder.")
        val_pos = set(_split_indices(len(kept), split.val_ratio, split.seed))
        n_tr = len(kept) - len(val_pos)
        say(f"  {f.stem}: {len(kept)} tiles -> {n_tr} train / {len(val_pos)} val")
        for k, (cname, sub, sub_raw) in enumerate(kept):
            split_name = "val" if k in val_pos else "train"
            emit(split_name, sub, sub_raw, f"{f.stem}_{cname}")
    return stats


def _split_indices(n: int, val_ratio: float, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = min(max(1, round(val_ratio * n)), n - 1)
    return idx[:n_val].tolist()


def convert_dataset(name: str, inputs, spec: LabelSpec | None,
                    classes: list[dict], ignore_values: list[int],
                    staging_root: Path, *, val_inputs=None,
                    split: SplitConfig | None = None,
                    intensity_norm: str = "max", progress=None) -> Path:
    """Convert `inputs` (files and/or folders) into a staged canonical dataset.

    inputs: a path or list of paths (files or folders) to use as the source.
    val_inputs: optional explicit validation source -> "provided" split.
    split: how to derive train/val when val_inputs is None (default: auto).
    classes: [{"index", "source_value", "name"}] — built by the Datasets page.
    """
    from . import analysis

    split = split or SplitConfig()
    name = sanitize_name(name)
    out_root = staging_root / name
    value_to_index = {int(c["source_value"]): int(c["index"]) for c in classes}
    say = progress or (lambda s: None)

    train_files = expand_inputs(inputs)
    if not train_files:
        raise FileNotFoundError(f"No supported point-cloud files in {inputs}")
    if val_inputs:
        strategy = "provided"
        val_files = expand_inputs(val_inputs)
        if not val_files:
            raise FileNotFoundError(f"No supported point-cloud files in {val_inputs}")
    else:
        strategy = resolve_strategy(split.strategy, len(train_files))
        val_files = None
    say(f"source: {len(train_files)} file(s), split strategy = {strategy}")

    scene_stats = _plan_and_convert(train_files, val_files, strategy, split, spec,
                                    value_to_index, out_root, intensity_norm, say)
    if not scene_stats["train"] or not scene_stats["val"]:
        raise ValueError("Conversion produced an empty train or val split — check "
                         "the split settings or supply an explicit val folder.")

    splits = {sp: {"scenes": [s["scene"] for s in scene_stats[sp]],
                   "total_points": sum(s["n_points"] for s in scene_stats[sp])}
              for sp in ("train", "val")}

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

    # Several source values may be combined into one class (same index/name), so
    # the number of classes is the count of UNIQUE indices, not table rows.
    idx_to_name: dict[int, str] = {}
    for c in classes:
        idx_to_name.setdefault(int(c["index"]), c["name"])

    # Collapse combined classes into ONE entry per index (counts are keyed by
    # index, so every combined row already carries the full total — take it once
    # and gather the source values). This is what the rare-class warnings and the
    # UI read, so a combined class shows added-together, not duplicated.
    by_index: dict[int, dict] = {}
    for c in classes:
        i = int(c["index"])
        e = by_index.get(i)
        if e is None:
            by_index[i] = {"index": i, "name": c["name"], "source_values": [int(c["source_value"])],
                           "train_count": int(c["train_count"]), "val_count": int(c["val_count"])}
        else:
            e["source_values"].append(int(c["source_value"]))
    classes = [by_index[i] for i in sorted(by_index)]

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
            "inputs": [str(p) for p in (inputs if isinstance(inputs, (list, tuple)) else [inputs])],
            "val_inputs": [str(p) for p in (val_inputs or [])],
            "split_strategy": strategy,
            "val_ratio": split.val_ratio, "split_seed": split.seed,
            "tile_m": split.tile_m,
            "label_kind": spec.kind if spec else None,
            "label_field": spec.field if spec else "",
            "truth_dir": spec.truth_dir if spec else "",
            "intensity_norm": intensity_norm,
            "hag_source": "source_dimension" if all(s["has_hag"] for s in all_stats) else "",
            "ignore_values": [int(v) for v in ignore_values],
        },
        "classes": classes,
        "num_classes": len(idx_to_name),
        "class_names": [idx_to_name[i] for i in sorted(idx_to_name)],
        "has_rgb": all(s["has_rgb"] for s in all_stats),
        "has_intensity": all(s["has_intensity"] for s in all_stats),
        "has_return_number": all(s["has_return_number"] for s in all_stats),
        "has_hag": all(s["has_hag"] for s in all_stats),
        "splits": splits,
        "stats": stats,
    }
    meta["recommendations"] = analysis.recommend(meta)

    with open(out_root / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    say(f"staged dataset -> {out_root}")
    return out_root


def convert_infer_job(job_id: str, input_dir: str, staging_root: Path, progress=None,
                      intensity_norm: str = "p95", hag: bool = False) -> Path:
    """Label-less conversion for inference-only jobs -> <staging>/_infer/<job_id>/.

    intensity_norm MUST match what the weights were trained with (max -> [0,1] for
    canonical --dataset runs; p95 -> [0,2] for the IEEE scripts) — a mismatch feeds
    the net out-of-distribution intensity and tanks accuracy for every checkpoint.
    `hag=True` (for *_hag weights trained on real PDAL HeightAboveGround) computes a
    per-point "hag" channel via SMRF->hag_nn when PDAL is available, so inference
    reproduces the trained feature instead of a z-min proxy; falls back to the proxy
    (with a warning) when PDAL is missing.
    """
    say = progress or (lambda s: None)
    if hag:
        from . import pretrain
        if pretrain.pdal_available():
            say("  computing real PDAL HeightAboveGround (SMRF -> hag_nn) per scene …")
        else:
            say("  ⚠ HAG requested but PDAL isn't installed here — using a z-min proxy "
                "(degraded for a model trained on real HAG).")
            hag = False
    out_root = staging_root / "_infer" / job_id
    files = discover_scenes(input_dir)
    if not files:
        raise FileNotFoundError(f"No supported point-cloud files in {input_dir}")
    scenes, stats = [], []
    for path in files:
        out_path = out_root / "scenes" / (path.stem + ".npz")
        say(f"  converting {path.name} ...")
        st = convert_scene(path, None, {}, out_path, intensity_norm=intensity_norm,
                           compute_hag=hag)
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


def add_hag_to_dataset(src_dir, out_dir, *, skip_ground: bool = False,
                       hag_filter: str = "hag_nn", progress=None) -> Path:
    """Add a per-tile HeightAboveGround channel to an already-converted dataset.

    Reads each train|val/*.npz, recomputes HAG with PDAL (SMRF -> hag) on the
    tile's OWN points (the inference twin pretrain.hag_for_cloud, kept in RAM),
    and writes a sibling dataset with a `hag` key added to every tile. Returns
    out_dir.

    Per-tile ground is weaker than whole-cloud SMRF on tiles with little bare
    earth; tiles where PDAL can't return an aligned result keep no `hag` key and
    the *_hag trainer falls back to its z-min proxy for them, so `has_hag` records
    whether ALL tiles got real HAG.
    """
    from . import pretrain

    say = progress or (lambda s: None)
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    meta_path = src_dir / "dataset_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{src_dir} is not a converted dataset (no dataset_meta.json)")
    if not pretrain.pdal_available():
        raise RuntimeError("PDAL is not installed — cannot compute HeightAboveGround.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    n_tiles = n_hag = 0
    for split in ("train", "val"):
        sdir = src_dir / split
        if not sdir.is_dir():
            continue
        tiles = sorted(sdir.glob("*.npz"))
        say(f"[{split}] {len(tiles)} tile(s)")
        for npz_path in tiles:
            with np.load(npz_path) as z:
                data = {k: z[k] for k in z.files}
            cloud = Cloud(
                xyz=data["xyz"].astype(np.float64),
                rgb=data.get("rgb"),
                intensity=data.get("intensity"),
                return_number=data.get("return_number"),
            )
            h = pretrain.hag_for_cloud(cloud, skip_ground=skip_ground, hag_filter=hag_filter)
            if h is not None and len(h) == cloud.n:
                data["hag"] = h.astype(np.float32)
                n_hag += 1
            out_path = out_dir / split / npz_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, **data)
            n_tiles += 1
            say(f"  {split}/{npz_path.name}: "
                f"{'HAG' if 'hag' in data else 'proxy'} ({cloud.n:,} pts)")
    if n_tiles == 0:
        raise ValueError(f"No npz tiles under {src_dir}/train|val — convert the dataset first.")

    meta["name"] = sanitize_name(out_dir.name)
    meta["has_hag"] = (n_hag == n_tiles)
    meta.setdefault("source", {})["hag_source"] = "per_tile_smrf"
    meta["created_utc"] = datetime.now(timezone.utc).isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    say(f"✓ HAG dataset -> {out_dir}  ({n_hag}/{n_tiles} tiles with real HAG)")
    return out_dir


# --------------------------------------------------------------------- self-check

def _selfcheck():
    """Synthesize a single labeled cloud, tile-split it, and assert the splits
    partition the points (no leakage, none lost above the tile floor)."""
    import tempfile

    rng = np.random.RandomState(0)
    n = 20000
    xyz = rng.uniform(0, 100, size=(n, 3)).astype(np.float64)
    cls = rng.randint(0, 3, size=n).astype(np.float64)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        np.savez(td / "scene.npz", xyz=xyz, classification=cls)
        classes = [{"index": i, "source_value": i, "name": f"c{i}"} for i in range(3)]
        out = convert_dataset("selftest", td / "scene.npz",
                              LabelSpec(kind="field", field="classification"),
                              classes, [], td / "staging",
                              split=SplitConfig(strategy="tile", tile_m=20.0))
        meta = json.loads((out / "dataset_meta.json").read_text())
        tr, va = meta["splits"]["train"], meta["splits"]["val"]
        assert tr["scenes"] and va["scenes"], "both splits must be non-empty"
        assert tr["total_points"] + va["total_points"] == n, "points must be conserved"
        # scene-split path
        for k in range(3):
            np.savez(td / f"s{k}.npz", xyz=xyz, classification=cls)
        out2 = convert_dataset("selftest2", td, LabelSpec(field="classification"),
                               classes, [], td / "staging",
                               split=SplitConfig(strategy="scene"))
        m2 = json.loads((out2 / "dataset_meta.json").read_text())
        assert m2["splits"]["train"]["scenes"] and m2["splits"]["val"]["scenes"]
    print("dataset self-check OK")


if __name__ == "__main__":
    _selfcheck()
