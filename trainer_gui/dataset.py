"""Convert user folders/files into the canonical dataset the scripts consume.

Layout: <staging>/<name>/{dataset_meta.json, train|val|test/<scene>.npz} — xyz f64,
label i32 (-1 = ignore), optional rgb u8 / intensity f32 / return_number f32.
Inference jobs use the same npz minus `label`, under scenes/.
The split is decided ONCE here (point-count fractions over whole scenes, or tiles
of a single cloud reassembled into holey per-split clouds) and recorded in
dataset_meta.json; trainers read the folders verbatim.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .readers import (SUPPORTED_EXTS, Cloud, crs_unit_factor, list_label_fields,
                      read_points)


@dataclass
class LabelSpec:
    """Label source. kind="field": named field in the cloud. kind="file": companion
    file at truth_dir / (scene name with src_suffix -> dst_suffix), one label/point."""
    kind: str = "field"            # "field" | "file"
    field: str = ""                # for kind="field"
    truth_dir: str = ""            # for kind="file"
    src_suffix: str = "_PC3.txt"
    dst_suffix: str = "_CLS.txt"


@dataclass
class SplitConfig:
    """val_frac/test_frac: target point-count fractions. mode "balanced" mirrors
    the global class mix (+ rare-class presence); "random" fills by points.
    seam_buffer_m/tile_m: single-cloud only. strategy "provided" = explicit folders."""
    val_frac: float = 0.15
    test_frac: float = 0.15
    mode: str = "balanced"        # "balanced" | "random"
    seed: int = 42
    seam_buffer_m: float = 1.0
    tile_m: float = 50.0
    strategy: str = "auto"        # "auto" | "provided"


def discover_scenes(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if folder.is_file():
        return [folder] if folder.suffix.lower() in SUPPORTED_EXTS else []
    top = sorted(p for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if top:
        return top
    # converted datasets keep their scenes under train/val/test
    sub = [p for split in ("train", "val", "test") if (folder / split).is_dir()
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


def feat_key(field: str) -> str:
    """Raw source-field name -> the npz key suffix (feat_<this>). Lowercase,
    non-alphanumeric runs collapse to one "_", edges stripped."""
    s = re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")
    if not s:
        raise ValueError(f"feature field name '{field}' is empty after sanitizing")
    return s


# intensity/return_number keep canonical npz keys (not feat_*); baking is still opt-in.
_CANON_SPELLINGS = {
    "intensity": "intensity", "scalar_intensity": "intensity",
    "return_number": "return_number", "returnnumber": "return_number",
    "scalar_return_number": "return_number", "scalar_returnnumber": "return_number",
    "ret_num": "return_number",
}


def canonical_channel(field: str) -> str | None:
    """Field name -> canonical channel name, or None for an ordinary feat_* field."""
    return _CANON_SPELLINGS.get((field or "").strip().lower().replace(" ", "_"))


def split_feature_fields(fields: list[str] | None) -> tuple[set[str], list[str]]:
    """Prep selection -> (canonical channels wanted, remaining feat_* fields)."""
    canon = {c for c in (canonical_channel(f) for f in (fields or [])) if c}
    rest = [f for f in (fields or []) if canonical_channel(f) is None]
    return canon, rest


def sanitize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_")
    if not s:
        raise ValueError("Dataset name is empty after sanitizing")
    return s.lower()


# ---- splitting: atoms = whole scenes or grid tiles; targets = point-count fractions

_SPLITS = ("train", "val", "test")


def _subset(cloud: Cloud, raw: np.ndarray | None, mask: np.ndarray):
    """A boolean-masked copy of a cloud (+ its labels)."""
    sub = Cloud(
        xyz=cloud.xyz[mask],
        rgb=cloud.rgb[mask] if cloud.rgb is not None else None,
        intensity=cloud.intensity[mask] if cloud.intensity is not None else None,
        return_number=cloud.return_number[mask] if cloud.return_number is not None else None,
        fields={k: v[mask] for k, v in cloud.fields.items()},
        crs_wkt=cloud.crs_wkt,
    )
    return sub, (raw[mask] if raw is not None else None)


def _tile_hists(inv: np.ndarray, n_tiles: int, raw, value_to_index: dict,
                num_classes: int) -> np.ndarray:
    """(n_tiles, C) per-class point counts, one vectorized pass."""
    hist = np.zeros((n_tiles, max(num_classes, 1)), dtype=np.int64)
    if raw is None or num_classes == 0:
        return hist[:, :num_classes] if num_classes else np.zeros((n_tiles, 0), np.int64)
    idx = np.full(len(raw), -1, dtype=np.int64)
    for src_val, i in value_to_index.items():
        if 0 <= int(i) < num_classes:
            idx[raw == src_val] = int(i)
    v = idx >= 0
    return np.bincount(inv[v] * num_classes + idx[v],
                       minlength=n_tiles * num_classes).reshape(n_tiles, num_classes)


def _hist_from_counts(class_counts: dict, num_classes: int) -> np.ndarray:
    """Per-class counts from a {index: count} dict (the _convert_one stat)."""
    h = np.zeros(num_classes, dtype=np.int64)
    for c, cnt in class_counts.items():
        if 0 <= int(c) < num_classes:
            h[int(c)] = int(cnt)
    return h


def _guarantee_presence(assign, hist, allowed):
    """Best-effort: move atoms so every globally-present class appears in every active split."""
    n, C = hist.shape
    H = hist.sum(0)
    for c in range(C):
        if H[c] <= 0:
            continue
        for s in range(3):
            if not allowed[s]:
                continue
            if any(hist[i, c] > 0 and assign[i] == s for i in range(n)):
                continue
            cand = [i for i in range(n) if hist[i, c] > 0 and assign[i] != s]
            if not cand:
                continue
            # prefer an atom whose source split keeps class c after it leaves
            keeps = lambda i: any(hist[j, c] > 0 and assign[j] == assign[i] and j != i
                                  for j in range(n))
            cand.sort(key=lambda i: (not keeps(i), -hist[i, c]))
            assign[cand[0]] = s


def _guarantee_nonempty(assign, pts, fr):
    """Every split with a positive target gets >= 1 atom."""
    pts = np.asarray(pts, dtype=np.float64)
    for s in range(3):
        if fr[s] <= 0 or (assign == s).any():
            continue
        donor = max(range(3), key=lambda d: int((assign == d).sum()))
        cand = np.where(assign == donor)[0]
        if len(cand) <= 1:
            continue
        assign[cand[int(np.argmin(pts[cand]))]] = s


def allocate_splits(pts, hist, val_frac: float, test_frac: float,
                    mode: str = "balanced", seed: int = 42) -> np.ndarray:
    """Assign atoms to train(0)/val(1)/test(2) approximating the point fractions.
    "random": greedy fill by point deficit. "balanced": rarity-first iterative
    stratification so each split mirrors the global class mix."""
    pts = np.asarray(pts, dtype=np.float64)
    hist = np.asarray(hist, dtype=np.float64)
    n = len(pts)
    C = hist.shape[1] if (hist.ndim == 2 and hist.size) else 0
    fr = np.array([max(0.0, 1.0 - val_frac - test_frac), val_frac, test_frac])
    allowed = fr > 0                                   # never assign to a 0-target split
    rng = np.random.RandomState(int(seed))
    assign = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return assign

    if mode != "balanced" or C == 0:
        target = fr * float(pts.sum())
        got = np.zeros(3)
        for i in rng.permutation(n):
            deficit = target - got
            deficit[~allowed] = -np.inf
            s = int(np.argmax(deficit))
            assign[i] = s
            got[s] += pts[i]
    else:
        H = hist.sum(0)
        target = np.outer(fr, H)                       # (3, C) point-count targets
        got = np.zeros((3, C))
        # critical class per atom = its rarest present class; feed the most-starved split
        crit = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            pres = np.where(hist[i] > 0)[0]
            if len(pres):
                crit[i] = int(pres[np.argmin(H[pres])])
        rarest = np.array([H[crit[i]] if crit[i] >= 0 else np.inf for i in range(n)])
        order = np.lexsort((-pts, rarest))             # rarest class first, big first
        for i in order:
            need = target - got
            primary = need[:, crit[i]] if crit[i] >= 0 else need.sum(1)
            key = primary + 1e-9 * need.sum(1) + rng.random_sample(3) * 1e-12
            key[~allowed] = -np.inf
            s = int(np.argmax(key))
            assign[i] = s
            got[s] += hist[i]
        _guarantee_presence(assign, hist.astype(np.int64), allowed)

    _guarantee_nonempty(assign, pts, fr)
    return assign


def _atomize(cloud: Cloud, tile_m: float):
    """XY grid tiles for a single cloud -> (keys, uniq, inv)."""
    xy = cloud.xyz[:, :2]
    keys = np.floor((xy - xy.min(0)) / max(float(tile_m), 1e-6)).astype(np.int64)
    # 1-D packed code: ~10x faster than unique(axis=0) on multi-M clouds
    span = int(keys[:, 1].max()) + 1
    code, inv = np.unique(keys[:, 0] * span + keys[:, 1], return_inverse=True)
    uniq = np.column_stack([code // span, code % span])
    return keys, uniq, inv


def _seam_drop(xy, keys, uniq, split_of_point, tile_m: float,
               seam_buffer_m: float) -> np.ndarray:
    """Keep-mask: drop points within seam_buffer_m of a neighbouring tile in a
    different split. ponytail: O(n) tile-edge approximation — can over-drop
    slightly, never leaks; effective buffer caps at tile_m."""
    n = len(xy)
    keep = np.ones(n, dtype=bool)
    if seam_buffer_m <= 0:
        return keep
    tile = max(float(tile_m), 1e-6)
    b = min(float(seam_buffer_m), tile)
    k0 = uniq.min(0)
    grid = np.full(uniq.max(0) - k0 + 3, -1, dtype=np.int64)   # +1 pad each side; -1 = empty
    kx = keys[:, 0] - k0[0] + 1
    ky = keys[:, 1] - k0[1] + 1
    grid[kx, ky] = split_of_point
    u = xy - xy.min(0) - keys * tile                           # offset in [0, tile)
    near = {-1: (u < b), 1: (u > tile - b), 0: np.ones((n, 2), dtype=bool)}
    drop = np.zeros(n, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nb = grid[kx + dx, ky + dy]
            drop |= (near[dx][:, 0] & near[dy][:, 1]
                     & (nb >= 0) & (nb != split_of_point))
    return ~drop


# --------------------------------------------------------------------- conversion

def _crs_check(crs_wkt: str | None, xyz: np.ndarray, name: str) -> str | None:
    """The pipeline is meter-denominated: geographic CRS raises, non-meter units
    (or degree-looking coords with no CRS) return a warning string, else None."""
    if crs_wkt:
        try:
            from pyproj import CRS
            crs = CRS.from_wkt(crs_wkt)
        except Exception:
            return None
        if crs.is_geographic:
            raise ValueError(
                f"{name}: cloud is in a geographic CRS ({crs.name}) — coordinates "
                "are lon/lat DEGREES, but tile sizes, radii and grids are meters. "
                "Reproject to a projected CRS (e.g. UTM) and re-convert.")
        try:
            unit = crs.axis_info[0].unit_name
            factor = float(crs.axis_info[0].unit_conversion_factor)
        except (AttributeError, IndexError):
            return None
        if abs(factor - 1.0) > 1e-6:
            # readers._read_las already scaled xyz by this factor at ingest.
            return (f"ℹ {name}: CRS unit is '{unit}' ({factor:g} m) — coordinates "
                    f"auto-scaled to meters for processing; exports restore the "
                    f"source units.")
        return None
    xy = xyz[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    if ((hi - lo).max() < 10.0 and abs(lo[0]) <= 360 and abs(hi[0]) <= 360
            and abs(lo[1]) <= 90 and abs(hi[1]) <= 90):
        return (f"⚠ {name}: no CRS, and the coordinates fit lon/lat bounds with a "
                "tiny extent — if these are degrees, reproject to a projected "
                "(meter) CRS before converting; meter-denominated settings would "
                "otherwise be wrong.")
    return None


def _hag_from_cloud(cloud: Cloud) -> np.ndarray | None:
    """Source HeightAboveGround/HAG field when one exists and aligns 1:1 — wins
    over recomputation; stored as feat_hag."""
    for name, arr in cloud.fields.items():
        key = name.lower().replace("_", "")
        if key in ("heightaboveground", "hag"):
            h = np.asarray(arr, dtype=np.float32).reshape(-1)
            if len(h) == cloud.n:
                return h
    return None


def _apply_rgb_mapping(cloud: Cloud, rgb_fields) -> Cloud:
    """Explicit-only color: set rgb from the mapped columns, or strip reader-auto
    color when unmapped. The inference path never calls this (keeps auto color)."""
    if not rgb_fields:
        cloud.rgb = None
        return cloud
    cols = []
    for fld in rgb_fields:
        key = next((k for k in cloud.fields if k.lower() == str(fld).lower()), None)
        if key is None:
            raise ValueError(f"RGB column '{fld}' not in the cloud "
                             f"(have: {sorted(cloud.fields)})")
        cols.append(np.asarray(cloud.fields[key], np.float64).reshape(-1))
    c = np.column_stack(cols)
    if c.max() > 255:            # 16-bit color
        c = c / 257.0
    elif 0 < c.max() <= 1.0:     # unit-range floats
        c = c * 255.0
    cloud.rgb = np.clip(c, 0, 255).astype(np.uint8)
    return cloud


def _convert_one(cloud: Cloud, raw: np.ndarray | None, value_to_index: dict[int, int],
                 out_path: Path, intensity_norm: str = "p95",
                 compute_hag: bool = False, ground_value: int | None = None,
                 hag_filter: str = "grid",
                 feature_fields: list[str] | None = None,
                 geo_features: list[str] | None = None,
                 geo_radius: float = 1.0) -> dict:
    """Write one (already-read, already-cropped) cloud to a canonical npz.
    compute_hag stores feat_hag when an aligned result is possible; a scene
    without it blocks runs that require the channel."""
    # xyz stays float64 — UTM-scale coords quantize to 0.5 m in float32;
    # trainers origin-shift to a local frame before their own casts.
    crs_warning = _crs_check(cloud.crs_wkt, cloud.xyz, out_path.stem)
    out: dict[str, np.ndarray] = {"xyz": cloud.xyz.astype(np.float64)}
    if cloud.crs_wkt:
        # 0-d unicode array, ferried scene npz -> pred npz -> export add_crs
        out["crs_wkt"] = np.asarray(cloud.crs_wkt)

    class_counts: dict[int, int] = {}
    # no class mapping = inference: raw only locates ground, don't write labels
    if raw is not None and value_to_index:
        label = np.full(cloud.n, -1, dtype=np.int32)
        for src_val, idx in value_to_index.items():
            label[raw == src_val] = idx
        out["label"] = label
        vals, cnts = np.unique(label[label >= 0], return_counts=True)
        class_counts = {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}

    if cloud.rgb is not None:
        out["rgb"] = cloud.rgb
    # canonical channels bake only when selected — no implicit channels
    canon_wanted, feature_fields = split_feature_fields(feature_fields)
    raw_imax = None
    if "intensity" in canon_wanted and cloud.intensity is not None:
        if intensity_norm == "p95":
            denom = max(float(np.percentile(cloud.intensity, 95)), 1.0)
            out["intensity"] = np.clip(cloud.intensity / denom, 0.0, 2.0).astype(np.float32)
            raw_imax = denom
        else:
            raw_imax = max(float(cloud.intensity.max()), 1.0)
            out["intensity"] = (cloud.intensity / raw_imax).astype(np.float32)
    if "return_number" in canon_wanted and cloud.return_number is not None:
        out["return_number"] = cloud.return_number.astype(np.float32)
    source_hag = _hag_from_cloud(cloud)
    if source_hag is not None:
        out["feat_hag"] = source_hag.astype(np.float32)
    if compute_hag and "feat_hag" not in out:
        from . import pretrain
        gmask = (raw == int(ground_value)) if (raw is not None and ground_value is not None) else None
        h = pretrain.hag_for_cloud(cloud, ground_mask=gmask, hag_filter=hag_filter)
        if h is not None and len(h) == cloud.n:
            out["feat_hag"] = h.astype(np.float32)

    # feat_geo_<nm>, stored raw; a ferried whole-cloud value wins over recomputation
    if geo_features:
        from . import pretrain
        have = {nm: np.asarray(cloud.fields[f"feat_geo_{nm.lower()}"], np.float32)
                for nm in geo_features if f"feat_geo_{nm.lower()}" in cloud.fields}
        todo = [nm for nm in geo_features if nm not in have]
        if todo:
            have.update(pretrain.geo_features_for_cloud(cloud.xyz, todo, geo_radius))
        for nm in geo_features:
            out[f"feat_geo_{nm.lower()}"] = have[nm]

    # feat_<name>: v/p95(|v|) clipped to [-2, 2]; a missing field fails loud
    feature_scales: dict[str, float] = {}
    for fld in (feature_fields or []):
        key = next((k for k in cloud.fields if k.lower() == fld.lower()), None)
        if key is None:
            raise ValueError(f"{out_path.stem}: feature field '{fld}' not found "
                             f"(have: {sorted(cloud.fields)})")
        v = np.asarray(cloud.fields[key])
        if v.ndim != 1 or not np.issubdtype(v.dtype, np.number):
            raise ValueError(f"{out_path.stem}: feature field '{fld}' is not numeric 1-D "
                             f"(dtype {v.dtype}, shape {v.shape})")
        v = v.astype(np.float32)
        scale = max(float(np.percentile(np.abs(v), 95)), 1e-6)
        out[f"feat_{feat_key(fld)}"] = np.clip(v / scale, -2.0, 2.0).astype(np.float32)
        feature_scales[feat_key(fld)] = scale

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
        "has_intensity": "intensity" in out,
        "has_return_number": "return_number" in out,
        "has_hag": "feat_hag" in out,
        "feature_scales": feature_scales,
        "crs_wkt": cloud.crs_wkt,
        "crs_warning": crs_warning,
    }


def convert_scene(path: Path, spec: LabelSpec | None, value_to_index: dict[int, int],
                  out_path: Path, intensity_norm: str = "p95",
                  compute_hag: bool = False, ground_value: int | None = None,
                  hag_filter: str = "grid",
                  feature_fields: list[str] | None = None,
                  geo_features: list[str] | None = None,
                  geo_radius: float = 1.0) -> dict:
    """Read one source file and convert the whole cloud (no cropping)."""
    cloud = read_points(path)
    raw = read_labels(path, cloud, spec) if spec is not None else None
    return _convert_one(cloud, raw, value_to_index, out_path, intensity_norm,
                        compute_hag=compute_hag, ground_value=ground_value,
                        hag_filter=hag_filter, feature_fields=feature_fields,
                        geo_features=geo_features, geo_radius=geo_radius)


def _available_ram_bytes() -> int | None:
    """Available physical RAM (bytes), stdlib only; None when undeterminable."""
    try:
        if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = _MS()
            m.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return int(m.ullAvailPhys)
        except Exception:                             # noqa: BLE001 — any ctypes failure = unknown
            pass
    return None


# fat per-worker RAM estimate (LAZ inflates 10-20x + transient copies);
# overshooting only costs parallelism, never a crash
_RAM_PER_FILE_FACTOR = 30
_RAM_HEADROOM = 0.7


def _worker_cap_detail(files: list[Path]) -> tuple[int, str]:
    """(thread count that won't OOM, human reason for the log)."""
    cores = os.cpu_count() or 4
    cap = min(cores, max(len(files), 1))
    reason = f"cores={cores}, files={len(files)}"
    ram = _available_ram_bytes()
    if ram and files:
        biggest = max((f.stat().st_size for f in files), default=0)
        if biggest > 0:
            mem_cap = max(1, int(ram * _RAM_HEADROOM // (biggest * _RAM_PER_FILE_FACTOR)))
            if mem_cap < cap:
                cap = mem_cap
                reason = (f"RAM-limited: {ram / 1e9:.1f} GB free, "
                          f"~{biggest * _RAM_PER_FILE_FACTOR / 1e9:.1f} GB/worker")
    return cap, reason


def _convert_many(files: list[Path], dest_for, spec, value_to_index, intensity_norm, say,
                  *, compute_hag, ground_value, hag_filter,
                  feature_fields: list[str] | None = None,
                  geo_features: list[str] | None = None, geo_radius: float = 1.0,
                  rgb_fields: list[str] | None = None,
                  max_workers: int | None = None) -> list[dict]:
    """Convert each file concurrently; stat dicts return in INPUT order (the caller
    feeds allocate_splits positionally). Threads: read/PDAL/savez drop the GIL."""
    def work(f: Path) -> dict:
        cloud = _apply_rgb_mapping(read_points(f), rgb_fields)
        raw = read_labels(f, cloud, spec) if spec is not None else None
        out_path = dest_for(f)
        st = _convert_one(cloud, raw, value_to_index, out_path, intensity_norm,
                          compute_hag=compute_hag, ground_value=ground_value,
                          hag_filter=hag_filter, feature_fields=feature_fields,
                          geo_features=geo_features, geo_radius=geo_radius)
        st["scene"] = out_path.name
        return st

    if max_workers is not None:
        workers, why = max_workers, "forced"
    else:
        workers, why = _worker_cap_detail(files)
    mode = "PARALLEL" if workers > 1 else "SERIAL"
    say(f"  {mode}: {workers} worker(s) [{why}]")
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f, st in zip(files, ex.map(work, files)):   # map preserves input order
            say(f"  converted {f.name}")
            out.append(st)
    return out


def _plan_and_convert(input_files: list[Path], val_files: list[Path] | None,
                      test_files: list[Path] | None, split: SplitConfig,
                      spec: LabelSpec | None, value_to_index: dict[int, int],
                      num_classes: int, out_root: Path, intensity_norm: str, say, *,
                      compute_hag: bool = False, ground_value: int | None = None,
                      hag_filter: str = "grid",
                      feature_fields: list[str] | None = None,
                      geo_features: list[str] | None = None, geo_radius: float = 1.0,
                      rgb_fields: list[str] | None = None,
                      max_workers: int | None = None) -> dict:
    """Convert sources into out_root/{train,val,test}/*.npz; returns per-split
    stat lists. Explicit val/test folders are used verbatim; the rest is allocated."""
    stats = {sp: [] for sp in _SPLITS}

    def emit(split_name: str, cloud: Cloud, raw, scene_name: str, hag_already=False):
        out_path = out_root / split_name / f"{scene_name}.npz"
        st = _convert_one(cloud, raw, value_to_index, out_path, intensity_norm,
                          compute_hag=compute_hag and not hag_already,
                          ground_value=ground_value, hag_filter=hag_filter,
                          feature_fields=feature_fields,
                          geo_features=geo_features, geo_radius=geo_radius)
        st["scene"] = out_path.name
        stats[split_name].append(st)

    for split_name, files in (("val", val_files), ("test", test_files)):
        if files:
            stats[split_name].extend(_convert_many(
                files, lambda f, sn=split_name: out_root / sn / f"{f.stem}.npz",
                spec, value_to_index, intensity_norm, say, compute_hag=compute_hag,
                ground_value=ground_value, hag_filter=hag_filter,
                feature_fields=feature_fields, geo_features=geo_features,
                geo_radius=geo_radius, rgb_fields=rgb_fields,
                max_workers=max_workers))
    vfrac = 0.0 if val_files else split.val_frac
    tfrac = 0.0 if test_files else split.test_frac

    if vfrac <= 0.0 and tfrac <= 0.0:                  # everything explicit -> all inputs train
        stats["train"].extend(_convert_many(
            input_files, lambda f: out_root / "train" / f"{f.stem}.npz",
            spec, value_to_index, intensity_norm, say, compute_hag=compute_hag,
            ground_value=ground_value, hag_filter=hag_filter,
            feature_fields=feature_fields, geo_features=geo_features,
            geo_radius=geo_radius, rgb_fields=rgb_fields,
            max_workers=max_workers))
        return stats

    # single cloud: tile as measurement, allocate tiles, reassemble per split
    if len(input_files) == 1:
        f = input_files[0]
        say(f"  single-cloud split {f.name} (tile-measure -> holey clouds) ...")
        cloud = _apply_rgb_mapping(read_points(f), rgb_fields)
        raw = read_labels(f, cloud, spec) if spec is not None else None
        if compute_hag and _hag_from_cloud(cloud) is None:
            from . import pretrain                      # whole-cloud HAG, ferried via fields
            gmask = (raw == int(ground_value)) if (raw is not None and ground_value is not None) else None
            h = pretrain.hag_for_cloud(cloud, ground_mask=gmask, hag_filter=hag_filter)
            if h is not None and len(h) == cloud.n:
                cloud.fields["HeightAboveGround"] = h.astype(np.float32)
        if geo_features:
            # whole-cloud once (seam-consistent); after HAG so PDAL never sees the columns
            from . import pretrain
            say(f"  computing {len(geo_features)} geometric feature(s) whole-cloud "
                f"(r={geo_radius:g} m) ...")
            for nm, v in pretrain.geo_features_for_cloud(
                    cloud.xyz, geo_features, geo_radius).items():
                cloud.fields[f"feat_geo_{nm.lower()}"] = v
        keys, uniq, inv = _atomize(cloud, split.tile_m)
        pts = np.bincount(inv, minlength=len(uniq))
        hist = _tile_hists(inv, len(uniq), raw, value_to_index, num_classes)
        assign = allocate_splits(pts, hist, vfrac, tfrac, split.mode, split.seed)
        sop = assign[inv]
        keep = _seam_drop(cloud.xyz[:, :2], keys, uniq, sop,
                          split.tile_m, split.seam_buffer_m)
        for s, split_name in enumerate(_SPLITS):
            mask = (sop == s) & keep
            if not mask.any():
                continue
            sub, sub_raw = _subset(cloud, raw, mask)
            say(f"  {split_name}: {int(mask.sum()):,} pts")
            emit(split_name, sub, sub_raw, f.stem, hag_already=True)
        return stats

    # folder of clouds: convert to a pool, allocate whole scenes, move
    say(f"  converting {len(input_files)} scene(s), then allocating by point count ...")
    pool = _convert_many(input_files, lambda f: out_root / "_pool" / f"{f.stem}.npz",
                         spec, value_to_index, intensity_norm, say, compute_hag=compute_hag,
                         ground_value=ground_value, hag_filter=hag_filter,
                         feature_fields=feature_fields, geo_features=geo_features,
                         geo_radius=geo_radius, rgb_fields=rgb_fields,
                         max_workers=max_workers)
    for st in pool:                                    # input order kept -> deterministic alloc
        st["_pool_path"] = out_root / "_pool" / st["scene"]
    pts = [st["n_points"] for st in pool]
    hist = [_hist_from_counts(st["class_counts"], num_classes) for st in pool]
    assign = allocate_splits(pts, hist, vfrac, tfrac, split.mode, split.seed)
    for st, a in zip(pool, assign):
        split_name = _SPLITS[a]
        dest = out_root / split_name / st["scene"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(st.pop("_pool_path")), str(dest))
        stats[split_name].append(st)
    pool_dir = out_root / "_pool"
    if pool_dir.is_dir():
        try:
            pool_dir.rmdir()
        except OSError:
            pass
    return stats


def convert_dataset(name: str, inputs, spec: LabelSpec | None,
                    classes: list[dict], ignore_values: list[int],
                    staging_root: Path, *, val_inputs=None, test_inputs=None,
                    split: SplitConfig | None = None,
                    intensity_norm: str = "p95", compute_hag: bool = False,
                    ground_value: int | None = None,
                    hag_filter: str = "grid",
                    feature_fields: list[str] | None = None,
                    geo_features: list[str] | None = None,
                    geo_radius: float = 1.0,
                    rgb_fields: list[str] | None = None,
                    max_workers: int | None = None,
                    progress=None) -> Path:
    """Convert `inputs` (files/folders) into a staged canonical dataset.

    classes: [{"index", "source_value", "name"}].
    feature_fields: source fields baked into every scene as feat_<name>.
    geo_features: jakteristics names, computed from xyz within geo_radius m.
    compute_hag: bake feat_hag; ground_value labels are the ONLY ground source
        when set, else CSF detects ground (grid heuristic is the PDAL-less
        fallback); hag_filter = "grid" | "hag_nn" | "hag_delaunay".
    """
    from . import analysis

    split = split or SplitConfig()
    name = sanitize_name(name)
    out_root = staging_root / name
    value_to_index = {int(c["source_value"]): int(c["index"]) for c in classes}
    say = progress or (lambda s: None)

    num_classes = (max(value_to_index.values()) + 1) if value_to_index else 0
    input_files = expand_inputs(inputs)
    if not input_files:
        raise FileNotFoundError(f"No supported point-cloud files in {inputs}")
    val_files = expand_inputs(val_inputs) if val_inputs else None
    test_files = expand_inputs(test_inputs) if test_inputs else None
    if val_inputs and not val_files:
        raise FileNotFoundError(f"No supported point-cloud files in {val_inputs}")
    if test_inputs and not test_files:
        raise FileNotFoundError(f"No supported point-cloud files in {test_inputs}")
    strategy = "provided" if (val_files or test_files) else "auto"
    say(f"source: {len(input_files)} file(s); split={split.mode} "
        f"val={split.val_frac:.0%} test={split.test_frac:.0%} seed={split.seed}"
        + (" (+ explicit folders)" if strategy == "provided" else ""))

    use_csf = False
    if compute_hag:
        from . import pretrain
        if hag_filter != "grid" and not pretrain.pdal_available():
            say(f"⚠ {hag_filter} needs PDAL (not installed) - using the grid method instead.")
            hag_filter = "grid"
        # labels win outright; else CSF; grid heuristic only without PDAL
        use_csf = ground_value is None and pretrain.pdal_available()
        if ground_value is None and not use_csf:
            say("⚠ no ground class set and PDAL (CSF) not installed - "
                "falling back to the grid detection heuristic.")
        src = (f"ground=class {ground_value}" if ground_value is not None
               else ("CSF" if use_csf else "grid detection"))
        say(f"computing HeightAboveGround inline ({src} -> {hag_filter}) …")
    if geo_features:
        say(f"computing geometric feature(s) inline (jakteristics, "
            f"r={geo_radius:g} m): {', '.join(geo_features)} …")
    scene_stats = _plan_and_convert(input_files, val_files, test_files, split, spec,
                                    value_to_index, num_classes, out_root,
                                    intensity_norm, say, compute_hag=compute_hag,
                                    ground_value=ground_value,
                                    hag_filter=hag_filter,
                                    feature_fields=feature_fields,
                                    geo_features=geo_features,
                                    geo_radius=geo_radius,
                                    rgb_fields=rgb_fields,
                                    max_workers=max_workers)
    for sp in _SPLITS:
        if not scene_stats[sp]:
            raise ValueError(f"Conversion produced an empty {sp} split - lower the "
                             f"val/test fractions, add more data, or supply explicit folders.")

    splits = {sp: {"scenes": [s["scene"] for s in scene_stats[sp]],
                   "total_points": sum(s["n_points"] for s in scene_stats[sp]),
                   "per_class": {str(i): sum(s["class_counts"].get(i, 0)
                                             for s in scene_stats[sp])
                                 for i in range(num_classes)}}
              for sp in _SPLITS}

    all_stats = [s for sp in _SPLITS for s in scene_stats[sp]]
    for w in sorted({s["crs_warning"] for s in all_stats if s.get("crs_warning")}):
        say(w)
    if len({s.get("crs_wkt") for s in all_stats} - {None}) > 1:
        say("⚠ scenes carry MORE THAN ONE CRS — the model would train on mixed "
            "zones/unit systems; reproject all sources to one CRS for a clean dataset.")
    total_pts = sum(s["n_points"] for s in all_stats)
    total_area = sum(s["area_m2"] for s in all_stats)
    density = total_pts / max(total_area, 1.0)
    spacing = (total_area / max(total_pts, 1)) ** 0.5
    for c in classes:
        for sp in _SPLITS:
            c[f"{sp}_count"] = sum(s["class_counts"].get(int(c["index"]), 0)
                                   for s in scene_stats[sp])

    # several source values may combine into one class: count unique indices
    idx_to_name: dict[int, str] = {}
    for c in classes:
        idx_to_name.setdefault(int(c["index"]), c["name"])

    # collapse combined classes to one entry per index (counts already total per index)
    by_index: dict[int, dict] = {}
    for c in classes:
        i = int(c["index"])
        e = by_index.get(i)
        if e is None:
            by_index[i] = {"index": i, "name": c["name"],
                           "source_values": [int(c["source_value"])],
                           **{f"{sp}_count": int(c[f"{sp}_count"]) for sp in _SPLITS}}
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
    if feature_fields:
        stats["feature_scales"] = {s["scene"]: s["feature_scales"] for s in all_stats}
    # hag_src: how HAG was produced; inference reproduces the method from this
    if not all(s["has_hag"] for s in all_stats):
        hag_src = ""
    elif not compute_hag:
        hag_src = "source_dimension"
    elif ground_value is not None:
        hag_src = f"{hag_filter}+labels"
    elif use_csf:
        hag_src = f"{hag_filter}+csf"
    else:
        hag_src = "grid"
    meta = {
        "schema_version": 2,
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "inputs": [str(p) for p in (inputs if isinstance(inputs, (list, tuple)) else [inputs])],
            "val_inputs": [str(p) for p in (val_inputs or [])],
            "test_inputs": [str(p) for p in (test_inputs or [])],
            "split_strategy": strategy,
            "label_kind": spec.kind if spec else None,
            "label_field": spec.field if spec else "",
            "truth_dir": spec.truth_dir if spec else "",
            "intensity_norm": intensity_norm,
            "hag_source": hag_src,
            "rgb_fields": rgb_fields,
            "hag_ground_value": (int(ground_value) if ground_value is not None else None),
            "hag_use_csf": bool(use_csf),
            # source_field "@geo:<name>"/"@hag:<method>" = computed channels,
            # recomputed at inference. Canonical intensity/return_number ride the
            # has_* flags, never this list.
            "feature_channels": ([{"name": feat_key(f), "source_field": f, "norm": "p95abs"}
                                  for f in split_feature_fields(feature_fields)[1]]
                                 + [{"name": f"geo_{nm.lower()}",
                                     "source_field": f"@geo:{nm}", "norm": "raw",
                                     "radius": float(geo_radius)}
                                    for nm in (geo_features or [])]
                                 + ([{"name": "hag", "source_field": f"@hag:{hag_src}",
                                      "norm": "raw"}] if hag_src else [])),
            "ignore_values": [int(v) for v in ignore_values],
        },
        "split": {
            "mode": split.mode,
            "seed": int(split.seed),
            "seam_buffer_m": float(split.seam_buffer_m),
            "atom_unit": "scene" if (len(input_files) > 1 or strategy == "provided") else "tile",
            "requested": {"train": round(1.0 - split.val_frac - split.test_frac, 4),
                          "val": split.val_frac, "test": split.test_frac},
            "achieved": {sp: round(splits[sp]["total_points"] / max(total_pts, 1), 4)
                         for sp in _SPLITS},
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


# classification-bearing field names, best first (matches datasets_page._scan_labels)
_GROUND_FIELD_CANDIDATES = ("classification", "Classification", "scalar_label",
                            "label", "class")


def convert_infer_job(job_id: str, input_dir: str, staging_root: Path, progress=None,
                      intensity_norm: str = "p95", hag: bool = False,
                      hag_filter: str = "grid", ground_value: int | None = None,
                      feature_fields: list[str] | None = None,
                      geo_features: list[str] | None = None,
                      geo_radius: float = 1.0,
                      out_dir: Path | None = None) -> Path:
    """Label-less conversion for inference jobs -> <staging>/_infer/<job_id>/ (or
    out_dir). The container mount stays /datasets/_infer/<job> either way.
    intensity_norm MUST match the weights' training norm — a mismatch feeds
    out-of-distribution intensity. hag/ground_value behave as on the Datasets
    page; a scene without ground_value degrades to detection, never to no HAG."""
    say = progress or (lambda s: None)
    # infer scenes always carry intensity/return_number when the source has them;
    # the checkpoint's FEAT_CHANNELS decides what's consumed
    feature_fields = list(feature_fields or []) + ["intensity", "return_number"]
    if hag:
        from . import pretrain
        if hag_filter != "grid" and not pretrain.pdal_available():
            say(f"  ⚠ {hag_filter} needs PDAL (not installed) - using the grid method "
                "instead (approximate HAG).")
            hag_filter = "grid"
        src = (f"ground=class {ground_value}" if ground_value is not None
               else ("CSF" if pretrain.pdal_available() else "grid detection"))
        say(f"  computing HeightAboveGround per scene ({src} -> {hag_filter}) …")
    if geo_features:
        say(f"  computing geometric feature(s) per scene (jakteristics, "
            f"r={geo_radius:g} m): {', '.join(geo_features)} …")
    out_root = Path(out_dir) if out_dir else (staging_root / "_infer" / job_id)
    files = discover_scenes(input_dir)
    if not files:
        raise FileNotFoundError(f"No supported point-cloud files in {input_dir}")
    # one spec for the whole job so "ground" can't change fields mid-job
    spec = None
    if hag and ground_value is not None:
        fields = list_label_fields(files[0])
        field = next((f for f in _GROUND_FIELD_CANDIDATES if f in fields), None)
        if field is None:
            raise ValueError(
                f"Ground class {ground_value} was set, but {files[0].name} carries no "
                f"classification field (has: {sorted(fields)}). Clear the ground class "
                f"to detect ground instead.")
        spec = LabelSpec(kind="field", field=field)
        say(f"  ground from the '{field}' field, value {ground_value}")
    scenes, stats = [], []
    for path in files:
        out_path = out_root / "scenes" / (path.stem + ".npz")
        say(f"  converting {path.name} ...")
        st = convert_scene(path, spec, {}, out_path, intensity_norm=intensity_norm,
                           compute_hag=hag, ground_value=ground_value,
                           hag_filter=hag_filter, feature_fields=feature_fields,
                           geo_features=geo_features, geo_radius=geo_radius)
        st["scene"] = out_path.name
        st["source_file"] = str(path)
        if st.get("crs_warning"):
            say("  " + st["crs_warning"])
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


# e57 deliberately absent: ASTM E2807 defines no classification attribute
PRED_EXPORT_FORMATS = ("las", "laz", "ply", "txt", "csv")


def export_predictions(pred_dir, fmt: str, progress=None, class_map=None,
                       unclass_threshold=None) -> list[Path]:
    """Export each <name>_pred.npz as <name>_pred.<fmt> (xyz + classification
    [+ confidence]). class_map remaps model indices to source values. Points below
    unclass_threshold export as class 1 (0 if 1 is taken; 255 with no class_map).
    The npz keeps the raw prediction so a new threshold re-exports without re-running."""
    fmt = fmt.lower().lstrip(".")
    if fmt not in PRED_EXPORT_FORMATS:
        raise ValueError(f"unsupported prediction format '{fmt}' "
                         f"(one of {', '.join(PRED_EXPORT_FORMATS)})")
    say = progress or (lambda s: None)
    unclass = 1                             # ASPRS "Unclassified"
    if unclass_threshold is not None and class_map and 1 in class_map.values():
        unclass = 0
        say("  (class 1 is taken by the model's own classes — low-confidence "
            "points export as class 0 instead)")
    elif unclass_threshold is not None and not class_map:
        # ponytail: raw indices make 0 and 1 real classes; 255 is free under the
        # uint8 clip below. Breaks at K>255, which the cast already can't do.
        unclass = 255
        say("  (no class map — export carries raw model indices; low-confidence "
            "points export as 255)")
    written: list[Path] = []
    no_crs: list[str] = []
    legend_said = False
    for src in sorted(Path(pred_dir).glob("*_pred.npz")):
        with np.load(src) as d:
            xyz = np.asarray(d["xyz"], np.float64)
            cls = np.asarray(d["classification"], np.int64)
            conf = (np.asarray(d["confidence"], np.float32)
                    if "confidence" in d.files else None)
            crs_wkt = str(d["crs_wkt"]) if "crs_wkt" in d.files else None
            member = (np.asarray(d["dominant_member"], np.uint8)
                      if "dominant_member" in d.files else None)
            member_names = ([str(s) for s in d["member_names"]]
                            if "member_names" in d.files else None)
        if member is not None and fmt in ("las", "laz") and not legend_said:
            legend_said = True
            say("  ens_member field (ensemble's dominant model per point): "
                + (", ".join(f"{i}={n}" for i, n in enumerate(member_names))
                   if member_names else "member indices in ensemble input order"))
        if crs_wkt is None and fmt in ("las", "laz"):
            no_crs.append(src.name)
        f = crs_unit_factor(crs_wkt)
        if f != 1.0:
            # Scenes are meter-scaled at ingest (readers._read_las); deliverables
            # go back to the source frame so they overlay the input clouds.
            xyz = xyz / f
            say(f"  {src.name}: restoring source CRS units (÷{f:g})")
        if class_map:
            lut = np.arange(max(int(cls.max(initial=0)), max(class_map)) + 1)
            for i, v in class_map.items():
                lut[i] = v
            cls = lut[cls]
        low = 0
        if unclass_threshold is not None and conf is not None:
            mask = conf < unclass_threshold
            cls[mask] = unclass
            low = int(mask.sum())
        # ponytail: uint8 classification (LAS pf6 / uchar); >255 classes would clip.
        cls = np.clip(cls, 0, 255).astype(np.uint8)
        dst = src.with_suffix(f".{fmt}")
        _write_pred(dst, xyz, cls, fmt, confidence=conf, crs_wkt=crs_wkt,
                    member=member)
        written.append(dst)
        say(f"  {src.name} -> {dst.name} ({len(xyz):,} pts"
            + (f"; {low:,} below confidence {unclass_threshold:g} -> class {unclass}"
               if unclass_threshold is not None and conf is not None else "") + ")")
    if no_crs:
        say(f"  ⚠ no CRS recorded in {', '.join(no_crs)} — those {fmt} deliverables "
            "carry no georeferencing (predictions from before CRS support, or "
            "sources without one). Re-running the inference job re-stages with CRS.")
    return written


def _write_pred(dst: Path, xyz: np.ndarray, cls: np.ndarray, fmt: str,
                confidence=None, crs_wkt=None, member=None):
    """One classified cloud -> dst. confidence rides in las/laz (Extra Bytes) and
    txt/csv; member (ensemble dominant model) as las/laz "ens_member"; crs_wkt is
    restored as the las/laz WKT VLR. ply stays classification-only."""
    if fmt in ("las", "laz"):
        import laspy
        h = laspy.LasHeader(point_format=6, version="1.4")
        if confidence is not None:
            h.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.float32))
        if member is not None:
            h.add_extra_dim(laspy.ExtraBytesParams(name="ens_member", type=np.uint8))
        if crs_wkt:
            try:
                from pyproj import CRS
                h.add_crs(CRS.from_wkt(crs_wkt))
            except Exception:
                pass    # a CRS-less deliverable beats no deliverable
        h.offsets = xyz.min(0)
        h.scales = [0.001] * 3
        las = laspy.LasData(h)
        las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        las.classification = cls
        if confidence is not None:
            las.confidence = confidence
        if member is not None:
            las.ens_member = member
        las.write(str(dst))
    elif fmt == "ply":
        # double, not float: UTM-scale coords quantize to 0.5 m in a float32 parse
        header = ("ply\nformat ascii 1.0\n" + f"element vertex {len(xyz)}\n"
                  "property double x\nproperty double y\nproperty double z\n"
                  "property uchar classification\nend_header")
        np.savetxt(dst, np.column_stack([xyz, cls]),
                   fmt=["%.3f"] * 3 + ["%d"], header=header, comments="")
    elif fmt == "txt":                                       # x y z class [confidence]
        cols = [xyz, cls] + ([confidence] if confidence is not None else [])
        np.savetxt(dst, np.column_stack(cols), fmt="%.3f %.3f %.3f %d"
                   + (" %.3f" if confidence is not None else ""))
    elif fmt == "csv":
        cols = [xyz, cls] + ([confidence] if confidence is not None else [])
        np.savetxt(dst, np.column_stack(cols), delimiter=",",
                   fmt=["%.3f"] * 3 + ["%d"]
                   + (["%.3f"] if confidence is not None else []),
                   header="x,y,z,classification"
                   + (",confidence" if confidence is not None else ""),
                   comments="")


# --------------------------------------------------------------------- self-check

def _selfcheck():
    """Allocator hits targets + both build paths materialize three disjoint splits."""
    import tempfile

    # allocator: balanced mode hits point targets and spreads a rare class
    C = 3
    pts = [1000] * 7 + [400, 400, 400]       # 10 atoms
    hist = np.zeros((10, C), dtype=np.int64)
    for i in range(7):
        hist[i, 0] = 700; hist[i, 1] = 300
    hist[7, 2] = hist[8, 2] = hist[9, 2] = 400   # rare class: one carrier per split
    a = allocate_splits(pts, hist, 0.2, 0.2, "balanced", 42)
    assert set(a.tolist()) == {0, 1, 2}, "all three splits used"
    got = np.zeros((3, C))
    for i, s in enumerate(a):
        got[s] += hist[i]
    for s in range(3):
        assert got[s, 2] > 0, "rare class present in every split (presence guarantee)"
    tot = sum(pts)
    train_frac = sum(pts[i] for i in range(10) if a[i] == 0) / tot
    assert 0.45 <= train_frac <= 0.75, f"train point-frac ~0.6, got {train_frac:.2f}"
    # random mode: no crash, all assigned, three splits non-empty for these sizes
    ar = allocate_splits(pts, hist, 0.2, 0.2, "random", 1)
    assert (ar >= 0).all() and set(ar.tolist()) == {0, 1, 2}

    rng = np.random.RandomState(0)
    n = 40000
    xyz = rng.uniform(0, 100, size=(n, 3)).astype(np.float64)
    cls = rng.randint(0, 3, size=n).astype(np.float64)
    classes = [{"index": i, "source_value": i, "name": f"c{i}"} for i in range(3)]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # single cloud -> 3 holey clouds (tile-measure + seam buffer, points only removed)
        np.savez(td / "scene.npz", xyz=xyz, classification=cls)
        out = convert_dataset("selftest", td / "scene.npz",
                              LabelSpec(kind="field", field="classification"),
                              [dict(c) for c in classes], [], td / "staging",
                              split=SplitConfig(val_frac=0.2, test_frac=0.2,
                                                seam_buffer_m=1.0, tile_m=20.0))
        meta = json.loads((out / "dataset_meta.json").read_text())
        assert meta["schema_version"] == 2 and "split" in meta
        assert set(meta["splits"]) == set(_SPLITS)
        for sp in _SPLITS:
            assert meta["splits"][sp]["scenes"], f"{sp} non-empty"
            assert list((out / sp).glob("*.npz")), f"{sp}/ materialized"
            assert all(f"{sp}_count" in c for c in meta["classes"])
        kept = sum(meta["splits"][sp]["total_points"] for sp in _SPLITS)
        assert kept <= n, "seam buffer only removes points"
        assert kept >= 0.5 * n, "seam buffer must not delete most points"
        # folder of clouds -> whole-scene 3-way; no scene in two splits; pool cleaned
        multi = td / "multi"; multi.mkdir()
        for k in range(9):
            np.savez(multi / f"s{k}.npz", xyz=xyz, classification=cls)
        out2 = convert_dataset("selftest2", multi, LabelSpec(field="classification"),
                               [dict(c) for c in classes], [], td / "staging",
                               split=SplitConfig(val_frac=0.2, test_frac=0.2))
        m2 = json.loads((out2 / "dataset_meta.json").read_text())
        names = {sp: set(m2["splits"][sp]["scenes"]) for sp in _SPLITS}
        union = set().union(*names.values())
        assert len(union) == 9 and sum(len(names[sp]) for sp in _SPLITS) == 9, \
            "every scene assigned exactly once"
        assert not (out2 / "_pool").exists(), "pool dir cleaned up"
        # provided val folder -> test allocated from inputs
        vdir = td / "vprov"; vdir.mkdir()
        np.savez(vdir / "v0.npz", xyz=xyz, classification=cls)
        out3 = convert_dataset("selftest3", multi, LabelSpec(field="classification"),
                               [dict(c) for c in classes], [], td / "staging",
                               val_inputs=[str(vdir)],
                               split=SplitConfig(val_frac=0.2, test_frac=0.2))
        m3 = json.loads((out3 / "dataset_meta.json").read_text())
        assert m3["source"]["split_strategy"] == "provided"
        assert m3["splits"]["val"]["scenes"] == ["v0.npz"] and m3["splits"]["test"]["scenes"]
    print("dataset self-check OK")


if __name__ == "__main__":
    _selfcheck()
