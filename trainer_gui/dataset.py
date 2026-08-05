"""Convert user folders/files into the canonical dataset the scripts consume.

Layout: <staging>/<name>/{dataset_meta.json, train|val|test/<scene>.npz} - xyz f64,
label i32 (-1 = ignore), optional rgb u8 / intensity f32 / return_number f32, with
inference jobs using the same npz minus `label` as flat <stem>_input.npz files
next to job.json (predictions merge into the same npz). The split is decided
ONCE here (point-count fractions over whole scenes, or tiles of a single cloud
reassembled into holey per-split clouds) and recorded in dataset_meta.json;
trainers read the folders verbatim.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .readers import (SUPPORTED_EXTS, Cloud, list_label_fields, read_points,
                      restore_to_source)


@dataclass
class LabelSpec:
    """Label source: a named field in the cloud."""
    field: str = ""


@dataclass
class SplitConfig:
    """val_frac/test_frac: target point-count fractions. mode "balanced" mirrors
    the global class mix (+ rare-class presence); "random" fills by points.
    seam_buffer_m/tile_m are single-cloud only, and strategy "provided" = explicit
    folders. seed is fixed so identical builds reproduce the identical split and
    trainer prep-cache signatures stay valid across rebuilds; recorded in
    dataset_meta.json."""
    val_frac: float = 0.15
    test_frac: float = 0.15
    mode: str = "balanced"
    seed: int = 0
    seam_buffer_m: float = 1.0
    tile_m: float = 50.0
    strategy: str = "auto"


def discover_scenes(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if folder.is_file():
        return [folder] if folder.suffix.lower() in SUPPORTED_EXTS else []
    top = sorted(p for p in folder.iterdir()
                 if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if top:
        return top
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
    if spec.field not in cloud.fields:
        raise ValueError(f"{path.name}: label field '{spec.field}' not found "
                         f"(have: {sorted(cloud.fields)})")
    raw = np.asarray(cloud.fields[spec.field])
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


_CANON_SPELLINGS = {
    "intensity": "intensity", "scalar_intensity": "intensity",
    "return_number": "return_number", "returnnumber": "return_number",
    "scalar_return_number": "return_number", "scalar_returnnumber": "return_number",
    "ret_num": "return_number",
}


def canonical_channel(field: str) -> str | None:
    """Field name -> canonical channel name, or None for an ordinary feat_* field."""
    s = (field or "").strip().lower().replace(" ", "_")
    return _CANON_SPELLINGS.get(s)


def split_feature_fields(fields: list[str] | None) -> tuple[dict, list[str]]:
    """Prep selection -> ({canonical channel: explicit source column | None},
    remaining feat_* entries). 'name=Column' binds a source column (Infer page
    channel mapping); the first binding for a canonical channel wins."""
    canon: dict[str, str | None] = {}
    rest: list[str] = []
    for f in (fields or []):
        tgt, _, src = f.partition("=")
        c = canonical_channel(tgt)
        if c:
            canon.setdefault(c, src or None)
        else:
            rest.append(f)
    return canon, rest


def sanitize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_")
    if not s:
        raise ValueError("Dataset name is empty after sanitizing")
    return s.lower()


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
        source_crs_wkt=cloud.source_crs_wkt,
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
                    mode: str, seed: int) -> np.ndarray:
    """Assign atoms to train(0)/val(1)/test(2) approximating the point fractions.
    "random": greedy fill by point deficit. "balanced": rarity-first iterative
    stratification so each split mirrors the global class mix."""
    pts = np.asarray(pts, dtype=np.float64)
    hist = np.asarray(hist, dtype=np.float64)
    n = len(pts)
    C = hist.shape[1] if (hist.ndim == 2 and hist.size) else 0
    fr = np.array([max(0.0, 1.0 - val_frac - test_frac), val_frac, test_frac])
    allowed = fr > 0
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
        target = np.outer(fr, H)
        got = np.zeros((3, C))
        crit = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            pres = np.where(hist[i] > 0)[0]
            if len(pres):
                crit[i] = int(pres[np.argmin(H[pres])])
        rarest = np.array([H[crit[i]] if crit[i] >= 0 else np.inf for i in range(n)])
        order = np.lexsort((-pts, rarest))
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
    span = int(keys[:, 1].max()) + 1
    code, inv = np.unique(keys[:, 0] * span + keys[:, 1], return_inverse=True)
    uniq = np.column_stack([code // span, code % span])
    return keys, uniq, inv


def _seam_drop(xy, keys, uniq, split_of_point, tile_m: float,
               seam_buffer_m: float) -> np.ndarray:
    """Keep-mask: drop points within seam_buffer_m of a neighbouring tile in a
    different split. ponytail: O(n) tile-edge approximation - can over-drop
    slightly, never leaks; effective buffer caps at tile_m."""
    n = len(xy)
    keep = np.ones(n, dtype=bool)
    if seam_buffer_m <= 0:
        return keep
    tile = max(float(tile_m), 1e-6)
    b = min(float(seam_buffer_m), tile)
    k0 = uniq.min(0)
    grid = np.full(uniq.max(0) - k0 + 3, -1, dtype=np.int64)
    kx = keys[:, 0] - k0[0] + 1
    ky = keys[:, 1] - k0[1] + 1
    grid[kx, ky] = split_of_point
    u = xy - xy.min(0) - keys * tile
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


def _savez_fast(path: Path, arrays: dict) -> None:
    """npz at zlib level 1: ~3x faster than savez_compressed's level 6 for ~10%
    bigger files; np.load-compatible. Streams members straight into the zip so
    GB-scale scenes never hold a second in-RAM copy."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name, a in arrays.items():
            with zf.open(name + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(a))


def _crs_label(wkt: str | None) -> str:
    """Short 'AUTH:CODE' (or CRS name) for a WKT, for the reproject info line."""
    try:
        from pyproj import CRS
        crs = CRS.from_wkt(wkt)
        auth = crs.to_authority()
        return f"{auth[0]}:{auth[1]}" if auth else crs.name
    except Exception:
        return "?"


def _crs_report(crs_wkt: str | None, source_crs_wkt: str | None,
                xyz: np.ndarray, name: str) -> str | None:
    """Coords are meter-projected by read_points before this runs. source_crs_wkt
    set = a reprojection happened (info line). No CRS + degree-looking coords is a
    D1 hard block naming the declare-EPSG remedy; no CRS at metric magnitude passes."""
    if source_crs_wkt:
        return f"ℹ {name}: reprojected {_crs_label(source_crs_wkt)} -> {_crs_label(crs_wkt)}"
    if crs_wkt:
        return None
    xy = xyz[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    if ((hi - lo).max() < 10.0 and abs(lo[0]) <= 360 and abs(hi[0]) <= 360
            and abs(lo[1]) <= 90 and abs(hi[1]) <= 90):
        raise ValueError(
            f"{name}: no CRS and the coordinates look like lon/lat degrees, but the "
            "pipeline is meter-denominated. Declare the source CRS in the EPSG field "
            "on the prep/inference page so it can be reprojected to meters.")
    return None


def _hag_from_cloud(cloud: Cloud) -> np.ndarray | None:
    """Source HeightAboveGround/HAG field when one exists and aligns 1:1 - wins
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
    if c.max() > 255:
        c = c / 257.0
    elif 0 < c.max() <= 1.0:
        c = c * 255.0
    cloud.rgb = np.clip(c, 0, 255).astype(np.uint8)
    return cloud


def _convert_one(cloud: Cloud, raw: np.ndarray | None, value_to_index: dict[int, int],
                 out_path: Path, intensity_norm: str = "p95",
                 compute_hag: bool = False, ground_value: int | None = None,
                 hag_filter: str = "grid",
                 ground_method: str = "labels",
                 feature_fields: list[str] | None = None,
                 geo_features: list[str] | None = None,
                 geo_k: int = 100, missing_fields: str = "raise") -> dict:
    """Write one (already-read, already-cropped) cloud to a canonical npz.
    compute_hag stores feat_hag when an aligned result is possible; a scene
    without it blocks runs that require the channel. missing_fields='skip'
    (inference): a requested field the file lacks is skipped and reported in
    stats['skipped_fields'] instead of raising - the trainer feeds zeros."""
    # xyz stays float64: UTM-scale coords quantize to 0.5 m in float32
    crs_warning = _crs_report(cloud.crs_wkt, cloud.source_crs_wkt, cloud.xyz, out_path.stem)
    out: dict[str, np.ndarray] = {"xyz": cloud.xyz.astype(np.float64)}
    # crs_wkt = processing CRS of the stored coords; source_crs_wkt present = a transform occurred (the round-trip signal), both ferry scene npz -> pred npz
    if cloud.crs_wkt:
        out["crs_wkt"] = np.asarray(cloud.crs_wkt)
    if cloud.source_crs_wkt:
        out["source_crs_wkt"] = np.asarray(cloud.source_crs_wkt)

    class_counts: dict[int, int] = {}
    if raw is not None and value_to_index:
        label = np.full(cloud.n, -1, dtype=np.int32)
        for src_val, idx in value_to_index.items():
            label[raw == src_val] = idx
        out["label"] = label
        vals, cnts = np.unique(label[label >= 0], return_counts=True)
        class_counts = {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}

    if cloud.rgb is not None:
        out["rgb"] = cloud.rgb
    canon_wanted, feature_fields = split_feature_fields(feature_fields)

    def _column(src: str) -> np.ndarray:
        key = next((k for k in cloud.fields if k.lower() == src.lower()), None)
        if key is None:
            raise ValueError(f"{out_path.stem}: feature field '{src}' not found "
                             f"(have: {sorted(cloud.fields)})")
        v = np.asarray(cloud.fields[key])
        if v.ndim != 1 or not np.issubdtype(v.dtype, np.number):
            raise ValueError(f"{out_path.stem}: feature field '{src}' is not numeric 1-D "
                             f"(dtype {v.dtype}, shape {v.shape})")
        return v.astype(np.float32)

    def _canon_source(name, attr):
        """The channel's array: an explicitly bound column wins over the
        format-standard field (Infer page 'name=Column' mapping)."""
        src = canon_wanted.get(name)
        return _column(src) if src else attr

    raw_imax = None
    cloud_intensity = _canon_source("intensity", cloud.intensity) \
        if "intensity" in canon_wanted else None
    if cloud_intensity is not None:
        if intensity_norm == "p95":
            denom = max(float(np.percentile(cloud_intensity, 95)), 1.0)
            out["intensity"] = np.clip(cloud_intensity / denom, 0.0, 2.0).astype(np.float32)
            raw_imax = denom
        else:
            raw_imax = max(float(cloud_intensity.max()), 1.0)
            out["intensity"] = (cloud_intensity / raw_imax).astype(np.float32)
    cloud_retnum = _canon_source("return_number", cloud.return_number) \
        if "return_number" in canon_wanted else None
    if cloud_retnum is not None:
        out["return_number"] = cloud_retnum.astype(np.float32)
    source_hag = _hag_from_cloud(cloud)
    if source_hag is not None:
        out["feat_hag"] = source_hag.astype(np.float32)
    hag_warning = None
    if compute_hag and "feat_hag" not in out:
        from . import pretrain
        gmask = (raw == int(ground_value)) if (raw is not None and ground_value is not None) else None
        hag_notes: list = []
        h = pretrain.hag_for_cloud(cloud, ground_mask=gmask, hag_filter=hag_filter, ground_method=ground_method,
                                   notes=hag_notes)
        if h is not None and len(h) == cloud.n:
            out["feat_hag"] = h.astype(np.float32)
        if hag_notes:
            hag_warning = f"⚠ {out_path.stem}: {hag_notes[0]}"

    if geo_features:
        from . import pretrain
        have = {nm: np.asarray(cloud.fields[f"feat_geo_{nm.lower()}"], np.float32)
                for nm in geo_features if f"feat_geo_{nm.lower()}" in cloud.fields}
        todo = [nm for nm in geo_features if nm not in have]
        if todo:
            have.update(pretrain.geo_features_for_cloud(cloud.xyz, todo, geo_k))
        for nm in geo_features:
            out[f"feat_geo_{nm.lower()}"] = have[nm]

    feature_scales: dict[str, float] = {}
    skipped_fields: list[str] = []
    for fld in (feature_fields or []):
        # verbatim-first: a real column literally named 'a=b' wins over the
        # 'target=Source' remap parse (Infer page channel mapping)
        tgt, src = fld, fld
        if "=" in fld and not any(k.lower() == fld.lower() for k in cloud.fields):
            tgt, _, src = fld.partition("=")
        try:
            v = _column(src or tgt)
        except ValueError:
            if missing_fields != "skip":
                raise
            skipped_fields.append(src or tgt)
            continue
        scale = max(float(np.percentile(np.abs(v), 95)), 1e-6)
        out[f"feat_{feat_key(tgt)}"] = np.clip(v / scale, -2.0, 2.0).astype(np.float32)
        feature_scales[feat_key(tgt)] = scale

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _savez_fast(out_path, out)

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
        "skipped_fields": skipped_fields,
        "crs_wkt": cloud.crs_wkt,
        "source_crs_wkt": cloud.source_crs_wkt,
        "crs_warning": crs_warning,
        "hag_warning": hag_warning,
    }


def convert_scene(path: Path, spec: LabelSpec | None, value_to_index: dict[int, int],
                  out_path: Path, intensity_norm: str = "p95",
                  compute_hag: bool = False, ground_value: int | None = None,
                  hag_filter: str = "grid",
                  ground_method: str = "labels",
                  feature_fields: list[str] | None = None,
                  geo_features: list[str] | None = None,
                  geo_k: int = 100,
                  declared_crs_epsg: int | None = None,
                  missing_fields: str = "raise") -> dict:
    """Read one source file and convert the whole cloud (no cropping)."""
    cloud = read_points(path, declared_crs_epsg)
    raw = read_labels(path, cloud, spec) if spec is not None else None
    return _convert_one(cloud, raw, value_to_index, out_path, intensity_norm,
                        compute_hag=compute_hag, ground_value=ground_value,
                        hag_filter=hag_filter, ground_method=ground_method, feature_fields=feature_fields,
                        geo_features=geo_features, geo_k=geo_k,
                        missing_fields=missing_fields)


def _convert_many(files: list[Path], dest_for, spec, value_to_index, intensity_norm, say,
                  *, compute_hag, ground_value, hag_filter, ground_method,
                  feature_fields: list[str] | None = None,
                  geo_features: list[str] | None = None, geo_k: int = 100,
                  rgb_fields: list[str] | None = None,
                  declared_crs_epsg: int | None = None) -> list[dict]:
    """Convert each file; stat dicts return in INPUT order (the caller
    feeds allocate_splits positionally)."""
    out = []
    for f in files:
        cloud = _apply_rgb_mapping(read_points(f, declared_crs_epsg), rgb_fields)
        raw = read_labels(f, cloud, spec) if spec is not None else None
        out_path = dest_for(f)
        st = _convert_one(cloud, raw, value_to_index, out_path, intensity_norm,
                          compute_hag=compute_hag, ground_value=ground_value,
                          hag_filter=hag_filter, ground_method=ground_method, feature_fields=feature_fields,
                          geo_features=geo_features, geo_k=geo_k)
        st["scene"] = out_path.name
        say(f"  converted {f.name}")
        out.append(st)
    return out


def _plan_and_convert(input_files: list[Path], val_files: list[Path] | None,
                      test_files: list[Path] | None, split: SplitConfig,
                      spec: LabelSpec | None, value_to_index: dict[int, int],
                      num_classes: int, out_root: Path, intensity_norm: str, say, *,
                      compute_hag: bool = False, ground_value: int | None = None,
                      hag_filter: str = "grid",
                      ground_method: str = "labels",
                      feature_fields: list[str] | None = None,
                      geo_features: list[str] | None = None, geo_k: int = 100,
                      rgb_fields: list[str] | None = None,
                      declared_crs_epsg: int | None = None) -> dict:
    """Convert sources into out_root/{train,val,test}/*.npz; returns per-split
    stat lists. Explicit val/test folders are used verbatim; the rest is allocated."""
    stats = {sp: [] for sp in _SPLITS}

    def emit(split_name: str, cloud: Cloud, raw, scene_name: str):
        out_path = out_root / split_name / f"{scene_name}.npz"
        # HAG/geo were already baked whole-cloud before tiling
        st = _convert_one(cloud, raw, value_to_index, out_path, intensity_norm,
                          compute_hag=False,
                          ground_value=ground_value, hag_filter=hag_filter, ground_method=ground_method,
                          feature_fields=feature_fields,
                          geo_features=geo_features, geo_k=geo_k)
        st["scene"] = out_path.name
        stats[split_name].append(st)

    for split_name, files in (("val", val_files), ("test", test_files)):
        if files:
            stats[split_name].extend(_convert_many(
                files, lambda f, sn=split_name: out_root / sn / f"{f.stem}.npz",
                spec, value_to_index, intensity_norm, say, compute_hag=compute_hag,
                ground_value=ground_value, hag_filter=hag_filter, ground_method=ground_method,
                feature_fields=feature_fields, geo_features=geo_features,
                geo_k=geo_k, rgb_fields=rgb_fields,
                declared_crs_epsg=declared_crs_epsg))
    vfrac = 0.0 if val_files else split.val_frac
    tfrac = 0.0 if test_files else split.test_frac

    if vfrac <= 0.0 and tfrac <= 0.0:
        stats["train"].extend(_convert_many(
            input_files, lambda f: out_root / "train" / f"{f.stem}.npz",
            spec, value_to_index, intensity_norm, say, compute_hag=compute_hag,
            ground_value=ground_value, hag_filter=hag_filter, ground_method=ground_method,
            feature_fields=feature_fields, geo_features=geo_features,
            geo_k=geo_k, rgb_fields=rgb_fields,
            declared_crs_epsg=declared_crs_epsg))
        return stats

    if len(input_files) == 1:
        f = input_files[0]
        say(f"  single-cloud split {f.name} (tile-measure -> holey clouds) ...")
        cloud = _apply_rgb_mapping(read_points(f, declared_crs_epsg), rgb_fields)
        raw = read_labels(f, cloud, spec) if spec is not None else None
        if compute_hag and _hag_from_cloud(cloud) is None:
            from . import pretrain
            gmask = (raw == int(ground_value)) if (raw is not None and ground_value is not None) else None
            hag_notes: list = []
            h = pretrain.hag_for_cloud(cloud, ground_mask=gmask, hag_filter=hag_filter, ground_method=ground_method,
                                       notes=hag_notes)
            if h is not None and len(h) == cloud.n:
                cloud.fields["HeightAboveGround"] = h.astype(np.float32)
            for hn in hag_notes:
                say(f"  ⚠ {f.stem}: {hn}")
        if geo_features:
            from . import pretrain
            say(f"  computing {len(geo_features)} geometric feature(s) whole-cloud "
                f"(pgeof optimal, k≤{geo_k}) ...")
            for nm, v in pretrain.geo_features_for_cloud(
                    cloud.xyz, geo_features, geo_k).items():
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
            emit(split_name, sub, sub_raw, f.stem)
        return stats

    say(f"  converting {len(input_files)} scene(s), then allocating by point count ...")
    pool = _convert_many(input_files, lambda f: out_root / "_pool" / f"{f.stem}.npz",
                         spec, value_to_index, intensity_norm, say, compute_hag=compute_hag,
                         ground_value=ground_value, hag_filter=hag_filter, ground_method=ground_method,
                         feature_fields=feature_fields, geo_features=geo_features,
                         geo_k=geo_k, rgb_fields=rgb_fields,
                         declared_crs_epsg=declared_crs_epsg)
    for st in pool:
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
        except OSError as e:
            say(f"  ! {pool_dir} could not be removed ({e}); it still holds "
                f"converted scenes that never reached their split folder")
    return stats


def convert_dataset(name: str, inputs, spec: LabelSpec | None,
                    classes: list[dict], ignore_values: list[int],
                    staging_root: Path, *, val_inputs=None, test_inputs=None,
                    split: SplitConfig | None = None,
                    intensity_norm: str = "p95", compute_hag: bool = False,
                    ground_value: int | None = None,
                    hag_filter: str = "grid",
                    ground_method: str = "labels",
                    feature_fields: list[str] | None = None,
                    geo_features: list[str] | None = None,
                    geo_k: int = 100,
                    rgb_fields: list[str] | None = None,
                    declared_crs_epsg: int | None = None,
                    progress=None) -> Path:
    """Convert `inputs` (files/folders) into a staged canonical dataset.

    classes = [{"index", "source_value", "name"}]; feature_fields = source fields
    baked into every scene as feat_<name>; geo_features = pgeof names computed from
    xyz via Weinmann optimal neighborhood (search ceiling geo_k neighbors).
    compute_hag bakes feat_hag, where ground_value labels are the ONLY ground
    source when set (else CSF detects ground, with the grid heuristic as the
    PDAL-less fallback) and hag_filter = "grid" | "hag_nn" | "hag_delaunay".
    """
    split = split or SplitConfig()
    name = sanitize_name(name)
    out_root = staging_root / name
    if out_root.is_dir() and any(out_root.iterdir()):
        raise FileExistsError(
            f"Dataset '{name}' already exists at {out_root}. Rebuilding on top of it "
            f"would mix old and new scenes across train/val/test. Delete that folder, "
            f"or build under a new dataset name.")
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

    if compute_hag:
        from . import pretrain
        if hag_filter != "grid" and not pretrain.pdal_available():
            say(f"⚠ {hag_filter} needs PDAL (not installed) - using the grid method instead.")
            hag_filter = "grid"
        if ground_method in ("csf", "smrf") and not pretrain.pdal_available():
            say(f"⚠ {pretrain.GROUND_LABELS[ground_method]} needs PDAL (not installed) - "
                "using Z-min proxy (percentile-Z raster) instead.")
            ground_method = "zmin"
        if ground_method == "labels" and ground_value is None:
            raise ValueError("HAG 'Base off ground layer' needs a ground class - set one on "
                             "the Classes table, or pick CSF / SMRF / Z-min proxy.")
        src = pretrain.GROUND_LABELS[ground_method] + (
            f" (class {ground_value})" if ground_method == "labels" else "")
        say(f"computing HeightAboveGround inline ({src} -> {hag_filter}) …")
    if geo_features:
        say(f"computing geometric feature(s) inline (pgeof optimal, "
            f"k≤{geo_k}): {', '.join(geo_features)} …")
    scene_stats = _plan_and_convert(input_files, val_files, test_files, split, spec,
                                    value_to_index, num_classes, out_root,
                                    intensity_norm, say, compute_hag=compute_hag,
                                    ground_value=ground_value,
                                    hag_filter=hag_filter, ground_method=ground_method,
                                    feature_fields=feature_fields,
                                    geo_features=geo_features,
                                    geo_k=geo_k,
                                    rgb_fields=rgb_fields,
                                    declared_crs_epsg=declared_crs_epsg)
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
    for w in sorted({s["hag_warning"] for s in all_stats if s.get("hag_warning")}):
        say(w)
    if len({s.get("crs_wkt") for s in all_stats} - {None}) > 1:
        say("ℹ scenes span multiple processing-CRS zones. Fine: coords are meters "
            "and trainers frame-invariant.")
    total_pts = sum(s["n_points"] for s in all_stats)
    total_area = sum(s["area_m2"] for s in all_stats)
    density = total_pts / max(total_area, 1.0)
    spacing = (total_area / max(total_pts, 1)) ** 0.5
    for c in classes:
        for sp in _SPLITS:
            c[f"{sp}_count"] = sum(s["class_counts"].get(int(c["index"]), 0)
                                   for s in scene_stats[sp])

    idx_to_name: dict[int, str] = {}
    for c in classes:
        idx_to_name.setdefault(int(c["index"]), c["name"])

    by_index: dict[int, dict] = {}
    for c in classes:
        i = int(c["index"])
        e = by_index.get(i)
        if e is None:
            by_index[i] = {"index": i, "name": c["name"],
                           "source_values": [int(c["source_value"])],
                           "asprs_value": (int(c["asprs_value"])
                                           if c.get("asprs_value") is not None else None),
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
    if not all(s["has_hag"] for s in all_stats):
        hag_src = ""
    elif not compute_hag:
        hag_src = "source_dimension"
    elif ground_method == "zmin":
        hag_src = "zmin"
    else:
        hag_src = f"{hag_filter}+{ground_method}"
    meta = {
        "schema_version": 2,
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "inputs": [str(p) for p in (inputs if isinstance(inputs, (list, tuple)) else [inputs])],
            "val_inputs": [str(p) for p in (val_inputs or [])],
            "test_inputs": [str(p) for p in (test_inputs or [])],
            "split_strategy": strategy,
            "label_field": spec.field if spec else "",
            "intensity_norm": intensity_norm,
            "hag_source": hag_src,
            "rgb_fields": rgb_fields,
            "hag_ground_value": (int(ground_value) if ground_value is not None else None),
            "hag_ground_method": (ground_method if compute_hag else None),
            # source_field "@geo:<name>"/"@hag:<method>" = computed channels, recomputed at inference; canonical intensity/return_number ride the has_* flags instead
            "feature_channels": ([{"name": feat_key(f.partition("=")[0]),
                                   "source_field": f.partition("=")[2] or f,
                                   "norm": "p95abs"}
                                  for f in split_feature_fields(feature_fields)[1]]
                                 + [{"name": f"geo_{nm.lower()}",
                                     "source_field": f"@geo:{nm}", "norm": "raw",
                                     "k": int(geo_k)}
                                    for nm in (geo_features or [])]
                                 + ([{"name": "hag", "source_field": f"@hag:{hag_src}",
                                      "norm": "raw"}] if hag_src else [])),
            "declared_crs_epsg": (int(declared_crs_epsg)
                                  if declared_crs_epsg is not None else None),
            "crs": {s["scene"]: {"proc": s.get("crs_wkt"),
                                 "source": s.get("source_crs_wkt")} for s in all_stats},
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

    # atomic: a crash mid-write must never corrupt the dataset's manifest
    tmp = out_root / "dataset_meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, out_root / "dataset_meta.json")
    say(f"staged dataset -> {out_root}")
    return out_root


_GROUND_FIELD_CANDIDATES = ("classification", "Classification", "scalar_label",
                            "label", "class")


def convert_infer_job(job_id: str, input_dir: str, staging_root: Path, progress=None,
                      intensity_norm: str = "p95", hag: bool = False,
                      hag_filter: str = "grid", ground_value: int | None = None,
                      ground_method: str = "labels",
                      feature_fields: list[str] | None = None,
                      geo_features: list[str] | None = None,
                      geo_k: int = 100,
                      declared_crs_epsg: int | None = None,
                      out_dir: Path | None = None) -> Path:
    """Label-less conversion for inference: each input file becomes
    <out_dir>/<stem>_input.npz (flat, next to where predictions will merge in)
    plus one job.json manifest. input_dir may be a single file or a folder.
    intensity_norm MUST match the weights' training norm - a mismatch feeds
    out-of-distribution intensity. hag/ground_value behave as on the Datasets
    page; a scene without ground_value degrades to detection, never to no HAG."""
    say = progress or (lambda s: None)
    feature_fields = list(feature_fields or []) + ["intensity", "return_number"]
    if hag:
        from . import pretrain
        if hag_filter != "grid" and not pretrain.pdal_available():
            say(f"  ⚠ {hag_filter} needs PDAL (not installed) - using the grid method "
                "instead (approximate HAG).")
            hag_filter = "grid"
        if ground_method in ("csf", "smrf") and not pretrain.pdal_available():
            say(f"  ⚠ {pretrain.GROUND_LABELS[ground_method]} needs PDAL - using Z-min "
                "proxy instead.")
            ground_method = "zmin"
        src = pretrain.GROUND_LABELS[ground_method] + (
            f" (class {ground_value})" if ground_method == "labels" else "")
        say(f"  computing HeightAboveGround per scene ({src} -> {hag_filter}) …")
    if geo_features:
        say(f"  computing geometric feature(s) per scene (pgeof optimal, "
            f"k≤{geo_k}): {', '.join(geo_features)} …")
    out_root = Path(out_dir) if out_dir else (staging_root / "_infer" / job_id)
    inp = Path(input_dir)
    files = [inp] if inp.is_file() else discover_scenes(input_dir)
    if not files:
        raise FileNotFoundError(f"No supported point-cloud files in {input_dir}")
    spec = None
    if hag and ground_value is not None:
        fields = list_label_fields(files[0])
        field = next((f for f in _GROUND_FIELD_CANDIDATES if f in fields), None)
        if field is None:
            raise ValueError(
                f"Ground class {ground_value} was set, but {files[0].name} carries no "
                f"classification field (has: {sorted(fields)}). Clear the ground class "
                f"to detect ground instead.")
        spec = LabelSpec(field=field)
        say(f"  ground from the '{field}' field, value {ground_value}")
    scenes, stats = [], []
    for path in files:
        out_path = out_root / (path.stem + "_input.npz")
        say(f"  converting {path.name} ...")
        st = convert_scene(path, spec, {}, out_path, intensity_norm=intensity_norm,
                           compute_hag=hag, ground_value=ground_value,
                           hag_filter=hag_filter, ground_method=ground_method, feature_fields=feature_fields,
                           geo_features=geo_features, geo_k=geo_k,
                           declared_crs_epsg=declared_crs_epsg,
                           missing_fields="skip")
        st["scene"] = out_path.name
        st["source_file"] = str(path)
        if st.get("crs_warning"):
            say("  " + st["crs_warning"])
        if st.get("hag_warning"):
            say("  " + st["hag_warning"])
        if st.get("skipped_fields"):
            say(f"  ⚠ {path.stem}: field(s) not in this file - not baked, the "
                f"model receives zeros: {', '.join(st['skipped_fields'])}")
        scenes.append(out_path.name)
        stats.append(st)
    update_job_json(out_root, "staging", {
        "schema_version": 2,
        "job_id": job_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_dir),
        "scenes": scenes,
        "sources": {s["scene"]: s["source_file"] for s in stats},
        "intensity_norm": intensity_norm, "hag": bool(hag),
        "hag_filter": hag_filter, "ground_method": ground_method,
        "ground_value": ground_value, "feature_fields": feature_fields,
        "geo_features": geo_features, "geo_k": geo_k,
        "declared_crs_epsg": declared_crs_epsg,
    })
    say(f"staged inference job -> {out_root}")
    return out_root


# e57 deliberately absent: ASTM E2807 defines no classification attribute
PRED_EXPORT_FORMATS = ("las", "laz", "ply", "txt", "csv")


def update_job_json(job_dir, section, doc):
    """job.json is the single per-job manifest (staging/infer/postproc/
    panoptic/report); each stage owns one section, atomically rewritten."""
    path = Path(job_dir) / "job.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[section] = doc
    tmp = path.with_name("job.json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return doc


def infer_stem(path) -> str:
    """tile_input.npz / tile_pred.npz -> tile (the source-file stem)."""
    return re.sub(r"_(input|pred)$", "", Path(path).stem)


def pred_files(pred_dir) -> list[Path]:
    """Prediction-bearing npz files in a job dir: merged *_input.npz (the
    one-file layout) plus standalone *_pred.npz (ensemble members, Modal
    pulls); when both exist for a stem the merged file wins. The job.json
    staging manifest is consulted first (manifests over globs); glob covers
    manifest-less dirs."""
    pred_dir = Path(pred_dir)
    try:
        job = appstate_read(pred_dir / "job.json") or {}
    except (OSError, ValueError):
        job = {}   # corrupt manifest: fall back to the glob rather than hide npz
    names = job.get("staging", {}).get("scenes") or []
    listed = [pred_dir / n for n in names]
    inputs = (listed if listed and all(q.is_file() for q in listed)
              else sorted(pred_dir.glob("*_input.npz")))
    by_stem: dict[str, Path] = {}
    for q in inputs:
        by_stem.setdefault(infer_stem(q), q)
    for q in sorted(pred_dir.glob("*_pred.npz")):
        by_stem.setdefault(infer_stem(q), q)
    return sorted(by_stem.values())


def appstate_read(path):
    """appstate.read_json's contract without the import (dataset.py stays
    GUI-framework-free for the shipped runtime): None only when the file isn't
    there; unreadable or malformed raises."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def merge_predictions(job_dir, pred_src_dir, progress=None) -> int:
    """Fold standalone <stem>_pred.npz (a Modal pull) into the job's
    <stem>_input.npz; the pred file is deleted after the fold so the job dir
    ends at one npz per input. Returns the number of scenes merged."""
    say = progress or (lambda s: None)
    n = 0
    for pf in sorted(Path(pred_src_dir).glob("*_pred.npz")):
        scene = Path(job_dir) / (infer_stem(pf) + "_input.npz")
        if not scene.exists():
            say(f"  ⚠ {pf.name}: no matching *_input.npz in {job_dir}; kept as-is")
            continue
        with np.load(scene) as z:
            d = {k: z[k] for k in z.files}
        # fresh predictions invalidate any previous postproc/clustering state
        for stale in ("classification_orig", "confidence_orig", "instance", "probs"):
            d.pop(stale, None)
        with np.load(pf) as z:
            for k in z.files:
                # scene already holds geometry/channels; take predictions only
                if k not in ("xyz", "intensity", "crs_wkt", "source_crs_wkt"):
                    d[k] = z[k]
        tmp = scene.with_name(scene.stem + ".tmp.npz")
        np.savez(tmp, **d)
        os.replace(tmp, scene)
        pf.unlink()
        n += 1
        say(f"  merged predictions into {scene.name}")
    return n


# ASPRS LAS 1.4 standard codes by class-name keyword; first hit wins, so the
# low/medium vegetation aliases must outrank the generic vegetation ones
ASPRS_ALIASES = [
    (2, ("ground", "terrain", "bare earth", "bareearth")),
    (3, ("low veg", "grass", "shrub")),
    (4, ("med veg", "medium veg", "bush", "hedge")),
    (5, ("veg", "tree", "canopy", "forest", "foliage")),
    (6, ("building", "roof", "structure", "facade", "house")),
    (7, ("noise", "low point", "outlier")),
    (9, ("water", "river", "lake", "pond", "sea", "pool")),
    (10, (r"rail\b", "railway", "railroad")),
    (11, ("road", "pavement", "asphalt", "sidewalk")),
    (14, ("wire", "power line", "powerline", "conductor", "cable", "catenary")),
    (15, ("tower", "pylon", "transmission")),
    (17, ("bridge", "overpass", "viaduct")),
    (1, ("unclassified", "unknown", "unlabeled", "unlabelled", "clutter", "default")),
]
ASPRS_USER_BASE = 64  # first user-definable code (LAS 1.4)


def asprs_code_for(name: str) -> int | None:
    """The ASPRS code whose alias starts a word in the class name ('veg'
    matches 'high_vegetation'); None = no standard code for this name.
    Alias entries are regex fragments anchored at a word start."""
    n = re.sub(r"[_\-./]+", " ", str(name).lower()).strip()
    for code, keys in ASPRS_ALIASES:
        if any(re.search(rf"\b{k}", n) for k in keys):
            return code
    return None


def _cls_int(c, key: str) -> int:
    """One int field of a dataset-meta classes entry ('source_value' honours the
    source_values list). ValueError naming the entry when it's unusable."""
    if not isinstance(c, dict):
        v = c
    elif key == "source_value":
        src = c.get("source_values") or [c.get("source_value")]
        v = src[0] if src else None
    else:
        v = c.get(key)
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(f"dataset meta class entry {c!r} has no usable '{key}' "
                         f"(got {v!r}); fix that entry in dataset_meta.json, or "
                         f"set Output codes to 'raw model indices', then export "
                         f"again.") from None


def export_class_map(meta_classes, mode: str) -> dict | None:
    """{model index: exported LAS code} from a dataset meta's classes list.
    mode 'asprs': the class's stored asprs_value, else a name match, else its
    source value (collisions fall into the 64+ user band). mode 'source': the
    training data's source values. Returns None ONLY for mode 'raw' (and for an
    empty classes list); a malformed entry raises ValueError naming it, because
    falling back to raw indices writes them into a file GIS reads as ASPRS."""
    if mode == "raw" or not meta_classes:
        return None
    cs = sorted(meta_classes, key=lambda c: _cls_int(c, "index"))
    if mode == "source":
        return {_cls_int(c, "index"): _cls_int(c, "source_value") for c in cs}
    cmap = {_cls_int(c, "index"): (_cls_int(c, "asprs_value")
                                   if c.get("asprs_value") is not None
                                   else asprs_code_for(c.get("name", "")))
            for c in cs}
    taken = {v for v in cmap.values() if v is not None}
    nxt = ASPRS_USER_BASE
    for c in cs:
        i = _cls_int(c, "index")
        if cmap[i] is not None:
            continue
        src = c.get("source_values") or [c.get("source_value")]
        # no source value at all is legal here - it just falls into the user band
        v = _cls_int(c, "source_value") if src and src[0] is not None else None
        if v is None or v in taken:
            v = max(nxt, ASPRS_USER_BASE)
            while v in taken:
                v += 1
            nxt = v + 1
        cmap[i] = v
        taken.add(v)
    return cmap


def class_map_for_export(meta: dict | None, mode: str, say, subject: str) -> dict | None:
    """export_class_map over a dataset meta, logging the raw-indices fallback via
    `say`; `subject` names the weights/run in the log line. Only a meta with no
    classes falls back - a malformed one raises (see export_class_map)."""
    classes = (meta or {}).get("classes", [])
    if mode != "raw" and not classes:
        say(f"[export] no dataset meta for {subject}; exported codes are raw "
            f"model indices. Register that dataset on the Datasets page "
            f"(Add existing…) to export real class codes.")
    return export_class_map(classes, mode)


def _rewrite_original(orig: Path, dst: Path, cls, confidence=None, member=None,
                      instance=None, extra_dims=None):
    """The original las/laz re-emitted with the predicted classification (+
    extra dims): every source dimension, VLR and CRS survives. COPC metadata is
    stripped (the rewrite invalidates the hierarchy: output is plain las/laz).
    Raises on point-count mismatch - the caller falls back to the xyz-only
    writer."""
    import laspy
    las = laspy.read(str(orig))
    if len(las.points) != len(cls):
        raise ValueError(f"{len(las.points):,} points vs {len(cls):,} predictions")
    if int(cls.max(initial=0)) > 31 and las.header.point_format.id < 6:
        # 5-bit classification below format 6; formats 6/7 so colour survives the upconvert
        target = 7 if "red" in las.point_format.dimension_names else 6
        las = laspy.convert(las, point_format_id=target)
    for lst in (las.header.vlrs, las.header.evlrs or []):
        lst[:] = [v for v in lst if getattr(v, "user_id", "") != "copc"]
    named = [("confidence", confidence, np.float32),
             ("ens_member", member, np.uint8),
             ("instance_id", instance, np.uint32)]
    named += [(n, a, t) for n, (a, t) in (extra_dims or {}).items()]
    for name, arr, typ in named:
        if arr is not None:
            if name not in las.point_format.dimension_names:
                las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=typ))
            setattr(las, name, arr)
    las.classification = cls
    las.write(str(dst))


def _prob_extra_dims(probs, entropy, margin, prob_dims, class_names):
    """las/laz Extra Bytes derived from the saved (N, C) softmax: normalized
    entropy + top-2 margin as float32, optional per-class dims quantized to
    uint8 (p*255) since C float dims would double the file."""
    p = np.asarray(probs, np.float32)
    p /= np.maximum(p.sum(-1, keepdims=True), 1e-12)
    out = {}
    if entropy:
        h = -(p * np.log(np.maximum(p, 1e-12))).sum(-1) / np.log(p.shape[1])
        out["entropy"] = (h.astype(np.float32), np.float32)
    if margin:
        top2 = np.partition(p, -2, axis=-1)[:, -2:]
        out["margin"] = ((top2[:, 1] - top2[:, 0]).astype(np.float32), np.float32)
    if prob_dims:
        names = list(class_names or []) + [str(i) for i in range(len(class_names or []),
                                                                 p.shape[1])]
        for i in range(p.shape[1]):
            safe = "".join(c if c.isalnum() else "_" for c in str(names[i]))
            out[f"prob_{safe}"[:32]] = ((p[:, i] * 255).round().astype(np.uint8),
                                        np.uint8)
    return out


def export_predictions(pred_dir, fmt: str, progress=None, class_map=None,
                       unclass_threshold=None, originals=None,
                       entropy=False, margin=False, prob_dims=False,
                       class_names=None) -> list[Path]:
    """Export each inferred npz as <stem>_pred.<fmt>, with class_map
    remapping model indices to output codes. las/laz with a matching original
    las/laz in `originals` ({input stem: path}) carries every source dimension
    over; anything else writes xyz + classification [+ confidence]. Points
    below unclass_threshold export as class 1 (0 if 1 is taken; 255 with no
    class_map). The npz keeps the raw prediction so a new threshold re-exports
    without re-running. entropy/margin/prob_dims add per-point uncertainty
    Extra Bytes (las/laz only) derived from the npz "probs" key, which exists
    when inference ran with TT_SAVE_PROBS=1."""
    fmt = fmt.lower().lstrip(".")
    if fmt not in PRED_EXPORT_FORMATS:
        raise ValueError(f"unsupported prediction format '{fmt}' "
                         f"(one of {', '.join(PRED_EXPORT_FORMATS)})")
    say = progress or (lambda s: None)
    unclass = 1
    if unclass_threshold is not None and class_map and 1 in class_map.values():
        unclass = 0
        say("  (class 1 is taken by the model's own classes; low-confidence "
            "points export as class 0 instead)")
    elif unclass_threshold is not None and not class_map:
        unclass = 255
        say("  (no class map: export carries raw model indices; low-confidence "
            "points export as 255)")
    written: list[Path] = []
    legend_said = False
    for src in pred_files(pred_dir):
        with np.load(src) as d:
            if "classification" not in d.files:
                say(f"  ⚠ {src.name}: staged scene without predictions yet - "
                    f"run inference first; skipped.")
                continue
            xyz = np.asarray(d["xyz"], np.float64)
            cls = np.asarray(d["classification"], np.int64)
            conf = (np.asarray(d["confidence"], np.float32)
                    if "confidence" in d.files else None)
            crs_wkt = str(d["crs_wkt"]) if "crs_wkt" in d.files else None
            source_wkt = str(d["source_crs_wkt"]) if "source_crs_wkt" in d.files else None
            member = (np.asarray(d["dominant_member"], np.uint8)
                      if "dominant_member" in d.files else None)
            member_names = ([str(s) for s in d["member_names"]]
                            if "member_names" in d.files else None)
            instance = (np.asarray(d["instance"], np.uint32)
                        if "instance" in d.files else None)
            want_probs = entropy or margin or prob_dims
            probs = (np.asarray(d["probs"], np.float32)
                     if want_probs and "probs" in d.files else None)
        extra_dims = None
        if want_probs and fmt in ("las", "laz"):
            if probs is None:
                say(f"  ⚠ {src.name}: no probs in the npz (inference ran without "
                    f"save-probs); skipping uncertainty dims.")
            else:
                extra_dims = _prob_extra_dims(probs, entropy, margin, prob_dims,
                                              class_names)
        if member is not None and fmt in ("las", "laz") and not legend_said:
            legend_said = True
            say("  ens_member field (ensemble's dominant model per point): "
                + (", ".join(f"{i}={n}" for i, n in enumerate(member_names))
                   if member_names else "member indices in ensemble input order"))
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
        cls = np.clip(cls, 0, 255).astype(np.uint8)
        dst = src.with_name(f"{infer_stem(src)}_pred.{fmt}")
        carried = False
        orig = (originals or {}).get(infer_stem(src))
        if fmt in ("las", "laz") and orig and Path(orig).suffix.lower() in (".las", ".laz"):
            try:
                _rewrite_original(Path(orig), dst, cls, confidence=conf,
                                  member=member, instance=instance,
                                  extra_dims=extra_dims)
                carried = True
            except Exception as e:
                say(f"  ⚠ {Path(orig).name}: can't carry source dimensions "
                    f"({e}); writing xyz + classification only.")
        if not carried:
            _write_pred(dst, xyz, cls, fmt, say=say, confidence=conf, crs_wkt=crs_wkt,
                        source_crs_wkt=source_wkt, member=member, instance=instance,
                        extra_dims=extra_dims)
        written.append(dst)
        say(f"  {src.name} -> {dst.name} ({len(xyz):,} pts"
            + ("; source dimensions carried over" if carried else "")
            + (f"; {low:,} below confidence {unclass_threshold:g} -> class {unclass}"
               if unclass_threshold is not None and conf is not None else "") + ")")
    return written


def prediction_report(pred_dir, class_names=None, unclass_threshold=None,
                      margin_threshold=0.1, progress=None) -> dict:
    """Job-level QA summary over every inferred npz: per-class counts and mean
    confidence, plus mean entropy and contested-point counts when probs were
    saved. Written to job.json's "report" section and returned. Uncertainty
    stats are model confidence, not calibrated probability."""
    say = progress or (lambda s: None)
    names = list(class_names or [])
    scenes = {}
    for src in pred_files(pred_dir):
        with np.load(src) as d:
            if "classification" not in d.files:
                continue
            cls = np.asarray(d["classification"], np.int64)
            conf = (np.asarray(d["confidence"], np.float32)
                    if "confidence" in d.files else None)
            probs = (np.asarray(d["probs"], np.float32)
                     if "probs" in d.files else None)
            inst = np.asarray(d["instance"]) if "instance" in d.files else None
        n = len(cls)
        per_class = {}
        for c in np.unique(cls):
            m = cls == c
            label = names[c] if c < len(names) else str(int(c))
            per_class[label] = {
                "points": int(m.sum()),
                "fraction": round(float(m.mean()), 4),
                "mean_confidence": (round(float(conf[m].mean()), 4)
                                    if conf is not None else None),
            }
        entry = {"points": n, "classes": per_class}
        if conf is not None and unclass_threshold is not None:
            entry["below_unclass_threshold"] = int((conf < unclass_threshold).sum())
        if probs is not None:
            p = probs / np.maximum(probs.sum(-1, keepdims=True), 1e-12)
            h = -(p * np.log(np.maximum(p, 1e-12))).sum(-1) / np.log(p.shape[1])
            top2 = np.partition(p, -2, axis=-1)[:, -2:]
            entry["mean_entropy"] = round(float(h.mean()), 4)
            entry["contested_points"] = int(
                ((top2[:, 1] - top2[:, 0]) < margin_threshold).sum())
        if inst is not None:
            entry["instances"] = int(inst.max(initial=0))
        scenes[infer_stem(src)] = entry
    if not scenes:
        raise RuntimeError(f"no inferred npz files in {pred_dir}")
    total = sum(s["points"] for s in scenes.values())
    report = {"schema": 1, "class_names": names, "total_points": total,
              "margin_threshold": margin_threshold,
              "unclass_threshold": unclass_threshold, "scenes": scenes}
    update_job_json(pred_dir, "report", report)
    agg = {}
    for s in scenes.values():
        for label, e in s["classes"].items():
            agg[label] = agg.get(label, 0) + e["points"]
    say(f"  report: {total:,} pts | " + ", ".join(
        f"{k} {v / total:.0%}" for k, v in sorted(agg.items(), key=lambda x: -x[1])))
    return report


def _write_pred(dst: Path, xyz: np.ndarray, cls: np.ndarray, fmt: str,
                confidence=None, crs_wkt=None, source_crs_wkt=None, member=None,
                instance=None, extra_dims=None, say=print):
    """One classified cloud -> dst, with xyz inverse-transformed back to the source
    frame via the shared keystone restore (raises the D2 legacy block). The embedded
    las/laz WKT VLR is the SOURCE CRS so deliverables overlay the input clouds.
    confidence rides in las/laz (Extra Bytes) and txt/csv, member (ensemble dominant
    model) as las/laz "ens_member", instance (panoptic, 0 = stuff) as las/laz
    "instance_id" and a txt/csv column, and ply stays classification-only."""
    xyz = restore_to_source(xyz, crs_wkt, source_crs_wkt)
    geo_wkt = source_crs_wkt or crs_wkt
    if fmt in ("las", "laz"):
        import laspy
        h = laspy.LasHeader(point_format=6, version="1.4")
        if confidence is not None:
            h.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.float32))
        if member is not None:
            h.add_extra_dim(laspy.ExtraBytesParams(name="ens_member", type=np.uint8))
        if instance is not None:
            h.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type=np.uint32))
        for name, (_arr, typ) in (extra_dims or {}).items():
            h.add_extra_dim(laspy.ExtraBytesParams(name=name, type=typ))
        if geo_wkt:
            try:
                from pyproj import CRS
                h.add_crs(CRS.from_wkt(geo_wkt))
            except Exception as e:
                # written unreferenced, it silently fails to overlay the input
                say(f"  ! {dst.name}: could not embed the CRS ({e}); the file "
                    f"is written WITHOUT georeferencing")
        h.offsets = xyz.min(0)
        h.scales = [0.001] * 3
        las = laspy.LasData(h)
        las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        las.classification = cls
        if confidence is not None:
            las.confidence = confidence
        if member is not None:
            las.ens_member = member
        if instance is not None:
            las.instance_id = instance
        for name, (arr, _typ) in (extra_dims or {}).items():
            setattr(las, name, arr)
        las.write(str(dst))
    elif fmt == "ply":
        header = ("ply\nformat ascii 1.0\n" + f"element vertex {len(xyz)}\n"
                  "property double x\nproperty double y\nproperty double z\n"
                  "property uchar classification\nend_header")
        np.savetxt(dst, np.column_stack([xyz, cls]),
                   fmt=["%.3f"] * 3 + ["%d"], header=header, comments="")
    elif fmt == "txt":
        cols = [xyz, cls] + ([confidence] if confidence is not None else []) \
            + ([instance] if instance is not None else [])
        np.savetxt(dst, np.column_stack(cols), fmt="%.3f %.3f %.3f %d"
                   + (" %.3f" if confidence is not None else "")
                   + (" %d" if instance is not None else ""))
    elif fmt == "csv":
        cols = [xyz, cls] + ([confidence] if confidence is not None else []) \
            + ([instance] if instance is not None else [])
        np.savetxt(dst, np.column_stack(cols), delimiter=",",
                   fmt=["%.3f"] * 3 + ["%d"]
                   + (["%.3f"] if confidence is not None else [])
                   + (["%d"] if instance is not None else []),
                   header="x,y,z,classification"
                   + (",confidence" if confidence is not None else "")
                   + (",instance" if instance is not None else ""),
                   comments="")
