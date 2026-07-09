"""Convert user folders/files into the canonical dataset the scripts consume.

Canonical layout (staged locally, then `modal volume put terminal-datasets ...`):
  <staging>/<name>/
    dataset_meta.json
    train/<scene>.npz     # xyz f32, label i32 (-1 = ignore), [rgb u8, intensity f32 0..1,
    val/<scene>.npz       #   return_number f32]
    test/<scene>.npz
Inference jobs use the same npz minus `label`, under scenes/.

The dataset stage decides a 3-way train/val/test split ONCE and materializes the
three folders; the training scripts read them verbatim (val = selection holdout,
test = final report) and never re-carve their own. The split is a property of the
DATASET, recorded in dataset_meta.json — not a per-script constant.

The split targets are POINT-COUNT fractions for val and test (train = remainder),
approximated greedily over atoms. A FOLDER of clouds splits by whole scenes; a
SINGLE cloud is tiled only as a MEASUREMENT tool, the tiles allocated to splits
and then reassembled into ONE (holey) whole-cloud npz per split with a small seam
buffer discarded to limit leakage. mode="balanced" mirrors the global class mix in
every split (and guarantees rare-class presence); mode="random" fills by point
count alone. The converter is otherwise format-/layout-agnostic; explicit
train/val/test folders ("provided") bypass allocation for whichever splits exist.
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

from .readers import SUPPORTED_EXTS, Cloud, list_label_fields, read_points


@dataclass
class LabelSpec:
    """Where ground-truth labels come from.

    kind="field": a named field in the cloud file itself (LAS dim, PLY prop,
        "column N" of an ASCII/npy file). This is the general/default case.
    kind="file": a companion label file alongside each cloud, one label per point.
        The companion path = truth_dir / scene_name with src_suffix replaced by
        dst_suffix (e.g. a "_PC3.txt" cloud paired with a "_CLS.txt" label file).
    """
    kind: str = "field"            # "field" | "file"
    field: str = ""                # for kind="field"
    truth_dir: str = ""            # for kind="file"
    src_suffix: str = "_PC3.txt"
    dst_suffix: str = "_CLS.txt"


@dataclass
class SplitConfig:
    """How the dataset stage derives the train/val/test split.

    The split is decided ONCE here and materialized as three whole-scene folders;
    the training scripts read those verbatim. val_frac/test_frac are TARGET
    POINT-COUNT fractions (each >= 0.05 in the UI; train = remainder), approximated
    greedily over atoms — whole scenes for a folder of clouds, or grid tiles for a
    single cloud (reassembled into one holey npz per split).

    mode          : "balanced" (greedy stratification so each split mirrors the
                    global class mix, + a rare-class presence guarantee) or
                    "random" (class-blind point-count fill).
    seed          : RNG seed (shown in the UI; default 42).
    seam_buffer_m : single-cloud only — discard points within this 2-D distance of
                    a boundary between atoms assigned to different splits.
    tile_m        : single-cloud measurement grid size (atoms), metres.
    strategy      : "auto" (allocate) or "provided" (explicit val/test folders; any
                    split not provided is allocated from the inputs).
    """
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
    # A converted dataset has no clouds at the top level — its scenes live under
    # train/, val/ and test/ (the canonical npz layout). Look there so HAG, Scan
    # and Analyze accept a converted dataset folder, not just a flat folder of clouds.
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


def sanitize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_")
    if not s:
        raise ValueError("Dataset name is empty after sanitizing")
    return s.lower()


# --------------------------------------------------------------------- splitting
#
# Atoms (the indivisible units the allocator assigns) are whole scenes for a folder
# of clouds, or grid tiles for a single cloud. Targets are POINT-COUNT fractions.

_SPLITS = ("train", "val", "test")


def _subset(cloud: Cloud, raw: np.ndarray | None, mask: np.ndarray):
    """A boolean-masked copy of a cloud (+ its labels)."""
    sub = Cloud(
        xyz=cloud.xyz[mask],
        rgb=cloud.rgb[mask] if cloud.rgb is not None else None,
        intensity=cloud.intensity[mask] if cloud.intensity is not None else None,
        return_number=cloud.return_number[mask] if cloud.return_number is not None else None,
        fields={k: v[mask] for k, v in cloud.fields.items()},
    )
    return sub, (raw[mask] if raw is not None else None)


def _tile_hists(inv: np.ndarray, n_tiles: int, raw, value_to_index: dict,
                num_classes: int) -> np.ndarray:
    """(n_tiles, C) per-class point counts, one vectorized pass: map raw->index
    once for the whole cloud, then bincount the (tile, class) pairs."""
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
    """Best-effort (balanced mode): make every globally-present class appear in
    every ACTIVE split, moving an atom rich in a missing class over."""
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
    """Every split with a positive target gets >= 1 atom (steal the smallest atom
    from the split that has the most)."""
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
    """Assign atoms to train(0)/val(1)/test(2) approximating val_frac/test_frac of
    TOTAL POINTS. pts[i] = atom point count; hist[i] = per-class point counts.

    mode="random"   : random order; greedily fill the split with the largest
                      remaining point deficit (class-blind).
    mode="balanced" : rarity-first order; place each atom in the split whose
                      resulting per-class point deviation is smallest (each split
                      mirrors the global class mix), then guarantee every present
                      class appears in every split where atoms allow.
    """
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
        # critical class per atom = its rarest globally-present class; place the
        # atom in the split most STARVED of that class (iterative stratification).
        crit = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            pres = np.where(hist[i] > 0)[0]
            if len(pres):
                crit[i] = int(pres[np.argmin(H[pres])])
        rarest = np.array([H[crit[i]] if crit[i] >= 0 else np.inf for i in range(n)])
        order = np.lexsort((-pts, rarest))             # rarest class first, big first
        for i in order:
            need = target - got                        # remaining desired (3, C)
            primary = need[:, crit[i]] if crit[i] >= 0 else need.sum(1)
            key = primary + 1e-9 * need.sum(1) + rng.random_sample(3) * 1e-12
            key[~allowed] = -np.inf
            s = int(np.argmax(key))                    # most-starved active split
            assign[i] = s
            got[s] += hist[i]
        _guarantee_presence(assign, hist.astype(np.int64), allowed)

    _guarantee_nonempty(assign, pts, fr)
    return assign


def _atomize(cloud: Cloud, tile_m: float):
    """XY grid tiles for a single cloud -> (keys, uniq, inv): per-point integer
    cell coords, the unique cells, and each point's tile id (row into uniq)."""
    xy = cloud.xyz[:, :2]
    keys = np.floor((xy - xy.min(0)) / max(float(tile_m), 1e-6)).astype(np.int64)
    # 1-D packed code instead of unique(axis=0): ~10x faster on multi-M clouds
    # (axis=0 sorts rows as structured voids).
    span = int(keys[:, 1].max()) + 1
    code, inv = np.unique(keys[:, 0] * span + keys[:, 1], return_inverse=True)
    uniq = np.column_stack([code // span, code % span])
    return keys, uniq, inv


def _seam_drop(xy, keys, uniq, split_of_point, tile_m: float,
               seam_buffer_m: float) -> np.ndarray:
    """Keep-mask: drop points within seam_buffer_m of a NEIGHBOURING tile that
    belongs to a different split, so the reassembled per-split clouds don't touch.

    ponytail: tile-edge approximation — pure arithmetic on each point's offset
    inside its tile plus a dense neighbour-split lookup, O(n), no KD-tree. It's a
    superset of the exact within-buffer-of-a-foreign-POINT set (a foreign point's
    tile region is never farther than the point), so it can only over-drop a
    little, never leak. Ceiling: only the 8 adjacent tiles are checked, so the
    effective buffer caps at tile_m."""
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

def _hag_from_cloud(cloud: Cloud) -> np.ndarray | None:
    """Return a source HeightAboveGround/HAG field when one exists and aligns 1:1.

    A cloud that already carries HAG (e.g. LAS/LAZ with a `HeightAboveGround`
    extra dim, which readers.py exposes as a numeric field) keeps it verbatim —
    it wins over recomputation. *_hag models require a real HAG feature and
    refuse to run without one.
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
                 compute_hag: bool = False, ground_value: int | None = None,
                 use_smrf: bool = True, hag_filter: str = "grid") -> dict:
    """Write one (already-read, already-cropped) cloud to a canonical npz.

    compute_hag: also store a per-point real HeightAboveGround ("hag", SMRF->hag)
    aligned to xyz — for *_hag weights trained on real HAG, or to bake HAG into a
    dataset during tiling. Computing it here (points already in RAM) is a single
    pass — no reload/re-write round trip. Skipped when PDAL can't produce an
    aligned result; the scene then has no "hag" key and *_hag models refuse it."""
    out: dict[str, np.ndarray] = {"xyz": cloud.xyz.astype(np.float32)}

    class_counts: dict[int, int] = {}
    # No class mapping = inference: `raw` was read only to locate ground (see
    # ground_value below), so writing an all-(-1) label array would be a lie.
    if raw is not None and value_to_index:
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
        gmask = (raw == int(ground_value)) if (raw is not None and ground_value is not None) else None
        h = pretrain.hag_for_cloud(cloud, ground_mask=gmask, use_smrf=use_smrf,
                                   hag_filter=hag_filter)
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
                  compute_hag: bool = False, ground_value: int | None = None,
                  use_smrf: bool = True, hag_filter: str = "grid") -> dict:
    """Read one source file and convert the whole cloud (no cropping)."""
    cloud = read_points(path)
    raw = read_labels(path, cloud, spec) if spec is not None else None
    return _convert_one(cloud, raw, value_to_index, out_path, intensity_norm,
                        compute_hag=compute_hag, ground_value=ground_value,
                        use_smrf=use_smrf, hag_filter=hag_filter)


def _available_ram_bytes() -> int | None:
    """Best-effort available physical RAM (bytes), stdlib only, cross-platform.
    None when it can't be determined — caller then falls back to a core-only cap."""
    try:                                              # Linux: free pages * page size
        if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        pass
    if os.name == "nt":                               # Windows: GlobalMemoryStatusEx
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


# A worker decompresses one cloud and makes a few transient copies (a structured
# array for PDAL, the npz buffer). LAZ inflates ~10-20x; 30x the largest input
# file is a deliberately fat per-worker estimate — overshooting only costs
# parallelism, never a crash. Headroom leaves RAM for the OS, Qt, and PDAL itself.
_RAM_PER_FILE_FACTOR = 30
_RAM_HEADROOM = 0.7


def _worker_cap_detail(files: list[Path]) -> tuple[int, str]:
    """(thread count that won't OOM, human reason). Cap = min(cores, files,
    RAM_budget / est-per-cloud); reason names the binding constraint for the log."""
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


def _worker_cap(files: list[Path]) -> int:
    """Thread count that won't OOM. Falls back to the core/file cap when RAM is unreadable."""
    return _worker_cap_detail(files)[0]


def _convert_many(files: list[Path], dest_for, spec, value_to_index, intensity_norm, say,
                  *, compute_hag, ground_value, use_smrf, hag_filter,
                  max_workers: int | None = None) -> list[dict]:
    """Read + convert each file to its npz CONCURRENTLY; returns the stat dicts in
    INPUT order. Order matters: the caller feeds it to allocate_splits positionally,
    so the split must not depend on which scene's PDAL/SMRF finishes first.

    Threads (not processes): read_points, hag_for_cloud's PDAL execute, and
    savez_compressed all drop the GIL, so this scales with cores. Workers are
    capped at the file count so we never hold more clouds in RAM than there are
    scenes. say() is a queued Qt signal (thread-safe), but we only call it from
    this thread as map yields in order — clean, ordered progress for free."""
    def work(f: Path) -> dict:
        cloud = read_points(f)
        raw = read_labels(f, cloud, spec) if spec is not None else None
        out_path = dest_for(f)
        st = _convert_one(cloud, raw, value_to_index, out_path, intensity_norm,
                          compute_hag=compute_hag, ground_value=ground_value,
                          use_smrf=use_smrf, hag_filter=hag_filter)
        st["scene"] = out_path.name
        return st

    if max_workers is not None:                       # caller override wins
        workers, why = max_workers, "forced"
    else:
        workers, why = _worker_cap_detail(files)       # RAM/core-safe Auto
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
                      use_smrf: bool = True, hag_filter: str = "grid",
                      max_workers: int | None = None) -> dict:
    """Convert sources into out_root/{train,val,test}/*.npz; returns
    {"train": [stats], "val": [stats], "test": [stats]}.

    Explicit val_files/test_files are used verbatim; whichever of val/test is NOT
    provided is allocated from input_files by point count (mode-aware). A single
    input cloud is tiled-as-measurement and reassembled into one holey npz/split."""
    stats = {sp: [] for sp in _SPLITS}

    def emit(split_name: str, cloud: Cloud, raw, scene_name: str, hag_already=False):
        out_path = out_root / split_name / f"{scene_name}.npz"
        st = _convert_one(cloud, raw, value_to_index, out_path, intensity_norm,
                          compute_hag=compute_hag and not hag_already,
                          ground_value=ground_value, use_smrf=use_smrf,
                          hag_filter=hag_filter)
        st["scene"] = out_path.name
        stats[split_name].append(st)

    # Explicit folders verbatim; their fractions drop out of the allocation.
    for split_name, files in (("val", val_files), ("test", test_files)):
        if files:
            stats[split_name].extend(_convert_many(
                files, lambda f, sn=split_name: out_root / sn / f"{f.stem}.npz",
                spec, value_to_index, intensity_norm, say, compute_hag=compute_hag,
                ground_value=ground_value, use_smrf=use_smrf, hag_filter=hag_filter,
                max_workers=max_workers))
    vfrac = 0.0 if val_files else split.val_frac
    tfrac = 0.0 if test_files else split.test_frac

    if vfrac <= 0.0 and tfrac <= 0.0:                  # everything explicit -> all train
        stats["train"].extend(_convert_many(
            input_files, lambda f: out_root / "train" / f"{f.stem}.npz",
            spec, value_to_index, intensity_norm, say, compute_hag=compute_hag,
            ground_value=ground_value, use_smrf=use_smrf, hag_filter=hag_filter,
            max_workers=max_workers))
        return stats

    # Single cloud -> tile as a measurement, allocate tiles, reassemble per split.
    if len(input_files) == 1:
        f = input_files[0]
        say(f"  single-cloud split {f.name} (tile-measure -> holey clouds) ...")
        cloud = read_points(f)
        raw = read_labels(f, cloud, spec) if spec is not None else None
        if compute_hag and _hag_from_cloud(cloud) is None:
            from . import pretrain                      # whole-cloud HAG, ferried via fields
            gmask = (raw == int(ground_value)) if (raw is not None and ground_value is not None) else None
            h = pretrain.hag_for_cloud(cloud, ground_mask=gmask, use_smrf=use_smrf,
                                       hag_filter=hag_filter)
            if h is not None and len(h) == cloud.n:
                cloud.fields["HeightAboveGround"] = h.astype(np.float32)
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

    # Folder of clouds -> convert all to a pool, allocate WHOLE scenes, move.
    say(f"  converting {len(input_files)} scene(s), then allocating by point count ...")
    pool = _convert_many(input_files, lambda f: out_root / "_pool" / f"{f.stem}.npz",
                         spec, value_to_index, intensity_norm, say, compute_hag=compute_hag,
                         ground_value=ground_value, use_smrf=use_smrf, hag_filter=hag_filter,
                         max_workers=max_workers)
    for st in pool:                                    # map kept input order -> deterministic alloc
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
                    intensity_norm: str = "max", compute_hag: bool = False,
                    ground_value: int | None = None, use_smrf: bool = True,
                    hag_filter: str = "grid", max_workers: int | None = None,
                    progress=None) -> Path:
    """Convert `inputs` (files and/or folders) into a staged canonical dataset with
    materialized train/val/test folders.

    inputs: a path or list of paths (files or folders) to use as the source.
    val_inputs/test_inputs: optional explicit val/test sources -> "provided" split;
        any split not given is allocated from `inputs` by point count.
    split: how to allocate train/val/test (fractions, mode, seed) when a split
        isn't provided explicitly (default: SplitConfig()).
    classes: [{"index", "source_value", "name"}] — built by the Datasets page.
    compute_hag: bake a per-scene HeightAboveGround channel into every scene in
        the SAME pass as conversion. hag_filter picks the method: "grid" (fast
        raster approximation, no PDAL, default) or "hag_nn"/"hag_delaunay"
        (accurate PDAL path). ground_value names the source label value that
        means GROUND — when set, those labels are the ONLY ground source (SMRF
        never runs; holes are nearest-filled/interpolated). With no ground_value
        the method detects ground itself: the grid's opening heuristic, or SMRF
        for the PDAL filters (use_smrf).
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

    if compute_hag:
        from . import pretrain
        if hag_filter != "grid" and not pretrain.pdal_available():
            say(f"⚠ {hag_filter} needs PDAL (not installed) - using the grid method instead.")
            hag_filter = "grid"
        # Trusted ground labels win outright; the grid method detects for itself.
        use_smrf = use_smrf and ground_value is None and hag_filter != "grid"
        src = (f"ground=class {ground_value}" if ground_value is not None
               else ("SMRF" if use_smrf else "grid detection"))
        say(f"computing HeightAboveGround inline ({src} -> {hag_filter}) …")
    scene_stats = _plan_and_convert(input_files, val_files, test_files, split, spec,
                                    value_to_index, num_classes, out_root,
                                    intensity_norm, say, compute_hag=compute_hag,
                                    ground_value=ground_value, use_smrf=use_smrf,
                                    hag_filter=hag_filter, max_workers=max_workers)
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
    total_pts = sum(s["n_points"] for s in all_stats)
    total_area = sum(s["area_m2"] for s in all_stats)
    density = total_pts / max(total_area, 1.0)
    spacing = (total_area / max(total_pts, 1)) ** 0.5
    for c in classes:
        for sp in _SPLITS:
            c[f"{sp}_count"] = sum(s["class_counts"].get(int(c["index"]), 0)
                                   for s in scene_stats[sp])

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
    # How every scene's HAG was produced (recorded for the GUI + reproducibility;
    # inference reads it back through the run manifest to reproduce the METHOD).
    if not all(s["has_hag"] for s in all_stats):
        hag_src = ""
    elif not compute_hag:
        hag_src = "source_dimension"          # HAG came from a dim already in the cloud
    elif ground_value is not None:
        hag_src = f"{hag_filter}+labels"
    else:
        hag_src = "grid" if hag_filter == "grid" else f"{hag_filter}+smrf"
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
            "hag_ground_value": (int(ground_value) if ground_value is not None else None),
            "hag_use_smrf": bool(use_smrf),
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


# Field names that plausibly carry a classification, best first. Same order the
# Datasets page offers when it picks a label field (see datasets_page._scan_labels).
_GROUND_FIELD_CANDIDATES = ("classification", "Classification", "scalar_label",
                            "label", "class")


def convert_infer_job(job_id: str, input_dir: str, staging_root: Path, progress=None,
                      intensity_norm: str = "p95", hag: bool = False,
                      hag_filter: str = "grid", ground_value: int | None = None,
                      out_dir: Path | None = None) -> Path:
    """Label-less conversion for inference-only jobs -> <staging>/_infer/<job_id>/,
    or to `out_dir` when given (the caller nests it under the owning dataset, e.g.
    <dataset>/infer/<job_id>). The container mount name stays /datasets/_infer/<job>
    regardless — only the host folder changes.

    intensity_norm MUST match what the weights were trained with (max -> [0,1], or
    p95 -> [0,2] for weights trained that way) — a mismatch feeds the net
    out-of-distribution intensity and tanks accuracy for every checkpoint.

    `hag=True` (required by *_hag weights) computes a per-point "hag" channel with
    `hag_filter`. `ground_value` names the classification value that means ground —
    those points become the ONLY ground source, exactly as on the Datasets page.
    Left None, ground is detected instead (grid's opening heuristic, or PDAL SMRF).
    A PDAL filter without PDAL installed falls back to grid.

    use_smrf is deliberately not a parameter: hag_for_cloud already forces it off
    once a usable ground mask arrives, and a scene that happens to contain none of
    `ground_value` degrades to detection rather than producing no HAG at all.
    """
    say = progress or (lambda s: None)
    if hag:
        from . import pretrain
        if hag_filter != "grid" and not pretrain.pdal_available():
            say(f"  ⚠ {hag_filter} needs PDAL (not installed) - using the grid method "
                "instead (approximate HAG).")
            hag_filter = "grid"
        src = (f"ground=class {ground_value}" if ground_value is not None
               else ("grid detection" if hag_filter == "grid" else "SMRF"))
        say(f"  computing HeightAboveGround per scene ({src} -> {hag_filter}) …")
    out_root = Path(out_dir) if out_dir else (staging_root / "_infer" / job_id)
    files = discover_scenes(input_dir)
    if not files:
        raise FileNotFoundError(f"No supported point-cloud files in {input_dir}")
    # One spec for the whole job: per-file resolution would let scenes key off
    # different fields and silently change what "ground" means mid-job. A file that
    # lacks the chosen field fails loud in read_labels.
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
                           hag_filter=hag_filter)
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


# Prediction export formats the Inference page offers. All carry xyz +
# classification only (no colour — the palette is the viewer's concern, not the
# deliverable's). e57 is deliberately absent: ASTM E2807 defines no
# classification attribute, so the format can't hold the one column we export.
PRED_EXPORT_FORMATS = ("las", "laz", "ply", "txt", "csv")


def export_predictions(pred_dir, fmt: str, progress=None) -> list[Path]:
    """Write each inferred scene (<name>_pred.npz — xyz + classification, straight
    from the training scripts) as <name>_pred.<fmt> carrying xyz + classification.
    The inferred data is transformed directly into the chosen type — there is no
    intermediate coloured PLY to render and reparse. The source .npz is removed on
    success. Returns the written paths."""
    fmt = fmt.lower().lstrip(".")
    if fmt not in PRED_EXPORT_FORMATS:
        raise ValueError(f"unsupported prediction format '{fmt}' "
                         f"(one of {', '.join(PRED_EXPORT_FORMATS)})")
    say = progress or (lambda s: None)
    written: list[Path] = []
    for src in sorted(Path(pred_dir).glob("*_pred.npz")):
        with np.load(src) as d:
            xyz = np.asarray(d["xyz"], np.float64)
            cls = np.asarray(d["classification"], np.int64)
        # ponytail: uint8 classification (LAS pf6 / uchar); >255 classes would clip.
        cls = np.clip(cls, 0, 255).astype(np.uint8)
        dst = src.with_suffix(f".{fmt}")
        _write_pred(dst, xyz, cls, fmt)
        src.unlink()                        # npz is never a target format
        written.append(dst)
        say(f"  {src.name} -> {dst.name} ({len(xyz):,} pts)")
    return written


def _write_pred(dst: Path, xyz: np.ndarray, cls: np.ndarray, fmt: str):
    """One classified cloud -> dst. xyz float, cls uint8; nothing else."""
    if fmt in ("las", "laz"):
        import laspy
        h = laspy.LasHeader(point_format=6, version="1.4")   # 8-bit classification
        h.offsets = xyz.min(0)
        h.scales = [0.001] * 3
        las = laspy.LasData(h)
        las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        las.classification = cls
        las.write(str(dst))
    elif fmt == "ply":
        header = ("ply\nformat ascii 1.0\n" + f"element vertex {len(xyz)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar classification\nend_header")
        np.savetxt(dst, np.column_stack([xyz, cls]),
                   fmt=["%.3f"] * 3 + ["%d"], header=header, comments="")
    elif fmt == "txt":                                       # x y z class
        np.savetxt(dst, np.column_stack([xyz, cls]), fmt="%.3f %.3f %.3f %d")
    elif fmt == "csv":
        np.savetxt(dst, np.column_stack([xyz, cls]), delimiter=",",
                   fmt=["%.3f"] * 3 + ["%d"], header="x,y,z,classification",
                   comments="")


# --------------------------------------------------------------------- self-check

def _selfcheck():
    """Assert the 3-way allocator hits point-count targets, mirrors the class mix,
    and that single-cloud (holey, seam-buffered) and folder (whole-scene) builds
    materialize three disjoint, non-empty splits."""
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
