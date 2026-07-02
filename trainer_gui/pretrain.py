"""Pretraining preprocessing — PDAL HAG augmentation + model-ready tiling.

Two standalone functions backing the Pretraining tab:

- `add_hag`: run PDAL on each LAS/LAZ cloud to compute HeightAboveGround and
  write it back out as a new dimension (georeferencing preserved). Ground can
  come from SMRF (use_smrf), from a labeled ground class (ground_class), or both
  unioned so SMRF fills the holes the labels miss (e.g. under buildings). Each
  cloud gets a `.json` sidecar with the executed pipeline + HAG stats; the folder
  gets a `pretrain_summary.json`.

- `tile_for_model`: produce train-ready tiles for a chosen backbone by staging
  the folder to the canonical npz layout (`dataset.convert_scene`) and running
  the existing, tested tilers (`prep.prep_dataset`). HAG stays in the LAZ from
  `add_hag`; the npz tiles keep the scripts' byte-layout contract unchanged.

`pdal` is imported lazily so this module (and the GUI) load fine without it —
only `add_hag` needs it. `tile_for_model` is pure numpy/laspy.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import dataset, prep
from .dataset import LabelSpec
from .readers import SUPPORTED_EXTS, read_points

LAS_EXTS = {".las", ".laz"}
HAG_FILTERS = ("hag_nn", "hag_delaunay")     # PDAL filters (add_hag + the accurate path)
HAG_METHODS = ("grid",) + HAG_FILTERS        # hag_for_cloud methods; "grid" = fast default

# SMRF overwrites Classification (ground->2). When the label lives in that dim,
# add_hag(preserve_class_as=...) ferries it here BEFORE SMRF so a downstream
# converter can read the real labels back after HAG.
HAG_PRESERVED_CLASS_DIM = "label_src"


def pdal_available() -> bool:
    """Whether python-pdal can be imported (Stage A is disabled without it)."""
    try:
        import pdal  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- Stage A: HAG

def _structured_from_cloud(cloud) -> np.ndarray:
    """Pack a readers.py Cloud into a PDAL-dimension-named structured array, so
    non-LAS inputs (txt/csv/ply/…) can be fed straight into a PDAL pipeline and
    written out as LAS/LAZ."""
    n = cloud.n
    dt = [("X", "f8"), ("Y", "f8"), ("Z", "f8")]
    cols = {"X": cloud.xyz[:, 0], "Y": cloud.xyz[:, 1], "Z": cloud.xyz[:, 2]}
    if cloud.intensity is not None:
        dt.append(("Intensity", "u2"))
        cols["Intensity"] = np.clip(cloud.intensity, 0, 65535).astype(np.uint16)
    if cloud.return_number is not None:
        # SMRF rejects ReturnNumber/NumberOfReturns of 0; keep them consistent (>=1).
        rn = np.clip(cloud.return_number, 1, 255).astype(np.uint8)
        dt.append(("ReturnNumber", "u1"))
        cols["ReturnNumber"] = rn
        dt.append(("NumberOfReturns", "u1"))
        cols["NumberOfReturns"] = rn
    if cloud.rgb is not None:
        for i, name in enumerate(("Red", "Green", "Blue")):
            dt.append((name, "u2"))
            cols[name] = cloud.rgb[:, i].astype(np.uint16)
    used = {name.lower() for name, _ in dt}
    for raw_name, values in cloud.fields.items():
        arr0 = np.asarray(values)
        if (arr0.ndim != 1 or len(arr0) != n
                or not np.issubdtype(arr0.dtype, np.number)):
            continue
        lname = raw_name.lower()
        dim = "Classification" if lname == "classification" else re.sub(
            r"[^A-Za-z0-9_]", "_", raw_name.strip())
        if not dim:
            continue
        if dim[0].isdigit():
            dim = "field_" + dim
        if dim.lower() in used:
            continue
        if dim == "Classification":
            dt.append((dim, "u1"))
            cols[dim] = np.clip(np.rint(arr0), 0, 255).astype(np.uint8)
        else:
            dt.append((dim, "f8"))
            cols[dim] = arr0.astype(np.float64)
        used.add(dim.lower())
    # Always carry a Classification dim so SMRF / ground-assignment have somewhere
    # to write (0 = unclassified when the source had no classification field).
    if "classification" not in used:
        dt.append(("Classification", "u1"))
        cols["Classification"] = np.zeros(n, dtype=np.uint8)
    arr = np.empty(n, dtype=dt)
    for k, v in cols.items():
        arr[k] = v
    return arr


def _hag_pipeline(reader_stage, out_path: str, *, ground_class: int | None = None,
                  use_smrf: bool = True, hag_filter: str = "hag_nn",
                  has_header: bool = True, preserve_class_as: str | None = None,
                  smrf_cell: float = 2.0) -> list:
    """[reader] -> [ferry] -> [smrf] -> [assign ground] -> filters.<hag> -> writers.las.

    Ground for the HAG filter is built from, in combination:
      - use_smrf: PDAL SMRF detects ground (sets Classification=2).
      - ground_class: the cloud's OWN label value for ground is re-asserted as
        Classification=2 AFTER SMRF — so labeled ground is UNIONed with SMRF's,
        and SMRF fills the holes the labels miss (missing ground returns, e.g.
        under buildings) without discarding the trusted labels.
    SMRF overwrites Classification, so when a ground_class is trusted its original
    values are stashed first (preserve_class_as, else HAG_PRESERVED_CLASS_DIM) and
    the assign reads ground back from that stash.

    `reader_stage` is the readers.las dict for LAS/LAZ inputs, or None when the
    points are injected as a numpy array. writers.las writes every extra dim
    (incl. HeightAboveGround), LAS 1.4; the header is forwarded only when there is
    a LAS reader to forward it from.
    """
    # Stash original Classification when we need to read a labeled ground back
    # after SMRF clobbers it (and/or when the caller wants labels preserved).
    stash = preserve_class_as or (HAG_PRESERVED_CLASS_DIM if ground_class is not None else None)
    stages: list = [reader_stage] if reader_stage else []
    if stash:
        stages.append({"type": "filters.ferry",
                       "dimensions": f"Classification=>{stash}"})
    if use_smrf:
        # cell = ground-raster resolution; coarser than PDAL's 1.0 m default is
        # ~(1.0/cell)^2 cheaper in SMRF and fine for a HAG reference surface.
        stages.append({"type": "filters.smrf", "cell": smrf_cell})
    if ground_class is not None:
        # Re-assert the labeled ground as class 2 (union with SMRF when it ran).
        stages.append({"type": "filters.assign",
                       "value": f"Classification = 2 WHERE {stash} == {int(ground_class)}"})
    stages.append({"type": f"filters.{hag_filter}"})
    writer = {"type": "writers.las", "filename": out_path,
              "minor_version": 4, "extra_dims": "all"}
    if has_header:
        writer["forward"] = "all"
    stages.append(writer)
    return stages


def add_hag(in_dir: str | Path, out_dir: str | Path, *, ground_class: int | None = None,
            use_smrf: bool = True, hag_filter: str = "hag_nn", smrf_cell: float = 2.0,
            preserve_class_as: str | None = None, max_workers: int | None = None,
            progress=None) -> dict:
    """Add a HeightAboveGround dim to every cloud in `in_dir` -> `out_dir`.

    LAS/LAZ are read by PDAL directly (CRS + dims preserved). Other supported
    formats (txt/csv/xyz/pts/ply/pcd/npy/npz) are read via readers.py and
    transformed to LAS/LAZ on the way through the HAG pipeline. Outputs are
    `.laz` with a HeightAboveGround dim + a per-file `.json` sidecar.

    Files are processed concurrently (PDAL drops the GIL in execute), so this
    scales with cores when there's more than one scene; smrf_cell trades ground
    resolution for SMRF speed. Returns a summary dict (also written to
    out_dir/pretrain_summary.json).
    """
    import pdal

    if hag_filter not in HAG_FILTERS:
        raise ValueError(f"hag_filter must be one of {HAG_FILTERS}, got {hag_filter!r}")
    if ground_class is not None:
        use_smrf = False              # trusted labels -> SMRF never runs
    say = progress or (lambda s: None)
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = dataset.discover_scenes(in_dir)   # accepts a folder OR a single file
    if not files:
        raise FileNotFoundError(f"No supported point-cloud files in {in_dir}")

    say(f"{len(files)} cloud(s) -> {out_dir}"
        + ("   (SMRF ground-classify)" if use_smrf else "")
        + (f"   (+ labeled ground = class {ground_class})" if ground_class is not None else ""))

    def _process_one(path: Path) -> dict:
        out_path = out_dir / (path.stem + ".laz")
        is_las = path.suffix.lower() in LAS_EXTS
        if is_las:
            reader = {"type": "readers.las", "filename": str(path)}
            arrays = None
        else:
            reader = None
            arrays = [_structured_from_cloud(read_points(path))]
        stages = _hag_pipeline(reader, str(out_path), ground_class=ground_class,
                               use_smrf=use_smrf, hag_filter=hag_filter, smrf_cell=smrf_cell,
                               has_header=is_las, preserve_class_as=preserve_class_as)
        pipe = (pdal.Pipeline(json.dumps(stages), arrays=arrays) if arrays
                else pdal.Pipeline(json.dumps(stages)))
        n = pipe.execute()
        arr = pipe.arrays[0]
        hag = np.asarray(arr["HeightAboveGround"], np.float64)
        n_ground = int((np.asarray(arr["Classification"]) == 2).sum())
        st = {
            "scene": path.name,
            "output": out_path.name,
            "n_points": int(n),
            "n_ground": n_ground,
            "hag_min": float(hag.min()),
            "hag_mean": float(hag.mean()),
            "hag_max": float(hag.max()),
        }
        with open(out_dir / (path.stem + ".json"), "w", encoding="utf-8") as f:
            json.dump({"pipeline": stages, **st}, f, indent=2)
        return st

    # Each SMRF holds its whole cloud in RAM — clamp by RAM + cores so a big
    # multi-file run can't OOM (one policy, shared with the converter).
    workers = max_workers or dataset._worker_cap(files)
    per_file = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_one, p): p for p in files}
        for fut in as_completed(futs):
            st = fut.result()
            per_file.append(st)
            say(f"  ✓ {st['scene']} -> {st['output']}: {st['n_points']:,} pts, "
                f"{st['n_ground']:,} ground, HAG {st['hag_min']:.2f}..{st['hag_max']:.2f} m")
    per_file.sort(key=lambda s: s["scene"])   # completion order -> stable report

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "ground_class": ground_class,
        "use_smrf": use_smrf,
        "hag_filter": hag_filter,
        "n_files": len(per_file),
        "total_points": sum(s["n_points"] for s in per_file),
        "files": per_file,
    }
    with open(out_dir / "pretrain_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    say(f"✓ HAG added for {len(per_file)} cloud(s) -> {out_dir}")
    return summary


# Grid-HAG knobs. cell = ground-raster resolution (error ~ cell x slope);
# open window = largest structure whose roof-only cells get rejected; relief =
# terrain rise the opening may flatten before a cell counts as non-ground.
# ponytail: fixed heuristics sized for buildings on gentle terrain — promote to
# parameters if a dataset's terrain/structures fight them.
GRID_HAG_CELL_M = 2.0
GRID_HAG_OPEN_M = 35.0
GRID_HAG_RELIEF_M = 2.5
_GRID_HAG_MAX_DIM = 4096          # grow the cell instead of allocating a huge raster


def hag_grid_for_cloud(cloud, *, ground_mask=None,
                       cell: float = GRID_HAG_CELL_M) -> "np.ndarray | None":
    """Approximate per-point HeightAboveGround from a rasterized ground surface.
    Pure numpy/scipy, no PDAL, a few O(n) passes — the fast default method.

    ground_mask given (trusted labels): the raster is the per-cell mean Z of the
    labeled ground points; label-less cells (e.g. under buildings) are filled
    from the NEAREST labeled cell — this replaces the old SMRF union as the
    hole-filler, so SMRF never runs when a ground class exists.
    No mask (detection): per-cell low-percentile Z, then a grey opening rejects
    cells whose lowest return sits on a structure (roofs); rejected + empty
    cells are nearest-filled the same way.

    HAG = z - bilinear(raster). Error is ~cell x slope: a feature channel, not
    survey ground — the trainers' accepted fallback is z-scene-min, far cruder.
    Returns float32 aligned 1:1 to cloud.xyz, or None when it can't produce one
    (caller falls back to that proxy)."""
    try:
        from scipy import ndimage
    except Exception:  # noqa: BLE001
        return None
    n = cloud.n
    if n == 0:
        return None
    xy = cloud.xyz[:, :2].astype(np.float64)
    z = cloud.xyz[:, 2].astype(np.float64)
    mn = xy.min(0)
    ext = xy.max(0) - mn
    cell = max(float(cell), float(ext.max()) / _GRID_HAG_MAX_DIM, 1e-6)
    ij = np.floor((xy - mn) / cell).astype(np.int64)
    dims = (int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1)
    flat = ij[:, 0] * dims[1] + ij[:, 1]
    ncell = dims[0] * dims[1]

    gm = None
    if ground_mask is not None:
        gm = np.asarray(ground_mask, dtype=bool).reshape(-1)
        if len(gm) != n or not gm.any():
            gm = None                              # unusable mask -> detection

    grid = np.zeros(ncell, np.float64)
    if gm is not None:                             # labeled: mean ground Z per cell
        cnt = np.bincount(flat[gm], minlength=ncell)
        zsum = np.bincount(flat[gm], weights=z[gm], minlength=ncell)
        valid = cnt > 0
        grid[valid] = zsum[valid] / cnt[valid]
        g2, v2 = grid.reshape(dims), valid.reshape(dims)
    else:                                          # detect: low-percentile Z + opening
        # One int64 argsort instead of a two-key lexsort (~2.5x faster): pack
        # cell id with 20-bit-quantized Z (<= mm resolution), so sorting the key
        # orders by cell, Z ascending within.
        zq = ((z - z.min()) * ((1 << 20) - 1) / max(float(z.max() - z.min()), 1e-9))
        order = np.argsort((flat << 20) | zq.astype(np.int64))
        counts = np.bincount(flat, minlength=ncell)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        valid = counts > 0
        pick = starts[valid] + (0.05 * (counts[valid] - 1)).astype(np.int64)
        grid[valid] = z[order][pick]               # 5th-pctile resists low noise
        g2, v2 = grid.reshape(dims), valid.reshape(dims)
        # Nearest-fill empties FIRST so the opening sees a full surface, then
        # reject cells the opening lowered by more than the relief tolerance:
        # their lowest return sits on a structure, not the ground.
        if (~v2).any():
            near = ndimage.distance_transform_edt(~v2, return_distances=False,
                                                  return_indices=True)
            g2 = g2[tuple(near)]
        size = max(int(round(GRID_HAG_OPEN_M / cell)), 3)
        opened = ndimage.grey_opening(g2, size=size, mode="nearest")
        v2 = v2 & ((g2 - opened) <= GRID_HAG_RELIEF_M)
        if not v2.any():
            return None                            # nothing survives -> no ground

    if (~v2).any():                                # nearest-fill holes (final surface)
        near = ndimage.distance_transform_edt(~v2, return_distances=False,
                                              return_indices=True)
        g2 = np.where(v2, g2, 0.0)[tuple(near)]
    coords = np.ascontiguousarray(((xy - mn) / cell - 0.5).T)
    ground_at = ndimage.map_coordinates(g2, coords, order=1, mode="nearest")
    return (z - ground_at).astype(np.float32)


def hag_for_cloud(cloud, *, ground_mask=None, use_smrf: bool = True,
                  hag_filter: str = "grid", smrf_cell: float = 2.0) -> "np.ndarray | None":
    """Per-point HeightAboveGround aligned 1:1 to cloud.xyz — the ONE engine
    behind dataset builds, per-tile HAG and inference (so train and infer can't
    diverge in method).

    hag_filter picks the method: "grid" (default, hag_grid_for_cloud — fast
    raster approximation, no PDAL) or a PDAL filter ("hag_nn"/"hag_delaunay",
    the accurate path). Ground comes from ONE source, never a union:
      - ground_mask: points the dataset LABELS as ground. Trusted labels win —
        SMRF NEVER runs when a usable mask exists (holes the labels miss are
        spanned by the grid's nearest-fill / the PDAL filter's interpolation).
      - use_smrf (PDAL filters only, no mask): SMRF detects ground. The grid
        method does its own detection and ignores use_smrf.

    Returns float32 length cloud.n, or None when no method can produce an
    aligned result (caller falls back to a z-min proxy — never worse than
    before). smrf+hag preserve input order; the length + first/last-X guard
    rejects the pathological case where they don't."""
    if hag_filter not in HAG_METHODS:
        raise ValueError(f"hag_filter must be one of {HAG_METHODS}, got {hag_filter!r}")
    if ground_mask is not None:
        ground_mask = np.asarray(ground_mask, dtype=bool).reshape(-1)
        if len(ground_mask) != cloud.n:
            return None
        if not ground_mask.any():
            ground_mask = None                        # unusable mask -> detection
        else:
            use_smrf = False                          # trusted labels -> no SMRF, ever
    if hag_filter == "grid":
        return hag_grid_for_cloud(cloud, ground_mask=ground_mask)
    if not pdal_available():
        return None
    if ground_mask is None and not use_smrf:
        return None                                   # nothing to anchor HAG to
    import pdal
    try:
        arr = _structured_from_cloud(cloud)           # always has a Classification dim
        ground = np.zeros(cloud.n, dtype=bool)
        if ground_mask is not None:
            ground |= ground_mask
        if use_smrf:                                  # detect ground (no labels case)
            smrf = pdal.Pipeline(json.dumps([{"type": "filters.smrf", "cell": smrf_cell}]), arrays=[arr])
            smrf.execute()
            sarr = smrf.arrays[0]
            if len(sarr) == cloud.n:
                ground |= (np.asarray(sarr["Classification"]) == 2)
        if not ground.any():
            return None                               # no ground points -> no HAG
        arr["Classification"] = np.where(ground, 2, 1).astype(arr["Classification"].dtype)
        pipe = pdal.Pipeline(json.dumps([{"type": f"filters.{hag_filter}"}]), arrays=[arr])
        pipe.execute()
        out = pipe.arrays[0]
        if len(out) != cloud.n or "HeightAboveGround" not in (out.dtype.names or ()):
            return None
        ax = np.asarray(out["X"], np.float64)
        if not (np.isclose(ax[0], cloud.xyz[0, 0]) and np.isclose(ax[-1], cloud.xyz[-1, 0])):
            return None   # PDAL reordered the points -> can't pair, fall back to proxy
        return np.asarray(out["HeightAboveGround"], np.float32)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- Stage B: tile

def tile_for_model(in_dir: str | Path, out_dir: str | Path, backbone_key: str,
                   params: dict, progress=None) -> Path:
    """Tile a folder of clouds into train-ready npz tiles for `backbone_key`.

    Stages the folder to the canonical npz layout under <out_dir>/_staged/train,
    reading per-point labels from the LAS `classification` dim (value 0 /
    unclassified -> ignore, the same default as the Datasets page), then runs
    the existing per-backbone tiler. Returns the prep dir (<...>/prep/<tag>).
    """
    say = progress or (lambda s: None)
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if not prep.supports_local_prep(backbone_key):
        raise ValueError(f"Tiling not supported for backbone '{backbone_key}'")

    files = dataset.discover_scenes(in_dir)
    if not files:
        raise FileNotFoundError(f"No supported point-cloud files in {in_dir}")

    spec = LabelSpec(kind="field", field="classification")
    say(f"scanning labels across {min(len(files), 8)} of {len(files)} scene(s) …")
    counts = dataset.scan_label_values(files, spec)
    # Identity map over observed classes; 0/unclassified is ignored (-1).
    value_to_index = {v: i for i, v in enumerate(sorted(c for c in counts if c != 0))}
    if not value_to_index:
        raise ValueError("No trainable classes found in the `classification` dim "
                         "(only value 0 / unclassified present).")
    say(f"classes (source->index): {value_to_index}   ·   ignored: [0]")

    staged = out_dir / "_staged"
    say(f"staging {len(files)} scene(s) -> {staged / 'train'} …")
    for path in files:
        out_path = staged / "train" / (path.stem + ".npz")
        say(f"  {path.name} …")
        dataset.convert_scene(path, spec, value_to_index, out_path)

    tag = prep.prep_tag(backbone_key, params)
    say(f"tiling -> prep/{tag} …")
    prep_dir = prep.prep_dataset(backbone_key, staged, params, progress=say)
    say(f"✓ tiles ready -> {prep_dir}")
    return prep_dir


# --------------------------------------------------------------------- self-check

def _selfcheck():
    """_hag_pipeline stage ordering (no PDAL needed): ferry-before-smrf when
    preserving labels, ground re-asserted AFTER smrf (union), writer always last."""
    s = _hag_pipeline({"type": "readers.las"}, "/x.laz", use_smrf=True,
                      hag_filter="hag_nn", has_header=True,
                      preserve_class_as=HAG_PRESERVED_CLASS_DIM)
    assert [st["type"] for st in s] == [
        "readers.las", "filters.ferry", "filters.smrf",
        "filters.hag_nn", "writers.las"], s
    assert s[1]["dimensions"] == f"Classification=>{HAG_PRESERVED_CLASS_DIM}"
    assert s[-1]["forward"] == "all"          # LAS reader -> forward the header
    # labeled ground + SMRF fill: stash labels, smrf, re-assert ground=class, hag
    u = _hag_pipeline(None, "/x.laz", ground_class=6, use_smrf=True,
                      hag_filter="hag_nn", has_header=False)
    assert [st["type"] for st in u] == [
        "filters.ferry", "filters.smrf", "filters.assign",
        "filters.hag_nn", "writers.las"], u
    assert u[2]["value"] == f"Classification = 2 WHERE {HAG_PRESERVED_CLASS_DIM} == 6"
    # labeled ground only (no smrf), array input -> no header to forward
    only = _hag_pipeline(None, "/x.laz", ground_class=2, use_smrf=False,
                         hag_filter="hag_nn", has_header=False)
    assert [st["type"] for st in only] == [
        "filters.ferry", "filters.assign", "filters.hag_nn", "writers.las"], only
    assert "forward" not in only[-1]          # array input -> no header to forward

    # grid HAG numerics (no PDAL needed): flat ground at z=0 with a 10x10 m
    # "roof" at z=10 and no ground returns beneath it. Detection must reject the
    # roof cells (opening) and nearest-fill them; the labeled path must get the
    # same answer from the mask alone.
    from .readers import Cloud
    g = np.mgrid[0:60, 0:60].reshape(2, -1).T.astype(np.float64)
    on_roof = (g[:, 0] >= 25) & (g[:, 0] < 35) & (g[:, 1] >= 25) & (g[:, 1] < 35)
    xyz = np.vstack([np.column_stack([g[~on_roof], np.zeros((~on_roof).sum())]),
                     np.column_stack([g[on_roof], np.full(on_roof.sum(), 10.0)])])
    ng = int((~on_roof).sum())
    for h in (hag_grid_for_cloud(Cloud(xyz=xyz)),                       # detection
              hag_grid_for_cloud(Cloud(xyz=xyz),                        # labeled
                                 ground_mask=np.arange(len(xyz)) < ng)):
        assert h is not None and len(h) == len(xyz)
        assert abs(float(h[:ng].mean())) < 0.3, "ground HAG ~0"
        assert float(h[ng:].min()) > 8.0, "roof HAG ~10 (cells rejected + filled)"
    assert hag_for_cloud(Cloud(xyz=xyz), hag_filter="grid") is not None
    print("pretrain self-check OK")


if __name__ == "__main__":
    _selfcheck()
