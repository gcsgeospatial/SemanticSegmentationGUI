"""Pretraining preprocessing — PDAL HAG augmentation + model-ready tiling.

Two standalone functions backing the Pretraining tab:

- `add_hag`: run PDAL on each LAS/LAZ cloud to compute HeightAboveGround and
  write it back out as a new dimension (georeferencing preserved). Ground is
  auto-classified with SMRF first (hag_nn needs ground points), unless the
  caller says it's already classified. Each cloud gets a `.json` sidecar with
  the executed pipeline + HAG stats; the folder gets a `pretrain_summary.json`.

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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import dataset, prep
from .dataset import LabelSpec
from .readers import SUPPORTED_EXTS, read_points

LAS_EXTS = {".las", ".laz"}
HAG_FILTERS = ("hag_nn", "hag_delaunay")

# SMRF overwrites Classification (ground->2). When the label lives in that dim we
# ferry it here BEFORE SMRF so the converter can read the real labels back after
# HAG. Must match datasets_page._hag_conversion_spec, which reads this dim.
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
    arr = np.empty(n, dtype=dt)
    for k, v in cols.items():
        arr[k] = v
    return arr


def _hag_pipeline(reader_stage, out_path: str, skip_ground: bool,
                  hag_filter: str, has_header: bool,
                  preserve_class_as: str | None = None) -> list:
    """[reader] -> [ferry] -> [filters.smrf] -> filters.<hag> -> writers.las.

    `reader_stage` is the readers.las dict for LAS/LAZ inputs, or None when the
    points are injected as a numpy array (the first stage is then the filter).
    `preserve_class_as` copies Classification into that extra dim before SMRF can
    overwrite it (used when the user's labels live in Classification).
    writers.las options mirror kpconv-pdal/utils/las.py:write_las — write every
    extra dim (incl. HeightAboveGround), LAS 1.4; forward the header only when
    there is a LAS reader to forward it from.
    """
    stages: list = [reader_stage] if reader_stage else []
    if preserve_class_as:
        # SMRF rewrites Classification (ground->2); copy the user's labels aside
        # first so the converter can read them back after HAG.
        stages.append({"type": "filters.ferry",
                       "dimensions": f"Classification=>{preserve_class_as}"})
    if not skip_ground:
        # SMRF sets Classification=2 on ground; hag_nn/hag_delaunay use it.
        stages.append({"type": "filters.smrf"})
    stages.append({"type": f"filters.{hag_filter}"})
    writer = {"type": "writers.las", "filename": out_path,
              "minor_version": 4, "extra_dims": "all"}
    if has_header:
        writer["forward"] = "all"
    stages.append(writer)
    return stages


def add_hag(in_dir: str | Path, out_dir: str | Path, *, skip_ground: bool = False,
            hag_filter: str = "hag_nn", preserve_class_as: str | None = None,
            progress=None) -> dict:
    """Add a HeightAboveGround dim to every cloud in `in_dir` -> `out_dir`.

    LAS/LAZ are read by PDAL directly (CRS + dims preserved). Other supported
    formats (txt/csv/xyz/pts/ply/pcd/npy/npz) are read via readers.py and
    transformed to LAS/LAZ on the way through the HAG pipeline. Outputs are
    `.laz` with a HeightAboveGround dim + a per-file `.json` sidecar.

    Returns a summary dict (also written to out_dir/pretrain_summary.json).
    """
    import pdal

    if hag_filter not in HAG_FILTERS:
        raise ValueError(f"hag_filter must be one of {HAG_FILTERS}, got {hag_filter!r}")
    say = progress or (lambda s: None)
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = dataset.discover_scenes(in_dir)   # accepts a folder OR a single file
    if not files:
        raise FileNotFoundError(f"No supported point-cloud files in {in_dir}")

    say(f"{len(files)} cloud(s) -> {out_dir}"
        + ("" if skip_ground else "   (SMRF ground-classify first)"))
    per_file = []
    for path in files:
        out_path = out_dir / (path.stem + ".laz")
        is_las = path.suffix.lower() in LAS_EXTS
        say(f"  {path.name} -> {out_path.name}"
            + ("" if is_las else "   (txt/other -> LAZ)") + " …")
        if is_las:
            reader = {"type": "readers.las", "filename": str(path)}
            arrays = None
        else:
            reader = None
            arrays = [_structured_from_cloud(read_points(path))]
        stages = _hag_pipeline(reader, str(out_path), skip_ground, hag_filter,
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
        per_file.append(st)
        with open(out_dir / (path.stem + ".json"), "w", encoding="utf-8") as f:
            json.dump({"pipeline": stages, **st}, f, indent=2)
        say(f"    {n:,} pts, {n_ground:,} ground, "
            f"HAG {st['hag_min']:.2f}..{st['hag_max']:.2f} m")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "skip_ground": skip_ground,
        "hag_filter": hag_filter,
        "n_files": len(per_file),
        "total_points": sum(s["n_points"] for s in per_file),
        "files": per_file,
    }
    with open(out_dir / "pretrain_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    say(f"✓ HAG added for {len(per_file)} cloud(s) -> {out_dir}")
    return summary


def hag_for_cloud(cloud) -> "np.ndarray | None":
    """Per-point HeightAboveGround (SMRF -> hag_nn) aligned 1:1 to cloud.xyz, for
    INFERENCE (the twin of add_hag's per-file laz output — same stages, kept in RAM).

    Returns float32 of length cloud.n, or None if PDAL is unavailable or the
    pipeline drops/reorders points (the caller then falls back to a z-min proxy, so
    a missing/odd PDAL is never worse than today's behaviour). smrf+hag_nn are
    point-wise filters that preserve input order; the length + first/last-X guard
    rejects the pathological case where they don't."""
    if not pdal_available():
        return None
    import pdal
    try:
        arr_in = _structured_from_cloud(cloud)
        stages = [{"type": "filters.smrf"}, {"type": "filters.hag_nn"}]
        pipe = pdal.Pipeline(json.dumps(stages), arrays=[arr_in])
        pipe.execute()
        arr = pipe.arrays[0]
        if len(arr) != cloud.n or "HeightAboveGround" not in (arr.dtype.names or ()):
            return None
        ax = np.asarray(arr["X"], np.float64)
        if not (np.isclose(ax[0], cloud.xyz[0, 0]) and np.isclose(ax[-1], cloud.xyz[-1, 0])):
            return None   # PDAL reordered the points -> can't pair, fall back to proxy
        return np.asarray(arr["HeightAboveGround"], np.float32)
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
    preserving labels, smrf skipped when ground is given, writer always last."""
    s = _hag_pipeline({"type": "readers.las"}, "/x.laz", skip_ground=False,
                      hag_filter="hag_nn", has_header=True,
                      preserve_class_as=HAG_PRESERVED_CLASS_DIM)
    assert [st["type"] for st in s] == [
        "readers.las", "filters.ferry", "filters.smrf",
        "filters.hag_nn", "writers.las"], s
    assert s[1]["dimensions"] == f"Classification=>{HAG_PRESERVED_CLASS_DIM}"
    assert s[-1]["forward"] == "all"          # LAS reader -> forward the header
    skip = _hag_pipeline(None, "/x.laz", skip_ground=True, hag_filter="hag_nn",
                         has_header=False)
    assert [st["type"] for st in skip] == ["filters.hag_nn", "writers.las"]
    assert "forward" not in skip[-1]          # array input -> no header to forward
    print("pretrain self-check OK")


if __name__ == "__main__":
    _selfcheck()
