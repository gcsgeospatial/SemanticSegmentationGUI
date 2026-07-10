"""HeightAboveGround for one cloud, by two interchangeable methods.

`hag_for_cloud` is the single engine behind every HAG channel in the app: the
Datasets page bakes it into a dataset during conversion, and the Inference page
bakes it into the scenes a *_hag model is about to label.

Two independent axes, never mixed:
  - ground SOURCE: a caller-supplied ground_mask (the file's own ground class,
    trusted labels) wins outright; with no mask, SMRF detects ground — for every
    interpolation method. Never a union of the two.
  - HAG INTERPOLATION: "grid" (pure numpy/scipy raster — fast, no PDAL) or
    hag_nn / hag_delaunay (PDAL filters — accurate, needs python-pdal).

Without PDAL there is no SMRF; the grid method then falls back to its own
percentile+opening detection heuristic. Returns None on any failure — callers
then write no "hag" key, and the *_hag models refuse to run.

`pdal` is imported lazily, so this module (and the GUI) load fine without it.
"""

from __future__ import annotations

import json
import re

import numpy as np

HAG_FILTERS = ("hag_nn", "hag_delaunay")     # PDAL filters (the accurate path)
HAG_METHODS = ("grid",) + HAG_FILTERS        # hag_for_cloud methods; "grid" = fast default


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

    ground_mask given (trusted labels or SMRF-detected ground): the raster is
    the per-cell mean Z of the masked points; mask-less cells (e.g. under
    buildings) are filled from the NEAREST masked cell — interpolation fills
    the gaps, never a second ground source.
    No mask (the no-PDAL fallback): per-cell low-percentile Z, then a grey opening rejects
    cells whose lowest return sits on a structure (roofs); rejected + empty
    cells are nearest-filled the same way.

    HAG = z - bilinear(raster). Error is ~cell x slope: a feature channel, not
    survey ground. Returns float32 aligned 1:1 to cloud.xyz, or None when it
    can't produce one — the scene then carries no "hag" key and the *_hag models
    refuse it (there is no fallback height)."""
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


def smrf_ground_mask(cloud, cell: float = 2.0) -> "np.ndarray | None":
    """Detect ground with PDAL's SMRF filter -> boolean mask aligned 1:1 to
    cloud.xyz, or None (no PDAL, pipeline failure, or reordered output). The
    only ground-detection path when the input carries no ground class."""
    if not pdal_available():
        return None
    import pdal
    try:
        arr = _structured_from_cloud(cloud)
        smrf = pdal.Pipeline(json.dumps([{"type": "filters.smrf", "cell": cell}]),
                             arrays=[arr])
        smrf.execute()
        sarr = smrf.arrays[0]
        if len(sarr) != cloud.n:
            return None
        mask = np.asarray(sarr["Classification"]) == 2
        return mask if mask.any() else None
    except Exception:  # noqa: BLE001
        return None


def hag_for_cloud(cloud, *, ground_mask=None, hag_filter: str = "grid",
                  smrf_cell: float = 2.0) -> "np.ndarray | None":
    """Per-point HeightAboveGround aligned 1:1 to cloud.xyz — the ONE engine
    behind dataset builds, per-tile HAG and inference (so train and infer can't
    diverge in method).

    Ground SOURCE (one, never a union): ground_mask — the file's own ground
    class, trusted outright — else SMRF detection, for every method. No PDAL
    means no SMRF: the grid method then falls back to its own detection
    heuristic, the PDAL filters return None.
    hag_filter picks the INTERPOLATION: "grid" (default, hag_grid_for_cloud —
    fast raster approximation, no PDAL) or a PDAL filter ("hag_nn" /
    "hag_delaunay", the accurate path).

    Returns float32 length cloud.n, or None when no method can produce an
    aligned result — callers then write no "hag" key, and the *_hag models fail
    loudly rather than train or predict on a fabricated height. smrf+hag preserve
    input order; the length + first/last-X guard rejects the pathological case
    where they don't."""
    if hag_filter not in HAG_METHODS:
        raise ValueError(f"hag_filter must be one of {HAG_METHODS}, got {hag_filter!r}")
    if ground_mask is not None:
        ground_mask = np.asarray(ground_mask, dtype=bool).reshape(-1)
        if len(ground_mask) != cloud.n or not ground_mask.any():
            ground_mask = None                        # unusable mask -> detection
    if ground_mask is None:
        ground_mask = smrf_ground_mask(cloud, smrf_cell)   # None without PDAL
    if hag_filter == "grid":
        # mask=None here means no labels AND no SMRF -> the grid heuristic.
        return hag_grid_for_cloud(cloud, ground_mask=ground_mask)
    if ground_mask is None or not pdal_available():
        return None                                   # nothing to anchor HAG to
    import pdal
    try:
        arr = _structured_from_cloud(cloud)           # always has a Classification dim
        arr["Classification"] = np.where(ground_mask, 2, 1).astype(
            arr["Classification"].dtype)
        pipe = pdal.Pipeline(json.dumps([{"type": f"filters.{hag_filter}"}]), arrays=[arr])
        pipe.execute()
        out = pipe.arrays[0]
        if len(out) != cloud.n or "HeightAboveGround" not in (out.dtype.names or ()):
            return None
        ax = np.asarray(out["X"], np.float64)
        if not (np.isclose(ax[0], cloud.xyz[0, 0]) and np.isclose(ax[-1], cloud.xyz[-1, 0])):
            return None   # PDAL reordered the points -> can't pair them back up
        return np.asarray(out["HeightAboveGround"], np.float32)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------- self-check

def _selfcheck():
    """Grid HAG numerics (no PDAL needed): flat ground at z=0 with a 10x10 m
    "roof" at z=10 and no ground returns beneath it. Detection must reject the
    roof cells (opening) and nearest-fill them; the labeled path must get the
    same answer from the mask alone."""
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
