"""HeightAboveGround — hag_for_cloud is the one engine behind every feat_hag.

Two independent axes, never mixed: ground SOURCE (caller mask wins, else CSF;
grid's own heuristic only without PDAL) and INTERPOLATION ("grid" numpy raster,
or PDAL hag_nn/hag_delaunay). Returns None on any failure — the scene then has
no hag key and *_hag models refuse to run. pdal imports lazily.
"""

from __future__ import annotations

import json
import re

import numpy as np

HAG_FILTERS = ("hag_nn", "hag_delaunay")     # PDAL filters (accurate path)
HAG_METHODS = ("grid",) + HAG_FILTERS

# jakteristics feature names, exact spellings
GEO_FEATURES = ("eigenvalue_sum", "omnivariance", "eigenentropy", "anisotropy",
                "planarity", "linearity", "PCA1", "PCA2",
                "surface_variation", "sphericity", "verticality")


def pdal_available() -> bool:
    try:
        import pdal  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def jakteristics_available() -> bool:
    try:
        import jakteristics  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- Stage A: HAG

def _structured_from_cloud(cloud) -> np.ndarray:
    """Pack a Cloud into a PDAL-dimension-named structured array."""
    n = cloud.n
    dt = [("X", "f8"), ("Y", "f8"), ("Z", "f8")]
    cols = {"X": cloud.xyz[:, 0], "Y": cloud.xyz[:, 1], "Z": cloud.xyz[:, 2]}
    if cloud.intensity is not None:
        dt.append(("Intensity", "u2"))
        cols["Intensity"] = np.clip(cloud.intensity, 0, 65535).astype(np.uint16)
    if cloud.return_number is not None:
        # ground filters reject ReturnNumber/NumberOfReturns of 0
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
    # always carry a Classification dim so CSF/ground-assignment can write
    if "classification" not in used:
        dt.append(("Classification", "u1"))
        cols["Classification"] = np.zeros(n, dtype=np.uint8)
    arr = np.empty(n, dtype=dt)
    for k, v in cols.items():
        arr[k] = v
    return arr


# ponytail: fixed grid-HAG heuristics sized for buildings on gentle terrain —
# promote to parameters if a dataset's terrain/structures fight them
GRID_HAG_CELL_M = 2.0
GRID_HAG_OPEN_M = 35.0
GRID_HAG_RELIEF_M = 2.5
_GRID_HAG_MAX_DIM = 4096          # grow the cell instead of a huge raster

# Past this nearest-fill distance the ground raster is extrapolating, not
# interpolating (a building wider than its ground context, or detection
# failure) — worth a per-scene warning, not silence.
GRID_HAG_GAP_WARN_M = 50.0

# CSF decimation cell: only a cell's lowest point can be ground, so CSF runs on
# the min-Z-per-cell low envelope — 10-50x fewer points, and a cleaner envelope
# in low vegetation (the cell minimum is the most-likely-ground return).
# Coverage is unchanged (every cell keeps a point), so the cloth always spans a
# building with its surrounding ground, however large the footprint. Must stay
# well under the CSF cloth resolution (~1-2 m) so the cloth has support.
CSF_DECIM_CELL_M = 0.5


def hag_grid_for_cloud(cloud, *, ground_mask=None,
                       cell: float = GRID_HAG_CELL_M,
                       notes: "list | None" = None) -> "np.ndarray | None":
    """HAG from a rasterized ground surface (numpy/scipy, no PDAL). With a mask:
    per-cell mean Z, holes nearest-filled (never a second ground source). Without:
    low-percentile Z + grey opening rejects roof cells. Error ~ cell x slope —
    a feature channel, not survey ground. float32 (n,) or None.
    notes (a list, when given) collects warning strings — e.g. a nearest-fill
    gap past GRID_HAG_GAP_WARN_M, where HAG is extrapolated."""
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
        # packed cell-id|quantized-Z argsort: ~2.5x faster than lexsort
        zq = ((z - z.min()) * ((1 << 20) - 1) / max(float(z.max() - z.min()), 1e-9))
        order = np.argsort((flat << 20) | zq.astype(np.int64))
        counts = np.bincount(flat, minlength=ncell)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        valid = counts > 0
        pick = starts[valid] + (0.05 * (counts[valid] - 1)).astype(np.int64)
        grid[valid] = z[order][pick]               # 5th-pctile resists low noise
        g2, v2 = grid.reshape(dims), valid.reshape(dims)
        # nearest-fill first so the opening sees a full surface, then reject
        # cells the opening lowered past the relief tolerance (roofs)
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
        dist, near = ndimage.distance_transform_edt(~v2, return_indices=True)
        g2 = np.where(v2, g2, 0.0)[tuple(near)]
        gap = float(dist.max()) * cell
        if notes is not None and gap > GRID_HAG_GAP_WARN_M:
            notes.append(f"HAG ground nearest-filled across a ~{gap:.0f} m gap "
                         f"(building wider than its ground context, or no ground "
                         f"detected there) — feat_hag in that area is extrapolated")
    coords = np.ascontiguousarray(((xy - mn) / cell - 0.5).T)
    ground_at = ndimage.map_coordinates(g2, coords, order=1, mode="nearest")
    return (z - ground_at).astype(np.float32)


def _min_z_per_cell(xyz, cell: float = CSF_DECIM_CELL_M) -> np.ndarray:
    """Indices of each cell's lowest point (ascending) — the low envelope."""
    xy = xyz[:, :2]
    z = xyz[:, 2]
    ij = np.floor((xy - xy.min(0)) / cell).astype(np.int64)
    flat = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
    # packed cell-id|quantized-Z argsort (same trick as hag_grid_for_cloud):
    # first occurrence per cell in the sorted order = that cell's minimum
    zq = ((z - z.min()) * ((1 << 20) - 1) / max(float(z.max() - z.min()), 1e-9))
    order = np.argsort((flat << 20) | zq.astype(np.int64))
    fo = flat[order]
    keep = order[np.concatenate(([True], fo[1:] != fo[:-1]))]
    return np.sort(keep)


def csf_ground_mask(cloud) -> "np.ndarray | None":
    """Ground mask via PDAL CSF, or None. CSF over SMRF: no window-size contract,
    so large flat roofs can't be absorbed into ground (the SMRF failure mode).

    CSF sees only the min-Z-per-cell decimation (CSF_DECIM_CELL_M) — points that
    aren't their cell's low point can't be ground, so this drops nothing valid,
    runs the cloth on 10-50x fewer points, and denoises the low-vegetation
    envelope. The returned mask is full-length; non-envelope points are simply
    False (not a ground source), which the interpolation handles natively."""
    if not pdal_available():
        return None
    import pdal
    from .readers import Cloud
    try:
        keep = _min_z_per_cell(cloud.xyz)
        # fresh sub-Cloud: geometry (+ returns for CSF's last-return logic) only,
        # so a stale source Classification can never leak into the mask
        sub = Cloud(xyz=cloud.xyz[keep],
                    return_number=(cloud.return_number[keep]
                                   if cloud.return_number is not None else None))
        arr = _structured_from_cloud(sub)
        csf = pdal.Pipeline(json.dumps([{"type": "filters.csf"}]), arrays=[arr])
        csf.execute()
        sarr = csf.arrays[0]
        ax = np.asarray(sarr["X"], np.float64)
        if (len(sarr) != len(keep)
                or not (np.isclose(ax[0], sub.xyz[0, 0])
                        and np.isclose(ax[-1], sub.xyz[-1, 0]))):
            return None   # PDAL dropped/reordered points -> can't map the mask back
        gm = np.asarray(sarr["Classification"]) == 2
        if not gm.any():
            return None
        mask = np.zeros(cloud.n, dtype=bool)
        mask[keep[gm]] = True
        return mask
    except Exception:  # noqa: BLE001
        return None


def hag_for_cloud(cloud, *, ground_mask=None,
                  hag_filter: str = "grid",
                  notes: "list | None" = None) -> "np.ndarray | None":
    """The ONE HAG engine (dataset builds, tiles, inference — methods can't
    diverge). Ground source: ground_mask wins, else CSF, never a union.
    hag_filter picks interpolation. float32 (n,) or None — never fabricated.
    notes collects per-scene warnings (grid path's large fill gaps)."""
    if hag_filter not in HAG_METHODS:
        raise ValueError(f"hag_filter must be one of {HAG_METHODS}, got {hag_filter!r}")
    if ground_mask is not None:
        ground_mask = np.asarray(ground_mask, dtype=bool).reshape(-1)
        if len(ground_mask) != cloud.n or not ground_mask.any():
            ground_mask = None                        # unusable mask -> detection
    if ground_mask is None:
        ground_mask = csf_ground_mask(cloud)          # None without PDAL
    if hag_filter == "grid":
        return hag_grid_for_cloud(cloud, ground_mask=ground_mask, notes=notes)
    if ground_mask is None or not pdal_available():
        return None                                   # nothing to anchor HAG to
    import pdal
    try:
        arr = _structured_from_cloud(cloud)
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


# ------------------------------------------------ Stage A': geometric features

def geo_features_for_cloud(xyz, names, radius: float = 1.0) -> "dict[str, np.ndarray]":
    """{jak_name: float32 (n,)}; NaN -> 0. Raises (never soft-None): geo channels
    are always explicitly requested, so failures must be loud."""
    bad = [n for n in names if n not in GEO_FEATURES]
    if bad:
        raise ValueError(f"unknown geometric feature(s) {bad}; "
                         f"valid: {list(GEO_FEATURES)}")
    try:
        from jakteristics import compute_features
    except ImportError as e:
        raise RuntimeError("Geometric feature channels need the 'jakteristics' "
                           "package (pip install jakteristics).") from e
    pts = np.ascontiguousarray(np.asarray(xyz, dtype=np.float64))
    # num_threads = all cores; fine while conversion forces max_workers=1
    feats = compute_features(pts, search_radius=float(radius),
                             feature_names=list(names))
    return {nm: np.nan_to_num(feats[:, i], nan=0.0, posinf=0.0,
                              neginf=0.0).astype(np.float32)
            for i, nm in enumerate(names)}


# --------------------------------------------------------------------- self-check

def _selfcheck():
    """Grid HAG: detection and labeled paths agree on a flat scene with a roof."""
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
