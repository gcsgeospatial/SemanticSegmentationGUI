"""Read every supported point-cloud format into one Cloud shape; `fields` holds
every per-point 1-D numeric array so any can be offered as the label source."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_EXTS = {".las", ".laz", ".ply", ".txt", ".csv", ".xyz", ".pts", ".pcd", ".npy", ".npz"}
ASCII_EXTS = {".txt", ".csv", ".xyz", ".pts"}


@dataclass
class Cloud:
    xyz: np.ndarray                      # (N, 3) float64
    rgb: np.ndarray | None = None        # (N, 3) uint8
    intensity: np.ndarray | None = None  # (N,) float32, raw (not normalized)
    return_number: np.ndarray | None = None
    fields: dict = field(default_factory=dict)  # name -> (N,) array, label candidates
    crs_wkt: str | None = None           # CRS of the STORED coords (the processing CRS)
    source_crs_wkt: str | None = None    # original CRS, set ONLY when a transform occurred

    @property
    def n(self) -> int:
        return len(self.xyz)


# --- CRS reprojection: every xyz downstream is projected & meter-denominated ---
# pyproj is a GUI-env dep; keep it lazy. normalize_to_meters (ingest) and
# restore_to_source (export) are inverses built from the same Transformer pair.

def _horizontal_crs(crs):
    """Horizontal sub-CRS of a compound CRS, else the CRS itself."""
    return crs.sub_crs_list[0] if crs.is_compound else crs


def vertical_unit_factor(crs) -> float:
    """Meters per source vertical unit: the vertical-axis unit of a compound/3D
    CRS, else a PROJECTED horizontal's linear unit. z beside geographic (angular)
    coords is metres, never radians — only a projected horizontal lends its unit."""
    for ax in crs.axis_info:
        if ax.direction.lower() in ("up", "down"):
            return float(ax.unit_conversion_factor)
    hor = _horizontal_crs(crs)
    if hor.is_projected and hor.axis_info:
        return float(hor.axis_info[0].unit_conversion_factor)
    return 1.0


def _is_meter_horizontal(crs) -> bool:
    hor = _horizontal_crs(crs)
    return hor.is_projected and abs(hor.axis_info[0].unit_conversion_factor - 1.0) <= 1e-9


def _wkt_looks_meter_projected(wkt: str) -> bool:
    """Cheap no-pyproj sniff: a projected WKT whose length unit is the metre."""
    up = wkt.upper()
    return ("PROJCRS" in up or "PROJCS" in up) and ('"METRE"' in up or '"METER"' in up)


def _estimate_utm(crs, xyz, CRS, Transformer):
    """UTM zone for the cloud centroid (PROJ db is bundled, no network)."""
    from pyproj.aoi import AreaOfInterest
    from pyproj.database import query_utm_crs_info
    hor = _horizontal_crs(crs)
    x, y = float(np.median(xyz[:, 0])), float(np.median(xyz[:, 1]))
    if hor.is_geographic:
        lon, lat = x, y
    else:
        lon, lat = Transformer.from_crs(hor, CRS.from_epsg(4326), always_xy=True).transform(x, y)
    info = query_utm_crs_info("WGS 84", area_of_interest=AreaOfInterest(lon, lat, lon, lat))
    if not info:
        raise ValueError(f"could not estimate a UTM zone for centroid lon/lat {lon:.5f},{lat:.5f}")
    return CRS.from_authority(info[0].auth_name, info[0].code)


def normalize_to_meters(xyz, source_crs_wkt):
    """Reproject a cloud into a projected, meter-denominated CRS.
    Returns (xyz_meters, proc_wkt, source_wkt); source_wkt is set ONLY when a
    transform occurred (its presence is the exact round-trip signal). Identity
    fast-path (already projected+meter, or no/unparseable CRS) keeps coords
    byte-identical. xy via pyproj Transformer(always_xy=True); z by vertical unit."""
    xyz = np.asarray(xyz, np.float64)
    if not source_crs_wkt:
        return xyz, None, None
    try:
        from pyproj import CRS, Transformer
    except ImportError:
        if _wkt_looks_meter_projected(source_crs_wkt):
            return xyz, source_crs_wkt, None
        raise ValueError("pyproj is required to reproject this source CRS to meters "
                         "— install pyproj in the GUI environment")
    try:
        crs = CRS.from_wkt(source_crs_wkt)
    except Exception as e:
        raise ValueError("source CRS WKT is present but unparseable, so the cloud "
                         "cannot be reprojected to meters — declare the EPSG code on "
                         "the prep/inference page to override it") from e
    if _is_meter_horizontal(crs):
        return xyz, source_crs_wkt, None       # identity fast-path: bytes unchanged
    utm = _estimate_utm(crs, xyz, CRS, Transformer)
    x, y = Transformer.from_crs(_horizontal_crs(crs), utm, always_xy=True).transform(
        xyz[:, 0], xyz[:, 1])
    z = xyz[:, 2] * vertical_unit_factor(crs)
    return np.column_stack([x, y, z]).astype(np.float64), utm.to_wkt(), source_crs_wkt


def is_legacy_unit_scale_pred(crs_wkt, source_crs_wkt) -> bool:
    """D2 detector: a stored non-meter processing CRS with no source_crs_wkt is a
    pred npz written under the old unit-scale contract; exporters hard-block on it."""
    if source_crs_wkt is not None or not crs_wkt:
        return False
    try:
        from pyproj import CRS
        return not _is_meter_horizontal(CRS.from_wkt(crs_wkt))
    except Exception:
        return False


def restore_to_source(xyz, proc_crs_wkt, source_crs_wkt):
    """Inverse of normalize_to_meters: proc-meter coords -> source frame, exact
    round-trip via the same Transformer pair (direction=INVERSE). Every exporter
    calls this. source_crs_wkt None = no transform at ingest (proc IS the source
    frame) — but a non-meter proc with no source is a legacy pred: hard block (D2)."""
    xyz = np.asarray(xyz, np.float64)
    if source_crs_wkt is None:
        if is_legacy_unit_scale_pred(proc_crs_wkt, source_crs_wkt):
            raise ValueError("prediction npz predates the CRS reprojection contract "
                             "(non-meter CRS with no source_crs_wkt) — re-run the "
                             "inference job to re-stage it under the new CRS contract")
        return xyz
    from pyproj import CRS, Transformer
    source = CRS.from_wkt(source_crs_wkt)
    tx = Transformer.from_crs(_horizontal_crs(source),
                              _horizontal_crs(CRS.from_wkt(proc_crs_wkt)), always_xy=True)
    x, y = tx.transform(xyz[:, 0], xyz[:, 1], direction="INVERSE")
    z = xyz[:, 2] / vertical_unit_factor(source)
    return np.column_stack([x, y, z]).astype(np.float64)


def read_points(path: str | Path, declared_crs_epsg: int | None = None) -> Cloud:
    ext = Path(path).suffix.lower()
    if ext in (".las", ".laz"):
        cloud = _read_las(path)
    elif ext == ".ply":
        cloud = _read_ply(path)
    elif ext in ASCII_EXTS:
        cloud = _read_ascii(path)
    elif ext == ".pcd":
        cloud = _read_pcd(path)
    elif ext in (".npy", ".npz"):
        cloud = _read_numpy(path)
    else:
        raise ValueError(f"Unsupported point-cloud format: {ext} ({path})")
    # D1 remedy: a declared EPSG fills in for a cloud carrying no CRS, then rides
    # the same reprojection-to-meters as an embedded one (formats other than
    # las/laz never carry a CRS, so this is their only route to georeferencing)
    if declared_crs_epsg is not None and cloud.crs_wkt is None:
        from pyproj import CRS
        wkt = CRS.from_epsg(int(declared_crs_epsg)).to_wkt()
        cloud.xyz, cloud.crs_wkt, cloud.source_crs_wkt = normalize_to_meters(cloud.xyz, wkt)
    return cloud


def list_label_fields(path: str | Path) -> list[str]:
    """Cheap-ish probe: names usable as the ground-truth label source."""
    return sorted(read_points(path).fields.keys())


# ---------------------------------------------------------------- las / laz

def _read_las(path) -> Cloud:
    import laspy

    las = laspy.read(str(path))
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    try:        # GeoTIFF-key or WKT VLR -> pyproj CRS; malformed/absent -> None
        crs = las.header.parse_crs()
    except Exception:
        crs = None

    dims = {d.name.lower() for d in las.point_format.dimensions}
    rgb = None
    if {"red", "green", "blue"}.issubset(dims):
        rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.float64)
        if rgb.max() > 255:  # 16-bit color
            rgb = rgb / 257.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    intensity = np.asarray(las.intensity, np.float32) if "intensity" in dims else None
    ret = np.asarray(las.return_number, np.float32) if "return_number" in dims else None

    fields = {}
    # red/green/blue stay in fields: color is explicit-only, columns must be mappable
    skip = {"x", "y", "z"}
    for d in las.point_format.dimensions:
        name = d.name.lower()
        if name in skip:
            continue
        try:
            arr = np.asarray(las[d.name])
        except Exception:
            continue
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            fields[d.name] = arr
    wkt = crs.to_wkt() if crs is not None else None
    xyz, proc_wkt, source_wkt = normalize_to_meters(xyz, wkt)
    if source_wkt is not None:      # reprojected: heights are vertical, scale by vertical unit
        vf = vertical_unit_factor(crs)
        for k in list(fields):
            if k.lower().replace("_", "") in ("heightaboveground", "hag"):
                fields[k] = np.asarray(fields[k], np.float64) * vf
    return Cloud(xyz=xyz, rgb=rgb, intensity=intensity, return_number=ret, fields=fields,
                 crs_wkt=proc_wkt, source_crs_wkt=source_wkt)


# ---------------------------------------------------------------- ply

def _read_ply(path) -> Cloud:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = set(v.dtype.names)
    xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)

    rgb = None
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([v["red"], v["green"], v["blue"]], -1).astype(np.float64)
        if rgb.max() > 255:
            rgb = rgb / 257.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    intensity = None
    for key in ("intensity", "scalar_intensity", "scalar_Intensity"):
        if key in names:
            intensity = np.asarray(v[key], np.float32)
            break

    ret = None
    for key in ("return_number", "scalar_return_number", "scalar_ReturnNumber"):
        if key in names:
            ret = np.asarray(v[key], np.float32)
            break

    fields = {}
    skip = {"x", "y", "z", "nx", "ny", "nz", "alpha"}
    for name in v.dtype.names:
        if name in skip:
            continue
        arr = np.asarray(v[name])
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            fields[name] = arr
    return Cloud(xyz=xyz, rgb=rgb, intensity=intensity, return_number=ret, fields=fields)


# ---------------------------------------------------------------- ascii

def _sniff_delimiter(path) -> str | None:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
    return "," if "," in line else None  # None -> whitespace


def _read_ascii(path) -> Cloud:
    """Columns: 0-2 = x,y,z; col 3 defaults to intensity, col 4 to return number
    (a common ASCII LiDAR column order); every extra column is exposed as a label
    candidate ("column 3", "column 4", ...)."""
    arr = np.loadtxt(str(path), delimiter=_sniff_delimiter(path), dtype=np.float64, ndmin=2)
    if arr.shape[1] < 3:
        raise ValueError(f"{path}: ASCII cloud needs at least 3 columns (x y z), "
                         f"found {arr.shape[1]}")
    xyz = arr[:, :3]
    intensity = arr[:, 3].astype(np.float32) if arr.shape[1] >= 4 else None
    ret = arr[:, 4].astype(np.float32) if arr.shape[1] >= 5 else None
    fields = {f"column {c}": arr[:, c] for c in range(3, arr.shape[1])}
    return Cloud(xyz=xyz, intensity=intensity, return_number=ret, fields=fields)


# ---------------------------------------------------------------- pcd

def _read_pcd(path) -> Cloud:
    import open3d as o3d

    pc = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pc.points, np.float64)
    rgb = None
    if pc.has_colors():
        rgb = np.clip(np.asarray(pc.colors) * 255.0, 0, 255).astype(np.uint8)
    intensity = None
    try:  # the legacy reader drops non-xyz/rgb attrs; the tensor reader keeps them
        t = o3d.t.io.read_point_cloud(str(path))
        if "intensity" in t.point:
            intensity = t.point["intensity"].numpy().reshape(-1).astype(np.float32)
    except Exception:
        pass
    return Cloud(xyz=xyz, rgb=rgb, intensity=intensity)


# ---------------------------------------------------------------- npy / npz

def _read_numpy(path) -> Cloud:
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        arr = np.load(str(path))
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"{path}: expected an (N, >=3) array, got {arr.shape}")
        fields = {f"column {c}": arr[:, c] for c in range(3, arr.shape[1])}
        intensity = arr[:, 3].astype(np.float32) if arr.shape[1] >= 4 else None
        return Cloud(xyz=arr[:, :3].astype(np.float64), intensity=intensity, fields=fields)

    z = np.load(str(path))
    xyz = None
    for key in ("xyz", "points", "coord", "coords"):
        if key in z:
            xyz = np.asarray(z[key], np.float64)
            break
    if xyz is None or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"{path}: no (N,3) 'xyz'/'points'/'coord' array found")
    rgb = None
    for key in ("rgb", "color", "colors"):
        if key in z:
            c = np.asarray(z[key], np.float64)
            rgb = np.clip(c * 255.0 if c.max() <= 1.0 else c, 0, 255).astype(np.uint8)
            break
    intensity = np.asarray(z["intensity"], np.float32) if "intensity" in z else None
    ret = None
    for key in ("return_number", "ret_num"):
        if key in z:
            ret = np.asarray(z[key], np.float32)
            break
    fields = {}
    used = {"xyz", "points", "coord", "coords", "rgb", "color", "colors", "intensity",
            "return_number", "ret_num", "crs_wkt", "source_crs_wkt"}
    for key in z.files:
        if key in used:
            continue
        arr = np.asarray(z[key])
        if arr.ndim == 1 and len(arr) == len(xyz) and np.issubdtype(arr.dtype, np.number):
            fields[key] = arr
    # a staged/pred npz already carries its processing (+ source) CRS — surface it
    # so read_points reports the real CRS and declared-EPSG fill never overrides it
    crs_wkt = str(z["crs_wkt"]) if "crs_wkt" in z.files else None
    source_crs_wkt = str(z["source_crs_wkt"]) if "source_crs_wkt" in z.files else None
    return Cloud(xyz=xyz, rgb=rgb, intensity=intensity, return_number=ret, fields=fields,
                 crs_wkt=crs_wkt, source_crs_wkt=source_crs_wkt)


# ---------------------------------------------------------------- self-check

def _selfcheck() -> None:
    """python readers.py — reproject round-trip, identity, and z-unit invariants."""
    from pyproj import CRS
    rng = np.random.default_rng(0)

    # EPSG:2263 (US-ft) reprojects to UTM meters and restores sub-mm.
    ft = CRS.from_epsg(2263).to_wkt()
    src = np.column_stack([rng.uniform(9.8e5, 9.9e5, 500),
                           rng.uniform(2.0e5, 2.1e5, 500),
                           rng.uniform(0, 100, 500)])
    m, proc, s = normalize_to_meters(src, ft)
    assert s == ft and proc is not None, "ftUS must record a transform"
    back = restore_to_source(m, proc, s)
    err = np.abs(back - src).max()
    assert err < 1e-3, f"ftUS round-trip {err} ft exceeds sub-mm"

    # projected meter source is the identity fast-path (bytes unchanged, no source).
    utm = CRS.from_epsg(32618).to_wkt()
    um = np.column_stack([rng.uniform(5e5, 5.1e5, 300), rng.uniform(4.5e6, 4.51e6, 300),
                          rng.uniform(0, 50, 300)])
    m2, proc2, s2 = normalize_to_meters(um, utm)
    assert s2 is None and proc2 == utm, "meter UTM must be identity"
    assert m2 is um or np.array_equal(m2, um), "identity must not change bytes"
    assert np.array_equal(restore_to_source(m2, proc2, s2), um), "identity restore is a no-op"

    # compound ftUS-horizontal / metre-vertical: xy moves, z is unchanged.
    comp = CRS.from_user_input("EPSG:2263+5703").to_wkt()
    cz = src[:, 2].copy()
    mc, procc, sc = normalize_to_meters(src, comp)
    assert sc == comp, "compound must record a transform"
    assert np.allclose(mc[:, 2], cz), "metre-vertical z must be unchanged"
    assert not np.allclose(mc[:, 0], src[:, 0]), "xy must move"
    bc = restore_to_source(mc, procc, sc)
    assert np.abs(bc - src).max() < 1e-3, "compound round-trip exceeds sub-mm"

    # geographic (2D lon/lat): xy reprojects, z beside degrees stays metres.
    geo = CRS.from_epsg(4326).to_wkt()
    gsrc = np.column_stack([rng.uniform(-73.99, -73.90, 300),
                            rng.uniform(40.70, 40.78, 300), rng.uniform(0, 100, 300)])
    mg, procg, sg = normalize_to_meters(gsrc, geo)
    assert sg == geo and procg is not None, "geographic must record a transform"
    assert np.allclose(mg[:, 2], gsrc[:, 2]), "geographic z must stay metres (not radians)"
    assert np.abs(restore_to_source(mg, procg, sg)[:, :2] - gsrc[:, :2]).max() < 1e-7

    # D2: a legacy non-meter pred (proc ftUS, no source) hard-blocks at restore.
    try:
        restore_to_source(src, ft, None)
        raise AssertionError("legacy ftUS pred must hard-block")
    except ValueError as e:
        assert "re-run the inference job" in str(e)
    assert is_legacy_unit_scale_pred(ft, None) and not is_legacy_unit_scale_pred(utm, None)

    print("readers self-check OK")


if __name__ == "__main__":
    _selfcheck()
