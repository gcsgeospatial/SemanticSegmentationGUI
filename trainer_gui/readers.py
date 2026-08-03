"""Read every supported point-cloud format into one Cloud shape; `fields` holds
every per-point 1-D numeric array so any can be offered as the label source."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SUPPORTED_EXTS = {".las", ".laz", ".ply", ".txt", ".csv", ".xyz", ".pts", ".pcd", ".npy", ".npz"}
ASCII_EXTS = {".txt", ".csv", ".xyz", ".pts"}


@dataclass
class Cloud:
    xyz: np.ndarray
    rgb: np.ndarray | None = None
    intensity: np.ndarray | None = None
    return_number: np.ndarray | None = None
    fields: dict = field(default_factory=dict)
    crs_wkt: str | None = None
    source_crs_wkt: str | None = None

    @property
    def n(self) -> int:
        return len(self.xyz)


# pyproj is a GUI-env dep, so keep it lazy; normalize_to_meters (ingest) and restore_to_source (export) are inverses built from the same Transformer pair

def _horizontal_crs(crs):
    """Horizontal sub-CRS of a compound CRS, else the CRS itself."""
    return crs.sub_crs_list[0] if crs.is_compound else crs


def vertical_unit_factor(crs) -> float:
    """Meters per source vertical unit: the vertical-axis unit of a compound/3D
    CRS, else a PROJECTED horizontal's linear unit. z beside geographic (angular)
    coords is metres, never radians - only a projected horizontal lends its unit."""
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
    transform occurred (its presence is the exact round-trip signal). The identity
    fast-path (already projected+meter, or no/unparseable CRS) keeps coords
    byte-identical; xy goes via pyproj Transformer(always_xy=True), z by vertical
    unit."""
    xyz = np.asarray(xyz, np.float64)
    if not source_crs_wkt:
        return xyz, None, None
    from pyproj import CRS, Transformer
    try:
        crs = CRS.from_wkt(source_crs_wkt)
    except Exception as e:
        raise ValueError("source CRS WKT is present but unparseable, so the cloud "
                         "cannot be reprojected to meters. Declare the EPSG code on "
                         "the prep/inference page to override it") from e
    if _is_meter_horizontal(crs):
        return xyz, source_crs_wkt, None
    utm = _estimate_utm(crs, xyz, CRS, Transformer)
    x, y = Transformer.from_crs(_horizontal_crs(crs), utm, always_xy=True).transform(
        xyz[:, 0], xyz[:, 1])
    z = xyz[:, 2] * vertical_unit_factor(crs)
    out = np.column_stack([x, y, z]).astype(np.float64)
    # pyproj returns inf (not an error) for points outside the target zone, and a single inf poisons the cache, surfacing later as a cryptic CUDA gather assert
    bad = int((~np.isfinite(out[:, :2])).any(1).sum())
    if bad:
        raise ValueError(
            f"reprojecting to meters produced {bad} non-finite XY point(s): they fall "
            "outside the target UTM zone (no-data XY sentinels, or the wrong source CRS). "
            "Remove those points or declare the correct EPSG on the prep/inference page")
    return out, utm.to_wkt(), source_crs_wkt


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
    frame) - but a non-meter proc with no source is a legacy pred: hard block (D2)."""
    xyz = np.asarray(xyz, np.float64)
    if source_crs_wkt is None:
        if is_legacy_unit_scale_pred(proc_crs_wkt, source_crs_wkt):
            raise ValueError("prediction npz predates the CRS reprojection contract "
                             "(non-meter CRS with no source_crs_wkt); re-run the "
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
    # D1 remedy: a declared EPSG fills in for a cloud carrying no CRS and rides the same reprojection-to-meters (formats other than las/laz never carry one)
    if declared_crs_epsg is not None and cloud.crs_wkt is None:
        from pyproj import CRS
        wkt = CRS.from_epsg(int(declared_crs_epsg)).to_wkt()
        cloud.xyz, cloud.crs_wkt, cloud.source_crs_wkt = normalize_to_meters(cloud.xyz, wkt)
    return cloud


def list_label_fields(path: str | Path) -> list[str]:
    """Cheap probe: names usable as the ground-truth label source."""
    return sorted(probe_points(path).fields.keys())


def probe_points(path: str | Path):
    """read_points stand-in for pick-time probing. las/laz answer from the header
    alone - dimension names, std-channel presence, CRS, bbox corners as .xyz -
    so a multi-GB file never blocks the GUI thread or touches RAM. The corners
    ride the real normalize_to_meters, keeping the CRS story identical to a full
    read. Consumers may only use field NAMES and channel presence: .fields values
    are None and array attrs are 0-length sentinels."""
    # ponytail: other formats still full-read; move the probe to a worker if big PLYs surface
    p = Path(path)
    if p.suffix.lower() not in (".las", ".laz"):
        return read_points(p)
    import laspy

    with laspy.open(str(p)) as f:
        h = f.header
        if h.point_count == 0:
            return read_points(p)
        try:
            crs = h.parse_crs()
        except Exception:
            crs = None
        dims = [d.name for d in h.point_format.dimensions]
        corners = np.array([h.mins, h.maxs], np.float64)
    lower = {d.lower() for d in dims}
    corners, proc_wkt, source_wkt = normalize_to_meters(
        corners, crs.to_wkt() if crs is not None else None)
    flag = np.empty(0, np.float32)
    return SimpleNamespace(
        xyz=corners,
        fields={d: None for d in dims if d.lower() not in ("x", "y", "z")},
        intensity=flag if "intensity" in lower else None,
        rgb=flag if {"red", "green", "blue"} <= lower else None,
        return_number=flag if "return_number" in lower else None,
        crs_wkt=proc_wkt, source_crs_wkt=source_wkt)


def _read_las(path) -> Cloud:
    import laspy

    las = laspy.read(str(path))
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    try:
        crs = las.header.parse_crs()
    except Exception:
        crs = None

    dims = {d.name.lower() for d in las.point_format.dimensions}
    rgb = None
    if {"red", "green", "blue"}.issubset(dims):
        rgb = np.column_stack([las.red, las.green, las.blue]).astype(np.float64)
        if rgb.max() > 255:
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
    if source_wkt is not None:
        vf = vertical_unit_factor(crs)
        for k in list(fields):
            if k.lower().replace("_", "") in ("heightaboveground", "hag"):
                fields[k] = np.asarray(fields[k], np.float64) * vf
    return Cloud(xyz=xyz, rgb=rgb, intensity=intensity, return_number=ret, fields=fields,
                 crs_wkt=proc_wkt, source_crs_wkt=source_wkt)


_PDAL_SKIP = {"x", "y", "z", "nx", "ny", "nz", "normalx", "normaly", "normalz", "alpha"}


def _read_pdal(path, reader_type) -> Cloud:
    """ply/pcd via a single-stage PDAL pipeline (PDAL already ships for HAG)."""
    import json

    import pdal

    pipe = pdal.Pipeline(json.dumps([{"type": reader_type, "filename": str(path)}]))
    pipe.execute()
    if not len(pipe.arrays) or not len(pipe.arrays[0]):
        raise ValueError(f"{path}: no points read")
    v = pipe.arrays[0]
    low = {n.lower(): n for n in v.dtype.names}

    def _pick(*keys):
        for k in keys:
            if k in low:
                return np.asarray(v[low[k]])
        return None

    xyz = np.stack([v[low["x"]], v[low["y"]], v[low["z"]]], -1).astype(np.float64)
    rgb = None
    if {"red", "green", "blue"} <= low.keys():
        rgb = np.stack([v[low["red"]], v[low["green"]], v[low["blue"]]], -1).astype(np.float64)
        if rgb.max() > 255:  # 16-bit color scaled to 8-bit
            rgb = rgb / 257.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    intensity = _pick("intensity", "scalar_intensity")
    if intensity is not None:
        intensity = intensity.astype(np.float32)
    ret = _pick("returnnumber", "return_number", "scalar_return_number", "scalar_returnnumber")
    if ret is not None:
        ret = ret.astype(np.float32)
    fields = {}
    for name in v.dtype.names:
        if name.lower() in _PDAL_SKIP:
            continue
        arr = np.asarray(v[name])
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            fields[name] = arr
    return Cloud(xyz=xyz, rgb=rgb, intensity=intensity, return_number=ret, fields=fields)


def _read_ply(path) -> Cloud:
    return _read_pdal(path, "readers.ply")


def _sniff_delimiter(path) -> str | None:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
    return "," if "," in line else None


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


def _read_pcd(path) -> Cloud:
    return _read_pdal(path, "readers.pcd")


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
    # a staged/pred npz already carries its processing (+ source) CRS, so surface it and the declared-EPSG fill can never override it
    crs_wkt = str(z["crs_wkt"]) if "crs_wkt" in z.files else None
    source_crs_wkt = str(z["source_crs_wkt"]) if "source_crs_wkt" in z.files else None
    return Cloud(xyz=xyz, rgb=rgb, intensity=intensity, return_number=ret, fields=fields,
                 crs_wkt=crs_wkt, source_crs_wkt=source_crs_wkt)
