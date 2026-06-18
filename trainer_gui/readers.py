"""Read point clouds of every supported format into one in-memory shape.

Formats: .las/.laz (laspy), .ply (plyfile), ASCII .txt/.csv/.xyz/.pts (numpy),
.pcd (open3d), .npy/.npz (numpy). Each reader returns a Cloud; `fields` holds
every per-point 1-D numeric array found, so the GUI can offer any of them as
the ground-truth label source.
"""

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

    @property
    def n(self) -> int:
        return len(self.xyz)


def read_points(path: str | Path) -> Cloud:
    ext = Path(path).suffix.lower()
    if ext in (".las", ".laz"):
        return _read_las(path)
    if ext == ".ply":
        return _read_ply(path)
    if ext in ASCII_EXTS:
        return _read_ascii(path)
    if ext == ".pcd":
        return _read_pcd(path)
    if ext in (".npy", ".npz"):
        return _read_numpy(path)
    raise ValueError(f"Unsupported point-cloud format: {ext} ({path})")


def list_label_fields(path: str | Path) -> list[str]:
    """Cheap-ish probe: names usable as the ground-truth label source."""
    return sorted(read_points(path).fields.keys())


# ---------------------------------------------------------------- las / laz

def _read_las(path) -> Cloud:
    import laspy

    las = laspy.read(str(path))
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

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
    skip = {"x", "y", "z", "red", "green", "blue"}
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
    return Cloud(xyz=xyz, rgb=rgb, intensity=intensity, return_number=ret, fields=fields)


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

    fields = {}
    skip = {"x", "y", "z", "red", "green", "blue", "nx", "ny", "nz", "alpha"}
    for name in v.dtype.names:
        if name in skip:
            continue
        arr = np.asarray(v[name])
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            fields[name] = arr
    return Cloud(xyz=xyz, rgb=rgb, intensity=intensity, fields=fields)


# ---------------------------------------------------------------- ascii

def _sniff_delimiter(path) -> str | None:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
    return "," if "," in line else None  # None -> whitespace


def _read_ascii(path) -> Cloud:
    """Columns: 0-2 = x,y,z; col 3 defaults to intensity, col 4 to return number
    (the IEEE _PC3.txt convention); every extra column is exposed as a label
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
    return Cloud(xyz=xyz, rgb=rgb)


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
    fields = {}
    used = {"xyz", "points", "coord", "coords", "rgb", "color", "colors", "intensity"}
    for key in z.files:
        if key in used:
            continue
        arr = np.asarray(z[key])
        if arr.ndim == 1 and len(arr) == len(xyz) and np.issubdtype(arr.dtype, np.number):
            fields[key] = arr
    return Cloud(xyz=xyz, rgb=rgb, intensity=intensity, fields=fields)
