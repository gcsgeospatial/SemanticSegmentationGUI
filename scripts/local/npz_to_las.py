"""Convert a trainer npz (scene or prediction) to .las for CloudCompare:
xyz -> cloud, rgb -> color, every other per-point channel -> a float32
Extra Bytes field listed by name.

Usage:
  python npz_to_las.py scene.npz [more.npz ...]     # writes sibling .las
  python npz_to_las.py --self-test
"""
import sys

if sys.version_info[0] < 3:
    sys.exit("npz_to_las.py needs python3 (plain 'python' here is 2.x)")

import numpy as np


def npz_to_las(path):
    import laspy
    z = np.load(path)
    xyz = np.asarray(z["xyz"], np.float64)
    rgb = np.asarray(z["rgb"]) if "rgb" in z.files else None

    h = laspy.LasHeader(point_format=7 if rgb is not None else 6, version="1.4")
    # npz_ prefix when shadowing a native LAS dim: the integer native slots
    # would truncate the normalized floats
    taken = set(h.point_format.dimension_names)
    extras = {}
    for k in z.files:
        if k in ("xyz", "rgb", "crs_wkt"):
            continue
        v = np.asarray(z[k])
        if v.ndim == 1 and len(v) == len(xyz) and np.issubdtype(v.dtype, np.number):
            extras[("npz_" + k if k in taken else k)[:32]] = v.astype(np.float32)
    for k in extras:
        h.add_extra_dim(laspy.ExtraBytesParams(name=k, type=np.float32))
    if "crs_wkt" in z.files:
        try:
            from pyproj import CRS
            crs = CRS.from_wkt(str(z["crs_wkt"]))
            h.add_crs(crs)
            # npz coords are meters (scaled at ingest); a non-meter CRS (ft
            # state planes) needs them back in source units to match its WKT.
            if not crs.is_geographic:
                f = float(crs.axis_info[0].unit_conversion_factor)
                if abs(f - 1.0) > 1e-6:
                    xyz = xyz / f         # h.offsets is set below from this xyz
        except Exception:
            pass                          # a CRS-less viewable file beats none
    h.offsets = xyz.min(0)
    h.scales = [0.001] * 3
    las = laspy.LasData(h)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        scale = 257 if rgb.max() <= 255 else 1      # las color is 16-bit
        las.red, las.green, las.blue = (rgb * scale).astype(np.uint16).T
    for k, v in extras.items():
        setattr(las, k, v)
    out = str(path)[: -len(".npz")] + ".las"
    las.write(out)
    # ponytail: .format, not f-string -- keeps the file py2-parseable so the
    # version guard above can fire instead of a bare SyntaxError.
    print("  {} -> {} ({:,} pts; fields: {})".format(
        path, out, len(xyz), ", ".join(extras) or "none"))
    return out


def self_test():
    import tempfile
    from pathlib import Path
    import laspy
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.npz"
        np.savez(p, xyz=np.random.rand(10, 3) * 100,
                 rgb=np.random.randint(0, 256, (10, 3)),
                 intensity=np.linspace(0, 2, 10).astype(np.float32),
                 feat_hag=np.linspace(0, 30, 10).astype(np.float32),
                 label=np.array([-1] * 5 + [2] * 5, np.int32))
        las = laspy.read(npz_to_las(p))
        assert np.allclose(np.asarray(las.feat_hag), np.linspace(0, 30, 10), atol=1e-5)
        assert np.asarray(las.label).tolist() == [-1] * 5 + [2] * 5   # f32 keeps -1
        assert np.asarray(las.red).max() > 255                        # 16-bit color
        # native-name collision lands under npz_ with full float precision
        assert np.allclose(np.asarray(las.npz_intensity), np.linspace(0, 2, 10), atol=1e-6)
    print("self-test OK")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--self-test"]:
        self_test()
    elif args:
        for a in args:
            npz_to_las(a)
    else:
        sys.exit(__doc__)
