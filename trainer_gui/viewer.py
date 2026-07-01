"""Standalone Open3D point-cloud viewer (child process — never share Qt's event loop).

Usage:
  python -m trainer_gui.viewer <cloud>          one .ply/.npz/.las/.txt/... file
  python -m trainer_gui.viewer <folder>         every cloud in a folder, in a grid
  python -m trainer_gui.viewer <pred.ply> --gt <truth>   error map vs ground truth
      (truth = a class-coloured .ply, an .npz, or a folder searched by scene
      name). Points whose predicted class differs from the ground truth are
      coloured YELLOW; correct points are greyscale, shaded by intensity (from the
      prediction npz or a sibling scene; `--intensity FILE` to point at it) so
      structure stays visible.

Points are coloured strictly by class (Open3D's point_color_option is forced to
Color, never the Z gradient). The colour key is printed to stdout; the matplotlib
fallback also draws it in-figure (the Open3D GL window can't overlay text).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

YELLOW = (1.0, 1.0, 0.0)
GREY = 0.55

# A "key" is a list of (label, rgb01) pairs driving the on-screen colour legend.


def _key_from_rgb(rgb_u8: np.ndarray) -> list[tuple[str, tuple]]:
    """Colour key for a palette-coloured cloud that carries no class names: label
    each matched class index generically ('class i'). [] when nothing matches (a
    raw RGB scene gets no spurious legend)."""
    from .palette import class_from_rgb, palette_for
    cls = class_from_rgb(rgb_u8)
    present = sorted({int(c) for c in np.unique(cls) if c >= 0})
    if not present:
        return []
    pal = palette_for(max(present) + 1).astype(np.float64) / 255.0
    return [(f"class {i}", tuple(pal[i].tolist())) for i in present]


def _key_for_names(rgb_u8: np.ndarray, names: list) -> list[tuple[str, tuple]]:
    """Colour key for a prediction coloured with palette_for(len(names)) — the exact
    palette the training scripts bake into prediction PLYs. Returns the full class
    list when any point matches that palette, so a chosen dataset's class scheme
    labels the prediction; [] if nothing matches (a raw RGB scene), so plain clouds
    get no spurious legend."""
    from .palette import palette_for
    pal = palette_for(len(names))
    matched = any(
        np.any((rgb_u8[:, 0] == c[0]) & (rgb_u8[:, 1] == c[1]) & (rgb_u8[:, 2] == c[2]))
        for c in pal)
    if not matched:
        return []
    return [(names[i], tuple((pal[i] / 255.0).tolist())) for i in range(len(names))]


def _npz_class(z) -> tuple[np.ndarray, str] | tuple[None, None]:
    """An npz's per-point class array + its key name: prediction (`pred`) or
    ground-truth/dataset (`label`), whichever is present. -1 stays -1 (ignore)."""
    for k in ("pred", "label"):
        if k in z:
            return np.asarray(z[k], np.int64).reshape(-1), k
    return None, None


def _load(path: Path, names: list | None = None):
    """One file -> (xyz float64 (N,3), rgb01 (N,3)|None, key). `names` is a chosen
    dataset's class names (a palette pick): used for the legend, and for npz
    predictions whose file carries no class_names; the file's own class_names win."""
    if path.suffix.lower() == ".npz":
        z = np.load(str(path), allow_pickle=False)
        cls, _ = _npz_class(z)
        if cls is not None and "xyz" in z:
            from .palette import generic_names, palette_for
            xyz = np.asarray(z["xyz"], np.float64)
            names_used = ([str(n) for n in z["class_names"]] if "class_names" in z
                          else (names or generic_names(int(cls.max()) + 1)))
            pal = palette_for(max(int(cls.max()) + 1, len(names_used))).astype(np.float64) / 255.0
            colors = np.full((len(xyz), 3), GREY)          # -1/ignore -> grey
            ok = cls >= 0
            colors[ok] = pal[np.clip(cls[ok], 0, len(pal) - 1)]
            key = [(names_used[i] if i < len(names_used) else f"class_{i}", tuple(pal[i].tolist()))
                   for i in sorted(set(cls[ok].tolist()))]
            return xyz, colors, key
    from .readers import read_points
    cloud = read_points(path)
    if cloud.rgb is not None:
        key = _key_for_names(cloud.rgb, names) if names else _key_from_rgb(cloud.rgb)
        return cloud.xyz, cloud.rgb.astype(np.float64) / 255.0, key
    if cloud.intensity is not None:          # no class colours -> shade by intensity
        v = _intensity_value(cloud.intensity)
        return cloud.xyz, np.stack([v, v, v], axis=1), []
    return cloud.xyz, None, []


def _load_folder(folder: Path, names: list | None = None):
    """Every cloud in a folder, centred and tiled into a grid so they're all
    visible at once -> (xyz, rgb01|None, key). `names` = a chosen dataset's class
    names for the legend (see _load)."""
    from .readers import SUPPORTED_EXTS, read_points
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".ply")
    if not files:
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if not files:
        raise FileNotFoundError(f"no point-cloud files in {folder}")

    clouds = []
    for p in files:
        try:
            c = read_points(p)
        except Exception as e:  # noqa: BLE001 — one bad file shouldn't sink the view
            print(f"  skip {p.name}: {e}")
            continue
        rgb = c.rgb.astype(np.float64) / 255.0 if c.rgb is not None else None
        if rgb is None and c.intensity is not None:   # no colours -> shade by intensity
            v = _intensity_value(c.intensity)
            rgb = np.stack([v, v, v], axis=1)
        clouds.append((c.xyz, rgb))
    if not clouds:
        raise FileNotFoundError(f"no readable clouds in {folder}")

    ncol = max(1, int(math.ceil(math.sqrt(len(clouds)))))
    dx = max((np.ptp(c[:, 0]) for c, _ in clouds), default=1.0) * 1.15
    dy = max((np.ptp(c[:, 1]) for c, _ in clouds), default=1.0) * 1.15
    xs, rgbs, any_rgb = [], [], False
    for i, (cz, rgb) in enumerate(clouds):
        r, col = divmod(i, ncol)
        xs.append(cz - cz.mean(0) + np.array([col * dx, -r * dy, 0.0]))
        if rgb is not None:
            rgbs.append(rgb)
            any_rgb = True
        else:
            rgbs.append(np.full((len(cz), 3), 0.6))
    xyz = np.vstack(xs)
    rgb = np.vstack(rgbs) if any_rgb else None
    if rgb is not None:
        rgb_u8 = (rgb * 255).round().astype(np.uint8)
        key = _key_for_names(rgb_u8, names) if names else _key_from_rgb(rgb_u8)
    else:
        key = []
    print(f"  {len(clouds)} clouds tiled in a {ncol}-wide grid")
    return xyz, rgb, key


def _intensity_value(intensity: np.ndarray) -> np.ndarray:
    """(N,) intensity -> (N,) brightness in [0.15, 0.85], robustly scaled by its
    own 2nd–98th percentile so the cloud reads like a LiDAR intensity image."""
    inten = np.asarray(intensity, np.float64).reshape(-1)
    lo, hi = np.percentile(inten, [2, 98])
    if hi <= lo:
        return np.full(len(inten), 0.5)
    return 0.15 + 0.7 * np.clip((inten - lo) / (hi - lo), 0, 1)


def error_colors(pred: np.ndarray, gt: np.ndarray, intensity: np.ndarray | None = None) -> np.ndarray:
    """Per-point colours for a prediction-vs-GT error map: wrong predictions are
    YELLOW, everything else (correct + unlabeled GT) is grey. When `intensity` is
    given BOTH vary with it — grey = (v,v,v), yellow = (v,v,0), the same brightness
    with blue dropped — so the cloud reads like an intensity image with the errors
    picked out in yellow."""
    wrong = (gt >= 0) & (pred != gt)
    if intensity is not None:
        v = _intensity_value(intensity)
        colors = np.stack([v, v, v], axis=1)   # correct/unlabeled: grey by intensity
        colors[wrong, 2] = 0.0                 # wrong: drop blue -> yellow by intensity
    else:
        colors = np.full((len(gt), 3), GREY)
        colors[wrong] = YELLOW
    return colors


def _load_intensity(path: Path) -> np.ndarray | None:
    """Per-point intensity from any readable cloud (npz 'intensity', ASCII col 3,
    LAS intensity, …), or None."""
    from .readers import read_points
    try:
        inten = read_points(path).intensity
    except Exception:  # noqa: BLE001 — intensity grading is optional, never block
        return None
    return None if inten is None else np.asarray(inten, np.float64).reshape(-1)


def _autofind_intensity(pred_path: Path, scene: str) -> np.ndarray | None:
    """Find the scene's input cloud (which carries intensity) near the prediction
    by matching its name — predictions live in predictions/, the inputs usually in
    a sibling scenes/. Returns intensity or None (grading is optional)."""
    names = [f"{scene}.npz", f"{scene}.txt"]
    dirs = [pred_path.parent] + [p / "scenes" for p in list(pred_path.parents)[:4]]
    for d in dirs:
        for nm in names:
            f = d / nm
            if f.exists() and (inten := _load_intensity(f)) is not None:
                print(f"  intensity grading from {f}")
                return inten
    return None


def _read_gt(gt_path: Path, scene: str) -> np.ndarray:
    """Ground-truth class indices for a scene. gt_path is either a file —
    a class-coloured .ply or an .npz with label/pred — or a folder searched by
    scene name (_gt.ply / .ply)."""
    from .palette import class_from_rgb
    from .readers import read_points

    if gt_path.is_dir():
        for cand in (f"{scene}_gt.ply", f"{scene}.ply", f"{scene}_pred.ply"):
            if (gt_path / cand).exists():
                gt_path = gt_path / cand
                break
        else:
            raise FileNotFoundError(f"no ground truth for '{scene}' in {gt_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"ground-truth file not found: {gt_path}")

    ext = gt_path.suffix.lower()
    if ext == ".ply":
        cloud = read_points(gt_path)
        if cloud.rgb is None:
            raise ValueError(f"{gt_path.name}: PLY has no colours to decode classes from")
        return class_from_rgb(cloud.rgb)
    if ext == ".npz":
        g, _ = _npz_class(np.load(str(gt_path), allow_pickle=False))
        if g is None:
            raise ValueError(f"{gt_path.name}: npz has no 'label'/'pred'")
        return g
    raise ValueError(f"{gt_path.name}: unsupported ground-truth file - use a "
                     f"class-coloured .ply or an .npz with 'label'/'pred'")


def _load_pred(pred_path: Path):
    """(xyz, pred_classes, intensity|None, scene) from a prediction .ply
    (class-coloured) or .npz (with pred/label + xyz)."""
    from .palette import class_from_rgb
    from .readers import read_points

    inten = None
    if pred_path.suffix.lower() == ".npz":
        z = np.load(str(pred_path), allow_pickle=False)
        pred, _ = _npz_class(z)
        if pred is None or "xyz" not in z:
            raise ValueError(f"{pred_path.name}: npz has no 'pred'/'label' to compare")
        xyz = np.asarray(z["xyz"], np.float64)
        if "intensity" in z:
            inten = np.asarray(z["intensity"], np.float64).reshape(-1)
    else:
        cloud = read_points(pred_path)
        if cloud.rgb is None:
            raise ValueError(f"{pred_path.name}: no colours, so the predicted class "
                             f"can't be read - pass a prediction .ply or .npz")
        xyz, pred = cloud.xyz, class_from_rgb(cloud.rgb)
        inten = None if cloud.intensity is None else np.asarray(cloud.intensity, np.float64)

    scene = pred_path.stem
    for suffix in ("_pred", "_gt"):
        scene = scene.replace(suffix, "")
    return xyz, pred, inten, scene


def prediction_metrics(pred_path, gt_path) -> dict:
    """Overall accuracy + mIoU + per-class IoU of a prediction cloud against
    ground truth, scored on points that carry a GT label. mIoU averages only the
    classes present in GT or prediction (absent classes don't drag it to zero)."""
    pred_path, gt_path = Path(pred_path), Path(gt_path)
    _, pred, _, scene = _load_pred(pred_path)
    gt = _read_gt(gt_path, scene)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    has = gt >= 0
    labeled = int(has.sum())
    acc = float((pred[has] == gt[has]).sum()) / max(labeled, 1)
    present = sorted({int(c) for c in np.unique(pred[has])} | {int(c) for c in np.unique(gt[has])})
    present = [c for c in present if c >= 0]   # drop the undecodable (-1) class from mIoU
    ious = {}
    for c in present:
        inter = int(((pred == c) & (gt == c) & has).sum())
        union = int((((pred == c) | (gt == c)) & has).sum())
        ious[c] = inter / union if union else 0.0
    miou = float(np.mean(list(ious.values()))) if ious else 0.0
    return {"scene": scene, "accuracy": acc, "miou": miou,
            "labeled": labeled, "per_class_iou": ious}


def compare_clouds(pred_path: Path, gt_path: Path, intensity_path: Path | None = None):
    """Error map for a prediction (.ply class-coloured, or .npz with pred/label)
    against ground truth. Correct points are shaded by intensity (so structure
    stays visible), wrong points are YELLOW.
    gt_path: a folder (matched by scene name) or a single file — a class-coloured
    .ply or an .npz.
    intensity_path: optional cloud to read intensity from; otherwise it's taken
    from the prediction npz, or auto-found from a sibling scene. -> (xyz, colors, key)."""
    xyz, pred, inten, scene = _load_pred(pred_path)
    if inten is None and intensity_path is not None:
        inten = _load_intensity(Path(intensity_path))
    if inten is None:
        inten = _autofind_intensity(pred_path, scene)
    gt = _read_gt(gt_path, scene)

    n = min(len(xyz), len(pred), len(gt))
    if not (len(xyz) == len(pred) == len(gt)):
        print(f"  warning: point counts differ (cloud {len(xyz):,}, "
              f"gt {len(gt):,}) - comparing the first {n:,}")
    xyz, pred, gt = xyz[:n], pred[:n], gt[:n]
    inten_n = inten[:n] if inten is not None and len(inten) >= n else None
    colors = error_colors(pred, gt, inten_n)

    has = gt >= 0
    labeled = int(has.sum())
    wrong = int((has & (pred != gt)).sum())
    acc = 1.0 - wrong / max(labeled, 1)
    print(f"  {scene}: accuracy {acc:.4f}  ({wrong:,} of {labeled:,} labeled pts wrong)")
    correct_label = "correct (shaded by intensity)" if inten_n is not None else "correct"
    key = [("wrong prediction", YELLOW), (correct_label, (GREY, GREY, GREY))]
    return xyz, colors, key


# ----------------------------------------------------------------------- export

def _save_ply(path: Path, xyz, rgb):
    """Write xyz + rgb01 colours to an ASCII PLY (rgb None -> uniform grey)."""
    rgb_u8 = ((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8) if rgb is not None
              else np.full((len(xyz), 3), int(GREY * 255), np.uint8))
    arr = np.column_stack([np.asarray(xyz, np.float64), rgb_u8])
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), arr, fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"],
               header=header, comments="")


# ----------------------------------------------------------------------- display

def _parse_palette(spec: str | None):
    """'r,g,b;r,g,b;…' -> [[r,g,b], …] (one colour per class index), or None.
    Passed by the GUI's Configure Palette so the viewer paints the user's colours."""
    if not spec:
        return None
    out = [[int(x) for x in part.split(",")] for part in spec.split(";") if part.strip()]
    return out or None


def _recolor_to_palette(rgb01: np.ndarray, names: list, palette: list):
    """Recolour a class-coloured cloud to a custom per-class palette and rebuild the
    legend. Predictions are baked with palette_for(N)[class]; map each point's baked
    colour back to its class index, then paint it `palette[index]`. Returns
    (rgb01, key), or None when the cloud isn't palette-coloured (raw RGB scenes are
    left untouched, so they don't get a bogus recolour or legend)."""
    from .palette import palette_for
    rgb_u8 = (np.clip(np.asarray(rgb01), 0, 1) * 255).round().astype(np.int64)
    base = palette_for(len(palette))
    out = np.array(rgb01, np.float64, copy=True)
    matched = False
    for bc, cc in zip(base, palette):
        m = (rgb_u8[:, 0] == bc[0]) & (rgb_u8[:, 1] == bc[1]) & (rgb_u8[:, 2] == bc[2])
        if m.any():
            out[m] = np.asarray(cc, np.float64) / 255.0
            matched = True
    if not matched:
        return None
    key = [(names[i] if i < len(names) else f"class_{i}",
            tuple((np.asarray(palette[i], np.float64) / 255.0).tolist()))
           for i in range(len(palette))]
    return out, key


def _print_key(title, key):
    if key:
        print(f"{title} - colour key:")
        for label, (r, g, b) in key:
            print(f"  {label:<18} rgb({int(r*255)},{int(g*255)},{int(b*255)})")


def _show(xyz, rgb, key, title, max_points):
    if len(xyz) > max_points:
        keep = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz = xyz[keep]
        rgb = rgb[keep] if rgb is not None else None
        print(f"(subsampled to {max_points:,} of the original points)")
    _print_key(title, key)
    if rgb is None:
        print("⚠ no class/colour data in this file - drawing it uniform grey, NOT by class. "
              "(An input/raw scene carries no labels; class colours come from a prediction "
              "PLY or an npz with a 'pred'/'label' key.)")
    print(f"{len(xyz):,} points. Drag to rotate, scroll to zoom, close window to exit.")
    xyz = xyz - xyz.mean(0, keepdims=True)

    try:
        import open3d as o3d
    except Exception as e:   # not just ImportError: a libstdc++/GL load failure too
        print(f"  (open3d unavailable: {e} - falling back to the matplotlib viewer)")
        o3d = None

    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # Always set per-point colours: class palette if we have it, else uniform
        # grey — never let Open3D fall back to its height-gradient default.
        pcd.colors = o3d.utility.Vector3dVector(
            rgb if rgb is not None else np.full((len(xyz), 3), GREY))
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=820)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = 2.5
        opt.background_color = np.asarray([1.0, 1.0, 1.0])
        # Colour strictly by each point's own RGB (the class palette) — never by
        # the Z coordinate, whose gradient looked like a height map.
        opt.point_color_option = o3d.visualization.PointColorOption.Color
        vis.run()
        vis.destroy_window()
        return

    _show_mpl(xyz, rgb, key, title)


def _show_mpl(xyz, rgb, key, title):
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except Exception as e:   # e.g. libstdc++ CXXABI mismatch on Linux
        print(f"\n⚠ No 3D viewer available - open3d couldn't load and matplotlib "
              f"failed too ({e}).\n  The colour key above tells you how each class "
              f"is coloured; the prediction file is at the path in the title. On "
              f"Linux this is usually a libstdc++ mismatch - launch via `pixi run "
              f"gui` (or `pixi run python -m trainer_gui.viewer …`) so the env's "
              f"libstdc++ is used.")
        return

    if len(xyz) > 150_000:
        keep = np.random.default_rng(0).choice(len(xyz), 150_000, replace=False)
        xyz, rgb = xyz[keep], (rgb[keep] if rgb is not None else None)
    fig = plt.figure(title, figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    c = rgb if rgb is not None else np.full((len(xyz), 3), GREY)   # never colour by height
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.3, c=c, linewidths=0)
    ax.set_box_aspect((np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), max(np.ptp(xyz[:, 2]), 1.0)))
    if key:
        ax.legend(handles=[mpatches.Patch(color=col, label=lab) for lab, col in key],
                  loc="upper right", fontsize=8)
    plt.show()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a cloud file, or a folder of clouds")
    ap.add_argument("--gt", default=None,
                    help="ground truth (a class-coloured .ply, an .npz, or a folder) "
                         "- turns PATH into a prediction and shows an error map vs this GT")
    ap.add_argument("--intensity", default=None,
                    help="with --gt: cloud to read per-point intensity from, used to "
                         "shade the correct points (default: the prediction's own "
                         "intensity, or a sibling scene found by name)")
    ap.add_argument("--class-names", default=None,
                    help="comma-separated class names (a dataset's palette) for the "
                         "colour legend, and to label predictions whose file carries no "
                         "class names; colours come from the shared categorical palette")
    ap.add_argument("--palette", default=None,
                    help="custom per-class colours 'r,g,b;r,g,b;…' (by class index) to "
                         "recolour a class-coloured cloud and its legend (the GUI's "
                         "Configure Palette); defaults to the baked categorical palette")
    ap.add_argument("--max-points", type=int, default=8_000_000,
                    help="random subsample cap for display")
    ap.add_argument("--save", default=None,
                    help="write the cloud (with the colours that would be shown - e.g. "
                         "the --gt error map) to this .ply and exit, instead of opening a window")
    args = ap.parse_args(argv)

    names = ([s.strip() for s in args.class_names.split(",") if s.strip()]
             if args.class_names else None)
    palette = _parse_palette(args.palette)
    path = Path(args.path)
    if args.gt:
        xyz, rgb, key = compare_clouds(path, Path(args.gt),
                                       Path(args.intensity) if args.intensity else None)
        title = f"{path.stem} vs ground truth"
    elif path.is_dir():
        xyz, rgb, key = _load_folder(path, names)
        title = path.name
    else:
        xyz, rgb, key = _load(path, names)
        title = path.name
    # Apply a custom palette (Configure Palette): recolour the class-coloured cloud
    # to the user's colours. Not for --gt (that's a yellow/grey error map).
    if palette and names and rgb is not None and not args.gt:
        recolored = _recolor_to_palette(rgb, names, palette)
        if recolored is not None:
            rgb, key = recolored
    if args.save:
        _save_ply(Path(args.save), xyz, rgb)
        print(f"saved {len(xyz):,} points -> {args.save}")
        return 0
    _show(xyz, rgb, key, title, args.max_points)
    return 0


if __name__ == "__main__":
    sys.exit(main())
