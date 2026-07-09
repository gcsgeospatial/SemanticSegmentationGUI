"""Cross-model majority-vote ensemble over trainer_gui prediction files.

All three backbones (PTv3, KPConvX, RandLA-Net) predict on the SAME raw input
points and their output reaches your folder as {scene}_pred.<fmt> (las/laz/ply/
txt/csv after the GUI export, or the container's raw {scene}_pred.npz), so
ensembling is a per-point vote, independent of architecture, dataset, or class
count. Ensembling 2-3 models is the contest-era trick that survived into modern
recipes (~+2 mIoU for 3 models on the 2024 Waymo winner).

Scenes are matched by filename (minus extension) across the input folders.
Points are matched by row when the clouds are identical (the normal case); if a
model's cloud differs (e.g. it filtered points), its labels are carried onto
the first model's points by nearest neighbour. Vote ties go to the earliest-
listed model — put your strongest model first. Output is written in the first
input's format.

Usage:
  python ensemble_vote.py --inputs predsA predsB [predsC ...] --out out_dir
  python ensemble_vote.py --self-test

# ponytail: hard-label majority vote; upgrade path is saving per-class probs
# from inference and averaging those instead.
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def vote(labels):
    """labels: (M, N) int array, one row per model. Returns (N,) voted labels;
    ties go to the earliest model whose label has the max vote count."""
    labels = np.asarray(labels)
    m, n = labels.shape
    c = int(labels.max()) + 1
    counts = np.zeros((n, c), np.int32)
    for row in labels:
        counts[np.arange(n), row] += 1
    mx = counts.max(1)
    out = np.full(n, -1, labels.dtype)
    for row in labels:                       # priority order = input order
        sel = (out < 0) & (counts[np.arange(n), row] == mx)
        out[sel] = row[sel]
    return out


def load_pred(path):
    """xyz, labels, intensity(|None) from any prediction file. npz is read
    natively (container-safe); exported formats go through the GUI's readers."""
    if path.endswith(".npz"):
        z = np.load(path)
        return (z["xyz"].astype(np.float32), z["classification"].astype(np.int64),
                z["intensity"] if "intensity" in z.files else None)
    sys.path.insert(0, REPO_ROOT)
    from trainer_gui.readers import read_points
    c = read_points(path)
    lab = c.fields.get("classification")
    if lab is None and len(c.fields) == 1:   # single label-candidate field
        lab = next(iter(c.fields.values()))
    if lab is None:
        raise ValueError(f"{path}: no 'classification' field (has {list(c.fields)})")
    return c.xyz.astype(np.float32), np.asarray(lab, np.int64), c.intensity


def write_out(path, xyz, cls, intensity):
    if path.endswith(".npz"):
        d = {"xyz": xyz, "classification": cls.astype(np.int32)}
        if intensity is not None:
            d["intensity"] = intensity
        np.savez(path, **d)
        return
    sys.path.insert(0, REPO_ROOT)
    from trainer_gui.dataset import _write_pred
    from pathlib import Path
    _write_pred(Path(path), np.asarray(xyz, np.float64),
                np.clip(cls, 0, 255).astype(np.uint8), path.rsplit(".", 1)[1])


def align(ref_xyz, xyz, lab):
    """Map lab onto ref_xyz. Row-aligned when the clouds match; NN otherwise."""
    if len(xyz) == len(ref_xyz) and np.allclose(xyz, ref_xyz, atol=1e-4):
        return lab
    from scipy.spatial import cKDTree      # lazy: only the mismatch path needs it
    return lab[cKDTree(xyz).query(ref_xyz)[1]]


PRED_EXTS = (".npz", ".las", ".laz", ".ply", ".txt", ".csv")


def _scan(d):
    """{stem: path} of prediction files in a folder (any supported extension)."""
    out = {}
    for p in sorted(glob.glob(f"{d}/*_pred.*")):
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() in PRED_EXTS:
            out[stem] = p
    return out


def ensemble(input_dirs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    scans = [_scan(d) for d in input_dirs]
    if not scans[0]:
        raise FileNotFoundError(f"no *_pred.<npz|las|laz|ply|txt|csv> under {input_dirs[0]}")
    done = 0
    for stem, ref_path in scans[0].items():
        others = [s.get(stem) for s in scans[1:]]
        if any(p is None for p in others):
            miss = [d for d, s in zip(input_dirs[1:], scans[1:]) if stem not in s]
            print(f"  skip {stem}: missing in {miss}")
            continue
        ref_xyz, ref_lab, ref_itn = load_pred(ref_path)
        rows = [ref_lab]
        for p in others:
            xyz, lab, _ = load_pred(p)
            rows.append(align(ref_xyz, xyz, lab))
        voted = vote(np.stack(rows))
        out_path = f"{out_dir}/{os.path.basename(ref_path)}"
        write_out(out_path, ref_xyz, voted, ref_itn)
        agree = float(np.mean([np.mean(r == voted) for r in rows]))
        print(f"  {os.path.basename(out_path)}: {len(voted)} pts, "
              f"mean model-vote agreement {agree:.3f}")
        done += 1
    print(f"ensembled {done}/{len(scans[0])} scene(s) -> {out_dir}")


def self_test():
    # majority wins
    assert vote(np.array([[1, 2, 3], [1, 2, 0], [0, 2, 3]])).tolist() == [1, 2, 3]
    # all-way tie -> first model wins
    assert vote(np.array([[1], [2], [3]])).tolist() == [1]
    # two-way tie between models 2+3 vs model 1: first max-count holder wins
    assert vote(np.array([[5], [7], [7]])).tolist() == [7]
    # identical clouds align by row even with permuted-looking labels
    xyz = np.random.rand(100, 3).astype(np.float32)
    assert (align(xyz, xyz, np.arange(100)) == np.arange(100)).all()
    # mismatched cloud falls back to NN mapping
    sub = xyz[::2]
    out = align(xyz, sub, np.arange(len(sub)))
    assert len(out) == 100 and (out[::2] == np.arange(50)).all()
    print("self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", nargs="+", help="2+ prediction folders, strongest model first")
    ap.add_argument("--out", help="output folder for the voted predictions")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    elif a.inputs and a.out:
        if len(a.inputs) < 2:
            ap.error("--inputs needs at least 2 folders")
        ensemble(a.inputs, a.out)
    else:
        ap.error("either --self-test or both --inputs and --out")
