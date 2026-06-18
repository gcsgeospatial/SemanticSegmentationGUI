#!/usr/bin/env python3
"""Combine IEEE ground-truth labels with the raw point clouds into coloured PLYs.

For each <scene>_PC3.txt (x,y,z,intensity,return) it reads the matching
<scene>_CLS.txt (one ASPRS code per line, index-aligned) and writes
<scene>_gt.ply coloured by class with the SAME palette the prediction PLYs use,
so ground truth and predictions look identical in the 3D viewer / folder view.

Defaults to the IEEE Validate-Track4 split -> Modal/groundtruth/. Point --pc/--cls
at the Train dirs for the training split:

  python make_groundtruth_ply.py \
      --pc  ".../IEEE/Train-Track4/Track4" \
      --cls ".../IEEE/Train-Track4-Truth/Track4-Truth"
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# Matches trainer_gui.palette.IEEE_PALETTE (contiguous class index -> RGB).
IEEE_PALETTE = np.array([
    [139, 90, 43],    # 0 Ground  — brown
    [34, 160, 34],    # 1 Trees   — green
    [200, 60, 60],    # 2 Building— red
    [40, 110, 220],   # 3 Water   — blue
    [235, 225, 60],   # 4 Bridge  — yellow
], dtype=np.uint8)
ASPRS = [2, 5, 6, 9, 17]                 # CLS.txt code for each class index
UNLABELED = np.array([110, 110, 110], dtype=np.uint8)   # ASPRS 0 / unmapped -> grey

IEEE_ROOT = r"C:\Users\OrionHoch\Desktop\LabeledDatasets\Aerial\Non-Color\IEEE"


def colors_for(codes: np.ndarray) -> np.ndarray:
    lut = np.full(256, -1, np.int64)
    for i, a in enumerate(ASPRS):
        lut[a] = i
    idx = lut[np.clip(codes, 0, 255)]
    out = np.tile(UNLABELED, (len(idx), 1))
    out[idx >= 0] = IEEE_PALETTE[idx[idx >= 0]]
    return out


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    arr = np.column_stack([xyz.astype(np.float32), rgb.astype(np.int32)])
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header")
    np.savetxt(path, arr, fmt=["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"],
               header=header, comments="")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pc", default=os.path.join(IEEE_ROOT, "Validate-Track4", "Track4"),
                    help="folder of <scene>_PC3.txt point clouds")
    ap.add_argument("--cls", default=os.path.join(IEEE_ROOT, "Validate-Track4-Truth"),
                    help="folder of <scene>_CLS.txt label files")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "groundtruth"),
                    help="output folder for the coloured PLYs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pcs = sorted(glob.glob(os.path.join(args.pc, "*_PC3.txt")))
    if not pcs:
        raise SystemExit(f"no *_PC3.txt files in {args.pc}")
    print(f"{len(pcs)} scene(s)  {args.pc} + {args.cls}  ->  {args.out}")

    written = 0
    for pc_path in pcs:
        name = os.path.basename(pc_path).replace("_PC3.txt", "")
        cls_path = os.path.join(args.cls, f"{name}_CLS.txt")
        if not os.path.exists(cls_path):
            print(f"  skip {name}: no {name}_CLS.txt in {args.cls}")
            continue
        xyz = np.loadtxt(pc_path, delimiter=",", usecols=(0, 1, 2))
        codes = np.loadtxt(cls_path).astype(int).reshape(-1)
        n = min(len(xyz), len(codes))
        if n != len(xyz) or n != len(codes):
            print(f"  {name}: count mismatch (pc {len(xyz):,}, cls {len(codes):,}) — using {n:,}")
        write_ply(os.path.join(args.out, f"{name}_gt.ply"), xyz[:n], colors_for(codes[:n]))
        written += 1
        print(f"  {name}: {n:,} pts -> {name}_gt.ply")
    print(f"done — {written} PLY(s) in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
