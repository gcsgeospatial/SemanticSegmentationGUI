"""Class-index -> RGB palettes shared by the viewer and the training scripts' PLY output."""

from __future__ import annotations

import numpy as np

# Matches the PALETTE arrays in the modal_train_* scripts (IEEE 5-class).
IEEE_PALETTE = np.array([
    [139, 90, 43],    # Ground — brown
    [34, 160, 34],    # Trees — green
    [200, 60, 60],    # Building — red
    [40, 110, 220],   # Water — blue
    [235, 225, 60],   # Bridge — yellow
], dtype=np.uint8)

# Generic categorical palette for arbitrary class counts (first 5 = IEEE colors
# so IEEE-trained runs look identical everywhere).
_EXTENDED = np.array([
    [139, 90, 43], [34, 160, 34], [200, 60, 60], [40, 110, 220], [235, 225, 60],
    [150, 80, 200], [240, 140, 40], [70, 200, 200], [220, 100, 170], [120, 120, 120],
    [90, 140, 60], [180, 180, 90], [60, 60, 160], [200, 170, 130], [100, 220, 120],
    [230, 70, 110], [50, 160, 110], [170, 110, 60], [110, 170, 230], [240, 200, 160],
], dtype=np.uint8)


def palette_for(num_classes: int) -> np.ndarray:
    """(num_classes, 3) uint8 colors; cycles if more classes than base colors."""
    reps = -(-num_classes // len(_EXTENDED))
    return np.tile(_EXTENDED, (reps, 1))[:num_classes]


# IEEE GRSS 2019 Track 4: contiguous class index <-> ASPRS LAS code (the values
# in the dataset's *_CLS.txt ground-truth files). Index i has color IEEE_PALETTE[i].
IEEE_CLASS_NAMES = ["Ground", "Trees", "Building", "Water", "Bridge"]
IEEE_ASPRS = [2, 5, 6, 9, 17]


def class_from_rgb(rgb: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Recover class indices from a class-coloured PLY's colors (-1 where a point's
    color isn't an exact palette match). The training scripts colour each point
    palette_for(num_classes)[class]; matching against the FULL categorical palette
    (the default) losslessly inverts that for any class count up to len(_EXTENDED),
    so >5-class predictions decode correctly (not all -1 like an IEEE-only match).
    Pass num_classes to restrict the match to exactly that many colours."""
    rgb = np.asarray(rgb)
    pal = _EXTENDED if not num_classes else palette_for(num_classes)
    out = np.full(len(rgb), -1, np.int64)
    for i, p in enumerate(pal):
        out[(rgb[:, 0] == p[0]) & (rgb[:, 1] == p[1]) & (rgb[:, 2] == p[2])] = i
    return out


def asprs_to_index(asprs: np.ndarray) -> np.ndarray:
    """(N,) ASPRS CLS codes -> contiguous IEEE class index (-1 for 0/unlabeled)."""
    lut = np.full(256, -1, np.int64)
    for i, a in enumerate(IEEE_ASPRS):
        lut[a] = i
    return lut[np.clip(np.asarray(asprs, np.int64), 0, 255)]
