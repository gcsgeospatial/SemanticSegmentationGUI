"""Class-index -> RGB palettes shared by the viewer and the training scripts' PLY output."""

from __future__ import annotations

import numpy as np

# Generic categorical palette for arbitrary class counts. It's stable across runs,
# so a given class index always paints the same colour everywhere (viewer legend,
# the training scripts' prediction PLYs, the GUI swatches).
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


def generic_names(num_classes: int) -> list[str]:
    """Generic class labels ('class 0', 'class 1', …) for clouds/runs/datasets that
    carry no class names of their own."""
    return [f"class {i}" for i in range(int(num_classes))]


def class_from_rgb(rgb: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Recover class indices from a class-coloured PLY's colors (-1 where a point's
    color isn't an exact palette match). The training scripts colour each point
    palette_for(num_classes)[class]; matching against the FULL categorical palette
    (the default) losslessly inverts that for any class count up to len(_EXTENDED),
    so multi-class predictions decode correctly. Pass num_classes to restrict the
    match to exactly that many colours."""
    rgb = np.asarray(rgb)
    pal = _EXTENDED if not num_classes else palette_for(num_classes)
    out = np.full(len(rgb), -1, np.int64)
    for i, p in enumerate(pal):
        out[(rgb[:, 0] == p[0]) & (rgb[:, 1] == p[1]) & (rgb[:, 2] == p[2])] = i
    return out
