"""Guard: every backbone's grid default must sit inside its own grid_clamp band.

The Infer page seeds its grid widget from this default when a run.json is absent,
so a default below the clamp floor silently runs inference far finer than the
model was trained at.  Run: python tests/test_fix_H_backbones.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer_gui.backbones import BACKBONES

for key, b in BACKBONES.items():
    gp = next((p for p in b.params if p.recommend_key == "grid"), None)
    assert gp is not None, f"{key}: no grid param"
    lo, hi = b.grid_clamp
    assert lo <= gp.default <= hi, \
        f"{key}: grid default {gp.default} outside clamp {b.grid_clamp}"

print("ok")
