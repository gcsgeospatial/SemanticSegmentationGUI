"""
KPConvX-L (cold) + HAG entry point — thin wrapper since the 2026-07-09 merge.

The 1,200-line copy-paste twin is gone: the HAG feature channel now lives in
local_train_kpconvx_cold.py behind --hag (one trainer, one drift surface). This
file remains only because the launch contract is by filename — the GUI
(local_cli) and the modal shell run local_train_<backbone.key>.py — so it
forces --hag / hag=True and delegates everything to the base script.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_train_kpconvx_cold as _base


def train_kpconvx_hag(*args, **kw):
    kw["hag"] = True
    return _base.train_kpconvx(*args, **kw)


def main():
    sys.argv.insert(1, "--hag")
    _base.main()


if __name__ == "__main__":
    main()
