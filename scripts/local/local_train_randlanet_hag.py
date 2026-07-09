"""
RandLA-Net + HAG entry point — thin wrapper since the 2026-07-09 merge.

The 1,100-line copy-paste twin is gone: the HAG feature channel now lives in
local_train_randlanet.py behind --hag (one trainer, one drift surface). This
file remains only because the launch contract is by filename — the GUI
(local_cli) and the modal shell run local_train_<backbone.key>.py — so it
forces --hag / hag=True and delegates everything to the base script.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_train_randlanet as _base


def train_randlanet_hag(*args, **kw):
    kw["hag"] = True
    return _base.train_randlanet(*args, **kw)


def main():
    sys.argv.insert(1, "--hag")
    _base.main()


if __name__ == "__main__":
    main()
