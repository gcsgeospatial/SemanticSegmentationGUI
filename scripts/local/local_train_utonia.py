"""
Utonia entry point — thin wrapper over the shared pcssl fine-tuner.

Utonia (ICML'26, Pointcept/Utonia, "Toward One Encoder for All Point Clouds")
is a cross-domain self-supervised pretrained encoder-only PTv3 (with 3D RoPE
inside attention), API-identical to Concerto/Sonata; the whole fine-tune/infer
pipeline lives in local_train_concerto.py. This file exists because the launch
contract is by filename — the GUI (local_cli) and the modal shell run
local_train_<backbone.key>.py — so it only swaps the package / HuggingFace
constants and delegates.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_train_concerto as _base

# Applied per CALL, not at import — several entry points may be imported into
# one process (the smoke test does), and an import-time overwrite would leave
# the shared core stuck on whichever wrapper imported last.
_CFG = dict(PKG="utonia", HF_NAME="utonia",
            HF_REPO="Pointcept/Utonia", BB_KEY="utonia")


def train_pcssl(*args, **kw):
    _base.__dict__.update(_CFG)
    return _base.train_pcssl(*args, **kw)


def main():
    _base.__dict__.update(_CFG)
    _base.main()


if __name__ == "__main__":
    main()
