"""
Sonata entry point — thin wrapper over the shared pcssl fine-tuner.

Sonata (CVPR'25 Highlight, facebookresearch/sonata) is a self-supervised
pretrained encoder-only PTv3, API-identical to Concerto/Utonia; the whole
fine-tune/infer pipeline lives in local_train_concerto.py. This file exists
because the launch contract is by filename — the GUI (local_cli) and the
modal shell run local_train_<backbone.key>.py — so it only swaps the
package / HuggingFace constants and delegates.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_train_concerto as _base

# Applied per CALL, not at import — several entry points may be imported into
# one process (the smoke test does), and an import-time overwrite would leave
# the shared core stuck on whichever wrapper imported last.
_CFG = dict(PKG="sonata", HF_NAME="sonata",
            HF_REPO="facebook/sonata", BB_KEY="sonata")


def train_pcssl(*args, **kw):
    _base.__dict__.update(_CFG)
    return _base.train_pcssl(*args, **kw)


def main():
    _base.__dict__.update(_CFG)
    _base.main()


if __name__ == "__main__":
    main()
