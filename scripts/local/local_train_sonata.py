"""Sonata entry point — thin wrapper over local_train_concerto.py (the launch
contract is by filename); only swaps the package/HF constants."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_train_concerto as _base

# applied per CALL, not at import — several wrappers may share one process
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
