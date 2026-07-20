"""Self-checks for the B-kp-trainers fixes (run: python tests/test_fix_b_kp_trainers.py).

The schedules live as closures inside train_*(), which needs torch+CUDA, so the
formulas are mirrored here against the modules' own constants: the point is that
the DEFAULT run is unchanged while short runs still traverse the curve.
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts", "local"))
import local_train_kpconv as kp
import local_train_kpconvx_cold as kpx


def kpx_lr(ep, n_epochs):
    s = n_epochs / 100.0
    raise_, plateau = max(1, round(kpx.CYC_RAISE * s)), round(kpx.CYC_PLATEAU * s)
    dec10 = kpx.CYC_DECREASE10 * s
    if ep < raise_:
        return kpx.CYC_LR0 * (kpx.CYC_LR1 / kpx.CYC_LR0) ** (ep / raise_)
    if ep < raise_ + plateau:
        return kpx.CYC_LR1
    return kpx.CYC_LR1 * 0.1 ** ((ep - raise_ - plateau) / dec10)


# default 100-epoch run is untouched (scale == 1.0 hits the quoted 30/5/120)
assert (max(1, round(kpx.CYC_RAISE * 1.0)), round(kpx.CYC_PLATEAU * 1.0),
        kpx.CYC_DECREASE10 * 1.0) == (30, 5, 120.0)
# ...and both run lengths actually reach the peak lr
for n in (2, 100):
    assert max(kpx_lr(e, n) for e in range(n)) == kpx.CYC_LR1, n

# KPConv: per-epoch decay completes exactly one /10 over the run, and the
# 150-epoch default reproduces the module constant bit-for-bit.
assert 0.1 ** (1.0 / kp.N_EPOCHS) == kp.LR_DECAY
for n in (2, 150):
    assert abs(kp.SGD_LR0 * (0.1 ** (1.0 / n)) ** n - kp.SGD_LR0 * 0.1) < 1e-12

# The cache-signature fix is covered by asserting the modules actually name the
# class-layout keys — comparing two literal dicts here would only test that dict
# inequality works. _cache_signature is a closure inside train_*(), so this reads
# the source rather than calling it.
import inspect
for m in (kp, kpx):
    src = inspect.getsource(m)
    body = src[src.index("def _cache_signature"):]
    body = body[:body.index("\n    def ", 1)]     # to the next sibling closure
    for key in ('"num_classes"', '"class_names"'):
        assert key in body, f"{m.__name__}: {key} missing from _cache_signature"

print("ok")
