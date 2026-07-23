"""kp-trainer LR-schedule self-checks (run: python tests/test_fix_b_kp_trainers.py).

The real schedules are closures inside train_*() (needs torch+CUDA), so the
formulas are mirrored here against the modules' constants.
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


# scale 1.0 keeps the stock 30/5/120 schedule
assert (max(1, round(kpx.CYC_RAISE * 1.0)), round(kpx.CYC_PLATEAU * 1.0),
        kpx.CYC_DECREASE10 * 1.0) == (30, 5, 120.0)
# both run lengths reach the peak lr
for n in (2, 100):
    assert max(kpx_lr(e, n) for e in range(n)) == kpx.CYC_LR1, n

# KPConv: per-epoch decay = exactly one /10 over the run
assert 0.1 ** (1.0 / kp.N_EPOCHS) == kp.LR_DECAY
for n in (2, 150):
    assert abs(kp.SGD_LR0 * (0.1 ** (1.0 / n)) ** n - kp.SGD_LR0 * 0.1) < 1e-12

# _cache_signature is a closure, so read the source for the class-layout keys
import inspect
for m in (kp, kpx):
    src = inspect.getsource(m)
    body = src[src.index("def _cache_signature"):]
    body = body[:body.index("\n    def ", 1)]     # to the next sibling closure
    for key in ('"num_classes"', '"class_names"'):
        assert key in body, f"{m.__name__}: {key} missing from _cache_signature"

print("ok")
