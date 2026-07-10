"""Density-domain-generalization primitives, shared by every local_train_* script
(and, when bundled, the modal_train_* wrappers).

The whole problem reduces to one scalar: occupancy  o = rho * g^2  (mean points per
voxel cell of size g, for areal density rho). A voxel subsample keeps <=1 point/cell,
so it CAPS density at 1/g^2. Consequence:
  * o >= 1  -> every cell filled, the subsampled cloud is ~a function of the surface,
              not of rho -> the backbone is density-invariant for free.
  * o <  1  -> cells go empty (Poisson holes), invariance breaks. This is the whole gap.
The valve is one-way: you can always thin a dense cloud DOWN to g0 (safe), you can
never invent the points a sparse cloud never captured (ill-posed). So everything here
is built around the SPARSEST density you must serve.

These are pure-numpy/scipy except `adabn_recalibrate` (torch). Run `python density.py`
for the self-checks.
"""
import os

import numpy as np

__all__ = [
    "effective_grid", "voxel_first_idx",
    "local_density_logdk", "adabn_recalibrate",
    "env_bool", "env_float", "env_int", "env_str",
]


# --------------------------------------------------------------------------- #
# Env-var overrides. The GUI "Density generalization (advanced)" panel saves a
# per-dataset DG config and passes it to the run as DG_* env vars; each script
# reads them in its config block via, e.g.
#     DG_DENSITY_AUG = dg.env_bool("DG_DENSITY_AUG", globals()["DG_DENSITY_AUG"])
# so every flag is controllable from the GUI (or a direct `DG_*=1 python ...`)
# without becoming a CLI argument. globals()[...] avoids the assign-before-ref
# UnboundLocalError when the local shadows the module constant.
# --------------------------------------------------------------------------- #
def env_bool(name, default):
    v = os.environ.get(name)
    return bool(default) if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else float(default)


def env_int(name, default):
    v = os.environ.get(name)
    return int(float(v)) if v not in (None, "") else int(default)


def env_str(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


# --------------------------------------------------------------------------- #
# D1 — density / grid jitter: pick a per-tile effective grid g_eff >= g0.
# Coarsening g lowers output density (~1/g_eff^2), i.e. drives occupancy o<1 so
# the model is trained across the density range it will meet at inference.
# --------------------------------------------------------------------------- #
def effective_grid(g0, coarsen_max=2.5, p_native=0.5, rng=None):
    """Per-tile effective grid. Returns g0 with prob p_native (a full-occupancy
    anchor every batch), else log-uniform in [g0, g0*coarsen_max].

    coarsen_max ties to the sparsest density you must serve: to reach output
    density rho_min from a model grid g0, set coarsen_max = 1/(g0*sqrt(rho_min)).
    Coarsening only — you cannot densify a subsampled tile (the one-way valve).
    """
    rng = rng or np.random.default_rng()
    if coarsen_max <= 1.0 or rng.random() < p_native:
        return float(g0)
    return float(g0) * float(np.exp(rng.uniform(0.0, np.log(coarsen_max))))


# --------------------------------------------------------------------------- #
# D0 / D0b — canonicalize to a grid: first point per g-cell. Used both to
# resample a training tile to g_eff (D1) and to thin a too-dense inference
# cloud down to the model's trained g0 (D0b). 'first' (not barycenter) keeps a
# real measurement and lets companion arrays / labels slice by the same index.
# --------------------------------------------------------------------------- #
def voxel_first_idx(xyz, g):
    """Indices of the first point falling in each g-sized voxel. Slice xyz AND
    every per-point companion (labels, intensity, hag, ...) by these indices to
    get a density-canonicalized cloud at <=1 point per g-cell."""
    keys = np.floor(np.asarray(xyz)[:, :3] / float(g)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)            # preserve original point order


# --------------------------------------------------------------------------- #
# D3b — explicit local-density feature so the net learns a density-CONDITIONAL
# boundary instead of one entangled with density. log of the k-th NN distance is
# the cleanest scalar: d_k ~ rho^(-1/2), so log d_k ~ -0.5 log rho (a clean,
# bounded density coordinate). Feed it as an extra input channel (pair with D1
# augmentation, or it sees no variation to learn from).
# --------------------------------------------------------------------------- #
def local_density_logdk(xyz, k=8):
    """Per-point log distance to the k-th nearest neighbour (natural log).
    Larger = sparser. Returns float32 array, shape (N,)."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz)[:, :3]
    n = len(xyz)
    if n <= 1:
        return np.zeros(n, np.float32)
    kk = min(k, n - 1)
    d, _ = cKDTree(xyz).query(xyz, k=kk + 1)   # +1: self is the 0-distance hit
    dk = d[:, -1]
    return np.log(np.maximum(dk, 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------- #
# D2b — AdaBN: re-estimate BatchNorm running stats on the (unlabeled) target so
# the frozen source stats stop mis-normalizing at a different density. No labels,
# no backprop. In 3D one tile = millions of points, so this is reliable even at
# batch size 1. Single highest-ROI label-free patch.
# --------------------------------------------------------------------------- #
def adabn_recalibrate(model, batches, forward, momentum=None, reset=True):
    """Refresh BN running mean/var over target `batches`.

    model    : nn.Module (BatchNorm layers anywhere inside).
    batches  : iterable of inputs to feed `forward`.
    forward  : callable(model, batch) -> runs a forward pass (output ignored).
    momentum : BN momentum while accumulating. None (default) -> torch's
               momentum=None mode, a CUMULATIVE average over all batches — the
               PreciseBN estimator (Yan et al., "Rethinking 'Batch' in
               BatchNorm"), unbiased in ~30-50 batches. A float -> exponential
               average with that momentum (with reset=True, (1-m)^N of the
               zeroed init survives N batches — feed enough of them).
    reset    : if True, zero the running stats first so the estimate is purely
               target-driven (pure AdaBN); if False, the existing stats act as a
               source prior that the target updates ease into (source-prior mixing).

    Leaves the model in eval() with adapted stats. Returns the model.
    """
    import torch
    import torch.nn as nn
    bns = [m for m in model.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                             nn.SyncBatchNorm))]
    saved = []
    for bn in bns:
        saved.append((bn.training, bn.momentum))
        if reset and bn.track_running_stats:
            if bn.running_mean is not None:
                bn.running_mean.zero_()
            if bn.running_var is not None:
                bn.running_var.fill_(1.0)
            bn.num_batches_tracked.zero_()
        bn.momentum = momentum          # float -> exponential avg; None -> cumulative
        bn.train()                      # train mode == update running stats on forward
    with torch.no_grad():
        for batch in batches:
            forward(model, batch)
    for bn, (was_training, mom) in zip(bns, saved):
        bn.momentum = mom
        bn.train(was_training)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Self-checks: the smallest things that fail if the math is wrong.
# --------------------------------------------------------------------------- #
def _demo():
    rng = np.random.default_rng(0)

    # effective_grid: native anchor returns g0; jitter stays in [g0, g0*max].
    g0 = 0.8
    assert effective_grid(g0, coarsen_max=1.0, rng=rng) == g0           # no jitter
    gs = [effective_grid(g0, 2.5, p_native=0.0, rng=rng) for _ in range(2000)]
    assert all(g0 <= g <= g0 * 2.5 + 1e-9 for g in gs)
    assert max(gs) > g0 * 2.0                                            # range exercised

    # voxel_first_idx: canonicalizing a DENSE cloud to g0 lands output occupancy
    # ~1 (<=1 pt/cell) and lowers density; a denser input -> same output count.
    side = 20.0
    dense = rng.uniform(0, side, size=(40000, 3)); dense[:, 2] = 0.0     # ~100 pts/m^2
    idx = voxel_first_idx(dense, g0)
    cells = (side / g0) ** 2
    out_o = len(idx) / cells
    assert 0.6 < out_o <= 1.0, out_o                                     # ~1 pt per cell
    denser = rng.uniform(0, side, size=(120000, 3)); denser[:, 2] = 0.0
    assert abs(len(voxel_first_idx(denser, g0)) - len(idx)) < 0.1 * len(idx)  # cap: 3x density, ~same out

    # local_density_logdk: sparser cloud -> larger log d_k.
    sparse = rng.uniform(0, side, size=(2000, 3)); sparse[:, 2] = 0.0
    assert local_density_logdk(sparse).mean() > local_density_logdk(dense).mean()
    assert local_density_logdk(np.zeros((1, 3))).shape == (1,)          # degenerate ok

    # env overrides: default when unset; parsed when set.
    os.environ.pop("DG_TEST_X", None)
    assert env_bool("DG_TEST_X", True) is True and env_int("DG_TEST_X", 8) == 8
    os.environ["DG_TEST_X"] = "off"; assert env_bool("DG_TEST_X", True) is False
    os.environ["DG_TEST_X"] = "2.5"; assert env_float("DG_TEST_X", 1.0) == 2.5
    os.environ["DG_TEST_X"] = "3"; assert env_int("DG_TEST_X", 0) == 3
    os.environ.pop("DG_TEST_X", None)

    print("density.py self-checks passed")


if __name__ == "__main__":
    _demo()
