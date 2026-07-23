"""Density-domain-generalization primitives for the local_train_* scripts.
Occupancy o = rho*g^2; o < 1 breaks density invariance, and coarsening is
one-way (thin dense, never densify sparse). `python density.py` self-checks."""
import os

import numpy as np

__all__ = [
    "effective_grid", "voxel_first_idx",
    "local_density_logdk", "adabn_recalibrate",
    "env_bool", "env_float", "env_int", "env_str",
]


# DG_* env overrides (set by the GUI panel or directly), never CLI args.
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


# D1 — density/grid jitter: per-tile effective grid g_eff >= g0.
def effective_grid(g0, coarsen_max=2.5, p_native=0.5, rng=None):
    """g0 with prob p_native, else log-uniform in [g0, g0*coarsen_max]
    (coarsen_max = 1/(g0*sqrt(rho_min)) reaches output density rho_min)."""
    rng = rng or np.random.default_rng()
    if coarsen_max <= 1.0 or rng.random() < p_native:
        return float(g0)
    return float(g0) * float(np.exp(rng.uniform(0.0, np.log(coarsen_max))))


# D0/D0b — canonicalize to a grid: first point per g-cell.
def voxel_first_idx(xyz, g):
    """Indices of the first point per g-voxel; slice every per-point companion
    array by them too."""
    keys = np.floor(np.asarray(xyz)[:, :3] / float(g)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)            # preserve original point order


# D3b — local-density input channel: log d_k ~ -0.5 log rho (pair with D1).
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


# D2b — AdaBN: re-estimate BN running stats on the unlabeled target.
def adabn_recalibrate(model, batches, forward, momentum=None, reset=True):
    """Refresh BN running mean/var over target `batches` via forward(model, b).
    momentum None = cumulative (PreciseBN); float = exponential. reset zeroes
    stats first (pure AdaBN). Leaves model.eval(); returns the model."""
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


def _demo():
    rng = np.random.default_rng(0)

    # effective_grid: native anchor returns g0; jitter stays in [g0, g0*max].
    g0 = 0.8
    assert effective_grid(g0, coarsen_max=1.0, rng=rng) == g0           # no jitter
    gs = [effective_grid(g0, 2.5, p_native=0.0, rng=rng) for _ in range(2000)]
    assert all(g0 <= g <= g0 * 2.5 + 1e-9 for g in gs)
    assert max(gs) > g0 * 2.0                                            # range exercised

    # voxel_first_idx: dense cloud canonicalizes to ~1 pt/cell
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
