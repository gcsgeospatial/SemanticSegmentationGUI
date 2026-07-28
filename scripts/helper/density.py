"""Density-domain-generalization primitives for the local_train_* scripts.
Occupancy o = rho*g^2; o < 1 breaks density invariance, and coarsening is
one-way (thin dense, never densify sparse)."""
import os

import numpy as np

__all__ = [
    "effective_grid", "voxel_first_idx",
    "local_density_logdk", "adabn_recalibrate", "apcotta_adapt",
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


# D1 - density/grid jitter: per-tile effective grid g_eff >= g0.
def effective_grid(g0, coarsen_max=2.5, p_native=0.5, rng=None):
    """g0 with prob p_native, else log-uniform in [g0, g0*coarsen_max]
    (coarsen_max = 1/(g0*sqrt(rho_min)) reaches output density rho_min)."""
    rng = rng or np.random.default_rng()
    if coarsen_max <= 1.0 or rng.random() < p_native:
        return float(g0)
    return float(g0) * float(np.exp(rng.uniform(0.0, np.log(coarsen_max))))


# D0/D0b - canonicalize to a grid: first point per g-cell.
def voxel_first_idx(xyz, g):
    """Indices of the first point per g-voxel; slice every per-point companion
    array by them too."""
    keys = np.floor(np.asarray(xyz)[:, :3] / float(g)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)


# D3b - local-density input channel: log d_k ~ -0.5 log rho (pair with D1).
def local_density_logdk(xyz, k=8):
    """Per-point log distance to the k-th nearest neighbour (natural log).
    Larger = sparser. Returns float32 array, shape (N,)."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz)[:, :3]
    n = len(xyz)
    if n <= 1:
        return np.zeros(n, np.float32)
    kk = min(k, n - 1)
    d, _ = cKDTree(xyz).query(xyz, k=kk + 1)
    dk = d[:, -1]
    return np.log(np.maximum(dk, 1e-6)).astype(np.float32)


# D2b - AdaBN: re-estimate BN running stats on the unlabeled target.
def adabn_recalibrate(model, batches, forward, momentum=None, reset=True):
    """Refresh BN running mean/var over target `batches` via forward(model, b).
    momentum None = cumulative (PreciseBN); float = exponential, and reset zeroes
    stats first (pure AdaBN). Leaves model.eval(); returns the model."""
    import torch
    import torch.nn as nn
    bns = [m for m in model.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                             nn.SyncBatchNorm))]
    saved = []
    for bn in bns:
        # stats snapshot: zero batches must not leave zeroed running stats
        stats = ((bn.running_mean.clone(), bn.running_var.clone(),
                  bn.num_batches_tracked.clone())
                 if bn.track_running_stats else None)
        saved.append((bn.training, bn.momentum, stats))
        if reset and bn.track_running_stats:
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
            bn.num_batches_tracked.zero_()
        bn.momentum = momentum
        bn.train()
    n = 0
    with torch.no_grad():
        for batch in batches:
            forward(model, batch)
            n += 1
    for bn, (was_training, mom, stats) in zip(bns, saved):
        bn.momentum = mom
        bn.train(was_training)
        if n == 0 and stats is not None:
            bn.running_mean.copy_(stats[0])
            bn.running_var.copy_(stats[1])
            bn.num_batches_tracked.copy_(stats[2])
    model.eval()
    if n == 0:
        raise RuntimeError(
            "AdaBN got zero target batches (every chunk was under the minimum "
            "point count); rerun with adaptation Off or infer on larger scenes")
    return model


# D2c - APCoTTA (arXiv:2505.09971) reduced to the offline one-shot job shape:
# AdaBN stats refresh + entropy-filtered entropy minimization on BN affine
# params + stochastic restore toward source weights (their RPI).
def apcotta_adapt(model, batches, logits_fn, lr=1e-3, ent_frac=0.4,
                  restore_p=0.01):
    """Adapt over target `batches` via logits_fn(model, batch) -> (N, C).
    Only points with entropy < ent_frac*ln(C) contribute loss (official code
    uses 0.8 nats absolute ~= 0.36*ln C at their 9 classes; 0.4 scales with C);
    each step restores every trainable element to its source value with prob
    restore_p. Strict superset of adabn_recalibrate. Leaves model.eval()."""
    # ponytail: BN-affine-only updates, cumulative running stats (paper keeps
    # per-batch stats), no gradient layer selection or weak/strong consistency
    # views; add the full recipe if a survey still underperforms plain AdaBN.
    import torch
    import torch.nn as nn
    bns = [m for m in model.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                             nn.SyncBatchNorm))]
    params = [p for bn in bns for p in (bn.weight, bn.bias) if p is not None]
    if not params:
        raise RuntimeError(
            "APCoTTA needs BatchNorm layers with affine params; this model has "
            "none - use plain inference (or AdaBN on a BN-based backbone)")
    source = [p.detach().clone() for p in params]
    grad_state = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in grad_state:
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    saved = []
    for bn in bns:
        # stats snapshot: zero batches must not leave zeroed running stats
        stats = ((bn.running_mean.clone(), bn.running_var.clone(),
                  bn.num_batches_tracked.clone())
                 if bn.track_running_stats else None)
        saved.append((bn.training, bn.momentum, stats))
        if bn.track_running_stats:
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)
            bn.num_batches_tracked.zero_()
        bn.momentum = None
        bn.train()
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    n_step = n_skip = 0
    with torch.enable_grad():
        for batch in batches:
            logits = logits_fn(model, batch).float()
            logp = torch.log_softmax(logits, -1)
            ent = -(logp.exp() * logp).sum(-1)
            keep = ent < ent_frac * float(np.log(logits.shape[-1]))
            if not keep.any():
                n_skip += 1
                continue
            opt.zero_grad()
            ent[keep].mean().backward()
            opt.step()
            with torch.no_grad():
                for p, s in zip(params, source):
                    m = torch.rand_like(p) < restore_p
                    p[m] = s[m]
            n_step += 1
    for p, req in grad_state:
        p.requires_grad_(req)
    for bn, (was_training, mom, stats) in zip(bns, saved):
        bn.momentum = mom
        bn.train(was_training)
        if n_step + n_skip == 0 and stats is not None:
            bn.running_mean.copy_(stats[0])
            bn.running_var.copy_(stats[1])
            bn.num_batches_tracked.copy_(stats[2])
    model.eval()
    if n_step + n_skip == 0:
        raise RuntimeError(
            "APCoTTA got zero target batches (every chunk was under the minimum "
            "point count); rerun with adaptation Off or infer on larger scenes")
    print(f"  [infer] APCoTTA: {n_step} adaptation step(s), "
          f"{n_skip} batch(es) skipped as all-high-entropy", flush=True)
    return model
