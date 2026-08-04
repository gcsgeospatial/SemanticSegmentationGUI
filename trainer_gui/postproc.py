"""Semantic post-processing over an inference predictions folder: KNN
soft-vote smoothing (RangeNet++-style), connected-component sieve (minimum
mapping unit), and confidence-gated HAG/geometry rules for the classic ALS
confusion pairs (lasclassify lineage).

Every run recomputes from the immutable originals (probs +
classification_orig/confidence_orig, stored on first run), so re-running with
different knobs is deterministic and reversible; only classification and
confidence are rewritten (plus any stale 'instance' key is dropped - ids from
a previous clustering no longer match relabeled points). Same atomic-rewrite
contract as panoptic."""

from __future__ import annotations

import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# HAG bands / geometry thresholds for the rule pass (metres, unitless ratios)
RULE_GROUND_HAG_M = 0.1     # low-veg this close to the DTM is ground
RULE_LOWVEG_HAG_M = 0.35    # ground this far above the DTM is low vegetation
RULE_HIGH_HAG_M = 2.0       # lasclassify's elevated-object threshold
RULE_PLANAR_MIN = 0.55      # planar enough to be roof/building
RULE_SCATTER_MIN = 0.4      # scattered enough to be vegetation

_KNN_CHUNK = 200_000        # (chunk, k+1, C) float32 gather stays sub-GB
_SIEVE_GRAPH_K = 8          # bounded-degree adjacency: a full radius pair
                            # list is O(n*density*r^2) and OOMs on a dominant
                            # ALS class; 8-NN chains connect dense blobs fine


def replace_retry(tmp, dst, attempts=6):
    """os.replace with a short backoff - Windows denies the swap while a
    reader (plot page, external viewer) briefly holds the destination open."""
    import time
    for i in range(attempts):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.25 * (i + 1))


def _knn_vote(xyz, probs, labels, conf, k, radius):
    """Gaussian-weighted neighbor soft vote -> (labels, conf). Small k and the
    hard radius cutoff are load-bearing: large neighborhoods bleed labels
    across interleaved-class interfaces instead of cleaning them. Points whose
    whole neighborhood carries zero probability mass (ensemble-uncovered or
    backfilled regions) keep their incoming label/conf."""
    n = len(xyz)
    p32 = np.asarray(probs, np.float32)
    tree = cKDTree(xyz)
    sigma = radius / 2.0
    out_lab = np.asarray(labels, np.int64).copy()
    out_conf = np.asarray(conf, np.float32).copy()
    for s in range(0, n, _KNN_CHUNK):
        d, nn = tree.query(xyz[s:s + _KNN_CHUNK], k=k + 1,
                           distance_upper_bound=radius, workers=-1)
        valid = np.isfinite(d)
        w = np.where(valid, np.exp(-0.5 * (d / sigma) ** 2), 0.0).astype(np.float32)
        nn = np.where(valid, nn, 0)
        votes = (w[:, :, None] * p32[nn]).sum(1)
        mass = votes.sum(1)
        has = mass > 0
        votes[has] /= mass[has, None]
        rows = np.flatnonzero(has) + s
        out_lab[rows] = votes[has].argmax(1)
        out_conf[rows] = votes[has].max(1)
    return out_lab, out_conf


def _sieve(xyz, labels, radius, min_pts, num_classes):
    """Dissolve same-class connected components under min_pts into the
    majority label of their non-dissolved neighbors. Islands with no such
    neighbor in range keep their label. Returns (labels, changed count)."""
    small = np.zeros(len(xyz), bool)
    for c in np.unique(labels):
        idxc = np.flatnonzero(labels == c)
        if len(idxc) < 2:
            small[idxc] = len(idxc) < min_pts
            continue
        pts = xyz[idxc]
        tree_c = cKDTree(pts)
        kq = min(_SIEVE_GRAPH_K + 1, len(idxc))
        rows, cols = [], []
        for s in range(0, len(idxc), _KNN_CHUNK):
            d, nn = tree_c.query(pts[s:s + _KNN_CHUNK], k=kq,
                                 distance_upper_bound=radius, workers=-1)
            v = np.isfinite(d)
            r = np.repeat(np.arange(s, s + len(d)), kq).reshape(d.shape)
            rows.append(r[v])
            cols.append(nn[v])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        m = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)),
                       shape=(len(idxc), len(idxc)))
        _, comp = connected_components(m, directed=False)
        sizes = np.bincount(comp)
        small[idxc[sizes[comp] < min_pts]] = True
    sm = np.flatnonzero(small)
    if not len(sm):
        return labels, 0
    # 16 neighbors within 4 sieve-radii: wide enough to reach past a dissolved
    # blob to real surroundings, small enough to stay local
    k = 16
    d, nn = cKDTree(xyz).query(xyz[sm], k=k,
                               distance_upper_bound=4.0 * radius, workers=-1)
    valid = np.isfinite(d)
    nn = np.where(valid, nn, 0)
    ok = valid & ~small[nn]          # only non-dissolved points get a vote
    votes = np.zeros((len(sm), num_classes), np.int32)
    lab_nn = labels[nn]
    for j in range(k):
        vj = ok[:, j]
        np.add.at(votes, (np.flatnonzero(vj), lab_nn[vj, j]), 1)
    has = votes.sum(1) > 0
    new = labels.copy()
    new[sm[has]] = votes[has].argmax(1)
    return new, int((new != labels).sum())


def _rules(labels, conf, data, roles, gate, say):
    """Confidence-gated confusion-pair rules; each rule only fires when its
    class roles resolve and its features exist. Channels come from the merged
    npz itself (the one-file layout). The HAG gate runs before any geometry is
    computed - pgeof over a full scene is minutes, feat_hag is a key read.
    Returns (labels, changed)."""
    veg, bld = roles.get("veg"), roles.get("building")
    gnd, lveg = roles.get("ground"), roles.get("low_veg")
    if "feat_hag" not in data:
        say("    rules: no feat_hag in the npz - stage with HAG enabled to "
            "unlock the HAG rules (ensemble outputs carry no channels)")
        return labels, 0
    hag = np.asarray(data["feat_hag"], np.float32)
    pl = data.get("feat_geo_planarity")
    sc = data.get("feat_geo_scattering")
    need_geo = veg is not None and bld is not None and (pl is None or sc is None)
    xyz = np.asarray(data["xyz"], np.float64) if need_geo else None
    if xyz is not None:
        from . import pretrain
        say(f"    computing planarity/scattering (pgeof) for {len(xyz):,} pts…")
        geo = pretrain.geo_features_for_cloud(xyz, ("planarity", "scattering"),
                                              geo_k=30)
        pl = pl if pl is not None else geo["planarity"]
        sc = sc if sc is not None else geo["scattering"]
    new = labels.copy()
    low = conf < gate
    if veg is not None and bld is not None:
        # planar AND scattered points end as vegetation - scattering wins,
        # matching the lasclassify precedence (buildings must not be rugged)
        m = low & (new == veg) & (hag >= RULE_HIGH_HAG_M) & (pl >= RULE_PLANAR_MIN)
        new[m] = bld
        m = low & (new == bld) & (hag >= RULE_HIGH_HAG_M) & (sc >= RULE_SCATTER_MIN)
        new[m] = veg
    if gnd is not None and lveg is not None:
        m = low & (new == lveg) & (hag <= RULE_GROUND_HAG_M)
        new[m] = gnd
        m = low & (new == gnd) & (hag >= RULE_LOWVEG_HAG_M) & (hag < RULE_HIGH_HAG_M)
        new[m] = lveg
    return new, int((new != labels).sum())


def run_postproc(pred_dir, knn=None, sieve=None, rules=None,
                 progress=None) -> dict:
    """Apply the enabled passes to every inferred npz in pred_dir, in the
    fixed order KNN vote -> sieve -> rules. knn={"k","radius"},
    sieve={"radius","min_pts"}, rules={"gate","roles":{role: index|None}}.
    With no passes enabled this still restores classification/confidence from
    the stored originals (and drops stale instance ids) - the 'reset' path.
    Atomic per-file rewrite; job.json's "postproc" section records settings +
    change counts."""
    from .dataset import pred_files, update_job_json
    say = progress or (lambda s: None)
    files = []
    for f in pred_files(pred_dir):
        with np.load(f) as d:
            if "classification" in d.files:
                files.append(f)
    if not files:
        raise RuntimeError(f"no inferred npz files in {pred_dir} - run "
                           "Inference on this folder first, then come back")
    stats = []
    for f in files:
        with np.load(f) as d:
            data = {key: d[key] for key in d.files}
        xyz = np.asarray(data["xyz"], np.float64)
        if "classification_orig" not in data:
            data["classification_orig"] = data["classification"].copy()
            data["confidence_orig"] = data.get(
                "confidence", np.zeros(len(xyz), np.float32)).copy()
        labels = np.asarray(data["classification_orig"], np.int64).copy()
        conf = np.asarray(data["confidence_orig"], np.float32).copy()
        probs = data.get("probs")
        num_classes = (probs.shape[1] if probs is not None
                       else int(labels.max(initial=0)) + 1)
        st = {"scene": f.name, "knn": 0, "sieve": 0, "rules": 0}
        if knn:
            if probs is None:
                say(f"  {f.name}: KNN vote skipped - no probs in the npz; "
                    "enable 'Save class probabilities' and re-run inference")
            else:
                say(f"  {f.name}: KNN soft vote (k={knn['k']}, "
                    f"r={knn['radius']:g} m)…")
                new, conf = _knn_vote(xyz, probs, labels, conf,
                                      int(knn["k"]), float(knn["radius"]))
                st["knn"] = int((new != labels).sum())
                labels = new
        if sieve:
            say(f"  {f.name}: sieve (r={sieve['radius']:g} m, "
                f"min {sieve['min_pts']} pts)…")
            labels, st["sieve"] = _sieve(xyz, labels, float(sieve["radius"]),
                                         int(sieve["min_pts"]), num_classes)
        if rules:
            say(f"  {f.name}: geometry rules (gate={rules['gate']:g})…")
            labels, st["rules"] = _rules(labels, conf, data, rules["roles"],
                                         float(rules["gate"]), say)
        data["classification"] = labels.astype(np.int32)
        data["confidence"] = conf.astype(np.float32)
        data.pop("instance", None)   # stale vs the rewritten labels
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, **data)
        replace_retry(tmp, f)
        say(f"  {f.name}: changed knn={st['knn']:,} sieve={st['sieve']:,} "
            f"rules={st['rules']:,} of {len(xyz):,} pts")
        stats.append(st)
    update_job_json(pred_dir, "postproc", {
        "knn": knn, "sieve": sieve,
        "rules": ({"gate": rules["gate"], "roles": rules["roles"]}
                  if rules else None),
        "scenes": stats})
    return {"files": len(files),
            "changed": int(sum(s["knn"] + s["sieve"] + s["rules"]
                               for s in stats))}
