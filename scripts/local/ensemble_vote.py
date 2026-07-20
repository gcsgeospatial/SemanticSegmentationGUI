"""Cross-model ensemble over trainer_gui prediction files.

All backbones predict on the SAME raw input points and their output reaches
your folder as {scene}_pred.<fmt> (las/laz/ply/txt/csv after the GUI export, or
the container's raw {scene}_pred.npz), so ensembling is per-point, independent
of architecture. Ensembling 2-3 models is the contest-era trick that survived
into modern recipes (~+2 mIoU for 3 models on the 2024 Waymo winner).

Voting: when every matched scene npz carries the saved class distribution
("probs", written under TT_SAVE_PROBS=1), the distributions are AVERAGED and
argmax wins (soft vote — confidence = max of the average). Otherwise it falls
back to a confidence-weighted hard vote: each model's label weighted by its
per-point confidence (1.0 when absent), confidence = the winning weight share.
Every output also carries "agreement": the fraction of models whose own label
matches the final one. Class compatibility is clamped via each folder's
infer_run.json when present.

Scenes are matched by filename (minus extension) across the input folders.
Points are matched by row when the clouds are identical (the normal case); if a
model's cloud differs (e.g. it filtered points), its rows are carried onto the
first model's points by nearest neighbour. Hard-vote ties go to the earliest-
listed model — put your strongest model first. Output is written in the first
input's format.

Usage:
  python ensemble_vote.py --inputs predsA predsB [predsC ...] --out out_dir
  python ensemble_vote.py --self-test
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def soft_vote(probs, labels=None):
    """probs: (M, N, C) normalized distributions, one per model. Returns
    ((N,) argmax labels, (N,) f32 confidence = max of the averaged distribution).
    A model that never covered a point leaves an all-zero row there (predict_points
    NN-fills the label only), so average over the COVERING models, not over M —
    otherwise a lone member's 0.99 is diluted to 0.33 and the export threshold
    rewrites the point to Unclassified. Points no model covered have no
    distribution at all; they fall back to a plain vote over `labels`."""
    probs = np.asarray(probs, np.float32)
    cov = (probs.sum(-1) > 0).sum(0)                  # (N,) models that saw each point
    p = probs.sum(0) / np.maximum(cov, 1)[:, None]
    lab, conf = p.argmax(1), p.max(1).astype(np.float32)
    miss = cov == 0
    if labels is not None and miss.any():
        labels = np.asarray(labels)[:, miss]
        lab[miss], conf[miss] = weighted_vote(labels, np.ones(labels.shape, np.float32))
    return lab, conf


def weighted_vote(labels, weights):
    """labels: (M, N) int, weights: (M, N) per-point confidence (1.0 = a plain
    vote). Returns ((N,) labels, (N,) f32 confidence = winning weight share).
    Ties go to the earliest model whose label holds the max weight."""
    labels = np.asarray(labels)
    weights = np.asarray(weights, np.float32)
    m, n = labels.shape
    c = int(labels.max()) + 1
    counts = np.zeros((n, c), np.float32)
    idx = np.arange(n)
    for row, w in zip(labels, weights):
        counts[idx, row] += w
    mx = counts.max(1)
    out = np.full(n, -1, labels.dtype)
    for row in labels:                       # priority order = input order
        sel = (out < 0) & (counts[idx, row] == mx)
        out[sel] = row[sel]
    conf = mx / np.maximum(counts.sum(1), 1e-9)
    return out, conf.astype(np.float32)


def agreement(labels, final):
    """(N,) f32 fraction of models whose own label equals the final label."""
    return (np.asarray(labels) == final).mean(0).astype(np.float32)


def load_pred(path):
    """(xyz, labels, intensity|None, confidence|None, probs|None) from any
    prediction file. npz is read natively (container-safe); exported formats go
    through the GUI's readers (probs never survive an export — npz only)."""
    if path.endswith(".npz"):
        z = np.load(path)
        # xyz float64: preds carry georeferenced (UTM ~1e6) coords — a float32
        # cast quantizes the voted deliverable's northing to 0.5 m steps.
        return (z["xyz"].astype(np.float64), z["classification"].astype(np.int64),
                z["intensity"] if "intensity" in z.files else None,
                z["confidence"].astype(np.float32) if "confidence" in z.files else None,
                z["probs"].astype(np.float32) if "probs" in z.files else None)
    sys.path.insert(0, REPO_ROOT)
    from trainer_gui.readers import read_points
    c = read_points(path)
    lab = c.fields.get("classification")
    if lab is None and len(c.fields) == 1:   # single label-candidate field
        lab = next(iter(c.fields.values()))
    if lab is None:
        raise ValueError(f"{path}: no 'classification' field (has {list(c.fields)})")
    conf = c.fields.get("confidence")
    return (c.xyz.astype(np.float64), np.asarray(lab, np.int64), c.intensity,
            np.asarray(conf, np.float32) if conf is not None else None, None)


def write_out(path, xyz, cls, intensity, confidence, agree, crs_wkt=None):
    if path.endswith(".npz"):
        d = {"xyz": xyz, "classification": cls.astype(np.int32),
             "confidence": np.asarray(confidence, np.float32),
             "agreement": np.asarray(agree, np.float32)}
        if intensity is not None:
            d["intensity"] = intensity
        if crs_wkt:
            d["crs_wkt"] = np.asarray(str(crs_wkt))
        np.savez(path, **d)
        return
    # ponytail: agreement is npz-only — dataset._write_pred has no agreement
    # column; add one there if the deliverable ever needs it.
    sys.path.insert(0, REPO_ROOT)
    from trainer_gui.dataset import _write_pred
    from pathlib import Path
    _write_pred(Path(path), np.asarray(xyz, np.float64),
                np.clip(cls, 0, 255).astype(np.uint8), path.rsplit(".", 1)[1],
                confidence=np.asarray(confidence, np.float32), crs_wkt=crs_wkt)


def align_idx(ref_xyz, xyz):
    """Row map from xyz onto ref_xyz: None when the clouds already match row-for-
    row (fast path); else NN indices so lab/conf/probs can all be re-rowed.
    rtol=0 is load-bearing: on UTM coords numpy's default rtol=1e-5 would make
    the tolerance ~50 m and any same-length cloud would take the fast path."""
    if len(xyz) == len(ref_xyz) and np.allclose(xyz, ref_xyz, rtol=0.0, atol=1e-4):
        return None
    from scipy.spatial import cKDTree      # lazy: only the mismatch path needs it
    return cKDTree(xyz).query(ref_xyz)[1]


PRED_EXTS = (".npz", ".las", ".laz", ".ply", ".txt", ".csv")


def _scan(d):
    """{stem: path} of prediction files in a folder (any supported extension)."""
    out = {}
    for p in sorted(glob.glob(f"{d}/*_pred.*")):
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() in PRED_EXTS:
            out[stem] = p
    return out


def _class_info(d):
    """(num_classes, class_names) from a folder's infer_run.json, or None when
    the folder has none (bare exported files) or it names no classes."""
    p = os.path.join(d, "infer_run.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    n, names = m.get("num_classes"), m.get("class_names") or []
    if n is None and not names:
        return None
    return int(n if n is not None else len(names)), list(names)


def check_class_clamp(input_dirs, log=print):
    """Refuse to mix models with different class sets. Dirs without an
    infer_run.json (bare exported files) skip the check with a printed note."""
    infos = {}
    for d in input_dirs:
        info = _class_info(d)
        if info is None:
            log(f"  note: no infer_run.json in {d} — class check skipped for it")
        else:
            infos[d] = info
    if len({(n, tuple(names)) for n, names in infos.values()}) > 1:
        raise ValueError(
            "class mismatch across ensemble inputs — every model must share one "
            "class set:\n" + "\n".join(f"  {d}: {n} classes {names}"
                                       for d, (n, names) in infos.items()))


def ensemble(input_dirs, out_dir, log=print):
    check_class_clamp(input_dirs, log)
    os.makedirs(out_dir, exist_ok=True)
    scans = [_scan(d) for d in input_dirs]
    if not scans[0]:
        raise FileNotFoundError(f"no *_pred.<npz|las|laz|ply|txt|csv> under {input_dirs[0]}")
    done = 0
    for stem, ref_path in scans[0].items():
        others = [s.get(stem) for s in scans[1:]]
        if any(p is None for p in others):
            miss = [d for d, s in zip(input_dirs[1:], scans[1:]) if stem not in s]
            log(f"  skip {stem}: missing in {miss}")
            continue
        ref_xyz, ref_lab, ref_itn, ref_conf, ref_probs = load_pred(ref_path)
        labs, confs, probs = [ref_lab], [ref_conf], [ref_probs]
        for p in others:
            xyz, lab, _, cf, pb = load_pred(p)
            idx = align_idx(ref_xyz, xyz)
            if idx is not None:
                lab = lab[idx]
                cf = cf[idx] if cf is not None else None
                pb = pb[idx] if pb is not None else None
            labs.append(lab)
            confs.append(cf)
            probs.append(pb)
        stacked = np.stack(labs)
        if all(p is not None for p in probs):
            log(f"  {stem}: soft vote (all {len(probs)} members carry probs)")
            voted, conf = soft_vote(np.stack(probs), stacked)
        else:                                            # fallback: weighted hard vote
            noprobs = [d for d, p in zip(input_dirs, probs) if p is None]
            # the two branches genuinely disagree — say which one ran
            log(f"  {stem}: weighted hard vote (no probs in {noprobs})")
            w = np.stack([c if c is not None else np.ones(len(ref_lab), np.float32)
                          for c in confs])
            voted, conf = weighted_vote(stacked, w)
        agree = agreement(stacked, voted)
        crs = None
        if ref_path.endswith(".npz"):    # ferry the reference member's CRS along
            with np.load(ref_path) as zr:
                crs = str(zr["crs_wkt"]) if "crs_wkt" in zr.files else None
        out_path = f"{out_dir}/{os.path.basename(ref_path)}"
        write_out(out_path, ref_xyz, voted, ref_itn, conf, agree, crs_wkt=crs)
        log(f"  {os.path.basename(out_path)}: {len(voted)} pts, "
            f"mean confidence {float(conf.mean()):.3f}, "
            f"mean agreement {float(agree.mean()):.3f}")
        done += 1
    log(f"ensembled {done}/{len(scans[0])} scene(s) -> {out_dir}")


def self_test():
    import shutil
    import tempfile

    # soft vote overturns the 2-1 hard majority: two unsure models vs one sure one
    sv = np.array([[[0.9, 0.1]], [[0.45, 0.55]], [[0.45, 0.55]]], np.float32)
    lab, conf = soft_vote(sv)
    hard, _ = weighted_vote(np.array([[0], [1], [1]]), np.ones((3, 1), np.float32))
    assert hard.tolist() == [1] and lab.tolist() == [0] and abs(conf[0] - 0.6) < 1e-6
    # coverage: pt0 all-covered, pt1 seen by model 0 only, pt2 seen by nobody
    cv = np.array([[[0.9, 0.1], [0.2, 0.8], [0.0, 0.0]],
                   [[0.45, 0.55], [0.0, 0.0], [0.0, 0.0]],
                   [[0.45, 0.55], [0.0, 0.0], [0.0, 0.0]]], np.float32)
    lab, conf = soft_vote(cv, np.array([[0, 1, 3], [1, 1, 3], [1, 1, 4]]))
    assert lab.tolist() == [0, 1, 3]          # pt2 keeps the NN-filled majority, not 0
    assert np.allclose(conf, [0.6, 0.8, 2 / 3])   # pt1 undiluted by the 2 non-covering
    # weighted hard vote: confidence flips the majority; conf = winning share
    lab, conf = weighted_vote(np.array([[0, 0], [1, 0], [1, 0]]),
                              np.array([[0.9, 1.0], [0.2, 1.0], [0.2, 1.0]], np.float32))
    assert lab.tolist() == [0, 0] and abs(conf[0] - 0.9 / 1.3) < 1e-6 and conf[1] == 1.0
    # unweighted all-way tie keeps the old semantics: earliest model wins
    assert weighted_vote(np.array([[1], [2], [3]]),
                         np.ones((3, 1), np.float32))[0].tolist() == [1]
    # agreement: exact per-point fraction of models matching the final label
    ag = agreement(np.array([[0, 0], [1, 0], [1, 1]]), np.array([0, 0]))
    assert np.allclose(ag, [1 / 3, 2 / 3]) and ag.dtype == np.float32
    # identical clouds take the row-aligned fast path; mismatched fall back to NN
    xyz = np.random.rand(100, 3).astype(np.float32)
    assert align_idx(xyz, xyz) is None
    idx = align_idx(xyz, xyz[::2])
    assert len(idx) == 100 and (idx[::2] == np.arange(50)).all()

    # end-to-end over folders: soft vote, clamp refusal, probs-missing fallback
    tmp = tempfile.mkdtemp(prefix="ensemble_vote_st_")
    try:
        x3 = np.random.rand(3, 3).astype(np.float32)
        P = [np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]], np.float32),
             np.array([[0.45, 0.55], [0.7, 0.3], [0.3, 0.7]], np.float32),
             np.array([[0.45, 0.55], [0.8, 0.2], [0.4, 0.6]], np.float32)]
        dirs = []
        for k, p in enumerate(P):
            d = os.path.join(tmp, f"m{k}")
            os.makedirs(d)
            np.savez(os.path.join(d, "s_pred.npz"), xyz=x3,
                     classification=p.argmax(1).astype(np.int32),
                     confidence=p.max(1), probs=p.astype(np.float16))
            with open(os.path.join(d, "infer_run.json"), "w", encoding="utf-8") as f:
                json.dump({"num_classes": 2, "class_names": ["a", "b"],
                           "backbone": f"m{k}"}, f)
            dirs.append(d)
        quiet = lambda s: None
        ensemble(dirs, os.path.join(tmp, "out"), log=quiet)
        z = np.load(os.path.join(tmp, "out", "s_pred.npz"))
        # point 0 is the overturn: hard majority 2-1 for class 1, soft vote -> 0
        assert z["classification"].tolist() == [0, 0, 1]
        assert z["confidence"].dtype == np.float32 and z["agreement"].dtype == np.float32
        assert np.allclose(z["confidence"], [0.6, 0.7, 0.7], atol=2e-3)   # f16 probs
        assert np.allclose(z["agreement"], [1 / 3, 1.0, 1.0])
        # clamp: mismatched class_names refuse loudly
        with open(os.path.join(dirs[1], "infer_run.json"), "w", encoding="utf-8") as f:
            json.dump({"num_classes": 2, "class_names": ["a", "c"]}, f)
        try:
            ensemble(dirs, os.path.join(tmp, "out2"), log=quiet)
            raise AssertionError("clamp did not refuse mismatched class_names")
        except ValueError:
            pass
        with open(os.path.join(dirs[1], "infer_run.json"), "w", encoding="utf-8") as f:
            json.dump({"num_classes": 2, "class_names": ["a", "b"]}, f)
        # one dir without probs -> confidence-weighted hard vote for every scene
        zp = np.load(os.path.join(dirs[2], "s_pred.npz"))
        slim = {k: zp[k] for k in zp.files if k != "probs"}
        np.savez(os.path.join(dirs[2], "s_pred.npz"), **slim)
        ensemble(dirs, os.path.join(tmp, "out3"), log=quiet)
        z3 = np.load(os.path.join(tmp, "out3", "s_pred.npz"))
        # point 0: w(0)=0.9 vs w(1)=0.55+0.55=1.1 -> label 1, share 1.1/2.0
        assert z3["classification"].tolist() == [1, 0, 1]
        assert np.allclose(z3["confidence"], [0.55, 1.0, 1.0], atol=1e-6)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", nargs="+", help="2+ prediction folders, strongest model first")
    ap.add_argument("--out", help="output folder for the voted predictions")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    elif a.inputs and a.out:
        if len(a.inputs) < 2:
            ap.error("--inputs needs at least 2 folders")
        ensemble(a.inputs, a.out)
    else:
        ap.error("either --self-test or both --inputs and --out")
