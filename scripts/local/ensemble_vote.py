"""Cross-model ensemble over trainer_gui *_pred.<fmt> files, matched by
filename across input folders. Soft vote (averaged "probs") when every member
carries them, else confidence-weighted hard vote; output adds "agreement" and
"dominant_member" (las/laz field "ens_member"), class sets clamped via
infer_run.json. Hard-vote ties go to the earliest input - put your strongest
model first; output uses the first input's format.

Usage:
  python ensemble_vote.py --inputs predsA predsB [predsC ...] --out out_dir
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def soft_vote(probs, labels=None):
    """(M, N, C) distributions -> ((N,) argmax labels, (N,) f32 confidence,
    (N, C) f32 fused distribution). Averages over the COVERING models only
    (all-zero row = not covered); points nobody covered fall back to a plain
    vote over `labels` (their fused row stays zero)."""
    probs = np.asarray(probs, np.float32)
    cov = (probs.sum(-1) > 0).sum(0)
    p = probs.sum(0) / np.maximum(cov, 1)[:, None]
    lab, conf = p.argmax(1), p.max(1).astype(np.float32)
    miss = cov == 0
    if labels is not None and miss.any():
        labels = np.asarray(labels)[:, miss]
        lab[miss], conf[miss] = weighted_vote(labels, np.ones(labels.shape, np.float32))
    return lab, conf, p


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
    for row in labels:
        sel = (out < 0) & (counts[idx, row] == mx)
        out[sel] = row[sel]
    conf = mx / np.maximum(counts.sum(1), 1e-9)
    return out, conf.astype(np.float32)


def agreement(labels, final):
    """(N,) f32 fraction of models whose own label equals the final label."""
    return (np.asarray(labels) == final).mean(0).astype(np.float32)


def dominant_member(voted, probs=None, labels=None, weights=None):
    """(N,) uint8 index of the member that most drove the final label (soft:
    prob mass on the voted class; hard: weight among matching members).
    Ties -> earliest; uncovered soft-vote points land on member 0."""
    voted = np.asarray(voted)
    if probs is not None:
        sup = np.asarray(probs, np.float32)[:, np.arange(len(voted)), voted]
    else:
        sup = np.where(np.asarray(labels) == voted,
                       np.asarray(weights, np.float32), 0.0)
    return sup.argmax(0).astype(np.uint8)


def load_pred(path):
    """(xyz, labels, intensity|None, confidence|None, probs|None) from any
    prediction file. npz is read natively (container-safe); exported formats go
    through the GUI's readers (probs never survive an export - npz only)."""
    if path.endswith(".npz"):
        z = np.load(path)
        # float64: a float32 cast quantizes UTM northing to 0.5 m steps
        return (z["xyz"].astype(np.float64), z["classification"].astype(np.int64),
                z["intensity"] if "intensity" in z.files else None,
                z["confidence"].astype(np.float32) if "confidence" in z.files else None,
                z["probs"].astype(np.float32) if "probs" in z.files else None)
    sys.path.insert(0, REPO_ROOT)
    from trainer_gui.readers import read_points
    c = read_points(path)
    lab = c.fields.get("classification")
    if lab is None and len(c.fields) == 1:
        lab = next(iter(c.fields.values()))
    if lab is None:
        raise ValueError(f"{path}: no 'classification' field (has {list(c.fields)})")
    conf = c.fields.get("confidence")
    return (c.xyz.astype(np.float64), np.asarray(lab, np.int64), c.intensity,
            np.asarray(conf, np.float32) if conf is not None else None, None)


def write_out(path, xyz, cls, intensity, confidence, agree, crs_wkt=None,
              dominant=None, member_names=None, source_crs_wkt=None,
              probs=None):
    if path.endswith(".npz"):
        d = {"xyz": xyz, "classification": cls.astype(np.int32),
             "confidence": np.asarray(confidence, np.float32),
             "agreement": np.asarray(agree, np.float32)}
        if probs is not None:   # fused distribution: post-vote smoothing input
            d["probs"] = np.asarray(probs, np.float16)
        if dominant is not None:
            d["dominant_member"] = np.asarray(dominant, np.uint8)
            if member_names:
                d["member_names"] = np.asarray(list(member_names))
        if intensity is not None:
            d["intensity"] = intensity
        if crs_wkt:
            d["crs_wkt"] = np.asarray(str(crs_wkt))
        if source_crs_wkt:                       # round-trip signal rides along
            d["source_crs_wkt"] = np.asarray(str(source_crs_wkt))
        np.savez(path, **d)
        return
    # ponytail: agreement is npz-only - dataset._write_pred has no agreement column; add one there if the deliverable ever needs it
    sys.path.insert(0, REPO_ROOT)
    from trainer_gui.dataset import _write_pred
    from pathlib import Path
    _write_pred(Path(path), np.asarray(xyz, np.float64),
                np.clip(cls, 0, 255).astype(np.uint8), path.rsplit(".", 1)[1],
                confidence=np.asarray(confidence, np.float32), crs_wkt=crs_wkt,
                source_crs_wkt=source_crs_wkt,
                member=np.asarray(dominant, np.uint8) if dominant is not None else None)


def align_idx(ref_xyz, xyz):
    """Row map from xyz onto ref_xyz: None when rows already match, else NN
    indices. rtol=0 is load-bearing - the default rtol=1e-5 is ~50 m on UTM."""
    if len(xyz) == len(ref_xyz) and np.allclose(xyz, ref_xyz, rtol=0.0, atol=1e-4):
        return None
    from scipy.spatial import cKDTree
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


def _member_name(d):
    """Display name for an ensemble member dir: its infer_run.json backbone,
    else the folder's basename."""
    p = os.path.join(d, "infer_run.json")
    try:
        with open(p, encoding="utf-8") as f:
            name = json.load(f).get("backbone")
        if name:
            return str(name)
    except (OSError, json.JSONDecodeError):
        pass
    return os.path.basename(os.path.normpath(d))


def check_class_clamp(input_dirs, log=print):
    """Refuse to mix models with different class sets. Dirs without an
    infer_run.json (bare exported files) skip the check with a printed note."""
    infos = {}
    for d in input_dirs:
        info = _class_info(d)
        if info is None:
            log(f"  note: no infer_run.json in {d}; class check skipped for it")
        else:
            infos[d] = info
    if len({(n, tuple(names)) for n, names in infos.values()}) > 1:
        raise ValueError(
            "class mismatch across ensemble inputs. Every model must share one "
            "class set:\n" + "\n".join(f"  {d}: {n} classes {names}"
                                       for d, (n, names) in infos.items()))


def ensemble(input_dirs, out_dir, log=print):
    check_class_clamp(input_dirs, log)
    os.makedirs(out_dir, exist_ok=True)
    scans = [_scan(d) for d in input_dirs]
    if not scans[0]:
        raise FileNotFoundError(f"no *_pred.<npz|las|laz|ply|txt|csv> under {input_dirs[0]}")
    names = [_member_name(d) for d in input_dirs]
    log("  dominant-member field: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)))
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
        fused = None
        if all(p is not None for p in probs):
            log(f"  {stem}: soft vote (all {len(probs)} members carry probs)")
            voted, conf, fused = soft_vote(np.stack(probs), stacked)
            dom = dominant_member(voted, probs=np.stack(probs))
        else:
            noprobs = [d for d, p in zip(input_dirs, probs) if p is None]
            # the two branches genuinely disagree - say which one ran
            log(f"  {stem}: weighted hard vote (no probs in {noprobs})")
            w = np.stack([c if c is not None else np.ones(len(ref_lab), np.float32)
                          for c in confs])
            voted, conf = weighted_vote(stacked, w)
            dom = dominant_member(voted, labels=stacked, weights=w)
        agree = agreement(stacked, voted)
        crs = src_crs = None
        if ref_path.endswith(".npz"):    # ferry the reference member's CRS pair along
            with np.load(ref_path) as zr:
                crs = str(zr["crs_wkt"]) if "crs_wkt" in zr.files else None
                src_crs = str(zr["source_crs_wkt"]) if "source_crs_wkt" in zr.files else None
        out_path = f"{out_dir}/{os.path.basename(ref_path)}"
        write_out(out_path, ref_xyz, voted, ref_itn, conf, agree, crs_wkt=crs,
                  dominant=dom, member_names=names, source_crs_wkt=src_crs,
                  probs=fused)
        log(f"  {os.path.basename(out_path)}: {len(voted)} pts, "
            f"mean confidence {float(conf.mean()):.3f}, "
            f"mean agreement {float(agree.mean()):.3f}")
        done += 1
    log(f"ensembled {done}/{len(scans[0])} scene(s) -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="2+ prediction folders, strongest model first")
    ap.add_argument("--out", required=True, help="output folder for the voted predictions")
    a = ap.parse_args()
    if len(a.inputs) < 2:
        ap.error("--inputs needs at least 2 folders")
    ensemble(a.inputs, a.out)
