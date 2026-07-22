"""Shared best-checkpoint selection for the local_train_* trainers.

Studies pick the checkpoint with the highest validation mIoU, not the last
epoch (Pointcept -> model_best.pth; the DFC aerial-LiDAR eval paper
arXiv:2603.22420 selects "the checkpoint with highest validation mIoU"). This
makes final_model.pth = best-by-val-mIoU while keeping the inference contract
(final_model.pth) unchanged.
"""
import contextlib
import csv
import json
import os
import time

# Path contract: inside containers (Modal) these stay the fixed /datasets +
# /outputs mounts; the pixi local backend has no container, so the GUI points
# them at the real host dirs via TT_* env vars. Unset vars = container layout,
# byte-for-byte.
DATASETS_ROOT = os.environ.get("TT_DATASETS_ROOT", "/datasets")
OUTPUTS_ROOT = os.environ.get("TT_OUTPUTS_ROOT", "/outputs")


def dataset_dir(name):
    """Root of a canonical dataset. TT_DATASET_DIR overrides for a staged
    dataset living outside the workspace (was an extra bind mount)."""
    return os.environ.get("TT_DATASET_DIR") or f"{DATASETS_ROOT}/{name}"


def infer_dir(job):
    """Inference job dir (scenes/ + predictions live here). TT_INFER_DIR
    overrides (was an extra bind mount)."""
    return os.environ.get("TT_INFER_DIR") or f"{DATASETS_ROOT}/_infer/{job}"


def write_pred(path, xyz, pred, intensity=None, confidence=None, probs=None,
               crs_wkt=None):
    """Write one inferred scene as a compact npz — xyz + per-point class index
    (+ optional intensity, per-point confidence = max of the normalized class
    distribution, and the full float16 distribution when TT_SAVE_PROBS=1). The
    host exporter (dataset.export_predictions) writes the user's chosen file
    type straight from this, so there's no intermediate coloured PLY to render
    and then reparse. Class is stored losslessly (the raw prediction), not
    encoded as palette colour."""
    import numpy as np
    # xyz stays float64: predictions carry georeferenced (UTM ~1e6) coords,
    # where a float32 cast quantizes northing to 0.5m steps in the deliverable.
    d = {"xyz": np.asarray(xyz, np.float64),
         "classification": np.asarray(pred, np.int32)}
    if intensity is not None:
        d["intensity"] = np.asarray(intensity, np.float32)
    if confidence is not None:
        d["confidence"] = np.asarray(confidence, np.float32)
    if probs is not None:
        d["probs"] = np.asarray(probs, np.float16)
    if crs_wkt:            # source CRS (WKT string) — the exporter georeferences
        d["crs_wkt"] = np.asarray(str(crs_wkt))     # the las/laz deliverable with it
    np.savez(path, **d)


def best_val_miou(val_csv):
    """Max val_miou already recorded (resume-safe seed). -1.0 if none."""
    if not os.path.exists(val_csv):
        return -1.0
    best = -1.0
    with open(val_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                best = max(best, float(row["val_miou"]))
            except (KeyError, ValueError):
                pass
    return best


class BestCheckpoint:
    """Track best val mIoU; write final_model.pth on a new best.

        best = BestCheckpoint(run_dir)            # seeds from val_metrics.csv
        if best.update(miou):                     # True on a new best
            torch.save(payload, best.final)       # model-specific payload
        ...
        best.finalize(lambda p: torch.save(last_payload, p))  # fallback if no val
    """

    def __init__(self, run_dir):
        self.final = os.path.join(run_dir, "final_model.pth")
        self.best = best_val_miou(os.path.join(run_dir, "val_metrics.csv"))

    def update(self, miou):
        if miou > self.best:
            self.best = miou
            return True
        return False

    def finalize(self, save_last):
        # final_model.pth already holds the best (written on improvement); only
        # save the last epoch if validation never ran (subset empty).
        if not os.path.exists(self.final):
            save_last(self.final)


STOP_SENTINEL = f"{OUTPUTS_ROOT}/STOP"   # module attr, not a default arg: smoke test repoints it


def clear_stop():
    """Delete a stale STOP sentinel at trainer startup so an old graceful-stop
    request can't kill a fresh run. ponytail: concurrent runs sharing one
    /outputs share the sentinel — accepted; press stop once per run."""
    try:
        os.remove(STOP_SENTINEL)
        print("  [stop] removed stale STOP sentinel", flush=True)
    except OSError:
        pass


def stop_requested(ep):
    """True when the GUI dropped /outputs/STOP: consume it and log. Called once
    per epoch at the loop tail; the trainer breaks and the NORMAL post-loop
    final-eval + finalize path runs (test_metrics.json, final_model.pth)."""
    if not os.path.exists(STOP_SENTINEL):
        return False
    try:
        os.remove(STOP_SENTINEL)
    except OSError:
        pass
    print(f"  [stop] STOP sentinel found — stopping after epoch {ep}; "
          f"running the final evaluation…", flush=True)
    return True


def _dg_block() -> dict | None:
    """DG settings that must travel WITH the weights, read from the same env the
    training process ran under. `logdk` changes the model's input width, so
    inference has to rebuild at that width and recompute the channel (with the same
    k) — record both so run.json is self-describing. AdaBN/TTA are inference-time
    choices made on the Infer page, NOT a property of the weights, so not here —
    write_infer_run records them per inference job instead."""
    try:
        import density as dg   # sibling helper; always importable when a trainer runs
    except ImportError:
        return None
    return {
        "density_aug": dg.env_bool("DG_DENSITY_AUG", False),
        "coarsen_max": dg.env_float("DG_COARSEN_MAX", 2.5),
        "p_native":    dg.env_float("DG_P_NATIVE", 0.5),
        "logdk":       dg.env_bool("DG_LOGDK_FEAT", False),
        "logdk_k":     dg.env_int("DG_LOGDK_K", 8),
    }


def _intensity_norm_from_meta(meta: dict) -> str:
    """Where convert_dataset records the intensity normalization: under
    meta['source']['intensity_norm'] (a top-level copy is tolerated for other
    writers). Default 'max'. Getting this wrong feeds inference a different
    intensity scale than training saw."""
    src = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    return src.get("intensity_norm") or meta.get("intensity_norm") or "max"


def write_run_manifest(run_dir, backbone, dataset=None, weights="final_model.pth"):
    """Finalize run.json — THE single record of a training run, next to the
    weights. Local inference reads ONLY this file (the user picks it explicitly):
    every input it needs is here, and the weights are its sibling `weights`. No
    searching, no path conventions.

    The trainer writes its raw config to run.json at train START; this call (at
    train end) MERGES the normalized manifest fields over it (so grid/chunk stay
    in sync across backbones, which name them differently) plus the dataset's
    intensity normalization — one file, raw config + manifest. A legacy
    run_config.json is read as the raw source when run.json doesn't exist yet.
    `backbone` is the backbone KEY (e.g. 'ptv3')."""
    rc = {}
    for fn in ("run.json", "run_config.json"):        # run_config.json = legacy runs
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    rc = json.load(f)
            except (OSError, ValueError):
                rc = {}
            break
    inorm = "p95"   # default; overridden below by the dataset's recorded intensity_norm
    if dataset:
        mp = f"{dataset_dir(dataset)}/dataset_meta.json"
        try:
            with open(mp, encoding="utf-8") as f:
                inorm = _intensity_norm_from_meta(json.load(f))
        except (OSError, ValueError):
            inorm = "max"
    dg = _dg_block()
    manifest = {
        "schema": "trainer_gui.run/2",
        "backbone": backbone,
        "weights": weights,
        "num_classes": rc.get("num_classes"),
        "class_names": rc.get("class_names"),
        "grid": rc.get("grid_size", rc.get("grid_m", rc.get("sub_grid_size", rc.get("grid")))),
        "chunk_xy": rc.get("chunk_xy", rc.get("chunk_xy_m")),
        "intensity_norm": inorm,
        # model-specific extras (absent/None when a backbone doesn't use them):
        "num_points": rc.get("num_points"),                              # RandLA sample size
        # density-generalization settings baked into the weights. `logdk` changes the
        # model input width, so inference MUST re-set DG_LOGDK_FEAT/_K to rebuild and
        # recompute the channel — that's why it travels with the weights here. AdaBN/TTA
        # are inference-time choices (Infer page), deliberately NOT recorded.
        "dg": dg,
    }
    doc = {**rc, **manifest}                          # manifest keys are authoritative
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return doc


def infer_meta(weights_path):
    """Inference metadata for a run, read from run.json beside the weights — the
    single self-contained manifest — so `run.json` + weights is enough on any host
    (no dependence on run_config.json). Falls back to a legacy run_config.json
    (normalizing its per-backbone key names) for older runs. Returns a normalized
    dict, or None when neither file is beside the weights (a bare .pth) so callers
    sniff/default. Missing individual fields are None."""
    d = os.path.dirname(weights_path)
    if os.path.basename(d) == "checkpoints":   # weights in runs/<id>/checkpoints/
        d = os.path.dirname(d)
    rj, rc_path = os.path.join(d, "run.json"), os.path.join(d, "run_config.json")
    if os.path.exists(rj):
        try:
            with open(rj, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            return None
        return {k: m.get(k) for k in ("num_classes", "class_names", "grid", "chunk_xy",
                                      "num_points", "dg",
                                      "features", "color_source", "hag_source")}
    if os.path.exists(rc_path):
        try:
            with open(rc_path, encoding="utf-8") as f:
                rc = json.load(f)
        except (OSError, ValueError):
            return None
        return {
            "num_classes": rc.get("num_classes"),
            "class_names": rc.get("class_names"),
            "grid": rc.get("grid_size", rc.get("grid_m", rc.get("sub_grid_size"))),
            "chunk_xy": rc.get("chunk_xy", rc.get("chunk_xy_m")),
            "num_points": rc.get("num_points"),
        }
    return None


# ------------------------------------------------------------ inference utils

def xy_chunk_groups(xyz, chunk_m, min_pts=1):
    """Index arrays grouping points into chunk_m x chunk_m XY windows (groups
    smaller than min_pts skipped) — one sort over packed window codes instead
    of a full-cloud boolean mask per window (O(n log n) vs O(n * windows))."""
    import numpy as np
    xy = np.asarray(xyz)[:, :2]
    if len(xy) == 0:
        return []
    ij = np.floor((xy - xy.min(0)) / float(chunk_m)).astype(np.int64)
    code = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
    order = np.argsort(code, kind="stable")
    sc = code[order]
    cuts = np.flatnonzero(sc[1:] != sc[:-1]) + 1
    return [g for g in np.split(order, cuts) if len(g) >= min_pts]


def voxel_unique(keys, return_inverse=False):
    """First-occurrence indices (and optionally the inverse map) of the unique
    integer ROWS of `keys` — identical output to np.unique(keys, axis=0,
    return_index=True[, return_inverse=True]) but ~10x faster: rows are packed
    into one int64 code (lexicographic order preserved, so even the ordering
    matches), and a 1-D unique does the work. The unique rows themselves are
    keys[first]. Falls back to axis=0 if the packed code would overflow."""
    import numpy as np
    keys = np.asarray(keys, dtype=np.int64)
    k = keys - keys.min(0)
    spans = k.max(0) + 1
    if float(np.prod(spans.astype(np.float64))) >= 2.0 ** 62:
        _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        return (first, inv) if return_inverse else first
    code = k[:, 0]
    for d in range(1, k.shape[1]):
        code = code * int(spans[d]) + k[:, d]
    _, first, inv = np.unique(code, return_index=True, return_inverse=True)
    return (first, inv) if return_inverse else first


def write_infer_run(run_dir, config, scene_stats):
    """infer_run.json — the single record of an inference job: the exact config
    used plus per-scene metrics ({scene, points, seconds}). Callers rewrite it
    after every scene, so a crash still leaves the completed scenes' numbers."""
    doc = dict(config)
    # Inference-time DG toggles: not a property of the weights (run.json keeps
    # those), but they ARE the record of this job — without them, sweep runs
    # differing only in AdaBN/TTA leave indistinguishable predictions.
    doc["adabn"] = os.environ.get("DG_INFER_ADABN") == "1"
    doc["tta_views"] = int(os.environ.get("DG_INFER_TTA", "0") or 0)
    doc["save_probs"] = os.environ.get("TT_SAVE_PROBS") == "1"
    doc["scenes"] = scene_stats
    doc["total_points"] = int(sum(s["points"] for s in scene_stats))
    doc["total_seconds"] = round(sum(float(s["seconds"]) for s in scene_stats), 3)
    with open(os.path.join(run_dir, "infer_run.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return doc


def exclude_class_idx(class_names):
    """EXCLUDE_CLASSES env (csv of class NAMES, e.g. "water,vehicle") -> sorted
    index list into this run's class_names. [] when unset. Unknown names fail
    loudly — a typo silently masking nothing is worse. At least one class must
    survive. (Legacy runs without class_names synthesize per-backbone fallback
    names; the error lists the accepted spellings.)"""
    names = [s.strip() for s in os.environ.get("EXCLUDE_CLASSES", "").split(",")
             if s.strip()]
    if not names:
        return []
    bad = [n for n in names if n not in class_names]
    if bad:
        raise ValueError(f"EXCLUDE_CLASSES names {bad} not in this run's "
                         f"classes {list(class_names)}")
    idx = sorted(class_names.index(n) for n in set(names))
    if len(idx) >= len(class_names):
        raise ValueError("EXCLUDE_CLASSES excludes every class — nothing left "
                         "to predict")
    print(f"  [infer] masking classes: {', '.join(names)} — masked points fall "
          f"to their next-best class; confidence is post-mask", flush=True)
    return idx


def apply_class_mask(prob, exclude_idx):
    """Zero the excluded columns of a normalized (N, C) prob matrix and
    renormalize (in place), so argmax/confidence fall to the next-best class.
    No-op on an empty list. Softmax probs are strictly positive, so the
    remaining mass can't be exactly zero."""
    if not exclude_idx:
        return prob
    import numpy as np
    prob[..., exclude_idx] = 0.0
    prob /= np.maximum(prob.sum(-1, keepdims=True), 1e-12)
    return prob


# ============================================================================
# Shared trainer pieces — the copy-paste core of the four local_train_* scripts,
# extracted verbatim. Everything a helper needs arrives as an argument (the
# scripts used to close over train_*() locals, which is why they copy-pasted);
# per-run config binds via the make_* factories. torch/numpy import inside the
# functions so this module stays importable on a GPU-less host (smoke test).
# ============================================================================

def gpu_name():
    """Real CUDA device name for logs/metadata."""
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------- losses

def lovasz_softmax_flat(probas, labels):
    """Lovász-Softmax (Berman et al. 2018): differentiable mIoU/Jaccard
    surrogate on (N, C) softmax probs / (N,) labels in [0, C), averaged over
    classes present in the batch."""
    import torch

    def _grad(gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1].clone()
        return jaccard

    if probas.numel() == 0:
        return probas.sum() * 0.0   # scalar 0, so an all-ignored batch can't crash
    losses = []
    for c in torch.unique(labels):
        fg = (labels == c).float()
        errors = (fg - probas[:, int(c)]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, _grad(fg[perm])))
    return torch.stack(losses).mean()


def focal_loss(logits, labels, gamma, class_weights=None):
    """alpha-balanced multiclass focal loss; masks ignore_index=-1 internally.
    alpha = class_weights (inverse-sqrt) when set. No label smoothing here."""
    import torch
    valid = labels >= 0
    if not valid.any():
        return logits.sum() * 0.0
    lg, lb = logits[valid], labels[valid]
    logp = torch.log_softmax(lg, dim=1)
    logpt = logp.gather(1, lb.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    loss = -((1.0 - pt) ** gamma) * logpt
    if class_weights is not None:
        loss = loss * class_weights[lb]
    return loss.mean()


def make_seg_loss(class_weights, label_smooth, use_focal, focal_gamma, lovasz_weight):
    """The shared total loss: weighted (label-smoothed) CE or focal, + Lovász.
    Guards all-ignored batches: CrossEntropyLoss(ignore_index=-1) returns NaN
    when every label is ignored (0/0 reduction); return a finite zero-grad
    value instead so backward() can never poison the weights."""
    import torch
    ce = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1,
                                   label_smoothing=label_smooth)

    def seg_loss(logits, labels):
        valid = labels >= 0
        if not valid.any():
            return logits.sum() * 0.0
        loss = (focal_loss(logits, labels, focal_gamma, class_weights)
                if use_focal else ce(logits, labels))
        if lovasz_weight > 0:
            probas = torch.softmax(logits[valid], dim=1)
            loss = loss + lovasz_weight * lovasz_softmax_flat(probas, labels[valid])
        return loss

    return seg_loss


# ------------------------------------------------- class balance / sampling

def drop_corrupt_tile(path):
    """Remove a truncated cached tile AND its scene's .done marker so the next
    prep re-tiles that scene. Interrupted runs can leave these behind: the
    modal shells commit volumes every 120s WHILE tiles are being written, so a
    preemption mid-commit can persist a half-uploaded npz behind a .done that
    made the same commit — and validate_cache checks the signature, not file
    integrity, so the poisoned cache is trusted forever without this."""
    import re
    print(f"  corrupt cached tile dropped: {path}", flush=True)
    try:
        os.remove(path)
    except OSError:
        pass
    m = re.match(r"(.+)_x-?\d+_y-?\d+\.npz$", os.path.basename(path))
    if m:
        try:
            os.remove(os.path.join(os.path.dirname(path), f"{m.group(1)}.done"))
        except OSError:
            pass


def scan_class_balance(tile_paths, num_classes, cache_path=None):
    """One pass over cached tiles' 'lab' arrays -> (class_counts, present_mask
    (n_tiles, C)). The per-tile np.load is the bottleneck (~45k tiles ->
    minutes), so read in parallel and optionally cache the raw scan (keyed on
    the tile set) — instant on every later launch."""
    import numpy as np
    names = np.array([os.path.basename(p) for p in tile_paths])
    if cache_path and os.path.exists(cache_path):
        try:
            cz = np.load(cache_path, allow_pickle=False)
            if (cz["tile_names"].shape == names.shape
                    and bool(np.all(cz["tile_names"] == names))
                    and int(cz["num_classes"]) == num_classes):
                print(f"  class balance: loaded cache ({len(tile_paths)} tiles)", flush=True)
                return cz["class_counts"].astype(np.int64), cz["present_mask"].astype(bool)
        except Exception as e:
            print(f"  class balance: ignoring unreadable cache ({e})", flush=True)

    def _scan(tp):
        try:
            lab = np.load(tp)["lab"]
        except Exception:
            return None            # truncated/corrupt tile — healed below
        v = lab[(lab >= 0) & (lab < num_classes)]
        return (np.bincount(v, minlength=num_classes).astype(np.int64)
                if v.size else np.zeros(num_classes, np.int64))

    from concurrent.futures import ThreadPoolExecutor
    print(f"  scanning {len(tile_paths)} train tiles for class balance (parallel)…",
          flush=True)
    per_tile = np.zeros((len(tile_paths), num_classes), np.int64)
    bad = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, counts in enumerate(ex.map(_scan, tile_paths)):
            if counts is None:
                bad.append(tile_paths[i])
            else:
                per_tile[i] = counts
    if bad:
        for p in bad:
            drop_corrupt_tile(p)
        raise RuntimeError(
            f"class-balance scan: {len(bad)} corrupt cached tile(s) removed and "
            "their scene(s) unmarked — rerun (Modal auto-retries) to re-tile them.")
    class_counts, present_mask = per_tile.sum(0), per_tile > 0
    if cache_path:
        try:
            np.savez(cache_path, tile_names=names, class_counts=class_counts,
                     present_mask=present_mask, num_classes=np.int64(num_classes))
            print(f"  class balance: cached scan -> {cache_path}", flush=True)
        except Exception as e:
            print(f"  class balance: could not write cache ({e})", flush=True)
    return class_counts, present_mask


def class_weights_np(class_counts, beta, cap, absent_to_one=False):
    """Inverse-frequency^beta weights (0.5 = inverse-sqrt, the RandLA standard),
    mean-normalized then clamped to [1/cap, cap]. absent_to_one pins classes
    with zero train points at 1.0 and mean-norms over present classes only
    (the PTv3 variant); off reproduces the KP scripts byte-for-byte."""
    import numpy as np
    freq = class_counts / max(int(class_counts.sum()), 1)
    w = (1.0 / np.maximum(freq, 1e-6)) ** beta
    if absent_to_one:
        w[class_counts == 0] = 1.0
        if (class_counts > 0).any():
            w = w / w[class_counts > 0].mean()
    else:
        w = w / w.mean()
    return np.clip(w, 1.0 / cap, cap)


def auto_rare_classes(class_counts, freq_frac):
    """Auto-rare rule shared by the tile trainers: present classes whose count
    is below freq_frac x the median present-class count."""
    import numpy as np
    present = class_counts[class_counts > 0]
    thresh = freq_frac * float(np.median(present)) if present.size else 0.0
    return [c for c in range(len(class_counts)) if 0 < class_counts[c] < thresh]


def make_tile_picker(train_tiles, rare_tiles, rare_prob):
    """P(rare_prob) draw from a rare-class tile, else uniform."""
    import numpy as np

    def pick_train_tile():
        if rare_tiles and np.random.rand() < rare_prob:
            return rare_tiles[np.random.randint(len(rare_tiles))]
        return train_tiles[np.random.randint(len(train_tiles))]

    return pick_train_tile


# ------------------------------------------------- canonical dataset + cache

def split_scenes(ds_root):
    """The dataset stage already materialized three whole-scene folders; read
    them verbatim and never re-carve a split. Returns three (name, pc_path,
    None) lists (the third slot is a legacy cls_path, always None: labels are
    embedded in the canonical .npz)."""
    import glob
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]

    def _items(split):
        return [(stem(p), p, None)
                for p in sorted(glob.glob(f"{ds_root}/{split}/*.npz"))]

    train, val, test = _items("train"), _items("val"), _items("test")
    if not train:
        raise FileNotFoundError(f"No canonical scenes under {ds_root}/train")
    return train, val, test


def validate_cache(prep_dir, sig, lists, legacy_pair):
    """Refuse to reuse a prep cache built with different settings instead of
    silently mixing incompatible data. Migrate a pre-validation cache by
    stamping .done markers for already-tiled scenes — legacy_pair(split_dir,
    name) returns that script's (output_glob, done_path). Returns True if the
    signature file was newly written."""
    import glob
    meta_path = f"{prep_dir}/cache_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        if old != sig:
            diffs = {k: [old.get(k), sig.get(k)]
                     for k in sorted(set(old) | set(sig)) if old.get(k) != sig.get(k)}
            raise RuntimeError(
                f"Preprocess cache at {prep_dir} was built with DIFFERENT settings "
                f"(mismatched: {diffs}). Reusing it would silently mix incompatible "
                f"data. Point PREP_DIR at a fresh path or delete the stale cache.")
        return False
    legacy = False
    for split, items in lists:
        d = f"{prep_dir}/{split}"
        for name, _, _ in items:
            pattern, done = legacy_pair(d, name)
            if glob.glob(pattern) and not os.path.exists(done):
                open(done, "w").close()
                legacy = True
    with open(meta_path, "w") as f:
        json.dump(sig, f, indent=2)
    if legacy:
        print(f"  migrated existing cache at {prep_dir}: stamped .done markers + "
              f"signature (assumed to match current settings).", flush=True)
    return True


# ------------------------------------------------------------- eval scoring

def score_ious(pred, lab, num_classes):
    """Per-class (intersection, union, gt_count) over already-valid-masked
    prediction/label arrays."""
    import numpy as np
    inter = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    gt = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        inter[c] = int(((pred == c) & (lab == c)).sum())
        union[c] = int(((pred == c) | (lab == c)).sum())
        gt[c] = int((lab == c).sum())
    return inter, union, gt


def eval_metrics(t_inter, t_union, t_gt, correct, total, class_names, t_start,
                 n_scenes, label, extra=None):
    """The metrics dict every evaluate() builds (acc, all-class + present-class
    mIoU, per-class IoU/GT counts), plus the one-line summary print. `extra`
    carries script-specific tail keys (skipped counts, protocol descriptors)."""
    import numpy as np
    num_classes = len(class_names)
    with np.errstate(invalid="ignore"):
        iou_per = t_inter / np.maximum(t_union, 1)
    gt_counts = [int(x) for x in t_gt.tolist()]
    present = [c for c in range(num_classes) if gt_counts[c] > 0]
    present_iou = [float(iou_per[c]) for c in present]
    present_mIoU = float(np.mean(present_iou)) if present_iou else 0.0
    extra = extra or {}
    m = {
        "overall_acc": correct / max(total, 1),
        "overall_mIoU": float(np.mean(iou_per)),
        "present_classes_mIoU": present_mIoU,
        "per_class_iou": {class_names[c]: float(iou_per[c]) for c in range(num_classes)},
        "per_class_gt_count": {class_names[c]: gt_counts[c] for c in range(num_classes)},
        "present_classes": [class_names[c] for c in present],
        "absent_classes": [class_names[c] for c in range(num_classes) if gt_counts[c] == 0],
        "total_test_seconds": time.time() - t_start,
        "num_scenes": n_scenes,
        "num_raw_points_scored": int(total),
        **extra,
    }
    skipped = {k.split("_")[1]: extra[k] for k in ("skipped_tiles", "skipped_scenes")
               if k in extra}
    print(f"  [{label}] acc={m['overall_acc']:.4f}  "
          f"mIoU({num_classes}-way)={m['overall_mIoU']:.4f}  "
          f"mIoU(present {len(present)})={m['present_classes_mIoU']:.4f}  "
          f"absent={m['absent_classes']}  raw_pts={total:,}"
          + ("  skipped(" + ",".join(f"{k}={v}" for k, v in skipped.items()) + ")"
             if skipped else ""), flush=True)
    return m


def append_val_row(val_csv, ep, m, class_names):
    """One val_metrics.csv row: epoch, acc, present-class mIoU, per-class IoUs."""
    ious = [m["per_class_iou"][n] for n in class_names]
    with open(val_csv, "a", newline="") as f:
        csv.writer(f).writerow([ep, f"{m['overall_acc']:.4f}",
                                f"{m['present_classes_mIoU']:.4f}"]
                               + [f"{x:.4f}" for x in ious])


# ---------------------------------------------------------- inference scenes

def scene_arrays(z, n):
    """(intensity, ret_num) from a canonical scene npz — THE one place the
    missing-channel fallbacks are decided, so train tiling and inference feed
    the same distribution (intensity -> RGB grayscale -> zeros;
    return_number/ret_num -> zeros)."""
    import numpy as np
    if "intensity" in z:
        intensity = z["intensity"].astype(np.float32)
    elif "rgb" in z:
        intensity = z["rgb"].astype(np.float32).mean(1) / 255.0
    else:
        intensity = np.zeros(n, np.float32)
    ret_num = (z["return_number"].astype(np.float32) if "return_number" in z
               else (z["ret_num"].astype(np.float32) if "ret_num" in z
                     else np.zeros(n, np.float32)))
    return intensity, ret_num


def run_infer_scenes(scenes, predict, pred_dir, run_dir, infer_cfg, cls_txt=False):
    """The --mode infer scene loop: predict(pc_path) ->
    (xyz, pred, intensity, confidence, probs) — confidence is the float32
    per-point max of the normalized class distribution (0.0 where nothing was
    predicted), probs the float16 distribution or None — written as
    <name>_pred.npz (+ optional _pred_CLS.txt), with the crash-safe per-scene
    infer_run.json rewrite."""
    import numpy as np
    print(f"  [infer] labeling {len(scenes)} scene(s) -> {pred_dir}", flush=True)
    scene_stats = []
    for pc_path in scenes:
        name = os.path.splitext(os.path.basename(pc_path))[0]
        t0 = time.time()
        xyz, pred, inten, conf, probs = predict(pc_path)
        try:                            # ferry the scene's CRS into the pred npz
            with np.load(pc_path) as z:
                crs = str(z["crs_wkt"]) if "crs_wkt" in z.files else None
        except OSError:                 # predict() is the scene's real reader —
            crs = None                  # a missing/odd file is its problem, not ours
        write_pred(f"{pred_dir}/{name}_pred.npz", xyz, pred, inten, conf, probs,
                   crs_wkt=crs)
        if cls_txt:
            np.savetxt(f"{pred_dir}/{name}_pred_CLS.txt", pred, fmt="%d")
        scene_stats.append({"scene": os.path.basename(pc_path),
                            "points": int(len(xyz)),
                            "seconds": round(time.time() - t0, 3)})
        write_infer_run(run_dir, infer_cfg, scene_stats)   # crash-safe: per scene
        print(f"  [infer] {name}: {len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
    # exact wording matters: the GUI's _localize_paths rewrites this line
    print(f"  [infer] done — predictions in "
          f"_infer/{os.path.basename(os.path.dirname(pred_dir))}/predictions",
          flush=True)


# ============================================================================
# KPConv-family shared pipeline — the prep/feature/eval core the kpconv and
# kpconvx_cold twins used to duplicate. Dataset feat_* channels are automatic:
# kp_tile_and_save caches every feat_* the canonical scene carries, and
# kp_make_build_feat feeds whichever ones the FEAT_CHANNELS spec names.
# ============================================================================

def kp_load_canonical(npz_path):
    """Canonical trainer_gui scene (.npz) -> (xyz, intensity, ret_num, lab,
    extras), extras = every feat_* channel the scene carries (Phase 2a writes
    them). Missing intensity/return_number fall back via scene_arrays — the
    single definition inference also uses, so train and infer see the same
    distribution. xyz is origin-offset (per-scene floor-min) before the
    float32 cast so projected (UTM) coords keep sub-meter precision."""
    import numpy as np
    z = np.load(npz_path)
    xyz = (z["xyz"] - np.floor(z["xyz"].min(0))).astype(np.float32)
    intensity, ret_num = scene_arrays(z, len(xyz))
    lab = z["label"].astype(np.int32) if "label" in z.files \
        else np.full(len(xyz), -1, np.int32)
    return xyz, intensity, ret_num, lab, scene_feats(z)


def scene_feats(z):
    """Every feat_* channel a scene npz carries. Pre-2026-07-13 conversions
    stored HAG under a bare 'hag' key — surface it as feat_hag so legacy
    datasets keep working without a re-convert/re-upload."""
    import numpy as np
    out = {k: z[k].astype(np.float32) for k in z.files if k.startswith("feat_")}
    if "feat_hag" not in out and "hag" in z.files:
        out["feat_hag"] = z["hag"].astype(np.float32)
    return out


def _grid_pool_t(p, a, l, voxel, num_classes):
    """Torch core of kp_grid_subsample: (points, attrs-or-None,
    labels-or-None) tensors on any device -> pooled tensors on that device.
    Voxel keys are raveled to one int64 so unique is a sort, not a row
    compare; sorted-flat order == np.unique(axis=0) lexicographic order, and
    float64 accumulators keep the old numpy numerics."""
    import torch
    k = torch.floor(p / voxel).long()
    k -= k.min(0).values
    m = k.max(0).values + 1
    flat = (k[:, 0] * m[1] + k[:, 1]) * m[2] + k[:, 2]
    inv = torch.unique(flat, return_inverse=True)[1]
    nv = int(inv.max()) + 1
    cnt = torch.bincount(inv, minlength=nv).double()
    sx = torch.zeros(nv, 3, dtype=torch.float64, device=p.device)
    sx.index_add_(0, inv, p.double()); sx /= cnt[:, None]
    sa = None
    if a is not None:
        sa = torch.zeros(nv, a.shape[1], dtype=torch.float64, device=p.device)
        sa.index_add_(0, inv, a.double()); sa /= cnt[:, None]
    sl = torch.full((nv,), -1, dtype=torch.int64, device=p.device)
    if l is not None:
        l = l.long()
        v = l >= 0
        oh = torch.bincount(inv[v] * num_classes + l[v],
                            minlength=nv * num_classes).reshape(nv, num_classes)
        has = oh.sum(1) > 0
        sl[has] = oh[has].argmax(1)
    return sx.float(), (sa.float() if sa is not None else None), sl


def kp_grid_subsample(xyz, attrs, lab, voxel, num_classes):
    """Voxel-grid subsample to `voxel` m: barycenter points, mean attrs,
    majority labels. Mirrors KPConv's grid_subsampling (the C++ op that
    produces the layer-0 cloud). Numpy wrapper over _grid_pool_t — pooled on
    CUDA when available, CPU torch otherwise (every trainer image ships
    torch)."""
    import numpy as np
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    sx, sa, sl = _grid_pool_t(t(xyz), t(attrs) if attrs is not None else None,
                              t(lab) if lab is not None else None,
                              voxel, num_classes)
    return (sx.cpu().numpy(),
            (sa.cpu().numpy() if sa is not None else None),
            sl.cpu().numpy())


def kp_augment(xyz, scale_min=0.9, scale_max=1.1, sym_x=True, noise=0.05):
    """Shared trainer_gui augmentation: vertical rotation, anisotropic scale
    with random x-flip, gaussian noise."""
    import numpy as np
    theta = np.random.rand() * 2 * np.pi
    cs, sn = np.cos(theta), np.sin(theta)
    R = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], np.float32)
    scale = np.random.uniform(scale_min, scale_max, 3).astype(np.float32)
    if sym_x and np.random.rand() < 0.5:
        scale[0] *= -1.0
    out = (xyz @ R.T) * scale
    out += np.random.normal(0, noise, out.shape).astype(np.float32)
    return out.astype(np.float32)


def make_prefetcher(make_batch, depth=2):
    """next() -> a ready training batch; `depth` background threads keep
    building the following ones, so tile loading + CPU pyramid assembly
    overlaps GPU compute (the same overlap upstream KPConvX gets from its
    DataLoader workers). Build errors re-raise at next(); call .shutdown()
    when the loop ends."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(depth)
    q = deque(ex.submit(make_batch) for _ in range(depth + 1))

    def nxt():
        q.append(ex.submit(make_batch))
        return q.popleft().result()

    nxt.shutdown = lambda: ex.shutdown(wait=False, cancel_futures=True)
    return nxt


def train_stride(chunk_xy):
    """Train-split tile stride: chunk_xy * TT_TRAIN_STRIDE (default 0.75 ->
    ~1.8x point duplication instead of the legacy 0.5's 4x; 1.0 = no overlap).
    Val/test always keep chunk_xy/2 — the eval's per-voxel voting needs the
    up-to-4 cover."""
    return chunk_xy * float(os.environ.get("TT_TRAIN_STRIDE", "0.75"))


def train_stride_tag():
    """Prep-dir suffix carrying the train-stride factor, so a factor change
    lands in a fresh cache instead of mixing grids. Empty for the legacy 0.5
    — those existing caches stay valid untagged."""
    f = float(os.environ.get("TT_TRAIN_STRIDE", "0.75"))
    return "" if f == 0.5 else f"_ts{f:g}"


def _savez_fast(path, **arrays):
    """np.savez_compressed but zlib level 1: ~2.4x faster writes for ~6%
    bigger tiles (measured on real tiles); np.load reads it unchanged."""
    import io
    import zipfile
    import numpy as np
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name, a in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(a))
            zf.writestr(name + ".npy", buf.getvalue())


@contextlib.contextmanager
def npz_save_pool():
    """`with npz_save_pool() as save:` -> save(path, **arrays) queues a
    _savez_fast on a thread pool — zlib releases the GIL, so tile
    writes scale with cores instead of serializing behind the tiling loop.
    Queue is bounded at 2x workers (backpressure: raw ptv3 tiles are ~10MB
    each and would otherwise pile up in RAM); worker exceptions surface on a
    later save() or at exit."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    workers = os.cpu_count() or 4
    ex = ThreadPoolExecutor(workers)
    pending = set()

    def save(path, **arrays):
        nonlocal pending
        if len(pending) >= workers * 2:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                f.result()
        pending.add(ex.submit(_savez_fast, path, **arrays))

    try:
        yield save
        for f in pending:
            f.result()
    finally:
        ex.shutdown(wait=True)


def tile_xy_indices(xyz_t, chunk_xy, stride):
    """Yield (x0, y0, idx) for every non-empty overlapping chunk_xy tile of a
    torch (N,3+) tensor; idx stays on xyz_t's device so callers gather with no
    round-trips (put the scene on CUDA once and the whole loop runs there).
    Points are sorted by x once so each column strip is a searchsorted slice,
    then each strip is sorted by y so tiles are slices too — replaces the old
    per-tile full-cloud boolean mask (O(tiles*N) -> O(N log N)). Tile origins
    come from np.arange exactly as before, so cached-tile filenames match."""
    import numpy as np
    import torch
    x, y = xyz_t[:, 0], xyz_t[:, 1]
    mins = (float(x.min()), float(y.min()))
    maxs = (float(x.max()), float(y.max()))
    ox = torch.argsort(x)
    xs = x[ox]
    for x0 in np.arange(mins[0], maxs[0], stride):
        lo, hi = torch.searchsorted(
            xs, torch.tensor([x0, x0 + chunk_xy], dtype=xs.dtype,
                             device=xs.device)).tolist()
        strip = ox[lo:hi]
        if len(strip) == 0:
            continue
        strip = strip[torch.argsort(y[strip])]
        ys = y[strip].contiguous()
        for y0 in np.arange(mins[1], maxs[1], stride):
            a, b = torch.searchsorted(
                ys, torch.tensor([y0, y0 + chunk_xy], dtype=ys.dtype,
                                 device=ys.device)).tolist()
            if b > a:
                yield x0, y0, strip[a:b]


def kp_tile_and_save(name, pc_path, out_dir, chunk_xy, stride, grid, num_classes):
    """One scene -> overlapping chunk_xy tiles, grid-subsampled and cached as
    .npz (xyz + intensity + ret_num [+ every feat_* the scene carries,
    mean-pooled like the other attrs] + lab). Returns the tile count, or None
    when the scene failed to load (left unmarked so it retries)."""
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    try:                              # canonical .npz (label + feat_* embedded)
        xyz, intensity, ret_num, lab, extras = kp_load_canonical(pc_path)
    except Exception as e:
        print(f"  skip {pc_path}: {e}", flush=True)
        return None
    intensity_n = np.clip(intensity, 0.0, 2.0).astype(np.float32)
    print(f"    {name}: {len(xyz):,} pts loaded in {time.time()-t0:.1f}s, tiling…",
          flush=True)
    # GPU-resident tiling: the scene's arrays move to the device ONCE; strip
    # slicing, per-tile gathers and voxel pooling all run there, and only the
    # small pooled tiles come back for the (thread-pooled) compressed writes.
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fnames = sorted(extras)              # deterministic cache column order
    P = torch.from_numpy(xyz).to(dev)
    A = torch.from_numpy(np.stack([intensity_n, ret_num]
                                  + [extras[n] for n in fnames],
                                  axis=1)).to(dev)
    L = torch.from_numpy(np.ascontiguousarray(lab)).to(dev)
    n_tiles = 0
    with npz_save_pool() as save:
        for x0, y0, idx in tile_xy_indices(P, chunk_xy, stride):
            # Low thresholds on purpose: water absorbs LiDAR, so pure-water
            # tiles are sparse — a higher cut would delete them from training.
            if len(idx) < 64:
                continue
            sx, sa, sl = _grid_pool_t(P[idx], A[idx], L[idx], grid, num_classes)
            if len(sx) < 32:
                continue
            sa = sa.cpu().numpy()
            tile = dict(
                xyz=sx.cpu().numpy(),
                intensity=sa[:, 0],
                ret_num=sa[:, 1],
            )
            for i, n in enumerate(fnames):
                tile[n] = sa[:, 2 + i]
            tile["lab"] = sl.cpu().numpy().astype(np.int32)
            save(os.path.join(out_dir, f"{name}_x{int(x0)}_y{int(y0)}.npz"), **tile)
            n_tiles += 1
    print(f"      -> {n_tiles} tiles", flush=True)
    return n_tiles


def kp_ensure_prep(prep_dir, ds_root, sig, tile_fn):
    """Idempotent prep for the KP twins: split folders read verbatim, cache
    signature validated (validate_cache), each un-.done scene tiled via
    tile_fn(name, pc_path, out_dir, split) — split so the caller can widen
    the train stride (train_stride) while val/test keep the voting overlap.
    Returns (train, val, test) scene lists."""
    print(f"  ensuring preprocessed cache -> {prep_dir}", flush=True)
    for split in ("train", "val", "test"):
        os.makedirs(f"{prep_dir}/{split}", exist_ok=True)
    train_list, val_list, test_list = split_scenes(ds_root)
    any_new = [validate_cache(
        prep_dir, sig,
        [("train", train_list), ("val", val_list), ("test", test_list)],
        lambda d, name: (f"{d}/{name}_x*.npz", f"{d}/{name}.done"))]

    def tile_remaining(items, out_dir, split):
        for name, pc_path, _cls in items:
            if os.path.exists(f"{out_dir}/{name}.done"):
                continue
            n = tile_fn(name, pc_path, out_dir, split)
            if n is not None:          # None == load failed; leave unmarked to retry
                open(f"{out_dir}/{name}.done", "w").close()
            any_new[0] = True

    for split, items in (("train", train_list), ("val", val_list), ("test", test_list)):
        print(f"  [{split}] {len(items)} scenes", flush=True)
        tile_remaining(items, f"{prep_dir}/{split}", split)
    print("  preprocessing cache updated." if any_new[0]
          else "  all scenes already cached.", flush=True)
    return train_list, val_list, test_list


def kp_make_build_feat(logdk_feat, logdk_k,
                       spec=("intensity", "return_number", "height")):
    """build_feat(xyz, intensity, ret_num, drop=False, extras=None)
    -> [1, *spec] (+ log d_k when D3b is on). The constant-1 bias is ALWAYS
    first and not part of the spec (default spec = the legacy
    [intensity, return_number, height] recipe).

    "height" is always tile-relative: z - min(z) over the tile. Real
    HeightAboveGround is the ordinary feat_hag channel, fed via `extras`
    like every feat_<name> dataset channel. With `drop`, zero the SENSOR
    channels (intensity / return_number / rgb) wherever they sit in the
    spec; geometry-derived channels (height, x/y/z, feat_*) always survive
    — they come from the coords and are never missing at inference."""
    import numpy as np
    import density as dg
    spec = list(spec)
    # ponytail: sensor set is the fixed names; a sensor-derived feat_* channel
    # (e.g. NDVI from a source dim) would need a per-channel flag in the
    # dataset catalog to join the drop.
    drop_idx = [i for i, n in enumerate(spec)
                if n in ("intensity", "return_number", "rgb")]

    def build_feat(xyz, intensity, ret_num, drop=False, extras=None):
        bias = np.ones((len(xyz), 1), np.float32)
        height = (xyz[:, 2] - xyz[:, 2].min()).astype(np.float32)
        src = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "height": height,
               "intensity": intensity, "return_number": ret_num,
               **(extras or {})}
        missing = [n for n in spec if n not in src]
        if missing:
            raise ValueError(f"feature channel(s) {missing} not available "
                             f"here; have {sorted(src)}")
        attrs = np.stack([src[n] for n in spec], axis=1).astype(np.float32)
        if drop and drop_idx:
            attrs[:, drop_idx] = 0.0   # feature-drop: sensor channels only
        cols = [bias, attrs]
        if logdk_feat:           # D3b: never dropped — the density signal to condition on
            cols.append(dg.local_density_logdk(xyz, logdk_k)[:, None])
        return np.concatenate(cols, axis=1).astype(np.float32)

    build_feat.spec = spec       # the KP helpers read which feat_* they must load
    return build_feat


def kp_make_sample_tile(build_feat, grid, max_pts, aug_color,
                        density_aug, coarsen_max, p_native):
    """sample_tile(tile_path, max_pts=None, min_pts=32, training=True) ->
    (augmented+centered xyz, features, labels) or None. Height comes from the
    original (pre-augmentation) z so it stays a meaningful
    height-above-tile-min feature."""
    import numpy as np
    import density as dg

    def sample_tile(tile_path, max_pts=max_pts, min_pts=32, training=True):
        z = np.load(tile_path)
        xyz, intensity, ret_num, lab = z["xyz"], z["intensity"], z["ret_num"], z["lab"]
        extras = feat_extras(z, build_feat.spec, os.path.basename(tile_path))
        if len(xyz) < min_pts:
            return None
        idx = np.arange(len(xyz))
        if len(idx) > max_pts:
            idx = np.random.choice(idx, max_pts, replace=False)
        xyz, intensity, ret_num, lab = xyz[idx], intensity[idx], ret_num[idx], lab[idx]
        extras = {n: v[idx] for n, v in extras.items()}
        # D1 density jitter: coarsen-only re-subsample; index-consistent across
        # all per-point arrays.
        if training and density_aug:
            g_eff = dg.effective_grid(grid, coarsen_max, p_native)
            if g_eff > grid:
                keep = dg.voxel_first_idx(xyz, g_eff)
                xyz, intensity, ret_num, lab = xyz[keep], intensity[keep], ret_num[keep], lab[keep]
                extras = {n: v[keep] for n, v in extras.items()}
        drop = (training and np.random.rand() > aug_color)
        feat = build_feat(xyz, intensity, ret_num, drop=drop, extras=extras)
        geo_xyz = kp_augment(xyz) if training else xyz
        geo_xyz = (geo_xyz - geo_xyz.mean(0)).astype(np.float32)
        return geo_xyz, feat, lab.astype(np.int64)

    return sample_tile


def kp_make_run_dir(variant):
    """Fresh timestamped run dir: <OUTPUTS_ROOT>/runs/<utc>_<variant>."""
    from datetime import datetime
    run_id = datetime.utcnow().strftime(f"%Y%m%d_%H%M%S_{variant}")
    run_dir = f"{OUTPUTS_ROOT}/runs/{run_id}"
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    return run_id, run_dir


def kp_find_latest_checkpoint(opt_type, feature_modes, arch_hash=None,
                              features=None, legacy_features=None,
                              skip_done=False):
    """Most recent run (run-ids are timestamps, so they sort) with checkpoints
    AND this script's recipe: optimizer type, feature_mode (the trainer's raw
    run.json value), the ordered feature spec (run.json "features"; a run
    without one is a legacy run = `legacy_features`) and, when given, the
    architecture hash. Same-width specs can differ in channel SEMANTICS
    (feat_hag vs tile height), so the names must match, not just the count.
    Returns (run_dir, ckpt_path, epoch) or None."""
    import glob

    def _ep(p):
        return int(os.path.basename(p)[2:5])   # ep149.pth -> 149

    for rd in sorted(glob.glob(f"{OUTPUTS_ROOT}/runs/*"), reverse=True):
        # skip_done: resume must not pick a COMPLETED run (find_latest_
        # unfinished_run's rule); EVAL_ONLY passes False because it *wants*
        # finished runs. Pre-DONE-marker runs are indistinguishable from
        # crashed ones and stay eligible — same behavior as before.
        if skip_done and os.path.exists(f"{rd}/DONE"):
            continue
        ckpts = glob.glob(f"{rd}/checkpoints/ep*.pth")
        if not ckpts:
            continue
        got_opt = fmode = ahash = None
        rc = {}
        for cfgp in (f"{rd}/run.json", f"{rd}/run_config.json"):   # legacy fallback
            try:
                with open(cfgp) as f:
                    rc = json.load(f)
                got_opt = rc.get("optimizer", {}).get("type")
                fmode = rc.get("feature_mode")
                ahash = rc.get("arch_hash")
                break
            except Exception:
                rc = {}
                continue
        if got_opt != opt_type:
            print(f"  resume: skipping {os.path.basename(rd)} "
                  f"(recipe mismatch: optimizer={got_opt})", flush=True)
            continue
        if fmode not in feature_modes:
            print(f"  resume: skipping {os.path.basename(rd)} "
                  f"(variant mismatch: feature_mode={fmode})", flush=True)
            continue
        if rc.get("hag_source"):
            # Deleted *_hag variant: its 'height' channel was real HAG, which the
            # native spec can't reproduce — same width, different semantics.
            print(f"  resume: skipping {os.path.basename(rd)} "
                  f"(legacy --hag run: hag_source={rc['hag_source']})", flush=True)
            continue
        if features is not None:
            got_feats = list(rc.get("features") or legacy_features or features)
            if got_feats != list(features):
                print(f"  resume: skipping {os.path.basename(rd)} "
                      f"(feature mismatch: {got_feats})", flush=True)
                continue
        if arch_hash is not None and ahash is not None and ahash != arch_hash:
            print(f"  resume: skipping {os.path.basename(rd)} "
                  f"(architecture mismatch: arch_hash={ahash})", flush=True)
            continue
        latest = max(ckpts, key=_ep)
        return rd, latest, _ep(latest)
    return None


def kp_make_evaluate(forward, build_feat, grid, chunk_xy, num_classes,
                     class_names):
    """The KP twins' voted eval, scored on the ORIGINAL raw points: per scene,
    run the model over its overlapping cached tiles (EVAL_BATCH tiles per
    forward), sum center-weighted softmax votes per voxel, argmax, then
    propagate to every raw point by nearest neighbour and score against the
    raw GT. forward(tiles) takes [(cxyz, feat)] and returns the per-tile
    (N, C) logits list; a failed tile is skipped."""
    import glob
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    def evaluate(scene_items, label):
        bs = max(1, int(os.environ.get("EVAL_BATCH", "4")))
        t_inter = np.zeros(num_classes, dtype=np.int64)
        t_union = np.zeros(num_classes, dtype=np.int64)
        t_gt = np.zeros(num_classes, dtype=np.int64)
        correct = total = 0
        n_scenes = n_skipped_tiles = n_skipped_scenes = 0
        t_test = time.time()

        def forward_group(group):
            # A failed batched forward falls back to tile-by-tile (an OOM also
            # halves bs for the rest of this pass); a tile that still fails
            # alone is skipped (None), the pre-batching per-tile semantics.
            nonlocal bs
            if len(group) > 1:
                try:
                    return forward([(c, f) for _, c, f in group])
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        bs = max(1, bs // 2)
            outs = []
            for _, c, f in group:
                try:
                    outs.append(forward([(c, f)])[0])
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                    outs.append(None)
            return outs

        with torch.no_grad():
            for name, pc_path, _cls, split_dir in scene_items:
                tiles = sorted(glob.glob(f"{split_dir}/{name}_x*.npz"))
                if not tiles:
                    n_skipped_scenes += 1
                    continue
                keys_l, log_l, xyz_l = [], [], []
                group = []

                def flush():
                    nonlocal n_skipped_tiles
                    for (xyz, _, _), lg in zip(group, forward_group(group)):
                        if lg is None:
                            n_skipped_tiles += 1
                            continue
                        # Soft votes tapered toward the tile border (truncated
                        # context).
                        e = np.exp(lg - lg.max(1, keepdims=True))
                        prob = e / e.sum(1, keepdims=True)
                        cxy = (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
                        d = np.abs(xyz[:, :2] - cxy).max(1)
                        wgt = np.clip(1.0 - d / (chunk_xy / 2.0), 0.05, 1.0) ** 2
                        keys_l.append(np.floor(xyz / grid).astype(np.int64))
                        log_l.append((prob * wgt[:, None]).astype(np.float32))
                        xyz_l.append(xyz.astype(np.float32))
                    group.clear()

                for tile in tiles:
                    try:
                        z = np.load(tile)
                        xyz = z["xyz"]
                    except Exception:
                        drop_corrupt_tile(tile)   # healed on the next prep
                        n_skipped_tiles += 1
                        continue
                    if len(xyz) < 32:
                        continue
                    feat = build_feat(xyz, z["intensity"], z["ret_num"],
                                      extras=feat_extras(z, build_feat.spec,
                                                         os.path.basename(tile)))
                    cxyz = (xyz - xyz.mean(0)).astype(np.float32)
                    group.append((xyz, cxyz, feat))
                    if len(group) >= bs:
                        flush()
                flush()
                if not keys_l:
                    n_skipped_scenes += 1
                    continue
                K = np.concatenate(keys_l); L = np.concatenate(log_l)
                P = np.concatenate(xyz_l)
                first, inv = voxel_unique(K, return_inverse=True)
                votes = np.zeros((len(first), num_classes), np.float64)
                np.add.at(votes, inv, L)
                pred_u = votes.argmax(1)
                rep_xyz = P[first]                      # one representative coord per voxel
                try:
                    raw_xyz, _, _, raw_lab, _ = kp_load_canonical(pc_path)
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped_scenes += 1
                    continue
                _, nn = cKDTree(rep_xyz).query(raw_xyz, workers=-1)
                raw_pred = pred_u[nn]
                v = raw_lab >= 0
                rp, rl = raw_pred[v], raw_lab[v]
                correct += int((rp == rl).sum()); total += int(v.sum())
                i_, u_, g_ = score_ious(rp, rl, num_classes)
                t_inter += i_; t_union += u_; t_gt += g_
                n_scenes += 1
        return eval_metrics(
            t_inter, t_union, t_gt, correct, total, class_names, t_test,
            n_scenes, label,
            extra={"skipped_tiles": n_skipped_tiles,
                   "skipped_scenes": n_skipped_scenes,
                   "scored_on": "raw_points",
                   "voted_overlap": True,
                   "vote_weighting": "center_tapered_softmax",
                   "reprojection": "nearest_voxel_representative_to_raw"})

    return evaluate


def kp_make_predict_points(forward_prob, build_feat, grid, chunk_xy,
                           num_classes, tta, save_probs=False, exclude_idx=None):
    """Sliding-window inference over already-normalized features (--mode infer);
    returns (pred, confidence, probs) per raw point — confidence is the max of
    the view-summed softmax normalized to sum 1 (0.0 where nothing was
    predicted), probs the float16 normalized distribution when save_probs else
    None. forward_prob(cxyz, feat) -> (N, C) softmax ndarray; an exception
    skips the window. D5 density-TTA: average softmax over `tta` extra
    density(scale) views. exclude_idx (EXCLUDE_CLASSES): classes masked out of
    the distribution before argmax — conf/probs are post-mask."""
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    feat_names = [n for n in getattr(build_feat, "spec", []) if n.startswith("feat_")]

    def predict_points(xyz, intensity_n, ret_num, extras=None):
        pred = np.full(len(xyz), -1, np.int64)
        conf = np.zeros(len(xyz), np.float32)
        probs = np.zeros((len(xyz), num_classes), np.float16) if save_probs else None
        n_done = n_skipped = 0
        last_err = None
        with torch.no_grad():
            for idx in xy_chunk_groups(xyz, chunk_xy, min_pts=64):
                cols = [intensity_n[idx], ret_num[idx]]
                cols += [extras[n][idx] for n in feat_names]   # feat_* ride along too
                attrs = np.stack(cols, axis=1).astype(np.float32)
                sx, sa, _ = kp_grid_subsample(xyz[idx], attrs, None, grid, num_classes)
                if len(sx) < 32:
                    continue
                sub_ex = {n: sa[:, 2 + i] for i, n in enumerate(feat_names)}
                feat = build_feat(sx, sa[:, 0], sa[:, 1], extras=sub_ex)
                base = (sx - sx.mean(0)).astype(np.float32)
                views = [1.0] + (list(np.linspace(0.85, 1.2, tta)) if tta else [])
                try:
                    prob = None
                    for s in views:
                        p = forward_prob((base * s).astype(np.float32), feat)
                        prob = p if prob is None else prob + p
                    # view sums exceed 1 — renormalize to a distribution
                    prob /= np.maximum(prob.sum(-1, keepdims=True), 1e-12)
                    prob = apply_class_mask(prob, exclude_idx)
                    sub_pred = prob.argmax(-1)
                except Exception as ex:
                    n_skipped += 1
                    last_err = ex
                    continue
                _, nn = cKDTree(sx).query(xyz[idx])
                pred[idx] = sub_pred[nn]
                conf[idx] = prob.max(-1)[nn]
                if save_probs:
                    probs[idx] = prob[nn].astype(np.float16)
                n_done += 1
        if n_skipped:
            print(f"  [infer] WARNING: {n_skipped} window(s) failed "
                  f"(last error: {last_err})", flush=True)
        if n_skipped and not n_done:
            # every window the scene HAD errored — filling would ship a
            # uniform, confidence-0 label map as if it were a prediction.
            raise RuntimeError(
                f"inference produced nothing: all {n_skipped} window(s) "
                f"failed (last error: {last_err})")
        miss = pred < 0
        if miss.any() and (~miss).any():
            _, nn = cKDTree(xyz[~miss]).query(xyz[miss])
            pred[miss] = pred[~miss][nn]           # conf/probs stay 0: no votes
        elif miss.any():
            # nothing predicted anywhere (tiny scene: every window skipped) —
            # fall back to the lowest NON-excluded class so EXCLUDE_CLASSES
            # holds even here; conf stays 0 so the export threshold flags it.
            pred[:] = min(set(range(num_classes)) - set(exclude_idx or ()))
        return pred, conf, probs

    return predict_points


def kp_make_target_batches(scenes, make_batch, build_feat, grid,
                           chunk_xy, num_classes, cap=30):
    """AdaBN (D2b) target-batch generator over inference scenes: same windows,
    subsample and features predict_points will see. make_batch(cxyz, feat) ->
    model batch; an exception skips the window."""
    import numpy as np

    feat_names = [n for n in getattr(build_feat, "spec", []) if n.startswith("feat_")]

    def target_batches():
        seen = 0
        for pc_path in scenes:
            if seen >= cap:
                return
            z = np.load(pc_path)
            # scene-local frame, matching what predict_points is fed
            txyz = (z["xyz"] - np.floor(z["xyz"].min(0))).astype(np.float32)
            tin, trn = scene_arrays(z, len(txyz))
            tex = feat_extras(z, feat_names, os.path.basename(pc_path))
            for idx in xy_chunk_groups(txyz, chunk_xy, min_pts=64):
                if seen >= cap:
                    return
                cols = ([tin[idx], trn[idx]]
                        + [tex[n][idx] for n in feat_names])
                attrs = np.stack(cols, 1).astype(np.float32)
                sx, sa, _ = kp_grid_subsample(txyz[idx], attrs, None, grid, num_classes)
                if len(sx) < 32:
                    continue
                # BN stats must see the same feature predict will be fed.
                sub_ex = {n: sa[:, 2 + i] for i, n in enumerate(feat_names)}
                feat = build_feat(sx, sa[:, 0], sa[:, 1], extras=sub_ex)
                cxyz = (sx - sx.mean(0)).astype(np.float32)
                try:
                    b = make_batch(cxyz, feat)
                except Exception:
                    continue
                seen += 1
                yield b

    return target_batches()


# ============================================================================
# PTv3-family shared pipeline — the prep/augment/eval core the ptv3 and
# pcssl (concerto/sonata/utonia) trainers used to duplicate. Same pattern as
# the kp_* family above: trainers pass small closures for the model-specific
# parts (color_src-aware load_canonical, build_feat, forward).
# ============================================================================

def ptv3_load_canonical(npz_path, color_src):
    """Canonical trainer_gui scene -> an (xyz, rgb, lab) tuple.
    color_src picks what fills the 3 color channels: intensity-first for new
    runs, rgb for old RGB-trained checkpoints at inference. Missing signals
    fall through (intensity -> rgb -> mid-gray). Intensity is npz-normalized
    (0..1 max / 0..2 p95); x255 puts it on the rgb scale the /255 sites expect.
    xyz is origin-offset (per-scene floor-min, kp_load_canonical's pattern)
    before the float32 cast: projected (UTM) coords are ~1e6 in magnitude,
    where float32 spacing is 0.5m AND a float32 mean is off by hundreds of
    meters — enough to blow the |xy|<=CHUNK_XY batch filter and empty every
    training batch."""
    import numpy as np
    z = np.load(npz_path)
    xyz = (z["xyz"] - np.floor(z["xyz"].min(0))).astype(np.float32)
    def _itn():
        return np.repeat((z["intensity"].astype(np.float32) * 255.0)[:, None], 3, axis=1)
    if color_src != "rgb" and "intensity" in z:
        rgb = _itn()
    elif "rgb" in z:
        rgb = z["rgb"].astype(np.float32)
    elif "intensity" in z:
        rgb = _itn()
    else:
        rgb = np.full((len(xyz), 3), 128.0, dtype=np.float32)
    lab = z["label"].astype(np.int64) if "label" in z \
        else np.full(len(xyz), -1, np.int64)
    # p95-normalized intensity spans [0,2] -> x255 up to 510: clip HERE so
    # the uint8 tile cache (train) and the float path (infer) agree — an
    # unclipped cast WRAPS the bright tail (306 -> 50).
    return xyz, np.clip(rgb, 0.0, 255.0), lab


def ptv3_tile_and_save(src_paths, out_dir, chunk_xy, stride, load_canonical):
    """Scenes -> overlapping chunk_xy tiles cached as .npz (xyz + the 3 baked
    color channels + lab + every feat_* the scene carries)."""
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    for fi, src in enumerate(src_paths):
        scene = os.path.splitext(os.path.basename(src))[0]
        t0 = time.time()
        try:
            xyz, rgb, lab = load_canonical(src)
        except Exception as e:
            print(f"  skip {src}: {e}", flush=True); continue
        # Every feat_* channel the scene carries rides into the cache tiles
        # (feat_hag included — HAG is an ordinary feature channel; legacy
        # bare-'hag' scenes surface it via scene_feats).
        extras = scene_feats(np.load(src)) if src.endswith(".npz") else {}
        print(f"    [{fi+1}/{len(src_paths)}] {scene}: {len(xyz):,} pts "
              f"loaded in {time.time()-t0:.1f}s, tiling…", flush=True)
        # ptv3 tiles are RAW slices (no pooling), so only the tile-index math
        # runs on the device; gathers stay on the host arrays.
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        P = torch.from_numpy(np.ascontiguousarray(xyz)).to(dev)
        n_tiles = 0
        with npz_save_pool() as save:
            for x0, y0, idx in tile_xy_indices(P, chunk_xy, stride):
                if len(idx) < 2048: continue
                m = idx.cpu().numpy()
                tile = {"xyz": xyz[m].astype(np.float32),
                        "rgb": rgb[m].astype(np.uint8),   # load_canonical clips to [0,255]
                        "lab": lab[m].astype(np.int32)}
                for n, v in extras.items():
                    tile[n] = v[m].astype(np.float32)
                save(f"{out_dir}/{scene}_x{int(x0)}_y{int(y0)}.npz", **tile)
                n_tiles += 1
        print(f"      -> {n_tiles} tiles", flush=True)


def ptv3_ensure_prep(prep_dir, ds_root, chunk_xy, stride, load_canonical):
    # Per-scene idempotency via prefix-match on existing tiles.
    import glob
    os.makedirs(f"{prep_dir}/train", exist_ok=True)
    os.makedirs(f"{prep_dir}/val",   exist_ok=True)
    os.makedirs(f"{prep_dir}/test",  exist_ok=True)
    print(f"  ensuring preprocessed cache -> {prep_dir}", flush=True)
    # PREP_DIR encodes color/feat/chunk/stride but NOT the class layout, and a
    # dataset rebuilt under the same name keeps this folder — so stamp the
    # signature (KP/RandLA's validate_cache) or reordered classes silently
    # reuse tiles labeled in the OLD index space. Empty lists: ptv3 keeps its
    # glob idempotency, there are no .done markers to migrate.
    meta = {}
    try:
        with open(f"{ds_root}/dataset_meta.json") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        pass
    sp = meta.get("split", {}) if isinstance(meta.get("split"), dict) else {}
    validate_cache(prep_dir, {"pipeline": "ptv3",
                              "chunk_xy": chunk_xy,
                              "stride": stride,
                              "train_stride": train_stride(chunk_xy),
                              "num_classes": meta.get("num_classes"),
                              "class_names": meta.get("class_names"),
                              "split_seed": sp.get("seed"),
                              "split_mode": sp.get("mode")}, [], None)
    any_new = [False]
    def already_tiled(out_dir, scene):
        return bool(glob.glob(f"{out_dir}/{scene}_x*.npz"))
    def tile_remaining(src_paths, out_dir, chunk, stride):
        for src in src_paths:
            scene = os.path.splitext(os.path.basename(src))[0]
            if already_tiled(out_dir, scene): continue
            ptv3_tile_and_save([src], out_dir, chunk, stride, load_canonical)
            any_new[0] = True
    # CANONICAL: the dataset stage already materialized the 3-way split;
    # tile each folder into its own PREP folder verbatim (no re-carving).
    train_paths = sorted(glob.glob(f"{ds_root}/train/*.npz"))
    val_paths   = sorted(glob.glob(f"{ds_root}/val/*.npz"))
    test_paths  = sorted(glob.glob(f"{ds_root}/test/*.npz"))
    if not train_paths:
        raise FileNotFoundError(f"No canonical scenes under {ds_root}/train")
    print(f"  [train] {len(train_paths)} canonical scenes", flush=True)
    tile_remaining(train_paths, f"{prep_dir}/train", chunk_xy,
                   train_stride(chunk_xy))
    # stride (not chunk_xy) so val/test tiles overlap and the final
    # eval can vote per-voxel over the up-to-4 covering tiles.
    print(f"  [val] {len(val_paths)} canonical scenes", flush=True)
    tile_remaining(val_paths, f"{prep_dir}/val", chunk_xy, stride)
    print(f"  [test] {len(test_paths)} canonical scenes", flush=True)
    tile_remaining(test_paths, f"{prep_dir}/test", chunk_xy, stride)
    if any_new[0]:
        print("  preprocessing cache updated.", flush=True)
    else:
        print("  all scenes already cached.", flush=True)


def ptv3_check_spec(spec, arch):
    """FEAT_CHANNELS sanity for the PTv3-family input layout: one 3-wide
    rgb-OR-intensity color slot, x/y/z, dataset feat_* channels."""
    bad = [n for n in spec
           if n not in ("x", "y", "z", "rgb", "intensity")
           and not n.startswith("feat_")]
    if bad:
        raise ValueError(f"{arch} can't feed {bad}; supported: x, y, z, "
                         f"rgb/intensity (one 3-wide color slot) plus "
                         f"dataset feat_* channels")
    if "rgb" in spec and "intensity" in spec:
        raise ValueError(f"{arch} has ONE 3-wide color slot — pick rgb OR "
                         f"intensity in FEAT_CHANNELS, not both")


def ptv3_augment_xyz(xyz, rot_z, rot_xy, scale_min, scale_max, flip_p,
                     jitter_sigma, jitter_clip):
    """PTv3 outdoor augmentation suite: full z-yaw, gentle x/y tilt,
    isotropic scale, per-axis flip, jitter."""
    import numpy as np
    az = (np.random.rand() * 2 - 1) * np.pi * rot_z
    ax = (np.random.rand() * 2 - 1) * np.pi * rot_xy
    ay = (np.random.rand() * 2 - 1) * np.pi * rot_xy
    cz, sz = np.cos(az), np.sin(az)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    out = xyz @ (Rz @ Ry @ Rx).T
    out = out * np.random.uniform(scale_min, scale_max)
    if np.random.rand() < flip_p:
        out[:, 0] = -out[:, 0]
    if np.random.rand() < flip_p:
        out[:, 1] = -out[:, 1]
    out += np.clip(np.random.normal(0, jitter_sigma, out.shape),
                   -jitter_clip, jitter_clip)
    return out.astype(np.float32)


def ptv3_lr_at(ep, base_lr, warmup_pct, n_epochs):
    """PTv3 outdoor schedule: short linear warmup to base_lr, then cosine
    decay to ~0. Set on the param groups each epoch — no scheduler state to
    restore, so RESUME is trivial (the KPConvX cold recipe's lr_at pattern)."""
    import numpy as np
    warm = max(1, int(round(warmup_pct * n_epochs)))
    if ep < warm:
        return base_lr * (ep + 1) / warm
    prog = (ep - warm) / max(1, n_epochs - warm)
    return float(0.5 * base_lr * (1.0 + np.cos(np.pi * prog)))


RESUME_RECIPE_KEYS = ("grid_size", "chunk_xy", "features", "n_epochs",
                      "num_classes", "class_names")


def find_latest_unfinished_run(suffix, cfg=None):
    """Latest UNFINISHED run dir ending in `suffix` with checkpoints (resume
    target). Runs marked DONE are skipped, so a completed experiment isn't
    re-resumed on the next launch — but a crashed/retried one is picked
    straight back up. `cfg` is the fresh run's config dict: a candidate whose
    run.json disagrees on RESUME_RECIPE_KEYS is skipped (kp_find_latest_
    checkpoint's rule), so a resumed run never keeps publishing a run.json its
    weights don't match — on mismatch the caller falls through to a fresh run
    and writes a correct one. Returns (run_dir, ckpt_path, epoch) or None."""
    import glob
    def _ep(p):
        return int(os.path.basename(p)[2:5])
    def _n(v):                      # run.json round-trips tuples as lists
        return list(v) if isinstance(v, (list, tuple)) else v
    for rd in sorted(glob.glob(f"{OUTPUTS_ROOT}/runs/*"), reverse=True):
        if not rd.endswith(suffix):
            continue
        if os.path.exists(f"{rd}/DONE"):
            continue
        if cfg is not None:
            try:
                with open(f"{rd}/run.json") as f:
                    rc = json.load(f)
            except (OSError, ValueError):
                rc = {}
            bad = {k: rc.get(k) for k in RESUME_RECIPE_KEYS
                   if _n(rc.get(k)) != _n(cfg.get(k))}
            if bad:
                print(f"  resume: skipping {os.path.basename(rd)} "
                      f"(recipe mismatch: {bad})", flush=True)
                continue
        ckpts = glob.glob(f"{rd}/checkpoints/ep*.pth")
        if ckpts:
            latest = max(ckpts, key=_ep)
            return rd, latest, _ep(latest)
    return None


def ptv3_make_evaluate(forward, build_feat, feat_spec, grid, chunk_xy,
                       num_classes, class_names):
    """The PTv3-family voted eval. Per-SCENE overlap voting scored on the
    ORIGINAL raw points (the protocol KPConvX/RandLA use). Per scene: forward
    its overlapping tiles (stride chunk_xy/2, each point in up to 4 tiles;
    EVAL_BATCH tiles per forward via the multi-entry offset), sum
    center-tapered softmax votes per grid voxel, argmax, then NN-propagate
    each voxel's prediction to the raw cloud and score against raw GT.
    forward(batch_dict) -> (N, C) logits tensor; scene_items are
    (name, load_raw, split_dir) triples."""
    import glob
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    def evaluate(scene_items, label):
        bs = max(1, int(os.environ.get("EVAL_BATCH", "4")))
        t_inter = np.zeros(num_classes, dtype=np.int64)
        t_union = np.zeros(num_classes, dtype=np.int64)
        t_gt    = np.zeros(num_classes, dtype=np.int64)
        correct = total = 0
        n_scenes = n_skipped_tiles = n_skipped_scenes = 0
        t_test = time.time()

        def _run(group):
            # gc stays per-tile min-subtracted (NOT the training path's global
            # min): identical serialization codes to the unbatched call, and
            # the offset-derived batch index keeps clouds apart in spconv.
            lens = [len(g[2]) for g in group]
            coord = torch.from_numpy(np.concatenate([g[2] for g in group])).cuda()
            featt = torch.from_numpy(np.concatenate([g[3] for g in group])).cuda()
            gc = np.ascontiguousarray(np.concatenate([g[4] for g in group]))
            grid_coord = torch.from_numpy(gc).long().cuda()
            offset = torch.tensor(np.cumsum(lens), dtype=torch.long).cuda()
            lg = forward({"coord": coord, "grid_coord": grid_coord,
                          "feat": featt, "offset": offset}
                         ).cpu().numpy().astype(np.float32)
            return np.split(lg, np.cumsum(lens)[:-1])

        def forward_group(group):
            # OOM on the batched forward halves bs for the rest of this pass
            # and retries tile-by-tile; a tile that still OOMs alone is
            # skipped. Non-OOM RuntimeErrors re-raise (abort eval), as before.
            nonlocal bs
            if len(group) > 1:
                try:
                    return _run(group)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
            outs = []
            for g in group:
                try:
                    outs.append(_run([g])[0])
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    outs.append(None)
            return outs

        with torch.no_grad():
            for name, load_raw, split_dir in scene_items:
                tiles = sorted(glob.glob(f"{split_dir}/{name}_x*.npz"))
                if not tiles:
                    n_skipped_scenes += 1; continue
                keys_l, vote_l, xyz_l = [], [], []
                group = []

                def flush():
                    nonlocal n_skipped_tiles
                    for (xyz, inverse, *_), lg in zip(group, forward_group(group)):
                        if lg is None:
                            n_skipped_tiles += 1
                            continue
                        e = np.exp(lg - lg.max(1, keepdims=True))
                        prob = (e / e.sum(1, keepdims=True))[inverse]  # per original pt
                        cxy = (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
                        d = np.abs(xyz[:, :2] - cxy).max(1)
                        wgt = np.clip(1.0 - d / (chunk_xy / 2.0), 0.05, 1.0) ** 2
                        keys_l.append(np.floor(xyz / grid).astype(np.int64))
                        vote_l.append((prob * wgt[:, None]).astype(np.float32))
                        xyz_l.append(xyz.astype(np.float32))
                    group.clear()

                for tile in tiles:
                    try:
                        z = np.load(tile)
                        xyz, rgb = z["xyz"].astype(np.float32), z["rgb"]
                    except Exception:
                        drop_corrupt_tile(tile)   # healed on the next prep
                        n_skipped_tiles += 1
                        continue
                    if len(xyz) < 64:
                        continue
                    ex = feat_extras(z, feat_spec, os.path.basename(tile))
                    # float64 mean: exact centering even for legacy global-UTM
                    # tiles, where a float32 mean is hundreds of meters off
                    cxyz = (xyz - xyz.mean(0, keepdims=True, dtype=np.float64)
                            ).astype(np.float32)
                    ok = (np.isfinite(cxyz).all(1)
                          & (np.abs(cxyz[:, :2]).max(1) <= chunk_xy)
                          & (np.abs(cxyz[:, 2]) <= 200.0))
                    if int(ok.sum()) < 64:
                        continue
                    xyz, rgb, cxyz = xyz[ok], rgb[ok], cxyz[ok]
                    ex = {n: v[ok] for n, v in ex.items()}
                    vk = np.floor(cxyz / grid).astype(np.int64)
                    first, inverse = voxel_unique(vk, return_inverse=True)
                    vx = cxyz[first].astype(np.float32)
                    feat = build_feat(vx, rgb[first].astype(np.float32) / 255.0,
                                      {n: v[first] for n, v in ex.items()})
                    gc = vk[first] - vk[first].min(0)        # unique, dedup-consistent
                    group.append((xyz, inverse, vx, feat, gc))
                    if len(group) >= bs:
                        flush()
                flush()
                if not keys_l:
                    n_skipped_scenes += 1; continue
                K = np.concatenate(keys_l); V = np.concatenate(vote_l); P = np.concatenate(xyz_l)
                ufirst, uinv = voxel_unique(K, return_inverse=True)
                votes = np.zeros((len(ufirst), num_classes), np.float64)
                np.add.at(votes, uinv, V)
                pred_u  = votes.argmax(1)
                rep_xyz = P[ufirst]                      # one raw coord per voxel
                # Reproject voxel predictions onto the raw scene cloud + raw GT.
                try:
                    raw_xyz, _, raw_lab = load_raw()
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped_scenes += 1; continue
                _, nn = cKDTree(rep_xyz).query(raw_xyz, workers=-1)
                raw_pred = pred_u[nn]
                v = (raw_lab >= 0) & (raw_lab < num_classes)
                rp, rl = raw_pred[v], raw_lab[v]
                correct += int((rp == rl).sum()); total += int(v.sum())
                i_, u_, g_ = score_ious(rp, rl, num_classes)
                t_inter += i_; t_union += u_; t_gt += g_
                n_scenes += 1
        return eval_metrics(
            t_inter, t_union, t_gt, correct, total, class_names, t_test,
            n_scenes, label,
            extra={"skipped_tiles": n_skipped_tiles,
                   "skipped_scenes": n_skipped_scenes,
                   "scored_on": "raw_points",
                   "voted_overlap": True,
                   "vote_weighting": "center_tapered_softmax",
                   "reprojection": "nearest_voxel_representative_to_raw"})

    return evaluate


# ============================================================================
# Cross-trainer plumbing shared by all local trainers and modal shells
# (this file is baked into every image as /root/train_common.py).
# ============================================================================

# Input-feature spec vocabulary (FEAT_CHANNELS env / run.json "features").
# Widths: rgb = 3, everything else 1 (PTv3 additionally expands a
# single-channel color entry to 3 — its arch rule). HAG is the ordinary
# feat_hag dataset channel; only `logdk` stays outside the spec (driven by
# DG_LOGDK_FEAT, appended after the spec channels).
FEAT_VOCAB = ("x", "y", "z", "height", "intensity", "return_number", "rgb")


def parse_feat_spec(env_value, legacy_default):
    """FEAT_CHANNELS csv ("a,b,c") -> ordered channel-name list. Empty/unset ->
    the trainer's exact legacy default. Valid names: FEAT_VOCAB or feat_<name>
    (a dataset feature channel stored under that npz key)."""
    import re
    names = [s.strip() for s in (env_value or "").split(",") if s.strip()]
    if not names:
        return list(legacy_default)
    bad = [n for n in names if n not in FEAT_VOCAB
           and not re.fullmatch(r"feat_[A-Za-z0-9_]+", n)]
    if bad:
        raise ValueError(
            f"unknown FEAT_CHANNELS name(s) {bad}: valid names are "
            f"{list(FEAT_VOCAB)} or feat_<name> dataset channels")
    return names


def feat_spec_tag(spec, legacy):
    """Short PREP_DIR suffix for a non-default feature spec, so custom-spec
    prep caches never collide with (or invalidate) the legacy ones. "" when
    the spec IS the legacy default — every existing cache path stays valid."""
    import hashlib
    if list(spec) == list(legacy):
        return ""
    return "_f" + hashlib.sha1(",".join(spec).encode()).hexdigest()[:6]


def feat_extras(z, spec, where):
    """The feat_* arrays `spec` needs from an npz (tile or scene). A missing
    key is a clear error naming the channel and what IS available."""
    import numpy as np
    out = {}
    for n in spec:
        if not n.startswith("feat_"):
            continue
        if n not in z.files:
            if n == "feat_hag" and "hag" in z.files:   # pre-2026-07-13 scene key
                out[n] = z["hag"].astype(np.float32)
                continue
            avail = [k for k in z.files if k.startswith("feat_")]
            raise ValueError(
                f"{where} has no '{n}' channel (available feat_*: "
                f"{avail or 'none'}). Rebuild the dataset/prep cache with this "
                f"feature or drop it from FEAT_CHANNELS.")
        out[n] = z[n].astype(np.float32)
    return out

# One central table: trainer global name -> (env var, density.env_* parser).
# Trainers opt in by name via env_overrides(); unknown names fail loudly.
_ENV_KNOBS = {
    "DG_DENSITY_AUG":   ("DG_DENSITY_AUG",       "env_bool"),
    "DG_COARSEN_MAX":   ("DG_COARSEN_MAX",       "env_float"),
    "DG_P_NATIVE":      ("DG_P_NATIVE",          "env_float"),
    "DG_LOGDK_FEAT":    ("DG_LOGDK_FEAT",        "env_bool"),
    "DG_LOGDK_K":       ("DG_LOGDK_K",           "env_int"),
    "DG_INFER_ADABN":   ("DG_INFER_ADABN",       "env_bool"),
    "DG_INFER_TTA":     ("DG_INFER_TTA",         "env_int"),
    "EVAL_VOTES":       ("EVAL_VOTES",           "env_int"),
    "VAL_EVERY":        ("VAL_EVERY",            "env_int"),
    "USE_FOCAL":        ("LOSS_FOCAL",           "env_bool"),
    "FOCAL_GAMMA":      ("LOSS_FOCAL_GAMMA",     "env_float"),
    "CLASS_WEIGHTING":  ("LOSS_CLASS_WEIGHTING", "env_bool"),
    "WEIGHT_BETA":      ("LOSS_WEIGHT_BETA",     "env_float"),
    "RARE_OVERSAMPLE":  ("RARE_OVERSAMPLE",      "env_bool"),
    "RARE_CENTER_PROB": ("RARE_CENTER_PROB",     "env_float"),
    "KP_AGGREGATION":   ("KP_AGGREGATION",       "env_str"),
    "KP_NORM":          ("KP_NORM",              "env_str"),
    "FEAT_CHANNELS":    ("FEAT_CHANNELS",        "env_str"),
}


def env_overrides(g, names):
    """Env-overridable trainer knobs (the GUI's DG / loss / class-balance panels
    export DG_*/LOSS_*/RARE_*/EVAL_*/KP_* into the trainer env). Returns the
    values for `names` in order, each defaulting to g[name]:

        (DG_DENSITY_AUG, ..., RARE_OVERSAMPLE) = tc.env_overrides(globals(), [
            "DG_DENSITY_AUG", ..., "RARE_OVERSAMPLE"])
    """
    import density as dg   # sibling helper, baked into every image next to this file
    out = []
    for name in names:
        env_key, parser = _ENV_KNOBS[name]
        out.append(getattr(dg, parser)(env_key, g[name]))
    return tuple(out)


def load_dataset_meta(dataset):
    """Load <dataset_dir>/dataset_meta.json.
    Returns (ds_meta, num_classes, class_names)."""
    meta_path = f"{dataset_dir(dataset)}/dataset_meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found — build the dataset "
                                f"with the trainer_gui app first.")
    with open(meta_path) as f:
        ds_meta = json.load(f)
    return ds_meta, int(ds_meta["num_classes"]), list(ds_meta["class_names"])


def load_ckpt_safe(path, map_location="cpu"):
    """torch.load with weights_only=True (a hand-picked .pth can't run code on
    load) and the shared re-export hint on failure."""
    import torch   # lazy: this module stays importable on a torch-less host
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load weights '{path}': {e}\n"
            f"  (loaded safely with weights_only=True — a full-model pickle or a "
            f"checkpoint from another script is rejected; re-export as a state_dict.)"
        ) from e


def modal_shell_run(script, flag_vals, env_json, volumes):
    """The whole body of every modal_train_* shell's train function: build the
    trainer command from (flag, value) pairs (None skipped), merge --env-json
    overrides into the subprocess env, and commit the volumes every 120s plus
    once on exit — so a crash/preemption still leaves the latest state on the
    volumes for the local trainer's auto-resume on retry. Volumes arrive as
    arguments and are duck-typed (.commit()): this module must not import modal
    (the local trainers import it inside Modal-free containers)."""
    import subprocess
    import sys
    import threading

    cmd = [sys.executable, script]
    for flag, val in flag_vals:
        if val is not None:
            cmd += [flag, str(val)]
    env = dict(os.environ)
    if env_json:
        ov = {str(k): str(v) for k, v in json.loads(env_json).items()}
        env.update(ov)
        print("[modal-shell] env overrides: " + " ".join(sorted(ov)), flush=True)
    print("[modal-shell] " + " ".join(cmd), flush=True)

    # ponytail: time-based commit; the trainer's 2-checkpoint retention covers
    # the rare case of snapshotting a half-written .pth.
    stop = threading.Event()

    def _commit_loop():
        while not stop.wait(120):
            for v in volumes:
                v.commit()

    threading.Thread(target=_commit_loop, daemon=True).start()
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        stop.set()
        for v in volumes:
            v.commit()


def _demo():  # ponytail: one runnable check -- `python train_common.py`
    import tempfile
    d = tempfile.mkdtemp()
    b = BestCheckpoint(d)
    assert b.update(0.5) and not b.update(0.4) and b.update(0.6)
    csv_path = os.path.join(d, "val_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_acc", "val_miou"])
        w.writerow([0, 0.9, 0.7])
        w.writerow([1, 0.9, 0.65])
    assert abs(best_val_miou(csv_path) - 0.7) < 1e-9
    assert not BestCheckpoint(d).update(0.69)   # resume seeds from csv -> no regress
    open(b.final, "w").close()
    b.finalize(lambda p: (_ for _ in ()).throw(AssertionError("should not save_last")))

    # write_run_manifest: normalized fields merged over the raw config (read from
    # run.json; a legacy run_config.json works as the raw source too).
    with open(os.path.join(d, "run_config.json"), "w") as f:
        json.dump({"num_classes": 7, "class_names": list("abcdefg"),
                   "grid_m": 2.0, "chunk_xy_m": 100.0,
                   "features": ["intensity", "return_number", "height", "feat_hag"]}, f)
    m = write_run_manifest(d, "kpconvx_cold")   # no dataset -> p95
    assert m["backbone"] == "kpconvx_cold" and m["weights"] == "final_model.pth"
    assert m["grid"] == 2.0 and m["chunk_xy"] == 100.0 and m["num_classes"] == 7
    assert m["intensity_norm"] == "p95"
    assert "hag_source" not in m and m.get("feature_mode") != "hag"
    assert m["grid_m"] == 2.0                       # raw config survives the merge
    assert os.path.exists(os.path.join(d, "run.json"))
    # second call reads the merged run.json itself (idempotent; no run_config needed)
    os.remove(os.path.join(d, "run_config.json"))
    m_again = write_run_manifest(d, "kpconvx_cold")
    assert m_again["grid"] == 2.0 and m_again["grid_m"] == 2.0
    # DG round-trip: defaults (no env) -> logdk off; env on -> recorded + readable back
    assert m["dg"] is not None and m["dg"]["logdk"] is False
    os.environ["DG_LOGDK_FEAT"] = "1"; os.environ["DG_LOGDK_K"] = "12"
    m2 = write_run_manifest(d, "kpconvx_cold")
    assert m2["dg"]["logdk"] is True and m2["dg"]["logdk_k"] == 12
    assert infer_meta(os.path.join(d, "final_model.pth"))["dg"]["logdk"] is True
    os.environ.pop("DG_LOGDK_FEAT"); os.environ.pop("DG_LOGDK_K")

    # intensity_norm lives under dataset_meta['source'] — read it from there
    assert _intensity_norm_from_meta({"source": {"intensity_norm": "p95"}}) == "p95"
    assert _intensity_norm_from_meta({"intensity_norm": "p95"}) == "p95"   # tolerate top-level
    assert _intensity_norm_from_meta({"source": {}}) == "max"              # default

    # infer_meta reads run.json beside the weights (None for a bare .pth);
    # a feat_hag spec round-trips through "features" like any other channel.
    # hag_source is surfaced (None for post-clean-break runs) so trainers can
    # loudly REJECT legacy --hag weights — same width, different semantics.
    im = infer_meta(os.path.join(d, "final_model.pth"))
    assert im and im["num_classes"] == 7 and im["grid"] == 2.0
    assert im["features"] == ["intensity", "return_number", "height", "feat_hag"]
    assert im["hag_source"] is None and im["class_names"] == list("abcdefg")
    assert infer_meta(os.path.join(tempfile.mkdtemp(), "bare.pth")) is None

    # tile_xy_indices == the old brute-force per-tile mask, tile for tile
    import numpy as np
    import torch as _torch
    rng = np.random.RandomState(1)
    pts = rng.rand(5_000, 3).astype(np.float32) * [300, 300, 10]
    got = {(x0, y0): set(idx.tolist())
           for x0, y0, idx in tile_xy_indices(_torch.from_numpy(pts), 100.0, 50.0)}
    mins, maxs = pts[:, :2].min(0), pts[:, :2].max(0)
    for x0 in np.arange(mins[0], maxs[0], 50.0):
        for y0 in np.arange(mins[1], maxs[1], 50.0):
            m = ((pts[:, 0] >= x0) & (pts[:, 0] < x0 + 100.0) &
                 (pts[:, 1] >= y0) & (pts[:, 1] < y0 + 100.0))
            ref = set(np.nonzero(m)[0].tolist())
            assert got.get((x0, y0), set()) == ref

    # corrupt cached tile -> scan drops it + the scene's .done and raises so
    # the (auto-retried) next run re-tiles the scene instead of dying forever
    cd = tempfile.mkdtemp()
    np.savez(f"{cd}/good_x0_y0.npz", lab=np.array([0, 1], np.int32))
    open(f"{cd}/bad_x0_y50.npz", "w").write("not a zip")
    open(f"{cd}/bad.done", "w").close()
    try:
        scan_class_balance([f"{cd}/good_x0_y0.npz", f"{cd}/bad_x0_y50.npz"], 2)
        raise AssertionError("corrupt tile must raise after healing")
    except RuntimeError as e:
        assert "corrupt" in str(e)
    assert not os.path.exists(f"{cd}/bad_x0_y50.npz")   # tile dropped
    assert not os.path.exists(f"{cd}/bad.done")          # scene unmarked
    assert os.path.exists(f"{cd}/good_x0_y0.npz")        # good tile untouched

    # ptv3_load_canonical: global-UTM float64 scenes come back scene-local —
    # centering a 30m tile then must land within |xy|<=40 (the batch filter
    # that emptied every utonia batch when a float32 mean of ~5e6-magnitude
    # coords was off by hundreds of meters).
    utm = np.array([620900.0, 4849000.0, 170.0]) + \
        np.random.RandomState(2).rand(20_000, 3) * [30, 30, 5]
    np.savez(f"{cd}/utm.npz", xyz=utm, label=np.zeros(len(utm), np.int32))
    lx, _, _ = ptv3_load_canonical(f"{cd}/utm.npz", "intensity")
    assert lx.dtype == np.float32 and 0 <= lx.min() and lx.max() < 40.0
    c = (lx - lx.mean(0, keepdims=True, dtype=np.float64)).astype(np.float32)
    assert (np.abs(c[:, :2]).max(1) <= 40.0).all()      # every point survives
    assert np.allclose(lx + np.floor(utm.min(0)), utm, atol=1e-2)  # invertible

    # make_prefetcher: batches arrive in order, shutdown is clean
    _pn = iter(range(100))
    pf = make_prefetcher(lambda: next(_pn), depth=2)
    assert pf() == 0 and pf() == 1 and pf() == 2
    pf.shutdown()

    # kp_grid_subsample (torch-pooled) == brute-force per-voxel reference,
    # voxel order included (sorted-flat == np.unique(axis=0) lexicographic)
    pts2 = rng.rand(2_000, 3).astype(np.float32) * 10 - 5   # negatives too
    at2 = rng.rand(2_000, 2).astype(np.float32)
    lb2 = rng.randint(-1, 4, 2_000).astype(np.int32)
    sx2, sa2, sl2 = kp_grid_subsample(pts2, at2, lb2, 1.0, 4)
    vk = np.floor(pts2 / 1.0).astype(np.int64)
    uk = np.unique(vk, axis=0)
    assert len(sx2) == len(uk)
    for i, kk in enumerate(uk):
        m = (vk == kk).all(1)
        assert np.allclose(sx2[i], pts2[m].mean(0), atol=1e-5)
        assert np.allclose(sa2[i], at2[m].mean(0), atol=1e-5)
        vl = lb2[m][lb2[m] >= 0]
        assert sl2[i] == (np.bincount(vl, minlength=4).argmax() if len(vl) else -1)

    # voxel_unique == np.unique(axis=0) exactly (order included), negatives too
    rng = np.random.RandomState(0)
    for dims in (2, 3):
        keys = rng.randint(-50, 50, size=(20_000, dims))
        ref_u, ref_first, ref_inv = np.unique(keys, axis=0, return_index=True,
                                              return_inverse=True)
        first, inv = voxel_unique(keys, return_inverse=True)
        assert np.array_equal(first, ref_first) and np.array_equal(inv, ref_inv.reshape(-1))
        assert np.array_equal(keys[first], ref_u)
    # xy_chunk_groups == the per-window boolean-mask reference
    xyz = rng.uniform(0, 300, size=(30_000, 3))
    got = {tuple(sorted(g.tolist())) for g in xy_chunk_groups(xyz, 100.0, min_pts=64)}
    mins = xyz[:, :2].min(0)
    ref = set()
    for x0 in np.arange(mins[0], xyz[:, 0].max() + 100.0, 100.0):
        for y0 in np.arange(mins[1], xyz[:, 1].max() + 100.0, 100.0):
            m = ((xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + 100.0)
                 & (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + 100.0))
            if m.sum() >= 64:
                ref.add(tuple(sorted(np.where(m)[0].tolist())))
    assert got == ref, (len(got), len(ref))
    # write_infer_run: config + per-scene metrics + totals round-trip
    doc = write_infer_run(d, {"backbone": "x"},
                          [{"scene": "a.npz", "points": 10, "seconds": 1.5},
                           {"scene": "b.npz", "points": 5, "seconds": 0.5}])
    assert doc["total_points"] == 15 and doc["total_seconds"] == 2.0
    assert json.load(open(os.path.join(d, "infer_run.json")))["backbone"] == "x"

    # ---- shared trainer pieces -------------------------------------------
    import torch
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(50, 4, generator=g)
    labels = torch.randint(0, 4, (50,), generator=g)
    # seg_loss == CE + Lovász by construction; all-ignored batch -> finite 0
    seg = make_seg_loss(None, 0.0, False, 2.0, 1.0)
    ref = (torch.nn.functional.cross_entropy(logits, labels)
           + lovasz_softmax_flat(torch.softmax(logits, 1), labels))
    assert torch.allclose(seg(logits, labels), ref)
    assert float(seg(logits, torch.full((50,), -1))) == 0.0
    # focal(gamma=0, no alpha) == plain CE; perfect preds -> Lovász ~ 0
    assert torch.allclose(focal_loss(logits, labels, 0.0),
                          torch.nn.functional.cross_entropy(logits, labels))
    hot = torch.nn.functional.one_hot(labels, 4).float() * 1e6
    assert float(lovasz_softmax_flat(torch.softmax(hot, 1), labels)) < 1e-3

    counts = np.array([1000, 1000, 10, 0], np.int64)
    assert auto_rare_classes(counts, 0.5) == [2]        # below 0.5 x median, present
    w = class_weights_np(counts[:3], 0.5, 5.0)          # no absent class
    assert w[2] > w[0] and w.max() <= 5.0 and w.min() >= 0.2
    wa = class_weights_np(counts, 0.5, 5.0, absent_to_one=True)
    assert wa[3] <= wa[2] and wa[2] > wa[0]             # absent class not up-weighted
    pick = make_tile_picker(["a", "b"], ["r"], 1.0)
    assert pick() == "r"
    assert make_tile_picker(["a"], [], 1.0)() == "a"    # no rare tiles -> uniform

    i_, u_, g_ = score_ious(np.array([0, 0, 1]), np.array([0, 1, 1]), 2)
    assert list(i_) == [1, 1] and list(u_) == [2, 2] and list(g_) == [1, 2]
    m_ev = eval_metrics(i_, u_, g_, 2, 3, ["a", "b"], time.time(), 1, "demo",
                        extra={"skipped_scenes": 0, "scored_on": "raw_points"})
    assert (abs(m_ev["overall_mIoU"] - 0.5) < 1e-9 and m_ev["scored_on"] == "raw_points"
            and m_ev["present_classes"] == ["a", "b"])
    vd = tempfile.mkdtemp()
    append_val_row(f"{vd}/v.csv", 3, m_ev, ["a", "b"])
    assert "3,0.6667,0.5000,0.5000,0.5000" in open(f"{vd}/v.csv").read()

    # ---- KP-family pipeline ----------------------------------------------
    # grid_subsample: two coincident-voxel points merge (mean attrs, majority lab)
    sx, sa, sl = kp_grid_subsample(
        np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [5.0, 5.0, 5.0]], np.float32),
        np.array([[0.0], [1.0], [2.0]], np.float32),
        np.array([1, 1, 0], np.int64), 1.0, 3)
    assert len(sx) == 2 and abs(sa[0, 0] - 0.5) < 1e-6 and list(sl) == [1, 0]
    assert kp_augment(sx).shape == sx.shape

    # feature spec: empty -> legacy default; names validated against the vocab
    assert parse_feat_spec("", ["intensity", "return_number", "height"]) \
        == ["intensity", "return_number", "height"]
    assert parse_feat_spec(" intensity , feat_ndvi ", []) == ["intensity", "feat_ndvi"]
    try:
        parse_feat_spec("bogus", [])
        raise AssertionError("unknown spec name must raise")
    except ValueError as e:
        assert "bogus" in str(e) and "intensity" in str(e)   # lists the vocabulary

    bf = kp_make_build_feat(False, 8)
    xyz10 = np.random.RandomState(0).rand(10, 3).astype(np.float32)
    f = bf(xyz10, np.ones(10, np.float32), np.zeros(10, np.float32))
    assert f.shape == (10, 4) and np.all(f[:, 0] == 1.0)
    fd = bf(xyz10, np.ones(10, np.float32), np.ones(10, np.float32), drop=True)
    assert np.all(fd[:, 1:3] == 0.0)                    # drop zeroes intensity+ret_num
    assert np.allclose(fd[:, 3], xyz10[:, 2] - xyz10[:, 2].min())  # geometry survives
    # spec-ordered assembly (bias always first, not in the spec)
    bfs = kp_make_build_feat(False, 8, spec=["height", "intensity", "feat_q"])
    fq = np.arange(10, dtype=np.float32)
    fs = bfs(xyz10, np.full(10, 0.5, np.float32), np.zeros(10, np.float32),
             extras={"feat_q": fq})
    assert fs.shape == (10, 4) and np.all(fs[:, 0] == 1.0)
    assert np.allclose(fs[:, 1], xyz10[:, 2] - xyz10[:, 2].min())   # native height
    assert np.all(fs[:, 2] == 0.5) and np.array_equal(fs[:, 3], fq)
    try:
        bfs(xyz10, fq, fq)                              # spec wants feat_q -> must raise
        raise AssertionError("missing extras must raise")
    except ValueError as e:
        assert "feat_q" in str(e)

    # end-to-end prep -> sample -> voted eval on a synthetic canonical dataset
    ds = tempfile.mkdtemp()
    rng = np.random.RandomState(1)
    for split in ("train", "val", "test"):
        os.makedirs(f"{ds}/{split}")
        np.savez(f"{ds}/{split}/s0.npz",
                 xyz=rng.uniform(0, 60, (4000, 3)).astype(np.float32),
                 intensity=rng.rand(4000).astype(np.float32),
                 feat_demo=np.full(4000, 0.25, np.float32),
                 label=rng.randint(0, 3, 4000).astype(np.int32))
    prep = os.path.join(ds, "prep")
    sig = {"pipeline": "demo", "grid": 2.0}
    tile_fn = lambda name, pc, outd, split: kp_tile_and_save(
        name, pc, outd, 30.0,
        train_stride(30.0) if split == "train" else 15.0, 2.0, 3)
    tr, va, te = kp_ensure_prep(prep, ds, sig, tile_fn)
    assert [n for n, _, _ in tr] == ["s0"] and os.path.exists(f"{prep}/train/s0.done")
    import glob as _glob
    train_tiles = sorted(_glob.glob(f"{prep}/train/*.npz"))
    assert train_tiles
    kp_ensure_prep(prep, ds, sig, tile_fn)              # idempotent re-run
    try:
        kp_ensure_prep(prep, ds, {**sig, "grid": 9.0}, tile_fn)
        raise AssertionError("stale cache must be refused")
    except RuntimeError:
        pass
    cc, pm = scan_class_balance(train_tiles, 3, cache_path=f"{prep}/cb.npz")
    cc2, _ = scan_class_balance(train_tiles, 3, cache_path=f"{prep}/cb.npz")
    assert cc.sum() > 0 and np.array_equal(cc, cc2) and pm.shape == (len(train_tiles), 3)

    st = kp_make_sample_tile(bf, 2.0, 500, 0.8, False, 2.5, 0.5)
    s = st(train_tiles[0], training=False)
    assert s and s[0].shape == (len(s[2]), 3) and s[1].shape[1] == 4
    assert abs(s[0].mean()) < 1e-3                      # centered
    # feat_* channels ride the tile cache (mean-pooled) and land where the
    # spec puts them; requesting an absent one is a clear error naming it.
    zt = np.load(train_tiles[0])
    assert "feat_demo" in zt.files and np.allclose(zt["feat_demo"], 0.25)
    st3 = kp_make_sample_tile(kp_make_build_feat(False, 8,
                                                 spec=["intensity", "feat_demo"]),
                              2.0, 500, 1.0, False, 2.5, 0.5)
    s3 = st3(train_tiles[0], training=False)
    assert s3[1].shape[1] == 3 and np.allclose(s3[1][:, 2], 0.25)
    try:
        feat_extras(zt, ["feat_nope"], "t0")
        raise AssertionError("absent feat_* must raise")
    except ValueError as e:
        assert "feat_nope" in str(e) and "feat_demo" in str(e)

    fwd = lambda tiles: [np.tile([5.0, 0.0, 0.0], (len(c), 1)).astype(np.float32)
                         for c, _ in tiles]
    ev = kp_make_evaluate(fwd, bf, 2.0, 30.0, 3, ["a", "b", "c"])
    os.environ["EVAL_BATCH"] = "1"
    m_kp = ev([("s0", f"{ds}/val/s0.npz", None, f"{prep}/val")], "demo")
    assert m_kp["num_scenes"] == 1 and m_kp["per_class_gt_count"]["a"] > 0
    assert abs(m_kp["per_class_iou"]["a"] - m_kp["per_class_gt_count"]["a"]
               / sum(m_kp["per_class_gt_count"].values())) < 0.02   # all-a predictions
    # batched grouping must not change the metrics
    os.environ["EVAL_BATCH"] = "3"
    m_kp3 = ev([("s0", f"{ds}/val/s0.npz", None, f"{prep}/val")], "demo")
    del os.environ["EVAL_BATCH"]
    assert (m_kp3["per_class_iou"] == m_kp["per_class_iou"]
            and m_kp3["overall_acc"] == m_kp["overall_acc"])
    pp = kp_make_predict_points(
        lambda cxyz, feat: np.tile([1.0, 0.0, 0.0], (len(cxyz), 1)).astype(np.float32),
        bf, 2.0, 30.0, 3, 0, save_probs=True)
    z0 = np.load(f"{ds}/val/s0.npz")
    pr, cf, pb = pp(z0["xyz"], z0["intensity"], np.zeros(4000, np.float32))
    assert pr.shape == (4000,) and set(pr.tolist()) == {0}
    assert cf.shape == (4000,) and cf.dtype == np.float32 and cf.max() <= 1.0 + 1e-6
    assert pb.shape == (4000, 3) and pb.dtype == np.float16

    # scene_arrays fallbacks + the infer scene loop
    zi, zr = scene_arrays({"files": []} and np.load(f"{ds}/val/s0.npz"), 4000)
    assert zi.shape == (4000,) and np.all(zr == 0.0)
    ij = tempfile.mkdtemp()
    os.makedirs(f"{ij}/predictions")
    run_infer_scenes([f"{ds}/val/s0.npz"],
                     lambda p: (z0["xyz"], pr, z0["intensity"], cf, pb),
                     f"{ij}/predictions", ij, {"backbone": "demo"}, cls_txt=True)
    assert (os.path.exists(f"{ij}/predictions/s0_pred.npz")
            and os.path.exists(f"{ij}/predictions/s0_pred_CLS.txt")
            and json.load(open(f"{ij}/infer_run.json"))["total_points"] == 4000)
    zp = np.load(f"{ij}/predictions/s0_pred.npz")
    assert (zp["confidence"].dtype == np.float32
            and zp["probs"].dtype == np.float16 and zp["probs"].shape == (4000, 3))
    assert "crs_wkt" not in zp.files                 # scene had none -> pred has none

    # crs_wkt ferry: a scene npz carrying a CRS string lands it in the pred npz
    np.savez(f"{ij}/c0.npz", xyz=z0["xyz"], intensity=z0["intensity"],
             crs_wkt=np.asarray('PROJCS["demo"]'))
    run_infer_scenes([f"{ij}/c0.npz"],
                     lambda p: (z0["xyz"], pr, z0["intensity"], cf, None),
                     f"{ij}/predictions", ij, {"backbone": "demo"})
    with np.load(f"{ij}/predictions/c0_pred.npz") as zc:
        assert str(zc["crs_wkt"]) == 'PROJCS["demo"]'

    # env_overrides: env wins over the module default, order preserved
    os.environ["LOSS_FOCAL_GAMMA"] = "3.5"
    uf, fg = env_overrides({"USE_FOCAL": False, "FOCAL_GAMMA": 2.0},
                           ["USE_FOCAL", "FOCAL_GAMMA"])
    assert uf is False and abs(fg - 3.5) < 1e-9
    del os.environ["LOSS_FOCAL_GAMMA"]

    # modal_shell_run: skips None flags, always commits volumes on exit
    class _V:
        n = 0
        def commit(self):
            _V.n += 1
    modal_shell_run("-V", [("--unused", None)], None, [_V()])
    assert _V.n == 1
    print("ok")


if __name__ == "__main__":
    _demo()
