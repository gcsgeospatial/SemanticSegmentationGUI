"""Shared best-checkpoint selection for the local_train_* trainers.

Studies pick the checkpoint with the highest validation mIoU, not the last
epoch (Pointcept -> model_best.pth; the DFC aerial-LiDAR eval paper
arXiv:2603.22420 selects "the checkpoint with highest validation mIoU"). This
makes final_model.pth = best-by-val-mIoU while keeping the inference contract
(final_model.pth) unchanged.
"""
import csv
import json
import os
import time


def write_pred(path, xyz, pred, intensity=None):
    """Write one inferred scene as a compact npz — xyz + per-point class index
    (+ optional intensity). The host exporter (dataset.export_predictions) writes
    the user's chosen file type straight from this, so there's no intermediate
    coloured PLY to render and then reparse. Class is stored losslessly (the raw
    prediction), not encoded as palette colour."""
    import numpy as np
    d = {"xyz": np.asarray(xyz, np.float32),
         "classification": np.asarray(pred, np.int32)}
    if intensity is not None:
        d["intensity"] = np.asarray(intensity, np.float32)
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


def _dg_block() -> dict | None:
    """DG settings that must travel WITH the weights, read from the same env the
    training process ran under. `logdk` changes the model's input width, so
    inference has to rebuild at that width and recompute the channel (with the same
    k) — record both so run.json is self-describing. AdaBN/TTA are inference-time
    choices made on the Infer page, NOT a property of the weights, so not here."""
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
        mp = f"/datasets/{dataset}/dataset_meta.json"
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
        "feature_mode": "hag" if "hag" in backbone else "native",
        "hag_source": rc.get("hag_source"),
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
                                      "hag_source", "num_points", "dg")}
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
            "hag_source": rc.get("hag_source"),
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
    doc["scenes"] = scene_stats
    doc["total_points"] = int(sum(s["points"] for s in scene_stats))
    doc["total_seconds"] = round(sum(float(s["seconds"]) for s in scene_stats), 3)
    with open(os.path.join(run_dir, "infer_run.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return doc


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
        lab = np.load(tp)["lab"]
        v = lab[(lab >= 0) & (lab < num_classes)]
        return (np.bincount(v, minlength=num_classes).astype(np.int64)
                if v.size else np.zeros(num_classes, np.int64))

    from concurrent.futures import ThreadPoolExecutor
    print(f"  scanning {len(tile_paths)} train tiles for class balance (parallel)…",
          flush=True)
    per_tile = np.zeros((len(tile_paths), num_classes), np.int64)
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, counts in enumerate(ex.map(_scan, tile_paths)):
            per_tile[i] = counts
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
    """(intensity, ret_num) from an inference scene npz, with the shared
    fallbacks (missing intensity -> 0.5; return_number/ret_num -> zeros)."""
    import numpy as np
    intensity = (z["intensity"].astype(np.float32) if "intensity" in z
                 else np.full(n, 0.5, np.float32))
    ret_num = (z["return_number"].astype(np.float32) if "return_number" in z
               else (z["ret_num"].astype(np.float32) if "ret_num" in z
                     else np.zeros(n, np.float32)))
    return intensity, ret_num


def scene_hag(z, pc_path, n, enabled):
    """Real per-point HAG from an inference scene npz (--hag); None when plain.
    convert_infer_job writes it when the HAG box is ticked."""
    import numpy as np
    if not enabled:
        return None
    if "hag" not in z.files or len(z["hag"]) != n:
        raise ValueError(
            f"{os.path.basename(pc_path)} has no per-point 'hag' channel, which this "
            f"HAG model requires. Tick 'Compute Height-Above-Ground' on the Inference "
            f"page and run again.")
    return z["hag"].astype(np.float32)


def run_infer_scenes(scenes, predict, pred_dir, run_dir, infer_cfg, cls_txt=False):
    """The --mode infer scene loop: predict(pc_path) -> (xyz, pred, intensity),
    written as <name>_pred.npz (+ optional _pred_CLS.txt), with the crash-safe
    per-scene infer_run.json rewrite."""
    import numpy as np
    print(f"  [infer] labeling {len(scenes)} scene(s) -> {pred_dir}", flush=True)
    scene_stats = []
    for pc_path in scenes:
        name = os.path.splitext(os.path.basename(pc_path))[0]
        t0 = time.time()
        xyz, pred, inten = predict(pc_path)
        write_pred(f"{pred_dir}/{name}_pred.npz", xyz, pred, inten)
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
# kpconvx_cold twins used to duplicate. To add a new per-point feature channel
# (e.g. linearity), touch kp_tile_and_save (cache it) and kp_make_build_feat
# (feed it) — nothing else.
# ============================================================================

def kp_load_canonical(npz_path):
    """Canonical trainer_gui scene (.npz) -> (xyz, intensity, ret_num, lab).
    xyz is origin-offset (per-scene floor-min) before the float32 cast so
    projected (UTM) coords keep sub-meter precision."""
    import numpy as np
    z = np.load(npz_path)
    xyz = (z["xyz"] - np.floor(z["xyz"].min(0))).astype(np.float32)
    if "intensity" in z.files:
        intensity = z["intensity"].astype(np.float32)
    elif "rgb" in z.files:
        intensity = z["rgb"].astype(np.float32).mean(1) / 255.0
    else:
        intensity = np.zeros(len(xyz), np.float32)
    ret_num = np.zeros(len(xyz), np.float32)
    lab = z["label"].astype(np.int32) if "label" in z.files \
        else np.full(len(xyz), -1, np.int32)
    return xyz, intensity, ret_num, lab


def kp_grid_subsample(xyz, attrs, lab, voxel, num_classes):
    """Voxel-grid subsample to `voxel` m: barycenter points, mean attrs,
    majority labels. Mirrors KPConv's grid_subsampling (the C++ op that
    produces the layer-0 cloud) in numpy so the prep cache never needs the
    compiled extensions."""
    import numpy as np
    keys = np.floor(xyz / voxel).astype(np.int64)
    inv = voxel_unique(keys, return_inverse=True)[1]
    nv = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=nv).astype(np.float64)
    sx = np.zeros((nv, 3)); np.add.at(sx, inv, xyz); sx /= cnt[:, None]
    sa = None
    if attrs is not None:
        sa = np.zeros((nv, attrs.shape[1])); np.add.at(sa, inv, attrs); sa /= cnt[:, None]
    sl = np.full(nv, -1, np.int64)
    if lab is not None:
        oh = np.zeros((nv, num_classes)); v = lab >= 0
        np.add.at(oh, (inv[v], lab[v]), 1)
        has = oh.sum(1) > 0
        sl[has] = oh[has].argmax(1)
    return (sx.astype(np.float32),
            (sa.astype(np.float32) if sa is not None else None),
            sl)


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


def kp_tile_and_save(name, pc_path, out_dir, chunk_xy, stride, grid, num_classes, hag_on):
    """One scene -> overlapping chunk_xy tiles, grid-subsampled and cached as
    .npz (xyz + intensity + ret_num [+ hag] + lab). Returns the tile count, or
    None when the scene failed to load (left unmarked so it retries)."""
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    try:                              # canonical .npz (label + optional hag embedded)
        xyz, intensity, ret_num, lab = kp_load_canonical(pc_path)
        hag = None
        if hag_on:   # --hag: the scene's real per-point HAG (startup guard vouches for it)
            z = np.load(pc_path)
            if "hag" not in z.files or len(z["hag"]) != len(xyz):
                raise ValueError("missing its 'hag' channel, which --hag requires; "
                                 "rebuild the dataset with HAG enabled")
            hag = z["hag"].astype(np.float32)
    except Exception as e:
        print(f"  skip {pc_path}: {e}", flush=True)
        return None
    intensity_n = np.clip(intensity, 0.0, 2.0).astype(np.float32)
    print(f"    {name}: {len(xyz):,} pts loaded in {time.time()-t0:.1f}s, "
          + (f"HAG {hag.min():.1f}..{hag.max():.1f}m, " if hag_on else "") + "tiling…",
          flush=True)
    mins = xyz[:, :2].min(0); maxs = xyz[:, :2].max(0)
    n_tiles = 0
    for x0 in np.arange(mins[0], maxs[0], stride):
        for y0 in np.arange(mins[1], maxs[1], stride):
            mask = (
                (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + chunk_xy) &
                (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + chunk_xy)
            )
            # Low thresholds on purpose: water absorbs LiDAR, so pure-water
            # tiles are sparse — a higher cut would delete them from training.
            if mask.sum() < 64:
                continue
            attrs = np.stack([intensity_n[mask], ret_num[mask]]
                             + ([hag[mask]] if hag_on else []), axis=1).astype(np.float32)
            sx, sa, sl = kp_grid_subsample(xyz[mask], attrs, lab[mask], grid, num_classes)
            if len(sx) < 32:
                continue
            tile = dict(
                xyz=sx.astype(np.float32),
                intensity=sa[:, 0].astype(np.float32),
                ret_num=sa[:, 1].astype(np.float32),
            )
            if hag_on:
                tile["hag"] = sa[:, 2].astype(np.float32)
            tile["lab"] = sl.astype(np.int32)
            np.savez_compressed(
                os.path.join(out_dir, f"{name}_x{int(x0)}_y{int(y0)}.npz"), **tile)
            n_tiles += 1
    print(f"      -> {n_tiles} tiles", flush=True)
    return n_tiles


def kp_ensure_prep(prep_dir, ds_root, sig, tile_fn):
    """Idempotent prep for the KP twins: split folders read verbatim, cache
    signature validated (validate_cache), each un-.done scene tiled via
    tile_fn(name, pc_path, out_dir). Returns (train, val, test) scene lists."""
    print(f"  ensuring preprocessed cache -> {prep_dir}", flush=True)
    for split in ("train", "val", "test"):
        os.makedirs(f"{prep_dir}/{split}", exist_ok=True)
    train_list, val_list, test_list = split_scenes(ds_root)
    any_new = [validate_cache(
        prep_dir, sig,
        [("train", train_list), ("val", val_list), ("test", test_list)],
        lambda d, name: (f"{d}/{name}_x*.npz", f"{d}/{name}.done"))]

    def tile_remaining(items, out_dir):
        for name, pc_path, _cls in items:
            if os.path.exists(f"{out_dir}/{name}.done"):
                continue
            n = tile_fn(name, pc_path, out_dir)
            if n is not None:          # None == load failed; leave unmarked to retry
                open(f"{out_dir}/{name}.done", "w").close()
            any_new[0] = True

    for split, items in (("train", train_list), ("val", val_list), ("test", test_list)):
        print(f"  [{split}] {len(items)} scenes", flush=True)
        tile_remaining(items, f"{prep_dir}/{split}")
    print("  preprocessing cache updated." if any_new[0]
          else "  all scenes already cached.", flush=True)
    return train_list, val_list, test_list


def kp_hag_of(z):
    """Real per-point HeightAboveGround for a cached --hag tile. Call only when
    HAG; the plain recipe wants build_feat's native height instead."""
    if "hag" not in z.files:
        raise ValueError("A --hag tile is missing its 'hag' channel. Rebuild the "
                         "dataset with Height-Above-Ground enabled.")
    return z["hag"].astype("float32")


def kp_make_build_feat(logdk_feat, logdk_k):
    """build_feat(xyz, intensity, ret_num, hag=None, drop=False) ->
    [1, intensity, return_number, height] (+ log d_k when D3b is on).

    height is the passed per-point `hag` array (--hag: real HeightAboveGround);
    when None this is the PLAIN recipe, whose 4th channel is z - min(z) over
    the tile. With `drop`, zero the non-bias channels (feature-drop)."""
    import numpy as np
    import density as dg

    def build_feat(xyz, intensity, ret_num, hag=None, drop=False):
        bias = np.ones((len(xyz), 1), np.float32)
        if hag is None:
            hag = (xyz[:, 2] - xyz[:, 2].min()).astype(np.float32)   # native height
        attrs = np.concatenate([intensity[:, None], ret_num[:, None],
                                hag[:, None]], axis=1).astype(np.float32)
        if drop:
            attrs[:, 1:] = 0.0   # keep intensity; drop ret_num/height
        cols = [bias, attrs]
        if logdk_feat:           # D3b: never dropped — the density signal to condition on
            cols.append(dg.local_density_logdk(xyz, logdk_k)[:, None])
        return np.concatenate(cols, axis=1).astype(np.float32)

    return build_feat


def kp_make_sample_tile(build_feat, hag_on, grid, max_pts, aug_color,
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
        hag = kp_hag_of(z) if hag_on else None
        if len(xyz) < min_pts:
            return None
        idx = np.arange(len(xyz))
        if len(idx) > max_pts:
            idx = np.random.choice(idx, max_pts, replace=False)
        xyz, intensity, ret_num, lab = xyz[idx], intensity[idx], ret_num[idx], lab[idx]
        if hag_on:
            hag = hag[idx]
        # D1 density jitter: coarsen-only re-subsample; index-consistent across
        # all per-point arrays.
        if training and density_aug:
            g_eff = dg.effective_grid(grid, coarsen_max, p_native)
            if g_eff > grid:
                keep = dg.voxel_first_idx(xyz, g_eff)
                xyz, intensity, ret_num, lab = xyz[keep], intensity[keep], ret_num[keep], lab[keep]
                if hag_on:
                    hag = hag[keep]
        drop = (training and np.random.rand() > aug_color)
        feat = build_feat(xyz, intensity, ret_num, hag, drop=drop)
        geo_xyz = kp_augment(xyz) if training else xyz
        geo_xyz = (geo_xyz - geo_xyz.mean(0)).astype(np.float32)
        return geo_xyz, feat, lab.astype(np.int64)

    return sample_tile


def kp_make_run_dir(variant):
    """Fresh timestamped run dir: /outputs/runs/<utc>_<variant>."""
    from datetime import datetime
    run_id = datetime.utcnow().strftime(f"%Y%m%d_%H%M%S_{variant}")
    run_dir = f"/outputs/runs/{run_id}"
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    return run_id, run_dir


def kp_find_latest_checkpoint(opt_type, feature_modes, arch_hash=None):
    """Most recent run (run-ids are timestamps, so they sort) with checkpoints
    AND this script's recipe: optimizer type, feature_mode (any accepted
    spelling — run.json holds the raw value until write_run_manifest merges the
    normalized one over it) and, when given, the architecture hash. Returns
    (run_dir, ckpt_path, epoch) or None."""
    import glob

    def _ep(p):
        return int(os.path.basename(p)[2:5])   # ep149.pth -> 149

    for rd in sorted(glob.glob("/outputs/runs/*"), reverse=True):
        ckpts = glob.glob(f"{rd}/checkpoints/ep*.pth")
        if not ckpts:
            continue
        got_opt = fmode = ahash = None
        for cfgp in (f"{rd}/run.json", f"{rd}/run_config.json"):   # legacy fallback
            try:
                with open(cfgp) as f:
                    rc = json.load(f)
                got_opt = rc.get("optimizer", {}).get("type")
                fmode = rc.get("feature_mode")
                ahash = rc.get("arch_hash")
                break
            except Exception:
                continue
        if got_opt != opt_type:
            print(f"  resume: skipping {os.path.basename(rd)} "
                  f"(recipe mismatch: optimizer={got_opt})", flush=True)
            continue
        if fmode not in feature_modes:
            print(f"  resume: skipping {os.path.basename(rd)} "
                  f"(variant mismatch: feature_mode={fmode})", flush=True)
            continue
        if arch_hash is not None and ahash is not None and ahash != arch_hash:
            print(f"  resume: skipping {os.path.basename(rd)} "
                  f"(architecture mismatch: arch_hash={ahash})", flush=True)
            continue
        latest = max(ckpts, key=_ep)
        return rd, latest, _ep(latest)
    return None


def kp_make_evaluate(forward, build_feat, hag_on, grid, chunk_xy, num_classes,
                     class_names):
    """The KP twins' voted eval, scored on the ORIGINAL raw points: per scene,
    run the model over its overlapping cached tiles, sum center-weighted
    softmax votes per voxel, argmax, then propagate to every raw point by
    nearest neighbour and score against the raw GT. forward(cxyz, feat) ->
    (N, C) logits ndarray; an exception skips the tile."""
    import glob
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    def evaluate(scene_items, label):
        t_inter = np.zeros(num_classes, dtype=np.int64)
        t_union = np.zeros(num_classes, dtype=np.int64)
        t_gt = np.zeros(num_classes, dtype=np.int64)
        correct = total = 0
        n_scenes = n_skipped_tiles = n_skipped_scenes = 0
        t_test = time.time()
        with torch.no_grad():
            for name, pc_path, _cls, split_dir in scene_items:
                tiles = sorted(glob.glob(f"{split_dir}/{name}_x*.npz"))
                if not tiles:
                    n_skipped_scenes += 1
                    continue
                keys_l, log_l, xyz_l = [], [], []
                for tile in tiles:
                    z = np.load(tile)
                    xyz = z["xyz"]
                    if len(xyz) < 32:
                        continue
                    feat = build_feat(xyz, z["intensity"], z["ret_num"],
                                      kp_hag_of(z) if hag_on else None)
                    cxyz = (xyz - xyz.mean(0)).astype(np.float32)
                    try:
                        lg = forward(cxyz, feat)
                    except Exception:
                        n_skipped_tiles += 1
                        continue
                    # Soft votes tapered toward the tile border (truncated context).
                    e = np.exp(lg - lg.max(1, keepdims=True))
                    prob = e / e.sum(1, keepdims=True)
                    cxy = (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
                    d = np.abs(xyz[:, :2] - cxy).max(1)
                    wgt = np.clip(1.0 - d / (chunk_xy / 2.0), 0.05, 1.0) ** 2
                    keys_l.append(np.floor(xyz / grid).astype(np.int64))
                    log_l.append((prob * wgt[:, None]).astype(np.float32))
                    xyz_l.append(xyz.astype(np.float32))
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
                    raw_xyz, _, _, raw_lab = kp_load_canonical(pc_path)
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped_scenes += 1
                    continue
                _, nn = cKDTree(rep_xyz).query(raw_xyz)
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
                           num_classes, tta):
    """Sliding-window inference over already-normalized features; returns
    per-raw-point class indices (--mode infer). forward_prob(cxyz, feat) ->
    (N, C) softmax ndarray; an exception skips the window. D5 density-TTA:
    average softmax over `tta` extra density(scale) views."""
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    def predict_points(xyz, intensity_n, ret_num, hag=None):
        pred = np.full(len(xyz), -1, np.int64)
        with torch.no_grad():
            for idx in xy_chunk_groups(xyz, chunk_xy, min_pts=64):
                cols = [intensity_n[idx], ret_num[idx]]
                if hag is not None:
                    cols.append(hag[idx])          # carry real HAG through the voxel mean
                attrs = np.stack(cols, axis=1).astype(np.float32)
                sx, sa, _ = kp_grid_subsample(xyz[idx], attrs, None, grid, num_classes)
                if len(sx) < 32:
                    continue
                sub_hag = sa[:, 2] if hag is not None else None
                feat = build_feat(sx, sa[:, 0], sa[:, 1], sub_hag)
                base = (sx - sx.mean(0)).astype(np.float32)
                views = [1.0] + (list(np.linspace(0.85, 1.2, tta)) if tta else [])
                try:
                    prob = None
                    for s in views:
                        p = forward_prob((base * s).astype(np.float32), feat)
                        prob = p if prob is None else prob + p
                    sub_pred = prob.argmax(-1)
                except Exception:
                    continue
                _, nn = cKDTree(sx).query(xyz[idx])
                pred[idx] = sub_pred[nn]
        miss = pred < 0
        if miss.any() and (~miss).any():
            _, nn = cKDTree(xyz[~miss]).query(xyz[miss])
            pred[miss] = pred[~miss][nn]
        return np.clip(pred, 0, num_classes - 1)

    return predict_points


def kp_make_target_batches(scenes, make_batch, build_feat, hag_on, grid,
                           chunk_xy, num_classes, cap=30):
    """AdaBN (D2b) target-batch generator over inference scenes: same windows,
    subsample and features predict_points will see. make_batch(cxyz, feat) ->
    model batch; an exception skips the window."""
    import numpy as np

    def target_batches():
        seen = 0
        for pc_path in scenes:
            if seen >= cap:
                return
            z = np.load(pc_path)
            txyz = z["xyz"].astype(np.float32)
            tin, trn = scene_arrays(z, len(txyz))
            thag = scene_hag(z, pc_path, len(txyz), hag_on)
            for idx in xy_chunk_groups(txyz, chunk_xy, min_pts=64):
                if seen >= cap:
                    return
                cols = [tin[idx], trn[idx]] + ([thag[idx]] if hag_on else [])
                attrs = np.stack(cols, 1).astype(np.float32)
                sx, sa, _ = kp_grid_subsample(txyz[idx], attrs, None, grid, num_classes)
                if len(sx) < 32:
                    continue
                # BN stats must see the same feature predict will be fed.
                feat = build_feat(sx, sa[:, 0], sa[:, 1], sa[:, 2] if hag_on else None)
                cxyz = (sx - sx.mean(0)).astype(np.float32)
                try:
                    b = make_batch(cxyz, feat)
                except Exception:
                    continue
                seen += 1
                yield b

    return target_batches()


# ============================================================================
# Cross-trainer plumbing shared by all four local trainers and all 8 modal
# shells (this file is baked into every image as /root/train_common.py).
# ============================================================================

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
    "USE_FOCAL":        ("LOSS_FOCAL",           "env_bool"),
    "FOCAL_GAMMA":      ("LOSS_FOCAL_GAMMA",     "env_float"),
    "CLASS_WEIGHTING":  ("LOSS_CLASS_WEIGHTING", "env_bool"),
    "WEIGHT_BETA":      ("LOSS_WEIGHT_BETA",     "env_float"),
    "RARE_OVERSAMPLE":  ("RARE_OVERSAMPLE",      "env_bool"),
    "RARE_CENTER_PROB": ("RARE_CENTER_PROB",     "env_float"),
    "KP_AGGREGATION":   ("KP_AGGREGATION",       "env_str"),
    "KP_NORM":          ("KP_NORM",              "env_str"),
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


def load_dataset_meta(dataset, hag, no_hag_hint):
    """Load /datasets/<dataset>/dataset_meta.json and gate --hag on has_hag.
    no_hag_hint: trainer-specific tail of the --hag error naming the plain
    sibling to use instead. Returns (ds_meta, num_classes, class_names,
    hag_source)."""
    meta_path = f"/datasets/{dataset}/dataset_meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found — build the dataset "
                                f"with the trainer_gui app first.")
    with open(meta_path) as f:
        ds_meta = json.load(f)
    if hag and not ds_meta.get("has_hag"):
        raise ValueError(
            f"--hag needs a dataset with a real HeightAboveGround channel, but "
            f"'{dataset}' has none (has_hag=false). Rebuild it with the Datasets "
            f"page 'Compute Height-Above-Ground' box, or {no_hag_hint}")
    # Legacy datasets recorded no method; the PDAL nearest-neighbour filter was
    # the only one back then.
    hag_source = (((ds_meta.get("source") or {}).get("hag_source") or "pdal_hag_nn")
                  if hag else None)
    return ds_meta, int(ds_meta["num_classes"]), list(ds_meta["class_names"]), hag_source


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
                   "grid_m": 2.0, "chunk_xy_m": 100.0, "hag_source": "pdal_hag_nn"}, f)
    m = write_run_manifest(d, "kpconvx_cold_hag")   # no dataset -> p95
    assert m["backbone"] == "kpconvx_cold_hag" and m["weights"] == "final_model.pth"
    assert m["grid"] == 2.0 and m["chunk_xy"] == 100.0 and m["num_classes"] == 7
    assert m["intensity_norm"] == "p95" and m["feature_mode"] == "hag"
    assert m["hag_source"] == "pdal_hag_nn"
    assert m["grid_m"] == 2.0                       # raw config survives the merge
    assert os.path.exists(os.path.join(d, "run.json"))
    # second call reads the merged run.json itself (idempotent; no run_config needed)
    os.remove(os.path.join(d, "run_config.json"))
    m_again = write_run_manifest(d, "kpconvx_cold_hag")
    assert m_again["grid"] == 2.0 and m_again["grid_m"] == 2.0
    # DG round-trip: defaults (no env) -> logdk off; env on -> recorded + readable back
    assert m["dg"] is not None and m["dg"]["logdk"] is False
    os.environ["DG_LOGDK_FEAT"] = "1"; os.environ["DG_LOGDK_K"] = "12"
    m2 = write_run_manifest(d, "kpconvx_cold_hag")
    assert m2["dg"]["logdk"] is True and m2["dg"]["logdk_k"] == 12
    assert infer_meta(os.path.join(d, "final_model.pth"))["dg"]["logdk"] is True
    os.environ.pop("DG_LOGDK_FEAT"); os.environ.pop("DG_LOGDK_K")

    # intensity_norm lives under dataset_meta['source'] — read it from there
    assert _intensity_norm_from_meta({"source": {"intensity_norm": "p95"}}) == "p95"
    assert _intensity_norm_from_meta({"intensity_norm": "p95"}) == "p95"   # tolerate top-level
    assert _intensity_norm_from_meta({"source": {}}) == "max"              # default

    # infer_meta reads run.json beside the weights (None for a bare .pth)
    im = infer_meta(os.path.join(d, "final_model.pth"))
    assert im and im["num_classes"] == 7 and im["grid"] == 2.0
    assert im["hag_source"] == "pdal_hag_nn" and im["class_names"] == list("abcdefg")
    assert infer_meta(os.path.join(tempfile.mkdtemp(), "bare.pth")) is None

    # voxel_unique == np.unique(axis=0) exactly (order included), negatives too
    import numpy as np
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

    bf = kp_make_build_feat(False, 8)
    xyz10 = np.random.RandomState(0).rand(10, 3).astype(np.float32)
    f = bf(xyz10, np.ones(10, np.float32), np.zeros(10, np.float32))
    assert f.shape == (10, 4) and np.all(f[:, 0] == 1.0)
    assert np.all(bf(xyz10, np.ones(10, np.float32), np.ones(10, np.float32),
                     drop=True)[:, 2:] == 0.0)          # drop keeps bias+intensity

    # end-to-end prep -> sample -> voted eval on a synthetic canonical dataset
    ds = tempfile.mkdtemp()
    rng = np.random.RandomState(1)
    for split in ("train", "val", "test"):
        os.makedirs(f"{ds}/{split}")
        np.savez(f"{ds}/{split}/s0.npz",
                 xyz=rng.uniform(0, 60, (4000, 3)).astype(np.float32),
                 intensity=rng.rand(4000).astype(np.float32),
                 label=rng.randint(0, 3, 4000).astype(np.int32))
    prep = os.path.join(ds, "prep")
    sig = {"pipeline": "demo", "grid": 2.0}
    tile_fn = lambda name, pc, outd: kp_tile_and_save(
        name, pc, outd, 30.0, 15.0, 2.0, 3, False)
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

    st = kp_make_sample_tile(bf, False, 2.0, 500, 0.8, False, 2.5, 0.5)
    s = st(train_tiles[0], training=False)
    assert s and s[0].shape == (len(s[2]), 3) and s[1].shape[1] == 4
    assert abs(s[0].mean()) < 1e-3                      # centered

    fwd = lambda cxyz, feat: np.tile([5.0, 0.0, 0.0], (len(cxyz), 1)).astype(np.float32)
    ev = kp_make_evaluate(fwd, bf, False, 2.0, 30.0, 3, ["a", "b", "c"])
    m_kp = ev([("s0", f"{ds}/val/s0.npz", None, f"{prep}/val")], "demo")
    assert m_kp["num_scenes"] == 1 and m_kp["per_class_gt_count"]["a"] > 0
    assert abs(m_kp["per_class_iou"]["a"] - m_kp["per_class_gt_count"]["a"]
               / sum(m_kp["per_class_gt_count"].values())) < 0.02   # all-a predictions
    pp = kp_make_predict_points(
        lambda cxyz, feat: np.tile([1.0, 0.0, 0.0], (len(cxyz), 1)).astype(np.float32),
        bf, 2.0, 30.0, 3, 0)
    z0 = np.load(f"{ds}/val/s0.npz")
    pr = pp(z0["xyz"], z0["intensity"], np.zeros(4000, np.float32))
    assert pr.shape == (4000,) and set(pr.tolist()) == {0}

    # scene_arrays fallbacks + scene_hag gate + the infer scene loop
    zi, zr = scene_arrays({"files": []} and np.load(f"{ds}/val/s0.npz"), 4000)
    assert zi.shape == (4000,) and np.all(zr == 0.0)
    try:
        scene_hag(np.load(f"{ds}/val/s0.npz"), "s0.npz", 4000, True)
        raise AssertionError("missing hag must raise")
    except ValueError:
        pass
    np.savez(f"{ds}/hag_scene.npz", xyz=z0["xyz"], hag=np.ones(4000, np.float32))
    zh = np.load(f"{ds}/hag_scene.npz")
    assert scene_hag(zh, "h.npz", 4000, True).dtype == np.float32   # happy path
    assert scene_hag(zh, "h.npz", 4000, False) is None
    ij = tempfile.mkdtemp()
    os.makedirs(f"{ij}/predictions")
    run_infer_scenes([f"{ds}/val/s0.npz"],
                     lambda p: (z0["xyz"], pr, z0["intensity"]),
                     f"{ij}/predictions", ij, {"backbone": "demo"}, cls_txt=True)
    assert (os.path.exists(f"{ij}/predictions/s0_pred.npz")
            and os.path.exists(f"{ij}/predictions/s0_pred_CLS.txt")
            and json.load(open(f"{ij}/infer_run.json"))["total_points"] == 4000)

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
