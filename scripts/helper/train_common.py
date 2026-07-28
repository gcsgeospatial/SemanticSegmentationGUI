"""Shared helpers for the local_train_* trainers.
final_model.pth = best-by-val-mIoU checkpoint (arXiv:2603.22420 protocol)."""
import contextlib
import csv
import json
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        with contextlib.suppress(Exception):
            _s.reconfigure(errors="replace")

DATASETS_ROOT = os.environ.get("TT_DATASETS_ROOT", "/datasets")
OUTPUTS_ROOT = os.environ.get("TT_OUTPUTS_ROOT", "/outputs")


def dataset_dir(name):
    """Canonical dataset root; TT_DATASET_DIR overrides."""
    return os.environ.get("TT_DATASET_DIR") or f"{DATASETS_ROOT}/{name}"


def infer_dir(job):
    """Inference job dir (scenes/ + predictions); TT_INFER_DIR overrides."""
    return os.environ.get("TT_INFER_DIR") or f"{DATASETS_ROOT}/_infer/{job}"


def resolve_weights_path(weights):
    """Absolute weights path: as given if absolute, else under OUTPUTS_ROOT."""
    return weights if os.path.isabs(weights) else f"{OUTPUTS_ROOT}/{weights}"


def write_pred(path, xyz, pred, intensity=None, confidence=None, probs=None,
               crs_wkt=None, source_crs_wkt=None):
    """Inferred-scene npz: xyz + classification (+ intensity, confidence,
    probs when TT_SAVE_PROBS=1). dataset.export_predictions reads this."""
    import numpy as np
    # float64: a float32 cast quantizes UTM northing to 0.5m steps
    d = {"xyz": np.asarray(xyz, np.float64),
         "classification": np.asarray(pred, np.int32)}
    if intensity is not None:
        d["intensity"] = np.asarray(intensity, np.float32)
    if confidence is not None:
        d["confidence"] = np.asarray(confidence, np.float32)
    if probs is not None:
        d["probs"] = np.asarray(probs, np.float16)
    if crs_wkt:
        d["crs_wkt"] = np.asarray(str(crs_wkt))
    if source_crs_wkt:
        d["source_crs_wkt"] = np.asarray(str(source_crs_wkt))
    np.savez(path, **d)


class DatasetExhausted(RuntimeError):
    """Deterministic 'no usable tiles' signal - must escape the broad
    train-loop excepts that swallow ordinary per-batch failures."""


class NonFiniteXYZ(ValueError):
    """Raised by require_finite_xyz: a hard data error, never retryable and
    never to be demoted to a warn-and-skip."""


def _replace_retry(tmp, path):
    """os.replace + Windows retry ladder: a reader (the GUI plot/inference page
    on a live run) briefly holds the destination; readers release in seconds."""
    for wait in (0.5, 1.0, 2.0, 4.0, None):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if wait is None:
                raise PermissionError(
                    f"{path} is locked by another process (the GUI plotting or "
                    "inference page open on a run that is still training?). "
                    "Close the reader and retry")
            time.sleep(wait)


def atomic_torch_save(obj, path):
    """torch.save via tmp + os.replace so a mid-write kill (Modal preemption,
    OOM) can't leave a truncated .pth for AUTO_RESUME to trip over."""
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    _replace_retry(tmp, path)


def atomic_json_save(doc, path):
    """json twin of atomic_torch_save: run.json gates resume, so a truncated
    write must never orphan a run's checkpoints."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    _replace_retry(tmp, path)


def validated_latest_ckpt(ckpts, ep_of):
    """Newest checkpoint that actually loads; skips truncated files so a
    corrupt max-epoch .pth can't permanently wedge resume."""
    import torch
    for p in sorted(ckpts, key=ep_of, reverse=True):
        try:
            torch.load(p, map_location="cpu", weights_only=True)
            return p
        except Exception as e:
            print(f"  resume: skipping corrupt checkpoint "
                  f"{os.path.basename(p)} ({e})", flush=True)
    return None


VAL_FULL_NOTE = ("  val: full raw-scored eval over every val scene, same protocol "
                 "as the test pass (PROXY_SAMPLING=full - slower than a proxy "
                 "pass, but directly comparable to the final numbers)")


def row_protocol(m):
    """Which val_metrics.csv scale a metrics dict belongs to. proxy_val stamps
    m['protocol']; the raw-scored full evals don't."""
    return "proxy" if "protocol" in m else "full"


def ranking_protocol(sampling):
    """PROXY_SAMPLING -> the row protocol that crowns final_model.pth.
    'full' (Full Eval) ranks on the same raw-scored eval as the test pass
    (slow, directly comparable); 'coverage' (Proxy Eval) ranks on the
    fixed-budget proxy."""
    return "full" if sampling == "full" else "proxy"


def best_val_miou(val_csv, protocol="proxy"):
    """Max val_miou over rows of `protocol` only (resume-safe seed). -1.0 if
    none. The other protocol scores on another scale and would freeze
    final_model.pth forever."""
    if not os.path.exists(val_csv):
        return -1.0
    best = -1.0
    with open(val_csv, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("protocol") != protocol:
                continue
            try:
                best = max(best, float(row["val_miou"]))
            except (KeyError, TypeError, ValueError):
                pass
    return best


class BestCheckpoint:
    """Track best val mIoU (seeded from val_metrics.csv); update(m) is True on
    a new best and ignores metrics from the other protocol, so callers can hand
    it every eval; finalize(save_last) saves last epoch only if val never ran."""

    def __init__(self, run_dir, protocol="proxy"):
        # every trainer builds one at train start, so this is the shared gate
        # keeping the inference-only channel kill out of training batches
        if zero_channels():
            raise RuntimeError(
                "TT_ZERO_CHANNELS is an inference-only control (set by the Infer "
                "page's Input channels table); unset it before training - a "
                "zeroed channel would silently poison every training batch.")
        self.protocol = protocol
        self.final = os.path.join(run_dir, "final_model.pth")
        self.best = best_val_miou(os.path.join(run_dir, "val_metrics.csv"),
                                  protocol)

    def update(self, m):
        if row_protocol(m) != self.protocol:
            return False
        miou = m["present_classes_mIoU"]
        if miou > self.best:
            self.best = miou
            return True
        return False

    def finalize(self, save_last):
        if not os.path.exists(self.final):
            save_last(self.final)


STOP_SENTINEL = f"{OUTPUTS_ROOT}/STOP"


def clear_stop():
    """Delete a stale STOP sentinel at startup; concurrent runs sharing one
    /outputs share the sentinel."""
    try:
        os.remove(STOP_SENTINEL)
        print("  [stop] removed stale STOP sentinel", flush=True)
    except OSError:
        pass


def stop_requested(ep):
    """Consume /outputs/STOP if present; the trainer breaks into the normal
    post-loop final-eval + finalize path."""
    if not os.path.exists(STOP_SENTINEL):
        return False
    try:
        os.remove(STOP_SENTINEL)
    except OSError:
        pass
    print(f"  [stop] STOP sentinel found. Stopping after epoch {ep}; "
          f"running the final evaluation…", flush=True)
    return True


def _dg_block() -> dict | None:
    """DG settings that travel WITH the weights (logdk changes model input
    width - inference rebuilds with the same k). AdaBN/TTA are per-job."""
    try:
        import density as dg
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
    """intensity_norm from meta['source'] (top-level tolerated); default 'max'.
    Wrong value = inference sees a different intensity scale than training."""
    src = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    return src.get("intensity_norm") or meta.get("intensity_norm") or "max"


def write_run_manifest(run_dir, backbone, dataset=None, weights="final_model.pth"):
    """Finalize run.json - the single record inference reads, beside the
    weights. Merges normalized manifest fields over the trainer's raw config
    in run.json. `backbone` = key."""
    rc = {}
    p = os.path.join(run_dir, "run.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                rc = json.load(f)
        except (OSError, ValueError):
            rc = {}
    inorm = "p95"
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
        "num_points": rc.get("num_points"),
        "dg": dg,
    }
    doc = {**rc, **manifest}
    atomic_json_save(doc, os.path.join(run_dir, "run.json"))
    return doc


def infer_meta(weights_path):
    """Normalized inference metadata from run.json beside the weights.
    None for a bare .pth; missing fields None."""
    d = os.path.dirname(weights_path)
    if os.path.basename(d) == "checkpoints":
        d = os.path.dirname(d)
    rj = os.path.join(d, "run.json")
    if not os.path.exists(rj):
        return None
    try:
        with open(rj, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    return {k: m.get(k) for k in ("num_classes", "class_names", "grid", "chunk_xy",
                                  "num_points", "dg",
                                  "features", "color_source")}


def xy_chunk_groups(xyz, chunk_m, min_pts=1):
    """Index groups over chunk_m XY windows via one packed-code sort
    (O(n log n)); groups smaller than min_pts skipped."""
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


VOXEL_GPU_MIN = 1_000_000


def voxel_unique(keys, return_inverse=False, gpu=True):
    """np.unique(keys, axis=0, return_index[, return_inverse]) equivalent,
    ~10x faster via packed int64 codes (order matches); axis=0 on overflow.
    Big inputs sort on CUDA when available - indices come back identical.
    gpu=False from threads that run concurrent with a model forward: a sort
    OOM falls back safely, but the ALLOCATION can push the forward into OOM."""
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
    if gpu and len(code) >= VOXEL_GPU_MIN:
        got = _voxel_unique_cuda(code, return_inverse)
        if got is not None:
            return got
    _, first, inv = np.unique(code, return_index=True, return_inverse=True)
    return (first, inv) if return_inverse else first


def _voxel_unique_cuda(code, return_inverse):
    """torch.unique(sorted) + amin-scatter first-occurrence == np.unique's
    stable-mergesort indices, exactly; None (no cuda / OOM) -> numpy path."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        ct = torch.from_numpy(code).cuda()
        _, inv = torch.unique(ct, sorted=True, return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), len(code),
                           dtype=torch.int64, device=ct.device)
        first.scatter_reduce_(0, inv, torch.arange(len(code), device=ct.device),
                              reduce="amin")
        first = first.cpu().numpy()
        return (first, inv.cpu().numpy()) if return_inverse else first
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        return None


def write_infer_run(run_dir, config, scene_stats):
    """infer_run.json: exact config + per-scene {scene, points, seconds};
    rewritten after every scene so a crash keeps completed numbers."""
    doc = dict(config)
    doc["adabn"] = os.environ.get("DG_INFER_ADABN") == "1"
    doc["apcotta"] = os.environ.get("DG_INFER_APCOTTA") == "1"
    doc["tta_views"] = int(os.environ.get("DG_INFER_TTA", "0") or 0)
    doc["save_probs"] = os.environ.get("TT_SAVE_PROBS") == "1"
    doc["scenes"] = scene_stats
    doc["total_points"] = int(sum(s["points"] for s in scene_stats))
    doc["total_seconds"] = round(sum(float(s["seconds"]) for s in scene_stats), 3)
    with open(os.path.join(run_dir, "infer_run.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return doc


def exclude_class_idx(class_names):
    """EXCLUDE_CLASSES env (csv of class names) -> sorted index list; [] when
    unset. Unknown names raise; at least one class must survive."""
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
        raise ValueError("EXCLUDE_CLASSES excludes every class; nothing left "
                         "to predict")
    print(f"  [infer] masking classes: {', '.join(names)}; masked points fall "
          f"to their next-best class (confidence is post-mask)", flush=True)
    return idx


def apply_class_mask(prob, exclude_idx):
    """Zero excluded prob columns and renormalize in place; no-op on []."""
    if not exclude_idx:
        return prob
    import numpy as np
    prob[..., exclude_idx] = 0.0
    prob /= np.maximum(prob.sum(-1, keepdims=True), 1e-12)
    return prob


def gpu_name():
    """Real CUDA device name for logs/metadata."""
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def lovasz_softmax_flat(probas, labels):
    """Lovász-Softmax (Berman et al. 2018) on (N, C) probs / (N,) labels."""
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
        return probas.sum() * 0.0
    losses = []
    for c in torch.unique(labels):
        fg = (labels == c).float()
        errors = (fg - probas[:, int(c)]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, _grad(fg[perm])))
    return torch.stack(losses).mean()


def focal_loss(logits, labels, gamma, class_weights=None):
    """Alpha-balanced multiclass focal loss; masks ignore_index=-1."""
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
    """Weighted (label-smoothed) CE or focal, + Lovász. All-ignored batches
    return a finite zero-grad value (CE ignore_index=-1 would give NaN)."""
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


def drop_corrupt_tile(path):
    """Remove a truncated cached tile and its scene's .done so the next prep
    re-tiles it (Modal preemption mid-commit can persist half-written npz)."""
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


def _tile_key(p):
    """Cache identity of one tile: a re-prepped tile reuses its basename."""
    try:
        st = os.stat(p)
        return f"{os.path.basename(p)}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        return f"{os.path.basename(p)}|0|0"


def scan_class_balance(tile_paths, num_classes, cache_path=None, with_counts=False):
    """Parallel scan of cached tiles' 'lab' -> (class_counts, present_mask) and,
    with_counts, the per-tile (tiles, classes) count matrix; optionally cached,
    keyed on every tile's (basename, mtime, size)."""
    import numpy as np
    keys = np.array([_tile_key(p) for p in tile_paths])
    if cache_path and os.path.exists(cache_path):
        try:
            cz = np.load(cache_path, allow_pickle=False)
            if "tile_keys" not in cz.files or "per_tile" not in cz.files:
                raise ValueError("pre-per-tile cache format")
            if (cz["tile_keys"].shape == keys.shape
                    and bool(np.all(cz["tile_keys"] == keys))
                    and int(cz["num_classes"]) == num_classes):
                print(f"  class balance: loaded cache ({len(tile_paths)} tiles)", flush=True)
                pt = cz["per_tile"].astype(np.int64)
                return (pt.sum(0), pt > 0, pt) if with_counts else (pt.sum(0), pt > 0)
        except Exception as e:
            print(f"  class balance: ignoring unreadable cache ({e})", flush=True)

    def _scan(tp):
        try:
            lab = np.load(tp)["lab"]
        except Exception:
            return None
        v = lab[(lab >= 0) & (lab < num_classes)]
        return (np.bincount(v, minlength=num_classes).astype(np.int64)
                if v.size else np.zeros(num_classes, np.int64))

    from concurrent.futures import ThreadPoolExecutor
    print(f"  scanning {len(tile_paths)} tiles for class balance (parallel)…",
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
            "their scene(s) unmarked. Rerun (Modal auto-retries) to re-tile them.")
    class_counts, present_mask = per_tile.sum(0), per_tile > 0
    if cache_path:
        try:
            np.savez(cache_path, tile_keys=keys, per_tile=per_tile,
                     num_classes=np.int64(num_classes))
            print(f"  class balance: cached scan -> {cache_path}", flush=True)
        except Exception as e:
            print(f"  class balance: could not write cache ({e})", flush=True)
    return ((class_counts, present_mask, per_tile) if with_counts
            else (class_counts, present_mask))


def class_weights_np(class_counts, beta, cap, absent_to_one=False):
    """Inverse-frequency^beta weights, mean-normalized, clamped to [1/cap, cap].
    absent_to_one pins zero-count classes at 1.0 (PTv3 variant)."""
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
    """Present classes below freq_frac x the median present-class count."""
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


_PROXY_FLOOR_PTS = 4096
_PROXY_FLOOR_TILES = 3
PROXY_PROTOCOL_TILES = "proxy_tiles_v2"
PROXY_PROTOCOL_SPHERES = "proxy_spheres_v2"


@contextlib.contextmanager
def fixed_np_seed(seed=20260724):
    """Deterministic global np.random for a startup block, caller state restored.
    Only safe before worker/prefetch threads exist."""
    import numpy as np
    st = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(st)


def pick_proxy_tiles(val_tiles, num_classes, budget, mode="coverage",
                     class_names=None, cache_path=None, viable=None):
    """Choose the mid-training Proxy Eval tile subset: an even stride over the
    val tiles, then greedily add the richest tiles (add-only, at most
    _PROXY_FLOOR_TILES per class) until every val-present class clears a point
    floor. viable(path)->bool pre-filters tiles the batch path would drop.
    Returns (tile_paths, report).
    'full' picks the same subset; the run ranks on the raw-scored Full Eval
    instead, so the result goes unused (kept so a switch back is cheap)."""
    import numpy as np
    if mode not in ("coverage", "full"):
        raise ValueError(f"PROXY_SAMPLING={mode!r} is not a sampling mode. Set "
                         "PROXY_SAMPLING to 'full' or 'coverage'.")
    if not val_tiles:
        raise RuntimeError("no val tiles to proxy-score - re-run prep with a "
                           "non-empty val split")
    cname = lambda c: (class_names[c] if class_names else str(c))
    bname = [os.path.basename(p) for p in val_tiles]
    _, _, per_tile = scan_class_balance(val_tiles, num_classes,
                                        cache_path=cache_path, with_counts=True)
    total = per_tile.sum(0)
    inventory = [c for c in range(num_classes) if total[c] > 0]

    picked, seen, ok, why = [], set(), {}, []

    def _ok(i):
        if viable is None:
            return True
        if i not in ok:
            try:
                ok[i] = bool(viable(val_tiles[i]))
            except (NonFiniteXYZ, DatasetExhausted):
                raise
            except Exception as e:
                ok[i] = False
                if not why:
                    why.append(f"{bname[i]}: {e}")
        return ok[i]

    def _add(i):
        if i in seen or not _ok(i):
            return False
        seen.add(i); picked.append(i)
        return True

    def _richest(c):
        cand = sorted((i for i in range(len(val_tiles))
                       if per_tile[i, c] > 0 and i not in seen),
                      key=lambda i: (-int(per_tile[i, c]), bname[i]))
        return next((i for i in cand if _ok(i)), None)

    stride = max(1, len(val_tiles) // budget)
    order = list(range(0, len(val_tiles), stride))
    order += [i for i in range(len(val_tiles)) if i not in set(order)]

    def _fill(n):
        for i in order:
            if len(picked) >= n:
                break
            _add(i)

    _fill(budget)
    counts = (per_tile[picked].sum(0) if picked else np.zeros(num_classes, np.int64))
    rarest = sorted(inventory, key=lambda c: (int(total[c]), cname(c)))
    covers = {}
    for c in rarest:
        for _ in range(_PROXY_FLOOR_TILES):
            if counts[c] >= _PROXY_FLOOR_PTS:
                break
            got = _richest(c)
            if got is None or not _add(got):
                break
            covers.setdefault(bname[got], []).append(cname(c))
            counts += per_tile[got]

    paths = [val_tiles[i] for i in sorted(picked)]
    if not paths:
        raise RuntimeError(
            f"no val tile survived the batch-path viability check "
            f"({sum(1 for v in ok.values() if not v)}/{len(val_tiles)} rejected"
            + (f"; first: {why[0]}" if why else "")
            + "), so every checkpoint would proxy-score mIoU 0 and "
            "final_model.pth would freeze at the first val pass. Delete this "
            "dataset's prep cache and re-run prep, then relaunch.")
    counts = per_tile[picked].sum(0)
    short = {cname(c): [int(counts[c]), _PROXY_FLOOR_PTS]
             for c in inventory if counts[c] < _PROXY_FLOOR_PTS}
    rep = {"mode": mode, "budget": budget, "n_tiles": len(paths),
           "inventory": inventory, "floor_points": _PROXY_FLOOR_PTS,
           "tiles": [bname[i] for i in sorted(picked)], "covers": covers,
           "per_class_picked": {cname(c): int(counts[c]) for c in inventory},
           "shortfall": short}
    rep["text"] = (
        f"  proxy val: mode={mode}  tiles={len(paths)}/{budget}  "
        f"inventory={len(inventory)} class(es)  picked GT: "
        + ", ".join(f"{cname(c)}={int(counts[c]):,}" for c in inventory)
        + (("  BELOW FLOOR(" + str(_PROXY_FLOOR_PTS) + "): "
            + ", ".join(f"{n}={v[0]:,}" for n, v in short.items())) if short else ""))
    return paths, rep


def split_scenes(ds_root):
    """Read the dataset's three split folders verbatim (never re-carve).
    Returns (name, pc_path) lists."""
    import glob
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]

    def _items(split):
        return [(stem(p), p)
                for p in sorted(glob.glob(f"{ds_root}/{split}/*.npz"))]

    train, val, test = _items("train"), _items("val"), _items("test")
    if not train:
        raise FileNotFoundError(f"No canonical scenes under {ds_root}/train")
    return train, val, test


def validate_cache(prep_dir, sig):
    """Refuse a prep cache built with different settings. True if the
    signature was newly written."""
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
    with open(meta_path, "w") as f:
        json.dump(sig, f, indent=2)
    return True


def score_ious(pred, lab, num_classes):
    """Per-class (intersection, union, gt_count) over already-valid-masked
    prediction/label arrays; one bincount pass instead of 3 masks per class.
    pred must be non-negative (argmax output) - bincount raises otherwise."""
    import numpy as np
    pred = np.asarray(pred, dtype=np.int64)
    lab = np.asarray(lab, dtype=np.int64)
    ok = (lab >= 0) & (lab < num_classes)
    cm = np.bincount(lab[ok] * num_classes + pred[ok],
                     minlength=num_classes * num_classes
                     ).reshape(num_classes, num_classes)
    inter = cm.diagonal().copy()
    gt = cm.sum(1)
    npred = np.bincount(pred, minlength=num_classes)[:num_classes]
    union = gt + npred - inter
    return inter, union, gt


class VoxelVoteAccum:
    """Streaming per-voxel vote accumulator for the voted full evals: tiles
    append per-point (voxel_key, weighted_prob, xyz); past max_rows the buffer
    reduces to per-voxel sums, so a 100M-point scene peaks at O(unique voxels
    + buffer) RAM instead of O(all overlapping tile points at once). Stable
    order keeps rep xyz == first point ever seen per voxel; votes stay the
    same float64 bincount sums (summation grouping aside)."""

    def __init__(self, num_classes, max_rows=None):
        self.C = num_classes
        self.max_rows = max_rows or int(os.environ.get("EVAL_VOTE_BUFFER",
                                                       "20000000"))
        self.keys, self.votes, self.xyz = [], [], []
        self.rows = 0
        self.dirty = False

    def add(self, keys, votes, xyz):
        self.keys.append(keys); self.votes.append(votes); self.xyz.append(xyz)
        self.rows += len(keys)
        self.dirty = True
        if self.rows > self.max_rows:
            self._compact()
            self.max_rows = max(self.max_rows, self.rows + self.rows // 2)

    def _compact(self):
        import numpy as np
        K = self.keys[0] if len(self.keys) == 1 else np.concatenate(self.keys)
        first, inv = voxel_unique(K, return_inverse=True)
        V = self.votes[0] if len(self.votes) == 1 else np.concatenate(self.votes)
        votes = np.stack([np.bincount(inv, weights=V[:, c], minlength=len(first))
                          for c in range(self.C)], axis=1)
        P = self.xyz[0] if len(self.xyz) == 1 else np.concatenate(self.xyz)
        self.keys, self.votes, self.xyz = [K[first]], [votes], [P[first]]
        self.rows = len(first)
        self.dirty = False

    def result(self):
        """(per-voxel argmax pred, representative raw xyz) or None if empty."""
        if not self.rows:
            return None
        if self.dirty:
            self._compact()
        return self.votes[0].argmax(1), self.xyz[0]


def score_raw_from_voxels(rep_xyz, pred_u, raw_xyz, raw_lab, num_classes,
                          chunk=8_000_000):
    """NN-propagate per-voxel preds to labeled raw points and score, chunked:
    a one-shot query over 100M points materializes ~GB of float64 distances +
    int64 indices; chunks cap that, and unlabeled/out-of-range points are
    never queried at all. Labels outside [0, num_classes) now drop from acc's
    denominator too (KP/randlanet used to count them; PTv3 never did).
    Returns (inter, union, gt, correct, total)."""
    import numpy as np
    from scipy.spatial import cKDTree
    tree = cKDTree(rep_xyz)
    inter = np.zeros(num_classes, np.int64)
    union = np.zeros(num_classes, np.int64)
    gt = np.zeros(num_classes, np.int64)
    correct = total = 0
    for s in range(0, len(raw_xyz), chunk):
        rl = raw_lab[s:s + chunk]
        v = (rl >= 0) & (rl < num_classes)
        if not v.any():
            continue
        _, nn = tree.query(raw_xyz[s:s + chunk][v], workers=-1)
        rp, rl = pred_u[nn], rl[v]
        correct += int((rp == rl).sum()); total += len(rl)
        i_, u_, g_ = score_ious(rp, rl, num_classes)
        inter += i_; union += u_; gt += g_
    return inter, union, gt, correct, total


def load_xyz_label(npz_path):
    """Slim eval-scoring loader: xyz + label only - skips the intensity/
    ret_num/feat_*/rgb channels a 100M-point scene would otherwise
    materialize (GBs) just to score predictions."""
    import numpy as np
    z = np.load(npz_path)
    raw = z["xyz"]
    require_finite_xyz(raw, os.path.basename(npz_path))
    xyz = (raw - np.floor(raw.min(0))).astype(np.float32)
    lab = z["label"].astype(np.int32) if "label" in z.files \
        else np.full(len(xyz), -1, np.int32)
    return xyz, lab


def eval_metrics(t_inter, t_union, t_gt, correct, total, class_names, t_start,
                 n_scenes, label, extra=None, force_present=None):
    """Shared metrics dict (acc, mIoU variants, per-class IoU/GT) + summary
    print; `extra` carries script-specific tail keys. force_present = class
    indices scored even with no GT in this sample (the val inventory), so a
    collapsed class scores 0 instead of vanishing from the denominator."""
    import numpy as np
    num_classes = len(class_names)
    with np.errstate(invalid="ignore"):
        iou_per = t_inter / np.maximum(t_union, 1)
    gt_counts = [int(x) for x in t_gt.tolist()]
    present = [c for c in range(num_classes) if gt_counts[c] > 0]
    forced = [c for c in sorted(set(force_present or ())) if gt_counts[c] == 0]
    scored = sorted(present + forced)
    scored_iou = [float(iou_per[c]) for c in scored]
    present_mIoU = float(np.mean(scored_iou)) if scored_iou else 0.0
    extra = extra or {}
    m = {
        "overall_acc": correct / max(total, 1),
        "overall_mIoU": float(np.mean(iou_per)),
        "present_classes_mIoU": present_mIoU,
        "per_class_iou": {class_names[c]: float(iou_per[c]) for c in range(num_classes)},
        "per_class_gt_count": {class_names[c]: gt_counts[c] for c in range(num_classes)},
        "present_classes": [class_names[c] for c in present],
        "scored_classes": [class_names[c] for c in scored],
        "absent_classes": [class_names[c] for c in range(num_classes)
                           if gt_counts[c] == 0 and c not in forced],
        "forced_zero_classes": [class_names[c] for c in forced],
        "total_test_seconds": time.time() - t_start,
        "num_scenes": n_scenes,
        "num_raw_points_scored": int(total),
        **extra,
    }
    skipped = {k.split("_")[1]: extra[k] for k in ("skipped_tiles", "skipped_scenes")
               if k in extra}
    print(f"  [{label}] acc={m['overall_acc']:.4f}  "
          f"mIoU({num_classes}-way)={m['overall_mIoU']:.4f}  "
          # "mIoU(present N)" wording is load-bearing: the GUI's VAL_RE parses it
          f"mIoU(present {len(scored)})={m['present_classes_mIoU']:.4f}  "
          + f"absent={m['absent_classes']}  raw_pts={total:,}"
          + (f"  forced0={m['forced_zero_classes']}" if forced else "")
          + ("  skipped(" + ",".join(f"{k}={v}" for k, v in skipped.items()) + ")"
             if skipped else ""), flush=True)
    return m


def init_val_csv(val_csv, class_names):
    """Write the val_metrics.csv header if absent."""
    if os.path.exists(val_csv):
        return
    cols = (["epoch", "val_acc", "val_miou"]
            + [f"iou_{n}" for n in class_names] + ["protocol"])
    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(cols)


def append_val_row(val_csv, ep, m, class_names):
    """One val_metrics.csv row: epoch, acc, present-class mIoU, per-class IoUs,
    protocol. Only rows matching the run's ranking protocol seed
    BestCheckpoint - see row_protocol/ranking_protocol."""
    ious = [m["per_class_iou"][n] for n in class_names]
    with open(val_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ep, f"{m['overall_acc']:.4f}",
                                f"{m['present_classes_mIoU']:.4f}"]
                               + [f"{x:.4f}" for x in ious]
                               + [row_protocol(m)])


def _proxy_remedy(run_dir):
    """Both files must go together, and each costs something - say what."""
    return (f"Unset AUTO_RESUME to launch a fresh run, or delete BOTH "
            f"{run_dir}/val_metrics.csv "
            f"(this wipes the run's GUI plot history) and "
            f"{run_dir}/final_model.pth (the run is un-inferrable and "
            f"un-packageable until a later epoch re-crowns it), then relaunch. "
            f"Close the GUI inference page first - Windows holds the weights "
            f"open and the delete fails with WinError 32.")


def proxy_guard(run_dir, report, protocol, class_names, ranking="proxy"):
    """Stamp/verify the checkpoint-ranking protocol of run_dir - call ONCE at
    startup right after BestCheckpoint (VAL_EVERY would leave epochs unguarded).
    Hashes tile BASENAMES only: absolute paths differ across win/linux/Modal and
    would falsely wedge a cross-backend resume. Raises RuntimeError when the
    run's rows were ranked under another protocol or the checkpoint they
    crowned is gone.
    ranking='full' stores a bare marker - the full eval covers the whole val
    split, so there is no tile subset to pin. A proxy-mode signature keeps its
    exact pre-'full' shape so in-flight proxy runs still resume."""
    import hashlib
    sig = ({"ranking": "full"} if ranking == "full" else
           {"protocol": protocol, "mode": report["mode"],
            "floor_points": report["floor_points"],
            "inventory": [class_names[c] for c in report["inventory"]],
            "tiles_sha1": hashlib.sha1(
                "\n".join(sorted(report["tiles"])).encode("utf-8")).hexdigest()[:16]})
    path = f"{run_dir}/proxy_val.json"
    rows = other = 0
    val_csv = f"{run_dir}/val_metrics.csv"
    if os.path.exists(val_csv):
        with open(val_csv, newline="", encoding="utf-8", errors="replace") as f:
            raw = list(csv.reader(f))
        for r in raw[1:]:
            if not r:
                continue
            if r[-1] == ranking:
                rows += 1
            else:
                other += 1
    old = None
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = None
    # rows on the other scale under a signature that NAMES that scale = a mid-run switch
    switched = (bool(other) and isinstance(old, dict)
                and old.get("ranking", "proxy") != ranking)
    if not rows and not switched:
        atomic_json_save(sig, path)
        return True
    if old != sig:
        was = old.get("ranking", "proxy") if isinstance(old, dict) else None
        if was is not None and was != ranking:
            back = old.get("mode", "coverage") if was == "proxy" else "full"
            raise RuntimeError(
                f"{run_dir} ranks checkpoints on the {was} validation protocol "
                f"but PROXY_SAMPLING now asks for {ranking}; the two score on "
                f"different scales, so the resumed run would chase a seed it can "
                f"never match. Set PROXY_SAMPLING={back!r} to continue this run. "
                f"{_proxy_remedy(run_dir)}")
        bad = ("unreadable proxy_val.json" if not isinstance(old, dict) else
               {k: [old.get(k), sig[k]] for k in sig if old.get(k) != sig[k]})
        raise RuntimeError(
            f"{run_dir} holds {rows} proxy val row(s) ranked under a different "
            f"proxy protocol (changed: {bad}). Resuming would rank checkpoints "
            f"on two scales at once. {_proxy_remedy(run_dir)} (A run whose only "
            f"rows come from EVAL_ONLY is full-protocol and never trips this.)")
    if rows and not os.path.exists(f"{run_dir}/final_model.pth"):
        raise RuntimeError(
            f"{run_dir} holds {rows} {ranking} val row(s) but no final_model.pth: "
            f"the resume seed already sits above the crowned checkpoint, which "
            f"is gone, so the run would publish last-epoch weights instead. "
            f"Delete {run_dir}/val_metrics.csv (this wipes the run's GUI plot "
            f"history) so the seed drops back to -1.0 and a later epoch re-crowns "
            f"the run, or unset AUTO_RESUME to launch a fresh run.")
    return True


def require_finite_xyz(xyz, where):
    """Hard preflight: one non-finite coord poisons origin-shift/KNN and only
    surfaces later as a cryptic CUDA gather assert deep in the net."""
    import numpy as np
    bad = int((~np.isfinite(xyz)).any(1).sum())
    if bad:
        raise NonFiniteXYZ(
            f"{where}: {bad}/{len(xyz)} points have non-finite xyz (inf/NaN). "
            "Re-ingest the source data (CRS reprojection now rejects non-finite "
            "output) or remove the scene, then delete its prep cache and re-run")


def proxy_val(batches, forward, num_classes, class_names, label, n_units,
              protocol, inventory=None):
    """Fixed-budget val on subsampled points (the upstream RandLA/KPConv/
    Pointcept protocol): ranks checkpoints between full evals at O(1) cost in
    dataset size. batches yields (model_batch, label_tensor); forward(batch)
    -> (N, num_classes) logits, and `inventory` (pick_proxy_tiles' val class
    inventory) is the mIoU denominator, so a collapsed class scores 0 instead
    of dropping out. Numbers are NOT comparable to the raw-scored full
    protocol."""
    import numpy as np
    import torch
    t_i = np.zeros(num_classes, np.int64)
    t_u = np.zeros(num_classes, np.int64)
    t_g = np.zeros(num_classes, np.int64)
    correct = total = 0
    t0 = time.time()
    with torch.no_grad():
        for batch, lab in batches:
            pred = forward(batch).reshape(-1, num_classes).argmax(-1).cpu().numpy()
            lv = lab.reshape(-1).cpu().numpy()
            v = (lv >= 0) & (lv < num_classes)
            p, l = pred[v], lv[v]
            correct += int((p == l).sum()); total += int(v.sum())
            i_, u_, g_ = score_ious(p, l, num_classes)
            t_i += i_; t_u += u_; t_g += g_
    m = eval_metrics(t_i, t_u, t_g, correct, total, class_names, t0,
                     n_units, label,
                     extra={"protocol": protocol,
                            "scored_on": "subsampled_points"},
                     force_present=inventory)
    low = {class_names[c]: int(t_g[c]) for c in (inventory or ())
           if int(t_g[c]) < _PROXY_FLOOR_PTS}
    m["scored_below_floor"] = low
    if low:
        print(f"  [{label}] scored GT below the {_PROXY_FLOOR_PTS}-point floor: "
              + ", ".join(f"{n}={v:,}" for n, v in low.items()), flush=True)
    return m


def zero_channels() -> set:
    """Channels the user chose to feed as constant zeros (TT_ZERO_CHANNELS,
    comma-separated - set by the Infer page channel table)."""
    return {c.strip() for c in os.environ.get("TT_ZERO_CHANNELS", "").split(",")
            if c.strip()}


def scene_arrays(z, n):
    """(intensity, ret_num) from a scene npz - the ONE place missing-channel
    fallbacks are decided (intensity -> rgb gray -> zeros; ret_num -> zeros).
    TT_ZERO_CHANNELS entries are zeroed even when the data exists."""
    import numpy as np
    zc = zero_channels()
    if "intensity" in zc:
        intensity = np.zeros(n, np.float32)
    elif "intensity" in z:
        intensity = z["intensity"].astype(np.float32)
    elif "rgb" in z and "rgb" not in zc:   # a killed rgb never leaks through the fallback
        intensity = z["rgb"].astype(np.float32).mean(1) / 255.0
    else:
        intensity = np.zeros(n, np.float32)
    if "return_number" in zc:
        ret_num = np.zeros(n, np.float32)
    else:
        ret_num = (z["return_number"].astype(np.float32) if "return_number" in z
                   else (z["ret_num"].astype(np.float32) if "ret_num" in z
                         else np.zeros(n, np.float32)))
    return intensity, ret_num


def run_infer_scenes(scenes, predict, pred_dir, run_dir, infer_cfg, cls_txt=False):
    """--mode infer loop: predict(pc_path) -> (xyz, pred, intensity, conf,
    probs), written as <name>_pred.npz (+ _pred_CLS.txt) with the crash-safe
    per-scene infer_run.json rewrite."""
    import numpy as np
    # backstop for the GUI's probe-based pre-skip: any feat_* the run needs
    # that the scenes don't carry rides as zeros - missing data never aborts
    need = [n for n in (infer_cfg.get("features") or []) if n.startswith("feat_")]
    if need and scenes:
        # union across ALL scenes: a mixed folder must never abort mid-run
        missing = set()
        for pc_path in scenes:
            with np.load(pc_path) as z0:
                missing.update(n for n in need if n not in z0.files)
        if missing:
            os.environ["TT_ZERO_CHANNELS"] = ",".join(sorted(zero_channels() | missing))
            print(f"  [channels] not in the converted scenes -> fed as zeros: "
                  f"{', '.join(sorted(missing))}", flush=True)
    zc = zero_channels()
    if zc:   # permanent provenance: these predictions ran without these inputs
        infer_cfg = {**infer_cfg, "zeroed_channels": sorted(zc)}
    print(f"  [infer] labeling {len(scenes)} scene(s) -> {pred_dir}", flush=True)
    scene_stats = []
    for pc_path in scenes:
        name = os.path.splitext(os.path.basename(pc_path))[0]
        t0 = time.time()
        xyz, pred, inten, conf, probs = predict(pc_path)
        try:
            with np.load(pc_path) as z:
                crs = str(z["crs_wkt"]) if "crs_wkt" in z.files else None
                src = str(z["source_crs_wkt"]) if "source_crs_wkt" in z.files else None
        except OSError:
            crs = src = None
        write_pred(f"{pred_dir}/{name}_pred.npz", xyz, pred, inten, conf, probs,
                   crs_wkt=crs, source_crs_wkt=src)
        if cls_txt:
            np.savetxt(f"{pred_dir}/{name}_pred_CLS.txt", pred, fmt="%d")
        scene_stats.append({"scene": os.path.basename(pc_path),
                            "points": int(len(xyz)),
                            "seconds": round(time.time() - t0, 3)})
        write_infer_run(run_dir, infer_cfg, scene_stats)
        print(f"  [infer] {name}: {len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
    # exact wording matters: the GUI's _localize_paths rewrites this line
    print(f"  [infer] done: predictions in "
          f"_infer/{os.path.basename(os.path.dirname(pred_dir))}/predictions",
          flush=True)


def kp_load_canonical(npz_path):
    """Scene npz -> (xyz, intensity, ret_num, lab, feat_* extras). xyz is
    origin-offset before the float32 cast (UTM sub-meter precision)."""
    import numpy as np
    z = np.load(npz_path)
    raw = z["xyz"]
    require_finite_xyz(raw, os.path.basename(npz_path))
    xyz = (raw - np.floor(raw.min(0))).astype(np.float32)
    intensity, ret_num = scene_arrays(z, len(xyz))
    lab = z["label"].astype(np.int32) if "label" in z.files \
        else np.full(len(xyz), -1, np.int32)
    return xyz, intensity, ret_num, lab, scene_feats(z)


def scene_feats(z):
    """Every feat_* channel a scene npz carries."""
    import numpy as np
    return {k: z[k].astype(np.float32) for k in z.files if k.startswith("feat_")}


def _grid_pool_t(p, a, l, voxel, num_classes):
    """Torch core of kp_grid_subsample; raveled int64 keys keep
    np.unique(axis=0) order, float64 accumulators keep the numpy numerics."""
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
    """Voxel-grid subsample: barycenter points, mean attrs, majority labels
    (KPConv's grid_subsampling); CUDA when available."""
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
    """next() -> ready batch; `depth` threads prefetch. Errors re-raise at
    next(); call .shutdown() when the loop ends."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(depth)
    q = deque(ex.submit(make_batch) for _ in range(depth + 1))

    def nxt():
        q.append(ex.submit(make_batch))
        return q.popleft().result()

    nxt.shutdown = lambda: ex.shutdown(wait=False, cancel_futures=True)
    return nxt


def prefetch_map(fn, items, depth=None):
    """Ordered map with a bounded thread-prefetch window: overlaps CPU tile
    prep (zlib/scipy drop the GIL) with the consumer's GPU work. Errors in fn
    re-raise at the corresponding yield."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    depth = depth or min(4, os.cpu_count() or 4)
    with ThreadPoolExecutor(depth) as ex:
        q = deque()
        for x in items:
            q.append(ex.submit(fn, x))
            if len(q) > depth:
                yield q.popleft().result()
        while q:
            yield q.popleft().result()


def train_stride(chunk_xy):
    """Train tile stride = chunk_xy * TT_TRAIN_STRIDE (default 0.75).
    Val/test keep chunk_xy/2 - per-voxel voting needs the overlap."""
    return chunk_xy * float(os.environ.get("TT_TRAIN_STRIDE", "0.75"))


def train_stride_tag():
    """Prep-dir suffix for the stride factor."""
    # uniform tag for every factor: old untagged 0.5-era caches re-prep
    return f"_ts{float(os.environ.get('TT_TRAIN_STRIDE', '0.75')):g}"


def _savez_fast(path, **arrays):
    """np.savez_compressed at zlib level 1 (~2.4x faster, ~6% bigger)."""
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
    """save(path, **arrays) queues _savez_fast on a thread pool (zlib drops
    the GIL). Bounded queue for RAM backpressure; errors surface on save()/exit."""
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
    """Yield (x0, y0, idx) per non-empty chunk_xy tile; idx stays on device.
    np.arange origins keep cached-tile filenames stable."""
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
    """Scene -> overlapping grid-subsampled tiles cached as .npz (xyz,
    intensity, ret_num, feat_*, lab). None when the load failed (retries)."""
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    try:
        xyz, intensity, ret_num, lab, extras = kp_load_canonical(pc_path)
    except NonFiniteXYZ:
        raise
    except Exception as e:
        print(f"  skip {pc_path}: {e}", flush=True)
        return None
    intensity_n = np.clip(intensity, 0.0, 2.0).astype(np.float32)
    print(f"    {name}: {len(xyz):,} pts loaded in {time.time()-t0:.1f}s, tiling…",
          flush=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fnames = sorted(extras)
    P = torch.from_numpy(xyz).to(dev)
    A = torch.from_numpy(np.stack([intensity_n, ret_num]
                                  + [extras[n] for n in fnames],
                                  axis=1)).to(dev)
    L = torch.from_numpy(np.ascontiguousarray(lab)).to(dev)
    n_tiles = 0
    with npz_save_pool() as save:
        for x0, y0, idx in tile_xy_indices(P, chunk_xy, stride):
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
    """Idempotent KP prep: validate cache signature, tile each un-.done scene
    via tile_fn(name, pc_path, out_dir, split). Returns (train, val, test)."""
    print(f"  ensuring preprocessed cache -> {prep_dir}", flush=True)
    for split in ("train", "val", "test"):
        os.makedirs(f"{prep_dir}/{split}", exist_ok=True)
    train_list, val_list, test_list = split_scenes(ds_root)
    any_new = [validate_cache(prep_dir, sig)]

    def tile_remaining(items, out_dir, split):
        for name, pc_path in items:
            if os.path.exists(f"{out_dir}/{name}.done"):
                continue
            n = tile_fn(name, pc_path, out_dir, split)
            if n is not None:
                open(f"{out_dir}/{name}.done", "w").close()
            any_new[0] = True

    for split, items in (("train", train_list), ("val", val_list), ("test", test_list)):
        print(f"  [{split}] {len(items)} scenes", flush=True)
        tile_remaining(items, f"{prep_dir}/{split}", split)
    print("  preprocessing cache updated." if any_new[0]
          else "  all scenes already cached.", flush=True)
    return train_list, val_list, test_list


def kp_make_build_feat(logdk_feat, logdk_k,
                       spec=("intensity", "return_number")):
    """build_feat(xyz, intensity, ret_num, drop=(), extras=None) -> [1, *spec]
    (+ log d_k). Bias always first, never dropped; every spec channel IS
    droppable."""
    import numpy as np
    import density as dg
    spec = list(spec)

    def build_feat(xyz, intensity, ret_num, drop=(), extras=None):
        bias = np.ones((len(xyz), 1), np.float32)
        src = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
               "intensity": intensity, "return_number": ret_num,
               **(extras or {})}
        missing = [n for n in spec if n not in src]
        if missing:
            raise ValueError(f"feature channel(s) {missing} not available "
                             f"here; have {sorted(src)}")
        attrs = np.stack([src[n] for n in spec], axis=1).astype(np.float32)
        if len(drop):
            attrs[:, list(drop)] = 0.0
        cols = [bias, attrs]
        if logdk_feat:
            cols.append(dg.local_density_logdk(xyz, logdk_k)[:, None])
        return np.concatenate(cols, axis=1).astype(np.float32)

    build_feat.spec = spec
    return build_feat


def kp_make_sample_tile(build_feat, grid, max_pts, aug_color,
                        density_aug, coarsen_max, p_native):
    """sample_tile(tile_path, ...) -> (augmented+centered xyz, feat, lab) or
    None. aug_color = per-channel KEEP probability, one independent coin per
    channel per training tile."""
    import numpy as np
    import density as dg

    def sample_tile(tile_path, max_pts=max_pts, min_pts=32, training=True):
        z = np.load(tile_path)
        require_finite_xyz(z["xyz"], os.path.basename(tile_path))
        xyz, intensity, ret_num, lab = z["xyz"], z["intensity"], z["ret_num"], z["lab"]
        extras = feat_extras(z, build_feat.spec, os.path.basename(tile_path))
        if len(xyz) < min_pts:
            return None
        idx = np.arange(len(xyz))
        if len(idx) > max_pts:
            idx = np.random.choice(idx, max_pts, replace=False)
        xyz, intensity, ret_num, lab = xyz[idx], intensity[idx], ret_num[idx], lab[idx]
        extras = {n: v[idx] for n, v in extras.items()}
        if training and density_aug:
            g_eff = dg.effective_grid(grid, coarsen_max, p_native)
            if g_eff > grid:
                keep = dg.voxel_first_idx(xyz, g_eff)
                xyz, intensity, ret_num, lab = xyz[keep], intensity[keep], ret_num[keep], lab[keep]
                extras = {n: v[keep] for n, v in extras.items()}
        drop = (np.flatnonzero(np.random.rand(len(build_feat.spec)) > aug_color)
                if training else ())
        feat = build_feat(xyz, intensity, ret_num, drop=drop, extras=extras)
        geo_xyz = kp_augment(xyz) if training else xyz
        geo_xyz = (geo_xyz - geo_xyz.mean(0)).astype(np.float32)
        return geo_xyz, feat, lab.astype(np.int64)

    return sample_tile


def kp_make_run_dir(variant):
    """Fresh timestamped run dir: <OUTPUTS_ROOT>/runs/<utc>_<variant>."""
    from datetime import datetime, timezone
    run_id = datetime.now(timezone.utc).strftime(f"%Y%m%d_%H%M%S_{variant}")
    run_dir = f"{OUTPUTS_ROOT}/runs/{run_id}"
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    return run_id, run_dir


def kp_find_latest_checkpoint(opt_type, feature_modes, arch_hash=None,
                              features=None, skip_done=False):
    """Most recent run with checkpoints AND this script's recipe (optimizer,
    feature_mode, ordered feature spec - names, not width - and arch hash).
    Returns (run_dir, ckpt_path, epoch) or None."""
    import glob

    def _ep(p):
        return int(os.path.basename(p)[2:5])

    for rd in sorted(glob.glob(f"{OUTPUTS_ROOT}/runs/*"), reverse=True):
        if skip_done and os.path.exists(f"{rd}/DONE"):
            continue
        ckpts = glob.glob(f"{rd}/checkpoints/ep*.pth")
        if not ckpts:
            continue
        got_opt = fmode = ahash = None
        rc = {}
        try:
            with open(f"{rd}/run.json") as f:
                rc = json.load(f)
            got_opt = rc.get("optimizer", {}).get("type")
            fmode = rc.get("feature_mode")
            ahash = rc.get("arch_hash")
        except Exception:
            rc = {}
        if got_opt != opt_type:
            print(f"  resume: skipping {os.path.basename(rd)} "
                  f"(recipe mismatch: optimizer={got_opt})", flush=True)
            continue
        if fmode not in feature_modes:
            print(f"  resume: skipping {os.path.basename(rd)} "
                  f"(variant mismatch: feature_mode={fmode})", flush=True)
            continue
        if features is not None:
            got_feats = list(rc.get("features") or features)
            if got_feats != list(features):
                print(f"  resume: skipping {os.path.basename(rd)} "
                      f"(feature mismatch: {got_feats})", flush=True)
                continue
        if arch_hash is not None and ahash is not None and ahash != arch_hash:
            print(f"  resume: skipping {os.path.basename(rd)} "
                  f"(architecture mismatch: arch_hash={ahash})", flush=True)
            continue
        latest = validated_latest_ckpt(ckpts, _ep)
        if latest is None:
            continue
        return rd, latest, _ep(latest)
    return None


def kp_make_evaluate(forward, build_feat, grid, chunk_xy, num_classes,
                     class_names):
    """KP voted eval scored on the ORIGINAL raw points: center-weighted
    softmax votes per voxel, argmax, NN-propagate to raw, score vs raw GT.
    forward([(cxyz, feat)]) -> per-tile (N, C) logits list; scene_items are
    (name, pc_path, split_dir) triples."""
    import glob
    import numpy as np
    import torch

    def evaluate(scene_items, label):
        bs0 = bs = max(1, int(os.environ.get("EVAL_BATCH", "4")))
        t_inter = np.zeros(num_classes, dtype=np.int64)
        t_union = np.zeros(num_classes, dtype=np.int64)
        t_gt = np.zeros(num_classes, dtype=np.int64)
        correct = total = 0
        n_scenes = n_skipped_tiles = n_skipped_scenes = 0
        t_test = time.time()

        def prep_tile(tile):
            try:
                z = np.load(tile)
                xyz = z["xyz"]
            except Exception:
                drop_corrupt_tile(tile)
                return "corrupt"
            if len(xyz) < 32:
                return None
            feat = build_feat(xyz, z["intensity"], z["ret_num"],
                              extras=feat_extras(z, build_feat.spec,
                                                 os.path.basename(tile)))
            return xyz, (xyz - xyz.mean(0)).astype(np.float32), feat

        def forward_group(group):
            nonlocal bs
            if len(group) > 1:
                try:
                    return forward([(c, f) for _, c, f in group])
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
            outs = []
            for _, c, f in group:
                try:
                    outs.append(forward([(c, f)])[0])
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    outs.append(None)
            return outs

        with torch.no_grad():
            for name, pc_path, split_dir in scene_items:
                bs = bs0
                tiles = sorted(glob.glob(f"{split_dir}/{name}_x*.npz"))
                if not tiles:
                    n_skipped_scenes += 1
                    continue
                acc = VoxelVoteAccum(num_classes)
                group = []

                def flush():
                    nonlocal n_skipped_tiles
                    for (xyz, _, _), lg in zip(group, forward_group(group)):
                        if lg is None:
                            n_skipped_tiles += 1
                            continue
                        e = np.exp(lg - lg.max(1, keepdims=True))
                        prob = e / e.sum(1, keepdims=True)
                        cxy = (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
                        d = np.abs(xyz[:, :2] - cxy).max(1)
                        wgt = np.clip(1.0 - d / (chunk_xy / 2.0), 0.05, 1.0) ** 2
                        acc.add(np.floor(xyz / grid).astype(np.int64),
                                (prob * wgt[:, None]).astype(np.float32),
                                xyz.astype(np.float32))
                    group.clear()

                for item in prefetch_map(prep_tile, tiles):
                    if item == "corrupt":
                        n_skipped_tiles += 1
                        continue
                    if item is None:
                        continue
                    group.append(item)
                    if len(group) >= bs:
                        flush()
                flush()
                got = acc.result()
                if got is None:
                    n_skipped_scenes += 1
                    continue
                pred_u, rep_xyz = got
                del acc
                try:
                    raw_xyz, raw_lab = load_xyz_label(pc_path)
                except NonFiniteXYZ:
                    raise
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped_scenes += 1
                    continue
                i_, u_, g_, c_, t_ = score_raw_from_voxels(
                    rep_xyz, pred_u, raw_xyz, raw_lab, num_classes)
                correct += c_; total += t_
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
    """Sliding-window inference -> (pred, confidence, probs) per raw point.
    forward_prob(cxyz, feat) -> (N, C) softmax; exceptions skip the window.
    exclude_idx masks classes pre-argmax; conf/probs are post-mask."""
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
                cols += [extras[n][idx] for n in feat_names]
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
            raise RuntimeError(
                f"inference produced nothing: all {n_skipped} window(s) "
                f"failed (last error: {last_err})")
        miss = pred < 0
        if miss.any() and (~miss).any():
            _, nn = cKDTree(xyz[~miss]).query(xyz[miss])
            pred[miss] = pred[~miss][nn]
        elif miss.any():
            pred[:] = min(set(range(num_classes)) - set(exclude_idx or ()))
        return pred, conf, probs

    return predict_points


def kp_make_target_batches(scenes, make_batch, build_feat, grid,
                           chunk_xy, num_classes, cap=30):
    """AdaBN target batches over inference scenes - same windows/features
    predict_points will see. make_batch(cxyz, feat) -> model batch."""
    import numpy as np

    feat_names = [n for n in getattr(build_feat, "spec", []) if n.startswith("feat_")]

    def target_batches():
        seen = 0
        for pc_path in scenes:
            if seen >= cap:
                return
            z = np.load(pc_path)
            require_finite_xyz(z["xyz"], os.path.basename(pc_path))
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


def kp_resume_ladder(infer, eval_only, want_resume, find_latest, infer_input,
                     variant, opt_type, n_epochs):
    """train/eval/infer mode ladder -> (run_dir, resume_ckpt, start_epoch)."""
    resume_info = find_latest() if (want_resume or eval_only) else None
    if infer:
        run_dir = infer_dir(infer_input)
        os.makedirs(os.environ.get("TT_PRED_DIR") or f"{run_dir}/predictions",
                    exist_ok=True)
        return run_dir, None, 0
    if resume_info:
        run_dir, resume_ckpt, resume_epoch = resume_info
        run_id = os.path.basename(run_dir)
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        start_epoch = resume_epoch + 1
        verb = "EVAL-ONLY on" if eval_only else "RESUMING"
        print(f"  {verb} {run_id} from {os.path.basename(resume_ckpt)}"
              + ("" if eval_only else f" -> starting at epoch {start_epoch}/{n_epochs}"),
              flush=True)
        return run_dir, resume_ckpt, start_epoch
    if eval_only:
        raise RuntimeError(f"eval mode: no {opt_type}-recipe run with "
                           f"checkpoints found under /outputs")
    _, run_dir = kp_make_run_dir(variant)
    return run_dir, None, 0


def kp_load_mode_weights(net, optim, resume_ckpt, start_epoch, eval_only,
                         infer, weights, run_dir, n_epochs):
    """Resume/eval/infer weight loading -> the (possibly bumped) start_epoch."""
    import torch
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location="cuda", weights_only=True)
        net.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        print(f"  resumed weights{' + optimizer' if 'optim' in ckpt else ''} "
              f"at epoch {start_epoch}", flush=True)
    if eval_only:
        fm = (resolve_weights_path(weights)
              if weights else f"{run_dir}/final_model.pth")
        if weights and not os.path.exists(fm):
            raise FileNotFoundError(f"--weights not found: {fm}")
        if os.path.exists(fm):
            net.load_state_dict(torch.load(fm, map_location="cuda", weights_only=True)["model"])
            print(f"  EVAL-ONLY: loaded {fm}", flush=True)
        start_epoch = n_epochs
    if infer:
        fm = (resolve_weights_path(weights)
              if weights else None)
        if not fm or not os.path.exists(fm):
            raise FileNotFoundError(f"--mode infer requires --weights; not found: {fm}")
        ck = load_ckpt_safe(fm, map_location="cuda")
        # bare state-dict .pth (no 'model' wrapper) stays loadable for shared weights
        sd = ck["model"] if (isinstance(ck, dict) and "model" in ck) else ck
        net.load_state_dict(sd)
        ep_tag = ck.get("epoch", "?") if isinstance(ck, dict) else "?"
        print(f"  [infer] loaded {weights} (best-val epoch {ep_tag})", flush=True)
        start_epoch = n_epochs
    return start_epoch


def init_metrics_csv(run_dir):
    """metrics.csv with the shared per-epoch header (append-safe on resume)."""
    metrics_csv = f"{run_dir}/metrics.csv"
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "train_acc", "train_iou", "lr",
                "sec_per_iter", "sec_per_epoch", "gpu_mem_mb",
            ])
    return metrics_csv


def kp_proxy_batches(proxy_samples, make_batch, proxy_rep, err_msg):
    """Pre-drawn proxy samples -> (batch, labels); err_msg(bn, why, e) words
    the per-backbone rebuild-the-prep remedy."""
    for bn, s in proxy_samples:
        try:
            b, lab_t = make_batch([s])
        except Exception as e:
            why = ", ".join(proxy_rep["covers"].get(bn, [])) or "the stride base"
            raise RuntimeError(err_msg(bn, why, e)) from e
        yield b, lab_t


def kp_make_run_eval(net, forward, evaluate, make_batch, sample_tile,
                     pick_train_tile, best, val_csv, run_dir, val_items,
                     test_items, val_list, test_list, num_classes, class_names,
                     val_full, eval_only, proxy_batches, proxy_tiles,
                     proxy_rep):
    """Shared KP eval driver: mid-training proxy/full val with best tracking,
    plus the final PreciseBN + val/test pass that writes test_metrics.json.
    forward(batch) -> logits; proxy_batches() -> (batch, labels) generator."""
    import torch
    import density as dg

    def run_eval(ep, write_json=False):
        if not write_json:
            net.eval()
            m = (evaluate(val_items, f"val@ep{ep}") if val_full else
                 proxy_val(proxy_batches(), forward,
                           num_classes, class_names, f"val@ep{ep}",
                           len(proxy_tiles), PROXY_PROTOCOL_TILES,
                           inventory=proxy_rep["inventory"]))
            net.train()
            if best.update(m):
                atomic_torch_save({"model": net.state_dict(), "epoch": ep},
                                  best.final)
            append_val_row(val_csv, ep, m, class_names)
            return m
        if not eval_only:
            def _bn_batches(n=48):
                made = fails = 0
                while made < n:
                    if fails >= 200:
                        raise DatasetExhausted(
                            "PreciseBN: 200 consecutive tile failures - training "
                            "tiles can't build batches; re-run dataset prep")
                    s = sample_tile(pick_train_tile(), training=False)
                    if s is None:
                        fails += 1
                        continue
                    try:
                        b, _ = make_batch([s])
                    except Exception:
                        fails += 1
                        continue
                    fails = 0
                    made += 1
                    yield b
            dg.adabn_recalibrate(net, _bn_batches(),
                                 forward=lambda mdl, b: forward(b))
        net.eval()
        m = evaluate(val_items, f"val@ep{ep}")
        # deliberately no best.update, even in full mode: AdaBN above recalibrates BN, so this row is not comparable to the mid-training ones
        append_val_row(val_csv, ep, m, class_names)
        swapped = not eval_only and os.path.exists(best.final)
        if swapped:
            live_state = {k: v.clone() for k, v in net.state_dict().items()}
            net.load_state_dict(torch.load(best.final, map_location="cuda",
                                           weights_only=True)["model"])
            net.eval()
        mt = evaluate(test_items, f"test@ep{ep}")
        if swapped:
            net.load_state_dict(live_state)
        with open(f"{run_dir}/test_metrics.json", "w", encoding="utf-8") as fj:
            json.dump({"val": m, "test": mt,
                       "val_scenes": [n for n, _ in val_list],
                       "test_scenes": [n for n, _ in test_list]}, fj, indent=2)
        net.train()
        return m

    return run_eval


def kp_run_infer(run_dir, net, forward, kp_batch, build_feat, predict_points,
                 backbone, note_name, weights, infer_input, grid, chunk_xy,
                 grid_cli, chunk_cli, num_classes, class_names, feat_spec,
                 exc_idx, infer_adabn, neighbor_limits=None, infer_apcotta=False):
    """Dataset-free inference over <run_dir>/scenes: optional AdaBN or APCoTTA
    then per-scene predict via run_infer_scenes. neighbor_limits goes into the
    infer config only when given (kpconv restores its pyramid crop)."""
    import glob
    from datetime import datetime, timezone
    import numpy as np
    import density as dg

    if not infer_input:
        raise ValueError("--mode infer requires --infer-input <job_id>")
    if (grid_cli is not None and grid_cli != grid) or \
       (chunk_cli is not None and chunk_cli != chunk_xy):
        print(f"  [infer] note: {note_name} uses its trained geometry "
              f"(grid={grid}, chunk={chunk_xy}); --grid/--chunk-xy ignored.", flush=True)
    net.eval()
    scenes = sorted(glob.glob(f"{run_dir}/scenes/*.npz"))
    if not scenes:
        raise FileNotFoundError(f"No scenes under {run_dir}/scenes")
    pred_dir = os.environ.get("TT_PRED_DIR") or f"{run_dir}/predictions"
    infer_cfg = {"backbone": backbone, "mode": "infer",
                 "weights": weights,
                 "infer_input": infer_input, "num_classes": num_classes,
                 "class_names": class_names, "grid": grid, "chunk_xy": chunk_xy}
    if neighbor_limits is not None:
        infer_cfg["neighbor_limits"] = neighbor_limits
    infer_cfg.update({"features": feat_spec, "gpu": gpu_name(),
                      "exclude_classes": [class_names[i] for i in exc_idx],
                      "started_utc": datetime.now(timezone.utc).isoformat()})
    if infer_adabn and infer_apcotta:
        raise ValueError("DG_INFER_ADABN and DG_INFER_APCOTTA are both set; "
                         "unset one - they are alternative adaptation modes")
    if infer_apcotta:
        print("  [infer] APCoTTA: adapting BN on target tiles...", flush=True)
        dg.apcotta_adapt(
            net,
            kp_make_target_batches(scenes, kp_batch, build_feat,
                                   grid, chunk_xy, num_classes),
            logits_fn=lambda m, b: forward(b))
    elif infer_adabn:
        print("  [infer] AdaBN: recomputing BN stats on target tiles...", flush=True)
        dg.adabn_recalibrate(
            net,
            kp_make_target_batches(scenes, kp_batch, build_feat,
                                   grid, chunk_xy, num_classes),
            forward=lambda m, b: forward(b))
        net.eval()

    def _predict(pc_path):
        z = np.load(pc_path)
        raw = z["xyz"]
        require_finite_xyz(raw, os.path.basename(pc_path))
        xyz = (raw - np.floor(raw.min(0))).astype(np.float32)
        intensity_n, ret_num = scene_arrays(z, len(xyz))
        extras = feat_extras(z, feat_spec, os.path.basename(pc_path))
        pred, conf, probs = predict_points(xyz, intensity_n, ret_num,
                                           extras=extras)
        return raw, pred, intensity_n, conf, probs

    run_infer_scenes(scenes, _predict, pred_dir, run_dir, infer_cfg, cls_txt=True)


def kp_train_loop(net, optim, forward, seg_loss, make_batch, sample_tile,
                  pick_train_tile, lr_at, run_eval, best, run_dir, metrics_csv,
                  num_classes, start_epoch, n_epochs, epoch_steps, pack_n,
                  accum, checkpoint_gap, val_every, eval_only, grad_clip_fn,
                  reg_fn=None):
    """Shared KP training loop: prefetching, ACCUM steps with the CUDA-assert
    re-raise, epoch metrics, periodic checkpoints/val, final PreciseBN eval and
    best finalize. reg_fn() adds a regularizer to the loss (kpconv deform);
    grad_clip_fn() clips each optimizer step."""
    import traceback
    import numpy as np
    import torch

    LOG_EVERY = 50
    AMP = os.environ.get("TT_AMP") == "1"

    def _draw():
        for _ in range(1000):
            s = sample_tile(pick_train_tile(), training=True)
            if s is not None:
                return s
        raise DatasetExhausted(
            "1000 consecutive empty tile draws - training tiles are empty or "
            "too small; re-run dataset prep")
    prefetch = (make_prefetcher(
        lambda: make_batch([_draw() for _ in range(pack_n)]),
        depth=int(os.environ.get("TT_PREFETCH", "2")))
        if start_epoch < n_epochs else None)
    print(f"  starting at epoch {start_epoch}, up to {n_epochs}, "
          f"{epoch_steps} steps/epoch, pack {pack_n} x accum {accum}"
          f"{' [bf16 autocast]' if AMP else ''}", flush=True)
    t_run = time.time()
    ep = n_epochs - 1
    for ep in range(start_epoch, n_epochs):
        cur_lr = lr_at(ep)
        for g in optim.param_groups:
            g["lr"] = cur_lr * g.get("lr_mult", 1.0)
        net.train()
        ep_loss = 0.0
        ep_conf = torch.zeros(num_classes, num_classes, dtype=torch.long,
                              device="cuda")
        t_ep = time.time()
        n_steps = n_fwd = n_failed = 0
        print(f"  ep {ep:3d} starting (lr={cur_lr:.2e})…", flush=True)
        for step in range(epoch_steps):
            optim.zero_grad()
            n_ok = 0
            for _ in range(accum):
                try:
                    batch, lab_t = prefetch()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                        logits = forward(batch)
                        loss_seg = seg_loss(logits, lab_t)
                        reg = reg_fn() if reg_fn is not None else None
                        loss = (loss_seg if reg is None else loss_seg + reg) / accum
                    if not torch.isfinite(loss):
                        n_failed += 1
                        continue
                    loss.backward()
                    n_ok += 1; n_fwd += 1
                    ep_loss += loss.item() * accum
                    pred = logits.argmax(-1)
                    m = lab_t >= 0
                    ep_conf += torch.bincount(
                        lab_t[m] * num_classes + pred[m],
                        minlength=num_classes * num_classes,
                    ).reshape(num_classes, num_classes)
                except Exception as e:
                    if isinstance(e, (DatasetExhausted, NonFiniteXYZ)):
                        raise
                    # a device-side assert poisons the CUDA context (every later forward fails too), so re-raise with the remedy instead of burning the epoch
                    if (isinstance(e, RuntimeError)
                            and "out of memory" not in str(e)
                            and any(s in str(e) for s in
                                    ("CUDA error", "device-side assert",
                                     "illegal memory access"))):
                        raise RuntimeError(
                            "unrecoverable CUDA error in forward. If it is an "
                            "index/scatter-gather assert, a scene likely has "
                            "non-finite coords: re-ingest the data and delete the "
                            f"prep cache. Original: {e}") from e
                    n_failed += 1
                    if n_failed == 1:
                        print(f"  forward failed (first occurrence, step {step}): {e}",
                              flush=True)
                        traceback.print_exc()
            if n_ok:
                grad_clip_fn()
                optim.step()
                n_steps += 1
                if n_steps % LOG_EVERY == 0:
                    print(f"    ep {ep:3d} step {n_steps:4d}: "
                          f"loss={ep_loss/max(n_fwd,1):.4f}", flush=True)
        if n_steps == 0:
            raise RuntimeError(f"epoch {ep}: 0 optimizer steps ({n_failed} failed forwards).")
        if n_failed:
            print(f"  ep {ep:3d} note: {n_failed} failed forwards", flush=True)
        sec_per_epoch = time.time() - t_ep
        sec_per_iter = sec_per_epoch / max(n_steps, 1)
        conf = ep_conf.cpu().numpy()
        ep_inter = np.diag(conf)
        ep_union = conf.sum(0) + conf.sum(1) - ep_inter
        train_acc = int(np.trace(conf)) / max(int(conf.sum()), 1)
        with np.errstate(invalid="ignore"):
            train_iou = float(np.mean(ep_inter / np.maximum(ep_union, 1)))
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6
        row = [ep, f"{ep_loss/max(n_fwd,1):.6f}", f"{train_acc:.4f}",
               f"{train_iou:.4f}", f"{cur_lr:.6e}", f"{sec_per_iter:.4f}",
               f"{sec_per_epoch:.2f}", f"{gpu_mem:.1f}"]
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        print(f"  ep {ep:3d}: loss={ep_loss/max(n_fwd,1):.4f} acc={train_acc:.4f} "
              f"miou={train_iou:.4f} lr={cur_lr:.2e} s/iter={sec_per_iter:.3f} "
              f"s/epoch={sec_per_epoch:.1f}", flush=True)
        if (ep + 1) % checkpoint_gap == 0:
            atomic_torch_save({"model": net.state_dict(),
                               "optim": optim.state_dict(), "epoch": ep},
                              f"{run_dir}/checkpoints/ep{ep:03d}.pth")
        stop = stop_requested(ep)
        if (ep + 1) % val_every == 0 and ep != n_epochs - 1 and not stop:
            run_eval(ep)
        if stop:
            break

    if prefetch:
        prefetch.shutdown()

    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(ep, write_json=True)
    if not eval_only:
        best.finalize(lambda p: atomic_torch_save(
            {"model": net.state_dict(), "epoch": ep}, p))
        open(f"{run_dir}/DONE", "w").close()
    print(f"  total wall-clock: {(time.time() - t_run)/3600:.2f} h")


def ptv3_load_canonical(npz_path, color_src):
    """Scene npz -> (xyz, rgb, lab). color_src picks the 3 color channels
    (fallbacks intensity -> rgb -> mid-gray; intensity x255 to rgb scale).
    xyz origin-offset before the float32 cast (UTM precision + batch filter)."""
    import numpy as np
    z = np.load(npz_path)
    raw = z["xyz"]
    require_finite_xyz(raw, os.path.basename(npz_path))
    xyz = (raw - np.floor(raw.min(0))).astype(np.float32)
    def _itn():
        return np.repeat((z["intensity"].astype(np.float32) * 255.0)[:, None], 3, axis=1)
    # pick the source over REAL presence first; a zeroed pick DISABLES the color
    # input (neutral mid-gray, parity with scene_arrays' true zeros) - never
    # substitute the other live source for a channel the user killed
    if color_src != "rgb" and "intensity" in z:
        src = "intensity"
    elif "rgb" in z:
        src = "rgb"
    elif "intensity" in z:
        src = "intensity"
    else:
        src = None
    zc = zero_channels()
    if src in zc and not getattr(ptv3_load_canonical, "_zc_logged", False):
        ptv3_load_canonical._zc_logged = True
        print(f"  [channels] color input '{src}' zeroed -> neutral mid-gray constant",
              flush=True)
    if src is None or src in zc:
        rgb = np.full((len(xyz), 3), 128.0, dtype=np.float32)
    elif src == "rgb":
        rgb = z["rgb"].astype(np.float32)
    else:
        rgb = _itn()
    lab = z["label"].astype(np.int64) if "label" in z \
        else np.full(len(xyz), -1, np.int64)
    # clip here: an unclipped uint8 cast WRAPS the p95 bright tail (306 -> 50)
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
        except NonFiniteXYZ:
            raise
        except Exception as e:
            print(f"  skip {src}: {e}", flush=True); continue
        extras = scene_feats(np.load(src)) if src.endswith(".npz") else {}
        print(f"    [{fi+1}/{len(src_paths)}] {scene}: {len(xyz):,} pts "
              f"loaded in {time.time()-t0:.1f}s, tiling…", flush=True)
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        P = torch.from_numpy(np.ascontiguousarray(xyz)).to(dev)
        cols = {"rgb": rgb, "lab": lab, **extras}
        G = None
        if dev == "cuda":
            try:
                G = {n: torch.from_numpy(np.ascontiguousarray(v)).to(dev)
                     for n, v in cols.items()}
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()

        def _tile_scene(G):
            n_tiles = 0
            with npz_save_pool() as save:
                for x0, y0, idx in tile_xy_indices(P, chunk_xy, stride):
                    if len(idx) < 2048: continue
                    if G is None:
                        m = idx.cpu().numpy()
                        cut, txyz = {n: v[m] for n, v in cols.items()}, xyz[m]
                    else:
                        cut = {n: t[idx].cpu().numpy() for n, t in G.items()}
                        txyz = P[idx].cpu().numpy()
                    tile = {"xyz": txyz.astype(np.float32),
                            "rgb": cut["rgb"].astype(np.uint8),
                            "lab": cut["lab"].astype(np.int32)}
                    for n in extras:
                        tile[n] = cut[n].astype(np.float32)
                    save(f"{out_dir}/{scene}_x{int(x0)}_y{int(y0)}.npz", **tile)
                    n_tiles += 1
            return n_tiles

        try:
            n_tiles = _tile_scene(G)
        except RuntimeError as e:
            if G is None or "out of memory" not in str(e).lower():
                raise
            G = None
            torch.cuda.empty_cache()
            n_tiles = _tile_scene(None)
        print(f"      -> {n_tiles} tiles", flush=True)


def ptv3_ensure_prep(prep_dir, ds_root, chunk_xy, stride, load_canonical):
    import glob
    os.makedirs(f"{prep_dir}/train", exist_ok=True)
    os.makedirs(f"{prep_dir}/val",   exist_ok=True)
    os.makedirs(f"{prep_dir}/test",  exist_ok=True)
    print(f"  ensuring preprocessed cache -> {prep_dir}", flush=True)
    # signature stamp: PREP_DIR doesn't encode the class layout, so a rebuilt dataset would silently reuse old-index tiles
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
                              "split_mode": sp.get("mode")})
    any_new = [False]
    def already_tiled(out_dir, scene):
        return bool(glob.glob(f"{out_dir}/{scene}_x*.npz"))
    def tile_remaining(src_paths, out_dir, chunk, stride):
        for src in src_paths:
            scene = os.path.splitext(os.path.basename(src))[0]
            if already_tiled(out_dir, scene): continue
            ptv3_tile_and_save([src], out_dir, chunk, stride, load_canonical)
            any_new[0] = True
    train_paths = sorted(glob.glob(f"{ds_root}/train/*.npz"))
    val_paths   = sorted(glob.glob(f"{ds_root}/val/*.npz"))
    test_paths  = sorted(glob.glob(f"{ds_root}/test/*.npz"))
    if not train_paths:
        raise FileNotFoundError(f"No canonical scenes under {ds_root}/train")
    print(f"  [train] {len(train_paths)} canonical scenes", flush=True)
    tile_remaining(train_paths, f"{prep_dir}/train", chunk_xy,
                   train_stride(chunk_xy))
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
        raise ValueError(f"{arch} has ONE 3-wide color slot; pick rgb OR "
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
    """Linear warmup then cosine decay; stateless, so resume is trivial."""
    import numpy as np
    warm = max(1, int(round(warmup_pct * n_epochs)))
    if ep < warm:
        return base_lr * (ep + 1) / warm
    prog = (ep - warm) / max(1, n_epochs - warm)
    return float(0.5 * base_lr * (1.0 + np.cos(np.pi * prog)))


RESUME_RECIPE_KEYS = ("grid_size", "chunk_xy", "features", "n_epochs",
                      "num_classes", "class_names")


def find_latest_unfinished_run(suffix, cfg=None):
    """Latest unfinished run ending in `suffix` with checkpoints. DONE runs
    and RESUME_RECIPE_KEYS mismatches vs `cfg` are skipped (weights must match
    the run.json they publish). Returns (run_dir, ckpt_path, epoch) or None."""
    import glob
    def _ep(p):
        return int(os.path.basename(p)[2:5])
    def _n(v):
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
            latest = validated_latest_ckpt(ckpts, _ep)
            if latest is None:
                continue
            return rd, latest, _ep(latest)
    return None


def ptv3_make_evaluate(forward, build_feat, feat_spec, grid, chunk_xy,
                       num_classes, class_names):
    """PTv3-family voted eval scored on the ORIGINAL raw points (KP protocol).
    forward(batch_dict) -> (N, C) logits; scene_items are (name, load_raw,
    split_dir) triples; load_raw() -> (xyz, label) only."""
    import glob
    import numpy as np
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def evaluate(scene_items, label):
        bs0 = bs = max(1, int(os.environ.get("EVAL_BATCH", "4")))
        t_inter = np.zeros(num_classes, dtype=np.int64)
        t_union = np.zeros(num_classes, dtype=np.int64)
        t_gt    = np.zeros(num_classes, dtype=np.int64)
        correct = total = 0
        n_scenes = n_skipped_tiles = n_skipped_scenes = 0
        t_test = time.time()

        def _run(group):
            lens = [len(g[2]) for g in group]
            coord = torch.from_numpy(np.concatenate([g[2] for g in group])).to(dev)
            featt = torch.from_numpy(np.concatenate([g[3] for g in group])).to(dev)
            gc = np.ascontiguousarray(np.concatenate([g[4] for g in group]))
            grid_coord = torch.from_numpy(gc).long().to(dev)
            offset = torch.tensor(np.cumsum(lens), dtype=torch.long).to(dev)
            lg = forward({"coord": coord, "grid_coord": grid_coord,
                          "feat": featt, "offset": offset}
                         ).cpu().numpy().astype(np.float32)
            return np.split(lg, np.cumsum(lens)[:-1])

        def prep_tile(tile):
            try:
                z = np.load(tile)
                xyz, rgb = z["xyz"].astype(np.float32), z["rgb"]
            except Exception:
                drop_corrupt_tile(tile)
                return "corrupt"
            if len(xyz) < 64:
                return None
            ex = feat_extras(z, feat_spec, os.path.basename(tile))
            cxyz = (xyz - xyz.mean(0, keepdims=True, dtype=np.float64)
                    ).astype(np.float32)
            ok = (np.isfinite(cxyz).all(1)
                  & (np.abs(cxyz[:, :2]).max(1) <= chunk_xy)
                  & (np.abs(cxyz[:, 2]) <= 200.0))
            if int(ok.sum()) < 64:
                return None
            xyz, rgb, cxyz = xyz[ok], rgb[ok], cxyz[ok]
            ex = {n: v[ok] for n, v in ex.items()}
            vk = np.floor(cxyz / grid).astype(np.int64)
            first, inverse = voxel_unique(vk, return_inverse=True, gpu=False)
            vx = cxyz[first].astype(np.float32)
            feat = build_feat(vx, rgb[first].astype(np.float32) / 255.0,
                              {n: v[first] for n, v in ex.items()})
            gc = vk[first] - vk[first].min(0)
            return xyz, inverse, vx, feat, gc

        def forward_group(group):
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
                bs = bs0
                tiles = sorted(glob.glob(f"{split_dir}/{name}_x*.npz"))
                if not tiles:
                    n_skipped_scenes += 1; continue
                acc = VoxelVoteAccum(num_classes)
                group = []

                def flush():
                    nonlocal n_skipped_tiles
                    for (xyz, inverse, *_), lg in zip(group, forward_group(group)):
                        if lg is None:
                            n_skipped_tiles += 1
                            continue
                        e = np.exp(lg - lg.max(1, keepdims=True))
                        prob = (e / e.sum(1, keepdims=True))[inverse]
                        cxy = (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
                        d = np.abs(xyz[:, :2] - cxy).max(1)
                        wgt = np.clip(1.0 - d / (chunk_xy / 2.0), 0.05, 1.0) ** 2
                        acc.add(np.floor(xyz / grid).astype(np.int64),
                                (prob * wgt[:, None]).astype(np.float32),
                                xyz.astype(np.float32))
                    group.clear()

                for item in prefetch_map(prep_tile, tiles):
                    if item == "corrupt":
                        n_skipped_tiles += 1
                        continue
                    if item is None:
                        continue
                    group.append(item)
                    if len(group) >= bs:
                        flush()
                flush()
                got = acc.result()
                if got is None:
                    n_skipped_scenes += 1; continue
                pred_u, rep_xyz = got
                del acc
                try:
                    raw_xyz, raw_lab = load_raw()
                except NonFiniteXYZ:
                    raise
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped_scenes += 1; continue
                i_, u_, g_, c_, t_ = score_raw_from_voxels(
                    rep_xyz, pred_u, raw_xyz, raw_lab, num_classes)
                correct += c_; total += t_
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


def ptv3_build_feat(feat_spec, cxyz, rgbf, extras=None, drop=(),
                    logdk_feat=False, logdk_k=8, normal_block=False):
    """Spec-ordered features (+ 3-wide zero normal block for the SSL stems,
    + optional log d_k). `drop` = spec indices to zero (train-time feature
    dropout); the normal slot and log d_k never drop."""
    import numpy as np
    import density as dg
    cols = []
    for i, n in enumerate(feat_spec):
        if n in ("rgb", "intensity"):
            c = rgbf
        elif n in ("x", "y", "z"):
            c = cxyz[:, "xyz".index(n):"xyz".index(n) + 1]
        else:
            c = extras[n][:, None]
        cols.append(np.zeros_like(c, dtype=np.float32) if i in drop else c)
    if normal_block:
        cols.append(np.zeros((len(cxyz), 3), np.float32))
    if logdk_feat:
        cols.append(dg.local_density_logdk(cxyz, logdk_k)[:, None])
    return np.concatenate(cols, axis=1).astype(np.float32)


def ptv3_in_bounds(xyz, chunk_xy):
    """Finite points inside the chunk window (200 m |z| guard)."""
    import numpy as np
    return (np.isfinite(xyz).all(1)
            & (np.abs(xyz[:, :2]).max(1) <= chunk_xy)
            & (np.abs(xyz[:, 2]) <= 200.0))


def ptv3_viable(path, chunk_xy):
    """>=64 in-bounds points after centering - the batcher's keep rule."""
    import numpy as np
    xyz = np.load(path)["xyz"].astype(np.float32)
    xyz = (xyz - xyz.mean(0, keepdims=True, dtype=np.float64)).astype(np.float32)
    return int(ptv3_in_bounds(xyz, chunk_xy).sum()) >= 64


def ptv3_scene_of(path):
    """Tile path -> scene name (strips the _x<...> tile suffix)."""
    b = os.path.basename(path)
    return b.rsplit("_x", 1)[0]


def ptv3_raw_loader(ds_root, split, name):
    """Deferred raw-scene loader for the eval scene items."""
    return lambda: load_xyz_label(f"{ds_root}/{split}/{name}.npz")


def ptv3_eval_items(ds_root, prep_dir, hold, test_tiles):
    """(val_items, test_items) scene triples for ptv3_make_evaluate."""
    val_items = [(n, ptv3_raw_loader(ds_root, "val", n), f"{prep_dir}/val")
                 for n in sorted(hold)]
    test_items = [(n, ptv3_raw_loader(ds_root, "test", n), f"{prep_dir}/test")
                  for n in sorted({ptv3_scene_of(p) for p in test_tiles})]
    print(f"  eval set: {len(val_items)} holdout(val) + {len(test_items)} test scenes",
          flush=True)
    return val_items, test_items


def ptv3_make_batcher(build_feat, feat_spec, grid, chunk_xy, augment_xyz,
                      aug_enable, aug_color, rare_oversample, rare_cols,
                      rare_center_prob, density_aug, coarsen_max, p_native):
    """to_ptv3_batch(tiles, training=True) -> (Point batch dict, labels)."""
    import numpy as np
    import torch
    import density as dg

    def to_ptv3_batch(tiles_for_batch, training=True):
        coords, feats, labels, offsets, grid_coords = [], [], [], [], []
        running = 0
        for tile in tiles_for_batch:
            z = np.load(tile)
            xyz, rgb, lab = z["xyz"], z["rgb"], z["lab"]
            ex = feat_extras(z, feat_spec, os.path.basename(tile))
            # 80k memory bound: a random ~30m crop while training, deterministic decimation for the proxy val (an uncropped tile OOMs outside the train loop's OOM guard)
            if len(xyz) > 80000:
                if not training:
                    idx = np.arange(0, len(xyz), -(-len(xyz) // 80000))
                else:
                    c = None
                    if (rare_oversample and rare_cols
                            and np.random.rand() < rare_center_prob):
                        ridx = np.where(np.isin(lab, rare_cols))[0]
                        if len(ridx):
                            c = xyz[ridx[np.random.randint(len(ridx))]]
                    if c is None:
                        c = xyz[np.random.randint(len(xyz))]
                    d2 = np.sum((xyz[:, :2] - c[:2]) ** 2, axis=1)
                    idx = np.where(d2 < 15.0 ** 2)[0]
                    if len(idx) > 80000:
                        idx = np.random.choice(idx, 80000, replace=False)
                xyz, rgb, lab = xyz[idx], rgb[idx], lab[idx]
                ex = {n: v[idx] for n, v in ex.items()}
            xyz = xyz.astype(np.float32)
            if training and aug_enable:
                xyz = augment_xyz(xyz)
            # float64 mean: a float32 mean at UTM magnitudes empties the window cut
            xyz = (xyz - xyz.mean(0, keepdims=True, dtype=np.float64)
                   ).astype(np.float32)
            ok = ptv3_in_bounds(xyz, chunk_xy)
            if int(ok.sum()) < 64:
                continue
            xyz = xyz[ok]; rgb = rgb[ok]; lab = lab[ok]
            ex = {n: v[ok] for n, v in ex.items()}
            # grid_coord MUST come from the same keys used to dedup: a different phase can collapse two voxels onto one grid_coord (CUDA assert)
            g_eff = (dg.effective_grid(grid, coarsen_max, p_native)
                     if (training and density_aug) else grid)
            keys = np.floor(xyz / g_eff).astype(np.int64)
            uniq = voxel_unique(keys)
            xyz = xyz[uniq]; rgb = rgb[uniq]; lab = lab[uniq]
            ex = {n: v[uniq] for n, v in ex.items()}
            xyz = xyz.astype(np.float32)
            fdrop = (np.flatnonzero(np.random.rand(len(feat_spec)) > aug_color)
                     if training else ())
            feat = build_feat(xyz, rgb.astype(np.float32) / 255.0, ex, drop=fdrop)
            coords.append(xyz); feats.append(feat); labels.append(lab)
            grid_coords.append(keys[uniq])
            running += len(xyz)
            offsets.append(running)
        coord = torch.from_numpy(np.concatenate(coords).astype(np.float32)).cuda()
        feat  = torch.from_numpy(np.concatenate(feats).astype(np.float32)).cuda()
        label = torch.from_numpy(np.concatenate(labels).astype(np.int64)).cuda()
        offset = torch.tensor(offsets, dtype=torch.long).cuda()
        gc = np.concatenate(grid_coords)
        gc -= gc.min(0, keepdims=True)
        grid_coord = torch.from_numpy(np.ascontiguousarray(gc)).long().cuda()
        return {"coord": coord, "grid_coord": grid_coord,
                "feat": feat, "offset": offset}, label

    return to_ptv3_batch


def ptv3_make_predict_scene(forward_logits, load_canonical, build_feat,
                            feat_spec, grid, chunk_xy, infer_tta,
                            num_classes, exclude_idx=None):
    """predict_scene(path) -> (raw xyz, pred, gray, conf, probs): chunked
    voxel inference (+ optional scale TTA); unpredicted points backfill from
    the nearest predicted neighbour."""
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    save_probs = os.environ.get("TT_SAVE_PROBS") == "1"

    def _predict_scene(scene_path):
        xyz, rgb, _ = load_canonical(scene_path)
        z0 = np.load(scene_path)
        ex0 = feat_extras(z0, feat_spec, os.path.basename(scene_path))
        pred = np.full(len(xyz), -1, np.int64)
        conf = np.zeros(len(xyz), np.float32)
        probs = np.zeros((len(xyz), num_classes), np.float16) if save_probs else None
        with torch.no_grad():
            for idx in xy_chunk_groups(xyz, chunk_xy, min_pts=64):
                w0 = (xyz[idx] - xyz[idx].mean(0)).astype(np.float32)
                rgbf = rgb[idx].astype(np.float32) / 255.0
                exw = {n: v[idx] for n, v in ex0.items()}
                views = [1.0] + (list(np.linspace(0.85, 1.2, infer_tta))
                                 if infer_tta else [])
                pprob = None
                for s in views:
                    w = (w0 * s).astype(np.float32)
                    keys = np.floor(w / grid).astype(np.int64)
                    first, inverse = voxel_unique(keys, return_inverse=True)
                    vx = w[first]
                    feat = build_feat(vx, rgbf[first],
                                      {n: v[first] for n, v in exw.items()})
                    coord = torch.from_numpy(vx).cuda()
                    featt = torch.from_numpy(feat).cuda()
                    offset = torch.tensor([len(vx)], dtype=torch.long).cuda()
                    gc = keys[first] - keys[first].min(0)
                    grid_coord = torch.from_numpy(np.ascontiguousarray(gc)).long().cuda()
                    lg = forward_logits({"coord": coord, "grid_coord": grid_coord,
                                         "feat": featt, "offset": offset})
                    vp = torch.softmax(lg.float(), -1).cpu().numpy()[inverse]
                    pprob = vp if pprob is None else pprob + vp
                pprob /= np.maximum(pprob.sum(-1, keepdims=True), 1e-12)
                pprob = apply_class_mask(pprob, exclude_idx)
                pred[idx] = pprob.argmax(-1)
                conf[idx] = pprob.max(-1)
                if save_probs:
                    probs[idx] = pprob.astype(np.float16)
        miss = pred < 0
        if miss.any() and (~miss).any():
            _, nn = cKDTree(xyz[~miss]).query(xyz[miss])
            pred[miss] = pred[~miss][nn]
        elif miss.any():
            pred[:] = min(set(range(num_classes)) - set(exclude_idx or ()))
        return z0["xyz"], pred, rgb[:, 0] / 255.0, conf, probs

    return _predict_scene


def ptv3_proxy_batches(proxy_tiles, batch_size, to_batch, viable, proxy_rep,
                       chunk_xy, prep_dir):
    """Zero-arg generator over the proxy-val tiles; a tile the batcher drops
    is a hard protocol error."""
    def _proxy_batches():
        for i in range(0, len(proxy_tiles), batch_size):
            chunk = proxy_tiles[i:i + batch_size]
            batch, lab = to_batch(chunk, training=False)
            if len(batch["offset"]) != len(chunk):
                bad = ([os.path.basename(t) for t in chunk if not viable(t)]
                       or [os.path.basename(t) for t in chunk])
                raise RuntimeError(
                    f"proxy val tile(s) {bad} were dropped at batch time (<64 "
                    f"points within chunk_xy={chunk_xy}m of the tile centre); "
                    f"they cover {[proxy_rep['covers'].get(n, []) for n in bad]}, "
                    f"whose score would silently vanish from the ranking metric. "
                    f"Delete {prep_dir} and relaunch to re-prep the val tiles.")
            yield batch, lab
    return _proxy_batches


def ptv3_make_run_eval(backbone, head, evaluate, forward_logits, proxy_batches,
                       proxy_tiles, proxy_rep, val_full, val_items, test_items,
                       val_csv, best, save_best, set_train_mode, num_classes,
                       names, run_dir):
    """run_eval(ep, write_json=False): proxy/full val + best-checkpoint
    update; the final write_json call also test-evals the best weights and
    writes test_metrics.json."""
    import torch

    def run_eval(ep, write_json=False):
        backbone.eval(); head.eval()
        if not write_json:
            m = (evaluate(val_items, f"val@ep{ep}") if val_full else
                 proxy_val(proxy_batches(), forward_logits, num_classes,
                           names, f"val@ep{ep}", len(proxy_tiles),
                           PROXY_PROTOCOL_TILES,
                           inventory=proxy_rep["inventory"]))
            # weights before the csv row: a kill between them must not seed a best final_model.pth can never match
            if best.update(m):
                save_best(ep)
            append_val_row(val_csv, ep, m, names)
            set_train_mode()
            return m
        m = evaluate(val_items, f"val@ep{ep}")
        # deliberately no best.update: the last epoch is never a crown candidate, matching the AdaBN trainers
        append_val_row(val_csv, ep, m, names)
        swapped = os.path.exists(best.final)
        if swapped:
            live_backbone = {k: v.clone() for k, v in backbone.state_dict().items()}
            live_head = {k: v.clone() for k, v in head.state_dict().items()}
            bckpt = torch.load(best.final, map_location="cuda", weights_only=True)
            backbone.load_state_dict(bckpt["backbone"]); head.load_state_dict(bckpt["head"])
            backbone.eval(); head.eval()
        mt = evaluate(test_items, f"test@ep{ep}")
        if swapped:
            backbone.load_state_dict(live_backbone); head.load_state_dict(live_head)
        with open(f"{run_dir}/test_metrics.json", "w", encoding="utf-8") as fj:
            json.dump({"val": m, "test": mt,
                       "val_scenes": [n for n, _, _ in val_items],
                       "test_scenes": [n for n, _, _ in test_items]}, fj, indent=2)
        set_train_mode()
        return m

    return run_eval


def ptv3_train_loop(backbone, head, optim, seg_loss, forward_logits,
                    make_train_batch, run_eval, best, save_best,
                    set_train_mode, run_dir, run_id, metrics_csv, start_epoch,
                    n_epochs, steps, batch_size, num_classes, base_lr,
                    warmup_pct, grad_clip, checkpoint_gap, val_every,
                    ckpt_extra=None):
    """The PTv3-family epoch loop: prefetched steps with an OOM guard, one
    metrics.csv row + periodic checkpoint per epoch, val/stop cadence, then
    the final combined eval, best crowning and DONE marker."""
    import glob
    import numpy as np
    import torch

    LOG_EVERY = 20
    AMP = os.environ.get("TT_AMP") == "1"
    prefetch = (make_prefetcher(make_train_batch,
                                depth=int(os.environ.get("TT_PREFETCH", "2")))
                if start_epoch < n_epochs else None)
    print(f"  starting at epoch {start_epoch}, up to {n_epochs}, "
          f"{steps} steps/epoch (batch {batch_size})"
          f"{' [bf16 autocast]' if AMP else ''}", flush=True)
    t_run = time.time()
    ep = n_epochs - 1
    for ep in range(start_epoch, n_epochs):
        cur_lr = ptv3_lr_at(ep, base_lr, warmup_pct, n_epochs)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        set_train_mode()
        ep_loss = 0.0
        ep_conf = torch.zeros(num_classes, num_classes, dtype=torch.long,
                              device="cuda")
        t_ep = time.time(); t_chunk = t_ep; n_steps = 0; last_log_step = 0
        n_oom = 0
        print(f"  ep {ep:3d} starting (lr={cur_lr:.2e})…", flush=True)
        for step in range(steps):
            try:
                batch, label = prefetch()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                    logits = forward_logits(batch)
                    loss = seg_loss(logits, label)
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()), grad_clip)
                optim.step()
                ep_loss += loss.item(); n_steps += 1
                if n_steps % LOG_EVERY == 0:
                    dt = time.time() - t_chunk
                    print(f"    ep {ep:3d} step {n_steps:4d}: "
                          f"loss={ep_loss/n_steps:.4f} "
                          f"{(n_steps-last_log_step)/max(dt,1e-6):.2f} it/s", flush=True)
                    t_chunk = time.time(); last_log_step = n_steps
                pred = logits.argmax(-1)
                m = label >= 0
                ep_conf += torch.bincount(
                    label[m] * num_classes + pred[m],
                    minlength=num_classes * num_classes,
                ).reshape(num_classes, num_classes)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    n_oom += 1
                    torch.cuda.empty_cache(); continue
                raise
        if n_steps == 0:
            raise RuntimeError(f"epoch {ep}: 0 optimizer steps ({n_oom} OOM steps); "
                               f"lower --batch or --chunk-xy.")
        if n_oom:
            print(f"  ep {ep:3d} note: {n_oom} OOM steps skipped", flush=True)
        sec_per_iter = (time.time() - t_ep) / max(n_steps, 1)
        sec_per_epoch = time.time() - t_ep
        conf = ep_conf.cpu().numpy()
        ep_inter = np.diag(conf)
        ep_union = conf.sum(0) + conf.sum(1) - ep_inter
        train_acc = int(np.trace(conf)) / max(int(conf.sum()), 1)
        with np.errstate(invalid="ignore"):
            train_iou = float(np.mean(ep_inter / np.maximum(ep_union, 1)))
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                ep, ep_loss / max(n_steps, 1), f"{train_acc:.4f}",
                f"{train_iou:.4f}", f"{cur_lr:.6e}", f"{sec_per_iter:.4f}",
                f"{sec_per_epoch:.2f}", f"{gpu_mem:.1f}",
            ])
        print(f"  ep {ep:3d}: loss={ep_loss/max(n_steps,1):.4f} "
              f"acc={train_acc:.4f} miou={train_iou:.4f} lr={cur_lr:.2e} "
              f"s/iter={sec_per_iter:.3f} s/ep={sec_per_epoch:.1f}", flush=True)
        if (ep + 1) % checkpoint_gap == 0 or ep == n_epochs - 1:
            atomic_torch_save({"backbone": backbone.state_dict(),
                               "head": head.state_dict(),
                               "optim": optim.state_dict(), "epoch": ep,
                               **(ckpt_extra or {})},
                              f"{run_dir}/checkpoints/ep{ep:03d}.pth")
            for old in sorted(glob.glob(f"{run_dir}/checkpoints/ep*.pth"))[:-2]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        stop = stop_requested(ep)
        if (ep + 1) % val_every == 0 and ep != n_epochs - 1 and not stop:
            run_eval(ep)
        if stop:
            break

    if prefetch:
        prefetch.shutdown()

    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(ep, write_json=True)
    best.finalize(lambda p: save_best(ep))
    print(f"  total wall-clock {(time.time() - t_run)/3600:.2f} h")

    open(f"{run_dir}/DONE", "w").close()
    print(f"  run complete -> {run_id}", flush=True)


FEAT_VOCAB = ("x", "y", "z", "intensity", "return_number", "rgb")


def parse_feat_spec(env_value, default_spec):
    """FEAT_CHANNELS csv -> ordered names; empty -> the trainer's default
    spec. Valid: FEAT_VOCAB or feat_<name>."""
    import re
    names = [s.strip() for s in (env_value or "").split(",") if s.strip()]
    if not names:
        return list(default_spec)
    # feat_intensity/feat_return_number ARE the canonical channels (old datasets
    # baked the raw column under a feat_ alias); collapse the distinction
    fix = {"feat_intensity": "intensity", "feat_return_number": "return_number",
           "feat_returnnumber": "return_number", "feat_ret_num": "return_number"}
    names = [fix.get(n.lower(), n) for n in names]
    bad = [n for n in names if n not in FEAT_VOCAB
           and not re.fullmatch(r"feat_[A-Za-z0-9_]+", n)]
    if bad:
        raise ValueError(
            f"unknown FEAT_CHANNELS name(s) {bad}: valid names are "
            f"{list(FEAT_VOCAB)} or feat_<name> dataset channels")
    return names


def feat_spec_tag(spec, default_spec):
    """PREP_DIR suffix for a non-default spec; "" when spec == the default so
    existing cache paths stay valid."""
    import hashlib
    if list(spec) == list(default_spec):
        return ""
    return "_f" + hashlib.sha1(",".join(spec).encode()).hexdigest()[:6]


def feat_extras(z, spec, where):
    """The feat_* arrays `spec` needs from an npz; missing key raises naming
    what IS available. TT_ZERO_CHANNELS entries are fed as zeros instead
    (user-disabled input), present or not."""
    import numpy as np
    zc = zero_channels()
    out = {}
    for n in spec:
        if not n.startswith("feat_"):
            continue
        if n in zc:
            out[n] = np.zeros(len(z["xyz"]), np.float32)
            continue
        if n not in z.files:
            avail = [k for k in z.files if k.startswith("feat_")]
            raise ValueError(
                f"{where} has no '{n}' channel (available feat_*: "
                f"{avail or 'none'}). Rebuild the dataset/prep cache with this "
                f"feature or drop it from FEAT_CHANNELS.")
        out[n] = z[n].astype(np.float32)
    return out

_ENV_KNOBS = {
    "DG_DENSITY_AUG":   ("DG_DENSITY_AUG",       "env_bool"),
    "DG_COARSEN_MAX":   ("DG_COARSEN_MAX",       "env_float"),
    "DG_P_NATIVE":      ("DG_P_NATIVE",          "env_float"),
    "DG_LOGDK_FEAT":    ("DG_LOGDK_FEAT",        "env_bool"),
    "DG_LOGDK_K":       ("DG_LOGDK_K",           "env_int"),
    "DG_INFER_ADABN":   ("DG_INFER_ADABN",       "env_bool"),
    "DG_INFER_APCOTTA": ("DG_INFER_APCOTTA",     "env_bool"),
    "DG_INFER_TTA":     ("DG_INFER_TTA",         "env_int"),
    "EVAL_VOTES":       ("EVAL_VOTES",           "env_int"),
    "VAL_EVERY":        ("VAL_EVERY",            "env_int"),
    "USE_FOCAL":        ("LOSS_FOCAL",           "env_bool"),
    "FOCAL_GAMMA":      ("LOSS_FOCAL_GAMMA",     "env_float"),
    "CLASS_WEIGHTING":  ("LOSS_CLASS_WEIGHTING", "env_bool"),
    "WEIGHT_BETA":      ("LOSS_WEIGHT_BETA",     "env_float"),
    "RARE_OVERSAMPLE":  ("RARE_OVERSAMPLE",      "env_bool"),
    "RARE_CENTER_PROB": ("RARE_CENTER_PROB",     "env_float"),
    "PROXY_SAMPLING":   ("PROXY_SAMPLING",       "env_str"),
    "KP_AGGREGATION":   ("KP_AGGREGATION",       "env_str"),
    "KP_NORM":          ("KP_NORM",              "env_str"),
    "FEAT_CHANNELS":    ("FEAT_CHANNELS",        "env_str"),
}


def env_overrides(g, names):
    """Values for `names` in order, each env-overridable (the GUI exports
    DG_*/LOSS_*/RARE_*/EVAL_*/KP_*), defaulting to g[name]."""
    import density as dg
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
        raise FileNotFoundError(f"{meta_path} not found. Build the dataset "
                                f"with the trainer_gui app first.")
    with open(meta_path) as f:
        ds_meta = json.load(f)
    return ds_meta, int(ds_meta["num_classes"]), list(ds_meta["class_names"])


def load_ckpt_safe(path, map_location="cpu"):
    """torch.load with weights_only=True + the shared re-export hint."""
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load weights '{path}': {e}\n"
            f"  (loaded safely with weights_only=True; a full-model pickle or a "
            f"checkpoint from another script is rejected; re-export as a state_dict.)"
        ) from e


def modal_retry_marker(fcid, marker_root, commit):
    """First attempt drops a marker file (markers accumulate on the outputs
    volume); a marker already present means a Modal retry, so auto-resume.
    `commit` is the volume's commit callable - this module must not import modal."""
    marker = f"{marker_root}/{fcid}" if fcid else ""
    if not fcid or os.path.exists(marker):
        os.environ["TT_MODAL_RETRY"] = os.environ["AUTO_RESUME"] = "1"
    else:
        os.makedirs(marker_root, exist_ok=True)
        open(marker, "w").close()
        commit()


def modal_shell_run(script, flag_vals, env_json, volumes, reload=None):
    """Body of every modal_train_* shell: build the trainer command (None
    flags skipped), merge --env-json into the env, commit volumes every 120s +
    on exit. `reload` (the outputs volume's .reload) is called after each commit
    so an externally-uploaded /outputs/STOP sentinel becomes visible in the
    container. Volumes are duck-typed - this module must not import modal."""
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

    stop = threading.Event()

    def _commit_loop():
        warned = False
        while not stop.wait(120):
            for v in volumes:
                v.commit()
            if reload is not None:
                try:
                    reload()
                except Exception as e:  # open handles can block reload; STOP just waits a cycle
                    if not warned:
                        print(f"[modal-shell] volume reload failed ({e}); "
                              f"STOP delivery delayed to the next pass", flush=True)
                        warned = True

    threading.Thread(target=_commit_loop, daemon=True).start()
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        stop.set()
        for v in volumes:
            v.commit()
