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
    final_model.pth forever; pre-protocol-column rows never seed."""
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
    """Delete a stale STOP sentinel at startup. ponytail: concurrent runs
    sharing one /outputs share the sentinel - press stop once per run."""
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
    (legacy run_config.json accepted as raw source). `backbone` = key."""
    rc = {}
    for fn in ("run.json", "run_config.json"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    rc = json.load(f)
            except (OSError, ValueError):
                rc = {}
            break
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
    """Normalized inference metadata from run.json (legacy run_config.json
    fallback) beside the weights. None for a bare .pth; missing fields None."""
    d = os.path.dirname(weights_path)
    if os.path.basename(d) == "checkpoints":
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
    Returns (name, pc_path, None) lists - third slot is a legacy cls_path."""
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
    """Refuse a prep cache built with different settings; stamps .done markers
    to migrate pre-validation caches. True if the signature was newly written."""
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
    """Write the val_metrics.csv header if absent; upgrade a pre-protocol
    header in place (rewrite: the trailing column must exist or appended rows
    misalign). Existing rows stay short a field -> they read as non-proxy."""
    cols = (["epoch", "val_acc", "val_miou"]
            + [f"iou_{n}" for n in class_names] + ["protocol"])
    rows = []
    if os.path.exists(val_csv):
        with open(val_csv, newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.reader(f))
        if rows and rows[0][-1:] == ["protocol"]:
            return
    rows[:1] = [cols]
    tmp = val_csv + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    _replace_retry(tmp, val_csv)


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
    would falsely wedge a cross-backend resume. Returns False when run_dir must
    be abandoned for a fresh one (pre-upgrade run: v1 rows, or ranking rows with
    no signature); raises RuntimeError when its rows were ranked under another
    protocol or the checkpoint they crowned is gone.
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
    rows = pre = other = 0
    val_csv = f"{run_dir}/val_metrics.csv"
    if os.path.exists(val_csv):
        with open(val_csv, newline="", encoding="utf-8", errors="replace") as f:
            raw = list(csv.reader(f))
        hdr = raw[0] if raw else []
        v1 = hdr[-1:] != ["protocol"]
        for r in raw[1:]:
            if not r:
                continue
            if v1 or len(r) < len(hdr):
                pre += 1
            elif r[-1] == ranking:
                rows += 1
            else:
                other += 1
    if pre:
        print(f"  resume: skipping {os.path.basename(run_dir)} "
              f"({pre} val row(s) from a pre-protocol-column run)", flush=True)
        return False
    old = None
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = None
    # rows on the other scale under a signature that NAMES that scale = a mid-run
    # switch; an unreadable signature stays lenient, as it was before 'full' existed
    switched = (bool(other) and isinstance(old, dict)
                and old.get("ranking", "proxy") != ranking)
    if not rows and not switched:
        atomic_json_save(sig, path)
        return True
    if not os.path.exists(path):
        print(f"  resume: skipping {os.path.basename(run_dir)} "
              f"({ranking}-protocol rows from a pre-signature run)", flush=True)
        return False
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


def scene_arrays(z, n):
    """(intensity, ret_num) from a scene npz - the ONE place missing-channel
    fallbacks are decided (intensity -> rgb gray -> zeros; ret_num -> zeros)."""
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
    """--mode infer loop: predict(pc_path) -> (xyz, pred, intensity, conf,
    probs), written as <name>_pred.npz (+ _pred_CLS.txt) with the crash-safe
    per-scene infer_run.json rewrite."""
    import numpy as np
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
    """Every feat_* channel a scene npz carries; legacy bare 'hag' surfaces
    as feat_hag."""
    import numpy as np
    out = {k: z[k].astype(np.float32) for k in z.files if k.startswith("feat_")}
    if "feat_hag" not in out and "hag" in z.files:
        out["feat_hag"] = z["hag"].astype(np.float32)
    return out


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
    """Prep-dir suffix for the stride factor; "" for the legacy 0.5 so
    existing caches stay valid untagged."""
    f = float(os.environ.get("TT_TRAIN_STRIDE", "0.75"))
    return "" if f == 0.5 else f"_ts{f:g}"


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
    any_new = [validate_cache(
        prep_dir, sig,
        [("train", train_list), ("val", val_list), ("test", test_list)],
        lambda d, name: (f"{d}/{name}_x*.npz", f"{d}/{name}.done"))]

    def tile_remaining(items, out_dir, split):
        for name, pc_path, _cls in items:
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
    droppable. "height" is legacy-only (old run.json specs); real HAG = feat_hag."""
    import numpy as np
    import density as dg
    spec = list(spec)

    def build_feat(xyz, intensity, ret_num, drop=(), extras=None):
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
                              features=None, legacy_features=None,
                              skip_done=False):
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
        for cfgp in (f"{rd}/run.json", f"{rd}/run_config.json"):
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
        latest = validated_latest_ckpt(ckpts, _ep)
        if latest is None:
            continue
        return rd, latest, _ep(latest)
    return None


def kp_make_evaluate(forward, build_feat, grid, chunk_xy, num_classes,
                     class_names):
    """KP voted eval scored on the ORIGINAL raw points: center-weighted
    softmax votes per voxel, argmax, NN-propagate to raw, score vs raw GT.
    forward([(cxyz, feat)]) -> per-tile (N, C) logits list."""
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
            for name, pc_path, _cls, split_dir in scene_items:
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


FEAT_VOCAB = ("x", "y", "z", "height", "intensity", "return_number", "rgb")


def parse_feat_spec(env_value, legacy_default):
    """FEAT_CHANNELS csv -> ordered names; empty -> the trainer's legacy
    default. Valid: FEAT_VOCAB or feat_<name>."""
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
    """PREP_DIR suffix for a non-default spec; "" when spec == legacy so
    existing cache paths stay valid."""
    import hashlib
    if list(spec) == list(legacy):
        return ""
    return "_f" + hashlib.sha1(",".join(spec).encode()).hexdigest()[:6]


def feat_extras(z, spec, where):
    """The feat_* arrays `spec` needs from an npz; missing key raises naming
    what IS available."""
    import numpy as np
    out = {}
    for n in spec:
        if not n.startswith("feat_"):
            continue
        if n not in z.files:
            if n == "feat_hag" and "hag" in z.files:
                out[n] = z["hag"].astype(np.float32)
                continue
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


def modal_shell_run(script, flag_vals, env_json, volumes):
    """Body of every modal_train_* shell: build the trainer command (None
    flags skipped), merge --env-json into the env, commit volumes every 120s +
    on exit. Volumes are duck-typed (.commit()) - this module must not import modal."""
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


def _demo():
    import tempfile
    d = tempfile.mkdtemp()
    px = lambda v: {"present_classes_mIoU": v, "protocol": "proxy_tiles_v2"}
    fu = lambda v: {"present_classes_mIoU": v}
    assert row_protocol(px(0.5)) == "proxy" and row_protocol(fu(0.5)) == "full"
    assert ranking_protocol("full") == "full"
    assert ranking_protocol("coverage") == "proxy"
    b = BestCheckpoint(d)
    assert b.update(px(0.5)) and not b.update(px(0.4)) and b.update(px(0.6))
    # a full-protocol metric never crowns a proxy-ranked run, however high
    assert not b.update(fu(0.99)) and abs(b.best - 0.6) < 1e-9
    csv_path = os.path.join(d, "val_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_acc", "val_miou", "protocol"])
        w.writerow([0, 0.9, 0.7, "proxy"])
        w.writerow([1, 0.9, 0.65, "proxy"])
        w.writerow([1, 0.9, 0.95, "full"])
    assert abs(best_val_miou(csv_path) - 0.7) < 1e-9
    assert abs(best_val_miou(csv_path, "full") - 0.95) < 1e-9
    assert not BestCheckpoint(d).update(px(0.69))
    # a full-ranked run seeds off the full rows and ignores the proxy ones
    bf = BestCheckpoint(d, "full")
    assert abs(bf.best - 0.95) < 1e-9 and not bf.update(px(0.99))
    assert bf.update(fu(0.96)) and not bf.update(fu(0.94))
    open(b.final, "w").close()
    b.finalize(lambda p: (_ for _ in ()).throw(AssertionError("should not save_last")))

    with open(os.path.join(d, "run_config.json"), "w") as f:
        json.dump({"num_classes": 7, "class_names": list("abcdefg"),
                   "grid_m": 2.0, "chunk_xy_m": 100.0,
                   "features": ["intensity", "return_number", "height", "feat_hag"]}, f)
    m = write_run_manifest(d, "kpconvx_cold")
    assert m["backbone"] == "kpconvx_cold" and m["weights"] == "final_model.pth"
    assert m["grid"] == 2.0 and m["chunk_xy"] == 100.0 and m["num_classes"] == 7
    assert m["intensity_norm"] == "p95"
    assert "hag_source" not in m and m.get("feature_mode") != "hag"
    assert m["grid_m"] == 2.0
    assert os.path.exists(os.path.join(d, "run.json"))
    os.remove(os.path.join(d, "run_config.json"))
    m_again = write_run_manifest(d, "kpconvx_cold")
    assert m_again["grid"] == 2.0 and m_again["grid_m"] == 2.0
    assert m["dg"] is not None and m["dg"]["logdk"] is False
    os.environ["DG_LOGDK_FEAT"] = "1"; os.environ["DG_LOGDK_K"] = "12"
    m2 = write_run_manifest(d, "kpconvx_cold")
    assert m2["dg"]["logdk"] is True and m2["dg"]["logdk_k"] == 12
    assert infer_meta(os.path.join(d, "final_model.pth"))["dg"]["logdk"] is True
    os.environ.pop("DG_LOGDK_FEAT"); os.environ.pop("DG_LOGDK_K")

    assert _intensity_norm_from_meta({"source": {"intensity_norm": "p95"}}) == "p95"
    assert _intensity_norm_from_meta({"intensity_norm": "p95"}) == "p95"
    assert _intensity_norm_from_meta({"source": {}}) == "max"

    im = infer_meta(os.path.join(d, "final_model.pth"))
    assert im and im["num_classes"] == 7 and im["grid"] == 2.0
    assert im["features"] == ["intensity", "return_number", "height", "feat_hag"]
    assert im["hag_source"] is None and im["class_names"] == list("abcdefg")
    assert infer_meta(os.path.join(tempfile.mkdtemp(), "bare.pth")) is None

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

    cd = tempfile.mkdtemp()
    np.savez(f"{cd}/good_x0_y0.npz", lab=np.array([0, 1], np.int32))
    open(f"{cd}/bad_x0_y50.npz", "w").write("not a zip")
    open(f"{cd}/bad.done", "w").close()
    try:
        scan_class_balance([f"{cd}/good_x0_y0.npz", f"{cd}/bad_x0_y50.npz"], 2)
        raise AssertionError("corrupt tile must raise after healing")
    except RuntimeError as e:
        assert "corrupt" in str(e)
    assert not os.path.exists(f"{cd}/bad_x0_y50.npz")
    assert not os.path.exists(f"{cd}/bad.done")
    assert os.path.exists(f"{cd}/good_x0_y0.npz")

    utm = np.array([620900.0, 4849000.0, 170.0]) + \
        np.random.RandomState(2).rand(20_000, 3) * [30, 30, 5]
    np.savez(f"{cd}/utm.npz", xyz=utm, label=np.zeros(len(utm), np.int32))
    lx, _, _ = ptv3_load_canonical(f"{cd}/utm.npz", "intensity")
    assert lx.dtype == np.float32 and 0 <= lx.min() and lx.max() < 40.0
    c = (lx - lx.mean(0, keepdims=True, dtype=np.float64)).astype(np.float32)
    assert (np.abs(c[:, :2]).max(1) <= 40.0).all()
    assert np.allclose(lx + np.floor(utm.min(0)), utm, atol=1e-2)

    _pn = iter(range(100))
    pf = make_prefetcher(lambda: next(_pn), depth=2)
    assert pf() == 0 and pf() == 1 and pf() == 2
    pf.shutdown()

    pts2 = rng.rand(2_000, 3).astype(np.float32) * 10 - 5
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

    rng = np.random.RandomState(0)
    for dims in (2, 3):
        keys = rng.randint(-50, 50, size=(20_000, dims))
        ref_u, ref_first, ref_inv = np.unique(keys, axis=0, return_index=True,
                                              return_inverse=True)
        first, inv = voxel_unique(keys, return_inverse=True)
        assert np.array_equal(first, ref_first) and np.array_equal(inv, ref_inv.reshape(-1))
        assert np.array_equal(keys[first], ref_u)
    if _torch.cuda.is_available():
        big = rng.randint(-40, 40, size=(50_000, 3))
        ref_f, ref_i = voxel_unique(big, return_inverse=True)
        keep, globals()["VOXEL_GPU_MIN"] = VOXEL_GPU_MIN, 1
        try:
            gf, gi = voxel_unique(big, return_inverse=True)
        finally:
            globals()["VOXEL_GPU_MIN"] = keep
        assert np.array_equal(gf, ref_f) and np.array_equal(gi, ref_i)
    assert list(prefetch_map(lambda x: x * x, range(20), depth=3)) \
        == [x * x for x in range(20)]
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
    doc = write_infer_run(d, {"backbone": "x"},
                          [{"scene": "a.npz", "points": 10, "seconds": 1.5},
                           {"scene": "b.npz", "points": 5, "seconds": 0.5}])
    assert doc["total_points"] == 15 and doc["total_seconds"] == 2.0
    assert json.load(open(os.path.join(d, "infer_run.json")))["backbone"] == "x"

    import torch
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(50, 4, generator=g)
    labels = torch.randint(0, 4, (50,), generator=g)
    seg = make_seg_loss(None, 0.0, False, 2.0, 1.0)
    ref = (torch.nn.functional.cross_entropy(logits, labels)
           + lovasz_softmax_flat(torch.softmax(logits, 1), labels))
    assert torch.allclose(seg(logits, labels), ref)
    assert float(seg(logits, torch.full((50,), -1))) == 0.0
    assert torch.allclose(focal_loss(logits, labels, 0.0),
                          torch.nn.functional.cross_entropy(logits, labels))
    hot = torch.nn.functional.one_hot(labels, 4).float() * 1e6
    assert float(lovasz_softmax_flat(torch.softmax(hot, 1), labels)) < 1e-3

    counts = np.array([1000, 1000, 10, 0], np.int64)
    assert auto_rare_classes(counts, 0.5) == [2]
    w = class_weights_np(counts[:3], 0.5, 5.0)
    assert w[2] > w[0] and w.max() <= 5.0 and w.min() >= 0.2
    wa = class_weights_np(counts, 0.5, 5.0, absent_to_one=True)
    assert wa[3] <= wa[2] and wa[2] > wa[0]
    pick = make_tile_picker(["a", "b"], ["r"], 1.0)
    assert pick() == "r"
    assert make_tile_picker(["a"], [], 1.0)() == "a"

    i_, u_, g_ = score_ious(np.array([0, 0, 1]), np.array([0, 1, 1]), 2)
    assert list(i_) == [1, 1] and list(u_) == [2, 2] and list(g_) == [1, 2]
    rs = np.random.RandomState(7)
    pr5, la5 = rs.randint(0, 5, 3000), rs.randint(0, 6, 3000)
    i5, u5, g5 = score_ious(pr5, la5, 5)
    for c in range(5):
        assert (i5[c] == ((pr5 == c) & (la5 == c)).sum()
                and u5[c] == ((pr5 == c) | (la5 == c)).sum()
                and g5[c] == (la5 == c).sum())
    m_ev = eval_metrics(i_, u_, g_, 2, 3, ["a", "b"], time.time(), 1, "demo",
                        extra={"skipped_scenes": 0, "scored_on": "raw_points"})
    assert (abs(m_ev["overall_mIoU"] - 0.5) < 1e-9 and m_ev["scored_on"] == "raw_points"
            and m_ev["present_classes"] == ["a", "b"])
    vd = tempfile.mkdtemp()
    append_val_row(f"{vd}/v.csv", 3, m_ev, ["a", "b"])
    assert "3,0.6667,0.5000,0.5000,0.5000,full" in open(f"{vd}/v.csv").read()
    v1 = f"{vd}/v1.csv"
    with open(v1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_acc", "val_miou", "iou_a", "iou_b"])
        w.writerow([0, 0.9, 0.88, 0.9, 0.86])
    assert best_val_miou(v1) == -1.0
    init_val_csv(v1, ["a", "b"])
    append_val_row(v1, 4, dict(m_ev, protocol=PROXY_PROTOCOL_TILES), ["a", "b"])
    init_val_csv(v1, ["a", "b"])
    assert best_val_miou(v1) == 0.5 and len(open(v1).read().split()) == 3

    i3, u3, g3 = score_ious(np.array([0, 0, 1]), np.array([0, 1, 1]), 3)
    m_inv = eval_metrics(i3, u3, g3, 2, 3, ["a", "b", "c"], time.time(), 1,
                         "demo", force_present=[0, 1, 2])
    assert (abs(m_inv["present_classes_mIoU"] - 1 / 3) < 1e-9
            and m_inv["forced_zero_classes"] == ["c"]
            and m_inv["scored_classes"] == ["a", "b", "c"]
            and m_inv["absent_classes"] == [])
    m_nof = eval_metrics(i3, u3, g3, 2, 3, ["a", "b", "c"], time.time(), 1, "demo")
    assert m_nof["present_classes_mIoU"] == 0.5
    assert m_nof["forced_zero_classes"] == [] and m_nof["absent_classes"] == ["c"]

    gd = tempfile.mkdtemp()
    rep = {"mode": "coverage", "floor_points": 4096, "inventory": [0, 1],
           "tiles": ["t1.npz", "t0.npz"]}
    gcsv = f"{gd}/val_metrics.csv"
    init_val_csv(gcsv, ["a", "b"])
    assert proxy_guard(gd, rep, PROXY_PROTOCOL_TILES, ["a", "b"])
    append_val_row(gcsv, 0, dict(m_ev, protocol=PROXY_PROTOCOL_TILES), ["a", "b"])
    try:
        proxy_guard(gd, rep, PROXY_PROTOCOL_TILES, ["a", "b"])
        raise AssertionError("proxy rows with no final_model.pth must block")
    except RuntimeError as e:
        assert ("final_model.pth" in str(e) and "plot history" in str(e)
                and "delete BOTH" not in str(e))
    open(f"{gd}/final_model.pth", "w").close()
    assert proxy_guard(gd, rep, PROXY_PROTOCOL_TILES, ["a", "b"])
    assert proxy_guard(gd, {**rep, "tiles": ["t1.npz", "t0.npz"]},
                       PROXY_PROTOCOL_TILES, ["a", "b"])
    try:
        proxy_guard(gd, {**rep, "mode": "density"}, PROXY_PROTOCOL_TILES, ["a", "b"])
        raise AssertionError("a mode switch must block")
    except RuntimeError as e:
        assert "EVAL_ONLY" in str(e) and "val_metrics.csv" in str(e)
    os.remove(f"{gd}/proxy_val.json")
    assert not proxy_guard(gd, rep, PROXY_PROTOCOL_TILES, ["a", "b"])
    open(f"{gd}/proxy_val.json", "w").write("{not json")
    try:
        proxy_guard(gd, rep, PROXY_PROTOCOL_TILES, ["a", "b"])
        raise AssertionError("an unparseable signature with proxy rows must block")
    except RuntimeError as e:
        assert "unreadable proxy_val.json" in str(e)
    nr = tempfile.mkdtemp()
    init_val_csv(f"{nr}/val_metrics.csv", ["a", "b"])
    append_val_row(f"{nr}/val_metrics.csv", 0, m_ev, ["a", "b"])
    open(f"{nr}/proxy_val.json", "w").write("{not json")
    assert proxy_guard(nr, {**rep, "mode": "density"}, PROXY_PROTOCOL_TILES, ["a", "b"])
    assert json.load(open(f"{nr}/proxy_val.json"))["mode"] == "density"
    ur = tempfile.mkdtemp()
    with open(f"{ur}/val_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_acc", "val_miou", "iou_a", "iou_b"])
        w.writerow([40, 0.9, 0.81, 0.9, 0.72])
    open(f"{ur}/final_model.pth", "w").close()
    assert not proxy_guard(ur, rep, PROXY_PROTOCOL_TILES, ["a", "b"])
    init_val_csv(f"{ur}/val_metrics.csv", ["a", "b"])
    assert not proxy_guard(ur, rep, PROXY_PROTOCOL_TILES, ["a", "b"])
    assert not os.path.exists(f"{ur}/proxy_val.json")

    # ranking='full': no tile subset to pin, and the two scales must not mix
    fd = tempfile.mkdtemp()
    fcsv = f"{fd}/val_metrics.csv"
    init_val_csv(fcsv, ["a", "b"])
    assert proxy_guard(fd, rep, PROXY_PROTOCOL_TILES, ["a", "b"], "full")
    assert json.load(open(f"{fd}/proxy_val.json")) == {"ranking": "full"}
    append_val_row(fcsv, 0, m_ev, ["a", "b"])            # m_ev has no protocol -> a 'full' row
    open(f"{fd}/final_model.pth", "w").close()
    assert proxy_guard(fd, rep, PROXY_PROTOCOL_TILES, ["a", "b"], "full")
    for mode in ("coverage", "density"):
        try:
            proxy_guard(fd, {**rep, "mode": mode}, PROXY_PROTOCOL_TILES, ["a", "b"])
            raise AssertionError(f"full -> {mode} must block")
        except RuntimeError as e:
            # the remedy names the setting that CONTINUES this run, not the new one
            assert "PROXY_SAMPLING='full'" in str(e) and "different scales" in str(e)
    # and the reverse: a proxy-ranked run (readable signature) may not go full
    pd_ = tempfile.mkdtemp()
    init_val_csv(f"{pd_}/val_metrics.csv", ["a", "b"])
    proxy_guard(pd_, rep, PROXY_PROTOCOL_TILES, ["a", "b"])
    append_val_row(f"{pd_}/val_metrics.csv", 0,
                   dict(m_ev, protocol=PROXY_PROTOCOL_TILES), ["a", "b"])
    open(f"{pd_}/final_model.pth", "w").close()
    try:
        proxy_guard(pd_, rep, PROXY_PROTOCOL_TILES, ["a", "b"], "full")
        raise AssertionError("proxy -> full must block")
    except RuntimeError as e:
        assert "PROXY_SAMPLING='coverage'" in str(e) and "different scales" in str(e)
    # a full-ranked run with rows but no crowned checkpoint blocks like a proxy one
    fd2 = tempfile.mkdtemp()
    init_val_csv(f"{fd2}/val_metrics.csv", ["a", "b"])
    proxy_guard(fd2, rep, PROXY_PROTOCOL_TILES, ["a", "b"], "full")
    append_val_row(f"{fd2}/val_metrics.csv", 0, m_ev, ["a", "b"])
    try:
        proxy_guard(fd2, rep, PROXY_PROTOCOL_TILES, ["a", "b"], "full")
        raise AssertionError("full rows with no final_model.pth must block")
    except RuntimeError as e:
        assert "final_model.pth" in str(e) and "full val row" in str(e)

    import ntpath
    import posixpath
    wd, pxd = tempfile.mkdtemp(), tempfile.mkdtemp()
    proxy_guard(wd, {**rep, "tiles": [ntpath.basename(p) for p
                                      in (r"D:\prep\t1.npz", r"D:\prep\t0.npz")]},
                PROXY_PROTOCOL_TILES, ["a", "b"])
    proxy_guard(pxd, {**rep, "tiles": [posixpath.basename(p) for p
                                       in ("/vol/prep/t0.npz", "/vol/prep/t1.npz")]},
                PROXY_PROTOCOL_TILES, ["a", "b"])
    wsig = json.load(open(f"{wd}/proxy_val.json", encoding="utf-8"))
    assert wsig == json.load(open(f"{pxd}/proxy_val.json", encoding="utf-8"))
    assert "/" not in json.dumps(wsig) and "\\" not in json.dumps(wsig)

    ptd = tempfile.mkdtemp()
    pnm = ["ground", "veg", "pole", "wire"]
    for i in range(12):
        lab = [np.zeros(6000, np.int32)]
        if i >= 9:
            lab.append(np.full(3000, 1, np.int32))
        if i == 8:
            lab.append(np.full(5000, 2, np.int32))
        if i == 7:
            lab.append(np.full(5000, 3, np.int32))
        np.savez(f"{ptd}/t{i:02d}.npz", lab=np.concatenate(lab))
    pvt = sorted(f"{ptd}/t{i:02d}.npz" for i in range(12))
    pcb = f"{ptd}/cb.npz"
    pcov, cr = pick_proxy_tiles(pvt, 4, 8, "coverage", pnm, cache_path=pcb)
    assert cr["inventory"] == [0, 1, 2, 3] and cr["shortfall"] == {}
    assert all(v >= cr["floor_points"] for v in cr["per_class_picked"].values())
    assert cr["covers"] == {"t08.npz": ["pole"], "t09.npz": ["veg"], "t10.npz": ["veg"]}
    assert 8 < cr["n_tiles"] <= 8 + _PROXY_FLOOR_TILES * len(cr["inventory"])
    assert [os.path.basename(p) for p in pcov] == cr["tiles"]
    psh = [pvt[i] for i in np.random.RandomState(5).permutation(12)]
    assert pick_proxy_tiles(sorted(psh), 4, 8, "coverage", pnm)[1] == cr
    assert pick_proxy_tiles(pvt, 4, 8, "coverage", pnm, cache_path=pcb)[1] == cr
    _, sr = pick_proxy_tiles(pvt, 4, 8, "coverage", pnm, cache_path=pcb,
                             viable=lambda p: not p.endswith("t08.npz"))
    assert sr["shortfall"] == {"pole": [0, sr["floor_points"]]}
    assert "t08.npz" not in sr["tiles"]
    assert pick_proxy_tiles(pvt, 4, 50, "coverage", pnm,
                            cache_path=pcb)[1]["n_tiles"] == 12
    try:
        pick_proxy_tiles([], 4, 8)
        raise AssertionError("an empty val split must raise")
    except RuntimeError as e:
        assert "non-empty val split" in str(e)
    assert pick_proxy_tiles(pvt, 4, 2, "coverage", pnm,
                            cache_path=pcb)[1]["shortfall"] == {}
    try:
        pick_proxy_tiles(pvt, 4, 8, "coverage", pnm, viable=lambda p: False)
        raise AssertionError("an all-non-viable val split must raise")
    except RuntimeError as e:
        assert "12/12 rejected" in str(e) and "final_model.pth" in str(e)
    try:
        pick_proxy_tiles(pvt, 4, 8, "rare", pnm)
        raise AssertionError("an unknown sampling mode must raise")
    except ValueError as e:
        assert "PROXY_SAMPLING" in str(e) and "coverage" in str(e)

    sx, sa, sl = kp_grid_subsample(
        np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [5.0, 5.0, 5.0]], np.float32),
        np.array([[0.0], [1.0], [2.0]], np.float32),
        np.array([1, 1, 0], np.int64), 1.0, 3)
    assert len(sx) == 2 and abs(sa[0, 0] - 0.5) < 1e-6 and list(sl) == [1, 0]
    assert kp_augment(sx).shape == sx.shape

    assert parse_feat_spec("", ["intensity", "return_number", "height"]) \
        == ["intensity", "return_number", "height"]
    assert parse_feat_spec(" intensity , feat_ndvi ", []) == ["intensity", "feat_ndvi"]
    try:
        parse_feat_spec("bogus", [])
        raise AssertionError("unknown spec name must raise")
    except ValueError as e:
        assert "bogus" in str(e) and "intensity" in str(e)

    bf = kp_make_build_feat(False, 8)
    xyz10 = np.random.RandomState(0).rand(10, 3).astype(np.float32)
    f = bf(xyz10, np.ones(10, np.float32), np.zeros(10, np.float32))
    assert f.shape == (10, 3) and np.all(f[:, 0] == 1.0)
    fd = bf(xyz10, np.ones(10, np.float32), np.ones(10, np.float32), drop=[0])
    assert np.all(fd[:, 1] == 0.0) and np.all(fd[:, 2] == 1.0)
    assert np.all(fd[:, 0] == 1.0)
    bfl = kp_make_build_feat(False, 8, spec=["intensity", "return_number", "height"])
    fl = bfl(xyz10, np.ones(10, np.float32), np.ones(10, np.float32))
    assert np.allclose(fl[:, 3], xyz10[:, 2] - xyz10[:, 2].min())
    fh = bfl(xyz10, np.ones(10, np.float32), np.ones(10, np.float32), drop=[2])
    assert np.all(fh[:, 3] == 0.0) and np.all(fh[:, 1] == 1.0)
    bfs = kp_make_build_feat(False, 8, spec=["height", "intensity", "feat_q"])
    fq = np.arange(10, dtype=np.float32)
    fs = bfs(xyz10, np.full(10, 0.5, np.float32), np.zeros(10, np.float32),
             extras={"feat_q": fq})
    assert fs.shape == (10, 4) and np.all(fs[:, 0] == 1.0)
    assert np.allclose(fs[:, 1], xyz10[:, 2] - xyz10[:, 2].min())
    assert np.all(fs[:, 2] == 0.5) and np.array_equal(fs[:, 3], fq)
    try:
        bfs(xyz10, fq, fq)
        raise AssertionError("missing extras must raise")
    except ValueError as e:
        assert "feat_q" in str(e)

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
    kp_ensure_prep(prep, ds, sig, tile_fn)
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
    assert s and s[0].shape == (len(s[2]), 3) and s[1].shape[1] == 3
    assert abs(s[0].mean()) < 1e-3
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
               / sum(m_kp["per_class_gt_count"].values())) < 0.02
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

    pd_ = tempfile.mkdtemp()
    rp = np.random.RandomState(4)
    pxyz = (rp.uniform(0, 60, (20_000, 3)) * [1, 1, 0.1]).astype(np.float32)
    prgb = rp.randint(0, 256, (20_000, 3)).astype(np.uint8)
    plab = rp.randint(0, 3, 20_000).astype(np.int32)
    np.savez(f"{pd_}/pv.npz", xyz=pxyz, rgb=prgb, label=plab)
    load_c = lambda p: ((lambda z: (z["xyz"].astype(np.float32), z["rgb"],
                                    z["label"].astype(np.int32)))(np.load(p)))
    ptv3_tile_and_save([f"{pd_}/pv.npz"], f"{pd_}/tiles", 30.0, 15.0, load_c)
    tls = sorted(_glob.glob(f"{pd_}/tiles/pv_x*.npz"))
    assert tls
    lut = {p.tobytes(): i for i, p in enumerate(pxyz)}
    for tp in tls:
        zt = np.load(tp)
        ii = np.array([lut[p.tobytes()] for p in zt["xyz"]])
        assert (np.array_equal(zt["rgb"], prgb[ii])
                and np.array_equal(zt["lab"], plab[ii]))
        assert (zt["xyz"][:, :2].max(0) - zt["xyz"][:, :2].min(0) <= 30.0).all()
    bfp = lambda vx, rgbn, extras: np.concatenate([vx, rgbn], 1).astype(np.float32)
    fwdp = lambda b: _torch.nn.functional.pad(
        _torch.full((len(b["feat"]), 1), 5.0), (0, 2))
    evp = ptv3_make_evaluate(fwdp, bfp, [], 2.0, 30.0, 3, ["a", "b", "c"])
    load_raw = lambda: (pxyz, plab)
    os.environ["EVAL_BATCH"] = "1"
    mp1 = evp([("pv", load_raw, f"{pd_}/tiles")], "demo")
    os.environ["EVAL_BATCH"] = "3"
    mp3 = evp([("pv", load_raw, f"{pd_}/tiles")], "demo")
    del os.environ["EVAL_BATCH"]
    assert mp1["num_scenes"] == 1 and mp1["per_class_gt_count"]["a"] > 0
    assert abs(mp1["overall_acc"] - mp1["per_class_gt_count"]["a"]
               / sum(mp1["per_class_gt_count"].values())) < 1e-9
    assert (mp3["per_class_iou"] == mp1["per_class_iou"]
            and mp3["overall_acc"] == mp1["overall_acc"])

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
    assert "crs_wkt" not in zp.files and "source_crs_wkt" not in zp.files

    np.savez(f"{ij}/c0.npz", xyz=z0["xyz"], intensity=z0["intensity"],
             crs_wkt=np.asarray('PROJCS["demo"]'))
    run_infer_scenes([f"{ij}/c0.npz"],
                     lambda p: (z0["xyz"], pr, z0["intensity"], cf, None),
                     f"{ij}/predictions", ij, {"backbone": "demo"})
    with np.load(f"{ij}/predictions/c0_pred.npz") as zc:
        assert str(zc["crs_wkt"]) == 'PROJCS["demo"]'
        assert "source_crs_wkt" not in zc.files

    np.savez(f"{ij}/c1.npz", xyz=z0["xyz"], intensity=z0["intensity"],
             crs_wkt=np.asarray('PROJCS["utm"]'),
             source_crs_wkt=np.asarray('GEOGCS["src"]'))
    run_infer_scenes([f"{ij}/c1.npz"],
                     lambda p: (z0["xyz"], pr, z0["intensity"], cf, None),
                     f"{ij}/predictions", ij, {"backbone": "demo"})
    with np.load(f"{ij}/predictions/c1_pred.npz") as zc:
        assert str(zc["crs_wkt"]) == 'PROJCS["utm"]'
        assert str(zc["source_crs_wkt"]) == 'GEOGCS["src"]'

    os.environ["LOSS_FOCAL_GAMMA"] = "3.5"
    uf, fg = env_overrides({"USE_FOCAL": False, "FOCAL_GAMMA": 2.0},
                           ["USE_FOCAL", "FOCAL_GAMMA"])
    assert uf is False and abs(fg - 3.5) < 1e-9
    del os.environ["LOSS_FOCAL_GAMMA"]

    class _V:
        n = 0
        def commit(self):
            _V.n += 1
    modal_shell_run("-V", [("--unused", None)], None, [_V()])
    assert _V.n == 1

    rng2 = np.random.RandomState(7)
    C = 5
    ks = rng2.randint(0, 9, (30_000, 3)).astype(np.int64)
    vs = rng2.rand(30_000, C).astype(np.float32)
    ps = rng2.rand(30_000, 3).astype(np.float32)
    acc = VoxelVoteAccum(C, max_rows=1_000)
    for s in range(0, 30_000, 700):
        acc.add(ks[s:s + 700], vs[s:s + 700], ps[s:s + 700])
    pred_a, rep_a = acc.result()
    first, inv = voxel_unique(ks, return_inverse=True)
    ref_votes = np.stack([np.bincount(inv, weights=vs[:, c], minlength=len(first))
                          for c in range(C)], axis=1)
    assert np.array_equal(rep_a, ps[first])
    assert np.array_equal(pred_a, ref_votes.argmax(1))
    assert VoxelVoteAccum(C).result() is None

    from scipy.spatial import cKDTree as _KD
    rep = rng2.rand(400, 3).astype(np.float32)
    pu = rng2.randint(0, C, 400)
    raw = rng2.rand(9_000, 3).astype(np.float32)
    lab = rng2.randint(-1, C + 1, 9_000).astype(np.int32)
    _, nn = _KD(rep).query(raw)
    v = (lab >= 0) & (lab < C)
    ri, ru, rg = score_ious(pu[nn][v], lab[v], C)
    i_, u_, g_, c_, t_ = score_raw_from_voxels(rep, pu, raw, lab, C, chunk=512)
    assert (np.array_equal(i_, ri) and np.array_equal(u_, ru)
            and np.array_equal(g_, rg))
    assert t_ == int(v.sum()) and c_ == int((pu[nn][v] == lab[v]).sum())
    print("ok")


if __name__ == "__main__":
    _demo()
