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
    print("ok")


if __name__ == "__main__":
    _demo()
