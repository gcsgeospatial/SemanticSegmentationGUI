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


def _intensity_norm_from_meta(meta: dict) -> str:
    """Where convert_dataset records the intensity normalization: under
    meta['source']['intensity_norm'] (a top-level copy is tolerated for other
    writers). Default 'max'. Getting this wrong feeds inference a different
    intensity scale than training saw."""
    src = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    return src.get("intensity_norm") or meta.get("intensity_norm") or "max"


def write_run_manifest(run_dir, backbone, dataset=None, weights="final_model.pth"):
    """Write run.json — THE single, self-contained inference manifest, next to the
    weights. Local inference reads ONLY this file (the user picks it explicitly):
    every input it needs is here, and the weights are its sibling `weights`. No
    searching, no path conventions.

    Derived from the run_config.json already written in `run_dir` (so the normalized
    fields stay in sync across backbones, which name grid/chunk differently) plus the
    dataset's intensity normalization. `backbone` is the backbone KEY (e.g. 'ptv3')."""
    rc = {}
    rc_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(rc_path):
        try:
            with open(rc_path, encoding="utf-8") as f:
                rc = json.load(f)
        except (OSError, ValueError):
            rc = {}
    inorm = "p95"   # the IEEE scripts normalize intensity to p95; canonical = dataset's
    if dataset:
        mp = f"/datasets/{dataset}/dataset_meta.json"
        try:
            with open(mp, encoding="utf-8") as f:
                inorm = _intensity_norm_from_meta(json.load(f))
        except (OSError, ValueError):
            inorm = "max"
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
        "label_map_asprs_to_index": rc.get("label_map_asprs_to_index"),  # ASPRS remap / IEEE flag
        "num_points": rc.get("num_points"),                              # RandLA sample size
    }
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


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
                                      "hag_source", "label_map_asprs_to_index", "num_points")}
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
            "label_map_asprs_to_index": rc.get("label_map_asprs_to_index"),
            "num_points": rc.get("num_points"),
        }
    return None


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

    # write_run_manifest: normalized run.json derived from run_config.json
    with open(os.path.join(d, "run_config.json"), "w") as f:
        json.dump({"num_classes": 7, "class_names": list("abcdefg"),
                   "grid_m": 2.0, "chunk_xy_m": 100.0, "hag_source": "pdal_hag_nn"}, f)
    m = write_run_manifest(d, "kpconvx_cold_hag")   # no dataset -> p95
    assert m["backbone"] == "kpconvx_cold_hag" and m["weights"] == "final_model.pth"
    assert m["grid"] == 2.0 and m["chunk_xy"] == 100.0 and m["num_classes"] == 7
    assert m["intensity_norm"] == "p95" and m["feature_mode"] == "hag"
    assert m["hag_source"] == "pdal_hag_nn"
    assert os.path.exists(os.path.join(d, "run.json"))

    # intensity_norm lives under dataset_meta['source'] — read it from there
    assert _intensity_norm_from_meta({"source": {"intensity_norm": "p95"}}) == "p95"
    assert _intensity_norm_from_meta({"intensity_norm": "p95"}) == "p95"   # tolerate top-level
    assert _intensity_norm_from_meta({"source": {}}) == "max"              # default

    # infer_meta reads run.json beside the weights (None for a bare .pth)
    im = infer_meta(os.path.join(d, "final_model.pth"))
    assert im and im["num_classes"] == 7 and im["grid"] == 2.0
    assert im["hag_source"] == "pdal_hag_nn" and im["class_names"] == list("abcdefg")
    assert infer_meta(os.path.join(tempfile.mkdtemp(), "bare.pth")) is None
    print("ok")


if __name__ == "__main__":
    _demo()
