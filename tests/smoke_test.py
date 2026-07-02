"""Local smoke test for the trainer_gui pipeline — no Modal calls, no GPU.

Run:  python tests/smoke_test.py   (from the trainer_gui/ project dir)

Covers: synthetic LAZ/PLY/ASCII scenes -> canonical conversion (field + companion
label specs), npz contract, meta + recommendations, inference-job conversion,
local prep caches (the live ptv3/randlanet cold paths), density-generalization
manifest round-trip, training-log parsing, viewer loading, local Docker backend.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
# train scripts moved under scripts/{local,modal,helper}; keep the bare module
# names importable (import_module("local_train_ptv3"), _modal_shim, …).
for _d in ("scripts/local", "scripts/modal", "scripts/helper"):
    sys.path.insert(0, str(_ROOT / _d))

from trainer_gui import analysis, dataset  # noqa: E402
from trainer_gui.dataset import LabelSpec  # noqa: E402

RNG = np.random.default_rng(7)
CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(("  [ok]  " if cond else "  [FAIL] ") + name)


def make_xyz(n=20_000, extent=100.0):
    xyz = RNG.uniform(0, extent, (n, 3))
    xyz[:, 2] *= 0.1
    return xyz


def write_laz(path, xyz, labels, intensity):
    import laspy
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(0)
    header.scales = [0.001] * 3
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = labels.astype(np.uint8)
    las.intensity = intensity.astype(np.uint16)
    las.red = (RNG.uniform(0, 65535, len(xyz))).astype(np.uint16)
    las.green = (RNG.uniform(0, 65535, len(xyz))).astype(np.uint16)
    las.blue = (RNG.uniform(0, 65535, len(xyz))).astype(np.uint16)
    las.write(str(path))


def write_ply(path, xyz, labels):
    from plyfile import PlyData, PlyElement
    arr = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                    ("label", "i4")])
    arr["x"], arr["y"], arr["z"] = xyz.T.astype(np.float32)
    arr["red"] = arr["green"] = arr["blue"] = RNG.integers(0, 255, len(xyz), dtype=np.uint8)
    arr["label"] = labels
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def write_ascii(pc_path, cls_path, xyz, labels, intensity):
    pc = np.column_stack([xyz, intensity, np.ones(len(xyz))])
    np.savetxt(pc_path, pc, delimiter=",", fmt="%.3f")
    np.savetxt(cls_path, labels, fmt="%d")


def main():
    # Windows consoles default to cp1252; any non-ASCII in test or progress
    # output (✓, …, ²) would otherwise crash the run on the print, not the logic.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    tmp = Path(tempfile.mkdtemp(prefix="trainer_gui_smoke_"))
    print(f"workdir: {tmp}")
    try:
        # ---------------- LAZ dataset with field labels (value 0 ignored)
        # Provide explicit val + test folders so the three materialized splits are
        # deterministic (train keeps scene0/scene1; val + test come verbatim).
        laz_root = tmp / "laz_src"
        for split, k in (("train", 2), ("val", 1), ("test", 1)):
            d = laz_root / split
            d.mkdir(parents=True)
            for i in range(k):
                xyz = make_xyz()
                labels = RNG.choice([0, 2, 5, 6], len(xyz), p=[0.1, 0.5, 0.3, 0.1])
                write_laz(d / f"scene{i}.laz", xyz, labels, RNG.uniform(0, 4000, len(xyz)))

        spec = LabelSpec(kind="field", field="classification")
        files = dataset.discover_scenes(laz_root / "train")
        check("laz: discovered scenes", len(files) == 2)
        counts = dataset.scan_label_values(files, spec)
        check("laz: scanned values {0,2,5,6}", set(counts) == {0, 2, 5, 6})

        classes = [{"index": 0, "source_value": 2, "name": "Ground"},
                   {"index": 1, "source_value": 5, "name": "Veg"},
                   {"index": 2, "source_value": 6, "name": "Building"}]
        staged = dataset.convert_dataset("laz_demo", str(laz_root / "train"), spec, classes,
                                         [0], tmp / "staging",
                                         val_inputs=[str(laz_root / "val")],
                                         test_inputs=[str(laz_root / "test")], progress=print)
        z = np.load(staged / "train" / "scene0.npz")
        check("laz npz: keys", {"xyz", "label", "rgb", "intensity"}.issubset(z.files))
        check("laz npz: xyz f32 (N,3)", z["xyz"].dtype == np.float32 and z["xyz"].shape[1] == 3)
        check("laz npz: label i32 in {-1,0,1,2}",
              z["label"].dtype == np.int32 and set(np.unique(z["label"])) <= {-1, 0, 1, 2})
        check("laz npz: ignored value 0 -> -1", (z["label"] == -1).sum() > 0)
        check("laz npz: intensity normalized 0..1",
              0.0 <= z["intensity"].min() and z["intensity"].max() <= 1.0)
        meta = json.loads((staged / "dataset_meta.json").read_text())
        check("meta: num_classes 3", meta["num_classes"] == 3)
        check("meta: class names ordered", meta["class_names"] == ["Ground", "Veg", "Building"])
        check("meta: schema v2 + split block",
              meta["schema_version"] == 2
              and {"mode", "seed", "requested", "achieved"} <= set(meta.get("split", {})))
        check("meta: three materialized splits (train/val/test)",
              set(meta["splits"]) == {"train", "val", "test"}
              and all(meta["splits"][sp]["scenes"] for sp in ("train", "val", "test"))
              and all(sorted((staged / sp).glob("*.npz")) for sp in ("train", "val", "test")))
        check("meta: per-class train/val/test counts populated",
              meta["classes"][0]["train_count"] > 0
              and all(f"{sp}_count" in meta["classes"][0] for sp in ("train", "val", "test")))
        check("meta: recommendations for all backbones",
              "ptv3" in meta["recommendations"] and "kpconvx_cold" in meta["recommendations"])
        grid = meta["recommendations"]["ptv3"]["grid"]
        # 20k pts over ~100x100m -> 2 pts/m² -> spacing .7m -> 3x = 2.1 clamped to 0.6
        check(f"meta: ptv3 grid clamped to band (got {grid})", 0.05 <= grid <= 0.6)
        check("meta: kpconvx grid clamped to its band",
              0.5 <= meta["recommendations"]["kpconvx_cold"]["grid"] <= 3.0)

        # ---------------- scene split: a folder of clouds is split by WHOLE scenes
        # (no tiling) into three disjoint train/val/test folders.
        sc_src = tmp / "scene_src"
        sc_src.mkdir()
        for i in range(3):
            xyz = make_xyz()
            labels = RNG.choice([0, 2, 5, 6], len(xyz), p=[0.1, 0.5, 0.3, 0.1])
            write_laz(sc_src / f"region{i}.laz", xyz, labels, RNG.uniform(0, 4000, len(xyz)))
        staged_sc = dataset.convert_dataset(
            "laz_scene", str(sc_src), spec, classes, [0], tmp / "staging",
            split=dataset.SplitConfig(val_frac=0.34, test_frac=0.33, mode="random", seed=42))
        msc = json.loads((staged_sc / "dataset_meta.json").read_text())
        tr_sc = msc["splits"]["train"]["scenes"]
        va_sc = msc["splits"]["val"]["scenes"]
        te_sc = msc["splits"]["test"]["scenes"]
        check("scene-split: whole scenes, no tiling (one npz per source file)",
              len(tr_sc) + len(va_sc) + len(te_sc) == 3
              and bool(tr_sc) and bool(va_sc) and bool(te_sc))
        check("scene-split: no scene shared across splits (leak-free)",
              not (set(tr_sc) & set(va_sc)) and not (set(tr_sc) & set(te_sc))
              and not (set(va_sc) & set(te_sc)))
        check("scene-split: meta records split mode + seed",
              msc["split"]["mode"] == "random" and msc["split"]["seed"] == 42)
        # Parallel conversion must not change the split: same folder + seed -> same
        # scene->split mapping (guards the order-preserving map in _convert_many).
        staged_sc2 = dataset.convert_dataset(
            "laz_scene2", str(sc_src), spec, classes, [0], tmp / "staging",
            split=dataset.SplitConfig(val_frac=0.34, test_frac=0.33, mode="random", seed=42))
        msc2 = json.loads((staged_sc2 / "dataset_meta.json").read_text())
        check("scene-split: deterministic across runs (parallel-safe)",
              all(sorted(msc["splits"][sp]["scenes"]) == sorted(msc2["splits"][sp]["scenes"])
                  for sp in ("train", "val", "test")))
        # Worker cap stays sane: >=1, never more than the file count (3 here).
        wc = dataset._worker_cap(list(sc_src.glob("*.laz")))
        check(f"convert: worker cap RAM/core-clamped (got {wc})", 1 <= wc <= 3)

        # ---------------- single-cloud split: one cloud is tile-measured and
        # reassembled into three holey train/val/test clouds (seam_buffer=0 here so
        # points are conserved; a positive buffer would drop a seam to limit leakage).
        staged_sp = dataset.convert_dataset(
            "laz_spatial", str(laz_root / "train" / "scene0.laz"), spec, classes, [0],
            tmp / "staging", split=dataset.SplitConfig(val_frac=0.3, test_frac=0.3,
                                                       mode="random", seed=42,
                                                       seam_buffer_m=0.0, tile_m=20.0))
        msp = json.loads((staged_sp / "dataset_meta.json").read_text())
        nt = len(msp["splits"]["train"]["scenes"])
        nv = len(msp["splits"]["val"]["scenes"])
        nte = len(msp["splits"]["test"]["scenes"])
        check("single-cloud split: one holey cloud per split (train/val/test)",
              nt == 1 and nv == 1 and nte == 1)
        check("single-cloud split: points conserved (seam_buffer=0 -> 20k in -> 20k out)",
              msp["splits"]["train"]["total_points"]
              + msp["splits"]["val"]["total_points"]
              + msp["splits"]["test"]["total_points"] == 20_000)
        check("single-cloud split: meta marks tile atoms", msp["split"]["atom_unit"] == "tile")

        # ---------------- PLY dataset with field labels
        ply_root = tmp / "ply_src"
        for split, k in (("train", 2), ("val", 1)):
            d = ply_root / split
            d.mkdir(parents=True)
            for i in range(k):
                xyz = make_xyz()
                write_ply(d / f"tile{i}.ply", xyz, RNG.choice([1, 2, 3], len(xyz)))
        spec_ply = LabelSpec(kind="field", field="label")
        classes_ply = [{"index": i, "source_value": v, "name": f"c{v}"}
                       for i, v in enumerate([1, 2, 3])]
        staged_ply = dataset.convert_dataset("ply_demo", str(ply_root / "train"), spec_ply,
                                             classes_ply, [], tmp / "staging",
                                             val_inputs=[str(ply_root / "val")])
        zp = np.load(sorted((staged_ply / "train").glob("*.npz"))[0])
        check("ply npz: rgb present u8", zp["rgb"].dtype == np.uint8)
        check("ply npz: all labels mapped", set(np.unique(zp["label"])) == {0, 1, 2})

        # ---------------- ASCII dataset with companion label files (per-scene)
        txt_root = tmp / "txt_src"
        truth = txt_root / "truth"
        truth.mkdir(parents=True)
        for split, k in (("train", 2), ("val", 1)):
            d = txt_root / split
            d.mkdir(parents=True)
            for i in range(k):
                xyz = make_xyz(8_000)
                labels = RNG.choice([0, 2, 6], len(xyz))
                write_ascii(d / f"JAX_{split}{i}_PC3.txt", truth / f"JAX_{split}{i}_CLS.txt",
                            xyz, labels, RNG.uniform(0, 100, len(xyz)))
        spec_txt = LabelSpec(kind="file", truth_dir=str(truth),
                             src_suffix="_PC3.txt", dst_suffix="_CLS.txt")
        counts_txt = dataset.scan_label_values(dataset.discover_scenes(txt_root / "train"), spec_txt)
        check("ascii: companion labels scanned", set(counts_txt) == {0, 2, 6})
        classes_txt = [{"index": 0, "source_value": 2, "name": "Ground"},
                       {"index": 1, "source_value": 6, "name": "Building"}]
        staged_txt = dataset.convert_dataset("txt_demo", str(txt_root / "train"), spec_txt,
                                             classes_txt, [0], tmp / "staging",
                                             val_inputs=[str(txt_root / "val")])
        zt = np.load(sorted((staged_txt / "train").glob("*.npz"))[0])
        check("ascii npz: intensity + return_number captured",
              "intensity" in zt.files and "return_number" in zt.files)

        # ---------------- inference job conversion (no labels)
        job = dataset.convert_infer_job("test_job", str(txt_root / "val"), tmp / "staging")
        zi = np.load(job / "scenes" / "JAX_val0_PC3.npz")
        check("infer npz: no label key", "label" not in zi.files)
        check("infer: job_meta written", (job / "job_meta.json").exists())
        check("infer npz: p95 intensity clipped to [0,2]",
              0.0 <= zi["intensity"].min() and zi["intensity"].max() <= 2.0)

        # ---------------- backbones + appstate (imports used below)
        from trainer_gui import appstate
        from trainer_gui.backbones import BACKBONES, infer_backbones

        # intensity normalization modes differ: p95 scales hotter than max
        one = dataset.discover_scenes(txt_root / "val")[0]
        dataset.convert_scene(one, None, {}, tmp / "imax.npz", intensity_norm="max")
        dataset.convert_scene(one, None, {}, tmp / "ip95.npz", intensity_norm="p95")
        imax, ip95 = np.load(tmp / "imax.npz")["intensity"], np.load(tmp / "ip95.npz")["intensity"]
        check("convert: max-norm intensity in [0,1]", imax.max() <= 1.0 + 1e-6)
        check("convert: p95-norm intensity >= max-norm and <= 2",
              ip95.max() >= imax.max() and ip95.max() <= 2.0)

        # ---------------- backbones: infer-readiness contracts
        check("backbones: folder-infer set = the 6 cold/hag scripts",
              set(infer_backbones()) == {"ptv3", "randlanet", "ptv3_hag", "randlanet_hag",
                                         "kpconvx_cold", "kpconvx_cold_hag"})
        check("backbones: randlanet uses --sub-grid and has no --chunk-xy",
              BACKBONES["randlanet"].grid_flag == "sub-grid"
              and not BACKBONES["randlanet"].has_chunk)
        check("backbones: ptv3 uses --grid and has --chunk-xy",
              BACKBONES["ptv3"].grid_flag == "grid" and BACKBONES["ptv3"].has_chunk)
        check("backbones: kpconvx is folder-inferable and canonical-trainable",
              BACKBONES["kpconvx_cold"].folder_infer and BACKBONES["kpconvx_cold"].ready
              and BACKBONES["kpconvx_cold_hag"].folder_infer
              and BACKBONES["kpconvx_cold_hag"].ready)
        check("backbones: outputs volumes follow the renamed apps",
              BACKBONES["ptv3"].outputs_volume == "ptv3-outputs"
              and BACKBONES["randlanet"].outputs_volume == "randlanet-cold-outputs"
              and BACKBONES["ptv3_hag"].outputs_volume == "ptv3-hag-outputs")
        check("backbones: each carries a recommended GPU + min VRAM (Train specs bar)",
              all(b.rec_gpu and b.min_vram_gb > 0 for b in BACKBONES.values())
              and BACKBONES["kpconvx_cold"].min_vram_gb >= BACKBONES["randlanet"].min_vram_gb)

        # ---------------- density generalization: env mapping + run.json round-trip
        import train_common

        # dg_config_to_env emits TRAIN-time vars only (density aug + the logdk channel,
        # which changes the input width). AdaBN/TTA are inference-time (set on the Infer
        # page), so they must NOT leak into the train launch env.
        dg_cfg = {"density_aug": True, "coarsen_max": 2.5, "p_native": 0.5,
                  "logdk": True, "logdk_k": 12, "adabn": True, "tta": 3}
        dg_env = analysis.dg_config_to_env(dg_cfg)
        check("dg: config_to_env emits train-time aug + logdk vars",
              dg_env.get("DG_DENSITY_AUG") == "1" and dg_env.get("DG_LOGDK_FEAT") == "1"
              and dg_env.get("DG_LOGDK_K") == "12")
        check("dg: config_to_env does NOT emit inference-time AdaBN/TTA",
              "DG_INFER_ADABN" not in dg_env and "DG_INFER_TTA" not in dg_env)
        check("dg: empty config -> baseline (no vars)", analysis.dg_config_to_env({}) == {})

        # loss/class-balance: only non-default knobs become LOSS_*/RARE_* env
        base = {"focal": False, "focal_gamma": 2.0, "class_weighting": True,
                "weight_beta": 0.5, "rare_oversample": True}
        check("loss: all-default config emits nothing", analysis.loss_config_to_env(base) == {})
        le = analysis.loss_config_to_env({**base, "focal": True, "focal_gamma": 3.0,
                                          "weight_beta": 1.0, "rare_oversample": False})
        check("loss: focal+gamma+beta+rare overrides map to env",
              le == {"LOSS_FOCAL": "1", "LOSS_FOCAL_GAMMA": "3.0",
                     "LOSS_WEIGHT_BETA": "1.0", "RARE_OVERSAMPLE": "0"})
        check("loss: gamma is ignored unless focal is on",
              "LOSS_FOCAL_GAMMA" not in analysis.loss_config_to_env({**base, "focal_gamma": 3.0}))

        # write_run_manifest bakes the DG block (read from the training env) into run.json
        # so a logdk model is self-describing; infer_meta reads it back at inference.
        dg_run = tmp / "dg_run"
        dg_run.mkdir()
        (dg_run / "run_config.json").write_text(json.dumps(
            {"num_classes": 5, "class_names": list("abcde"), "grid_m": 2.0}), encoding="utf-8")
        os.environ["DG_LOGDK_FEAT"], os.environ["DG_LOGDK_K"] = "1", "12"
        try:
            man = train_common.write_run_manifest(str(dg_run), "kpconvx_cold")
        finally:
            os.environ.pop("DG_LOGDK_FEAT", None)
            os.environ.pop("DG_LOGDK_K", None)
        check("dg: run.json records the dg block (logdk + k) from the train env",
              man["dg"]["logdk"] is True and man["dg"]["logdk_k"] == 12)
        im = train_common.infer_meta(str(dg_run / "final_model.pth"))
        check("dg: infer_meta reads the dg block back (self-describing weights)",
              im["dg"]["logdk"] is True and im["dg"]["logdk_k"] == 12)
        check("dg: baseline run (no DG env) records logdk off",
              train_common.write_run_manifest(str(dg_run), "ptv3")["dg"]["logdk"] is False)

        # Train page's Modal-presence check parses `modal volume ls --json` entries
        # whose basename key varies by CLI version — lock the parser.
        from trainer_gui.pages.infer_page import _entry_name, _localize_paths
        check("infer: _entry_name reads path/Filename/name across CLI shapes",
              _entry_name({"path": "/ds/dataset_meta.json"}) == "dataset_meta.json"
              and _entry_name({"Filename": "train/"}) == "train"
              and _entry_name({"name": "a.npz"}) == "a.npz"
              and _entry_name({}) == "")
        # Local runs must show the host output folder, not the container's /datasets
        # bind-mount path — both the leading-slash and bare forms the scripts print.
        check("infer: _localize_paths maps container paths to the host output folder",
              _localize_paths("labeling -> /datasets/_infer/J/predictions/s_pred.ply",
                              "J", r"C:\out", r"C:\stage") == r"labeling -> C:\out/s_pred.ply"
              and _localize_paths("done — predictions in _infer/J/predictions",
                                  "J", r"C:\out", r"C:\stage") == r"done — predictions in C:\out")

        # ---------------- prediction export (Inference page format picker)
        # The scripts' coloured *_pred.ply -> chosen format carrying xyz +
        # classification ONLY (no RGB); class indices invert losslessly from the
        # palette colours; the coloured source ply is consumed.
        from trainer_gui.palette import palette_for
        exd = tmp / "pred_export"
        exd.mkdir()
        ex_cls = np.array([0, 1, 2, 3, 4, 2, 0], np.int64)
        hdr7 = ("ply\nformat ascii 1.0\nelement vertex 7\nproperty float x\n"
                "property float y\nproperty float z\nproperty uchar red\n"
                "property uchar green\nproperty uchar blue\nend_header")

        def _mk_pred():
            np.savetxt(exd / "sceneX_pred.ply",
                       np.column_stack([make_xyz(7), palette_for(5)[ex_cls]]),
                       fmt=["%.3f"] * 3 + ["%d"] * 3, header=hdr7, comments="")

        import laspy
        _mk_pred()
        las_out = dataset.export_predictions(exd, "las")[0]
        check("export: las carries the palette-inverted classes, source ply consumed",
              las_out.suffix == ".las"
              and list(laspy.read(str(las_out)).classification) == list(ex_cls)
              and not (exd / "sceneX_pred.ply").exists())
        _mk_pred()
        csv_out = dataset.export_predictions(exd, "csv")[0]
        rows = np.genfromtxt(csv_out, delimiter=",", names=True)
        check("export: csv has an x,y,z,classification header + the classes",
              set(rows.dtype.names) == {"x", "y", "z", "classification"}
              and list(rows["classification"].astype(int)) == list(ex_cls))
        _mk_pred()
        from plyfile import PlyData
        ply_out = dataset.export_predictions(exd, "ply")[0]
        pv = PlyData.read(str(ply_out))["vertex"]
        check("export: ply rewrite drops RGB, keeps only a classification column",
              {p.name for p in pv.properties} == {"x", "y", "z", "classification"}
              and list(np.asarray(pv["classification"]).astype(int)) == list(ex_cls))

        # ---------------- plots: read val curves, build figures, average runs
        from trainer_gui import plots
        run = tmp / "runs_demo" / "ptv3" / "20260101_000000_demo_ptv3"
        run.mkdir(parents=True)
        (run / "val_metrics.csv").write_text(
            "epoch,val_acc,val_miou,iou_Ground,iou_Trees\n"
            "9,0.80,0.40,0.70,0.30\n19,0.90,0.55,0.80,0.50\n", encoding="utf-8")
        (run / "metrics.csv").write_text(
            "epoch,train_loss,train_iou\n0,0.90,0.30\n1,0.50,0.60\n", encoding="utf-8")
        (run / "run_config.json").write_text(json.dumps(
            {"backbone": "PTv3", "dataset": "demo", "class_names": ["Ground", "Trees"]}),
            encoding="utf-8")
        (run / "test_metrics.json").write_text(json.dumps(
            {"val": {"overall_mIoU": 0.55, "per_class_iou": {"Ground": 0.8, "Trees": 0.5}},
             "test": {"overall_mIoU": 0.50, "per_class_iou": {"Ground": 0.75, "Trees": 0.45}}}),
            encoding="utf-8")
        ex, ey = plots.val_series(run, "val_miou")
        check("plots: val_series reads (epoch, val_miou)", ex == [9, 19] and ey == [0.40, 0.55])
        check("plots: available metrics include per-class IoU",
              set(plots.available_metrics(run)) >= {"val_miou", "val_acc", "iou_Ground", "iou_Trees"})
        check("plots: available metrics also include metrics.csv columns",
              {"train_loss", "train_iou"}.issubset(plots.available_metrics(run)))
        check("plots: series reads a metrics.csv (training) column",
              plots.series(run, "train_loss") == ([0, 1], [0.90, 0.50]))
        check("plots: discover_runs walks backbone subdirs", run in plots.discover_runs(tmp / "runs_demo"))
        check("plots: single_run_figure builds a dashboard", len(plots.single_run_figure(run).get_axes()) >= 4)
        run2 = tmp / "runs_demo" / "ptv3" / "20260102_000000_demo_ptv3"
        shutil.copytree(run, run2)
        fig = plots.multi_run_figure([run, run2], "val_miou", show_runs=True, show_avg=True)
        line_labels = [str(ln.get_label()) for ln in fig.get_axes()[0].get_lines()]
        check("plots: multi_run_figure overlays runs + an average line",
              len(line_labels) == 3 and any("average" in s for s in line_labels))

        # ---------------- local prep caches (live cold backbones: ptv3, randlanet)
        from trainer_gui import prep

        # ptv3 (cold): rgb tiles. prep mirrors the dataset's three materialized
        # splits verbatim (val = selection holdout, test = final report) — it does
        # NOT re-derive its own val.
        pd = prep.prep_dataset("ptv3", staged, {"chunk-xy": 50.0}, progress=print)
        check("prep ptv3: dir name", pd.name == "ptv3_cold_chunk50")
        train_tiles = sorted((pd / "train").glob("*.npz"))
        val_tiles = sorted((pd / "val").glob("*.npz"))
        test_tiles = sorted((pd / "test").glob("*.npz"))
        check("prep ptv3: tiles in train + val + test",
              bool(train_tiles) and bool(val_tiles) and bool(test_tiles))
        check("prep ptv3: cold tile keys (rgb layout)",
              set(np.load(train_tiles[0]).files) == {"xyz", "rgb", "lab"})

        # randlanet (cold): whole-scene grid subsample, fewer points than input
        pr = prep.prep_dataset("randlanet", staged, {"sub-grid": 0.3})
        check("prep randlanet: dir name", pr.name == "randlanet_cold_grid30")
        zr = np.load(next((pr / "train").glob("*.npz")))
        zin = np.load(next((staged / "train").glob("*.npz")))
        check("prep randlanet: keys + subsampled",
              set(zr.files) == {"xyz", "rgb", "lab"} and len(zr["xyz"]) < len(zin["xyz"]))

        # idempotency: a second run does no extra work and keeps tile counts
        n_before = len(list((pd / "train").glob("*.npz")))
        prep.prep_dataset("ptv3", staged, {"chunk-xy": 50.0})
        check("prep: idempotent re-run",
              len(list((pd / "train").glob("*.npz"))) == n_before)

        # the live backbones without a local prep path are correctly unsupported
        check("prep: kpconvx_cold + hag variants have no local prep",
              not prep.supports_local_prep("kpconvx_cold")
              and not prep.supports_local_prep("ptv3_hag")
              and not prep.supports_local_prep("randlanet_hag"))

        # ---------------- Pretraining: tile_for_model (no PDAL needed)
        from trainer_gui import pretrain

        pt_out = tmp / "pretrain_tile"
        prep_dir = pretrain.tile_for_model(str(laz_root / "train"), pt_out,
                                           "ptv3", {"chunk-xy": 50.0}, progress=print)
        check("pretrain tile: prep dir tag", prep_dir.name == "ptv3_cold_chunk50")
        check("pretrain tile: staged under out/_staged",
              (pt_out / "_staged" / "train").exists())
        pt_tiles = sorted((prep_dir / "train").glob("*.npz"))
        check("pretrain tile: train tiles produced", len(pt_tiles) > 0)
        zpt = np.load(pt_tiles[0])
        check("pretrain tile: tile keys match prep contract",
              set(zpt.files) == {"xyz", "rgb", "lab"})

        # ---------------- Pretraining: add_hag (skipped if PDAL absent)
        try:
            import pdal  # noqa: F401
            have_pdal = True
        except Exception:  # noqa: BLE001
            have_pdal = False
        if have_pdal:
            hag_src = tmp / "hag_src"
            hag_src.mkdir()
            xyz = make_xyz(20_000)
            # Low points are ground (class 2) so hag_nn has a surface to measure.
            labels = np.where(xyz[:, 2] < 0.5, 2, 6).astype(np.uint8)
            write_laz(hag_src / "scene_h.las", xyz, labels, RNG.uniform(0, 4000, len(xyz)))
            hag_out = tmp / "hag_out"
            summary = pretrain.add_hag(str(hag_src), hag_out, ground_class=2,
                                       use_smrf=False, hag_filter="hag_nn", progress=print)
            check("pretrain hag: output laz written", (hag_out / "scene_h.laz").exists())
            sidecar = hag_out / "scene_h.json"
            check("pretrain hag: json sidecar written", sidecar.exists())
            sj = json.loads(sidecar.read_text())
            check("pretrain hag: sidecar has hag stats",
                  {"hag_min", "hag_mean", "hag_max"}.issubset(sj))
            check("pretrain hag: summary json written",
                  (hag_out / "pretrain_summary.json").exists() and summary["n_files"] == 1)

            # txt input: a clear ground plane + elevated points so SMRF finds ground
            txt_hag = tmp / "hag_txt_src"
            txt_hag.mkdir()
            ground = make_xyz(10_000)
            ground[:, 2] = RNG.uniform(0, 0.05, len(ground))      # flat ground ~0 m
            above = make_xyz(10_000)
            above[:, 2] = RNG.uniform(3, 8, len(above))           # elevated points
            gxyz = np.vstack([ground, above])
            pc = np.column_stack([gxyz, RNG.uniform(0, 100, len(gxyz)),
                                  np.ones(len(gxyz))])             # x,y,z,intensity,ret
            np.savetxt(txt_hag / "scene_t_PC3.txt", pc, delimiter=",", fmt="%.3f")
            txt_out = tmp / "hag_txt_out"
            pretrain.add_hag(str(txt_hag), txt_out, use_smrf=True,
                             hag_filter="hag_nn", progress=print)
            check("pretrain hag: txt input -> laz", (txt_out / "scene_t_PC3.laz").exists())
            zt_hag = json.loads((txt_out / "scene_t_PC3.json").read_text())
            check("pretrain hag: txt sidecar HAG spans ground..elevated",
                  zt_hag["hag_max"] > 2.0)

        else:
            print("  - pretrain hag: skipped (pdal not installed)")

        # per-tile HAG over a converted dataset -> sibling <name>_hag with a hag
        # key. Default method is now "grid" (raster approximation) — no PDAL.
        hag_ds = dataset.add_hag_to_dataset(staged, tmp / "staging" / "laz_demo_hag",
                                            progress=print)
        zhd = np.load(hag_ds / "train" / "scene0.npz")
        check("dataset: add_hag_to_dataset adds a per-tile hag channel",
              "hag" in zhd.files and zhd["hag"].shape[0] == zhd["xyz"].shape[0])
        check("dataset: add_hag covers the test split too",
              "hag" in np.load(hag_ds / "test" / "scene0.npz").files)
        mhd = json.loads((hag_ds / "dataset_meta.json").read_text())
        check("dataset: hag dataset meta tags per_tile source + keeps classes",
              mhd["source"]["hag_source"] == "per_tile_grid"
              and mhd["num_classes"] == 3 and isinstance(mhd["has_hag"], bool))

        # inline HAG: bake it during tiling in one pass (no separate reload).
        # ground_value=2 (the "Ground" source value) -> the labels are the ONLY
        # ground source (SMRF never runs); grid nearest-fills label gaps.
        staged_ih = dataset.convert_dataset(
            "laz_hag_inline", str(laz_root / "train"), spec, classes, [0], tmp / "staging",
            val_inputs=[str(laz_root / "val")], test_inputs=[str(laz_root / "test")],
            compute_hag=True, ground_value=2, use_smrf=True, progress=print)
        zih = np.load(staged_ih / "train" / "scene0.npz")
        check("convert_dataset(compute_hag): tiles carry a hag channel inline",
              "hag" in zih.files and zih["hag"].shape[0] == zih["xyz"].shape[0])
        mih = json.loads((staged_ih / "dataset_meta.json").read_text())
        check("convert_dataset(compute_hag): meta records grid+labels ground + has_hag",
              mih["has_hag"] is True
              and mih["source"]["hag_source"] == "grid+labels"
              and mih["source"]["hag_ground_value"] == 2
              and mih["source"]["hag_use_smrf"] is False)

        # ---------------- analysis.scan_folder on raw files
        stats = analysis.scan_folder(dataset.discover_scenes(laz_root / "train"))
        check("analysis: density > 0", stats["mean_pts_per_m2"] > 0)
        check("analysis: rgb detected", stats["has_rgb"])

        # ---------------- log parser (needs QCoreApplication for signals)
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication.instance() or QCoreApplication([])
        from trainer_gui.jobs import LogParser
        seen = {"epochs": [], "run": None}
        p = LogParser()
        p.epoch.connect(lambda m: seen["epochs"].append(m))
        p.run_id.connect(lambda r: seen.update(run=r))
        p.feed("  ep  12: loss=0.4321 acc=0.9123 miou=0.7012 s/iter=0.123 s/ep=61.4\n")
        p.feed("  [predict] labeling 1 scene(s) -> /outputs/runs/20260611_010101_demo_ptv3/predictions\n")
        check("parser: epoch line parsed", seen["epochs"]
              and seen["epochs"][0]["epoch"] == 12 and abs(seen["epochs"][0]["miou"] - 0.7012) < 1e-9)
        check("parser: run id extracted", seen["run"] == "20260611_010101_demo_ptv3")

        # ---------------- viewer loader on a synthetic prediction npz
        pred_path = tmp / "scene_pred.npz"
        np.savez_compressed(pred_path, xyz=make_xyz(1000).astype(np.float32),
                            pred=RNG.integers(0, 3, 1000).astype(np.int32),
                            class_names=np.array(["Ground", "Veg", "Building"]))
        from trainer_gui.viewer import _load
        xyz, rgb01, key = _load(pred_path)
        check("viewer: pred npz loads colored with a 3-class colour key",
              rgb01 is not None and len(key) == 3
              and all(isinstance(lab, str) and len(col) == 3 for lab, col in key))

        # canonical dataset tile (xyz + label, -1=ignore) must colour by class too
        from trainer_gui.palette import palette_for
        lbl_path = tmp / "scene_label.npz"
        lab = np.array([-1, 0, 1, 2, 0], np.int32)
        np.savez_compressed(lbl_path, xyz=make_xyz(5).astype(np.float32), label=lab)
        _, rgb_l, key_l = _load(lbl_path)
        pal3 = palette_for(3)
        check("viewer: label npz colours by class, -1 stays grey",
              rgb_l is not None
              and np.allclose(rgb_l[1], pal3[0] / 255.0)     # class 0 -> palette[0]
              and np.allclose(rgb_l[0], 0.55)                # ignore -> grey
              and len(key_l) == 3)

        # ---------------- existing repo prediction as a viewer fixture
        jax = Path(__file__).resolve().parents[2] / "JAX_066_pred.ply"
        if jax.exists():
            xyz2, rgb2, _ = _load(jax)
            check("viewer: JAX_066_pred.ply loads with colors",
                  len(xyz2) > 0 and rgb2 is not None)

        # ---------------- ground-truth comparison (error map)
        from trainer_gui.palette import class_from_rgb, palette_for
        from trainer_gui.viewer import compare_clouds, error_colors
        # class_from_rgb losslessly inverts the categorical palette used to colour
        # prediction PLYs (so a class-coloured cloud decodes back to class indices).
        cls = np.array([0, 1, 2, 3, 4, 0, 2], np.int64)
        check("palette: class_from_rgb inverts palette_for colours",
              list(class_from_rgb(palette_for(5)[cls])) == list(cls))
        ec = error_colors(np.array([0, 1, 2, 3]), np.array([0, 2, 2, -1]))
        check("viewer: error_colors yellow on mismatch, grey on correct + no-GT",
              tuple(ec[1]) == (1.0, 1.0, 0.0)        # wrong -> yellow
              and abs(ec[0][0] - 0.55) < 1e-9        # correct -> grey
              and abs(ec[2][0] - 0.55) < 1e-9        # correct -> grey
              and abs(ec[3][0] - 0.55) < 1e-9)       # no GT  -> grey
        ec_i = error_colors(np.array([0, 1, 2, 3]), np.array([9, 9, 2, 3]),  # pt0,1 wrong
                            intensity=np.array([0.0, 1.0, 0.0, 1.0]))
        check("viewer: both wrong (yellow) and correct (grey) vary with intensity",
              ec_i[0][2] == 0.0 and ec_i[1][2] == 0.0        # wrong -> yellow (no blue)
              and ec_i[0][0] == ec_i[0][1]                   # wrong is yellow (R=G)
              and ec_i[0][0] != ec_i[1][0]                   # wrong brightness varies w/ intensity
              and ec_i[2][2] > 0.0 and ec_i[3][2] > 0.0      # correct -> grey (has blue)
              and ec_i[2][0] != ec_i[3][0])                  # correct brightness varies
        # end-to-end: a palette-coloured prediction PLY vs a class-coloured .ply GT
        cmp = tmp / "cmp"
        cmp.mkdir()
        pal5 = palette_for(5)
        hdr = ("ply\nformat ascii 1.0\nelement vertex 7\nproperty float x\nproperty float y\n"
               "property float z\nproperty uchar red\nproperty uchar green\n"
               "property uchar blue\nend_header")
        pred_cls = np.array([0, 1, 2, 3, 4, 0, 2], np.int64)
        np.savetxt(cmp / "scene900_pred.ply",
                   np.column_stack([make_xyz(7).astype(np.float32), pal5[pred_cls]]),
                   fmt=["%.3f"] * 3 + ["%d"] * 3, header=hdr, comments="")
        gt_cls = np.array([0, 1, 2, 3, 4, 0, 1], np.int64)   # only pt 6 differs (pred 2 vs gt 1)
        np.savetxt(cmp / "scene900_gt.ply",
                   np.column_stack([make_xyz(7).astype(np.float32), pal5[gt_cls]]),
                   fmt=["%.3f"] * 3 + ["%d"] * 3, header=hdr, comments="")
        _, vc_ply, ckey = compare_clouds(cmp / "scene900_pred.ply", cmp / "scene900_gt.ply")
        check("viewer: compare_clouds flags exactly the mismatched point yellow",
              int((np.abs(vc_ply - np.array([1.0, 1.0, 0.0])).sum(1) < 1e-9).sum()) == 1)
        check("viewer: compare key is just wrong (yellow) + correct (grey)",
              ckey == [("wrong prediction", (1.0, 1.0, 0.0)), ("correct", (0.55, 0.55, 0.55))])

        # chosen-dataset palette (Inference menu): a prediction coloured with the
        # shared categorical palette is labelled with the picked class names, and a
        # raw RGB cloud gets no spurious legend.
        from trainer_gui.palette import palette_for
        from trainer_gui.viewer import _key_for_names, _load
        names3, pal3 = ["A", "B", "C"], palette_for(3)
        kc = _key_for_names(pal3[np.array([0, 1, 2, 1, 0])], names3)   # palette-coloured
        check("viewer: _key_for_names labels a chosen-palette cloud with all class names",
              [n for n, _ in kc] == names3 and np.allclose(kc[0][1], pal3[0] / 255.0))
        check("viewer: _key_for_names gives no key for a non-palette (raw RGB) cloud",
              _key_for_names(np.array([[1, 2, 3], [4, 5, 6]], np.uint8), names3) == [])
        nn = tmp / "pred_nonames.npz"
        np.savez_compressed(nn, xyz=make_xyz(5).astype(np.float32),
                            pred=np.array([0, 1, 2, 1, 0], np.int32))   # no class_names
        _, _, k_nn = _load(nn, ["X", "Y", "Z"])
        check("viewer: _load applies chosen class names to an npz lacking class_names",
              [n for n, _ in k_nn] == ["X", "Y", "Z"])

        # ================= LOCAL (Docker) backend =================
        import importlib

        import _modal_shim
        from trainer_gui import local_cli
        BB = BACKBONES

        # the modal shim must import every training script (no torch / no cloud)
        # and expose its @app.local_entrypoint main + the recorded Image recipe.
        SCRIPTS = ["modal_train_ptv3", "modal_train_ptv3_hag", "modal_train_randlanet",
                   "modal_train_randlanet_hag", "modal_train_kpconvx_cold",
                   "modal_train_kpconvx_cold_hag"]
        shim_ok, recipe_ok = True, True
        for nm in SCRIPTS:
            _modal_shim.install()
            for m in list(sys.modules):
                if m.startswith("modal_train_"):
                    sys.modules.pop(m, None)
            try:
                mod = importlib.import_module(nm)
            except Exception:  # noqa: BLE001
                shim_ok = False
                continue
            app = _modal_shim.App.last
            img = (app.image if app else None) or getattr(mod, "image", None)
            if app is None or getattr(app, "entrypoint", None) is None:
                shim_ok = False
            if img is None or not getattr(img, "steps", None):
                recipe_ok = False
        check("local: modal shim imports all 6 train scripts + finds main", shim_ok)
        check("local: every script records an Image recipe for Dockerfile gen", recipe_ok)
        # the shim's catch-all __getattr__ must RAISE on dunders, else inspect.getmodule
        # reads modal.__file__ as a function and torch's import blows up (endswith bug).
        _modal_shim.install()
        check("local: shim raises on dunder lookups (inspect.getmodule-safe)",
              not hasattr(sys.modules["modal"], "__file__")
              and sys.modules["modal"].Volume is not None)

        # decoupling: every local_train_X.py imports with NO modal (torch lives in
        # the body) and exposes its train fn with no-op Volume stand-ins.
        LOCAL = ["local_train_ptv3", "local_train_ptv3_hag", "local_train_randlanet",
                 "local_train_randlanet_hag", "local_train_kpconvx_cold",
                 "local_train_kpconvx_cold_hag"]
        local_ok = True
        for nm in LOCAL:
            try:
                lm = importlib.import_module(nm)
            except Exception:  # noqa: BLE001
                local_ok = False
                continue
            fn = next((getattr(lm, n) for n in dir(lm) if n.startswith("train_")), None)
            # Volumes are a Modal concept; the local scripts must carry NO stand-ins.
            if fn is None or hasattr(lm, "outputs_volume") or hasattr(lm, "datasets_volume"):
                local_ok = False
        check("local: all 6 local_train_*.py import + expose train fn + no volume stubs", local_ok)
        # --mode infer labels arbitrary clouds from weights, so it must NOT demand a
        # --dataset (the class count/names come from the checkpoint + its run.json).
        # Calling with dataset=None must fault PAST the dataset gate (missing weights /
        # Docker-only deps), never with the old "--dataset is required" error.
        infer_gate_ok = True
        for nm in LOCAL:
            lm = importlib.import_module(nm)
            fn = next((getattr(lm, n) for n in dir(lm) if n.startswith("train_")), None)
            try:
                fn(dataset=None, mode="infer")   # no weights -> should fault in the infer branch
                infer_gate_ok = False            # must raise (never silently proceed)
            except Exception as e:               # noqa: BLE001
                if "dataset is required" in str(e).lower():
                    infer_gate_ok = False
        check("local: --mode infer runs dataset-free (all 6 backbones)", infer_gate_ok)
        with open(importlib.import_module("local_train_ptv3").__file__, encoding="utf-8") as f:
            _lt_src = f.read()
        check("local: local_train_ptv3.py has no 'import modal' (source is decoupled)",
              "import modal" not in _lt_src and "_NoVol" not in _lt_src
              and "modal" not in _lt_src.lower())
        # inversion: the modal wrapper bakes its local script in + shells out to it.
        _modal_shim.install()
        for m in list(sys.modules):
            if m.startswith("modal_train_"):
                sys.modules.pop(m, None)
        importlib.import_module("modal_train_ptv3")
        _steps = _modal_shim.App.last.image.steps
        check("local: modal wrapper bakes its local_train script into the image",
              any(k == "copy_file" and "local_train_ptv3.py" in p["src"] for k, p in _steps))

        # gen_dockerfiles: recipe -> a buildable Dockerfile (cuda base, ctx, quoting)
        import tools.gen_dockerfiles as gen
        df, contexts = gen.build_dockerfile("ptv3", "modal_train_ptv3.py")
        check("gen: ptv3 Dockerfile uses a CUDA devel base + python bootstrap",
              "FROM nvidia/cuda:" in df and "-devel-ubuntu22.04" in df
              and "ln -sf /usr/bin/python3 /usr/bin/python" in df)
        check("gen: Dockerfile pins a heredoc-capable frontend on line 1",
              df.splitlines()[0] == "# syntax=docker/dockerfile:1")
        check("gen: shell-unsafe pip version specs are quoted",
              "'numpy<2.0'" in df and "'pandas<3'" in df)
        check("gen: model repo COPY uses a build-context matching build_all",
              "COPY --from=ptv3src . /opt/ptv3" in df and "ptv3src" in contexts)
        check("gen: local_train script is NOT baked (bind-mounted at /workspace locally)",
              "local_train" not in df)
        dfr, _ = gen.build_dockerfile("randlanet", "modal_train_randlanet.py")
        check("gen: multi-line run command emitted as a RUN heredoc",
              "RUN <<'TT_EOT'" in dfr and "build_ext --inplace" in dfr)

        # cross-platform app dir: APPDATA overrides everywhere; else native per-OS
        check("appstate: _app_base honors APPDATA on every platform",
              appstate._app_base("linux", {"APPDATA": "/x"}) == Path("/x")
              and appstate._app_base("win32", {"APPDATA": "/x"}) == Path("/x"))
        check("appstate: _app_base uses XDG/.config on Linux, Library on macOS, LOCALAPPDATA on win",
              appstate._app_base("linux", {"XDG_CONFIG_HOME": "/cfg"}) == Path("/cfg")
              and appstate._app_base("linux", {}) == Path.home() / ".config"
              and appstate._app_base("darwin", {}) == Path.home() / "Library" / "Application Support"
              and appstate._app_base("win32", {"LOCALAPPDATA": "/la"}) == Path("/la"))

        # local_cli + appstate exec mode/config, isolated to a throwaway APPDATA
        _old_appdata = os.environ.get("APPDATA")
        os.environ["APPDATA"] = str(tmp / "appdata_local")
        try:
            check("appstate: exec mode defaults to modal", appstate.get_exec_mode() == "modal")
            appstate.set_exec_mode("local")
            check("appstate: exec mode persists to local", appstate.get_exec_mode() == "local")
            check("appstate: selectable_datasets == the saved registry (no builtins)",
                  appstate.selectable_datasets() == appstate.known_datasets())

            # delete_dataset: forget a saved entry AND remove its staged copy on disk
            del_dir = tmp / "to_delete_ds"
            del_dir.mkdir()
            (del_dir / "dataset_meta.json").write_text("{}", encoding="utf-8")
            appstate.remember_dataset("to_delete", {"staged_dir": str(del_dir),
                                                    "uploaded": False})
            check("appstate: dataset registered before delete",
                  "to_delete" in appstate.known_datasets())
            appstate.delete_dataset("to_delete")
            check("appstate: delete_dataset forgets the entry + deletes its staged dir",
                  "to_delete" not in appstate.known_datasets() and not del_dir.exists())

            # Even when on-disk removal fails (locked files on Windows), the entry
            # MUST still be forgotten so the GUI list refreshes — forget happens first.
            import shutil as _sh
            stuck = tmp / "stuck_ds"
            stuck.mkdir()
            appstate.remember_dataset("stuck", {"staged_dir": str(stuck), "uploaded": False})
            _orig_rmtree = _sh.rmtree
            _sh.rmtree = lambda *a, **k: (_ for _ in ()).throw(OSError("locked"))
            try:
                _, err = appstate.delete_dataset("stuck")
            finally:
                _sh.rmtree = _orig_rmtree
            check("appstate: delete_dataset forgets the entry even when rmtree fails",
                  "stuck" not in appstate.known_datasets() and err)

            # Registry defaults to our GHCR org when never set (pulling works out
            # of the box); TT_REGISTRY/env or an explicit value override it.
            _saved_reg = os.environ.pop("TT_REGISTRY", None)
            check("appstate: registry defaults to ghcr.io/gcsgeospatial when unset",
                  appstate.local_config()["registry"] == "ghcr.io/gcsgeospatial")
            if _saved_reg is not None:
                os.environ["TT_REGISTRY"] = _saved_reg
            cfg = appstate.local_config()
            check("appstate: local_config fills datasets/outputs roots + gpus",
                  bool(cfg["datasets_root"]) and bool(cfg["outputs_root"])
                  and isinstance(cfg["images"], dict) and cfg["gpus"] == "all")
            # registry="" here = opt out of the default, so the bare local tag shows.
            appstate.set_local_config({**cfg, "images": {"ptv3": "myimg:1"},
                                       "gpus": "0", "registry": ""})
            check("appstate: local_config overrides round-trip",
                  appstate.local_config()["images"]["ptv3"] == "myimg:1"
                  and appstate.local_config()["gpus"] == "0")

            prog, args = local_cli.run_script(
                "modal_train_ptv3.py", {"dataset": "myds", "grid": 0.05, "epochs": 250},
                BB["ptv3"], repo_root="/repo", gpu="A100")
            joined = " ".join(args)
            check("local_cli: docker run with ipc=host + gpus + workspace/datasets/outputs mounts",
                  args[0] == "run" and "--ipc=host" in args and "--gpus" in args
                  and "/repo:/workspace" in args
                  and ":/datasets" in joined and ":/outputs" in joined)
            check("local_cli: invokes the decoupled scripts/local/local_train_<key>.py with the flags",
                  "scripts/local/local_train_ptv3.py" in args and "local_run.py" not in args
                  and "--dataset" in args and "myds" in args and "--grid" in args)
            _, args_o = local_cli.run_script(
                "modal_train_ptv3.py", {"dataset": "d"}, BB["ptv3"],
                repo_root="/repo", outputs_root="/myout")
            check("local_cli: outputs_root override binds the chosen folder to /outputs",
                  "/myout:/outputs" in " ".join(args_o))
            check("local_cli: image tag override respected (else trainer-local-<key>)",
                  local_cli.image_for(BB["ptv3"]) == "myimg:1"
                  and local_cli.image_for(BB["randlanet"]) == "trainer-local-randlanet")

            # registry distribution: a configured registry makes the default tag
            # pullable (build once, pull anywhere); local-only tags must be built.
            appstate.set_local_config({**appstate.local_config(), "images": {},
                                       "registry": "ghcr.io/acme"})
            check("local_cli: registry prefixes the default image tag + marks it pullable",
                  local_cli.image_for(BB["randlanet"]) == "ghcr.io/acme/trainer-local-randlanet"
                  and local_cli.is_pullable(BB["randlanet"]))
            ok_pull, _ = local_cli.image_preflight(BB["randlanet"])   # absent but pullable
            check("local_cli: missing-but-pullable image is allowed to run (docker auto-pulls)",
                  ok_pull is True)

            # GUI image manager: pull() targets the same tag image_for/run use, and
            # all_statuses reports one row per backbone with the manager's contract
            # (works whether or not docker is installed on this box).
            pprog, pargs = local_cli.pull(BB["randlanet"])
            check("local_cli: pull() = `docker pull <image_for tag>`",
                  pargs == ["pull", local_cli.image_for(BB["randlanet"])])
            st = local_cli.all_statuses()
            check("local_cli: all_statuses has one contract-shaped row per backbone",
                  len(st) == len(BB)
                  and all({"key", "label", "tag", "present", "pullable", "docker"} <= set(s)
                          for s in st)
                  and {s["key"] for s in st} == set(BB))
            appstate.set_local_config({**appstate.local_config(), "registry": ""})
            ok_local, msg_local = local_cli.image_preflight(BB["randlanet"])  # absent, local-only
            check("local_cli: missing local-only image blocks with a build hint",
                  ok_local is False and "build_all" in msg_local
                  and not local_cli.is_pullable(BB["randlanet"]))

            # backbone selection: unset = all; an explicit list filters in local mode only.
            check("appstate: backbones all enabled by default (unset)",
                  appstate.enabled_backbones() is None
                  and appstate.backbone_enabled("ptv3_hag"))
            appstate.set_enabled_backbones(["ptv3", "randlanet", "kpconvx_cold"])
            check("appstate: explicit selection hides the others in local mode",
                  appstate.backbone_enabled("randlanet")
                  and not appstate.backbone_enabled("ptv3_hag"))
            appstate.set_exec_mode("modal")
            check("appstate: selection does NOT filter in modal mode",
                  appstate.backbone_enabled("ptv3_hag"))
            appstate.set_exec_mode("local")
        finally:
            if _old_appdata is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = _old_appdata

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [n for n, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED:", *failed, sep="\n  - ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
