"""Local smoke test for the trainer_gui pipeline — no Modal calls, no GPU.

Run:  python tests/smoke_test.py   (from the trainer_gui/ project dir)

Covers: synthetic LAZ/PLY/ASCII scenes -> canonical conversion (field + companion
label specs), npz contract, meta + recommendations, inference-job conversion,
training-log parsing, viewer loading.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer_gui import analysis, dataset  # noqa: E402
from trainer_gui.dataset import LabelSpec  # noqa: E402

RNG = np.random.default_rng(7)
CHECKS = []


def check(name, cond):
    CHECKS.append((name, bool(cond)))
    print(("  ✓ " if cond else "  ✗ ") + name)


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
    tmp = Path(tempfile.mkdtemp(prefix="trainer_gui_smoke_"))
    print(f"workdir: {tmp}")
    try:
        # ---------------- LAZ dataset with field labels (ASPRS-ish: 0 ignored)
        laz_root = tmp / "laz_src"
        for split, k in (("train", 2), ("val", 1)):
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
        staged = dataset.convert_dataset("laz_demo", str(laz_root / "train"),
                                         str(laz_root / "val"), spec, classes, [0],
                                         tmp / "staging", progress=print)
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
        check("meta: counts populated", meta["classes"][0]["train_count"] > 0)
        check("meta: recommendations for all backbones",
              "ptv3_warm" in meta["recommendations"] and "kpconvx_cold" in meta["recommendations"])
        grid = meta["recommendations"]["ptv3_warm"]["grid"]
        # 20k pts over ~100x100m -> 2 pts/m² -> spacing .7m -> 3x = 2.1 clamped to 0.6
        check(f"meta: ptv3 grid clamped to band (got {grid})", 0.05 <= grid <= 0.6)
        check("meta: octformer got a depth",
              9 <= meta["recommendations"]["octformer_warm"]["octree_depth"] <= 12)

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
        staged_ply = dataset.convert_dataset("ply_demo", str(ply_root / "train"),
                                             str(ply_root / "val"), spec_ply, classes_ply,
                                             [], tmp / "staging")
        zp = np.load(staged_ply / "train" / "tile0.npz")
        check("ply npz: rgb present u8", zp["rgb"].dtype == np.uint8)
        check("ply npz: all labels mapped", set(np.unique(zp["label"])) == {0, 1, 2})

        # ---------------- ASCII dataset with companion label files (IEEE layout)
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
        staged_txt = dataset.convert_dataset("txt_demo", str(txt_root / "train"),
                                             str(txt_root / "val"), spec_txt, classes_txt,
                                             [0], tmp / "staging")
        zt = np.load(staged_txt / "train" / "JAX_train0_PC3.npz")
        check("ascii npz: intensity + return_number captured",
              "intensity" in zt.files and "return_number" in zt.files)

        # ---------------- inference job conversion (no labels)
        job = dataset.convert_infer_job("test_job", str(txt_root / "val"), tmp / "staging")
        zi = np.load(job / "scenes" / "JAX_val0_PC3.npz")
        check("infer npz: no label key", "label" not in zi.files)
        check("infer: job_meta written", (job / "job_meta.json").exists())
        check("infer npz: p95 intensity clipped to [0,2] (matches IEEE training)",
              0.0 <= zi["intensity"].min() and zi["intensity"].max() <= 2.0)

        # ---------------- built-in IEEE / IEEE HAG known datasets
        from trainer_gui import appstate
        from trainer_gui.backbones import BACKBONES
        kd = appstate.known_datasets()
        check("appstate: IEEE + IEEE HAG are built-in known datasets",
              kd.get("IEEE", {}).get("builtin") is True
              and kd.get("IEEE HAG", {}).get("backbones") == ["ptv3_hag", "randlanet_hag"])
        check("appstate: built-in dataset backbones all exist + are trainable",
              all(k in BACKBONES and BACKBONES[k].ready
                  for ds in ("IEEE", "IEEE HAG") for k in kd[ds]["backbones"]))

        # intensity normalization modes differ: p95 scales hotter than max
        one = dataset.discover_scenes(txt_root / "val")[0]
        dataset.convert_scene(one, None, {}, tmp / "imax.npz", intensity_norm="max")
        dataset.convert_scene(one, None, {}, tmp / "ip95.npz", intensity_norm="p95")
        imax, ip95 = np.load(tmp / "imax.npz")["intensity"], np.load(tmp / "ip95.npz")["intensity"]
        check("convert: max-norm intensity in [0,1]", imax.max() <= 1.0 + 1e-6)
        check("convert: p95-norm intensity >= max-norm and <= 2",
              ip95.max() >= imax.max() and ip95.max() <= 2.0)

        # ---------------- backbones: infer-readiness contracts
        from trainer_gui.backbones import BACKBONES, infer_backbones
        check("backbones: folder-infer set = the 6 IEEE cold/hag scripts",
              set(infer_backbones()) == {"ptv3", "randlanet", "ptv3_hag", "randlanet_hag",
                                         "kpconvx_cold", "kpconvx_cold_hag"})
        check("backbones: randlanet uses --sub-grid and has no --chunk-xy",
              BACKBONES["randlanet"].grid_flag == "sub-grid"
              and not BACKBONES["randlanet"].has_chunk)
        check("backbones: ptv3 uses --grid and has --chunk-xy",
              BACKBONES["ptv3"].grid_flag == "grid" and BACKBONES["ptv3"].has_chunk)
        check("backbones: kpconvx is folder-inferable but not canonical-trainable",
              BACKBONES["kpconvx_cold"].folder_infer and not BACKBONES["kpconvx_cold"].ready
              and BACKBONES["kpconvx_cold_hag"].folder_infer)
        check("backbones: IEEE outputs volumes (not stale stpls3d)",
              BACKBONES["ptv3"].outputs_volume == "ptv3-ieee-outputs"
              and BACKBONES["randlanet"].outputs_volume == "randlanet-cold-ieee-outputs"
              and BACKBONES["ptv3_hag"].outputs_volume == "ptv3-ieee-hag-outputs")

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
            {"backbone": "PTv3", "dataset": "IEEE", "class_names": ["Ground", "Trees"]}),
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

        # ---------------- local prep caches (must match the scripts' layouts)
        from trainer_gui import prep

        # ptv3_warm: tiles with intensity/ret_num keys, three splits, seed-42 holdout
        pd = prep.prep_dataset("ptv3_warm", staged, {"chunk-xy": 50.0}, progress=print)
        check("prep ptv3_warm: dir name", pd.name == "ptv3_warm_chunk50")
        train_tiles = sorted((pd / "train").glob("*.npz"))
        val_tiles = sorted((pd / "val").glob("*.npz"))
        test_tiles = sorted((pd / "test").glob("*.npz"))
        check("prep ptv3_warm: tiles in all splits",
              train_tiles and val_tiles and test_tiles)
        zt = np.load(train_tiles[0])
        check("prep ptv3_warm: tile keys",
              set(zt.files) == {"xyz", "intensity", "ret_num", "lab"})
        # the holdout must match the scripts' _split_scenes (seed 42, 1/5 capped at 10)
        names = sorted(p.stem for p in (staged / "train").glob("*.npz"))
        rng = np.random.RandomState(42)
        idx = np.arange(len(names))
        rng.shuffle(idx)
        n_hold = min(10, max(1, len(names) // 5))
        expect_val = sorted(names[i] for i in idx[:n_hold])
        got_val = sorted({p.stem.rsplit("_x", 1)[0] for p in val_tiles})
        check("prep ptv3_warm: holdout split matches script seed", got_val == expect_val)

        # randlanet_warm: whole-scene subsample, fewer points than input
        pr = prep.prep_dataset("randlanet_warm", staged, {"sub-grid": 0.3})
        check("prep randlanet_warm: dir name", pr.name == "randlanet_warm_grid30")
        zr = np.load(next((pr / "train").glob("*.npz")))
        zin = np.load(next((staged / "train").glob("*.npz")))
        check("prep randlanet_warm: keys + subsampled",
              set(zr.files) == {"xyz", "intensity", "ret_num", "lab"}
              and len(zr["xyz"]) < len(zin["xyz"]))

        # octformer (cold): rgb + normals tiles, two splits only
        # (50 m chunks — the synthetic scenes are too sparse for 25 m tiles to
        # clear the 2048-point minimum, exactly as the remote script would skip)
        po = prep.prep_dataset("octformer", staged, {"chunk-xy": 50.0})
        check("prep octformer cold: dir name", po.name == "octformer_cold_chunk50")
        check("prep octformer cold: no val split (two-dir layout)",
              not (po / "val").exists())
        zo = np.load(next((po / "train").glob("*.npz")))
        check("prep octformer cold: keys incl. normals",
              set(zo.files) == {"xyz", "rgb", "nrm", "lab"})
        nrm = zo["nrm"]
        check("prep octformer cold: normals unit-ish",
              np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-4))

        # kpconvx_warm: 30 m tiles, min 1024 pts
        pk = prep.prep_dataset("kpconvx_warm", staged, {"chunk-xy": 30.0})
        check("prep kpconvx_warm: dir name", pk.name == "kpconvx_warm_chunk30")
        check("prep kpconvx_warm: tiles exist", any((pk / "train").glob("*.npz")))

        # idempotency: second run does no extra work and keeps tile counts
        n_before = len(list((pd / "train").glob("*.npz")))
        prep.prep_dataset("ptv3_warm", staged, {"chunk-xy": 50.0})
        check("prep: idempotent re-run",
              len(list((pd / "train").glob("*.npz"))) == n_before)

        check("prep: kpconvx_cold correctly unsupported",
              not prep.supports_local_prep("kpconvx_cold"))

        # ---------------- Pretraining: tile_for_model (no PDAL needed)
        from trainer_gui import pretrain

        pt_out = tmp / "pretrain_tile"
        prep_dir = pretrain.tile_for_model(str(laz_root / "train"), pt_out,
                                           "ptv3_warm", {"chunk-xy": 50.0}, progress=print)
        check("pretrain tile: prep dir tag", prep_dir.name == "ptv3_warm_chunk50")
        check("pretrain tile: staged under out/_staged",
              (pt_out / "_staged" / "train").exists())
        pt_tiles = sorted((prep_dir / "train").glob("*.npz"))
        check("pretrain tile: train tiles produced", len(pt_tiles) > 0)
        zpt = np.load(pt_tiles[0])
        check("pretrain tile: tile keys match prep contract",
              set(zpt.files) == {"xyz", "intensity", "ret_num", "lab"})

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
            summary = pretrain.add_hag(str(hag_src), hag_out, skip_ground=True,
                                       hag_filter="hag_nn", progress=print)
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
            pretrain.add_hag(str(txt_hag), txt_out, skip_ground=False,
                             hag_filter="hag_nn", progress=print)
            check("pretrain hag: txt input -> laz", (txt_out / "scene_t_PC3.laz").exists())
            zt_hag = json.loads((txt_out / "scene_t_PC3.json").read_text())
            check("pretrain hag: txt sidecar HAG spans ground..elevated",
                  zt_hag["hag_max"] > 2.0)
        else:
            print("  - pretrain hag: skipped (pdal not installed)")

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
        lbl_path = tmp / "scene_label.npz"
        lab = np.array([-1, 0, 1, 2, 0], np.int32)
        np.savez_compressed(lbl_path, xyz=make_xyz(5).astype(np.float32), label=lab)
        _, rgb_l, key_l = _load(lbl_path)
        from trainer_gui.palette import IEEE_PALETTE
        check("viewer: label npz colours by IEEE class, -1 stays grey",
              rgb_l is not None
              and np.allclose(rgb_l[1], IEEE_PALETTE[0] / 255.0)
              and np.allclose(rgb_l[0], 0.55)               # ignore -> grey
              and [lab for lab, _ in key_l] == ["Ground", "Trees", "Building"])

        # ---------------- existing repo prediction as a viewer fixture
        jax = Path(__file__).resolve().parents[2] / "JAX_066_pred.ply"
        if jax.exists():
            xyz2, rgb2, _ = _load(jax)
            check("viewer: JAX_066_pred.ply loads with colors",
                  len(xyz2) > 0 and rgb2 is not None)

        # ---------------- ground-truth comparison (error map)
        from trainer_gui import palette
        from trainer_gui.viewer import compare_clouds, error_colors
        cls = np.array([0, 1, 2, 3, 4, 0, 2], np.int64)
        check("palette: class_from_rgb inverts IEEE_PALETTE colors",
              list(palette.class_from_rgb(palette.IEEE_PALETTE[cls])) == list(cls))
        check("palette: asprs_to_index maps codes + 0->unlabeled",
              list(palette.asprs_to_index(np.array([2, 5, 6, 9, 17, 0]))) == [0, 1, 2, 3, 4, -1])
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
        # end-to-end: synthetic IEEE prediction PLY (palette-coloured) + GT CLS.txt
        cmp = tmp / "cmp"
        cmp.mkdir()
        ply_cls = np.array([0, 1, 2, 3, 4, 0, 2], np.int64)
        arr = np.column_stack([make_xyz(7).astype(np.float32), palette.IEEE_PALETTE[ply_cls]])
        hdr = ("ply\nformat ascii 1.0\nelement vertex 7\nproperty float x\nproperty float y\n"
               "property float z\nproperty uchar red\nproperty uchar green\n"
               "property uchar blue\nend_header")
        np.savetxt(cmp / "JAX_900_pred.ply", arr, fmt=["%.3f"] * 3 + ["%d"] * 3,
                   header=hdr, comments="")
        np.savetxt(cmp / "JAX_900_CLS.txt", np.array([2, 5, 6, 9, 17, 2, 5]), fmt="%d")
        _, vc, ckey = compare_clouds(cmp / "JAX_900_pred.ply", cmp)   # only pt 6 (pred 2 vs GT 1) differs
        n_yellow = int((np.abs(vc - np.array([1.0, 1.0, 0.0])).sum(1) < 1e-9).sum())
        check("viewer: compare_clouds flags exactly the mismatched point yellow", n_yellow == 1)
        # GT can also be a class-coloured .ply (e.g. make_groundtruth_ply output)
        gt_cls = palette.asprs_to_index(np.array([2, 5, 6, 9, 17, 2, 5]))   # -> [0,1,2,3,4,0,1]
        np.savetxt(cmp / "JAX_900_gt.ply",
                   np.column_stack([make_xyz(7).astype(np.float32), palette.IEEE_PALETTE[gt_cls]]),
                   fmt=["%.3f"] * 3 + ["%d"] * 3, header=hdr, comments="")
        _, vc_ply, _ = compare_clouds(cmp / "JAX_900_pred.ply", cmp / "JAX_900_gt.ply")
        check("viewer: compare accepts a class-coloured .ply ground truth",
              int((np.abs(vc_ply - np.array([1.0, 1.0, 0.0])).sum(1) < 1e-9).sum()) == 1)
        check("viewer: compare key is just wrong (yellow) + correct (grey)",
              ckey == [("wrong prediction", (1.0, 1.0, 0.0)), ("correct", (0.55, 0.55, 0.55))])
        from trainer_gui.viewer import _ieee_key
        check("viewer: _ieee_key lists all 5 classes (Water + Bridge included)",
              [n for n, _ in _ieee_key()] == ["Ground", "Trees", "Building", "Water", "Bridge"])

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
