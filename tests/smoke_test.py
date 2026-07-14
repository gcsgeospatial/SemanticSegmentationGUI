"""Local smoke test for the trainer_gui pipeline — no Modal calls, no GPU.

Run:  python tests/smoke_test.py   (from the trainer_gui/ project dir)

Covers: synthetic LAZ/PLY/ASCII scenes -> canonical conversion (field + companion
label specs), npz contract, meta + recommendations, inference-job conversion
(incl. opt-in HAG), density-generalization manifest round-trip, training-log
parsing, ground-truth comparison stats, Dockerfile generation, local Docker backend.

Every check here guards code the shipped app reaches. If a function's only caller
is this file, delete the function — don't add a check for it.
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


def write_laz(path, xyz, labels, intensity, extra=None):
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
    for name, vals in (extra or {}).items():   # float Extra Bytes dims
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
        las[name] = vals.astype(np.float32)
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
        check("laz npz: xyz f64 (N,3)", z["xyz"].dtype == np.float64 and z["xyz"].shape[1] == 3)
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
        # 20k pts over ~100x100m -> 2 pts/m² -> spacing .71m -> 1.25x = 0.88 (in band)
        check(f"meta: ptv3 grid clamped to band (got {grid})", 0.10 <= grid <= 2.0)
        check("meta: kpconvx grid clamped to its band",
              0.4 <= meta["recommendations"]["kpconvx_cold"]["grid"] <= 2.0)

        # ---------------- analysis.recommend: aerial density sweep 0.5 -> 1000 pts/m2.
        # Occupancy o = rho*g^2 must stay >= 1 (the density-invariant band, see
        # scripts/DENSITY_DG.md), grids shrink monotonically as density grows, and
        # every value stays inside its UI band so the Train page never clamps.
        # The whole sweep is folded into two checks — one failing (rho, backbone)
        # is named in the message, so per-cell checks bought nothing but noise.
        DENSITIES = (0.5, 2.0, 10.0, 50.0, 100.0, 250.0, 1000.0)
        recs = {rho: analysis.recommend({"stats": {"mean_pts_per_m2": rho,
                                                   "mean_spacing_m": rho ** -0.5}})
                for rho in DENSITIES}
        def _in_band(rho, k, rec):
            dflt = next(int(p.default) for p in analysis.BACKBONES[k].params
                        if p.flag == "batch")
            return (rho * rec["grid"] ** 2 >= 0.99 and 1 <= rec["batch"] <= dflt
                    and (10 <= rec["chunk_xy"] <= 200 if "chunk_xy" in rec else True))

        bad = [f"rho={rho} {k}" for rho, r in recs.items() for k, rec in r.items()
               if not _in_band(rho, k, rec)]
        check("recommend: o>=1, batch<=default, chunk in band across the sweep"
              + (" - bad: " + ", ".join(bad) if bad else ""), not bad)
        check("recommend: grids shrink monotonically with density",
              all(recs[hi][k]["grid"] <= recs[lo][k]["grid"] + 1e-9
                  for lo, hi in zip(DENSITIES, DENSITIES[1:]) for k in recs[hi]))
        r05 = analysis.recommend({"stats": {"mean_pts_per_m2": 0.5,
                                            "mean_spacing_m": 0.5 ** -0.5}})
        check("recommend: kpconvx reproduces the proven sparse recipe (g=2.0, chunk=100)",
              r05["kpconvx_cold"]["grid"] == 2.0 and r05["kpconvx_cold"]["chunk_xy"] == 100)
        r100 = analysis.recommend({"stats": {"mean_pts_per_m2": 100.0,
                                             "mean_spacing_m": 0.1}})
        check("recommend: randlanet num_points adapts with density (pyramid-friendly)",
              r05["randlanet"]["num_points"] == 8192
              and r100["randlanet"]["num_points"] == 45056
              and r05["randlanet"]["num_points"] % 4096 == 0)
        check("recommend: ptv3 tiles stay clear of the script's 80k train crop",
              all(analysis.recommend({"stats": {"mean_pts_per_m2": d,
                                                "mean_spacing_m": d ** -0.5}}
                                     )["ptv3"]["chunk_xy"] ** 2 * d <= 80_000
                  for d in (2.0, 10.0, 50.0)))
        dg_wide = analysis.dg_recommend(1000.0, 2.0)   # UAV-trained, QL2 inference
        # need comes from the grid the model ACTUALLY trains at (ptv3 pinned at
        # 0.15 by the crop), not the raw density ratio: 1/(0.15*sqrt(2)) ~ 4.7
        check("dg: dense-trained gap sized to the real (pinned) grid, not the ratio",
              4.5 <= dg_wide["coarsen_max"] <= 5.0 and dg_wide["p_native"] == 0.35
              and dg_wide["tta"] == 4 and dg_wide["logdk"]
              and "exceeds the aug range" not in dg_wide["rationale"])
        dg_over = analysis.dg_recommend(50.0, 0.5)     # beyond the aug range
        check("dg: gap beyond the aug range caps at 6 + retrain advice",
              dg_over["coarsen_max"] == 6.0
              and "exceeds the aug range" in dg_over["rationale"])
        dg_mid = analysis.dg_recommend(8.0, 2.0)       # QL1 -> QL2: modest 4x
        check("dg: modest gap stays in the aug range",
              1.5 <= dg_mid["coarsen_max"] <= 2.0
              and "exceeds the aug range" not in dg_mid["rationale"])
        check("dg: no aug when the pinned training grid already covers the target",
              analysis.dg_recommend(100.0, 50.0)["density_aug"] is False)

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
        wc = dataset._worker_cap_detail(list(sc_src.glob("*.laz")))[0]
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
        from trainer_gui.backbones import BACKBONES

        # intensity normalization modes differ: p95 scales hotter than max
        one = dataset.discover_scenes(txt_root / "val")[0]
        dataset.convert_scene(one, None, {}, tmp / "imax.npz", intensity_norm="max")
        dataset.convert_scene(one, None, {}, tmp / "ip95.npz", intensity_norm="p95")
        imax, ip95 = np.load(tmp / "imax.npz")["intensity"], np.load(tmp / "ip95.npz")["intensity"]
        check("convert: max-norm intensity in [0,1]", imax.max() <= 1.0 + 1e-6)
        check("convert: p95-norm intensity >= max-norm and <= 2",
              ip95.max() >= imax.max() and ip95.max() <= 2.0)

        # ---------------- backbones: registry contracts
        check("backbones: registry = the 4 cold scripts + 3 pcssl encoders",
              set(BACKBONES) == {"ptv3", "randlanet", "kpconvx_cold", "kpconv",
                                 "concerto", "sonata", "utonia"})
        # pcssl family: one shared trainer; the sonata/utonia wrappers must
        # override the core's PKG/HF/BB_KEY constants per CALL, never at
        # import (several entry points share one process here).
        import local_train_concerto as _pc
        import local_train_sonata as _ps
        import local_train_utonia as _pu
        check("backbones: pcssl wrappers don't clobber the shared core at import",
              _pc.BB_KEY == "concerto" and _pc.HF_NAME == "concerto_base"
              and _ps._CFG["BB_KEY"] == "sonata"
              and _pu._CFG["HF_REPO"] == "Pointcept/Utonia")
        check("backbones: pcssl entries expose --freeze-encoder + --grid + --chunk-xy",
              all(any(p.flag == "freeze-encoder" for p in BACKBONES[k].params)
                  and BACKBONES[k].grid_flag == "grid" and BACKBONES[k].has_chunk
                  for k in ("concerto", "sonata", "utonia")))
        check("backbones: randlanet uses --sub-grid and has no --chunk-xy",
              BACKBONES["randlanet"].grid_flag == "sub-grid"
              and not BACKBONES["randlanet"].has_chunk)
        check("backbones: ptv3 uses --grid and has --chunk-xy",
              BACKBONES["ptv3"].grid_flag == "grid" and BACKBONES["ptv3"].has_chunk)
        check("backbones: outputs volumes follow the renamed apps",
              BACKBONES["ptv3"].outputs_volume == "ptv3-outputs"
              and BACKBONES["randlanet"].outputs_volume == "randlanet-cold-outputs"
              and BACKBONES["kpconvx_cold"].outputs_volume == "kpconvx-cold-outputs")
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

        # ---------------- graceful stop: the /outputs/STOP sentinel contract
        # (GUI touches it, trainer consumes it at epoch end and breaks into the
        # normal final-eval + finalize path).
        _orig_sentinel = train_common.STOP_SENTINEL
        stop_f = tmp / "STOP"
        train_common.STOP_SENTINEL = str(stop_f)
        try:
            check("stop: no sentinel -> not requested", not train_common.stop_requested(3))
            stop_f.touch()
            train_common.clear_stop()
            check("stop: clear_stop removes a stale sentinel", not stop_f.exists())
            stop_f.touch()
            check("stop: sentinel -> requested AND consumed",
                  train_common.stop_requested(3) and not stop_f.exists())
        finally:
            train_common.STOP_SENTINEL = _orig_sentinel

        # ---------------- class masking: EXCLUDE_CLASSES parse + prob renorm
        mask_names = ["ground", "veg", "building"]
        os.environ["EXCLUDE_CLASSES"] = "veg"
        try:
            check("mask: name csv -> index list",
                  train_common.exclude_class_idx(mask_names) == [1])
            os.environ["EXCLUDE_CLASSES"] = "bogus"
            try:
                train_common.exclude_class_idx(mask_names)
                bad_raised = False
            except ValueError:
                bad_raised = True
            check("mask: unknown name fails loudly", bad_raised)
            os.environ["EXCLUDE_CLASSES"] = "ground,veg,building"
            try:
                train_common.exclude_class_idx(mask_names)
                all_raised = False
            except ValueError:
                all_raised = True
            check("mask: excluding every class fails loudly", all_raised)
        finally:
            os.environ.pop("EXCLUDE_CLASSES", None)
        check("mask: unset env -> no exclusions",
              train_common.exclude_class_idx(mask_names) == [])
        mp = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]], np.float32)
        mm = train_common.apply_class_mask(mp.copy(), [0])
        check("mask: excluded column zeroed, rows renormalized, argmax = next best",
              np.all(mm[:, 0] == 0.0)
              and np.allclose(mm.sum(1), 1.0, atol=1e-6)
              and list(mm.argmax(1)) == [1, 1]
              and np.allclose(mm.max(1), [0.6, 0.6 / 0.9], atol=1e-6))
        check("mask: empty exclusion is identity",
              np.array_equal(train_common.apply_class_mask(mp.copy(), []), mp))
        import inspect
        check("mask: kp_make_predict_points accepts exclude_idx",
              "exclude_idx" in inspect.signature(train_common.kp_make_predict_points).parameters)

        # Train page's Modal-presence check parses `modal volume ls --json` entries
        # whose basename key varies by CLI version — lock the parser.
        from trainer_gui.pages.infer_page import _entry_name, _localize_paths, _parse_run_ref
        # The Inference run box accepts the train log's copy string verbatim
        # (<volume>/runs/<id>), plus every older shape that already worked.
        check("infer: _parse_run_ref handles pasted volume paths and bare ids",
              _parse_run_ref("ptv3-hag-outputs/runs/20260709_195750_ieee-test_ptv3_hag")
              == ("ptv3-hag-outputs", "20260709_195750_ieee-test_ptv3_hag")
              and _parse_run_ref("runs/20260709_1957") == ("", "20260709_1957")
              and _parse_run_ref("20260709_1957  (ptv3_hag)") == ("", "20260709_1957")
              and _parse_run_ref("/vol/runs/id/") == ("vol", "id")
              and _parse_run_ref("") == ("", ""))
        # Modal CLI removed `volume get -f` (only --force remains); `put` keeps -f.
        from trainer_gui import modal_cli
        check("modal_cli: volume get spells out --force (short -f was removed)",
              "--force" in modal_cli.volume_get("v", "runs/x", "d")[1]
              and "-f" not in modal_cli.volume_get("v", "runs/x", "d")[1])
        from trainer_gui.pages.infer_page import _manifest_in
        mdir = tmp / "dl_run"
        mdir.mkdir()
        (mdir / "run.json").write_text('{"backbone": "ptv3"}', encoding="utf-8")
        check("infer: _manifest_in reads run.json from a downloaded run folder",
              _manifest_in(mdir)["backbone"] == "ptv3"
              and _manifest_in(tmp / "no_such_run") is None)
        # Post-conversion channel report: flags scenes missing intensity, BLOCKS
        # on a missing feat_hag (an ordinary hard feat_* requirement) with the
        # HAG-box hint — the scene npz is the model's literal input.
        from trainer_gui.pages.infer_page import _scene_channel_report
        cdir = tmp / "chan_job" / "scenes"
        cdir.mkdir(parents=True)
        np.savez(cdir / "with_i.npz", xyz=make_xyz(5),
                 intensity=np.linspace(0, 1, 5, dtype=np.float32),
                 feat_hag=np.ones(5, np.float32))
        np.savez(cdir / "no_i.npz", xyz=make_xyz(5))
        rep_lines, rep_block = _scene_channel_report(
            tmp / "chan_job", features=["intensity", "feat_hag"])
        rep = "\n".join(rep_lines)
        check("infer: channel report flags missing intensity + blocks on feat_hag",
              "no_i" in rep and "with_i" not in rep.split("intensity channel in:")[1].split("\n")[0]
              and "feat_hag" in rep and "Compute Height-Above-Ground" in rep
              and rep_block is True)
        hag_ok_lines, hag_ok_block = _scene_channel_report(
            tmp / "chan_job", features=["intensity"])
        check("infer: channel report passes when feat_hag isn't in the spec",
              hag_ok_block is False
              and not any("feat_hag" in s for s in hag_ok_lines))

        # ---------------- feature channels (Phase 2): user-picked extra dims ride
        # as feat_<name> npz keys, meta records the raw-field mapping, inference
        # bakes them, and the channel report BLOCKS scenes missing a required one.
        feat_root = tmp / "feat_src"
        for split, k in (("train", 2), ("val", 1)):
            d = feat_root / split
            d.mkdir(parents=True)
            for i in range(k):
                xyz = make_xyz(4_000)
                labels = RNG.choice([0, 2, 5, 6], len(xyz))
                write_laz(d / f"f{i}.laz", xyz, labels, RNG.uniform(0, 4000, len(xyz)),
                          extra={"eig_lin": RNG.uniform(0, 1, len(xyz))})
        staged_ft = dataset.convert_dataset("feat_demo", str(feat_root / "train"), spec,
                                            classes, [0], tmp / "staging",
                                            val_inputs=[str(feat_root / "val")],
                                            feature_fields=["eig_lin"])
        zft = np.load(sorted((staged_ft / "train").glob("*.npz"))[0])
        check("feat: staged npz carries feat_eig_lin f32 within [-2,2]",
              "feat_eig_lin" in zft.files and zft["feat_eig_lin"].dtype == np.float32
              and zft["feat_eig_lin"].min() >= -2.0 and zft["feat_eig_lin"].max() <= 2.0)
        mft = json.loads((staged_ft / "dataset_meta.json").read_text())
        check("feat: meta source.feature_channels records name + source_field + norm",
              mft["source"]["feature_channels"]
              == [{"name": "eig_lin", "source_field": "eig_lin", "norm": "p95abs"}])
        job_ft = dataset.convert_infer_job("feat_job", str(feat_root / "val"),
                                           tmp / "staging", feature_fields=["eig_lin"])
        zjf = np.load(next((job_ft / "scenes").glob("*.npz")))
        check("feat: convert_infer_job bakes feat_eig_lin, still no label key",
              "feat_eig_lin" in zjf.files and "label" not in zjf.files)
        try:
            dataset.convert_infer_job("feat_missing", str(laz_root / "val"),
                                      tmp / "staging", feature_fields=["eig_lin"])
            feat_raised = False
        except ValueError:
            feat_raised = True
        check("feat: requesting a source field the inputs lack raises ValueError",
              feat_raised)
        FSPEC = ["x", "y", "z", "intensity", "feat_eig_lin"]
        ok_lines, ok_block = _scene_channel_report(job_ft, features=FSPEC)
        check("feat: channel report passes scenes carrying the required channel",
              ok_block is False and not any("missing" in s and "feat_eig_lin" in s
                                            for s in ok_lines))
        bad_lines, bad_block = _scene_channel_report(tmp / "chan_job", features=FSPEC)
        check("feat: channel report blocks + names scenes missing feat_eig_lin",
              bad_block is True
              and any("feat_eig_lin" in s and "no_i" in s and "with_i" in s
                      for s in bad_lines))
        check("feat: parse_feat_spec resolves an ordered csv spec",
              train_common.parse_feat_spec("intensity,feat_eig_lin", [])
              == ["intensity", "feat_eig_lin"])
        try:
            train_common.parse_feat_spec("bogus", [])
            pfs_raised = False
        except ValueError:
            pfs_raised = True
        check("feat: parse_feat_spec raises on an unknown channel name", pfs_raised)

        # ---------------- computed geometric channels (jakteristics): engine
        # numerics, meta record, single-cloud ferry, infer recompute + gating.
        from trainer_gui import pretrain
        gp = np.mgrid[0:40, 0:40].reshape(2, -1).T.astype(np.float64) * 0.4
        geo_plane = np.c_[gp, np.zeros(len(gp))]              # 16x16 m flat plane
        geo_wall = np.c_[np.full(len(gp), 30.0), gp]          # vertical plane at x=30
        geo_xyz = np.vstack([geo_plane, geo_wall])
        g1 = pretrain.geo_features_for_cloud(
            geo_xyz, ["planarity", "linearity", "verticality", "PCA1"], 1.0)
        g2 = pretrain.geo_features_for_cloud(
            geo_xyz, ["planarity", "linearity", "verticality", "PCA1"], 1.0)
        n_pl = len(geo_plane)
        check("geo: engine deterministic, float32, finite",
              all(np.array_equal(g1[k], g2[k]) for k in g1)
              and all(v.dtype == np.float32 and np.isfinite(v).all()
                      for v in g1.values()))
        check("geo: plane reads planar+horizontal, wall reads vertical",
              g1["planarity"][:n_pl].mean() > 0.8
              and g1["verticality"][:n_pl].mean() < 0.2
              and g1["verticality"][n_pl:].mean() > 0.8)
        try:
            pretrain.geo_features_for_cloud(geo_xyz, ["bogus"], 1.0)
            geo_raised = False
        except ValueError:
            geo_raised = True
        check("geo: unknown feature name raises ValueError", geo_raised)

        staged_geo = dataset.convert_dataset(
            "geo_demo", str(feat_root / "train"), spec, classes, [0], tmp / "staging",
            val_inputs=[str(feat_root / "val")], feature_fields=["eig_lin"],
            geo_features=["planarity", "PCA1"], geo_radius=1.5)
        zg = np.load(sorted((staged_geo / "train").glob("*.npz"))[0])
        check("geo: staged npz carries feat_geo_planarity + feat_geo_pca1 (f32, finite)",
              all(k in zg.files and zg[k].dtype == np.float32
                  and np.isfinite(zg[k]).all()
                  for k in ("feat_geo_planarity", "feat_geo_pca1")))
        mg = json.loads((staged_geo / "dataset_meta.json").read_text())
        check("geo: meta feature_channels records @geo: sentinel + raw norm + radius",
              mg["source"]["feature_channels"]
              == [{"name": "eig_lin", "source_field": "eig_lin", "norm": "p95abs"},
                  {"name": "geo_planarity", "source_field": "@geo:planarity",
                   "norm": "raw", "radius": 1.5},
                  {"name": "geo_pca1", "source_field": "@geo:PCA1",
                   "norm": "raw", "radius": 1.5}])

        # Single-cloud split: values must be the WHOLE-cloud computation ferried
        # into the tiles (exact float match), not per-split recomputes.
        staged_gsp = dataset.convert_dataset(
            "geo_spatial", str(laz_root / "train" / "scene0.laz"), spec, classes, [0],
            tmp / "staging", split=dataset.SplitConfig(val_frac=0.3, test_frac=0.3,
                                                       mode="random", seed=42,
                                                       seam_buffer_m=0.0, tile_m=20.0),
            geo_features=["planarity"])
        cloud_g = dataset.read_points(laz_root / "train" / "scene0.laz")
        whole = pretrain.geo_features_for_cloud(cloud_g.xyz, ["planarity"], 1.0)["planarity"]
        by_xy = {tuple(np.round(p, 3)): v for p, v in zip(cloud_g.xyz, whole)}
        ferry_ok = True
        for sp_name in ("train", "val", "test"):
            zs = np.load(next((staged_gsp / sp_name).glob("*.npz")))
            for p, v in zip(zs["xyz"], zs["feat_geo_planarity"]):
                if by_xy.get(tuple(np.round(np.asarray(p, np.float64), 3))) != v:
                    ferry_ok = False
                    break
        check("geo: single-cloud split ferries whole-cloud values exactly", ferry_ok)

        job_geo = dataset.convert_infer_job("geo_job", str(feat_root / "val"),
                                            tmp / "staging",
                                            geo_features=["planarity"], geo_radius=1.0)
        zgj = np.load(next((job_geo / "scenes").glob("*.npz")))
        check("geo: convert_infer_job recomputes feat_geo_planarity, no label key",
              "feat_geo_planarity" in zgj.files and "label" not in zgj.files)
        GSPEC = ["x", "y", "z", "feat_geo_planarity"]
        gok_lines, gok_block = _scene_channel_report(job_geo, features=GSPEC)
        check("geo: channel report passes a job carrying the computed channel",
              gok_block is False)
        gbad_lines, gbad_block = _scene_channel_report(tmp / "chan_job", features=GSPEC)
        check("geo: channel report blocks a job missing the computed channel",
              gbad_block is True and any("feat_geo_planarity" in s for s in gbad_lines))
        check("geo: parse_feat_spec accepts feat_geo_* names",
              train_common.parse_feat_spec("intensity,feat_geo_pca1", [])
              == ["intensity", "feat_geo_pca1"])
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
        # The scripts' *_pred.npz (xyz + class, straight from inference) -> chosen
        # format carrying xyz + classification; the inferred data is transformed
        # directly into the target type (no coloured PLY); the source npz is KEPT
        # (raw indices + confidence) so a new threshold re-exports without inference.
        exd = tmp / "pred_export"
        exd.mkdir()
        ex_cls = np.array([0, 1, 2, 3, 4, 2, 0], np.int64)
        ex_xyz = make_xyz(7)

        def _mk_pred():
            # scripts now write inferred data as a compact npz (xyz + class); export
            # writes the chosen type straight from it — no intermediate coloured PLY.
            np.savez(exd / "sceneX_pred.npz", xyz=ex_xyz.astype(np.float32),
                     classification=ex_cls.astype(np.int32))

        import laspy
        _mk_pred()
        las_out = dataset.export_predictions(exd, "las")[0]
        check("export: las carries the inferred classes, source npz kept for re-export",
              las_out.suffix == ".las"
              and list(laspy.read(str(las_out)).classification) == list(ex_cls)
              and (exd / "sceneX_pred.npz").exists())
        csv_out = dataset.export_predictions(exd, "csv")[0]
        rows = np.genfromtxt(csv_out, delimiter=",", names=True)
        check("export: csv has an x,y,z,classification header + the classes",
              set(rows.dtype.names) == {"x", "y", "z", "classification"}
              and list(rows["classification"].astype(int)) == list(ex_cls))
        from plyfile import PlyData
        ply_out = dataset.export_predictions(exd, "ply")[0]
        pv = PlyData.read(str(ply_out))["vertex"]
        check("export: ply carries only xyz + a classification column",
              {p.name for p in pv.properties} == {"x", "y", "z", "classification"}
              and list(np.asarray(pv["classification"]).astype(int)) == list(ex_cls))
        (exd / "sceneX_pred.npz").unlink()

        # write_pred (Phase 0): confidence + probs ride in the npz with the
        # dtype/shape contract the export threshold below depends on.
        train_common.write_pred(exd / "sceneC_pred.npz", ex_xyz, ex_cls,
                                confidence=np.linspace(0.1, 1.0, 7, dtype=np.float32),
                                probs=RNG.random((7, 5)).astype(np.float16))
        with np.load(exd / "sceneC_pred.npz") as zc:
            check("export: write_pred carries confidence f32 (N,) + probs f16 (N,C)",
                  zc["confidence"].dtype == np.float32 and zc["confidence"].shape == (7,)
                  and zc["probs"].dtype == np.float16 and zc["probs"].shape == (7, 5))
        (exd / "sceneC_pred.npz").unlink()

        # confidence threshold + class remap at export: model indices map to the
        # dataset's source values; points under the cut become ASPRS class 1
        # (Unclassified); confidence rides into the LAS as an Extra Bytes dim.
        from trainer_gui import readers
        np.savez(exd / "sceneT_pred.npz", xyz=make_xyz(3).astype(np.float32),
                 classification=np.array([0, 1, 2], np.int32),
                 confidence=np.array([0.9, 0.3, 0.8], np.float32))
        thr_out = dataset.export_predictions(exd, "las", class_map={0: 10, 1: 11, 2: 12},
                                             unclass_threshold=0.5)[0]
        tcloud = readers.read_points(thr_out)
        check("export: remap + threshold -> low-confidence point exports as class 1",
              list(tcloud.fields["classification"].astype(int)) == [10, 1, 12])
        check("export: confidence survives the LAS round-trip (extra-bytes dim)",
              np.allclose(tcloud.fields["confidence"], [0.9, 0.3, 0.8], atol=1e-6))
        thr0_out = dataset.export_predictions(exd, "las", class_map={0: 10, 1: 11, 2: 12},
                                              unclass_threshold=0.0)[0]
        c0 = readers.read_points(thr0_out).fields["classification"].astype(int)
        check("export: re-export of the kept npz at threshold 0.0 marks nothing",
              list(c0) == [10, 11, 12])

        # ---------------- ensemble vote (Phase 3): soft vote over saved probs,
        # class clamp, confidence-weighted fallback, agreement. Drives the
        # functions the GUI's ensemble mode calls; the Qt member loop itself is
        # GUI-side and not smoke-testable.
        import ensemble_vote as evote
        _q = lambda s: None   # quiet log for the ensemble() calls

        # soft vote overturns a 2-1 hard majority when the majority is unsure
        sv_lab, sv_conf = evote.soft_vote(np.array(
            [[[0.9, 0.1]], [[0.45, 0.55]], [[0.45, 0.55]]], np.float32))
        hv_lab, _ = evote.weighted_vote(np.array([[0], [1], [1]]),
                                        np.ones((3, 1), np.float32))
        check("ensemble: soft vote overturns the 2-1 hard majority via confidence",
              hv_lab.tolist() == [1] and sv_lab.tolist() == [0]
              and abs(float(sv_conf[0]) - 0.6) < 1e-6)
        wv_lab, wv_conf = evote.weighted_vote(
            np.array([[0, 0], [1, 0], [1, 0]]),
            np.array([[0.9, 1.0], [0.2, 1.0], [0.2, 1.0]], np.float32))
        check("ensemble: weighted-hard vote flips on confidence; conf = winning share",
              wv_lab.tolist() == [0, 0] and abs(float(wv_conf[0]) - 0.9 / 1.3) < 1e-6
              and float(wv_conf[1]) == 1.0)

        # synthetic 3-model trio end-to-end: infer_run.json clamp + npz payload
        ens = tmp / "ens"
        ex3 = make_xyz(3).astype(np.float32)
        EP = [np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]], np.float32),
              np.array([[0.45, 0.55], [0.7, 0.3], [0.3, 0.7]], np.float32),
              np.array([[0.45, 0.55], [0.8, 0.2], [0.4, 0.6]], np.float32)]
        for k, p in enumerate(EP):
            d = ens / f"m{k}"
            d.mkdir(parents=True)
            np.savez(d / "s_pred.npz", xyz=ex3,
                     classification=p.argmax(1).astype(np.int32),
                     confidence=p.max(1), probs=p.astype(np.float16))
            (d / "infer_run.json").write_text(json.dumps(
                {"num_classes": 2, "class_names": ["a", "b"], "backbone": f"m{k}"}),
                encoding="utf-8")
        edirs = [str(ens / f"m{k}") for k in range(3)]
        evote.ensemble(edirs, str(ens / "out"), log=_q)
        with np.load(ens / "out" / "s_pred.npz") as ez:
            check("ensemble: output npz carries confidence f32 + agreement f32",
                  ez["confidence"].dtype == np.float32
                  and ez["agreement"].dtype == np.float32)
            check("ensemble: 3-model soft vote — labels, confidence, agreement exact",
                  ez["classification"].tolist() == [0, 0, 1]
                  and np.allclose(ez["confidence"], [0.6, 0.7, 0.7], atol=2e-3)
                  and np.allclose(ez["agreement"], [1 / 3, 1.0, 1.0]))
        # clamp: mismatched class_names across infer_run.json refuse loudly
        (ens / "m1" / "infer_run.json").write_text(json.dumps(
            {"num_classes": 2, "class_names": ["a", "c"]}), encoding="utf-8")
        try:
            evote.ensemble(edirs, str(ens / "out2"), log=_q)
            clamp_raised = False
        except ValueError:
            clamp_raised = True
        check("ensemble: class clamp refuses mismatched class_names", clamp_raised)
        (ens / "m1" / "infer_run.json").write_text(json.dumps(
            {"num_classes": 2, "class_names": ["a", "b"]}), encoding="utf-8")
        # one member without probs -> confidence-weighted hard vote for the scene
        with np.load(ens / "m2" / "s_pred.npz") as zslim:
            slim = {k: zslim[k] for k in zslim.files if k != "probs"}
        np.savez(ens / "m2" / "s_pred.npz", **slim)
        evote.ensemble(edirs, str(ens / "out3"), log=_q)
        with np.load(ens / "out3" / "s_pred.npz") as ez3:
            # point 0: w(0)=0.9 vs w(1)=0.55+0.55 -> label 1, share 1.1/2.0
            check("ensemble: probs missing anywhere -> weighted-hard fallback",
                  ez3["classification"].tolist() == [1, 0, 1]
                  and np.allclose(ez3["confidence"], [0.55, 1.0, 1.0], atol=1e-6))

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

        # inline HAG: bake it during tiling in one pass (no separate reload).
        # ground_value=2 (the "Ground" source value) -> the labels are the ONLY
        # ground source (SMRF never runs); grid nearest-fills label gaps.
        staged_ih = dataset.convert_dataset(
            "laz_hag_inline", str(laz_root / "train"), spec, classes, [0], tmp / "staging",
            val_inputs=[str(laz_root / "val")], test_inputs=[str(laz_root / "test")],
            compute_hag=True, ground_value=2, progress=print)
        zih = np.load(staged_ih / "train" / "scene0.npz")
        check("convert_dataset(compute_hag): tiles carry a feat_hag channel inline",
              "feat_hag" in zih.files
              and zih["feat_hag"].shape[0] == zih["xyz"].shape[0])
        mih = json.loads((staged_ih / "dataset_meta.json").read_text())
        check("convert_dataset(compute_hag): meta records grid+labels ground + has_hag",
              mih["has_hag"] is True
              and mih["source"]["hag_source"] == "grid+labels"
              and mih["source"]["hag_ground_value"] == 2
              and mih["source"]["hag_use_smrf"] is False)
        check("convert_dataset(compute_hag): feature_channels catalogs the hag entry",
              {"name": "hag", "source_field": "@hag:grid+labels", "norm": "raw"}
              in mih["source"]["feature_channels"])

        # convert_infer_job with the HAG box ticked + a ground class: resolves the
        # 'classification' field off files[0], masks ground by value, bakes a
        # feat_hag channel — and STILL writes no label key (empty value_to_index).
        # Guards the whole opt-in HAG path; grid needs no PDAL.
        job_h = dataset.convert_infer_job("hag_job", str(laz_root / "val"),
                                          tmp / "staging", hag=True, hag_filter="grid",
                                          ground_value=2, progress=print)
        zjh = np.load(next((job_h / "scenes").glob("*.npz")))
        check("convert_infer_job(hag, ground_value): feat_hag baked, no label key",
              "feat_hag" in zjh.files and "label" not in zjh.files
              and zjh["feat_hag"].shape[0] == zjh["xyz"].shape[0])
        # Ground SOURCE and interpolation are separate axes now; the old smrf_fill
        # union is gone — a ground class means labels only, no second source.
        import inspect
        from trainer_gui import pretrain
        check("hag: smrf_fill union removed from the whole chain",
              "smrf_fill" not in inspect.signature(dataset.convert_infer_job).parameters
              and "smrf_fill" not in inspect.signature(pretrain.hag_for_cloud).parameters
              and "use_smrf" not in inspect.signature(pretrain.hag_for_cloud).parameters)
        # Ground from the labels must NOT equal ground from detection — otherwise
        # ground_value silently isn't reaching hag_for_cloud's ground_mask.
        job_d = dataset.convert_infer_job("hag_detect_job", str(laz_root / "val"),
                                          tmp / "staging", hag=True, hag_filter="grid")
        zjd = np.load(next((job_d / "scenes").glob("*.npz")))
        check("convert_infer_job(ground_value): labeled ground != detected ground",
              "feat_hag" in zjd.files
              and not np.allclose(zjh["feat_hag"], zjd["feat_hag"]))
        # HAG off = the default: not one wasted ground pass, no feat_hag channel.
        job_n = dataset.convert_infer_job("nohag_job", str(laz_root / "val"),
                                          tmp / "staging")
        zjn = np.load(next((job_n / "scenes").glob("*.npz")))
        check("convert_infer_job(): HAG is opt-in — no feat_hag channel by default",
              "feat_hag" not in zjn.files)

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

        # ---------------- ground-truth comparison stats (Inference page)
        # analysis.prediction_metrics scores explicit per-point classifications
        # only (a prediction npz 'classification'/'pred' vs a GT npz 'label' /
        # LAS classification) — RGB-palette decoding is gone with the viewer.
        gtc = tmp / "gtcmp"
        gtc.mkdir()
        np.savez(gtc / "sceneY_pred.npz", xyz=make_xyz(5).astype(np.float32),
                 classification=np.array([0, 1, 2, 2, 0], np.int32))
        np.savez(gtc / "sceneY_gt.npz", xyz=make_xyz(5).astype(np.float32),
                 label=np.array([0, 1, 1, 2, -1], np.int32))
        gm = analysis.prediction_metrics(gtc / "sceneY_pred.npz", gtc / "sceneY_gt.npz")
        check("analysis: prediction_metrics scores accuracy + per-class IoU on labeled pts",
              gm["scene"] == "sceneY" and gm["labeled"] == 4
              and abs(gm["accuracy"] - 0.75) < 1e-9
              and gm["per_class_iou"] == {0: 1.0, 1: 0.5, 2: 0.5}
              and abs(gm["miou"] - 2 / 3) < 1e-9)

        # ================= LOCAL (Docker) backend =================
        import importlib

        import _modal_shim
        from trainer_gui import local_cli
        BB = BACKBONES

        # the modal shim must import every training script (no torch / no cloud)
        # and expose its @app.local_entrypoint main + the recorded Image recipe.
        # Derived from the registry so a new backbone is covered automatically.
        SCRIPTS = [Path(b.script).stem for b in BB.values()]
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
        check(f"local: modal shim imports all {len(SCRIPTS)} train scripts + finds main", shim_ok)
        check("local: every script records an Image recipe for Dockerfile gen", recipe_ok)
        # the shim's catch-all __getattr__ must RAISE on dunders, else inspect.getmodule
        # reads modal.__file__ as a function and torch's import blows up (endswith bug).
        _modal_shim.install()
        check("local: shim raises on dunder lookups (inspect.getmodule-safe)",
              not hasattr(sys.modules["modal"], "__file__")
              and sys.modules["modal"].Volume is not None)

        # decoupling: every local_train_X.py imports with NO modal (torch lives in
        # the body) and exposes its train fn with no-op Volume stand-ins.
        LOCAL = [s.replace("modal_train_", "local_train_") for s in SCRIPTS]
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
        check(f"local: all {len(LOCAL)} local_train_*.py import + expose train fn + no volume stubs", local_ok)
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
        check(f"local: --mode infer runs dataset-free (all {len(LOCAL)} backbones)", infer_gate_ok)
        # Phase 0 contract: every backbone's infer path funnels through
        # train_common.run_infer_scenes, so each produced *_pred.npz must carry a
        # per-point max-softmax confidence in (0, 1] (the export threshold reads it).
        # The trainers above fault at missing weights before predicting, so drive
        # the shared loop with a stub predict instead of a GPU run.
        il_dir = tmp / "infer_loop"
        il_dir.mkdir()
        _il_conf = np.array([0.4, 1.0, 0.7], np.float32)
        train_common.run_infer_scenes(
            ["a.npz", "b.npz"],
            lambda p: (make_xyz(3).astype(np.float32), np.array([0, 1, 0], np.int64),
                       None, _il_conf, np.full((3, 2), 0.5, np.float16)),
            str(il_dir), str(il_dir), {"mode": "infer"})
        il_preds = sorted(il_dir.glob("*_pred.npz"))
        check("local: infer loop writes confidence in (0,1] into every *_pred.npz",
              len(il_preds) == 2
              and all("confidence" in (z := np.load(f)).files
                      and 0.0 < float(z["confidence"].min())
                      and float(z["confidence"].max()) <= 1.0 for f in il_preds))
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
        df = gen.build_dockerfile("ptv3", "modal_train_ptv3.py")
        check("gen: ptv3 Dockerfile uses a CUDA devel base + python bootstrap",
              "FROM nvidia/cuda:" in df and "-devel-ubuntu22.04" in df
              and "ln -sf /usr/bin/python3 /usr/bin/python" in df)
        check("gen: Dockerfile pins a heredoc-capable frontend on line 1",
              df.splitlines()[0] == "# syntax=docker/dockerfile:1")
        check("gen: shell-unsafe pip version specs are quoted",
              "'numpy<2.0'" in df and "'pandas<3'" in df)
        check("gen: model repo is a pinned upstream clone (no build-contexts)",
              "git clone https://github.com/Pointcept/PointTransformerV3.git /opt/ptv3" in df
              and "checkout --detach" in df and "COPY --from" not in df)
        check("gen: local_train script is NOT baked (bind-mounted at /workspace locally)",
              "local_train" not in df)
        dfr = gen.build_dockerfile("randlanet", "modal_train_randlanet.py")
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

            # delete removes only the data; runs/ + infer/ are kept for record keeping
            keep_ds = tmp / "keep_ds"
            (keep_ds / "train").mkdir(parents=True)
            (keep_ds / "dataset_meta.json").write_text("{}", encoding="utf-8")
            (keep_ds / "runs" / "r1").mkdir(parents=True)
            (keep_ds / "infer" / "j1").mkdir(parents=True)
            appstate.remember_dataset("keep_ds", {"staged_dir": str(keep_ds), "uploaded": False})
            _, kerr = appstate.delete_dataset("keep_ds")
            check("appstate: delete_dataset keeps runs/ + infer/, removes only data",
                  not kerr and not (keep_ds / "train").exists()
                  and not (keep_ds / "dataset_meta.json").exists()
                  and (keep_ds / "runs" / "r1").is_dir() and (keep_ds / "infer" / "j1").is_dir())
            check("appstate: kept runs re-registered on the Plotting page",
                  str(keep_ds / "runs") in appstate.get("plot_extra_roots", []))

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

            # ---- workspace re-root: one root owns every dataset; runs + infer nest
            # under it, host-side only (container paths /datasets, /outputs unchanged).
            ws = tmp / "ws"
            appstate.set_workspace(str(ws))
            # A prior check pinned datasets_root to a resolved path; clear it so we
            # exercise the real default (blank -> derives from the workspace).
            appstate.set_local_config({**appstate.local_config(), "datasets_root": ""})
            check("appstate: workspace_dir + datasets_root default to the set workspace",
                  appstate.workspace_dir() == ws
                  and appstate.local_config()["datasets_root"] == str(ws))
            # base /datasets = workspace (so /datasets/<name> resolves with no extra
            # mount); the dataset's own folder bound to /outputs => runs at <ds>/runs/<id>.
            ds_root = ws / "myds"
            _, wargs = local_cli.run_script(
                "modal_train_ptv3.py", {"dataset": "myds"}, BB["ptv3"],
                repo_root="/repo", outputs_root=str(ds_root))
            wj = " ".join(wargs)
            check("local_cli: workspace -> /datasets and the dataset folder -> /outputs",
                  f"{ws.as_posix()}:/datasets" in wj and f"{ds_root.as_posix()}:/outputs" in wj)
            check("appstate: scratch_infer_dir sits under the workspace",
                  appstate.scratch_infer_dir() == ws / "_scratch" / "infer")
            # infer job nested under its owning dataset via out_dir (container path fixed)
            job2 = dataset.convert_infer_job("ijob", str(txt_root / "val"),
                                             ws, out_dir=ds_root / "infer" / "ijob")
            check("dataset: convert_infer_job(out_dir=) nests scenes under <ds>/infer/<job>",
                  job2 == ds_root / "infer" / "ijob"
                  and (job2 / "scenes").is_dir() and (job2 / "job_meta.json").exists())
            appstate.remember_dataset("myds", {"staged_dir": str(ds_root),
                                               "meta_path": str(ds_root / "dataset_meta.json"),
                                               "uploaded": False})
            (ds_root / "runs").mkdir(parents=True, exist_ok=True)
            check("appstate: dataset_root + dataset_run_roots resolve inside the workspace",
                  appstate.dataset_root("myds") == ds_root
                  and (ds_root / "runs") in appstate.dataset_run_roots())
            appstate.forget_dataset("myds")

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
                  and appstate.backbone_enabled("kpconv"))
            appstate.set_enabled_backbones(["ptv3", "randlanet", "kpconvx_cold"])
            check("appstate: explicit selection hides the others in local mode",
                  appstate.backbone_enabled("randlanet")
                  and not appstate.backbone_enabled("kpconv"))
            appstate.set_exec_mode("modal")
            check("appstate: selection does NOT filter in modal mode",
                  appstate.backbone_enabled("kpconv"))
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
