"""
Modal training script for KPConvX-L on IEEE GRSS 2019 Track 4 — COLD-START,
train_LAS-matched, IEEE-NATIVE-FEATURES + HAG variant.

This is the HAG twin of modal_train_kpconvx_cold.py. EVERYTHING is identical
except the 4th feature channel: the cold script feeds a crude height proxy
(z - tile_min_z); this script feeds the real PDAL HeightAboveGround dimension
computed by the trainer_gui Pretraining tab (SMRF ground-classify -> hag_nn).
Same 4 input channels, same architecture/param-count -> a clean A/B test of
"tile-relative height" vs "true height above ground".

  This file       : 4 features = [1, intensity, return_number, HAG].
                    HAG is read per-point from /data/IEEE/HAG/{Train,Validate}/
                    <scene>_PC3.laz (HeightAboveGround dim) and paired to the
                    raw _PC3.txt points.
  Cold sibling    : modal_train_kpconvx_cold.py — 4th channel = z - tile_min_z.
  Geo sibling     : modal_train_kpconvx_cold_geo.py — the train_LAS feature set
                    [1, Linearity, Planarity, Scattering, Verticality].

DATA AND EVALUATION stay train_LAS-faithful (2.0 m grid, conv radius 2.5 cells,
BN momentum 0.02, the same augmentation suite, IEEE label mapping). The TRAINING
ENGINE is KPConvX's own recipe (experiments/S3DIS/train_S3DIS.py), not the 2019
KPConv SGD recipe — the architecture was tuned under it:
  - AdamW (weight_decay 0.05) + 1-cycle LR (1e-4 -> 5e-3 raise over 30 epochs,
    5-epoch plateau, /10 per 120 epochs) + label smoothing 0.2
  - packed batches: PACK_N tiles concatenated per forward (lengths-aware
    pyramid), x ACCUM grad accumulation -> effective batch 8 clouds/step
  - 100 m tiles: KPConvX guidance keeps input-region/grid ~ 50; at a 2.0 m grid
    that is ~100 m. The old 50 m tiles starved the two deepest stages (40/80 m
    receptive fields) and gave water no shoreline context.
  - class-weighted smoothed CE + mild rare-tile oversampling (Water/Bridge)
  - held-out val pass every VAL_EVERY epochs (eval mode, no weight updates),
    appended to val_metrics.csv so convergence can be watched over time
  - final val/test eval votes over overlapping tiles (stride 50): per-voxel
    logit sums across up to 4 covering tiles

RUNTIME: 150 epochs x 300 steps x 8 tiles = 360,000 tile forwards of ~2-5k pts.
Checkpoints (model+optimizer) every 10 epochs; if the 24 h timeout hits, set
RESUME=True and relaunch to continue the same run.

ASPRS code -> contiguous class index (class 0 ignored):
  {0:-1, 2:0 Ground, 5:1 Trees, 6:2 Building, 9:3 Water, 17:4 Bridge}

Prereq: upload the HAGTEST output (the Pretraining tab's LAZ-with-HAG) to the
ieee-data volume so the loader can find HeightAboveGround per scene:
    modal volume put ieee-data "C:/Users/OrionHoch/Desktop/HAGTEST/Train"    /IEEE/HAG/Train
    modal volume put ieee-data "C:/Users/OrionHoch/Desktop/HAGTEST/Validate" /IEEE/HAG/Validate

Usage:
    modal run modal_train_kpconvx_cold_hag.py
    modal run modal_train_kpconvx_cold_hag.py --mode eval     # re-score the latest
        # run's weights with the (center-weighted) voted eval — no training.
    modal run modal_train_kpconvx_cold_hag.py --mode eval --weights runs/<id>/final_model.pth
"""

from typing import Optional

import modal

# ============================================================================
# Configuration
# ============================================================================
APP_NAME      = "kpconvx-cold-ieee-hag"
FEATURE_MODE  = "native_hag"  # [1, intensity, return_number, HAG (height above ground)]
GPU_TYPE      = "A100"
N_EPOCHS      = 200          # extended from 150 (val mIoU still rising); the
                             # 1-cycle lr keeps decaying smoothly past 150
EPOCH_STEPS   = 300          # optimizer steps / epoch (KPConvX S3DIS: 300)
PACK_N        = 4            # tiles packed per forward (KPConvX batch_size = 4)
ACCUM         = 2            # grad-accumulated forwards / step -> effective batch 8 tiles
CHECKPOINT_GAP = 10          # checkpoint frequency (epochs); saves model + optimizer
VAL_EVERY     = 10           # held-out val pass every N epochs (no weight updates)
VAL_SUBSET    = 200          # tiles used in the periodic val pass (evenly spaced)
TIMEOUT_HOURS = 24

# Resume: when True, continue the most recent AdamW-recipe run in the outputs
# volume (same run dir, appended metrics) instead of starting fresh. On, to
# extend the 150-epoch run to 200. Set back to False for a fresh run.
RESUME = False

# Class-balanced loss + rare-class oversampling. This deliberately diverges from
# train_LAS.py's segloss_balance='none' to stop the rare Water/Bridge classes
# from collapsing to 0 IoU under plain cross-entropy.
CLASS_WEIGHTING = True
WEIGHT_BETA     = 0.5        # 0 = uniform, 1 = full inverse-freq; 0.5 = inverse SQRT
                            # frequency (w = 1/sqrt(freq)) — the RandLA-Net /
                            # SemanticKITTI standard: sub-linear, so rare classes
                            # are boosted without raw 1/freq's instability.
WEIGHT_CAP      = 5.0        # clamp each weight to [1/CAP, CAP] after mean-normalising
# Lovász-Softmax: tractable surrogate that optimizes mIoU (Jaccard) directly and
# weights every class equally, countering CE's majority bias on rare classes.
# Total loss = <pointwise> + LOVASZ_WEIGHT * lovasz_softmax.
# Set 0.0 to disable.
LOVASZ_WEIGHT   = 1.0
# Focal loss (Lin et al. 2017): when USE_FOCAL, the pointwise term is
# alpha-balanced focal loss instead of weighted (label-smoothed) CE. alpha
# reuses the inverse-sqrt class weights; (1-p_t)^gamma down-weights easy points
# so hard + rare points dominate. gamma=0 == weighted CE. NOTE: the focal path
# does NOT apply LABEL_SMOOTH (focal's modulation is the regulariser); set
# USE_FOCAL=False to fall back to the smoothed weighted-CE term.
USE_FOCAL       = False
FOCAL_GAMMA     = 2.0
RARE_OVERSAMPLE = True
RARE_CLASSES    = [3, 4]     # Water, Bridge (contiguous class indices)
# 0.5 + cap 10 overcooked (Building 0.74->0.61, rare tiles memorised): dialed back.
RARE_TILE_PROB  = 0.25       # P(draw the next train tile from a rare-class tile)

NUM_CLASSES   = 5            # IEEE: Ground, Trees, Building, Water, Bridge
INPUT_CHANNELS = 4           # [1, intensity, return_number, HAG]
N_VAL_HOLDOUT = 10
HOLDOUT_SEED  = 42

# Geometry — matches train_LAS.py LASConfig.
GRID          = 2.0          # first_subsampling_dl: layer-0 voxel grid (m)
KP_RADIUS     = 2.5          # conv_radius (grid cells); first_radius = GRID*KP_RADIUS
RADIUS_SCALING = 2.0         # standard KPConv per-layer doubling

# KPConvX guidance: input region ~ 50x the subsampling grid (S3DIS: 2.1 m at
# 0.04 m). At GRID=2.0 that is ~100 m. Stride 50 everywhere: train tiles overlap
# for coverage, and val/test tiles overlap so the final eval can vote (each
# point is covered by up to 4 tiles).
CHUNK_XY      = 100.0
STRIDE        = 50.0

# Augmentation (train_LAS.py LASConfig).
AUG_SCALE_MIN = 0.9
AUG_SCALE_MAX = 1.1
AUG_SYMMETRY_X = True        # augment_symmetries = [True, False, False]
AUG_NOISE     = 0.05         # augment_noise
AUG_COLOR     = 0.8          # augment_color: P(keep features) = 0.8

# Optimizer — KPConvX's own recipe (experiments/S3DIS/train_S3DIS.py).
WEIGHT_DECAY  = 0.05
CYC_LR0       = 1e-4         # start (and floor) lr of the 1-cycle schedule
CYC_LR1       = 5e-3         # peak lr
CYC_RAISE     = 30           # epochs raising lr0 -> lr1 (exponential)
CYC_PLATEAU   = 5            # epochs held at peak
CYC_DECREASE10 = 120         # epochs per /10 decay after the plateau
LABEL_SMOOTH  = 0.2          # KPConvX smooth_labels
GRAD_CLIP     = 100.0
BN_MOMENTUM   = 0.02

DATA_ROOT      = "/data/IEEE"
TRAIN_PC_DIR   = f"{DATA_ROOT}/Train-Track4/Track4"
TRAIN_CLS_DIR  = f"{DATA_ROOT}/Train-Track4-Truth/Track4-Truth"
TEST_PC_DIR    = f"{DATA_ROOT}/Validate-Track4/Track4"
TEST_CLS_DIR   = f"{DATA_ROOT}/Validate-Track4-Truth"
PRED_PC_DIR    = f"{DATA_ROOT}/Test-Track4/Test-Track4"
# Per-point HeightAboveGround from the Pretraining tab (HAGTEST upload). Train +
# val-holdout scenes come from Train-Track4 (HAG/Train); test from HAG/Validate.
HAG_TRAIN_DIR  = f"{DATA_ROOT}/HAG/Train"
HAG_TEST_DIR   = f"{DATA_ROOT}/HAG/Validate"
N_PREDICT      = 1
PREP_DIR       = f"{DATA_ROOT}/prep/kpconvx_cold_hag_grid20_c100_origin"

CLASS_NAMES = ["Ground", "Trees", "Building", "Water", "Bridge"]
LABEL_MAP   = {0: -1, 2: 0, 5: 1, 6: 2, 9: 3, 17: 4}

# ============================================================================
# Modal image
# ============================================================================
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential", "cmake", "libgl1", "libglib2.0-0", "ninja-build")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "numpy<2.0",
        "scipy",
        "scikit-learn",
        "easydict",
        "h5py",
        "matplotlib",
        "timm",
        "pykeops",
        "tqdm",
        "tensorboard",
        "pandas<3",
        "laspy",
        "lazrs",          # LAZ backend so laspy can read the HAG .laz scenes
        index_url="https://download.pytorch.org/whl/cu121",
        extra_index_url="https://pypi.org/simple",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

# Mount the KPConvX standalone repo into the image at /opt/kpconvx.
# (Path corrected from the warm script — the repo now lives under Modal_H3D.)
image = image.add_local_dir(
    "C:/Users/OrionHoch/Desktop/Modal_H3D/ml-kpconvx/Standalone/KPConvX",
    "/opt/kpconvx",
    copy=True,
)
image = image.run_commands(
    "cd /opt/kpconvx/cpp_wrappers/cpp_subsampling && python setup.py build_ext --inplace",
    "cd /opt/kpconvx/cpp_wrappers/cpp_neighbors && python setup.py build_ext --inplace",
    "touch /opt/kpconvx/cpp_wrappers/__init__.py "
    "      /opt/kpconvx/cpp_wrappers/cpp_subsampling/__init__.py "
    "      /opt/kpconvx/cpp_wrappers/cpp_neighbors/__init__.py",
)

data_volume     = modal.Volume.from_name("ieee-data",            create_if_missing=True)
outputs_volume  = modal.Volume.from_name(f"{APP_NAME}-outputs",  create_if_missing=True)
datasets_volume = modal.Volume.from_name("terminal-datasets",    create_if_missing=True)

# ============================================================================
# Training function
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_volume, "/outputs": outputs_volume,
             "/datasets": datasets_volume},
    cpu=8,
    memory=49152,
    timeout=TIMEOUT_HOURS * 3600,
)
def train_kpconvx(mode: str = "train", weights: Optional[str] = None,
                  infer_input: Optional[str] = None, grid: Optional[float] = None,
                  chunk_xy: Optional[float] = None):
    import os, sys, time, json, csv, glob, traceback
    from datetime import datetime
    import numpy as np
    import torch
    import laspy
    from scipy.spatial import cKDTree

    sys.path.insert(0, "/opt/kpconvx")
    EVAL_ONLY = (mode == "eval")
    INFER     = (mode == "infer")   # arbitrary-folder inference (trainer_gui)

    LABEL_LUT = np.full(256, -1, dtype=np.int32)
    for raw, mapped in LABEL_MAP.items():
        LABEL_LUT[raw] = mapped

    # ------------------------------------------------------------------------
    # HAG: per-point HeightAboveGround read from the Pretraining-tab .laz files
    # and paired to the raw _PC3.txt points.
    # ------------------------------------------------------------------------
    def _read_hag_laz(laz_path):
        las = laspy.read(laz_path)
        names = list(las.point_format.dimension_names)
        hname = next((n for n in names if n.lower() in ("heightaboveground", "hag")), None)
        if hname is None:
            raise KeyError(f"{laz_path}: no HeightAboveGround dim (have {names})")
        hag = np.asarray(las[hname], dtype=np.float32)
        laz_xyz = np.column_stack([las.x, las.y, las.z])   # float64: absolute UTM
        return hag, laz_xyz                                 # needs full precision for pairing

    def load_hag(name, hag_dir, xyz_ref):
        """HAG aligned to xyz_ref's point order. PDAL preserves point order, so
        when counts match we index-pair (verified on a sample); otherwise we
        fall back to nearest-neighbor pairing on xyz. Returns float32 (N,)."""
        laz_path = f"{hag_dir}/{name}_PC3.laz"
        if not os.path.exists(laz_path):
            raise FileNotFoundError(
                f"HAG file missing: {laz_path}. Upload the Pretraining-tab output:\n"
                f'  modal volume put ieee-data "<HAGTEST>/Train" /IEEE/HAG/Train\n'
                f'  modal volume put ieee-data "<HAGTEST>/Validate" /IEEE/HAG/Validate')
        hag, laz_xyz = _read_hag_laz(laz_path)
        # xyz_ref is offset to a per-scene origin (precision fix) while the laz
        # carries absolute coords; subtract each array's own per-axis min so the
        # index-pair guard and NN fallback compare in a common full-precision frame.
        ref0 = xyz_ref.astype(np.float64) - xyz_ref.min(0)
        laz0 = laz_xyz - laz_xyz.min(0)
        if len(hag) == len(xyz_ref):
            k = min(2048, len(hag))
            s = np.random.RandomState(0).choice(len(hag), k, replace=False)
            if np.allclose(laz0[s], ref0[s], atol=0.05):
                return hag
        nn = cKDTree(laz0).query(ref0)[1]
        return hag[nn].astype(np.float32)

    # ------------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------------
    def grid_subsample(xyz, attrs, lab, voxel):
        """Voxel-grid subsample to `voxel` m: barycenter points, mean attrs,
        majority labels. Mirrors KPConv-PyTorch's grid_subsampling (the C++ op
        that produces the first_subsampling_dl=2.0 layer-0 cloud)."""
        keys = np.floor(xyz / voxel).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        nv = int(inv.max()) + 1
        cnt = np.bincount(inv, minlength=nv).astype(np.float64)
        sx = np.zeros((nv, 3)); np.add.at(sx, inv, xyz); sx /= cnt[:, None]
        sa = None
        if attrs is not None:
            sa = np.zeros((nv, attrs.shape[1])); np.add.at(sa, inv, attrs); sa /= cnt[:, None]
        sl = np.full(nv, -1, np.int64)
        if lab is not None:
            oh = np.zeros((nv, NUM_CLASSES)); v = lab >= 0
            np.add.at(oh, (inv[v], lab[v]), 1)
            has = oh.sum(1) > 0
            sl[has] = oh[has].argmax(1)
        return (sx.astype(np.float32),
                (sa.astype(np.float32) if sa is not None else None),
                sl)

    def augment(xyz):
        """train_LAS.py augmentation_transform: vertical rotation, anisotropic
        scale in [0.9,1.1] with random x-flip, gaussian noise 0.05."""
        theta = np.random.rand() * 2 * np.pi
        cs, sn = np.cos(theta), np.sin(theta)
        R = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], np.float32)
        scale = np.random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX, 3).astype(np.float32)
        if AUG_SYMMETRY_X and np.random.rand() < 0.5:
            scale[0] *= -1.0
        out = (xyz @ R.T) * scale
        out += np.random.normal(0, AUG_NOISE, out.shape).astype(np.float32)
        return out.astype(np.float32)

    # ------------------------------------------------------------------------
    # IEEE preprocessing -> 2.0 m subsampled .npz chunks (xyz + intensity/ret)
    # ------------------------------------------------------------------------
    def load_ieee(pc_path, cls_path):
        # PC3.txt: x, y, z, intensity, returnNumber  (CSV, no RGB)
        pc = np.loadtxt(pc_path, delimiter=",")                 # float64 (full precision)
        # Per-scene origin offset before the float32 cast: projected (UTM) coords
        # otherwise quantize to ~0.25-0.5 m on northing (float32 ~7 sig digits),
        # corrupting sub-meter geometry. Deterministic offset, so cached tiles and
        # the eval-time raw reload share one coordinate frame.
        xyz       = (pc[:, :3] - np.floor(pc[:, :3].min(0))).astype(np.float32)
        intensity = pc[:, 3].astype(np.float32)
        ret_num   = pc[:, 4].astype(np.float32) if pc.shape[1] >= 5 else np.zeros(len(pc), np.float32)
        lab_raw   = np.loadtxt(cls_path, dtype=np.int32).reshape(-1)
        if len(lab_raw) != len(xyz):
            raise ValueError(f"point/label mismatch in {pc_path}: "
                             f"{len(xyz)} pts vs {len(lab_raw)} labels")
        lab = LABEL_LUT[np.clip(lab_raw, 0, 255)].astype(np.int32)
        return xyz, intensity, ret_num, lab

    def tile_and_save(name, pc_path, cls_path, out_dir, chunk_xy, stride, hag_dir):
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        try:
            xyz, intensity, ret_num, lab = load_ieee(pc_path, cls_path)
            hag = load_hag(name, hag_dir, xyz)
        except Exception as e:
            print(f"  skip {pc_path}: {e}", flush=True); return None
        # Robust per-scene normalisation: dividing by the max lets one hot return
        # rescale the whole scene, so "water-grade" intensity meant different
        # numbers in different scenes. p95 + clip keeps the scale comparable.
        i_p95 = max(float(np.percentile(intensity, 95)), 1.0)
        intensity_n = np.clip(intensity / i_p95, 0.0, 2.0).astype(np.float32)
        print(f"    {name}: {len(xyz):,} pts loaded in {time.time()-t0:.1f}s, "
              f"intensity p95={i_p95:.1f}, HAG {hag.min():.1f}..{hag.max():.1f}m, tiling…",
              flush=True)
        mins = xyz[:, :2].min(0); maxs = xyz[:, :2].max(0)
        x0s = np.arange(mins[0], maxs[0], stride)
        y0s = np.arange(mins[1], maxs[1], stride)
        n_tiles = 0
        for x0 in x0s:
            for y0 in y0s:
                mask = (
                    (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + chunk_xy) &
                    (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + chunk_xy)
                )
                # Low thresholds on purpose: water absorbs LiDAR, so pure-water
                # tiles are sparse — the old 256-pt cut deleted them from training.
                if mask.sum() < 64:
                    continue
                attrs = np.stack([intensity_n[mask], ret_num[mask], hag[mask]],
                                 axis=1).astype(np.float32)
                sx, sa, sl = grid_subsample(xyz[mask], attrs, lab[mask], GRID)
                if len(sx) < 32:
                    continue
                np.savez_compressed(
                    os.path.join(out_dir, f"{name}_x{int(x0)}_y{int(y0)}.npz"),
                    xyz=sx.astype(np.float32),
                    intensity=sa[:, 0].astype(np.float32),
                    ret_num=sa[:, 1].astype(np.float32),
                    hag=sa[:, 2].astype(np.float32),
                    lab=sl.astype(np.int32),
                )
                n_tiles += 1
        print(f"      -> {n_tiles} tiles", flush=True)
        return n_tiles

    def _split_scenes():
        train_pc = sorted(glob.glob(f"{TRAIN_PC_DIR}/*_PC3.txt"))
        if not train_pc:
            raise FileNotFoundError(
                f"No *_PC3.txt under {TRAIN_PC_DIR}. Upload the IEEE dataset to the "
                f"ieee-data volume first, e.g.:\n"
                f"  modal volume put ieee-data "
                f'"C:\\Users\\OrionHoch\\Desktop\\LabledDatasets\\IEEE" /IEEE')
        names_all = [os.path.basename(p).replace("_PC3.txt", "") for p in train_pc]
        rng = np.random.RandomState(HOLDOUT_SEED)
        idx = np.arange(len(names_all)); rng.shuffle(idx)
        val_names = sorted(names_all[i] for i in idx[:N_VAL_HOLDOUT])
        train_names = sorted(n for n in names_all if n not in val_names)
        def _pair(names, pc_dir, cls_dir):
            return [(n, f"{pc_dir}/{n}_PC3.txt", f"{cls_dir}/{n}_CLS.txt") for n in names]
        test_pc = sorted(glob.glob(f"{TEST_PC_DIR}/*_PC3.txt"))
        test_names = sorted(os.path.basename(p).replace("_PC3.txt", "") for p in test_pc)
        return (
            _pair(train_names, TRAIN_PC_DIR, TRAIN_CLS_DIR),
            _pair(val_names,   TRAIN_PC_DIR, TRAIN_CLS_DIR),
            _pair(test_names,  TEST_PC_DIR,  TEST_CLS_DIR),
        )

    def _cache_signature():
        # Everything that changes what a cached tile contains. A mismatch means
        # the cache is stale/leaky and must not be silently reused.
        return {
            "format_version": 1,
            "pipeline": "kpconvx_cold_hag",
            "grid": GRID,
            "chunk_xy": CHUNK_XY,
            "stride": STRIDE,
            "holdout_seed": HOLDOUT_SEED,
            "n_val_holdout": N_VAL_HOLDOUT,
            "label_map": {str(k): v for k, v in sorted(LABEL_MAP.items())},
            "min_pts_mask": 64,
            "min_pts_sub": 32,
            "intensity_norm": "p95_clip2",
            "feature_recipe": "bias,intensity,ret_num,hag",
            "hag_source": "pdal_hag_nn",
            "hag_train_dir": HAG_TRAIN_DIR,
            "hag_test_dir": HAG_TEST_DIR,
        }

    def _validate_cache(lists):
        """Refuse to reuse a cache built with different settings (grid, stride,
        split seed, label map, feature recipe, HAG source …) instead of silently
        mixing incompatible data. Migrate a pre-validation cache by stamping
        .done markers for already-tiled scenes. Returns True if the signature
        file was newly written (so the caller commits the volume)."""
        meta_path = f"{PREP_DIR}/cache_meta.json"
        cur = _cache_signature()
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                old = json.load(f)
            if old != cur:
                diffs = {k: [old.get(k), cur.get(k)]
                         for k in sorted(set(old) | set(cur)) if old.get(k) != cur.get(k)}
                raise RuntimeError(
                    f"Preprocess cache at {PREP_DIR} was built with DIFFERENT settings "
                    f"(mismatched: {diffs}). Reusing it would silently mix incompatible "
                    f"data. Point PREP_DIR at a fresh path or delete the stale cache.")
            return False
        legacy = False
        for split, items in lists:
            d = f"{PREP_DIR}/{split}"
            for name, _, _ in items:
                if glob.glob(f"{d}/{name}_x*.npz") and not os.path.exists(f"{d}/{name}.done"):
                    open(f"{d}/{name}.done", "w").close(); legacy = True
        with open(meta_path, "w") as f:
            json.dump(cur, f, indent=2)
        if legacy:
            print(f"  migrated existing cache at {PREP_DIR}: stamped .done markers + "
                  f"signature (assumed to match current settings).", flush=True)
        return True

    def ensure_prep():
        print(f"  ensuring preprocessed cache -> {PREP_DIR}", flush=True)
        for split in ("train", "val", "test"):
            os.makedirs(f"{PREP_DIR}/{split}", exist_ok=True)
        train_list, val_list, test_list = _split_scenes()
        any_new = [_validate_cache([("train", train_list), ("val", val_list),
                                    ("test", test_list)])]

        def already_tiled(out_dir, name):
            # A scene counts as cached only once its .done marker exists, so a run
            # interrupted mid-scene is re-tiled rather than silently left partial.
            return os.path.exists(f"{out_dir}/{name}.done")

        def tile_remaining(items, out_dir, chunk_xy, stride, hag_dir):
            for name, pc_path, cls_path in items:
                if already_tiled(out_dir, name):
                    continue
                n = tile_and_save(name, pc_path, cls_path, out_dir, chunk_xy, stride, hag_dir)
                if n is not None:          # None == load failed; leave unmarked to retry
                    open(f"{out_dir}/{name}.done", "w").close()
                any_new[0] = True

        # Train + val-holdout scenes are Train-Track4 -> HAG/Train; the test
        # split is Validate-Track4 -> HAG/Validate.
        print(f"  [train] {len(train_list)} scenes", flush=True)
        tile_remaining(train_list, f"{PREP_DIR}/train", CHUNK_XY, STRIDE, HAG_TRAIN_DIR)
        # val/test also tile at stride 50 so the final eval can vote over the
        # up-to-4 overlapping tiles covering each point.
        print(f"  [val] {len(val_list)} scenes", flush=True)
        tile_remaining(val_list, f"{PREP_DIR}/val", CHUNK_XY, STRIDE, HAG_TRAIN_DIR)
        print(f"  [test] {len(test_list)} scenes", flush=True)
        tile_remaining(test_list, f"{PREP_DIR}/test", CHUNK_XY, STRIDE, HAG_TEST_DIR)
        if any_new[0]:
            data_volume.commit()
            print("  preprocessing committed.", flush=True)
        else:
            print("  all scenes already cached.", flush=True)
        return train_list, val_list, test_list

    def make_run_dir():
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_ieee_kpconvx_cold_hag")
        run_dir = f"/outputs/runs/{run_id}"
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        return run_id, run_dir

    def find_latest_checkpoint():
        """Most recent run (run-ids are timestamps, so they sort) that has
        checkpoints AND was trained with this script's recipe — old SGD-recipe
        runs in the volume are not valid warm starts. Returns
        (run_dir, ckpt_path, epoch) or None."""
        def _ep(p):
            return int(os.path.basename(p)[2:5])   # ep149.pth -> 149
        for rd in sorted(glob.glob("/outputs/runs/*"), reverse=True):
            ckpts = glob.glob(f"{rd}/checkpoints/ep*.pth")
            if not ckpts:
                continue
            try:
                with open(f"{rd}/run_config.json") as f:
                    opt_type = json.load(f).get("optimizer", {}).get("type")
            except Exception:
                opt_type = None
            if opt_type != "AdamW":
                print(f"  resume: skipping {os.path.basename(rd)} "
                      f"(recipe mismatch: optimizer={opt_type})", flush=True)
                continue
            latest = max(ckpts, key=_ep)
            return rd, latest, _ep(latest)
        return None

    print("=" * 70)
    print(f"  KPConvX-L  IEEE Track 4  COLD/{FEATURE_MODE}  "
          f"({GPU_TYPE}, {N_EPOCHS} ep, {EPOCH_STEPS} steps, "
          f"pack {PACK_N} x accum {ACCUM})")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}  device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    train_list, val_list, test_list = ([], [], []) if INFER else ensure_prep()

    resume_info = find_latest_checkpoint() if (RESUME or EVAL_ONLY) else None
    if INFER:
        # Inference-only: fresh *_infer run dir, weights loaded after net build.
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_infer")
        run_dir = f"/outputs/runs/{run_id}"
        os.makedirs(f"{run_dir}/predictions", exist_ok=True)
        resume_ckpt, start_epoch = None, 0
    elif resume_info:
        run_dir, resume_ckpt, resume_epoch = resume_info
        run_id = os.path.basename(run_dir)
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        start_epoch = resume_epoch + 1
        verb = "EVAL-ONLY on" if EVAL_ONLY else "RESUMING"
        print(f"  {verb} {run_id} from {os.path.basename(resume_ckpt)}"
              + ("" if EVAL_ONLY else f" -> starting at epoch {start_epoch}/{N_EPOCHS}"),
              flush=True)
    else:
        if EVAL_ONLY:
            raise RuntimeError("eval mode: no AdamW-recipe run with checkpoints "
                               "found on the outputs volume")
        run_id, run_dir = make_run_dir()
        resume_ckpt, start_epoch = None, 0
    if resume_ckpt is None and not INFER:
        with open(f"{run_dir}/run_config.json", "w") as f:
            json.dump({
            "backbone": "KPConvX-L",
            "warm_start": False,
            "feature_mode": FEATURE_MODE,
            "input_channels": INPUT_CHANNELS,
            "comparison_target": "kpconv-pdal/train_LAS.py (deformable KPFCNN)",
            "dataset": "IEEE GRSS 2019 DFC Track 4",
            "n_epochs": N_EPOCHS, "epoch_steps": EPOCH_STEPS,
            "pack_n": PACK_N, "accum": ACCUM,
            "grid_m": GRID, "kp_radius": KP_RADIUS, "radius_scaling": RADIUS_SCALING,
            "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
            "label_map_asprs_to_index": LABEL_MAP,
            "chunk_xy_m": CHUNK_XY, "stride_m": STRIDE,
            "optimizer": {"type": "AdamW", "weight_decay": WEIGHT_DECAY,
                          "cyc_lr0": CYC_LR0, "cyc_lr1": CYC_LR1,
                          "cyc_raise": CYC_RAISE, "cyc_plateau": CYC_PLATEAU,
                          "cyc_decrease10": CYC_DECREASE10,
                          "label_smoothing": LABEL_SMOOTH,
                          "grad_clip": GRAD_CLIP, "bn_momentum": BN_MOMENTUM},
            "class_balance": {"weighting": CLASS_WEIGHTING, "beta": WEIGHT_BETA,
                              "weight_scheme": "inv_sqrt_freq" if WEIGHT_BETA == 0.5
                              else f"inv_freq^{WEIGHT_BETA}",
                              "cap": WEIGHT_CAP, "rare_tile_prob": RARE_TILE_PROB},
            "loss": {"pointwise": "focal" if USE_FOCAL else "weighted_ce",
                     "focal_gamma": FOCAL_GAMMA if USE_FOCAL else None,
                     "ce_weighted": CLASS_WEIGHTING,
                     "label_smoothing": 0.0 if USE_FOCAL else LABEL_SMOOTH,
                     "lovasz_softmax_weight": LOVASZ_WEIGHT},
            "train_scenes": [n for n, _, _ in train_list],
            "val_scenes":   [n for n, _, _ in val_list],
            "test_scenes":  [n for n, _, _ in test_list],
            "holdout_seed": HOLDOUT_SEED,
        }, f, indent=2)

    # ------------------------------------------------------------------------
    # Build model — KPConvX-L preset, train_LAS-matched geometry, random init.
    # ------------------------------------------------------------------------
    from utils.config import init_cfg
    from models.KPNext import KPNeXt

    cfg = init_cfg()
    cfg.data.name           = "IEEE"
    cfg.data.task           = "cloud_segmentation"
    cfg.data.num_classes    = NUM_CLASSES
    cfg.data.dim            = 3
    cfg.data.label_values   = list(range(NUM_CLASSES))
    cfg.data.ignored_labels = []
    cfg.data.pred_values    = list(range(NUM_CLASSES))

    cfg.model.layer_blocks  = (3, 3, 9, 12, 3)
    cfg.model.kp_mode       = "kpconvx"
    cfg.model.shell_sizes   = [1, 14, 42]
    cfg.model.kp_radius     = KP_RADIUS
    cfg.model.kp_sigma      = KP_RADIUS
    cfg.model.kp_influence  = "linear"
    cfg.model.kp_aggregation = "nearest"
    cfg.model.kp_fixed      = "center"
    cfg.model.conv_groups   = -1
    cfg.model.share_kp      = True
    cfg.model.init_channels = 64
    cfg.model.channel_scaling = 1.41
    cfg.model.norm          = "batch"
    cfg.model.bn_momentum   = BN_MOMENTUM
    cfg.model.in_sub_size   = GRID
    cfg.model.in_sub_mode   = "grid"
    cfg.model.radius_scaling = RADIUS_SCALING
    cfg.model.grid_pool     = True
    cfg.model.decoder_layer = True
    cfg.model.upsample_n    = 3
    cfg.model.drop_path_rate = 0.3
    cfg.model.input_channels = INPUT_CHANNELS  # [1, intensity, ret_num, HAG]
    cfg.model.neighbor_limits = [12, 16, 20, 20, 20]
    cfg.model.use_strided_conv = True
    cfg.model.kpx_upcut     = False
    cfg.model.kpx_expansion = 8
    cfg.model.inv_groups    = 8
    cfg.model.inv_grp_norm  = True
    cfg.model.inv_act       = "sigmoid"
    cfg.model.first_inv_layer = 1
    cfg.model.kpinv_reduc   = 1

    cfg.data.init_sub_size  = GRID
    cfg.data.init_sub_mode  = "grid"

    net = KPNeXt(cfg).cuda()
    print(f"  Model params: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    print(f"  first_radius={net.first_radius:.2f} m  subsample_size={net.subsample_size:.2f} m  "
          f"num_layers={net.num_layers}", flush=True)

    optim = torch.optim.AdamW(net.parameters(), lr=CYC_LR0, weight_decay=WEIGHT_DECAY)
    # loss_fn is built after the tile scan below (needs class counts for weights).

    def lr_at(ep):
        """KPConvX 1-cycle: exponential raise lr0->lr1 over CYC_RAISE epochs,
        hold CYC_PLATEAU epochs, then /10 every CYC_DECREASE10 epochs. Set
        directly on the param groups each epoch (resume-friendly)."""
        if ep < CYC_RAISE:
            return CYC_LR0 * (CYC_LR1 / CYC_LR0) ** (ep / CYC_RAISE)
        if ep < CYC_RAISE + CYC_PLATEAU:
            return CYC_LR1
        return CYC_LR1 * 0.1 ** ((ep - CYC_RAISE - CYC_PLATEAU) / CYC_DECREASE10)

    # Resume from the latest checkpoint: restore weights (and optimizer state,
    # which checkpoints now include). LR is set per-epoch from lr_at().
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location="cuda")
        net.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        print(f"  resumed weights{' + optimizer' if 'optim' in ckpt else ''} "
              f"at epoch {start_epoch}", flush=True)

    if EVAL_ONLY:
        # Prefer explicit --weights, then the run's final_model.pth, else keep
        # the checkpoint weights loaded above. start_epoch = N_EPOCHS empties
        # the training loop, so the script falls straight to the final eval.
        fm = f"/outputs/{weights}" if weights else f"{run_dir}/final_model.pth"
        if weights and not os.path.exists(fm):
            raise FileNotFoundError(f"--weights not found on outputs volume: {fm}")
        if os.path.exists(fm):
            net.load_state_dict(torch.load(fm, map_location="cuda")["model"])
            print(f"  EVAL-ONLY: loaded {fm}", flush=True)
        start_epoch = N_EPOCHS

    if INFER:
        fm = f"/outputs/{weights}" if weights else None
        if not fm or not os.path.exists(fm):
            raise FileNotFoundError(f"--mode infer requires --weights; not found: {fm}")
        ck = torch.load(fm, map_location="cuda")
        net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        print(f"  [infer] loaded {weights}", flush=True)
        start_epoch = N_EPOCHS

    train_tiles = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    val_tiles   = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
    test_tiles  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
    print(f"  train_tiles: {len(train_tiles)}   val_tiles: {len(val_tiles)}   "
          f"test_tiles: {len(test_tiles)}", flush=True)
    if not train_tiles and not INFER:
        raise RuntimeError("No training tiles after preprocessing — check the IEEE upload.")

    # --- class-balanced loss + rare-class oversampling ----------------------
    # One pass over the training tiles: count labels for inverse-frequency class
    # weights, and flag tiles containing Water/Bridge so we can oversample them.
    print("  scanning train tiles for class balance…", flush=True)
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    rare_tiles = []
    for tp in train_tiles:
        lab = np.load(tp)["lab"]
        v = lab[lab >= 0]
        if v.size:
            class_counts += np.bincount(v, minlength=NUM_CLASSES)
            if RARE_OVERSAMPLE and np.isin(v, RARE_CLASSES).any():
                rare_tiles.append(tp)
    print(f"  class counts: {dict(zip(CLASS_NAMES, class_counts.tolist()))}", flush=True)
    print(f"  rare-class tiles: {len(rare_tiles)} / {len(train_tiles)}", flush=True)

    if CLASS_WEIGHTING:
        # beta=0.5 -> w = 1/sqrt(freq) (inverse-sqrt-frequency): sub-linear, so
        # rare classes are boosted without raw 1/freq's instability. Mean-norm,
        # then capped.
        freq = class_counts / max(int(class_counts.sum()), 1)
        w = (1.0 / np.maximum(freq, 1e-6)) ** WEIGHT_BETA
        w = w / w.mean()                                    # keep loss scale ~1
        w = np.clip(w, 1.0 / WEIGHT_CAP, WEIGHT_CAP)
        class_weights = torch.tensor(w, dtype=torch.float32).cuda()
        print(f"  class weights: "
              f"{dict(zip(CLASS_NAMES, [round(float(x), 3) for x in w]))}", flush=True)
    else:
        class_weights = None
    # Torch-native equivalent of KPConvX's SmoothCrossEntropyLoss (correct
    # ignore_index=-1 and weight normalisation, same smoothing).
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1,
                                        label_smoothing=LABEL_SMOOTH)

    # --- Lovász-Softmax (Berman et al. 2018): differentiable mIoU/Jaccard
    # surrogate, per-class and equally weighted, added on top of CE. Pure-torch
    # flat implementation (logits are already (N, C), labels (N,)). -----------
    def _lovasz_grad(gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1].clone()
        return jaccard

    def lovasz_softmax_flat(probas, labels):
        # probas: (N, C) softmax probs; labels: (N,) in [0, C). Averaged over
        # classes present in the batch.
        if probas.numel() == 0:
            return probas.sum() * 0.0
        losses = []
        for c in torch.unique(labels):
            fg = (labels == c).float()
            class_pred = probas[:, int(c)]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
        if not losses:
            return probas.sum() * 0.0
        return torch.stack(losses).mean()

    # alpha-balanced multiclass focal loss; masks ignore_index=-1 internally.
    # alpha = class_weights (inverse-sqrt) when set. No label smoothing here.
    def focal_loss(logits, labels):
        valid = labels >= 0
        if not valid.any():
            return logits.sum() * 0.0
        lg, lb = logits[valid], labels[valid]
        logp = torch.log_softmax(lg, dim=1)
        logpt = logp.gather(1, lb.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** FOCAL_GAMMA) * logpt
        if class_weights is not None:
            loss = loss * class_weights[lb]
        return loss.mean()

    def seg_loss(logits, labels):
        # Guard all-ignored packs: CrossEntropyLoss(ignore_index=-1) returns NaN
        # when every label is ignored (0/0 reduction). Return a finite, zero-grad
        # value instead so backward() can never poison the weights with NaN.
        valid = labels >= 0
        if not valid.any():
            return logits.sum() * 0.0
        loss = focal_loss(logits, labels) if USE_FOCAL else loss_fn(logits, labels)
        if LOVASZ_WEIGHT > 0:
            probas = torch.softmax(logits[valid], dim=1)
            loss = loss + LOVASZ_WEIGHT * lovasz_softmax_flat(probas, labels[valid])
        return loss

    def pick_train_tile():
        if rare_tiles and np.random.rand() < RARE_TILE_PROB:
            return rare_tiles[np.random.randint(len(rare_tiles))]
        return train_tiles[np.random.randint(len(train_tiles))]

    def _hag_of(z, xyz):
        """Per-point HAG for a cached tile. Real HeightAboveGround when present
        (HAG-prepped tiles); else the z-tile-min proxy (e.g. the Test-Track4
        predict demo, whose scenes have no HAG laz)."""
        if "hag" in z.files:
            return z["hag"].astype(np.float32)
        return (xyz[:, 2] - xyz[:, 2].min()).astype(np.float32)

    def build_feat(xyz, intensity, ret_num, hag, drop=False):
        """[1, intensity, return_number, HAG]. HAG = real HeightAboveGround
        (PDAL hag_nn). With `drop`, zero the non-bias channels (train_LAS
        feature-drop, P=1-augment_color=0.2)."""
        bias = np.ones((len(xyz), 1), np.float32)
        attrs = np.concatenate([intensity[:, None], ret_num[:, None],
                                hag[:, None]], axis=1).astype(np.float32)
        if drop:
            attrs[:, 1:] = 0.0   # keep intensity (water's main cue); drop ret_num/HAG
        return np.concatenate([bias, attrs], axis=1).astype(np.float32)

    def sample_tile(tile_path, max_pts=60000, min_pts=32, training=True):
        z = np.load(tile_path)
        xyz, intensity, ret_num, lab = z["xyz"], z["intensity"], z["ret_num"], z["lab"]
        hag = _hag_of(z, xyz)
        if len(xyz) < min_pts:
            return None
        idx = np.arange(len(xyz))
        if len(idx) > max_pts:
            idx = np.random.choice(idx, max_pts, replace=False)
        xyz, intensity, ret_num, lab, hag = \
            xyz[idx], intensity[idx], ret_num[idx], lab[idx], hag[idx]
        # HAG comes straight from the cached per-point height-above-ground.
        drop = (training and np.random.rand() > AUG_COLOR)
        feat = build_feat(xyz, intensity, ret_num, hag, drop=drop)
        geo_xyz = augment(xyz) if training else xyz
        geo_xyz = (geo_xyz - geo_xyz.mean(0)).astype(np.float32)
        return geo_xyz, feat, lab.astype(np.int64)

    from utils.torch_pyramid import build_full_pyramid

    class _KPBatch:
        def __init__(self, in_dict): self.in_dict = in_dict
        def device(self): return self.in_dict.points[0].device

    def make_kp_pack(samples):
        """Pack one or more (xyz, feat, lab) clouds into a single pyramid batch
        — KPConvX pack mode: concatenated points with per-cloud lengths, exactly
        how the repo's own collate batches clouds (neighbor search and pooling
        are lengths-aware, so clouds never mix)."""
        pts     = torch.from_numpy(np.ascontiguousarray(
                      np.concatenate([s[0] for s in samples]))).float()
        feats   = torch.from_numpy(np.ascontiguousarray(
                      np.concatenate([s[1] for s in samples]))).float()
        lengths = torch.tensor([len(s[0]) for s in samples], dtype=torch.long)
        pyr = build_full_pyramid(
            pts, lengths,
            net.num_layers, net.subsample_size, net.first_radius,
            net.radius_scaling, net.neighbor_limits, net.upsample_n,
            sub_mode=net.in_sub_mode, grid_pool_mode=net.grid_pool,
        )
        pyr.features = feats
        for k, v in list(pyr.items()):
            if isinstance(v, list):
                pyr[k] = [t.cuda() if torch.is_tensor(t) else t for t in v]
            elif torch.is_tensor(v):
                pyr[k] = v.cuda()
        lab_t = None
        if samples[0][2] is not None:
            lab_t = torch.from_numpy(np.ascontiguousarray(
                        np.concatenate([s[2] for s in samples]))).long().cuda()
        return _KPBatch(pyr), lab_t

    def _predict_points(xyz, intensity_n, ret_num):
        """Sliding-window KPConvX inference over already-normalized features;
        returns per-raw-point class indices. Shared by --mode infer (npz input)
        and the Test-Track4 predict demo (txt input). Neither carries a HAG laz,
        so the 4th channel uses the z-tile-min proxy (per-subtile)."""
        pred = np.full(len(xyz), -1, np.int64)
        mins, maxs = xyz[:, :2].min(0), xyz[:, :2].max(0)
        with torch.no_grad():
            for x0 in np.arange(mins[0], maxs[0] + CHUNK_XY, CHUNK_XY):
                for y0 in np.arange(mins[1], maxs[1] + CHUNK_XY, CHUNK_XY):
                    m = ((xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + CHUNK_XY) &
                         (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + CHUNK_XY))
                    if m.sum() < 64:
                        continue
                    idx = np.where(m)[0]
                    attrs = np.stack([intensity_n[idx], ret_num[idx]], axis=1).astype(np.float32)
                    sx, sa, _ = grid_subsample(xyz[idx], attrs, None, GRID)
                    if len(sx) < 32:
                        continue
                    proxy_hag = (sx[:, 2] - sx[:, 2].min()).astype(np.float32)
                    feat = build_feat(sx, sa[:, 0], sa[:, 1], proxy_hag)
                    cxyz = (sx - sx.mean(0)).astype(np.float32)
                    try:
                        batch, _ = make_kp_pack([(cxyz, feat, None)])
                        sub_pred = net(batch).argmax(-1).cpu().numpy()
                    except Exception:
                        continue
                    _, nn = cKDTree(sx).query(xyz[idx])
                    pred[idx] = sub_pred[nn]
        miss = pred < 0
        if miss.any() and (~miss).any():
            _, nn = cKDTree(xyz[~miss]).query(xyz[miss])
            pred[miss] = pred[~miss][nn]
        return np.clip(pred, 0, NUM_CLASSES - 1)

    # ------------------------------------------------------------------------
    # Arbitrary-folder inference (trainer_gui): label the npz scenes uploaded to
    # terminal-datasets:/_infer/<job>/scenes/ and write predictions, then stop.
    # The uploaded scenes carry no HAG laz -> the 4th channel falls back to the
    # z-tile-min proxy (same as the Test-Track4 demo). KPConvX geometry is fixed
    # to the trained values; any --grid/--chunk-xy passed is informational only.
    # ------------------------------------------------------------------------
    if INFER:
        if not infer_input:
            raise ValueError("--mode infer requires --infer-input <job_id>")
        if (grid is not None and grid != GRID) or (chunk_xy is not None and chunk_xy != CHUNK_XY):
            print(f"  [infer] note: KPConvX uses its trained geometry "
                  f"(grid={GRID}, chunk={CHUNK_XY}); --grid/--chunk-xy ignored.", flush=True)
        print("  [infer] note: uploaded scenes have no HAG laz -> HAG channel uses "
              "the z-tile-min proxy (folder inference is best-effort vs HAG-prepped data).",
              flush=True)
        net.eval()
        IDX_TO_ASPRS = np.array([2, 5, 6, 9, 17], dtype=np.int32)
        PALETTE = np.array([[139, 90, 43], [34, 160, 34], [200, 60, 60],
                            [40, 110, 220], [235, 225, 60]], dtype=np.int32)

        def _write_ply(path, pxyz, pred_idx, intensity=None):
            cols = [pxyz.astype(np.float32), PALETTE[pred_idx]]
            props = ("property float x\nproperty float y\nproperty float z\n"
                     "property uchar red\nproperty uchar green\nproperty uchar blue\n")
            fmt = ["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"]
            if intensity is not None:        # carry per-point intensity for the viewer
                cols.append(np.asarray(intensity, np.float32).reshape(-1, 1))
                props += "property float intensity\n"
                fmt.append("%.5f")
            header = "ply\nformat ascii 1.0\n" + f"element vertex {len(pxyz)}\n" + props + "end_header"
            np.savetxt(path, np.column_stack(cols), fmt=fmt, header=header, comments="")

        scenes = sorted(glob.glob(f"/datasets/_infer/{infer_input}/scenes/*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scenes under /datasets/_infer/{infer_input}/scenes")
        pred_dir = f"{run_dir}/predictions"
        with open(f"{run_dir}/run_config.json", "w") as f:
            json.dump({"backbone": "KPConvX-L-HAG", "mode": "infer", "weights": weights,
                       "infer_input": infer_input, "num_classes": NUM_CLASSES,
                       "class_names": CLASS_NAMES, "grid": GRID, "chunk_xy": CHUNK_XY,
                       "hag": "proxy_z_tile_min", "gpu": GPU_TYPE,
                       "scenes": [os.path.basename(s) for s in scenes]}, f, indent=2)
        print(f"  [infer] labeling {len(scenes)} scene(s) -> {run_dir}/predictions", flush=True)
        for pc_path in scenes:
            name = os.path.splitext(os.path.basename(pc_path))[0]
            t0 = time.time()
            z = np.load(pc_path)
            xyz = z["xyz"].astype(np.float32)
            # convert_infer_job already p95-normalized intensity to [0,2].
            intensity_n = (z["intensity"].astype(np.float32) if "intensity" in z
                           else np.full(len(xyz), 0.5, np.float32))
            ret_num = (z["return_number"].astype(np.float32) if "return_number" in z
                       else (z["ret_num"].astype(np.float32) if "ret_num" in z
                             else np.zeros(len(xyz), np.float32)))
            pred = _predict_points(xyz, intensity_n, ret_num)
            _write_ply(f"{pred_dir}/{name}_pred.ply", xyz, pred, intensity_n)
            np.savetxt(f"{pred_dir}/{name}_pred_CLS.txt", IDX_TO_ASPRS[pred], fmt="%d")
            print(f"  [infer] {name}: {len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
        outputs_volume.commit()
        print(f"  [infer] done — predictions in runs/{run_id}/predictions", flush=True)
        return

    metrics_csv = f"{run_dir}/metrics.csv"
    if not os.path.exists(metrics_csv):     # keep prior rows when resuming
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "train_acc", "train_iou", "lr",
                "sec_per_epoch", "gpu_mem_mb",
            ])

    # ------------------------------------------------------------------------
    # Periodic held-out validation (eval mode, no_grad — weights untouched).
    # Appends to val_metrics.csv and commits the volume, so progress can be
    # watched mid-run with:  modal volume get <app>-outputs runs/<id>/val_metrics.csv
    # ------------------------------------------------------------------------
    val_csv = f"{run_dir}/val_metrics.csv"
    if not os.path.exists(val_csv):
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "val_acc", "val_miou"] +
                                   [f"iou_{n}" for n in CLASS_NAMES])

    # Seeded random subset. NOT a strided slice: tiles sort scene-major with y
    # innermost, so [::k] picks the same two y-stripes of every scene — a
    # spatially biased (optimistic) sample. Seeded choice is representative
    # and still identical across epochs.
    _rs = np.random.RandomState(0)
    val_subset = [val_tiles[i] for i in sorted(_rs.choice(
        len(val_tiles), min(VAL_SUBSET, len(val_tiles)), replace=False))]

    def quick_val(ep):
        net.eval()
        inter = np.zeros(NUM_CLASSES, np.int64)
        union = np.zeros(NUM_CLASSES, np.int64)
        correct = total = 0
        t0 = time.time()
        with torch.no_grad():
            for tile in val_subset:
                s = sample_tile(tile, training=False)
                if s is None:
                    continue
                cxyz, feat, lab = s
                try:
                    batch, _ = make_kp_pack([(cxyz, feat, None)])
                    pred = net(batch).argmax(-1).cpu().numpy()
                except Exception:
                    continue
                v = lab >= 0
                correct += int((pred[v] == lab[v]).sum())
                total   += int(v.sum())
                for c in range(NUM_CLASSES):
                    inter[c] += int(((pred == c) & (lab == c) & v).sum())
                    union[c] += int((((pred == c) | (lab == c)) & v).sum())
        with np.errstate(invalid="ignore"):
            ious = inter / np.maximum(union, 1)
        acc = correct / max(total, 1)
        miou = float(ious.mean())
        with open(val_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{acc:.4f}", f"{miou:.4f}"] +
                                   [f"{x:.4f}" for x in ious])
        print(f"  [val @ ep {ep:3d}] acc={acc:.4f} mIoU={miou:.4f}  " +
              "  ".join(f"{n}={x:.3f}" for n, x in zip(CLASS_NAMES, ious)) +
              f"  ({time.time()-t0:.0f}s)", flush=True)
        outputs_volume.commit()

    # ------------------------------------------------------------------------
    # Train loop — KPConvX recipe: EPOCH_STEPS optimizer steps of
    # PACK_N tiles/forward x ACCUM accumulated forwards.
    # ------------------------------------------------------------------------
    LOG_EVERY = 50
    print(f"  starting at epoch {start_epoch}, up to {N_EPOCHS}, "
          f"{EPOCH_STEPS} steps/epoch, pack {PACK_N} x accum {ACCUM}", flush=True)
    t_run = time.time()
    for ep in range(start_epoch, N_EPOCHS):
        cur_lr = lr_at(ep)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        net.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        ep_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        ep_union = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_ep = time.time()
        n_steps = n_fwd = n_failed = 0
        print(f"  ep {ep:3d} starting (lr={cur_lr:.2e})…", flush=True)
        for step in range(EPOCH_STEPS):
            optim.zero_grad()
            n_ok = 0
            for _ in range(ACCUM):
                samples = []
                while len(samples) < PACK_N:
                    s = sample_tile(pick_train_tile(), training=True)
                    if s is not None:
                        samples.append(s)
                try:
                    batch, lab_t = make_kp_pack(samples)
                    logits = net(batch)
                    loss = seg_loss(logits, lab_t) / ACCUM
                    # Defense in depth: skip any non-finite loss (NaN/Inf from an
                    # all-ignored pack or a numerical blow-up) before backward, so
                    # one bad gradient can't poison the weights — as RandLA does.
                    if not torch.isfinite(loss):
                        n_failed += 1
                        continue
                    loss.backward()
                    n_ok += 1; n_fwd += 1
                    ep_loss += loss.item() * ACCUM
                    pred = logits.argmax(-1)
                    m = lab_t >= 0
                    ep_correct += (pred[m] == lab_t[m]).sum().item()
                    ep_total   += int(m.sum())
                    for c in range(NUM_CLASSES):
                        ep_inter[c] += ((pred == c) & (lab_t == c)).sum().item()
                        ep_union[c] += (((pred == c) | (lab_t == c)) & m).sum().item()
                except Exception as e:
                    n_failed += 1
                    if n_failed == 1:
                        print(f"  forward failed (first occurrence, step {step}): {e}",
                              flush=True)
                        traceback.print_exc()
            if n_ok:
                torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
                optim.step()
                n_steps += 1
                if n_steps % LOG_EVERY == 0:
                    print(f"    ep {ep:3d} step {n_steps:4d}: "
                          f"loss={ep_loss/max(n_fwd,1):.4f}", flush=True)
        if n_steps == 0:
            raise RuntimeError(f"epoch {ep}: 0 optimizer steps — {n_failed} failed forwards.")
        if n_failed:
            print(f"  ep {ep:3d} note: {n_failed} failed forwards", flush=True)
        sec_per_epoch = time.time() - t_ep
        train_acc = ep_correct / max(ep_total, 1)
        with np.errstate(invalid="ignore"):
            train_iou = float(np.mean(ep_inter / np.maximum(ep_union, 1)))
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, f"{ep_loss/max(n_fwd,1):.6f}", f"{train_acc:.4f}",
                f"{train_iou:.4f}", f"{cur_lr:.6e}", f"{sec_per_epoch:.2f}", f"{gpu_mem:.1f}",
            ])
        print(f"  ep {ep:3d}: loss={ep_loss/max(n_fwd,1):.4f} acc={train_acc:.4f} "
              f"miou={train_iou:.4f} lr={cur_lr:.2e} s/epoch={sec_per_epoch:.1f}", flush=True)
        if (ep + 1) % CHECKPOINT_GAP == 0:
            torch.save({"model": net.state_dict(), "optim": optim.state_dict(),
                        "epoch": ep},
                       f"{run_dir}/checkpoints/ep{ep:03d}.pth")
            outputs_volume.commit()
        if (ep + 1) % VAL_EVERY == 0 or ep == N_EPOCHS - 1:
            quick_val(ep)

    if not EVAL_ONLY:
        torch.save({"model": net.state_dict(), "epoch": N_EPOCHS - 1},
                   f"{run_dir}/final_model.pth")

    # ------------------------------------------------------------------------
    # Test — train-holdout val + Validate-Track4 (with GT)
    # ------------------------------------------------------------------------
    net.eval()
    def evaluate(scene_list, split_dir, label):
        """Final eval scored on the ORIGINAL raw points (official protocol).
        Per scene: run the model over its overlapping cached tiles (stride 50,
        each point in up to 4 tiles), sum CENTER-WEIGHTED SOFTMAX votes per 2 m
        voxel and take the argmax, then propagate each voxel's prediction to
        every raw point by nearest neighbour and score against the raw GT.
        Reprojecting to raw points removes the 2 m voxel-resolution bias and the
        arbitrary first-duplicate voxel-GT of the old voxel scoring."""
        t_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_union = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_gt    = np.zeros(NUM_CLASSES, dtype=np.int64)
        correct = total = 0
        n_scenes = n_skipped_tiles = n_skipped_scenes = 0
        t_test = time.time()
        with torch.no_grad():
            for name, pc_path, cls_path in scene_list:
                tiles = sorted(glob.glob(f"{split_dir}/{name}_x*.npz"))
                if not tiles:
                    n_skipped_scenes += 1; continue
                keys_l, log_l, xyz_l = [], [], []
                for tile in tiles:
                    z = np.load(tile)
                    xyz = z["xyz"]
                    if len(xyz) < 32:
                        continue
                    feat = build_feat(xyz, z["intensity"], z["ret_num"], _hag_of(z, xyz))
                    cxyz = (xyz - xyz.mean(0)).astype(np.float32)
                    try:
                        batch, _ = make_kp_pack([(cxyz, feat, None)])
                        lg = net(batch).cpu().numpy().astype(np.float32)
                    except Exception:
                        n_skipped_tiles += 1; continue
                    # Soft votes tapered toward the tile border (truncated context).
                    e = np.exp(lg - lg.max(1, keepdims=True))
                    prob = e / e.sum(1, keepdims=True)
                    cxy = (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
                    d = np.abs(xyz[:, :2] - cxy).max(1)
                    wgt = np.clip(1.0 - d / (CHUNK_XY / 2.0), 0.05, 1.0) ** 2
                    keys_l.append(np.floor(xyz / GRID).astype(np.int64))
                    log_l.append((prob * wgt[:, None]).astype(np.float32))
                    xyz_l.append(xyz.astype(np.float32))
                if not keys_l:
                    n_skipped_scenes += 1; continue
                K = np.concatenate(keys_l); L = np.concatenate(log_l)
                P = np.concatenate(xyz_l)
                uniq, first, inv = np.unique(K, axis=0, return_index=True, return_inverse=True)
                votes = np.zeros((len(uniq), NUM_CLASSES), np.float64)
                np.add.at(votes, inv, L)
                pred_u  = votes.argmax(1)
                rep_xyz = P[first]                      # one representative coord per voxel
                # Reproject voxel predictions onto the raw scene cloud + raw GT.
                try:
                    raw_xyz, _, _, raw_lab = load_ieee(pc_path, cls_path)
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped_scenes += 1; continue
                _, nn = cKDTree(rep_xyz).query(raw_xyz)
                raw_pred = pred_u[nn]
                v = raw_lab >= 0
                rp, rl = raw_pred[v], raw_lab[v]
                correct += int((rp == rl).sum()); total += int(v.sum())
                for c in range(NUM_CLASSES):
                    t_inter[c] += int(((rp == c) & (rl == c)).sum())
                    t_union[c] += int(((rp == c) | (rl == c)).sum())
                    t_gt[c]    += int((rl == c).sum())
                n_scenes += 1
        with np.errstate(invalid="ignore"):
            iou_per = t_inter / np.maximum(t_union, 1)
        gt_counts = [int(x) for x in t_gt.tolist()]
        present = [c for c in range(NUM_CLASSES) if gt_counts[c] > 0]
        present_iou = [float(iou_per[c]) for c in present]
        present_mIoU = float(np.mean(present_iou)) if present_iou else 0.0
        m = {
            "overall_acc": correct / max(total, 1),
            "overall_mIoU": float(np.mean(iou_per)),
            "present_classes_mIoU": present_mIoU,
            "per_class_iou": {CLASS_NAMES[c]: float(iou_per[c]) for c in range(NUM_CLASSES)},
            "per_class_gt_count": {CLASS_NAMES[c]: gt_counts[c] for c in range(NUM_CLASSES)},
            "present_classes": [CLASS_NAMES[c] for c in present],
            "absent_classes":  [CLASS_NAMES[c] for c in range(NUM_CLASSES) if gt_counts[c] == 0],
            "total_test_seconds": time.time() - t_test,
            "num_scenes": n_scenes,
            "num_raw_points_scored": int(total),
            "skipped_tiles": n_skipped_tiles,
            "skipped_scenes": n_skipped_scenes,
            "scored_on": "raw_points",
            "voted_overlap": True,
            "vote_weighting": "center_tapered_softmax",
            "reprojection": "nearest_voxel_representative_to_raw",
        }
        print(f"  [{label}] acc={m['overall_acc']:.4f}  mIoU(5-way)={m['overall_mIoU']:.4f}  "
              f"mIoU(present {len(present)})={m['present_classes_mIoU']:.4f}  "
              f"absent={m['absent_classes']}  raw_pts={total:,}  "
              f"skipped(tiles={n_skipped_tiles},scenes={n_skipped_scenes})", flush=True)
        return m

    print("  evaluating on train-holdout val + Validate-Track4 test…", flush=True)
    test_metrics = {
        "val":  evaluate(val_list,  f"{PREP_DIR}/val",  "val"),
        "test": evaluate(test_list, f"{PREP_DIR}/test", "test"),
        "val_scenes":  [n for n, _, _ in val_list],
        "test_scenes": [n for n, _, _ in test_list],
    }
    with open(f"{run_dir}/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"  total wall-clock: {(time.time() - t_run)/3600:.2f} h")
    outputs_volume.commit()

    # ------------------------------------------------------------------------
    # Inference demo: label the first N_PREDICT Test-Track4 scenes (no GT).
    # ------------------------------------------------------------------------
    IDX_TO_ASPRS = np.array([2, 5, 6, 9, 17], dtype=np.int32)
    PALETTE = np.array([[139, 90, 43], [34, 160, 34], [200, 60, 60],
                        [40, 110, 220], [235, 225, 60]], dtype=np.int32)

    def _write_cls(path, pred_idx):
        np.savetxt(path, IDX_TO_ASPRS[pred_idx], fmt="%d")

    def _write_ply(path, xyz, pred_idx, intensity=None):
        cols = [xyz.astype(np.float32), PALETTE[pred_idx]]
        props = ("property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fmt = ["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"]
        if intensity is not None:            # carry per-point intensity for the viewer
            cols.append(np.asarray(intensity, np.float32).reshape(-1, 1))
            props += "property float intensity\n"
            fmt.append("%.5f")
        header = "ply\nformat ascii 1.0\n" + f"element vertex {len(xyz)}\n" + props + "end_header"
        np.savetxt(path, np.column_stack(cols), fmt=fmt, header=header, comments="")

    def _predict_scene(pc_path):
        pc = np.loadtxt(pc_path, delimiter=",")                 # float64 (full precision)
        # Per-scene origin offset before float32 cast (precision; see load_ieee).
        xyz = (pc[:, :3] - np.floor(pc[:, :3].min(0))).astype(np.float32)
        intensity = pc[:, 3].astype(np.float32)
        ret_num = pc[:, 4].astype(np.float32) if pc.shape[1] >= 5 else np.zeros(len(pc), np.float32)
        i_p95 = max(float(np.percentile(intensity, 95)), 1.0)
        intensity_n = np.clip(intensity / i_p95, 0.0, 2.0)
        # Test-Track4 has no HAG laz -> _predict_points uses the z-tile-min proxy.
        return xyz, _predict_points(xyz, intensity_n, ret_num), intensity_n

    try:
        net.eval()
        pred_dir = f"{run_dir}/predictions"
        os.makedirs(pred_dir, exist_ok=True)
        scenes = [] if EVAL_ONLY else sorted(glob.glob(f"{PRED_PC_DIR}/*_PC3.txt"))[:N_PREDICT]
        print(f"  [predict] labeling {len(scenes)} Test-Track4 scene(s) -> {pred_dir}", flush=True)
        for pc_path in scenes:
            name = os.path.basename(pc_path).replace("_PC3.txt", "")
            t0 = time.time()
            xyz, pred, inten = _predict_scene(pc_path)
            _write_cls(f"{pred_dir}/{name}_pred_CLS.txt", pred)
            _write_ply(f"{pred_dir}/{name}_pred.ply", xyz, pred, inten)
            print(f"  [predict] {name}: {len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
        outputs_volume.commit()
    except Exception as e:
        print(f"  [predict] skipped (model is saved): {e}", flush=True)
        traceback.print_exc()


@app.local_entrypoint()
def main(mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None, grid: Optional[float] = None,
         chunk_xy: Optional[float] = None):
    what = {"eval": "eval-only re-score", "infer": f"infer({weights})"}.get(mode, "train")
    print(f"Launching {APP_NAME} [{what}] on {GPU_TYPE} for up to {TIMEOUT_HOURS}h.")
    train_kpconvx.remote(mode=mode, weights=weights, infer_input=infer_input,
                         grid=grid, chunk_xy=chunk_xy)
