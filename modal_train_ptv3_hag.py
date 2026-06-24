"""
Modal training script for PointTransformerV3 on IEEE GRSS 2019 DFC Track 4 —
HAG variant.

This is the HAG twin of modal_train_ptv3.py. Identical in every way except it
appends a real PDAL HeightAboveGround feature (computed by the trainer_gui
Pretraining tab: SMRF ground-classify -> hag_nn) as an extra input channel:
in_channels 6 -> 7 ([xyz, rgb(=intensity grayscale), HAG]). HAG is read per
point from /data/IEEE/HAG/{Train,Validate}/<scene>_PC3.laz and paired to the
raw _PC3.txt points. Run it head-to-head against modal_train_ptv3.py to measure
HAG's contribution.

The default (no --dataset) path trains on the same raw IEEE Track 4 airborne
LiDAR — and uses the same ASPRS->index label map, per-scene p95 intensity
normalization, and scene holdout — as modal_train_kpconvx_cold.py, so PTv3 can
be compared head-to-head against KPConvX/KPConv on identical data. (It was
previously wired to STPLS3D; switched 2026-06-15.)

Prereq: upload the HAGTEST output (Pretraining tab) to the ieee-data volume:
    modal volume put ieee-data "C:/Users/OrionHoch/Desktop/HAGTEST/Train"    /IEEE/HAG/Train
    modal volume put ieee-data "C:/Users/OrionHoch/Desktop/HAGTEST/Validate" /IEEE/HAG/Validate
The --dataset (canonical trainer_gui) path has no HAG laz, so it falls back to a
z-scene-min proxy for the 7th channel (keeps in_channels fixed at 7).

Uses the standalone PTv3 model.py from testSem/PointTransformerV3 directly
(no full Pointcept install). The fast path uses FlashAttention; if the GPU
doesn't support it the script falls back to PTv3 with enable_flash=False.

----------------------------------------------------------------------------
ACCURACY / ANTI-OVERFITTING RECIPE (2026-06-15)
----------------------------------------------------------------------------
This script carries the same generalization machinery proven out in
modal_train_kpconvx_cold.py, reconciled with PointTransformerV3's *own*
published outdoor-LiDAR recipe (Wu et al., CVPR 2024, supp. Tab. 12-16):

  - DATA AUGMENTATION (was completely absent — the biggest overfit driver):
    PTv3's outdoor suite — full z-rotation, small x/y tilt (±π/64), isotropic
    scale [0.9,1.1], per-axis flip (p=0.5), gaussian jitter (σ=0.005, clip 0.02).
  - LOSS = weighted CE (+ optional focal / label smoothing) + Lovász-Softmax.
    PTv3's outdoor loss is literally CE + Lovász (equal weight); Lovász is a
    differentiable mIoU surrogate that weights every class equally — the same
    term added to the KPConvX recipe.
  - CLASS BALANCE: inverse-sqrt-frequency class weights (mean-normalized,
    capped) + rare-class tile oversampling (rare classes auto-detected from the
    training-set frequency histogram for canonical datasets).
  - OPTIMIZER tuned to PTv3's paper: AdamW lr 2e-3, weight_decay 5e-3 (NOT the
    0.05 it had — 10x too strong), drop_path 0.3 stochastic depth, ~2-epoch
    warmup via OneCycle.
  - PERIODIC HELD-OUT VALIDATION every VAL_EVERY epochs (eval mode, no grad) ->
    val_metrics.csv, committed mid-run so the train/val gap is watchable live.
  - OVERLAP-VOTING EVAL: val/test tiles cut at stride CHUNK_XY/2 so each point
    is covered by up to 4 tiles; per-voxel center-tapered softmax votes are
    summed before argmax (the old eval took a single-tile argmax and even
    random-cropped large tiles, dropping points).
  - RESUMABLE checkpoints (model + optimizer) every CHECKPOINT_GAP epochs; set
    RESUME=True to continue the latest matching run after a timeout.

Usage:
    modal volume create ieee-data
    modal volume put ieee-data "C:/Users/OrionHoch/Desktop/LabledDatasets/IEEE" /IEEE
    modal run --detach modal_train_ptv3.py
    modal app logs ptv3-ieee

----------------------------------------------------------------------------
Training-terminal integration: running with no flags trains on IEEE Track 4.
Extra flags (see modal_train_ptv3_warm.py for details):

  --dataset NAME                          canonical trainer_gui dataset on the
                                          terminal-datasets volume
  --grid / --chunk-xy / --epochs / --batch / --steps-per-epoch / --predict-n
  --mode infer --weights runs/<id>/final_model.pth --infer-input <job_id>

GPU type / timeout come from TT_GPU / TT_TIMEOUT_HOURS env vars.
"""

import os
from typing import Optional

import modal

# ============================================================================
# Configuration
# ============================================================================
APP_NAME      = "ptv3-ieee-hag"
GPU_TYPE      = os.environ.get("TT_GPU", "A100")
N_EPOCHS      = 100            # PTv3 outdoor recipe (Wu et al., CVPR 2024): ~50
                             # epochs x 500 steps. Override per-run with --epochs;
                             # for a cheap end-to-end check pass --epochs 2.
BATCH_SIZE    = 4
TIMEOUT_HOURS = int(os.environ.get("TT_TIMEOUT_HOURS", "24"))

# IEEE GRSS 2019 DFC Track 4 (default, no --dataset). Same data contract as
# modal_train_kpconvx_cold.py so the two are directly comparable: ASPRS code ->
# contiguous class index (class 0 ignored), 5 classes.
NUM_CLASSES   = 5
CLASS_NAMES   = ["Ground", "Trees", "Building", "Water", "Bridge"]
LABEL_MAP     = {0: -1, 2: 0, 5: 1, 6: 2, 9: 3, 17: 4}
GRID_SIZE     = 0.5            # voxel grid (m). PTv3's 0.05 m outdoor default is
                               # for dense near-sensor LiDAR; IEEE airborne is
                               # ~2 pts/m² (~0.7 m spacing), so a 5 cm grid is a
                               # no-op downsample AND leaves PTv3's sparse
                               # positional-encoding conv with empty kernels (no
                               # point falls within a few voxels of another).
                               # 0.5 m ≈ the actual point spacing. Override --grid.
USE_FLASH_ATTN = False  # ponytail: flash serialized-attn patch-gather OOBs mid-train
                        # (the "device-side assert" crash). Off + patch_size 128 is the
                        # documented Pointcept fix. Re-enable w/ patch 128 if you want speed.
HOLDOUT_SEED   = 42
N_VAL_HOLDOUT  = 10            # held-out train scenes for val (matches KPConvX cold)

# ----------------------------------------------------------------------------
# Regularization / optimizer — PTv3's published outdoor-LiDAR recipe
# (Wu et al., CVPR 2024, supplementary Tab. 13). The original script used a
# generic AdamW + OneCycle with weight_decay 0.05; the paper uses 5e-3.
# ----------------------------------------------------------------------------
DROP_PATH     = 0.3      # stochastic depth (PTv3 outdoor default)
BASE_LR       = 2e-3     # PTv3 outdoor base lr (also OneCycle peak here)
WEIGHT_DECAY  = 5e-3     # PTv3 outdoor AdamW wd (NOT 0.05)
WARMUP_PCT    = 0.04     # ~2 epochs of 50 warmup (PTv3 uses a 2-epoch warmup)
GRAD_CLIP     = 1.0

# Augmentation — PTv3 outdoor suite. The original to_ptv3_batch did NONE; for a
# transformer on a handful of large scenes that is the dominant overfit cause.
AUG_ENABLE       = True
AUG_ROT_Z        = 1.0          # z angle ~ U(-pi, pi) * AUG_ROT_Z (full yaw)
AUG_ROT_XY       = 1.0 / 64.0   # x,y tilt ~ U(-pi, pi) * this (gentle ±~2.8 deg)
AUG_SCALE_MIN    = 0.9
AUG_SCALE_MAX    = 1.1
AUG_FLIP_P       = 0.5          # per-axis (x, y) coordinate flip probability
AUG_JITTER_SIGMA = 0.005        # gaussian per-point noise (m)
AUG_JITTER_CLIP  = 0.02         # clip jitter to +/- this (m)

# Loss = weighted CE (+ optional focal / label smoothing) + Lovász-Softmax,
# mirroring modal_train_kpconvx_cold.py. PTv3's own outdoor loss is CE + Lovász.
CLASS_WEIGHTING  = True
WEIGHT_BETA      = 0.5     # 0.5 = inverse-SQRT-frequency (sub-linear, stable)
WEIGHT_CAP       = 5.0     # clamp each weight to [1/CAP, CAP] after mean-norm
LABEL_SMOOTH     = 0.0     # PTv3 leans on Lovász, not smoothing (KPConvX used 0.2)
LOVASZ_WEIGHT    = 1.0     # total = <pointwise> + LOVASZ_WEIGHT * lovasz_softmax
USE_FOCAL        = False   # True -> alpha-balanced focal instead of weighted CE
FOCAL_GAMMA      = 2.0

# Rare-class tile oversampling. RARE_CLASSES=None auto-detects rare classes from
# the train-set frequency histogram (classes below RARE_FREQ_FRAC x median freq).
RARE_OVERSAMPLE  = True
RARE_CLASSES     = None
RARE_FREQ_FRAC   = 0.5
RARE_TILE_PROB   = 0.25    # P(draw the next train tile from a rare-class tile)

# Periodic held-out validation + checkpoint/resume cadence.
VAL_EVERY        = 10      # FULL voted eval over the combined eval set every N
                          # epochs (no weight updates). Heavier than the old
                          # subset val pass — raise this if it costs too much.
CHECKPOINT_GAP   = 3       # checkpoint (model + optimizer) frequency, epochs
RESUME           = False   # force-resume the latest matching run (see AUTO_RESUME)
AUTO_RESUME      = False    # auto-continue an unfinished run (no DONE marker) on
                          # relaunch / Modal auto-retry, so an intermittent crash
                          # never loses the run — only epochs since last checkpoint

DATA_ROOT      = "/data/IEEE"
TRAIN_PC_DIR   = f"{DATA_ROOT}/Train-Track4/Track4"            # *_PC3.txt (x,y,z,intensity,ret)
TRAIN_CLS_DIR  = f"{DATA_ROOT}/Train-Track4-Truth/Track4-Truth"  # *_CLS.txt (ASPRS codes)
TEST_PC_DIR    = f"{DATA_ROOT}/Validate-Track4/Track4"
TEST_CLS_DIR   = f"{DATA_ROOT}/Validate-Track4-Truth"
PRED_PC_DIR    = f"{DATA_ROOT}/Test-Track4/Test-Track4"        # no GT — for the predict demo
# Per-point HeightAboveGround from the Pretraining tab (HAGTEST upload). Train +
# val-holdout scenes -> HAG/Train; the test split (Validate-Track4) -> HAG/Validate.
HAG_TRAIN_DIR  = f"{DATA_ROOT}/HAG/Train"
HAG_TEST_DIR   = f"{DATA_ROOT}/HAG/Validate"
PREP_DIR       = f"{DATA_ROOT}/prep/ptv3_ieee_hag_grid05_origin"

DATASETS_ROOT = "/datasets"   # terminal-datasets volume (trainer_gui canonical datasets)

# ============================================================================
# Image
# ============================================================================
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential", "cmake", "ninja-build", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.0",
        "torchvision==0.20.0",
        "numpy<2.0",
        "scipy",
        "scikit-learn",
        "plyfile",
        "tqdm",
        "tensorboard",
        "addict",
        "einops",
        "timm",
        "pandas<3",
        "laspy",
        "lazrs",          # LAZ backend so laspy can read the HAG .laz scenes
        index_url="https://download.pytorch.org/whl/cu124",
        extra_index_url="https://pypi.org/simple",
    )
    .pip_install(
        "spconv-cu124",
        "torch-scatter",
        "torch-cluster",
        find_links="https://data.pyg.org/whl/torch-2.5.0+cu124.html",
    )
    .run_commands(
        # FlashAttention: install the prebuilt wheel matching this image exactly
        # (torch 2.5 / cu12 / cp310 / cxx11abiFALSE — PyTorch's pip wheels use the
        # pre-cxx11 ABI). Plain `pip install flash-attn` compiles from source and
        # reliably fails on debian_slim here; the matched wheel installs in
        # seconds. build_model still probes `import flash_attn` and falls back to
        # standard attention if it's ever absent.
        "pip install --no-deps "
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

image = image.add_local_dir(
    "C:/Users/OrionHoch/Desktop/testSem/PointTransformerV3",
    "/opt/ptv3",
    copy=True,
)
# model.py uses a package-relative import (`from .serialization import encode`),
# so it must be imported as `ptv3.model`, not top-level `model`. Make /opt/ptv3
# a package; we add /opt (its parent) to sys.path at runtime.
image = image.run_commands("touch /opt/ptv3/__init__.py")

data_volume     = modal.Volume.from_name("ieee-data",           create_if_missing=True)
outputs_volume  = modal.Volume.from_name(f"{APP_NAME}-outputs", create_if_missing=True)
datasets_volume = modal.Volume.from_name(
    os.environ.get("TT_DATASET_VOLUME", "terminal-datasets"), create_if_missing=True)


# ============================================================================
# Training function
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_volume, "/outputs": outputs_volume,
             DATASETS_ROOT: datasets_volume},
    cpu=8,
    memory=49152,
    timeout=TIMEOUT_HOURS * 3600,
    # Auto-restart the container on failure (e.g. an intermittent CUDA device-
    # side assert from spconv, which poisons the context and cannot be caught
    # in-process). Each retry auto-resumes from the latest checkpoint, so an
    # intermittent crash costs only the epochs since the last checkpoint.
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def train_ptv3(dataset: Optional[str] = None, grid: Optional[float] = None,
               epochs: Optional[int] = None, batch: Optional[int] = None,
               steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
               mode: str = "train", weights: Optional[str] = None,
               infer_input: Optional[str] = None, predict_n: Optional[int] = None):
    import os, sys, time, json, csv, glob, traceback
    from datetime import datetime
    import numpy as np
    import torch
    import laspy
    from plyfile import PlyData
    from scipy.spatial import cKDTree

    sys.path.insert(0, "/opt")          # so `import ptv3.model` resolves

    # ------------------------------------------------------------------------
    # HAG: per-point HeightAboveGround read from the Pretraining-tab .laz files
    # (SMRF -> hag_nn) and paired to the raw scene points. PDAL preserves point
    # order, so index-pairing works when counts match (verified on a sample);
    # otherwise nearest-neighbor pairing on xyz.
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

    # --- resolve config: CLI args override the module defaults ---------------
    GRID_SIZE   = grid if grid is not None else globals()["GRID_SIZE"]
    N_EPOCHS    = epochs if epochs is not None else globals()["N_EPOCHS"]
    BATCH_SIZE  = batch if batch is not None else globals()["BATCH_SIZE"]
    STEPS       = steps_per_epoch if steps_per_epoch is not None else 500
    CHUNK_XY    = chunk_xy if chunk_xy is not None else 50.0
    STRIDE      = CHUNK_XY / 2.0
    N_PREDICT   = predict_n if predict_n is not None else 1
    NUM_CLASSES = globals()["NUM_CLASSES"]
    CLASS_NAMES = None

    # IEEE ASPRS-code -> contiguous-index lookup (used by the default-path loaders).
    LABEL_MAP = globals()["LABEL_MAP"]
    LABEL_LUT = np.full(256, -1, dtype=np.int64)
    for raw, mapped in LABEL_MAP.items():
        LABEL_LUT[int(raw)] = mapped

    ds_root = f"{DATASETS_ROOT}/{dataset}" if dataset else None
    if ds_root:
        meta_path = f"{ds_root}/dataset_meta.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{meta_path} not found — upload the dataset "
                                    f"with the trainer_gui app first.")
        with open(meta_path) as f:
            ds_meta = json.load(f)
        NUM_CLASSES = int(ds_meta["num_classes"])
        CLASS_NAMES = list(ds_meta["class_names"])
        PREP_DIR = f"{ds_root}/prep/ptv3_cold_hag_chunk{int(CHUNK_XY)}"
    else:
        CLASS_NAMES = list(globals()["CLASS_NAMES"])   # IEEE Track 4 default
        PREP_DIR = globals()["PREP_DIR"]

    # --- Preprocessing ------------------------------------------------------
    def load_ply(path):
        ply = PlyData.read(path)
        v = ply["vertex"]
        names = set(p.name for p in v.properties)
        xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
        if {"red", "green", "blue"}.issubset(names):
            rgb = np.stack([v["red"], v["green"], v["blue"]], -1).astype(np.float32)
        else:
            rgb = np.full((xyz.shape[0], 3), 128.0, dtype=np.float32)
        for key in ("scalar_label", "label", "class"):
            if key in names:
                lab = np.asarray(v[key]).astype(np.int64); break
        else:
            lab = np.zeros(xyz.shape[0], dtype=np.int64)
        return xyz, rgb, lab

    def load_canonical(npz_path):
        """Canonical trainer_gui scene -> the (xyz, rgb, lab) tuple load_ply produces.
        Missing rgb falls back to mid-gray (same as load_ply)."""
        z = np.load(npz_path)
        xyz = z["xyz"].astype(np.float32)
        if "rgb" in z:
            rgb = z["rgb"].astype(np.float32)
        elif "intensity" in z:
            rgb = np.repeat((z["intensity"].astype(np.float32) * 255.0)[:, None], 3, axis=1)
        else:
            rgb = np.full((len(xyz), 3), 128.0, dtype=np.float32)
        lab = z["label"].astype(np.int64) if "label" in z \
            else np.full(len(xyz), -1, np.int64)
        return xyz, rgb, lab

    def _loadtxt_fast(path, delimiter=None):
        """pandas' C parser reads big IEEE .txt scenes ~20-50x faster than
        np.loadtxt — the eval-time raw-reload bottleneck (each multi-million-point
        scene took minutes under loadtxt). Falls back to loadtxt if pandas is
        unavailable or the parse fails. delimiter=None -> whitespace-separated."""
        try:
            import pandas as pd
            if delimiter is None:
                df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
            else:
                df = pd.read_csv(path, header=None, sep=delimiter)
            return df.to_numpy()
        except Exception:
            return np.loadtxt(path, delimiter=delimiter)

    def _ieee_xyz_rgb(pc_path):
        """IEEE PC3.txt -> (xyz, rgb) where rgb carries per-scene p95-normalized
        intensity as grayscale (the one cue that separates water; same robust
        normalization as the KPConvX cold script). Intensity rides the 3 color
        channels so the model's in_channels=6 (coord+color) is unchanged."""
        pc = _loadtxt_fast(pc_path, ",")                        # float64 (full precision)
        # Subtract a per-scene origin before the float32 cast: projected (UTM)
        # coords otherwise quantize to ~0.25-0.5 m on the northing axis (float32
        # has ~7 sig digits). The offset is deterministic (floor of the per-scene
        # min), so cached tiles and the eval-time raw reload share one frame.
        xyz = (pc[:, :3] - np.floor(pc[:, :3].min(0))).astype(np.float32)
        intensity = pc[:, 3].astype(np.float32) if pc.shape[1] > 3 \
            else np.zeros(len(pc), np.float32)
        i_p95 = max(float(np.percentile(intensity, 95)), 1.0)
        i_norm = np.clip(intensity / i_p95, 0.0, 1.0)
        rgb = np.repeat((i_norm * 255.0)[:, None], 3, axis=1).astype(np.float32)
        return xyz, rgb

    def load_ieee(pc_path, cls_path):
        """Labeled IEEE scene -> (xyz, rgb, lab) with ASPRS codes mapped to
        contiguous indices (unmapped/ignored -> -1)."""
        xyz, rgb = _ieee_xyz_rgb(pc_path)
        lab_raw = _loadtxt_fast(cls_path).reshape(-1).astype(np.int64)
        if len(lab_raw) != len(xyz):
            raise ValueError(f"point/label mismatch in {pc_path}: "
                             f"{len(xyz)} pts vs {len(lab_raw)} labels")
        lab = LABEL_LUT[np.clip(lab_raw, 0, 255)].astype(np.int64)
        return xyz, rgb, lab

    def load_scene(path):
        # canonical .npz -> trainer_gui loader; IEEE PC3.txt (no GT) -> intensity
        # as grayscale + lab=-1 (for the predict demo); else PLY.
        if path.endswith(".npz"):
            return load_canonical(path)
        if path.endswith("_PC3.txt"):
            xyz, rgb = _ieee_xyz_rgb(path)
            return xyz, rgb, np.full(len(xyz), -1, np.int64)
        return load_ply(path)

    def tile_and_save(src_paths, out_dir, chunk_xy, stride):
        os.makedirs(out_dir, exist_ok=True)
        for fi, src in enumerate(src_paths):
            scene = os.path.splitext(os.path.basename(src))[0]
            t0 = time.time()
            try:
                xyz, rgb, lab = load_scene(src)
            except Exception as e:
                print(f"  skip {src}: {e}", flush=True); continue
            print(f"    [{fi+1}/{len(src_paths)}] {scene}: {len(xyz):,} pts "
                  f"loaded in {time.time()-t0:.1f}s, tiling…", flush=True)
            mins, maxs = xyz[:, :2].min(0), xyz[:, :2].max(0)
            n_tiles = 0
            for x0 in np.arange(mins[0], maxs[0], stride):
                for y0 in np.arange(mins[1], maxs[1], stride):
                    m = ((xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + chunk_xy) &
                         (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + chunk_xy))
                    if m.sum() < 2048: continue
                    out_path = f"{out_dir}/{scene}_x{int(x0)}_y{int(y0)}.npz"
                    np.savez_compressed(out_path,
                        xyz=xyz[m].astype(np.float32),
                        rgb=rgb[m].astype(np.uint8),
                        lab=lab[m].astype(np.int32))
                    n_tiles += 1
            print(f"      -> {n_tiles} tiles", flush=True)

    def tile_ieee_scene(name, pc_path, cls_path, out_dir, chunk_xy, stride, hag_dir, min_pts=512):
        """Tile one labeled IEEE scene into xyz/rgb/hag/lab npz windows. min_pts is
        deliberately low (vs the 2048 ply/canonical floor): water absorbs LiDAR,
        so pure-water tiles are sparse and a high cut would delete them — the
        same lesson the KPConvX cold log records."""
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        try:
            xyz, rgb, lab = load_ieee(pc_path, cls_path)
            hag = load_hag(name, hag_dir, xyz)
        except Exception as e:
            print(f"  skip {pc_path}: {e}", flush=True); return
        print(f"    {name}: {len(xyz):,} pts loaded in {time.time()-t0:.1f}s, "
              f"HAG {hag.min():.1f}..{hag.max():.1f}m, tiling…", flush=True)
        mins, maxs = xyz[:, :2].min(0), xyz[:, :2].max(0)
        n_tiles = 0
        for x0 in np.arange(mins[0], maxs[0], stride):
            for y0 in np.arange(mins[1], maxs[1], stride):
                m = ((xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + chunk_xy) &
                     (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + chunk_xy))
                if m.sum() < min_pts:
                    continue
                np.savez_compressed(f"{out_dir}/{name}_x{int(x0)}_y{int(y0)}.npz",
                    xyz=xyz[m].astype(np.float32),
                    rgb=rgb[m].astype(np.uint8),
                    hag=hag[m].astype(np.float32),
                    lab=lab[m].astype(np.int32))
                n_tiles += 1
        print(f"      -> {n_tiles} tiles", flush=True)

    def ensure_prep():
        # Per-scene idempotency via prefix-match on existing tiles.
        os.makedirs(f"{PREP_DIR}/train", exist_ok=True)
        os.makedirs(f"{PREP_DIR}/test",  exist_ok=True)
        print(f"  ensuring preprocessed cache -> {PREP_DIR}", flush=True)
        any_new = [False]
        def already_tiled(out_dir, scene):
            return bool(glob.glob(f"{out_dir}/{scene}_x*.npz"))
        def tile_remaining(src_paths, out_dir, chunk, stride):
            for src in src_paths:
                scene = os.path.splitext(os.path.basename(src))[0]
                if already_tiled(out_dir, scene): continue
                tile_and_save([src], out_dir, chunk, stride)
                any_new[0] = True
        if ds_root:
            train_paths = sorted(glob.glob(f"{ds_root}/train/*.npz"))
            test_paths  = sorted(glob.glob(f"{ds_root}/val/*.npz"))
            if not train_paths:
                raise FileNotFoundError(f"No canonical scenes under {ds_root}/train")
            print(f"  [train] {len(train_paths)} canonical scenes", flush=True)
            tile_remaining(train_paths, f"{PREP_DIR}/train", CHUNK_XY, STRIDE)
            print(f"  [test] {len(test_paths)} canonical scenes", flush=True)
            # stride = STRIDE (not CHUNK_XY) so test tiles overlap and the final
            # eval can vote per-voxel over the up-to-4 covering tiles.
            tile_remaining(test_paths, f"{PREP_DIR}/test", CHUNK_XY, STRIDE)
        else:
            # IEEE Track 4: tile every train scene (the val holdout is carved out
            # later by scene name) + every Validate scene; both at stride STRIDE
            # so train tiles overlap for coverage and val/test tiles can be voted.
            def _ieee_remaining(pc_dir, cls_dir, out_dir, hag_dir):
                pcs = sorted(glob.glob(f"{pc_dir}/*_PC3.txt"))
                for pc in pcs:
                    name = os.path.basename(pc).replace("_PC3.txt", "")
                    if already_tiled(out_dir, name):
                        continue
                    cls = f"{cls_dir}/{name}_CLS.txt"
                    tile_ieee_scene(name, pc, cls, out_dir, CHUNK_XY, STRIDE, hag_dir)
                    any_new[0] = True
                return len(pcs)
            # Train + val-holdout -> HAG/Train; the test split -> HAG/Validate.
            n_tr = _ieee_remaining(TRAIN_PC_DIR, TRAIN_CLS_DIR, f"{PREP_DIR}/train",
                                   HAG_TRAIN_DIR)
            if n_tr == 0:
                raise FileNotFoundError(
                    f"No *_PC3.txt under {TRAIN_PC_DIR}. Upload the IEEE dataset to the "
                    f"ieee-data volume first, e.g.:\n  modal volume put ieee-data "
                    f'"C:\\Users\\OrionHoch\\Desktop\\LabledDatasets\\IEEE" /IEEE')
            print(f"  [train] {n_tr} IEEE Train-Track4 scenes", flush=True)
            n_te = _ieee_remaining(TEST_PC_DIR, TEST_CLS_DIR, f"{PREP_DIR}/test",
                                   HAG_TEST_DIR)
            print(f"  [test] {n_te} IEEE Validate-Track4 scenes", flush=True)
        if any_new[0]:
            (datasets_volume if ds_root else data_volume).commit()
            print("  preprocessing committed.", flush=True)
        else:
            print("  all scenes already cached.", flush=True)

    # --- Model builder + prediction helpers ----------------------------------
    from ptv3.model import PointTransformerV3

    def _hag_of(z_or_files, xyz):
        """Per-point HAG for a cached tile: real HeightAboveGround when the tile
        carries it (HAG-prepped IEEE tiles), else the z-scene-min proxy (canonical
        --dataset tiles and the no-GT predict scenes have no HAG laz)."""
        files = z_or_files.files if hasattr(z_or_files, "files") else z_or_files
        if files is not None and "hag" in files:
            return z_or_files["hag"].astype(np.float32)
        return (xyz[:, 2] - xyz[:, 2].min()).astype(np.float32)

    def build_model(num_classes):
        # FlashAttention is optional — the image installs it best-effort
        # (`pip install flash-attn || echo ...`), so it may be absent. PTv3's
        # SerializedAttention asserts when enable_flash=True but flash_attn can't
        # be imported, so probe it and fall back to standard attention (identical
        # results, just slower) instead of crashing the whole run.
        use_flash = USE_FLASH_ATTN
        if use_flash:
            try:
                import flash_attn  # noqa: F401
                print("  flash_attn enabled.", flush=True)
            except Exception:
                use_flash = False
                print("  flash_attn unavailable — falling back to "
                      "enable_flash=False (slower attention).", flush=True)
        backbone = PointTransformerV3(
            in_channels=7,                # xyz + rgb + HAG
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(128, 128, 128, 128, 128),   # 1024 overflowed the flash pad-gather
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(128, 128, 128, 128),
            drop_path=DROP_PATH,
            enable_flash=use_flash,
            cls_mode=False,
        ).cuda()
        head = torch.nn.Linear(64, num_classes).cuda()
        return backbone, head

    from scipy.spatial import cKDTree
    BASE_PALETTE = np.array([
        [139, 90, 43], [34, 160, 34], [200, 60, 60], [40, 110, 220], [235, 225, 60],
        [150, 80, 200], [240, 140, 40], [70, 200, 200], [220, 100, 170], [120, 120, 120],
        [90, 140, 60], [180, 180, 90], [60, 60, 160], [200, 170, 130], [100, 220, 120],
        [230, 70, 110], [50, 160, 110], [170, 110, 60], [110, 170, 230], [240, 200, 160],
    ], dtype=np.int32)

    def _palette(num_classes):
        reps = -(-num_classes // len(BASE_PALETTE))
        return np.tile(BASE_PALETTE, (reps, 1))[:num_classes]

    def _write_ply(path, xyz, pred_idx, palette, intensity=None):
        cols = [xyz.astype(np.float32), palette[pred_idx]]
        props = ("property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fmt = ["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"]
        if intensity is not None:            # carry per-point intensity for the viewer
            cols.append(np.asarray(intensity, np.float32).reshape(-1, 1))
            props += "property float intensity\n"
            fmt.append("%.5f")
        header = "ply\nformat ascii 1.0\n" + f"element vertex {len(xyz)}\n" + props + "end_header"
        np.savetxt(path, np.column_stack(cols), fmt=fmt, header=header, comments="")

    def make_predict_scene(backbone, head, num_classes):
        def _predict_scene(scene_path):
            # Tile into CHUNK_XY windows, voxel-downsample tracking the inverse
            # map, scatter per-voxel predictions back, NN-fill stragglers.
            xyz, rgb, _ = load_scene(scene_path)
            # No HAG laz for the predict scenes (IEEE Test-Track4 / canonical val)
            # -> z-scene-min proxy for the 7th channel (best-effort demo only).
            hag = (xyz[:, 2] - xyz[:, 2].min()).astype(np.float32)
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
                        w = (xyz[idx] - xyz[idx].mean(0)).astype(np.float32)
                        keys = np.floor(w / GRID_SIZE).astype(np.int64)
                        _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                                      return_inverse=True)
                        vx = w[first]
                        feat = np.concatenate(
                            [vx, rgb[idx][first].astype(np.float32) / 255.0,
                             hag[idx][first, None].astype(np.float32)],
                            axis=1).astype(np.float32)
                        coord = torch.from_numpy(vx).cuda()
                        featt = torch.from_numpy(feat).cuda()
                        offset = torch.tensor([len(vx)], dtype=torch.long).cuda()
                        gc = keys[first] - keys[first].min(0)   # unique, dedup-consistent
                        grid_coord = torch.from_numpy(np.ascontiguousarray(gc)).long().cuda()
                        point = backbone({"coord": coord, "grid_coord": grid_coord,
                                          "feat": featt, "offset": offset})
                        fe = point["feat"] if isinstance(point, dict) else point.feat
                        voxel_pred = head(fe).argmax(-1).cpu().numpy()
                        pred[idx] = voxel_pred[inverse]
            miss = pred < 0
            if miss.any() and (~miss).any():
                _, nn = cKDTree(xyz[~miss]).query(xyz[miss])
                pred[miss] = pred[~miss][nn]
            # rgb carries the p95-normalized intensity grayscale the model saw.
            return xyz, np.clip(pred, 0, num_classes - 1), rgb[:, 0] / 255.0
        return _predict_scene

    # ==========================================================================
    # INFERENCE-ONLY MODE
    # ==========================================================================
    if mode == "infer":
        if not weights or not infer_input:
            raise ValueError("--mode infer requires --weights and --infer-input")
        wpath = f"/outputs/{weights}"
        if not os.path.exists(wpath):
            raise FileNotFoundError(f"weights not found on outputs volume: {wpath}")
        ckpt = torch.load(wpath, map_location="cpu", weights_only=False)
        bsd, hsd = ckpt["backbone"], ckpt["head"]
        num_classes = int(hsd["weight"].shape[0])
        class_names = [f"class_{i}" for i in range(num_classes)]
        run_cfg_path = os.path.join(os.path.dirname(os.path.dirname(wpath)), "run_config.json") \
            if os.path.basename(os.path.dirname(wpath)) == "checkpoints" \
            else os.path.join(os.path.dirname(wpath), "run_config.json")
        if os.path.exists(run_cfg_path):
            with open(run_cfg_path) as f:
                class_names = json.load(f).get("class_names") or class_names

        backbone, head = build_model(num_classes)
        backbone.load_state_dict(bsd)
        head.load_state_dict(hsd)
        backbone.eval(); head.eval()
        print(f"  [infer] loaded {weights} ({num_classes} classes)", flush=True)

        scenes = sorted(glob.glob(f"{DATASETS_ROOT}/_infer/{infer_input}/scenes/*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scenes under {DATASETS_ROOT}/_infer/{infer_input}/scenes")

        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_infer")
        # Predictions live next to the input scenes on the shared terminal-datasets
        # volume (not the per-backbone outputs volume), so inference output lands in
        # one consistent place no matter which model produced it.
        run_dir = f"{DATASETS_ROOT}/_infer/{infer_input}"
        pred_dir = f"{run_dir}/predictions"
        os.makedirs(pred_dir, exist_ok=True)
        with open(f"{run_dir}/run_config.json", "w") as f:
            json.dump({"backbone": "PTv3", "mode": "infer", "weights": weights,
                       "infer_input": infer_input, "num_classes": num_classes,
                       "class_names": class_names, "grid_size": GRID_SIZE,
                       "chunk_xy": CHUNK_XY, "gpu": GPU_TYPE,
                       "scenes": [os.path.basename(s) for s in scenes]}, f, indent=2)

        predict_scene = make_predict_scene(backbone, head, num_classes)
        palette = _palette(num_classes)
        print(f"  [infer] labeling {len(scenes)} scene(s) -> {pred_dir}", flush=True)
        for pc_path in scenes:
            name = os.path.splitext(os.path.basename(pc_path))[0]
            t0 = time.time()
            xyz, pred, inten = predict_scene(pc_path)
            _write_ply(f"{pred_dir}/{name}_pred.ply", xyz, pred, palette, inten)
            print(f"  [infer] {name}: {len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
        datasets_volume.commit()
        print(f"  [infer] done — predictions in _infer/{infer_input}/predictions", flush=True)
        return

    # ==========================================================================
    # TRAINING MODE
    # ==========================================================================
    print("=" * 70)
    print(f"  PTv3  {dataset or 'IEEE_Track4'}  ({GPU_TYPE}, {N_EPOCHS} ep, batch {BATCH_SIZE})")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    ensure_prep()

    tag = dataset or "ieee"

    def lr_at(ep):
        """PTv3 outdoor schedule: short linear warmup to BASE_LR, then cosine
        decay to ~0. Set on the param groups each epoch — no scheduler state to
        restore, so RESUME is trivial (the KPConvX cold recipe's lr_at pattern)."""
        warm = max(1, int(round(WARMUP_PCT * N_EPOCHS)))
        if ep < warm:
            return BASE_LR * (ep + 1) / warm
        prog = (ep - warm) / max(1, N_EPOCHS - warm)
        return float(0.5 * BASE_LR * (1.0 + np.cos(np.pi * prog)))

    def _find_latest_run():
        """Latest UNFINISHED run dir for this tag with checkpoints (resume target).
        Runs marked DONE are skipped, so a completed experiment isn't re-resumed
        on the next launch — but a crashed/retried one is picked straight back up."""
        def _ep(p):
            return int(os.path.basename(p)[2:5])
        for rd in sorted(glob.glob("/outputs/runs/*"), reverse=True):
            if not rd.endswith(f"{tag}_ptv3_hag"):
                continue
            if os.path.exists(f"{rd}/DONE"):
                continue
            ckpts = glob.glob(f"{rd}/checkpoints/ep*.pth")
            if ckpts:
                latest = max(ckpts, key=_ep)
                return rd, latest, _ep(latest)
        return None

    resume_info = _find_latest_run() if (RESUME or AUTO_RESUME) else None
    if resume_info:
        run_dir, resume_ckpt, resume_epoch = resume_info
        run_id = os.path.basename(run_dir)
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        start_epoch = resume_epoch + 1
        print(f"  RESUMING {run_id} from {os.path.basename(resume_ckpt)} "
              f"-> epoch {start_epoch}/{N_EPOCHS}", flush=True)
    else:
        run_id = datetime.utcnow().strftime(f"%Y%m%d_%H%M%S_{tag}_ptv3_hag")
        run_dir = f"/outputs/runs/{run_id}"
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        resume_ckpt, start_epoch = None, 0
        with open(f"{run_dir}/run_config.json", "w") as f:
            json.dump({
                "backbone": "PTv3", "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
                "dataset": dataset or "IEEE_Track4",
                "mode": mode, "gpu": GPU_TYPE,
                "comparison_target": None if dataset
                    else "modal_train_kpconvx_cold.py (KPConvX-L) on IEEE Track 4",
                "label_map_asprs_to_index": None if dataset else LABEL_MAP,
                "n_val_holdout": None if dataset else N_VAL_HOLDOUT,
                "num_classes": NUM_CLASSES, "grid_size": GRID_SIZE,
                "class_names": CLASS_NAMES,
                "in_channels": 7,
                "features": ["x", "y", "z", "rgb_r", "rgb_g", "rgb_b", "HAG"],
                "hag_source": "PDAL hag_nn (trainer_gui Pretraining tab)",
                "chunk_xy": CHUNK_XY, "stride": STRIDE,
                "steps_per_epoch": STEPS,
                "flash_attn": USE_FLASH_ATTN,
                "holdout_seed": HOLDOUT_SEED,
                "drop_path": DROP_PATH,
                "optimizer": {"type": "AdamW", "base_lr": BASE_LR,
                              "weight_decay": WEIGHT_DECAY, "warmup_pct": WARMUP_PCT,
                              "schedule": "warmup+cosine", "grad_clip": GRAD_CLIP},
                "augmentation": {"enable": AUG_ENABLE, "rot_z": AUG_ROT_Z,
                                 "rot_xy": AUG_ROT_XY,
                                 "scale": [AUG_SCALE_MIN, AUG_SCALE_MAX],
                                 "flip_p": AUG_FLIP_P, "jitter_sigma": AUG_JITTER_SIGMA,
                                 "jitter_clip": AUG_JITTER_CLIP},
                "loss": {"pointwise": "focal" if USE_FOCAL else "weighted_ce",
                         "focal_gamma": FOCAL_GAMMA if USE_FOCAL else None,
                         "class_weighting": CLASS_WEIGHTING, "weight_beta": WEIGHT_BETA,
                         "weight_cap": WEIGHT_CAP, "label_smoothing": LABEL_SMOOTH,
                         "lovasz_softmax_weight": LOVASZ_WEIGHT},
                "class_balance": {"rare_oversample": RARE_OVERSAMPLE,
                                  "rare_classes": RARE_CLASSES,
                                  "rare_freq_frac": RARE_FREQ_FRAC,
                                  "rare_tile_prob": RARE_TILE_PROB},
            }, f, indent=2)

    # --- Model --------------------------------------------------------------
    backbone, head = build_model(NUM_CLASSES)
    print(f"  Params: {sum(p.numel() for p in backbone.parameters()):,}")

    optim = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=BASE_LR, weight_decay=WEIGHT_DECAY,
    )
    if resume_ckpt is not None:
        rckpt = torch.load(resume_ckpt, map_location="cuda", weights_only=False)
        backbone.load_state_dict(rckpt["backbone"]); head.load_state_dict(rckpt["head"])
        if "optim" in rckpt:
            optim.load_state_dict(rckpt["optim"])
        print(f"  resumed weights{' + optimizer' if 'optim' in rckpt else ''}", flush=True)
    # loss / class weights are built after the train-tile class scan below.

    # --- Data ---------------------------------------------------------------
    def _scene_of(p):
        b = os.path.basename(p)
        return b.rsplit("_x", 1)[0]
    all_train_tiles = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    all_test_tiles  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
    # Both IEEE (default) and canonical datasets use the same split: hold out a
    # few train scenes by name for in-distribution validation; the test set is
    # the dedicated test tiles (IEEE Validate-Track4 / the canonical val/ folder).
    scene_names = sorted({_scene_of(p) for p in all_train_tiles})
    rng = np.random.RandomState(HOLDOUT_SEED)
    idx = np.arange(len(scene_names))
    rng.shuffle(idx)
    n_hold = min(N_VAL_HOLDOUT, max(1, len(scene_names) // 5))
    hold = {scene_names[i] for i in idx[:n_hold]}
    synth_train_tiles = [f for f in all_train_tiles if _scene_of(f) not in hold]
    synth_test_tiles  = [f for f in all_train_tiles if _scene_of(f) in hold]
    real_train_tiles, real_test_tiles = [], all_test_tiles
    print(f"  train: {len(synth_train_tiles)}   val(holdout {n_hold} scenes): "
          f"{len(synth_test_tiles)}   test: {len(real_test_tiles)}", flush=True)

    # --- class balance: scan training tiles for inverse-frequency weights +
    # rare-class tile flags. The per-tile np.load over the Modal volume is the
    # bottleneck (~45k tiles -> minutes), so read in parallel AND cache the raw
    # scan to PREP_DIR (keyed on the tile set) — instant on every later launch. -
    train_pool = synth_train_tiles + real_train_tiles
    cache_path = f"{PREP_DIR}/class_balance_cache.npz"
    pool_names = np.array([os.path.basename(p) for p in train_pool])

    def _scan_tile(tp):
        lab = np.load(tp)["lab"]
        v = lab[(lab >= 0) & (lab < NUM_CLASSES)]
        return (np.bincount(v, minlength=NUM_CLASSES).astype(np.int64)
                if v.size else np.zeros(NUM_CLASSES, np.int64))

    class_counts = present_mask = None
    if os.path.exists(cache_path):
        try:
            cz = np.load(cache_path, allow_pickle=False)
            if (cz["tile_names"].shape == pool_names.shape
                    and bool(np.all(cz["tile_names"] == pool_names))
                    and int(cz["num_classes"]) == NUM_CLASSES):
                class_counts = cz["class_counts"].astype(np.int64)
                present_mask = cz["present_mask"].astype(bool)
                print(f"  class balance: loaded cache ({len(train_pool)} tiles)", flush=True)
        except Exception as e:
            print(f"  class balance: ignoring unreadable cache ({e})", flush=True)

    if class_counts is None:
        from concurrent.futures import ThreadPoolExecutor
        print(f"  scanning {len(train_pool)} train tiles for class balance (parallel)…",
              flush=True)
        per_tile = np.zeros((len(train_pool), NUM_CLASSES), np.int64)
        with ThreadPoolExecutor(max_workers=32) as _ex:
            for i, counts in enumerate(_ex.map(_scan_tile, train_pool)):
                per_tile[i] = counts
        class_counts = per_tile.sum(0)
        present_mask = per_tile > 0
        try:
            np.savez(cache_path, tile_names=pool_names, class_counts=class_counts,
                     present_mask=present_mask, num_classes=np.int64(NUM_CLASSES))
            (datasets_volume if ds_root else data_volume).commit()
            print(f"  class balance: cached scan -> {cache_path}", flush=True)
        except Exception as e:
            print(f"  class balance: could not write cache ({e})", flush=True)

    def _name(c):
        return CLASS_NAMES[c] if CLASS_NAMES else c
    print(f"  class counts: {dict(zip([_name(c) for c in range(NUM_CLASSES)], class_counts.tolist()))}",
          flush=True)

    # Rare classes: explicit RARE_CLASSES, else auto — present classes whose
    # frequency is below RARE_FREQ_FRAC x the median present-class frequency.
    if RARE_CLASSES is not None:
        rare_set = set(RARE_CLASSES)
    elif RARE_OVERSAMPLE:
        present_freq = class_counts[class_counts > 0]
        thresh = RARE_FREQ_FRAC * float(np.median(present_freq)) if present_freq.size else 0.0
        rare_set = {c for c in range(NUM_CLASSES)
                    if 0 < class_counts[c] < thresh}
    else:
        rare_set = set()
    rare_cols = sorted(rare_set)
    if RARE_OVERSAMPLE and rare_cols:
        rare_tiles = [train_pool[i] for i in np.nonzero(present_mask[:, rare_cols].any(1))[0]]
    else:
        rare_tiles = []
    print(f"  rare classes: {sorted(_name(c) for c in rare_set)}  "
          f"({len(rare_tiles)}/{len(train_pool)} tiles)", flush=True)

    if CLASS_WEIGHTING:
        # w = 1/sqrt(freq) (inverse-sqrt-frequency): sub-linear, mean-normalized,
        # capped to [1/CAP, CAP]. Same scheme as the KPConvX cold script.
        freq = class_counts / max(int(class_counts.sum()), 1)
        w = (1.0 / np.maximum(freq, 1e-6)) ** WEIGHT_BETA
        w[class_counts == 0] = 1.0          # don't up-weight absent classes
        w = w / w[class_counts > 0].mean() if (class_counts > 0).any() else w
        w = np.clip(w, 1.0 / WEIGHT_CAP, WEIGHT_CAP)
        class_weights = torch.tensor(w, dtype=torch.float32).cuda()
        print(f"  class weights: "
              f"{dict(zip([_name(c) for c in range(NUM_CLASSES)], [round(float(x),3) for x in w]))}",
              flush=True)
    else:
        class_weights = None

    # Torch-native weighted (label-smoothed) CE — matches KPConvX's SmoothCE.
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1,
                                        label_smoothing=LABEL_SMOOTH)

    # --- Lovász-Softmax (Berman et al. 2018): differentiable mIoU surrogate,
    # per-class equally weighted, added on top of CE — PTv3's outdoor loss is
    # literally CE + Lovász. Pure-torch flat form (logits (N,C), labels (N,)). --
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
        if probas.numel() == 0:
            return probas.sum() * 0.0
        losses = []
        for c in torch.unique(labels):
            fg = (labels == c).float()
            errors = (fg - probas[:, int(c)]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
        if not losses:
            return probas.sum() * 0.0
        return torch.stack(losses).mean()

    # alpha-balanced multiclass focal loss (alpha = inverse-sqrt class weights).
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
        loss = focal_loss(logits, labels) if USE_FOCAL else ce_loss(logits, labels)
        if LOVASZ_WEIGHT > 0:
            valid = labels >= 0
            if valid.any():
                probas = torch.softmax(logits[valid], dim=1)
                loss = loss + LOVASZ_WEIGHT * lovasz_softmax_flat(probas, labels[valid])
        return loss

    def pick_train_tile():
        if rare_tiles and np.random.rand() < RARE_TILE_PROB:
            return rare_tiles[np.random.randint(len(rare_tiles))]
        return synth_train_tiles[np.random.randint(len(synth_train_tiles))]

    # --- PTv3 outdoor augmentation suite (was entirely absent before) ---------
    def augment_xyz(xyz):
        """Full z-yaw, gentle x/y tilt, isotropic scale, per-axis flip, jitter."""
        az = (np.random.rand() * 2 - 1) * np.pi * AUG_ROT_Z
        ax = (np.random.rand() * 2 - 1) * np.pi * AUG_ROT_XY
        ay = (np.random.rand() * 2 - 1) * np.pi * AUG_ROT_XY
        cz, sz = np.cos(az), np.sin(az)
        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
        out = xyz @ (Rz @ Ry @ Rx).T
        out = out * np.random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX)
        if np.random.rand() < AUG_FLIP_P:
            out[:, 0] = -out[:, 0]
        if np.random.rand() < AUG_FLIP_P:
            out[:, 1] = -out[:, 1]
        out += np.clip(np.random.normal(0, AUG_JITTER_SIGMA, out.shape),
                       -AUG_JITTER_CLIP, AUG_JITTER_CLIP)
        return out.astype(np.float32)

    def to_ptv3_batch(tiles_for_batch, training=True):
        # PTv3 takes a dict with: coord, grid_coord, feat, offset
        coords, feats, labels, offsets, grid_coords = [], [], [], [], []
        running = 0
        for tile in tiles_for_batch:
            z = np.load(tile)
            xyz, rgb, lab = z["xyz"], z["rgb"], z["lab"]
            hag = _hag_of(z, xyz)                      # real HAG or z-min proxy
            # random crop ~30m for memory (train only — eval keeps full tiles)
            if training and len(xyz) > 80000:
                c = xyz[np.random.randint(len(xyz))]
                d2 = np.sum((xyz[:, :2] - c[:2]) ** 2, axis=1)
                idx = np.where(d2 < 15.0 ** 2)[0]
                if len(idx) > 80000:
                    idx = np.random.choice(idx, 80000, replace=False)
                xyz, rgb, lab, hag = xyz[idx], rgb[idx], lab[idx], hag[idx]
            xyz = xyz.astype(np.float32)
            if training and AUG_ENABLE:
                xyz = augment_xyz(xyz)
            xyz = xyz - xyz.mean(0, keepdims=True)
            # Drop non-finite + far-outlier points: a single bad coordinate (NaN/
            # Inf or a stray faraway return) blows up the voxel grid and spconv's
            # indices, triggering a CUDA index device-assert that kills the run.
            ok = (np.isfinite(xyz).all(1)
                  & (np.abs(xyz[:, :2]).max(1) <= CHUNK_XY)
                  & (np.abs(xyz[:, 2]) <= 200.0))
            if int(ok.sum()) < 64:
                continue
            xyz = xyz[ok]; rgb = rgb[ok]; lab = lab[ok]; hag = hag[ok]
            # voxel-grid downsample to GRID_SIZE. Take grid_coord from the SAME
            # integer keys used to dedup, so it is unique per cloud and phase-
            # consistent with the kept points. Recomputing it as
            # floor((coord-min)/GRID) uses a different phase and can collapse two
            # distinct voxels onto one grid_coord — which corrupts spconv's
            # submanifold rulebook (out-of-bounds gather -> CUDA device assert).
            keys = np.floor(xyz / GRID_SIZE).astype(np.int64)
            _, uniq = np.unique(keys, axis=0, return_index=True)
            xyz = xyz[uniq]; rgb = rgb[uniq]; lab = lab[uniq]; hag = hag[uniq]
            # feat = augmented/centered coords + color/intensity (intensity is
            # carried in the rgb channels for LiDAR datasets) + HAG (meters).
            feat = np.concatenate(
                [xyz.astype(np.float32), rgb.astype(np.float32) / 255.0,
                 hag[:, None].astype(np.float32)], axis=1)
            coords.append(xyz); feats.append(feat); labels.append(lab)
            grid_coords.append(keys[uniq])
            running += len(xyz)
            offsets.append(running)
        coord = torch.from_numpy(np.concatenate(coords).astype(np.float32)).cuda()
        feat  = torch.from_numpy(np.concatenate(feats).astype(np.float32)).cuda()
        label = torch.from_numpy(np.concatenate(labels).astype(np.int64)).cuda()
        offset = torch.tensor(offsets, dtype=torch.long).cuda()
        gc = np.concatenate(grid_coords)
        gc -= gc.min(0, keepdims=True)                 # non-negative for spconv
        grid_coord = torch.from_numpy(np.ascontiguousarray(gc)).long().cuda()
        return {"coord": coord, "grid_coord": grid_coord,
                "feat": feat, "offset": offset}, label

    # --- Train loop ---------------------------------------------------------
    metrics_csv = f"{run_dir}/metrics.csv"
    if not os.path.exists(metrics_csv):       # keep prior rows when resuming
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss", "train_acc", "val_acc",
                "train_iou", "val_iou", "lr", "sec_per_iter", "sec_per_epoch",
                "gpu_mem_mb",
            ])

    # --- Raw-scene loaders + ONE combined eval set ---------------------------
    # val (train-holdout scenes) and test (the dedicated test split) are scored
    # together as a single evaluation set. Each item carries its own tile dir so
    # evaluate() votes per scene regardless of which split it came from.
    def _raw_loader(split, name):
        """Closure -> (xyz, rgb, lab) for the ORIGINAL raw scene, so the voted
        voxel predictions can be reprojected onto raw points + raw GT. `name` is a
        parameter, so each closure binds its own scene (no loop late-binding bug).
        HAG isn't needed here — raw scoring only uses raw xyz + raw GT."""
        if ds_root:
            sub = "train" if split == "val" else "val"   # val holdout = train scenes
            return lambda: load_canonical(f"{ds_root}/{sub}/{name}.npz")
        if split == "val":
            return lambda: load_ieee(f"{TRAIN_PC_DIR}/{name}_PC3.txt",
                                     f"{TRAIN_CLS_DIR}/{name}_CLS.txt")
        return lambda: load_ieee(f"{TEST_PC_DIR}/{name}_PC3.txt",
                                 f"{TEST_CLS_DIR}/{name}_CLS.txt")

    eval_items = (
        [(n, _raw_loader("val", n), f"{PREP_DIR}/train") for n in sorted(hold)] +
        [(n, _raw_loader("test", n), f"{PREP_DIR}/test")
         for n in sorted({_scene_of(p) for p in real_test_tiles})]
    )
    print(f"  eval set: {len(eval_items)} scenes "
          f"({len(hold)} holdout + {len(eval_items) - len(hold)} test)", flush=True)

    def evaluate(scene_items, label):
        """Per-SCENE overlap voting scored on the ORIGINAL raw points — the
        official protocol KPConvX/RandLA use, so all three are directly
        comparable. For each scene: forward its overlapping tiles (stride
        CHUNK_XY/2, each point in up to 4 tiles), sum center-tapered softmax
        votes per GRID voxel, argmax, then NN-propagate each voxel's prediction
        to the raw cloud and score against raw GT.

        Fixes vs the old eval: (1) voting is per-scene, so voxel keys from
        different scenes can never collide (the old eval concatenated every
        scene onto one global voxel grid — silently corrupting any dataset whose
        scenes share a coordinate frame); (2) scoring is on raw points, not on
        GRID voxels (the old eval scored at voxel resolution against an arbitrary
        first-point voxel label), so the number means the same thing as KPConvX's."""
        t_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_union = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_gt    = np.zeros(NUM_CLASSES, dtype=np.int64)
        correct = total = 0
        n_scenes = n_skipped_tiles = n_skipped_scenes = 0
        t_test = time.time()
        with torch.no_grad():
            for name, load_raw, split_dir in scene_items:
                tiles = sorted(glob.glob(f"{split_dir}/{name}_x*.npz"))
                if not tiles:
                    n_skipped_scenes += 1; continue
                keys_l, vote_l, xyz_l = [], [], []
                for tile in tiles:
                    z = np.load(tile)
                    xyz, rgb = z["xyz"].astype(np.float32), z["rgb"]
                    if len(xyz) < 64:
                        continue
                    hag = _hag_of(z, xyz)
                    cxyz = xyz - xyz.mean(0, keepdims=True)
                    ok = (np.isfinite(cxyz).all(1)
                          & (np.abs(cxyz[:, :2]).max(1) <= CHUNK_XY)
                          & (np.abs(cxyz[:, 2]) <= 200.0))
                    if int(ok.sum()) < 64:
                        continue
                    xyz, rgb, cxyz, hag = xyz[ok], rgb[ok], cxyz[ok], hag[ok]
                    vk = np.floor(cxyz / GRID_SIZE).astype(np.int64)
                    _, first, inverse = np.unique(vk, axis=0, return_index=True,
                                                  return_inverse=True)
                    vx = cxyz[first].astype(np.float32)
                    feat = np.concatenate(
                        [vx, rgb[first].astype(np.float32) / 255.0,
                         hag[first, None].astype(np.float32)], axis=1).astype(np.float32)
                    coord = torch.from_numpy(vx).cuda()
                    featt = torch.from_numpy(feat).cuda()
                    offset = torch.tensor([len(vx)], dtype=torch.long).cuda()
                    gc = vk[first] - vk[first].min(0)        # unique, dedup-consistent
                    grid_coord = torch.from_numpy(np.ascontiguousarray(gc)).long().cuda()
                    try:
                        point = backbone({"coord": coord, "grid_coord": grid_coord,
                                          "feat": featt, "offset": offset})
                        fe = point["feat"] if isinstance(point, dict) else point.feat
                        lg = head(fe).cpu().numpy().astype(np.float32)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache(); n_skipped_tiles += 1; continue
                        raise
                    ex = np.exp(lg - lg.max(1, keepdims=True))
                    prob = (ex / ex.sum(1, keepdims=True))[inverse]   # per original pt
                    cxy = (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
                    d = np.abs(xyz[:, :2] - cxy).max(1)
                    wgt = np.clip(1.0 - d / (CHUNK_XY / 2.0), 0.05, 1.0) ** 2
                    keys_l.append(np.floor(xyz / GRID_SIZE).astype(np.int64))
                    vote_l.append((prob * wgt[:, None]).astype(np.float32))
                    xyz_l.append(xyz.astype(np.float32))
                if not keys_l:
                    n_skipped_scenes += 1; continue
                K = np.concatenate(keys_l); V = np.concatenate(vote_l); P = np.concatenate(xyz_l)
                uniq, ufirst, uinv = np.unique(K, axis=0, return_index=True, return_inverse=True)
                votes = np.zeros((len(uniq), NUM_CLASSES), np.float64)
                np.add.at(votes, uinv, V)
                pred_u  = votes.argmax(1)
                rep_xyz = P[ufirst]                      # one raw coord per voxel
                # Reproject voxel predictions onto the raw scene cloud + raw GT.
                try:
                    raw_xyz, _, raw_lab = load_raw()
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped_scenes += 1; continue
                _, nn = cKDTree(rep_xyz).query(raw_xyz)
                raw_pred = pred_u[nn]
                v = (raw_lab >= 0) & (raw_lab < NUM_CLASSES)
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
        absent  = [c for c in range(NUM_CLASSES) if gt_counts[c] == 0]
        present_iou = [float(iou_per[c]) for c in present]
        present_mIoU = float(np.mean(present_iou)) if present_iou else 0.0
        m = {
            "overall_acc": correct / max(total, 1),
            "overall_mIoU": float(np.mean(iou_per)),
            "present_classes_mIoU": present_mIoU,
            "per_class_iou": {_name(c): float(iou_per[c]) for c in range(NUM_CLASSES)},
            "per_class_gt_count": {_name(c): gt_counts[c] for c in range(NUM_CLASSES)},
            "present_classes": [_name(c) for c in present],
            "absent_classes": [_name(c) for c in absent],
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
        print(f"  [{label}] acc={m['overall_acc']:.4f}  "
              f"mIoU({NUM_CLASSES}-way)={m['overall_mIoU']:.4f}  "
              f"mIoU(present {len(present)})={m['present_classes_mIoU']:.4f}  "
              f"absent={m['absent_classes']}  raw_pts={total:,}  "
              f"skipped(tiles={n_skipped_tiles},scenes={n_skipped_scenes})", flush=True)
        return m

    # --- Periodic evaluation: the REAL voted eval (not a cheap proxy) over the
    # whole combined eval set every VAL_EVERY epochs, appended to val_metrics.csv
    # so the curve is watchable mid-run. NOTE: this is far heavier than the old
    # quick_val — it forwards every overlapping tile of every eval scene. Raise
    # VAL_EVERY if it costs too much wall-clock. ------------------------------
    val_csv = f"{run_dir}/val_metrics.csv"
    if not os.path.exists(val_csv):
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "val_acc", "val_miou"] +
                [f"iou_{_name(c)}" for c in range(NUM_CLASSES)])

    def run_eval(ep, write_json=False):
        backbone.eval(); head.eval()
        m = evaluate(eval_items, f"eval@ep{ep}")
        ious = [m["per_class_iou"][_name(c)] for c in range(NUM_CLASSES)]
        with open(val_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{m['overall_acc']:.4f}",
                                    f"{m['overall_mIoU']:.4f}"] + [f"{x:.4f}" for x in ious])
        if write_json:
            # val + test are one combined set now; write the same metrics under
            # both keys so the trainer_gui (which reads test_metrics["val"] and
            # ["test"]) keeps rendering without changes.
            with open(f"{run_dir}/test_metrics.json", "w") as fj:
                json.dump({"val": m, "test": m,
                           "eval_scenes": [n for n, _, _ in eval_items]}, fj, indent=2)
        outputs_volume.commit()
        backbone.train(); head.train()
        return m

    LOG_EVERY = 20  # intra-epoch heartbeat
    print(f"  starting at epoch {start_epoch}, up to {N_EPOCHS}, "
          f"{STEPS} steps/epoch (batch {BATCH_SIZE})", flush=True)
    t_run = time.time()
    for ep in range(start_epoch, N_EPOCHS):
        cur_lr = lr_at(ep)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        backbone.train(); head.train()
        ep_loss = ep_correct = ep_total = 0
        ep_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        ep_union = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_ep = time.time(); t_chunk = t_ep; n_steps = 0; last_log_step = 0
        print(f"  ep {ep:3d} starting (lr={cur_lr:.2e})…", flush=True)
        for step in range(STEPS):
            picks = [pick_train_tile() for _ in range(BATCH_SIZE)]
            try:
                batch, label = to_ptv3_batch(picks, training=True)
                point = backbone(batch)
                feat = point["feat"] if isinstance(point, dict) else point.feat
                logits = head(feat)
                loss = seg_loss(logits, label)
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()), GRAD_CLIP)
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
                ep_correct += (pred[m] == label[m]).sum().item()
                ep_total   += int(m.sum())
                for c in range(NUM_CLASSES):
                    ep_inter[c] += ((pred == c) & (label == c)).sum().item()
                    ep_union[c] += (((pred == c) | (label == c)) & m).sum().item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache(); continue
                raise
        sec_per_iter = (time.time() - t_ep) / max(n_steps, 1)
        sec_per_epoch = time.time() - t_ep
        train_acc = ep_correct / max(ep_total, 1)
        with np.errstate(invalid="ignore"):
            train_iou = float(np.mean(ep_inter / np.maximum(ep_union, 1)))
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, ep_loss / max(n_steps, 1), "", f"{train_acc:.4f}", "",
                f"{train_iou:.4f}", "", f"{cur_lr:.6e}", f"{sec_per_iter:.4f}",
                f"{sec_per_epoch:.2f}", f"{gpu_mem:.1f}",
            ])
        print(f"  ep {ep:3d}: loss={ep_loss/max(n_steps,1):.4f} "
              f"acc={train_acc:.4f} miou={train_iou:.4f} lr={cur_lr:.2e} "
              f"s/iter={sec_per_iter:.3f} s/ep={sec_per_epoch:.1f}", flush=True)
        if (ep + 1) % CHECKPOINT_GAP == 0 or ep == N_EPOCHS - 1:
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "optim": optim.state_dict(), "epoch": ep},
                       f"{run_dir}/checkpoints/ep{ep:03d}.pth")
            # keep only the 2 newest checkpoints (~0.5 GB each with optimizer)
            for old in sorted(glob.glob(f"{run_dir}/checkpoints/ep*.pth"))[:-2]:
                try:
                    os.remove(old)
                except OSError:
                    pass
            outputs_volume.commit()
        if (ep + 1) % VAL_EVERY == 0 and ep != N_EPOCHS - 1:
            run_eval(ep)               # last epoch handled by the final eval below

    torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                "epoch": N_EPOCHS - 1}, f"{run_dir}/final_model.pth")

    # --- Final evaluation: the real voted eval over the combined eval set,
    # written to test_metrics.json (the same number run_eval logs periodically). -
    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(N_EPOCHS - 1, write_json=True)
    print(f"  total wall-clock {(time.time() - t_run)/3600:.2f} h")

    # ------------------------------------------------------------------------
    # Prediction demo: canonical -> first N_PREDICT val scenes (PLY + npz);
    # IEEE default -> first N_PREDICT Test-Track4 scenes (no GT), written as an
    # ASPRS-coded *_pred_CLS.txt (like the KPConvX cold script) + a colored PLY.
    # Best-effort.
    # ------------------------------------------------------------------------
    try:
        backbone.eval(); head.eval()
        pred_dir = f"{run_dir}/predictions"
        os.makedirs(pred_dir, exist_ok=True)
        predict_scene = make_predict_scene(backbone, head, NUM_CLASSES)
        palette = _palette(NUM_CLASSES)
        if ds_root:
            scenes = sorted(glob.glob(f"{ds_root}/val/*.npz"))[:N_PREDICT]
            print(f"  [predict] labeling {len(scenes)} scene(s) -> {pred_dir}", flush=True)
            for pc_path in scenes:
                name = os.path.splitext(os.path.basename(pc_path))[0]
                t0 = time.time()
                xyz, pred, inten = predict_scene(pc_path)
                _write_ply(f"{pred_dir}/{name}_pred.ply", xyz, pred, palette, inten)
                np.savez_compressed(f"{pred_dir}/{name}_pred.npz",
                                    xyz=xyz.astype(np.float32), pred=pred.astype(np.int32),
                                    intensity=inten.astype(np.float32),
                                    class_names=np.array(CLASS_NAMES))
                print(f"  [predict] {name}: {len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
        else:
            IDX_TO_ASPRS = np.array([2, 5, 6, 9, 17], dtype=np.int32)  # index -> ASPRS code
            scenes = sorted(glob.glob(f"{PRED_PC_DIR}/*_PC3.txt"))[:N_PREDICT]
            print(f"  [predict] labeling {len(scenes)} Test-Track4 scene(s) -> {pred_dir}",
                  flush=True)
            for pc_path in scenes:
                name = os.path.basename(pc_path).replace("_PC3.txt", "")
                t0 = time.time()
                xyz, pred, inten = predict_scene(pc_path)
                np.savetxt(f"{pred_dir}/{name}_pred_CLS.txt", IDX_TO_ASPRS[pred], fmt="%d")
                _write_ply(f"{pred_dir}/{name}_pred.ply", xyz, pred, palette, inten)
                print(f"  [predict] {name}: {len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
        outputs_volume.commit()
    except Exception as e:
        print(f"  [predict] skipped (model is saved): {e}", flush=True)
        traceback.print_exc()

    # Mark the run complete so AUTO_RESUME won't re-resume it on the next launch
    # (a crashed/retried run has no DONE and is picked back up automatically).
    open(f"{run_dir}/DONE", "w").close()
    outputs_volume.commit()
    print(f"  run complete -> {run_id}", flush=True)


@app.local_entrypoint()
def main(dataset: Optional[str] = None, grid: Optional[float] = None,
         epochs: Optional[int] = None, batch: Optional[int] = None,
         steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
         mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None, predict_n: Optional[int] = None):
    what = f"infer({weights})" if mode == "infer" else f"train({dataset or 'IEEE Track 4'})"
    print(f"Launching {APP_NAME} [{what}] on {GPU_TYPE} for up to {TIMEOUT_HOURS}h.")
    train_ptv3.remote(dataset=dataset, grid=grid, epochs=epochs, batch=batch,
                      steps_per_epoch=steps_per_epoch, chunk_xy=chunk_xy, mode=mode,
                      weights=weights, infer_input=infer_input, predict_n=predict_n)
