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

import os
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
# Per-point HeightAboveGround from the Pretraining tab (HAGTEST upload). Train +
# val-holdout scenes come from Train-Track4 (HAG/Train); test from HAG/Validate.
HAG_TRAIN_DIR  = f"{DATA_ROOT}/HAG/Train"
HAG_TEST_DIR   = f"{DATA_ROOT}/HAG/Validate"
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

image = image.add_local_file("local_train_kpconvx_cold_hag.py", "/root/local_train_kpconvx_cold_hag.py")
image = image.add_local_file("train_common.py", "/root/train_common.py")

data_volume     = modal.Volume.from_name("ieee-data",            create_if_missing=True)
outputs_volume  = modal.Volume.from_name(f"{APP_NAME}-outputs",  create_if_missing=True)
datasets_volume = modal.Volume.from_name(
    os.environ.get("TT_DATASET_VOLUME", "terminal-datasets"), create_if_missing=True)

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
def train_kpconvx(dataset: Optional[str] = None, mode: str = "train",
                  weights: Optional[str] = None,
                  infer_input: Optional[str] = None, grid: Optional[float] = None,
                  chunk_xy: Optional[float] = None, epochs: Optional[int] = None,
                  batch: Optional[int] = None, steps_per_epoch: Optional[int] = None):
    """Modal shell: provision the GPU container + volumes, then run the LOCAL
    trainer. All training/inference logic lives in local_train_kpconvx_cold_hag.py — this only
    shells out to it, so local and cloud run byte-identical code."""
    import subprocess
    import sys
    import threading

    cmd = [sys.executable, "/root/local_train_kpconvx_cold_hag.py"]
    for _flag, _val in (
        ("--dataset", dataset),
        ("--mode", mode),
        ("--weights", weights),
        ("--infer-input", infer_input),
        ("--grid", grid),
        ("--chunk-xy", chunk_xy),
        ("--epochs", epochs),
        ("--batch", batch),
        ("--steps-per-epoch", steps_per_epoch),
    ):
        if _val is not None:
            cmd += [_flag, str(_val)]
    print("[modal-shell] " + " ".join(cmd), flush=True)

    # Persist checkpoints + prep cache mid-run so an uncatchable spconv CUDA
    # device-assert (the reason this function has retries) still leaves the
    # latest state on the volumes for the local trainer's auto-resume on the
    # retry. ponytail: time-based commit; the trainer's 2-checkpoint retention
    # covers the rare case of snapshotting a half-written .pth.
    _stop = threading.Event()

    def _commit_loop():
        while not _stop.wait(120):
            data_volume.commit()
            outputs_volume.commit()
            datasets_volume.commit()

    _t = threading.Thread(target=_commit_loop, daemon=True)
    _t.start()
    try:
        subprocess.run(cmd, check=True)
    finally:
        _stop.set()
        data_volume.commit()
        outputs_volume.commit()
        datasets_volume.commit()


@app.local_entrypoint()
def main(dataset: Optional[str] = None, mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None, grid: Optional[float] = None,
         chunk_xy: Optional[float] = None, epochs: Optional[int] = None,
         batch: Optional[int] = None, steps_per_epoch: Optional[int] = None):
    what = {"eval": "eval-only re-score", "infer": f"infer({weights})"}.get(
        mode, f"train({dataset or 'IEEE Track 4'})")
    print(f"Launching {APP_NAME} [{what}] on {GPU_TYPE} for up to {TIMEOUT_HOURS}h.")
    train_kpconvx.remote(dataset=dataset, mode=mode, weights=weights, infer_input=infer_input,
                         grid=grid, chunk_xy=chunk_xy, epochs=epochs, batch=batch,
                         steps_per_epoch=steps_per_epoch)
