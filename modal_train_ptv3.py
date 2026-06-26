"""
Modal training script for PointTransformerV3 on IEEE GRSS 2019 DFC Track 4.

The default (no --dataset) path trains on the same raw IEEE Track 4 airborne
LiDAR — and uses the same ASPRS->index label map, per-scene p95 intensity
normalization, and scene holdout — as modal_train_kpconvx_cold.py, so PTv3 can
be compared head-to-head against KPConvX/KPConv on identical data. (It was
previously wired to STPLS3D; switched 2026-06-15.)

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
  --grid / --chunk-xy / --epochs / --batch / --steps-per-epoch
  --mode infer --weights runs/<id>/final_model.pth --infer-input <job_id>

GPU type / timeout come from TT_GPU / TT_TIMEOUT_HOURS env vars.
"""

import os
from typing import Optional

import modal

# ============================================================================
# Configuration
# ============================================================================
APP_NAME      = "ptv3-ieee"
GPU_TYPE      = os.environ.get("TT_GPU", "A100")
N_EPOCHS      = 100             # smoke test; PTv3 outdoor recipe trains ~50 epochs
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
USE_FLASH_ATTN = False   # PTv3 runs on standard attention; no flash-attn dep
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
VAL_EVERY        = 10      # held-out val pass every N epochs (no weight updates)
VAL_SUBSET       = 200     # tiles in the periodic val pass (seeded random subset)
CHECKPOINT_GAP   = 3       # checkpoint (model + optimizer) frequency, epochs
RESUME           = True   # force-resume the latest matching run (see AUTO_RESUME)
AUTO_RESUME      = False    # auto-continue an unfinished run (no DONE marker) on
                          # relaunch / Modal auto-retry, so an intermittent crash
                          # never loses the run — only epochs since last checkpoint
DEBUG_CUDA       = False   # ONE-OFF diagnostic: CUDA_LAUNCH_BLOCKING + dump the
                          # offending batch's stats + no retries (set True to debug).
STEM_KERNEL      = 5       # PTv3's faithful 5x5x5 stem. (Shrinking it to 3 was a
                          # failed workaround for the cu124 spconv conv-backward
                          # device-assert; that bug is fixed properly by moving to
                          # the spconv-cu118 build, so the original stem is restored.
                          # Set 3 only if a large-kernel spconv issue ever recurs.)

DATA_ROOT      = "/data/IEEE"
TRAIN_PC_DIR   = f"{DATA_ROOT}/Train-Track4/Track4"            # *_PC3.txt (x,y,z,intensity,ret)
TRAIN_CLS_DIR  = f"{DATA_ROOT}/Train-Track4-Truth/Track4-Truth"  # *_CLS.txt (ASPRS codes)
TEST_PC_DIR    = f"{DATA_ROOT}/Validate-Track4/Track4"
TEST_CLS_DIR   = f"{DATA_ROOT}/Validate-Track4-Truth"
PREP_DIR       = f"{DATA_ROOT}/prep/ptv3_ieee_grid05_origin"

DATASETS_ROOT = "/datasets"   # terminal-datasets volume (trainer_gui canonical datasets)

# ============================================================================
# Image
# ============================================================================

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential", "cmake", "ninja-build", "libgl1", "libglib2.0-0")
    # Canonical PTv3/Pointcept stack: torch 2.1 + CUDA 11.8 + spconv-cu118. The
    # cu124 build's sparse-conv backward device-asserted on valid input and its
    # implicit-GEMM couldn't NVRTC-compile (missing cumm headers); spconv-cu118 is
    # the mature, battle-tested build PTv3 is actually developed against. No
    # flash-attn — PTv3 runs fine on standard attention (USE_FLASH_ATTN=False).
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
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
        index_url="https://download.pytorch.org/whl/cu118",
        extra_index_url="https://pypi.org/simple",
    )
    .pip_install(
        "spconv-cu118",
        "torch-scatter",
        "torch-cluster",
        find_links="https://data.pyg.org/whl/torch-2.1.0+cu118.html",
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

image = image.add_local_file("local_train_ptv3.py", "/root/local_train_ptv3.py")
image = image.add_local_file("train_common.py", "/root/train_common.py")

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
    retries=modal.Retries(max_retries=0 if DEBUG_CUDA else 10,
                          backoff_coefficient=1.0, initial_delay=5.0),
)
def train_ptv3(dataset: Optional[str] = None, grid: Optional[float] = None,
               epochs: Optional[int] = None, batch: Optional[int] = None,
               steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
               mode: str = "train", weights: Optional[str] = None,
               infer_input: Optional[str] = None):
    """Modal shell: provision the GPU container + volumes, then run the LOCAL
    trainer. All training/inference logic lives in local_train_ptv3.py — this only
    shells out to it, so local and cloud run byte-identical code."""
    import subprocess
    import sys
    import threading

    cmd = [sys.executable, "/root/local_train_ptv3.py"]
    for _flag, _val in (
        ("--dataset", dataset),
        ("--grid", grid),
        ("--epochs", epochs),
        ("--batch", batch),
        ("--steps-per-epoch", steps_per_epoch),
        ("--chunk-xy", chunk_xy),
        ("--mode", mode),
        ("--weights", weights),
        ("--infer-input", infer_input),
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
def main(dataset: Optional[str] = None, grid: Optional[float] = None,
         epochs: Optional[int] = None, batch: Optional[int] = None,
         steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
         mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None):
    what = f"infer({weights})" if mode == "infer" else f"train({dataset or 'IEEE Track 4'})"
    print(f"Launching {APP_NAME} [{what}] on {GPU_TYPE} for up to {TIMEOUT_HOURS}h.")
    train_ptv3.remote(dataset=dataset, grid=grid, epochs=epochs, batch=batch,
                      steps_per_epoch=steps_per_epoch, chunk_xy=chunk_xy, mode=mode,
                      weights=weights, infer_input=infer_input)
