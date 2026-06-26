"""
Modal training script for RandLA-Net (PyTorch) on IEEE GRSS 2019 Track 4 —
COLD-START + HAG variant.

This is the HAG twin of modal_train_randlanet.py. Identical in every way except
it appends a real PDAL HeightAboveGround feature (computed by the trainer_gui
Pretraining tab: SMRF ground-classify -> hag_nn) as an extra input channel:
IN_DIM 5 -> 6 ([xyz, intensity, return_number, HAG]); fc0 is rebuilt at 6->8.
HAG is read per point from /data/IEEE/HAG/{Train,Validate}/<scene>_PC3.laz and
paired to the raw _PC3.txt points. Run it head-to-head against
modal_train_randlanet.py to measure HAG's contribution.

Prereq: upload the HAGTEST output (Pretraining tab) to the ieee-data volume:
    modal volume put ieee-data "C:/Users/OrionHoch/Desktop/HAGTEST/Train"    /IEEE/HAG/Train
    modal volume put ieee-data "C:/Users/OrionHoch/Desktop/HAGTEST/Validate" /IEEE/HAG/Validate
The --dataset (canonical trainer_gui) path has no HAG laz, so it falls back to a
z-scene-min proxy for the 6th channel (keeps IN_DIM fixed at 6).

Random initialization — no pretrained weights. Architecture and features are
identical to the warm sibling (modal_train_randlanet_warm.py), making
warm-vs-cold a clean initialization comparison.

IEEE 2019 Track 4 layout:
  Train-Track4/Track4/JAX_*_PC3.txt, OMA_*_PC3.txt        (110 scenes)
  Train-Track4-Truth/Track4-Truth/JAX_*_CLS.txt, ...      (per-point labels)
  Validate-Track4/Track4/*.txt                            (10 scenes)
  Validate-Track4-Truth/*.txt                             (per-point labels)
  Test-Track4 has no GT — we ignore it.

Each PC3.txt row is `x, y, z, intensity, returnNumber` (comma-separated).
Each CLS.txt is one ASPRS LAS class code per line. We remap
{0:-1 (ignore), 2:0 Ground, 5:1 Trees, 6:2 Building, 9:3 Water, 17:4 Bridge}.

REVISION 2026-06-12 (ported from the KPConvX cold-run fixes):
  - features are [xyz, intensity, return_number] (5 ch). Intensity is
    water's only reliable cue; fc0 is built at 5->8 (cold start, so there is
    no pretrained-stem constraint).
  - p95-clipped per-scene intensity normalization (new PREP_DIR)
  - train augmentation (vertical rotation, x-flip, isotropic scale): was none
  - class-weighted CE + rare-class-centered sphere sampling (rare = train
    frequency < 2%, so canonical --dataset runs work too)
  - held-out val pass every VAL_EVERY epochs -> val_metrics.csv (no weight
    updates), checkpoints include optimizer state
  - final eval now covers every subsampled point of every val/test scene in
    spatially-sorted blocks (the old eval scored ONE random sphere per scene)

Validate-Track4 (10 scenes) is the test set. 10 Train-Track4 scenes picked
deterministically (seed=42) form an in-distribution validation holdout.

The warm-start sibling script is modal_train_randlanet_warm.py.

----------------------------------------------------------------------------
Training-terminal integration: running with no flags behaves exactly as the
original. Extra flags (see modal_train_ptv3_warm.py for details):

  --dataset NAME                          canonical trainer_gui dataset on the
                                          terminal-datasets volume
  --sub-grid / --num-points / --epochs / --batch / --steps-per-epoch
  --mode infer --weights runs/<id>/final_model.pth --infer-input <job_id>

GPU type / timeout come from TT_GPU / TT_TIMEOUT_HOURS env vars.
"""

import os
from typing import Optional

import modal

# ============================================================================
# Configuration
# ============================================================================
APP_NAME      = "randlanet-cold-ieee-hag"
GPU_TYPE      = os.environ.get("TT_GPU", "A10G")   # RandLA is light, A10G handles it
N_EPOCHS      = 100              # was 5 (smoke test); 250-300 for a full run
BATCH_SIZE    = 6
VAL_BATCH     = 12
TIMEOUT_HOURS = int(os.environ.get("TT_TIMEOUT_HOURS", "24"))

NUM_CLASSES   = 5                # IEEE Track 4: Ground, Trees, Building, Water, Bridge
NUM_POINTS    = 45056            # 4096*11, RandLA SemKITTI default
SUB_GRID_SIZE = 0.30             # 30 cm — IEEE LiDAR is sparser (~2 pts/m²) than KITTI
IN_DIM        = 6                # [x, y, z, intensity, return_number, HAG]
N_VAL_HOLDOUT = 10               # number of train scenes held out for val
HOLDOUT_SEED  = 42

# Class balance (rare classes derived from train frequency, so this also works
# for canonical --dataset runs with arbitrary class sets).
CLASS_WEIGHTING  = True
WEIGHT_BETA      = 0.5           # inverse-frequency exponent. 0.5 == inverse
                                 # SQRT frequency (w = 1/sqrt(freq)), the
                                 # RandLA-Net / SemanticKITTI standard: sub-linear
                                 # so rare classes are boosted without the
                                 # exploding-gradient instability of raw 1/freq.
WEIGHT_CAP       = 5.0           # clamp weights to [1/CAP, CAP] after mean-norm
# Lovász-Softmax: a tractable surrogate that optimizes mIoU (Jaccard) directly
# and weights every class equally, so it counters CE's majority-class bias on
# rare classes. Total loss = <pointwise> + LOVASZ_WEIGHT * lovasz_softmax.
# Set to 0.0 to disable (recovers a pointwise-only loss).
LOVASZ_WEIGHT    = 1.0
# Focal loss (Lin et al. 2017): when USE_FOCAL, the pointwise term is
# alpha-balanced focal loss instead of weighted cross-entropy. alpha reuses the
# inverse-sqrt class weights; (1-p_t)^gamma down-weights easy/well-classified
# points so hard + rare points dominate the gradient. gamma=0 == weighted CE.
# USE_FOCAL=False reverts the pointwise term to weighted CE.
USE_FOCAL        = False
FOCAL_GAMMA      = 2.0
RARE_OVERSAMPLE  = True
RARE_FREQ_THRESH = 0.02          # classes under 2% of train points count as rare
RARE_CENTER_PROB = 0.25          # P(center the next train sphere on a rare-class point)
VAL_EVERY        = 10            # held-out val pass every N epochs (no weight updates)
VAL_BATCHES      = 16            # batches per periodic val pass

DATA_ROOT      = "/data/IEEE"
TRAIN_PC_DIR   = f"{DATA_ROOT}/Train-Track4/Track4"
TRAIN_CLS_DIR  = f"{DATA_ROOT}/Train-Track4-Truth/Track4-Truth"
TEST_PC_DIR    = f"{DATA_ROOT}/Validate-Track4/Track4"
TEST_CLS_DIR   = f"{DATA_ROOT}/Validate-Track4-Truth"
# Per-point HeightAboveGround from the Pretraining tab (HAGTEST upload). Train +
# val-holdout scenes -> HAG/Train; the test split (Validate-Track4) -> HAG/Validate.
HAG_TRAIN_DIR  = f"{DATA_ROOT}/HAG/Train"
HAG_TEST_DIR   = f"{DATA_ROOT}/HAG/Validate"
PREP_DIR       = f"{DATA_ROOT}/prep/randlanet_hag_grid30_p95_origin"   # p95 intensity norm + HAG

# ASPRS LAS code -> contiguous 0..4 IEEE class index. Class 0 (Unclassified)
# maps to -1 and is ignored by the CE loss.
CLASS_NAMES = ["Ground", "Trees", "Building", "Water", "Bridge"]
LABEL_MAP   = {0: -1, 2: 0, 5: 1, 6: 2, 9: 3, 17: 4}

DATASETS_ROOT = "/datasets"   # terminal-datasets volume (trainer_gui canonical datasets)

# ============================================================================
# Image
# ============================================================================

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential", "cmake", "ninja-build", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "numpy<2.0",
        "scipy",
        "scikit-learn",
        "tqdm",
        "tensorboard",
        "pyyaml",
        "matplotlib",
        "Cython",
        "pandas<3",
        "laspy",
        "lazrs",          # LAZ backend so laspy can read the HAG .laz scenes
        index_url="https://download.pytorch.org/whl/cu121",
        extra_index_url="https://pypi.org/simple",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

image = image.add_local_dir(
    "C:/Users/OrionHoch/Desktop/testSem/RandLA-Net-pytorch",
    "/opt/randlanet",
    copy=True,
)

# Compile cpp wrappers and nearest_neighbors at image-build time. The upstream
# setup.py lists knn.pyx, which newer Cython/distutils mangle; the repo ships a
# pre-cythonized knn.cpp, so we rewrite setup.py to build from it directly.
_NN_SETUP = r"""
from setuptools import setup, Extension
import numpy
setup(
    name='nearest_neighbors',
    ext_modules=[Extension(
        'nearest_neighbors',
        sources=['knn.cpp', 'knn_.cxx'],
        include_dirs=['./', numpy.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11', '-fopenmp'],
        extra_link_args=['-std=c++11', '-fopenmp'],
    )],
)
"""

image = image.run_commands(
    f"cat > /opt/randlanet/utils/nearest_neighbors/setup.py <<'PY'\n{_NN_SETUP}\nPY",
    "cd /opt/randlanet/utils/nearest_neighbors && python setup.py build_ext --inplace",
    "mkdir -p /opt/randlanet/utils/nearest_neighbors/lib/python "
    " && touch /opt/randlanet/utils/nearest_neighbors/__init__.py "
    "          /opt/randlanet/utils/nearest_neighbors/lib/__init__.py "
    "          /opt/randlanet/utils/nearest_neighbors/lib/python/__init__.py "
    " && cp /opt/randlanet/utils/nearest_neighbors/nearest_neighbors*.so "
    "       /opt/randlanet/utils/nearest_neighbors/lib/python/",
    "cd /opt/randlanet/utils/cpp_wrappers/cpp_subsampling && python setup.py build_ext --inplace",
    "touch /opt/randlanet/utils/cpp_wrappers/__init__.py "
    "      /opt/randlanet/utils/cpp_wrappers/cpp_subsampling/__init__.py",
)

image = image.add_local_file("scripts/local/local_train_randlanet_hag.py", "/root/local_train_randlanet_hag.py")
image = image.add_local_file("scripts/helper/train_common.py", "/root/train_common.py")

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
             DATASETS_ROOT: datasets_volume},
    cpu=8,
    memory=32768,
    timeout=TIMEOUT_HOURS * 3600,
)
def train_randlanet(dataset: Optional[str] = None, sub_grid: Optional[float] = None,
                    num_points: Optional[int] = None, epochs: Optional[int] = None,
                    batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                    mode: str = "train", weights: Optional[str] = None,
                    infer_input: Optional[str] = None):
    """Modal shell: provision the GPU container + volumes, then run the LOCAL
    trainer. All training/inference logic lives in local_train_randlanet_hag.py — this only
    shells out to it, so local and cloud run byte-identical code."""
    import subprocess
    import sys
    import threading

    cmd = [sys.executable, "/root/local_train_randlanet_hag.py"]
    for _flag, _val in (
        ("--dataset", dataset),
        ("--sub-grid", sub_grid),
        ("--num-points", num_points),
        ("--epochs", epochs),
        ("--batch", batch),
        ("--steps-per-epoch", steps_per_epoch),
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
def main(dataset: Optional[str] = None, sub_grid: Optional[float] = None,
         num_points: Optional[int] = None, epochs: Optional[int] = None,
         batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
         mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None):
    # .remote() keeps the local CLI attached so logs stream in real time.
    # Pair with `modal run --detach ...` if you want to close the terminal
    # mid-run; you can then reattach with `modal app logs {APP_NAME} -f`
    # while the app is still active.
    what = f"infer({weights})" if mode == "infer" else f"train({dataset or 'IEEE legacy'})"
    print(f"Launching {APP_NAME} [{what}] on {GPU_TYPE} for up to {TIMEOUT_HOURS}h.")
    train_randlanet.remote(dataset=dataset, sub_grid=sub_grid, num_points=num_points,
                           epochs=epochs, batch=batch, steps_per_epoch=steps_per_epoch,
                           mode=mode, weights=weights, infer_input=infer_input)
