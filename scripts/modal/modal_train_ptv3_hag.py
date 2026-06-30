"""
Modal shell for PointTransformerV3 — HAG variant, thin subprocess wrapper.

HAG twin of modal_train_ptv3.py: the local trainer appends an extra HAG channel
(in_channels 6 -> 7). Canonical --dataset runs without a real HAG feature fall
back to a z-scene-min proxy for the 7th channel (handled in the local trainer).

Provisions a GPU container + the outputs / terminal-datasets volumes, then shells
out to the local trainer (scripts/local/local_train_ptv3_hag.py) so local and
cloud run byte-identical code. Trains on a canonical trainer_gui dataset passed
via --dataset (staged on the terminal-datasets volume).

  --dataset NAME                          canonical trainer_gui dataset
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
APP_NAME      = "ptv3-hag"
GPU_TYPE      = os.environ.get("TT_GPU", "A100")
TIMEOUT_HOURS = int(os.environ.get("TT_TIMEOUT_HOURS", "24"))

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

image = image.add_local_file("scripts/local/local_train_ptv3_hag.py", "/root/local_train_ptv3_hag.py")
image = image.add_local_file("scripts/helper/train_common.py", "/root/train_common.py")

outputs_volume  = modal.Volume.from_name(f"{APP_NAME}-outputs", create_if_missing=True)
datasets_volume = modal.Volume.from_name(
    os.environ.get("TT_DATASET_VOLUME", "terminal-datasets"), create_if_missing=True)


# ============================================================================
# Training function
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/outputs": outputs_volume, DATASETS_ROOT: datasets_volume},
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
               infer_input: Optional[str] = None):
    """Modal shell: provision the GPU container + volumes, then run the LOCAL
    trainer. All training/inference logic lives in local_train_ptv3_hag.py — this only
    shells out to it, so local and cloud run byte-identical code."""
    import subprocess
    import sys
    import threading

    cmd = [sys.executable, "/root/local_train_ptv3_hag.py"]
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
            outputs_volume.commit()
            datasets_volume.commit()

    _t = threading.Thread(target=_commit_loop, daemon=True)
    _t.start()
    try:
        subprocess.run(cmd, check=True)
    finally:
        _stop.set()
        outputs_volume.commit()
        datasets_volume.commit()


@app.local_entrypoint()
def main(dataset: Optional[str] = None, grid: Optional[float] = None,
         epochs: Optional[int] = None, batch: Optional[int] = None,
         steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
         mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None):
    what = f"infer({weights})" if mode == "infer" else f"train({dataset})"
    print(f"Launching {APP_NAME} [{what}] on {GPU_TYPE} for up to {TIMEOUT_HOURS}h.")
    train_ptv3.remote(dataset=dataset, grid=grid, epochs=epochs, batch=batch,
                      steps_per_epoch=steps_per_epoch, chunk_xy=chunk_xy, mode=mode,
                      weights=weights, infer_input=infer_input)
