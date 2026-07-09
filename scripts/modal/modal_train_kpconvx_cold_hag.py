"""
Modal shell for KPConvX-L, COLD-START + HAG — thin subprocess wrapper.

HAG twin of modal_train_kpconvx_cold.py: the local trainer feeds a real
HeightAboveGround channel instead of the tile-relative height proxy (same 4 input
channels). Canonical --dataset runs without a real HAG feature fall back to the
height proxy (handled in the local trainer). Random init (no warm-start).

Provisions a GPU container + the outputs / terminal-datasets volumes, then shells
out to the local trainer (scripts/local/local_train_kpconvx_cold_hag.py) so local
and cloud run byte-identical code. Trains on a canonical trainer_gui dataset
passed via --dataset (staged on the terminal-datasets volume).

  --dataset NAME                          canonical trainer_gui dataset
  --grid / --chunk-xy / --epochs / --batch / --steps-per-epoch
  --mode eval --weights runs/<id>/final_model.pth     # voted re-score, no train
  --mode infer --weights runs/<id>/final_model.pth --infer-input <job_id>

GPU type / timeout come from TT_GPU / TT_TIMEOUT_HOURS env vars.
"""

import os
from typing import Optional

import modal

# ============================================================================
# Configuration
# ============================================================================
APP_NAME      = "kpconvx-cold-hag"
GPU_TYPE      = "A100"
TIMEOUT_HOURS = 24

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
# Model source: pinned upstream clone (the standalone KPConvX subtree) —
# portable (no local checkout needed to build). Bump the SHA deliberately:
# it IS the architecture version.
image = image.run_commands(
    "git clone https://github.com/apple/ml-kpconvx.git /tmp/ml-kpconvx"
    " && git -C /tmp/ml-kpconvx checkout --detach 54e644a9f3bddd4c344a58193897a44582b0fea4"
    " && mv /tmp/ml-kpconvx/Standalone/KPConvX /opt/kpconvx"
    " && rm -rf /tmp/ml-kpconvx",
)
image = image.run_commands(
    "cd /opt/kpconvx/cpp_wrappers/cpp_subsampling && python setup.py build_ext --inplace",
    "cd /opt/kpconvx/cpp_wrappers/cpp_neighbors && python setup.py build_ext --inplace",
    "touch /opt/kpconvx/cpp_wrappers/__init__.py "
    "      /opt/kpconvx/cpp_wrappers/cpp_subsampling/__init__.py "
    "      /opt/kpconvx/cpp_wrappers/cpp_neighbors/__init__.py",
)

image = image.add_local_file("scripts/local/local_train_kpconvx_cold_hag.py", "/root/local_train_kpconvx_cold_hag.py")
# the hag entry point is a thin wrapper since the merge — ship the real trainer too
image = image.add_local_file("scripts/local/local_train_kpconvx_cold.py", "/root/local_train_kpconvx_cold.py")
image = image.add_local_file("scripts/helper/train_common.py", "/root/train_common.py")
# density.py: the DG/env-knob helper every local trainer imports (`import density
# as dg`) — without it cloud runs die on ModuleNotFoundError at startup.
image = image.add_local_file("scripts/helper/density.py", "/root/density.py")

outputs_volume  = modal.Volume.from_name(f"{APP_NAME}-outputs",  create_if_missing=True)
datasets_volume = modal.Volume.from_name(
    os.environ.get("TT_DATASET_VOLUME", "terminal-datasets"), create_if_missing=True)

# ============================================================================
# Training function
# ============================================================================
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/outputs": outputs_volume, "/datasets": datasets_volume},
    cpu=8,
    memory=49152,
    timeout=TIMEOUT_HOURS * 3600,
)
def train_kpconvx(dataset: Optional[str] = None, mode: str = "train",
                  weights: Optional[str] = None,
                  infer_input: Optional[str] = None, grid: Optional[float] = None,
                  chunk_xy: Optional[float] = None, epochs: Optional[int] = None,
                  batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                  env_json: Optional[str] = None):
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
    env = dict(os.environ)
    if env_json:
        import json
        _ov = {str(k): str(v) for k, v in json.loads(env_json).items()}
        env.update(_ov)
        print("[modal-shell] env overrides: " + " ".join(sorted(_ov)), flush=True)
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
        subprocess.run(cmd, check=True, env=env)
    finally:
        _stop.set()
        outputs_volume.commit()
        datasets_volume.commit()


@app.local_entrypoint()
def main(dataset: Optional[str] = None, mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None, grid: Optional[float] = None,
         chunk_xy: Optional[float] = None, epochs: Optional[int] = None,
         batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                  env_json: Optional[str] = None):
    what = {"eval": "eval-only re-score", "infer": f"infer({weights})"}.get(
        mode, f"train({dataset})")
    print(f"Launching {APP_NAME} [{what}] on {GPU_TYPE} for up to {TIMEOUT_HOURS}h.")
    train_kpconvx.remote(dataset=dataset, mode=mode, weights=weights, infer_input=infer_input,
                         grid=grid, chunk_xy=chunk_xy, epochs=epochs, batch=batch,
                         steps_per_epoch=steps_per_epoch, env_json=env_json)
