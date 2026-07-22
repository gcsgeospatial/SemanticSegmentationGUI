"""
Modal shell for the ORIGINAL KPConv (deformable KPFCNN), COLD-START — thin
subprocess wrapper.

Random init (no warm-start). Provisions a GPU container + the outputs /
terminal-datasets volumes, then shells out to the local trainer
(scripts/local/local_train_kpconv.py) so local and cloud run byte-identical
code. Trains on a canonical trainer_gui dataset passed via --dataset (staged on
the terminal-datasets volume).

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
APP_NAME      = "kpconv"
GPU_TYPE      = os.environ.get("TT_GPU", "A100")
TIMEOUT_HOURS = int(os.environ.get("TT_TIMEOUT_HOURS", "24"))

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
        "numpy<2.0",       # keeps numpy.distutils importable for the cpp setup.py builds
        "scipy",
        "scikit-learn",
        "matplotlib",      # hard import of kernels/kernel_points.py (model build)
        "tqdm",
        "tensorboard",
        "pandas<3",
        index_url="https://download.pytorch.org/whl/cu121",
        extra_index_url="https://pypi.org/simple",
    )
    # MPLBACKEND=Agg: kernel_points.py imports pyplot at module top — never
    # probe a GUI backend in the container.
    .env({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"})
)

# Model source: pinned upstream clone (HuguesTHOMAS/KPConv-PyTorch) — portable
# (no local checkout needed to build). Bump the SHA deliberately: it IS the
# architecture version.
image = image.run_commands(
    "git clone https://github.com/HuguesTHOMAS/KPConv-PyTorch.git /opt/kpconv"
    " && git -C /opt/kpconv checkout --detach d19c575d3fa9fcfd5a74845b5b27aac7e50472c7",
)
image = image.run_commands(
    "cd /opt/kpconv/cpp_wrappers/cpp_subsampling && python setup.py build_ext --inplace",
    "cd /opt/kpconv/cpp_wrappers/cpp_neighbors && python setup.py build_ext --inplace",
    "touch /opt/kpconv/cpp_wrappers/__init__.py "
    "      /opt/kpconv/cpp_wrappers/cpp_subsampling/__init__.py "
    "      /opt/kpconv/cpp_wrappers/cpp_neighbors/__init__.py",
    # Pre-solve the one kernel disposition our config uses (K=15, 3D, 'center').
    # load_kernels caches it at UNIT scale (radius-independent), so this single
    # file serves every KPConv block; the trainer builds the net under
    # _cwd(/opt/kpconv), which is where this relative cache path is found.
    "cd /opt/kpconv && MPLBACKEND=Agg python -c "
    "\"from kernels.kernel_points import load_kernels; "
    "load_kernels(1.0, 15, dimension=3, fixed='center')\"",
)

image = image.add_local_file("scripts/local/local_train_kpconv.py", "/root/local_train_kpconv.py")
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
    # Auto-restart the container on failure (preemption / intermittent CUDA
    # crash). Each retry auto-resumes from the latest checkpoint (marker below),
    # so a preemption costs only the epochs since the last checkpoint.
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def train_kpconv(dataset: Optional[str] = None, mode: str = "train",
                 weights: Optional[str] = None,
                 infer_input: Optional[str] = None, grid: Optional[float] = None,
                 chunk_xy: Optional[float] = None, epochs: Optional[int] = None,
                 batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                 env_json: Optional[str] = None):
    """Modal shell: provision the GPU container + volumes, then run the LOCAL
    trainer. All training/inference logic lives in local_train_kpconv.py — this
    only shells out to it, so local and cloud run byte-identical code."""
    import sys
    sys.path.insert(0, "/root")
    # Resume only on Modal's OWN retries (preemption / crash), never on a user
    # relaunch -- parity with local, where a fresh launch is a fresh run. The
    # function-call id is stable across retries but new for every `modal run`,
    # so attempt 1 just drops a marker and any later attempt of the same call
    # resumes. No id (shouldn't happen in a container) -> resume, the safe side.
    # ponytail: markers accumulate under /outputs/.attempts (bytes each, never
    # cleaned) -- delete the dir if it ever bothers anyone.
    fcid = modal.current_function_call_id()
    marker = f"/outputs/.attempts/{fcid}" if fcid else ""
    if not fcid or os.path.exists(marker):
        os.environ["TT_MODAL_RETRY"] = os.environ["AUTO_RESUME"] = "1"
    else:
        os.makedirs("/outputs/.attempts", exist_ok=True)
        open(marker, "w").close()
        outputs_volume.commit()

    import train_common
    train_common.modal_shell_run(
        "/root/local_train_kpconv.py",
        [
            ("--dataset", dataset),
            ("--mode", mode),
            ("--weights", weights),
            ("--infer-input", infer_input),
            ("--grid", grid),
            ("--chunk-xy", chunk_xy),
            ("--epochs", epochs),
            ("--batch", batch),
            ("--steps-per-epoch", steps_per_epoch),
        ],
        env_json,
        [outputs_volume, datasets_volume],
    )


@app.local_entrypoint()
def main(dataset: Optional[str] = None, mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None, grid: Optional[float] = None,
         chunk_xy: Optional[float] = None, epochs: Optional[int] = None,
         batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
         env_json: Optional[str] = None):
    what = {"eval": "eval-only re-score", "infer": f"infer({weights})"}.get(
        mode, f"train({dataset})")
    print(f"Launching {APP_NAME} [{what}] on {GPU_TYPE} for up to {TIMEOUT_HOURS}h.")
    train_kpconv.remote(dataset=dataset, mode=mode, weights=weights, infer_input=infer_input,
                        grid=grid, chunk_xy=chunk_xy, epochs=epochs, batch=batch,
                        steps_per_epoch=steps_per_epoch, env_json=env_json)
