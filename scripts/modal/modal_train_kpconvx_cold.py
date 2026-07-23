"""Modal shell for KPConvX-L (cold-start) — shells out to
local_train_kpconvx_cold.py so local and cloud run identical code. Flags:
--dataset --grid --chunk-xy --epochs --batch --steps-per-epoch; --mode
eval|infer --weights --infer-input. GPU/timeout from TT_GPU / TT_TIMEOUT_HOURS."""

import os
from typing import Optional

import modal

APP_NAME      = "kpconvx-cold"
GPU_TYPE      = os.environ.get("TT_GPU", "A100")
TIMEOUT_HOURS = int(os.environ.get("TT_TIMEOUT_HOURS", "24"))

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
        "torch-cluster",
        "tqdm",
        "tensorboard",
        "pandas<3",
        index_url="https://download.pytorch.org/whl/cu121",
        extra_index_url="https://pypi.org/simple",
        find_links="https://data.pyg.org/whl/torch-2.3.0+cu121.html",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

# pinned clone of the keops-free fork (torch-cluster neighbor search) —
# the SHA IS the architecture version
image = image.run_commands(
    "git clone https://github.com/orion-hoch/ml-kpconvx-windows-acessible.git /tmp/ml-kpconvx"
    " && git -C /tmp/ml-kpconvx checkout --detach b2cd23ccac54342780980124b8b9e419e339672d"
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

image = image.add_local_file("scripts/local/local_train_kpconvx_cold.py", "/root/local_train_kpconvx_cold.py")
image = image.add_local_file("scripts/helper/train_common.py", "/root/train_common.py")
image = image.add_local_file("scripts/helper/density.py", "/root/density.py")

outputs_volume  = modal.Volume.from_name(f"{APP_NAME}-outputs",  create_if_missing=True)
datasets_volume = modal.Volume.from_name(
    os.environ.get("TT_DATASET_VOLUME", "terminal-datasets"), create_if_missing=True)

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/outputs": outputs_volume, "/datasets": datasets_volume},
    cpu=8,
    memory=49152,
    timeout=TIMEOUT_HOURS * 3600,
    # auto-restart on failure; each retry auto-resumes from the last checkpoint
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def train_kpconvx(dataset: Optional[str] = None, mode: str = "train",
                  weights: Optional[str] = None,
                  infer_input: Optional[str] = None, grid: Optional[float] = None,
                  chunk_xy: Optional[float] = None, epochs: Optional[int] = None,
                  batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                  env_json: Optional[str] = None):
    """Shell out to the local trainer — local and cloud run identical code."""
    import sys
    sys.path.insert(0, "/root")
    # resume only on Modal's OWN retries (call id stable across retries, new
    # per `modal run`). ponytail: /outputs/.attempts markers never cleaned.
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
        "/root/local_train_kpconvx_cold.py",
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
    train_kpconvx.remote(dataset=dataset, mode=mode, weights=weights, infer_input=infer_input,
                         grid=grid, chunk_xy=chunk_xy, epochs=epochs, batch=batch,
                         steps_per_epoch=steps_per_epoch, env_json=env_json)
