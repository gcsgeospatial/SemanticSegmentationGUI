"""Modal shell for PointTransformerV3 — shells out to local_train_ptv3.py so
local and cloud run identical code. Flags: --dataset --grid --chunk-xy
--epochs --batch --steps-per-epoch; --mode infer --weights --infer-input.
GPU/timeout from TT_GPU / TT_TIMEOUT_HOURS."""

import os
from typing import Optional

import modal

APP_NAME      = "ptv3"
GPU_TYPE      = os.environ.get("TT_GPU", "A100")
TIMEOUT_HOURS = int(os.environ.get("TT_TIMEOUT_HOURS", "24"))

DATASETS_ROOT = "/datasets"   # terminal-datasets volume

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential", "cmake", "ninja-build", "libgl1", "libglib2.0-0")
    # torch 2.1 + cu118 + spconv-cu118: the cu124 build device-asserts and
    # fails NVRTC — cu118 is what PTv3 is developed against
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

# pinned upstream clone — the SHA IS the architecture version
image = image.run_commands(
    "git clone https://github.com/Pointcept/PointTransformerV3.git /opt/ptv3"
    " && git -C /opt/ptv3 checkout --detach 3229e9b7de1770c8ad17c316f8e349982de509f8"
    " && rm -rf /opt/ptv3/.git",
)
# package-relative imports: import as ptv3.model (/opt on sys.path)
image = image.run_commands("touch /opt/ptv3/__init__.py")

image = image.add_local_file("scripts/local/local_train_ptv3.py", "/root/local_train_ptv3.py")
image = image.add_local_file("scripts/helper/train_common.py", "/root/train_common.py")
image = image.add_local_file("scripts/helper/density.py", "/root/density.py")

outputs_volume  = modal.Volume.from_name(f"{APP_NAME}-outputs", create_if_missing=True)
datasets_volume = modal.Volume.from_name(
    os.environ.get("TT_DATASET_VOLUME", "terminal-datasets"), create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/outputs": outputs_volume, DATASETS_ROOT: datasets_volume},
    cpu=8,
    memory=49152,
    timeout=TIMEOUT_HOURS * 3600,
    # auto-restart on failure; each retry auto-resumes from the last checkpoint
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def train_ptv3(dataset: Optional[str] = None, grid: Optional[float] = None,
               epochs: Optional[int] = None, batch: Optional[int] = None,
               steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
               mode: str = "train", weights: Optional[str] = None,
               infer_input: Optional[str] = None,
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
        "/root/local_train_ptv3.py",
        [
            ("--dataset", dataset),
            ("--grid", grid),
            ("--epochs", epochs),
            ("--batch", batch),
            ("--steps-per-epoch", steps_per_epoch),
            ("--chunk-xy", chunk_xy),
            ("--mode", mode),
            ("--weights", weights),
            ("--infer-input", infer_input),
        ],
        env_json,
        [outputs_volume, datasets_volume],
    )


@app.local_entrypoint()
def main(dataset: Optional[str] = None, grid: Optional[float] = None,
         epochs: Optional[int] = None, batch: Optional[int] = None,
         steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
         mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None,
               env_json: Optional[str] = None):
    what = f"infer({weights})" if mode == "infer" else f"train({dataset})"
    print(f"Launching {APP_NAME} [{what}] on {GPU_TYPE} for up to {TIMEOUT_HOURS}h.")
    train_ptv3.remote(dataset=dataset, grid=grid, epochs=epochs, batch=batch,
                      steps_per_epoch=steps_per_epoch, chunk_xy=chunk_xy, mode=mode,
                      weights=weights, infer_input=infer_input, env_json=env_json)
