"""Modal shell for PointTransformerV3 - shells out to local_train_ptv3.py so
local and cloud run identical code. Flags: --dataset --grid --chunk-xy
--epochs --batch --steps-per-epoch; --mode infer --weights --infer-input.
GPU/timeout from TT_GPU / TT_TIMEOUT_HOURS."""

import os
import sys
from typing import Optional

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local run
sys.path.insert(0, "/root")                                     # in-container
import _shell

APP_NAME = "ptv3"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential", "cmake", "ninja-build", "libgl1", "libglib2.0-0")
    # torch 2.1 + cu118 + spconv-cu118: the cu124 build device-asserts and fails NVRTC, and cu118 is what PTv3 is developed against
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

# pinned upstream clone - the SHA IS the architecture version
image = image.run_commands(
    "git clone https://github.com/Pointcept/PointTransformerV3.git /opt/ptv3"
    " && git -C /opt/ptv3 checkout --detach 3229e9b7de1770c8ad17c316f8e349982de509f8"
    " && rm -rf /opt/ptv3/.git",
)
# package-relative imports: import as ptv3.model (/opt on sys.path)
image = image.run_commands("touch /opt/ptv3/__init__.py")

image = image.add_local_file("scripts/local/local_train_ptv3.py", "/root/local_train_ptv3.py")
image = image.add_local_file("scripts/helper/train_common.py", "/root/train_common.py")
image = image.add_local_file("scripts/modal/_shell.py", "/root/_shell.py")

app, outputs_volume, datasets_volume, _fn_kwargs, _launch = _shell.setup(APP_NAME)


@app.function(image=image, **_fn_kwargs)
def train_ptv3(dataset: Optional[str] = None, grid: Optional[float] = None,
               epochs: Optional[int] = None, batch: Optional[int] = None,
               steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
               mode: str = "train", weights: Optional[str] = None,
               infer_input: Optional[str] = None,
               env_json: Optional[str] = None):
    """Shell out to the local trainer - local and cloud run identical code."""
    import sys
    sys.path.insert(0, "/root")
    import train_common
    train_common.modal_entry(
        modal.current_function_call_id(), "/root/local_train_ptv3.py",
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
        env_json, outputs_volume, datasets_volume)


@app.local_entrypoint()
def main(dataset: Optional[str] = None, grid: Optional[float] = None,
         epochs: Optional[int] = None, batch: Optional[int] = None,
         steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
         mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None,
         env_json: Optional[str] = None):
    _launch(train_ptv3, **locals())
