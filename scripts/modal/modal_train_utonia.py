"""Modal shell for Utonia - shells out to local_train_utonia.py so local and
cloud run identical code. Flags: --dataset --grid --chunk-xy --epochs --batch
--steps-per-epoch --freeze-encoder; --mode infer --weights --infer-input.
GPU/timeout from TT_GPU / TT_TIMEOUT_HOURS."""

import os
import sys
from typing import Optional

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local run
sys.path.insert(0, "/root")                                     # in-container
import _shell

APP_NAME = "utonia"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential", "cmake", "ninja-build", "libgl1", "libglib2.0-0")
    # torch 2.5 + cu124 + spconv-cu124 (upstream combo); no flash-attn, the trainer uses the enable_flash=False fallback (HF cache: /outputs/hf_cache)
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
        "huggingface_hub",
        "packaging",
        index_url="https://download.pytorch.org/whl/cu124",
        extra_index_url="https://pypi.org/simple",
    )
    .pip_install(
        "spconv-cu124",
        "torch-scatter",
        find_links="https://data.pyg.org/whl/torch-2.5.0+cu124.html",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

# pinned upstream clone - the SHA IS the architecture version
image = image.run_commands(
    "git clone https://github.com/Pointcept/Utonia.git /opt/utonia"
    " && git -C /opt/utonia checkout --detach da776a0bd3a48c6df83ac2ae0e27b26141cc7e31"
    " && rm -rf /opt/utonia/.git",
)

image = image.add_local_file("scripts/local/local_train_utonia.py", "/root/local_train_utonia.py")
image = image.add_local_file("scripts/local/local_train_ptv3.py", "/root/local_train_ptv3.py")
image = image.add_local_file("scripts/helper/train_common.py", "/root/train_common.py")
image = image.add_local_file("scripts/modal/_shell.py", "/root/_shell.py")

app, outputs_volume, datasets_volume, _fn_kwargs, _launch = _shell.setup(APP_NAME)


@app.function(image=image, **_fn_kwargs)
def train_utonia(dataset: Optional[str] = None, grid: Optional[float] = None,
                 epochs: Optional[int] = None, batch: Optional[int] = None,
                 steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
                 mode: str = "train", weights: Optional[str] = None,
                 infer_input: Optional[str] = None,
                 freeze_encoder: Optional[int] = None,
                 env_json: Optional[str] = None):
    """Shell out to the local trainer - local and cloud run identical code."""
    import sys
    sys.path.insert(0, "/root")
    import train_common
    train_common.modal_entry(
        modal.current_function_call_id(), "/root/local_train_utonia.py",
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
            ("--freeze-encoder", freeze_encoder),
        ],
        env_json, outputs_volume, datasets_volume)


@app.local_entrypoint()
def main(dataset: Optional[str] = None, grid: Optional[float] = None,
         epochs: Optional[int] = None, batch: Optional[int] = None,
         steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
         mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None,
         freeze_encoder: Optional[int] = None,
         env_json: Optional[str] = None):
    _launch(train_utonia, **locals())
