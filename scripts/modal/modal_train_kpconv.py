"""Modal shell for the original KPConv (cold-start) - shells out to
local_train_kpconv.py so local and cloud run identical code. Flags: --dataset
--grid --chunk-xy --epochs --batch --steps-per-epoch; --mode eval|infer
--weights --infer-input. GPU/timeout from TT_GPU / TT_TIMEOUT_HOURS."""

import os
import sys
from typing import Optional

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local run
sys.path.insert(0, "/root")                                     # in-container
import _shell

APP_NAME = "kpconv"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential", "cmake", "libgl1", "libglib2.0-0", "ninja-build")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "numpy<2.0",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "tensorboard",
        "pandas<3",
        index_url="https://download.pytorch.org/whl/cu121",
        extra_index_url="https://pypi.org/simple",
    )
    # MPLBACKEND=Agg: kernel_points.py imports pyplot at module top
    .env({"PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"})
)

# pinned upstream clone - the SHA IS the architecture version
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
    # pre-solve the K=15 'center' kernel disposition (unit scale, cached)
    "cd /opt/kpconv && MPLBACKEND=Agg python -c "
    "\"from kernels.kernel_points import load_kernels; "
    "load_kernels(1.0, 15, dimension=3, fixed='center')\"",
)

image = image.add_local_file("scripts/local/local_train_kpconv.py", "/root/local_train_kpconv.py")
image = image.add_local_file("scripts/helper/train_common.py", "/root/train_common.py")
image = image.add_local_file("scripts/modal/_shell.py", "/root/_shell.py")

app, outputs_volume, datasets_volume, _fn_kwargs, _launch = _shell.setup(APP_NAME)


@app.function(image=image, **_fn_kwargs)
def train_kpconv(dataset: Optional[str] = None, mode: str = "train",
                 weights: Optional[str] = None,
                 infer_input: Optional[str] = None, grid: Optional[float] = None,
                 chunk_xy: Optional[float] = None, epochs: Optional[int] = None,
                 batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                 env_json: Optional[str] = None):
    """Shell out to the local trainer - local and cloud run identical code."""
    import sys
    sys.path.insert(0, "/root")
    import train_common
    train_common.modal_entry(
        modal.current_function_call_id(), "/root/local_train_kpconv.py",
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
        env_json, outputs_volume, datasets_volume)


@app.local_entrypoint()
def main(dataset: Optional[str] = None, mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None, grid: Optional[float] = None,
         chunk_xy: Optional[float] = None, epochs: Optional[int] = None,
         batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
         env_json: Optional[str] = None):
    _launch(train_kpconv, **locals())
