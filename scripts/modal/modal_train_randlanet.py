"""
Modal shell for RandLA-Net (PyTorch), COLD-START — thin subprocess wrapper.

Random initialization (no pretrained weights). Provisions a GPU container + the
outputs / terminal-datasets volumes, then shells out to the local trainer
(scripts/local/local_train_randlanet.py) so local and cloud run byte-identical
code. Trains on a canonical trainer_gui dataset passed via --dataset (staged on
the terminal-datasets volume).

  --dataset NAME                          canonical trainer_gui dataset
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
APP_NAME      = "randlanet-cold"
GPU_TYPE      = os.environ.get("TT_GPU", "A10G")   # RandLA is light, A10G handles it
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
        index_url="https://download.pytorch.org/whl/cu121",
        extra_index_url="https://pypi.org/simple",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

# Model source: pinned upstream clone — portable (no local checkout needed to
# build). Bump the SHA deliberately: it IS the architecture version.
image = image.run_commands(
    "git clone https://github.com/tsunghan-wu/RandLA-Net-pytorch.git /opt/randlanet"
    " && git -C /opt/randlanet checkout --detach 75adeacdb796db07e69ba990c36409c5d3ee886b"
    " && rm -rf /opt/randlanet/.git",
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

image = image.add_local_file("scripts/local/local_train_randlanet.py", "/root/local_train_randlanet.py")
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
    volumes={"/outputs": outputs_volume, DATASETS_ROOT: datasets_volume},
    cpu=8,
    memory=32768,
    timeout=TIMEOUT_HOURS * 3600,
)
def train_randlanet(dataset: Optional[str] = None, sub_grid: Optional[float] = None,
                    num_points: Optional[int] = None, epochs: Optional[int] = None,
                    batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                    mode: str = "train", weights: Optional[str] = None,
                    infer_input: Optional[str] = None,
               env_json: Optional[str] = None):
    """Modal shell: provision the GPU container + volumes, then run the LOCAL
    trainer. All training/inference logic lives in local_train_randlanet.py — this only
    shells out to it, so local and cloud run byte-identical code."""
    import sys
    sys.path.insert(0, "/root")
    import train_common
    train_common.modal_shell_run(
        "/root/local_train_randlanet.py",
        [
            ("--dataset", dataset),
            ("--sub-grid", sub_grid),
            ("--num-points", num_points),
            ("--epochs", epochs),
            ("--batch", batch),
            ("--steps-per-epoch", steps_per_epoch),
            ("--mode", mode),
            ("--weights", weights),
            ("--infer-input", infer_input),
        ],
        env_json,
        [outputs_volume, datasets_volume],
    )


@app.local_entrypoint()
def main(dataset: Optional[str] = None, sub_grid: Optional[float] = None,
         num_points: Optional[int] = None, epochs: Optional[int] = None,
         batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
         mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None,
               env_json: Optional[str] = None):
    # .remote() keeps the local CLI attached so logs stream in real time.
    # Pair with `modal run --detach ...` if you want to close the terminal
    # mid-run; you can then reattach with `modal app logs {APP_NAME} -f`
    # while the app is still active.
    what = f"infer({weights})" if mode == "infer" else f"train({dataset})"
    print(f"Launching {APP_NAME} [{what}] on {GPU_TYPE} for up to {TIMEOUT_HOURS}h.")
    train_randlanet.remote(dataset=dataset, sub_grid=sub_grid, num_points=num_points,
                           epochs=epochs, batch=batch, steps_per_epoch=steps_per_epoch,
                           mode=mode, weights=weights, infer_input=infer_input, env_json=env_json)
