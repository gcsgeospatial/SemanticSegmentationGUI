"""Modal shell for RandLA-Net (cold-start) - shells out to
local_train_randlanet.py so local and cloud run identical code. Flags:
--dataset --sub-grid --num-points --epochs --batch --steps-per-epoch;
--mode infer --weights --infer-input. GPU/timeout from TT_GPU / TT_TIMEOUT_HOURS."""

import os
import sys
from typing import Optional

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local run
sys.path.insert(0, "/root")                                     # in-container
import _shell

APP_NAME = "randlanet-cold"

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

# pinned upstream clone - the SHA IS the architecture version
image = image.run_commands(
    "git clone https://github.com/tsunghan-wu/RandLA-Net-pytorch.git /opt/randlanet"
    " && git -C /opt/randlanet checkout --detach 75adeacdb796db07e69ba990c36409c5d3ee886b"
    " && rm -rf /opt/randlanet/.git",
)

# upstream setup.py lists knn.pyx (newer Cython mangles it); build from the shipped pre-cythonized knn.cpp instead
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
image = image.add_local_file("scripts/modal/_shell.py", "/root/_shell.py")

app, outputs_volume, datasets_volume, _fn_kwargs, _launch = _shell.setup(
    APP_NAME, gpu_default="A10G", memory=32768)


@app.function(image=image, **_fn_kwargs)
def train_randlanet(dataset: Optional[str] = None, sub_grid: Optional[float] = None,
                    num_points: Optional[int] = None, epochs: Optional[int] = None,
                    batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                    mode: str = "train", weights: Optional[str] = None,
                    infer_input: Optional[str] = None,
                    env_json: Optional[str] = None):
    """Shell out to the local trainer - local and cloud run identical code."""
    import sys
    sys.path.insert(0, "/root")
    import train_common
    train_common.modal_entry(
        modal.current_function_call_id(), "/root/local_train_randlanet.py",
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
        env_json, outputs_volume, datasets_volume)


@app.local_entrypoint()
def main(dataset: Optional[str] = None, sub_grid: Optional[float] = None,
         num_points: Optional[int] = None, epochs: Optional[int] = None,
         batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
         mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None,
         env_json: Optional[str] = None):
    _launch(train_randlanet, **locals())
