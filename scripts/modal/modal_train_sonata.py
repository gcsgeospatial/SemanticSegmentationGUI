"""
Modal shell for Sonata (pretrained Pointcept-SSL encoder) — thin subprocess wrapper.

Provisions a GPU container + the outputs / terminal-datasets volumes, then shells
out to the local trainer (scripts/local/local_train_sonata.py, a thin wrapper
over local_train_concerto.py) so local and cloud run byte-identical code.
Fine-tunes the sonata pretrained encoder (HuggingFace facebook/sonata, cached
on the outputs volume) on a canonical trainer_gui dataset passed via --dataset.

  --dataset NAME                          canonical trainer_gui dataset
  --grid / --chunk-xy / --epochs / --batch / --steps-per-epoch
  --freeze-encoder 0|1                    1 = linear probe (head only)
  --mode infer --weights runs/<id>/final_model.pth --infer-input <job_id>

GPU type / timeout come from TT_GPU / TT_TIMEOUT_HOURS env vars.
"""

import os
from typing import Optional

import modal

# ============================================================================
# Configuration
# ============================================================================
APP_NAME      = "sonata"
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
    # Concerto/Sonata/Utonia stack: torch 2.5 + CUDA 12.4 + spconv-cu124 (the
    # upstream environment.yml combo). No flash-attn — the trainer runs the
    # upstream enable_flash=False fallback (standard attention, patch 1024),
    # exactly like the upstream demos on non-flash setups. huggingface_hub
    # pulls the pretrained checkpoint at first run (cached under
    # /outputs/hf_cache, so it persists on the outputs volume).
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

# Model source: pinned upstream clone — portable (no local checkout needed to
# build). Bump the SHA deliberately: it IS the architecture version. The repo
# root holds the sonata/ package dir; the trainer adds /opt/sonata to
# sys.path and imports sonata.model.
image = image.run_commands(
    "git clone https://github.com/facebookresearch/sonata.git /opt/sonata"
    " && git -C /opt/sonata checkout --detach 18c09ff8d713494f78a8213792262b910977a65d"
    " && rm -rf /opt/sonata/.git",
)

# the by-filename entry point + the shared pcssl core it delegates to
image = image.add_local_file("scripts/local/local_train_sonata.py", "/root/local_train_sonata.py")
image = image.add_local_file("scripts/local/local_train_concerto.py", "/root/local_train_concerto.py")
image = image.add_local_file("scripts/helper/train_common.py", "/root/train_common.py")
# density.py: the DG/env-knob helper every local trainer imports (`import density
# as dg`) — without it cloud runs die on ModuleNotFoundError at startup.
image = image.add_local_file("scripts/helper/density.py", "/root/density.py")

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
    # in-process). Each retry auto-resumes from the latest checkpoint (the
    # shared concerto trainer's AUTO_RESUME machinery), so an intermittent
    # crash costs only the epochs since the last checkpoint.
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def train_sonata(dataset: Optional[str] = None, grid: Optional[float] = None,
                 epochs: Optional[int] = None, batch: Optional[int] = None,
                 steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
                 mode: str = "train", weights: Optional[str] = None,
                 infer_input: Optional[str] = None,
                 freeze_encoder: Optional[int] = None,
                 env_json: Optional[str] = None):
    """Modal shell: provision the GPU container + volumes, then run the LOCAL
    trainer. All training/inference logic lives in local_train_concerto.py
    (via the local_train_sonata.py wrapper) — this only shells out to it, so
    local and cloud run byte-identical code."""
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
        "/root/local_train_sonata.py",
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
        env_json,
        [outputs_volume, datasets_volume],
    )


@app.local_entrypoint()
def main(dataset: Optional[str] = None, grid: Optional[float] = None,
         epochs: Optional[int] = None, batch: Optional[int] = None,
         steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
         mode: str = "train", weights: Optional[str] = None,
         infer_input: Optional[str] = None,
         freeze_encoder: Optional[int] = None,
         env_json: Optional[str] = None):
    what = f"infer({weights})" if mode == "infer" else f"train({dataset})"
    print(f"Launching {APP_NAME} [{what}] on {GPU_TYPE} for up to {TIMEOUT_HOURS}h.")
    train_sonata.remote(dataset=dataset, grid=grid, epochs=epochs, batch=batch,
                        steps_per_epoch=steps_per_epoch, chunk_xy=chunk_xy, mode=mode,
                        weights=weights, infer_input=infer_input,
                        freeze_encoder=freeze_encoder, env_json=env_json)
