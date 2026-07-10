"""Builders for the LOCAL (Docker) execution path — the mirror of modal_cli.

Each backbone's local_train_*.py runs DIRECTLY inside its Docker image (built by
tools/gen_dockerfiles.py) — no modal, no shim — with host dirs bind-mounted to
the paths the trainer reads:

    /workspace  <- repo root (runs scripts/local/local_train_*.py from here)
    /datasets   <- staging root (canonical datasets + _infer/<job> live here)
    /outputs    <- the chosen output folder (training writes runs/<id>/..., weights read here)

No upload/download: the data is already on the host, and predictions/checkpoints
land straight back in the bind-mounted host dirs. Returns (program, args) for
JobRunner — exactly like modal_cli — so the GUI dispatch is backend-agnostic.
The modal_train_*.py scripts are now thin shells that subprocess these same
local_train_*.py inside a Modal container, so local and cloud run one codebase.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import appstate


def _posix(p: str) -> str:
    """Host path with forward slashes for `docker -v` (C:\\x -> C:/x; posix is
    unchanged) — matters when a Windows GUI drives a remote Linux daemon (L7)."""
    return Path(p).as_posix()


def docker_exe() -> str:
    return shutil.which("docker") or "docker"


def have_docker() -> bool:
    return shutil.which("docker") is not None


def image_for(backbone) -> str:
    """Resolve the image tag: an explicit per-backbone override wins; else a
    configured registry prefix turns the default into a pullable tag; else the
    bare local-build name."""
    cfg = appstate.local_config()
    override = cfg["images"].get(backbone.key)
    if override:
        return override
    base = f"trainer-local-{backbone.key}"
    reg = (cfg.get("registry") or "").rstrip("/")
    return f"{reg}/{base}" if reg else base


def is_pullable(backbone) -> bool:
    """True if image_for() is a registry tag `docker run` can auto-pull (a registry
    prefix is configured, or the override names a registry host)."""
    if appstate.local_config().get("registry"):
        return True
    tag = image_for(backbone)
    head = tag.split("/", 1)[0]
    return "/" in tag and ("." in head or ":" in head)   # host has a dot or port


def effective_image(backbone, here: "set[str] | None" = None) -> str:
    """The tag `docker run` should actually use. An explicit override always wins;
    otherwise prefer a locally-built bare `trainer-local-<key>` image when one is
    present, even if a registry prefix is configured — so a host that ran
    build_all.sh uses ITS image instead of (failing to) pull the registry tag (M3).
    Falls back to the configured tag when no local build exists."""
    cfg = appstate.local_config()
    if cfg["images"].get(backbone.key):
        return image_for(backbone)            # explicit override wins
    tag = image_for(backbone)
    bare = f"trainer-local-{backbone.key}"
    if tag != bare:                           # a registry prefix is in play
        here = present_images() if here is None else here
        if bare in here or f"{bare}:latest" in here:
            return bare
    return tag


def image_preflight(backbone) -> tuple[bool, str]:
    """(proceed, message) for the image. proceed=False blocks the launch; a
    non-empty message is shown either way — an FYI when docker will auto-pull, an
    error+how-to-fix when there's nothing to build or pull. Prefers a locally-built
    image over the registry tag (M3), so a host that built its own image isn't sent
    to pull (and fail on) a registry tag it never pushed."""
    here = present_images()
    img = effective_image(backbone, here)
    if img in here or f"{img}:latest" in here:
        if img != image_for(backbone):
            return True, (f"[local] using locally-built '{img}' "
                          f"(not pulling '{image_for(backbone)}').")
        return True, ""
    tag = image_for(backbone)
    if is_pullable(backbone):
        return True, (f"[local] '{tag}' not present locally - docker will pull it from the "
                      f"registry now (run `docker login` first if it's private).")
    return False, (f"[local] image '{tag}' isn't built on this host. Build it "
                   f"(bash docker/build_all.sh) - or set a registry (TT_REGISTRY or "
                   f"local_config['registry']) and `docker pull` it - then run again.")


def gpu_preflight() -> tuple[bool, str]:
    """(proceed, message) for GPU availability. The trainers are CUDA-only (every
    script calls .cuda()), so block when GPUs are disabled in local_config, and warn
    when no NVIDIA stack is detectable — otherwise docker fails with a cryptic
    'could not select device driver' (M4). There is no CPU inference path."""
    cfg = appstate.local_config()
    if not cfg.get("gpus"):
        return False, ("[local] GPUs are disabled (local_config['gpus']=''), but these "
                       "models require CUDA. Set gpus='all' (or a device id) and install "
                       "the NVIDIA Container Toolkit, then run again.")
    if shutil.which("nvidia-smi") is None:
        return True, ("[local] ⚠ no 'nvidia-smi' on PATH - if this host lacks an NVIDIA GPU "
                      "+ Container Toolkit the run will fail ('could not select device "
                      "driver'). These models are CUDA-only (no CPU fallback).")
    return True, ""


def _mounts(cfg: dict, repo_root: str, extra_mounts, outputs_root: str = "") -> list[str]:
    m = ["-v", f"{_posix(repo_root)}:/workspace",
         "-v", f"{_posix(cfg['datasets_root'])}:/datasets",
         "-v", f"{_posix(outputs_root or cfg['outputs_root'])}:/outputs"]
    for host, container in (extra_mounts or []):
        m += ["-v", f"{_posix(host)}:{container}"]
    return m


def run_script(script: str, flags: dict, backbone, *, repo_root: str,
               gpu: str = "", extra_mounts=None,
               outputs_root: str = "", env: "dict | None" = None) -> tuple[str, list[str]]:
    """`docker run --rm --gpus all -v ... <image> python scripts/local/local_train_<key>.py --flags`.

    `script` (the modal_train_*.py name) is accepted for call-site parity with
    modal_cli but unused: locally we run the decoupled local_train_<key>.py
    directly. `flags` are the same kebab-case flags; `extra_mounts` is a list of
    (host, container) bind pairs (e.g. a local .pth into /outputs). `outputs_root`
    overrides the host dir bound to /outputs (the user-picked output folder).
    """
    cfg = appstate.local_config()
    # --ipc=host gives the container the host's /dev/shm. Docker's default is 64 MB,
    # which PyTorch DataLoader workers (num_workers>0) overflow on larger batches ->
    # "Bus error … insufficient shared memory" SIGBUS at the epoch boundary.
    # ponytail: host IPC = zero tuning, never too small; if sharing the host IPC
    # namespace is ever undesirable, swap for ["--shm-size", "8g"].
    args = ["run", "--rm", "--ipc=host", "-w", "/workspace"]
    if cfg.get("gpus"):
        args += ["--gpus", str(cfg["gpus"])]
    args += _mounts(cfg, repo_root, extra_mounts, outputs_root=outputs_root)
    if gpu:
        args += ["-e", f"TT_GPU={gpu}"]          # cosmetic locally; keeps log parity
    for k, v in (env or {}).items():             # e.g. DG_* density-generalization flags
        args += ["-e", f"{k}={v}"]
    args += list(cfg.get("extra_args", []))
    args += [effective_image(backbone), "python", f"scripts/local/local_train_{backbone.key}.py"]
    for key, val in flags.items():
        if val is None:
            continue
        args += [f"--{key}", str(val)]
    return docker_exe(), args


def pull(backbone) -> tuple[str, list[str]]:
    """`docker pull <tag>` for a backbone's image. Reuses image_for so the pulled
    tag is exactly the one `docker run` will look for."""
    return docker_exe(), ["pull", image_for(backbone)]


def present_images() -> set[str]:
    """`repo:tag` of every image present locally, from ONE `docker images` call —
    the bulk path for the GUI's status refresh (per-image `docker image inspect`
    calls can each block to their timeout when the daemon is down)."""
    try:
        r = subprocess.run([docker_exe(), "images", "--format", "{{.Repository}}:{{.Tag}}"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return set(r.stdout.split()) if r.returncode == 0 else set()


def all_statuses(progress=None) -> list[dict]:
    """Per-backbone image status for the GUI image manager (one dict each):
    {key,label,tag,present,pullable,docker}. Runs in a FuncWorker thread (one
    `docker images` call), so a slow/down daemon never freezes the window. When
    docker is absent every row is reported missing+un-pullable. `progress` is the
    FuncWorker hook (unused here)."""
    from .backbones import BACKBONES
    if not have_docker():
        return [{"key": k, "label": b.label, "tag": image_for(b),
                 "present": False, "pullable": False, "docker": False}
                for k, b in BACKBONES.items()]
    here = present_images()

    def _present(b):   # M3: a locally-built bare image counts as present too
        return any(t in here or f"{t}:latest" in here
                   for t in (image_for(b), f"trainer-local-{b.key}"))
    return [{"key": k, "label": b.label, "tag": image_for(b),
             "present": _present(b),
             "pullable": is_pullable(b), "docker": True}
            for k, b in BACKBONES.items()]


def preview(program: str, args: list[str]) -> str:
    """One-line shell preview for the log (the exact command JobRunner will run)."""
    return program + " " + " ".join(args)
