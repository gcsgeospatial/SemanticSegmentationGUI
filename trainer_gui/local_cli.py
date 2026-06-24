"""Builders for the LOCAL (Docker) execution path — the mirror of modal_cli.

Each modal_train_*.py runs unmodified inside its backbone's Docker image (built
by tools/gen_dockerfiles.py) via local_run.py, with host dirs bind-mounted to
the paths the scripts already hardcode:

    /workspace  <- repo root (the scripts + local_run.py + _modal_shim.py)
    /datasets   <- staging root (canonical datasets + _infer/<job> live here)
    /outputs    <- local runs dir (training writes runs/<id>/..., weights read here)
    /data       <- raw IEEE data (only the built-in no-`--dataset` path needs it)

No upload/download: the data is already on the host, and predictions/checkpoints
land straight back in the bind-mounted host dirs. Returns (program, args) for
JobRunner — exactly like modal_cli — so the GUI dispatch is backend-agnostic.
"""

from __future__ import annotations

import shutil
import subprocess

from . import appstate


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


def image_available(backbone) -> bool:
    """True if the backbone's Docker image is present locally (built or pulled)."""
    try:
        r = subprocess.run([docker_exe(), "image", "inspect", image_for(backbone)],
                           capture_output=True, timeout=15)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def is_pullable(backbone) -> bool:
    """True if image_for() is a registry tag `docker run` can auto-pull (a registry
    prefix is configured, or the override names a registry host)."""
    if appstate.local_config().get("registry"):
        return True
    tag = image_for(backbone)
    head = tag.split("/", 1)[0]
    return "/" in tag and ("." in head or ":" in head)   # host has a dot or port


def image_preflight(backbone) -> tuple[bool, str]:
    """(proceed, message) for the image. proceed=False blocks the launch; a
    non-empty message is shown either way — an FYI when docker will auto-pull, an
    error+how-to-fix when there's nothing to build or pull. Distinguishes a
    not-built local image from a registry tag, so the user gets the right hint
    instead of docker's cryptic 'pull access denied'."""
    if image_available(backbone):
        return True, ""
    tag = image_for(backbone)
    if is_pullable(backbone):
        return True, (f"[local] '{tag}' not present locally — docker will pull it from the "
                      f"registry now (run `docker login` first if it's private).")
    return False, (f"[local] image '{tag}' isn't built on this host. Build it "
                   f"(bash docker/build_all.sh) — or set a registry (TT_REGISTRY or "
                   f"local_config['registry']) and `docker pull` it — then run again.")


def _mounts(cfg: dict, repo_root: str, extra_mounts) -> list[str]:
    m = ["-v", f"{repo_root}:/workspace",
         "-v", f"{cfg['datasets_root']}:/datasets",
         "-v", f"{cfg['outputs_root']}:/outputs"]
    if cfg.get("data_root"):
        m += ["-v", f"{cfg['data_root']}:/data"]
    for host, container in (extra_mounts or []):
        m += ["-v", f"{host}:{container}"]
    return m


def run_script(script: str, flags: dict, backbone, *, repo_root: str,
               gpu: str = "", extra_mounts=None) -> tuple[str, list[str]]:
    """`docker run --rm --gpus all -v ... <image> python local_run.py script --flags`.

    `flags` are the same kebab-case flags modal_cli.run_script takes; `extra_mounts`
    is a list of (host, container) bind pairs (e.g. a local .pth into /outputs).
    """
    cfg = appstate.local_config()
    args = ["run", "--rm", "-w", "/workspace"]
    if cfg.get("gpus"):
        args += ["--gpus", str(cfg["gpus"])]
    args += _mounts(cfg, repo_root, extra_mounts)
    if gpu:
        args += ["-e", f"TT_GPU={gpu}"]          # cosmetic locally; keeps log parity
    args += list(cfg.get("extra_args", []))
    args += [image_for(backbone), "python", "local_run.py", script]
    for key, val in flags.items():
        if val is None:
            continue
        args += [f"--{key}", str(val)]
    return docker_exe(), args


def preview(program: str, args: list[str]) -> str:
    """One-line shell preview for the log (the exact command JobRunner will run)."""
    return program + " " + " ".join(args)
