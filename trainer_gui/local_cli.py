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

from . import appstate


def docker_exe() -> str:
    return shutil.which("docker") or "docker"


def have_docker() -> bool:
    return shutil.which("docker") is not None


def image_for(backbone) -> str:
    return appstate.local_config()["images"].get(backbone.key) or f"trainer-local-{backbone.key}"


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
