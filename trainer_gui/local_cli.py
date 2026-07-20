"""Builders for the LOCAL (pixi) execution path — the mirror of modal_cli.

Each backbone's local_train_*.py runs DIRECTLY on the host inside its pixi
environment (envs/pixi.toml — one env per backbone, env name = backbone key
with '_' -> '-'; heavy deps + the trainer-src-<key> model-source package live
there). No container and no bind mounts: the trainers read real host paths
from the TT_* env vars (the train_common path contract):

    TT_DATASETS_ROOT  <- staging root (canonical datasets + _infer/<job>)
    TT_OUTPUTS_ROOT   <- the chosen output folder (runs/<id> lands here)
    TT_DATASET_DIR / TT_INFER_DIR / TT_PRED_DIR <- out-of-workspace overrides

run_script returns (program, args, env) for JobRunner (args like modal_cli,
env passed as extra_env) so the GUI dispatch stays backend-agnostic. The
modal_train_*.py scripts still subprocess these same local_train_*.py inside
a Modal container, so local and cloud run one codebase.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from . import appstate

# local_cli lives in trainer_gui/ inside the repo; the envs workspace sits
# beside it. Editable installs run from the checkout, so this always resolves.
_REPO = Path(__file__).resolve().parents[1]


def pixi_exe() -> str:
    return shutil.which("pixi") or "pixi"


def have_pixi() -> bool:
    return shutil.which("pixi") is not None


def runnable() -> bool:
    """True when this host can actually execute a local run: pixi on PATH and
    linux (the training envs are linux-64/CUDA-only). Elsewhere the GUI prints
    the exact command instead of running it — the dev-box dry-run flow."""
    return have_pixi() and sys.platform.startswith("linux")


def env_name(backbone) -> str:
    """pixi environment name for a backbone: the key, '_' -> '-' (pixi forbids
    underscores in environment names; e.g. kpconvx_cold -> kpconvx-cold)."""
    return backbone.key.replace("_", "-")


def manifest_path(repo_root: str = "") -> str:
    return str(Path(repo_root or _REPO) / "envs" / "pixi.toml")


def env_dir(backbone, repo_root: str = "") -> Path:
    return Path(repo_root or _REPO) / "envs" / ".pixi" / "envs" / env_name(backbone)


def installed(backbone, repo_root: str = "") -> bool:
    """Cheap install check: a solved+installed pixi env has conda-meta/ on
    disk. Pure directory scan — no subprocess, can't hang."""
    return (env_dir(backbone, repo_root) / "conda-meta").is_dir()


def env_preflight(backbone, repo_root: str = "") -> tuple[bool, str]:
    """(proceed, message) for the backbone's pixi env. Blocks when the env
    isn't installed yet — a first install downloads multi-GB CUDA wheels, which
    shouldn't happen as a surprise side effect of pressing Launch."""
    if installed(backbone, repo_root):
        return True, ""
    return False, (f"[local] pixi env '{env_name(backbone)}' isn't installed on this "
                   f"host. Install it from Configure model… (or run `pixi install "
                   f"--manifest-path envs/pixi.toml -e {env_name(backbone)}`), then "
                   f"launch again.")


def gpu_preflight() -> tuple[bool, str]:
    """(proceed, message) for GPU availability. The trainers are CUDA-only
    (every script calls .cuda()), so block when GPUs are disabled in
    local_config, and warn when no NVIDIA driver is detectable. There is no
    CPU inference path."""
    cfg = appstate.local_config()
    if not cfg.get("gpus"):
        return False, ("[local] GPUs are disabled (local_config['gpus']=''), but these "
                       "models require CUDA. Set gpus='all' (or a device id) and run "
                       "again.")
    if shutil.which("nvidia-smi") is None:
        return True, ("[local] ⚠ no 'nvidia-smi' on PATH - if this host lacks an NVIDIA "
                      "GPU + driver the run will fail. These models are CUDA-only "
                      "(no CPU fallback).")
    return True, ""


def run_env(*, outputs_root: str = "", dataset_dir: str = "", infer_dir: str = "",
            pred_dir: str = "", gpu: str = "", env: "dict | None" = None) -> dict:
    """The extra_env dict for JobRunner: the TT_* path contract + CUDA device
    selection (replaces docker's -v mounts and --gpus)."""
    cfg = appstate.local_config()
    out = {"TT_DATASETS_ROOT": str(cfg["datasets_root"]),
           "TT_OUTPUTS_ROOT": str(outputs_root or cfg["outputs_root"])}
    if dataset_dir:
        out["TT_DATASET_DIR"] = str(dataset_dir)
    if infer_dir:
        out["TT_INFER_DIR"] = str(infer_dir)
    if pred_dir:
        out["TT_PRED_DIR"] = str(pred_dir)
    gpus = str(cfg.get("gpus") or "")
    if gpus and gpus != "all":   # 'all' = don't restrict; '' is blocked upstream
        out["CUDA_VISIBLE_DEVICES"] = gpus.removeprefix("device=")
    if gpu:
        out["TT_GPU"] = gpu                  # cosmetic locally; keeps log parity
    out.update(env or {})                    # e.g. DG_* density-generalization flags
    return out


def run_script(script: str, flags: dict, backbone, *, repo_root: str = "",
               gpu: str = "", outputs_root: str = "", dataset_dir: str = "",
               infer_dir: str = "", pred_dir: str = "",
               env: "dict | None" = None) -> tuple[str, list[str], dict]:
    """`pixi run -e <env> python scripts/local/local_train_<key>.py --flags`.

    `script` (the modal_train_*.py name) is accepted for call-site parity with
    modal_cli but unused: locally we run the decoupled local_train_<key>.py
    directly. Returns (program, args, env) — pass env to JobRunner extra_env.
    `--locked` = the committed pixi.lock is the contract; a manifest edit
    without re-locking fails loudly instead of silently re-solving."""
    args = ["run", "--manifest-path", manifest_path(repo_root), "--locked",
            "-e", env_name(backbone),
            "python", f"scripts/local/local_train_{backbone.key}.py"]
    for key, val in flags.items():
        if val is None:
            continue
        args += [f"--{key}", str(val)]
    return pixi_exe(), args, run_env(outputs_root=outputs_root,
                                     dataset_dir=dataset_dir, infer_dir=infer_dir,
                                     pred_dir=pred_dir, gpu=gpu, env=env)


def install(backbone, repo_root: str = "") -> tuple[str, list[str]]:
    """`pixi install -e <env>` for a backbone's environment — the 'build/pull'
    action of the env manager UI (streamed by the same runner that ran
    docker pull)."""
    return pixi_exe(), ["install", "--manifest-path", manifest_path(repo_root),
                        "--locked", "-e", env_name(backbone)]


def installed_weights(backbone, repo_root: str = "") -> list[tuple[str, str]]:
    """(name, final_model.pth path) for every trainer-weights-* conda package
    installed in the backbone's env ($PREFIX/share/trainer-weights/<name>/).
    The Infer page lists these beside filesystem .pth picks."""
    root = env_dir(backbone, repo_root) / "share" / "trainer-weights"
    return sorted((d.name, str(d / "final_model.pth"))
                  for d in root.iterdir() if (d / "final_model.pth").is_file()) \
        if root.is_dir() else []


def all_statuses(progress=None) -> list[dict]:
    """Per-backbone env status for the GUI env manager (one dict each):
    {key,label,env,installed,pixi}. A pure directory scan — instant, so a
    FuncWorker thread stays trivial. `progress` is the FuncWorker hook
    (unused here)."""
    from .backbones import BACKBONES
    pixi = have_pixi()
    return [{"key": k, "label": b.label, "env": env_name(b),
             "installed": installed(b), "pixi": pixi}
            for k, b in BACKBONES.items()]


def preview(program: str, args: list[str], env: "dict | None" = None) -> str:
    """One-line shell preview for the log (the exact command JobRunner will
    run, with the TT_* env prefix that docker -v mounts used to express)."""
    pre = " ".join(f"{k}={v}" for k, v in (env or {}).items())
    return (pre + " " if pre else "") + program + " " + " ".join(args)
