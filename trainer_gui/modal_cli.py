"""Modal CLI command builders; every function returns (program, args) for JobRunner."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

DATASETS_VOLUME = "terminal-datasets"


def modal_exe() -> str:
    return shutil.which("modal") or "modal"


def volume_create(volume: str) -> tuple[str, list[str]]:
    # errors if the volume exists — run as JobRunner's `pre` step, exit ignored
    return modal_exe(), ["volume", "create", volume]


def volume_put(volume: str, local_path: str, remote_path: str) -> tuple[str, list[str]]:
    # -f: overwrite existing remote files (re-uploads after edits)
    return modal_exe(), ["volume", "put", "-f", volume, local_path, remote_path]


def volume_get(volume: str, remote_path: str, local_path: str) -> tuple[str, list[str]]:
    # --force spelled out: newer Modal CLIs dropped -f on `get`
    return modal_exe(), ["volume", "get", "--force", volume, remote_path, local_path]


def volume_ls(volume: str, remote_path: str = "/") -> tuple[str, list[str]]:
    return modal_exe(), ["volume", "ls", "--json", volume, remote_path]


def run_script(script: str, flags: dict, detach: bool = False,
               env: dict | None = None) -> tuple[str, list[str]]:
    """`modal run [--detach] script.py --flag value ...` (kebab-case keys); env
    rides as one --env-json flag, mirroring local_cli's extra_env."""
    args = ["run"]
    if detach:
        args.append("--detach")
    args.append(script)
    for key, val in flags.items():
        if val is None:
            continue
        args += [f"--{key}", str(val)]
    if env:
        args += ["--env-json", json.dumps({k: str(v) for k, v in env.items()})]
    return modal_exe(), args


# ---- thin synchronous helpers (background threads only — they block) ----

def fetch_run_manifest(volume: str, run_id: str, timeout: int = 60) -> dict | None:
    """Blocking: read runs/<run_id>/run.json (legacy run_config.json) off an
    outputs volume; None if absent/unreadable."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    for fn in ("run.json", "run_config.json"):
        with tempfile.TemporaryDirectory() as td:
            prog, args = volume_get(volume, f"runs/{run_id}/{fn}", td)
            try:
                out = subprocess.run([prog] + args, capture_output=True, text=True,
                                     timeout=timeout, encoding="utf-8",
                                     errors="replace", env=env)
            except (OSError, subprocess.TimeoutExpired):
                return None
            dest = os.path.join(td, fn)
            if out.returncode == 0 and os.path.isfile(dest):
                try:
                    with open(dest, encoding="utf-8") as f:
                        return json.load(f)
                except (OSError, json.JSONDecodeError):
                    return None
    return None


def list_volume_entries(volume: str, remote_path: str = "/", timeout: int = 60) -> list[dict]:
    """Blocking `modal volume ls --json`; returns [] if the path doesn't exist."""
    prog, args = volume_ls(volume, remote_path)
    # force UTF-8: modal's ✓/box chars crash under Windows cp1252
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        out = subprocess.run([prog] + args, capture_output=True, text=True,
                             timeout=timeout, encoding="utf-8", errors="replace", env=env)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []
