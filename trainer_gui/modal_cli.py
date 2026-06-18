"""Builders for the Modal CLI commands the GUI launches (always via QProcess).

Every function returns (program, args) so callers can hand it straight to
JobRunner. The `modal` executable is resolved from PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess

DATASETS_VOLUME = "terminal-datasets"


def modal_exe() -> str:
    return shutil.which("modal") or "modal"


def volume_put(volume: str, local_path: str, remote_path: str) -> tuple[str, list[str]]:
    # -f: overwrite existing remote files (re-uploads after edits)
    return modal_exe(), ["volume", "put", "-f", volume, local_path, remote_path]


def volume_get(volume: str, remote_path: str, local_path: str) -> tuple[str, list[str]]:
    return modal_exe(), ["volume", "get", "-f", volume, remote_path, local_path]


def volume_ls(volume: str, remote_path: str = "/") -> tuple[str, list[str]]:
    return modal_exe(), ["volume", "ls", "--json", volume, remote_path]


def run_script(script: str, flags: dict, detach: bool = False) -> tuple[str, list[str]]:
    """`modal run [--detach] script.py --flag value ...` — flags use kebab-case keys."""
    args = ["run"]
    if detach:
        args.append("--detach")
    args.append(script)
    for key, val in flags.items():
        if val is None:
            continue
        args += [f"--{key}", str(val)]
    return modal_exe(), args


def app_logs(app_name: str) -> tuple[str, list[str]]:
    return modal_exe(), ["app", "logs", app_name]


# ---- thin synchronous helpers (background threads only — they block) ----

def list_volume_entries(volume: str, remote_path: str = "/", timeout: int = 60) -> list[dict]:
    """Blocking `modal volume ls --json`; returns [] if the path doesn't exist."""
    prog, args = volume_ls(volume, remote_path)
    try:
        out = subprocess.run([prog] + args, capture_output=True, text=True,
                             timeout=timeout, encoding="utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []
