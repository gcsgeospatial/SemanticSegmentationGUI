"""Persisted app state (known datasets, last-used params, run history).

Stored as JSON in the per-OS app dir (%APPDATA% on Windows, $XDG_CONFIG_HOME or
~/.config on Linux, ~/Library/Application Support on macOS). Staging and
downloaded run artifacts also live there so the repo stays clean.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _app_base(platform: str, environ) -> Path:
    """Native per-OS base dir for app data. APPDATA is honored on EVERY platform
    so it stays a single override knob (tests set it); otherwise pick the native
    location for the OS."""
    if environ.get("APPDATA"):
        return Path(environ["APPDATA"])
    home = Path.home()
    if platform == "win32":
        return Path(environ.get("LOCALAPPDATA") or home)
    if platform == "darwin":
        return home / "Library" / "Application Support"
    return Path(environ.get("XDG_CONFIG_HOME") or (home / ".config"))


def app_dir() -> Path:
    d = _app_base(sys.platform, os.environ) / "trainer_gui"
    d.mkdir(parents=True, exist_ok=True)
    return d


def staging_dir() -> Path:
    d = app_dir() / "staging"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir() -> Path:
    d = app_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def local_runs_dir() -> Path:
    """Where local (Docker) training writes runs/<id>/... — bind-mounted /outputs."""
    d = app_dir() / "local_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- execution mode: "modal" (cloud) | "local" (Docker on a GPU host) --------

def get_exec_mode() -> str:
    return "local" if get("exec_mode") == "local" else "modal"


def set_exec_mode(mode: str) -> None:
    put("exec_mode", "local" if mode == "local" else "modal")


# Defaults for the local backend. Roots default to the dirs the GUI already
# uses (so a converted dataset / inference job is immediately reachable); every
# value is overridable from state.json["local_config"] for I/O modularity.
_DEFAULT_LOCAL_CONFIG = {
    "images": {},          # backbone.key -> docker image tag (default trainer-local-<key>)
    "datasets_root": "",   # host -> /datasets (default: staging_dir())
    "outputs_root": "",    # host -> /outputs  (default: local_runs_dir())
    "data_root": "",       # host -> /data     (built-in IEEE raw data; optional)
    "gpus": "all",         # docker --gpus value ("all" | "0" | "" to disable)
    "extra_args": [],      # extra `docker run` args
}


def local_config() -> dict:
    cfg = {**_DEFAULT_LOCAL_CONFIG, **get("local_config", {})}
    cfg["datasets_root"] = cfg["datasets_root"] or str(staging_dir())
    cfg["outputs_root"] = cfg["outputs_root"] or str(local_runs_dir())
    return cfg


def set_local_config(cfg: dict) -> None:
    put("local_config", cfg)


_STATE_PATH = None  # resolved lazily so tests can monkeypatch APPDATA


def _state_path() -> Path:
    return app_dir() / "state.json"


def load_state() -> dict:
    try:
        with open(_state_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    with open(_state_path(), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def get(key: str, default: Any = None) -> Any:
    return load_state().get(key, default)


def put(key: str, value: Any) -> None:
    state = load_state()
    state[key] = value
    save_state(state)


# ---- datasets registry: name -> {meta_path, staged_dir, uploaded: bool} ----

# Built-in datasets that already live on the ieee-data Modal volume. The IEEE
# training scripts read them via their no-`--dataset` default (real data, real
# per-point HAG for the HAG variants). These are virtual registry entries — never
# written to state.json, so they always appear and can't be forgotten. `builtin`
# makes the Train page skip `--dataset` (run that default); `backbones` restricts
# the model list to the scripts whose default path actually targets this data.
BUILTIN_DATASETS = {
    "IEEE": {
        "builtin": True, "uploaded": True, "meta_path": "",
        "backbones": ["ptv3", "randlanet", "kpconvx_cold"],
        "note": "Raw IEEE GRSS 2019 Track 4 (ieee-data volume) — the scripts' "
                "default. 5 classes: Ground/Trees/Building/Water/Bridge.",
    },
    "IEEE HAG": {
        "builtin": True, "uploaded": True, "meta_path": "",
        "backbones": ["ptv3_hag", "randlanet_hag", "kpconvx_cold_hag"],
        "note": "Raw IEEE Track 4 + real per-point HeightAboveGround "
                "(ieee-data:/IEEE/HAG) — trains the HAG model variants.",
    },
}


def known_datasets() -> dict:
    # Builtins last so the reserved IEEE names always resolve to the builtin entry.
    return {**get("datasets", {}), **BUILTIN_DATASETS}


def remember_dataset(name: str, info: dict) -> None:
    ds = known_datasets()
    ds[name] = info
    put("datasets", ds)


def forget_dataset(name: str) -> None:
    ds = known_datasets()
    ds.pop(name, None)
    put("datasets", ds)
