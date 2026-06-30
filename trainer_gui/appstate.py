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


def default_download_dir() -> Path:
    """A *findable* default for downloaded artifacts — the user's Downloads folder
    (or home if that's missing). Deliberately NOT a hidden app dir: settings live
    under %APPDATA%/.config, but downloads the user has to open should land where
    they'll actually look. It's only the default — every download path is an
    editable, user-pickable field."""
    dl = Path.home() / "Downloads"
    return dl if dl.exists() else Path.home()


# ---- execution mode: "modal" (cloud) | "local" (Docker on a GPU host) --------

def get_exec_mode() -> str:
    return "local" if get("exec_mode") == "local" else "modal"


def set_exec_mode(mode: str) -> None:
    put("exec_mode", "local" if mode == "local" else "modal")


# The org we publish the trainer-local-* images under, on GitHub Container
# Registry. Used as the registry when the user hasn't set one — so the GUI pulls
# from it out of the box. Override via local_config['registry'] or TT_REGISTRY;
# clear it to "" explicitly for local-build-only (no pulling).
DEFAULT_REGISTRY = "ghcr.io/gcsgeospatial"

# Defaults for the local backend. Roots default to the dirs the GUI already
# uses (so a converted dataset / inference job is immediately reachable); every
# value is overridable from state.json["local_config"] for I/O modularity.
_DEFAULT_LOCAL_CONFIG = {
    "images": {},          # backbone.key -> docker image tag (default trainer-local-<key>)
    "registry": "",        # registry prefix, e.g. "ghcr.io/you" -> pull instead of build
    "datasets_root": "",   # host -> /datasets (default: staging_dir())
    "outputs_root": "",    # host -> /outputs  (default: local_runs_dir())
    "gpus": "all",         # docker --gpus value ("all" | "0" | "" to disable)
    "extra_args": [],      # extra `docker run` args
}


def local_config() -> dict:
    saved = get("local_config", {})
    cfg = {**_DEFAULT_LOCAL_CONFIG, **saved}
    cfg["datasets_root"] = cfg["datasets_root"] or str(staging_dir())
    cfg["outputs_root"] = cfg["outputs_root"] or str(local_runs_dir())
    # Registry precedence: a saved value (even "") wins; else TT_REGISTRY (set it
    # once in the env, no JSON edit); else our DEFAULT_REGISTRY so pulling works
    # out of the box. "registry" absent from saved = never set -> use the default;
    # present-and-empty = the user opted out (local builds only), so leave it.
    if "registry" not in saved:
        cfg["registry"] = os.environ.get("TT_REGISTRY", "") or DEFAULT_REGISTRY
    else:
        cfg["registry"] = cfg["registry"] or os.environ.get("TT_REGISTRY", "")
    return cfg


def set_local_config(cfg: dict) -> None:
    put("local_config", cfg)


# ---- which backbones to show in local mode (hide images your driver can't run) --

def enabled_backbones():
    """Backbone keys enabled for local mode, or None = all. Lets you hide a
    backbone whose Docker image you can't run (e.g. a cu124 image on an older
    driver) or simply don't use. Stored explicitly once the user picks."""
    val = get("local_config", {}).get("enabled_backbones")
    return None if val is None else set(val)


def set_enabled_backbones(keys) -> None:
    cfg = {**get("local_config", {}), "enabled_backbones": list(keys)}
    put("local_config", cfg)


def backbone_enabled(key: str) -> bool:
    """True if this backbone should appear. Only filters in local mode; an unset
    selection means all are enabled."""
    if get_exec_mode() != "local":
        return True
    en = enabled_backbones()
    return en is None or key in en


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

def known_datasets() -> dict:
    return get("datasets", {})


def selectable_datasets() -> dict:
    """Datasets offered for a job — the saved registry (every dataset is converted
    on the Datasets page, so all of them are selectable)."""
    return known_datasets()


def remember_dataset(name: str, info: dict) -> None:
    ds = known_datasets()
    ds[name] = info
    put("datasets", ds)


def forget_dataset(name: str) -> None:
    ds = known_datasets()
    ds.pop(name, None)
    put("datasets", ds)


def delete_dataset(name: str) -> None:
    """Forget a saved dataset AND delete its staged copy on disk, plus any per-
    dataset overrides keyed by name. Best-effort: a missing/empty staged_dir or a
    failed rmtree is ignored so the registry entry still goes away. Never touches a
    builtin (none remain, but guard anyway)."""
    info = known_datasets().get(name, {})
    if info.get("builtin"):
        return
    staged = info.get("staged_dir", "")
    if staged and os.path.isdir(staged):
        import shutil
        shutil.rmtree(staged, ignore_errors=True)
    forget_dataset(name)
    for key in ("dg_config", "palette_overrides", "palette_name_overrides"):
        allc = get(key, {})
        if name in allc:
            allc.pop(name, None)
            put(key, allc)
