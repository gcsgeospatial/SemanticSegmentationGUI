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


def workspace_dir() -> Path:
    """The single root that owns every dataset and, nested inside each, its runs/
    and infer/ output. Set once via the first-launch prompt (set_workspace);
    until then it falls back to staging_dir() so nothing breaks. This is the host
    dir bound to /datasets, so /datasets/<name> resolves for every dataset."""
    w = get("workspace")
    d = Path(w) if w else staging_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_workspace(path: str) -> None:
    put("workspace", str(path))


def dataset_root(name: str) -> Path:
    """A dataset's on-disk root = its registered staged_dir (may live outside the
    workspace for pre-existing datasets — zero-move keeps them where they are)."""
    return Path(known_datasets()[name]["staged_dir"])


def dataset_run_roots() -> list[Path]:
    """`<dataset>/runs` for every registered dataset still on disk — the roots the
    Plotting page scans for local training runs."""
    roots = []
    for name, info in known_datasets().items():
        staged = info.get("staged_dir", "")
        if staged and os.path.isdir(staged):
            roots.append(Path(staged) / "runs")
    return roots


def scratch_infer_dir() -> Path:
    """Where inference from a loose .pth (no linked dataset) lands — a findable
    spot in the workspace instead of forcing a dataset pick."""
    return workspace_dir() / "_scratch" / "infer"


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
    "datasets_root": "",   # host -> /datasets (default: workspace_dir())
    "outputs_root": "",    # host -> /outputs  (default: local_runs_dir())
    "gpus": "all",         # docker --gpus value ("all" | "0" | "" to disable)
    "extra_args": [],      # extra `docker run` args
}


def local_config() -> dict:
    saved = get("local_config", {})
    cfg = {**_DEFAULT_LOCAL_CONFIG, **saved}
    cfg["datasets_root"] = cfg["datasets_root"] or str(workspace_dir())
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


# A dataset folder's record-keeping subdirs: kept when the dataset is deleted so
# past training runs + inference outputs survive. Everything else is "data".
_KEEP_ON_DELETE = ("runs", "infer")


def _rmtree_force(path: Path) -> str:
    """Remove a file or a whole dir tree, best-effort. On Windows retry once after
    clearing the read-only bit that blocks rmtree. Returns "" on success, else the
    error string — NEVER raises (a throw would skip the caller's list refresh)."""
    import shutil
    import stat

    def _rm():
        shutil.rmtree(path) if path.is_dir() else path.unlink()

    try:
        _rm()
    except OSError:
        try:
            for p in ([path, *path.rglob("*")] if path.is_dir() else [path]):
                try:
                    os.chmod(p, stat.S_IWRITE)
                except OSError:
                    pass
            _rm()
        except OSError as e:
            return str(e)
    return ""


def delete_dataset(name: str) -> tuple[str, str]:
    """Forget a saved dataset AND delete its DATA on disk (train/val/test, meta,
    prep cache), plus any per-dataset overrides keyed by name. Its runs/ and infer/
    subdirs are KEPT for record keeping (and re-registered on the Plotting page).
    Returns (staged_dir, error): error is "" when the data is gone (or there was
    none), else the reason it couldn't be removed (locked/in-use file — common on
    Windows). The registry entry + overrides are dropped regardless, so the list
    stays consistent even if files linger. Every dataset is deletable."""
    info = known_datasets().get(name, {})
    staged = info.get("staged_dir", "")

    # Forget the entry + overrides FIRST so the list always reflects the delete,
    # even if the on-disk cleanup below fails (locked files are common on Windows).
    forget_dataset(name)
    for key in ("dg_config", "palette_overrides", "palette_name_overrides"):
        allc = get(key, {})
        if name in allc:
            allc.pop(name, None)
            put(key, allc)

    # Best-effort disk removal — must NEVER raise. Delete only the data; if runs/
    # or infer/ are present, keep them (and the folder) and remove data per-child;
    # otherwise there's nothing to preserve, so drop the whole folder.
    err = ""
    if staged and os.path.isdir(staged):
        root = Path(staged)
        if any(c.name in _KEEP_ON_DELETE for c in root.iterdir()):
            for child in root.iterdir():
                if child.name not in _KEEP_ON_DELETE:
                    err = _rmtree_force(child) or err
            _register_kept_runs(root)
        else:
            err = _rmtree_force(root)
    return (staged, err)


def _register_kept_runs(root: Path) -> None:
    """A deleted dataset's registry entry is gone, so dataset_run_roots() no longer
    covers its kept runs/. Add it to the Plotting page's extra roots so those runs
    stay visible for record keeping."""
    runs = root / "runs"
    try:
        has_runs = runs.is_dir() and any(runs.iterdir())
    except OSError:
        return
    if has_runs:
        extra = get("plot_extra_roots", [])
        if str(runs) not in extra:
            extra.append(str(runs))
            put("plot_extra_roots", extra)
