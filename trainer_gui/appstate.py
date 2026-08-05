"""Persisted app state, stored as JSON in the per-OS app dir; staging and
downloaded run artifacts live there too."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _app_base(platform: str, environ) -> Path:
    """Per-OS app-data base."""
    home = Path.home()
    if platform == "win32":
        return Path(environ.get("APPDATA") or environ.get("LOCALAPPDATA") or home)
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
    """The root owning every dataset (host dir bound to /datasets); falls back to
    staging_dir() until the first-launch prompt sets it."""
    w = get("workspace")
    d = Path(w) if w else staging_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_workspace(path: str) -> None:
    put("workspace", str(path))


def dataset_run_roots() -> list[Path]:
    """<dataset>/runs for every registered dataset still on disk."""
    roots = []
    for name, info in known_datasets().items():
        staged = info.get("staged_dir", "")
        if staged and os.path.isdir(staged):
            roots.append(Path(staged) / "runs")
    return roots


def run_roots(repo_root: str | None = None) -> list[Path]:
    """Every place local runs live - the ONE discovery source for the
    Inference run picker (each root walked one level deep)."""
    roots = [*dataset_run_roots(), workspace_dir() / "inference", runs_dir()]
    if repo_root:
        roots.append(Path(repo_root) / "runs")
    return roots


def runs_dir() -> Path:
    d = app_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def local_runs_dir() -> Path:
    """Where local training writes runs/<id>/... - the TT_OUTPUTS_ROOT default."""
    d = app_dir() / "local_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- execution mode: "modal" (cloud) | "local" (pixi env on a GPU host) ------

def get_exec_mode() -> str:
    return "local" if get("exec_mode") == "local" else "modal"


def set_exec_mode(mode: str) -> None:
    put("exec_mode", "local" if mode == "local" else "modal")


def modal_datasets_volume() -> str:
    """Modal volume datasets upload to; user-set on the Datasets page."""
    return (get("modal_datasets_volume") or os.environ.get("TT_DATASET_VOLUME")
            or "terminal-datasets")


def set_modal_datasets_volume(name: str) -> None:
    put("modal_datasets_volume", name.strip())


def modal_outputs_volume() -> str:
    """Custom Modal outputs volume; '' = the per-model '<app>-outputs' default."""
    return (get("modal_outputs_volume") or os.environ.get("TT_OUTPUTS_VOLUME")
            or "").strip()


def set_modal_outputs_volume(name: str) -> None:
    put("modal_outputs_volume", name.strip())


def read_json(path) -> dict | None:
    """Parse a JSON file; None ONLY when it isn't there. Unreadable or malformed
    raises (OSError / json.JSONDecodeError) - a caller that wants to treat those
    as absent must catch and say so itself."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _state_path() -> Path:
    return app_dir() / "state.json"


_quarantined = ""


def state_quarantine() -> str:
    """Path a corrupt state.json was renamed to this session ("" = none); the
    GUI reports it at startup, since the app came up with empty settings."""
    return _quarantined


def load_state() -> dict:
    global _quarantined
    path = _state_path()
    try:
        return read_json(path) or {}
    except (OSError, ValueError) as bad_state:
        # never return {} in place: put() would write it back over the registry
        n = 1
        while (path.parent / f"{path.name}.bad-{n}").exists():
            n += 1
        dest = path.parent / f"{path.name}.bad-{n}"
        try:
            os.replace(path, dest)
        except OSError as e:
            raise RuntimeError(
                f"{path} is unreadable ({bad_state}) and cannot be renamed out of "
                f"the way ({e}). Close whatever is holding it, or delete it by "
                f"hand, then relaunch.") from e
        _quarantined = str(dest)
        return {}


def save_state(state: dict) -> None:
    # atomic: a crash mid-write must never take the whole app state with it
    path = _state_path()
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def get(key: str, default: Any = None) -> Any:
    return load_state().get(key, default)


def put(key: str, value: Any) -> None:
    state = load_state()
    state[key] = value
    save_state(state)


# ---- datasets registry: name -> {meta_path, staged_dir, uploaded: bool} ----

def known_datasets() -> dict:
    return get("datasets", {})


def remember_dataset(name: str, info: dict) -> None:
    ds = known_datasets()
    ds[name] = info
    put("datasets", ds)


def forget_dataset(name: str) -> None:
    ds = known_datasets()
    ds.pop(name, None)
    put("datasets", ds)


_KEEP_ON_DELETE = ("runs", "infer")


def _rmtree_force(path: Path) -> str:
    """Best-effort remove (Windows read-only retry). Returns "" or the error -
    never raises (a throw would skip the caller's list refresh)."""
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
    """Forget a dataset and delete its data (runs/ and infer/ are kept). Returns
    (staged_dir, error); the registry entry is dropped even if files linger."""
    info = known_datasets().get(name, {})
    staged = info.get("staged_dir", "")

    # forget first so the list reflects the delete even if cleanup fails
    forget_dataset(name)

    err = ""
    if staged and os.path.isdir(staged):
        root = Path(staged)
        if any(c.name in _KEEP_ON_DELETE for c in root.iterdir()):
            for child in root.iterdir():
                if child.name not in _KEEP_ON_DELETE:
                    err = _rmtree_force(child) or err
        else:
            err = _rmtree_force(root)
    return (staged, err)


def is_run_dir(d: Path) -> bool:
    d = Path(d)
    return d.is_dir() and any((d / fn).exists()
                              for fn in ("run.json", "metrics.csv",
                                         "val_metrics.csv"))


def discover_runs(root: Path) -> list[Path]:
    """Run dirs at `root` or one level below it (the layout downloads use:
    <runs>/<backbone>/<run_id>/). Sorted newest-name-first, de-duplicated."""
    root = Path(root)
    if not root.exists():
        return []
    found: set[Path] = set()
    if is_run_dir(root):
        found.add(root)
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if is_run_dir(child):
            found.add(child)
        else:
            for grand in child.iterdir():
                if is_run_dir(grand):
                    found.add(grand)
    return sorted(found, key=lambda p: p.name, reverse=True)
