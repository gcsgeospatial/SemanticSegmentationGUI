"""In-container launcher: run a modal_train_*.py script locally via the shim.

Used inside each backbone's Docker image (with /data /outputs /datasets bind-
mounted to host dirs):

    python local_run.py modal_train_ptv3.py --dataset X --grid 0.05 --epochs 250

It shims `modal`, imports the script, finds its @app.local_entrypoint `main`,
maps the kebab `--flags` to main()'s kwargs (typed from its annotations), and
calls it. `main` then runs `train_X.remote(...)`, which the shim executes in
THIS process — the body's hardcoded /data /outputs /datasets are the mounts.

Standalone & generic: no per-backbone code, reuses each `main`'s own signature.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import typing

import _modal_shim


def _parse_flags(argv: list[str]) -> dict:
    """`--flag value` / bare `--flag` -> {flag: value | True}. Kebab keys kept."""
    out: dict = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            key = tok[2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[key] = argv[i + 1]
                i += 2
            else:
                out[key] = True
                i += 1
        else:
            i += 1
    return out


def _coerce(val, annot):
    if val is True or annot is inspect.Parameter.empty or annot is None:
        return val
    base = annot
    args = typing.get_args(annot)
    if args:   # Optional[X] / Union[X, None] -> X
        base = next((a for a in args if a is not type(None)), annot)
    if base is bool:
        return str(val).strip().lower() in ("1", "true", "yes", "on")
    if base is int:
        return int(float(val))     # tolerate "50.0"
    if base is float:
        return float(val)
    return str(val)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python local_run.py <script.py> [--flag value ...]", file=sys.stderr)
        return 2
    script = argv[0]
    flags = _parse_flags(argv[1:])

    _modal_shim.install()
    here = os.path.dirname(os.path.abspath(script)) or "."
    if here not in sys.path:
        sys.path.insert(0, here)
    module_name = os.path.splitext(os.path.basename(script))[0]
    mod = importlib.import_module(module_name)

    app = _modal_shim.App.last
    entry = getattr(app, "entrypoint", None) if app else None
    if entry is None:
        entry = getattr(mod, "main", None)
    if entry is None:
        print(f"{script}: no @app.local_entrypoint found", file=sys.stderr)
        return 1

    sig = inspect.signature(entry)
    kwargs = {}
    for name, p in sig.parameters.items():
        flag = name.replace("_", "-")
        raw = flags.get(flag, flags.get(name))
        if raw is not None:
            kwargs[name] = _coerce(raw, p.annotation)

    print(f"[local_run] {script} -> main({kwargs})", flush=True)
    entry(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
