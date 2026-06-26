"""One-shot splitter: decouple each modal_train_X.py into

  * local_train_X.py  — the pure trainer/inferencer (no `import modal`). The
    ~1000-line body is copied VERBATIM; the three modal Volumes it closes over
    become local no-ops (data is on the bind-mounted /data /outputs /datasets),
    and a generated argparse `main()` replaces the @app.local_entrypoint.

  * modal_train_X.py  — rewritten as a THIN modal shell: same App / Image /
    Volumes / @app.function decorator, but the body just `subprocess`-runs the
    local script inside the GPU container (and commits the volumes), so all
    training logic lives in ONE place — the local script.

The six scripts share one exact shape (verified), so this marker-based slicer
handles all of them. Run once per script, then this tool can be deleted:

    python tools/split_local.py modal_train_ptv3.py
    python tools/split_local.py --all

ponytail: structural markers, not a full AST rewrite — the shape is uniform and
git is the undo. The body text is never reflowed, only relocated.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import typing

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SCRIPTS = [
    "modal_train_ptv3.py",
    "modal_train_ptv3_hag.py",
    "modal_train_randlanet.py",
    "modal_train_randlanet_hag.py",
    "modal_train_kpconvx_cold.py",
    "modal_train_kpconvx_cold_hag.py",
]


def _base_type(annot):
    """Optional[int] / int / ... -> the concrete leaf type (or None if unknown)."""
    base = annot
    for a in typing.get_args(annot):
        if a is not type(None):
            base = a
            break
    return base if base in (int, float, bool, str) else None


def _entry_params(stem: str):
    """Import the script under the modal shim and read its @app.local_entrypoint
    signature — the source of truth for the flags both new files must mirror."""
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "scripts", "helper"))   # _modal_shim moved here
    import _modal_shim
    _modal_shim.install()
    for m in list(sys.modules):
        if m.startswith("modal_train_"):
            sys.modules.pop(m, None)
    importlib.import_module(stem)
    entry = _modal_shim.App.last.entrypoint
    return list(inspect.signature(entry).parameters.values())


def _argparse_block(fn_name: str, params, doc: str) -> str:
    need_bool = any(_base_type(p.annotation) is bool for p in params)
    lines = []
    if need_bool:
        lines += ["def _as_bool(s):",
                  '    return str(s).strip().lower() in ("1", "true", "yes", "on")',
                  "", ""]
    lines += ["def main():",
              "    import argparse",
              f"    ap = argparse.ArgumentParser(description={doc!r})"]
    for p in params:
        flag = "--" + p.name.replace("_", "-")
        t = _base_type(p.annotation)
        default = None if p.default is inspect.Parameter.empty else p.default
        typ = {int: "int", float: "float", bool: "_as_bool"}.get(t)
        bits = [repr(flag)]
        if typ:
            bits.append(f"type={typ}")
        bits.append(f"default={default!r}")
        lines.append(f"    ap.add_argument({', '.join(bits)})")
    lines += ["    args = ap.parse_args()",
              f"    {fn_name}(**vars(args))",
              "", "",
              'if __name__ == "__main__":',
              "    main()",
              ""]
    return "\n".join(lines)


def _subprocess_block(local_script: str, params, vol_names) -> str:
    flags = "\n".join(
        f'        ("--{p.name.replace("_", "-")}", {p.name}),' for p in params)
    commit = "\n".join(f"            {v}.commit()" for v in vol_names)
    commit_final = "\n".join(f"        {v}.commit()" for v in vol_names)
    return f'''    """Modal shell: provision the GPU container + volumes, then run the LOCAL
    trainer. All training/inference logic lives in {local_script} — this only
    shells out to it, so local and cloud run byte-identical code."""
    import subprocess
    import sys
    import threading

    cmd = [sys.executable, "/root/{local_script}"]
    for _flag, _val in (
{flags}
    ):
        if _val is not None:
            cmd += [_flag, str(_val)]
    print("[modal-shell] " + " ".join(cmd), flush=True)

    # Persist checkpoints + prep cache mid-run so an uncatchable spconv CUDA
    # device-assert (the reason this function has retries) still leaves the
    # latest state on the volumes for the local trainer's auto-resume on the
    # retry. ponytail: time-based commit; the trainer's 2-checkpoint retention
    # covers the rare case of snapshotting a half-written .pth.
    _stop = threading.Event()

    def _commit_loop():
        while not _stop.wait(120):
{commit}

    _t = threading.Thread(target=_commit_loop, daemon=True)
    _t.start()
    try:
        subprocess.run(cmd, check=True)
    finally:
        _stop.set()
{commit_final}
'''


def split(script: str) -> None:
    stem = os.path.splitext(script)[0]
    key = stem.replace("modal_train_", "")
    path = os.path.join(REPO, script)
    local_name = f"local_train_{key}.py"
    local_path = os.path.join(REPO, local_name)

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    def find(pred, start=0):
        for i in range(start, len(lines)):
            if pred(lines[i]):
                return i
        raise RuntimeError(f"{script}: marker not found")

    i_app = find(lambda l: l.startswith("app = modal.App("))
    i_def = find(lambda l: l.startswith("def train_"), i_app)
    i_sig_end = find(lambda l: l.rstrip().endswith("):"), i_def)
    i_entry = find(lambda l: l.startswith("@app.local_entrypoint"), i_sig_end)

    if any("[modal-shell]" in l for l in lines[i_sig_end + 1:i_entry]):
        print(f"  ! {script} is already a thin modal shell — skipping to avoid "
              f"clobbering. `git checkout {script}` to re-split from scratch.")
        return

    header = [l for l in lines[:i_app] if l.strip() != "import modal"]
    scaffold = lines[i_app:i_def]
    sig = lines[i_def:i_sig_end + 1]
    body = lines[i_sig_end + 1:i_entry]
    entry = lines[i_entry:]

    fn_name = sig[0].split("(")[0].replace("def ", "").strip()
    params = _entry_params(stem)

    vol_names = [l.split("=")[0].strip()
                 for l in scaffold if "modal.Volume.from_name" in l and "=" in l]

    # ---- local_train_X.py : header (no modal) + no-op vols + body + argparse ----
    novol = (
        "\n# Local stand-ins for the modal Volumes the body commits to: the data is\n"
        "# already on the bind-mounted /data /outputs /datasets, so there is nothing\n"
        "# to upload. (The modal shell does the real commits when it runs this.)\n"
        "class _NoVol:\n"
        "    def commit(self, *a, **k): pass\n"
        "    def reload(self, *a, **k): pass\n\n\n"
        + ", ".join(vol_names) + " = " + ", ".join("_NoVol()" for _ in vol_names) + "\n\n\n"
    )
    local_doc = f"Local {key} trainer/inferencer (no modal)."
    local_text = (
        "".join(header).rstrip() + "\n\n"
        + novol
        + "".join(sig) + "".join(body).rstrip() + "\n\n\n"
        + _argparse_block(fn_name, params, local_doc)
    )
    with open(local_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(local_text)

    # ---- modal_train_X.py : header(+modal) + scaffold(+add_local_file) + shell --
    i_first_vol = next(k for k, l in enumerate(scaffold)
                       if "modal.Volume.from_name" in l)
    add_file = (f'image = image.add_local_file("{local_name}", '
                f'"/root/{local_name}")\n\n')
    scaffold2 = scaffold[:i_first_vol] + [add_file] + scaffold[i_first_vol:]

    modal_header = lines[:i_app]   # keep `import modal`
    modal_text = (
        "".join(modal_header).rstrip() + "\n\n"
        + "".join(scaffold2)
        + "".join(sig)
        + _subprocess_block(local_name, params, vol_names)
        + "\n\n"
        + "".join(entry).rstrip() + "\n"
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(modal_text)

    print(f"  {script}: body {len(body)} lines -> {local_name}; "
          f"modal shell rewritten ({len(params)} flags: "
          f"{', '.join(p.name for p in params)})")


def main(argv):
    targets = SCRIPTS if (argv and argv[0] == "--all") else argv
    if not targets:
        print(__doc__)
        return 2
    for s in targets:
        split(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
