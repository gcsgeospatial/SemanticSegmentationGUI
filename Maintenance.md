# Maintenance

A working manual for this repo: what the layers are, which seams are contracts
you may not move, and how to do the recurring jobs — adding an architecture,
adding a dependency, turning a local trainer into its cloud twin, publishing
weights.

Companion doc: `../inference_tui/Maintenance.md` covers the `sem` CLI. It ships
its own copy of the trainer code, which diverges from this one deliberately —
there is no automatic sync between the two. Read that doc before changing
anything under `scripts/`, because a change here reaches CLI users only if
someone carries it across.

---

## 1. The sixty-second model

Three layers. Nothing crosses a layer boundary except files, a command line,
and environment variables.

```
trainer_gui/         PySide6 desktop app.  Never imports torch. Never touches CUDA.
   |  builds (program, args, env)
   v
trainer_gui/local_cli.py   ->  pixi run -e <env> python scripts/local/local_train_<key>.py ...
trainer_gui/modal_cli.py   ->  modal run scripts/modal/modal_train_<key>.py ...
   |
   v
scripts/local/local_train_*.py   the trainers. One per architecture.
scripts/helper/train_common.py   the shared trainer machinery.
```

The GUI runs Python 3.13 with numpy 2 and Qt. A trainer runs Python 3.10 or
3.11 with numpy <2, a specific torch build and a specific CUDA. They cannot
share a process, and they don't try to. If you ever want to `import torch`
inside `trainer_gui/`, stop — that makes the app uninstallable on every machine
that doesn't already have that exact stack.

The Modal side is the same idea applied again: `scripts/modal/modal_train_*.py`
is a *shell*. It builds a container image, copies the local trainer into it,
and runs it as a subprocess. Local and cloud execute the same trainer code;
there is no separate cloud implementation and there should never be one.

---

## 2. Repo map

| Path | What it is |
|---|---|
| `trainer_gui/main.py` | Entry point + main window (three-tab stack). |
| `trainer_gui/pages/` | `datasets_page`, `train_page`, `infer_page` — one file per screen. |
| `trainer_gui/backbones.py` | The architecture registry: script path, Modal app name, tunable parameters and ranges. |
| `trainer_gui/dataset.py` | Raw clouds → canonical dataset; inference-job staging, prediction merge, export. |
| `trainer_gui/readers.py` | Format I/O (las/laz/ply/pcd/txt/npz) and CRS capture/restore. |
| `trainer_gui/pretrain.py` | Height-above-ground and geometric-feature engines. (Feature computation, not pretrained weights — the name is historical.) |
| `trainer_gui/postproc.py`, `panoptic.py` | Re-runnable cleanup passes; ALPINE instance clustering. |
| `trainer_gui/jobs.py` | `JobRunner` (QProcess), `LogParser` (log regexes), `FuncWorker` (background thread). |
| `trainer_gui/local_cli.py`, `modal_cli.py` | Command builders; mirrors of each other. |
| `trainer_gui/appstate.py` | Persisted JSON state in the per-OS app dir; workspace and dataset registry. |
| `scripts/local/` | The trainers. Source of truth for all training and inference logic. |
| `scripts/modal/` | Cloud shells. Own the image, GPU, volumes, flag forwarding — nothing else. |
| `scripts/helper/train_common.py` | Shared: tiling, prep caching, eval, checkpoints, manifests, TTA, losses, resume. |
| `envs/pixi.toml` | Per-architecture **training** environments. A separate pixi workspace on purpose. |
| `pixi.toml` (root) | The GUI's own environment only. |
| `tools/package_weights.py` | A finished run → a `trainer-weights-*` conda package. Gitignored maintainer tooling. |

### Tooling kept outside the published tree

Excluded via `.gitignore` and currently living one directory up in `Modal/old/`:

- `conda-recipes/` — the `trainer-src-*` recipes that package each pinned
  upstream architecture source as a conda package.
- `devtools/check_env_sync.py` + `_modal_shim.py` — the drift checker that
  proves `envs/pixi.toml` and those recipes still agree with the Modal image
  definitions.

`scripts/modal/README.md` and `envs/pixi.toml` still reference these by their
former in-repo paths. Either move them back and re-gitignore, or fix the
comments — but know they exist and are load-bearing before touching
environments.

---

## 3. The contracts

These are the interfaces between layers. Artifacts already exist on disk and on
Modal volumes that were written against them, so treat them as fixed and extend
only additively.

### 3.1 A dataset directory

```
<workspace>/<name>/
    dataset_meta.json          classes, split record, source info, intensity_norm, CRS
    train/<scene>.npz
    val/<scene>.npz
    test/<scene>.npz
```

Each `.npz`: `xyz` (f64), `label` (i32, `-1` = ignore), optional `rgb` (u8),
`intensity` (f32), `return_number` (f32), plus baked channels (`feat_hag`,
`feat_geo_*`, `feat_<custom>`).

The split is decided once, at build time, by point-count fractions over whole
scenes (or over tiles of a single cloud), and recorded. Trainers read the
folders verbatim and never re-split. The seed is fixed so a rebuild reproduces
the identical split, which is what keeps trainer prep-cache signatures valid
across rebuilds.

### 3.2 `run.json`

Written by `train_common.write_run_manifest`, sits beside `final_model.pth`,
and is the only thing inference reads to reconstruct how a model was trained.

```json
{ "schema": "trainer_gui.run/2", "backbone": "utonia", "weights": "final_model.pth",
  "num_classes": 8, "class_names": [...], "grid": 0.25, "chunk_xy": 50.0,
  "intensity_norm": "p95", "num_points": null, "features": [...],
  "color_source": "intensity", "dg": { "logdk": false, "logdk_k": 8, ... } }
```

**Additive-only, permanently.** Published weights packages carry a copy of this
file, including ones on machines you can't reach. New optional fields are fine;
renaming a field or changing its meaning breaks every model already in the
field.

`backbone` must equal a key in `backbones.py::BACKBONES` — that's how the
Inference page auto-selects the architecture for a picked run.

A run directory is otherwise minimal by design: `run.json`, `final_model.pth`,
one `metrics.csv`, `checkpoints/`. Test metrics and finish state fold into
`run.json`. Adding sidecar files re-creates sprawl that was deliberately
removed.

### 3.3 An inference job

One directory holds the whole job:

```
<output>/
    <stem>_input.npz      staged channels AND the merged predictions
    job.json              the job record (staging config, scenes, results)
    <stem>_pred.laz       the export
```

No separate scenes directory, no per-job home, no side JSONs.

### 3.4 Environment variables

Flags are the stable public surface; environment variables are the knob
surface. `local_cli` sets them in the child process directly; `modal_cli` packs
them into one `--env-json` argument that `train_common.modal_shell_run` unpacks
back into the container subprocess's environment. Same variables either way.

| Prefix | Owns | Examples |
|---|---|---|
| `TT_` | Paths, volumes, runtime toggles | `TT_DATASETS_ROOT`, `TT_OUTPUTS_ROOT`, `TT_DATASET_DIR`, `TT_INFER_DIR`, `TT_PRED_DIR`, `TT_GPU`, `TT_TIMEOUT_HOURS`, `TT_SAVE_PROBS`, `TT_INFER_OVERLAP`, `TT_ZERO_CHANNELS`, `TT_AMP`, `TT_PREFETCH` |
| `DG_` | Density generalization + test-time adaptation | `DG_DENSITY_AUG`, `DG_COARSEN_MAX`, `DG_P_NATIVE`, `DG_LOGDK_FEAT`, `DG_LOGDK_K`, `DG_INFER_TTA`, `DG_INFER_TTA_RIGID`, `DG_INFER_ADABN`, `DG_INFER_APCOTTA` |
| `LOSS_`, `RARE_` | Loss shaping, rare-class oversampling | `LOSS_CLASS_WEIGHTING`, `LOSS_FOCAL`, `RARE_OVERSAMPLE`, `RARE_TILE_PROB` |
| `PROXY_`, `EVAL_` | Validation protocol, eval batching | `PROXY_TILES`, `PROXY_SAMPLING`, `EVAL_VOTES`, `EVAL_ONLY` |
| `FEAT_CHANNELS`, `EXCLUDE_CLASSES` | Input spec, class masking | |
| `AUTO_RESUME` | Set by the Modal retry marker | |

Read them through `train_common.env_bool/env_float/env_int/env_str`, never
`os.environ` directly — the coercion rules live there.

`DG_LOGDK_FEAT` is special: it changes the model's input width, so it travels
with the weights in `run.json["dg"]` and inference re-applies it automatically.
AdaBN/APCoTTA/TTA are per-job and do not.

### 3.5 Log lines the GUI parses

`jobs.py` scrapes stdout with three regexes. Change these print statements in a
trainer and the GUI's live metrics and run-id detection go dark, silently.

```
ep  12: loss=0.4321 acc=0.9123 miou=0.7012 s/iter=0.123 s/ep=61.4
[val@ep9] acc=0.91 mIoU(5-way)=0.70 mIoU(present 4)=0.74
run complete -> 20260805_101500_ptv3
RESUMING 20260805_101500_ptv3        (or: EVAL-ONLY on <id>)
```

`test@epN` deliberately does not match the val regex.

---

## 4. Turning a local trainer into its Modal version

The rule: **all logic lives in `scripts/local/`; the Modal shell owns only
deployment.** A shell owns the image (dependencies + pinned upstream source),
GPU/timeout/retries, volume mounts, flag forwarding, and the `--env-json`
passthrough. An `if` statement in a shell almost always belongs in the local
trainer.

```python
"""Modal shell for <Arch> - shells out to local_train_<key>.py."""
import os, sys
from typing import Optional
import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local run
sys.path.insert(0, "/root")                                     # in-container
import _shell

APP_NAME = "<key-with-dashes>"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", ...)
    .pip_install("torch==X.Y.Z", ..., index_url="https://download.pytorch.org/whl/cuNNN",
                 extra_index_url="https://pypi.org/simple")
    .env({"PYTHONUNBUFFERED": "1"})
)

# pinned upstream clone - the SHA IS the architecture version
image = image.run_commands(
    "git clone https://github.com/<org>/<repo>.git /opt/<pkg>"
    " && git -C /opt/<pkg> checkout --detach <40-char-sha>"
    " && rm -rf /opt/<pkg>/.git")

image = image.add_local_file("scripts/local/local_train_<key>.py", "/root/local_train_<key>.py")
image = image.add_local_file("scripts/helper/train_common.py", "/root/train_common.py")
image = image.add_local_file("scripts/modal/_shell.py", "/root/_shell.py")

app, outputs_volume, datasets_volume, _fn_kwargs, _launch = _shell.setup(APP_NAME)

@app.function(image=image, **_fn_kwargs)
def train_<key>(dataset: Optional[str] = None, ..., env_json: Optional[str] = None):
    import sys; sys.path.insert(0, "/root")
    import train_common
    train_common.modal_entry(
        modal.current_function_call_id(), "/root/local_train_<key>.py",
        [("--dataset", dataset), ("--grid", grid), ..., ("--mode", mode),
         ("--weights", weights), ("--infer-input", infer_input)],
        env_json, outputs_volume, datasets_volume)

@app.local_entrypoint()
def main(dataset: Optional[str] = None, ..., env_json: Optional[str] = None):
    _launch(train_<key>, **locals())
```

Behavior that skeleton buys you:

- `_shell.setup()` centralizes app, volumes, GPU (`TT_GPU`), timeout
  (`TT_TIMEOUT_HOURS`), CPU/memory and a 10-retry policy. Ten retries is safe
  because each retry auto-resumes.
- `train_common.modal_entry()` drops a marker file on the outputs volume keyed
  by the Modal function-call id. A marker that already exists means "this is a
  retry", so it sets `AUTO_RESUME=1` and the trainer picks up the last
  checkpoint instead of restarting.
- `modal_shell_run` commits both volumes every 120 s and reloads the outputs
  volume each cycle. That reload is how **graceful stop** works: the Train page
  uploads a `/STOP` file, the container sees it within ~2 minutes, finishes the
  epoch, runs the full final evaluation, and finalizes. Stop must never mean
  kill.
- `train_common` never imports modal; volumes are duck-typed (`.commit()`,
  `.reload()`) so the local path passes nothing.
- Flags whose value is `None` are dropped, so the local trainer's own defaults
  win. Keep every parameter `Optional[...] = None` all the way down.

**Keeping image and local environment in sync.** The Modal image recipe is the
source of truth for training dependencies; the same pins must appear in the
matching `[feature.<key>]` of `envs/pixi.toml`, and the git SHA must match the
`trainer-src-<key>` conda recipe. The drift checker imports each Modal script
under an offline shim, re-derives the pins and SHA, and diffs all three. Run it
after any dependency change — the failure it prevents is "trains fine on Modal,
crashes on the user's local GPU".

---

## 5. Adding an architecture

Two flavors; decide which you have first.

### Flavor A — a wrapper over an existing trainer

Same architecture, different pretrained weights (the Concerto/Sonata/Utonia
case over PTv3). About twenty lines:

```python
# scripts/local/local_train_<key>.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_train_ptv3 as _base

# applied per CALL, not at import - several wrappers may share one process
_CFG = dict(PKG="<pkg>", HF_NAME="<hf>", HF_REPO="<org>/<Repo>",
            BB_KEY="<key>", BASE_LR=6e-4)

def train_pcssl(*args, **kw):
    _base.__dict__.update(_CFG); return _base.train_pcssl(*args, **kw)

def main():
    _base.__dict__.update(_CFG); _base.main()

if __name__ == "__main__":
    main()
```

"Per call, not at import" is load-bearing: the wrappers mutate globals on a
shared module, so applying config at import time lets two wrappers in one
process silently train each other's model. Then do steps 2–6 below.

### Flavor B — a new architecture

1. **`scripts/local/local_train_<key>.py`.** Follow `local_train_ptv3.py`'s
   shape: module-level ALL-CAPS defaults, one `train_<key>(...)` taking every
   knob as `Optional[...] = None`, and a `main()` whose argparse mirrors the
   signature exactly. Import `train_common` by relative path — the runtime
   ships as a flat tree and relies on `__file__`-relative resolution.

   Pull as much as possible from `train_common`: tiling and prep caching, class
   balance scan, rare-class picking, proxy validation, loss construction, the
   checkpoint/resume ladder, evaluation, overlap voting, TTA, manifest writing.
   Writing a second copy of something that already exists there is the signal to
   hoist instead.

   Non-negotiable for a new trainer: emit the four log lines from §3.5 exactly;
   call `write_run_manifest()` at the end; support
   `--mode infer --weights ... --infer-input <job_id>`; honor `AUTO_RESUME` and
   the `/STOP` sentinel via `stop_requested()`; save checkpoints with
   `atomic_torch_save` and JSON with `atomic_json_save`.

2. **`scripts/modal/modal_train_<key>.py`** — the skeleton in §4.

3. **`trainer_gui/backbones.py`** — one `Backbone(...)`:

   ```python
   Backbone(key="<key>", label="<Display Name>",
            script="scripts/modal/modal_train_<key>.py",
            app_name="<key-with-dashes>", rec_gpu="A100",
            params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.02, 3.0,
                              step=0.05, decimals=2)] + _common(150, 4))
   ```

   `key` is the hinge: it names the pixi environment (`_` → `-`), the local
   script filename, and the value written to `run.json["backbone"]`. The
   `params` list drives the Train page form automatically; add a `PARAM_TIPS`
   entry for any new flag so its tooltip isn't blank. Grid and tile defaults
   stay at the published ALS operating point (0.25 m / 50 m); only batch scales
   with hardware, because VRAM is the only thing it actually controls.

4. **`envs/pixi.toml`** — a `[feature.<key>]` block plus an `[environments]`
   entry. Copy the closest existing feature and change the pins. What catches
   people:
   - `platforms` must reference the named CUDA variants
     (`linux-64-cuda121`, `win-64-cuda121`), not bare platforms.
   - Set `<KEY>_SRC` in **both** `[feature.<key>.activation.env]` and
     `[feature.<key>.target.win-64.activation.env]` — cmd.exe expands `%VAR%`,
     not `$VAR`, and the conda packages ship only `.sh` activation scripts.
   - `[feature.<key>.pypi-options]` mirrors the Modal `pip_install` call's
     `index_url` / `extra_index_url` / `find_links`, with
     `index-strategy = "unsafe-best-match"` and `no-build = true` (wheels only,
     so the lock solves from any platform).
   - `[environments]` names use dashes.

5. **`trainer-src-<key>` conda recipe** (in the external tooling directory,
   §2) — packages the pinned upstream source to `$PREFIX/opt/<pkg>` and writes
   the activation script. Version bumps when the SHA moves. Build and upload
   with the `pkg` environment's tasks, and register the backbone in the drift
   checker's table.

6. **`trainer_gui/pages/train_page.py`** — add the key to `_FEAT_STANDARD`
   (which channels the architecture can consume), and to `_PTV3_LIKE` /
   `_XYZ_IMPLICIT` if it has a single 3-wide color slot or an implicit xyz
   prefix in its feature spec. This is the only place in the GUI with hardcoded
   backbone keys; everything else iterates `BACKBONES`. Keep it that way.

Then validate (§7).

---

## 6. Environments and locks

Two pixi workspaces, deliberately separate:

- **root `pixi.toml`** — the GUI only (Python 3.13, PySide6, PDAL, laspy,
  pyproj, pgeof, modal).
- **`envs/pixi.toml`** — one environment per architecture plus a `pkg` tooling
  environment. Separate because pixi locks every environment of a workspace
  together, and a win-64 GUI solve must not drag several linux-64 CUDA stacks
  along with it.

Dependency changes land atomically: both platforms built and uploaded, and
`pixi.lock` updated, in the same commit. A lock referencing an artifact nobody
uploaded is a broken install for everyone who pulls.

`local_cli` runs `pixi run --frozen` — install exactly what the lock says,
never re-solve. (`--locked` is avoided: pixi ≤ 0.73 false-positives on
multi-environment PyPI index attribution.) A first install of a training
environment pulls multi-GB CUDA wheels, so `env_preflight()` hard-blocks with
the exact `pixi install` command rather than doing it as a side effect of
pressing Launch. That reflects the house rule: **preflights block, and every
message names the exact remedy** — one cause per message, no warn-and-proceed.

---

## 7. Publishing weights

`tools/package_weights.py` turns a finished `runs/<id>/` into a noarch conda
package `trainer-weights-<dataset>-<backbone>` containing `final_model.pth`,
`run.json`, and a generated `NOTICE.md`. Any number of these can exist — one
per dataset × architecture × training round — and they all install side by side
into `$PREFIX/share/trainer-weights/<name>/`.

```
pixi run --manifest-path envs/pixi.toml -e pkg package-weights <run_dir>
pixi run --manifest-path envs/pixi.toml -e pkg upload
```

- `--no-include-recipe` is passed on purpose: without it rattler-build embeds a
  second copy of the staged source under `info/recipe/`, doubling a
  multi-hundred-MB artifact.
- Retraining the same dataset and architecture produces a **new version**, never
  a rebuild of an existing one (§9).
- `BACKBONE_NOTICES` in that script is the license ledger. An architecture with
  no entry gets a loud `UNVERIFIED_NOTICE` and a `LicenseRef-Unverified`
  license — a deliberate publication blocker. The Pointcept-family encoders are
  CC-BY-NC 4.0 and anything finetuned from them inherits the non-commercial
  restriction. Verify and record upstream terms before publishing a new
  architecture's weights.

Consumers: the GUI's Inference page lists anything installed under
`share/trainer-weights/`, and the `sem` CLI installs these packages as its
products.

---

## 8. Validating a change

There is no GPU on most working machines and no test suite in the repo, so
validation is a ladder — say which rung you reached:

1. **GPU run** — the only rung that proves anything about training.
2. **CPU imitation** — the same code path on tiny synthetic inputs. Most
   trainers are CUDA-only (`.cuda()` with no fallback), so in practice this
   covers `dataset.py`, `pretrain.py`, `postproc.py`, `readers.py`.
3. **Compile + review** — `compileall`, adversarial reading, and checking
   upstream docs for any API you pinned. This is the floor, not a pass.

Cheap checks that catch most regressions:

```
python -m compileall -q scripts trainer_gui
python -c "import trainer_gui.backbones as b; print(list(b.BACKBONES))"
python <tooling dir>/check_env_sync.py                     # after any dependency change
pixi run --manifest-path envs/pixi.toml -e <env> sanity    # per-architecture import smoke
```

Announce expected runtime before starting anything CPU-heavy on someone else's
machine; a full conversion or geo-feature pass runs for many minutes.

---

## 9. Traps

- **Never rebuild a published conda version.** Rebuilding overwrites the
  artifact but not its repodata entry, wedging every installer on a hash
  mismatch. Bump instead.
- **`run.json` is permanent and additive-only** (§3.2).
- **`height` is permanently removed.** A `height` channel once existed; it is
  gone from the vocabulary, feature builder, defaults and UI, and must not
  return under that name.
- **Windows:** multiprocessing uses spawn, so anything handed to a worker must
  be picklable (module-level functions, not closures). Killing a job needs
  `taskkill /PID <pid> /T /F` — `QProcess.kill()` reaches only the direct child
  and orphans the trainer holding the GPU. Force `PYTHONUTF8=1` and
  `PYTHONIOENCODING=utf-8` on every child; Modal's box-drawing output crashes
  cp1252 consoles.
- **Prep caches are signature-validated.** A mismatch is supposed to trigger a
  rebuild; if a stale cache survives anyway, that's a bug to chase, not to work
  around. The manual fix is deleting the prep directory.
- **CUDA assertions must re-raise.** A device-side assert leaves the context
  unusable; swallowing it produces an endless stream of misleading downstream
  errors.
- **Pixi environment names can't contain underscores.** `kpconvx_cold` (key) ↔
  `kpconvx-cold` (environment, Modal app). `local_cli.env_name()` is the only
  place that translation belongs.
- **Modal images are expensive to rebuild.** After an image recipe change the
  next `modal run` rebuilds, which takes a long time and surprises a user
  mid-workflow. Warn, or pre-warm it yourself.

---

## 10. Conventions

Decisions already made; reverse them knowingly, not by accident.

- Contract, UX, and "should this exist at all" questions are owner decisions —
  present options rather than shipping a direction. Mechanical work proceeds on
  its own.
- Defaults are sensible, pre-selected, visible, and one click to undo. Making
  everything opt-in was tried and produced worse results for real users.
- Preflights block; messages name the remedy; one cause per message.
- Dedupe aggressively into `train_common.py`. The older "duplication between
  trainers is deliberate" rule is dead — it produced seven divergent copies of
  the same evaluation bug.
- Comments cover non-obvious constraints in one terse line. If a comment runs
  to a paragraph, the code usually wants simplifying instead.
- Prefer deleting to keeping "just in case". Era-compat layers, legacy artifact
  readers, and testing branches were removed on purpose; supporting old
  artifact formats measured more expensive than regenerating them.
