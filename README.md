# Training Terminal

Desktop GUI (PySide6) for training and running point-cloud semantic-segmentation
models. Bring a folder of point clouds, pick a model, train it, run inference,
view the predicted cloud — all in one window.

Two execution backends: **Local (Docker)** and **Modal (cloud)** — switched in
the sidebar. Both run the same `scripts/local/` trainers (the Modal shells
subprocess them on a cloud GPU; see `scripts/modal/README.md`). Everything
below is the local path.

## Run it

```powershell
cd trainer_gui
pixi install     # one-time: builds .pixi env (python 3.11 + open3d + pdal + qt)
pixi run gui     # launch
pixi run test    # smoke checks, no Docker/Modal needed
```

Python is pinned to 3.11 (Open3D has no 3.13 wheels). [pixi](https://pixi.sh)
does the rest on win-64 and linux-64. In the sidebar, set **Execution backend →
Local (Docker)**.

<details><summary>pip alternative (Python 3.10–3.12)</summary>

```powershell
cd trainer_gui
pip install -e .
trainer-gui        # or: python -m trainer_gui
```
</details>

## How it runs training/inference

The GUI never trains in-process. It builds a `docker run` command and executes
it. One Docker image per model, each with the CUDA stack + model source **baked
in** — the image is self-contained.

```
docker run --rm --gpus all --ipc=host -w /workspace \
  -v <repo>:/workspace  -v <staging>:/datasets  -v <output>:/outputs \
  trainer-local-<model>  python scripts/local/local_train_<model>.py --flags...
```

Three host dirs are bind-mounted; nothing is uploaded or downloaded — inputs are
already on the host, and checkpoints/predictions land straight back in `/outputs`:

| Mount | Host dir | Holds |
|---|---|---|
| `/workspace` | repo root | the `scripts/local/*.py` that run |
| `/datasets` | staging root | canonical datasets + `_infer/<job>` inputs |
| `/outputs` | your chosen folder | `runs/<id>/` weights + predictions |

Code lives in `trainer_gui/local_cli.py`. **No Docker on PATH → the GUI prints
the exact command instead of running it** (dry-run — this dev box is Intel Arc,
no CUDA).

## The scripts

```
scripts/local/    the real trainers/inferencers — plain argparse, no modal.
                  Edit these. Run standalone: python scripts/local/local_train_ptv3.py --dataset X
scripts/modal/    thin shells that bake the local twin into a modal.Image
                  and subprocess it in the cloud (train_common.modal_shell_run).
scripts/helper/   train_common.py (shared training/manifest logic),
                  density.py, _modal_shim.py (used only by gen_dockerfiles.py)
```

Each `local_train_<model>.py` takes the same kebab-case flags the GUI fills in
(`--dataset --grid --epochs --batch ...`) and one `--mode infer` path.

## The Docker images

One image per model, generated — never hand-write a Dockerfile:

```bash
python tools/gen_dockerfiles.py          # regenerate docker/*.Dockerfile + build/pull/push scripts
                                          # RE-RUN after editing any script's image recipe
bash docker/build_all.sh                  # build — works on any machine with Docker + Buildx
```

Model sources are **git-cloned at pinned SHAs during the build** — no local
model checkouts and no GPU needed at build time (an NVIDIA GPU is still
required at **run** time; the images are CUDA-based). Every other machine just
needs the built image.

**Distribute build-once-run-anywhere:**

```bash
TT_REGISTRY=ghcr.io/<you> bash docker/push_all.sh   # build machine
TT_REGISTRY=ghcr.io/<you> bash docker/pull_all.sh   # any other machine
```

Set `TT_REGISTRY` (or `local_config["registry"]` in the app state JSON) before
launching and `docker run` auto-pulls missing images. Registry = the images;
use Hugging Face / S3 for the `.pth` weights. Full details: `docker/README.md`.

## Models

| Model | key | Local script |
|---|---|---|
| PTv3 | `ptv3` | `scripts/local/local_train_ptv3.py` |
| PTv3 + HAG | `ptv3_hag` | `scripts/local/local_train_ptv3_hag.py` |
| RandLA-Net | `randlanet` | `scripts/local/local_train_randlanet.py` |
| RandLA-Net + HAG | `randlanet_hag` | `scripts/local/local_train_randlanet_hag.py` |
| KPConvX-L | `kpconvx_cold` | `scripts/local/local_train_kpconvx_cold.py` |
| KPConvX-L + HAG | `kpconvx_cold_hag` | `scripts/local/local_train_kpconvx_cold_hag.py` |
| KPConv | `kpconv` | `scripts/local/local_train_kpconv.py` |
| KPConv + HAG | `kpconv_hag` | `scripts/local/local_train_kpconv_hag.py` |

`_hag` = an extra **HeightAboveGround** input channel (ground = the file's ground
class when set, else PDAL SMRF detection; interpolated by grid / `hag_nn` / `hag_delaunay`).
The `_hag` scripts are thin wrappers that run the base script with `--hag` — one
trainer per backbone, so recipe changes land in one file.

A `_hag` model **requires a real HAG channel**: train it on a dataset built with
**Compute Height-Above-Ground**, and tick the same box on the Inference page. There
is no substitute height — a missing `hag` channel is a hard error, not a silent
fallback. Checkpoints from before this rule (their `run.json` records
`hag_source: z_minus_scene_min_proxy`) learned a stand-in height and must be
retrained; the Inference page refuses them.

## The normal path: Datasets → Train → Inference

Three pages, front to back. Each one fills one of the bind-mounts above.

### 1. Datasets — turn raw clouds into a trainable dataset

Builds `/datasets/<name>` (the canonical `.npz` splits). Three numbered sections,
top to bottom:

1. **New dataset** — give it a **Name**, point **Input** at a file or a folder of
   clouds (`las/laz`, `ply`, ASCII `txt/csv/xyz/pts`, `pcd`, `npy/npz`). The
   **Label field** dropdown auto-probes the first file for label fields (e.g.
   `classification`, `scalar_label`) — pick which one holds the class. **Output
   folder** defaults to the app staging dir.
2. **Classes** — click **Scan label values**: every distinct label value shows up
   as a row (value, points seen, editable class name, a **Train** checkbox).
   Rename classes, **uncheck** any value that means "unknown" (→ ignore-label,
   dropped from loss + mIoU), select rows + **Combine** to merge several values
   into one class. **Analyze density** prints pts/m², spacing and a suggested tile
   size, and pre-computes the per-model param recommendations the Train page uses.
3. **Train / val / test split** — set **Validation** and **Test** fractions
   (train = the remainder), **Split mode** (Balanced mirrors the class mix /
   Random fills by point count), and the **seed** (default 42). Already have
   split folders? Tick **Separate train/val/test folders (use as-is)**. Optional
   **Compute Height-Above-Ground (HAG)** bakes a per-point HAG channel (ground =
   your labeled ground class when set, else SMRF detection; never a mix) — only
   the `_hag` models read it.

Hit **Build dataset**. It writes `train/ val/ test/` `.npz` to staging; progress
streams in the console. Done → the dataset appears under **Saved Datasets** and is
ready to pick on the Train page.

### 2. Train — run a model on the dataset

Writes `/outputs/runs/<id>/` (weights + logs + `run.json`).

- Pick the **Dataset** (the status line confirms *✓ train/val/test standard met*)
  and a **Model**. **Configure model…** opens a popup showing that model's Docker
  image status, a **Pull** button, and the registry field.
- **Parameters** are pre-filled from the density analysis (**★** = recommended) —
  grid/sub-grid, epochs, batch, steps/epoch, tile size. All editable.
- **Smoke run** (2 epochs × 50 steps) validates a new dataset end-to-end fast.
- Optional: per-run **Domain generalization** (train robust to a different
  inference density) and **Loss & class balance** knobs.
- Set the **Output folder** (bound to `/outputs`), hit **Launch training**. The
  exact `docker run` is echoed to the log; then logs stream live and per-epoch
  **Loss / Acc / mIoU** fill the metrics table. **Stop process** kills it.

On finish, each run dir has the weights and a **`run.json`** — the single file
Inference needs.

### 3. Inference — label new clouds with a trained model

Reads a run's `run.json` + weights, writes predictions to a host folder.

- **Weights → From a training run**: **Browse** to the run's **`run.json`**. It
  auto-fills the backbone, grid, tile size and intensity norm, ticks
  **Compute Height-Above-Ground** when the run trained with it (you can untick),
  and points **Weights file** at the sibling `.pth` (override if you want). (Or
  pick **Local .pth file** and choose the architecture yourself.)
- **Input**: a folder or single cloud to label. Grid/tile come from the run.
  **Compute Height-Above-Ground** is off by default and only the `_hag` models use
  it; when on, name the **ground class** (e.g. `2`) if your clouds already carry a
  ground classification, else ground is detected. Optional label-free
  **density adapt (AdaBN)** / **density TTA** for inference at a different density
  than training. Set the **Output folder** for predictions.
- **Run inference** converts the input to canonical scenes, then `docker run
  --mode infer` with the scenes bind-mounted and predictions written straight to
  your output folder (no upload/download).
- View results in place: **View a point cloud…**, **Compare to ground truth…**
  (prints accuracy + per-class mIoU, paints mismatches), **Export comparison
  PLY…**, and **Class colours & names…** to set the legend.
- **Ensemble 2-3 models** (typically ~+2 mIoU): run inference on the same input
  with each trained model (a separate output folder per model), then
  majority-vote the folders —
  `python scripts/local/ensemble_vote.py --inputs predsA predsB [predsC] --out voted`
  (strongest model first; it wins vote ties). Works across backbones and
  formats: all three models predict the same raw points, and the voter reads
  the exported `*_pred.las/laz/ply/txt/csv` (or raw `.npz`) directly.

## Canonical dataset format

`<staging>/<name>/` (bind-mounted to `/datasets/<name>`):

- `dataset_meta.json` — classes (source value → index → name), per-split counts,
  density stats, per-model recommendations, and the recorded split (mode, seed,
  fractions).
- `train/<scene>.npz`, `val/<scene>.npz`, `test/<scene>.npz` — `xyz` (f32),
  `label` (i32, −1 = ignored), plus `rgb`/`intensity`/`return_number`/`hag` when
  the source has them.

The Datasets page carves all three splits **once** (val % / test % sliders each
≥ 5 %, train takes the rest; balanced or random; seeded, default 42). A folder
splits by whole scenes; a single cloud is tiled and reassembled per split with a
seam buffer discarded to limit leakage. Trainers read the three folders verbatim
and never re-split.

## Repo layout

```
trainer_gui/     the PySide6 app (pip/pixi package) — pages/, local_cli.py, ...
scripts/local/   the real trainers/inferencers (run in Docker)
scripts/modal/   thin shells (see its README.md)
scripts/helper/  train_common.py, density.py, _modal_shim.py
docker/          generated Dockerfiles + build/pull/push scripts
tools/           gen_dockerfiles.py
```

## Tests

```powershell
cd trainer_gui
python tests/smoke_test.py     # or: pixi run test
```
