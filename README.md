# Training Terminal

Desktop GUI (PySide6) that unifies the `modal_train_*.py` point-cloud training
scripts: bring your own dataset, pick a model, train or run inference on Modal,
and view the predicted point cloud — all in one place.

## Install + run (pixi — recommended)

The env is pinned to Python 3.11 because Open3D ships no wheels for 3.13
(which is what a current base conda gives you). [pixi](https://pixi.sh) handles
all of it on **Windows and Linux** (`win-64` + `linux-64`) — the commands are
identical on both (the block below is PowerShell; use any shell):

```powershell
cd trainer_gui
pixi install        # one-time: creates .pixi env with python 3.11 + open3d
pixi run gui        # launch the app
pixi run test       # local pipeline checks
```

First launch only: authenticate Modal inside the env with
`pixi run modal token new` (or reuse an existing `modal` on PATH — the GUI
calls whichever it finds first).

<details><summary>pip alternative (needs Python 3.10–3.12)</summary>

```powershell
cd trainer_gui
pip install -e .
trainer-gui          # or: python -m trainer_gui
```
</details>

## Workflow

1. **Datasets page** — pick a training folder and a validation folder of point
   clouds (`las/laz`, `ply`, ASCII `txt/csv/xyz/pts`, `pcd`, `npy/npz`). Choose
   where the ground-truth labels live: a field inside each file (e.g. LAS
   `classification`, PLY `label`) or a companion label file per scene (the IEEE
   `_PC3.txt` / `_CLS.txt` layout). Scan label values, name your classes, and
   uncheck any value that means *unknown* (those become ignore-labels, excluded
   from the loss and mIoU). *Analyze* measures point density and pre-computes
   per-model parameter recommendations. *Convert + Upload* writes the canonical
   dataset and pushes it to the `terminal-datasets` Modal volume.

1b. **Pretraining page** *(optional pre-step)* — prepare raw clouds before you
   convert a dataset. **Add HAG**: point at a folder of clouds (`las/laz`, or
   `txt/csv/xyz/pts/ply/pcd/npy/npz` — non-LAS inputs are transformed to LAZ on
   the way through) and PDAL computes *HeightAboveGround* (SMRF
   ground-classification → `hag_nn`), writing each cloud back out as `.laz` with
   a new `HeightAboveGround` dimension plus a `.json` sidecar (pipeline + HAG
   stats). **Tile for a
   model**: point at a folder, pick a backbone, and it produces train-ready
   tiles using the same tiling logic the Train page expects. Everything writes
   to a local folder you then point the Datasets page at.

2. **Train page** — pick the dataset + model. Grid size, tile size and batch are
   pre-filled from the density analysis (★ = recommended); everything stays
   editable. "Smoke run" does 2 epochs × 50 steps on an A10G (~pennies) to
   validate a new dataset end-to-end. Logs stream live; per-epoch loss/acc/mIoU
   fill the metrics table. Detached runs survive closing the app — re-attach
   any time with the button (runs `modal app logs <app>`).

   **Prep tiles locally** (on by default): tiling, grid subsampling and
   OctFormer's normal estimation run on your machine, and the finished cache is
   uploaded to `terminal-datasets:/<dataset>/prep/<tag>/` before launch. The
   script's remote `ensure_prep()` finds everything cached and skips straight
   to training — no Modal CPU time spent preprocessing. Needs the dataset's
   local staged copy (i.e. converted on this machine); otherwise it falls back
   to remote prep automatically. Note: prep depends on the tile/grid size, so
   changing those params re-preps under a new cache tag.

3. **Runs page** — list runs on each model's outputs volume, download artifacts,
   inspect per-class IoU, render the metrics dashboard (via the repo's
   `plot_run_metrics.py`), and open predictions in the 3D viewer.

4. **Inference page** — pick weights (a finished run, or a local `.pth` which
   gets uploaded), point at a folder of new clouds, run `--mode infer` on
   Modal, and the predictions download automatically for viewing.

## Wired models

| Model | Script | Status |
|---|---|---|
| PTv3 (cold) | `scripts/modal/modal_train_ptv3.py` | ready |
| PTv3 + HAG | `scripts/modal/modal_train_ptv3_hag.py` | ready |
| RandLA-Net (cold) | `scripts/modal/modal_train_randlanet.py` | ready |
| RandLA-Net + HAG | `scripts/modal/modal_train_randlanet_hag.py` | ready |
| KPConvX-L (cold) | `scripts/modal/modal_train_kpconvx_cold.py` | ready |
| KPConvX-L + HAG | `scripts/modal/modal_train_kpconvx_cold_hag.py` | ready |

Every script still runs standalone — `modal run scripts/modal/modal_train_X.py`
with no flags reproduces the original IEEE behavior. The GUI just passes
`--dataset/--grid/--epochs/...` flags and sets `TT_GPU` / `TT_TIMEOUT_HOURS`.

## Repo layout

```
scripts/modal/   thin Modal entrypoints (`modal run …`; bake + subprocess the local twin)
scripts/local/   the actual trainers/inferencers (run directly in Docker, no modal)
scripts/helper/  shared bits: train_common.py, _modal_shim.py, make_groundtruth_ply.py
trainer_gui/     the PySide6 desktop app (the pip-installed package)
docker/          generated Dockerfiles + build/pull/push scripts
tools/           gen_dockerfiles.py (+ the one-shot split_local.py)
```

## Canonical dataset format

`%APPDATA%/trainer_gui/staging/<name>/` (uploaded to `terminal-datasets:/<name>`):

- `dataset_meta.json` — classes (source value → index → name), counts, density
  stats, per-model recommendations
- `train/<scene>.npz`, `val/<scene>.npz` — `xyz` (f32), `label` (i32, −1 =
  ignored), plus `rgb` (u8), `intensity` (f32, normalized 0–1) and
  `return_number` (f32) when the source has them

Your **val/** folder is used as the *test* set; an in-distribution validation
holdout is carved deterministically from train/.

## Tests

```powershell
cd trainer_gui
python tests/smoke_test.py
```
