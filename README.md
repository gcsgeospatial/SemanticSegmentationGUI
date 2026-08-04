# Training Terminal

Desktop GUI (PySide6) for training and running point-cloud semantic-segmentation
models: bring a folder of point clouds, build a dataset, train a model, run
inference. Three pages: **Datasets → Train → Inference**.

Two execution backends, switched in the sidebar: **Local (pixi)** and
**Modal (cloud)**. Both run the same `scripts/local/` trainers (the Modal
shells subprocess them on a cloud GPU; see `scripts/modal/README.md`).

## Run it

```powershell
cd trainer_gui
pixi install     # one-time
pixi run gui     # launch
```

pip alternative (Python 3.10+): `pip install -e .` then `trainer-gui`.

## How it runs training/inference

The GUI never trains in-process. Locally it runs each trainer inside a
per-model pixi environment (`envs/pixi.toml`, locked in `envs/pixi.lock` -
one env per backbone, CUDA/torch + pinned model sources as `trainer-src-*`
conda packages from a public prefix.dev channel). Host dirs pass via env vars
(the `train_common` contract; unset = the fixed container paths Modal uses):

| Env var | Holds |
|---|---|
| `TT_DATASETS_ROOT` | staging root with the canonical datasets |
| `TT_OUTPUTS_ROOT` | `runs/<id>/` weights + training artifacts |
| `TT_DATASET_DIR` / `TT_INFER_DIR` / `TT_PRED_DIR` | per-run overrides |
| `TT_TRAIN_STRIDE` | train-tile stride as a fraction of chunk_xy (default 0.75) |

No pixi on PATH (or not a CUDA host) → the GUI prints the exact command
instead of running it.

## Models

| Model | key | Local script |
|---|---|---|
| PTv3 | `ptv3` | `scripts/local/local_train_ptv3.py` |
| RandLA-Net | `randlanet` | `scripts/local/local_train_randlanet.py` |
| KPConvX-L | `kpconvx_cold` | `scripts/local/local_train_kpconvx_cold.py` |
| KPConv | `kpconv` | `scripts/local/local_train_kpconv.py` |
| Concerto (pretrained encoder) | `concerto` | `scripts/local/local_train_concerto.py` |
| Sonata (pretrained encoder) | `sonata` | `scripts/local/local_train_sonata.py` |
| Utonia (pretrained encoder) | `utonia` | `scripts/local/local_train_utonia.py` |

`concerto`/`sonata`/`utonia` fine-tune Pointcept self-supervised encoders
(shared trainer; the latter two are thin wrappers). Their HuggingFace weights
are **CC-BY-NC 4.0 (non-commercial)**.

**HeightAboveGround** is a feature channel (`feat_hag`), not a model variant:
bake it at dataset build or inference staging (ground source: labeled class /
CSF / SMRF / Z-min proxy; grid, NN, or Delaunay interpolation). Runs trained
with it record it in `run.json "features"`; inference recomputes it from the
input clouds.

## The three pages

- **Datasets** builds `<staging>/<name>/` - scan label values into named
  classes, optional HAG, one recorded train/val/test split
  (`dataset_meta.json` + `train|val|test/<scene>.npz`; trainers read the
  folders verbatim and never re-split).
- **Train** picks dataset + model, pre-fills per-model parameters, launches
  the pixi run and streams logs. A finished run dir holds the weights,
  `run.json` (the single manifest inference needs - config, class names,
  final test metrics, finished flag), and `metrics.csv` (per-epoch train +
  val rows).
- **Inference** takes weights (a training run, an installed
  `trainer-weights-*` package, or a bare `.pth`) and an input file/folder.
  Each input converts to `<stem>_input.npz` in the job dir and predictions
  merge into that same file - one npz per input plus one `job.json`
  manifest. Optional cleanup passes (KNN smoothing, sieve, geometry rules),
  ALPINE instance clustering, and export to las/laz/ply/txt/csv. An
  **Ensemble** group runs 2-3 models over one staged job and majority-votes
  the result (`scripts/local/ensemble_vote.py` works standalone too).

## Repo layout

```
trainer_gui/     the PySide6 app - pages/, local_cli.py, ...
scripts/local/   the real trainers/inferencers (plain argparse; run standalone)
scripts/modal/   thin cloud shells (see its README.md)
scripts/helper/  train_common.py (shared training/manifest logic)
envs/            pixi.toml + pixi.lock - one training env per backbone
```
