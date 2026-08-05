# Training Terminal

A desktop app for point-cloud semantic segmentation: turn your classified
point clouds into a dataset, train a model on it, and classify new clouds
with the result — three tabs, no code.

Runs on your own NVIDIA GPU, or on [Modal](https://modal.com) cloud GPUs with
the switch in the top bar.

## Install & run

```
cd trainer_gui
pixi install
pixi run gui
```

[pixi](https://pixi.sh) handles everything on Windows and Linux. On first
launch you pick a workspace folder — all datasets, training runs, and
inference results live there.

## 1 · Datasets

Point at classified clouds (`las/laz`, `ply`, `txt/csv/xyz/pts`, `pcd`,
`npy/npz`). The app finds your label field, lists every class value it sees,
and lets you rename, merge, or exclude classes. Pick a validation/test
fraction and Build — the split is carved once, recorded, and never changes
under you. Coordinates are reprojected to meters automatically and restored
on export.

Optional per-point channels you can bake in:

- **Height Above Ground** — ground found from your ground class, or detected
  with [CSF (cloth simulation)](https://github.com/jianboqi/CSF) or
  [SMRF](https://pdal.io/en/latest/stages/filters.smrf.html) via
  [PDAL](https://pdal.io), or a fast lowest-point proxy; interpolated by
  grid, nearest-neighbor, or Delaunay.
- **Geometric features** (planarity, linearity, scattering, verticality…) —
  computed with [pgeof](https://github.com/drprojects/point_geometric_features)
  at the optimal neighborhood size.

**Duplicate & edit** clones an existing dataset's whole setup into the form
so you can add a channel and rebuild without redoing your choices.

## 2 · Train

Pick the dataset and a model; sensible per-model defaults are pre-filled and
everything is editable. Logs stream live; you can stop gracefully at the
nearest checkpoint. A finished run is one folder holding the weights, its
manifest, and its metric history — that folder is all inference needs.

| Model | What it is |
|---|---|
| [PTv3](https://github.com/Pointcept/PointTransformerV3) | Point Transformer V3 ([paper](https://arxiv.org/abs/2312.10035)) |
| [RandLA-Net](https://github.com/QingyongHu/RandLA-Net) | lightweight large-scale net ([paper](https://arxiv.org/abs/1911.11236)) |
| [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) | kernel point convolution ([paper](https://arxiv.org/abs/1904.08889)) |
| [KPConvX-L](https://github.com/apple/ml-kpconvx) | modernized KPConv with kernel attention |
| Concerto / [Sonata](https://github.com/facebookresearch/sonata) / Utonia | fine-tuned self-supervised [Pointcept](https://github.com/Pointcept/Pointcept)-family encoders — **weights are CC-BY-NC 4.0 (non-commercial)** |

Optional training extras: class balancing, focal/Lovász loss options, rare-class
oversampling, and density-generalization augmentation for inferring at a
different point density than you trained on.

## 3 · Inference

Pick weights (a run, an installed model package, or a bare `.pth`), pick an
input file or folder, Run. Everything for a job lands in one folder: one
`.npz` per input (the converted channels *and* the predictions in the same
file — open it in any viewer to explore both) plus a `job.json` record and
your exported result.

- **Export**: classified `las/laz` (source coordinates and every original
  dimension carried over), or `txt/csv/ply`; a confidence threshold can send
  unsure points to *unclassified*; optional per-point uncertainty fields.
- **Accuracy boosters**: test-time augmentation views, overlapped tile
  voting, and test-time adaptation (AdaBN, [paper](https://arxiv.org/abs/1603.04779),
  and APCoTTA) on the BatchNorm architectures (RandLA-Net / KPConv / KPConvX).
- **Cleanup passes** (re-runnable any time without re-inference): KNN
  probability smoothing (in the spirit of
  [RangeNet++](https://github.com/PRBonn/rangenet_lib)), small-island
  removal, and confidence-gated height/planarity rules for the classic
  ground/vegetation/building confusions.
- **Instances**: cluster chosen classes into individual objects with
  [ALPINE](https://github.com/valeoai/Alpine) ([paper](https://arxiv.org/abs/2503.13203)) —
  training-free, exported as an `instance_id` field.
- **Ensemble**: run 2–3 trained models over one input and majority-vote —
  typically ~+2 mIoU over the best single model.

## Licensing

MIT for the app. Models keep their upstream licenses (see the table);
the Pointcept-encoder weights are non-commercial.
