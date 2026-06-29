# Density Domain-Generalization — implementation spec

Goal: one trained model that segments point clouds at densities it was **not** trained on
(denser **and** sparser), as close to true cross-density generalization as the information
allows. Engineering substrate only — no dataset/brand specifics.

## The theory in one screen

Occupancy `o = rho * g^2` (mean points per voxel cell of size `g`, areal density `rho`).
A voxel subsample keeps <=1 point/cell, so it **caps density at `1/g^2`**.

- `o >= 1`: cells filled, the subsampled cloud is ~a function of the *surface*, not `rho`
  -> the backbone is density-invariant **for free**.
- `o < 1`: Poisson holes appear, invariance breaks. **This is the entire gap.**
- One-way valve: you can thin a dense cloud **down** to `g0` (safe), never invent points a
  sparse cloud never captured (ill-posed). So design around the **sparsest** density served.

The cliff has one name: a SUM-type aggregation makes `E[activation] ∝ n ∝ o` once `o<1`, which
de-calibrates the frozen BatchNorm stats; error grows with `|o_test − o_train|`.

`1/g^2` per backbone: KPConvX g=2.0 -> 0.25 pt/m^2 ; PTv3 g=0.5 -> 4 ; RandLA g=0.3 -> 11.
So KPConvX stays in the invariant band on sparse data longest; PTv3/RandLA fall below `o=1`
first and must coarsen `g` (trading resolution) to serve very sparse clouds.

## Decisions (all flag-gated; defaults reproduce current behaviour exactly)

| ID | Decision | Mechanism → why inference improves | Where | Status |
|----|----------|-----------------------------------|-------|--------|
| D0 | canonical `g0 = 1/sqrt(rho_min)`; subsample train+infer to it | puts the sparsest served cloud at `o=1`, inside the invariant band | config `GRID*`; inference subsample | **partly built-in** (KPConvX tiles at GRID; PTv3/RandLA voxelize) |
| D0b | inference: thin dense cloud to `g0` | removes the only shift on the dense side, exactly (`o->1`) | infer path | **built-in for KPConvX** (`_predict_points` grid_subsample); wire for PTv3/RandLA if raw>grid |
| D0c | keep metric coords, float64-before-center | avoids a self-made density/scale confound | already done (origin offset, float64 load) | **already present** |
| D1 | density/grid jitter augmentation | widens `p_train(o)` so `o_test` lands in the calibrated band — the bulk lever, only one that helps sparse | batch assembly (train) | **WIRE** `density.effective_grid` |
| D2a | KPConvX sum→mean / inverse-density aggregation | stops `E[activation] ∝ o` at source; removes the cliff | `cfg.model.kp_aggregation` (line 619) | **stage** (config-reachable; retrain+GPU) |
| D2b | AdaBN: recompute BN stats on target at inference | re-aligns frozen source stats to target `o`; label-free, exact for moments 1–2 | infer path | **WIRE** `density.adabn_recalibrate` |
| D2c | BN→GroupNorm in magnitude-sensitive blocks | per-sample norm has no cross-sample stat to be wrong | `cfg.model.norm` (line 625) / lib | **stage** (config/lib; retrain) |
| D2d | RandLA: normalize LocSE rel-coords by mean nbr dist | fixes fixed-k geometry leak (`r_k ∝ rho^-1/2`) | `/opt/randlanet` LocSE | **stage** (library patch; retrain) |
| D3a | real HAG from robust DTM | per-point, density-invariant by construction; strongest single input | `*_hag.py` variants | **present** (audit KPConvX_hag adds not replaces) |
| D3b | explicit `log d_k` input (+ optional FiLM) | net learns density-*conditional* boundary, not entangled; answers D2b's residual | feature assembly + `INPUT_CHANNELS` | **WIRE input** `density.local_density_logdk`; FiLM = stage(lib) |
| D4 | density-consistency loss `f(x)≈f(decimate(x))` | flattens the feature manifold along density at the same points; uses unlabeled pts | train step | **stage** (in-script, biggest; separate BN per branch) |
| D5 | density-TTA: average softmax over resamplings | decorrelated density views -> lower posterior variance -> fewer flipped boundary pts | infer path | **WIRE** |

## Shared flags (add to each script's config block, defaults = current behaviour)

```python
# --- density domain-generalization (density.py) ---
DG_DENSITY_AUG   = False   # D1: per-tile effective-grid jitter during training
DG_COARSEN_MAX   = 2.5     # = 1/(g0*sqrt(rho_min)); sweep output density down by this factor
DG_P_NATIVE      = 0.5     # P(tile kept at native grid) — the full-occupancy anchor
DG_LOGDK_FEAT    = False   # D3b: append log k-th-NN distance as an input channel (needs +1 in-dim, retrain)
DG_LOGDK_K       = 8
DG_INFER_ADABN   = False   # D2b: recompute BN stats on the target tiles before predicting
DG_INFER_TTA     = 0       # D5: number of extra density/scale views to average (0 = off)
```

All four "WIRE" rows are reachable in the local training scripts and need no library edits.
`density.py` (helper dir) holds the verified primitives; `python density.py` runs its self-checks.

## Wiring map (file:line landmarks)

KPConvX `local_train_kpconvx_cold.py`: grid_subsample 244 · build_feat/INPUT_CHANNELS 102 ·
model build 601-648 · train forward 1143-1146 · `_predict_points` 871-902 · infer block 910-967.
PTv3 `local_train_ptv3.py`: voxel dedup 918 · `augment_xyz` 863 · build_model 423-455 ·
train forward 1139-1156 · `to_ptv3_batch` 884-937 · `_predict_scene` 481-520.
RandLA `local_train_randlanet.py`: grid_subsample 329 · build_net 181-188 · train forward
1010-1021 · `tf_map` 469-504 · `_predict_scene` 597-663.
`*_hag.py` variants mirror their base; same edits, plus the `hag` companion array must be
sliced by the same index whenever points are dropped/subsampled.

## Build & validation order

1. **Free / label-free first (no retrain):** D2b AdaBN + D5 TTA in the infer path. Run on a
   different-density cloud, compare to baseline — this *measures how much density hurts today*.
2. **Bulk retrain lever:** D1 jitter (+ optionally D3b input). Retrain, compare on the same
   held-out different-density cloud (NOT a same-density val split — that moves opposite).
3. **Deeper retrains, isolated ablations:** D2a/D2c (config), D4 (consistency), D2d (RandLA lib).
4. Each as an isolated A/B, never stacked, so each gain is attributable.

Hard ceiling: nothing recovers detail absent from a natively-sparse `o<1` cloud. Dense side is
fully solvable (canonicalize); sparse side is bounded by the `e^-o` empty-cell residual — train
tolerance (D1) and average variance (D5), do not expect to canonicalize it away.

## Status — wired 2026-06-28 (all flag-gated, defaults reproduce prior behaviour)

`density.py` self-checks pass (`pixi run python scripts/helper/density.py`); all 6 scripts
`py_compile`-clean. Verified by compile + helper unit self-checks only — NOT yet run on GPU/data.

| Change | kpconvx | kpconvx_hag | ptv3 | ptv3_hag | randla | randla_hag |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| config flags + `import density as dg` | ok | ok | ok | ok | ok | ok |
| **D1** density/grid jitter (train) | ok | ok (hag sliced) | ok | ok (hag via uniq) | ok | ok (hag via sel) |
| **D5** density-TTA (infer) | ok | ok | ok | ok | ok | ok |
| **D2b** AdaBN (infer) | ok | ok (hag proxy) | n/a¹ | n/a¹ | ok | ok (hag proxy) |
| **D3b** log d_k input channel | ok | ok | ok | ok | ok | ok |
| **D2a/D2c** aggregation/norm flags | ok | ok | — | — | — | — |

The hag AdaBN target-batch builders carry a z-min HAG proxy in the feature (BN-stat
estimation doesn't need exact HAG). ¹ PTv3 BN is pooling/stem only, so BN-TTA barely helps.
**D3b** (`DG_LOGDK_FEAT`) appends `log` of the k-th-NN distance as an input channel and bumps the
model input dim by 1 — retrain-only (old weights won't load); KPConvX feeds it through the single
`build_feat`, PTv3/RandLA at every inline feature site (train/eval/infer/AdaBN). **D2a/D2c**
(`KP_AGGREGATION`/`KP_NORM`) surface the KPConvX `cfg` knobs as module constants — change + retrain to A/B.

**Still staged (NOT wired, with the reason each is held):**
- **D4 consistency loss** — in-script (dual-branch `f(x)≈f(decimate(x))`, stop-grad logit-KL on the
  shared index subset, separate BN per branch). Held because correct per-point index alignment
  through each backbone's internal pyramid/voxel re-subsample can't be verified without a GPU run,
  and there's no runnable check possible here. Recipe is fully specified (Tier-4 above); it's a
  short, well-scoped job once a GPU is available — just not safe to ship blind.
- **D2d RandLA LocSE rel-coord normalization** and **D3b FiLM modulation** — live inside the external
  pip packages (`/opt/kpconvx`, `/opt`, `/opt/randlanet`), not in this repo. Need the library vendored
  (or a monkey-patch shim) before they can be wired.
- **D0/D0b explicit knob** — all three backbones already grid-canonicalize density at inference, so a
  separate knob is a no-op for the current grids; expose only if a deploy needs a g0 ≠ the model grid.

**How to turn on — three ways (all the same `DG_*` knobs).** The GUI is split by lifecycle:
- **Train-time** (`density_aug`, `logdk`) — bake into the weights:
1. **GUI**: Datasets page -> select a saved dataset -> **"Density generalization (advanced)"** panel.
   It reads the dataset's stored density, takes your **target inference density**, and **Recommend**
   fills the train toggles (`analysis.dg_recommend`); **Save to dataset** persists it (`appstate`
   `dg_config`). The Train page injects them as `DG_*` env at launch (local docker `-e`). At the end of
   training `write_run_manifest` reads the same env and records a **`dg` block in `run.json`** (logdk +
   k), so the model is self-describing.
- **Inference-time** (`adabn`, `tta`) — label-free, no retrain, applies to any model:
1. **GUI**: Inference page -> Input box -> **AdaBN** / **Density TTA** toggles. The launch
   (`_infer_dg_env`) also re-injects `DG_LOGDK_FEAT`/`_K` **recovered from the run.json `dg` block**, so
   a logdk-trained model rebuilds at the right width and recomputes the channel automatically. (A bare
   `.pth` with no run.json can't recover logdk — pick the run.json, or the load fails loudly.)
- Either lifecycle, **directly**:
2. **Env var**: `DG_DENSITY_AUG=1 DG_INFER_ADABN=1 python local_train_kpconvx_cold.py ...`.
   Each script reads `DG_*` in its config block via `dg.env_bool/float/int/str(name, globals()[name])`.
3. **Edit the constant** in the script's config block.

Cheapest first read on how much density hurts today — no retrain: `DG_INFER_ADABN=1` and/or
`DG_INFER_TTA=3` on a different-density cloud. Bulk lever (retrain): `DG_DENSITY_AUG=1`, tune
`DG_COARSEN_MAX` to `1/(g0*sqrt(rho_min))` (the GUI's Recommend computes `sqrt(train/infer)` for you).

**Plumbing:** `scripts/helper/density.py` `env_*` helpers + per-script env-shadow block;
`train_common.py` `_dg_block`/`write_run_manifest` (records the `dg` block in run.json) + `infer_meta`
(reads it back); `trainer_gui/analysis.py` `dg_recommend`/`dg_config_to_env` (train-time only);
`appstate.get/set_dg_config`; `pages/datasets_page.py` `_density_gen_box` (train-time toggles);
`pages/train_page.py` injection; `pages/infer_page.py` `_infer_dg_env` (AdaBN/TTA toggles + logdk
recovery); `local_cli.run_script(env=)` (used by BOTH train and infer launches).
Modal note: env lands client-side — the modal shell still needs to forward `DG_*` into the
container subprocess (local docker already does via `-e`). Modal is frozen (`scripts/modal/DEPRECATED.md`).
