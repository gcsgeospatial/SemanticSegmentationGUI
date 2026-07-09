"""
Local training script for KPConvX-L on a canonical trainer_gui --dataset
(3-folder train/val/test), COLD-START, native-features variant.

  Features  : 4 = [1, intensity, return_number, height], where height is
              tile-relative (z - tile_min_z). No geometric-feature
              computation — uses the attributes the cloud carries.
  --hag     : swaps the 4th channel to real PDAL HeightAboveGround, which the
              dataset (or the Inference page's HAG box) must supply.
              Same 4 channels / param count -> a clean A/B of tile-relative
              height vs true height above ground. Replaces the old copy-paste
              local_train_kpconvx_cold_hag.py twin (now a thin wrapper
              forcing --hag).

Random init (no warm-start). Geometry: 2.0 m grid, conv radius 2.5 cells, BN
momentum 0.02, with KPConvX's own training recipe (experiments/S3DIS):
  - AdamW (weight_decay 0.05) + 1-cycle LR (1e-4 -> 5e-3 raise over 30 epochs,
    5-epoch plateau, /10 per 120 epochs) + label smoothing 0.2
  - packed batches: PACK_N tiles concatenated per forward (lengths-aware
    pyramid), x ACCUM grad accumulation -> effective batch 8 clouds/step
  - 100 m tiles (input region ~ 50x the 2.0 m grid) at stride 50 so train tiles
    overlap for coverage and val/test tiles overlap for the voted eval
  - class-weighted smoothed CE (+ Lovasz) + mild rare-class tile oversampling
  - held-out val pass every VAL_EVERY epochs (eval mode, no weight updates),
    appended to val_metrics.csv so convergence can be watched over time
  - final val/test eval votes over overlapping tiles (stride 50): per-voxel
    softmax sums across up to 4 covering tiles

Checkpoints (model+optimizer) every CHECKPOINT_GAP epochs; set RESUME=True to
continue the most recent run.

Usage:
    python local_train_kpconvx_cold.py --dataset <name>
    python local_train_kpconvx_cold.py --dataset <name> --hag   # real-HAG 4th channel
    python local_train_kpconvx_cold.py --dataset <name> --mode eval \
        --weights runs/<id>/final_model.pth   # re-score with the voted eval
    python local_train_kpconvx_cold.py --mode infer --infer-input <job> \
        --weights runs/<id>/final_model.pth   # label staged scenes
"""

import os
from typing import Optional


def gpu_name() -> str:
    """Real CUDA device name for logs/metadata (replaces the old fixed cloud GPU_TYPE)."""
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


# ============================================================================
# Configuration
# ============================================================================
APP_NAME      = "kpconvx-cold"
FEATURE_MODE  = "native"     # [1, intensity, return_number, height]
N_EPOCHS      = 100          # extended from 150 (val mIoU still rising); the
                             # 1-cycle lr keeps decaying smoothly past 150
EPOCH_STEPS   = 300          # optimizer steps / epoch (KPConvX S3DIS: 300)
PACK_N        = 4            # tiles packed per forward (KPConvX batch_size = 4)
ACCUM         = 2            # grad-accumulated forwards / step -> effective batch 8 tiles
CHECKPOINT_GAP = 10          # checkpoint frequency (epochs); saves model + optimizer
VAL_EVERY     = 10           # held-out val pass every N epochs (no weight updates)
TIMEOUT_HOURS = 24

# Resume: when True, continue the most recent AdamW-recipe run in the outputs
# dir (same run dir, appended metrics) instead of starting fresh. On, to
# extend the 150-epoch run to 200. Set back to False for a fresh run.
RESUME = False

# Class-balanced loss + rare-class oversampling. This deliberately diverges from
# train_LAS.py's segloss_balance='none' to stop the rare Water/Bridge classes
# from collapsing to 0 IoU under plain cross-entropy.
CLASS_WEIGHTING = True
WEIGHT_BETA     = 0.5        # 0 = uniform, 1 = full inverse-freq; 0.5 = inverse SQRT
                            # frequency (w = 1/sqrt(freq)) — the RandLA-Net /
                            # SemanticKITTI standard: sub-linear, so rare classes
                            # are boosted without raw 1/freq's instability.
WEIGHT_CAP      = 5.0        # clamp each weight to [1/CAP, CAP] after mean-normalising
# Lovász-Softmax: tractable surrogate that optimizes mIoU (Jaccard) directly and
# weights every class equally, countering CE's majority bias on rare classes.
# Total loss = <pointwise> + LOVASZ_WEIGHT * lovasz_softmax.
# Set 0.0 to disable.
LOVASZ_WEIGHT   = 1.0
# Focal loss (Lin et al. 2017): when USE_FOCAL, the pointwise term is
# alpha-balanced focal loss instead of weighted (label-smoothed) CE. alpha
# reuses the inverse-sqrt class weights; (1-p_t)^gamma down-weights easy points
# so hard + rare points dominate. gamma=0 == weighted CE. NOTE: the focal path
# does NOT apply LABEL_SMOOTH (focal's modulation is the regulariser); set
# USE_FOCAL=False to fall back to the smoothed weighted-CE term.
USE_FOCAL       = False
FOCAL_GAMMA     = 2.0
RARE_OVERSAMPLE = True
RARE_CLASSES    = None       # explicit class indices, or None -> auto from train
                             # frequency (same rule as the PTv3 script), so any
                             # dataset's minority classes are picked up unchanged
RARE_FREQ_FRAC  = 0.5        # auto-rare threshold: freq < frac x median present freq
# 0.5 + cap 10 overcooked (Building 0.74->0.61, rare tiles memorised): dialed back.
RARE_TILE_PROB  = 0.25       # P(draw the next train tile from a rare-class tile)

INPUT_CHANNELS = 4           # [1, intensity, return_number, height]

# Geometry — matches train_LAS.py LASConfig.
GRID          = 2.0          # first_subsampling_dl: layer-0 voxel grid (m)
KP_RADIUS     = 2.5          # conv_radius (grid cells); first_radius = GRID*KP_RADIUS
RADIUS_SCALING = 2.0         # standard KPConv per-layer doubling

# KPConvX guidance: input region ~ 50x the subsampling grid (S3DIS: 2.1 m at
# 0.04 m). At GRID=2.0 that is ~100 m. Stride 50 everywhere: train tiles overlap
# for coverage, and val/test tiles overlap so the final eval can vote (each
# point is covered by up to 4 tiles).
CHUNK_XY      = 100.0
STRIDE        = 50.0

# Augmentation (train_LAS.py LASConfig).
AUG_SCALE_MIN = 0.9
AUG_SCALE_MAX = 1.1
AUG_SYMMETRY_X = True        # augment_symmetries = [True, False, False]
AUG_NOISE     = 0.05         # augment_noise
AUG_COLOR     = 0.8          # augment_color: P(keep features) = 0.8

# --- density domain-generalization (scripts/helper/density.py; see DENSITY_DG.md) ---
# All default to current behaviour. o = rho*g^2; the model is density-invariant for
# o>=1 and breaks for o<1, so D1 trains across the o-range and D2b/D5 patch inference.
DG_DENSITY_AUG = False   # D1: per-tile coarsen the loaded tile to a jittered grid (train only)
DG_COARSEN_MAX = 2.5     # = 1/(GRID*sqrt(rho_min)); how far down in density to sweep
DG_P_NATIVE    = 0.5     # P(tile kept at native GRID) — the full-occupancy anchor
DG_INFER_ADABN = False   # D2b: recompute BN stats on the target tiles before predicting
DG_INFER_TTA   = 0       # D5: # extra density(scale) views to average at inference (0=off)
# D3b: explicit local-density input channel (log k-th-NN distance) so the net learns a
# density-CONDITIONAL boundary instead of one entangled with density. Bumps input_channels
# by 1 -> retrain (old weights won't load). Pair with DG_DENSITY_AUG so rho actually varies.
# (FiLM modulation on this scalar is the stronger form but lives in the KPConvX library.)
DG_LOGDK_FEAT  = False
DG_LOGDK_K     = 8
# D2a/D2c (staged): KPConvX aggregation + norm knobs. Defaults = current behaviour; valid
# alternatives depend on the KPConvX lib (e.g. norm "group" for GroupNorm). Change+retrain to A/B.
KP_AGGREGATION = "nearest"   # D2a: neighbour aggregation mode
KP_NORM        = "batch"     # D2c: "batch" | (lib-dependent, e.g. "group")

# Optimizer — KPConvX's own recipe (experiments/S3DIS/train_S3DIS.py).
WEIGHT_DECAY  = 0.05
CYC_LR0       = 1e-4         # start (and floor) lr of the 1-cycle schedule
CYC_LR1       = 5e-3         # peak lr
CYC_RAISE     = 30           # epochs raising lr0 -> lr1 (exponential)
CYC_PLATEAU   = 5            # epochs held at peak
CYC_DECREASE10 = 120         # epochs per /10 decay after the plateau
LABEL_SMOOTH  = 0.2          # KPConvX smooth_labels
GRAD_CLIP     = 100.0
BN_MOMENTUM   = 0.02

def train_kpconvx(dataset: Optional[str] = None, mode: str = "train",
                  weights: Optional[str] = None,
                  infer_input: Optional[str] = None, grid: Optional[float] = None,
                  chunk_xy: Optional[float] = None, epochs: Optional[int] = None,
                  batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                  hag: bool = False):
    import os, sys, time, json, csv, glob, traceback
    from datetime import datetime
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helper"))
    import density as dg
    import train_common as tc
    # DG flags: env-overridable (GUI "Density generalization" panel / DG_*=1 in the shell).
    # globals()[...] reads the module default; the local shadow is what the nested closures use.
    DG_DENSITY_AUG = dg.env_bool("DG_DENSITY_AUG", globals()["DG_DENSITY_AUG"])
    DG_COARSEN_MAX = dg.env_float("DG_COARSEN_MAX", globals()["DG_COARSEN_MAX"])
    DG_P_NATIVE    = dg.env_float("DG_P_NATIVE", globals()["DG_P_NATIVE"])
    DG_LOGDK_FEAT  = dg.env_bool("DG_LOGDK_FEAT", globals()["DG_LOGDK_FEAT"])
    DG_LOGDK_K     = dg.env_int("DG_LOGDK_K", globals()["DG_LOGDK_K"])
    DG_INFER_ADABN = dg.env_bool("DG_INFER_ADABN", globals()["DG_INFER_ADABN"])
    DG_INFER_TTA   = dg.env_int("DG_INFER_TTA", globals()["DG_INFER_TTA"])
    # Loss / class-balance overrides (GUI "Loss & class balance" panel -> LOSS_*/
    # RARE_* env; mirrors the DG env pattern). Unset env -> the script constants.
    USE_FOCAL       = dg.env_bool("LOSS_FOCAL", globals()["USE_FOCAL"])
    FOCAL_GAMMA     = dg.env_float("LOSS_FOCAL_GAMMA", globals()["FOCAL_GAMMA"])
    CLASS_WEIGHTING = dg.env_bool("LOSS_CLASS_WEIGHTING", globals()["CLASS_WEIGHTING"])
    WEIGHT_BETA     = dg.env_float("LOSS_WEIGHT_BETA", globals()["WEIGHT_BETA"])
    RARE_OVERSAMPLE = dg.env_bool("RARE_OVERSAMPLE", globals()["RARE_OVERSAMPLE"])
    KP_AGGREGATION = dg.env_str("KP_AGGREGATION", globals()["KP_AGGREGATION"])
    KP_NORM        = dg.env_str("KP_NORM", globals()["KP_NORM"])

    sys.path.insert(0, "/opt/kpconvx")
    EVAL_ONLY = (mode == "eval")
    INFER     = (mode == "infer")   # arbitrary-folder inference (trainer_gui)

    # --dataset (a canonical 3-folder trainer_gui dataset) is required for
    # training and eval. Folder inference (--mode infer) is the only dataset-free
    # path: it labels staged scenes from trained weights.
    if dataset is None and not INFER:
        raise ValueError("--dataset is required: pass a canonical trainer_gui "
                         "dataset name (train/val/test folders). The only "
                         "dataset-free path is --mode infer.")

    # --- resolve run config: CLI flags override the module defaults --------------
    # Geometry + schedule knobs the GUI sends as --grid / --chunk-xy / --epochs /
    # --batch / --steps-per-epoch. Assigned as LOCALS so every
    # nested helper (grid_subsample, the train loop, evaluate, …) picks them up
    # via closure. --batch maps to PACK_N (tiles packed per forward).
    GRID        = grid if grid is not None else globals()["GRID"]
    CHUNK_XY    = chunk_xy if chunk_xy is not None else globals()["CHUNK_XY"]
    STRIDE      = CHUNK_XY / 2.0
    N_EPOCHS    = epochs if epochs is not None else globals()["N_EPOCHS"]
    EPOCH_STEPS = steps_per_epoch if steps_per_epoch is not None else globals()["EPOCH_STEPS"]
    PACK_N      = batch if batch is not None else globals()["PACK_N"]
    HAG = bool(hag)   # --hag: local vars named `hag` below are per-point ARRAYS
    # --hag swaps the 4th feature channel (still 4 ch): shadow FEATURE_MODE so the
    # banner / run.json / resume filter all track the variant.
    FEATURE_MODE = "native_hag" if HAG else globals()["FEATURE_MODE"]

    # --dataset NAME selects a canonical trainer_gui dataset under /datasets
    # (bind-mounted); NUM_CLASSES / CLASS_NAMES / PREP_DIR are locals so the
    # whole function tracks the dataset's class layout.
    HAG_SOURCE = None
    ds_root = f"/datasets/{dataset}" if dataset else None
    if ds_root:
        meta_path = f"{ds_root}/dataset_meta.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{meta_path} not found — build the dataset "
                                    f"with the trainer_gui app first.")
        with open(meta_path) as f:
            ds_meta = json.load(f)
        NUM_CLASSES = int(ds_meta["num_classes"])
        CLASS_NAMES = list(ds_meta["class_names"])
        if HAG and not ds_meta.get("has_hag"):
            raise ValueError(
                f"--hag needs a dataset with a real HeightAboveGround channel, but "
                f"'{dataset}' has none (has_hag=false). Rebuild it with the Datasets "
                f"page 'Compute Height-Above-Ground' box, or train the plain KPConvX-L "
                f"(its 4th channel is the tile-relative height, which needs no HAG).")
        # --hag: the dataset's real HAG. (Legacy datasets recorded no method; the
        # PDAL nearest-neighbour filter was the only one back then.)
        HAG_SOURCE = (((ds_meta.get("source") or {}).get("hag_source") or "pdal_hag_nn")
                      if HAG else None)
        # --hag gets its own cache family (tiles carry a "hag" array).
        PREP_DIR = f"{ds_root}/prep/kpconvx_cold{'_hag' if HAG else ''}_grid{GRID:g}_c{int(CHUNK_XY)}"
    else:
        # Folder inference (--mode infer): no --dataset. Reproduce the TRAINED
        # geometry + class layout from the run's run.json (the self-contained
        # manifest beside the weights; legacy run_config.json is the fallback).
        # INFER never reads cached tiles, so PREP_DIR is an unused placeholder.
        NUM_CLASSES = 5
        CLASS_NAMES = [f"class {i}" for i in range(NUM_CLASSES)]
        PREP_DIR = "/outputs/_infer_unused"
        if INFER and weights:
            import os as _os, sys as _sys
            _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "helper"))
            from train_common import infer_meta
            meta = infer_meta(f"/outputs/{weights}")
            if meta:
                # M5: never load cross-variant weights — both variants are 4-ch, so
                # no shape check can catch it (run.json's hag_source tags them).
                if HAG and not meta.get("hag_source"):
                    raise ValueError(
                        "These weights are from a plain KPConvX-L run (no 'hag_source' in "
                        "run.json): its 4th input channel is the native tile-relative "
                        "height, not real HAG. Re-run inference with the plain "
                        "'KPConvX-L' backbone.")
                if not HAG and meta.get("hag_source"):
                    raise ValueError(
                        "These weights are from a KPConvX-L + HAG run (run.json has "
                        "'hag_source'): its 4th input channel is HeightAboveGround, not the "
                        "native height this script feeds. Re-run inference with the "
                        "'KPConvX-L + HAG' backbone.")
                HAG_SOURCE = meta.get("hag_source")   # record what the weights were trained on
                NUM_CLASSES = int(meta.get("num_classes") or NUM_CLASSES)
                CLASS_NAMES = list(meta.get("class_names") or
                                   [f"class {i}" for i in range(NUM_CLASSES)])
                if meta.get("grid") is not None: GRID = float(meta["grid"])
                if meta.get("chunk_xy") is not None: CHUNK_XY = float(meta["chunk_xy"])
                STRIDE = CHUNK_XY / 2.0

    # ------------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------------
    def grid_subsample(xyz, attrs, lab, voxel):
        """Voxel-grid subsample to `voxel` m: barycenter points, mean attrs,
        majority labels. Mirrors KPConv-PyTorch's grid_subsampling (the C++ op
        that produces the first_subsampling_dl=2.0 layer-0 cloud)."""
        keys = np.floor(xyz / voxel).astype(np.int64)
        inv = tc.voxel_unique(keys, return_inverse=True)[1]
        nv = int(inv.max()) + 1
        cnt = np.bincount(inv, minlength=nv).astype(np.float64)
        sx = np.zeros((nv, 3)); np.add.at(sx, inv, xyz); sx /= cnt[:, None]
        sa = None
        if attrs is not None:
            sa = np.zeros((nv, attrs.shape[1])); np.add.at(sa, inv, attrs); sa /= cnt[:, None]
        sl = np.full(nv, -1, np.int64)
        if lab is not None:
            oh = np.zeros((nv, NUM_CLASSES)); v = lab >= 0
            np.add.at(oh, (inv[v], lab[v]), 1)
            has = oh.sum(1) > 0
            sl[has] = oh[has].argmax(1)
        return (sx.astype(np.float32),
                (sa.astype(np.float32) if sa is not None else None),
                sl)

    def augment(xyz):
        """train_LAS.py augmentation_transform: vertical rotation, anisotropic
        scale in [0.9,1.1] with random x-flip, gaussian noise 0.05."""
        theta = np.random.rand() * 2 * np.pi
        cs, sn = np.cos(theta), np.sin(theta)
        R = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], np.float32)
        scale = np.random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX, 3).astype(np.float32)
        if AUG_SYMMETRY_X and np.random.rand() < 0.5:
            scale[0] *= -1.0
        out = (xyz @ R.T) * scale
        out += np.random.normal(0, AUG_NOISE, out.shape).astype(np.float32)
        return out.astype(np.float32)

    # ------------------------------------------------------------------------
    # Preprocessing -> 2.0 m subsampled .npz chunks (xyz + intensity/ret)
    # ------------------------------------------------------------------------
    def load_canonical(npz_path):
        """Canonical trainer_gui scene (.npz) -> (xyz, intensity, ret_num, lab).
        xyz is origin-offset (per-scene floor-min) before the float32 cast so
        projected (UTM) coords keep sub-meter precision; intensity is already
        GUI-normalized (the clip below just bounds it); no return-number channel
        (zeros); labels are already contiguous indices."""
        z = np.load(npz_path)
        xyz = (z["xyz"] - np.floor(z["xyz"].min(0))).astype(np.float32)
        if "intensity" in z.files:
            intensity = z["intensity"].astype(np.float32)
        elif "rgb" in z.files:
            intensity = z["rgb"].astype(np.float32).mean(1) / 255.0
        else:
            intensity = np.zeros(len(xyz), np.float32)
        ret_num = np.zeros(len(xyz), np.float32)
        lab = z["label"].astype(np.int32) if "label" in z.files \
            else np.full(len(xyz), -1, np.int32)
        return xyz, intensity, ret_num, lab

    def tile_and_save(name, pc_path, cls_path, out_dir, chunk_xy, stride):
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        try:                              # canonical .npz (label + optional hag embedded)
            xyz, intensity, ret_num, lab = load_canonical(pc_path)
            hag = None
            if HAG:   # --hag: the scene's real per-point HAG (startup guard vouches for it)
                z = np.load(pc_path)
                if "hag" not in z.files or len(z["hag"]) != len(xyz):
                    raise ValueError("missing its 'hag' channel, which --hag requires; "
                                     "rebuild the dataset with HAG enabled")
                hag = z["hag"].astype(np.float32)
        except Exception as e:
            print(f"  skip {pc_path}: {e}", flush=True); return None
        # Canonical intensity is already GUI-normalized, so pass it through (just clip).
        intensity_n = np.clip(intensity, 0.0, 2.0).astype(np.float32)
        print(f"    {name}: {len(xyz):,} pts loaded in {time.time()-t0:.1f}s, "
              + (f"HAG {hag.min():.1f}..{hag.max():.1f}m, " if HAG else "") + "tiling…",
              flush=True)
        mins = xyz[:, :2].min(0); maxs = xyz[:, :2].max(0)
        x0s = np.arange(mins[0], maxs[0], stride)
        y0s = np.arange(mins[1], maxs[1], stride)
        n_tiles = 0
        for x0 in x0s:
            for y0 in y0s:
                mask = (
                    (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + chunk_xy) &
                    (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + chunk_xy)
                )
                # Low thresholds on purpose: water absorbs LiDAR, so pure-water
                # tiles are sparse — the old 256-pt cut deleted them from training.
                if mask.sum() < 64:
                    continue
                attrs = np.stack([intensity_n[mask], ret_num[mask]]
                                 + ([hag[mask]] if HAG else []), axis=1).astype(np.float32)
                sx, sa, sl = grid_subsample(xyz[mask], attrs, lab[mask], GRID)
                if len(sx) < 32:
                    continue
                tile = dict(
                    xyz=sx.astype(np.float32),
                    intensity=sa[:, 0].astype(np.float32),
                    ret_num=sa[:, 1].astype(np.float32),
                )
                if HAG:   # --hag caches carry the subsampled hag column; without
                    tile["hag"] = sa[:, 2].astype(np.float32)   # --hag the tile layout is exactly pre-merge
                tile["lab"] = sl.astype(np.int32)
                np.savez_compressed(
                    os.path.join(out_dir, f"{name}_x{int(x0)}_y{int(y0)}.npz"), **tile)
                n_tiles += 1
        print(f"      -> {n_tiles} tiles", flush=True)
        return n_tiles

    def _split_scenes():
        # The dataset stage already materialized three whole-scene folders
        # (val = selection holdout, test = final report); read them verbatim and
        # never re-carve a split. The third tuple slot (cls_path) is always None
        # (labels are embedded in the canonical .npz).
        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        train_npz = sorted(glob.glob(f"{ds_root}/train/*.npz"))
        if not train_npz:
            raise FileNotFoundError(f"No canonical scenes under {ds_root}/train")
        val_npz  = sorted(glob.glob(f"{ds_root}/val/*.npz"))
        test_npz = sorted(glob.glob(f"{ds_root}/test/*.npz"))
        return ([(stem(p), p, None) for p in train_npz],
                [(stem(p), p, None) for p in val_npz],
                [(stem(p), p, None) for p in test_npz])

    def _cache_signature():
        # Everything that changes what a cached tile contains. A mismatch means
        # the cache is stale/leaky and must not be silently reused. The split is a
        # property of the DATASET (3 materialized folders), so the signature
        # records the dataset's split identity.
        sp = ds_meta.get("split", {})
        # --hag is a separate cache family (pipeline / recipe / hag_source below
        # match the old _hag script), so every existing cache stays valid.
        sig = {
            "format_version": 1,
            "pipeline": "kpconvx_cold_hag" if HAG else "kpconvx_cold",
            "grid": GRID,
            "chunk_xy": CHUNK_XY,
            "stride": STRIDE,
            "split_seed": sp.get("seed"),
            "split_mode": sp.get("mode"),
            "min_pts_mask": 64,
            "min_pts_sub": 32,
            "intensity_norm": "p95_clip2",
            "feature_recipe": ("bias,intensity,ret_num,hag" if HAG
                               else "bias,intensity,ret_num,height"),
        }
        if HAG:
            sig["hag_source"] = HAG_SOURCE
        return sig

    def _validate_cache(lists):
        """Refuse to reuse a cache built with different settings (grid, stride,
        split seed, label map, feature recipe …) instead of silently mixing
        incompatible data. Migrate a pre-validation cache by stamping .done
        markers for already-tiled scenes. Returns True if the signature file was
        newly written (so the caller can report fresh work)."""
        meta_path = f"{PREP_DIR}/cache_meta.json"
        cur = _cache_signature()
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                old = json.load(f)
            if old != cur:
                diffs = {k: [old.get(k), cur.get(k)]
                         for k in sorted(set(old) | set(cur)) if old.get(k) != cur.get(k)}
                raise RuntimeError(
                    f"Preprocess cache at {PREP_DIR} was built with DIFFERENT settings "
                    f"(mismatched: {diffs}). Reusing it would silently mix incompatible "
                    f"data. Point PREP_DIR at a fresh path or delete the stale cache.")
            return False
        legacy = False
        for split, items in lists:
            d = f"{PREP_DIR}/{split}"
            for name, _, _ in items:
                if glob.glob(f"{d}/{name}_x*.npz") and not os.path.exists(f"{d}/{name}.done"):
                    open(f"{d}/{name}.done", "w").close(); legacy = True
        with open(meta_path, "w") as f:
            json.dump(cur, f, indent=2)
        if legacy:
            print(f"  migrated existing cache at {PREP_DIR}: stamped .done markers + "
                  f"signature (assumed to match current settings).", flush=True)
        return True

    def ensure_prep():
        print(f"  ensuring preprocessed cache -> {PREP_DIR}", flush=True)
        for split in ("train", "val", "test"):
            os.makedirs(f"{PREP_DIR}/{split}", exist_ok=True)
        train_list, val_list, test_list = _split_scenes()
        any_new = [_validate_cache([("train", train_list), ("val", val_list),
                                    ("test", test_list)])]

        def already_tiled(out_dir, name):
            # A scene counts as cached only once its .done marker exists, so a run
            # interrupted mid-scene is re-tiled rather than silently left partial.
            return os.path.exists(f"{out_dir}/{name}.done")

        def tile_remaining(items, out_dir, chunk_xy, stride):
            for name, pc_path, cls_path in items:
                if already_tiled(out_dir, name):
                    continue
                n = tile_and_save(name, pc_path, cls_path, out_dir, chunk_xy, stride)
                if n is not None:          # None == load failed; leave unmarked to retry
                    open(f"{out_dir}/{name}.done", "w").close()
                any_new[0] = True

        print(f"  [train] {len(train_list)} scenes", flush=True)
        tile_remaining(train_list, f"{PREP_DIR}/train", CHUNK_XY, STRIDE)
        # val/test also tile at stride 50 so the final eval can vote over the
        # up-to-4 overlapping tiles covering each point.
        print(f"  [val] {len(val_list)} scenes", flush=True)
        tile_remaining(val_list, f"{PREP_DIR}/val", CHUNK_XY, STRIDE)
        print(f"  [test] {len(test_list)} scenes", flush=True)
        tile_remaining(test_list, f"{PREP_DIR}/test", CHUNK_XY, STRIDE)
        if any_new[0]:
            print("  preprocessing cache updated.", flush=True)
        else:
            print("  all scenes already cached.", flush=True)
        return train_list, val_list, test_list

    def make_run_dir():
        run_id = datetime.utcnow().strftime(
            "%Y%m%d_%H%M%S_kpconvx_cold_hag" if HAG
            else "%Y%m%d_%H%M%S_kpconvx_cold_native")
        run_dir = f"/outputs/runs/{run_id}"
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        return run_id, run_dir

    def find_latest_checkpoint():
        """Most recent run (run-ids are timestamps, so they sort) that has
        checkpoints AND was trained with this script's recipe — old SGD-recipe
        runs in the outputs dir are not valid warm starts. Returns
        (run_dir, ckpt_path, epoch) or None."""
        def _ep(p):
            return int(os.path.basename(p)[2:5])   # ep149.pth -> 149
        for rd in sorted(glob.glob("/outputs/runs/*"), reverse=True):
            ckpts = glob.glob(f"{rd}/checkpoints/ep*.pth")
            if not ckpts:
                continue
            opt_type = fmode = None
            for _cfg in (f"{rd}/run.json", f"{rd}/run_config.json"):   # legacy fallback
                try:
                    with open(_cfg) as f:
                        _rc = json.load(f)
                    opt_type = _rc.get("optimizer", {}).get("type")
                    fmode = _rc.get("feature_mode")
                    break
                except Exception:
                    continue
            if opt_type != "AdamW":
                print(f"  resume: skipping {os.path.basename(rd)} "
                      f"(recipe mismatch: optimizer={opt_type})", flush=True)
                continue
            # Both variants live in this script since the --hag merge: never resume
            # across them (run.json feature_mode: "native" vs "native_hag").
            if fmode != FEATURE_MODE:
                print(f"  resume: skipping {os.path.basename(rd)} "
                      f"(variant mismatch: feature_mode={fmode})", flush=True)
                continue
            latest = max(ckpts, key=_ep)
            return rd, latest, _ep(latest)
        return None

    print("=" * 70)
    print(f"  KPConvX-L  {dataset or 'infer'}  COLD/{FEATURE_MODE}  "
          f"({gpu_name()}, {N_EPOCHS} ep, {EPOCH_STEPS} steps, "
          f"pack {PACK_N} x accum {ACCUM})")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}  device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    train_list, val_list, test_list = ([], [], []) if INFER else ensure_prep()

    resume_info = find_latest_checkpoint() if (RESUME or EVAL_ONLY) else None
    if INFER:
        # Inference-only: fresh *_infer run dir, weights loaded after net build.
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_infer")
        # Predictions live next to the input scenes under /datasets/_infer (not the
        # per-model runs dir), so inference output is in one consistent place.
        run_dir = f"/datasets/_infer/{infer_input}"
        os.makedirs(f"{run_dir}/predictions", exist_ok=True)
        resume_ckpt, start_epoch = None, 0
    elif resume_info:
        run_dir, resume_ckpt, resume_epoch = resume_info
        run_id = os.path.basename(run_dir)
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        start_epoch = resume_epoch + 1
        verb = "EVAL-ONLY on" if EVAL_ONLY else "RESUMING"
        print(f"  {verb} {run_id} from {os.path.basename(resume_ckpt)}"
              + ("" if EVAL_ONLY else f" -> starting at epoch {start_epoch}/{N_EPOCHS}"),
              flush=True)
    else:
        if EVAL_ONLY:
            raise RuntimeError("eval mode: no AdamW-recipe run with checkpoints "
                               "found under /outputs")
        run_id, run_dir = make_run_dir()
        resume_ckpt, start_epoch = None, 0
    if resume_ckpt is None and not INFER:
        with open(f"{run_dir}/run.json", "w") as f:
            json.dump({
            "backbone": "KPConvX-L",
            "warm_start": False,
            "feature_mode": FEATURE_MODE,
            "input_channels": INPUT_CHANNELS,
            "dataset": dataset,
            **({"hag_source": HAG_SOURCE} if HAG else {}),   # --hag: tags the variant
            "n_epochs": N_EPOCHS, "epoch_steps": EPOCH_STEPS,
            "pack_n": PACK_N, "accum": ACCUM,
            "grid_m": GRID, "kp_radius": KP_RADIUS, "radius_scaling": RADIUS_SCALING,
            "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
            "chunk_xy_m": CHUNK_XY, "stride_m": STRIDE,
            "optimizer": {"type": "AdamW", "weight_decay": WEIGHT_DECAY,
                          "cyc_lr0": CYC_LR0, "cyc_lr1": CYC_LR1,
                          "cyc_raise": CYC_RAISE, "cyc_plateau": CYC_PLATEAU,
                          "cyc_decrease10": CYC_DECREASE10,
                          "label_smoothing": LABEL_SMOOTH,
                          "grad_clip": GRAD_CLIP, "bn_momentum": BN_MOMENTUM},
            "class_balance": {"weighting": CLASS_WEIGHTING, "beta": WEIGHT_BETA,
                              "weight_scheme": "inv_sqrt_freq" if WEIGHT_BETA == 0.5
                              else f"inv_freq^{WEIGHT_BETA}",
                              "cap": WEIGHT_CAP, "rare_tile_prob": RARE_TILE_PROB,
                              "rare_classes": RARE_CLASSES if RARE_CLASSES is not None
                              else "auto", "rare_freq_frac": RARE_FREQ_FRAC},
            "loss": {"pointwise": "focal" if USE_FOCAL else "weighted_ce",
                     "focal_gamma": FOCAL_GAMMA if USE_FOCAL else None,
                     "ce_weighted": CLASS_WEIGHTING,
                     "label_smoothing": 0.0 if USE_FOCAL else LABEL_SMOOTH,
                     "lovasz_softmax_weight": LOVASZ_WEIGHT},
            "train_scenes": [n for n, _, _ in train_list],
            "val_scenes":   [n for n, _, _ in val_list],
            "test_scenes":  [n for n, _, _ in test_list],
        }, f, indent=2)

    # ------------------------------------------------------------------------
    # Build model — KPConvX-L preset, train_LAS-matched geometry, random init.
    # ------------------------------------------------------------------------
    from utils.config import init_cfg
    from models.KPNext import KPNeXt

    cfg = init_cfg()
    cfg.data.name           = dataset or "infer"
    cfg.data.task           = "cloud_segmentation"
    cfg.data.num_classes    = NUM_CLASSES
    cfg.data.dim            = 3
    cfg.data.label_values   = list(range(NUM_CLASSES))
    cfg.data.ignored_labels = []
    cfg.data.pred_values    = list(range(NUM_CLASSES))

    cfg.model.layer_blocks  = (3, 3, 9, 12, 3)
    cfg.model.kp_mode       = "kpconvx"
    cfg.model.shell_sizes   = [1, 14, 42]
    cfg.model.kp_radius     = KP_RADIUS
    cfg.model.kp_sigma      = KP_RADIUS
    cfg.model.kp_influence  = "linear"
    cfg.model.kp_aggregation = KP_AGGREGATION
    cfg.model.kp_fixed      = "center"
    cfg.model.conv_groups   = -1
    cfg.model.share_kp      = True
    cfg.model.init_channels = 64
    cfg.model.channel_scaling = 1.41
    cfg.model.norm          = KP_NORM
    cfg.model.bn_momentum   = BN_MOMENTUM
    cfg.model.in_sub_size   = GRID
    cfg.model.in_sub_mode   = "grid"
    cfg.model.radius_scaling = RADIUS_SCALING
    cfg.model.grid_pool     = True
    cfg.model.decoder_layer = True
    cfg.model.upsample_n    = 3
    cfg.model.drop_path_rate = 0.3
    cfg.model.input_channels = INPUT_CHANNELS + (1 if DG_LOGDK_FEAT else 0)  # +log d_k (D3b)
    cfg.model.neighbor_limits = [12, 16, 20, 20, 20]
    cfg.model.use_strided_conv = True
    cfg.model.kpx_upcut     = False
    cfg.model.kpx_expansion = 8
    cfg.model.inv_groups    = 8
    cfg.model.inv_grp_norm  = True
    cfg.model.inv_act       = "sigmoid"
    cfg.model.first_inv_layer = 1
    cfg.model.kpinv_reduc   = 1

    cfg.data.init_sub_size  = GRID
    cfg.data.init_sub_mode  = "grid"

    net = KPNeXt(cfg).cuda()
    print(f"  Model params: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    print(f"  first_radius={net.first_radius:.2f} m  subsample_size={net.subsample_size:.2f} m  "
          f"num_layers={net.num_layers}", flush=True)

    optim = torch.optim.AdamW(net.parameters(), lr=CYC_LR0, weight_decay=WEIGHT_DECAY)
    # loss_fn is built after the tile scan below (needs class counts for weights).

    def lr_at(ep):
        """KPConvX 1-cycle: exponential raise lr0->lr1 over CYC_RAISE epochs,
        hold CYC_PLATEAU epochs, then /10 every CYC_DECREASE10 epochs. Set
        directly on the param groups each epoch (resume-friendly)."""
        if ep < CYC_RAISE:
            return CYC_LR0 * (CYC_LR1 / CYC_LR0) ** (ep / CYC_RAISE)
        if ep < CYC_RAISE + CYC_PLATEAU:
            return CYC_LR1
        return CYC_LR1 * 0.1 ** ((ep - CYC_RAISE - CYC_PLATEAU) / CYC_DECREASE10)

    # Resume from the latest checkpoint: restore weights (and optimizer state,
    # which checkpoints now include). LR is set per-epoch from lr_at().
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location="cuda", weights_only=True)
        net.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        print(f"  resumed weights{' + optimizer' if 'optim' in ckpt else ''} "
              f"at epoch {start_epoch}", flush=True)

    if EVAL_ONLY:
        # Prefer explicit --weights, then the run's final_model.pth, else keep
        # the checkpoint weights loaded above. start_epoch = N_EPOCHS empties
        # the training loop, so the script falls straight to the final eval.
        fm = f"/outputs/{weights}" if weights else f"{run_dir}/final_model.pth"
        if weights and not os.path.exists(fm):
            raise FileNotFoundError(f"--weights not found under /outputs: {fm}")
        if os.path.exists(fm):
            net.load_state_dict(torch.load(fm, map_location="cuda", weights_only=True)["model"])
            print(f"  EVAL-ONLY: loaded {fm}", flush=True)
        start_epoch = N_EPOCHS

    if INFER:
        fm = f"/outputs/{weights}" if weights else None
        if not fm or not os.path.exists(fm):
            raise FileNotFoundError(f"--mode infer requires --weights; not found: {fm}")
        try:   # weights_only=True: a hand-picked .pth can't run code on load
            ck = torch.load(fm, map_location="cuda", weights_only=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load weights '{fm}': {e}\n"
                f"  (loaded safely with weights_only=True — a full-model pickle or a "
                f"checkpoint from another script is rejected; re-export as a state_dict.)"
            ) from e
        net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        print(f"  [infer] loaded {weights} (final_model = best-val epoch "
              f"{ck.get('epoch', '?') if isinstance(ck, dict) else '?'})", flush=True)
        start_epoch = N_EPOCHS

    train_tiles = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    val_tiles   = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
    test_tiles  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
    print(f"  train_tiles: {len(train_tiles)}   val_tiles: {len(val_tiles)}   "
          f"test_tiles: {len(test_tiles)}", flush=True)
    if not train_tiles and not INFER:
        raise RuntimeError("No training tiles after preprocessing — check the dataset.")

    # --- class-balanced loss + rare-class oversampling ----------------------
    # One pass over the training tiles: count labels for inverse-frequency class
    # weights, then flag tiles containing any rare class so we can oversample
    # them. Rare = explicit RARE_CLASSES, else auto — present classes whose
    # count is below RARE_FREQ_FRAC x the median present-class count (data-
    # driven, so this works for any dataset's class set; was hardcoded [3, 4]).
    print("  scanning train tiles for class balance…", flush=True)
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    tile_counts = []                       # (path, per-class bincount) per tile
    for tp in train_tiles:
        lab = np.load(tp)["lab"]
        v = lab[lab >= 0]
        cnt = (np.bincount(v, minlength=NUM_CLASSES) if v.size
               else np.zeros(NUM_CLASSES, np.int64))
        class_counts += cnt
        tile_counts.append((tp, cnt))
    print(f"  class counts: {dict(zip(CLASS_NAMES, class_counts.tolist()))}", flush=True)
    if RARE_CLASSES is not None:
        rare_classes = list(RARE_CLASSES)
    else:
        present = class_counts[class_counts > 0]
        thresh = RARE_FREQ_FRAC * float(np.median(present)) if present.size else 0.0
        rare_classes = [c for c in range(NUM_CLASSES) if 0 < class_counts[c] < thresh]
    rare_tiles = ([tp for tp, cnt in tile_counts if cnt[rare_classes].any()]
                  if (RARE_OVERSAMPLE and rare_classes) else [])
    print(f"  rare classes: {[CLASS_NAMES[c] for c in rare_classes]}", flush=True)
    print(f"  rare-class tiles: {len(rare_tiles)} / {len(train_tiles)}", flush=True)

    if CLASS_WEIGHTING:
        # beta=0.5 -> w = 1/sqrt(freq) (inverse-sqrt-frequency): sub-linear, so
        # rare classes are boosted without raw 1/freq's instability. Mean-norm,
        # then capped.
        freq = class_counts / max(int(class_counts.sum()), 1)
        w = (1.0 / np.maximum(freq, 1e-6)) ** WEIGHT_BETA
        w = w / w.mean()                                    # keep loss scale ~1
        w = np.clip(w, 1.0 / WEIGHT_CAP, WEIGHT_CAP)
        class_weights = torch.tensor(w, dtype=torch.float32).cuda()
        print(f"  class weights: "
              f"{dict(zip(CLASS_NAMES, [round(float(x), 3) for x in w]))}", flush=True)
    else:
        class_weights = None
    # Torch-native equivalent of KPConvX's SmoothCrossEntropyLoss (correct
    # ignore_index=-1 and weight normalisation, same smoothing).
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1,
                                        label_smoothing=LABEL_SMOOTH)

    # --- Lovász-Softmax (Berman et al. 2018): differentiable mIoU/Jaccard
    # surrogate, per-class and equally weighted, added on top of CE. Pure-torch
    # flat implementation (logits are already (N, C), labels (N,)). -----------
    def _lovasz_grad(gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1].clone()
        return jaccard

    def lovasz_softmax_flat(probas, labels):
        # probas: (N, C) softmax probs; labels: (N,) in [0, C). Averaged over
        # classes present in the batch.
        if probas.numel() == 0:
            return probas.sum() * 0.0
        losses = []
        for c in torch.unique(labels):
            fg = (labels == c).float()
            class_pred = probas[:, int(c)]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
        if not losses:
            return probas.sum() * 0.0
        return torch.stack(losses).mean()

    # alpha-balanced multiclass focal loss; masks ignore_index=-1 internally.
    # alpha = class_weights (inverse-sqrt) when set. No label smoothing here.
    def focal_loss(logits, labels):
        valid = labels >= 0
        if not valid.any():
            return logits.sum() * 0.0
        lg, lb = logits[valid], labels[valid]
        logp = torch.log_softmax(lg, dim=1)
        logpt = logp.gather(1, lb.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** FOCAL_GAMMA) * logpt
        if class_weights is not None:
            loss = loss * class_weights[lb]
        return loss.mean()

    def seg_loss(logits, labels):
        # Guard all-ignored packs: CrossEntropyLoss(ignore_index=-1) returns NaN
        # when every label is ignored (0/0 reduction). Return a finite, zero-grad
        # value instead so backward() can never poison the weights with NaN.
        valid = labels >= 0
        if not valid.any():
            return logits.sum() * 0.0
        loss = focal_loss(logits, labels) if USE_FOCAL else loss_fn(logits, labels)
        if LOVASZ_WEIGHT > 0:
            probas = torch.softmax(logits[valid], dim=1)
            loss = loss + LOVASZ_WEIGHT * lovasz_softmax_flat(probas, labels[valid])
        return loss

    def pick_train_tile():
        if rare_tiles and np.random.rand() < RARE_TILE_PROB:
            return rare_tiles[np.random.randint(len(rare_tiles))]
        return train_tiles[np.random.randint(len(train_tiles))]

    def _hag_of(z, xyz):
        """Real per-point HeightAboveGround for a cached --hag tile. Call only when
        HAG; the plain recipe wants build_feat's native height instead."""
        if "hag" not in z.files:
            raise ValueError("A --hag tile is missing its 'hag' channel. Rebuild the "
                             "dataset with Height-Above-Ground enabled.")
        return z["hag"].astype(np.float32)

    def build_feat(xyz, intensity, ret_num, hag=None, drop=False):
        """[1, intensity, return_number, height].

        height is the passed per-point `hag` array (--hag: real HeightAboveGround).
        When hag is None this is the PLAIN recipe, whose 4th channel is genuinely
        z - min(z) over the tile — a trained-on native feature, not a HAG stand-in.
        With `drop`, zero the non-bias channels (train_LAS feature-drop,
        P=1-augment_color=0.2)."""
        bias = np.ones((len(xyz), 1), np.float32)
        if hag is None:
            hag = (xyz[:, 2] - xyz[:, 2].min()).astype(np.float32)   # native height
        attrs = np.concatenate([intensity[:, None], ret_num[:, None],
                                hag[:, None]], axis=1).astype(np.float32)
        if drop:
            attrs[:, 1:] = 0.0   # keep intensity (water's main cue); drop ret_num/height
        cols = [bias, attrs]
        if DG_LOGDK_FEAT:        # D3b: never dropped — it is the density signal to condition on
            cols.append(dg.local_density_logdk(xyz, DG_LOGDK_K)[:, None])
        return np.concatenate(cols, axis=1).astype(np.float32)

    def sample_tile(tile_path, max_pts=60000, min_pts=32, training=True):
        z = np.load(tile_path)
        xyz, intensity, ret_num, lab = z["xyz"], z["intensity"], z["ret_num"], z["lab"]
        hag = _hag_of(z, xyz) if HAG else None   # --hag: cached per-point array
        if len(xyz) < min_pts:
            return None
        idx = np.arange(len(xyz))
        if len(idx) > max_pts:
            idx = np.random.choice(idx, max_pts, replace=False)
        xyz, intensity, ret_num, lab = xyz[idx], intensity[idx], ret_num[idx], lab[idx]
        if HAG:
            hag = hag[idx]
        # D1 density jitter: re-subsample the (already GRID-tiled) cloud to a coarser
        # effective grid so the model trains across the density range it will infer on.
        # Coarsen-only (one-way valve); index-consistent across all per-point arrays.
        if training and DG_DENSITY_AUG:
            g_eff = dg.effective_grid(GRID, DG_COARSEN_MAX, DG_P_NATIVE)
            if g_eff > GRID:
                keep = dg.voxel_first_idx(xyz, g_eff)
                xyz, intensity, ret_num, lab = xyz[keep], intensity[keep], ret_num[keep], lab[keep]
                if HAG:
                    hag = hag[keep]   # SAME index as xyz, so the feature never desyncs
        # height comes from the original (pre-augmentation) z so it stays a
        # meaningful height-above-tile-min feature; --hag feeds the cached HAG instead.
        drop = (training and np.random.rand() > AUG_COLOR)
        feat = build_feat(xyz, intensity, ret_num, hag, drop=drop)
        geo_xyz = augment(xyz) if training else xyz
        geo_xyz = (geo_xyz - geo_xyz.mean(0)).astype(np.float32)
        return geo_xyz, feat, lab.astype(np.int64)

    from utils.torch_pyramid import build_full_pyramid

    class _KPBatch:
        def __init__(self, in_dict): self.in_dict = in_dict
        def device(self): return self.in_dict.points[0].device

    def make_kp_pack(samples):
        """Pack one or more (xyz, feat, lab) clouds into a single pyramid batch
        — KPConvX pack mode: concatenated points with per-cloud lengths, exactly
        how the repo's own collate batches clouds (neighbor search and pooling
        are lengths-aware, so clouds never mix)."""
        pts     = torch.from_numpy(np.ascontiguousarray(
                      np.concatenate([s[0] for s in samples]))).float()
        feats   = torch.from_numpy(np.ascontiguousarray(
                      np.concatenate([s[1] for s in samples]))).float()
        lengths = torch.tensor([len(s[0]) for s in samples], dtype=torch.long)
        pyr = build_full_pyramid(
            pts, lengths,
            net.num_layers, net.subsample_size, net.first_radius,
            net.radius_scaling, net.neighbor_limits, net.upsample_n,
            sub_mode=net.in_sub_mode, grid_pool_mode=net.grid_pool,
        )
        pyr.features = feats
        for k, v in list(pyr.items()):
            if isinstance(v, list):
                pyr[k] = [t.cuda() if torch.is_tensor(t) else t for t in v]
            elif torch.is_tensor(v):
                pyr[k] = v.cuda()
        lab_t = None
        if samples[0][2] is not None:
            lab_t = torch.from_numpy(np.ascontiguousarray(
                        np.concatenate([s[2] for s in samples]))).long().cuda()
        return _KPBatch(pyr), lab_t

    def scene_hag(z, pc_path, n):
        """Real per-point HAG from an inference scene npz (--hag); None when plain.
        convert_infer_job writes it when the HAG box is ticked."""
        if not HAG:
            return None
        if "hag" not in z.files or len(z["hag"]) != n:
            raise ValueError(
                f"{os.path.basename(pc_path)} has no per-point 'hag' channel, which this "
                f"HAG model requires. Tick 'Compute Height-Above-Ground' on the Inference "
                f"page and run again.")
        return z["hag"].astype(np.float32)

    def _predict_points(xyz, intensity_n, ret_num, hag=None):
        """Sliding-window KPConvX inference over already-normalized features;
        returns per-raw-point class indices (used by --mode infer). `hag` is
        per-raw-point real HeightAboveGround (--hag); None -> the plain recipe's
        native tile-relative height."""
        pred = np.full(len(xyz), -1, np.int64)
        with torch.no_grad():
            # windows via one packed-code sort, not a full-cloud mask per window
            for idx in tc.xy_chunk_groups(xyz, CHUNK_XY, min_pts=64):
                cols = [intensity_n[idx], ret_num[idx]]
                if hag is not None:
                    cols.append(hag[idx])          # carry real HAG through the voxel mean
                attrs = np.stack(cols, axis=1).astype(np.float32)
                sx, sa, _ = grid_subsample(xyz[idx], attrs, None, GRID)
                if len(sx) < 32:
                    continue
                # None -> build_feat derives the plain recipe's native height from sx.
                sub_hag = sa[:, 2] if hag is not None else None
                feat = build_feat(sx, sa[:, 0], sa[:, 1], sub_hag)
                base = (sx - sx.mean(0)).astype(np.float32)
                # D5 density-TTA: isotropic coord scale s IS a density change
                # (o -> o/s^2). Average softmax over decorrelated density views;
                # views=[1.0] when off -> identical to the old single-view argmax.
                views = [1.0] + (list(np.linspace(0.85, 1.2, DG_INFER_TTA))
                                 if DG_INFER_TTA else [])
                try:
                    prob = None
                    for s in views:
                        batch, _ = make_kp_pack([((base * s).astype(np.float32), feat, None)])
                        p = torch.softmax(net(batch).float(), -1).cpu().numpy()
                        prob = p if prob is None else prob + p
                    sub_pred = prob.argmax(-1)
                except Exception:
                    continue
                _, nn = cKDTree(sx).query(xyz[idx])
                pred[idx] = sub_pred[nn]
        miss = pred < 0
        if miss.any() and (~miss).any():
            _, nn = cKDTree(xyz[~miss]).query(xyz[miss])
            pred[miss] = pred[~miss][nn]
        return np.clip(pred, 0, NUM_CLASSES - 1)

    # ------------------------------------------------------------------------
    # Arbitrary-folder inference (trainer_gui): label the npz scenes staged to
    # terminal-datasets:/_infer/<job>/scenes/ and write predictions, then stop.
    # KPConvX geometry (GRID, CHUNK_XY) is fixed to the trained values; any
    # --grid/--chunk-xy passed for CLI-compatibility is informational only.
    # ------------------------------------------------------------------------
    if INFER:
        if not infer_input:
            raise ValueError("--mode infer requires --infer-input <job_id>")
        if (grid is not None and grid != GRID) or (chunk_xy is not None and chunk_xy != CHUNK_XY):
            print(f"  [infer] note: KPConvX uses its trained geometry "
                  f"(grid={GRID}, chunk={CHUNK_XY}); --grid/--chunk-xy ignored.", flush=True)
        if HAG:
            print("  [infer] 4th channel = real HeightAboveGround; every scene must "
                  "carry a per-point 'hag' array.", flush=True)
        net.eval()
        scenes = sorted(glob.glob(f"/datasets/_infer/{infer_input}/scenes/*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scenes under /datasets/_infer/{infer_input}/scenes")
        pred_dir = f"{run_dir}/predictions"
        infer_cfg = {"backbone": "KPConvX-L-HAG" if HAG else "KPConvX-L", "mode": "infer",
                     "weights": weights,
                     "infer_input": infer_input, "num_classes": NUM_CLASSES,
                     "class_names": CLASS_NAMES, "grid": GRID, "chunk_xy": CHUNK_XY,
                     "hag": (HAG_SOURCE if HAG else False), "gpu": gpu_name(),
                     "started_utc": datetime.utcnow().isoformat() + "Z"}
        print(f"  [infer] labeling {len(scenes)} scene(s) -> {run_dir}/predictions", flush=True)
        if DG_INFER_ADABN:
            # D2b: re-estimate BN running stats on the target tiles (label-free) so the
            # source-density stats stop mis-normalizing at a different inference density.
            print("  [infer] AdaBN: recomputing BN stats on target tiles...", flush=True)

            def _target_batches(cap=30):
                seen = 0
                for pc_path in scenes:
                    if seen >= cap:
                        return
                    z = np.load(pc_path)
                    txyz = z["xyz"].astype(np.float32)
                    tin = (z["intensity"].astype(np.float32) if "intensity" in z
                           else np.full(len(txyz), 0.5, np.float32))
                    trn = (z["return_number"].astype(np.float32) if "return_number" in z
                           else (z["ret_num"].astype(np.float32) if "ret_num" in z
                                 else np.zeros(len(txyz), np.float32)))
                    thag = scene_hag(z, pc_path, len(txyz))
                    # windows via one packed-code sort, not a full-cloud mask per window
                    for idx in tc.xy_chunk_groups(txyz, CHUNK_XY, min_pts=64):
                        if seen >= cap:
                            return
                        cols = [tin[idx], trn[idx]] + ([thag[idx]] if HAG else [])
                        attrs = np.stack(cols, 1).astype(np.float32)
                        sx, sa, _ = grid_subsample(txyz[idx], attrs, None, GRID)
                        if len(sx) < 32:
                            continue
                        # BN stats must see the same feature predict will be fed.
                        feat = build_feat(sx, sa[:, 0], sa[:, 1],
                                          sa[:, 2] if HAG else None)
                        cxyz = (sx - sx.mean(0)).astype(np.float32)
                        try:
                            b, _ = make_kp_pack([(cxyz, feat, None)])
                        except Exception:
                            continue
                        seen += 1
                        yield b
            dg.adabn_recalibrate(net, _target_batches(), forward=lambda m, b: m(b))
            net.eval()
        scene_stats = []
        for pc_path in scenes:
            name = os.path.splitext(os.path.basename(pc_path))[0]
            t0 = time.time()
            z = np.load(pc_path)
            xyz = z["xyz"].astype(np.float32)
            # convert_infer_job already p95-normalized intensity to [0,2].
            intensity_n = (z["intensity"].astype(np.float32) if "intensity" in z
                           else np.full(len(xyz), 0.5, np.float32))
            ret_num = (z["return_number"].astype(np.float32) if "return_number" in z
                       else (z["ret_num"].astype(np.float32) if "ret_num" in z
                             else np.zeros(len(xyz), np.float32)))
            hag = scene_hag(z, pc_path, len(xyz))   # --hag: real HAG; plain: None
            pred = _predict_points(xyz, intensity_n, ret_num, hag=hag)
            tc.write_pred(f"{pred_dir}/{name}_pred.npz", xyz, pred, intensity_n)
            np.savetxt(f"{pred_dir}/{name}_pred_CLS.txt", pred, fmt="%d")
            scene_stats.append({"scene": os.path.basename(pc_path),
                                "points": int(len(xyz)),
                                "seconds": round(time.time() - t0, 3)})
            tc.write_infer_run(run_dir, infer_cfg, scene_stats)   # crash-safe: per scene
            print(f"  [infer] {name}: {len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
        print(f"  [infer] done — predictions in _infer/{infer_input}/predictions", flush=True)
        return

    metrics_csv = f"{run_dir}/metrics.csv"
    if not os.path.exists(metrics_csv):     # keep prior rows when resuming
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "train_acc", "train_iou", "lr",
                "sec_per_epoch", "gpu_mem_mb",
            ])

    # ------------------------------------------------------------------------
    # Periodic held-out validation (eval mode, no_grad — weights untouched).
    # Appends to val_metrics.csv, so progress can be
    # watched mid-run by tailing runs/<id>/val_metrics.csv under the outputs dir.
    # ------------------------------------------------------------------------
    val_csv = f"{run_dir}/val_metrics.csv"
    if not os.path.exists(val_csv):
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "val_acc", "val_miou"] +
                                   [f"iou_{n}" for n in CLASS_NAMES])

    # Held-out VAL + TEST sets, scored with the REAL voted eval below (each item
    # carries its tile split_dir). The periodic pass scores VAL only (no test
    # peeking); the final pass also scores TEST, written separately.
    val_items  = [(n, p, c, f"{PREP_DIR}/val")  for n, p, c in val_list]
    test_items = [(n, p, c, f"{PREP_DIR}/test") for n, p, c in test_list]
    print(f"  eval set: {len(val_items)} holdout(val) + {len(test_items)} test scenes",
          flush=True)

    def evaluate(scene_items, label):
        """Final eval scored on the ORIGINAL raw points (official protocol).
        Per scene: run the model over its overlapping cached tiles (stride 50,
        each point in up to 4 tiles), sum CENTER-WEIGHTED SOFTMAX votes per 2 m
        voxel and take the argmax, then propagate each voxel's prediction to
        every raw point by nearest neighbour and score against the raw GT.
        Reprojecting to raw points removes the 2 m voxel-resolution bias and the
        arbitrary first-duplicate voxel-GT of the old voxel scoring."""
        t_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_union = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_gt    = np.zeros(NUM_CLASSES, dtype=np.int64)
        correct = total = 0
        n_scenes = n_skipped_tiles = n_skipped_scenes = 0
        t_test = time.time()
        with torch.no_grad():
            for name, pc_path, cls_path, split_dir in scene_items:
                tiles = sorted(glob.glob(f"{split_dir}/{name}_x*.npz"))
                if not tiles:
                    n_skipped_scenes += 1; continue
                keys_l, log_l, xyz_l = [], [], []
                for tile in tiles:
                    z = np.load(tile)
                    xyz = z["xyz"]
                    if len(xyz) < 32:
                        continue
                    feat = build_feat(xyz, z["intensity"], z["ret_num"],
                                      _hag_of(z, xyz) if HAG else None)
                    cxyz = (xyz - xyz.mean(0)).astype(np.float32)
                    try:
                        batch, _ = make_kp_pack([(cxyz, feat, None)])
                        lg = net(batch).cpu().numpy().astype(np.float32)
                    except Exception:
                        n_skipped_tiles += 1; continue
                    # Soft votes tapered toward the tile border (truncated context).
                    e = np.exp(lg - lg.max(1, keepdims=True))
                    prob = e / e.sum(1, keepdims=True)
                    cxy = (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
                    d = np.abs(xyz[:, :2] - cxy).max(1)
                    wgt = np.clip(1.0 - d / (CHUNK_XY / 2.0), 0.05, 1.0) ** 2
                    keys_l.append(np.floor(xyz / GRID).astype(np.int64))
                    log_l.append((prob * wgt[:, None]).astype(np.float32))
                    xyz_l.append(xyz.astype(np.float32))
                if not keys_l:
                    n_skipped_scenes += 1; continue
                K = np.concatenate(keys_l); L = np.concatenate(log_l)
                P = np.concatenate(xyz_l)
                first, inv = tc.voxel_unique(K, return_inverse=True)
                votes = np.zeros((len(first), NUM_CLASSES), np.float64)
                np.add.at(votes, inv, L)
                pred_u  = votes.argmax(1)
                rep_xyz = P[first]                      # one representative coord per voxel
                # Reproject voxel predictions onto the raw scene cloud + raw GT.
                try:
                    raw_xyz, _, _, raw_lab = load_canonical(pc_path)
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped_scenes += 1; continue
                _, nn = cKDTree(rep_xyz).query(raw_xyz)
                raw_pred = pred_u[nn]
                v = raw_lab >= 0
                rp, rl = raw_pred[v], raw_lab[v]
                correct += int((rp == rl).sum()); total += int(v.sum())
                for c in range(NUM_CLASSES):
                    t_inter[c] += int(((rp == c) & (rl == c)).sum())
                    t_union[c] += int(((rp == c) | (rl == c)).sum())
                    t_gt[c]    += int((rl == c).sum())
                n_scenes += 1
        with np.errstate(invalid="ignore"):
            iou_per = t_inter / np.maximum(t_union, 1)
        gt_counts = [int(x) for x in t_gt.tolist()]
        present = [c for c in range(NUM_CLASSES) if gt_counts[c] > 0]
        present_iou = [float(iou_per[c]) for c in present]
        present_mIoU = float(np.mean(present_iou)) if present_iou else 0.0
        m = {
            "overall_acc": correct / max(total, 1),
            "overall_mIoU": float(np.mean(iou_per)),
            "present_classes_mIoU": present_mIoU,
            "per_class_iou": {CLASS_NAMES[c]: float(iou_per[c]) for c in range(NUM_CLASSES)},
            "per_class_gt_count": {CLASS_NAMES[c]: gt_counts[c] for c in range(NUM_CLASSES)},
            "present_classes": [CLASS_NAMES[c] for c in present],
            "absent_classes":  [CLASS_NAMES[c] for c in range(NUM_CLASSES) if gt_counts[c] == 0],
            "total_test_seconds": time.time() - t_test,
            "num_scenes": n_scenes,
            "num_raw_points_scored": int(total),
            "skipped_tiles": n_skipped_tiles,
            "skipped_scenes": n_skipped_scenes,
            "scored_on": "raw_points",
            "voted_overlap": True,
            "vote_weighting": "center_tapered_softmax",
            "reprojection": "nearest_voxel_representative_to_raw",
        }
        print(f"  [{label}] acc={m['overall_acc']:.4f}  mIoU(5-way)={m['overall_mIoU']:.4f}  "
              f"mIoU(present {len(present)})={m['present_classes_mIoU']:.4f}  "
              f"absent={m['absent_classes']}  raw_pts={total:,}  "
              f"skipped(tiles={n_skipped_tiles},scenes={n_skipped_scenes})", flush=True)
        return m

    import os as _os, sys as _sys   # scripts/helper is a sibling dir (flat /root in the container image)
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "helper"))
    from train_common import BestCheckpoint, write_run_manifest
    best = BestCheckpoint(run_dir)
    # the single inference manifest (run.json); the _hag key keeps the Infer
    # page mapping HAG runs back to the HAG entry point exactly as before.
    write_run_manifest(run_dir, "kpconvx_cold_hag" if HAG else "kpconvx_cold", dataset)

    def run_eval(ep, write_json=False):
        # Periodic pass scores the held-out VAL scenes only (no test peeking) and
        # selects the best checkpoint on val present-class mIoU. The final pass
        # (write_json) also scores the TEST set and writes both, separately.
        net.eval()
        m = evaluate(val_items, f"val@ep{ep}")
        ious = [m["per_class_iou"][CLASS_NAMES[c]] for c in range(NUM_CLASSES)]
        with open(val_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{m['overall_acc']:.4f}",
                                    f"{m['present_classes_mIoU']:.4f}"] + [f"{x:.4f}" for x in ious])
        if not EVAL_ONLY and best.update(m["present_classes_mIoU"]):
            torch.save({"model": net.state_dict(), "epoch": ep}, best.final)
        if write_json:
            mt = evaluate(test_items, f"test@ep{ep}")
            with open(f"{run_dir}/test_metrics.json", "w") as fj:
                json.dump({"val": m, "test": mt,
                           "val_scenes": [n for n, _, _ in val_list],
                           "test_scenes": [n for n, _, _ in test_list]}, fj, indent=2)
        net.train()
        return m

    # ------------------------------------------------------------------------
    # Train loop — KPConvX recipe: EPOCH_STEPS optimizer steps of
    # PACK_N tiles/forward x ACCUM accumulated forwards.
    # ------------------------------------------------------------------------
    LOG_EVERY = 50
    print(f"  starting at epoch {start_epoch}, up to {N_EPOCHS}, "
          f"{EPOCH_STEPS} steps/epoch, pack {PACK_N} x accum {ACCUM}", flush=True)
    t_run = time.time()
    for ep in range(start_epoch, N_EPOCHS):
        cur_lr = lr_at(ep)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        net.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        ep_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        ep_union = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_ep = time.time()
        n_steps = n_fwd = n_failed = 0
        print(f"  ep {ep:3d} starting (lr={cur_lr:.2e})…", flush=True)
        for step in range(EPOCH_STEPS):
            optim.zero_grad()
            n_ok = 0
            for _ in range(ACCUM):
                samples = []
                while len(samples) < PACK_N:
                    s = sample_tile(pick_train_tile(), training=True)
                    if s is not None:
                        samples.append(s)
                try:
                    batch, lab_t = make_kp_pack(samples)
                    logits = net(batch)
                    loss = seg_loss(logits, lab_t) / ACCUM
                    # Defense in depth: skip any non-finite loss (NaN/Inf from an
                    # all-ignored pack or a numerical blow-up) before backward, so
                    # one bad gradient can't poison the weights — as RandLA does.
                    if not torch.isfinite(loss):
                        n_failed += 1
                        continue
                    loss.backward()
                    n_ok += 1; n_fwd += 1
                    ep_loss += loss.item() * ACCUM
                    pred = logits.argmax(-1)
                    m = lab_t >= 0
                    ep_correct += (pred[m] == lab_t[m]).sum().item()
                    ep_total   += int(m.sum())
                    for c in range(NUM_CLASSES):
                        ep_inter[c] += ((pred == c) & (lab_t == c)).sum().item()
                        ep_union[c] += (((pred == c) | (lab_t == c)) & m).sum().item()
                except Exception as e:
                    n_failed += 1
                    if n_failed == 1:
                        print(f"  forward failed (first occurrence, step {step}): {e}",
                              flush=True)
                        traceback.print_exc()
            if n_ok:
                torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
                optim.step()
                n_steps += 1
                if n_steps % LOG_EVERY == 0:
                    print(f"    ep {ep:3d} step {n_steps:4d}: "
                          f"loss={ep_loss/max(n_fwd,1):.4f}", flush=True)
        if n_steps == 0:
            raise RuntimeError(f"epoch {ep}: 0 optimizer steps — {n_failed} failed forwards.")
        if n_failed:
            print(f"  ep {ep:3d} note: {n_failed} failed forwards", flush=True)
        sec_per_epoch = time.time() - t_ep
        train_acc = ep_correct / max(ep_total, 1)
        with np.errstate(invalid="ignore"):
            train_iou = float(np.mean(ep_inter / np.maximum(ep_union, 1)))
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, f"{ep_loss/max(n_fwd,1):.6f}", f"{train_acc:.4f}",
                f"{train_iou:.4f}", f"{cur_lr:.6e}", f"{sec_per_epoch:.2f}", f"{gpu_mem:.1f}",
            ])
        print(f"  ep {ep:3d}: loss={ep_loss/max(n_fwd,1):.4f} acc={train_acc:.4f} "
              f"miou={train_iou:.4f} lr={cur_lr:.2e} s/epoch={sec_per_epoch:.1f}", flush=True)
        if (ep + 1) % CHECKPOINT_GAP == 0:
            torch.save({"model": net.state_dict(), "optim": optim.state_dict(),
                        "epoch": ep},
                       f"{run_dir}/checkpoints/ep{ep:03d}.pth")
        if (ep + 1) % VAL_EVERY == 0 and ep != N_EPOCHS - 1:
            run_eval(ep)               # last epoch handled by the final eval below

    # --- Final evaluation: the real voted eval over the combined eval set,
    # written to test_metrics.json (the same number run_eval logs periodically). -
    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(N_EPOCHS - 1, write_json=True)
    if not EVAL_ONLY:
        best.finalize(lambda p: torch.save(
            {"model": net.state_dict(), "epoch": N_EPOCHS - 1}, p))
    print(f"  total wall-clock: {(time.time() - t_run)/3600:.2f} h")


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Local kpconvx_cold trainer/inferencer.')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--mode', default='train')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--infer-input', default=None)
    ap.add_argument('--grid', type=float, default=None)
    ap.add_argument('--chunk-xy', type=float, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--steps-per-epoch', type=int, default=None)
    ap.add_argument('--hag', action='store_true',
                    help='swap the 4th input channel to real HeightAboveGround '
                         '(replaces the old local_train_kpconvx_cold_hag.py)')
    args = ap.parse_args()
    # --dataset is required for training/eval; only --mode infer may omit it.
    if args.dataset is None and args.mode != 'infer':
        ap.error('--dataset is required (a canonical trainer_gui dataset name); '
                 'only --mode infer may omit it.')
    train_kpconvx(**vars(args))


if __name__ == "__main__":
    main()
