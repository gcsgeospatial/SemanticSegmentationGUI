"""
Local training script for KPConvX-L on a canonical trainer_gui --dataset
(3-folder train/val/test), COLD-START, native-features variant.

  Features  : 4 = [1, intensity, return_number, height], where height is
              tile-relative (z - tile_min_z). No geometric-feature
              computation — uses the attributes the cloud carries.
              FEAT_CHANNELS env overrides the spec (ordered csv incl. dataset
              feat_* channels, e.g. feat_hag for real HeightAboveGround);
              run.json "features" records the resolved list.

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
    python local_train_kpconvx_cold.py --dataset <name> --mode eval \
        --weights runs/<id>/final_model.pth   # re-score with the voted eval
    python local_train_kpconvx_cold.py --mode infer --infer-input <job> \
        --weights runs/<id>/final_model.pth   # label staged scenes
"""

from typing import Optional

# ============================================================================
# Configuration
# ============================================================================
FEATURE_MODE  = "native"     # [1, intensity, return_number, height]
N_EPOCHS      = 100
EPOCH_STEPS   = 300          # optimizer steps / epoch (KPConvX S3DIS: 300)
PACK_N        = 4            # tiles packed per forward (KPConvX batch_size = 4)
ACCUM         = 2            # grad-accumulated forwards / step -> effective batch 8 tiles
CHECKPOINT_GAP = 10          # checkpoint frequency (epochs); saves model + optimizer
VAL_EVERY     = 10           # held-out val pass every N epochs (no weight updates)

# Resume: when True, continue the most recent AdamW-recipe run in the outputs
# dir (same run dir, appended metrics) instead of starting fresh.
RESUME = False
# The cloud shell sets AUTO_RESUME=1 only on Modal's OWN retries (preemption /
# crash); locally a user can export it to continue after a Kill. Same contract
# as the PTv3-family trainers.
AUTO_RESUME = os.environ.get("AUTO_RESUME", "0") == "1"

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

# Input-feature spec (FEAT_CHANNELS env, GUI picker): comma-separated ordered
# names; "" = the legacy [intensity, return_number, height] recipe. The
# constant-1 bias channel is always first and not part of the spec.
FEAT_CHANNELS = ""

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

# Augmentation (train_LAS.py LASConfig): geometry (rotation/scale/x-flip/noise)
# is train_common.kp_augment's defaults; only the feature-drop probability is
# configured here.
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
                  batch: Optional[int] = None, steps_per_epoch: Optional[int] = None):
    import os, sys, time, json, csv, glob, traceback
    from datetime import datetime
    import numpy as np
    import torch


    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helper"))
    import density as dg
    import train_common as tc
    # Env-overridable knobs (GUI "Density generalization" + "Loss & class
    # balance" panels; see train_common._ENV_KNOBS). Local shadows: the nested
    # closures capture these, defaulting to the module constants.
    (DG_DENSITY_AUG, DG_COARSEN_MAX, DG_P_NATIVE, DG_LOGDK_FEAT, DG_LOGDK_K,
     DG_INFER_ADABN, DG_INFER_TTA, USE_FOCAL, FOCAL_GAMMA, CLASS_WEIGHTING,
     WEIGHT_BETA, RARE_OVERSAMPLE, KP_AGGREGATION, KP_NORM,
     VAL_EVERY, FEAT_CHANNELS) = tc.env_overrides(
        globals(), [
        "DG_DENSITY_AUG", "DG_COARSEN_MAX", "DG_P_NATIVE", "DG_LOGDK_FEAT",
        "DG_LOGDK_K", "DG_INFER_ADABN", "DG_INFER_TTA", "USE_FOCAL",
        "FOCAL_GAMMA", "CLASS_WEIGHTING", "WEIGHT_BETA", "RARE_OVERSAMPLE",
        "KP_AGGREGATION", "KP_NORM", "VAL_EVERY", "FEAT_CHANNELS"])

    # KPCONVX_SRC (conda trainer-src-kpconvx package activation) points at the
    # source tree; the container images bake it at /opt/kpconvx.
    sys.path.insert(0, os.environ.get("KPCONVX_SRC", "/opt/kpconvx"))
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
    FEATURE_MODE = globals()["FEATURE_MODE"]
    # The 1-cycle phases are quoted for the 100-epoch default; scale them with
    # N_EPOCHS so a short run still traverses the whole curve instead of dying
    # in the warmup ramp. At N_EPOCHS=100 the scale is 1.0, so the default
    # schedule (30/5/120) is bit-for-bit unchanged.
    _cyc_scale     = N_EPOCHS / 100.0
    CYC_RAISE      = max(1, round(globals()["CYC_RAISE"] * _cyc_scale))
    CYC_PLATEAU    = round(globals()["CYC_PLATEAU"] * _cyc_scale)
    CYC_DECREASE10 = globals()["CYC_DECREASE10"] * _cyc_scale
    # Input-feature spec: FEAT_CHANNELS env at train (the infer branch below
    # overrides it from run.json — env is ignored at infer). "height" is always
    # tile-relative; real HAG is the ordinary feat_hag dataset channel.
    FEAT_LEGACY = ["intensity", "return_number", "height"]
    FEAT_SPEC = (list(FEAT_LEGACY) if INFER    # env ignored at infer
                 else tc.parse_feat_spec(FEAT_CHANNELS, FEAT_LEGACY))

    # --dataset NAME selects a canonical trainer_gui dataset under /datasets
    # (bind-mounted); NUM_CLASSES / CLASS_NAMES / PREP_DIR are locals so the
    # whole function tracks the dataset's class layout.
    ds_root = tc.dataset_dir(dataset) if dataset else None
    if ds_root:
        ds_meta, NUM_CLASSES, CLASS_NAMES = tc.load_dataset_meta(dataset)
        # A custom feature spec gets its own cache family (legacy spec = ""
        # tag, old caches valid).
        PREP_DIR = (f"{ds_root}/prep/kpconvx_cold"
                    f"_grid{GRID:g}_c{int(CHUNK_XY)}"
                    f"{tc.feat_spec_tag(FEAT_SPEC, FEAT_LEGACY)}"
                    f"{tc.train_stride_tag()}")
    else:
        # Folder inference (--mode infer): no --dataset. Reproduce the TRAINED
        # geometry + class layout from the run's run.json (the self-contained
        # manifest beside the weights; legacy run_config.json is the fallback).
        # INFER never reads cached tiles, so PREP_DIR is an unused placeholder.
        NUM_CLASSES = 5
        CLASS_NAMES = [f"class {i}" for i in range(NUM_CLASSES)]
        PREP_DIR = f"{tc.OUTPUTS_ROOT}/_infer_unused"
        if INFER and weights:
            meta = tc.infer_meta(weights if os.path.isabs(weights)
                                 else f"{tc.OUTPUTS_ROOT}/{weights}")
            if meta:
                NUM_CLASSES = int(meta.get("num_classes") or NUM_CLASSES)
                CLASS_NAMES = list(meta.get("class_names") or
                                   [f"class {i}" for i in range(NUM_CLASSES)])
                if meta.get("grid") is not None: GRID = float(meta["grid"])
                if meta.get("chunk_xy") is not None: CHUNK_XY = float(meta["chunk_xy"])
                STRIDE = CHUNK_XY / 2.0
                # rebuild the EXACT assembly recorded with the weights (env is
                # ignored at infer); manifests without "features" = legacy runs.
                mf = meta.get("features")
                try:
                    FEAT_SPEC = (tc.parse_feat_spec(",".join(mf), FEAT_LEGACY)
                                 if mf else list(FEAT_LEGACY))
                except ValueError:
                    FEAT_SPEC = list(FEAT_LEGACY)
                if meta.get("hag_source"):
                    # ponytail: TEMPORARY shim (remove once legacy runs retire) —
                    # the deleted --hag variant's 'height' channel was real HAG.
                    # Feed the baked feat_hag channel in that slot: same width,
                    # right semantics. Scenes MUST be converted with HAG on.
                    FEAT_SPEC = ["feat_hag" if n == "height" else n for n in FEAT_SPEC]
                    print(f"  [legacy-hag] weights from the removed --hag variant "
                          f"(hag_source={meta['hag_source']}): 'height' -> feat_hag; "
                          f"input scenes must carry a baked feat_hag channel",
                          flush=True)

    if "rgb" in FEAT_SPEC:
        raise ValueError("the KPConvX tile pipeline has no rgb channel — use "
                         "intensity (rgb is folded into it when a scene has "
                         "no intensity)")
    # bias + spec channels (+ log d_k); every kp spec name is width 1.
    IN_CH = 1 + len(FEAT_SPEC)

    def _cache_signature():
        # Everything that changes what a cached tile contains. A mismatch means
        # the cache is stale/leaky and must not be silently reused. The split is a
        # property of the DATASET (3 materialized folders), so the signature
        # records the dataset's split identity.
        sp = ds_meta.get("split", {})
        # The spec-derived recipe string reproduces the legacy spellings
        # byte-for-byte, so every existing legacy-spec cache stays valid.
        return {
            # v2: tiles carry the scene's real return_number (was zero-filled)
            "format_version": 2,
            "pipeline": "kpconvx_cold",
            "grid": GRID,
            "chunk_xy": CHUNK_XY,
            "stride": STRIDE,
            "split_seed": sp.get("seed"),
            "split_mode": sp.get("mode"),
            "min_pts_mask": 64,
            "min_pts_sub": 32,
            "intensity_norm": "p95_clip2",
            # kp_tile_and_save bakes labels against the class layout: merging or
            # reordering classes must invalidate the cache.
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "feature_recipe": "bias," + ",".join(
                "ret_num" if n == "return_number" else n
                for n in FEAT_SPEC),
        }

    def ensure_prep():
        # train tiles at the (wider) TT_TRAIN_STRIDE; val/test keep chunk/2 so
        # the final eval can vote over the up-to-4 overlapping covering tiles.
        return tc.kp_ensure_prep(
            PREP_DIR, ds_root, _cache_signature(),
            lambda name, pc_path, out_dir, split: tc.kp_tile_and_save(
                name, pc_path, out_dir, CHUNK_XY,
                tc.train_stride(CHUNK_XY) if split == "train" else STRIDE,
                GRID, NUM_CLASSES))

    def find_latest_checkpoint():
        # Only same-recipe runs are valid resume targets: AdamW recipe AND this
        # exact feature spec.
        return tc.kp_find_latest_checkpoint("AdamW", {FEATURE_MODE},
                                            features=FEAT_SPEC,
                                            legacy_features=FEAT_LEGACY,
                                            skip_done=not EVAL_ONLY)

    print("=" * 70)
    print(f"  KPConvX-L  {dataset or 'infer'}  COLD/{FEATURE_MODE}  "
          f"({tc.gpu_name()}, {N_EPOCHS} ep, {EPOCH_STEPS} steps, "
          f"pack {PACK_N} x accum {ACCUM})")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}  device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    if not INFER:
        # Clear a stale STOP from an old run BEFORE the slow prep (tiling,
        # calibration): a stop clicked during startup must survive to the loop.
        tc.clear_stop()
    train_list, val_list, test_list = ([], [], []) if INFER else ensure_prep()

    resume_info = (find_latest_checkpoint()
                   if (RESUME or AUTO_RESUME or EVAL_ONLY) else None)
    if INFER:
        # Inference-only: fresh *_infer run dir, weights loaded after net build.
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_infer")
        # Predictions live next to the input scenes under the datasets _infer
        # tree (not the per-model runs dir), so inference output is in one
        # consistent place.
        run_dir = tc.infer_dir(infer_input)
        os.makedirs(os.environ.get("TT_PRED_DIR") or f"{run_dir}/predictions",
                    exist_ok=True)
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
        run_id, run_dir = tc.kp_make_run_dir("kpconvx_cold_native")
        resume_ckpt, start_epoch = None, 0
    if resume_ckpt is None and not INFER:
        with open(f"{run_dir}/run.json", "w") as f:
            json.dump({
            "backbone": "KPConvX-L",
            "warm_start": False,
            "feature_mode": FEATURE_MODE,
            "input_channels": IN_CH,
            # resolved input spec (bias/log-dk ride outside it) — inference
            # rebuilds this exact assembly from here.
            "features": FEAT_SPEC,
            "dataset": dataset,
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
    cfg.model.input_channels = IN_CH + (1 if DG_LOGDK_FEAT else 0)  # +log d_k (D3b)
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
    # seg_loss is built after the tile scan below (needs class counts for weights).

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
        fm = ((weights if os.path.isabs(weights) else f"{tc.OUTPUTS_ROOT}/{weights}")
              if weights else f"{run_dir}/final_model.pth")
        if weights and not os.path.exists(fm):
            raise FileNotFoundError(f"--weights not found: {fm}")
        if os.path.exists(fm):
            net.load_state_dict(torch.load(fm, map_location="cuda", weights_only=True)["model"])
            print(f"  EVAL-ONLY: loaded {fm}", flush=True)
        start_epoch = N_EPOCHS

    if INFER:
        fm = ((weights if os.path.isabs(weights) else f"{tc.OUTPUTS_ROOT}/{weights}")
              if weights else None)
        if not fm or not os.path.exists(fm):
            raise FileNotFoundError(f"--mode infer requires --weights; not found: {fm}")
        ck = tc.load_ckpt_safe(fm, map_location="cuda")
        net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        print(f"  [infer] loaded {weights} (final_model = best-val epoch "
              f"{ck.get('epoch', '?') if isinstance(ck, dict) else '?'})", flush=True)
        start_epoch = N_EPOCHS

    train_tiles = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    val_tiles   = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
    test_tiles  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
    if not INFER:
        print(f"  train_tiles: {len(train_tiles)}   val_tiles: {len(val_tiles)}   "
              f"test_tiles: {len(test_tiles)}", flush=True)
    if not train_tiles and not INFER:
        raise RuntimeError("No training tiles after preprocessing — check the dataset.")

    if INFER:
        # Inference never trains or samples tiles — skip the class-balance scan
        # (and its noisy zero-count logging); the loss/picker are train-only.
        seg_loss = pick_train_tile = None
    else:
        # --- class-balanced loss + rare-class oversampling ------------------
        # One (cached, parallel) pass over the training tiles: label counts for
        # inverse-frequency class weights, then flag tiles containing any rare
        # class so we can oversample them. Rare = explicit RARE_CLASSES, else auto
        # (below RARE_FREQ_FRAC x the median present-class count).
        class_counts, present_mask = tc.scan_class_balance(
            train_tiles, NUM_CLASSES,
            cache_path=f"{PREP_DIR}/class_balance_cache.npz")
        print(f"  class counts: {dict(zip(CLASS_NAMES, class_counts.tolist()))}",
              flush=True)
        rare_classes = (list(RARE_CLASSES) if RARE_CLASSES is not None
                        else tc.auto_rare_classes(class_counts, RARE_FREQ_FRAC))
        rare_tiles = ([train_tiles[i]
                       for i in np.nonzero(present_mask[:, rare_classes].any(1))[0]]
                      if (RARE_OVERSAMPLE and rare_classes) else [])
        print(f"  rare classes: {[CLASS_NAMES[c] for c in rare_classes]}", flush=True)
        print(f"  rare-class tiles: {len(rare_tiles)} / {len(train_tiles)}", flush=True)

        if CLASS_WEIGHTING:
            w = tc.class_weights_np(class_counts, WEIGHT_BETA, WEIGHT_CAP)
            class_weights = torch.tensor(w, dtype=torch.float32).cuda()
            print(f"  class weights: "
                  f"{dict(zip(CLASS_NAMES, [round(float(x), 3) for x in w]))}",
                  flush=True)
        else:
            class_weights = None
        # Shared loss recipe: weighted smoothed CE (or focal) + Lovász-Softmax,
        # with the all-ignored-pack NaN guard (see train_common.make_seg_loss).
        seg_loss = tc.make_seg_loss(class_weights, LABEL_SMOOTH, USE_FOCAL,
                                    FOCAL_GAMMA, LOVASZ_WEIGHT)
        pick_train_tile = tc.make_tile_picker(train_tiles, rare_tiles, RARE_TILE_PROB)

    # Feature recipe + tile sampler (shared KP-family pipeline; dataset feat_*
    # channels are cached by kp_tile_and_save and fed by the spec, nothing here).
    build_feat = tc.kp_make_build_feat(DG_LOGDK_FEAT, DG_LOGDK_K, FEAT_SPEC)
    sample_tile = tc.kp_make_sample_tile(
        build_feat, GRID, max_pts=60000, aug_color=AUG_COLOR,
        density_aug=DG_DENSITY_AUG, coarsen_max=DG_COARSEN_MAX,
        p_native=DG_P_NATIVE)

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

    def _kp_batch(cxyz, feat):
        return make_kp_pack([(cxyz, feat, None)])[0]

    # Sliding-window inference over already-normalized features (--mode infer),
    # with the D5 density-TTA view averaging (shared KP-family pipeline).
    SAVE_PROBS = os.environ.get("TT_SAVE_PROBS") == "1"
    EXC_IDX = tc.exclude_class_idx(CLASS_NAMES) if INFER else []
    _predict_points = tc.kp_make_predict_points(
        lambda cxyz, feat: torch.softmax(net(_kp_batch(cxyz, feat)).float(),
                                         -1).cpu().numpy(),
        build_feat, GRID, CHUNK_XY, NUM_CLASSES, DG_INFER_TTA,
        save_probs=SAVE_PROBS, exclude_idx=EXC_IDX)

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
        net.eval()
        scenes = sorted(glob.glob(f"{run_dir}/scenes/*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scenes under {run_dir}/scenes")
        pred_dir = os.environ.get("TT_PRED_DIR") or f"{run_dir}/predictions"
        infer_cfg = {"backbone": "KPConvX-L", "mode": "infer",
                     "weights": weights,
                     "infer_input": infer_input, "num_classes": NUM_CLASSES,
                     "class_names": CLASS_NAMES, "grid": GRID, "chunk_xy": CHUNK_XY,
                     "gpu": tc.gpu_name(),
                     "exclude_classes": [CLASS_NAMES[i] for i in EXC_IDX],
                     "started_utc": datetime.utcnow().isoformat() + "Z"}
        if DG_INFER_ADABN:
            # D2b: re-estimate BN running stats on the target tiles (label-free) so the
            # source-density stats stop mis-normalizing at a different inference density.
            print("  [infer] AdaBN: recomputing BN stats on target tiles...", flush=True)
            dg.adabn_recalibrate(
                net,
                tc.kp_make_target_batches(scenes, _kp_batch, build_feat,
                                          GRID, CHUNK_XY, NUM_CLASSES),
                forward=lambda m, b: m(b))
            net.eval()

        def _predict(pc_path):
            z = np.load(pc_path)
            # predict in the scene-local frame (kp_load_canonical's origin
            # shift — the frame the model trained in; global-UTM float32 both
            # quantizes coords and mis-centers windows), return the ORIGINAL
            # georeferenced coords as the deliverable.
            raw = z["xyz"]
            xyz = (raw - np.floor(raw.min(0))).astype(np.float32)
            # convert_infer_job already p95-normalized intensity to [0,2].
            intensity_n, ret_num = tc.scene_arrays(z, len(xyz))
            extras = tc.feat_extras(z, FEAT_SPEC, os.path.basename(pc_path))
            pred, conf, probs = _predict_points(xyz, intensity_n, ret_num,
                                                extras=extras)
            return raw, pred, intensity_n, conf, probs

        tc.run_infer_scenes(scenes, _predict, pred_dir, run_dir, infer_cfg, cls_txt=True)
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

    # Final/periodic eval scored on the ORIGINAL raw points: center-weighted
    # softmax votes over the overlapping cached tiles, per-voxel argmax,
    # NN-reprojected to the raw cloud (shared KP-family pipeline).
    def _fwd_eval(tiles):     # [(cxyz, feat)] -> per-tile logits list
        b, _ = make_kp_pack([(c, f, None) for c, f in tiles])
        lg = net(b).cpu().numpy().astype(np.float32)
        return np.split(lg, np.cumsum([len(c) for c, _ in tiles])[:-1])
    evaluate = tc.kp_make_evaluate(_fwd_eval, build_feat, GRID, CHUNK_XY,
                                   NUM_CLASSES, CLASS_NAMES)

    best = tc.BestCheckpoint(run_dir)
    tc.write_run_manifest(run_dir, "kpconvx_cold", dataset)

    def run_eval(ep, write_json=False):
        # Periodic pass scores the held-out VAL scenes only (no test peeking) and
        # selects the best checkpoint on val present-class mIoU. The final pass
        # (write_json) scores TEST on the BEST-TRACKED checkpoint, not whatever
        # epoch training happened to end on, so test_metrics.json reports the
        # model actually kept as final_model.pth. VAL stays on the current
        # (most-recent) weights, matching the periodic curve. The swap is skipped
        # in --mode eval (EVAL_ONLY): there "current" already IS the explicitly
        # requested --weights, and best.final may be a different checkpoint.
        #
        # PreciseBN (Yan et al., "Rethinking 'Batch' in BatchNorm"): eval-mode BN
        # uses EMA running stats that lag the fast-moving weights ("moment
        # staleness") — the failure where train stays smooth while val whipsaws
        # (ground<->water wholesale flips). Before scoring, re-estimate the stats
        # with the CURRENT frozen weights over clean train tiles, cumulative
        # average (momentum=None), single-tile packs to match evaluate()'s
        # forwards. best.update then both selects on and saves the precise stats.
        # Skipped in EVAL_ONLY: an explicitly requested --weights is scored
        # as-shipped, reproducing the curve that selected it.
        if not EVAL_ONLY:
            def _bn_batches(n=48):
                made = 0
                while made < n:
                    s = sample_tile(pick_train_tile(), training=False)
                    if s is None:
                        continue
                    try:
                        b, _ = make_kp_pack([s])
                    except Exception:
                        continue
                    made += 1
                    yield b
            dg.adabn_recalibrate(net, _bn_batches(), forward=lambda mdl, b: mdl(b))
        net.eval()
        m = evaluate(val_items, f"val@ep{ep}")
        tc.append_val_row(val_csv, ep, m, CLASS_NAMES)
        if not EVAL_ONLY and best.update(m["present_classes_mIoU"]):
            torch.save({"model": net.state_dict(), "epoch": ep}, best.final)
        if write_json:
            swapped = not EVAL_ONLY and os.path.exists(best.final)
            if swapped:
                live_state = {k: v.clone() for k, v in net.state_dict().items()}
                net.load_state_dict(torch.load(best.final, map_location="cuda",
                                               weights_only=True)["model"])
                net.eval()
            mt = evaluate(test_items, f"test@ep{ep}")
            if swapped:
                net.load_state_dict(live_state)
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
    # bf16 autocast (A100+): opt-in — it changes training numerics, so it's a
    # knob, not a default. No GradScaler needed for bf16.
    AMP = os.environ.get("TT_AMP") == "1"
    # Background pack building: tile np.load + feature assembly + the CPU
    # pyramid build (fill_pyramid dispatches its neighbor search by DEVICE, so
    # a CPU-built pyramid means CPU KNN) overlap the GPU fwd/bwd instead of
    # serializing in front of it — the same overlap upstream KPConvX gets from
    # its DataLoader workers.
    def _draw():
        while True:
            s = sample_tile(pick_train_tile(), training=True)
            if s is not None:
                return s
    prefetch = (tc.make_prefetcher(
        lambda: make_kp_pack([_draw() for _ in range(PACK_N)]),
        depth=int(os.environ.get("TT_PREFETCH", "2")))
        if start_epoch < N_EPOCHS else None)
    print(f"  starting at epoch {start_epoch}, up to {N_EPOCHS}, "
          f"{EPOCH_STEPS} steps/epoch, pack {PACK_N} x accum {ACCUM}"
          f"{' [bf16 autocast]' if AMP else ''}", flush=True)
    t_run = time.time()
    ep = N_EPOCHS - 1     # final-eval label when the loop never runs (EVAL_ONLY)
    for ep in range(start_epoch, N_EPOCHS):
        cur_lr = lr_at(ep)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        net.train()
        ep_loss = 0.0
        # Confusion matrix accumulated ON the GPU: the old per-class python
        # loop forced ~4*NUM_CLASSES device syncs per forward; this is one
        # bincount per forward and a single .cpu() per epoch.
        ep_conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long,
                              device="cuda")
        t_ep = time.time()
        n_steps = n_fwd = n_failed = 0
        print(f"  ep {ep:3d} starting (lr={cur_lr:.2e})…", flush=True)
        for step in range(EPOCH_STEPS):
            optim.zero_grad()
            n_ok = 0
            for _ in range(ACCUM):
                try:
                    batch, lab_t = prefetch()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
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
                    ep_conf += torch.bincount(
                        lab_t[m] * NUM_CLASSES + pred[m],
                        minlength=NUM_CLASSES * NUM_CLASSES,
                    ).reshape(NUM_CLASSES, NUM_CLASSES)
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
        conf = ep_conf.cpu().numpy()
        ep_inter = np.diag(conf)
        ep_union = conf.sum(0) + conf.sum(1) - ep_inter
        train_acc = int(np.trace(conf)) / max(int(conf.sum()), 1)
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
        stop = tc.stop_requested(ep)
        if (ep + 1) % VAL_EVERY == 0 and ep != N_EPOCHS - 1 and not stop:
            run_eval(ep)               # last epoch handled by the final eval below
        if stop:
            break                      # falls through to the final eval + finalize

    if prefetch:
        prefetch.shutdown()      # stop background pack builds during the eval

    # --- Final evaluation: the real voted eval over the combined eval set,
    # written to test_metrics.json (the same number run_eval logs periodically). -
    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(ep, write_json=True)
    if not EVAL_ONLY:
        best.finalize(lambda p: torch.save(
            {"model": net.state_dict(), "epoch": ep}, p))
        # Mark the run complete so AUTO_RESUME won't re-resume it on the next
        # launch (a crashed/retried run has no DONE and is picked back up).
        open(f"{run_dir}/DONE", "w").close()
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
    args = ap.parse_args()
    # --dataset is required for training/eval; only --mode infer may omit it.
    if args.dataset is None and args.mode != 'infer':
        ap.error('--dataset is required (a canonical trainer_gui dataset name); '
                 'only --mode infer may omit it.')
    train_kpconvx(**vars(args))


if __name__ == "__main__":
    main()
