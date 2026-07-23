"""KPConvX-L local trainer, cold start, on a canonical trainer_gui dataset.
Default features [1, intensity, return_number]; FEAT_CHANNELS env overrides,
run.json "features" records it. KPConvX's own S3DIS recipe (AdamW + 1-cycle,
packed batches, voted eval over overlapping tiles). Modes: train | eval
(--weights re-score) | infer (--weights --infer-input, dataset-free).
"""

import os
from typing import Optional

FEATURE_MODE  = "native"     # [1, intensity, return_number] (+ selected feat_*)
N_EPOCHS      = 100
EPOCH_STEPS   = 300          # optimizer steps / epoch (KPConvX S3DIS: 300)
PACK_N        = 4            # tiles packed per forward (KPConvX batch_size = 4)
ACCUM         = 2            # grad-accumulated forwards / step -> effective batch 8 tiles
CHECKPOINT_GAP = 10          # checkpoint frequency (epochs); saves model + optimizer
VAL_EVERY     = 10           # held-out val pass every N epochs (no weight updates)

RESUME = False               # continue the most recent AdamW-recipe run
# AUTO_RESUME=1: set by the cloud shell on Modal retries; exportable locally
AUTO_RESUME = os.environ.get("AUTO_RESUME", "0") == "1"

CLASS_WEIGHTING = True
WEIGHT_BETA     = 0.5        # 0.5 = inverse-sqrt frequency
WEIGHT_CAP      = 5.0        # clamp each weight to [1/CAP, CAP] after mean-norm
LOVASZ_WEIGHT   = 1.0        # + LOVASZ_WEIGHT * lovasz_softmax; 0 disables
USE_FOCAL       = False      # focal pointwise term (skips LABEL_SMOOTH)
FOCAL_GAMMA     = 2.0
RARE_OVERSAMPLE = True
RARE_CLASSES    = None       # explicit indices, or None -> auto from train freq
RARE_FREQ_FRAC  = 0.5        # auto-rare: freq < frac x median present freq
RARE_TILE_PROB  = 0.25       # 0.5 + cap 10 overcooked; dialed back

# FEAT_CHANNELS env: ordered csv; "" = [intensity, return_number].
# The constant-1 bias channel is always first and not part of the spec.
FEAT_CHANNELS = ""

# geometry — matches train_LAS.py LASConfig
GRID          = 2.0          # layer-0 voxel grid (m)
KP_RADIUS     = 2.5          # conv_radius (grid cells)
RADIUS_SCALING = 2.0

# input region ~50x the grid; stride chunk/2 so eval tiles overlap for voting
CHUNK_XY      = 100.0
STRIDE        = 50.0

AUG_COLOR     = 0.8          # per-channel keep prob (geometry aug = kp_augment defaults)

# density domain-generalization (scripts/helper/density.py; DENSITY_DG.md)
DG_DENSITY_AUG = False   # D1: per-tile coarsen to a jittered grid (train only)
DG_COARSEN_MAX = 2.5     # = 1/(GRID*sqrt(rho_min))
DG_P_NATIVE    = 0.5     # P(tile kept at native GRID)
DG_INFER_ADABN = False   # D2b: recompute BN stats on target tiles before predicting
DG_INFER_TTA   = 0       # D5: extra density(scale) views at inference (0=off)
# D3b: log k-th-NN-distance input channel; bumps input_channels (retrain)
DG_LOGDK_FEAT  = False
DG_LOGDK_K     = 8
KP_AGGREGATION = "nearest"   # D2a: neighbour aggregation mode
KP_NORM        = "batch"     # D2c: "batch" | lib-dependent (e.g. "group")

# optimizer — KPConvX's own S3DIS recipe
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
    from datetime import datetime, timezone
    import numpy as np
    import torch


    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helper"))
    import density as dg
    import train_common as tc
    # GUI env-overridable knobs (train_common._ENV_KNOBS); closures capture these
    (DG_DENSITY_AUG, DG_COARSEN_MAX, DG_P_NATIVE, DG_LOGDK_FEAT, DG_LOGDK_K,
     DG_INFER_ADABN, DG_INFER_TTA, USE_FOCAL, FOCAL_GAMMA, CLASS_WEIGHTING,
     WEIGHT_BETA, RARE_OVERSAMPLE, KP_AGGREGATION, KP_NORM,
     VAL_EVERY, FEAT_CHANNELS) = tc.env_overrides(
        globals(), [
        "DG_DENSITY_AUG", "DG_COARSEN_MAX", "DG_P_NATIVE", "DG_LOGDK_FEAT",
        "DG_LOGDK_K", "DG_INFER_ADABN", "DG_INFER_TTA", "USE_FOCAL",
        "FOCAL_GAMMA", "CLASS_WEIGHTING", "WEIGHT_BETA", "RARE_OVERSAMPLE",
        "KP_AGGREGATION", "KP_NORM", "VAL_EVERY", "FEAT_CHANNELS"])

    sys.path.insert(0, os.environ.get("KPCONVX_SRC", "/opt/kpconvx"))
    EVAL_ONLY = (mode == "eval")
    INFER     = (mode == "infer")

    if dataset is None and not INFER:
        raise ValueError("--dataset is required: pass a canonical trainer_gui "
                         "dataset name (train/val/test folders). The only "
                         "dataset-free path is --mode infer.")

    # --- resolve run config: CLI flags override the module defaults ----------
    GRID        = grid if grid is not None else globals()["GRID"]
    CHUNK_XY    = chunk_xy if chunk_xy is not None else globals()["CHUNK_XY"]
    STRIDE      = CHUNK_XY / 2.0
    N_EPOCHS    = epochs if epochs is not None else globals()["N_EPOCHS"]
    EPOCH_STEPS = steps_per_epoch if steps_per_epoch is not None else globals()["EPOCH_STEPS"]
    PACK_N      = batch if batch is not None else globals()["PACK_N"]
    FEATURE_MODE = globals()["FEATURE_MODE"]
    # scale the 1-cycle phases with N_EPOCHS so short runs traverse the whole curve
    _cyc_scale     = N_EPOCHS / 100.0
    CYC_RAISE      = max(1, round(globals()["CYC_RAISE"] * _cyc_scale))
    CYC_PLATEAU    = round(globals()["CYC_PLATEAU"] * _cyc_scale)
    CYC_DECREASE10 = globals()["CYC_DECREASE10"] * _cyc_scale
    # "height" is dead for new runs (real HAG = feat_hag); FEAT_LEGACY keeps it
    # ONLY so pre-spec checkpoints, whose weights expect that width, still infer.
    FEAT_LEGACY = ["intensity", "return_number", "height"]   # infer-only reconstruction
    FEAT_DEFAULT = ["intensity", "return_number"]
    FEAT_SPEC = (list(FEAT_LEGACY) if INFER    # env ignored at infer
                 else tc.parse_feat_spec(FEAT_CHANNELS, FEAT_DEFAULT))

    ds_root = tc.dataset_dir(dataset) if dataset else None
    if ds_root:
        ds_meta, NUM_CLASSES, CLASS_NAMES = tc.load_dataset_meta(dataset)
        # custom feature spec = its own cache family ("" tag keeps old caches valid)
        PREP_DIR = (f"{ds_root}/prep/kpconvx_cold"
                    f"_grid{GRID:g}_c{int(CHUNK_XY)}"
                    f"{tc.feat_spec_tag(FEAT_SPEC, FEAT_LEGACY)}"
                    f"{tc.train_stride_tag()}")
    else:
        # infer mode: geometry + class layout come from the run.json beside the
        # weights; PREP_DIR is an unused placeholder
        NUM_CLASSES = 5
        CLASS_NAMES = [f"class {i}" for i in range(NUM_CLASSES)]
        PREP_DIR = f"{tc.OUTPUTS_ROOT}/_infer_unused"
        if INFER and weights:
            meta = tc.infer_meta(tc.resolve_weights_path(weights))
            if meta:
                NUM_CLASSES = int(meta.get("num_classes") or NUM_CLASSES)
                CLASS_NAMES = list(meta.get("class_names") or
                                   [f"class {i}" for i in range(NUM_CLASSES)])
                if meta.get("grid") is not None: GRID = float(meta["grid"])
                if meta.get("chunk_xy") is not None: CHUNK_XY = float(meta["chunk_xy"])
                STRIDE = CHUNK_XY / 2.0
                # rebuild the exact assembly from run.json; env ignored at infer
                mf = meta.get("features")
                # present-but-malformed features must fail hard, not silently
                # fall back to legacy (surfaces later as a state_dict mismatch)
                FEAT_SPEC = (tc.parse_feat_spec(",".join(mf), FEAT_LEGACY)
                             if mf else list(FEAT_LEGACY))
                if meta.get("hag_source"):
                    # ponytail: TEMPORARY shim (remove once legacy runs retire) —
                    # the deleted --hag variant's 'height' was real HAG; feed
                    # feat_hag in that slot (scenes must be converted with HAG on)
                    FEAT_SPEC = ["feat_hag" if n == "height" else n for n in FEAT_SPEC]
                    print(f"  [legacy-hag] weights from the removed --hag variant "
                          f"(hag_source={meta['hag_source']}): 'height' -> feat_hag; "
                          f"input scenes must carry a baked feat_hag channel",
                          flush=True)

    if "rgb" in FEAT_SPEC:
        raise ValueError("the KPConvX tile pipeline has no rgb channel — use "
                         "intensity (rgb is folded into it when a scene has "
                         "no intensity)")
    IN_CH = 1 + len(FEAT_SPEC)   # bias + spec channels; kp names are width 1

    def _cache_signature():
        # everything that changes cached tile content; mismatch = rebuild
        sp = ds_meta.get("split", {})
        # recipe string reproduces legacy spellings so old caches stay valid
        return {
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
            "num_classes": NUM_CLASSES,   # class reorder/merge invalidates the cache
            "class_names": CLASS_NAMES,
            "feature_recipe": "bias," + ",".join(
                "ret_num" if n == "return_number" else n
                for n in FEAT_SPEC),
        }

    def ensure_prep():
        # train tiles at TT_TRAIN_STRIDE; val/test keep chunk/2 for the voted eval
        return tc.kp_ensure_prep(
            PREP_DIR, ds_root, _cache_signature(),
            lambda name, pc_path, out_dir, split: tc.kp_tile_and_save(
                name, pc_path, out_dir, CHUNK_XY,
                tc.train_stride(CHUNK_XY) if split == "train" else STRIDE,
                GRID, NUM_CLASSES))

    def find_latest_checkpoint():
        # only same-recipe (AdamW + exact feature spec) runs are valid resume targets
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
        # clear stale STOP before the slow prep; a stop clicked during startup survives
        tc.clear_stop()
    train_list, val_list, test_list = ([], [], []) if INFER else ensure_prep()

    resume_info = (find_latest_checkpoint()
                   if (RESUME or AUTO_RESUME or EVAL_ONLY) else None)
    if INFER:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_infer")
        # predictions live beside the input scenes, whatever model produced them
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
            "features": FEAT_SPEC,   # inference rebuilds this exact assembly
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

    # --- model: KPConvX-L preset, train_LAS-matched geometry, random init ----
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

    def lr_at(ep):
        """1-cycle: raise lr0->lr1, hold, then /10 every CYC_DECREASE10 epochs."""
        if ep < CYC_RAISE:
            return CYC_LR0 * (CYC_LR1 / CYC_LR0) ** (ep / CYC_RAISE)
        if ep < CYC_RAISE + CYC_PLATEAU:
            return CYC_LR1
        return CYC_LR1 * 0.1 ** ((ep - CYC_RAISE - CYC_PLATEAU) / CYC_DECREASE10)

    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location="cuda", weights_only=True)
        net.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        print(f"  resumed weights{' + optimizer' if 'optim' in ckpt else ''} "
              f"at epoch {start_epoch}", flush=True)

    if EVAL_ONLY:
        # --weights, else final_model.pth, else the checkpoint loaded above;
        # start_epoch = N_EPOCHS falls straight through to the final eval
        fm = (tc.resolve_weights_path(weights)
              if weights else f"{run_dir}/final_model.pth")
        if weights and not os.path.exists(fm):
            raise FileNotFoundError(f"--weights not found: {fm}")
        if os.path.exists(fm):
            net.load_state_dict(torch.load(fm, map_location="cuda", weights_only=True)["model"])
            print(f"  EVAL-ONLY: loaded {fm}", flush=True)
        start_epoch = N_EPOCHS

    if INFER:
        fm = (tc.resolve_weights_path(weights)
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
        seg_loss = pick_train_tile = None
    else:
        # --- class-balanced loss + rare-class oversampling ------------------
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
        seg_loss = tc.make_seg_loss(class_weights, LABEL_SMOOTH, USE_FOCAL,
                                    FOCAL_GAMMA, LOVASZ_WEIGHT)
        pick_train_tile = tc.make_tile_picker(train_tiles, rare_tiles, RARE_TILE_PROB)

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
        """Pack (xyz, feat, lab) clouds into one lengths-aware pyramid batch."""
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

    SAVE_PROBS = os.environ.get("TT_SAVE_PROBS") == "1"
    EXC_IDX = tc.exclude_class_idx(CLASS_NAMES) if INFER else []
    _predict_points = tc.kp_make_predict_points(
        lambda cxyz, feat: torch.softmax(net(_kp_batch(cxyz, feat)).float(),
                                         -1).cpu().numpy(),
        build_feat, GRID, CHUNK_XY, NUM_CLASSES, DG_INFER_TTA,
        save_probs=SAVE_PROBS, exclude_idx=EXC_IDX)

    # --- inference-only mode (geometry fixed to the trained values) ----------
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
                     "started_utc": datetime.now(timezone.utc).isoformat()}
        if DG_INFER_ADABN:
            # D2b: re-estimate BN running stats on the target tiles (label-free)
            print("  [infer] AdaBN: recomputing BN stats on target tiles...", flush=True)
            dg.adabn_recalibrate(
                net,
                tc.kp_make_target_batches(scenes, _kp_batch, build_feat,
                                          GRID, CHUNK_XY, NUM_CLASSES),
                forward=lambda m, b: m(b))
            net.eval()

        def _predict(pc_path):
            z = np.load(pc_path)
            # scene-local frame for compute; deliverable keeps original coords
            raw = z["xyz"]
            xyz = (raw - np.floor(raw.min(0))).astype(np.float32)
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

    val_csv = f"{run_dir}/val_metrics.csv"
    if not os.path.exists(val_csv):
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "val_acc", "val_miou"] +
                                   [f"iou_{n}" for n in CLASS_NAMES])

    val_items  = [(n, p, c, f"{PREP_DIR}/val")  for n, p, c in val_list]
    test_items = [(n, p, c, f"{PREP_DIR}/test") for n, p, c in test_list]
    print(f"  eval set: {len(val_items)} holdout(val) + {len(test_items)} test scenes",
          flush=True)

    # voted eval on raw points (shared KP-family pipeline)
    def _fwd_eval(tiles):     # [(cxyz, feat)] -> per-tile logits list
        b, _ = make_kp_pack([(c, f, None) for c, f in tiles])
        lg = net(b).cpu().numpy().astype(np.float32)
        return np.split(lg, np.cumsum([len(c) for c, _ in tiles])[:-1])
    evaluate = tc.kp_make_evaluate(_fwd_eval, build_feat, GRID, CHUNK_XY,
                                   NUM_CLASSES, CLASS_NAMES)

    best = tc.BestCheckpoint(run_dir)
    tc.write_run_manifest(run_dir, "kpconvx_cold", dataset)

    def run_eval(ep, write_json=False):
        # val scores current weights; the final call scores TEST on the
        # best-tracked checkpoint (what final_model.pth actually is).
        # PreciseBN: re-estimate BN stats with frozen weights before scoring.
        # Both skipped in EVAL_ONLY — explicit --weights is scored as-shipped.
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

    # --- train loop: EPOCH_STEPS steps of PACK_N tiles x ACCUM forwards ------
    LOG_EVERY = 50
    AMP = os.environ.get("TT_AMP") == "1"   # opt-in bf16 autocast
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
        # GPU-accumulated confusion matrix: one bincount/forward, one sync/epoch
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
                    # skip non-finite loss so one bad gradient can't poison the weights
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

    # final voted eval -> test_metrics.json
    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(ep, write_json=True)
    if not EVAL_ONLY:
        best.finalize(lambda p: torch.save(
            {"model": net.state_dict(), "epoch": ep}, p))
        # DONE marker: AUTO_RESUME skips completed runs
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
    if args.dataset is None and args.mode != 'infer':
        ap.error('--dataset is required (a canonical trainer_gui dataset name); '
                 'only --mode infer may omit it.')
    train_kpconvx(**vars(args))


if __name__ == "__main__":
    main()
