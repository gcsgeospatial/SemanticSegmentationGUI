"""KPConvX-L local trainer, cold start, on a canonical trainer_gui dataset,
using KPConvX's own S3DIS recipe (AdamW + 1-cycle, packed batches, voted eval
over overlapping tiles). Default features [1, intensity, return_number];
FEAT_CHANNELS env overrides and run.json "features" records it. Modes: train | eval
(--weights re-score) | infer (--weights --infer-input, dataset-free).
"""

import os
from typing import Optional


def _kpconvx_root() -> str:
    """KPConvX source root: KPCONVX_SRC or /opt/kpconvx (Modal image)."""
    env = os.environ.get("KPCONVX_SRC")
    for p in (env, "/opt/kpconvx"):
        if p and os.path.isdir(p):
            return p
    if env:
        raise RuntimeError(f"KPCONVX_SRC={env} is not a directory - fix the "
                           "env var or bake the source at /opt/kpconvx")
    raise RuntimeError(
        "KPConvX source not found: set KPCONVX_SRC to your KPConvX Standalone "
        "clone (the pixi env activation sets it) or bake it at /opt/kpconvx")


FEATURE_MODE  = "native"
N_EPOCHS      = 100
EPOCH_STEPS   = 300
PACK_N        = 4
ACCUM         = 2
CHECKPOINT_GAP = 10
VAL_EVERY     = 10

RESUME = False
AUTO_RESUME = os.environ.get("AUTO_RESUME", "0") == "1"

CLASS_WEIGHTING = True
WEIGHT_BETA     = 0.5
WEIGHT_CAP      = 5.0
LOVASZ_WEIGHT   = 1.0
USE_FOCAL       = False
FOCAL_GAMMA     = 2.0
RARE_OVERSAMPLE = True
RARE_CLASSES    = None
RARE_FREQ_FRAC  = 0.5
RARE_TILE_PROB  = 0.25

PROXY_TILES     = 48
PROXY_SAMPLING  = "full"

FEAT_CHANNELS = ""

GRID          = 0.25
KP_RADIUS     = 2.5
RADIUS_SCALING = 2.0

CHUNK_XY      = 25.0    # 100 x GRID: KPConv's in_radius = 50 x dl rule as a side
STRIDE        = 25.0

AUG_COLOR     = 0.8

DG_DENSITY_AUG = False
DG_COARSEN_MAX = 2.5
DG_P_NATIVE    = 0.5
DG_INFER_ADABN = False
DG_INFER_TTA   = 0
DG_LOGDK_FEAT  = False
DG_LOGDK_K     = 8
KP_AGGREGATION = "nearest"
KP_NORM        = "batch"

WEIGHT_DECAY  = 0.05
CYC_LR0       = 1e-4
CYC_LR1       = 5e-3
CYC_RAISE     = 30
CYC_PLATEAU   = 5
CYC_DECREASE10 = 120
LABEL_SMOOTH  = 0.2
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
    (DG_DENSITY_AUG, DG_COARSEN_MAX, DG_P_NATIVE, DG_LOGDK_FEAT, DG_LOGDK_K,
     DG_INFER_ADABN, DG_INFER_TTA, USE_FOCAL, FOCAL_GAMMA, CLASS_WEIGHTING,
     WEIGHT_BETA, RARE_OVERSAMPLE, KP_AGGREGATION, KP_NORM,
     VAL_EVERY, FEAT_CHANNELS, PROXY_SAMPLING) = tc.env_overrides(
        globals(), [
        "DG_DENSITY_AUG", "DG_COARSEN_MAX", "DG_P_NATIVE", "DG_LOGDK_FEAT",
        "DG_LOGDK_K", "DG_INFER_ADABN", "DG_INFER_TTA", "USE_FOCAL",
        "FOCAL_GAMMA", "CLASS_WEIGHTING", "WEIGHT_BETA", "RARE_OVERSAMPLE",
        "KP_AGGREGATION", "KP_NORM", "VAL_EVERY", "FEAT_CHANNELS",
        "PROXY_SAMPLING"])

    sys.path.insert(0, _kpconvx_root())
    EVAL_ONLY = (mode == "eval")
    INFER     = (mode == "infer")

    if dataset is None and not INFER:
        raise ValueError("--dataset is required: pass a canonical trainer_gui "
                         "dataset name (train/val/test folders). The only "
                         "dataset-free path is --mode infer.")

    GRID        = grid if grid is not None else globals()["GRID"]
    CHUNK_XY    = chunk_xy if chunk_xy is not None else globals()["CHUNK_XY"]
    STRIDE      = CHUNK_XY / 2.0
    N_EPOCHS    = epochs if epochs is not None else globals()["N_EPOCHS"]
    EPOCH_STEPS = steps_per_epoch if steps_per_epoch is not None else globals()["EPOCH_STEPS"]
    PACK_N      = batch if batch is not None else globals()["PACK_N"]
    FEATURE_MODE = globals()["FEATURE_MODE"]
    _cyc_scale     = N_EPOCHS / 100.0
    CYC_RAISE      = max(1, round(globals()["CYC_RAISE"] * _cyc_scale))
    CYC_PLATEAU    = round(globals()["CYC_PLATEAU"] * _cyc_scale)
    CYC_DECREASE10 = globals()["CYC_DECREASE10"] * _cyc_scale
    FEAT_DEFAULT = ["intensity", "return_number"]
    FEAT_SPEC = (list(FEAT_DEFAULT) if INFER
                 else tc.parse_feat_spec(FEAT_CHANNELS, FEAT_DEFAULT))

    ds_root = tc.dataset_dir(dataset) if dataset else None
    if ds_root:
        ds_meta, NUM_CLASSES, CLASS_NAMES = tc.load_dataset_meta(dataset)
        PREP_DIR = (f"{ds_root}/prep/kpconvx_cold"
                    f"_grid{GRID:g}_c{int(CHUNK_XY)}"
                    f"{tc.feat_spec_tag(FEAT_SPEC, FEAT_DEFAULT)}"
                    f"{tc.train_stride_tag()}")
    else:
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
                # LEGACY (delete before production): every pre-current weight
                # era routes through legacy_weights - translated tokens, z-min
                # slot, hag_source manifests, era-0 spec-less run_configs
                import legacy_weights as lw
                lspec, lnote = lw.kp_legacy_spec(tc.resolve_weights_path(weights))
                if lnote:
                    print(f"  [legacy] {lnote}", flush=True)
                FEAT_SPEC = lspec if lspec is not None else list(FEAT_DEFAULT)

    if "rgb" in FEAT_SPEC:
        raise ValueError("the KPConvX tile pipeline has no rgb channel. Use "
                         "intensity (rgb is folded into it when a scene has "
                         "no intensity)")
    IN_CH = 1 + len(FEAT_SPEC)

    def _cache_signature():
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
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "feature_recipe": "bias," + ",".join(
                "ret_num" if n == "return_number" else n
                for n in FEAT_SPEC),
        }

    def ensure_prep():
        return tc.kp_ensure_prep(
            PREP_DIR, ds_root, _cache_signature(),
            lambda name, pc_path, out_dir, split: tc.kp_tile_and_save(
                name, pc_path, out_dir, CHUNK_XY,
                tc.train_stride(CHUNK_XY) if split == "train" else STRIDE,
                GRID, NUM_CLASSES))

    def find_latest_checkpoint():
        return tc.kp_find_latest_checkpoint("AdamW", {FEATURE_MODE},
                                            features=FEAT_SPEC,
                                            legacy_features=FEAT_DEFAULT,
                                            skip_done=not EVAL_ONLY)

    print("=" * 70)
    print(f"  KPConvX-L  {dataset or 'infer'}  COLD/{FEATURE_MODE}  "
          f"({tc.gpu_name()}, {N_EPOCHS} ep, {EPOCH_STEPS} steps, "
          f"pack {PACK_N} x accum {ACCUM})")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}  device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    if not INFER:
        tc.clear_stop()
    train_list, val_list, test_list = ([], [], []) if INFER else ensure_prep()

    resume_info = (find_latest_checkpoint()
                   if (RESUME or AUTO_RESUME or EVAL_ONLY) else None)
    if INFER:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_infer")
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

    build_feat = tc.kp_make_build_feat(DG_LOGDK_FEAT, DG_LOGDK_K, FEAT_SPEC)
    if INFER:   # LEGACY (delete before production): z-min / extinct-token columns
        import legacy_weights as lw
        build_feat = lw.wrap_build_feat(build_feat)
    sample_tile = tc.kp_make_sample_tile(
        build_feat, GRID, max_pts=60000, aug_color=AUG_COLOR,
        density_aug=DG_DENSITY_AUG, coarsen_max=DG_COARSEN_MAX,
        p_native=DG_P_NATIVE)
    # curated proxy subset + protocol stamp BEFORE run.json: a pre-signature resume falls through to a fresh run dir, which run.json must then describe
    proxy_tiles = proxy_rep = best = val_tiles = None
    proxy_samples = []
    VAL_RANK = tc.ranking_protocol(PROXY_SAMPLING)
    VAL_FULL = VAL_RANK == "full"
    if not INFER:
        val_tiles = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
        tc.init_val_csv(f"{run_dir}/val_metrics.csv", CLASS_NAMES)
        best = tc.BestCheckpoint(run_dir, VAL_RANK)
    if not INFER and not EVAL_ONLY:
        with tc.fixed_np_seed():
            proxy_tiles, proxy_rep = tc.pick_proxy_tiles(
                val_tiles, NUM_CLASSES, PROXY_TILES, mode=PROXY_SAMPLING,
                class_names=CLASS_NAMES,
                cache_path=f"{PREP_DIR}/val_class_balance_cache.npz",
                viable=lambda p: sample_tile(p, training=False) is not None)
            # subsampled ONCE here, before the prefetch threads exist: sample_tile draws from the global np.random, so re-sampling each val pass would race them and fork the scored point set
            for p in proxy_tiles:
                s = sample_tile(p, training=False)
                if s is None:
                    bn = os.path.basename(p)
                    why = ", ".join(proxy_rep["covers"].get(bn, [])) or "the stride base"
                    raise RuntimeError(
                        f"proxy val tile {bn} (curated for {why}) has fewer than "
                        f"32 points: delete {PREP_DIR} and re-run the dataset prep")
                proxy_samples.append((os.path.basename(p), s))
        print(tc.VAL_FULL_NOTE if VAL_FULL else proxy_rep["text"], flush=True)
        if not tc.proxy_guard(run_dir, proxy_rep, tc.PROXY_PROTOCOL_TILES,
                              CLASS_NAMES, VAL_RANK):
            run_id, run_dir = tc.kp_make_run_dir("kpconvx_cold_native")
            resume_ckpt, start_epoch = None, 0
            tc.init_val_csv(f"{run_dir}/val_metrics.csv", CLASS_NAMES)
            best = tc.BestCheckpoint(run_dir, VAL_RANK)
            tc.proxy_guard(run_dir, proxy_rep, tc.PROXY_PROTOCOL_TILES,
                           CLASS_NAMES, VAL_RANK)

    if resume_ckpt is None and not INFER:
        with open(f"{run_dir}/run.json", "w") as f:
            json.dump({
            "backbone": "KPConvX-L",
            "warm_start": False,
            "feature_mode": FEATURE_MODE,
            "input_channels": IN_CH,
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
    cfg.model.input_channels = IN_CH + (1 if DG_LOGDK_FEAT else 0)
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
    test_tiles  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
    if not INFER:
        print(f"  train_tiles: {len(train_tiles)}   val_tiles: {len(val_tiles)}   "
              f"test_tiles: {len(test_tiles)}", flush=True)
    if not train_tiles and not INFER:
        raise RuntimeError("No training tiles after preprocessing. Check the dataset.")

    if INFER:
        seg_loss = pick_train_tile = None
    else:
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
            print("  [infer] AdaBN: recomputing BN stats on target tiles...", flush=True)
            dg.adabn_recalibrate(
                net,
                tc.kp_make_target_batches(scenes, _kp_batch, build_feat,
                                          GRID, CHUNK_XY, NUM_CLASSES),
                forward=lambda m, b: m(b))
            net.eval()

        def _predict(pc_path):
            z = np.load(pc_path)
            raw = z["xyz"]
            tc.require_finite_xyz(raw, os.path.basename(pc_path))
            xyz = (raw - np.floor(raw.min(0))).astype(np.float32)
            intensity_n, ret_num = tc.scene_arrays(z, len(xyz))
            extras = tc.feat_extras(z, FEAT_SPEC, os.path.basename(pc_path))
            pred, conf, probs = _predict_points(xyz, intensity_n, ret_num,
                                                extras=extras)
            return raw, pred, intensity_n, conf, probs

        tc.run_infer_scenes(scenes, _predict, pred_dir, run_dir, infer_cfg, cls_txt=True)
        return

    metrics_csv = f"{run_dir}/metrics.csv"
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "train_acc", "train_iou", "lr",
                "sec_per_epoch", "gpu_mem_mb",
            ])

    val_csv = f"{run_dir}/val_metrics.csv"

    val_items  = [(n, p, c, f"{PREP_DIR}/val")  for n, p, c in val_list]
    test_items = [(n, p, c, f"{PREP_DIR}/test") for n, p, c in test_list]
    print(f"  eval set: {len(val_items)} holdout(val) + {len(test_items)} test scenes",
          flush=True)

    def _fwd_eval(tiles):
        b, _ = make_kp_pack([(c, f, None) for c, f in tiles])
        lg = net(b).cpu().numpy().astype(np.float32)
        return np.split(lg, np.cumsum([len(c) for c, _ in tiles])[:-1])
    evaluate = tc.kp_make_evaluate(_fwd_eval, build_feat, GRID, CHUNK_XY,
                                   NUM_CLASSES, CLASS_NAMES)

    tc.write_run_manifest(run_dir, "kpconvx_cold", dataset)

    def _proxy_batches():
        for bn, s in proxy_samples:
            try:
                b, lab_t = make_kp_pack([s])
            except Exception as e:
                why = ", ".join(proxy_rep["covers"].get(bn, [])) or "the stride base"
                raise RuntimeError(
                    f"proxy val tile {bn} (curated for {why}) could not build "
                    f"a KPConvX batch ({e}): delete {PREP_DIR} and re-run the "
                    f"dataset prep.") from e
            yield b, lab_t

    def run_eval(ep, write_json=False):
        if not write_json:
            net.eval()
            m = (evaluate(val_items, f"val@ep{ep}") if VAL_FULL else
                 tc.proxy_val(_proxy_batches(), lambda b: net(b),
                              NUM_CLASSES, CLASS_NAMES, f"val@ep{ep}",
                              len(proxy_tiles), tc.PROXY_PROTOCOL_TILES,
                              inventory=proxy_rep["inventory"]))
            net.train()
            if best.update(m):
                tc.atomic_torch_save({"model": net.state_dict(), "epoch": ep},
                                     best.final)
            tc.append_val_row(val_csv, ep, m, CLASS_NAMES)
            return m
        if not EVAL_ONLY:
            def _bn_batches(n=48):
                made = fails = 0
                while made < n:
                    if fails >= 200:
                        raise tc.DatasetExhausted(
                            "PreciseBN: 200 consecutive tile failures - training "
                            "tiles can't build batches; re-run dataset prep")
                    s = sample_tile(pick_train_tile(), training=False)
                    if s is None:
                        fails += 1
                        continue
                    try:
                        b, _ = make_kp_pack([s])
                    except Exception:
                        fails += 1
                        continue
                    fails = 0
                    made += 1
                    yield b
            dg.adabn_recalibrate(net, _bn_batches(), forward=lambda mdl, b: mdl(b))
        net.eval()
        m = evaluate(val_items, f"val@ep{ep}")
        # deliberately no best.update, even in full mode: AdaBN above recalibrates BN, so this row is not comparable to the mid-training ones
        tc.append_val_row(val_csv, ep, m, CLASS_NAMES)
        swapped = not EVAL_ONLY and os.path.exists(best.final)
        if swapped:
            live_state = {k: v.clone() for k, v in net.state_dict().items()}
            net.load_state_dict(torch.load(best.final, map_location="cuda",
                                           weights_only=True)["model"])
            net.eval()
        mt = evaluate(test_items, f"test@ep{ep}")
        if swapped:
            net.load_state_dict(live_state)
        with open(f"{run_dir}/test_metrics.json", "w", encoding="utf-8") as fj:
            json.dump({"val": m, "test": mt,
                       "val_scenes": [n for n, _, _ in val_list],
                       "test_scenes": [n for n, _, _ in test_list]}, fj, indent=2)
        net.train()
        return m

    LOG_EVERY = 50
    AMP = os.environ.get("TT_AMP") == "1"
    def _draw():
        for _ in range(1000):
            s = sample_tile(pick_train_tile(), training=True)
            if s is not None:
                return s
        raise tc.DatasetExhausted(
            "1000 consecutive empty tile draws - training tiles are empty or "
            "too small; re-run dataset prep")
    prefetch = (tc.make_prefetcher(
        lambda: make_kp_pack([_draw() for _ in range(PACK_N)]),
        depth=int(os.environ.get("TT_PREFETCH", "2")))
        if start_epoch < N_EPOCHS else None)
    print(f"  starting at epoch {start_epoch}, up to {N_EPOCHS}, "
          f"{EPOCH_STEPS} steps/epoch, pack {PACK_N} x accum {ACCUM}"
          f"{' [bf16 autocast]' if AMP else ''}", flush=True)
    t_run = time.time()
    ep = N_EPOCHS - 1
    for ep in range(start_epoch, N_EPOCHS):
        cur_lr = lr_at(ep)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        net.train()
        ep_loss = 0.0
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
                    if isinstance(e, (tc.DatasetExhausted, tc.NonFiniteXYZ)):
                        raise
                    # a device-side assert poisons the CUDA context (every later forward fails too), so re-raise with the remedy instead of burning the epoch
                    if (isinstance(e, RuntimeError)
                            and "out of memory" not in str(e)
                            and any(s in str(e) for s in
                                    ("CUDA error", "device-side assert",
                                     "illegal memory access"))):
                        raise RuntimeError(
                            "unrecoverable CUDA error in forward. If it is an "
                            "index/scatter-gather assert, a scene likely has "
                            "non-finite coords: re-ingest the data and delete the "
                            f"prep cache. Original: {e}") from e
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
            raise RuntimeError(f"epoch {ep}: 0 optimizer steps ({n_failed} failed forwards).")
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
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                ep, f"{ep_loss/max(n_fwd,1):.6f}", f"{train_acc:.4f}",
                f"{train_iou:.4f}", f"{cur_lr:.6e}", f"{sec_per_epoch:.2f}", f"{gpu_mem:.1f}",
            ])
        print(f"  ep {ep:3d}: loss={ep_loss/max(n_fwd,1):.4f} acc={train_acc:.4f} "
              f"miou={train_iou:.4f} lr={cur_lr:.2e} s/epoch={sec_per_epoch:.1f}", flush=True)
        if (ep + 1) % CHECKPOINT_GAP == 0:
            tc.atomic_torch_save({"model": net.state_dict(),
                                  "optim": optim.state_dict(), "epoch": ep},
                                 f"{run_dir}/checkpoints/ep{ep:03d}.pth")
        stop = tc.stop_requested(ep)
        if (ep + 1) % VAL_EVERY == 0 and ep != N_EPOCHS - 1 and not stop:
            run_eval(ep)
        if stop:
            break

    if prefetch:
        prefetch.shutdown()

    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(ep, write_json=True)
    if not EVAL_ONLY:
        best.finalize(lambda p: tc.atomic_torch_save(
            {"model": net.state_dict(), "epoch": ep}, p))
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
