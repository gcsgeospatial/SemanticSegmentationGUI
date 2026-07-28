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

CHUNK_XY      = 25.0    # 100 x GRID as the tile side = KPConv's in_radius = 50 x dl rule

AUG_COLOR     = 0.8

DG_DENSITY_AUG = False
DG_COARSEN_MAX = 2.5
DG_P_NATIVE    = 0.5
DG_INFER_ADABN = False
DG_INFER_APCOTTA = False
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
    import os, sys, json, glob
    import numpy as np
    import torch


    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helper"))
    import train_common as tc
    (DG_DENSITY_AUG, DG_COARSEN_MAX, DG_P_NATIVE, DG_LOGDK_FEAT, DG_LOGDK_K,
     DG_INFER_ADABN, DG_INFER_APCOTTA, DG_INFER_TTA, USE_FOCAL, FOCAL_GAMMA,
     CLASS_WEIGHTING, WEIGHT_BETA, RARE_OVERSAMPLE, KP_AGGREGATION, KP_NORM,
     VAL_EVERY, FEAT_CHANNELS, PROXY_SAMPLING) = tc.env_overrides(
        globals(), [
        "DG_DENSITY_AUG", "DG_COARSEN_MAX", "DG_P_NATIVE", "DG_LOGDK_FEAT",
        "DG_LOGDK_K", "DG_INFER_ADABN", "DG_INFER_APCOTTA", "DG_INFER_TTA",
        "USE_FOCAL", "FOCAL_GAMMA", "CLASS_WEIGHTING", "WEIGHT_BETA",
        "RARE_OVERSAMPLE", "KP_AGGREGATION", "KP_NORM", "VAL_EVERY",
        "FEAT_CHANNELS", "PROXY_SAMPLING"])

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
                mf = meta.get("features")
                FEAT_SPEC = list(mf) if mf else list(FEAT_DEFAULT)

    if "rgb" in FEAT_SPEC:
        raise ValueError("the KPConvX tile pipeline has no rgb channel. Use "
                         "intensity (rgb is folded into it when a scene has "
                         "no intensity)")
    IN_CH = 1 + len(FEAT_SPEC)

    def _cache_signature():
        sp = ds_meta.get("split", {})
        # 'bias'/'ret_num' spellings are part of the cache key - changing them invalidates existing prep caches
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

    run_dir, resume_ckpt, start_epoch = tc.kp_resume_ladder(
        INFER, EVAL_ONLY, RESUME or AUTO_RESUME, find_latest_checkpoint,
        infer_input, "kpconvx_cold_native", "AdamW", N_EPOCHS)

    build_feat = tc.kp_make_build_feat(DG_LOGDK_FEAT, DG_LOGDK_K, FEAT_SPEC)
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
            "train_scenes": [n for n, _ in train_list],
            "val_scenes":   [n for n, _ in val_list],
            "test_scenes":  [n for n, _ in test_list],
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

    start_epoch = tc.kp_load_mode_weights(net, optim, resume_ckpt, start_epoch,
                                          EVAL_ONLY, INFER, weights, run_dir,
                                          N_EPOCHS)

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

    _forward = lambda b: net(b)
    SAVE_PROBS = os.environ.get("TT_SAVE_PROBS") == "1"
    EXC_IDX = tc.exclude_class_idx(CLASS_NAMES) if INFER else []
    _predict_points = tc.kp_make_predict_points(
        lambda cxyz, feat: torch.softmax(net(_kp_batch(cxyz, feat)).float(),
                                         -1).cpu().numpy(),
        build_feat, GRID, CHUNK_XY, NUM_CLASSES, DG_INFER_TTA,
        save_probs=SAVE_PROBS, exclude_idx=EXC_IDX)

    if INFER:
        tc.kp_run_infer(run_dir, net, _forward, _kp_batch, build_feat,
                        _predict_points, "KPConvX-L", "KPConvX", weights,
                        infer_input, GRID, CHUNK_XY, grid, chunk_xy,
                        NUM_CLASSES, CLASS_NAMES, FEAT_SPEC, EXC_IDX,
                        DG_INFER_ADABN, infer_apcotta=DG_INFER_APCOTTA)
        return

    metrics_csv = tc.init_metrics_csv(run_dir)

    val_csv = f"{run_dir}/val_metrics.csv"

    val_items  = [(n, p, f"{PREP_DIR}/val")  for n, p in val_list]
    test_items = [(n, p, f"{PREP_DIR}/test") for n, p in test_list]
    print(f"  eval set: {len(val_items)} holdout(val) + {len(test_items)} test scenes",
          flush=True)

    def _fwd_eval(tiles):
        b, _ = make_kp_pack([(c, f, None) for c, f in tiles])
        lg = net(b).cpu().numpy().astype(np.float32)
        return np.split(lg, np.cumsum([len(c) for c, _ in tiles])[:-1])
    evaluate = tc.kp_make_evaluate(_fwd_eval, build_feat, GRID, CHUNK_XY,
                                   NUM_CLASSES, CLASS_NAMES)

    tc.write_run_manifest(run_dir, "kpconvx_cold", dataset)

    run_eval = tc.kp_make_run_eval(
        net, _forward, evaluate, make_kp_pack, sample_tile, pick_train_tile,
        best, val_csv, run_dir, val_items, test_items, val_list, test_list,
        NUM_CLASSES, CLASS_NAMES, VAL_FULL, EVAL_ONLY,
        proxy_batches=lambda: tc.kp_proxy_batches(
            proxy_samples, make_kp_pack, proxy_rep,
            lambda bn, why, e: (
                f"proxy val tile {bn} (curated for {why}) could not build "
                f"a KPConvX batch ({e}): delete {PREP_DIR} and re-run the "
                f"dataset prep.")),
        proxy_tiles=proxy_tiles, proxy_rep=proxy_rep)

    tc.kp_train_loop(
        net, optim, _forward, seg_loss, make_kp_pack, sample_tile,
        pick_train_tile, lr_at, run_eval, best, run_dir, metrics_csv,
        NUM_CLASSES, start_epoch, N_EPOCHS, EPOCH_STEPS, PACK_N, ACCUM,
        CHECKPOINT_GAP, VAL_EVERY, EVAL_ONLY,
        grad_clip_fn=lambda: torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                            GRAD_CLIP))


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
