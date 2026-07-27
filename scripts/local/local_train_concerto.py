"""Shared fine-tuner for the Pointcept-SSL encoders (Concerto/Sonata/Utonia):
pretrained PTv3 encoder + upcast walk + linear seg head on the ptv3 pipeline,
with the sonata/utonia wrappers only swapping the PKG constants. Custom
FEAT_CHANNELS re-initializes the input stem (the pretrained stem needs the
legacy layout); weights are CC-BY-NC 4.0 and checkpoints embed the model config
so --mode infer works offline. Flags: --dataset --grid --chunk-xy --epochs
--batch --steps-per-epoch --freeze-encoder | --mode infer --weights
--infer-input.
"""

import os
from typing import Optional

# sonata/utonia wrappers overwrite these four globals; read via globals() at call time
PKG      = "concerto"
HF_NAME  = "concerto_base"
HF_REPO  = "Pointcept/Concerto"
BB_KEY   = "concerto"

N_EPOCHS      = 100
BATCH_SIZE    = 4

GRID_SIZE     = 0.25
USE_FLASH_ATTN = False
FREEZE_ENCODER = False

FEAT_CHANNELS = ""

DROP_PATH     = 0.3
BASE_LR       = 6e-4
WEIGHT_DECAY  = 5e-3
WARMUP_PCT    = 0.04
GRAD_CLIP     = 1.0

AUG_ENABLE       = True
AUG_ROT_Z        = 1.0
AUG_ROT_XY       = 1.0 / 64.0
AUG_SCALE_MIN    = 0.9
AUG_SCALE_MAX    = 1.1
AUG_FLIP_P       = 0.5
AUG_JITTER_SIGMA = 0.005
AUG_JITTER_CLIP  = 0.02
AUG_COLOR        = 0.8

DG_LOGDK_FEAT  = False
DG_LOGDK_K     = 8

DG_DENSITY_AUG = False
DG_COARSEN_MAX = 2.5
DG_P_NATIVE    = 0.5
DG_INFER_TTA   = 0

CLASS_WEIGHTING  = True
WEIGHT_BETA      = 0.5
WEIGHT_CAP       = 5.0
LABEL_SMOOTH     = 0.0
LOVASZ_WEIGHT    = 1.0
USE_FOCAL        = False
FOCAL_GAMMA      = 2.0

RARE_OVERSAMPLE  = True
RARE_CLASSES     = None
RARE_FREQ_FRAC   = 0.5
RARE_TILE_PROB   = 0.25
RARE_CENTER_PROB = 0.25

VAL_EVERY        = 10
PROXY_TILES      = 48
PROXY_SAMPLING   = "full"
CHECKPOINT_GAP   = 3
RESUME           = False
AUTO_RESUME      = os.environ.get("AUTO_RESUME", "0") == "1"

DATASETS_ROOT = os.environ.get("TT_DATASETS_ROOT", "/datasets")

def train_pcssl(dataset: Optional[str] = None, grid: Optional[float] = None,
                epochs: Optional[int] = None, batch: Optional[int] = None,
                steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
                mode: str = "train", weights: Optional[str] = None,
                infer_input: Optional[str] = None,
                freeze_encoder: Optional[int] = None):
    if dataset is None and mode != "infer":
        raise ValueError("--dataset is required: pass a canonical trainer_gui dataset "
                         "name materialized under /datasets. The only "
                         "dataset-free path is --mode infer.")
    import os, sys, time, json, csv, glob
    from datetime import datetime, timezone
    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helper"))
    import density as dg
    import train_common as tc
    (DG_DENSITY_AUG, DG_COARSEN_MAX, DG_P_NATIVE, DG_LOGDK_FEAT, DG_LOGDK_K,
     DG_INFER_TTA, USE_FOCAL, FOCAL_GAMMA, CLASS_WEIGHTING, WEIGHT_BETA,
     RARE_OVERSAMPLE, RARE_CENTER_PROB, VAL_EVERY, PROXY_SAMPLING,
     FEAT_CHANNELS) = tc.env_overrides(globals(), [
        "DG_DENSITY_AUG", "DG_COARSEN_MAX", "DG_P_NATIVE", "DG_LOGDK_FEAT",
        "DG_LOGDK_K", "DG_INFER_TTA", "USE_FOCAL", "FOCAL_GAMMA",
        "CLASS_WEIGHTING", "WEIGHT_BETA", "RARE_OVERSAMPLE", "RARE_CENTER_PROB",
        "VAL_EVERY", "PROXY_SAMPLING", "FEAT_CHANNELS"])

    PKG, HF_NAME, HF_REPO, BB_KEY = (globals()["PKG"], globals()["HF_NAME"],
                                     globals()["HF_REPO"], globals()["BB_KEY"])
    sys.path.insert(0, os.environ.get(f"{PKG.upper()}_SRC", f"/opt/{PKG}"))

    GRID_SIZE   = grid if grid is not None else globals()["GRID_SIZE"]
    N_EPOCHS    = epochs if epochs is not None else globals()["N_EPOCHS"]
    BATCH_SIZE  = batch if batch is not None else globals()["BATCH_SIZE"]
    STEPS       = steps_per_epoch if steps_per_epoch is not None else 500
    CHUNK_XY    = chunk_xy if chunk_xy is not None else 50.0
    STRIDE      = CHUNK_XY / 2.0
    FREEZE = bool(freeze_encoder if freeze_encoder is not None
                  else globals()["FREEZE_ENCODER"])
    color_src = "intensity"
    FEAT_LEGACY = ["x", "y", "z", "intensity"]
    FEAT_SPEC = list(FEAT_LEGACY)

    if dataset:
        ds_root = tc.dataset_dir(dataset)
        ds_meta, NUM_CLASSES, CLASS_NAMES = tc.load_dataset_meta(dataset)
        if not ds_meta.get("has_intensity"):
            color_src = "rgb" if ds_meta.get("has_rgb") else "gray"
        FEAT_LEGACY = ["x", "y", "z", "rgb" if color_src == "rgb" else "intensity"]
        FEAT_SPEC = (list(FEAT_LEGACY) if mode == "infer"
                     else tc.parse_feat_spec(FEAT_CHANNELS, FEAT_LEGACY))
        tc.ptv3_check_spec(FEAT_SPEC, "this backbone")
        if "rgb" in FEAT_SPEC:
            color_src = "rgb"
        elif "intensity" in FEAT_SPEC and ds_meta.get("has_intensity"):
            color_src = "intensity"
        PREP_DIR = (f"{ds_root}/prep/pcssl_{color_src}"
                    f"{tc.feat_spec_tag(FEAT_SPEC, FEAT_LEGACY)}_chunk{int(CHUNK_XY)}_loc"
                    f"{tc.train_stride_tag()}")

    def _in_ch(spec):
        return (sum(3 if n in ("rgb", "intensity") else 1 for n in spec) + 3
                + (1 if DG_LOGDK_FEAT else 0))
    IN_CH = _in_ch(FEAT_SPEC)

    def load_canonical(npz_path):
        return tc.ptv3_load_canonical(npz_path, color_src)

    import importlib
    _mdl = importlib.import_module(f"{PKG}.model")

    def _upcast_feat(point):
        """Upstream upcast walk: concat each pooling level's features back onto
        its parent -> per-input-point features of dim sum(enc_channels)."""
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point.feat

    def build_feat(cxyz, rgbf, extras=None, drop=()):
        """Spec-ordered features + 3-wide zero normal block + optional log d_k.
        `drop` = spec indices to zero (train-time feature dropout); the normal
        slot and log d_k never drop."""
        cols = []
        for i, n in enumerate(FEAT_SPEC):
            if n in ("rgb", "intensity"):
                c = rgbf
            elif n in ("x", "y", "z"):
                c = cxyz[:, "xyz".index(n):"xyz".index(n) + 1]
            else:
                c = extras[n][:, None]
            cols.append(np.zeros_like(c, dtype=np.float32) if i in drop else c)
        cols.append(np.zeros((len(cxyz), 3), np.float32))
        if DG_LOGDK_FEAT:
            cols.append(dg.local_density_logdk(cxyz, DG_LOGDK_K)[:, None])
        return np.concatenate(cols, axis=1).astype(np.float32)

    def _stem_is_pretrained():
        # only the exact pretraining layout maps onto the pretrained stem
        return (FEAT_SPEC == FEAT_LEGACY and not DG_LOGDK_FEAT)

    def build_model(num_classes, from_config=None):
        """Encoder + linear seg head. Training downloads the pretrained ckpt;
        from_config (embedded in our checkpoints) rebuilds offline.
        Returns (backbone, head, config, stem_pretrained)."""
        stem_pre = _stem_is_pretrained()
        if from_config is not None:
            config = dict(from_config)
            sd = None
        else:
            ckpt = _mdl.load(HF_NAME, repo_id=HF_REPO,
                             download_root=f"{tc.OUTPUTS_ROOT}/hf_cache/{PKG}",
                             ckpt_only=True)
            config = dict(ckpt["config"])
            sd = ckpt["state_dict"]
        n_stages = len(config.get("enc_depths", (3, 3, 3, 12, 3)))
        if not USE_FLASH_ATTN:
            config["enable_flash"] = False
            config["upcast_attention"] = True
            config["upcast_softmax"] = True
            config["enc_patch_size"] = [min(int(s), 1024) for s in
                                        config.get("enc_patch_size",
                                                   [1024] * n_stages)]
        config["in_channels"] = IN_CH
        config["drop_path"] = DROP_PATH
        config["freeze_encoder"] = False
        backbone = _mdl.PointTransformerV3(**config).cuda()
        if sd is not None:
            if not stem_pre:
                sd = {k: v for k, v in sd.items()
                      if not k.startswith("embedding.")}
            missing, unexpected = backbone.load_state_dict(sd, strict=False)
            bad = ([k for k in missing if not k.startswith("embedding.")]
                   + list(unexpected))
            if bad:
                raise RuntimeError(f"pretrained {HF_NAME} did not match the "
                                   f"rebuilt architecture: {bad[:8]}")
            print(f"  loaded pretrained {HF_NAME} "
                  f"({'pretrained' if stem_pre else 'custom (re-initialized)'} "
                  f"input stem, {IN_CH} channels)", flush=True)
        if FREEZE:
            for p in backbone.enc.parameters():
                p.requires_grad = False
            if stem_pre:
                for p in backbone.embedding.parameters():
                    p.requires_grad = False
        head_in = (int(sum(config.get("enc_channels", (48, 96, 192, 384, 512))))
                   if config.get("enc_mode")
                   else int(config.get("dec_channels", (96, 96, 192, 384))[0]))
        head = torch.nn.Linear(head_in, num_classes).cuda()
        return backbone, head, config, stem_pre

    from scipy.spatial import cKDTree

    def make_predict_scene(backbone, head, num_classes, exclude_idx=None):
        SAVE_PROBS = os.environ.get("TT_SAVE_PROBS") == "1"

        def _predict_scene(scene_path):
            xyz, rgb, _ = load_canonical(scene_path)
            z0 = np.load(scene_path)
            ex0 = tc.feat_extras(z0, FEAT_SPEC, os.path.basename(scene_path))
            pred = np.full(len(xyz), -1, np.int64)
            conf = np.zeros(len(xyz), np.float32)
            probs = np.zeros((len(xyz), num_classes), np.float16) if SAVE_PROBS else None
            with torch.no_grad():
                for idx in tc.xy_chunk_groups(xyz, CHUNK_XY, min_pts=64):
                    w0 = (xyz[idx] - xyz[idx].mean(0)).astype(np.float32)
                    rgbf = rgb[idx].astype(np.float32) / 255.0
                    exw = {n: v[idx] for n, v in ex0.items()}
                    views = [1.0] + (list(np.linspace(0.85, 1.2, DG_INFER_TTA))
                                     if DG_INFER_TTA else [])
                    pprob = None
                    for s in views:
                        w = (w0 * s).astype(np.float32)
                        keys = np.floor(w / GRID_SIZE).astype(np.int64)
                        first, inverse = tc.voxel_unique(keys, return_inverse=True)
                        vx = w[first]
                        feat = build_feat(vx, rgbf[first],
                                          {n: v[first] for n, v in exw.items()})
                        coord = torch.from_numpy(vx).cuda()
                        featt = torch.from_numpy(feat).cuda()
                        offset = torch.tensor([len(vx)], dtype=torch.long).cuda()
                        gc = keys[first] - keys[first].min(0)
                        grid_coord = torch.from_numpy(np.ascontiguousarray(gc)).long().cuda()
                        fe = _upcast_feat(backbone({"coord": coord,
                                                    "grid_coord": grid_coord,
                                                    "feat": featt,
                                                    "offset": offset}))
                        vp = torch.softmax(head(fe).float(), -1).cpu().numpy()[inverse]
                        pprob = vp if pprob is None else pprob + vp
                    pprob /= np.maximum(pprob.sum(-1, keepdims=True), 1e-12)
                    pprob = tc.apply_class_mask(pprob, exclude_idx)
                    pred[idx] = pprob.argmax(-1)
                    conf[idx] = pprob.max(-1)
                    if SAVE_PROBS:
                        probs[idx] = pprob.astype(np.float16)
            miss = pred < 0
            if miss.any() and (~miss).any():
                _, nn = cKDTree(xyz[~miss]).query(xyz[miss])
                pred[miss] = pred[~miss][nn]
            elif miss.any():
                pred[:] = min(set(range(num_classes)) - set(exclude_idx or ()))
            return z0["xyz"], pred, rgb[:, 0] / 255.0, conf, probs
        return _predict_scene

    if mode == "infer":
        if not weights or not infer_input:
            raise ValueError("--mode infer requires --weights and --infer-input")
        wpath = tc.resolve_weights_path(weights)
        if not os.path.exists(wpath):
            raise FileNotFoundError(f"weights not found: {wpath}")
        ckpt = tc.load_ckpt_safe(wpath, map_location="cpu")
        bsd, hsd = ckpt["backbone"], ckpt["head"]
        num_classes = int(hsd["weight"].shape[0])
        class_names = [f"class_{i}" for i in range(num_classes)]
        meta = tc.infer_meta(wpath)
        color_src = (meta or {}).get("color_source") or "rgb"
        if meta:
            class_names = meta.get("class_names") or class_names
            if meta.get("grid") is not None:
                GRID_SIZE = float(meta["grid"])
        FEAT_LEGACY = ["x", "y", "z", "rgb" if color_src == "rgb" else "intensity"]
        mf = (meta or {}).get("features")
        if mf:   # LEGACY (delete before production): era feature tokens
            import legacy_weights as lw
            mf = lw.translate_features(mf, hag_source=(meta or {}).get("hag_source"))
        if not mf:
            FEAT_SPEC = list(FEAT_LEGACY)
        else:
            if len(set(mf)) != len(mf):
                raise ValueError(f"run.json 'features' has duplicates: {mf}")
            FEAT_SPEC = tc.parse_feat_spec(",".join(mf), FEAT_LEGACY)
            tc.ptv3_check_spec(FEAT_SPEC, "this backbone")
        IN_CH = _in_ch(FEAT_SPEC)

        if "config" not in ckpt:
            raise ValueError(f"{weights} has no embedded model config "
                             f"(not a local_train_{BB_KEY}.py checkpoint?)")
        backbone, head, model_cfg, stem_pre = build_model(
            num_classes, from_config=ckpt["config"])
        backbone.load_state_dict(bsd)
        head.load_state_dict(hsd)
        backbone.eval(); head.eval()
        print(f"  [infer] loaded {weights} ({num_classes} classes; "
              f"final_model = best-val epoch {ckpt.get('epoch', '?')})", flush=True)

        run_dir = tc.infer_dir(infer_input)
        scenes = sorted(glob.glob(f"{run_dir}/scenes/*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scenes under {run_dir}/scenes")

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_infer")
        pred_dir = os.environ.get("TT_PRED_DIR") or f"{run_dir}/predictions"
        os.makedirs(pred_dir, exist_ok=True)
        exc_idx = tc.exclude_class_idx(class_names)
        infer_cfg = {"backbone": BB_KEY, "mode": "infer", "weights": weights,
                     "pretrained": HF_NAME,
                     "stem": "pretrained" if stem_pre else "custom",
                     "infer_input": infer_input, "num_classes": num_classes,
                     "class_names": class_names, "grid_size": GRID_SIZE,
                     "color_source": color_src, "features": FEAT_SPEC,
                     "chunk_xy": CHUNK_XY, "gpu": tc.gpu_name(),
                     "exclude_classes": [class_names[i] for i in exc_idx],
                     "started_utc": datetime.now(timezone.utc).isoformat()}

        predict_scene = make_predict_scene(backbone, head, num_classes,
                                           exclude_idx=exc_idx)
        tc.run_infer_scenes(scenes, predict_scene, pred_dir, run_dir, infer_cfg)
        return

    print("=" * 70)
    print(f"  {BB_KEY} [{HF_NAME}{', frozen encoder' if FREEZE else ''}]  "
          f"{dataset}  ({tc.gpu_name()}, {N_EPOCHS} ep, batch {BATCH_SIZE})")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    tc.clear_stop()
    tc.ptv3_ensure_prep(PREP_DIR, ds_root, CHUNK_XY, STRIDE, load_canonical)

    def _name(c):
        return CLASS_NAMES[c] if CLASS_NAMES else c
    names = [_name(c) for c in range(NUM_CLASSES)]

    def _in_bounds(xyz):
        return (np.isfinite(xyz).all(1)
                & (np.abs(xyz[:, :2]).max(1) <= CHUNK_XY)
                & (np.abs(xyz[:, 2]) <= 200.0))

    def _viable(p):
        xyz = np.load(p)["xyz"].astype(np.float32)
        xyz = (xyz - xyz.mean(0, keepdims=True, dtype=np.float64)).astype(np.float32)
        return int(_in_bounds(xyz).sum()) >= 64

    val_tiles = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
    VAL_RANK = tc.ranking_protocol(PROXY_SAMPLING)
    VAL_FULL = VAL_RANK == "full"
    proxy_tiles, proxy_rep = tc.pick_proxy_tiles(
        val_tiles, NUM_CLASSES, PROXY_TILES, mode=PROXY_SAMPLING,
        class_names=names, viable=_viable,
        cache_path=f"{PREP_DIR}/val_class_balance_cache.npz")
    print(tc.VAL_FULL_NOTE if VAL_FULL else proxy_rep["text"], flush=True)

    tag = dataset
    _pt = BB_KEY

    _recipe = {"grid_size": GRID_SIZE, "chunk_xy": CHUNK_XY, "features": FEAT_SPEC,
               "n_epochs": N_EPOCHS, "num_classes": NUM_CLASSES,
               "class_names": CLASS_NAMES}
    resume_info = (tc.find_latest_unfinished_run(f"{tag}_{_pt}", _recipe)
                   if (RESUME or AUTO_RESUME) else None)
    if resume_info and not tc.proxy_guard(resume_info[0], proxy_rep,
                                          tc.PROXY_PROTOCOL_TILES, names,
                                          VAL_RANK):
        resume_info = None
    if resume_info:
        run_dir, resume_ckpt, resume_epoch = resume_info
        run_id = os.path.basename(run_dir)
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        start_epoch = resume_epoch + 1
        print(f"  RESUMING {run_id} from {os.path.basename(resume_ckpt)} "
              f"-> epoch {start_epoch}/{N_EPOCHS}", flush=True)
    else:
        run_id = datetime.now(timezone.utc).strftime(f"%Y%m%d_%H%M%S_{tag}_{_pt}")
        run_dir = f"{tc.OUTPUTS_ROOT}/runs/{run_id}"
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        resume_ckpt, start_epoch = None, 0
        with open(f"{run_dir}/run.json", "w") as f:
            cfg = {
                "backbone": BB_KEY, "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
                "pretrained": HF_NAME, "hf_repo": HF_REPO,
                "stem": "pretrained" if _stem_is_pretrained() else "custom",
                "freeze_encoder": FREEZE,
                "dataset": dataset,
                "mode": mode, "gpu": tc.gpu_name(),
                "num_classes": NUM_CLASSES, "grid_size": GRID_SIZE,
                "class_names": CLASS_NAMES,
                "color_source": color_src,
                "features": FEAT_SPEC,
                "in_channels": IN_CH,
                "chunk_xy": CHUNK_XY, "stride": STRIDE,
                "steps_per_epoch": STEPS,
                "flash_attn": USE_FLASH_ATTN,
                "drop_path": DROP_PATH,
                "optimizer": {"type": "AdamW", "base_lr": BASE_LR,
                              "weight_decay": WEIGHT_DECAY, "warmup_pct": WARMUP_PCT,
                              "schedule": "warmup+cosine", "grad_clip": GRAD_CLIP},
                "augmentation": {"enable": AUG_ENABLE, "rot_z": AUG_ROT_Z,
                                 "rot_xy": AUG_ROT_XY,
                                 "scale": [AUG_SCALE_MIN, AUG_SCALE_MAX],
                                 "flip_p": AUG_FLIP_P, "jitter_sigma": AUG_JITTER_SIGMA,
                                 "jitter_clip": AUG_JITTER_CLIP},
                "loss": {"pointwise": "focal" if USE_FOCAL else "weighted_ce",
                         "focal_gamma": FOCAL_GAMMA if USE_FOCAL else None,
                         "class_weighting": CLASS_WEIGHTING, "weight_beta": WEIGHT_BETA,
                         "weight_cap": WEIGHT_CAP, "label_smoothing": LABEL_SMOOTH,
                         "lovasz_softmax_weight": LOVASZ_WEIGHT},
                "class_balance": {"rare_oversample": RARE_OVERSAMPLE,
                                  "rare_classes": RARE_CLASSES,
                                  "rare_freq_frac": RARE_FREQ_FRAC,
                                  "rare_tile_prob": RARE_TILE_PROB,
                                  "rare_center_prob": RARE_CENTER_PROB},
            }
            json.dump(cfg, f, indent=2)

    backbone, head, model_cfg, stem_pre = build_model(NUM_CLASSES)
    n_all = sum(p.numel() for p in backbone.parameters())
    n_train = (sum(p.numel() for p in backbone.parameters() if p.requires_grad)
               + sum(p.numel() for p in head.parameters()))
    print(f"  Params: {n_all:,} ({n_train:,} trainable)")

    def _set_backbone_mode():
        # requires_grad=False alone still lets BN stats update and DropPath fire; eval() is what actually stops both
        backbone.train(not FREEZE)
        if FREEZE and not stem_pre:
            backbone.embedding.train()

    optim = torch.optim.AdamW(
        [p for p in backbone.parameters() if p.requires_grad]
        + list(head.parameters()),
        lr=BASE_LR, weight_decay=WEIGHT_DECAY,
    )
    if resume_ckpt is not None:
        rckpt = torch.load(resume_ckpt, map_location="cuda", weights_only=True)
        backbone.load_state_dict(rckpt["backbone"]); head.load_state_dict(rckpt["head"])
        if "optim" in rckpt:
            optim.load_state_dict(rckpt["optim"])
        print(f"  resumed weights{' + optimizer' if 'optim' in rckpt else ''}", flush=True)

    def _scene_of(p):
        b = os.path.basename(p)
        return b.rsplit("_x", 1)[0]
    train_tiles = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    test_tiles  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
    hold = {_scene_of(p) for p in val_tiles}
    print(f"  train: {len(train_tiles)}   val(holdout {len(hold)} scenes): "
          f"{len(val_tiles)}   test: {len(test_tiles)}", flush=True)

    class_counts, present_mask = tc.scan_class_balance(
        train_tiles, NUM_CLASSES, cache_path=f"{PREP_DIR}/class_balance_cache.npz")

    print(f"  class counts: {dict(zip(names, class_counts.tolist()))}", flush=True)

    if RARE_CLASSES is not None:
        rare_set = set(RARE_CLASSES)
    elif RARE_OVERSAMPLE:
        rare_set = set(tc.auto_rare_classes(class_counts, RARE_FREQ_FRAC))
    else:
        rare_set = set()
    rare_cols = sorted(rare_set)
    if RARE_OVERSAMPLE and rare_cols:
        rare_tiles = [train_tiles[i] for i in np.nonzero(present_mask[:, rare_cols].any(1))[0]]
    else:
        rare_tiles = []
    print(f"  rare classes: {sorted(_name(c) for c in rare_set)}  "
          f"({len(rare_tiles)}/{len(train_tiles)} tiles)", flush=True)

    if CLASS_WEIGHTING:
        w = tc.class_weights_np(class_counts, WEIGHT_BETA, WEIGHT_CAP,
                                absent_to_one=True)
        class_weights = torch.tensor(w, dtype=torch.float32).cuda()
        print(f"  class weights: "
              f"{dict(zip(names, [round(float(x), 3) for x in w]))}", flush=True)
    else:
        class_weights = None

    seg_loss = tc.make_seg_loss(class_weights, LABEL_SMOOTH, USE_FOCAL,
                                FOCAL_GAMMA, LOVASZ_WEIGHT)
    pick_train_tile = tc.make_tile_picker(train_tiles, rare_tiles,
                                          RARE_TILE_PROB)

    def augment_xyz(xyz):
        return tc.ptv3_augment_xyz(xyz, AUG_ROT_Z, AUG_ROT_XY, AUG_SCALE_MIN,
                                   AUG_SCALE_MAX, AUG_FLIP_P,
                                   AUG_JITTER_SIGMA, AUG_JITTER_CLIP)

    def to_ptv3_batch(tiles_for_batch, training=True):
        coords, feats, labels, offsets, grid_coords = [], [], [], [], []
        running = 0
        for tile in tiles_for_batch:
            z = np.load(tile)
            xyz, rgb, lab = z["xyz"], z["rgb"], z["lab"]
            ex = tc.feat_extras(z, FEAT_SPEC, os.path.basename(tile))
            if len(xyz) > 80000:
                if not training:
                    idx = np.arange(0, len(xyz), -(-len(xyz) // 80000))
                else:
                    c = None
                    if (RARE_OVERSAMPLE and rare_cols
                            and np.random.rand() < RARE_CENTER_PROB):
                        ridx = np.where(np.isin(lab, rare_cols))[0]
                        if len(ridx):
                            c = xyz[ridx[np.random.randint(len(ridx))]]
                    if c is None:
                        c = xyz[np.random.randint(len(xyz))]
                    d2 = np.sum((xyz[:, :2] - c[:2]) ** 2, axis=1)
                    idx = np.where(d2 < 15.0 ** 2)[0]
                    if len(idx) > 80000:
                        idx = np.random.choice(idx, 80000, replace=False)
                xyz, rgb, lab = xyz[idx], rgb[idx], lab[idx]
                ex = {n: v[idx] for n, v in ex.items()}
            xyz = xyz.astype(np.float32)
            if training and AUG_ENABLE:
                xyz = augment_xyz(xyz)
            # float64 mean: a float32 mean at UTM magnitudes empties the window cut
            xyz = (xyz - xyz.mean(0, keepdims=True, dtype=np.float64)
                   ).astype(np.float32)
            ok = _in_bounds(xyz)
            if int(ok.sum()) < 64:
                continue
            xyz = xyz[ok]; rgb = rgb[ok]; lab = lab[ok]
            ex = {n: v[ok] for n, v in ex.items()}
            # grid_coord MUST come from the same keys used to dedup: a different phase can collapse two voxels onto one grid_coord (CUDA assert)
            g_eff = (dg.effective_grid(GRID_SIZE, DG_COARSEN_MAX, DG_P_NATIVE)
                     if (training and DG_DENSITY_AUG) else GRID_SIZE)
            keys = np.floor(xyz / g_eff).astype(np.int64)
            uniq = tc.voxel_unique(keys)
            xyz = xyz[uniq]; rgb = rgb[uniq]; lab = lab[uniq]
            ex = {n: v[uniq] for n, v in ex.items()}
            xyz = xyz.astype(np.float32)
            fdrop = (np.flatnonzero(np.random.rand(len(FEAT_SPEC)) > AUG_COLOR)
                     if training else ())
            feat = build_feat(xyz, rgb.astype(np.float32) / 255.0, ex, drop=fdrop)
            coords.append(xyz); feats.append(feat); labels.append(lab)
            grid_coords.append(keys[uniq])
            running += len(xyz)
            offsets.append(running)
        coord = torch.from_numpy(np.concatenate(coords).astype(np.float32)).cuda()
        feat  = torch.from_numpy(np.concatenate(feats).astype(np.float32)).cuda()
        label = torch.from_numpy(np.concatenate(labels).astype(np.int64)).cuda()
        offset = torch.tensor(offsets, dtype=torch.long).cuda()
        gc = np.concatenate(grid_coords)
        gc -= gc.min(0, keepdims=True)
        grid_coord = torch.from_numpy(np.ascontiguousarray(gc)).long().cuda()
        return {"coord": coord, "grid_coord": grid_coord,
                "feat": feat, "offset": offset}, label

    metrics_csv = f"{run_dir}/metrics.csv"
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss", "train_acc", "val_acc",
                "train_iou", "val_iou", "lr", "sec_per_iter", "sec_per_epoch",
                "gpu_mem_mb",
            ])

    def _raw_loader(split, name):
        return lambda: tc.load_xyz_label(f"{ds_root}/{split}/{name}.npz")

    val_items = [(n, _raw_loader("val", n), f"{PREP_DIR}/val") for n in sorted(hold)]
    test_items = [(n, _raw_loader("test", n), f"{PREP_DIR}/test")
                  for n in sorted({_scene_of(p) for p in test_tiles})]
    print(f"  eval set: {len(val_items)} holdout(val) + {len(test_items)} test scenes",
          flush=True)

    evaluate = tc.ptv3_make_evaluate(
        lambda batch: head(_upcast_feat(backbone(batch))), build_feat,
        FEAT_SPEC, GRID_SIZE, CHUNK_XY, NUM_CLASSES, names)

    val_csv = f"{run_dir}/val_metrics.csv"
    tc.init_val_csv(val_csv, names)

    best = tc.BestCheckpoint(run_dir, VAL_RANK)
    tc.proxy_guard(run_dir, proxy_rep, tc.PROXY_PROTOCOL_TILES, names, VAL_RANK)
    tc.write_run_manifest(run_dir, _pt, dataset)

    def _proxy_batches():
        for i in range(0, len(proxy_tiles), BATCH_SIZE):
            chunk = proxy_tiles[i:i + BATCH_SIZE]
            batch, lab = to_ptv3_batch(chunk, training=False)
            if len(batch["offset"]) != len(chunk):
                bad = ([os.path.basename(t) for t in chunk if not _viable(t)]
                       or [os.path.basename(t) for t in chunk])
                raise RuntimeError(
                    f"proxy val tile(s) {bad} were dropped at batch time (<64 "
                    f"points within chunk_xy={CHUNK_XY}m of the tile centre); "
                    f"they cover {[proxy_rep['covers'].get(n, []) for n in bad]}, "
                    f"whose score would silently vanish from the ranking metric. "
                    f"Delete {PREP_DIR} and relaunch to re-prep the val tiles.")
            yield batch, lab

    def _save_best(ep):
        tc.atomic_torch_save({"backbone": backbone.state_dict(),
                              "head": head.state_dict(), "epoch": ep,
                              "config": model_cfg}, best.final)

    def run_eval(ep, write_json=False):
        backbone.eval(); head.eval()
        if not write_json:
            m = (evaluate(val_items, f"val@ep{ep}") if VAL_FULL else
                 tc.proxy_val(_proxy_batches(),
                              lambda batch: head(_upcast_feat(backbone(batch))),
                              NUM_CLASSES, names, f"val@ep{ep}",
                              len(proxy_tiles), tc.PROXY_PROTOCOL_TILES,
                              inventory=proxy_rep["inventory"]))
            # weights before the csv row: a kill between them must not seed a best final_model.pth can never match
            if best.update(m):
                _save_best(ep)
            tc.append_val_row(val_csv, ep, m, names)
            _set_backbone_mode(); head.train()
            return m
        m = evaluate(val_items, f"val@ep{ep}")
        # deliberately no best.update: the last epoch is never a crown candidate, matching the AdaBN trainers
        tc.append_val_row(val_csv, ep, m, names)
        swapped = os.path.exists(best.final)
        if swapped:
            live_backbone = {k: v.clone() for k, v in backbone.state_dict().items()}
            live_head = {k: v.clone() for k, v in head.state_dict().items()}
            bckpt = torch.load(best.final, map_location="cuda", weights_only=True)
            backbone.load_state_dict(bckpt["backbone"]); head.load_state_dict(bckpt["head"])
            backbone.eval(); head.eval()
        mt = evaluate(test_items, f"test@ep{ep}")
        if swapped:
            backbone.load_state_dict(live_backbone); head.load_state_dict(live_head)
        with open(f"{run_dir}/test_metrics.json", "w", encoding="utf-8") as fj:
            json.dump({"val": m, "test": mt,
                       "val_scenes": [n for n, _, _ in val_items],
                       "test_scenes": [n for n, _, _ in test_items]}, fj, indent=2)
        _set_backbone_mode(); head.train()
        return m

    LOG_EVERY = 20
    AMP = os.environ.get("TT_AMP") == "1"
    prefetch = (tc.make_prefetcher(
        lambda: to_ptv3_batch([pick_train_tile() for _ in range(BATCH_SIZE)],
                              training=True),
        depth=int(os.environ.get("TT_PREFETCH", "2")))
        if start_epoch < N_EPOCHS else None)
    print(f"  starting at epoch {start_epoch}, up to {N_EPOCHS}, "
          f"{STEPS} steps/epoch (batch {BATCH_SIZE})"
          f"{' [bf16 autocast]' if AMP else ''}", flush=True)
    t_run = time.time()
    ep = N_EPOCHS - 1
    for ep in range(start_epoch, N_EPOCHS):
        cur_lr = tc.ptv3_lr_at(ep, BASE_LR, WARMUP_PCT, N_EPOCHS)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        _set_backbone_mode(); head.train()
        ep_loss = 0.0
        ep_conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long,
                              device="cuda")
        t_ep = time.time(); t_chunk = t_ep; n_steps = 0; last_log_step = 0
        n_oom = 0
        print(f"  ep {ep:3d} starting (lr={cur_lr:.2e})…", flush=True)
        for step in range(STEPS):
            try:
                batch, label = prefetch()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                    logits = head(_upcast_feat(backbone(batch)))
                    loss = seg_loss(logits, label)
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()), GRAD_CLIP)
                optim.step()
                ep_loss += loss.item(); n_steps += 1
                if n_steps % LOG_EVERY == 0:
                    dt = time.time() - t_chunk
                    print(f"    ep {ep:3d} step {n_steps:4d}: "
                          f"loss={ep_loss/n_steps:.4f} "
                          f"{(n_steps-last_log_step)/max(dt,1e-6):.2f} it/s", flush=True)
                    t_chunk = time.time(); last_log_step = n_steps
                pred = logits.argmax(-1)
                m = label >= 0
                ep_conf += torch.bincount(
                    label[m] * NUM_CLASSES + pred[m],
                    minlength=NUM_CLASSES * NUM_CLASSES,
                ).reshape(NUM_CLASSES, NUM_CLASSES)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    n_oom += 1
                    torch.cuda.empty_cache(); continue
                raise
        if n_steps == 0:
            raise RuntimeError(f"epoch {ep}: 0 optimizer steps ({n_oom} OOM steps); "
                               f"lower --batch or --chunk-xy.")
        if n_oom:
            print(f"  ep {ep:3d} note: {n_oom} OOM steps skipped", flush=True)
        sec_per_iter = (time.time() - t_ep) / max(n_steps, 1)
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
                ep, ep_loss / max(n_steps, 1), "", f"{train_acc:.4f}", "",
                f"{train_iou:.4f}", "", f"{cur_lr:.6e}", f"{sec_per_iter:.4f}",
                f"{sec_per_epoch:.2f}", f"{gpu_mem:.1f}",
            ])
        print(f"  ep {ep:3d}: loss={ep_loss/max(n_steps,1):.4f} "
              f"acc={train_acc:.4f} miou={train_iou:.4f} lr={cur_lr:.2e} "
              f"s/iter={sec_per_iter:.3f} s/ep={sec_per_epoch:.1f}", flush=True)
        if (ep + 1) % CHECKPOINT_GAP == 0 or ep == N_EPOCHS - 1:
            tc.atomic_torch_save({"backbone": backbone.state_dict(),
                                  "head": head.state_dict(),
                                  "optim": optim.state_dict(), "epoch": ep,
                                  "config": model_cfg},
                                 f"{run_dir}/checkpoints/ep{ep:03d}.pth")
            for old in sorted(glob.glob(f"{run_dir}/checkpoints/ep*.pth"))[:-2]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        stop = tc.stop_requested(ep)
        if (ep + 1) % VAL_EVERY == 0 and ep != N_EPOCHS - 1 and not stop:
            run_eval(ep)
        if stop:
            break

    if prefetch:
        prefetch.shutdown()

    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(ep, write_json=True)
    best.finalize(lambda p: _save_best(ep))
    print(f"  total wall-clock {(time.time() - t_run)/3600:.2f} h")

    open(f"{run_dir}/DONE", "w").close()
    print(f"  run complete -> {run_id}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Local Pointcept-SSL (concerto/sonata/utonia) fine-tuner/inferencer.')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--grid', type=float, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--steps-per-epoch', type=int, default=None)
    ap.add_argument('--chunk-xy', type=float, default=None)
    ap.add_argument('--mode', default='train')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--infer-input', default=None)
    ap.add_argument('--freeze-encoder', type=int, default=None,
                    help='1 = linear probe: freeze the pretrained encoder, '
                         'train only the seg head (0 = full fine-tune)')
    args = ap.parse_args()
    train_pcssl(**vars(args))


if __name__ == "__main__":
    main()
