"""PTv3-family local trainer on a canonical trainer_gui dataset. The base mode
is plain PTv3 (standalone model.py, standard attention) on PTv3's published
outdoor recipe; the concerto/sonata/utonia wrappers overwrite the PKG/HF
globals to fine-tune a pretrained Pointcept-SSL encoder (upcast walk + linear
seg head, CC-BY-NC 4.0 weights, checkpoints embed the model config so --mode
infer works offline). Color slot prefers intensity over RGB (run.json
"color_source" keeps old RGB checkpoints honest); FEAT_CHANNELS env overrides
the input spec, run.json "features" records it. Custom FEAT_CHANNELS
re-initializes a pretrained input stem (it needs the default layout).
Flags: --dataset --grid --chunk-xy --epochs --batch --steps-per-epoch
[--freeze-encoder] | --mode infer --weights --infer-input <job_id>.
"""

import os
from typing import Optional

# pcssl wrappers overwrite these; HF_NAME None = plain cold-start PTv3
PKG      = None
HF_NAME  = None
HF_REPO  = None
BB_KEY   = "ptv3"

N_EPOCHS      = 100
BATCH_SIZE    = 4

GRID_SIZE     = 0.25
USE_FLASH_ATTN = False
FREEZE_ENCODER = False

FEAT_CHANNELS = ""

DROP_PATH     = 0.3
BASE_LR       = 2e-3
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
AUTO_RESUME      = os.environ.get("AUTO_RESUME", "0") == "1"

def train_ptv3(dataset: Optional[str] = None, grid: Optional[float] = None,
               epochs: Optional[int] = None, batch: Optional[int] = None,
               steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
               mode: str = "train", weights: Optional[str] = None,
               infer_input: Optional[str] = None,
               freeze_encoder: Optional[int] = None):
    if dataset is None and mode != "infer":
        raise ValueError("--dataset is required: pass a canonical trainer_gui dataset "
                         "name materialized under /datasets. The only "
                         "dataset-free path is --mode infer.")
    import os, sys, json, glob
    from datetime import datetime, timezone
    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helper"))
    import train_common as tc
    (DG_DENSITY_AUG, DG_COARSEN_MAX, DG_P_NATIVE, DG_LOGDK_FEAT, DG_LOGDK_K,
     DG_INFER_TTA, USE_FOCAL, FOCAL_GAMMA, CLASS_WEIGHTING, WEIGHT_BETA,
     RARE_OVERSAMPLE, VAL_EVERY,
     FEAT_CHANNELS, PROXY_SAMPLING) = tc.env_overrides(globals(), [
        "DG_DENSITY_AUG", "DG_COARSEN_MAX", "DG_P_NATIVE", "DG_LOGDK_FEAT",
        "DG_LOGDK_K", "DG_INFER_TTA", "USE_FOCAL", "FOCAL_GAMMA",
        "CLASS_WEIGHTING", "WEIGHT_BETA", "RARE_OVERSAMPLE",
        "VAL_EVERY", "FEAT_CHANNELS", "PROXY_SAMPLING"])

    # wrappers overwrite the PKG globals per call - read them here, not at import
    PKG, HF_NAME, HF_REPO, BB_KEY = (globals()["PKG"], globals()["HF_NAME"],
                                     globals()["HF_REPO"], globals()["BB_KEY"])
    pretrained = HF_NAME is not None
    if pretrained:
        sys.path.insert(0, os.environ.get(f"{PKG.upper()}_SRC", f"/opt/{PKG}"))
    else:
        sys.path.insert(0, os.path.dirname(os.environ.get("PTV3_SRC", "/opt/ptv3")))
    spec_label = "this backbone" if pretrained else "PTv3"

    GRID_SIZE   = grid if grid is not None else globals()["GRID_SIZE"]
    N_EPOCHS    = epochs if epochs is not None else globals()["N_EPOCHS"]
    BATCH_SIZE  = batch if batch is not None else globals()["BATCH_SIZE"]
    STEPS       = steps_per_epoch if steps_per_epoch is not None else 500
    CHUNK_XY    = chunk_xy if chunk_xy is not None else 50.0
    STRIDE      = CHUNK_XY / 2.0
    FREEZE = bool(freeze_encoder if freeze_encoder is not None
                  else globals()["FREEZE_ENCODER"])
    color_src = "intensity"
    FEAT_DEFAULT = ["x", "y", "z", "intensity"]
    FEAT_SPEC = list(FEAT_DEFAULT)

    if dataset:
        ds_root = tc.dataset_dir(dataset)
        ds_meta, NUM_CLASSES, CLASS_NAMES = tc.load_dataset_meta(dataset)
        if not ds_meta.get("has_intensity"):
            color_src = "rgb" if ds_meta.get("has_rgb") else "gray"
        FEAT_DEFAULT = ["x", "y", "z", "rgb" if color_src == "rgb" else "intensity"]
        FEAT_SPEC = (list(FEAT_DEFAULT) if mode == "infer"
                     else tc.parse_feat_spec(FEAT_CHANNELS, FEAT_DEFAULT))
        tc.ptv3_check_spec(FEAT_SPEC, spec_label)
        if "rgb" in FEAT_SPEC:
            color_src = "rgb"
        elif "intensity" in FEAT_SPEC and ds_meta.get("has_intensity"):
            color_src = "intensity"
        PREP_DIR = (f"{ds_root}/prep/{'pcssl' if pretrained else 'ptv3'}_{color_src}"
                    f"{tc.feat_spec_tag(FEAT_SPEC, FEAT_DEFAULT)}_chunk{int(CHUNK_XY)}_loc"
                    f"{tc.train_stride_tag()}")

    def _in_ch(spec):
        # a pretrained stem expects a trailing 3-wide (zeroed) normal slot
        return (sum(3 if n in ("rgb", "intensity") else 1 for n in spec)
                + (3 if pretrained else 0) + (1 if DG_LOGDK_FEAT else 0))
    IN_CH = _in_ch(FEAT_SPEC)

    def load_canonical(npz_path):
        return tc.ptv3_load_canonical(npz_path, color_src)

    if pretrained:
        import importlib
        _mdl = importlib.import_module(f"{PKG}.model")
    else:
        from ptv3.model import PointTransformerV3
        # PTv3's 5x5x5 stem needs the spconv-cu118 build; cu124's backward asserts

    def _upcast_feat(point):
        """Upstream upcast walk: concat each pooling level's features back onto
        its parent -> per-input-point features of dim sum(enc_channels)."""
        while "pooling_parent" in point.keys():
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point.feat

    def _forward_logits(batch):
        if pretrained:
            return head(_upcast_feat(backbone(batch)))
        point = backbone(batch)
        return head(point["feat"] if isinstance(point, dict) else point.feat)

    def build_feat(cxyz, rgbf, extras=None, drop=()):
        return tc.ptv3_build_feat(FEAT_SPEC, cxyz, rgbf, extras, drop,
                                  DG_LOGDK_FEAT, DG_LOGDK_K,
                                  normal_block=pretrained)

    def _stem_is_pretrained():
        # only the exact pretraining layout maps onto the pretrained stem
        return pretrained and (FEAT_SPEC == FEAT_DEFAULT and not DG_LOGDK_FEAT)

    def build_model(num_classes, from_config=None):
        """(backbone, head, model_cfg, stem_pre). Plain PTv3 builds the fixed
        published arch (model_cfg None); pretrained downloads the HF checkpoint
        (or rebuilds offline from a config embedded in our checkpoints)."""
        if not pretrained:
            backbone = PointTransformerV3(
                in_channels=IN_CH,
                order=("z", "z-trans", "hilbert", "hilbert-trans"),
                stride=(2, 2, 2, 2),
                enc_depths=(2, 2, 2, 6, 2),
                enc_channels=(32, 64, 128, 256, 512),
                enc_num_head=(2, 4, 8, 16, 32),
                enc_patch_size=(1024, 1024, 1024, 1024, 1024),
                dec_depths=(2, 2, 2, 2),
                dec_channels=(64, 64, 128, 256),
                dec_num_head=(4, 4, 8, 16),
                dec_patch_size=(1024, 1024, 1024, 1024),
                drop_path=DROP_PATH,
                enable_flash=USE_FLASH_ATTN,
                cls_mode=False,
            ).cuda()
            head = torch.nn.Linear(64, num_classes).cuda()
            return backbone, head, None, False
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
        FEAT_DEFAULT = ["x", "y", "z", "rgb" if color_src == "rgb" else "intensity"]
        mf = (meta or {}).get("features")
        if not mf:
            FEAT_SPEC = list(FEAT_DEFAULT)
        else:
            if len(set(mf)) != len(mf):
                raise ValueError(f"run.json 'features' has duplicates: {mf}")
            FEAT_SPEC = tc.parse_feat_spec(",".join(mf), FEAT_DEFAULT)
            tc.ptv3_check_spec(FEAT_SPEC, spec_label)
        IN_CH = _in_ch(FEAT_SPEC)

        if pretrained and "config" not in ckpt:
            raise ValueError(f"{weights} has no embedded model config "
                             f"(not a local_train_{BB_KEY}.py checkpoint?)")
        backbone, head, model_cfg, stem_pre = build_model(
            num_classes, from_config=ckpt.get("config"))
        backbone.load_state_dict(bsd)
        head.load_state_dict(hsd)
        backbone.eval(); head.eval()
        print(f"  [infer] loaded {weights} ({num_classes} classes; "
              f"final_model = best-val epoch {ckpt.get('epoch', '?')})", flush=True)

        run_dir = tc.infer_dir(infer_input)
        scenes = tc.infer_scenes(run_dir)
        if not scenes:
            raise FileNotFoundError(f"No staged *_input.npz scenes in {run_dir}")

        pred_dir = os.environ.get("TT_PRED_DIR") or f"{run_dir}/predictions"
        os.makedirs(pred_dir, exist_ok=True)
        exc_idx = tc.exclude_class_idx(class_names)
        infer_cfg = {"backbone": BB_KEY if pretrained else "PTv3",
                     "mode": "infer", "weights": weights,
                     "infer_input": infer_input, "num_classes": num_classes,
                     "class_names": class_names, "grid_size": GRID_SIZE,
                     "color_source": color_src, "features": FEAT_SPEC,
                     "chunk_xy": CHUNK_XY, "gpu": tc.gpu_name(),
                     "exclude_classes": [class_names[i] for i in exc_idx],
                     "started_utc": datetime.now(timezone.utc).isoformat()}
        if pretrained:
            infer_cfg["pretrained"] = HF_NAME
            infer_cfg["stem"] = "pretrained" if stem_pre else "custom"

        predict_scene = tc.ptv3_make_predict_scene(
            _forward_logits, load_canonical, build_feat, FEAT_SPEC, GRID_SIZE,
            CHUNK_XY, DG_INFER_TTA, num_classes, exclude_idx=exc_idx)
        tc.run_infer_scenes(scenes, predict_scene, pred_dir, run_dir, infer_cfg)
        return

    print("=" * 70)
    if pretrained:
        print(f"  {BB_KEY} [{HF_NAME}{', frozen encoder' if FREEZE else ''}]  "
              f"{dataset}  ({tc.gpu_name()}, {N_EPOCHS} ep, batch {BATCH_SIZE})")
    else:
        print(f"  PTv3  {dataset}  ({tc.gpu_name()}, {N_EPOCHS} ep, batch {BATCH_SIZE})")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    tc.preflight_env()
    tc.clear_stop()
    tc.ptv3_ensure_prep(PREP_DIR, ds_root, CHUNK_XY, STRIDE, load_canonical)

    # the dataset stage decided the 3-way split - read verbatim, never re-carve
    train_tiles = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    val_tiles   = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
    test_tiles  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
    hold = {tc.ptv3_scene_of(p) for p in val_tiles}
    print(f"  train: {len(train_tiles)}   val(holdout {len(hold)} scenes): "
          f"{len(val_tiles)}   test: {len(test_tiles)}", flush=True)

    def _name(c):
        return CLASS_NAMES[c] if CLASS_NAMES else c
    names = [_name(c) for c in range(NUM_CLASSES)]

    def _viable(p):
        return tc.ptv3_viable(p, CHUNK_XY)

    VAL_RANK = tc.ranking_protocol(PROXY_SAMPLING)
    VAL_FULL = VAL_RANK == "full"
    proxy_tiles, proxy_rep = tc.pick_proxy_tiles(
        val_tiles, NUM_CLASSES, PROXY_TILES, mode=PROXY_SAMPLING,
        class_names=names, cache_path=f"{PREP_DIR}/val_class_balance_cache.npz",
        viable=_viable)
    print(tc.VAL_FULL_NOTE if VAL_FULL else proxy_rep["text"], flush=True)

    _recipe = {"grid_size": GRID_SIZE, "chunk_xy": CHUNK_XY, "features": FEAT_SPEC,
               "n_epochs": N_EPOCHS, "num_classes": NUM_CLASSES,
               "class_names": CLASS_NAMES}
    run_dir, run_id, resume_ckpt, start_epoch = tc.resume_ladder(
        f"{dataset}_{BB_KEY}", _recipe, AUTO_RESUME, proxy_rep,
        tc.PROXY_PROTOCOL_TILES, names, VAL_RANK, N_EPOCHS)
    if resume_ckpt is None:
        with open(f"{run_dir}/run.json", "w") as f:
            cfg = {
                "backbone": BB_KEY if pretrained else "PTv3",
                "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
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
            if pretrained:
                cfg.update({"pretrained": HF_NAME, "hf_repo": HF_REPO,
                            "stem": "pretrained" if _stem_is_pretrained() else "custom",
                            "freeze_encoder": FREEZE})
            json.dump(cfg, f, indent=2)

    backbone, head, model_cfg, stem_pre = build_model(NUM_CLASSES)
    if pretrained:
        n_all = sum(p.numel() for p in backbone.parameters())
        n_train = (sum(p.numel() for p in backbone.parameters() if p.requires_grad)
                   + sum(p.numel() for p in head.parameters()))
        print(f"  Params: {n_all:,} ({n_train:,} trainable)")
    else:
        print(f"  Params: {sum(p.numel() for p in backbone.parameters()):,}")

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

    to_ptv3_batch = tc.ptv3_make_batcher(
        build_feat, FEAT_SPEC, GRID_SIZE, CHUNK_XY, augment_xyz, AUG_ENABLE,
        AUG_COLOR, RARE_OVERSAMPLE, rare_cols, RARE_CENTER_PROB,
        DG_DENSITY_AUG, DG_COARSEN_MAX, DG_P_NATIVE)

    metrics_csv = tc.init_metrics_csv(run_dir, names)

    val_items, test_items = tc.ptv3_eval_items(ds_root, PREP_DIR, hold, test_tiles)

    evaluate = tc.ptv3_make_evaluate(_forward_logits, build_feat, FEAT_SPEC,
                                     GRID_SIZE, CHUNK_XY, NUM_CLASSES, names)

    val_csv = metrics_csv

    best = tc.BestCheckpoint(run_dir, VAL_RANK)
    tc.proxy_guard(run_dir, proxy_rep, tc.PROXY_PROTOCOL_TILES, names, VAL_RANK)
    tc.write_run_manifest(run_dir, BB_KEY, dataset)

    _proxy_batches = tc.ptv3_proxy_batches(proxy_tiles, BATCH_SIZE,
                                           to_ptv3_batch, _viable, proxy_rep,
                                           CHUNK_XY, PREP_DIR)

    def _save_best(ep, extra=None):
        state = {"backbone": backbone.state_dict(),
                 "head": head.state_dict(), "epoch": ep, **(extra or {})}
        if pretrained:
            state["config"] = model_cfg
        tc.atomic_torch_save(state, best.final)

    def _set_train_mode():
        if pretrained:
            _set_backbone_mode()
        else:
            backbone.train()
        head.train()

    run_eval = tc.ptv3_make_run_eval(
        backbone, head, evaluate, _forward_logits, _proxy_batches, proxy_tiles,
        proxy_rep, VAL_FULL, val_items, test_items, val_csv, best, _save_best,
        _set_train_mode, NUM_CLASSES, names, run_dir)

    tc.ptv3_train_loop(
        backbone, head, optim, seg_loss, _forward_logits,
        lambda: to_ptv3_batch([pick_train_tile() for _ in range(BATCH_SIZE)],
                              training=True),
        run_eval, best, _save_best, _set_train_mode, run_dir, run_id,
        metrics_csv, start_epoch, N_EPOCHS, STEPS, BATCH_SIZE, NUM_CLASSES,
        BASE_LR, WARMUP_PCT, GRAD_CLIP, CHECKPOINT_GAP, VAL_EVERY,
        ckpt_extra={"config": model_cfg} if pretrained else None)


# the pcssl wrappers import this module and call train_pcssl/main by these names
train_pcssl = train_ptv3


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Local PTv3-family trainer/inferencer (ptv3 + pcssl wrappers).')
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
                         'train only the seg head (0 = full fine-tune); '
                         'pcssl wrappers only')
    args = ap.parse_args()
    train_ptv3(**vars(args))


if __name__ == "__main__":
    main()
