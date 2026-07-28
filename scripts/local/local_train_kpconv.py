"""Original KPConv (deformable KPFCNN, KPConv-PyTorch) local trainer on a
canonical trainer_gui dataset, cold start; native SGD recipe + the shared
trainer_gui enhancements, reusing only Config, KPFCNN, p2p_fitting_regularizer
and segmentation_inputs from the upstream repo. neighborhood_limits
are calibrated once per prep cache and travel in run.json (inference restores
them). Modes: train | eval | infer.
"""

import contextlib
import hashlib
import os
import sys
from typing import Optional


def _kpconv_root() -> str:
    """KPConv-PyTorch source root: KPCONV_SRC or /opt/kpconv."""
    for p in (os.environ.get("KPCONV_SRC"), "/opt/kpconv"):
        if p and os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "KPConv-PyTorch source not found: set KPCONV_SRC or bake it at /opt/kpconv")


@contextlib.contextmanager
def _cwd(path):
    """load_kernels() caches kernel dispositions to a CWD-relative path -
    build the net under the KPConv root so the pre-baked cache is found."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


FEATURE_MODE  = "native"
N_EPOCHS      = 150
EPOCH_STEPS   = 300
PACK_N        = 3
ACCUM         = 2
CHECKPOINT_GAP = 10
VAL_EVERY     = 10
PROXY_TILES   = 48
PROXY_SAMPLING = "full"

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

FEAT_CHANNELS = ""

GRID          = 0.25
CONV_RADIUS   = 2.5
DEFORM_RADIUS = 5.0
KP_EXTENT     = 1.2
NUM_KP        = 15
FIRST_FEAT    = 64
DEFORMABLE    = True
CHUNK_XY      = 25.0    # 100 x GRID as the tile side = KPConv's in_radius = 50 x dl rule
MAX_TILE_PTS  = 40000

CALIB_TILES     = 100
CALIB_UNTOUCHED = 0.9
CALIB_MAX_PTS   = 20000

AUG_COLOR     = 0.8

DG_DENSITY_AUG = False
DG_COARSEN_MAX = 2.5
DG_P_NATIVE    = 0.5
DG_INFER_ADABN = False
DG_INFER_APCOTTA = False
DG_INFER_TTA   = 0
DG_LOGDK_FEAT  = False
DG_LOGDK_K     = 8

OPT_TYPE_STR  = "SGD"
SGD_LR0       = 1e-2
SGD_MOMENTUM  = 0.98
SGD_WD        = 1e-3
LR_DECAY      = 0.1 ** (1.0 / 150.0)
DEFORM_LR_FACTOR = 0.1
LABEL_SMOOTH  = 0.2
GRAD_CLIP     = 100.0
BN_MOMENTUM   = 0.02


def _arch(deformable: bool) -> list:
    """train_S3DIS.py architecture verbatim (deformable form): 22 blocks,
    4 strided -> num_layers=5, deform_layers=[F,F,F,T,T]."""
    d = "resnetb_deformable" if deformable else "resnetb"
    ds = d + "_strided"
    return ['simple', 'resnetb', 'resnetb_strided',
            'resnetb', 'resnetb', 'resnetb_strided',
            'resnetb', 'resnetb', 'resnetb_strided',
            d, d, ds, d, d,
            'nearest_upsample', 'unary', 'nearest_upsample', 'unary',
            'nearest_upsample', 'unary', 'nearest_upsample', 'unary']


def arch_hash(arch) -> str:
    return hashlib.sha1(",".join(arch).encode()).hexdigest()[:8]


def make_cfg(num_classes: int, in_features_dim: int, grid: float,
             conv_radius: float = CONV_RADIUS, deform_radius: float = DEFORM_RADIUS,
             kp_extent: float = KP_EXTENT, num_kp: int = NUM_KP,
             first_feat: int = FIRST_FEAT, deformable: bool = DEFORMABLE,
             dataset_name: str = "infer"):
    """In-code KPConv Config. cfg.__init__() is re-called at the end to derive
    num_layers/deform_layers from the assigned architecture (Config.load()'s
    own pattern; the re-call is idempotent)."""
    from utils.config import Config
    cfg = Config()
    cfg.dataset              = dataset_name
    cfg.dataset_task         = "cloud_segmentation"
    cfg.num_classes          = num_classes
    cfg.in_points_dim        = 3
    cfg.in_features_dim      = in_features_dim
    cfg.first_subsampling_dl = grid
    cfg.conv_radius          = conv_radius
    cfg.deform_radius        = deform_radius
    cfg.KP_extent            = kp_extent
    cfg.num_kernel_points    = num_kp
    cfg.first_features_dim   = first_feat
    cfg.use_batch_norm       = True
    cfg.batch_norm_momentum  = BN_MOMENTUM
    cfg.architecture         = _arch(deformable)
    cfg.modulated            = False
    cfg.fixed_kernel_points  = "center"
    cfg.KP_influence         = "linear"
    cfg.aggregation_mode     = "sum"
    cfg.deform_fitting_mode  = "point2point"
    cfg.deform_fitting_power = 1.0
    cfg.deform_lr_factor     = DEFORM_LR_FACTOR
    cfg.repulse_extent       = 1.0
    cfg.class_w              = []
    cfg.saving               = False
    cfg.saving_path          = None
    cfg.__init__()
    return cfg


def _run_json_beside(weights_path):
    """Raw run.json beside the weights (infer needs neighbor_limits + geometry)."""
    import json
    d = os.path.dirname(weights_path)
    if os.path.basename(d) == "checkpoints":
        d = os.path.dirname(d)
    p = os.path.join(d, "run.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def train_kpconv(dataset: Optional[str] = None, mode: str = "train",
                 weights: Optional[str] = None,
                 infer_input: Optional[str] = None, grid: Optional[float] = None,
                 chunk_xy: Optional[float] = None, epochs: Optional[int] = None,
                 batch: Optional[int] = None, steps_per_epoch: Optional[int] = None):
    import os, sys, time, json, glob
    import numpy as np
    import torch


    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helper"))
    import train_common as tc
    (DG_DENSITY_AUG, DG_COARSEN_MAX, DG_P_NATIVE, DG_LOGDK_FEAT, DG_LOGDK_K,
     DG_INFER_ADABN, DG_INFER_APCOTTA, DG_INFER_TTA, USE_FOCAL, FOCAL_GAMMA,
     CLASS_WEIGHTING, WEIGHT_BETA, RARE_OVERSAMPLE, VAL_EVERY,
     FEAT_CHANNELS, PROXY_SAMPLING) = tc.env_overrides(globals(), [
        "DG_DENSITY_AUG", "DG_COARSEN_MAX", "DG_P_NATIVE", "DG_LOGDK_FEAT",
        "DG_LOGDK_K", "DG_INFER_ADABN", "DG_INFER_APCOTTA", "DG_INFER_TTA",
        "USE_FOCAL", "FOCAL_GAMMA", "CLASS_WEIGHTING", "WEIGHT_BETA",
        "RARE_OVERSAMPLE", "VAL_EVERY", "FEAT_CHANNELS", "PROXY_SAMPLING"])

    sys.path.insert(0, _kpconv_root())
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
    LR_DECAY    = 0.1 ** (1.0 / N_EPOCHS)
    FEAT_DEFAULT = ["intensity", "return_number"]
    FEAT_SPEC = (list(FEAT_DEFAULT) if INFER
                 else tc.parse_feat_spec(FEAT_CHANNELS, FEAT_DEFAULT))
    CONV_RADIUS   = globals()["CONV_RADIUS"]
    DEFORM_RADIUS = globals()["DEFORM_RADIUS"]
    KP_EXTENT     = globals()["KP_EXTENT"]
    NUM_KP        = globals()["NUM_KP"]
    FIRST_FEAT    = globals()["FIRST_FEAT"]
    DEFORMABLE    = globals()["DEFORMABLE"]
    NEIGHBOR_LIMITS = None

    ds_root = tc.dataset_dir(dataset) if dataset else None
    if ds_root:
        ds_meta, NUM_CLASSES, CLASS_NAMES = tc.load_dataset_meta(dataset)
        PREP_DIR = (f"{ds_root}/prep/kpconv"
                    f"_grid{GRID:g}_c{int(CHUNK_XY)}"
                    f"{tc.feat_spec_tag(FEAT_SPEC, FEAT_DEFAULT)}"
                    f"{tc.train_stride_tag()}")
    else:
        NUM_CLASSES = 5
        CLASS_NAMES = [f"class {i}" for i in range(NUM_CLASSES)]
        PREP_DIR = f"{tc.OUTPUTS_ROOT}/_infer_unused"
        if INFER and weights:
            rj = _run_json_beside(tc.resolve_weights_path(weights))
            if rj:
                NUM_CLASSES = int(rj.get("num_classes") or NUM_CLASSES)
                CLASS_NAMES = list(rj.get("class_names") or
                                   [f"class {i}" for i in range(NUM_CLASSES)])
                if rj.get("grid") is not None: GRID = float(rj["grid"])
                if rj.get("chunk_xy") is not None: CHUNK_XY = float(rj["chunk_xy"])
                STRIDE = CHUNK_XY / 2.0
                CONV_RADIUS   = float(rj.get("conv_radius", CONV_RADIUS))
                DEFORM_RADIUS = float(rj.get("deform_radius", DEFORM_RADIUS))
                KP_EXTENT     = float(rj.get("kp_extent", KP_EXTENT))
                NUM_KP        = int(rj.get("num_kernel_points", NUM_KP))
                FIRST_FEAT    = int(rj.get("first_features_dim", FIRST_FEAT))
                DEFORMABLE    = bool(rj.get("deformable", DEFORMABLE))
                if rj.get("neighbor_limits"):
                    NEIGHBOR_LIMITS = [int(x) for x in rj["neighbor_limits"]]
                mf = rj.get("features")
                FEAT_SPEC = list(mf) if mf else list(FEAT_DEFAULT)

    if "rgb" in FEAT_SPEC:
        raise ValueError("the KPConv tile pipeline has no rgb channel. Use "
                         "intensity (rgb is folded into it when a scene has "
                         "no intensity)")
    IN_CH = 1 + len(FEAT_SPEC)

    def _cache_signature():
        sp = ds_meta.get("split", {})
        return {
            "format_version": 2,
            "pipeline": "kpconv",
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
            "conv_radius": CONV_RADIUS,
            "deform_radius": DEFORM_RADIUS,
            "arch_hash": arch_hash(_arch(DEFORMABLE)),
        }

    def ensure_prep():
        return tc.kp_ensure_prep(
            PREP_DIR, ds_root, _cache_signature(),
            lambda name, pc_path, out_dir, split: tc.kp_tile_and_save(
                name, pc_path, out_dir, CHUNK_XY,
                tc.train_stride(CHUNK_XY) if split == "train" else STRIDE,
                GRID, NUM_CLASSES))

    def find_latest_checkpoint():
        return tc.kp_find_latest_checkpoint(
            OPT_TYPE_STR, {FEATURE_MODE},
            arch_hash=arch_hash(_arch(DEFORMABLE)),
            features=FEAT_SPEC,
            skip_done=not EVAL_ONLY)

    print("=" * 70)
    print(f"  KPConv  {dataset or 'infer'}  COLD/{FEATURE_MODE}  "
          f"({tc.gpu_name()}, {N_EPOCHS} ep, {EPOCH_STEPS} steps, "
          f"pack {PACK_N} x accum {ACCUM}, {OPT_TYPE_STR})")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}  device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    if not INFER:
        tc.clear_stop()
    train_list, val_list, test_list = ([], [], []) if INFER else ensure_prep()

    run_dir, resume_ckpt, start_epoch = tc.kp_resume_ladder(
        INFER, EVAL_ONLY, RESUME or AUTO_RESUME, find_latest_checkpoint,
        infer_input, "kpconv_native", OPT_TYPE_STR, N_EPOCHS)

    train_tiles = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    val_tiles   = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
    test_tiles  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
    if not INFER:
        print(f"  train_tiles: {len(train_tiles)}   val_tiles: {len(val_tiles)}   "
              f"test_tiles: {len(test_tiles)}", flush=True)
    if not train_tiles and not INFER:
        raise RuntimeError("No training tiles after preprocessing. Check the dataset.")

    from models.architectures import KPFCNN, p2p_fitting_regularizer
    from datasets.common import PointCloudDataset

    IN_DIM = IN_CH + (1 if DG_LOGDK_FEAT else 0)
    cfg = make_cfg(NUM_CLASSES, IN_DIM, GRID,
                   conv_radius=CONV_RADIUS, deform_radius=DEFORM_RADIUS,
                   kp_extent=KP_EXTENT, num_kp=NUM_KP, first_feat=FIRST_FEAT,
                   deformable=DEFORMABLE, dataset_name=dataset or "infer")
    assert cfg.num_layers == 5, f"unexpected num_layers={cfg.num_layers}"

    class _Shim(PointCloudDataset):
        """Carrier for segmentation_inputs, which reads only .config and
        .neighborhood_limits; parent __init__ deliberately skipped."""
        def __init__(self, config, neighborhood_limits):
            self.config = config
            self.neighborhood_limits = neighborhood_limits

    class _KPBatch:
        """The six attributes KPFCNN.forward reads; segmentation_inputs' flat
        list is 5*L+2 long (no scales/rots tail), so L=(len-2)//5."""
        def __init__(self, li, L, dev="cuda"):
            g = lambda a: torch.from_numpy(a).to(dev)
            self.points    = [g(a) for a in li[0:L]]
            self.neighbors = [g(a) for a in li[L:2 * L]]
            self.pools     = [g(a) for a in li[2 * L:3 * L]]
            self.upsamples = [g(a) for a in li[3 * L:4 * L]]
            self.lengths   = [g(a) for a in li[4 * L:5 * L]]
            self.features  = g(li[5 * L])
            self.labels    = g(li[5 * L + 1])

    def make_kp_batch(samples):
        """Pack (xyz, feat, lab) clouds into one lengths-aware KPConv batch
        (C++ pyramid via segmentation_inputs, cropped to NEIGHBOR_LIMITS)."""
        pts = np.ascontiguousarray(np.concatenate([s[0] for s in samples]),
                                   dtype=np.float32)
        feats = np.ascontiguousarray(np.concatenate([s[1] for s in samples]),
                                     dtype=np.float32)
        lens = np.array([len(s[0]) for s in samples], dtype=np.int32)
        has_lab = samples[0][2] is not None
        lab = (np.ascontiguousarray(np.concatenate([s[2] for s in samples]),
                                    dtype=np.int64)
               if has_lab else np.zeros(len(pts), np.int64))
        li = _shim.segmentation_inputs(pts, feats, lab, lens)
        b = _KPBatch(li, (len(li) - 2) // 5)
        return b, (b.labels if has_lab else None)

    build_feat = tc.kp_make_build_feat(DG_LOGDK_FEAT, DG_LOGDK_K, FEAT_SPEC)
    sample_tile = tc.kp_make_sample_tile(
        build_feat, GRID, max_pts=MAX_TILE_PTS, aug_color=AUG_COLOR,
        density_aug=DG_DENSITY_AUG, coarsen_max=DG_COARSEN_MAX,
        p_native=DG_P_NATIVE)

    VAL_RANK = tc.ranking_protocol(PROXY_SAMPLING)
    VAL_FULL = VAL_RANK == "full"
    proxy_tiles = proxy_rep = None
    proxy_samples = []
    if not INFER and not EVAL_ONLY:
        with tc.fixed_np_seed():
            proxy_tiles, proxy_rep = tc.pick_proxy_tiles(
                val_tiles, NUM_CLASSES, PROXY_TILES, mode=PROXY_SAMPLING,
                class_names=CLASS_NAMES,
                cache_path=f"{PREP_DIR}/val_class_balance_cache.npz",
                viable=lambda p: sample_tile(p, training=False) is not None)
            # subsampled ONCE here, before the prefetch threads exist: sample_tile draws from the global np.random, so re-sampling each val pass would race them and fork the scored point set
            for tp in proxy_tiles:
                s = sample_tile(tp, training=False)
                if s is None:
                    bn = os.path.basename(tp)
                    why = ", ".join(proxy_rep["covers"].get(bn, [])) or "the stride base"
                    raise RuntimeError(
                        f"proxy val tile {bn} (curated for {why}) has fewer than "
                        f"32 points: delete {PREP_DIR} and re-run the dataset prep")
                proxy_samples.append((os.path.basename(tp), s))
        print(tc.VAL_FULL_NOTE if VAL_FULL else proxy_rep["text"], flush=True)
        if not tc.proxy_guard(run_dir, proxy_rep, tc.PROXY_PROTOCOL_TILES,
                              CLASS_NAMES, VAL_RANK):
            run_id, run_dir = tc.kp_make_run_dir("kpconv_native")
            resume_ckpt, start_epoch = None, 0
            tc.proxy_guard(run_dir, proxy_rep, tc.PROXY_PROTOCOL_TILES,
                           CLASS_NAMES, VAL_RANK)

    # neighborhood_limits: a train/infer mismatch silently changes the pyramid, so the limits travel in run.json and inference restores them
    def calibrate_neighbors(tiles, k=CALIB_TILES, untouched=CALIB_UNTOUCHED):
        """Per-layer limits keeping `untouched` of neighborhoods uncropped -
        KPConv's own calibration rule. The probe runs uncropped by construction."""
        hist_n = int(np.ceil(4 / 3 * np.pi * (cfg.deform_radius + 1) ** 3))
        hists = np.zeros((cfg.num_layers, hist_n), np.int64)
        probe = _Shim(cfg, [])
        take = np.random.choice(tiles, min(k, len(tiles)), replace=False)
        for tp in take:
            s = sample_tile(tp, max_pts=CALIB_MAX_PTS, training=False)
            if s is None:
                continue
            li = probe.segmentation_inputs(s[0], s[1], s[2].astype(np.int64),
                                           np.array([len(s[0])], np.int32))
            L = (len(li) - 2) // 5
            for layer, nb in enumerate(li[L:2 * L]):
                counts = np.sum(nb < nb.shape[0], axis=1)
                hists[layer] += np.bincount(counts, minlength=hist_n)[:hist_n]
        if not hists.sum():
            raise RuntimeError("neighbor calibration saw no usable tiles")
        cumsum = np.cumsum(hists.T, axis=0)
        limits = np.sum(cumsum < untouched * cumsum[-1, :], axis=0)
        return [max(1, int(x)) for x in limits]

    def _load_or_calibrate_limits():
        cache_p = f"{PREP_DIR}/neighbor_limits.json"
        key = {**_cache_signature(), "kp_extent": KP_EXTENT,
               "untouched": CALIB_UNTOUCHED, "calib_max_pts": CALIB_MAX_PTS}
        if os.path.exists(cache_p):
            try:
                with open(cache_p) as f:
                    doc = json.load(f)
                if doc.get("key") == key:
                    print(f"  neighbor_limits (cached): {doc['limits']}", flush=True)
                    return [int(x) for x in doc["limits"]]
            except Exception:
                pass
        print(f"  calibrating neighbor_limits on up to {CALIB_TILES} tiles "
              f"(one-off per prep cache)…", flush=True)
        t0 = time.time()
        limits = calibrate_neighbors(train_tiles)
        print(f"  neighbor_limits: {limits}  ({time.time()-t0:.1f}s)", flush=True)
        with open(cache_p, "w") as f:
            json.dump({"key": key, "limits": limits}, f, indent=2)
        return limits

    if INFER:
        if NEIGHBOR_LIMITS is None:
            print("  [infer] no neighbor_limits in run.json. Using exact "
                  "uncropped neighbor matrices (slower).", flush=True)
            NEIGHBOR_LIMITS = []
    elif resume_ckpt is not None:
        rj_prev = _run_json_beside(f"{run_dir}/final_model.pth")
        if rj_prev and rj_prev.get("neighbor_limits"):
            NEIGHBOR_LIMITS = [int(x) for x in rj_prev["neighbor_limits"]]
            print(f"  neighbor_limits (from resumed run): {NEIGHBOR_LIMITS}", flush=True)
        else:
            NEIGHBOR_LIMITS = _load_or_calibrate_limits()
    else:
        NEIGHBOR_LIMITS = _load_or_calibrate_limits()

    _shim = _Shim(cfg, NEIGHBOR_LIMITS)

    if resume_ckpt is None and not INFER:
        with open(f"{run_dir}/run.json", "w") as f:
            json.dump({
                "backbone": "KPConv",
                "feature_mode": FEATURE_MODE,
                "input_channels": IN_CH,
                "features": FEAT_SPEC,
                "dataset": dataset,
                "n_epochs": N_EPOCHS, "epoch_steps": EPOCH_STEPS,
                "pack_n": PACK_N, "accum": ACCUM,
                "grid_m": GRID, "chunk_xy_m": CHUNK_XY, "stride_m": STRIDE,
                "deformable": DEFORMABLE,
                "architecture": _arch(DEFORMABLE),
                "arch_hash": arch_hash(_arch(DEFORMABLE)),
                "conv_radius": CONV_RADIUS, "deform_radius": DEFORM_RADIUS,
                "kp_extent": KP_EXTENT, "num_kernel_points": NUM_KP,
                "first_features_dim": FIRST_FEAT,
                "neighbor_limits": NEIGHBOR_LIMITS,
                "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES,
                "optimizer": {
                    "type": "SGD", "lr0": SGD_LR0, "momentum": SGD_MOMENTUM,
                    "weight_decay": SGD_WD, "deform_lr_factor": DEFORM_LR_FACTOR,
                    "lr_decay_per_epoch": LR_DECAY,
                    "grad_clip_value": GRAD_CLIP, "bn_momentum": BN_MOMENTUM,
                    "label_smoothing": LABEL_SMOOTH},
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
                         "lovasz_softmax_weight": LOVASZ_WEIGHT,
                         "deform_reg": "p2p_fitting" if DEFORMABLE else None},
                "train_scenes": [n for n, _ in train_list],
                "val_scenes":   [n for n, _ in val_list],
                "test_scenes":  [n for n, _ in test_list],
            }, f, indent=2)

    with _cwd(_kpconv_root()):
        net = KPFCNN(cfg, lbl_values=list(range(NUM_CLASSES)), ign_lbls=[]).cuda()
    print(f"  Model params: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    print(f"  layers={cfg.num_layers}  deform={cfg.deform_layers}  "
          f"first_radius={GRID * CONV_RADIUS:.2f} m  grid={GRID:g} m  "
          f"neighbor_limits={NEIGHBOR_LIMITS}", flush=True)

    deform_params = [p for n, p in net.named_parameters() if "offset" in n]
    other_params  = [p for n, p in net.named_parameters() if "offset" not in n]
    groups = [{"params": other_params, "lr_mult": 1.0}]
    if deform_params:
        groups.append({"params": deform_params, "lr_mult": DEFORM_LR_FACTOR})
    optim = torch.optim.SGD(groups, lr=SGD_LR0, momentum=SGD_MOMENTUM,
                            weight_decay=SGD_WD)

    def lr_at(ep):
        """Closed-form exponential decay - resume lands on the right lr."""
        return SGD_LR0 * (LR_DECAY ** ep)

    start_epoch = tc.kp_load_mode_weights(net, optim, resume_ckpt, start_epoch,
                                          EVAL_ONLY, INFER, weights, run_dir,
                                          N_EPOCHS)

    if INFER:
        seg_loss = pick_train_tile = None
    else:
        class_counts, present_mask = tc.scan_class_balance(
            train_tiles, NUM_CLASSES,
            cache_path=f"{PREP_DIR}/class_balance_cache.npz")
        rare_classes = (list(RARE_CLASSES) if RARE_CLASSES is not None
                        else tc.auto_rare_classes(class_counts, RARE_FREQ_FRAC))
        rare_tiles = ([train_tiles[i]
                       for i in np.nonzero(present_mask[:, rare_classes].any(1))[0]]
                      if (RARE_OVERSAMPLE and rare_classes) else [])
        print(f"  class counts: {dict(zip(CLASS_NAMES, class_counts.tolist()))}", flush=True)
        print(f"  rare classes: {[CLASS_NAMES[c] for c in rare_classes]}", flush=True)
        print(f"  rare-class tiles: {len(rare_tiles)} / {len(train_tiles)}", flush=True)

        if CLASS_WEIGHTING:
            w = tc.class_weights_np(class_counts, WEIGHT_BETA, WEIGHT_CAP)
            class_weights = torch.tensor(w, dtype=torch.float32).cuda()
            print(f"  class weights: "
                  f"{dict(zip(CLASS_NAMES, [round(float(x), 3) for x in w]))}", flush=True)
        else:
            class_weights = None
        seg_loss = tc.make_seg_loss(class_weights, LABEL_SMOOTH, USE_FOCAL,
                                    FOCAL_GAMMA, LOVASZ_WEIGHT)
        pick_train_tile = tc.make_tile_picker(train_tiles, rare_tiles, RARE_TILE_PROB)

    def _kp_batch(cxyz, feat):
        return make_kp_batch([(cxyz, feat, None)])[0]

    _forward = lambda b: net(b, cfg)
    SAVE_PROBS = os.environ.get("TT_SAVE_PROBS") == "1"
    EXC_IDX = tc.exclude_class_idx(CLASS_NAMES) if INFER else []
    _predict_points = tc.kp_make_predict_points(
        lambda cxyz, feat: torch.softmax(net(_kp_batch(cxyz, feat), cfg).float(),
                                         -1).cpu().numpy(),
        build_feat, GRID, CHUNK_XY, NUM_CLASSES, DG_INFER_TTA,
        save_probs=SAVE_PROBS, exclude_idx=EXC_IDX)

    if INFER:
        tc.kp_run_infer(run_dir, net, _forward, _kp_batch, build_feat,
                        _predict_points, "KPConv", "KPConv", weights,
                        infer_input, GRID, CHUNK_XY, grid, chunk_xy,
                        NUM_CLASSES, CLASS_NAMES, FEAT_SPEC, EXC_IDX,
                        DG_INFER_ADABN, neighbor_limits=NEIGHBOR_LIMITS,
                        infer_apcotta=DG_INFER_APCOTTA)
        return

    metrics_csv = tc.init_metrics_csv(run_dir)

    val_csv = f"{run_dir}/val_metrics.csv"
    tc.init_val_csv(val_csv, CLASS_NAMES)

    val_items  = [(n, p, f"{PREP_DIR}/val")  for n, p in val_list]
    test_items = [(n, p, f"{PREP_DIR}/test") for n, p in test_list]
    print(f"  eval set: {len(val_items)} holdout(val) + {len(test_items)} test scenes",
          flush=True)

    def _fwd_eval(tiles):
        b, _ = make_kp_batch([(c, f, None) for c, f in tiles])
        lg = net(b, cfg).cpu().numpy().astype(np.float32)
        return np.split(lg, np.cumsum([len(c) for c, _ in tiles])[:-1])
    evaluate = tc.kp_make_evaluate(_fwd_eval, build_feat, GRID, CHUNK_XY,
                                   NUM_CLASSES, CLASS_NAMES)

    best = tc.BestCheckpoint(run_dir, VAL_RANK)
    tc.write_run_manifest(run_dir, "kpconv", dataset)

    run_eval = tc.kp_make_run_eval(
        net, _forward, evaluate, make_kp_batch, sample_tile, pick_train_tile,
        best, val_csv, run_dir, val_items, test_items, val_list, test_list,
        NUM_CLASSES, CLASS_NAMES, VAL_FULL, EVAL_ONLY,
        proxy_batches=lambda: tc.kp_proxy_batches(
            proxy_samples, make_kp_batch, proxy_rep,
            lambda bn, why, e: (
                f"proxy val tile {bn} (curated for {why}) could not build "
                f"a KPConv batch: delete {PREP_DIR} and re-run the dataset "
                f"prep. Original: {e}")),
        proxy_tiles=proxy_tiles, proxy_rep=proxy_rep)

    tc.kp_train_loop(
        net, optim, _forward, seg_loss, make_kp_batch, sample_tile,
        pick_train_tile, lr_at, run_eval, best, run_dir, metrics_csv,
        NUM_CLASSES, start_epoch, N_EPOCHS, EPOCH_STEPS, PACK_N, ACCUM,
        CHECKPOINT_GAP, VAL_EVERY, EVAL_ONLY,
        grad_clip_fn=lambda: torch.nn.utils.clip_grad_value_(net.parameters(),
                                                             GRAD_CLIP),
        reg_fn=lambda: p2p_fitting_regularizer(net))


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Local KPConv (original, deformable KPFCNN) trainer/inferencer.')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--mode', default='train')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--infer-input', default=None)
    ap.add_argument('--grid', type=float, default=None)
    ap.add_argument('--chunk-xy', type=float, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--steps-per-epoch', type=int, default=None)
    args = vars(ap.parse_args())
    if args['dataset'] is None and args['mode'] != 'infer':
        ap.error('--dataset is required (a canonical trainer_gui dataset name); '
                 'only --mode infer may omit it.')
    train_kpconv(**args)


if __name__ == "__main__":
    main()
