"""
Local training script for the ORIGINAL KPConv (KPConv-PyTorch, HuguesTHOMAS)
on a canonical trainer_gui --dataset (3-folder train/val/test), deformable
KPFCNN, cold start.

  Features  : 3 = [1, intensity, return_number] by default. FEAT_CHANNELS env
              overrides the spec (ordered csv incl. dataset feat_* channels,
              e.g. feat_hag for real HeightAboveGround); run.json "features"
              records the resolved list. The old tile-relative "height"
              channel is legacy-only (pre-spec checkpoints).

Model: the deformable KPFCNN (train_S3DIS.py architecture verbatim) built from
the KPConv-PyTorch repo at /opt/kpconv (env KPCONV_SRC overrides; the dev-box
clone is the host fallback). Reused from that repo: Config, KPFCNN,
p2p_fitting_regularizer, and PointCloudDataset.segmentation_inputs (the C++
neighbor/pooling pyramid) — none of its trainers, testers or dataset classes
(they carry hardcoded ../../Data paths, sentinel-file stop logic and
plt.show()-on-failure calibration).

Recipe: KPConv's NATIVE SGD (momentum 0.98, wd 1e-3, deformable-offset params
at 0.1x LR, clip_grad_value_ 100, lr *= 0.1**(1/150) per epoch), plus the
shared trainer_gui enhancements: class-weighted smoothed CE (+ Lovasz),
rare-class tile oversampling, packed batches x grad accumulation, held-out val
passes, voted raw-point eval, DG hooks, auto-resume, best-val-mIoU
final_model.pth.

KPConv-specific mechanics:
  - neighborhood_limits are CALIBRATED once per prep cache (90% of
    neighborhoods untouched — the repo's own rule) and recorded in run.json;
    inference restores them so the pyramid geometry matches the weights.
  - the deformable fitting/repulsion regularizer (p2p_fitting_regularizer) is
    added to the loss after each forward (it reads state the forward populates)
    and logged as the metrics.csv offset_loss column.
  - kernel dispositions cache to a CWD-RELATIVE path, so the model is built
    under _cwd(<kpconv root>) — the image pre-bakes the file; without the chdir
    the first run would write kernels/dispositions/ into the bind-mounted repo.

Usage:
    python local_train_kpconv.py --dataset <name>
    python local_train_kpconv.py --dataset <name> --mode eval \
        --weights runs/<id>/final_model.pth
    python local_train_kpconv.py --mode infer --infer-input <job> \
        --weights runs/<id>/final_model.pth
    python local_train_kpconv.py --self-test   # host check: no GPU, no C++
"""

import contextlib
import hashlib
import os
import sys
from typing import Optional


def _kpconv_root() -> str:
    """KPConv-PyTorch source root: /opt/kpconv inside the images; KPCONV_SRC or
    the dev-box clone for host runs (--self-test)."""
    for p in (os.environ.get("KPCONV_SRC"), "/opt/kpconv",
              os.path.join(os.path.expanduser("~"), "Desktop", "KPConv-PyTorch")):
        if p and os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "KPConv-PyTorch source not found: set KPCONV_SRC or bake it at /opt/kpconv")


@contextlib.contextmanager
def _cwd(path):
    """load_kernels() caches kernel dispositions to the RELATIVE path
    kernels/dispositions/ — under /workspace that would be the bind-mounted host
    repo. Build the net under the KPConv root so the (pre-baked) cache is found
    there instead. All our data paths are absolute, so the chdir is inert."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ============================================================================
# Configuration
# ============================================================================
FEATURE_MODE  = "native"     # [1, intensity, return_number] (+ selected feat_*)
N_EPOCHS      = 150
EPOCH_STEPS   = 300          # optimizer steps / epoch
PACK_N        = 3            # tiles packed per forward (deformable KPConv is
                             # heavier than KPConvX: 3, not 4)
ACCUM         = 2            # grad-accumulated forwards / step
CHECKPOINT_GAP = 10          # checkpoint frequency (epochs); saves model + optimizer
VAL_EVERY     = 10           # held-out val pass every N epochs (no weight updates)

# Resume: when True, continue the most recent same-recipe run in the outputs
# dir (same run dir, appended metrics) instead of starting fresh.
RESUME = False
# The cloud shell sets AUTO_RESUME=1 only on Modal's OWN retries (preemption /
# crash); locally a user can export it to continue after a Kill. Same contract
# as the PTv3-family trainers.
AUTO_RESUME = os.environ.get("AUTO_RESUME", "0") == "1"

# Class-balanced loss + rare-class oversampling (shared trainer_gui recipe).
CLASS_WEIGHTING = True
WEIGHT_BETA     = 0.5        # 0.5 = inverse-sqrt frequency
WEIGHT_CAP      = 5.0        # clamp each weight to [1/CAP, CAP] after mean-norm
LOVASZ_WEIGHT   = 1.0        # total loss = <pointwise> + w * lovasz_softmax
USE_FOCAL       = False
FOCAL_GAMMA     = 2.0
RARE_OVERSAMPLE = True
RARE_CLASSES    = None       # explicit class indices, or None -> auto from train freq
RARE_FREQ_FRAC  = 0.5        # auto-rare threshold: freq < frac x median present freq
RARE_TILE_PROB  = 0.25       # P(draw the next train tile from a rare-class tile)

# Input-feature spec (FEAT_CHANNELS env, GUI picker): comma-separated ordered
# names; "" = the default [intensity, return_number] recipe (no height). The
# constant-1 bias channel is always first and not part of the spec.
FEAT_CHANNELS = ""

# Geometry — original KPConv deformable KPFCNN (train_S3DIS.py constants),
# scaled to aerial LiDAR exactly like the KPConvX sibling (GRID 2.0 / 100 m tiles).
GRID          = 2.0          # first_subsampling_dl: layer-0 voxel grid (m)
CONV_RADIUS   = 2.5          # conv radius in grid cells (S3DIS: 2.5)
DEFORM_RADIUS = 5.0          # S3DIS deformable radius (6.0 is the ModelNet40
                             # config — heavier and not the segmentation recipe)
KP_EXTENT     = 1.2          # S3DIS KP_extent
NUM_KP        = 15           # kernel points / conv -> cache k_015_center_3D.ply
FIRST_FEAT    = 64           # 128 == released-S3DIS capacity, ~2x memory
DEFORMABLE    = True         # False -> rigid resnetb blocks everywhere; the
                             # p2p fitting regularizer then self-zeroes
CHUNK_XY      = 100.0        # tile size (m); ~50x GRID, mirrors the sibling
STRIDE        = 50.0         # tile stride: overlap for train coverage + voted eval
MAX_TILE_PTS  = 40000        # sample_tile cap (KPConvX uses 60000; the deformable
                             # neighbor-difference tensors are markedly heavier)

# neighborhood_limits calibration — mirrors KPConv's own 90%-untouched rule
# (datasets/S3DIS.py calibration) minus its sampler/plt.show()/1-0 hazards.
CALIB_TILES     = 100        # tiles sampled for the histogram
CALIB_UNTOUCHED = 0.9        # fraction of neighborhoods left uncropped
CALIB_MAX_PTS   = 20000      # the probe runs UNCROPPED by construction: keep it small

# Augmentation: geometry (rotation/scale/x-flip/noise) is train_common.
# kp_augment's defaults; only the feature-drop probability is configured here.
# (KPConv's own augmentation_transform is never called.)
AUG_COLOR     = 0.8          # per-channel keep prob: each feature channel independently zeroed with p 0.2

# --- density domain-generalization (scripts/helper/density.py; see DENSITY_DG.md) ---
DG_DENSITY_AUG = False   # D1: per-tile coarsen the loaded tile to a jittered grid (train only)
DG_COARSEN_MAX = 2.5
DG_P_NATIVE    = 0.5
DG_INFER_ADABN = False   # D2b: recompute BN stats on the target tiles before predicting
DG_INFER_TTA   = 0       # D5: # extra density(scale) views to average at inference (0=off)
DG_LOGDK_FEAT  = False   # D3b: +1 input channel (log k-th-NN distance) -> retrain
DG_LOGDK_K     = 8

# Optimizer — KPConv's NATIVE recipe (utils/trainer.py + train_S3DIS.py);
# recorded in run.json, and find_latest_checkpoint refuses to resume across
# recipes. (The AdamW 1-cycle recipe lives in local_train_kpconvx_cold.py.)
OPT_TYPE_STR  = "SGD"
SGD_LR0       = 1e-2
SGD_MOMENTUM  = 0.98
SGD_WD        = 1e-3
LR_DECAY      = 0.1 ** (1.0 / 150.0)   # closed-form per-epoch exponential decay
DEFORM_LR_FACTOR = 0.1       # offset params train at 0.1x LR (KPConv trainer.py)
LABEL_SMOOTH  = 0.2
GRAD_CLIP     = 100.0        # clip_grad_VALUE_ (KPConv's own clip), not norm
BN_MOMENTUM   = 0.02         # S3DIS batch_norm_momentum


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
    """KPConv Config built in-code from our recipe (no Config subclass files).

    Config.__init__ derives num_layers/deform_layers from self.architecture AT
    CONSTRUCTION TIME — and architecture is a CLASS attribute defaulting to [].
    So after assigning the instance fields we re-call cfg.__init__(), exactly
    what Config.load() does upstream (utils/config.py:275); it writes only the
    two derived fields, so the re-call is idempotent."""
    from utils.config import Config   # _kpconv_root() must be on sys.path
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
    cfg.class_w              = []    # class weights live in OUR seg_loss, not net.loss
    cfg.saving               = False
    cfg.saving_path          = None
    cfg.__init__()   # re-derive num_layers + deform_layers from the set architecture
    return cfg


def _run_json_beside(weights_path):
    """The raw (manifest-merged) run.json beside the weights. KPConv inference
    needs more than train_common.infer_meta's normalized keys — neighbor_limits
    and the net geometry — so read the full document."""
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
     WEIGHT_BETA, RARE_OVERSAMPLE, VAL_EVERY,
     FEAT_CHANNELS) = tc.env_overrides(globals(), [
        "DG_DENSITY_AUG", "DG_COARSEN_MAX", "DG_P_NATIVE", "DG_LOGDK_FEAT",
        "DG_LOGDK_K", "DG_INFER_ADABN", "DG_INFER_TTA", "USE_FOCAL",
        "FOCAL_GAMMA", "CLASS_WEIGHTING", "WEIGHT_BETA", "RARE_OVERSAMPLE",
        "VAL_EVERY", "FEAT_CHANNELS"])

    sys.path.insert(0, _kpconv_root())
    EVAL_ONLY = (mode == "eval")
    INFER     = (mode == "infer")   # arbitrary-folder inference (trainer_gui)

    if dataset is None and not INFER:
        raise ValueError("--dataset is required: pass a canonical trainer_gui "
                         "dataset name (train/val/test folders). The only "
                         "dataset-free path is --mode infer.")

    # --- resolve run config: CLI flags override the module defaults ------------
    GRID        = grid if grid is not None else globals()["GRID"]
    CHUNK_XY    = chunk_xy if chunk_xy is not None else globals()["CHUNK_XY"]
    STRIDE      = CHUNK_XY / 2.0
    N_EPOCHS    = epochs if epochs is not None else globals()["N_EPOCHS"]
    EPOCH_STEPS = steps_per_epoch if steps_per_epoch is not None else globals()["EPOCH_STEPS"]
    PACK_N      = batch if batch is not None else globals()["PACK_N"]
    FEATURE_MODE = globals()["FEATURE_MODE"]
    # LR_DECAY is quoted for the 150-epoch default; scale it with N_EPOCHS so a
    # short run still completes the same /10 decay instead of training at a
    # near-constant lr0. At N_EPOCHS=150 this reproduces the module constant.
    LR_DECAY    = 0.1 ** (1.0 / N_EPOCHS)
    # Input-feature spec: FEAT_CHANNELS env at train (the infer block below
    # overrides it from run.json — env is ignored at infer). "height" (the
    # tile-relative z proxy) is DEAD for new runs (removed 2026-07-22 — real
    # HAG is the feat_hag dataset channel); FEAT_LEGACY keeps it ONLY so
    # pre-spec checkpoints, whose weights expect that width, still infer.
    FEAT_LEGACY = ["intensity", "return_number", "height"]   # infer-only reconstruction
    FEAT_DEFAULT = ["intensity", "return_number"]            # train default, no height
    FEAT_SPEC = (list(FEAT_LEGACY) if INFER    # env ignored at infer
                 else tc.parse_feat_spec(FEAT_CHANNELS, FEAT_DEFAULT))
    # KPConv pyramid geometry: module constants, restored from run.json at infer
    # so the rebuilt model matches the weights exactly.
    CONV_RADIUS   = globals()["CONV_RADIUS"]
    DEFORM_RADIUS = globals()["DEFORM_RADIUS"]
    KP_EXTENT     = globals()["KP_EXTENT"]
    NUM_KP        = globals()["NUM_KP"]
    FIRST_FEAT    = globals()["FIRST_FEAT"]
    DEFORMABLE    = globals()["DEFORMABLE"]
    NEIGHBOR_LIMITS = None   # resolved below: run.json (infer/resume) or calibration

    ds_root = tc.dataset_dir(dataset) if dataset else None
    if ds_root:
        ds_meta, NUM_CLASSES, CLASS_NAMES = tc.load_dataset_meta(dataset)
        PREP_DIR = (f"{ds_root}/prep/kpconv"
                    f"_grid{GRID:g}_c{int(CHUNK_XY)}"
                    f"{tc.feat_spec_tag(FEAT_SPEC, FEAT_LEGACY)}"
                    f"{tc.train_stride_tag()}")
    else:
        # Folder inference (--mode infer): no --dataset. Reproduce the TRAINED
        # geometry, class layout AND neighbor_limits from the run's run.json (the
        # self-contained manifest beside the weights). INFER never reads cached
        # tiles, so PREP_DIR is an unused placeholder.
        NUM_CLASSES = 5
        CLASS_NAMES = [f"class {i}" for i in range(NUM_CLASSES)]
        PREP_DIR = f"{tc.OUTPUTS_ROOT}/_infer_unused"
        if INFER and weights:
            rj = _run_json_beside(weights if os.path.isabs(weights)
                                  else f"{tc.OUTPUTS_ROOT}/{weights}")
            if rj:
                NUM_CLASSES = int(rj.get("num_classes") or NUM_CLASSES)
                CLASS_NAMES = list(rj.get("class_names") or
                                   [f"class {i}" for i in range(NUM_CLASSES)])
                if rj.get("grid") is not None: GRID = float(rj["grid"])
                if rj.get("chunk_xy") is not None: CHUNK_XY = float(rj["chunk_xy"])
                STRIDE = CHUNK_XY / 2.0
                # Rebuild the IDENTICAL net + pyramid the weights trained with.
                CONV_RADIUS   = float(rj.get("conv_radius", CONV_RADIUS))
                DEFORM_RADIUS = float(rj.get("deform_radius", DEFORM_RADIUS))
                KP_EXTENT     = float(rj.get("kp_extent", KP_EXTENT))
                NUM_KP        = int(rj.get("num_kernel_points", NUM_KP))
                FIRST_FEAT    = int(rj.get("first_features_dim", FIRST_FEAT))
                DEFORMABLE    = bool(rj.get("deformable", DEFORMABLE))
                if rj.get("neighbor_limits"):
                    NEIGHBOR_LIMITS = [int(x) for x in rj["neighbor_limits"]]
                # rebuild the EXACT assembly recorded with the weights (env is
                # ignored at infer); manifests without "features" = legacy runs.
                mf = rj.get("features")
                try:
                    FEAT_SPEC = (tc.parse_feat_spec(",".join(mf), FEAT_LEGACY)
                                 if mf else list(FEAT_LEGACY))
                except ValueError:
                    FEAT_SPEC = list(FEAT_LEGACY)
                if rj.get("hag_source"):
                    # ponytail: TEMPORARY shim (remove once legacy runs retire) —
                    # the deleted --hag variant's 'height' channel was real HAG.
                    # Feed the baked feat_hag channel in that slot: same width,
                    # right semantics. Scenes MUST be converted with HAG on.
                    FEAT_SPEC = ["feat_hag" if n == "height" else n for n in FEAT_SPEC]
                    print(f"  [legacy-hag] weights from the removed --hag variant "
                          f"(hag_source={rj['hag_source']}): 'height' -> feat_hag; "
                          f"input scenes must carry a baked feat_hag channel",
                          flush=True)

    if "rgb" in FEAT_SPEC:
        raise ValueError("the KPConv tile pipeline has no rgb channel — use "
                         "intensity (rgb is folded into it when a scene has "
                         "no intensity)")
    # bias + spec channels (+ log d_k below); every kp spec name is width 1.
    IN_CH = 1 + len(FEAT_SPEC)

    def _cache_signature():
        # Everything that changes what a cached tile contains — plus the KPConv
        # pyramid geometry, because neighbor_limits.json is keyed by this same
        # signature and stale limits silently change the pyramid. The
        # spec-derived recipe string reproduces the legacy spellings
        # byte-for-byte, so every existing cache stays valid.
        sp = ds_meta.get("split", {})
        return {
            # v2: tiles carry the scene's real return_number (was zero-filled)
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
            # kp_tile_and_save bakes labels against the class layout: merging or
            # reordering classes must invalidate the cache.
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
        # train tiles at the (wider) TT_TRAIN_STRIDE; val/test keep chunk/2 so
        # the final eval can vote over the up-to-4 overlapping covering tiles.
        return tc.kp_ensure_prep(
            PREP_DIR, ds_root, _cache_signature(),
            lambda name, pc_path, out_dir, split: tc.kp_tile_and_save(
                name, pc_path, out_dir, CHUNK_XY,
                tc.train_stride(CHUNK_XY) if split == "train" else STRIDE,
                GRID, NUM_CLASSES))

    def find_latest_checkpoint():
        # Same-recipe runs only: optimizer type, feature spec AND the
        # architecture hash must all match.
        return tc.kp_find_latest_checkpoint(
            OPT_TYPE_STR, {FEATURE_MODE},
            arch_hash=arch_hash(_arch(DEFORMABLE)),
            features=FEAT_SPEC, legacy_features=FEAT_LEGACY,
            skip_done=not EVAL_ONLY)

    print("=" * 70)
    print(f"  KPConv  {dataset or 'infer'}  COLD/{FEATURE_MODE}  "
          f"({tc.gpu_name()}, {N_EPOCHS} ep, {EPOCH_STEPS} steps, "
          f"pack {PACK_N} x accum {ACCUM}, {OPT_TYPE_STR})")
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
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_infer")
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
            raise RuntimeError(f"eval mode: no {OPT_TYPE_STR}-recipe run with "
                               f"checkpoints found under /outputs")
        run_id, run_dir = tc.kp_make_run_dir("kpconv_native")
        resume_ckpt, start_epoch = None, 0

    train_tiles = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    val_tiles   = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
    test_tiles  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
    if not INFER:
        print(f"  train_tiles: {len(train_tiles)}   val_tiles: {len(val_tiles)}   "
              f"test_tiles: {len(test_tiles)}", flush=True)
    if not train_tiles and not INFER:
        raise RuntimeError("No training tiles after preprocessing — check the dataset.")

    # ------------------------------------------------------------------------
    # Config + input pipeline shims — the ONLY imports from KPConv-PyTorch are
    # Config, KPFCNN, p2p_fitting_regularizer and PointCloudDataset (for
    # segmentation_inputs, which reads only .config and .neighborhood_limits).
    # ------------------------------------------------------------------------
    from models.architectures import KPFCNN, p2p_fitting_regularizer
    from datasets.common import PointCloudDataset   # needs the compiled C++ wrappers

    IN_DIM = IN_CH + (1 if DG_LOGDK_FEAT else 0)   # +log d_k (D3b)
    cfg = make_cfg(NUM_CLASSES, IN_DIM, GRID,
                   conv_radius=CONV_RADIUS, deform_radius=DEFORM_RADIUS,
                   kp_extent=KP_EXTENT, num_kp=NUM_KP, first_feat=FIRST_FEAT,
                   deformable=DEFORMABLE, dataset_name=dataset or "infer")
    assert cfg.num_layers == 5, f"unexpected num_layers={cfg.num_layers}"

    class _Shim(PointCloudDataset):
        """Duck-typed carrier for segmentation_inputs: it reads ONLY .config and
        .neighborhood_limits (datasets/common.py:457,:332) — the parent __init__
        is deliberately skipped so none of the dataset machinery exists."""
        def __init__(self, config, neighborhood_limits):
            self.config = config
            self.neighborhood_limits = neighborhood_limits

    class _KPBatch:
        """The six attributes KPFCNN.forward + its blocks actually read. The
        flat list from segmentation_inputs is 5*L+2 long (points, neighbors,
        pools, upsamples, lengths, features, labels — no scales/rots tail, so
        L=(len-2)//5, unlike S3DISCustomBatch's (len-7)//5)."""
        def __init__(self, li, L, dev="cuda"):
            g = lambda a: torch.from_numpy(a).to(dev)
            self.points    = [g(a) for a in li[0:L]]          # f32
            self.neighbors = [g(a) for a in li[L:2 * L]]      # i64
            self.pools     = [g(a) for a in li[2 * L:3 * L]]  # i64
            self.upsamples = [g(a) for a in li[3 * L:4 * L]]  # i64
            self.lengths   = [g(a) for a in li[4 * L:5 * L]]  # i32
            self.features  = g(li[5 * L])
            self.labels    = g(li[5 * L + 1])

    def make_kp_batch(samples):
        """Pack one or more (xyz, feat, lab) clouds into a single KPConv batch:
        concatenated points with per-cloud lengths (batch_neighbors /
        batch_grid_subsampling are lengths-aware, so clouds never mix), then the
        C++ pyramid via segmentation_inputs, cropped to NEIGHBOR_LIMITS."""
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

    # ------------------------------------------------------------------------
    # Features + tile sampling (shared KP-family pipeline; also feeds the
    # calibration probe). Dataset feat_* channels are cached by
    # kp_tile_and_save and fed by the spec, nothing here.
    # ------------------------------------------------------------------------
    build_feat = tc.kp_make_build_feat(DG_LOGDK_FEAT, DG_LOGDK_K, FEAT_SPEC)
    sample_tile = tc.kp_make_sample_tile(
        build_feat, GRID, max_pts=MAX_TILE_PTS, aug_color=AUG_COLOR,
        density_aug=DG_DENSITY_AUG, coarsen_max=DG_COARSEN_MAX,
        p_native=DG_P_NATIVE)

    # ------------------------------------------------------------------------
    # neighborhood_limits — calibrate once per prep cache, reuse everywhere.
    # A train/infer mismatch silently changes the pyramid (no shape error), so
    # the limits travel in run.json and inference restores them above.
    # ------------------------------------------------------------------------
    def calibrate_neighbors(tiles, k=CALIB_TILES, untouched=CALIB_UNTOUCHED):
        """Per-layer neighbor-count limits keeping `untouched` of neighborhoods
        uncropped — KPConv's own calibration rule (datasets/S3DIS.py) minus the
        sampler, the plt.show() and the 1/0. The probe runs UNCROPPED by
        construction (that is what it measures), so it samples small tiles."""
        hist_n = int(np.ceil(4 / 3 * np.pi * (cfg.deform_radius + 1) ** 3))
        hists = np.zeros((cfg.num_layers, hist_n), np.int64)
        probe = _Shim(cfg, [])                      # empty limits => uncropped
        take = np.random.choice(tiles, min(k, len(tiles)), replace=False)
        for tp in take:
            s = sample_tile(tp, max_pts=CALIB_MAX_PTS, training=False)
            if s is None:
                continue
            li = probe.segmentation_inputs(s[0], s[1], s[2].astype(np.int64),
                                           np.array([len(s[0])], np.int32))
            L = (len(li) - 2) // 5
            for layer, nb in enumerate(li[L:2 * L]):   # input_neighbors per layer
                counts = np.sum(nb < nb.shape[0], axis=1)   # non-shadow neighbors
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
            print("  [infer] WARNING: no neighbor_limits in run.json — running with "
                  "UNCROPPED neighbor matrices (exact, but slower).", flush=True)
            NEIGHBOR_LIMITS = []
    elif resume_info:
        # Adopt the resumed/eval'd run's limits so the pyramid matches its weights.
        rj_prev = _run_json_beside(f"{run_dir}/final_model.pth")
        if rj_prev and rj_prev.get("neighbor_limits"):
            NEIGHBOR_LIMITS = [int(x) for x in rj_prev["neighbor_limits"]]
            print(f"  neighbor_limits (from resumed run): {NEIGHBOR_LIMITS}", flush=True)
        else:
            NEIGHBOR_LIMITS = _load_or_calibrate_limits()
    else:
        NEIGHBOR_LIMITS = _load_or_calibrate_limits()

    _shim = _Shim(cfg, NEIGHBOR_LIMITS)

    # --- raw run config at train START (write_run_manifest merges over it) ------
    if resume_ckpt is None and not INFER:
        with open(f"{run_dir}/run.json", "w") as f:
            json.dump({
                "backbone": "KPConv",
                "warm_start": False,
                "feature_mode": FEATURE_MODE,
                "input_channels": IN_CH,
                # resolved input spec (bias/log-dk ride outside it) —
                # inference rebuilds this exact assembly from here.
                "features": FEAT_SPEC,
                "dataset": dataset,
                "n_epochs": N_EPOCHS, "epoch_steps": EPOCH_STEPS,
                "pack_n": PACK_N, "accum": ACCUM,
                "grid_m": GRID, "chunk_xy_m": CHUNK_XY, "stride_m": STRIDE,
                # KPConv pyramid geometry — inference rebuilds the IDENTICAL model
                # and pyramid from these (see the --mode infer branch above).
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
                "train_scenes": [n for n, _, _ in train_list],
                "val_scenes":   [n for n, _, _ in val_list],
                "test_scenes":  [n for n, _, _ in test_list],
            }, f, indent=2)

    # ------------------------------------------------------------------------
    # Build model — original deformable KPFCNN, random init. Constructed under
    # the KPConv root: load_kernels caches dispositions to a RELATIVE path, and
    # only there is the pre-baked cache found (and the host repo left untouched).
    # ------------------------------------------------------------------------
    with _cwd(_kpconv_root()):
        net = KPFCNN(cfg, lbl_values=list(range(NUM_CLASSES)), ign_lbls=[]).cuda()
    print(f"  Model params: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    print(f"  layers={cfg.num_layers}  deform={cfg.deform_layers}  "
          f"first_radius={GRID * CONV_RADIUS:.2f} m  grid={GRID:g} m  "
          f"neighbor_limits={NEIGHBOR_LIMITS}", flush=True)

    # Two param groups — offset (deformable) params at DEFORM_LR_FACTOR x LR,
    # exactly KPConv's own trainer split; lr_mult rides along in the group dict.
    deform_params = [p for n, p in net.named_parameters() if "offset" in n]
    other_params  = [p for n, p in net.named_parameters() if "offset" not in n]
    groups = [{"params": other_params, "lr_mult": 1.0}]
    if deform_params:                       # empty when DEFORMABLE=False
        groups.append({"params": deform_params, "lr_mult": DEFORM_LR_FACTOR})
    optim = torch.optim.SGD(groups, lr=SGD_LR0, momentum=SGD_MOMENTUM,
                            weight_decay=SGD_WD)

    def lr_at(ep):
        """KPConv's closed-form exponential decay (resume at epoch N lands on
        the right lr without replaying the schedule)."""
        return SGD_LR0 * (LR_DECAY ** ep)

    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location="cuda", weights_only=True)
        net.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        print(f"  resumed weights{' + optimizer' if 'optim' in ckpt else ''} "
              f"at epoch {start_epoch}", flush=True)

    if EVAL_ONLY:
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

    if INFER:
        # Inference never trains or samples tiles — skip the class-balance scan
        # (and its noisy zero-count logging); the loss/picker are train-only.
        seg_loss = pick_train_tile = None
    else:
        # --- class-balanced loss + rare-class oversampling ------------------
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
        # Shared loss recipe: weighted smoothed CE (or focal) + Lovász-Softmax,
        # with the all-ignored-pack NaN guard (see train_common.make_seg_loss).
        seg_loss = tc.make_seg_loss(class_weights, LABEL_SMOOTH, USE_FOCAL,
                                    FOCAL_GAMMA, LOVASZ_WEIGHT)
        pick_train_tile = tc.make_tile_picker(train_tiles, rare_tiles, RARE_TILE_PROB)

    def _kp_batch(cxyz, feat):
        return make_kp_batch([(cxyz, feat, None)])[0]

    # Sliding-window inference over already-normalized features (--mode infer),
    # with the D5 density-TTA view averaging (shared KP-family pipeline).
    SAVE_PROBS = os.environ.get("TT_SAVE_PROBS") == "1"
    EXC_IDX = tc.exclude_class_idx(CLASS_NAMES) if INFER else []
    _predict_points = tc.kp_make_predict_points(
        lambda cxyz, feat: torch.softmax(net(_kp_batch(cxyz, feat), cfg).float(),
                                         -1).cpu().numpy(),
        build_feat, GRID, CHUNK_XY, NUM_CLASSES, DG_INFER_TTA,
        save_probs=SAVE_PROBS, exclude_idx=EXC_IDX)

    # ------------------------------------------------------------------------
    # Arbitrary-folder inference: label the staged npz scenes and stop.
    # ------------------------------------------------------------------------
    if INFER:
        if not infer_input:
            raise ValueError("--mode infer requires --infer-input <job_id>")
        if (grid is not None and grid != GRID) or (chunk_xy is not None and chunk_xy != CHUNK_XY):
            print(f"  [infer] note: KPConv uses its trained geometry "
                  f"(grid={GRID}, chunk={CHUNK_XY}); --grid/--chunk-xy ignored.", flush=True)
        net.eval()
        scenes = sorted(glob.glob(f"{run_dir}/scenes/*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scenes under {run_dir}/scenes")
        pred_dir = os.environ.get("TT_PRED_DIR") or f"{run_dir}/predictions"
        infer_cfg = {"backbone": "KPConv", "mode": "infer",
                     "weights": weights,
                     "infer_input": infer_input, "num_classes": NUM_CLASSES,
                     "class_names": CLASS_NAMES, "grid": GRID, "chunk_xy": CHUNK_XY,
                     "neighbor_limits": NEIGHBOR_LIMITS,
                     "gpu": tc.gpu_name(),
                     "exclude_classes": [CLASS_NAMES[i] for i in EXC_IDX],
                     "started_utc": datetime.utcnow().isoformat() + "Z"}
        if DG_INFER_ADABN:
            # D2b: re-estimate BN running stats on the target tiles (label-free).
            print("  [infer] AdaBN: recomputing BN stats on target tiles...", flush=True)
            dg.adabn_recalibrate(
                net,
                tc.kp_make_target_batches(scenes, _kp_batch, build_feat,
                                          GRID, CHUNK_XY, NUM_CLASSES),
                forward=lambda m, b: m(b, cfg))
            net.eval()

        def _predict(pc_path):
            z = np.load(pc_path)
            # predict in the scene-local frame (kp_load_canonical's origin
            # shift — the frame the model trained in; global-UTM float32 both
            # quantizes coords and mis-centers windows), return the ORIGINAL
            # georeferenced coords as the deliverable.
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
                "epoch", "train_loss", "train_acc", "train_iou", "offset_loss",
                "lr", "sec_per_epoch", "gpu_mem_mb",
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

    # Final/periodic eval scored on the ORIGINAL raw points: center-weighted
    # softmax votes over the overlapping cached tiles, per-voxel argmax,
    # NN-reprojected to the raw cloud (shared KP-family pipeline).
    def _fwd_eval(tiles):     # [(cxyz, feat)] -> per-tile logits list
        b, _ = make_kp_batch([(c, f, None) for c, f in tiles])
        lg = net(b, cfg).cpu().numpy().astype(np.float32)
        return np.split(lg, np.cumsum([len(c) for c, _ in tiles])[:-1])
    evaluate = tc.kp_make_evaluate(_fwd_eval, build_feat, GRID, CHUNK_XY,
                                   NUM_CLASSES, CLASS_NAMES)

    best = tc.BestCheckpoint(run_dir)
    tc.write_run_manifest(run_dir, "kpconv", dataset)

    def run_eval(ep, write_json=False):
        # Periodic pass scores held-out VAL only and selects the best checkpoint
        # on val present-class mIoU; the final pass (write_json) scores TEST on
        # the best-tracked checkpoint. The swap is skipped in --mode eval.
        #
        # PreciseBN (Yan et al., "Rethinking 'Batch' in BatchNorm"): eval-mode BN
        # runs on EMA stats that lag the fast-moving weights ("moment staleness"),
        # which whipsaws the val curve while train stays smooth. Re-estimate the
        # stats with the CURRENT frozen weights over clean train tiles (cumulative
        # average, single-tile batches to match evaluate()'s forwards) before
        # scoring; best.update then selects on and saves the precise stats.
        # Skipped in EVAL_ONLY: explicit --weights are scored as-shipped.
        if not EVAL_ONLY:
            def _bn_batches(n=48):
                made = 0
                while made < n:
                    s = sample_tile(pick_train_tile(), training=False)
                    if s is None:
                        continue
                    try:
                        b, _ = make_kp_batch([s])
                    except Exception:
                        continue
                    made += 1
                    yield b
            dg.adabn_recalibrate(net, _bn_batches(), forward=lambda mdl, b: mdl(b, cfg))
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
    # Train loop — EPOCH_STEPS optimizer steps of PACK_N tiles/forward x ACCUM.
    # ------------------------------------------------------------------------
    LOG_EVERY = 50
    # Same loop speedups as the kpconvx_cold twin: opt-in bf16 autocast,
    # background pack building (tile load + CPU pyramid overlap the GPU), and
    # a GPU-accumulated confusion matrix (one bincount/forward, one sync/epoch
    # instead of ~4*NUM_CLASSES syncs per forward).
    AMP = os.environ.get("TT_AMP") == "1"
    def _draw():
        while True:
            s = sample_tile(pick_train_tile(), training=True)
            if s is not None:
                return s
    prefetch = (tc.make_prefetcher(
        lambda: make_kp_batch([_draw() for _ in range(PACK_N)]),
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
            g["lr"] = cur_lr * g["lr_mult"]     # deform offsets stay at 0.1x
        net.train()
        ep_loss, ep_reg = 0.0, 0.0
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
                        logits = net(batch, cfg)
                        loss_seg = seg_loss(logits, lab_t)
                        # Deformable fitting/repulsion regularizer — reads state the
                        # forward populates (min_d2/deformed_KP), so it MUST come
                        # after net(...); plain 0 when there are no deformable blocks.
                        reg = p2p_fitting_regularizer(net)
                        loss = (loss_seg + reg) / ACCUM
                    if not torch.isfinite(loss):
                        n_failed += 1
                        continue
                    loss.backward()
                    n_ok += 1; n_fwd += 1
                    ep_loss += loss.item() * ACCUM
                    ep_reg  += float(reg.detach()) if torch.is_tensor(reg) else 0.0
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
                # KPConv's own clip: VALUE clip, not norm (utils/trainer.py).
                torch.nn.utils.clip_grad_value_(net.parameters(), GRAD_CLIP)
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
                f"{train_iou:.4f}", f"{ep_reg/max(n_fwd,1):.6f}",
                f"{cur_lr:.6e}", f"{sec_per_epoch:.2f}", f"{gpu_mem:.1f}",
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

    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(ep, write_json=True)
    if not EVAL_ONLY:
        best.finalize(lambda p: torch.save(
            {"model": net.state_dict(), "epoch": ep}, p))
        # Mark the run complete so AUTO_RESUME won't re-resume it on the next
        # launch (a crashed/retried run has no DONE and is picked back up).
        open(f"{run_dir}/DONE", "w").close()
    print(f"  total wall-clock: {(time.time() - t_run)/3600:.2f} h")


def _self_test():
    """Host-tier wiring check (no GPU, no compiled C++, no /datasets): the
    Config -> architecture -> num_layers/deform_layers derivation, the CPU
    KPFCNN build, and the offset param-group split. The make_kp_batch
    round-trip needs the compiled C++ wrappers, so that check lives in the
    container — any training/inference run exercises it immediately."""
    os.environ.setdefault("MPLBACKEND", "Agg")   # kernel_points imports pyplot
    sys.path.insert(0, _kpconv_root())

    for deformable in (True, False):
        cfg = make_cfg(num_classes=5, in_features_dim=4, grid=2.0,
                       deformable=deformable)
        assert cfg.num_layers == 5, cfg.num_layers
        want = [False, False, False, True, True] if deformable else [False] * 5
        assert cfg.deform_layers == want, cfg.deform_layers
    # split identity: segmentation_inputs returns 5*L+2 items -> L == num_layers
    assert (5 * cfg.num_layers + 2 - 2) // 5 == cfg.num_layers

    from models.architectures import KPFCNN
    cfg = make_cfg(num_classes=5, in_features_dim=4, grid=2.0, deformable=DEFORMABLE)
    with _cwd(_kpconv_root()):      # kernel disposition cache is CWD-relative
        net = KPFCNN(cfg, lbl_values=list(range(5)), ign_lbls=[])   # CPU build
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params > 0
    deform_params = [n for n, _ in net.named_parameters() if "offset" in n]
    other_params  = [n for n, _ in net.named_parameters() if "offset" not in n]
    assert bool(deform_params) == DEFORMABLE, deform_params[:3]
    assert len(deform_params) + len(other_params) == len(list(net.parameters()))
    print(f"ok — num_layers={cfg.num_layers} deform_layers={cfg.deform_layers} "
          f"params={n_params:,} offset_params={len(deform_params)}")


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
    ap.add_argument('--self-test', action='store_true',
                    help='host wiring check: config derivation + CPU model build; '
                         'no GPU, no compiled C++, no dataset')
    args = vars(ap.parse_args())
    if args.pop('self_test'):
        _self_test()
        return
    if args['dataset'] is None and args['mode'] != 'infer':
        ap.error('--dataset is required (a canonical trainer_gui dataset name); '
                 'only --mode infer may omit it.')
    train_kpconv(**args)


if __name__ == "__main__":
    main()
