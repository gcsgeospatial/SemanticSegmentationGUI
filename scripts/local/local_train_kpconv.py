"""
Local training script for the ORIGINAL KPConv (KPConv-PyTorch, HuguesTHOMAS)
on a canonical trainer_gui --dataset (3-folder train/val/test), deformable
KPFCNN, cold start.

  Features  : 4 = [1, intensity, return_number, height], where height is
              tile-relative (z - tile_min_z).
  --hag     : swaps the 4th channel to real PDAL HeightAboveGround, which the
              dataset (or the Inference page's HAG box) must supply. Same 4
              channels / param count -> a clean A/B of tile-relative height vs
              true height above ground.

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
    python local_train_kpconv.py --dataset <name> --hag
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


def gpu_name() -> str:
    """Real CUDA device name for logs/metadata."""
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


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
APP_NAME      = "kpconv"
FEATURE_MODE  = "native"     # [1, intensity, return_number, height]
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

INPUT_CHANNELS = 4           # [1, intensity, return_number, height]

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

# Augmentation (shared recipe; KPConv's own augmentation_transform is never called).
AUG_SCALE_MIN = 0.9
AUG_SCALE_MAX = 1.1
AUG_SYMMETRY_X = True
AUG_NOISE     = 0.05
AUG_COLOR     = 0.8          # P(keep features) = 0.8 -> feature-drop with p 0.2

# --- density domain-generalization (scripts/helper/density.py; see DENSITY_DG.md) ---
DG_DENSITY_AUG = False   # D1: per-tile coarsen the loaded tile to a jittered grid (train only)
DG_COARSEN_MAX = 2.5
DG_P_NATIVE    = 0.5
DG_INFER_ADABN = False   # D2b: recompute BN stats on the target tiles before predicting
DG_INFER_TTA   = 0       # D5: # extra density(scale) views to average at inference (0=off)
DG_LOGDK_FEAT  = False   # D3b: +1 input channel (log k-th-NN distance) -> retrain
DG_LOGDK_K     = 8

# Optimizer — KPConv's NATIVE recipe (utils/trainer.py + train_S3DIS.py).
# One constant swaps to the sibling AdamW 1-cycle; recorded in run.json either
# way, and find_latest_checkpoint refuses to resume across recipes.
OPTIMIZER     = "sgd"        # "sgd" | "adamw"
OPT_TYPE_STR  = "SGD" if OPTIMIZER == "sgd" else "AdamW"
SGD_LR0       = 1e-2
SGD_MOMENTUM  = 0.98
SGD_WD        = 1e-3
LR_DECAY      = 0.1 ** (1.0 / 150.0)   # closed-form per-epoch exponential decay
DEFORM_LR_FACTOR = 0.1       # offset params train at 0.1x LR (KPConv trainer.py)
# AdamW branch (KPConvX 1-cycle), kept so the optimizer swap stays one line:
WEIGHT_DECAY  = 0.05
CYC_LR0       = 1e-4
CYC_LR1       = 5e-3
CYC_RAISE     = 30
CYC_PLATEAU   = 5
CYC_DECREASE10 = 120
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
    # DG flags: env-overridable (GUI "Density generalization" panel / DG_*=1 in
    # the shell). globals()[...] reads the module default; the local shadow is
    # what the nested closures use.
    DG_DENSITY_AUG = dg.env_bool("DG_DENSITY_AUG", globals()["DG_DENSITY_AUG"])
    DG_COARSEN_MAX = dg.env_float("DG_COARSEN_MAX", globals()["DG_COARSEN_MAX"])
    DG_P_NATIVE    = dg.env_float("DG_P_NATIVE", globals()["DG_P_NATIVE"])
    DG_LOGDK_FEAT  = dg.env_bool("DG_LOGDK_FEAT", globals()["DG_LOGDK_FEAT"])
    DG_LOGDK_K     = dg.env_int("DG_LOGDK_K", globals()["DG_LOGDK_K"])
    DG_INFER_ADABN = dg.env_bool("DG_INFER_ADABN", globals()["DG_INFER_ADABN"])
    DG_INFER_TTA   = dg.env_int("DG_INFER_TTA", globals()["DG_INFER_TTA"])
    # Loss / class-balance overrides (GUI "Loss & class balance" panel).
    USE_FOCAL       = dg.env_bool("LOSS_FOCAL", globals()["USE_FOCAL"])
    FOCAL_GAMMA     = dg.env_float("LOSS_FOCAL_GAMMA", globals()["FOCAL_GAMMA"])
    CLASS_WEIGHTING = dg.env_bool("LOSS_CLASS_WEIGHTING", globals()["CLASS_WEIGHTING"])
    WEIGHT_BETA     = dg.env_float("LOSS_WEIGHT_BETA", globals()["WEIGHT_BETA"])
    RARE_OVERSAMPLE = dg.env_bool("RARE_OVERSAMPLE", globals()["RARE_OVERSAMPLE"])

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
    HAG = bool(hag)   # --hag: local vars named `hag` below are per-point ARRAYS
    FEATURE_MODE = "native_hag" if HAG else globals()["FEATURE_MODE"]
    # KPConv pyramid geometry: module constants, restored from run.json at infer
    # so the rebuilt model matches the weights exactly.
    CONV_RADIUS   = globals()["CONV_RADIUS"]
    DEFORM_RADIUS = globals()["DEFORM_RADIUS"]
    KP_EXTENT     = globals()["KP_EXTENT"]
    NUM_KP        = globals()["NUM_KP"]
    FIRST_FEAT    = globals()["FIRST_FEAT"]
    DEFORMABLE    = globals()["DEFORMABLE"]
    NEIGHBOR_LIMITS = None   # resolved below: run.json (infer/resume) or calibration

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
                f"page 'Compute Height-Above-Ground' box, or train the plain KPConv "
                f"(its 4th channel is the tile-relative height, which needs no HAG).")
        HAG_SOURCE = (((ds_meta.get("source") or {}).get("hag_source") or "pdal_hag_nn")
                      if HAG else None)
        PREP_DIR = f"{ds_root}/prep/kpconv{'_hag' if HAG else ''}_grid{GRID:g}_c{int(CHUNK_XY)}"
    else:
        # Folder inference (--mode infer): no --dataset. Reproduce the TRAINED
        # geometry, class layout AND neighbor_limits from the run's run.json (the
        # self-contained manifest beside the weights). INFER never reads cached
        # tiles, so PREP_DIR is an unused placeholder.
        NUM_CLASSES = 5
        CLASS_NAMES = [f"class {i}" for i in range(NUM_CLASSES)]
        PREP_DIR = "/outputs/_infer_unused"
        if INFER and weights:
            rj = _run_json_beside(f"/outputs/{weights}")
            if rj:
                # M5: never load cross-variant weights — both variants are 4-ch,
                # so no shape check can catch it (run.json's hag_source tags them).
                if HAG and not rj.get("hag_source"):
                    raise ValueError(
                        "These weights are from a plain KPConv run (no 'hag_source' in "
                        "run.json): its 4th input channel is the native tile-relative "
                        "height, not real HAG. Re-run inference with the plain "
                        "'KPConv' backbone.")
                if not HAG and rj.get("hag_source"):
                    raise ValueError(
                        "These weights are from a KPConv + HAG run (run.json has "
                        "'hag_source'): its 4th input channel is HeightAboveGround, not "
                        "the native height this script feeds. Re-run inference with the "
                        "'KPConv + HAG' backbone.")
                HAG_SOURCE = rj.get("hag_source")
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

    # ------------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------------
    def grid_subsample(xyz, attrs, lab, voxel):
        """Voxel-grid subsample to `voxel` m: barycenter points, mean attrs,
        majority labels. Mirrors KPConv's grid_subsampling (the C++ op that
        produces the first_subsampling_dl layer-0 cloud) in numpy so the prep
        cache never needs the compiled extensions."""
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
        """Shared trainer_gui augmentation: vertical rotation, anisotropic scale
        in [0.9,1.1] with random x-flip, gaussian noise 0.05. KPConv's own
        augmentation_transform (a PointCloudDataset method) is never called."""
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
    # Preprocessing -> GRID-subsampled .npz tiles (xyz + intensity/ret[/hag])
    # ------------------------------------------------------------------------
    def load_canonical(npz_path):
        """Canonical trainer_gui scene (.npz) -> (xyz, intensity, ret_num, lab).
        xyz is origin-offset (per-scene floor-min) before the float32 cast so
        projected (UTM) coords keep sub-meter precision."""
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
                # tiles are sparse — a higher cut would delete them from training.
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
                if HAG:
                    tile["hag"] = sa[:, 2].astype(np.float32)
                tile["lab"] = sl.astype(np.int32)
                np.savez_compressed(
                    os.path.join(out_dir, f"{name}_x{int(x0)}_y{int(y0)}.npz"), **tile)
                n_tiles += 1
        print(f"      -> {n_tiles} tiles", flush=True)
        return n_tiles

    def _split_scenes():
        # The dataset stage already materialized three whole-scene folders;
        # read them verbatim and never re-carve a split.
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
        # Everything that changes what a cached tile contains — plus the KPConv
        # pyramid geometry, because neighbor_limits.json is keyed by this same
        # signature and stale limits silently change the pyramid.
        sp = ds_meta.get("split", {})
        sig = {
            "format_version": 1,
            "pipeline": "kpconv_hag" if HAG else "kpconv",
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
            "conv_radius": CONV_RADIUS,
            "deform_radius": DEFORM_RADIUS,
            "arch_hash": arch_hash(_arch(DEFORMABLE)),
        }
        if HAG:
            sig["hag_source"] = HAG_SOURCE
        return sig

    def _validate_cache(lists):
        """Refuse to reuse a cache built with different settings instead of
        silently mixing incompatible data."""
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
            "%Y%m%d_%H%M%S_kpconv_hag" if HAG else "%Y%m%d_%H%M%S_kpconv_native")
        run_dir = f"/outputs/runs/{run_id}"
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        return run_id, run_dir

    def find_latest_checkpoint():
        """Most recent run (run-ids are timestamps, so they sort) that has
        checkpoints AND was trained with this script's recipe: optimizer type,
        feature_mode AND architecture must all match. Returns
        (run_dir, ckpt_path, epoch) or None."""
        def _ep(p):
            return int(os.path.basename(p)[2:5])   # ep149.pth -> 149
        my_hash = arch_hash(_arch(DEFORMABLE))
        for rd in sorted(glob.glob("/outputs/runs/*"), reverse=True):
            ckpts = glob.glob(f"{rd}/checkpoints/ep*.pth")
            if not ckpts:
                continue
            opt_type = fmode = ahash = None
            for _cfgp in (f"{rd}/run.json", f"{rd}/run_config.json"):
                try:
                    with open(_cfgp) as f:
                        _rc = json.load(f)
                    opt_type = _rc.get("optimizer", {}).get("type")
                    fmode = _rc.get("feature_mode")
                    ahash = _rc.get("arch_hash")
                    break
                except Exception:
                    continue
            if opt_type != OPT_TYPE_STR:
                print(f"  resume: skipping {os.path.basename(rd)} "
                      f"(recipe mismatch: optimizer={opt_type})", flush=True)
                continue
            # run.json holds the RAW value ("native"/"native_hag") until
            # write_run_manifest merges the normalized one over it ("native"/
            # "hag") — accept either spelling of this variant, or resume would
            # never match a manifest-merged HAG run.
            if fmode not in {FEATURE_MODE, "hag" if HAG else "native"}:
                print(f"  resume: skipping {os.path.basename(rd)} "
                      f"(variant mismatch: feature_mode={fmode})", flush=True)
                continue
            if ahash is not None and ahash != my_hash:
                print(f"  resume: skipping {os.path.basename(rd)} "
                      f"(architecture mismatch: arch_hash={ahash})", flush=True)
                continue
            latest = max(ckpts, key=_ep)
            return rd, latest, _ep(latest)
        return None

    print("=" * 70)
    print(f"  KPConv  {dataset or 'infer'}  COLD/{FEATURE_MODE}  "
          f"({gpu_name()}, {N_EPOCHS} ep, {EPOCH_STEPS} steps, "
          f"pack {PACK_N} x accum {ACCUM}, {OPT_TYPE_STR})")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}  device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    train_list, val_list, test_list = ([], [], []) if INFER else ensure_prep()

    resume_info = find_latest_checkpoint() if (RESUME or EVAL_ONLY) else None
    if INFER:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_infer")
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
            raise RuntimeError(f"eval mode: no {OPT_TYPE_STR}-recipe run with "
                               f"checkpoints found under /outputs")
        run_id, run_dir = make_run_dir()
        resume_ckpt, start_epoch = None, 0

    train_tiles = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    val_tiles   = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
    test_tiles  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
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

    IN_DIM = INPUT_CHANNELS + (1 if DG_LOGDK_FEAT else 0)   # +log d_k (D3b)
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
    # Features + tile sampling (shared recipe; also feeds the calibration probe)
    # ------------------------------------------------------------------------
    def _hag_of(z, xyz):
        """Real per-point HeightAboveGround for a cached --hag tile."""
        if "hag" not in z.files:
            raise ValueError("A --hag tile is missing its 'hag' channel. Rebuild the "
                             "dataset with Height-Above-Ground enabled.")
        return z["hag"].astype(np.float32)

    def build_feat(xyz, intensity, ret_num, hag=None, drop=False):
        """[1, intensity, return_number, height]. height is the passed per-point
        `hag` array (--hag: real HeightAboveGround); when None this is the PLAIN
        recipe, whose 4th channel is z - min(z) over the tile. With `drop`, zero
        the non-bias channels (feature-drop, P=1-AUG_COLOR)."""
        bias = np.ones((len(xyz), 1), np.float32)
        if hag is None:
            hag = (xyz[:, 2] - xyz[:, 2].min()).astype(np.float32)   # native height
        attrs = np.concatenate([intensity[:, None], ret_num[:, None],
                                hag[:, None]], axis=1).astype(np.float32)
        if drop:
            attrs[:, 1:] = 0.0   # keep intensity; drop ret_num/height
        cols = [bias, attrs]
        if DG_LOGDK_FEAT:        # D3b: never dropped — the density signal to condition on
            cols.append(dg.local_density_logdk(xyz, DG_LOGDK_K)[:, None])
        return np.concatenate(cols, axis=1).astype(np.float32)

    def sample_tile(tile_path, max_pts=MAX_TILE_PTS, min_pts=32, training=True):
        z = np.load(tile_path)
        xyz, intensity, ret_num, lab = z["xyz"], z["intensity"], z["ret_num"], z["lab"]
        hag = _hag_of(z, xyz) if HAG else None
        if len(xyz) < min_pts:
            return None
        idx = np.arange(len(xyz))
        if len(idx) > max_pts:
            idx = np.random.choice(idx, max_pts, replace=False)
        xyz, intensity, ret_num, lab = xyz[idx], intensity[idx], ret_num[idx], lab[idx]
        if HAG:
            hag = hag[idx]
        # D1 density jitter: coarsen-only re-subsample; index-consistent across
        # all per-point arrays.
        if training and DG_DENSITY_AUG:
            g_eff = dg.effective_grid(GRID, DG_COARSEN_MAX, DG_P_NATIVE)
            if g_eff > GRID:
                keep = dg.voxel_first_idx(xyz, g_eff)
                xyz, intensity, ret_num, lab = xyz[keep], intensity[keep], ret_num[keep], lab[keep]
                if HAG:
                    hag = hag[keep]
        # height comes from the original (pre-augmentation) z so it stays a
        # meaningful height-above-tile-min feature.
        drop = (training and np.random.rand() > AUG_COLOR)
        feat = build_feat(xyz, intensity, ret_num, hag, drop=drop)
        geo_xyz = augment(xyz) if training else xyz
        geo_xyz = (geo_xyz - geo_xyz.mean(0)).astype(np.float32)
        return geo_xyz, feat, lab.astype(np.int64)

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
                "input_channels": INPUT_CHANNELS,
                "dataset": dataset,
                **({"hag_source": HAG_SOURCE} if HAG else {}),
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
                "optimizer": (
                    {"type": "SGD", "lr0": SGD_LR0, "momentum": SGD_MOMENTUM,
                     "weight_decay": SGD_WD, "deform_lr_factor": DEFORM_LR_FACTOR,
                     "lr_decay_per_epoch": LR_DECAY,
                     "grad_clip_value": GRAD_CLIP, "bn_momentum": BN_MOMENTUM,
                     "label_smoothing": LABEL_SMOOTH}
                    if OPTIMIZER == "sgd" else
                    {"type": "AdamW", "weight_decay": WEIGHT_DECAY,
                     "cyc_lr0": CYC_LR0, "cyc_lr1": CYC_LR1,
                     "cyc_raise": CYC_RAISE, "cyc_plateau": CYC_PLATEAU,
                     "cyc_decrease10": CYC_DECREASE10,
                     "deform_lr_factor": DEFORM_LR_FACTOR,
                     "label_smoothing": LABEL_SMOOTH,
                     "grad_clip_value": GRAD_CLIP, "bn_momentum": BN_MOMENTUM}),
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
    # exactly KPConv's own trainer split. Built before the optimizer branch so
    # it survives either recipe; lr_mult rides along in the group dict.
    deform_params = [p for n, p in net.named_parameters() if "offset" in n]
    other_params  = [p for n, p in net.named_parameters() if "offset" not in n]
    groups = [{"params": other_params, "lr_mult": 1.0}]
    if deform_params:                       # empty when DEFORMABLE=False
        groups.append({"params": deform_params, "lr_mult": DEFORM_LR_FACTOR})
    if OPTIMIZER == "sgd":
        optim = torch.optim.SGD(groups, lr=SGD_LR0, momentum=SGD_MOMENTUM,
                                weight_decay=SGD_WD)
    else:
        optim = torch.optim.AdamW(groups, lr=CYC_LR0, weight_decay=WEIGHT_DECAY)

    def lr_at(ep):
        """SGD: KPConv's closed-form exponential decay (resume at epoch N lands
        on the right lr without replaying the schedule). AdamW: the KPConvX
        1-cycle, kept so the optimizer swap stays one constant."""
        if OPTIMIZER == "sgd":
            return SGD_LR0 * (LR_DECAY ** ep)
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

    # --- class-balanced loss + rare-class oversampling ----------------------
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
    if not INFER:
        print(f"  class counts: {dict(zip(CLASS_NAMES, class_counts.tolist()))}", flush=True)
    if RARE_CLASSES is not None:
        rare_classes = list(RARE_CLASSES)
    else:
        present = class_counts[class_counts > 0]
        thresh = RARE_FREQ_FRAC * float(np.median(present)) if present.size else 0.0
        rare_classes = [c for c in range(NUM_CLASSES) if 0 < class_counts[c] < thresh]
    rare_tiles = ([tp for tp, cnt in tile_counts if cnt[rare_classes].any()]
                  if (RARE_OVERSAMPLE and rare_classes) else [])
    if not INFER:
        print(f"  rare classes: {[CLASS_NAMES[c] for c in rare_classes]}", flush=True)
        print(f"  rare-class tiles: {len(rare_tiles)} / {len(train_tiles)}", flush=True)

    if CLASS_WEIGHTING:
        freq = class_counts / max(int(class_counts.sum()), 1)
        w = (1.0 / np.maximum(freq, 1e-6)) ** WEIGHT_BETA
        w = w / w.mean()                                    # keep loss scale ~1
        w = np.clip(w, 1.0 / WEIGHT_CAP, WEIGHT_CAP)
        class_weights = torch.tensor(w, dtype=torch.float32).cuda()
        if not INFER:
            print(f"  class weights: "
                  f"{dict(zip(CLASS_NAMES, [round(float(x), 3) for x in w]))}", flush=True)
    else:
        class_weights = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1,
                                        label_smoothing=LABEL_SMOOTH)

    # --- Lovász-Softmax: differentiable mIoU surrogate on (N, C) logits -------
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
        # when every label is ignored; return a finite zero-grad value instead.
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

    def scene_hag(z, pc_path, n):
        """Real per-point HAG from an inference scene npz (--hag); None when plain."""
        if not HAG:
            return None
        if "hag" not in z.files or len(z["hag"]) != n:
            raise ValueError(
                f"{os.path.basename(pc_path)} has no per-point 'hag' channel, which this "
                f"HAG model requires. Tick 'Compute Height-Above-Ground' on the Inference "
                f"page and run again.")
        return z["hag"].astype(np.float32)

    def _predict_points(xyz, intensity_n, ret_num, hag=None):
        """Sliding-window KPConv inference over already-normalized features;
        returns per-raw-point class indices (used by --mode infer)."""
        pred = np.full(len(xyz), -1, np.int64)
        with torch.no_grad():
            for idx in tc.xy_chunk_groups(xyz, CHUNK_XY, min_pts=64):
                cols = [intensity_n[idx], ret_num[idx]]
                if hag is not None:
                    cols.append(hag[idx])          # carry real HAG through the voxel mean
                attrs = np.stack(cols, axis=1).astype(np.float32)
                sx, sa, _ = grid_subsample(xyz[idx], attrs, None, GRID)
                if len(sx) < 32:
                    continue
                sub_hag = sa[:, 2] if hag is not None else None
                feat = build_feat(sx, sa[:, 0], sa[:, 1], sub_hag)
                base = (sx - sx.mean(0)).astype(np.float32)
                # D5 density-TTA: average softmax over density(scale) views.
                views = [1.0] + (list(np.linspace(0.85, 1.2, DG_INFER_TTA))
                                 if DG_INFER_TTA else [])
                try:
                    prob = None
                    for s in views:
                        batch, _ = make_kp_batch([((base * s).astype(np.float32), feat, None)])
                        p = torch.softmax(net(batch, cfg).float(), -1).cpu().numpy()
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
    # Arbitrary-folder inference: label the staged npz scenes and stop.
    # ------------------------------------------------------------------------
    if INFER:
        if not infer_input:
            raise ValueError("--mode infer requires --infer-input <job_id>")
        if (grid is not None and grid != GRID) or (chunk_xy is not None and chunk_xy != CHUNK_XY):
            print(f"  [infer] note: KPConv uses its trained geometry "
                  f"(grid={GRID}, chunk={CHUNK_XY}); --grid/--chunk-xy ignored.", flush=True)
        if HAG:
            print("  [infer] 4th channel = real HeightAboveGround; every scene must "
                  "carry a per-point 'hag' array.", flush=True)
        net.eval()
        scenes = sorted(glob.glob(f"/datasets/_infer/{infer_input}/scenes/*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scenes under /datasets/_infer/{infer_input}/scenes")
        pred_dir = f"{run_dir}/predictions"
        infer_cfg = {"backbone": "KPConv-HAG" if HAG else "KPConv", "mode": "infer",
                     "weights": weights,
                     "infer_input": infer_input, "num_classes": NUM_CLASSES,
                     "class_names": CLASS_NAMES, "grid": GRID, "chunk_xy": CHUNK_XY,
                     "neighbor_limits": NEIGHBOR_LIMITS,
                     "hag": (HAG_SOURCE if HAG else False), "gpu": gpu_name(),
                     "started_utc": datetime.utcnow().isoformat() + "Z"}
        print(f"  [infer] labeling {len(scenes)} scene(s) -> {run_dir}/predictions", flush=True)
        if DG_INFER_ADABN:
            # D2b: re-estimate BN running stats on the target tiles (label-free).
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
                    for idx in tc.xy_chunk_groups(txyz, CHUNK_XY, min_pts=64):
                        if seen >= cap:
                            return
                        cols = [tin[idx], trn[idx]] + ([thag[idx]] if HAG else [])
                        attrs = np.stack(cols, 1).astype(np.float32)
                        sx, sa, _ = grid_subsample(txyz[idx], attrs, None, GRID)
                        if len(sx) < 32:
                            continue
                        feat = build_feat(sx, sa[:, 0], sa[:, 1],
                                          sa[:, 2] if HAG else None)
                        cxyz = (sx - sx.mean(0)).astype(np.float32)
                        try:
                            b, _ = make_kp_batch([(cxyz, feat, None)])
                        except Exception:
                            continue
                        seen += 1
                        yield b
            dg.adabn_recalibrate(net, _target_batches(),
                                 forward=lambda m, b: m(b, cfg))
            net.eval()
        scene_stats = []
        for pc_path in scenes:
            name = os.path.splitext(os.path.basename(pc_path))[0]
            t0 = time.time()
            z = np.load(pc_path)
            xyz = z["xyz"].astype(np.float32)
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

    def evaluate(scene_items, label):
        """Final eval scored on the ORIGINAL raw points: per scene, run the
        model over its overlapping cached tiles, sum center-weighted softmax
        votes per voxel, argmax, then propagate to every raw point by nearest
        neighbour and score against the raw GT."""
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
                        batch, _ = make_kp_batch([(cxyz, feat, None)])
                        lg = net(batch, cfg).cpu().numpy().astype(np.float32)
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
        print(f"  [{label}] acc={m['overall_acc']:.4f}  mIoU(all)={m['overall_mIoU']:.4f}  "
              f"mIoU(present {len(present)})={m['present_classes_mIoU']:.4f}  "
              f"absent={m['absent_classes']}  raw_pts={total:,}  "
              f"skipped(tiles={n_skipped_tiles},scenes={n_skipped_scenes})", flush=True)
        return m

    best = tc.BestCheckpoint(run_dir)
    # the single inference manifest (run.json); the _hag key keeps the Infer
    # page mapping HAG runs back to the HAG entry point.
    tc.write_run_manifest(run_dir, "kpconv_hag" if HAG else "kpconv", dataset)

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
        ious = [m["per_class_iou"][CLASS_NAMES[c]] for c in range(NUM_CLASSES)]
        with open(val_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{m['overall_acc']:.4f}",
                                    f"{m['present_classes_mIoU']:.4f}"] + [f"{x:.4f}" for x in ious])
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
    print(f"  starting at epoch {start_epoch}, up to {N_EPOCHS}, "
          f"{EPOCH_STEPS} steps/epoch, pack {PACK_N} x accum {ACCUM}", flush=True)
    t_run = time.time()
    for ep in range(start_epoch, N_EPOCHS):
        cur_lr = lr_at(ep)
        for g in optim.param_groups:
            g["lr"] = cur_lr * g["lr_mult"]     # deform offsets stay at 0.1x
        net.train()
        ep_loss, ep_reg, ep_correct, ep_total = 0.0, 0.0, 0, 0
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
                    batch, lab_t = make_kp_batch(samples)
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
        train_acc = ep_correct / max(ep_total, 1)
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
        if (ep + 1) % VAL_EVERY == 0 and ep != N_EPOCHS - 1:
            run_eval(ep)               # last epoch handled by the final eval below

    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(N_EPOCHS - 1, write_json=True)
    if not EVAL_ONLY:
        best.finalize(lambda p: torch.save(
            {"model": net.state_dict(), "epoch": N_EPOCHS - 1}, p))
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
    ap.add_argument('--hag', action='store_true',
                    help='swap the 4th input channel to real HeightAboveGround')
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
