"""
Modal training script for RandLA-Net (PyTorch) on IEEE GRSS 2019 Track 4 —
COLD-START + HAG variant.

This is the HAG twin of modal_train_randlanet.py. Identical in every way except
it appends a real PDAL HeightAboveGround feature (computed by the trainer_gui
Pretraining tab: SMRF ground-classify -> hag_nn) as an extra input channel:
IN_DIM 5 -> 6 ([xyz, intensity, return_number, HAG]); fc0 is rebuilt at 6->8.
HAG is read per point from /data/IEEE/HAG/{Train,Validate}/<scene>_PC3.laz and
paired to the raw _PC3.txt points. Run it head-to-head against
modal_train_randlanet.py to measure HAG's contribution.

Prereq: upload the HAGTEST output (Pretraining tab) to the ieee-data volume:
    modal volume put ieee-data "C:/Users/OrionHoch/Desktop/HAGTEST/Train"    /IEEE/HAG/Train
    modal volume put ieee-data "C:/Users/OrionHoch/Desktop/HAGTEST/Validate" /IEEE/HAG/Validate
The --dataset (canonical trainer_gui) path has no HAG laz, so it falls back to a
z-scene-min proxy for the 6th channel (keeps IN_DIM fixed at 6).

Random initialization — no pretrained weights. Architecture and features are
identical to the warm sibling (modal_train_randlanet_warm.py), making
warm-vs-cold a clean initialization comparison.

IEEE 2019 Track 4 layout:
  Train-Track4/Track4/JAX_*_PC3.txt, OMA_*_PC3.txt        (110 scenes)
  Train-Track4-Truth/Track4-Truth/JAX_*_CLS.txt, ...      (per-point labels)
  Validate-Track4/Track4/*.txt                            (10 scenes)
  Validate-Track4-Truth/*.txt                             (per-point labels)
  Test-Track4 has no GT — we ignore it.

Each PC3.txt row is `x, y, z, intensity, returnNumber` (comma-separated).
Each CLS.txt is one ASPRS LAS class code per line. We remap
{0:-1 (ignore), 2:0 Ground, 5:1 Trees, 6:2 Building, 9:3 Water, 17:4 Bridge}.

REVISION 2026-06-12 (ported from the KPConvX cold-run fixes):
  - features are [xyz, intensity, return_number] (5 ch). Intensity is
    water's only reliable cue; fc0 is built at 5->8 (cold start, so there is
    no pretrained-stem constraint).
  - p95-clipped per-scene intensity normalization (new PREP_DIR)
  - train augmentation (vertical rotation, x-flip, isotropic scale): was none
  - class-weighted CE + rare-class-centered sphere sampling (rare = train
    frequency < 2%, so canonical --dataset runs work too)
  - held-out val pass every VAL_EVERY epochs -> val_metrics.csv (no weight
    updates), checkpoints include optimizer state
  - final eval now covers every subsampled point of every val/test scene in
    spatially-sorted blocks (the old eval scored ONE random sphere per scene)

Validate-Track4 (10 scenes) is the test set. 10 Train-Track4 scenes picked
deterministically (seed=42) form an in-distribution validation holdout.

The warm-start sibling script is modal_train_randlanet_warm.py.

----------------------------------------------------------------------------
Training-terminal integration: running with no flags behaves exactly as the
original. Extra flags (see modal_train_ptv3_warm.py for details):

  --dataset NAME                          canonical trainer_gui dataset on the
                                          terminal-datasets volume
  --sub-grid / --num-points / --epochs / --batch / --steps-per-epoch
  --mode infer --weights runs/<id>/final_model.pth --infer-input <job_id>

Timeout comes from the TT_TIMEOUT_HOURS env var.
"""

import os
from typing import Optional


def gpu_name() -> str:
    """Real CUDA device name for logs/metadata (replaces the old fixed cloud GPU_TYPE)."""
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def partition_scenes(pool, predefined_test, val_frac, test_frac, seed):
    """Deterministic scene-level train/val/test split (whole scenes, never tiles).

    pool            : scene names available for train (+ val, + test if none predefined).
    predefined_test : scene names already forming a dedicated test set (IEEE
                      Validate-Track4 / a canonical val/ folder). Non-empty -> it wins
                      and test_frac is ignored (a fixed test set must stay fixed).
    val_frac/test_frac : fractions of len(pool); each rounds to a whole scene count
                      (>=1 when positive and the pool has room).
    Returns (train, val, test) sorted name lists; always leaves >=1 train scene.
    """
    import numpy as np
    pool = sorted(pool)
    n = len(pool)
    want = lambda frac, avail: min(max(1, round(frac * n)) if frac > 0 else 0, max(0, avail))
    order = np.random.RandomState(seed).permutation(n)
    n_test = 0 if predefined_test else want(test_frac, n - 2)   # leave room for val + train
    n_val = want(val_frac, n - 1 - n_test)
    pick = lambda lo, hi: sorted(pool[i] for i in order[lo:hi])
    test = sorted(predefined_test) if predefined_test else pick(0, n_test)
    val = pick(n_test, n_test + n_val)
    train = pick(n_test + n_val, n)
    return train, val, test


# ============================================================================
# Configuration
# ============================================================================
APP_NAME      = "randlanet-cold-ieee-hag"
N_EPOCHS      = 100              # was 5 (smoke test); 250-300 for a full run
BATCH_SIZE    = 6
VAL_BATCH     = 12
TIMEOUT_HOURS = int(os.environ.get("TT_TIMEOUT_HOURS", "24"))

NUM_CLASSES   = 5                # IEEE Track 4: Ground, Trees, Building, Water, Bridge
NUM_POINTS    = 45056            # 4096*11, RandLA SemKITTI default
SUB_GRID_SIZE = 0.30             # 30 cm — IEEE LiDAR is sparser (~2 pts/m²) than KITTI
IN_DIM        = 6                # [x, y, z, intensity, return_number, HAG]
VAL_FRAC      = 0.15             # fraction of train scenes held out for in-distribution val
TEST_FRAC     = 0.15             # fraction held out for test — ONLY when no dedicated test
                                 # set exists (IEEE Validate-Track4 / canonical val/ wins)
HOLDOUT_SEED  = 42

# --- density domain-generalization (scripts/helper/density.py; see DENSITY_DG.md) ---
# o = rho*g^2; density-invariant for o>=1, breaks for o<1. RandLA's fixed-N absorbs a
# plain keep-fraction, so D1 jitters the SUB_GRID per sphere; D5 averages density views.
# D2b recomputes BN stats on the target (its AdaBN target batch carries a z-min HAG proxy).
DG_DENSITY_AUG = False   # D1: per-sphere coarser SUB_GRID during training
DG_COARSEN_MAX = 2.0     # = 1/(SUB_GRID_SIZE*sqrt(rho_min)); density sweep-down factor
DG_P_NATIVE    = 0.5     # P(sphere kept at native SUB_GRID_SIZE)
DG_INFER_ADABN = False   # D2b: recompute BN stats on target tiles before predicting (RandLA is pure-BN)
DG_INFER_TTA   = 0       # D5: # extra density(scale) views to average at inference (0=off)
# D3b: explicit local-density input channel (log k-th-NN distance) -> bumps IN_DIM 6->7
# (retrain; old fc0 weights won't load). Pair with DG_DENSITY_AUG so rho varies.
DG_LOGDK_FEAT  = False
DG_LOGDK_K     = 8

# Class balance (rare classes derived from train frequency, so this also works
# for canonical --dataset runs with arbitrary class sets).
CLASS_WEIGHTING  = True
WEIGHT_BETA      = 0.5           # inverse-frequency exponent. 0.5 == inverse
                                 # SQRT frequency (w = 1/sqrt(freq)), the
                                 # RandLA-Net / SemanticKITTI standard: sub-linear
                                 # so rare classes are boosted without the
                                 # exploding-gradient instability of raw 1/freq.
WEIGHT_CAP       = 5.0           # clamp weights to [1/CAP, CAP] after mean-norm
# Lovász-Softmax: a tractable surrogate that optimizes mIoU (Jaccard) directly
# and weights every class equally, so it counters CE's majority-class bias on
# rare classes. Total loss = <pointwise> + LOVASZ_WEIGHT * lovasz_softmax.
# Set to 0.0 to disable (recovers a pointwise-only loss).
LOVASZ_WEIGHT    = 1.0
# Focal loss (Lin et al. 2017): when USE_FOCAL, the pointwise term is
# alpha-balanced focal loss instead of weighted cross-entropy. alpha reuses the
# inverse-sqrt class weights; (1-p_t)^gamma down-weights easy/well-classified
# points so hard + rare points dominate the gradient. gamma=0 == weighted CE.
# USE_FOCAL=False reverts the pointwise term to weighted CE.
USE_FOCAL        = False
FOCAL_GAMMA      = 2.0
RARE_OVERSAMPLE  = True
RARE_FREQ_THRESH = 0.02          # classes under 2% of train points count as rare
RARE_CENTER_PROB = 0.25          # P(center the next train sphere on a rare-class point)
VAL_EVERY        = 10            # held-out val pass every N epochs (no weight updates)
VAL_BATCHES      = 16            # batches per periodic val pass

DATA_ROOT      = "/data/IEEE"
TRAIN_PC_DIR   = f"{DATA_ROOT}/Train-Track4/Track4"
TRAIN_CLS_DIR  = f"{DATA_ROOT}/Train-Track4-Truth/Track4-Truth"
TEST_PC_DIR    = f"{DATA_ROOT}/Validate-Track4/Track4"
TEST_CLS_DIR   = f"{DATA_ROOT}/Validate-Track4-Truth"
# Per-point HeightAboveGround from the Pretraining tab (HAGTEST upload). Train +
# val-holdout scenes -> HAG/Train; the test split (Validate-Track4) -> HAG/Validate.
HAG_TRAIN_DIR  = f"{DATA_ROOT}/HAG/Train"
HAG_TEST_DIR   = f"{DATA_ROOT}/HAG/Validate"
PREP_DIR       = f"{DATA_ROOT}/prep/randlanet_hag_grid30_p95_origin"   # p95 intensity norm + HAG

# ASPRS LAS code -> contiguous 0..4 IEEE class index. Class 0 (Unclassified)
# maps to -1 and is ignored by the CE loss.
CLASS_NAMES = ["Ground", "Trees", "Building", "Water", "Bridge"]
LABEL_MAP   = {0: -1, 2: 0, 5: 1, 6: 2, 9: 3, 17: 4}

DATASETS_ROOT = "/datasets"   # terminal-datasets volume (trainer_gui canonical datasets)

# ============================================================================
# Image
# ============================================================================


# Local stand-ins for the modal Volumes the body commits to: the data is
# already on the bind-mounted /data /outputs /datasets, so there is nothing
# to upload. (The modal shell does the real commits when it runs this.)
class _NoVol:
    def commit(self, *a, **k): pass
    def reload(self, *a, **k): pass


data_volume, outputs_volume, datasets_volume = _NoVol(), _NoVol(), _NoVol()


def train_randlanet(dataset: Optional[str] = None, sub_grid: Optional[float] = None,
                    num_points: Optional[int] = None, epochs: Optional[int] = None,
                    batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                    mode: str = "train", weights: Optional[str] = None,
                    infer_input: Optional[str] = None):
    import os, sys, time, json, csv, glob
    from datetime import datetime
    import numpy as np
    import torch
    import laspy
    import torch.nn as nn
    import torch.optim as optim
    from scipy.spatial import cKDTree
    from torch.utils.data import DataLoader, Dataset
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helper"))
    import density as dg
    # DG flags: env-overridable (GUI "Density generalization" panel / DG_*=1 in the shell).
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

    # --- resolve config: CLI args override the module defaults ---------------
    SUB_GRID_SIZE = sub_grid if sub_grid is not None else globals()["SUB_GRID_SIZE"]
    # D3b: effective input dim grows by 1 when the log-d_k channel is on. Shadow IN_DIM
    # here (before build_net) so build_net's default, the checkpoint in-dim check, and the
    # run-config all track it.
    IN_DIM = globals()["IN_DIM"] + (1 if DG_LOGDK_FEAT else 0)
    NUM_POINTS    = num_points if num_points is not None else globals()["NUM_POINTS"]
    N_EPOCHS      = epochs if epochs is not None else globals()["N_EPOCHS"]
    BATCH_SIZE    = batch if batch is not None else globals()["BATCH_SIZE"]
    STEPS         = steps_per_epoch if steps_per_epoch is not None else 500
    NUM_CLASSES   = globals()["NUM_CLASSES"]
    CLASS_NAMES   = globals()["CLASS_NAMES"]
    HAG_SOURCE    = "pdal_hag_nn"

    ds_root = f"{DATASETS_ROOT}/{dataset}" if dataset else None
    if ds_root:
        meta_path = f"{ds_root}/dataset_meta.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{meta_path} not found — upload the dataset "
                                    f"with the trainer_gui app first.")
        with open(meta_path) as f:
            ds_meta = json.load(f)
        NUM_CLASSES = int(ds_meta["num_classes"])
        CLASS_NAMES = list(ds_meta["class_names"])
        HAG_SOURCE = "pdal_hag_nn" if ds_meta.get("has_hag") else "z_minus_scene_min_proxy"
        # No "warm" in the name: the cold sibling shares this cache (identical prep).
        PREP_DIR = f"{ds_root}/prep/randlanet_hag_grid{int(round(SUB_GRID_SIZE * 100))}_p95"
    else:
        PREP_DIR = globals()["PREP_DIR"]

    # utils/metric.py calls sklearn.metrics.confusion_matrix(y_true, y_pred,
    # np.arange(...)) — the third positional was the `labels` argument in old
    # sklearn but became `sample_weight` in ≥0.24. Wrap it so the old call
    # still works.
    import sklearn.metrics as _skm
    _orig_cm = _skm.confusion_matrix
    def _cm_compat(y_true, y_pred, labels=None, **kwargs):
        return _orig_cm(y_true, y_pred, labels=labels, **kwargs)
    _skm.confusion_matrix = _cm_compat

    sys.path.insert(0, "/opt/randlanet")
    from network.RandLANet import Network
    import network.pytorch_utils as pt_utils
    from utils.metric import compute_acc, IoUCalculator
    from utils.data_process import DataProcessing as DP

    def build_net(num_classes_, in_dim=IN_DIM):
        """Network with fc0 rebuilt for in_dim channels. Upstream hardcodes
        Conv1d(3, 8) (xyz only); we feed [xyz, intensity, return_number]."""
        cfg.num_classes = num_classes_
        net_ = Network(cfg)
        if in_dim != 3:
            net_.fc0 = pt_utils.Conv1d(in_dim, 8, kernel_size=1, bn=True)
        return net_.to(device)

    # --- Lovász-Softmax (Berman et al. 2018): differentiable surrogate for the
    # mIoU/Jaccard index. Operates per-class on softmax probabilities, treating
    # every present class equally, which is exactly what CE fails to do under
    # heavy imbalance. Pure-PyTorch flat implementation (no extra deps). -------
    def _lovasz_grad(gt_sorted):
        # gradient of the Lovász extension of the Jaccard loss w.r.t. sorted errors
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1].clone()
        return jaccard

    def lovasz_softmax_flat(probas, labels):
        # probas: (N, C) softmax probabilities; labels: (N,) in [0, C). Averages
        # the per-class loss over classes actually present in the batch.
        if probas.numel() == 0:
            return probas * 0.0
        C = probas.size(1)
        losses = []
        for c in torch.unique(labels):
            fg = (labels == c).float()                 # foreground mask for class c
            class_pred = probas[:, int(c)]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            grad = _lovasz_grad(fg[perm])
            losses.append(torch.dot(errors_sorted, grad))
        if not losses:
            return probas.sum() * 0.0
        return torch.stack(losses).mean()

    # alpha-balanced multiclass focal loss. logits (M, C) / labels (M,) already
    # valid (no -1). alpha = _class_w (inverse-sqrt class weights) when set.
    def focal_loss(logits, labels):
        if logits.numel() == 0:
            return logits.sum() * 0.0
        logp = torch.log_softmax(logits, dim=1)
        logpt = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** FOCAL_GAMMA) * logpt
        if _class_w is not None:
            loss = loss * _class_w[labels]
        return loss.mean()

    # The upstream network/loss_func.compute_loss hardcodes label==0 as ignored
    # (SemanticKITTI convention). Our labels are already remapped to 0..K-1
    # at preprocessing with the ignored value(s) -> -1. Pointwise term is focal
    # or weighted CE (ignore_index=-1), optionally combined with Lovász.
    _class_w = None
    _ce = nn.CrossEntropyLoss(ignore_index=-1)
    def compute_loss(end_points, num_classes):
        logits = end_points["logits"].transpose(1, 2).reshape(-1, num_classes)
        labels = end_points["labels"].reshape(-1)
        valid_mask = (labels >= 0) & (labels < num_classes)
        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        if USE_FOCAL:
            loss = focal_loss(valid_logits, valid_labels)
        else:
            loss = _ce(valid_logits, valid_labels)
        if LOVASZ_WEIGHT > 0:
            probas = torch.softmax(valid_logits, dim=1)
            loss = loss + LOVASZ_WEIGHT * lovasz_softmax_flat(probas, valid_labels)
        end_points["valid_logits"] = valid_logits
        end_points["valid_labels"] = valid_labels
        end_points["loss"] = loss
        return loss, end_points

    # --- config (mirrors ConfigSemanticKITTI but tuned for sparse aerial) ----
    class Cfg:
        k_n = 16
        num_layers = 4
        num_points = NUM_POINTS
        num_classes = NUM_CLASSES
        sub_grid_size = SUB_GRID_SIZE
        batch_size = BATCH_SIZE
        val_batch_size = VAL_BATCH
        train_steps = STEPS
        val_steps = 50
        sub_sampling_ratio = [4, 4, 4, 4]
        d_out = [16, 64, 128, 256]
        num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]
        noise_init = 3.5
        max_epoch = N_EPOCHS
        learning_rate = 1e-2
        lr_decays = {i: 0.95 for i in range(0, N_EPOCHS + 1)}
        train_sum_dir = "train_log"
        saving = True
        saving_path = None
    cfg = Cfg()

    # --- loaders + label remap ----------------------------------------------
    # Build the ASPRS-code -> contiguous-index LUT once.
    LABEL_LUT = np.full(256, -1, dtype=np.int32)
    for raw, mapped in LABEL_MAP.items():
        LABEL_LUT[raw] = mapped

    # --- HAG: per-point HeightAboveGround from the Pretraining-tab .laz files,
    # paired to the raw scene points. PDAL preserves point order, so index-pair
    # when counts match (sample-verified); else nearest-neighbor on xyz. --------
    def _read_hag_laz(laz_path):
        las = laspy.read(laz_path)
        names = list(las.point_format.dimension_names)
        hname = next((n for n in names if n.lower() in ("heightaboveground", "hag")), None)
        if hname is None:
            raise KeyError(f"{laz_path}: no HeightAboveGround dim (have {names})")
        hag = np.asarray(las[hname], dtype=np.float32)
        laz_xyz = np.column_stack([las.x, las.y, las.z])   # float64: absolute UTM
        return hag, laz_xyz                                 # needs full precision for pairing

    def load_hag(name, hag_dir, xyz_ref):
        laz_path = f"{hag_dir}/{name}_PC3.laz"
        if not os.path.exists(laz_path):
            raise FileNotFoundError(
                f"HAG file missing: {laz_path}. Upload the Pretraining-tab output:\n"
                f'  modal volume put ieee-data "<HAGTEST>/Train" /IEEE/HAG/Train\n'
                f'  modal volume put ieee-data "<HAGTEST>/Validate" /IEEE/HAG/Validate')
        hag, laz_xyz = _read_hag_laz(laz_path)
        # xyz_ref is offset to a per-scene origin (precision fix) while the laz
        # carries absolute coords; subtract each array's own per-axis min so the
        # index-pair guard and NN fallback compare in a common full-precision frame.
        ref0 = xyz_ref.astype(np.float64) - xyz_ref.min(0)
        laz0 = laz_xyz - laz_xyz.min(0)
        if len(hag) == len(xyz_ref):
            k = min(2048, len(hag))
            s = np.random.RandomState(0).choice(len(hag), k, replace=False)
            if np.allclose(laz0[s], ref0[s], atol=0.05):
                return hag
        nn = cKDTree(laz0).query(ref0)[1]
        return hag[nn].astype(np.float32)

    def load_ieee(pc_path, cls_path):
        # PC3.txt: x, y, z, intensity, returnNumber  (CSV)
        # CLS.txt: one ASPRS class code per line
        pc = np.loadtxt(pc_path, delimiter=",")                 # float64 (full precision)
        # Per-scene origin offset before the float32 cast: projected (UTM) coords
        # otherwise quantize to ~0.25-0.5 m on northing (float32 ~7 sig digits),
        # corrupting sub-meter geometry. Deterministic offset, so cached tiles and
        # the eval-time raw reload share one coordinate frame.
        xyz       = (pc[:, :3] - np.floor(pc[:, :3].min(0))).astype(np.float32)
        intensity = pc[:, 3].astype(np.float32)
        ret_num   = pc[:, 4].astype(np.float32) if pc.shape[1] >= 5 else np.zeros(len(pc), np.float32)
        lab_raw   = np.loadtxt(cls_path, dtype=np.int32).reshape(-1)
        if len(lab_raw) != len(xyz):
            raise ValueError(f"point/label mismatch in {pc_path}: "
                             f"{len(xyz)} pts vs {len(lab_raw)} labels")
        lab = LABEL_LUT[np.clip(lab_raw, 0, 255)].astype(np.int32)
        # p95 + clip instead of raw values: one hot return must not rescale the
        # scene (water's intensity cue has to mean the same thing everywhere).
        i_p95 = max(float(np.percentile(intensity, 95)), 1.0)
        intensity = np.clip(intensity / i_p95, 0.0, 2.0).astype(np.float32)
        return xyz, intensity, ret_num, lab

    def load_canonical(npz_path):
        """Canonical trainer_gui scene -> the same tuple load_ieee produces.
        RandLA only consumes xyz + labels; the rest is kept for cache parity."""
        z = np.load(npz_path)
        xyz = z["xyz"].astype(np.float32)
        intensity = z["intensity"].astype(np.float32) if "intensity" in z \
            else np.full(len(xyz), 0.5, np.float32)
        ret_num = z["return_number"].astype(np.float32) if "return_number" in z \
            else np.zeros(len(xyz), np.float32)
        lab = z["label"].astype(np.int32) if "label" in z \
            else np.full(len(xyz), -1, np.int32)
        return xyz, intensity, ret_num, lab

    def load_scene(pc_path, cls_path=None):
        if pc_path.endswith(".npz"):
            return load_canonical(pc_path)
        return load_ieee(pc_path, cls_path)

    def load_canonical_hag(npz_path, xyz):
        z = np.load(npz_path)
        if "hag" in z.files and len(z["hag"]) == len(xyz):
            return z["hag"].astype(np.float32)
        return (xyz[:, 2] - xyz[:, 2].min()).astype(np.float32)

    def grid_subsample(xyz, intensity, ret_num, hag, lab, grid):
        keys = np.floor(xyz / grid).astype(np.int64)
        _, uniq = np.unique(keys, axis=0, return_index=True)
        return xyz[uniq], intensity[uniq], ret_num[uniq], hag[uniq], lab[uniq]

    def _split_scenes():
        """Deterministic scene-level split; returns (name, pc_path, cls_path) lists."""
        if ds_root:
            # VAL_FRAC of the train scenes -> val; the dataset's val/ folder is the
            # dedicated test set, or TEST_FRAC carves one from train when val/ is empty.
            train_npz = sorted(glob.glob(f"{ds_root}/train/*.npz"))
            if not train_npz:
                raise FileNotFoundError(f"No canonical scenes under {ds_root}/train")
            test_npz = sorted(glob.glob(f"{ds_root}/val/*.npz"))
            stem = lambda p: os.path.splitext(os.path.basename(p))[0]
            path_of = {stem(p): p for p in train_npz + test_npz}
            train_names, val_names, test_names = partition_scenes(
                [stem(p) for p in train_npz], [stem(p) for p in test_npz],
                VAL_FRAC, TEST_FRAC, HOLDOUT_SEED)
            return ([(n, path_of[n], None) for n in train_names],
                    [(n, path_of[n], None) for n in val_names],
                    [(n, path_of[n], None) for n in test_names])
        train_pc  = sorted(glob.glob(f"{TRAIN_PC_DIR}/*_PC3.txt"))
        if not train_pc:
            raise FileNotFoundError(
                f"No *_PC3.txt under {TRAIN_PC_DIR}. Upload the IEEE dataset to the "
                f"ieee-data volume first, e.g.:\n"
                f"  modal volume put ieee-data "
                f'"C:\\Users\\OrionHoch\\Desktop\\LabledDatasets\\IEEE" /IEEE\n'
                f"then verify: modal volume ls ieee-data /IEEE/Train-Track4/Track4")
        name_of = lambda p: os.path.basename(p).replace("_PC3.txt", "")
        test_pc = sorted(glob.glob(f"{TEST_PC_DIR}/*_PC3.txt"))
        train_names, val_names, test_names = partition_scenes(
            [name_of(p) for p in train_pc], [name_of(p) for p in test_pc],
            VAL_FRAC, TEST_FRAC, HOLDOUT_SEED)
        def _pair(names, pc_dir, cls_dir):
            return [(n, f"{pc_dir}/{n}_PC3.txt", f"{cls_dir}/{n}_CLS.txt")
                    for n in names]
        # carved-from-train test (no dedicated test set) lives in the TRAIN dirs
        test_dirs = (TEST_PC_DIR, TEST_CLS_DIR) if test_pc else (TRAIN_PC_DIR, TRAIN_CLS_DIR)
        return (
            _pair(train_names, TRAIN_PC_DIR, TRAIN_CLS_DIR),
            _pair(val_names,   TRAIN_PC_DIR, TRAIN_CLS_DIR),
            _pair(test_names,  *test_dirs),
        )

    def _cache_signature():
        # Everything that changes what a cached scene .npz contains. A mismatch
        # means the cache is stale/leaky and must not be silently reused.
        return {
            "format_version": 1,
            "pipeline": "randlanet_hag",
            "dataset": dataset or "IEEE_Track4",
            "sub_grid_size": SUB_GRID_SIZE,
            "num_classes": NUM_CLASSES,
            "holdout_seed": HOLDOUT_SEED,
            "val_frac": VAL_FRAC,
            "test_frac": TEST_FRAC,
            "label_map": (None if ds_root
                          else {str(k): v for k, v in sorted(LABEL_MAP.items())}),
            "feature_recipe": "xyz,intensity,return_number,hag",
            "hag_source": HAG_SOURCE,
            "hag_train_dir": (None if ds_root else HAG_TRAIN_DIR),
            "hag_test_dir": (None if ds_root else HAG_TEST_DIR),
        }

    def _validate_cache(lists):
        """Refuse to reuse a cache built with different settings (grid, split
        seed, label map, dataset, feature recipe, HAG source …) instead of
        silently mixing incompatible data. Migrate a pre-validation cache by
        stamping .done markers for already-saved scenes. Returns True if the
        signature file was newly written (so the caller commits the volume)."""
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
                    f"data. Point PREP_DIR / --dataset at a fresh path or delete the cache.")
            return False
        legacy = False
        for split, items in lists:
            d = f"{PREP_DIR}/{split}"
            for name, _, _ in items:
                npz = f"{d}/{name}.npz"
                if os.path.exists(npz) and not os.path.exists(npz + ".done"):
                    open(npz + ".done", "w").close(); legacy = True
        with open(meta_path, "w") as f:
            json.dump(cur, f, indent=2)
        if legacy:
            print(f"  migrated existing cache at {PREP_DIR}: stamped .done markers + "
                  f"signature (assumed to match current settings).", flush=True)
        return True

    def ensure_prep():
        # Per-scene idempotency keyed on a .done marker (written only after a
        # clean save) so a run interrupted mid-scene is redone, not left partial.
        print(f"  ensuring preprocessed cache -> {PREP_DIR}", flush=True)
        for split in ("train", "val", "test"):
            os.makedirs(f"{PREP_DIR}/{split}", exist_ok=True)
        train_list, val_list, test_list = _split_scenes()
        any_new = _validate_cache([("train", train_list), ("val", val_list),
                                   ("test", test_list)])
        for split, items in (("train", train_list), ("val", val_list),
                             ("test", test_list)):
            # train + val-holdout scenes are Train-Track4 -> HAG/Train; test is
            # Validate-Track4 -> HAG/Validate. A test set carved from train (no
            # dedicated Validate set) is Train-origin -> HAG/Train. Canonical
            # --dataset reads HAG from the npz, so hag_dir is unused there.
            dedicated_test = ds_root or glob.glob(f"{TEST_PC_DIR}/*_PC3.txt")
            hag_dir = (HAG_TEST_DIR if split == "test" and dedicated_test
                       else HAG_TRAIN_DIR)
            print(f"  [{split}] {len(items)} scenes", flush=True)
            for i, (name, pc_path, cls_path) in enumerate(items):
                out = f"{PREP_DIR}/{split}/{name}.npz"
                if os.path.exists(out + ".done"):
                    continue
                t0 = time.time()
                try:
                    xyz, intensity, ret_num, lab = load_scene(pc_path, cls_path)
                    n_in = len(xyz)
                    hag = (load_canonical_hag(pc_path, xyz)
                           if ds_root else load_hag(name, hag_dir, xyz))
                    xyz, intensity, ret_num, hag, lab = grid_subsample(
                        xyz, intensity, ret_num, hag, lab, SUB_GRID_SIZE)
                except Exception as e:
                    print(f"  skip {pc_path}: {e}", flush=True); continue
                np.savez_compressed(
                    out,
                    xyz=xyz.astype(np.float32),
                    intensity=intensity.astype(np.float32),
                    ret_num=ret_num.astype(np.float32),
                    hag=hag.astype(np.float32),
                    lab=lab.astype(np.int32))
                open(out + ".done", "w").close()      # mark complete after a clean write
                any_new = True
                print(f"    [{i+1}/{len(items)}] {name}: {n_in:,} -> "
                      f"{len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
        if any_new:
            (datasets_volume if ds_root else data_volume).commit()
            print("  preprocessing committed.", flush=True)
        else:
            print("  all scenes already cached.", flush=True)
        return train_list, val_list, test_list

    # --- batch assembly (standalone so inference can reuse it) ---------------
    def tf_map(batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        # Features = xyz (3ch) to match the SemKITTI pretrained encoder.
        features = batch_pc
        input_points, input_neighbors, input_pools, input_up = [], [], [], []
        for i in range(cfg.num_layers):
            neigh = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            sub_n = batch_pc.shape[1] // cfg.sub_sampling_ratio[i]
            sub_points = batch_pc[:, :sub_n, :]
            pool_i = neigh[:, :sub_n, :]
            up_i  = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc); input_neighbors.append(neigh)
            input_pools.append(pool_i);    input_up.append(up_i)
            batch_pc = sub_points
        flat = (input_points + input_neighbors + input_pools + input_up
                + [features, batch_label, batch_pc_idx, batch_cloud_idx])
        return flat

    def collate_fn(batch):
        # Items are (pc, extra_feats(3ch), label, point_idx, cloud_idx); the
        # network input is [xyz, intensity, return_number, HAG] = IN_DIM channels.
        pcs, feats, lbs, idxs, cinds = zip(*batch)
        pcs  = np.stack(pcs);  feats = np.stack(feats); lbs = np.stack(lbs)
        idxs = np.stack(idxs); cinds = np.stack(cinds)
        flat = tf_map(pcs, lbs, idxs, cinds)
        n = cfg.num_layers
        d = {"xyz": [], "neigh_idx": [], "sub_idx": [], "interp_idx": []}
        for t in flat[:n]:           d["xyz"].append(torch.from_numpy(t).float())
        for t in flat[n:2*n]:        d["neigh_idx"].append(torch.from_numpy(t).long())
        for t in flat[2*n:3*n]:      d["sub_idx"].append(torch.from_numpy(t).long())
        for t in flat[3*n:4*n]:      d["interp_idx"].append(torch.from_numpy(t).long())
        full_feat = np.concatenate([pcs, feats], axis=2)        # (B, N, IN_DIM)
        d["features"] = torch.from_numpy(full_feat).float().transpose(1, 2)
        d["labels"]   = torch.from_numpy(flat[4*n+1]).long()
        d["input_inds"] = torch.from_numpy(flat[4*n+2]).long()
        d["cloud_inds"] = torch.from_numpy(flat[4*n+3]).long()
        return d

    # --- Dataset ------------------------------------------------------------
    class Scenes(Dataset):
        def __init__(self, split, files=None, label=None):
            self.split = split
            self.label = label or split
            if files is None:
                files = sorted(glob.glob(f"{PREP_DIR}/{split}/*.npz"))
            self.files = files
            self.scenes = [self._load(f) for f in self.files]
            # rare_idx / rare_scenes are filled in after the class-frequency
            # scan (training mode only); until then sampling is uniform.
            self.rare_idx, self.rare_scenes = None, None
            print(f"  [{self.label}] {len(self.scenes)} scenes", flush=True)

        @staticmethod
        def _load(f):
            z = np.load(f)
            # HAG-prepped tiles carry "hag"; tolerate legacy caches via z-min proxy.
            hag = (z["hag"] if "hag" in z.files
                   else (z["xyz"][:, 2] - z["xyz"][:, 2].min()).astype(np.float32))
            return (z["xyz"], z["intensity"], z["ret_num"], hag, z["lab"])

        def set_rare_classes(self, rare_classes):
            self.rare_idx = [np.where(np.isin(lab, rare_classes))[0]
                             for _, _, _, _, lab in self.scenes]
            self.rare_scenes = [i for i, r in enumerate(self.rare_idx) if len(r)]

        def sample_sphere(self, cloud_idx, center_idx, augment=False, rng=np.random):
            xyz, intensity, ret_num, hag, lab = self.scenes[cloud_idx]
            center = xyz[center_idx:center_idx + 1]
            d2 = np.sum((xyz - center) ** 2, axis=1)
            sel = np.argpartition(d2, min(cfg.num_points, len(xyz) - 1))[:cfg.num_points]
            if len(sel) < cfg.num_points:
                sel = np.concatenate([sel, rng.choice(len(xyz), cfg.num_points - len(sel))])
            rng.shuffle(sel)
            # D1 density jitter: re-subsample the sphere to a coarser grid (fixed-N
            # preserved by padding from the sparser set). hag[sel] downstream stays
            # index-consistent because it is indexed by the same `sel`.
            if augment and DG_DENSITY_AUG:
                g_eff = dg.effective_grid(SUB_GRID_SIZE, DG_COARSEN_MAX, DG_P_NATIVE, rng=rng)
                if g_eff > SUB_GRID_SIZE:
                    keep = dg.voxel_first_idx(xyz[sel], g_eff)
                    sel = sel[keep]
                    if len(sel) < cfg.num_points:
                        sel = np.concatenate([sel, rng.choice(sel, cfg.num_points - len(sel))])
                    rng.shuffle(sel)
            pc = (xyz[sel] - center).astype(np.float32)
            if augment:
                theta = rng.rand() * 2 * np.pi
                cs, sn = np.cos(theta), np.sin(theta)
                R = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], np.float32)
                pc = pc @ R.T
                if rng.rand() < 0.5:
                    pc[:, 0] *= -1.0
                pc = pc * np.float32(rng.uniform(0.9, 1.1))
            feat2 = np.stack([intensity[sel], ret_num[sel], hag[sel]]
                             + ([dg.local_density_logdk(pc, DG_LOGDK_K)] if DG_LOGDK_FEAT else []),
                             axis=1).astype(np.float32)   # +log d_k (D3b) -> IN_DIM=7
            lb = lab[sel].astype(np.int64)
            return pc.astype(np.float32), feat2, lb, sel.astype(np.int32), \
                   np.array([cloud_idx], dtype=np.int32)

        def __len__(self):
            return cfg.train_steps * BATCH_SIZE if self.split == "train" else len(self.scenes)

        def __getitem__(self, idx):
            if self.split == "train":
                if (self.rare_scenes and RARE_OVERSAMPLE
                        and np.random.rand() < RARE_CENTER_PROB):
                    ci = self.rare_scenes[np.random.randint(len(self.rare_scenes))]
                    pick = int(self.rare_idx[ci][np.random.randint(len(self.rare_idx[ci]))])
                else:
                    ci = np.random.randint(len(self.scenes))
                    pick = np.random.randint(len(self.scenes[ci][0]))
                return self.sample_sphere(ci, pick, augment=True)
            return self.sample_sphere(idx, np.random.randint(len(self.scenes[idx][0])),
                                      augment=False)

    device = torch.device("cuda")

    # --- Prediction helpers (shared by post-training demo + infer mode) ------
    import traceback
    from scipy.spatial import cKDTree
    IDX_TO_ASPRS = np.array([2, 5, 6, 9, 17], dtype=np.int32)
    BASE_PALETTE = np.array([
        [139, 90, 43], [34, 160, 34], [200, 60, 60], [40, 110, 220], [235, 225, 60],
        [150, 80, 200], [240, 140, 40], [70, 200, 200], [220, 100, 170], [120, 120, 120],
        [90, 140, 60], [180, 180, 90], [60, 60, 160], [200, 170, 130], [100, 220, 120],
        [230, 70, 110], [50, 160, 110], [170, 110, 60], [110, 170, 230], [240, 200, 160],
    ], dtype=np.int32)

    def _palette(num_classes):
        reps = -(-num_classes // len(BASE_PALETTE))
        return np.tile(BASE_PALETTE, (reps, 1))[:num_classes]

    def _write_ply(path, xyz, pred_idx, palette, intensity=None):
        cols = [xyz.astype(np.float32), palette[pred_idx]]
        props = ("property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fmt = ["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"]
        if intensity is not None:            # carry per-point intensity for the viewer
            cols.append(np.asarray(intensity, np.float32).reshape(-1, 1))
            props += "property float intensity\n"
            fmt.append("%.5f")
        header = "ply\nformat ascii 1.0\n" + f"element vertex {len(xyz)}\n" + props + "end_header"
        np.savetxt(path, np.column_stack(cols), fmt=fmt, header=header, comments="")

    def make_predict_scene(net, num_classes):
        def _predict_scene(pc_path):
            # RandLA works on fixed NUM_POINTS samples: grid-subsample the scene,
            # spatially sort it for locality, predict it in NUM_POINTS blocks
            # (reusing the collate), then NN-propagate to all original points.
            if pc_path.endswith(".npz"):
                z = np.load(pc_path)
                xyz0 = z["xyz"].astype(np.float32)
                itn0 = z["intensity"].astype(np.float32) if "intensity" in z \
                    else np.full(len(xyz0), 0.5, np.float32)
                ret0 = z["return_number"].astype(np.float32) if "return_number" in z \
                    else (z["ret_num"].astype(np.float32) if "ret_num" in z
                          else np.zeros(len(xyz0), np.float32))
            else:
                pc = np.loadtxt(pc_path, delimiter=",")          # float64 (full precision)
                # Per-scene origin offset before float32 cast (precision fix).
                xyz0 = (pc[:, :3] - np.floor(pc[:, :3].min(0))).astype(np.float32)
                itn0 = pc[:, 3].astype(np.float32) if pc.shape[1] >= 4 \
                    else np.full(len(xyz0), 0.5, np.float32)
                i_p95 = max(float(np.percentile(itn0, 95)), 1.0)
                itn0 = np.clip(itn0 / i_p95, 0.0, 2.0).astype(np.float32)
                ret0 = pc[:, 4].astype(np.float32) if pc.shape[1] >= 5 \
                    else np.zeros(len(xyz0), np.float32)
            # Predict scenes (IEEE Test-Track4 / canonical infer npz) have no HAG
            # laz -> z-scene-min proxy for the 6th channel (best-effort demo).
            hag0 = (z["hag"].astype(np.float32)
                    if pc_path.endswith(".npz") and "hag" in z.files
                    else (xyz0[:, 2] - xyz0[:, 2].min()).astype(np.float32))
            keys = np.floor(xyz0 / SUB_GRID_SIZE).astype(np.int64)
            _, uniq = np.unique(keys, axis=0, return_index=True)
            sub_xyz = xyz0[uniq]
            order = np.lexsort((sub_xyz[:, 1], sub_xyz[:, 0]))   # rough spatial locality
            sub_sorted = sub_xyz[order]
            sub_itn = itn0[uniq][order]
            sub_ret = ret0[uniq][order]
            sub_hag = hag0[uniq][order]
            sub_pred = np.full(len(sub_xyz), -1, np.int64)
            N = cfg.num_points
            with torch.no_grad():
                for s in range(0, len(sub_sorted), N):
                    real = min(N, len(sub_sorted) - s)
                    if real < 64:
                        continue
                    block = sub_sorted[s:s + N]
                    f2 = np.stack([sub_itn[s:s + N], sub_ret[s:s + N], sub_hag[s:s + N]]
                                  + ([dg.local_density_logdk(block, DG_LOGDK_K)] if DG_LOGDK_FEAT else []),
                                  axis=1)   # D3b
                    orig = order[s:s + real].astype(np.int64)   # indices into sub_xyz
                    if real < N:                         # pad the final short block
                        pad = np.random.choice(real, N - real)
                        block = np.concatenate([block, block[pad]], axis=0)
                        f2 = np.concatenate([f2, f2[pad]], axis=0)
                        orig = np.concatenate([orig, np.full(N - real, -1, np.int64)])
                    # RandLA-Net's first-N subsampling requires randomized point
                    # order (see evaluate()); shuffle, then unshuffle on scatter.
                    perm = np.random.permutation(N)
                    block, f2, orig = block[perm], f2[perm], orig[perm]
                    pc0 = (block - block.mean(0, keepdims=True)).astype(np.float32)
                    # D5 density-TTA: isotropic scale s rescales the LocSE relative-coord
                    # magnitudes (a density view); average softmax over views. views=[1.0]
                    # when off -> identical to the old single-view argmax.
                    views = [1.0] + (list(np.linspace(0.85, 1.2, DG_INFER_TTA))
                                     if DG_INFER_TTA else [])
                    prob = None
                    for sv in views:
                        pc_c = (pc0 * sv).astype(np.float32)
                        item = (pc_c, f2.astype(np.float32), np.zeros(N, np.int64),
                                np.arange(N, dtype=np.int32), np.array([0], np.int32))
                        batch = collate_fn([item])
                        for k in ("features", "labels", "input_inds", "cloud_inds"):
                            batch[k] = batch[k].to(device)
                        for k in ("xyz", "neigh_idx", "sub_idx", "interp_idx"):
                            batch[k] = [t.to(device) for t in batch[k]]
                        lg = net(batch)["logits"].transpose(1, 2).reshape(-1, num_classes)
                        pp = torch.softmax(lg.float(), -1).cpu().numpy()
                        prob = pp if prob is None else prob + pp
                    p = prob.argmax(-1)
                    valid = orig >= 0
                    sub_pred[orig[valid]] = p[valid]
            valid = sub_pred >= 0
            nn = cKDTree(sub_xyz[valid]).query(xyz0)[1]
            pred = sub_pred[valid][nn]
            # itn0 is the p95-normalized intensity (the feature the net saw).
            return xyz0, np.clip(pred, 0, num_classes - 1), itn0
        return _predict_scene

    # ==========================================================================
    # INFERENCE-ONLY MODE
    # ==========================================================================
    if mode == "infer":
        if not weights or not infer_input:
            raise ValueError("--mode infer requires --weights and --infer-input")
        wpath = f"/outputs/{weights}"
        if not os.path.exists(wpath):
            raise FileNotFoundError(f"weights not found on outputs volume: {wpath}")
        try:   # weights_only=True: a hand-picked .pth can't run code on load
            ckpt = torch.load(wpath, map_location=device, weights_only=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load weights '{wpath}': {e}\n"
                f"  (loaded safely with weights_only=True — a full-model pickle or a "
                f"checkpoint from another script is rejected; re-export as a state_dict.)"
            ) from e
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        fc3_key = next((k for k in sd if k.startswith("fc3.") and k.endswith("weight")), None)
        num_classes = int(sd[fc3_key].shape[0]) if fc3_key else NUM_CLASSES
        class_names = [f"class_{i}" for i in range(num_classes)]
        import os as _os, sys as _sys   # read the run's run.json (single manifest) beside the weights
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "helper"))
        from train_common import infer_meta
        meta = infer_meta(wpath)
        if meta:
            class_names = meta.get("class_names") or class_names
            if meta.get("grid") is not None:
                SUB_GRID_SIZE = float(meta["grid"])

        fc0_key = next((k for k in sd if k.startswith("fc0.") and sd[k].dim() >= 2), None)
        ckpt_in_dim = int(sd[fc0_key].shape[1]) if fc0_key is not None else 3
        net = build_net(num_classes, in_dim=ckpt_in_dim)
        if ckpt_in_dim != IN_DIM:
            raise ValueError(
                f"checkpoint fc0 expects {ckpt_in_dim} input channels but this "
                f"script feeds {IN_DIM} ([xyz, intensity, return_number]) — "
                f"use weights trained by this script version")
        net.load_state_dict(sd)
        net.eval()
        print(f"  [infer] loaded {weights} ({num_classes} classes: {class_names}; "
              f"final_model = best-val epoch {ckpt.get('epoch', '?')})", flush=True)

        scenes = sorted(glob.glob(f"{DATASETS_ROOT}/_infer/{infer_input}/scenes/*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scenes under {DATASETS_ROOT}/_infer/{infer_input}/scenes")

        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_infer")
        # Predictions live next to the input scenes on the shared terminal-datasets
        # volume (not the per-backbone outputs volume), so inference output lands in
        # one consistent place no matter which model produced it.
        run_dir = f"{DATASETS_ROOT}/_infer/{infer_input}"
        pred_dir = f"{run_dir}/predictions"
        os.makedirs(pred_dir, exist_ok=True)
        with open(f"{run_dir}/run_config.json", "w") as f:
            json.dump({"backbone": "RandLA-Net", "mode": "infer", "weights": weights,
                       "infer_input": infer_input, "num_classes": num_classes,
                       "class_names": class_names, "sub_grid_size": SUB_GRID_SIZE,
                       "gpu": gpu_name(),
                       "scenes": [os.path.basename(s) for s in scenes]}, f, indent=2)

        predict_scene = make_predict_scene(net, num_classes)
        palette = _palette(num_classes)
        print(f"  [infer] labeling {len(scenes)} scene(s) -> {pred_dir}", flush=True)
        if DG_INFER_ADABN:
            # D2b: re-estimate BN running stats on the target tiles (label-free). f2 carries
            # the z-min HAG proxy in its 3rd column (IN_DIM=6) — BN stats don't need exact HAG.
            print("  [infer] AdaBN: recomputing BN stats on target tiles...", flush=True)

            def _target_batches(cap=30):
                seen = 0
                N = cfg.num_points
                for pc_path in scenes:
                    if seen >= cap:
                        return
                    z = np.load(pc_path)
                    xyz0 = z["xyz"].astype(np.float32)
                    itn0 = (z["intensity"].astype(np.float32) if "intensity" in z
                            else np.full(len(xyz0), 0.5, np.float32))
                    ret0 = (z["return_number"].astype(np.float32) if "return_number" in z
                            else (z["ret_num"].astype(np.float32) if "ret_num" in z
                                  else np.zeros(len(xyz0), np.float32)))
                    keys = np.floor(xyz0 / SUB_GRID_SIZE).astype(np.int64)
                    _, uniq = np.unique(keys, axis=0, return_index=True)
                    sx, si, sr = xyz0[uniq], itn0[uniq], ret0[uniq]
                    shag = (sx[:, 2] - sx[:, 2].min()).astype(np.float32)
                    for s0 in range(0, len(sx), N):
                        if seen >= cap:
                            return
                        real = min(N, len(sx) - s0)
                        if real < 64:
                            continue
                        block = sx[s0:s0 + N]
                        f2 = np.stack([si[s0:s0 + N], sr[s0:s0 + N], shag[s0:s0 + N]]
                                      + ([dg.local_density_logdk(block, DG_LOGDK_K)] if DG_LOGDK_FEAT else []),
                                      axis=1)   # D3b
                        if real < N:
                            pad = np.random.choice(real, N - real)
                            block = np.concatenate([block, block[pad]], 0)
                            f2 = np.concatenate([f2, f2[pad]], 0)
                        perm = np.random.permutation(N)
                        block, f2 = block[perm], f2[perm]
                        pc_c = (block - block.mean(0, keepdims=True)).astype(np.float32)
                        item = (pc_c, f2.astype(np.float32), np.zeros(N, np.int64),
                                np.arange(N, dtype=np.int32), np.array([0], np.int32))
                        batch = collate_fn([item])
                        for k in ("features", "labels", "input_inds", "cloud_inds"):
                            batch[k] = batch[k].to(device)
                        for k in ("xyz", "neigh_idx", "sub_idx", "interp_idx"):
                            batch[k] = [t.to(device) for t in batch[k]]
                        seen += 1
                        yield batch
            dg.adabn_recalibrate(net, _target_batches(), forward=lambda m, b: m(b))
            net.eval()
        for pc_path in scenes:
            name = os.path.splitext(os.path.basename(pc_path))[0]
            t0 = time.time()
            xyz, pred, inten = predict_scene(pc_path)
            _write_ply(f"{pred_dir}/{name}_pred.ply", xyz, pred, palette, inten)
            print(f"  [infer] {name}: {len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
        datasets_volume.commit()
        print(f"  [infer] done — predictions in _infer/{infer_input}/predictions", flush=True)
        return

    # ==========================================================================
    # TRAINING MODE
    # ==========================================================================
    print("=" * 70)
    print(f"  RandLA-Net  {dataset or 'IEEE Track 4'}  "
          f"({gpu_name()}, {N_EPOCHS} ep, batch {BATCH_SIZE})")
    print("=" * 70)
    train_list, val_list, test_list = ensure_prep()
    tag = dataset or "ieee"
    run_id = datetime.utcnow().strftime(f"%Y%m%d_%H%M%S_{tag}_randlanet_cold_hag")
    run_dir = f"/outputs/runs/{run_id}"
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    with open(f"{run_dir}/run_config.json", "w") as f:
        json.dump({
            "backbone": "RandLA-Net", "warm_start": False,
            "dataset": dataset or "IEEE GRSS 2019 DFC Track 4",
            "mode": mode, "gpu": gpu_name(),
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE, "num_points": NUM_POINTS,
            "sub_grid_size": SUB_GRID_SIZE, "in_dim": IN_DIM,
            "features": ["x", "y", "z", "intensity", "return_number", "HAG"],
            "hag_source": HAG_SOURCE,
            "steps_per_epoch": STEPS,
            "class_balance": {"weighting": CLASS_WEIGHTING, "beta": WEIGHT_BETA,
                              "weight_scheme": "inv_sqrt_freq" if WEIGHT_BETA == 0.5
                              else f"inv_freq^{WEIGHT_BETA}",
                              "cap": WEIGHT_CAP, "rare_freq_thresh": RARE_FREQ_THRESH,
                              "rare_center_prob": RARE_CENTER_PROB},
            "loss": {"pointwise": "focal" if USE_FOCAL else "weighted_ce",
                     "focal_gamma": FOCAL_GAMMA if USE_FOCAL else None,
                     "ce_weighted": CLASS_WEIGHTING,
                     "lovasz_softmax_weight": LOVASZ_WEIGHT},
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "label_map_asprs_to_index": None if dataset else LABEL_MAP,
            "train_scenes": [n for n, _, _ in train_list],
            "val_scenes":   [n for n, _, _ in val_list],
            "test_scenes":  [n for n, _, _ in test_list],
            "holdout_seed": HOLDOUT_SEED,
        }, f, indent=2)

    train_files = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    val_files   = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
    test_files  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))

    train_ds = Scenes("train", files=train_files)
    val_ds   = Scenes("val",   files=val_files,   label="val")
    test_ds  = Scenes("test",  files=test_files,  label="test")

    # --- class-balanced loss + rare-centered sphere sampling ----------------
    # Scan the (already in-RAM) train scenes: inverse-frequency class weights,
    # and per-scene rare-point indices so training spheres can be centered on
    # rare classes. Rare = train frequency < RARE_FREQ_THRESH.
    print("  scanning train scenes for class balance…", flush=True)
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, _, _, _, lab in train_ds.scenes:
        v = lab[lab >= 0]
        if v.size:
            class_counts += np.bincount(v, minlength=NUM_CLASSES)
    freq = class_counts / max(int(class_counts.sum()), 1)
    rare_classes = [c for c in range(NUM_CLASSES) if 0 < freq[c] < RARE_FREQ_THRESH]
    if RARE_OVERSAMPLE and rare_classes:
        train_ds.set_rare_classes(rare_classes)
    print(f"  class counts: {dict(zip(CLASS_NAMES, class_counts.tolist()))}", flush=True)
    print(f"  rare classes: {[CLASS_NAMES[c] for c in rare_classes]}", flush=True)
    if CLASS_WEIGHTING:
        # Inverse-frequency raised to WEIGHT_BETA. beta=0.5 -> w = 1/sqrt(freq),
        # the inverse-sqrt-frequency scheme (sub-linear: boosts rare classes
        # without the instability of raw 1/freq). Mean-normalized, then capped.
        w = (1.0 / np.maximum(freq, 1e-6)) ** WEIGHT_BETA
        w = w / w.mean()
        w = np.clip(w, 1.0 / WEIGHT_CAP, WEIGHT_CAP)
        # Rebind the weight tensor + loss used by compute_loss / focal_loss
        # (closures over _class_w and _ce in the enclosing scope).
        _class_w = torch.tensor(w, dtype=torch.float32).to(device)
        _ce = nn.CrossEntropyLoss(weight=_class_w, ignore_index=-1)
        print(f"  class weights: "
              f"{dict(zip(CLASS_NAMES, [round(float(x), 3) for x in w]))}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)

    net = build_net(NUM_CLASSES)
    print(f"  params: {sum(p.numel() for p in net.parameters()):,}")

    # Cold start: random initialization (no pretrained checkpoint).

    opt = optim.Adam(net.parameters(), lr=cfg.learning_rate)
    sched = optim.lr_scheduler.ExponentialLR(opt, 0.95)

    metrics_csv = f"{run_dir}/metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss", "train_acc", "val_acc",
            "train_iou", "val_iou", "sec_per_iter", "sec_per_epoch",
            "gpu_mem_mb",
        ])

    def _to_device(batch):
        for k in ("features", "labels", "input_inds", "cloud_inds"):
            batch[k] = batch[k].to(device)
        for k in ("xyz", "neigh_idx", "sub_idx", "interp_idx"):
            batch[k] = [t.to(device) for t in batch[k]]
        return batch

    # --- Periodic + final evaluation: the REAL full-coverage eval (not a cheap
    # subset proxy) over the held-out val set every VAL_EVERY epochs, appended to
    # val_metrics.csv so the val curve is the SAME protocol the final test reports
    # (full-coverage blocks, reprojected to raw points, scored on raw GT). Heavier
    # than the old quick_val — raise VAL_EVERY if it costs too much. ------------
    def evaluate(ds, name2src, label):
        """Full-coverage eval scored on the ORIGINAL raw points (official
        protocol). Each scene's 0.30 m subsampled points are predicted once via
        spatially-sorted NUM_POINTS blocks; those predictions are then
        propagated to the raw cloud by nearest neighbour and scored against the
        raw GT, instead of scoring the subsampled points (which is neither the
        benchmark protocol nor comparable across backbones)."""
        t_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_union = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_gt    = np.zeros(NUM_CLASSES, dtype=np.int64)
        correct = total = 0; t_test = time.time()
        n_scenes = n_skipped = 0
        N = cfg.num_points
        with torch.no_grad():
            for i, (xyz, intensity, ret_num, hag, lab) in enumerate(ds.scenes):
                order = np.lexsort((xyz[:, 1], xyz[:, 0]))   # rough spatial locality
                pred = np.full(len(xyz), -1, np.int64)
                pend_items, pend_blocks = [], []

                def flush():
                    nonlocal pend_items, pend_blocks
                    if not pend_items:
                        return
                    batch = _to_device(collate_fn(pend_items))
                    end_points = net(batch)
                    p = end_points["logits"].transpose(1, 2).argmax(-1).cpu().numpy()
                    for bi, orig in enumerate(pend_blocks):
                        valid = orig >= 0           # drop padded positions
                        pred[orig[valid]] = p[bi, valid]
                    pend_items, pend_blocks = [], []

                for s in range(0, len(order), N):
                    blk = order[s:s + N]
                    real = len(blk)
                    if real < 64:
                        continue
                    pts_blk = xyz[blk]
                    f2 = np.stack([intensity[blk], ret_num[blk], hag[blk]]
                                  + ([dg.local_density_logdk(pts_blk, DG_LOGDK_K)] if DG_LOGDK_FEAT else []),
                                  axis=1).astype(np.float32)   # D3b
                    orig = blk.astype(np.int64)
                    if real < N:                         # pad the final short block
                        pad = np.random.choice(real, N - real)
                        pts_blk = np.concatenate([pts_blk, pts_blk[pad]], axis=0)
                        f2 = np.concatenate([f2, f2[pad]], axis=0)
                        orig = np.concatenate([orig, np.full(N - real, -1, np.int64)])
                    # RandLA-Net subsamples by taking the FIRST points at each
                    # layer (tf_map), so the input MUST be shuffled — training
                    # spheres are (sample_sphere). Feeding the lexsort-ordered
                    # block collapses the multi-scale subsampling onto one corner
                    # and wrecks predictions. Shuffle, then track originals.
                    perm = np.random.permutation(N)
                    pts_blk, f2, orig = pts_blk[perm], f2[perm], orig[perm]
                    pc_c = (pts_blk - pts_blk.mean(0, keepdims=True)).astype(np.float32)
                    pend_items.append((pc_c, f2, np.zeros(N, np.int64),
                                       np.arange(N, dtype=np.int32),
                                       np.array([0], np.int32)))
                    pend_blocks.append(orig)
                    if len(pend_items) == VAL_BATCH:
                        flush()
                flush()
                # Reproject the subsampled-point predictions onto the raw cloud.
                name = os.path.splitext(os.path.basename(ds.files[i]))[0]
                src = name2src.get(name)
                got = pred >= 0
                if src is None or not got.any():
                    n_skipped += 1; continue
                try:
                    raw_xyz, _, _, raw_lab = load_scene(*src)
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped += 1; continue
                _, nn = cKDTree(xyz[got]).query(raw_xyz)
                raw_pred = pred[got][nn]
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
        absent  = [c for c in range(NUM_CLASSES) if gt_counts[c] == 0]
        present_iou = [float(iou_per[c]) for c in present]
        present_mIoU = float(np.mean(present_iou)) if present_iou else 0.0
        m = {
            "overall_acc": correct / max(total, 1),
            "overall_mIoU": float(np.mean(iou_per)),
            "present_classes_mIoU": present_mIoU,
            "per_class_iou": {CLASS_NAMES[c]: float(iou_per[c]) for c in range(NUM_CLASSES)},
            "per_class_gt_count": {CLASS_NAMES[c]: gt_counts[c] for c in range(NUM_CLASSES)},
            "present_classes": [CLASS_NAMES[c] for c in present],
            "absent_classes":  [CLASS_NAMES[c] for c in absent],
            "total_test_seconds": time.time() - t_test,
            "num_scenes": n_scenes,
            "num_raw_points_scored": int(total),
            "skipped_scenes": n_skipped,
            "scored_on": "raw_points",
            "full_coverage": True,
            "reprojection": "nearest_subsampled_point_to_raw",
        }
        print(f"  [{label}] acc={m['overall_acc']:.4f}  "
              f"mIoU({NUM_CLASSES}-way)={m['overall_mIoU']:.4f}  "
              f"mIoU(present {len(present)})={m['present_classes_mIoU']:.4f}  "
              f"absent={m['absent_classes']}  raw_pts={total:,}  skipped={n_skipped}", flush=True)
        return m

    val_src = {n: (p, c) for n, p, c in val_list}

    val_csv = f"{run_dir}/val_metrics.csv"
    with open(val_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "val_acc", "val_miou"] +
                               [f"iou_{n}" for n in CLASS_NAMES])

    import os as _os, sys as _sys   # scripts/helper is a sibling dir (flat /root in Modal)
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "helper"))
    from train_common import BestCheckpoint, write_run_manifest
    best = BestCheckpoint(run_dir)
    write_run_manifest(run_dir, "randlanet_hag", dataset)   # the single inference manifest (run.json)

    def run_eval(ep, write_json=False):
        net.eval()
        m = evaluate(val_ds, val_src, f"eval@ep{ep}")
        ious = [m["per_class_iou"][CLASS_NAMES[c]] for c in range(NUM_CLASSES)]
        with open(val_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{m['overall_acc']:.4f}",
                                    f"{m['present_classes_mIoU']:.4f}"] + [f"{x:.4f}" for x in ious])
        if best.update(m["present_classes_mIoU"]):
            torch.save({"model": net.state_dict(), "epoch": ep}, best.final)
        if write_json:
            m_test = evaluate(test_ds, {n: (p, c) for n, p, c in test_list}, "test")
            with open(f"{run_dir}/test_metrics.json", "w") as fj:
                json.dump({"val": m, "test": m_test,
                           "val_scenes": [n for n, _, _ in val_list],
                           "test_scenes": [n for n, _, _ in test_list]}, fj, indent=2)
        outputs_volume.commit()
        net.train()
        return m

    t_run = time.time()
    print(f"  starting {N_EPOCHS} epochs, {cfg.train_steps} steps/epoch", flush=True)
    LOG_EVERY = 20
    for ep in range(N_EPOCHS):
        net.train()
        iou_calc = IoUCalculator(cfg)
        ep_loss = 0.0; n_steps = 0; correct = total = 0
        t_ep = time.time()
        t_chunk = t_ep
        print(f"  ep {ep:3d} starting…", flush=True)
        for batch in train_loader:
            for k in ("features", "labels", "input_inds", "cloud_inds"):
                batch[k] = batch[k].to(device, non_blocking=True)
            for k in ("xyz", "neigh_idx", "sub_idx", "interp_idx"):
                batch[k] = [t.to(device, non_blocking=True) for t in batch[k]]
            end_points = net(batch)
            loss, end_points = compute_loss(end_points, NUM_CLASSES)
            # Skip non-finite batches so one bad gradient can't poison the
            # weights with NaN (which is unrecoverable), and clip gradients to
            # prevent the spike in the first place (RandLA at lr=1e-2 is prone
            # to it; the other backbones already clip).
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.item()); n_steps += 1
            acc, end_points = compute_acc(end_points)
            iou_calc.add_data(end_points)
            pred = end_points["logits"].transpose(1, 2).reshape(-1, NUM_CLASSES).argmax(-1)
            lbl  = end_points["labels"].reshape(-1)
            m = lbl >= 0
            correct += (pred[m] == lbl[m]).sum().item(); total += int(m.sum())
            if n_steps % LOG_EVERY == 0:
                dt = time.time() - t_chunk
                print(f"    ep {ep:3d} step {n_steps:4d}: "
                      f"loss={ep_loss/n_steps:.4f} acc={correct/max(total,1):.4f} "
                      f"{LOG_EVERY/dt:.2f} it/s", flush=True)
                t_chunk = time.time()
        mean_iou, _ = iou_calc.compute_iou()
        sched.step()
        sec_per_iter = (time.time() - t_ep) / max(n_steps, 1)
        sec_per_epoch = time.time() - t_ep
        train_acc = correct / max(total, 1)
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, f"{ep_loss/max(n_steps,1):.4f}", "", f"{train_acc:.4f}", "",
                f"{mean_iou:.4f}", "", f"{sec_per_iter:.4f}",
                f"{sec_per_epoch:.2f}", f"{gpu_mem:.1f}",
            ])
        print(f"  ep {ep:3d}: loss={ep_loss/max(n_steps,1):.4f} "
              f"acc={train_acc:.4f} miou={mean_iou:.4f} "
              f"s/iter={sec_per_iter:.3f} s/ep={sec_per_epoch:.1f}", flush=True)
        if (ep + 1) % 5 == 0:
            torch.save({"model": net.state_dict(), "optim": opt.state_dict(),
                        "epoch": ep},
                       f"{run_dir}/checkpoints/ep{ep:03d}.pth")
            outputs_volume.commit()
        if (ep + 1) % VAL_EVERY == 0 and ep != N_EPOCHS - 1:
            run_eval(ep)               # last epoch handled by the final eval below

    print("  final evaluation (val + test)…", flush=True)
    run_eval(N_EPOCHS - 1, write_json=True)
    best.finalize(lambda p: torch.save(
        {"model": net.state_dict(), "epoch": N_EPOCHS - 1}, p))
    print(f"  total wall-clock {(time.time() - t_run)/3600:.2f} h")
    outputs_volume.commit()


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Local randlanet_hag trainer/inferencer (no modal).')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--sub-grid', type=float, default=None)
    ap.add_argument('--num-points', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--steps-per-epoch', type=int, default=None)
    ap.add_argument('--mode', default='train')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--infer-input', default=None)
    args = ap.parse_args()
    train_randlanet(**vars(args))


if __name__ == "__main__":
    main()
