"""
Local training script for RandLA-Net (PyTorch) — COLD-START variant.

Trains on a canonical trainer_gui dataset (--dataset NAME) whose three
materialized train/val/test scene folders live under /datasets. Random
initialization (no pretrained weights).

Features are [xyz, intensity, return_number] (5 ch); fc0 is rebuilt 5->8.
FEAT_CHANNELS env overrides the spec (ordered csv incl. dataset feat_*
channels, e.g. feat_hag for real HeightAboveGround); run.json "features"
records the resolved list and inference rebuilds from it.
Per-scene p95-clipped intensity normalization, vertical-rotation / x-flip /
isotropic-scale train augmentation, class-weighted CE (+ optional focal /
Lovász) with rare-class-centered sphere sampling, a held-out val pass every
VAL_EVERY epochs (-> val_metrics.csv), and a final full-coverage eval that
scores every val/test scene on its raw points.

Flags:
  --dataset NAME    (required) canonical dataset under /datasets
  --sub-grid / --num-points / --epochs / --batch / --steps-per-epoch
  --mode infer --weights runs/<id>/final_model.pth --infer-input <job_id>
"""

from typing import Optional

# ============================================================================
# Configuration
# ============================================================================
N_EPOCHS      = 100              # default when --epochs is omitted; 250-300 for a full run
BATCH_SIZE    = 6
VAL_BATCH     = 12

NUM_POINTS    = 45056            # 4096*11, RandLA SemKITTI default
SUB_GRID_SIZE = 0.30             # 30 cm — sparse aerial LiDAR (~2 pts/m²) vs KITTI
# Input-feature spec (FEAT_CHANNELS env, GUI picker): comma-separated ordered
# names; "" = the legacy [x, y, z, intensity, return_number] recipe (IN_DIM 5).
# log d_k (DG_LOGDK_FEAT) appends after the spec, as always.
FEAT_CHANNELS = ""

# --- density domain-generalization (scripts/helper/density.py; see DENSITY_DG.md) ---
# o = rho*g^2; density-invariant for o>=1, breaks for o<1. RandLA's fixed-N absorbs a
# plain keep-fraction, so D1 jitters the SUB_GRID per sphere (the real density knob);
# D2b/D5 patch inference. All default to current behaviour.
DG_DENSITY_AUG = False   # D1: per-sphere coarser SUB_GRID during training
DG_COARSEN_MAX = 2.0     # = 1/(SUB_GRID_SIZE*sqrt(rho_min)); density sweep-down factor
DG_P_NATIVE    = 0.5     # P(sphere kept at native SUB_GRID_SIZE)
DG_INFER_ADABN = False   # D2b: recompute BN stats on target tiles before predicting (RandLA is pure-BN)
DG_INFER_TTA   = 0       # D5: # extra density(scale) views to average at inference (0=off)
EVAL_VOTES     = 2       # overlap-vote passes at eval/inference: pass v shifts the
                         # block grid by v/EVAL_VOTES of a block and softmax probs
                         # accumulate per point, so block-border points get a vote
                         # from a block that centers them (the same overlap voting
                         # the PTv3/KPConvX evals already do). 1 = old single-pass
                         # argmax at the old cost; each extra pass costs one more
                         # full forward over the scene.
# D3b: explicit local-density input channel (log k-th-NN distance) -> bumps IN_DIM 5->6
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

DATASETS_ROOT = "/datasets"   # bind-mounted datasets root (trainer_gui canonical datasets)

def train_randlanet(dataset: Optional[str] = None, sub_grid: Optional[float] = None,
                    num_points: Optional[int] = None, epochs: Optional[int] = None,
                    batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                    mode: str = "train", weights: Optional[str] = None,
                    infer_input: Optional[str] = None):
    if dataset is None and mode != "infer":
        raise ValueError("--dataset is required: pass a canonical trainer_gui "
                         "dataset name under /datasets. The only "
                         "dataset-free path is --mode infer.")
    import os, sys, time, json, csv, glob
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
     DG_INFER_ADABN, DG_INFER_TTA, EVAL_VOTES, USE_FOCAL, FOCAL_GAMMA,
     CLASS_WEIGHTING, WEIGHT_BETA, RARE_OVERSAMPLE,
     RARE_CENTER_PROB, VAL_EVERY, FEAT_CHANNELS) = tc.env_overrides(globals(), [
        "DG_DENSITY_AUG", "DG_COARSEN_MAX", "DG_P_NATIVE", "DG_LOGDK_FEAT",
        "DG_LOGDK_K", "DG_INFER_ADABN", "DG_INFER_TTA", "EVAL_VOTES",
        "USE_FOCAL", "FOCAL_GAMMA", "CLASS_WEIGHTING", "WEIGHT_BETA",
        "RARE_OVERSAMPLE", "RARE_CENTER_PROB", "VAL_EVERY", "FEAT_CHANNELS"])
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    # --- resolve config: CLI args override the module defaults ---------------
    SUB_GRID_SIZE = sub_grid if sub_grid is not None else globals()["SUB_GRID_SIZE"]
    # Input-feature spec: FEAT_CHANNELS env at train (the infer branch below
    # overrides it from run.json — env is ignored at infer). x/y/z come from
    # the recentered (augmented) coords, exactly the columns the old stack fed.
    FEAT_LEGACY = ["x", "y", "z", "intensity", "return_number"]
    FEAT_SPEC = (list(FEAT_LEGACY) if mode == "infer"   # env ignored at infer
                 else tc.parse_feat_spec(FEAT_CHANNELS, FEAT_LEGACY))
    bad = [n for n in FEAT_SPEC
           if n not in FEAT_LEGACY and not n.startswith("feat_")]
    if bad:
        raise ValueError(f"RandLA-Net can't feed {bad}; supported: "
                         f"{FEAT_LEGACY} plus dataset feat_* channels")
    # the non-coordinate spec channels — the columns cached feat2 stacks carry
    NONXYZ = [n for n in FEAT_SPEC if n not in ("x", "y", "z")]
    # Effective input dim: spec channels (all width 1) +1 log-d_k (D3b).
    # Shadowed here (before build_net is defined) so build_net's default, the
    # checkpoint in-dim check, and the run-config all track it.
    IN_DIM = len(FEAT_SPEC) + (1 if DG_LOGDK_FEAT else 0)
    NUM_POINTS    = num_points if num_points is not None else globals()["NUM_POINTS"]
    N_EPOCHS      = epochs if epochs is not None else globals()["N_EPOCHS"]
    BATCH_SIZE    = batch if batch is not None else globals()["BATCH_SIZE"]
    STEPS         = steps_per_epoch if steps_per_epoch is not None else 500
    if dataset:
        ds_root = f"{DATASETS_ROOT}/{dataset}"
        ds_meta, NUM_CLASSES, CLASS_NAMES = tc.load_dataset_meta(dataset)
        # A custom feature spec gets its own cache family (legacy spec = ""
        # tag, old caches valid).
        PREP_DIR = (f"{ds_root}/prep/randlanet"
                    f"_grid{int(round(SUB_GRID_SIZE * 100))}_p95"
                    f"{tc.feat_spec_tag(FEAT_SPEC, FEAT_LEGACY)}")
    else:
        # --mode infer: dataset-free. The real class count/names come from the
        # checkpoint (+ its run.json) in the inference branch; these are placeholders
        # so the inline `class Cfg` (num_classes = NUM_CLASSES) below has a value.
        ds_meta, NUM_CLASSES, CLASS_NAMES, PREP_DIR = {}, 0, [], None

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

    # The upstream network/loss_func.compute_loss hardcodes label==0 as ignored
    # (SemanticKITTI convention). Our labels are already remapped to 0..K-1
    # at preprocessing with the ignored value(s) -> -1. Pointwise term is focal
    # or weighted CE (ignore_index=-1), optionally combined with Lovász — the
    # loss primitives live in train_common (shared with the other backbones).
    _class_w = None
    _ce = nn.CrossEntropyLoss(ignore_index=-1)
    def compute_loss(end_points, num_classes):
        logits = end_points["logits"].transpose(1, 2).reshape(-1, num_classes)
        labels = end_points["labels"].reshape(-1)
        valid_mask = (labels >= 0) & (labels < num_classes)
        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask]
        if USE_FOCAL:
            loss = tc.focal_loss(valid_logits, valid_labels, FOCAL_GAMMA, _class_w)
        else:
            loss = _ce(valid_logits, valid_labels)
        if LOVASZ_WEIGHT > 0:
            probas = torch.softmax(valid_logits, dim=1)
            loss = loss + LOVASZ_WEIGHT * tc.lovasz_softmax_flat(probas, valid_labels)
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

    # --- canonical scene loader ---------------------------------------------
    def load_canonical(npz_path):
        """Canonical trainer_gui scene -> (xyz, intensity, return_number,
        label, extras); extras = every feat_* channel the scene carries
        (Phase 2a writes them, feat_hag included). Which channels the net
        actually eats is FEAT_SPEC's call; DG_LOGDK_FEAT appends more."""
        z = np.load(npz_path)
        # origin-offset (per-scene floor-min, kp_load_canonical's pattern)
        # before the float32 cast so projected (UTM) coords keep sub-meter
        # precision in the prep cache (float32 spacing at y~5e6 is 0.5m).
        xyz = (z["xyz"] - np.floor(z["xyz"].min(0))).astype(np.float32)
        intensity = z["intensity"].astype(np.float32) if "intensity" in z \
            else np.full(len(xyz), 0.5, np.float32)
        ret_num = z["return_number"].astype(np.float32) if "return_number" in z \
            else np.zeros(len(xyz), np.float32)
        lab = z["label"].astype(np.int32) if "label" in z \
            else np.full(len(xyz), -1, np.int32)
        extras = {k: z[k].astype(np.float32) for k in z.files
                  if k.startswith("feat_")}
        return xyz, intensity, ret_num, lab, extras

    def grid_subsample(xyz, intensity, ret_num, lab, extras, grid):
        keys = np.floor(xyz / grid).astype(np.int64)
        uniq = tc.voxel_unique(keys)
        return (xyz[uniq], intensity[uniq], ret_num[uniq], lab[uniq],
                {n: v[uniq] for n, v in extras.items()})

    def _cache_signature():
        # Everything that changes what a cached scene .npz contains. A mismatch
        # means the cache is stale/leaky and must not be silently reused.
        sig = {
            "format_version": 1,
            # v2: cached xyz is scene-local (load_canonical's origin shift) —
            # a global-frame cache must not be reused (0.5m y quantization).
            "coord_frame": "scene-local",
            "pipeline": "randlanet",
            "dataset": dataset,
            "sub_grid_size": SUB_GRID_SIZE,
            "num_classes": NUM_CLASSES,
            # spec-derived; reproduces the legacy string byte-for-byte for the
            # default spec, so every existing cache stays valid.
            "feature_recipe": ",".join(FEAT_SPEC).replace("x,y,z", "xyz"),
        }
        # The dataset carries the split decision in dataset_meta.json; fold its
        # seed/mode in so a re-split of the dataset invalidates the cache.
        sp = ds_meta.get("split", {}) if isinstance(ds_meta, dict) else {}
        sig["split_seed"] = sp.get("seed")
        sig["split_mode"] = sp.get("mode")
        return sig

    def ensure_prep():
        # Per-scene idempotency keyed on a .done marker (written only after a
        # clean save) so a run interrupted mid-scene is redone, not left partial.
        print(f"  ensuring preprocessed cache -> {PREP_DIR}", flush=True)
        for split in ("train", "val", "test"):
            os.makedirs(f"{PREP_DIR}/{split}", exist_ok=True)
        train_list, val_list, test_list = tc.split_scenes(ds_root)
        any_new = tc.validate_cache(
            PREP_DIR, _cache_signature(),
            [("train", train_list), ("val", val_list), ("test", test_list)],
            lambda d, name: (f"{d}/{name}.npz", f"{d}/{name}.npz.done"))
        for split, items in (("train", train_list), ("val", val_list),
                             ("test", test_list)):
            print(f"  [{split}] {len(items)} scenes", flush=True)
            for i, (name, pc_path, cls_path) in enumerate(items):
                out = f"{PREP_DIR}/{split}/{name}.npz"
                if os.path.exists(out + ".done"):
                    continue
                t0 = time.time()
                try:
                    xyz, intensity, ret_num, lab, extras = load_canonical(pc_path)
                    n_in = len(xyz)
                    # Every feat_* the scene carries rides through the
                    # subsample into the cache tiles (feat_hag included).
                    xyz, intensity, ret_num, lab, extras = grid_subsample(
                        xyz, intensity, ret_num, lab, extras, SUB_GRID_SIZE)
                except Exception as e:
                    print(f"  skip {pc_path}: {e}", flush=True); continue
                tile = dict(xyz=xyz.astype(np.float32),
                            intensity=intensity.astype(np.float32),
                            ret_num=ret_num.astype(np.float32),
                            lab=lab.astype(np.int32))
                for n, v in extras.items():
                    tile[n] = v.astype(np.float32)
                np.savez_compressed(out, **tile)
                open(out + ".done", "w").close()      # mark complete after a clean write
                any_new = True
                print(f"    [{i+1}/{len(items)}] {name}: {n_in:,} -> "
                      f"{len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
        if any_new:
            print("  preprocessing cache updated.", flush=True)
        else:
            print("  all scenes already cached.", flush=True)
        return train_list, val_list, test_list

    # --- batch assembly (standalone so inference can reuse it) ---------------
    def tf_map(batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
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
                + [batch_label, batch_pc_idx, batch_cloud_idx])
        return flat

    def collate_fn(batch):
        # Items are (pc, feat2, label, point_idx, cloud_idx); feat2 carries the
        # non-coordinate spec channels then log d_k. The network input is
        # assembled here in FEAT_SPEC order — x/y/z pulled from the (augmented,
        # centered) coords, exactly the columns the old [pc, feat2] concat fed.
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
        ax = {"x": 0, "y": 1, "z": 2}
        cols = [pcs[:, :, ax[n2]] if n2 in ax else feats[:, :, NONXYZ.index(n2)]
                for n2 in FEAT_SPEC]
        tail = feats[:, :, len(NONXYZ):]           # log d_k, appended as always
        full_feat = np.concatenate([np.stack(cols, axis=2), tail],
                                   axis=2)          # (B, N, IN_DIM)
        d["features"] = torch.from_numpy(full_feat).float().transpose(1, 2)
        d["labels"]   = torch.from_numpy(flat[4*n]).long()
        d["input_inds"] = torch.from_numpy(flat[4*n+1]).long()
        d["cloud_inds"] = torch.from_numpy(flat[4*n+2]).long()
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
            # the feat_* channels the spec needs — a miss is a clear error here
            extras = tc.feat_extras(z, FEAT_SPEC, os.path.basename(f))
            return (z["xyz"], z["intensity"], z["ret_num"], extras, z["lab"])

        def set_rare_classes(self, rare_classes):
            self.rare_idx = [np.where(np.isin(lab, rare_classes))[0]
                             for *_, lab in self.scenes]
            self.rare_scenes = [i for i, r in enumerate(self.rare_idx) if len(r)]

        def sample_sphere(self, cloud_idx, center_idx, augment=False, rng=np.random):
            xyz, intensity, ret_num, extras, lab = self.scenes[cloud_idx]
            center = xyz[center_idx:center_idx + 1]
            d2 = np.sum((xyz - center) ** 2, axis=1)
            sel = np.argpartition(d2, min(cfg.num_points, len(xyz) - 1))[:cfg.num_points]
            if len(sel) < cfg.num_points:
                sel = np.concatenate([sel, rng.choice(len(xyz), cfg.num_points - len(sel))])
            rng.shuffle(sel)
            # D1 density jitter: re-subsample the sphere to a coarser grid so the model
            # trains across the inference density range. Fixed-N is preserved by padding
            # from the (now sparser) point set, so the kNN graph sees sparser geometry.
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
            src = {"intensity": intensity, "return_number": ret_num, **extras}
            cols = ([src[n][sel] for n in NONXYZ]
                    + ([dg.local_density_logdk(pc, DG_LOGDK_K)] if DG_LOGDK_FEAT else []))
            feat2 = (np.stack(cols, axis=1) if cols
                     else np.zeros((len(sel), 0))).astype(np.float32)   # D3b: + log d_k on the (augmented) coords
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

    def make_predict_scene(net, num_classes, exclude_idx=None):
        SAVE_PROBS = os.environ.get("TT_SAVE_PROBS") == "1"

        def _predict_scene(pc_path):
            # RandLA works on fixed NUM_POINTS samples: grid-subsample the scene,
            # spatially sort it for locality, predict it in NUM_POINTS blocks
            # (reusing the collate), then NN-propagate to all original points.
            z = np.load(pc_path)
            # predict in the scene-local frame (kp_load_canonical's origin
            # shift pattern; global-UTM float32 quantizes y to 0.5m), return
            # the ORIGINAL georeferenced coords as the deliverable.
            raw0 = z["xyz"]
            xyz0 = (raw0 - np.floor(raw0.min(0))).astype(np.float32)
            itn0, ret0 = tc.scene_arrays(z, len(xyz0))
            ex0 = tc.feat_extras(z, FEAT_SPEC, os.path.basename(pc_path))
            keys = np.floor(xyz0 / SUB_GRID_SIZE).astype(np.int64)
            uniq = tc.voxel_unique(keys)
            sub_xyz = xyz0[uniq]
            order = np.lexsort((sub_xyz[:, 1], sub_xyz[:, 0]))   # rough spatial locality
            sub_itn0 = itn0[uniq]
            sub_ret0 = ret0[uniq]
            sub_ex0 = {n: v[uniq] for n, v in ex0.items()}
            # EVAL_VOTES soft-vote passes over offset block grids (same overlap
            # voting as the eval protocol; votes is n_sub x C float32).
            sub_votes = np.zeros((len(sub_xyz), num_classes), np.float32)
            N = cfg.num_points
            n_passes = max(int(EVAL_VOTES), 1)
            with torch.no_grad():
              for vp in range(n_passes):
                ov = np.roll(order, -(N * vp) // n_passes) if vp else order
                sub_sorted = sub_xyz[ov]
                sub_src = {"intensity": sub_itn0[ov], "return_number": sub_ret0[ov],
                           **{n2: v[ov] for n2, v in sub_ex0.items()}}
                for s in range(0, len(sub_sorted), N):
                    real = min(N, len(sub_sorted) - s)
                    if real < 64:
                        continue
                    block = sub_sorted[s:s + N]
                    cols = ([sub_src[n2][s:s + N] for n2 in NONXYZ]
                            + ([dg.local_density_logdk(block, DG_LOGDK_K)] if DG_LOGDK_FEAT else []))
                    f2 = (np.stack(cols, axis=1) if cols
                          else np.zeros((len(block), 0), np.float32))   # D3b
                    orig = ov[s:s + real].astype(np.int64)      # indices into sub_xyz
                    if real < N:                         # pad the final short block
                        pad = np.random.choice(real, N - real)
                        block = np.concatenate([block, block[pad]], axis=0)
                        f2 = np.concatenate([f2, f2[pad]], axis=0)
                        orig = np.concatenate([orig, np.full(N - real, -1, np.int64)])
                    # RandLA-Net's first-N subsampling requires randomized point
                    # order (training spheres shuffle); shuffle, unshuffle on scatter.
                    perm = np.random.permutation(N)
                    block, f2, orig = block[perm], f2[perm], orig[perm]
                    # float64 mean: raw inference scenes carry global-UTM
                    # coords, where a float32 mean mis-centers by ~500m
                    pc0 = (block - block.mean(0, keepdims=True, dtype=np.float64)
                           ).astype(np.float32)
                    # D5 density-TTA: isotropic scale s rescales the LocSE relative-coord
                    # magnitudes (a density view); average softmax over views. views=[1.0]
                    # when off -> identical to the old single-view behavior.
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
                    valid = orig >= 0
                    sub_votes[orig[valid]] += prob[valid]
            valid = sub_votes.sum(1) > 0
            if not valid.any():
                # tiny scene: every block was skipped (<64 pts), zero votes —
                # degrade like the KP/PTv3 paths instead of crashing the job.
                # Lowest NON-excluded class, confidence 0.
                fb = min(set(range(num_classes)) - set(exclude_idx or ()))
                return (raw0, np.full(len(xyz0), fb, np.int64), itn0,
                        np.zeros(len(xyz0), np.float32),
                        np.zeros((len(xyz0), num_classes), np.float16)
                        if SAVE_PROBS else None)
            nn = cKDTree(sub_xyz[valid]).query(xyz0)[1]
            vv = sub_votes[valid]                    # copy — safe to normalize in place
            vv /= vv.sum(1, keepdims=True)           # vote sums exceed 1 -> distribution
            vv = tc.apply_class_mask(vv, exclude_idx)
            pred = vv.argmax(1)[nn]
            conf = vv.max(1)[nn]
            probs = vv[nn].astype(np.float16) if SAVE_PROBS else None
            # itn0 is the p95-normalized intensity (the feature the net saw).
            return raw0, np.clip(pred, 0, num_classes - 1), itn0, conf, probs
        return _predict_scene

    # ==========================================================================
    # INFERENCE-ONLY MODE
    # ==========================================================================
    if mode == "infer":
        if not weights or not infer_input:
            raise ValueError("--mode infer requires --weights and --infer-input")
        wpath = f"/outputs/{weights}"
        if not os.path.exists(wpath):
            raise FileNotFoundError(f"weights not found under /outputs: {wpath}")
        ckpt = tc.load_ckpt_safe(wpath, map_location=device)
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        fc3_key = next((k for k in sd if k.startswith("fc3.") and k.endswith("weight")), None)
        num_classes = int(sd[fc3_key].shape[0]) if fc3_key else NUM_CLASSES
        class_names = [f"class_{i}" for i in range(num_classes)]
        # read the run's run.json (single manifest) beside the weights
        meta = tc.infer_meta(wpath)
        if meta:
            class_names = meta.get("class_names") or class_names
            if meta.get("grid") is not None:
                SUB_GRID_SIZE = float(meta["grid"])
        # rebuild the EXACT assembly recorded with the weights (env is ignored
        # at infer). Manifests without "features" = legacy runs.
        mf = (meta or {}).get("features") or []
        try:
            FEAT_SPEC = (tc.parse_feat_spec(",".join(mf), FEAT_LEGACY)
                         if mf else list(FEAT_LEGACY))
        except ValueError:
            FEAT_SPEC = list(FEAT_LEGACY)
        NONXYZ = [n for n in FEAT_SPEC if n not in ("x", "y", "z")]
        IN_DIM = len(FEAT_SPEC) + (1 if DG_LOGDK_FEAT else 0)

        fc0_key = next((k for k in sd if k.startswith("fc0.") and sd[k].dim() >= 2), None)
        ckpt_in_dim = int(sd[fc0_key].shape[1]) if fc0_key is not None else 3
        net = build_net(num_classes, in_dim=ckpt_in_dim)
        if ckpt_in_dim != IN_DIM:
            raise ValueError(
                f"checkpoint fc0 expects {ckpt_in_dim} input channels but this "
                f"script feeds {IN_DIM} ({FEAT_SPEC}"
                f"{' + logdk' if DG_LOGDK_FEAT else ''}) "
                f"— use weights trained with the same feature recipe")
        net.load_state_dict(sd)
        net.eval()
        print(f"  [infer] loaded {weights} ({num_classes} classes: {class_names}; "
              f"final_model = best-val epoch {ckpt.get('epoch', '?')})", flush=True)

        scenes = sorted(glob.glob(f"{DATASETS_ROOT}/_infer/{infer_input}/scenes/*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scenes under {DATASETS_ROOT}/_infer/{infer_input}/scenes")

        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_infer")
        # Predictions live next to the input scenes under the shared /datasets tree
        # (not the per-model runs dir), so inference output lands in
        # one consistent place no matter which model produced it.
        run_dir = f"{DATASETS_ROOT}/_infer/{infer_input}"
        pred_dir = f"{run_dir}/predictions"
        os.makedirs(pred_dir, exist_ok=True)
        exc_idx = tc.exclude_class_idx(class_names)
        infer_cfg = {"backbone": "RandLA-Net", "mode": "infer", "weights": weights,
                     "infer_input": infer_input, "num_classes": num_classes,
                     "class_names": class_names, "sub_grid_size": SUB_GRID_SIZE,
                     "gpu": tc.gpu_name(),
                     "exclude_classes": [class_names[i] for i in exc_idx],
                     "started_utc": datetime.utcnow().isoformat() + "Z"}

        predict_scene = make_predict_scene(net, num_classes, exclude_idx=exc_idx)
        if DG_INFER_ADABN:
            # D2b: re-estimate BN running stats on the target tiles (label-free) so the
            # source-density stats stop mis-normalizing at a different inference density.
            print("  [infer] AdaBN: recomputing BN stats on target tiles...", flush=True)

            def _target_batches(cap=30):
                seen = 0
                N = cfg.num_points
                for pc_path in scenes:
                    if seen >= cap:
                        return
                    z = np.load(pc_path)
                    xyz0 = z["xyz"].astype(np.float32)
                    itn0, ret0 = tc.scene_arrays(z, len(xyz0))
                    ex0 = tc.feat_extras(z, FEAT_SPEC, os.path.basename(pc_path))
                    keys = np.floor(xyz0 / SUB_GRID_SIZE).astype(np.int64)
                    uniq = tc.voxel_unique(keys)
                    sx = xyz0[uniq]
                    # BN stats must see the same feature the net will be fed at predict.
                    s_src = {"intensity": itn0[uniq], "return_number": ret0[uniq],
                             **{n: v[uniq] for n, v in ex0.items()}}
                    for s0 in range(0, len(sx), N):
                        if seen >= cap:
                            return
                        real = min(N, len(sx) - s0)
                        if real < 64:
                            continue
                        block = sx[s0:s0 + N]
                        cols = ([s_src[n][s0:s0 + N] for n in NONXYZ]
                                + ([dg.local_density_logdk(block, DG_LOGDK_K)] if DG_LOGDK_FEAT else []))
                        f2 = (np.stack(cols, axis=1) if cols
                              else np.zeros((len(block), 0), np.float32))   # D3b
                        if real < N:
                            pad = np.random.choice(real, N - real)
                            block = np.concatenate([block, block[pad]], 0)
                            f2 = np.concatenate([f2, f2[pad]], 0)
                        perm = np.random.permutation(N)
                        block, f2 = block[perm], f2[perm]
                        pc_c = (block - block.mean(0, keepdims=True,
                                                   dtype=np.float64)).astype(np.float32)
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
        tc.run_infer_scenes(scenes, predict_scene, pred_dir, run_dir, infer_cfg)
        return

    # ==========================================================================
    # TRAINING MODE
    # ==========================================================================
    print("=" * 70)
    print(f"  RandLA-Net  {dataset}  "
          f"({tc.gpu_name()}, {N_EPOCHS} ep, batch {BATCH_SIZE})")
    print("=" * 70)
    # Clear a stale STOP from an old run BEFORE the slow prep (conversion): a
    # stop clicked during startup must survive to the loop.
    tc.clear_stop()
    train_list, val_list, test_list = ensure_prep()
    tag = dataset
    run_id = datetime.utcnow().strftime(f"%Y%m%d_%H%M%S_{tag}_randlanet_cold")
    run_dir = f"/outputs/runs/{run_id}"
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    with open(f"{run_dir}/run.json", "w") as f:
        json.dump({
            "backbone": "RandLA-Net", "warm_start": False,
            "dataset": dataset,
            "mode": mode, "gpu": tc.gpu_name(),
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE, "num_points": NUM_POINTS,
            "sub_grid_size": SUB_GRID_SIZE, "in_dim": IN_DIM,
            # resolved input spec (log-dk rides outside it, driven by
            # DG_LOGDK_FEAT) — inference rebuilds this exact assembly from here.
            "features": FEAT_SPEC,
            "steps_per_epoch": STEPS,
            "eval_votes": EVAL_VOTES,
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
            "train_scenes": [n for n, _, _ in train_list],
            "val_scenes":   [n for n, _, _ in val_list],
            "test_scenes":  [n for n, _, _ in test_list],
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
    for *_, lab in train_ds.scenes:
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
        # Inverse-sqrt-frequency weights (shared scheme). Rebind the weight
        # tensor + loss used by compute_loss (closures over _class_w and _ce).
        w = tc.class_weights_np(class_counts, WEIGHT_BETA, WEIGHT_CAP)
        _class_w = torch.tensor(w, dtype=torch.float32).to(device)
        _ce = nn.CrossEntropyLoss(weight=_class_w, ignore_index=-1)
        print(f"  class weights: "
              f"{dict(zip(CLASS_NAMES, [round(float(x), 3) for x in w]))}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)

    net = build_net(NUM_CLASSES)
    print(f"  params: {sum(p.numel() for p in net.parameters()):,}")

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

    # ------------------------------------------------------------------------
    # Periodic held-out validation (eval mode, no_grad — weights untouched).
    # Same seeded spheres every pass, so rows are comparable across epochs.
    # Watch mid-run: tail runs/<id>/val_metrics.csv under the outputs dir.
    # ------------------------------------------------------------------------
    # --- Periodic + final evaluation: the REAL full-coverage eval (not a cheap
    # subset proxy) on the held-out val set every VAL_EVERY epochs, appended to
    # val_metrics.csv so the val curve uses the SAME protocol the final test does.
    # NOTE: far heavier than the old quick_val — every val scene is fully covered
    # and reprojected to raw points. Raise VAL_EVERY if it costs too much. -------
    val_csv = f"{run_dir}/val_metrics.csv"
    with open(val_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "val_acc", "val_miou"] +
                               [f"iou_{n}" for n in CLASS_NAMES])

    def evaluate(ds, name2src, label):
        """Full-coverage eval scored on the ORIGINAL raw points (official
        protocol). Each scene's 0.30 m subsampled points are predicted via
        spatially-sorted NUM_POINTS blocks, over EVAL_VOTES offset passes whose
        per-point softmax probs are summed (overlap voting — the protocol the
        PTv3/KPConvX evals use); the voted predictions are then propagated to
        the raw cloud by nearest neighbour and scored against the raw GT,
        instead of scoring the subsampled points (which is neither the
        benchmark protocol nor comparable across backbones)."""
        t_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_union = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_gt    = np.zeros(NUM_CLASSES, dtype=np.int64)
        correct = total = 0; t_test = time.time()
        n_scenes = n_skipped = 0
        N = cfg.num_points
        with torch.no_grad():
            for i, (xyz, intensity, ret_num, extras, lab) in enumerate(ds.scenes):
                src = {"intensity": intensity, "return_number": ret_num, **extras}
                order = np.lexsort((xyz[:, 1], xyz[:, 0]))   # rough spatial locality
                # EVAL_VOTES soft-vote passes over offset block grids (votes is
                # n_sub x C float32 — a few MB per class per million points).
                votes = np.zeros((len(xyz), NUM_CLASSES), np.float32)
                pend_items, pend_blocks = [], []

                def flush():
                    nonlocal pend_items, pend_blocks
                    if not pend_items:
                        return
                    batch = _to_device(collate_fn(pend_items))
                    end_points = net(batch)
                    p = torch.softmax(end_points["logits"].transpose(1, 2).float(),
                                      -1).cpu().numpy()
                    for bi, orig in enumerate(pend_blocks):
                        valid = orig >= 0           # drop padded positions
                        votes[orig[valid]] += p[bi, valid]
                    pend_items, pend_blocks = [], []

                n_passes = max(int(EVAL_VOTES), 1)
                for vp in range(n_passes):
                    ov = np.roll(order, -(N * vp) // n_passes) if vp else order
                    for s in range(0, len(ov), N):
                        blk = ov[s:s + N]
                        real = len(blk)
                        if real < 64:
                            continue
                        pts_blk = xyz[blk]
                        cols = ([src[n][blk] for n in NONXYZ]
                                + ([dg.local_density_logdk(pts_blk, DG_LOGDK_K)] if DG_LOGDK_FEAT else []))
                        f2 = (np.stack(cols, axis=1) if cols
                              else np.zeros((len(blk), 0))).astype(np.float32)   # D3b
                        orig = blk.astype(np.int64)
                        if real < N:                         # pad the final short block
                            pad = np.random.choice(real, N - real)
                            pts_blk = np.concatenate([pts_blk, pts_blk[pad]], axis=0)
                            f2 = np.concatenate([f2, f2[pad]], axis=0)
                            orig = np.concatenate([orig, np.full(N - real, -1, np.int64)])
                        # RandLA-Net subsamples by taking the FIRST points at each
                        # layer (tf_map), so the input MUST be shuffled — training
                        # spheres are (sample_sphere). The lexsort-ordered block
                        # collapses the multi-scale subsampling onto one corner and
                        # wrecks predictions. Shuffle, then track originals.
                        perm = np.random.permutation(N)
                        pts_blk, f2, orig = pts_blk[perm], f2[perm], orig[perm]
                        pc_c = (pts_blk - pts_blk.mean(0, keepdims=True,
                                                       dtype=np.float64)).astype(np.float32)
                        pend_items.append((pc_c, f2, np.zeros(N, np.int64),
                                           np.arange(N, dtype=np.int32),
                                           np.array([0], np.int32)))
                        pend_blocks.append(orig)
                        if len(pend_items) == VAL_BATCH:
                            flush()
                    flush()
                pred = votes.argmax(1)
                # Reproject the subsampled-point predictions onto the raw cloud.
                name = os.path.splitext(os.path.basename(ds.files[i]))[0]
                raw_src = name2src.get(name)
                got = votes.sum(1) > 0
                if raw_src is None or not got.any():
                    n_skipped += 1; continue
                try:
                    raw_xyz, _, _, raw_lab, _ = load_canonical(raw_src[0])
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped += 1; continue
                _, nn = cKDTree(xyz[got]).query(raw_xyz)
                raw_pred = pred[got][nn]
                v = raw_lab >= 0
                rp, rl = raw_pred[v], raw_lab[v]
                correct += int((rp == rl).sum()); total += int(v.sum())
                i_, u_, g_ = tc.score_ious(rp, rl, NUM_CLASSES)
                t_inter += i_; t_union += u_; t_gt += g_
                n_scenes += 1
        return tc.eval_metrics(
            t_inter, t_union, t_gt, correct, total, CLASS_NAMES, t_test,
            n_scenes, label,
            extra={"skipped_scenes": n_skipped,
                   "scored_on": "raw_points",
                   "full_coverage": True,
                   "reprojection": "nearest_subsampled_point_to_raw"})

    val_src = {n: (p, c) for n, p, c in val_list}

    best = tc.BestCheckpoint(run_dir)
    tc.write_run_manifest(run_dir, "randlanet", dataset)

    def run_eval(ep, write_json=False):
        # VAL always scores the current (most-recent) weights. The final
        # (write_json) call scores TEST on the BEST-TRACKED checkpoint instead,
        # so test_metrics.json reports the model actually kept as
        # final_model.pth, not whatever epoch training happened to end on.
        #
        # PreciseBN (Yan et al., "Rethinking 'Batch' in BatchNorm"): eval-mode BN
        # runs on EMA stats that lag the fast-moving weights ("moment staleness"),
        # which whipsaws the val curve while train stays smooth. Re-estimate the
        # stats with the CURRENT frozen weights over train_loader batches (the
        # exact distribution BN tracked, cumulative average) before scoring, so
        # best.update selects on and saves the precise stats.
        def _bn_batches(n=32):
            it = iter(train_loader)
            for _ in range(n):
                try:
                    batch = next(it)
                except StopIteration:
                    return
                for k in ("features", "labels", "input_inds", "cloud_inds"):
                    batch[k] = batch[k].to(device, non_blocking=True)
                for k in ("xyz", "neigh_idx", "sub_idx", "interp_idx"):
                    batch[k] = [t.to(device, non_blocking=True) for t in batch[k]]
                yield batch
        dg.adabn_recalibrate(net, _bn_batches(), forward=lambda mdl, b: mdl(b))
        net.eval()
        m = evaluate(val_ds, val_src, f"eval@ep{ep}")
        tc.append_val_row(val_csv, ep, m, CLASS_NAMES)
        if best.update(m["present_classes_mIoU"]):
            torch.save({"model": net.state_dict(), "epoch": ep}, best.final)
        if write_json:
            swapped = os.path.exists(best.final)
            if swapped:
                live_state = {k: v.clone() for k, v in net.state_dict().items()}
                net.load_state_dict(torch.load(best.final, map_location=device,
                                               weights_only=True)["model"])
                net.eval()
            m_test = evaluate(test_ds, {n: (p, c) for n, p, c in test_list}, "test")
            if swapped:
                net.load_state_dict(live_state)
            with open(f"{run_dir}/test_metrics.json", "w") as fj:
                json.dump({"val": m, "test": m_test,
                           "val_scenes": [n for n, _, _ in val_list],
                           "test_scenes": [n for n, _, _ in test_list]}, fj, indent=2)
        net.train()
        return m

    t_run = time.time()
    # Opt-in bf16 autocast (TT_AMP=1), matching the other trainers. Batch
    # prefetch is already covered here by the DataLoader's num_workers.
    AMP = os.environ.get("TT_AMP") == "1"
    print(f"  starting {N_EPOCHS} epochs, {cfg.train_steps} steps/epoch"
          f"{' [bf16 autocast]' if AMP else ''}", flush=True)
    LOG_EVERY = 20
    ep = N_EPOCHS - 1     # final-eval label when the loop never runs
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
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
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
        stop = tc.stop_requested(ep)
        if (ep + 1) % VAL_EVERY == 0 and ep != N_EPOCHS - 1 and not stop:
            run_eval(ep)               # last epoch handled by the final eval below
        if stop:
            break                      # falls through to the final eval + finalize

    # --- Final evaluation: the real full-coverage eval over val + test, written
    # to test_metrics.json (the same val number run_eval logs periodically). ----
    print("  final evaluation (val + test)…", flush=True)
    run_eval(ep, write_json=True)
    best.finalize(lambda p: torch.save(
        {"model": net.state_dict(), "epoch": ep}, p))
    print(f"  total wall-clock {(time.time() - t_run)/3600:.2f} h")


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Local randlanet trainer/inferencer.')
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
