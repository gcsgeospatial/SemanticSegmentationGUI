"""
Local training script for PointTransformerV3 on a canonical trainer_gui dataset.

Trains on the 3-folder --dataset path: a dataset materialized by the trainer_gui
app under /datasets as train/val/test scene folders (.npz) plus
a dataset_meta.json (num_classes / class_names). NUM_CLASSES / CLASS_NAMES come
from that metadata; val = in-distribution selection holdout, test = final report.

Uses the standalone PTv3 model.py from testSem/PointTransformerV3 directly (no
full Pointcept install), on standard attention (enable_flash=False).

Recipe (PTv3's published outdoor-LiDAR suite — Wu et al., CVPR 2024): full data
augmentation, loss = weighted CE (+ optional focal / label smoothing) + Lovász-
Softmax, inverse-sqrt-frequency class weights + rare-class tile oversampling,
AdamW (lr 2e-3, wd 5e-3) + warmup/cosine, overlap-voting eval scored on raw
points, periodic held-out validation + resumable checkpoints.

The 3 color channels prefer lidar intensity (normalized grayscale x3) over RGB
when the dataset has both — intensity separates water/asphalt/shadow where RGB
is ambiguous. RGB is the fallback, mid-gray the last resort. Old RGB-trained
checkpoints keep getting RGB at inference (run.json "color_source", absent =
pre-change = rgb). FEAT_CHANNELS env overrides the input spec (ordered csv
incl. dataset feat_* channels); run.json "features" records the resolved list
and inference rebuilds from it.

Usage:
    python local_train_ptv3.py --dataset NAME [--grid G] [--chunk-xy C]
        [--epochs N] [--batch B] [--steps-per-epoch S]
    python local_train_ptv3.py --dataset NAME --mode infer
        --weights runs/<id>/final_model.pth --infer-input <job_id>
"""

import os
from typing import Optional

# ============================================================================
# Configuration
# ============================================================================
N_EPOCHS      = 100             # default when --epochs is omitted; PTv3 outdoor recipe trains ~50 epochs
BATCH_SIZE    = 4

GRID_SIZE     = 0.5            # voxel grid (m). PTv3's 0.05 m outdoor default is
                               # for dense near-sensor LiDAR; airborne ALS is
                               # ~2 pts/m² (~0.7 m spacing), so a 5 cm grid is a
                               # no-op downsample AND leaves PTv3's sparse
                               # positional-encoding conv with empty kernels (no
                               # point falls within a few voxels of another).
                               # 0.5 m ≈ the actual point spacing. Override --grid.
USE_FLASH_ATTN = False   # PTv3 runs on standard attention — flash serialized-attn
                         # patch-gather OOB'd mid-train (disabled after a real
                         # crash, commit 98cc344; not a missing-dep workaround)

# Input-feature spec (FEAT_CHANNELS env, GUI picker): comma-separated ordered
# names; "" = the legacy [x, y, z, <color>] layout where <color> is "rgb" or
# "intensity" per color_src. PTv3's color slot is 3-wide: a single-channel
# "intensity" entry is expanded to 3 via the tile's baked rgb array (the arch
# rule) — run.json records the true single name plus "color_source" as always.
# log d_k (DG_LOGDK_FEAT) appends after the spec.
FEAT_CHANNELS = ""

# ----------------------------------------------------------------------------
# Regularization / optimizer — PTv3's published outdoor-LiDAR recipe
# (Wu et al., CVPR 2024, supplementary Tab. 13).
# ----------------------------------------------------------------------------
DROP_PATH     = 0.3      # stochastic depth (PTv3 outdoor default)
BASE_LR       = 2e-3     # PTv3 outdoor base lr (also OneCycle peak here)
WEIGHT_DECAY  = 5e-3     # PTv3 outdoor AdamW wd (NOT 0.05)
WARMUP_PCT    = 0.04     # 4% of the run (PTv3's 2-epoch warmup at its 50-epoch recipe)
GRAD_CLIP     = 1.0

# Augmentation — PTv3 outdoor suite. A transformer on a handful of large
# scenes overfits badly without it — don't disable casually.
AUG_ENABLE       = True
AUG_ROT_Z        = 1.0          # z angle ~ U(-pi, pi) * AUG_ROT_Z (full yaw)
AUG_ROT_XY       = 1.0 / 64.0   # x,y tilt ~ U(-pi, pi) * this (gentle ±~2.8 deg)
AUG_SCALE_MIN    = 0.9
AUG_SCALE_MAX    = 1.1
AUG_FLIP_P       = 0.5          # per-axis (x, y) coordinate flip probability
AUG_JITTER_SIGMA = 0.005        # gaussian per-point noise (m)
AUG_JITTER_CLIP  = 0.02         # clip jitter to +/- this (m)
AUG_COLOR        = 0.8          # per-channel keep prob: each spec entry (rgb = one
                                # entry, its 3 columns drop together) independently
                                # zeroed with p 0.2 per training tile — no channel
                                # is exempt, so none becomes a hard dependency

# D3b: explicit local-density input channel (log k-th-NN distance) -> bumps in_channels
# 6->7 (retrain). Pair with DG_DENSITY_AUG so rho varies. FiLM form lives in the ptv3 lib.
DG_LOGDK_FEAT  = False
DG_LOGDK_K     = 8

# --- density domain-generalization (scripts/helper/density.py; see DENSITY_DG.md) ---
# o = rho*g^2; density-invariant for o>=1, breaks for o<1. D1 jitters the voxel grid
# in training so the model spans the inference density range; D5 averages density
# (scale) views at inference. NOTE: D2b AdaBN is intentionally omitted for PTv3 — its
# BN is only in pooling/stem (LayerNorm elsewhere), so BN-recalibration barely helps.
DG_DENSITY_AUG = False   # D1: jitter GRID_SIZE per tile during training
DG_COARSEN_MAX = 2.5     # = 1/(GRID_SIZE*sqrt(rho_min)); density sweep-down factor
DG_P_NATIVE    = 0.5     # P(tile kept at native GRID_SIZE)
DG_INFER_TTA   = 0       # D5: # extra density(scale) views to average at inference (0=off)

# Loss = weighted CE (+ optional focal / label smoothing) + Lovász-Softmax,
# mirroring the kpconvx_cold trainer. PTv3's own outdoor loss is CE + Lovász.
CLASS_WEIGHTING  = True
WEIGHT_BETA      = 0.5     # 0.5 = inverse-SQRT-frequency (sub-linear, stable)
WEIGHT_CAP       = 5.0     # clamp each weight to [1/CAP, CAP] after mean-norm
LABEL_SMOOTH     = 0.0     # PTv3 leans on Lovász, not smoothing (KPConvX used 0.2)
LOVASZ_WEIGHT    = 1.0     # total = <pointwise> + LOVASZ_WEIGHT * lovasz_softmax
USE_FOCAL        = False   # True -> alpha-balanced focal instead of weighted CE
FOCAL_GAMMA      = 2.0

# Rare-class tile oversampling. RARE_CLASSES=None auto-detects rare classes from
# the train-set frequency histogram (classes below RARE_FREQ_FRAC x median freq).
RARE_OVERSAMPLE  = True
RARE_CLASSES     = None
RARE_FREQ_FRAC   = 0.5
RARE_TILE_PROB   = 0.25    # P(draw the next train tile from a rare-class tile)
RARE_CENTER_PROB = 0.25    # P(center the train crop on a rare-class point): the
                           # sample-level half of oversampling — a rare TILE is
                           # still mostly ground, so a uniform crop center would
                           # usually miss the rare points the tile was picked for
                           # (same idea as RandLA's rare-centered spheres)

# Periodic held-out validation + checkpoint/resume cadence.
VAL_EVERY        = 10      # held-out val pass every N epochs (no weight updates)
CHECKPOINT_GAP   = 3       # checkpoint (model + optimizer) frequency, epochs
RESUME           = False   # force-resume the latest matching run (see AUTO_RESUME)
# Auto-continue an unfinished run (no DONE marker) on relaunch/auto-retry, so an
# intermittent crash never loses the run — only epochs since last checkpoint.
# The cloud shells set AUTO_RESUME=1 (their retries=10 depends on it); local
# runs default off — a fresh local launch is a fresh run.
AUTO_RESUME      = os.environ.get("AUTO_RESUME", "0") == "1"

# Fixed mount inside cloud containers; the pixi local backend points
# TT_DATASETS_ROOT at the real host dir (see train_common path contract).
DATASETS_ROOT = os.environ.get("TT_DATASETS_ROOT", "/datasets")

def train_ptv3(dataset: Optional[str] = None, grid: Optional[float] = None,
               epochs: Optional[int] = None, batch: Optional[int] = None,
               steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
               mode: str = "train", weights: Optional[str] = None,
               infer_input: Optional[str] = None):
    if dataset is None and mode != "infer":
        raise ValueError("--dataset is required: pass a canonical trainer_gui dataset "
                         "name materialized under /datasets. The only "
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
     DG_INFER_TTA, USE_FOCAL, FOCAL_GAMMA, CLASS_WEIGHTING, WEIGHT_BETA,
     RARE_OVERSAMPLE, RARE_CENTER_PROB, VAL_EVERY,
     FEAT_CHANNELS) = tc.env_overrides(globals(), [
        "DG_DENSITY_AUG", "DG_COARSEN_MAX", "DG_P_NATIVE", "DG_LOGDK_FEAT",
        "DG_LOGDK_K", "DG_INFER_TTA", "USE_FOCAL", "FOCAL_GAMMA",
        "CLASS_WEIGHTING", "WEIGHT_BETA", "RARE_OVERSAMPLE", "RARE_CENTER_PROB",
        "VAL_EVERY", "FEAT_CHANNELS"])

    # so `import ptv3.model` resolves: PTV3_SRC (conda trainer-src-ptv3 package
    # activation) points AT the clone, so its parent goes on sys.path; the
    # container images bake the clone at /opt/ptv3.
    sys.path.insert(0, os.path.dirname(os.environ.get("PTV3_SRC", "/opt/ptv3")))

    # --- resolve config: CLI args override the module defaults ---------------
    GRID_SIZE   = grid if grid is not None else globals()["GRID_SIZE"]
    N_EPOCHS    = epochs if epochs is not None else globals()["N_EPOCHS"]
    BATCH_SIZE  = batch if batch is not None else globals()["BATCH_SIZE"]
    STEPS       = steps_per_epoch if steps_per_epoch is not None else 500
    CHUNK_XY    = chunk_xy if chunk_xy is not None else 50.0
    STRIDE      = CHUNK_XY / 2.0
    color_src = "intensity"   # what fills the 3 color channels; see module docstring
    FEAT_LEGACY = ["x", "y", "z", "intensity"]   # re-derived once color_src is known
    FEAT_SPEC = list(FEAT_LEGACY)                # --mode infer resolves from run.json

    # NUM_CLASSES / CLASS_NAMES come ONLY from the dataset's dataset_meta.json.
    # --mode infer is dataset-free: the inference branch below reads the class
    # count/names straight from the checkpoint (+ its run.json), so skip the meta.
    if dataset:
        ds_root = tc.dataset_dir(dataset)
        ds_meta, NUM_CLASSES, CLASS_NAMES = tc.load_dataset_meta(dataset)
        if not ds_meta.get("has_intensity"):
            color_src = "rgb" if ds_meta.get("has_rgb") else "gray"
        # Input-feature spec: FEAT_CHANNELS env; "" = the legacy layout for this
        # dataset's color_src. An explicit "rgb" spec forces real RGB into the
        # color slot even when the dataset also has intensity. Env is ignored
        # at infer (the inference branch resolves the spec from run.json).
        FEAT_LEGACY = ["x", "y", "z", "rgb" if color_src == "rgb" else "intensity"]
        FEAT_SPEC = (list(FEAT_LEGACY) if mode == "infer"
                     else tc.parse_feat_spec(FEAT_CHANNELS, FEAT_LEGACY))
        tc.ptv3_check_spec(FEAT_SPEC, "PTv3")
        if "rgb" in FEAT_SPEC:
            color_src = "rgb"
        elif "intensity" in FEAT_SPEC and ds_meta.get("has_intensity"):
            color_src = "intensity"
        # Keyed by color_src: tiles bake the color channels in, so RGB-era
        # caches ("ptv3_cold") are abandoned, not silently reused. A custom
        # feature spec gets its own family too (legacy spec = "" tag).
        # "_loc" = tiles in the scene-local frame (ptv3_load_canonical's
        # origin shift). Pre-shift caches stored global-UTM float32 tiles
        # (0.5m y quantization + the float32-mean centering bug) — a new dir
        # name abandons them instead of silently mixing frames.
        PREP_DIR = (f"{ds_root}/prep/ptv3_{color_src}"
                    f"{tc.feat_spec_tag(FEAT_SPEC, FEAT_LEGACY)}_chunk{int(CHUNK_XY)}_loc"
                    f"{tc.train_stride_tag()}")

    def _in_ch(spec):
        # spec widths: the color slot (rgb OR intensity) is 3 wide — PTv3
        # expands single-channel color to 3 (see build_feat) — the rest 1;
        # + log d_k (D3b) appended after the spec.
        return (sum(3 if n in ("rgb", "intensity") else 1 for n in spec)
                + (1 if DG_LOGDK_FEAT else 0))
    IN_CH = _in_ch(FEAT_SPEC)

    # --- Preprocessing ------------------------------------------------------
    def load_canonical(npz_path):
        return tc.ptv3_load_canonical(npz_path, color_src)

    # --- Model builder + prediction helpers ----------------------------------
    from ptv3.model import PointTransformerV3
    # PTv3's 5x5x5 stem needs the spconv-cu118 build — cu124's conv backward
    # device-asserts on it (the STEM_KERNEL=3 workaround lives in git history).

    def build_feat(cxyz, rgbf, extras=None, drop=()):
        """Spec-ordered PTv3 features: x/y/z from the (augmented, centered,
        voxel-deduped) coords, rgb/intensity = the tile's 3 baked color
        channels (single-channel intensity was expanded to 3 at prep — the
        arch rule), feat_* from `extras` (feat_hag included). log d_k appends
        after the spec. The legacy spec reproduces [cxyz, rgbf] exactly.
        `drop` = spec indices to zero (per-channel feature dropout, train
        time only); log d_k never drops."""
        cols = []
        for i, n in enumerate(FEAT_SPEC):
            if n in ("rgb", "intensity"):
                c = rgbf
            elif n in ("x", "y", "z"):
                c = cxyz[:, "xyz".index(n):"xyz".index(n) + 1]
            else:
                c = extras[n][:, None]
            cols.append(np.zeros_like(c, dtype=np.float32) if i in drop else c)
        if DG_LOGDK_FEAT:
            cols.append(dg.local_density_logdk(cxyz, DG_LOGDK_K)[:, None])
        return np.concatenate(cols, axis=1).astype(np.float32)

    def build_model(num_classes):
        backbone = PointTransformerV3(
            in_channels=IN_CH,   # spec channels (+ log d_k, D3b)
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
        return backbone, head

    from scipy.spatial import cKDTree

    def make_predict_scene(backbone, head, num_classes, exclude_idx=None):
        SAVE_PROBS = os.environ.get("TT_SAVE_PROBS") == "1"

        def _predict_scene(scene_path):
            # Tile into CHUNK_XY windows, voxel-downsample tracking the inverse
            # map, scatter per-voxel predictions back, NN-fill stragglers.
            xyz, rgb, _ = load_canonical(scene_path)   # scene-local frame
            z0 = np.load(scene_path)
            # the feat_* channels the spec needs (a miss is a clear error;
            # feat_hag is written by convert_infer_job like any other channel)
            ex0 = tc.feat_extras(z0, FEAT_SPEC, os.path.basename(scene_path))
            pred = np.full(len(xyz), -1, np.int64)
            conf = np.zeros(len(xyz), np.float32)
            probs = np.zeros((len(xyz), num_classes), np.float16) if SAVE_PROBS else None
            with torch.no_grad():
                # windows via one packed-code sort, not a full-cloud mask per window
                for idx in tc.xy_chunk_groups(xyz, CHUNK_XY, min_pts=64):
                    w0 = (xyz[idx] - xyz[idx].mean(0)).astype(np.float32)
                    rgbf = rgb[idx].astype(np.float32) / 255.0
                    exw = {n: v[idx] for n, v in ex0.items()}
                    # D5 density-TTA: isotropic scale s is a density change (o->o/s^2);
                    # re-voxelize per view, average per-point softmax. views=[1.0] when
                    # off -> identical to the old single-view argmax.
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
                        gc = keys[first] - keys[first].min(0)   # unique, dedup-consistent
                        grid_coord = torch.from_numpy(np.ascontiguousarray(gc)).long().cuda()
                        point = backbone({"coord": coord, "grid_coord": grid_coord,
                                          "feat": featt, "offset": offset})
                        fe = point["feat"] if isinstance(point, dict) else point.feat
                        vp = torch.softmax(head(fe).float(), -1).cpu().numpy()[inverse]
                        pprob = vp if pprob is None else pprob + vp
                    # view sums exceed 1 — renormalize to a distribution
                    pprob /= np.maximum(pprob.sum(-1, keepdims=True), 1e-12)
                    pprob = tc.apply_class_mask(pprob, exclude_idx)
                    pred[idx] = pprob.argmax(-1)
                    conf[idx] = pprob.max(-1)
                    if SAVE_PROBS:
                        probs[idx] = pprob.astype(np.float16)
            miss = pred < 0
            if miss.any() and (~miss).any():
                _, nn = cKDTree(xyz[~miss]).query(xyz[miss])
                pred[miss] = pred[~miss][nn]           # conf/probs stay 0: no votes
            elif miss.any():
                # nothing predicted (tiny scene) — lowest NON-excluded class so
                # EXCLUDE_CLASSES holds even here; conf stays 0.
                pred[:] = min(set(range(num_classes)) - set(exclude_idx or ()))
            # rgb carries the p95-normalized intensity grayscale the model saw.
            # Return the scene's ORIGINAL coords, not the origin-shifted ones
            # load_canonical computed on — the _pred.npz is the georeferenced
            # deliverable.
            return z0["xyz"], pred, rgb[:, 0] / 255.0, conf, probs
        return _predict_scene

    # ==========================================================================
    # INFERENCE-ONLY MODE
    # ==========================================================================
    if mode == "infer":
        if not weights or not infer_input:
            raise ValueError("--mode infer requires --weights and --infer-input")
        wpath = weights if os.path.isabs(weights) else f"{tc.OUTPUTS_ROOT}/{weights}"
        if not os.path.exists(wpath):
            raise FileNotFoundError(f"weights not found: {wpath}")
        ckpt = tc.load_ckpt_safe(wpath, map_location="cpu")
        bsd, hsd = ckpt["backbone"], ckpt["head"]
        num_classes = int(hsd["weight"].shape[0])
        class_names = [f"class_{i}" for i in range(num_classes)]
        # read the run's run.json (single manifest) beside the weights
        meta = tc.infer_meta(wpath)
        # Feed the checkpoint the color signal it was trained on. Manifests from
        # before intensity-first carry no color_source -> those runs saw RGB.
        color_src = (meta or {}).get("color_source") or "rgb"
        if meta:
            class_names = meta.get("class_names") or class_names
            if meta.get("grid") is not None:
                GRID_SIZE = float(meta["grid"])
        # rebuild the EXACT assembly recorded with the weights (env is ignored
        # at infer). Old manifests wrote no "features", or wrote the color slot
        # as 3 duplicate entries — those fall back to the legacy layout, which
        # is exactly what they trained on (color_src drives it).
        FEAT_LEGACY = ["x", "y", "z", "rgb" if color_src == "rgb" else "intensity"]
        mf = (meta or {}).get("features")
        try:
            FEAT_SPEC = (tc.parse_feat_spec(",".join(mf), FEAT_LEGACY)
                         if mf and len(set(mf)) == len(mf) else list(FEAT_LEGACY))
            tc.ptv3_check_spec(FEAT_SPEC, "PTv3")
        except ValueError:
            FEAT_SPEC = list(FEAT_LEGACY)
        IN_CH = _in_ch(FEAT_SPEC)

        backbone, head = build_model(num_classes)
        backbone.load_state_dict(bsd)
        head.load_state_dict(hsd)
        backbone.eval(); head.eval()
        print(f"  [infer] loaded {weights} ({num_classes} classes; "
              f"final_model = best-val epoch {ckpt.get('epoch', '?')})", flush=True)

        run_dir = tc.infer_dir(infer_input)
        scenes = sorted(glob.glob(f"{run_dir}/scenes/*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scenes under {run_dir}/scenes")

        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_infer")
        # Predictions live next to the input scenes under the shared datasets
        # tree (not the per-model runs dir), so inference output lands in
        # one consistent place no matter which model produced it.
        pred_dir = os.environ.get("TT_PRED_DIR") or f"{run_dir}/predictions"
        os.makedirs(pred_dir, exist_ok=True)
        exc_idx = tc.exclude_class_idx(class_names)
        infer_cfg = {"backbone": "PTv3", "mode": "infer", "weights": weights,
                     "infer_input": infer_input, "num_classes": num_classes,
                     "class_names": class_names, "grid_size": GRID_SIZE,
                     "color_source": color_src, "features": FEAT_SPEC,
                     "chunk_xy": CHUNK_XY, "gpu": tc.gpu_name(),
                     "exclude_classes": [class_names[i] for i in exc_idx],
                     "started_utc": datetime.utcnow().isoformat() + "Z"}

        predict_scene = make_predict_scene(backbone, head, num_classes,
                                           exclude_idx=exc_idx)
        tc.run_infer_scenes(scenes, predict_scene, pred_dir, run_dir, infer_cfg)
        return

    # ==========================================================================
    # TRAINING MODE
    # ==========================================================================
    print("=" * 70)
    print(f"  PTv3  {dataset}  ({tc.gpu_name()}, {N_EPOCHS} ep, batch {BATCH_SIZE})")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    # Clear a stale STOP from an old run BEFORE the slow prep (tiling): a stop
    # clicked during startup must survive to the loop.
    tc.clear_stop()
    tc.ptv3_ensure_prep(PREP_DIR, ds_root, CHUNK_XY, STRIDE, load_canonical)

    tag = dataset
    _pt = "ptv3"

    # Only the RESUME_RECIPE_KEYS subset of cfg: a candidate run.json that
    # disagrees is skipped, so a resume never republishes a manifest its
    # weights don't match. cfg itself is built below, in the fresh-run branch.
    _recipe = {"grid_size": GRID_SIZE, "chunk_xy": CHUNK_XY, "features": FEAT_SPEC,
               "n_epochs": N_EPOCHS, "num_classes": NUM_CLASSES,
               "class_names": CLASS_NAMES}
    resume_info = (tc.find_latest_unfinished_run(f"{tag}_{_pt}", _recipe)
                   if (RESUME or AUTO_RESUME) else None)
    if resume_info:
        run_dir, resume_ckpt, resume_epoch = resume_info
        run_id = os.path.basename(run_dir)
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        start_epoch = resume_epoch + 1
        print(f"  RESUMING {run_id} from {os.path.basename(resume_ckpt)} "
              f"-> epoch {start_epoch}/{N_EPOCHS}", flush=True)
    else:
        run_id = datetime.utcnow().strftime(f"%Y%m%d_%H%M%S_{tag}_{_pt}")
        run_dir = f"{tc.OUTPUTS_ROOT}/runs/{run_id}"
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        resume_ckpt, start_epoch = None, 0
        with open(f"{run_dir}/run.json", "w") as f:
            cfg = {
                "backbone": "PTv3", "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
                "dataset": dataset,
                "mode": mode, "gpu": tc.gpu_name(),
                "num_classes": NUM_CLASSES, "grid_size": GRID_SIZE,
                "class_names": CLASS_NAMES,
                "color_source": color_src,
                # resolved input spec: one rgb/intensity name = the 3-wide
                # color slot (PTv3 expands single-channel color to 3 — see
                # build_feat); log-dk rides outside it, driven by
                # DG_LOGDK_FEAT. Inference rebuilds this exact assembly.
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

    # --- Model --------------------------------------------------------------
    backbone, head = build_model(NUM_CLASSES)
    print(f"  Params: {sum(p.numel() for p in backbone.parameters()):,}")

    optim = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=BASE_LR, weight_decay=WEIGHT_DECAY,
    )
    if resume_ckpt is not None:
        rckpt = torch.load(resume_ckpt, map_location="cuda", weights_only=True)
        backbone.load_state_dict(rckpt["backbone"]); head.load_state_dict(rckpt["head"])
        if "optim" in rckpt:
            optim.load_state_dict(rckpt["optim"])
        print(f"  resumed weights{' + optimizer' if 'optim' in rckpt else ''}", flush=True)
    # loss / class weights are built after the train-tile class scan below.

    # --- Data ---------------------------------------------------------------
    def _scene_of(p):
        b = os.path.basename(p)
        return b.rsplit("_x", 1)[0]
    # CANONICAL: the dataset stage decided the 3-way split and materialized
    # train/val/test; read the three PREP folders verbatim and NEVER re-carve.
    # val = in-distribution selection holdout, test = final-report set.
    train_tiles = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    val_tiles   = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
    test_tiles  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
    hold = {_scene_of(p) for p in val_tiles}
    print(f"  train: {len(train_tiles)}   val(holdout {len(hold)} scenes): "
          f"{len(val_tiles)}   test: {len(test_tiles)}", flush=True)

    # --- class balance: one (parallel, PREP_DIR-cached) scan of the training
    # tiles -> inverse-frequency weights + rare-class tile flags. -------------
    class_counts, present_mask = tc.scan_class_balance(
        train_tiles, NUM_CLASSES, cache_path=f"{PREP_DIR}/class_balance_cache.npz")

    def _name(c):
        return CLASS_NAMES[c] if CLASS_NAMES else c
    names = [_name(c) for c in range(NUM_CLASSES)]
    print(f"  class counts: {dict(zip(names, class_counts.tolist()))}", flush=True)

    # Rare classes: explicit RARE_CLASSES, else auto — present classes whose
    # frequency is below RARE_FREQ_FRAC x the median present-class frequency.
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
        # Inverse-sqrt-frequency, absent classes pinned at 1.0 (never up-weight
        # a class with no train points), mean-normalized over present classes.
        w = tc.class_weights_np(class_counts, WEIGHT_BETA, WEIGHT_CAP,
                                absent_to_one=True)
        class_weights = torch.tensor(w, dtype=torch.float32).cuda()
        print(f"  class weights: "
              f"{dict(zip(names, [round(float(x), 3) for x in w]))}", flush=True)
    else:
        class_weights = None

    # Shared loss recipe: weighted (label-smoothed) CE or focal, + Lovász —
    # PTv3's outdoor loss is literally CE + Lovász (train_common.make_seg_loss).
    seg_loss = tc.make_seg_loss(class_weights, LABEL_SMOOTH, USE_FOCAL,
                                FOCAL_GAMMA, LOVASZ_WEIGHT)
    pick_train_tile = tc.make_tile_picker(train_tiles, rare_tiles,
                                          RARE_TILE_PROB)

    # --- PTv3 outdoor augmentation suite --------------------------------------
    def augment_xyz(xyz):
        return tc.ptv3_augment_xyz(xyz, AUG_ROT_Z, AUG_ROT_XY, AUG_SCALE_MIN,
                                   AUG_SCALE_MAX, AUG_FLIP_P,
                                   AUG_JITTER_SIGMA, AUG_JITTER_CLIP)

    def to_ptv3_batch(tiles_for_batch, training=True):
        # PTv3 takes a dict with: coord, grid_coord, feat, offset
        coords, feats, labels, offsets, grid_coords = [], [], [], [], []
        running = 0
        for tile in tiles_for_batch:
            z = np.load(tile)
            xyz, rgb, lab = z["xyz"], z["rgb"], z["lab"]
            ex = tc.feat_extras(z, FEAT_SPEC, os.path.basename(tile))
            # random crop ~30m for memory (train only — eval keeps full tiles).
            # With prob RARE_CENTER_PROB the crop centers on a rare-class point,
            # so the rare points a rare tile was drawn for actually land in-crop.
            if training and len(xyz) > 80000:
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
            # float64 mean: at global-UTM magnitudes a float32 mean is off by
            # hundreds of meters, which fails the |xy|<=CHUNK_XY cut below for
            # EVERY point and empties the whole batch (np.concatenate crash).
            xyz = (xyz - xyz.mean(0, keepdims=True, dtype=np.float64)
                   ).astype(np.float32)
            # Drop non-finite + far-outlier points: a single bad coordinate (NaN/
            # Inf or a stray faraway return) blows up the voxel grid and spconv's
            # indices, triggering a CUDA index device-assert that kills the run.
            ok = (np.isfinite(xyz).all(1)
                  & (np.abs(xyz[:, :2]).max(1) <= CHUNK_XY)
                  & (np.abs(xyz[:, 2]) <= 200.0))
            if int(ok.sum()) < 64:
                continue
            xyz = xyz[ok]; rgb = rgb[ok]; lab = lab[ok]
            ex = {n: v[ok] for n, v in ex.items()}
            # voxel-grid downsample to GRID_SIZE. Take grid_coord from the SAME
            # integer keys used to dedup, so it is unique per cloud and phase-
            # consistent with the kept points. Recomputing it as
            # floor((coord-min)/GRID) uses a different phase and can collapse two
            # distinct voxels onto one grid_coord — which corrupts spconv's
            # submanifold rulebook (out-of-bounds gather -> CUDA device assert).
            # D1 density jitter: coarsen the voxel grid per tile (train only) so the
            # model spans the inference density range. grid_coord still comes from the
            # dedup keys below, so the spconv submanifold invariant is preserved.
            g_eff = (dg.effective_grid(GRID_SIZE, DG_COARSEN_MAX, DG_P_NATIVE)
                     if (training and DG_DENSITY_AUG) else GRID_SIZE)
            keys = np.floor(xyz / g_eff).astype(np.int64)
            uniq = tc.voxel_unique(keys)
            xyz = xyz[uniq]; rgb = rgb[uniq]; lab = lab[uniq]
            ex = {n: v[uniq] for n, v in ex.items()}
            # feat = spec-ordered stack (augmented/centered coords, the baked
            # color channels, dataset feat_*) — see build_feat. Per-channel
            # feature dropout: train-time only, one independent coin per entry.
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
        gc -= gc.min(0, keepdims=True)                 # non-negative for spconv
        grid_coord = torch.from_numpy(np.ascontiguousarray(gc)).long().cuda()
        return {"coord": coord, "grid_coord": grid_coord,
                "feat": feat, "offset": offset}, label

    # --- Train loop ---------------------------------------------------------
    metrics_csv = f"{run_dir}/metrics.csv"
    if not os.path.exists(metrics_csv):       # keep prior rows when resuming
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss", "train_acc", "val_acc",
                "train_iou", "val_iou", "lr", "sec_per_iter", "sec_per_epoch",
                "gpu_mem_mb",
            ])

    # --- Periodic + final evaluation: the REAL voted eval (not a cheap proxy).
    # The periodic pass scores the held-out VAL scenes every VAL_EVERY epochs and
    # selects the best checkpoint on val present-class mIoU (NO test peeking); the
    # final pass also scores the TEST set, written separately to test_metrics.json.
    # NOTE: far heavier than the old quick_val — it forwards every overlapping
    # tile of every eval scene. Raise VAL_EVERY if it costs too much. -----------
    def _raw_loader(split, name):
        """Closure -> (xyz, rgb, lab) for the ORIGINAL raw scene, so the voted
        voxel predictions can be reprojected onto raw points + raw GT. `name` is a
        parameter, so each closure binds its own scene (no loop late-binding bug).
        split is 'val'|'test' -> the matching materialized dataset folder.
        Features aren't needed here — raw scoring only uses raw xyz + raw GT."""
        return lambda: load_canonical(f"{ds_root}/{split}/{name}.npz")

    val_items = [(n, _raw_loader("val", n), f"{PREP_DIR}/val") for n in sorted(hold)]
    test_items = [(n, _raw_loader("test", n), f"{PREP_DIR}/test")
                  for n in sorted({_scene_of(p) for p in test_tiles})]
    print(f"  eval set: {len(val_items)} holdout(val) + {len(test_items)} test scenes",
          flush=True)

    def _forward_logits(batch):
        point = backbone(batch)
        return head(point["feat"] if isinstance(point, dict) else point.feat)
    evaluate = tc.ptv3_make_evaluate(_forward_logits, build_feat, FEAT_SPEC,
                                     GRID_SIZE, CHUNK_XY, NUM_CLASSES, names)

    val_csv = f"{run_dir}/val_metrics.csv"
    if not os.path.exists(val_csv):
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "val_acc", "val_miou"] +
                [f"iou_{_name(c)}" for c in range(NUM_CLASSES)])

    best = tc.BestCheckpoint(run_dir)
    tc.write_run_manifest(run_dir, "ptv3", dataset)

    def run_eval(ep, write_json=False):
        # Periodic pass scores the held-out VAL scenes only (no test peeking) and
        # selects the best checkpoint on val present-class mIoU. The final pass
        # (write_json) scores TEST on the BEST-TRACKED checkpoint, not whatever
        # epoch training happened to end on, so test_metrics.json reports the
        # model actually kept as final_model.pth. VAL stays on the current
        # (most-recent) weights, matching the periodic curve.
        backbone.eval(); head.eval()
        m = evaluate(val_items, f"val@ep{ep}")
        tc.append_val_row(val_csv, ep, m, names)
        if best.update(m["present_classes_mIoU"]):
            torch.save({"backbone": backbone.state_dict(),
                        "head": head.state_dict(), "epoch": ep}, best.final)
        if write_json:
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
            with open(f"{run_dir}/test_metrics.json", "w") as fj:
                json.dump({"val": m, "test": mt,
                           "val_scenes": [n for n, _, _ in val_items],
                           "test_scenes": [n for n, _, _ in test_items]}, fj, indent=2)
        backbone.train(); head.train()
        return m

    LOG_EVERY = 20  # intra-epoch heartbeat
    # Same loop speedups as the KP trainers: opt-in bf16 autocast, background
    # batch building (tile load + batch assembly overlap the GPU), and a
    # GPU-accumulated confusion matrix (one bincount/step, one sync/epoch).
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
    ep = N_EPOCHS - 1     # final-eval label when the loop never runs
    for ep in range(start_epoch, N_EPOCHS):
        cur_lr = tc.ptv3_lr_at(ep, BASE_LR, WARMUP_PCT, N_EPOCHS)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        backbone.train(); head.train()
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
                    point = backbone(batch)
                    feat = point["feat"] if isinstance(point, dict) else point.feat
                    logits = head(feat)
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
            raise RuntimeError(f"epoch {ep}: 0 optimizer steps — {n_oom} OOM steps; "
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
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, ep_loss / max(n_steps, 1), "", f"{train_acc:.4f}", "",
                f"{train_iou:.4f}", "", f"{cur_lr:.6e}", f"{sec_per_iter:.4f}",
                f"{sec_per_epoch:.2f}", f"{gpu_mem:.1f}",
            ])
        print(f"  ep {ep:3d}: loss={ep_loss/max(n_steps,1):.4f} "
              f"acc={train_acc:.4f} miou={train_iou:.4f} lr={cur_lr:.2e} "
              f"s/iter={sec_per_iter:.3f} s/ep={sec_per_epoch:.1f}", flush=True)
        if (ep + 1) % CHECKPOINT_GAP == 0 or ep == N_EPOCHS - 1:
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(),
                        "optim": optim.state_dict(), "epoch": ep},
                       f"{run_dir}/checkpoints/ep{ep:03d}.pth")
            # keep only the 2 newest checkpoints (~0.5 GB each with optimizer)
            for old in sorted(glob.glob(f"{run_dir}/checkpoints/ep*.pth"))[:-2]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        stop = tc.stop_requested(ep)
        if (ep + 1) % VAL_EVERY == 0 and ep != N_EPOCHS - 1 and not stop:
            run_eval(ep)               # last epoch handled by the final eval below
        if stop:
            break   # falls through to the final eval + finalize (+ DONE marker)

    if prefetch:
        prefetch.shutdown()      # stop background batch builds during the eval

    # --- Final evaluation: the real voted eval over the combined eval set,
    # written to test_metrics.json (the same number run_eval logs periodically). -
    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(ep, write_json=True)
    best.finalize(lambda p: torch.save(
        {"backbone": backbone.state_dict(), "head": head.state_dict(),
         "epoch": ep}, p))
    print(f"  total wall-clock {(time.time() - t_run)/3600:.2f} h")

    # Mark the run complete so AUTO_RESUME won't re-resume it on the next launch
    # (a crashed/retried run has no DONE and is picked back up automatically).
    open(f"{run_dir}/DONE", "w").close()
    print(f"  run complete -> {run_id}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Local ptv3 trainer/inferencer.')
    ap.add_argument('--dataset', default=None)   # required to train; omitted for --mode infer
    ap.add_argument('--grid', type=float, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--steps-per-epoch', type=int, default=None)
    ap.add_argument('--chunk-xy', type=float, default=None)
    ap.add_argument('--mode', default='train')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--infer-input', default=None)
    args = ap.parse_args()
    train_ptv3(**vars(args))


if __name__ == "__main__":
    main()
