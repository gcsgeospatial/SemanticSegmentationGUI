"""
Local training script for PointTransformerV3 on a canonical trainer_gui dataset.

Trains on the 3-folder --dataset path: a dataset materialized by the trainer_gui
app onto the terminal-datasets volume as train/val/test scene folders (.npz) plus
a dataset_meta.json (num_classes / class_names). NUM_CLASSES / CLASS_NAMES come
from that metadata; val = in-distribution selection holdout, test = final report.

Uses the standalone PTv3 model.py from testSem/PointTransformerV3 directly (no
full Pointcept install). The fast path uses FlashAttention; if the GPU doesn't
support it the script falls back to PTv3 with enable_flash=False.

Recipe (PTv3's published outdoor-LiDAR suite — Wu et al., CVPR 2024): full data
augmentation, loss = weighted CE (+ optional focal / label smoothing) + Lovász-
Softmax, inverse-sqrt-frequency class weights + rare-class tile oversampling,
AdamW (lr 2e-3, wd 5e-3) + warmup/cosine, overlap-voting eval scored on raw
points, periodic held-out validation + resumable checkpoints.

Usage:
    python local_train_ptv3.py --dataset NAME [--grid G] [--chunk-xy C]
        [--epochs N] [--batch B] [--steps-per-epoch S]
    python local_train_ptv3.py --dataset NAME --mode infer
        --weights runs/<id>/final_model.pth --infer-input <job_id>

Timeout comes from the TT_TIMEOUT_HOURS env var.
"""

import os
from typing import Optional


def gpu_name() -> str:
    """Real CUDA device name for logs/metadata (replaces the old fixed cloud GPU_TYPE)."""
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


# ============================================================================
# Configuration
# ============================================================================
APP_NAME      = "ptv3"
N_EPOCHS      = 100             # smoke test; PTv3 outdoor recipe trains ~50 epochs
BATCH_SIZE    = 4
TIMEOUT_HOURS = int(os.environ.get("TT_TIMEOUT_HOURS", "24"))

GRID_SIZE     = 0.5            # voxel grid (m). PTv3's 0.05 m outdoor default is
                               # for dense near-sensor LiDAR; airborne ALS is
                               # ~2 pts/m² (~0.7 m spacing), so a 5 cm grid is a
                               # no-op downsample AND leaves PTv3's sparse
                               # positional-encoding conv with empty kernels (no
                               # point falls within a few voxels of another).
                               # 0.5 m ≈ the actual point spacing. Override --grid.
USE_FLASH_ATTN = False   # PTv3 runs on standard attention; no flash-attn dep

# ----------------------------------------------------------------------------
# Regularization / optimizer — PTv3's published outdoor-LiDAR recipe
# (Wu et al., CVPR 2024, supplementary Tab. 13). The original script used a
# generic AdamW + OneCycle with weight_decay 0.05; the paper uses 5e-3.
# ----------------------------------------------------------------------------
DROP_PATH     = 0.3      # stochastic depth (PTv3 outdoor default)
BASE_LR       = 2e-3     # PTv3 outdoor base lr (also OneCycle peak here)
WEIGHT_DECAY  = 5e-3     # PTv3 outdoor AdamW wd (NOT 0.05)
WARMUP_PCT    = 0.04     # ~2 epochs of 50 warmup (PTv3 uses a 2-epoch warmup)
GRAD_CLIP     = 1.0

# Augmentation — PTv3 outdoor suite. The original to_ptv3_batch did NONE; for a
# transformer on a handful of large scenes that is the dominant overfit cause.
AUG_ENABLE       = True
AUG_ROT_Z        = 1.0          # z angle ~ U(-pi, pi) * AUG_ROT_Z (full yaw)
AUG_ROT_XY       = 1.0 / 64.0   # x,y tilt ~ U(-pi, pi) * this (gentle ±~2.8 deg)
AUG_SCALE_MIN    = 0.9
AUG_SCALE_MAX    = 1.1
AUG_FLIP_P       = 0.5          # per-axis (x, y) coordinate flip probability
AUG_JITTER_SIGMA = 0.005        # gaussian per-point noise (m)
AUG_JITTER_CLIP  = 0.02         # clip jitter to +/- this (m)

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
# mirroring modal_train_kpconvx_cold.py. PTv3's own outdoor loss is CE + Lovász.
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

# Periodic held-out validation + checkpoint/resume cadence.
VAL_EVERY        = 10      # held-out val pass every N epochs (no weight updates)
CHECKPOINT_GAP   = 3       # checkpoint (model + optimizer) frequency, epochs
RESUME           = True   # force-resume the latest matching run (see AUTO_RESUME)
AUTO_RESUME      = False    # auto-continue an unfinished run (no DONE marker) on
                          # relaunch / Modal auto-retry, so an intermittent crash
                          # never loses the run — only epochs since last checkpoint
DEBUG_CUDA       = False   # ONE-OFF diagnostic: CUDA_LAUNCH_BLOCKING + dump the
                          # offending batch's stats + no retries (set True to debug).
STEM_KERNEL      = 5       # PTv3's faithful 5x5x5 stem. (Shrinking it to 3 was a
                          # failed workaround for the cu124 spconv conv-backward
                          # device-assert; that bug is fixed properly by moving to
                          # the spconv-cu118 build, so the original stem is restored.
                          # Set 3 only if a large-kernel spconv issue ever recurs.)

DATASETS_ROOT = "/datasets"   # terminal-datasets volume (trainer_gui canonical datasets)

# ============================================================================
# Image
# ============================================================================


# Local stand-ins for the modal Volumes the body commits to: the data is
# already on the bind-mounted /outputs /datasets, so there is nothing to
# upload. (The modal shell does the real commits when it runs this.)
class _NoVol:
    def commit(self, *a, **k): pass
    def reload(self, *a, **k): pass


outputs_volume, datasets_volume = _NoVol(), _NoVol()


def train_ptv3(dataset: Optional[str] = None, grid: Optional[float] = None,
               epochs: Optional[int] = None, batch: Optional[int] = None,
               steps_per_epoch: Optional[int] = None, chunk_xy: Optional[float] = None,
               mode: str = "train", weights: Optional[str] = None,
               infer_input: Optional[str] = None):
    if dataset is None and mode != "infer":
        raise ValueError("--dataset is required: pass a canonical trainer_gui dataset "
                         "name materialized on the terminal-datasets volume. The only "
                         "dataset-free path is --mode infer.")
    import os, sys, time, json, csv, glob, traceback
    from datetime import datetime
    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helper"))
    import density as dg
    # DG flags: env-overridable (GUI "Density generalization" panel / DG_*=1 in the shell).
    DG_DENSITY_AUG = dg.env_bool("DG_DENSITY_AUG", globals()["DG_DENSITY_AUG"])
    DG_COARSEN_MAX = dg.env_float("DG_COARSEN_MAX", globals()["DG_COARSEN_MAX"])
    DG_P_NATIVE    = dg.env_float("DG_P_NATIVE", globals()["DG_P_NATIVE"])
    DG_LOGDK_FEAT  = dg.env_bool("DG_LOGDK_FEAT", globals()["DG_LOGDK_FEAT"])
    DG_LOGDK_K     = dg.env_int("DG_LOGDK_K", globals()["DG_LOGDK_K"])
    DG_INFER_TTA   = dg.env_int("DG_INFER_TTA", globals()["DG_INFER_TTA"])
    # Loss / class-balance overrides (GUI "Loss & class balance" panel -> LOSS_*/
    # RARE_* env; mirrors the DG env pattern). Unset env -> the script constants.
    USE_FOCAL       = dg.env_bool("LOSS_FOCAL", globals()["USE_FOCAL"])
    FOCAL_GAMMA     = dg.env_float("LOSS_FOCAL_GAMMA", globals()["FOCAL_GAMMA"])
    CLASS_WEIGHTING = dg.env_bool("LOSS_CLASS_WEIGHTING", globals()["CLASS_WEIGHTING"])
    WEIGHT_BETA     = dg.env_float("LOSS_WEIGHT_BETA", globals()["WEIGHT_BETA"])
    RARE_OVERSAMPLE = dg.env_bool("RARE_OVERSAMPLE", globals()["RARE_OVERSAMPLE"])
    from plyfile import PlyData

    sys.path.insert(0, "/opt")          # so `import ptv3.model` resolves

    if globals().get("DEBUG_CUDA"):
        # Synchronous CUDA: the device-side assert then fires AT the real op, so
        # the Python traceback is correct (instead of surfacing async in spconv).
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        print("  [debug] CUDA_LAUNCH_BLOCKING=1 (synchronous; slower) — diagnostic run.",
              flush=True)

    # --- resolve config: CLI args override the module defaults ---------------
    GRID_SIZE   = grid if grid is not None else globals()["GRID_SIZE"]
    N_EPOCHS    = epochs if epochs is not None else globals()["N_EPOCHS"]
    BATCH_SIZE  = batch if batch is not None else globals()["BATCH_SIZE"]
    STEPS       = steps_per_epoch if steps_per_epoch is not None else 500
    CHUNK_XY    = chunk_xy if chunk_xy is not None else 50.0
    STRIDE      = CHUNK_XY / 2.0

    # NUM_CLASSES / CLASS_NAMES come ONLY from the dataset's dataset_meta.json.
    # --mode infer is dataset-free: the inference branch below reads the class
    # count/names straight from the checkpoint (+ its run.json), so skip the meta.
    if dataset:
        ds_root = f"{DATASETS_ROOT}/{dataset}"
        meta_path = f"{ds_root}/dataset_meta.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{meta_path} not found — upload the dataset "
                                    f"with the trainer_gui app first.")
        with open(meta_path) as f:
            ds_meta = json.load(f)
        NUM_CLASSES = int(ds_meta["num_classes"])
        CLASS_NAMES = list(ds_meta["class_names"])
        PREP_DIR = f"{ds_root}/prep/ptv3_cold_chunk{int(CHUNK_XY)}"

    # --- Preprocessing ------------------------------------------------------
    def load_ply(path):
        ply = PlyData.read(path)
        v = ply["vertex"]
        names = set(p.name for p in v.properties)
        xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
        if {"red", "green", "blue"}.issubset(names):
            rgb = np.stack([v["red"], v["green"], v["blue"]], -1).astype(np.float32)
        else:
            rgb = np.full((xyz.shape[0], 3), 128.0, dtype=np.float32)
        for key in ("scalar_label", "label", "class"):
            if key in names:
                lab = np.asarray(v[key]).astype(np.int64); break
        else:
            lab = np.zeros(xyz.shape[0], dtype=np.int64)
        return xyz, rgb, lab

    def load_canonical(npz_path):
        """Canonical trainer_gui scene -> the (xyz, rgb, lab) tuple load_ply produces.
        Missing rgb falls back to mid-gray (same as load_ply)."""
        z = np.load(npz_path)
        xyz = z["xyz"].astype(np.float32)
        if "rgb" in z:
            rgb = z["rgb"].astype(np.float32)
        elif "intensity" in z:
            rgb = np.repeat((z["intensity"].astype(np.float32) * 255.0)[:, None], 3, axis=1)
        else:
            rgb = np.full((len(xyz), 3), 128.0, dtype=np.float32)
        lab = z["label"].astype(np.int64) if "label" in z \
            else np.full(len(xyz), -1, np.int64)
        return xyz, rgb, lab

    def load_scene(path):
        # canonical .npz -> trainer_gui loader; else PLY.
        if path.endswith(".npz"):
            return load_canonical(path)
        return load_ply(path)

    def tile_and_save(src_paths, out_dir, chunk_xy, stride):
        os.makedirs(out_dir, exist_ok=True)
        for fi, src in enumerate(src_paths):
            scene = os.path.splitext(os.path.basename(src))[0]
            t0 = time.time()
            try:
                xyz, rgb, lab = load_scene(src)
            except Exception as e:
                print(f"  skip {src}: {e}", flush=True); continue
            print(f"    [{fi+1}/{len(src_paths)}] {scene}: {len(xyz):,} pts "
                  f"loaded in {time.time()-t0:.1f}s, tiling…", flush=True)
            mins, maxs = xyz[:, :2].min(0), xyz[:, :2].max(0)
            n_tiles = 0
            for x0 in np.arange(mins[0], maxs[0], stride):
                for y0 in np.arange(mins[1], maxs[1], stride):
                    m = ((xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + chunk_xy) &
                         (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + chunk_xy))
                    if m.sum() < 2048: continue
                    out_path = f"{out_dir}/{scene}_x{int(x0)}_y{int(y0)}.npz"
                    np.savez_compressed(out_path,
                        xyz=xyz[m].astype(np.float32),
                        rgb=rgb[m].astype(np.uint8),
                        lab=lab[m].astype(np.int32))
                    n_tiles += 1
            print(f"      -> {n_tiles} tiles", flush=True)

    def ensure_prep():
        # Per-scene idempotency via prefix-match on existing tiles.
        os.makedirs(f"{PREP_DIR}/train", exist_ok=True)
        os.makedirs(f"{PREP_DIR}/val",   exist_ok=True)
        os.makedirs(f"{PREP_DIR}/test",  exist_ok=True)
        print(f"  ensuring preprocessed cache -> {PREP_DIR}", flush=True)
        any_new = [False]
        def already_tiled(out_dir, scene):
            return bool(glob.glob(f"{out_dir}/{scene}_x*.npz"))
        def tile_remaining(src_paths, out_dir, chunk, stride):
            for src in src_paths:
                scene = os.path.splitext(os.path.basename(src))[0]
                if already_tiled(out_dir, scene): continue
                tile_and_save([src], out_dir, chunk, stride)
                any_new[0] = True
        # CANONICAL: the dataset stage already materialized the 3-way split;
        # tile each folder into its own PREP folder verbatim (no re-carving).
        train_paths = sorted(glob.glob(f"{ds_root}/train/*.npz"))
        val_paths   = sorted(glob.glob(f"{ds_root}/val/*.npz"))
        test_paths  = sorted(glob.glob(f"{ds_root}/test/*.npz"))
        if not train_paths:
            raise FileNotFoundError(f"No canonical scenes under {ds_root}/train")
        print(f"  [train] {len(train_paths)} canonical scenes", flush=True)
        tile_remaining(train_paths, f"{PREP_DIR}/train", CHUNK_XY, STRIDE)
        # stride = STRIDE (not CHUNK_XY) so val/test tiles overlap and the final
        # eval can vote per-voxel over the up-to-4 covering tiles.
        print(f"  [val] {len(val_paths)} canonical scenes", flush=True)
        tile_remaining(val_paths, f"{PREP_DIR}/val", CHUNK_XY, STRIDE)
        print(f"  [test] {len(test_paths)} canonical scenes", flush=True)
        tile_remaining(test_paths, f"{PREP_DIR}/test", CHUNK_XY, STRIDE)
        if any_new[0]:
            datasets_volume.commit()
            print("  preprocessing committed.", flush=True)
        else:
            print("  all scenes already cached.", flush=True)

    # --- Model builder + prediction helpers ----------------------------------
    from ptv3.model import PointTransformerV3

    # PTv3's stem is a 5x5x5 SubMConv (indice_key="stem", kernel_volume 125).
    # spconv's Native conv backward device-asserts on that large kernel on this
    # cu124 build (confirmed by the diagnostic: clean batch, OOB in
    # ConvGemmOps.indice_conv_backward), and the implicit-GEMM path can't
    # NVRTC-compile here. Every 3x3x3 conv (the xCPE) backward-passes fine, so
    # shrink ONLY the stem to STEM_KERNEL by patching the SubMConv3d ctor.
    if STEM_KERNEL and STEM_KERNEL != 5:
        import spconv.pytorch as _spc
        if not getattr(_spc.SubMConv3d, "_stem_patched", False):
            _orig_subm = _spc.SubMConv3d.__init__
            def _subm_init(self, *a, **kw):
                if kw.get("indice_key") == "stem":
                    kw["kernel_size"] = STEM_KERNEL
                _orig_subm(self, *a, **kw)
            _spc.SubMConv3d.__init__ = _subm_init
            _spc.SubMConv3d._stem_patched = True
            print(f"  [spconv] PTv3 stem SubMConv3d kernel 5 -> {STEM_KERNEL}", flush=True)

    def build_model(num_classes):
        # FlashAttention is optional — the image installs it best-effort
        # (`pip install flash-attn || echo ...`), so it may be absent. PTv3's
        # SerializedAttention asserts when enable_flash=True but flash_attn can't
        # be imported, so probe it and fall back to standard attention (identical
        # results, just slower) instead of crashing the whole run.
        use_flash = USE_FLASH_ATTN
        if use_flash:
            try:
                import flash_attn  # noqa: F401
                print("  flash_attn enabled.", flush=True)
            except Exception:
                use_flash = False
                print("  flash_attn unavailable — falling back to "
                      "enable_flash=False (slower attention).", flush=True)
        backbone = PointTransformerV3(
            in_channels=6 + (1 if DG_LOGDK_FEAT else 0),   # xyz + rgb (+ log d_k, D3b)
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
            enable_flash=use_flash,
            cls_mode=False,
        ).cuda()
        head = torch.nn.Linear(64, num_classes).cuda()
        return backbone, head

    from scipy.spatial import cKDTree
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

    def make_predict_scene(backbone, head, num_classes):
        def _predict_scene(scene_path):
            # Tile into CHUNK_XY windows, voxel-downsample tracking the inverse
            # map, scatter per-voxel predictions back, NN-fill stragglers.
            xyz, rgb, _ = load_scene(scene_path)
            pred = np.full(len(xyz), -1, np.int64)
            mins, maxs = xyz[:, :2].min(0), xyz[:, :2].max(0)
            with torch.no_grad():
                for x0 in np.arange(mins[0], maxs[0] + CHUNK_XY, CHUNK_XY):
                    for y0 in np.arange(mins[1], maxs[1] + CHUNK_XY, CHUNK_XY):
                        m = ((xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + CHUNK_XY) &
                             (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + CHUNK_XY))
                        if m.sum() < 64:
                            continue
                        idx = np.where(m)[0]
                        w0 = (xyz[idx] - xyz[idx].mean(0)).astype(np.float32)
                        rgbf = rgb[idx].astype(np.float32) / 255.0
                        # D5 density-TTA: isotropic scale s is a density change (o->o/s^2);
                        # re-voxelize per view, average per-point softmax. views=[1.0] when
                        # off -> identical to the old single-view argmax.
                        views = [1.0] + (list(np.linspace(0.85, 1.2, DG_INFER_TTA))
                                         if DG_INFER_TTA else [])
                        pprob = None
                        for s in views:
                            w = (w0 * s).astype(np.float32)
                            keys = np.floor(w / GRID_SIZE).astype(np.int64)
                            _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                                          return_inverse=True)
                            vx = w[first]
                            feat = np.concatenate(
                                [vx, rgbf[first]]
                                + ([dg.local_density_logdk(vx, DG_LOGDK_K)[:, None]] if DG_LOGDK_FEAT else []),
                                axis=1).astype(np.float32)
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
                        pred[idx] = pprob.argmax(-1)
            miss = pred < 0
            if miss.any() and (~miss).any():
                _, nn = cKDTree(xyz[~miss]).query(xyz[miss])
                pred[miss] = pred[~miss][nn]
            # rgb carries the p95-normalized intensity grayscale the model saw.
            return xyz, np.clip(pred, 0, num_classes - 1), rgb[:, 0] / 255.0
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
            ckpt = torch.load(wpath, map_location="cpu", weights_only=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load weights '{wpath}': {e}\n"
                f"  (loaded safely with weights_only=True — a full-model pickle or a "
                f"checkpoint from another script is rejected; re-export as a state_dict.)"
            ) from e
        bsd, hsd = ckpt["backbone"], ckpt["head"]
        num_classes = int(hsd["weight"].shape[0])
        class_names = [f"class_{i}" for i in range(num_classes)]
        import os as _os, sys as _sys   # read the run's run.json (single manifest) beside the weights
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "helper"))
        from train_common import infer_meta
        meta = infer_meta(wpath)
        if meta:
            class_names = meta.get("class_names") or class_names
            if meta.get("grid") is not None:
                GRID_SIZE = float(meta["grid"])

        backbone, head = build_model(num_classes)
        backbone.load_state_dict(bsd)
        head.load_state_dict(hsd)
        backbone.eval(); head.eval()
        print(f"  [infer] loaded {weights} ({num_classes} classes; "
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
            json.dump({"backbone": "PTv3", "mode": "infer", "weights": weights,
                       "infer_input": infer_input, "num_classes": num_classes,
                       "class_names": class_names, "grid_size": GRID_SIZE,
                       "chunk_xy": CHUNK_XY, "gpu": gpu_name(),
                       "scenes": [os.path.basename(s) for s in scenes]}, f, indent=2)

        predict_scene = make_predict_scene(backbone, head, num_classes)
        palette = _palette(num_classes)
        print(f"  [infer] labeling {len(scenes)} scene(s) -> {pred_dir}", flush=True)
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
    print(f"  PTv3  {dataset}  ({gpu_name()}, {N_EPOCHS} ep, batch {BATCH_SIZE})")
    print("=" * 70)
    print(f"  CUDA: {torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    ensure_prep()

    tag = dataset
    # Namespace runs by stem kernel: shrinking the stem (STEM_KERNEL != 5) changes
    # the architecture, so old 5x5x5-stem checkpoints are incompatible — give them
    # a distinct tag so AUTO_RESUME starts a fresh lineage instead of trying (and
    # failing) to load mismatched weights.
    _pt = "ptv3" if STEM_KERNEL == 5 else f"ptv3k{STEM_KERNEL}"

    def lr_at(ep):
        """PTv3 outdoor schedule: short linear warmup to BASE_LR, then cosine
        decay to ~0. Set on the param groups each epoch — no scheduler state to
        restore, so RESUME is trivial (the KPConvX cold recipe's lr_at pattern)."""
        warm = max(1, int(round(WARMUP_PCT * N_EPOCHS)))
        if ep < warm:
            return BASE_LR * (ep + 1) / warm
        prog = (ep - warm) / max(1, N_EPOCHS - warm)
        return float(0.5 * BASE_LR * (1.0 + np.cos(np.pi * prog)))

    def _find_latest_run():
        """Latest UNFINISHED run dir for this tag with checkpoints (resume target).
        Runs marked DONE are skipped, so a completed experiment isn't re-resumed
        on the next launch — but a crashed/retried one is picked straight back up."""
        def _ep(p):
            return int(os.path.basename(p)[2:5])
        for rd in sorted(glob.glob("/outputs/runs/*"), reverse=True):
            if not rd.endswith(f"{tag}_{_pt}"):
                continue
            if os.path.exists(f"{rd}/DONE"):
                continue
            ckpts = glob.glob(f"{rd}/checkpoints/ep*.pth")
            if ckpts:
                latest = max(ckpts, key=_ep)
                return rd, latest, _ep(latest)
        return None

    resume_info = _find_latest_run() if (RESUME or AUTO_RESUME) else None
    if resume_info:
        run_dir, resume_ckpt, resume_epoch = resume_info
        run_id = os.path.basename(run_dir)
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        start_epoch = resume_epoch + 1
        print(f"  RESUMING {run_id} from {os.path.basename(resume_ckpt)} "
              f"-> epoch {start_epoch}/{N_EPOCHS}", flush=True)
    else:
        run_id = datetime.utcnow().strftime(f"%Y%m%d_%H%M%S_{tag}_{_pt}")
        run_dir = f"/outputs/runs/{run_id}"
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
        resume_ckpt, start_epoch = None, 0
        with open(f"{run_dir}/run_config.json", "w") as f:
            cfg = {
                "backbone": "PTv3", "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
                "dataset": dataset,
                "mode": mode, "gpu": gpu_name(),
                "num_classes": NUM_CLASSES, "grid_size": GRID_SIZE,
                "class_names": CLASS_NAMES,
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
                                  "rare_tile_prob": RARE_TILE_PROB},
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
    synth_train_tiles = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    synth_test_tiles  = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))   # = val tiles
    real_test_tiles   = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))
    hold = {_scene_of(p) for p in synth_test_tiles}
    real_train_tiles = []
    val_tile_dir = f"{PREP_DIR}/val"
    test_raw_split, test_tile_dir = "test", f"{PREP_DIR}/test"
    print(f"  train: {len(synth_train_tiles)}   val(holdout {len(hold)} scenes): "
          f"{len(synth_test_tiles)}   test: {len(real_test_tiles)}", flush=True)

    # --- class balance: scan training tiles for inverse-frequency weights +
    # rare-class tile flags. The per-tile np.load over the Modal volume is the
    # bottleneck (~45k tiles -> minutes), so read in parallel AND cache the raw
    # scan to PREP_DIR (keyed on the tile set) — instant on every later launch. -
    train_pool = synth_train_tiles + real_train_tiles
    cache_path = f"{PREP_DIR}/class_balance_cache.npz"
    pool_names = np.array([os.path.basename(p) for p in train_pool])

    def _scan_tile(tp):
        lab = np.load(tp)["lab"]
        v = lab[(lab >= 0) & (lab < NUM_CLASSES)]
        return (np.bincount(v, minlength=NUM_CLASSES).astype(np.int64)
                if v.size else np.zeros(NUM_CLASSES, np.int64))

    class_counts = present_mask = None
    if os.path.exists(cache_path):
        try:
            cz = np.load(cache_path, allow_pickle=False)
            if (cz["tile_names"].shape == pool_names.shape
                    and bool(np.all(cz["tile_names"] == pool_names))
                    and int(cz["num_classes"]) == NUM_CLASSES):
                class_counts = cz["class_counts"].astype(np.int64)
                present_mask = cz["present_mask"].astype(bool)
                print(f"  class balance: loaded cache ({len(train_pool)} tiles)", flush=True)
        except Exception as e:
            print(f"  class balance: ignoring unreadable cache ({e})", flush=True)

    if class_counts is None:
        from concurrent.futures import ThreadPoolExecutor
        print(f"  scanning {len(train_pool)} train tiles for class balance (parallel)…",
              flush=True)
        per_tile = np.zeros((len(train_pool), NUM_CLASSES), np.int64)
        with ThreadPoolExecutor(max_workers=32) as _ex:
            for i, counts in enumerate(_ex.map(_scan_tile, train_pool)):
                per_tile[i] = counts
        class_counts = per_tile.sum(0)
        present_mask = per_tile > 0
        try:
            np.savez(cache_path, tile_names=pool_names, class_counts=class_counts,
                     present_mask=present_mask, num_classes=np.int64(NUM_CLASSES))
            datasets_volume.commit()
            print(f"  class balance: cached scan -> {cache_path}", flush=True)
        except Exception as e:
            print(f"  class balance: could not write cache ({e})", flush=True)

    def _name(c):
        return CLASS_NAMES[c] if CLASS_NAMES else c
    print(f"  class counts: {dict(zip([_name(c) for c in range(NUM_CLASSES)], class_counts.tolist()))}",
          flush=True)

    # Rare classes: explicit RARE_CLASSES, else auto — present classes whose
    # frequency is below RARE_FREQ_FRAC x the median present-class frequency.
    if RARE_CLASSES is not None:
        rare_set = set(RARE_CLASSES)
    elif RARE_OVERSAMPLE:
        present_freq = class_counts[class_counts > 0]
        thresh = RARE_FREQ_FRAC * float(np.median(present_freq)) if present_freq.size else 0.0
        rare_set = {c for c in range(NUM_CLASSES)
                    if 0 < class_counts[c] < thresh}
    else:
        rare_set = set()
    rare_cols = sorted(rare_set)
    if RARE_OVERSAMPLE and rare_cols:
        rare_tiles = [train_pool[i] for i in np.nonzero(present_mask[:, rare_cols].any(1))[0]]
    else:
        rare_tiles = []
    print(f"  rare classes: {sorted(_name(c) for c in rare_set)}  "
          f"({len(rare_tiles)}/{len(train_pool)} tiles)", flush=True)

    if CLASS_WEIGHTING:
        # w = 1/sqrt(freq) (inverse-sqrt-frequency): sub-linear, mean-normalized,
        # capped to [1/CAP, CAP]. Same scheme as the KPConvX cold script.
        freq = class_counts / max(int(class_counts.sum()), 1)
        w = (1.0 / np.maximum(freq, 1e-6)) ** WEIGHT_BETA
        w[class_counts == 0] = 1.0          # don't up-weight absent classes
        w = w / w[class_counts > 0].mean() if (class_counts > 0).any() else w
        w = np.clip(w, 1.0 / WEIGHT_CAP, WEIGHT_CAP)
        class_weights = torch.tensor(w, dtype=torch.float32).cuda()
        print(f"  class weights: "
              f"{dict(zip([_name(c) for c in range(NUM_CLASSES)], [round(float(x),3) for x in w]))}",
              flush=True)
    else:
        class_weights = None

    # Torch-native weighted (label-smoothed) CE — matches KPConvX's SmoothCE.
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1,
                                        label_smoothing=LABEL_SMOOTH)

    # --- Lovász-Softmax (Berman et al. 2018): differentiable mIoU surrogate,
    # per-class equally weighted, added on top of CE — PTv3's outdoor loss is
    # literally CE + Lovász. Pure-torch flat form (logits (N,C), labels (N,)). --
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
            errors = (fg - probas[:, int(c)]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
        if not losses:
            return probas.sum() * 0.0
        return torch.stack(losses).mean()

    # alpha-balanced multiclass focal loss (alpha = inverse-sqrt class weights).
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
        loss = focal_loss(logits, labels) if USE_FOCAL else ce_loss(logits, labels)
        if LOVASZ_WEIGHT > 0:
            valid = labels >= 0
            if valid.any():
                probas = torch.softmax(logits[valid], dim=1)
                loss = loss + LOVASZ_WEIGHT * lovasz_softmax_flat(probas, labels[valid])
        return loss

    def pick_train_tile():
        if rare_tiles and np.random.rand() < RARE_TILE_PROB:
            return rare_tiles[np.random.randint(len(rare_tiles))]
        return synth_train_tiles[np.random.randint(len(synth_train_tiles))]

    # --- PTv3 outdoor augmentation suite (was entirely absent before) ---------
    def augment_xyz(xyz):
        """Full z-yaw, gentle x/y tilt, isotropic scale, per-axis flip, jitter."""
        az = (np.random.rand() * 2 - 1) * np.pi * AUG_ROT_Z
        ax = (np.random.rand() * 2 - 1) * np.pi * AUG_ROT_XY
        ay = (np.random.rand() * 2 - 1) * np.pi * AUG_ROT_XY
        cz, sz = np.cos(az), np.sin(az)
        cx, sx = np.cos(ax), np.sin(ax)
        cy, sy = np.cos(ay), np.sin(ay)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
        out = xyz @ (Rz @ Ry @ Rx).T
        out = out * np.random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX)
        if np.random.rand() < AUG_FLIP_P:
            out[:, 0] = -out[:, 0]
        if np.random.rand() < AUG_FLIP_P:
            out[:, 1] = -out[:, 1]
        out += np.clip(np.random.normal(0, AUG_JITTER_SIGMA, out.shape),
                       -AUG_JITTER_CLIP, AUG_JITTER_CLIP)
        return out.astype(np.float32)

    def to_ptv3_batch(tiles_for_batch, training=True):
        # PTv3 takes a dict with: coord, grid_coord, feat, offset
        coords, feats, labels, offsets, grid_coords = [], [], [], [], []
        running = 0
        for tile in tiles_for_batch:
            z = np.load(tile)
            xyz, rgb, lab = z["xyz"], z["rgb"], z["lab"]
            # random crop ~30m for memory (train only — eval keeps full tiles)
            if training and len(xyz) > 80000:
                c = xyz[np.random.randint(len(xyz))]
                d2 = np.sum((xyz[:, :2] - c[:2]) ** 2, axis=1)
                idx = np.where(d2 < 15.0 ** 2)[0]
                if len(idx) > 80000:
                    idx = np.random.choice(idx, 80000, replace=False)
                xyz, rgb, lab = xyz[idx], rgb[idx], lab[idx]
            xyz = xyz.astype(np.float32)
            if training and AUG_ENABLE:
                xyz = augment_xyz(xyz)
            xyz = xyz - xyz.mean(0, keepdims=True)
            # Drop non-finite + far-outlier points: a single bad coordinate (NaN/
            # Inf or a stray faraway return) blows up the voxel grid and spconv's
            # indices, triggering a CUDA index device-assert that kills the run.
            ok = (np.isfinite(xyz).all(1)
                  & (np.abs(xyz[:, :2]).max(1) <= CHUNK_XY)
                  & (np.abs(xyz[:, 2]) <= 200.0))
            if int(ok.sum()) < 64:
                continue
            xyz = xyz[ok]; rgb = rgb[ok]; lab = lab[ok]
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
            _, uniq = np.unique(keys, axis=0, return_index=True)
            xyz = xyz[uniq]; rgb = rgb[uniq]; lab = lab[uniq]
            # feat = augmented/centered coords + color/intensity (intensity is
            # carried in the rgb channels for LiDAR canonical datasets).
            feat = np.concatenate(
                [xyz.astype(np.float32), rgb.astype(np.float32) / 255.0]
                + ([dg.local_density_logdk(xyz, DG_LOGDK_K)[:, None]] if DG_LOGDK_FEAT else []),
                axis=1)
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
        split is 'val'|'test' -> the matching materialized dataset folder."""
        return lambda: load_canonical(f"{ds_root}/{split}/{name}.npz")

    val_items = [(n, _raw_loader("val", n), val_tile_dir) for n in sorted(hold)]
    test_items = [(n, _raw_loader(test_raw_split, n), test_tile_dir)
                  for n in sorted({_scene_of(p) for p in real_test_tiles})]
    print(f"  eval set: {len(val_items)} holdout(val) + {len(test_items)} test scenes",
          flush=True)

    def evaluate(scene_items, label):
        """Per-SCENE overlap voting scored on the ORIGINAL raw points (the
        protocol KPConvX/RandLA use). Per scene: forward its overlapping tiles
        (stride CHUNK_XY/2, each point in up to 4 tiles), sum center-tapered
        softmax votes per GRID voxel, argmax, then NN-propagate each voxel's
        prediction to the raw cloud and score against raw GT."""
        t_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_union = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_gt    = np.zeros(NUM_CLASSES, dtype=np.int64)
        correct = total = 0
        n_scenes = n_skipped_tiles = n_skipped_scenes = 0
        t_test = time.time()
        with torch.no_grad():
            for name, load_raw, split_dir in scene_items:
                tiles = sorted(glob.glob(f"{split_dir}/{name}_x*.npz"))
                if not tiles:
                    n_skipped_scenes += 1; continue
                keys_l, vote_l, xyz_l = [], [], []
                for tile in tiles:
                    z = np.load(tile)
                    xyz, rgb = z["xyz"].astype(np.float32), z["rgb"]
                    if len(xyz) < 64:
                        continue
                    cxyz = xyz - xyz.mean(0, keepdims=True)
                    ok = (np.isfinite(cxyz).all(1)
                          & (np.abs(cxyz[:, :2]).max(1) <= CHUNK_XY)
                          & (np.abs(cxyz[:, 2]) <= 200.0))
                    if int(ok.sum()) < 64:
                        continue
                    xyz, rgb, cxyz = xyz[ok], rgb[ok], cxyz[ok]
                    vk = np.floor(cxyz / GRID_SIZE).astype(np.int64)
                    _, first, inverse = np.unique(vk, axis=0, return_index=True,
                                                  return_inverse=True)
                    vx = cxyz[first].astype(np.float32)
                    feat = np.concatenate(
                        [vx, rgb[first].astype(np.float32) / 255.0]
                        + ([dg.local_density_logdk(vx, DG_LOGDK_K)[:, None]] if DG_LOGDK_FEAT else []),
                        axis=1).astype(np.float32)
                    coord = torch.from_numpy(vx).cuda()
                    featt = torch.from_numpy(feat).cuda()
                    offset = torch.tensor([len(vx)], dtype=torch.long).cuda()
                    gc = vk[first] - vk[first].min(0)        # unique, dedup-consistent
                    grid_coord = torch.from_numpy(np.ascontiguousarray(gc)).long().cuda()
                    try:
                        point = backbone({"coord": coord, "grid_coord": grid_coord,
                                          "feat": featt, "offset": offset})
                        fe = point["feat"] if isinstance(point, dict) else point.feat
                        lg = head(fe).cpu().numpy().astype(np.float32)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache(); n_skipped_tiles += 1; continue
                        raise
                    ex = np.exp(lg - lg.max(1, keepdims=True))
                    prob = (ex / ex.sum(1, keepdims=True))[inverse]   # per original pt
                    cxy = (xyz[:, :2].min(0) + xyz[:, :2].max(0)) / 2
                    d = np.abs(xyz[:, :2] - cxy).max(1)
                    wgt = np.clip(1.0 - d / (CHUNK_XY / 2.0), 0.05, 1.0) ** 2
                    keys_l.append(np.floor(xyz / GRID_SIZE).astype(np.int64))
                    vote_l.append((prob * wgt[:, None]).astype(np.float32))
                    xyz_l.append(xyz.astype(np.float32))
                if not keys_l:
                    n_skipped_scenes += 1; continue
                K = np.concatenate(keys_l); V = np.concatenate(vote_l); P = np.concatenate(xyz_l)
                uniq, ufirst, uinv = np.unique(K, axis=0, return_index=True, return_inverse=True)
                votes = np.zeros((len(uniq), NUM_CLASSES), np.float64)
                np.add.at(votes, uinv, V)
                pred_u  = votes.argmax(1)
                rep_xyz = P[ufirst]                      # one raw coord per voxel
                # Reproject voxel predictions onto the raw scene cloud + raw GT.
                try:
                    raw_xyz, _, raw_lab = load_raw()
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped_scenes += 1; continue
                _, nn = cKDTree(rep_xyz).query(raw_xyz)
                raw_pred = pred_u[nn]
                v = (raw_lab >= 0) & (raw_lab < NUM_CLASSES)
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
            "per_class_iou": {_name(c): float(iou_per[c]) for c in range(NUM_CLASSES)},
            "per_class_gt_count": {_name(c): gt_counts[c] for c in range(NUM_CLASSES)},
            "present_classes": [_name(c) for c in present],
            "absent_classes": [_name(c) for c in absent],
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
        print(f"  [{label}] acc={m['overall_acc']:.4f}  "
              f"mIoU({NUM_CLASSES}-way)={m['overall_mIoU']:.4f}  "
              f"mIoU(present {len(present)})={m['present_classes_mIoU']:.4f}  "
              f"absent={m['absent_classes']}  raw_pts={total:,}  "
              f"skipped(tiles={n_skipped_tiles},scenes={n_skipped_scenes})", flush=True)
        return m

    val_csv = f"{run_dir}/val_metrics.csv"
    if not os.path.exists(val_csv):
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "val_acc", "val_miou"] +
                [f"iou_{_name(c)}" for c in range(NUM_CLASSES)])

    import os as _os, sys as _sys   # scripts/helper is a sibling dir (flat /root in Modal)
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "helper"))
    from train_common import BestCheckpoint, write_run_manifest
    best = BestCheckpoint(run_dir)
    write_run_manifest(run_dir, "ptv3", dataset)   # the single inference manifest (run.json)

    def run_eval(ep, write_json=False):
        # Periodic pass scores the held-out VAL scenes only (no test peeking) and
        # selects the best checkpoint on val present-class mIoU. The final pass
        # (write_json) also scores the TEST set and writes both, separately.
        backbone.eval(); head.eval()
        m = evaluate(val_items, f"val@ep{ep}")
        ious = [m["per_class_iou"][_name(c)] for c in range(NUM_CLASSES)]
        with open(val_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{m['overall_acc']:.4f}",
                                    f"{m['present_classes_mIoU']:.4f}"] + [f"{x:.4f}" for x in ious])
        if best.update(m["present_classes_mIoU"]):
            torch.save({"backbone": backbone.state_dict(),
                        "head": head.state_dict(), "epoch": ep}, best.final)
        if write_json:
            mt = evaluate(test_items, f"test@ep{ep}")
            with open(f"{run_dir}/test_metrics.json", "w") as fj:
                json.dump({"val": m, "test": mt,
                           "val_scenes": [n for n, _, _ in val_items],
                           "test_scenes": [n for n, _, _ in test_items]}, fj, indent=2)
        outputs_volume.commit()
        backbone.train(); head.train()
        return m

    LOG_EVERY = 20  # intra-epoch heartbeat
    print(f"  starting at epoch {start_epoch}, up to {N_EPOCHS}, "
          f"{STEPS} steps/epoch (batch {BATCH_SIZE})", flush=True)
    t_run = time.time()
    for ep in range(start_epoch, N_EPOCHS):
        cur_lr = lr_at(ep)
        for g in optim.param_groups:
            g["lr"] = cur_lr
        backbone.train(); head.train()
        ep_loss = ep_correct = ep_total = 0
        ep_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        ep_union = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_ep = time.time(); t_chunk = t_ep; n_steps = 0; last_log_step = 0
        print(f"  ep {ep:3d} starting (lr={cur_lr:.2e})…", flush=True)
        for step in range(STEPS):
            picks = [pick_train_tile() for _ in range(BATCH_SIZE)]
            _dbg = None
            try:
                batch, label = to_ptv3_batch(picks, training=True)
                if DEBUG_CUDA:
                    # Snapshot the batch on CPU BEFORE the forward (GPU still
                    # healthy). If the step then asserts, this is the culprit.
                    gc, co = batch["grid_coord"], batch["coord"]
                    _dbg = {"picks": [os.path.basename(p) for p in picks],
                            "n_pts": int(co.shape[0]), "offset": batch["offset"].tolist(),
                            "grid_min": gc.min(0).values.tolist(),
                            "grid_max": gc.max(0).values.tolist(),
                            "grid_unique": int(torch.unique(gc, dim=0).shape[0]),
                            "coord_min": [round(float(x), 2) for x in co.min(0).values.tolist()],
                            "coord_max": [round(float(x), 2) for x in co.max(0).values.tolist()],
                            "coord_finite": bool(torch.isfinite(co).all().item()),
                            "lab_min": int(label.min().item()), "lab_max": int(label.max().item())}
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
                ep_correct += (pred[m] == label[m]).sum().item()
                ep_total   += int(m.sum())
                for c in range(NUM_CLASSES):
                    ep_inter[c] += ((pred == c) & (label == c)).sum().item()
                    ep_union[c] += (((pred == c) | (label == c)) & m).sum().item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache(); continue
                if _dbg is not None:
                    print(f"  [debug] CRASH at ep {ep} step {step} — offending batch:\n"
                          f"          {json.dumps(_dbg)}", flush=True)
                raise
        sec_per_iter = (time.time() - t_ep) / max(n_steps, 1)
        sec_per_epoch = time.time() - t_ep
        train_acc = ep_correct / max(ep_total, 1)
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
            outputs_volume.commit()
        if (ep + 1) % VAL_EVERY == 0 and ep != N_EPOCHS - 1:
            run_eval(ep)               # last epoch handled by the final eval below

    # --- Final evaluation: the real voted eval over the combined eval set,
    # written to test_metrics.json (the same number run_eval logs periodically). -
    print("  final evaluation over the combined eval set…", flush=True)
    run_eval(N_EPOCHS - 1, write_json=True)
    best.finalize(lambda p: torch.save(
        {"backbone": backbone.state_dict(), "head": head.state_dict(),
         "epoch": N_EPOCHS - 1}, p))
    print(f"  total wall-clock {(time.time() - t_run)/3600:.2f} h")
    outputs_volume.commit()

    # Mark the run complete so AUTO_RESUME won't re-resume it on the next launch
    # (a crashed/retried run has no DONE and is picked back up automatically).
    open(f"{run_dir}/DONE", "w").close()
    outputs_volume.commit()
    print(f"  run complete -> {run_id}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Local ptv3 trainer/inferencer (no modal).')
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
