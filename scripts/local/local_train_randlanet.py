"""RandLA-Net local trainer, cold start, on a canonical trainer_gui dataset.
FEAT_CHANNELS env overrides the input spec; run.json "features" records it and
inference rebuilds from it. Flags: --dataset --sub-grid --num-points --epochs
--batch --steps-per-epoch | --mode infer --weights --infer-input <job_id>.
"""

import os
import sys
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "helper"))
import density as dg
import train_common as tc

N_EPOCHS      = 100
BATCH_SIZE    = 6
VAL_BATCH     = 12

NUM_POINTS    = 45056
SUB_GRID_SIZE = 0.25
FEAT_CHANNELS = ""
AUG_COLOR     = 0.8

DG_DENSITY_AUG = False
DG_COARSEN_MAX = 2.0
DG_P_NATIVE    = 0.5
DG_INFER_ADABN = False
DG_INFER_APCOTTA = False
DG_INFER_TTA   = 0
EVAL_VOTES     = 2
DG_LOGDK_FEAT  = False
DG_LOGDK_K     = 8

CLASS_WEIGHTING  = True
WEIGHT_BETA      = 0.5
WEIGHT_CAP       = 5.0
LOVASZ_WEIGHT    = 1.0
USE_FOCAL        = False
FOCAL_GAMMA      = 2.0
RARE_OVERSAMPLE  = True
RARE_FREQ_THRESH = 0.02
RARE_CENTER_PROB = 0.25
VAL_EVERY        = 10
PROXY_SAMPLING   = "full"
PROXY_TILES      = 48
PROXY_ANCHORS    = 3

RESUME           = False
AUTO_RESUME      = os.environ.get("AUTO_RESUME", "0") == "1"


def _randlanet_root() -> str:
    """RandLA-Net-pytorch source root: RANDLANET_SRC or /opt/randlanet (Modal image)."""
    env = os.environ.get("RANDLANET_SRC")
    for p in (env, "/opt/randlanet"):
        if p and os.path.isdir(p):
            return p
    if env:
        raise RuntimeError(f"RANDLANET_SRC={env} is not a directory - fix the "
                           "env var or bake the source at /opt/randlanet")
    raise RuntimeError(
        "RandLA-Net source not found: set RANDLANET_SRC to your RandLA-Net-pytorch "
        "clone (the pixi env activation sets it) or bake it at /opt/randlanet")


_DP = None
def _dp():
    """utils.data_process.DataProcessing from the source tree, imported once."""
    global _DP
    if _DP is None:
        root = _randlanet_root()
        if root not in sys.path:
            sys.path.insert(0, root)
        from utils.data_process import DataProcessing
        _DP = DataProcessing
    return _DP


# data plumbing stays at module level: Windows-spawn DataLoader workers pickle the dataset/collate by qualified name, so <locals> classes/closures can't cross
class Collater:
    """Batch assembly (KNN pyramid + tensor packing); picklable across spawn."""
    def __init__(self, num_layers, k_n, sub_sampling_ratio, feat_spec, nonxyz):
        self.num_layers = num_layers
        self.k_n = k_n
        self.sub_sampling_ratio = sub_sampling_ratio
        self.feat_spec = feat_spec
        self.nonxyz = nonxyz

    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        DP = _dp()
        input_points, input_neighbors, input_pools, input_up = [], [], [], []
        for i in range(self.num_layers):
            neigh = DP.knn_search(batch_pc, batch_pc, self.k_n)
            sub_n = batch_pc.shape[1] // self.sub_sampling_ratio[i]
            sub_points = batch_pc[:, :sub_n, :]
            pool_i = neigh[:, :sub_n, :]
            up_i  = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc); input_neighbors.append(neigh)
            input_pools.append(pool_i);    input_up.append(up_i)
            batch_pc = sub_points
        return (input_points + input_neighbors + input_pools + input_up
                + [batch_label, batch_pc_idx, batch_cloud_idx])

    def __call__(self, batch):
        pcs, feats, lbs, idxs, cinds = zip(*batch)
        pcs  = np.stack(pcs);  feats = np.stack(feats); lbs = np.stack(lbs)
        idxs = np.stack(idxs); cinds = np.stack(cinds)
        flat = self.tf_map(pcs, lbs, idxs, cinds)
        n = self.num_layers
        d = {"xyz": [], "neigh_idx": [], "sub_idx": [], "interp_idx": []}
        for t in flat[:n]:           d["xyz"].append(torch.from_numpy(t).float())
        for t in flat[n:2*n]:        d["neigh_idx"].append(torch.from_numpy(t).long())
        for t in flat[2*n:3*n]:      d["sub_idx"].append(torch.from_numpy(t).long())
        for t in flat[3*n:4*n]:      d["interp_idx"].append(torch.from_numpy(t).long())
        ax = {"x": 0, "y": 1, "z": 2}
        cols = [pcs[:, :, ax[n2]] if n2 in ax else feats[:, :, self.nonxyz.index(n2)]
                for n2 in self.feat_spec]
        tail = feats[:, :, len(self.nonxyz):]
        full_feat = np.concatenate([np.stack(cols, axis=2), tail],
                                   axis=2)
        d["features"] = torch.from_numpy(full_feat).float().transpose(1, 2)
        d["labels"]   = torch.from_numpy(flat[4*n]).long()
        d["input_inds"] = torch.from_numpy(flat[4*n+1]).long()
        d["cloud_inds"] = torch.from_numpy(flat[4*n+2]).long()
        return d


class Scenes(Dataset):
    """Scene cache + sphere sampler; o = sampling knobs (SimpleNamespace).
    Picklable across spawn: __getstate__ strips the cache, workers reload lazily."""
    def __init__(self, split, files, o, label=None):
        self.split = split
        self.o = o
        self.label = label or split
        self.files = files
        self.scenes = [self._load(f) for f in self.files]
        self.rare_idx, self.rare_scenes = None, None
        self._rng = None
        print(f"  [{self.label}] {len(self.scenes)} scenes", flush=True)

    def __getstate__(self):
        return {**self.__dict__, "scenes": None, "_rng": None}

    def _load(self, f):
        z = np.load(f)
        tc.require_finite_xyz(z["xyz"], os.path.basename(f))
        extras = tc.feat_extras(z, self.o.feat_spec, os.path.basename(f))
        return (z["xyz"], z["intensity"], z["ret_num"], extras, z["lab"])

    def _scene(self, i):
        if self.scenes is None:
            self.scenes = [None] * len(self.files)
        if self.scenes[i] is None:
            self.scenes[i] = self._load(self.files[i])
        return self.scenes[i]

    def set_rare_classes(self, rare_classes):
        self.rare_idx = [np.where(np.isin(lab, rare_classes))[0]
                         for *_, lab in self.scenes]
        self.rare_scenes = [i for i, r in enumerate(self.rare_idx) if len(r)]

    def sample_sphere(self, cloud_idx, center_idx, augment=False, rng=np.random):
        o = self.o
        xyz, intensity, ret_num, extras, lab = self._scene(cloud_idx)
        center = xyz[center_idx:center_idx + 1]
        d2 = np.sum((xyz - center) ** 2, axis=1)
        sel = np.argpartition(d2, min(o.num_points, len(xyz) - 1))[:o.num_points]
        if len(sel) < o.num_points:
            sel = np.concatenate([sel, rng.choice(len(xyz), o.num_points - len(sel))])
        rng.shuffle(sel)
        if augment and o.density_aug:
            g_eff = dg.effective_grid(o.sub_grid, o.coarsen_max, o.p_native, rng=rng)
            if g_eff > o.sub_grid:
                keep = dg.voxel_first_idx(xyz[sel], g_eff)
                sel = sel[keep]
                if len(sel) < o.num_points:
                    sel = np.concatenate([sel, rng.choice(sel, o.num_points - len(sel))])
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
        cols = [src[n][sel] for n in o.nonxyz]
        if augment:
            cols = [np.zeros_like(c) if rng.rand() > o.aug_color else c
                    for c in cols]
        if o.logdk_feat:
            cols.append(dg.local_density_logdk(pc, o.logdk_k))
        feat2 = (np.stack(cols, axis=1) if cols
                 else np.zeros((len(sel), 0))).astype(np.float32)
        lb = lab[sel].astype(np.int64)
        return pc.astype(np.float32), feat2, lb, sel.astype(np.int32), \
               np.array([cloud_idx], dtype=np.int32)

    def __len__(self):
        return self.o.steps * self.o.batch_size if self.split == "train" \
            else len(self.files)

    def __getitem__(self, idx):
        r = self._rng if self._rng is not None else np.random
        if self.split == "train":
            if (self.rare_scenes and self.o.rare_oversample
                    and r.rand() < self.o.rare_center_prob):
                ci = self.rare_scenes[r.randint(len(self.rare_scenes))]
                pick = int(self.rare_idx[ci][r.randint(len(self.rare_idx[ci]))])
            else:
                ci = r.randint(len(self.files))
                pick = r.randint(len(self._scene(ci)[0]))
            return self.sample_sphere(ci, pick, augment=True, rng=r)
        return self.sample_sphere(idx, r.randint(len(self._scene(idx)[0])),
                                  augment=False, rng=r)


def _worker_init(worker_id):
    # PyTorch reseeds python/torch per worker but NOT numpy; without this every worker replays identical sphere picks and augmentations
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.RandomState(info.seed % (2 ** 32))


def train_randlanet(dataset: Optional[str] = None, sub_grid: Optional[float] = None,
                    num_points: Optional[int] = None, epochs: Optional[int] = None,
                    batch: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                    mode: str = "train", weights: Optional[str] = None,
                    infer_input: Optional[str] = None):
    if dataset is None and mode != "infer":
        raise ValueError("--dataset is required: pass a canonical trainer_gui "
                         "dataset name under /datasets. The only "
                         "dataset-free path is --mode infer.")
    import time, json, csv, glob
    from datetime import datetime, timezone
    (DG_DENSITY_AUG, DG_COARSEN_MAX, DG_P_NATIVE, DG_LOGDK_FEAT, DG_LOGDK_K,
     DG_INFER_ADABN, DG_INFER_APCOTTA, DG_INFER_TTA, EVAL_VOTES, USE_FOCAL,
     FOCAL_GAMMA, CLASS_WEIGHTING, WEIGHT_BETA, RARE_OVERSAMPLE,
     RARE_CENTER_PROB, VAL_EVERY, FEAT_CHANNELS,
     PROXY_SAMPLING) = tc.env_overrides(globals(), [
        "DG_DENSITY_AUG", "DG_COARSEN_MAX", "DG_P_NATIVE", "DG_LOGDK_FEAT",
        "DG_LOGDK_K", "DG_INFER_ADABN", "DG_INFER_APCOTTA", "DG_INFER_TTA",
        "EVAL_VOTES", "USE_FOCAL", "FOCAL_GAMMA", "CLASS_WEIGHTING",
        "WEIGHT_BETA", "RARE_OVERSAMPLE", "RARE_CENTER_PROB", "VAL_EVERY",
        "FEAT_CHANNELS", "PROXY_SAMPLING"])
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    SUB_GRID_SIZE = sub_grid if sub_grid is not None else globals()["SUB_GRID_SIZE"]
    FEAT_DEFAULT = ["x", "y", "z", "intensity", "return_number"]
    FEAT_SPEC = (list(FEAT_DEFAULT) if mode == "infer"
                 else tc.parse_feat_spec(FEAT_CHANNELS, FEAT_DEFAULT))
    bad = [n for n in FEAT_SPEC
           if n not in FEAT_DEFAULT and not n.startswith("feat_")]
    if bad:
        raise ValueError(f"RandLA-Net can't feed {bad}; supported: "
                         f"{FEAT_DEFAULT} plus dataset feat_* channels")
    NONXYZ = [n for n in FEAT_SPEC if n not in ("x", "y", "z")]
    IN_DIM = len(FEAT_SPEC) + (1 if DG_LOGDK_FEAT else 0)
    NUM_POINTS    = num_points if num_points is not None else globals()["NUM_POINTS"]
    N_EPOCHS      = epochs if epochs is not None else globals()["N_EPOCHS"]
    BATCH_SIZE    = batch if batch is not None else globals()["BATCH_SIZE"]
    STEPS         = steps_per_epoch if steps_per_epoch is not None else 500
    if dataset:
        ds_root = tc.dataset_dir(dataset)
        ds_meta, NUM_CLASSES, CLASS_NAMES = tc.load_dataset_meta(dataset)
        PREP_DIR = (f"{ds_root}/prep/randlanet"
                    f"_grid{int(round(SUB_GRID_SIZE * 100))}_p95"
                    f"{tc.feat_spec_tag(FEAT_SPEC, FEAT_DEFAULT)}")
    else:
        ds_meta, NUM_CLASSES, CLASS_NAMES, PREP_DIR = {}, 0, [], None

    # pinned upstream calls confusion_matrix(y, p, labels) positionally; newer sklearn made labels keyword-only
    import sklearn.metrics as _skm
    _orig_cm = _skm.confusion_matrix
    def _cm_compat(y_true, y_pred, labels=None, **kwargs):
        return _orig_cm(y_true, y_pred, labels=labels, **kwargs)
    _skm.confusion_matrix = _cm_compat

    sys.path.insert(0, _randlanet_root())
    from network.RandLANet import Network
    import network.pytorch_utils as pt_utils
    from utils.metric import compute_acc, IoUCalculator

    def build_net(num_classes_, in_dim=IN_DIM):
        """Network with fc0 rebuilt for in_dim channels. Upstream hardcodes
        Conv1d(3, 8) (xyz only); we feed [xyz, intensity, return_number]."""
        cfg.num_classes = num_classes_
        net_ = Network(cfg)
        if in_dim != 3:
            net_.fc0 = pt_utils.Conv1d(in_dim, 8, kernel_size=1, bn=True)
        return net_.to(device)

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
    collate_fn = Collater(cfg.num_layers, cfg.k_n, cfg.sub_sampling_ratio,
                          FEAT_SPEC, NONXYZ)

    def load_canonical(npz_path):
        """Scene npz -> (xyz, intensity, return_number, label, feat_* extras)."""
        z = np.load(npz_path)
        raw = z["xyz"]
        tc.require_finite_xyz(raw, os.path.basename(npz_path))
        # origin shift before float32 cast keeps UTM coords sub-meter precise
        xyz = (raw - np.floor(raw.min(0))).astype(np.float32)
        intensity, ret_num = tc.scene_arrays(z, len(xyz))
        lab = z["label"].astype(np.int32) if "label" in z \
            else np.full(len(xyz), -1, np.int32)
        return xyz, intensity, ret_num, lab, tc.scene_feats(z)

    def grid_subsample(xyz, intensity, ret_num, lab, extras, grid):
        keys = np.floor(xyz / grid).astype(np.int64)
        uniq = tc.voxel_unique(keys)
        return (xyz[uniq], intensity[uniq], ret_num[uniq], lab[uniq],
                {n: v[uniq] for n, v in extras.items()})

    def _cache_signature():
        sig = {
            "format_version": 2,
            "coord_frame": "scene-local",
            "pipeline": "randlanet",
            "dataset": dataset,
            "sub_grid_size": SUB_GRID_SIZE,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "feature_recipe": ",".join(FEAT_SPEC).replace("x,y,z", "xyz"),
        }
        sp = ds_meta.get("split", {}) if isinstance(ds_meta, dict) else {}
        sig["split_seed"] = sp.get("seed")
        sig["split_mode"] = sp.get("mode")
        return sig

    def ensure_prep():
        print(f"  ensuring preprocessed cache -> {PREP_DIR}", flush=True)
        for split in ("train", "val", "test"):
            os.makedirs(f"{PREP_DIR}/{split}", exist_ok=True)
        train_list, val_list, test_list = tc.split_scenes(ds_root)
        any_new = tc.validate_cache(PREP_DIR, _cache_signature())
        for split, items in (("train", train_list), ("val", val_list),
                             ("test", test_list)):
            print(f"  [{split}] {len(items)} scenes", flush=True)
            for i, (name, pc_path) in enumerate(items):
                out = f"{PREP_DIR}/{split}/{name}.npz"
                if os.path.exists(out + ".done"):
                    continue
                t0 = time.time()
                try:
                    xyz, intensity, ret_num, lab, extras = load_canonical(pc_path)
                    n_in = len(xyz)
                    xyz, intensity, ret_num, lab, extras = grid_subsample(
                        xyz, intensity, ret_num, lab, extras, SUB_GRID_SIZE)
                except Exception as e:
                    raise RuntimeError(f"failed to load/subsample {pc_path}: {e}. "
                                       "Fix or remove the scene, then re-run") from e
                tile = dict(xyz=xyz.astype(np.float32),
                            intensity=intensity.astype(np.float32),
                            ret_num=ret_num.astype(np.float32),
                            lab=lab.astype(np.int32))
                for n, v in extras.items():
                    tile[n] = v.astype(np.float32)
                np.savez_compressed(out, **tile)
                open(out + ".done", "w").close()
                any_new = True
                print(f"    [{i+1}/{len(items)}] {name}: {n_in:,} -> "
                      f"{len(xyz):,} pts in {time.time()-t0:.1f}s", flush=True)
        if any_new:
            print("  preprocessing cache updated.", flush=True)
        else:
            print("  all scenes already cached.", flush=True)
        return train_list, val_list, test_list

    device = torch.device("cuda")

    from scipy.spatial import cKDTree

    def make_predict_scene(net, num_classes, exclude_idx=None):
        SAVE_PROBS = os.environ.get("TT_SAVE_PROBS") == "1"

        def _predict_scene(pc_path):
            z = np.load(pc_path)
            raw0 = z["xyz"]
            tc.require_finite_xyz(raw0, os.path.basename(pc_path))
            xyz0 = (raw0 - np.floor(raw0.min(0))).astype(np.float32)
            itn0, ret0 = tc.scene_arrays(z, len(xyz0))
            ex0 = tc.feat_extras(z, FEAT_SPEC, os.path.basename(pc_path))
            keys = np.floor(xyz0 / SUB_GRID_SIZE).astype(np.int64)
            uniq = tc.voxel_unique(keys)
            sub_xyz = xyz0[uniq]
            sub_src = {"intensity": itn0[uniq], "return_number": ret0[uniq],
                       **{n2: v[uniq] for n2, v in ex0.items()}}
            sub_votes = np.zeros((len(sub_xyz), num_classes), np.float32)
            N = cfg.num_points
            n_passes = max(int(EVAL_VOTES), 1)
            max_blocks = 2 * (len(sub_xyz) // N + 1)
            with torch.no_grad():
              for vp in range(n_passes):
                rng = np.random.RandomState(20250720 + vp)
                uncov = np.ones(len(sub_xyz), bool)
                for _ in range(max_blocks):
                    rem = np.flatnonzero(uncov)
                    if not len(rem):
                        break
                    seed = int(rng.choice(rem))
                    d2 = np.sum((sub_xyz - sub_xyz[seed]) ** 2, axis=1)
                    sel = np.argpartition(d2, min(N, len(sub_xyz) - 1))[:N]
                    uncov[sel] = False
                    real = len(sel)
                    if real < 64:
                        continue
                    block = sub_xyz[sel]
                    cols = ([sub_src[n2][sel] for n2 in NONXYZ]
                            + ([dg.local_density_logdk(block, DG_LOGDK_K)] if DG_LOGDK_FEAT else []))
                    f2 = (np.stack(cols, axis=1) if cols
                          else np.zeros((len(block), 0), np.float32))
                    orig = sel.astype(np.int64)
                    if real < N:
                        pad = np.random.choice(real, N - real)
                        block = np.concatenate([block, block[pad]], axis=0)
                        f2 = np.concatenate([f2, f2[pad]], axis=0)
                        orig = np.concatenate([orig, np.full(N - real, -1, np.int64)])
                    perm = np.random.permutation(N)
                    block, f2, orig = block[perm], f2[perm], orig[perm]
                    pc0 = (block - sub_xyz[seed]).astype(np.float32)
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
                fb = min(set(range(num_classes)) - set(exclude_idx or ()))
                return (raw0, np.full(len(xyz0), fb, np.int64), itn0,
                        np.zeros(len(xyz0), np.float32),
                        np.zeros((len(xyz0), num_classes), np.float16)
                        if SAVE_PROBS else None)
            nn = cKDTree(sub_xyz[valid]).query(xyz0, workers=-1)[1]
            vv = sub_votes[valid]
            vv /= vv.sum(1, keepdims=True)
            vv = tc.apply_class_mask(vv, exclude_idx)
            pred = vv.argmax(1)[nn]
            conf = vv.max(1)[nn]
            probs = vv[nn].astype(np.float16) if SAVE_PROBS else None
            return raw0, np.clip(pred, 0, num_classes - 1), itn0, conf, probs
        return _predict_scene

    if mode == "infer":
        if not weights or not infer_input:
            raise ValueError("--mode infer requires --weights and --infer-input")
        wpath = tc.resolve_weights_path(weights)
        if not os.path.exists(wpath):
            raise FileNotFoundError(f"weights not found: {wpath}")
        ckpt = tc.load_ckpt_safe(wpath, map_location=device)
        # accept wrapped ('model'/'model_state_dict'/'state_dict') and bare state dicts
        sd = (ckpt.get("model", ckpt.get("model_state_dict",
                                         ckpt.get("state_dict", ckpt)))
              if isinstance(ckpt, dict) else ckpt)
        fc3_key = next((k for k in sd if k.startswith("fc3.") and k.endswith("weight")), None)
        if fc3_key is None:
            raise ValueError(f"{weights}: fc3/fc0 layers missing - not a RandLA-Net state dict from this trainer")
        num_classes = int(sd[fc3_key].shape[0])
        class_names = [f"class_{i}" for i in range(num_classes)]
        meta = tc.infer_meta(wpath)
        if meta:
            class_names = meta.get("class_names") or class_names
            if meta.get("grid") is not None:
                SUB_GRID_SIZE = float(meta["grid"])
            if meta.get("num_points"):
                cfg.num_points = int(meta["num_points"])
                cfg.num_sub_points = [cfg.num_points // r
                                      for r in (4, 16, 64, 256)]
        fc0_key = next((k for k in sd if k.startswith("fc0.") and sd[k].dim() >= 2), None)
        if fc0_key is None:
            raise ValueError(f"{weights}: fc3/fc0 layers missing - not a RandLA-Net state dict from this trainer")
        ckpt_in_dim = int(sd[fc0_key].shape[1])
        mf = (meta or {}).get("features") or []
        FEAT_SPEC = (tc.parse_feat_spec(",".join(mf), FEAT_DEFAULT) if mf
                     else list(FEAT_DEFAULT))
        NONXYZ = [n for n in FEAT_SPEC if n not in ("x", "y", "z")]
        IN_DIM = len(FEAT_SPEC) + (1 if DG_LOGDK_FEAT else 0)
        collate_fn = Collater(cfg.num_layers, cfg.k_n, cfg.sub_sampling_ratio,
                              FEAT_SPEC, NONXYZ)
        net = build_net(num_classes, in_dim=ckpt_in_dim)
        if ckpt_in_dim != IN_DIM:
            raise ValueError(
                f"checkpoint fc0 expects {ckpt_in_dim} input channels but this "
                f"script feeds {IN_DIM} ({FEAT_SPEC}"
                f"{' + logdk' if DG_LOGDK_FEAT else ''}). "
                f"Use weights trained with the same feature recipe")
        net.load_state_dict(sd)
        net.eval()
        print(f"  [infer] loaded {weights} ({num_classes} classes: {class_names}; "
              f"final_model = best-val epoch {ckpt.get('epoch', '?')})", flush=True)

        run_dir = tc.infer_dir(infer_input)
        scenes = sorted(glob.glob(f"{run_dir}/scenes/*.npz"))
        if not scenes:
            raise FileNotFoundError(f"No scenes under {run_dir}/scenes")

        pred_dir = os.environ.get("TT_PRED_DIR") or f"{run_dir}/predictions"
        os.makedirs(pred_dir, exist_ok=True)
        exc_idx = tc.exclude_class_idx(class_names)
        infer_cfg = {"backbone": "RandLA-Net", "mode": "infer", "weights": weights,
                     "infer_input": infer_input, "num_classes": num_classes,
                     "class_names": class_names, "sub_grid_size": SUB_GRID_SIZE,
                     "features": FEAT_SPEC, "gpu": tc.gpu_name(),
                     "exclude_classes": [class_names[i] for i in exc_idx],
                     "started_utc": datetime.now(timezone.utc).isoformat()}

        predict_scene = make_predict_scene(net, num_classes, exclude_idx=exc_idx)
        if DG_INFER_ADABN and DG_INFER_APCOTTA:
            raise ValueError("DG_INFER_ADABN and DG_INFER_APCOTTA are both set; "
                             "unset one - they are alternative adaptation modes")
        if DG_INFER_ADABN or DG_INFER_APCOTTA:

            def _target_batches(cap=30):
                seen = 0
                N = cfg.num_points
                for pc_path in scenes:
                    if seen >= cap:
                        return
                    z = np.load(pc_path)
                    raw0 = z["xyz"]
                    tc.require_finite_xyz(raw0, os.path.basename(pc_path))
                    xyz0 = (raw0 - np.floor(raw0.min(0))).astype(np.float32)
                    itn0, ret0 = tc.scene_arrays(z, len(xyz0))
                    ex0 = tc.feat_extras(z, FEAT_SPEC, os.path.basename(pc_path))
                    keys = np.floor(xyz0 / SUB_GRID_SIZE).astype(np.int64)
                    uniq = tc.voxel_unique(keys)
                    sx = xyz0[uniq]
                    s_src = {"intensity": itn0[uniq], "return_number": ret0[uniq],
                             **{n: v[uniq] for n, v in ex0.items()}}
                    rng = np.random.RandomState(20250720)
                    uncov = np.ones(len(sx), bool)
                    for _ in range(2 * (len(sx) // N + 1)):
                        if seen >= cap:
                            return
                        rem = np.flatnonzero(uncov)
                        if not len(rem):
                            break
                        seed_i = int(rng.choice(rem))
                        d2 = np.sum((sx - sx[seed_i]) ** 2, axis=1)
                        sel = np.argpartition(d2, min(N, len(sx) - 1))[:N]
                        uncov[sel] = False
                        real = len(sel)
                        if real < 64:
                            continue
                        block = sx[sel]
                        cols = ([s_src[n][sel] for n in NONXYZ]
                                + ([dg.local_density_logdk(block, DG_LOGDK_K)] if DG_LOGDK_FEAT else []))
                        f2 = (np.stack(cols, axis=1) if cols
                              else np.zeros((len(block), 0), np.float32))
                        if real < N:
                            pad = np.random.choice(real, N - real)
                            block = np.concatenate([block, block[pad]], 0)
                            f2 = np.concatenate([f2, f2[pad]], 0)
                        perm = np.random.permutation(N)
                        block, f2 = block[perm], f2[perm]
                        pc_c = (block - sx[seed_i]).astype(np.float32)
                        item = (pc_c, f2.astype(np.float32), np.zeros(N, np.int64),
                                np.arange(N, dtype=np.int32), np.array([0], np.int32))
                        batch = collate_fn([item])
                        for k in ("features", "labels", "input_inds", "cloud_inds"):
                            batch[k] = batch[k].to(device)
                        for k in ("xyz", "neigh_idx", "sub_idx", "interp_idx"):
                            batch[k] = [t.to(device) for t in batch[k]]
                        seen += 1
                        yield batch
            if DG_INFER_APCOTTA:
                print("  [infer] APCoTTA: adapting BN on target tiles...", flush=True)
                dg.apcotta_adapt(
                    net, _target_batches(),
                    logits_fn=lambda m, b: m(b)["logits"].transpose(1, 2)
                                               .reshape(-1, num_classes))
            else:
                print("  [infer] AdaBN: recomputing BN stats on target tiles...", flush=True)
                dg.adabn_recalibrate(net, _target_batches(), forward=lambda m, b: m(b))
                net.eval()
        tc.run_infer_scenes(scenes, predict_scene, pred_dir, run_dir, infer_cfg)
        return

    print("=" * 70)
    print(f"  RandLA-Net  {dataset}  "
          f"({tc.gpu_name()}, {N_EPOCHS} ep, batch {BATCH_SIZE})")
    print("=" * 70)
    tc.clear_stop()
    train_list, val_list, test_list = ensure_prep()
    train_files = sorted(glob.glob(f"{PREP_DIR}/train/*.npz"))
    val_files   = sorted(glob.glob(f"{PREP_DIR}/val/*.npz"))
    test_files  = sorted(glob.glob(f"{PREP_DIR}/test/*.npz"))

    VAL_RANK = tc.ranking_protocol(PROXY_SAMPLING)
    VAL_FULL = VAL_RANK == "full"
    proxy_files, proxy_rep = tc.pick_proxy_tiles(
        val_files, NUM_CLASSES, PROXY_TILES, mode=PROXY_SAMPLING,
        class_names=CLASS_NAMES,
        cache_path=f"{PREP_DIR}/val_class_balance_cache.npz",
        viable=lambda p: np.load(p)["lab"].size >= 64)
    proxy_rep["tiles"].append(f"anchors={PROXY_ANCHORS} "
                              f"slots={cfg.val_steps * VAL_BATCH} "
                              f"num_points={cfg.num_points}")
    print(tc.VAL_FULL_NOTE if VAL_FULL else proxy_rep["text"], flush=True)

    tag = dataset
    _recipe = {"features": FEAT_SPEC, "n_epochs": N_EPOCHS,
               "num_classes": NUM_CLASSES, "class_names": CLASS_NAMES}
    resume_info = (tc.find_latest_unfinished_run(f"{tag}_randlanet_cold", _recipe)
                   if (RESUME or AUTO_RESUME) else None)
    if resume_info and not tc.proxy_guard(resume_info[0], proxy_rep,
                                          tc.PROXY_PROTOCOL_SPHERES, CLASS_NAMES,
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
        resume_ckpt, start_epoch = None, 0
        run_id = datetime.now(timezone.utc).strftime(f"%Y%m%d_%H%M%S_{tag}_randlanet_cold")
        run_dir = f"{tc.OUTPUTS_ROOT}/runs/{run_id}"
        os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    if resume_ckpt is None:
        with open(f"{run_dir}/run.json", "w") as f:
            json.dump({
                "backbone": "RandLA-Net",
                "dataset": dataset,
                "mode": mode, "gpu": tc.gpu_name(),
                "n_epochs": N_EPOCHS,
                "batch_size": BATCH_SIZE, "num_points": NUM_POINTS,
                "sub_grid_size": SUB_GRID_SIZE, "in_dim": IN_DIM,
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
                "train_scenes": [n for n, _ in train_list],
                "val_scenes":   [n for n, _ in val_list],
                "test_scenes":  [n for n, _ in test_list],
            }, f, indent=2)

    dso = SimpleNamespace(
        num_points=cfg.num_points, steps=cfg.train_steps, batch_size=BATCH_SIZE,
        feat_spec=FEAT_SPEC, nonxyz=NONXYZ, aug_color=AUG_COLOR,
        sub_grid=SUB_GRID_SIZE, density_aug=DG_DENSITY_AUG,
        coarsen_max=DG_COARSEN_MAX, p_native=DG_P_NATIVE,
        logdk_feat=DG_LOGDK_FEAT, logdk_k=DG_LOGDK_K,
        rare_oversample=RARE_OVERSAMPLE, rare_center_prob=RARE_CENTER_PROB)
    train_ds = Scenes("train", train_files, dso)
    val_ds   = Scenes("val",   val_files,   dso, label="val")
    test_ds  = Scenes("test",  test_files,  dso, label="test")

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
        w = tc.class_weights_np(class_counts, WEIGHT_BETA, WEIGHT_CAP)
        _class_w = torch.tensor(w, dtype=torch.float32).to(device)
        _ce = nn.CrossEntropyLoss(weight=_class_w, ignore_index=-1)
        print(f"  class weights: "
              f"{dict(zip(CLASS_NAMES, [round(float(x), 3) for x in w]))}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, collate_fn=collate_fn,
                              worker_init_fn=_worker_init,
                              pin_memory=True, drop_last=True)

    net = build_net(NUM_CLASSES)
    print(f"  params: {sum(p.numel() for p in net.parameters()):,}")

    opt = optim.Adam(net.parameters(), lr=cfg.learning_rate)
    if resume_ckpt is not None:
        rckpt = torch.load(resume_ckpt, map_location=device, weights_only=True)
        net.load_state_dict(rckpt["model"])
        if "optim" in rckpt:
            opt.load_state_dict(rckpt["optim"])
        print(f"  resumed weights{' + optimizer' if 'optim' in rckpt else ''} "
              f"at epoch {start_epoch}", flush=True)
    sched = optim.lr_scheduler.ExponentialLR(opt, 0.95)

    metrics_csv = tc.init_metrics_csv(run_dir)

    def _to_device(batch):
        for k in ("features", "labels", "input_inds", "cloud_inds"):
            batch[k] = batch[k].to(device)
        for k in ("xyz", "neigh_idx", "sub_idx", "interp_idx"):
            batch[k] = [t.to(device) for t in batch[k]]
        return batch

    val_csv = f"{run_dir}/val_metrics.csv"
    tc.init_val_csv(val_csv, CLASS_NAMES)

    def evaluate(ds, name2src, label):
        """Full-coverage eval scored on raw points: sphere-sweep the subsampled
        scene over EVAL_VOTES overlap passes, NN-propagate, score raw GT."""
        t_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_union = np.zeros(NUM_CLASSES, dtype=np.int64)
        t_gt    = np.zeros(NUM_CLASSES, dtype=np.int64)
        correct = total = 0; t_test = time.time()
        n_scenes = n_skipped = 0
        N = cfg.num_points
        with torch.no_grad():
            for i, (xyz, intensity, ret_num, extras, lab) in enumerate(ds.scenes):
                src = {"intensity": intensity, "return_number": ret_num, **extras}
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
                        valid = orig >= 0
                        votes[orig[valid]] += p[bi, valid]
                    pend_items, pend_blocks = [], []

                n_passes = max(int(EVAL_VOTES), 1)
                max_blocks = 2 * (len(xyz) // N + 1)
                for vp in range(n_passes):
                    rng = np.random.RandomState(20250720 + vp)
                    uncov = np.ones(len(xyz), bool)
                    for _ in range(max_blocks):
                        rem = np.flatnonzero(uncov)
                        if not len(rem):
                            break
                        seed = int(rng.choice(rem))
                        d2 = np.sum((xyz - xyz[seed]) ** 2, axis=1)
                        sel = np.argpartition(d2, min(N, len(xyz) - 1))[:N]
                        uncov[sel] = False
                        real = len(sel)
                        if real < 64:
                            continue
                        pts_blk = xyz[sel]
                        cols = ([src[n][sel] for n in NONXYZ]
                                + ([dg.local_density_logdk(pts_blk, DG_LOGDK_K)] if DG_LOGDK_FEAT else []))
                        f2 = (np.stack(cols, axis=1) if cols
                              else np.zeros((len(sel), 0))).astype(np.float32)
                        orig = sel.astype(np.int64)
                        if real < N:
                            pad = np.random.choice(real, N - real)
                            pts_blk = np.concatenate([pts_blk, pts_blk[pad]], axis=0)
                            f2 = np.concatenate([f2, f2[pad]], axis=0)
                            orig = np.concatenate([orig, np.full(N - real, -1, np.int64)])
                        perm = np.random.permutation(N)
                        pts_blk, f2, orig = pts_blk[perm], f2[perm], orig[perm]
                        pc_c = (pts_blk - xyz[seed]).astype(np.float32)
                        pend_items.append((pc_c, f2, np.zeros(N, np.int64),
                                           np.arange(N, dtype=np.int32),
                                           np.array([0], np.int32)))
                        pend_blocks.append(orig)
                        if len(pend_items) == VAL_BATCH:
                            flush()
                    flush()
                pred = votes.argmax(1)
                name = os.path.splitext(os.path.basename(ds.files[i]))[0]
                raw_src = name2src.get(name)
                got = votes.sum(1) > 0
                if raw_src is None:
                    print(f"  [{label}] skip {name}: no raw source found", flush=True)
                    n_skipped += 1; continue
                if not got.any():
                    print(f"  [{label}] skip {name}: no votes", flush=True)
                    n_skipped += 1; continue
                try:
                    raw_xyz, raw_lab = tc.load_xyz_label(raw_src)
                except tc.NonFiniteXYZ:
                    raise
                except Exception as ex:
                    print(f"  [{label}] skip {name}: raw reload failed: {ex}", flush=True)
                    n_skipped += 1; continue
                i_, u_, g_, c_, t_ = tc.score_raw_from_voxels(
                    xyz[got], pred[got], raw_xyz, raw_lab, NUM_CLASSES)
                correct += c_; total += t_
                t_inter += i_; t_union += u_; t_gt += g_
                n_scenes += 1
        return tc.eval_metrics(
            t_inter, t_union, t_gt, correct, total, CLASS_NAMES, t_test,
            n_scenes, label,
            extra={"skipped_scenes": n_skipped,
                   "scored_on": "raw_points",
                   "full_coverage": True,
                   "reprojection": "nearest_subsampled_point_to_raw"})

    val_src = {n: p for n, p in val_list}

    best = tc.BestCheckpoint(run_dir, VAL_RANK)
    tc.proxy_guard(run_dir, proxy_rep, tc.PROXY_PROTOCOL_SPHERES, CLASS_NAMES,
                   VAL_RANK)
    tc.write_run_manifest(run_dir, "randlanet", dataset)

    SLOTS = cfg.val_steps * VAL_BATCH
    inv = proxy_rep["inventory"]
    pidx = [val_ds.files.index(p) for p in proxy_files]
    cnt = np.stack([np.bincount(lab[(lab >= 0) & (lab < NUM_CLASSES)],
                                minlength=NUM_CLASSES)
                    for *_, lab in (val_ds.scenes[i] for i in pidx)])
    tot = cnt.sum(0)
    n_slot = {c: PROXY_ANCHORS for c in inv}
    anchors = []
    for c in sorted(inv, key=lambda c: (int(tot[c]), CLASS_NAMES[c])):
        cand = sorted((r for r in range(len(pidx)) if cnt[r, c] > 0),
                      key=lambda r: (-int(cnt[r, c]),
                                     os.path.basename(val_ds.files[pidx[r]])))
        for j, r in enumerate(cand):
            n = n_slot[c] // len(cand) + (j < n_slot[c] % len(cand))
            if not n:
                continue
            pts = np.flatnonzero(val_ds.scenes[pidx[r]][4] == c)
            anchors += [(pidx[r], int(p), c)
                        for p in pts[::max(1, len(pts) // n)][:n]]
    anchors = anchors[:SLOTS]
    print(f"  proxy val: {len(anchors)}/{SLOTS} class-anchored spheres, "
          f"{SLOTS - len(anchors)} from the scene rotation", flush=True)

    def _proxy_batches(rng):
        for s in range(cfg.val_steps):
            items = []
            for b in range(VAL_BATCH):
                slot = s * VAL_BATCH + b
                if slot < len(anchors):
                    ci, pick, c = anchors[slot]
                    npts = len(val_ds.scenes[ci][0])
                    if npts < 64 or pick >= npts:
                        raise RuntimeError(
                            f"proxy val scene {os.path.basename(val_ds.files[ci])} "
                            f"(curated for {CLASS_NAMES[c]}) has {npts} points at "
                            f"batch time: delete {PREP_DIR} and re-run the prep")
                else:
                    ci = slot % len(val_ds.files)
                    pick = int(rng.randint(len(val_ds.scenes[ci][0])))
                items.append(val_ds.sample_sphere(ci, pick, rng=rng))
            batch = _to_device(collate_fn(items))
            yield batch, batch["labels"]

    def run_eval(ep, write_json=False):
        if not write_json:
            net.eval()
            # no AdaBN here: it would mutate BN running stats mid-training
            m = (evaluate(val_ds, val_src, f"eval@ep{ep}") if VAL_FULL else
                 tc.proxy_val(
                     _proxy_batches(np.random.RandomState(20260724)),
                     lambda b: net(b)["logits"].transpose(1, 2).reshape(-1, NUM_CLASSES),
                     NUM_CLASSES, CLASS_NAMES, f"eval@ep{ep}",
                     cfg.val_steps, tc.PROXY_PROTOCOL_SPHERES, inventory=inv))
            net.train()
            # weights before the csv row: a kill between them must not seed a phantom best that final_model.pth can never match
            if best.update(m):
                tc.atomic_torch_save({"model": net.state_dict(), "epoch": ep},
                                     best.final)
            tc.append_val_row(val_csv, ep, m, CLASS_NAMES)
            return m
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
        # deliberately no best.update: in proxy mode these numbers are another scale, and in full mode AdaBN above has already moved the weights
        swapped = os.path.exists(best.final)
        if swapped:
            live_state = {k: v.clone() for k, v in net.state_dict().items()}
            net.load_state_dict(torch.load(best.final, map_location=device,
                                           weights_only=True)["model"])
            net.eval()
        m_test = evaluate(test_ds, {n: p for n, p in test_list}, "test")
        if swapped:
            net.load_state_dict(live_state)
        with open(f"{run_dir}/test_metrics.json", "w", encoding="utf-8") as fj:
            json.dump({"val": m, "test": m_test,
                       "val_scenes": [n for n, _ in val_list],
                       "test_scenes": [n for n, _ in test_list]}, fj, indent=2)
        net.train()
        return m

    t_run = time.time()
    AMP = os.environ.get("TT_AMP") == "1"
    print(f"  starting at epoch {start_epoch}, up to {N_EPOCHS}, "
          f"{cfg.train_steps} steps/epoch"
          f"{' [bf16 autocast]' if AMP else ''}", flush=True)
    LOG_EVERY = 20
    ep = N_EPOCHS - 1
    for ep in range(start_epoch, N_EPOCHS):
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
        cur_lr = opt.param_groups[0]["lr"]
        sched.step()
        sec_per_iter = (time.time() - t_ep) / max(n_steps, 1)
        sec_per_epoch = time.time() - t_ep
        train_acc = correct / max(total, 1)
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                ep, f"{ep_loss/max(n_steps,1):.4f}", f"{train_acc:.4f}",
                f"{mean_iou:.4f}", f"{cur_lr:.6e}", f"{sec_per_iter:.4f}",
                f"{sec_per_epoch:.2f}", f"{gpu_mem:.1f}",
            ])
        print(f"  ep {ep:3d}: loss={ep_loss/max(n_steps,1):.4f} "
              f"acc={train_acc:.4f} miou={mean_iou:.4f} "
              f"s/iter={sec_per_iter:.3f} s/ep={sec_per_epoch:.1f}", flush=True)
        if (ep + 1) % 5 == 0:
            tc.atomic_torch_save({"model": net.state_dict(),
                                  "optim": opt.state_dict(), "epoch": ep},
                                 f"{run_dir}/checkpoints/ep{ep:03d}.pth")
        stop = tc.stop_requested(ep)
        if (ep + 1) % VAL_EVERY == 0 and ep != N_EPOCHS - 1 and not stop:
            run_eval(ep)
        if stop:
            break

    print("  final evaluation (val + test)…", flush=True)
    run_eval(ep, write_json=True)
    best.finalize(lambda p: tc.atomic_torch_save(
        {"model": net.state_dict(), "epoch": ep}, p))
    print(f"  total wall-clock {(time.time() - t_run)/3600:.2f} h")

    open(f"{run_dir}/DONE", "w").close()
    print(f"  run complete -> {run_id}", flush=True)


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
