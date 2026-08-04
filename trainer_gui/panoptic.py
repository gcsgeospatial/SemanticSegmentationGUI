"""Training-free panoptic step: ALPINE clustering over inference predictions.

Alpine + fit_2d_box_modest are vendored from https://github.com/valeoai/Alpine
(MIT, Copyright (c) 2025 Valeo Comfort and Driving Assistance - Corentin
Sautier @ valeo.ai; box fitting extracted upstream from YurongYou/MODEST).
Paper: Sautier et al., "Clustering is back: Reaching state-of-the-art LiDAR
instance segmentation without training", arXiv:2503.13203.
Local changes: fixed the constructor's k-shadowing bug, dropped tqdm.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial import cKDTree as KDTree


def minimum_bounding_rectangle(points):
    """Smallest-area bounding rectangle of an nx2 point set (rotating calipers
    over the convex hull). Returns (4x2 corners, angle, area)."""
    pi2 = np.pi / 2.0
    hull_points = points[ConvexHull(points).vertices]
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.unique(np.abs(np.mod(angles, pi2)))
    rotations = np.vstack(
        [np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]
    ).T.reshape((-1, 2, 2))
    rot_points = np.dot(rotations, hull_points.T)
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)
    areas = (max_x - min_x) * (max_y - min_y)
    best = np.argmin(areas)
    x1, x2, y1, y2, r = max_x[best], min_x[best], max_y[best], min_y[best], rotations[best]
    rval = np.array([np.dot([x1, y2], r), np.dot([x2, y2], r),
                     np.dot([x2, y1], r), np.dot([x1, y1], r)])
    return rval, angles[best], areas[best]


def fit_2d_box_modest(ptc):
    corners, ry, _ = minimum_bounding_rectangle(ptc)
    box_length = np.linalg.norm(corners[0] - corners[1])
    box_width = np.linalg.norm(corners[0] - corners[-1])
    box_center = (corners[0] + corners[2]) / 2
    return box_center, box_length, box_width, ry


class Alpine:
    """ALPINE clustering: per thing class, connect kNN pairs closer than the
    class's box width in XY and take connected components; optionally split
    clusters whose fitted 2D box exceeds the prior (binary-searched threshold).

    thing_indexes: class indices to cluster (others keep instance 0 = stuff).
    thing_bboxes: {class index: [length_m, width_m]} standard object size.
    """

    def __init__(self, thing_indexes, thing_bboxes, k: int = 32,
                 split: bool = False, margin: float = 1.3):
        self.thing_indexes = thing_indexes
        for ci, v in thing_bboxes.items():   # upstream used `k` here, clobbering the kwarg
            if v[1] > v[0]:
                thing_bboxes[ci] = [v[1], v[0]]
        self.thing_bboxes = thing_bboxes
        self.k = k
        self.split = split
        self.margin = margin
        self.labels_ = None
        assert len(self.thing_indexes) == len(self.thing_bboxes)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = X[:, :2]
        offset = 1
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        for object_class in self.thing_indexes:
            mask = y == object_class
            if (n := mask.sum()) > 0:
                if n == 1:
                    self.labels_[mask] = offset
                    offset += 1
                    continue
                pc_mask = X[mask]
                clusters = self._clusterize(pc_mask, self.thing_bboxes[object_class][1],
                                            k=self.k)
                if self.split:
                    inst_mask = np.zeros(pc_mask.shape[0], dtype=np.int32)
                    for cid in range(clusters[0]):
                        mask_cluster = clusters[1] == cid
                        sub = self._split_cluster(pc_mask[mask_cluster],
                                                  self.thing_bboxes[object_class][1],
                                                  self.thing_bboxes[object_class],
                                                  margin=self.margin, k=self.k)
                        inst_mask[mask_cluster] = sub[1] + offset
                        offset += sub[0]
                    self.labels_[mask] = inst_mask
                else:
                    self.labels_[mask] = clusters[1] + offset
                    offset += clusters[0]
        return self

    def fit_predict(self, X: np.ndarray, y: np.ndarray):
        self.fit(X, y)
        return self.labels_

    def _clusterize(self, pc, th, k, dist=None, neighbors=None):
        pc = pc[:, :2]
        k = min((k, pc.shape[0] - 1))
        if neighbors is None:
            kdtree = KDTree(pc)
            dist, neighbors = kdtree.query(pc, k=k + 1)
            dist, neighbors = dist[:, 1:], neighbors[:, 1:]
        orig = np.vstack([np.arange(pc.shape[0]) for _ in range(neighbors.shape[1])]).T
        weights = (dist < th).astype("float")
        W = csr_matrix((weights.flatten(), (orig.flatten(), neighbors.flatten())),
                       shape=(pc.shape[0], pc.shape[0]))
        W = (W + W.T) / 2
        n, labels = connected_components(W, directed=False, return_labels=True)
        return n, labels

    def _split_cluster(self, pc, th, box_size, margin=1.3, k=32):
        if pc.shape[0] < 3:
            return 1, np.zeros(pc.shape[0], dtype=np.int32)
        try:
            _, x, y, _ = fit_2d_box_modest(pc[:, :2])
            if max(x, y) < box_size[0] * margin and min(x, y) < box_size[1] * margin:
                return 1, np.zeros(pc.shape[0], dtype=np.int32)
            dt = th / 2
            current_th = dt
            k_ = min((k, pc.shape[0] - 1))
            kdtree = KDTree(pc)
            dist, neighbors = kdtree.query(pc, k=k_ + 1)
            dist, neighbors = dist[:, 1:], neighbors[:, 1:]
            while dt > 1e-3:
                dt = dt / 2
                clusters = self._clusterize(pc, current_th, k, dist, neighbors)
                if clusters[0] == 1:
                    current_th -= dt
                elif clusters[0] == 2:
                    mask_1 = clusters[1] == 0
                    mask_2 = clusters[1] == 1
                    n, clusters[1][mask_1] = self._split_cluster(
                        pc[mask_1], current_th, box_size, margin=margin, k=k)
                    n2, cl = self._split_cluster(
                        pc[mask_2], current_th, box_size, margin=margin, k=k)
                    clusters[1][mask_2] = cl + n
                    return n + n2, clusters[1]
                else:
                    current_th += dt
        except QhullError:
            pass
        return 1, np.zeros(pc.shape[0], dtype=np.int32)


# ---------------------------------------------------------------- GUI step

def job_id_for(pred_dir: Path) -> str:
    """predictions_<job>[_m<k>|_ensemble][/predictions] -> <job>."""
    import re
    pred_dir = Path(pred_dir)
    name = pred_dir.parent.name if pred_dir.name == "predictions" else pred_dir.name
    name = re.sub(r"(_m\d+|_ensemble)$", "", name)
    return name[len("predictions_"):] if name.startswith("predictions_") else name


def class_names_for(pred_dir: Path) -> tuple[list | None, str]:
    """(class_names, source label): infer_run.json beside the npz (ensemble),
    run.json a level up (downloaded runs), else the staged infer job's
    infer_run.json (local runs). None when nothing names the classes."""
    from . import appstate
    pred_dir = Path(pred_dir)
    for p in (pred_dir / "infer_run.json", pred_dir.parent / "run.json",
              pred_dir.parent.parent / "run.json"):
        m = appstate.read_json(p)
        if m and m.get("class_names"):
            return list(m["class_names"]), p.name
    roots = [appstate.scratch_infer_dir()]
    for name in appstate.known_datasets():
        try:
            roots.append(appstate.dataset_root(name) / "infer")
        except (KeyError, OSError):
            pass
    for r in roots:
        m = appstate.read_json(r / job_id_for(pred_dir) / "infer_run.json")
        if m and m.get("class_names"):
            return list(m["class_names"]), "infer_run.json"
    return None, ""


def run_panoptic(pred_dir, things: dict, k: int = 32, split: bool = True,
                 margin: float = 1.3, progress=None) -> dict:
    """Cluster every *_pred.npz in pred_dir with ALPINE and write the result
    back as an additive 'instance' key (uint32, 0 = stuff, ids per scene).
    things = {model class index: (length_m, width_m)}. Atomic per-file rewrite
    so a crash never loses a prediction. Returns {files, instances}."""
    say = progress or (lambda s: None)
    files = sorted(Path(pred_dir).glob("*_pred.npz"))
    if not files:
        raise RuntimeError(f"no *_pred.npz files in {pred_dir} - run Inference "
                           "on this folder first, then come back")
    alp = Alpine(sorted(things), {c: [float(v[0]), float(v[1])] for c, v in things.items()},
                 k=k, split=split, margin=margin)
    total = 0
    for f in files:
        say(f"  {f.name}: clustering…")
        with np.load(f) as d:
            data = {key: d[key] for key in d.files}
        xyz = np.asarray(data["xyz"], np.float64)
        labels = alp.fit_predict(xyz[:, :2], np.asarray(data["classification"]))
        data["instance"] = labels.astype(np.uint32)
        n_inst = int(labels.max(initial=0))
        total += n_inst
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, **data)
        from .postproc import replace_retry
        replace_retry(tmp, f)
        say(f"  {f.name}: {n_inst:,} instances over {len(xyz):,} pts")
    return {"files": len(files), "instances": total}
