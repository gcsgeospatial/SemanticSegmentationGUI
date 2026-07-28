"""Run-artifact figures, drawn into Figure() directly (no pyplot - works headless
and embedded). Sources: val_metrics.csv, metrics.csv, test_metrics.json, run.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

METRIC_LABELS = {
    "val_miou": "validation mIoU", "val_acc": "validation accuracy",
    "train_loss": "train loss", "val_loss": "val loss",
    "train_acc": "train accuracy", "train_iou": "train mIoU", "val_iou": "val mIoU",
    "sec_per_epoch": "sec / epoch", "sec_per_iter": "sec / iter",
    "gpu_mem_mb": "GPU memory (MB)", "lr": "learning rate",
}


def metric_label(metric: str) -> str:
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]
    if metric.startswith("iou_"):
        return f"{metric[4:]} IoU"
    return metric


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_csv_columns(path: Path) -> dict[str, list]:
    """CSV -> {column: [floats, None for blanks]}; {} if the file is missing."""
    path = Path(path)
    if not path.exists():
        return {}
    # utf-8 matches the trainers' writers; errors="replace" tolerates legacy cp1252-written runs (only class-name header cells are ever non-ASCII)
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        cols: dict[str, list] = {n: [] for n in (reader.fieldnames or [])}
        for row in reader:
            for n in cols:
                # protocol stays a string: it selects rows, it isn't plotted
                cols[n].append(row[n] if n == "protocol" else _f(row[n]))
    return cols


def read_config(run_dir: Path) -> dict:
    p = Path(run_dir) / "run.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_test_metrics(run_dir: Path) -> dict:
    p = Path(run_dir) / "test_metrics.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _rows(cols: dict, metric: str, proxy: bool) -> tuple[list, list]:
    """(epochs, values) for the rows of one protocol. A blank/absent protocol
    (legacy runs, metrics.csv) counts as full."""
    prot = cols.get("protocol") or [None] * len(cols.get("epoch") or [])
    xy = [(e, v) for e, v, p in zip(cols["epoch"], cols[metric], prot)
          if e is not None and v is not None and (p == "proxy") == proxy]
    return [e for e, _ in xy], [v for _, v in xy]


def series(run_dir: Path, metric: str) -> tuple[list, list]:
    """(epochs, values) for `metric`, blanks dropped. Sourced from val_metrics.csv
    (the periodic val pass) when the column lives there, else metrics.csv (the
    per-epoch training curves). Proxy rows are the checkpoint-ranking curve; the
    final full raw-scored row is on another scale and never joins it."""
    for fname in ("val_metrics.csv", "metrics.csv"):
        cols = read_csv_columns(Path(run_dir) / fname)
        if metric in cols and cols.get("epoch"):
            xs, ys = _rows(cols, metric, proxy=True)
            if not xs:      # legacy / EVAL_ONLY runs: full-protocol rows only
                xs, ys = _rows(cols, metric, proxy=False)
            if xs:
                return xs, ys
    return [], []


def _has_data(cols: dict, name: str) -> bool:
    return any(v is not None for v in cols.get(name, []))


def available_metrics(run_dir: Path) -> list[str]:
    """Plottable metric keys for this run: validation curves (val_metrics.csv)
    first, then any populated per-epoch column from metrics.csv."""
    out: list[str] = []
    vcols = read_csv_columns(Path(run_dir) / "val_metrics.csv")
    out += [m for m in ("val_miou", "val_acc") if _has_data(vcols, m)]
    out += [c for c in vcols if c.startswith("iou_") and _has_data(vcols, c)]
    mcols = read_csv_columns(Path(run_dir) / "metrics.csv")
    out += [c for c in mcols if c != "epoch" and c not in out and _has_data(mcols, c)]
    return out


def is_run_dir(d: Path) -> bool:
    d = Path(d)
    return d.is_dir() and any((d / fn).exists()
                              for fn in ("val_metrics.csv", "run.json"))


def discover_runs(root: Path) -> list[Path]:
    """Run dirs at `root` or one level below it (the layout the GUI downloads into:
    <runs>/<backbone>/<run_id>/). Sorted newest-name-first, de-duplicated."""
    root = Path(root)
    if not root.exists():
        return []
    found: set[Path] = set()
    if is_run_dir(root):
        found.add(root)
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if is_run_dir(child):
            found.add(child)
        else:
            for grand in child.iterdir():
                if is_run_dir(grand):
                    found.add(grand)
    return sorted(found, key=lambda p: p.name, reverse=True)


def run_label(run_dir: Path) -> str:
    """Short legend label: backbone + the distinctive tail of the run id."""
    cfg = read_config(run_dir)
    name = Path(run_dir).name
    bb = cfg.get("backbone", "")
    return f"{name}" + (f"  [{bb}]" if bb else "")


# CVD-validated fixed hue order (dataviz six-checks, light surface); runs beyond
# the named hues fold to gray so hues are never cycled or repainted.
RUN_COLORS = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00", "#CC79A7"]
FOLD_COLOR = "#9a9a9a"


def run_colors(run_dirs) -> dict[str, str]:
    """Selection-order -> fixed hue, stable across every view in one redraw."""
    return {str(d): (RUN_COLORS[i] if i < len(RUN_COLORS) else FOLD_COLOR)
            for i, d in enumerate(run_dirs)}


def full_points(run_dir: Path, metric: str) -> tuple[list, list]:
    """(epochs, values) of raw-scored full-eval rows - the honest numbers, on a
    different scale than the proxy curve, so they draw as distinct star marks."""
    cols = read_csv_columns(Path(run_dir) / "val_metrics.csv")
    if metric in cols and cols.get("epoch"):
        return _rows(cols, metric, proxy=False)
    return [], []


def final_scores(run_dir: Path) -> dict:
    """{'val'|'test': {'acc','miou','per_class'}} from test_metrics.json."""
    tm = read_test_metrics(run_dir)
    out = {}
    for split in ("val", "test"):
        m = tm.get(split) or {}
        out[split] = {"acc": m.get("overall_acc"), "miou": m.get("overall_mIoU"),
                      "per_class": m.get("per_class_iou") or {}}
    return out


def _no_data(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color="#888", fontsize=9)


def multi_run_figure(run_dirs, metric: str = "val_miou", *, show_runs: bool = True,
                     show_avg: bool = True, fig: Figure | None = None) -> Figure:
    """Overlay `metric` vs epoch per run (entity-stable colors), full-eval rows
    as stars, optional mean ± std band across runs.

    The average is computed per epoch over whichever runs reported that epoch
    (val passes share a stride, so curves line up); a ±1 std band shows spread.
    """
    fig = fig or Figure(figsize=(9, 5.5))
    fig.clear()
    ax = fig.add_subplot(111)
    colors = run_colors(run_dirs)

    curves, starred = [], False
    for d in run_dirs:
        xs, ys = series(d, metric)
        if not xs:
            continue
        c = colors[str(d)]
        curves.append((run_label(d), dict(zip(xs, ys))))
        if show_runs:
            ax.plot(xs, ys, marker="o", ms=3.5, lw=2, color=c,
                    alpha=0.75 if (show_avg and len(run_dirs) > 1) else 1.0,
                    label=run_label(d))
            fx, fy = full_points(d, metric)
            if fx:
                ax.plot(fx, fy, "*", ms=13, color=c, mec="white", mew=0.8,
                        zorder=6, label="full eval (raw)" if not starred else None)
                starred = True

    if show_avg and len(curves) >= 2:
        epochs = sorted({e for _, m in curves for e in m})
        xs2 = np.array(epochs)
        mean = np.array([float(np.mean([m[e] for _, m in curves if e in m]))
                         for e in epochs])
        std = np.array([float(np.std([m[e] for _, m in curves if e in m]))
                        for e in epochs])
        ax.plot(xs2, mean, color="black", lw=2.6, zorder=5,
                label=f"average (n={len(curves)})")
        ax.fill_between(xs2, mean - std, mean + std, color="black", alpha=0.12,
                        zorder=1, label="±1 std")

    if not curves:
        _no_data(ax, "No val_metrics.csv in the selected run(s).")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"{metric_label(metric)} vs epoch")
    ax.grid(alpha=0.25)
    if curves:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


_DASH_PANELS = (("val_miou", "validation mIoU"), ("train_loss", "train loss"),
                ("lr", "learning rate"), ("sec_per_epoch", "sec / epoch"))


def dashboard_figure(run_dirs, fig: Figure | None = None) -> Figure:
    """2x2 training-dynamics small multiples: val mIoU (+ full-eval stars),
    train loss, lr (log), epoch time. One axis per measure - never dual-axis."""
    fig = fig or Figure(figsize=(9, 5.5))
    fig.clear()
    colors = run_colors(run_dirs)
    axes = fig.subplots(2, 2).ravel()
    handles = {}
    for ax, (metric, title) in zip(axes, _DASH_PANELS):
        drew = False
        for d in run_dirs:
            xs, ys = series(d, metric)
            if not xs:
                continue
            c = colors[str(d)]
            (line,) = ax.plot(xs, ys, lw=2, ms=3, marker="o", color=c)
            handles.setdefault(run_label(d), line)
            drew = True
            if metric == "val_miou":
                fx, fy = full_points(d, metric)
                if fx:
                    ax.plot(fx, fy, "*", ms=12, color=c, mec="white",
                            mew=0.8, zorder=6)
        if metric == "lr" and drew:
            ax.set_yscale("log")
        if not drew:
            _no_data(ax, f"no {title} data")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    axes[2].set_xlabel("epoch", fontsize=9)
    axes[3].set_xlabel("epoch", fontsize=9)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="lower center",
                   fontsize=8, ncol=min(len(handles), 3), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def class_iou_figure(run_dirs, fig: Figure | None = None) -> Figure:
    """Per-class final IoU (test split, val fallback) as horizontal bars; up to
    4 runs grouped side by side, classes sorted by the first run's score."""
    fig = fig or Figure(figsize=(9, 5.5))
    fig.clear()
    ax = fig.add_subplot(111)
    colors = run_colors(run_dirs)
    shown = list(run_dirs)[:4]

    per_run = []
    for d in shown:
        fs = final_scores(d)
        pc = fs["test"]["per_class"] or fs["val"]["per_class"]
        if pc:
            per_run.append((d, pc))
    if not per_run:
        _no_data(ax, "No test_metrics.json in the selected run(s) - "
                     "finish (or gracefully stop) a run to get final scores.")
        fig.tight_layout()
        return fig

    classes = sorted({c for _, pc in per_run for c in pc},
                     key=lambda c: -(per_run[0][1].get(c) or 0))
    ypos = np.arange(len(classes), dtype=float)
    h = 0.8 / len(per_run)
    for i, (d, pc) in enumerate(per_run):
        vals = [pc.get(c) or 0.0 for c in classes]
        c = colors[str(d)]
        bars = ax.barh(ypos - 0.4 + h * (i + 0.5), vals, height=h * 0.9,
                       color=c, label=run_label(d))
        if len(per_run) == 1:
            for b, v in zip(bars, vals):
                ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                        va="center", fontsize=8, color="#444")
    ax.set_yticks(ypos, classes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12 if len(per_run) == 1 else 1.02)
    ax.set_xlabel("IoU")
    dropped = len(list(run_dirs)) - len(shown)
    ax.set_title("Per-class IoU (final eval)"
                 + (f"  ·  first 4 of {len(list(run_dirs))} selected" if dropped > 0 else ""))
    ax.grid(alpha=0.25, axis="x")
    if len(per_run) > 1:
        # below the axes: with grouped bars no corner is reliably empty
        fig.legend(fontsize=8, loc="lower center", ncol=min(len(per_run), 3),
                   frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    else:
        fig.tight_layout()
    return fig


def leaderboard_figure(run_dirs, fig: Figure | None = None) -> Figure:
    """Final test mIoU per run (val fallback, marked), sorted best-first,
    direct-labeled; bars keep each run's selection color."""
    fig = fig or Figure(figsize=(9, 5.5))
    fig.clear()
    ax = fig.add_subplot(111)
    colors = run_colors(run_dirs)

    rows = []
    for d in run_dirs:
        fs = final_scores(d)
        if fs["test"]["miou"] is not None:
            rows.append((run_label(d), fs["test"]["miou"], colors[str(d)], ""))
        elif fs["val"]["miou"] is not None:
            rows.append((run_label(d), fs["val"]["miou"], colors[str(d)], " (val)"))
    if not rows:
        _no_data(ax, "No test_metrics.json in the selected run(s) - "
                     "finish (or gracefully stop) a run to get final scores.")
        fig.tight_layout()
        return fig

    rows.sort(key=lambda r: -r[1])
    ypos = np.arange(len(rows))
    ax.barh(ypos, [r[1] for r in rows], height=0.6, color=[r[2] for r in rows])
    ax.set_yticks(ypos, [r[0] + r[3] for r in rows])
    ax.invert_yaxis()
    for y, (_, v, _, _) in zip(ypos, rows):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9, color="#444")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("final mIoU (test; '(val)' = no test split)")
    ax.set_title("Run leaderboard")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    return fig
