"""Run-artifact figures, drawn into Figure() directly (no pyplot — works headless
and embedded). Sources: val_metrics.csv, metrics.csv, test_metrics.json,
run.json (run_config.json on legacy runs).
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
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols: dict[str, list] = {n: [] for n in (reader.fieldnames or [])}
        for row in reader:
            for n in cols:
                cols[n].append(_f(row[n]))
    return cols


def read_config(run_dir: Path) -> dict:
    # run.json is the single run record; run_config.json = legacy runs.
    for fn in ("run.json", "run_config.json"):
        p = Path(run_dir) / fn
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
    return {}


def read_test_metrics(run_dir: Path) -> dict:
    p = Path(run_dir) / "test_metrics.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def series(run_dir: Path, metric: str) -> tuple[list, list]:
    """(epochs, values) for `metric`, blanks dropped. Sourced from val_metrics.csv
    (the periodic val pass) when the column lives there, else metrics.csv (the
    per-epoch training curves)."""
    for fname in ("val_metrics.csv", "metrics.csv"):
        cols = read_csv_columns(Path(run_dir) / fname)
        if metric in cols and cols.get("epoch"):
            xs, ys = [], []
            for e, v in zip(cols["epoch"], cols[metric]):
                if e is not None and v is not None:
                    xs.append(e)
                    ys.append(v)
            if xs:
                return xs, ys
    return [], []


val_series = series   # back-compat alias


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
                              for fn in ("val_metrics.csv", "run.json", "run_config.json"))


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
        else:                                  # one more level (backbone subdirs)
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


# --------------------------------------------------------------------------- multi

def multi_run_figure(run_dirs, metric: str = "val_miou", *, show_runs: bool = True,
                     show_avg: bool = True, fig: Figure | None = None) -> Figure:
    """Overlay `metric` vs epoch for each run; optional mean ± std band across runs.

    The average is computed per epoch over whichever runs reported that epoch
    (val passes share a stride, so curves line up); a ±1 std band shows spread.
    """
    fig = fig or Figure(figsize=(9, 5.5))
    fig.clear()
    ax = fig.add_subplot(111)

    series = []   # (label, {epoch: value})
    for d in run_dirs:
        xs, ys = val_series(d, metric)
        if xs:
            series.append((run_label(d), dict(zip(xs, ys))))
            if show_runs:
                ax.plot(xs, ys, marker="o", ms=3, lw=1.3,
                        alpha=0.7 if (show_avg and len(run_dirs) > 1) else 1.0,
                        label=run_label(d))

    if show_avg and len(series) >= 2:
        epochs = sorted({e for _, m in series for e in m})
        xs2, mean, std = [], [], []
        for e in epochs:
            vals = [m[e] for _, m in series if e in m]
            xs2.append(e)
            mean.append(float(np.mean(vals)))
            std.append(float(np.std(vals)))
        xs2, mean, std = np.array(xs2), np.array(mean), np.array(std)
        ax.plot(xs2, mean, color="black", lw=2.6, zorder=5,
                label=f"average (n={len(series)})")
        ax.fill_between(xs2, mean - std, mean + std, color="black", alpha=0.12,
                        zorder=1, label="±1 std")

    if not series:
        ax.text(0.5, 0.5, "No val_metrics.csv in the selected run(s).",
                ha="center", va="center", transform=ax.transAxes, color="#888")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"{metric_label(metric)} vs epoch")
    ax.grid(alpha=0.3)
    if series:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- single

def single_run_figure(run_dir: Path, fig: Figure | None = None) -> Figure:
    """Compact 2x2 dashboard for one run, focused on the val curves the periodic
    val pass records (the live train metrics are already on the Train page)."""
    run_dir = Path(run_dir)
    fig = fig or Figure(figsize=(12, 8))
    fig.clear()
    cfg = read_config(run_dir)
    test = read_test_metrics(run_dir)
    axes = fig.subplots(2, 2)
    title = f"{cfg.get('backbone', 'model')}  |  {cfg.get('dataset', '')}  |  {run_dir.name}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # 1) val mIoU + val accuracy, with final val/test references
    ax = axes[0, 0]
    for metric, color in (("val_miou", "C0"), ("val_acc", "C1")):
        xs, ys = val_series(run_dir, metric)
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, color=color, label=metric_label(metric))
    for split, color in (("val", "C2"), ("test", "C3")):
        miou = (test.get(split) or {}).get("overall_mIoU")
        if miou is not None:
            ax.axhline(miou, color=color, ls=":", lw=1.5, label=f"{split} final mIoU={miou:.3f}")
    ax.set(title="Validation mIoU / accuracy", xlabel="epoch", ylim=(0, 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 2) per-class val IoU over epochs
    ax = axes[0, 1]
    cols = read_csv_columns(run_dir / "val_metrics.csv")
    epochs = cols.get("epoch") or []
    plotted = False
    for name in [c for c in cols if c.startswith("iou_")]:
        xs = [e for e, v in zip(epochs, cols[name]) if e is not None and v is not None]
        ys = [v for e, v in zip(epochs, cols[name]) if e is not None and v is not None]
        if xs:
            ax.plot(xs, ys, marker=".", ms=4, label=name[4:])
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no per-class val IoU", ha="center", va="center",
                transform=ax.transAxes, color="#888")
    ax.set(title="Per-class validation IoU", xlabel="epoch", ylim=(0, 1))
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=8, ncol=2)

    # 3) training-health: train loss + train mIoU
    ax = axes[1, 0]
    m = read_csv_columns(run_dir / "metrics.csv")
    me = m.get("epoch") or []
    if m.get("train_loss"):
        ax.plot([e for e in me], m["train_loss"], color="C0", label="train loss")
    ax.set(title="Training loss / mIoU", xlabel="epoch")
    ax.grid(alpha=0.3)
    if m.get("train_iou"):
        ax2 = ax.twinx()
        ax2.plot([e for e in me], m["train_iou"], color="C4", label="train mIoU")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("train mIoU", color="C4")
    ax.legend(fontsize=8, loc="center right")

    # 4) final per-class IoU bars (val + test)
    ax = axes[1, 1]
    ref = test.get("test") or test.get("val") or {}
    classes = list((ref.get("per_class_iou") or {}).keys())
    if classes:
        idx = np.arange(len(classes))
        w = 0.4
        val_pc = (test.get("val") or {}).get("per_class_iou", {})
        test_pc = (test.get("test") or {}).get("per_class_iou", {})
        if val_pc:
            ax.bar(idx - w / 2, [val_pc.get(c, 0) for c in classes], w, label="val", color="C2")
        if test_pc:
            ax.bar(idx + w / 2, [test_pc.get(c, 0) for c in classes], w, label="test", color="C3")
        ax.set_xticks(idx)
        ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set(title="Final per-class IoU", ylim=(0, 1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.set_title("Final per-class IoU (no test_metrics.json)")
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
