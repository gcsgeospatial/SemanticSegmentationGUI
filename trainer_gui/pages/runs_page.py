"""Runs page: list runs on the outputs volumes, download artifacts, show metrics
and the dashboard plot, open predictions in the viewer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel,
                               QListWidget, QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSplitter, QTabWidget, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import appstate, modal_cli, ui
from ..backbones import BACKBONES
from ..jobs import FuncWorker, JobRunner

PROJECT_DIR = str(Path(__file__).resolve().parents[2])   # trainer_gui/ project dir


class RunsPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.lister = FuncWorker(self)
        self.downloader = JobRunner(self)
        self._local_run_dir: Path | None = None

        root = QVBoxLayout(self)
        title = QLabel("Runs")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        sub = QLabel("Browse finished/ongoing runs per model, pull their artifacts down "
                     "and inspect metrics, plots and predictions.")
        sub.setWordWrap(True)
        sub.setObjectName("pageSub")
        root.addWidget(sub)

        top = QHBoxLayout()
        self.backbone_combo = QComboBox()
        for key, b in BACKBONES.items():
            self.backbone_combo.addItem(b.label, key)
        top.addWidget(QLabel("Model:"))
        top.addWidget(self.backbone_combo)
        self.refresh_btn = QPushButton("Refresh runs")
        self.refresh_btn.clicked.connect(self._refresh)
        top.addWidget(self.refresh_btn)
        self.download_btn = QPushButton("Download artifacts")
        self.download_btn.setObjectName("primary")
        self.download_btn.clicked.connect(self._download)
        top.addWidget(self.download_btn)
        top.addStretch()
        root.addLayout(top)

        list_col = QVBoxLayout()
        list_col.addWidget(QLabel("Runs on the outputs volume"))
        self.run_list = QListWidget()
        list_col.addWidget(self.run_list, 1)
        self.status = QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setObjectName("log")
        left = ui.vsplit(ui.wrap(list_col), self.status, sizes=[420, 160])

        self.tabs = QTabWidget()
        # metrics tab
        metrics_w = QWidget()
        ml = QVBoxLayout(metrics_w)
        self.summary_label = QLabel("Download a run to see its metrics.")
        self.summary_label.setWordWrap(True)
        ml.addWidget(self.summary_label)
        self.iou_table = QTableWidget(0, 4)
        self.iou_table.setHorizontalHeaderLabels(["Class", "Val IoU", "Test IoU", "Test GT points"])
        self.iou_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.iou_table.verticalHeader().setVisible(False)
        self.iou_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.iou_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        ml.addWidget(self.iou_table, 1)
        self.tabs.addTab(metrics_w, "Metrics")
        # plot tab
        self.plot_label = QLabel("The dashboard plot appears here after download.")
        self.plot_label.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(self.plot_label)
        scroll.setWidgetResizable(True)
        self.tabs.addTab(scroll, "Plot")
        # predictions tab
        pred_w = QWidget()
        pl = QVBoxLayout(pred_w)
        self.pred_list = QListWidget()
        pl.addWidget(self.pred_list, 1)
        view_btn = QPushButton("Open in 3D viewer")
        view_btn.clicked.connect(self._view_pred)
        pl.addWidget(view_btn)
        self.tabs.addTab(pred_w, "Predictions")
        self.tabs.setMinimumHeight(520)
        root.addWidget(ui.hsplit(left, self.tabs, sizes=[340, 740]), 1)

        self.downloader.output.connect(lambda s: self._say(s, newline=False))
        self.downloader.finished.connect(self._on_downloaded)
        self.lister.done.connect(self._on_listed)
        self.lister.error.connect(lambda tb: self._say(f"✗ {tb}"))

    # ------------------------------------------------------------- listing
    def _backbone(self):
        return BACKBONES[self.backbone_combo.currentData()]

    def _refresh(self):
        b = self._backbone()
        self.refresh_btn.setEnabled(False)
        self._say(f"Listing {b.outputs_volume}:runs/ …")

        def job(progress):
            return modal_cli.list_volume_entries(b.outputs_volume, "runs/")

        self.lister.start(job)

    def _on_listed(self, entries):
        self.refresh_btn.setEnabled(True)
        self.run_list.clear()
        names = sorted((Path(e.get("Filename", e.get("path", ""))).name
                        for e in entries if e), reverse=True)
        for n in names:
            if n:
                self.run_list.addItem(n)
        self._say(f"{self.run_list.count()} runs found.")

    # ------------------------------------------------------------- download
    def _download(self):
        items = self.run_list.selectedItems()
        if not items:
            self._say("Select a run first (Refresh runs if the list is empty).")
            return
        b = self._backbone()
        run_id = items[0].text()
        dest = appstate.runs_dir() / b.key
        dest.mkdir(parents=True, exist_ok=True)
        self._local_run_dir = dest / run_id
        self._say(f"Downloading runs/{run_id} -> {self._local_run_dir} …")
        self.download_btn.setEnabled(False)
        prog, args = modal_cli.volume_get(b.outputs_volume, f"runs/{run_id}", str(dest))
        self.downloader.start(prog, args, cwd=self.repo_root)

    def _on_downloaded(self, code: int):
        self.download_btn.setEnabled(True)
        if code != 0:
            self._say(f"✗ Download failed (exit {code}).")
            return
        self._say("✓ Downloaded.")
        self._show_metrics()
        self._make_plot()
        self._populate_preds()

    # ------------------------------------------------------------- metrics
    def _show_metrics(self):
        run_dir = self._local_run_dir
        tm_path = run_dir / "test_metrics.json"
        cfg_path = run_dir / "run_config.json"
        cfg = {}
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not tm_path.exists():
            self.summary_label.setText(
                f"{run_dir.name}: no test_metrics.json (run still going, or an inference run). "
                f"Backbone: {cfg.get('backbone', '?')}, mode: {cfg.get('mode', 'train')}.")
            self.iou_table.setRowCount(0)
            return
        tm = json.loads(tm_path.read_text(encoding="utf-8"))
        val, test = tm.get("val", {}), tm.get("test", {})
        self.summary_label.setText(
            f"{run_dir.name}   ·   {cfg.get('backbone', '?')} on {cfg.get('dataset', '?')}   ·   "
            f"val mIoU {val.get('overall_mIoU', 0):.3f} / acc {val.get('overall_acc', 0):.3f}   ·   "
            f"test mIoU {test.get('overall_mIoU', 0):.3f} / acc {test.get('overall_acc', 0):.3f}")
        classes = list((test.get("per_class_iou") or val.get("per_class_iou") or {}).keys())
        self.iou_table.setRowCount(len(classes))
        for r, cls in enumerate(classes):
            self.iou_table.setItem(r, 0, QTableWidgetItem(cls))
            for col, blob in ((1, val), (2, test)):
                iou = float(blob.get("per_class_iou", {}).get(cls, 0.0))
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setValue(round(iou * 100))
                bar.setFormat(f"{iou:.3f}")
                self.iou_table.setCellWidget(r, col, bar)
            gt = test.get("per_class_gt_count", {}).get(cls, "")
            item = QTableWidgetItem(f"{gt:,}" if isinstance(gt, int) else str(gt))
            item.setTextAlignment(Qt.AlignCenter)
            self.iou_table.setItem(r, 3, item)
        self.tabs.setCurrentIndex(0)

    # ------------------------------------------------------------- plot
    def _make_plot(self):
        run_dir = self._local_run_dir
        if not run_dir or not ((run_dir / "val_metrics.csv").exists()
                               or (run_dir / "metrics.csv").exists()):
            return
        try:
            from .. import plots
            png = run_dir / "metrics_plot.png"
            plots.single_run_figure(run_dir).savefig(str(png), dpi=120)
            self.plot_label.setPixmap(QPixmap(str(png)))
            self._say("✓ Plot rendered.")
        except Exception as e:  # noqa: BLE001
            self._say(f"(plot skipped: {e})")

    # ------------------------------------------------------------- predictions
    def _populate_preds(self):
        self.pred_list.clear()
        pred_dir = self._local_run_dir / "predictions"
        if not pred_dir.is_dir():
            return
        for p in sorted(pred_dir.iterdir()):
            if p.suffix.lower() in (".ply", ".npz"):
                self.pred_list.addItem(str(p))

    def _view_pred(self):
        items = self.pred_list.selectedItems()
        if not items:
            self._say("Select a prediction file first.")
            return
        QProcess.startDetached(sys.executable, ["-m", "trainer_gui.viewer", items[0].text()],
                               PROJECT_DIR)
        self._say(f"Opened viewer for {Path(items[0].text()).name}")

    # ------------------------------------------------------------- helpers
    def _say(self, text: str, newline: bool = True):
        self.status.moveCursor(QTextCursor.End)
        self.status.insertPlainText(text + ("\n" if newline else ""))
        self.status.moveCursor(QTextCursor.End)
