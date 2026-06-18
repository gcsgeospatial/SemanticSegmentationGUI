"""Train page: dataset + backbone + recommended params -> modal run, live logs."""

from __future__ import annotations

import json
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import analysis, appstate, modal_cli, prep, ui
from ..backbones import BACKBONES, GPU_CHOICES
from ..jobs import FuncWorker, JobRunner, LogParser


class TrainPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.runner = JobRunner(self)
        self.log_runner = JobRunner(self)   # for re-attaching to detached runs
        self.prep_worker = FuncWorker(self)  # local tiling/subsampling
        self.prep_uploader = JobRunner(self)
        self.parser = LogParser(self)
        self._param_widgets: dict[str, QWidget] = {}
        self._meta: dict | None = None
        self._last_run_id: str | None = None
        self._pending: dict | None = None   # launch args while prep/upload run

        root = QVBoxLayout(self)
        title = QLabel("Train")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        sub = QLabel("Pick a dataset and a model. Parameters are pre-filled from the dataset's "
                     "density analysis — edit anything before launching. Runs execute on Modal; "
                     "detached runs keep going if you close this app.")
        sub.setWordWrap(True)
        sub.setObjectName("pageSub")
        root.addWidget(sub)

        form_box = QGroupBox("Job")
        form = QFormLayout(form_box)
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_change)
        form.addRow("Dataset", self.dataset_combo)
        self.backbone_combo = QComboBox()  # populated per-dataset in _reload_backbones
        self.backbone_combo.currentIndexChanged.connect(self._rebuild_params)
        form.addRow("Model", self.backbone_combo)
        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(GPU_CHOICES)
        self.gpu_combo.setCurrentText("A100")
        form.addRow("GPU", self.gpu_combo)
        opts_row = QHBoxLayout()
        self.detach_chk = QCheckBox("Detached (survives closing the app)")
        self.detach_chk.setChecked(True)
        self.smoke_chk = QCheckBox("Smoke run (2 epochs × 50 steps on A10G)")
        opts_row.addWidget(self.detach_chk)
        opts_row.addWidget(self.smoke_chk)
        opts_row.addStretch()
        form.addRow("Options", _wrap(opts_row))
        self.prep_chk = QCheckBox("Prep tiles locally + upload (no Modal CPU time "
                                  "spent on preprocessing)")
        self.prep_chk.setChecked(True)
        form.addRow("", self.prep_chk)

        self.params_box = QGroupBox("Parameters (recommended values pre-filled)")
        self.params_form = QFormLayout(self.params_box)
        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color: #b25f00;")

        run_row = QHBoxLayout()
        self.launch_btn = QPushButton("Launch training")
        self.launch_btn.setObjectName("primary")
        self.launch_btn.clicked.connect(self._launch)
        run_row.addWidget(self.launch_btn)
        self.attach_btn = QPushButton("Re-attach to logs")
        self.attach_btn.clicked.connect(self._reattach)
        run_row.addWidget(self.attach_btn)
        self.stop_btn = QPushButton("Stop local process")
        self.stop_btn.clicked.connect(self._stop)
        run_row.addWidget(self.stop_btn)
        run_row.addStretch()

        # Config column (scrolls when squeezed so the log can take the room).
        config_col = QVBoxLayout()
        config_col.addWidget(form_box)
        config_col.addWidget(self.params_box)
        config_col.addWidget(self.warn_label)
        config_col.addLayout(run_row)
        config_col.addStretch()

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("log")
        self.log.setPlaceholderText("Modal logs appear here…")

        metrics_col = QVBoxLayout()
        metrics_col.addWidget(QLabel("Live epoch metrics"))
        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(["Epoch", "Loss", "Acc", "mIoU"])
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        metrics_col.addWidget(self.metrics_table, 1)

        # Drag the handles: config | (log | metrics) are all resizable.
        body = ui.hsplit(self.log, ui.wrap(metrics_col), sizes=[680, 340])
        root.addWidget(ui.vsplit(ui.scrollable(ui.wrap(config_col)), body,
                                 sizes=[400, 360]), 1)

        self.runner.output.connect(self._on_output)
        self.runner.finished.connect(self._on_finished)
        self.log_runner.output.connect(self._on_output)
        self.prep_worker.output.connect(self._append)
        self.prep_worker.done.connect(self._on_prepped)
        self.prep_worker.error.connect(self._on_prep_error)
        self.prep_uploader.output.connect(lambda s: self._append(s, newline=False))
        self.prep_uploader.finished.connect(self._on_prep_uploaded)
        self.parser.epoch.connect(self._on_epoch)
        self.parser.run_id.connect(self._on_run_id)

        self.reload_datasets()
        self._rebuild_params()

    # ------------------------------------------------------------- datasets
    def reload_datasets(self):
        current = self.dataset_combo.currentText()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for name in sorted(appstate.known_datasets()):
            self.dataset_combo.addItem(name)
        self.dataset_combo.blockSignals(False)
        if current:
            i = self.dataset_combo.findText(current)
            if i >= 0:
                self.dataset_combo.setCurrentIndex(i)
        self._on_dataset_change()

    def _on_dataset_change(self):
        self._meta = None
        name = self.dataset_combo.currentText()
        info = appstate.known_datasets().get(name, {})
        meta_path = info.get("meta_path", "")
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
        self._reload_backbones()

    def _reload_backbones(self):
        """Populate the model dropdown. Built-in IEEE datasets restrict it to the
        scripts whose no-`--dataset` default trains on that data."""
        allowed = appstate.known_datasets().get(
            self.dataset_combo.currentText(), {}).get("backbones")
        prev = self.backbone_combo.currentData()
        self.backbone_combo.blockSignals(True)
        self.backbone_combo.clear()
        for key, b in BACKBONES.items():
            if allowed and key not in allowed:
                continue
            self.backbone_combo.addItem(
                b.label + ("" if b.ready else "  (script not wired yet)"), key)
        i = self.backbone_combo.findData(prev)
        if i >= 0:
            self.backbone_combo.setCurrentIndex(i)
        self.backbone_combo.blockSignals(False)
        self._rebuild_params()

    # ------------------------------------------------------------- params form
    def _backbone(self):
        return BACKBONES[self.backbone_combo.currentData()]

    def _rebuild_params(self):
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_widgets.clear()
        b = self._backbone()
        recs = (self._meta or {}).get("recommendations", {}).get(b.key, {})
        for spec in b.params:
            value = recs.get(spec.recommend_key, spec.default) if spec.recommend_key else spec.default
            if spec.kind == "float":
                w = QDoubleSpinBox()
                w.setDecimals(spec.decimals)
                w.setSingleStep(spec.step)
                w.setRange(spec.lo, spec.hi)
                w.setValue(float(value))
            else:
                w = QSpinBox()
                w.setRange(int(spec.lo), int(spec.hi))
                w.setValue(int(value))
            label = spec.label + ("  ★" if spec.recommend_key and spec.recommend_key in recs else "")
            self.params_form.addRow(label, w)
            self._param_widgets[spec.flag] = w
        if self._meta:
            warns = analysis.warnings_for(self._meta)
            self.warn_label.setText("\n".join("⚠ " + w for w in warns))
        else:
            self.warn_label.setText("")

    # ------------------------------------------------------------- launch
    def _launch(self):
        b = self._backbone()
        if not b.ready:
            self._append(f"{b.label} isn't wired for CLI args yet — pick a ready model.")
            return
        name = self.dataset_combo.currentText()
        if not name:
            self._append("Create + upload a dataset on the Datasets page first.")
            return
        if self.runner.running or self.prep_worker.running or self.prep_uploader.running:
            self._append("A local process is already running.")
            return

        info = appstate.known_datasets().get(name, {})
        # Built-in IEEE datasets run the script's no-`--dataset` default (real data
        # already on the ieee-data volume — real HAG for the HAG variants).
        flags = {} if info.get("builtin") else {"dataset": name}
        for spec in b.params:
            w = self._param_widgets[spec.flag]
            flags[spec.flag] = w.value()
        gpu = self.gpu_combo.currentText()
        if self.smoke_chk.isChecked():
            flags["epochs"] = 2
            flags["steps-per-epoch"] = 50
            gpu = "A10G"

        self.log.clear()
        self.metrics_table.setRowCount(0)
        self._last_run_id = None
        self.launch_btn.setEnabled(False)
        self._pending = {"backbone": b, "flags": flags, "gpu": gpu, "dataset": name}

        # Optional local prep: tile/subsample here and upload the cache so the
        # GPU container finds everything already preprocessed and skips it.
        # Built-ins already have their prep on the ieee-data volume — skip it.
        if self.prep_chk.isChecked() and not info.get("builtin"):
            staged = appstate.known_datasets().get(name, {}).get("staged_dir", "")
            if not prep.supports_local_prep(b.key):
                self._append(f"[prep] {b.label} has no local prep path — the script "
                             f"will preprocess remotely.")
            elif not staged or not os.path.isdir(staged):
                self._append("[prep] No local staged copy of this dataset on this "
                             "machine — the script will preprocess remotely.")
            else:
                self._append(f"[prep] Building {b.label} cache locally from {staged} …")
                self.prep_worker.start(prep.prep_dataset, b.key, staged, dict(flags))
                return

        self._start_modal_run()

    def _on_prepped(self, prep_dir):
        p = self._pending
        if p is None:
            return
        tag = prep_dir.name
        remote = f"/{p['dataset']}/prep/{tag}"
        self._append(f"[prep] Uploading cache -> {modal_cli.DATASETS_VOLUME}:{remote} …")
        prog, args = modal_cli.volume_put(modal_cli.DATASETS_VOLUME, str(prep_dir), remote)
        self.prep_uploader.start(prog, args, cwd=self.repo_root)

    def _on_prep_error(self, tb: str):
        self._append(f"\n[prep] Local prep failed — falling back to remote "
                     f"preprocessing.\n{tb}")
        self._start_modal_run()

    def _on_prep_uploaded(self, code: int):
        if code != 0:
            self._append(f"[prep] Upload failed (exit {code}) — the script will "
                         f"preprocess remotely instead.")
        else:
            self._append("[prep] ✓ Cache uploaded — remote preprocessing will be skipped.")
        self._start_modal_run()

    def _start_modal_run(self):
        p, self._pending = self._pending, None
        if p is None:
            self.launch_btn.setEnabled(True)
            return
        prog, args = modal_cli.run_script(p["backbone"].script, p["flags"],
                                          detach=self.detach_chk.isChecked())
        self._append(f"\n$ modal {' '.join(args)}   [TT_GPU={p['gpu']}]\n")
        self.runner.start(prog, args, cwd=self.repo_root, extra_env={"TT_GPU": p["gpu"]})

    def _reattach(self):
        b = self._backbone()
        if self.log_runner.running:
            self.log_runner.terminate()
            return
        self._append(f"\n$ modal app logs {b.app_name}\n")
        prog, args = modal_cli.app_logs(b.app_name)
        self.log_runner.start(prog, args, cwd=self.repo_root)

    def _stop(self):
        stopped = False
        for r in (self.runner, self.log_runner):
            if r.running:
                r.terminate()
                stopped = True
        self._append("\n[stopped local process — a detached Modal run keeps going; "
                     "use `modal app stop <app>` to kill it remotely]" if stopped
                     else "\n[no local process running]")

    # ------------------------------------------------------------- stream
    def _on_output(self, text: str):
        self._append(text, newline=False)
        self.parser.feed(text)

    def _on_finished(self, code: int):
        self.launch_btn.setEnabled(True)
        if code == 0:
            extra = (f" Run id: {self._last_run_id}." if self._last_run_id else "")
            self._append(f"\n✓ Process finished.{extra} See the Runs page for artifacts.")
        else:
            self._append(f"\n✗ Process exited with code {code}.")

    def _on_epoch(self, m: dict):
        r = self.metrics_table.rowCount()
        self.metrics_table.insertRow(r)
        for col, val in enumerate((str(m["epoch"]), f"{m['loss']:.4f}",
                                   f"{m['acc']:.4f}", f"{m['miou']:.4f}")):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.metrics_table.setItem(r, col, item)
        self.metrics_table.scrollToBottom()

    def _on_run_id(self, run_id: str):
        self._last_run_id = run_id
        history = appstate.get("run_history", [])
        history.append({"run_id": run_id, "backbone": self._backbone().key,
                        "dataset": self.dataset_combo.currentText()})
        appstate.put("run_history", history[-200:])

    def _append(self, text: str, newline: bool = True):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text + ("\n" if newline else ""))
        self.log.moveCursor(QTextCursor.End)


def _wrap(layout) -> QWidget:
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w
