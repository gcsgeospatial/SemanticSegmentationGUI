"""Train page: dataset + model + params -> a training run, with live logs and
epoch metrics. Backends: local (pixi run) or Modal (cloud GPU)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QProgressBar, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import analysis, appstate, dataset, local_cli, modal_cli, theme, ui
from ..backbones import BACKBONES, GPU_CHOICES
from ..jobs import FuncWorker, JobRunner, LogParser
from ..logconsole import LogConsole

# Channels each arch CAN consume — a capability filter. "height" is removed —
# don't reintroduce it. The standard channels are pre-checked as a sensible,
# reversible default (default_feature_checks); geo/hag extras stay opt-in.
_FEAT_STANDARD = {
    "randlanet":    ["intensity", "return_number"],
    "kpconvx_cold": ["intensity", "return_number"],
    "kpconv":       ["intensity", "return_number"],
    # ptv3 family: one 3-wide color slot (rgb OR intensity), no return_number
    "ptv3":         ["intensity", "rgb"],
    "concerto":     ["intensity", "rgb"],
    "sonata":       ["intensity", "rgb"],
    "utonia":       ["intensity", "rgb"],
}

_PTV3_LIKE = ("ptv3", "concerto", "sonata", "utonia")
# these trainers' FEAT_CHANNELS spec expects x,y,z; the launcher prepends them
_XYZ_IMPLICIT = ("randlanet",) + _PTV3_LIKE


def default_feature_checks(base: str, std: list[str]) -> set[str]:
    """Channels to pre-check: the arch's standard inputs the dataset offers.
    ptv3-family has one 3-wide color slot, so collapse to intensity-first
    (rgb only when intensity is absent). geo/feat_hag extras aren't in `std`,
    so they stay unchecked."""
    if base in _PTV3_LIKE:
        for c in ("intensity", "rgb"):
            if c in std:
                return {c}
        return set()
    return set(std)

_TUNING_FLAGS = ("batch", "chunk-xy", "grid", "sub-grid")


class TrainPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.runner = JobRunner(self)
        self.pull_runner = JobRunner(self)
        self.status_worker = FuncWorker(self)
        self.modal_worker = FuncWorker(self)
        self.parser = LogParser(self)
        self._param_widgets: dict[str, QWidget] = {}
        self._meta: dict | None = None
        self._last_run_id: str | None = None
        self._out_root: str | None = None
        self._pending: dict | None = None
        self._last_statuses: dict = {}
        self._cfg_dialog: QDialog | None = None
        self._ds_ready = False
        self._built_sig: tuple | None = None
        self._key_rows: list[QWidget] = []
        self._run_live = False
        self._run_t0: float | None = None
        self._run_epochs = 0

        root = QVBoxLayout(self)
        title = QLabel("Train")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.sub.setObjectName("pageSub")
        root.addWidget(self.sub)

        form_box = QGroupBox("Job")
        form = self.form = QFormLayout(form_box)
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_change)
        form.addRow("Dataset", self.dataset_combo)
        self.ds_status = QLabel("")
        self.ds_status.setWordWrap(True)
        theme.set_accent(self.ds_status, "muted")
        form.addRow("", self.ds_status)
        self.backbone_combo = QComboBox()
        self.backbone_combo.currentIndexChanged.connect(self._rebuild_params)
        model_row = QHBoxLayout()
        model_row.addWidget(self.backbone_combo, 1)
        self.cfg_btn = QPushButton("Configure model…")
        self.cfg_btn.setToolTip("Pixi env status + install.")
        self.cfg_btn.clicked.connect(self._open_model_config)
        model_row.addWidget(self.cfg_btn)
        form.addRow("Model", ui.wrap(model_row))
        self.star_hint = QLabel("★ = dataset recommendation")
        theme.set_accent(self.star_hint, "muted")
        form.addRow("", self.star_hint)
        # Validation cadence (VAL_EVERY env; trainer default 10 emits nothing).
        self.val_every = QSpinBox()
        self.val_every.setRange(1, 100)
        self.val_every.setValue(10)
        self.val_every.setToolTip(
            "Run the held-out validation pass every N epochs. Lower N = slower "
            "training but finer best-checkpoint selection (the best model is "
            "picked among validated epochs only); higher N = faster, coarser.")
        self.val_every.valueChanged.connect(self._refresh_summaries)
        form.addRow("Validate every N epochs", self.val_every)
        form.addRow("Input features", self._features_row())
        self.detach_chk = QCheckBox("Detach (return immediately — launch several models in parallel)")
        self.detach_chk.setToolTip("Runs in the cloud without streaming logs here. Reattach with "
                                   "`modal app logs <app>`; the run id appears under runs/ on the "
                                   "model's outputs volume (the Inference page Run field is editable).")
        form.addRow("", self.detach_chk)
        self.gpu_combo = QComboBox()
        for g in GPU_CHOICES:
            self.gpu_combo.addItem(g)
        self.gpu_combo.setToolTip("Cloud GPU for this run (Modal only). Defaults to the "
                                  "model's recommendation when you switch models.")
        form.addRow("GPU (Modal)", self.gpu_combo)
        self.backbone_combo.currentIndexChanged.connect(self._sync_gpu_default)
        self.gpu_combo.currentIndexChanged.connect(self._refresh_summaries)

        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        theme.set_accent(self.warn_label, "warn")

        run_row = QHBoxLayout()
        self.launch_btn = QPushButton("Launch training")
        self.launch_btn.setObjectName("primary")
        self.launch_btn.clicked.connect(self._launch)
        run_row.addWidget(self.launch_btn)
        self.stop_ckpt_btn = QPushButton("⏹ Stop at nearest checkpoint")
        self.stop_ckpt_btn.setToolTip(
            "Cooperative stop (local runs): the trainer finishes the current epoch, "
            "then runs its normal final evaluation and best-checkpoint finalize "
            "(test_metrics.json, final_model.pth). Kill stays available meanwhile.")
        self.stop_ckpt_btn.clicked.connect(self._stop_graceful)
        run_row.addWidget(self.stop_ckpt_btn)
        self.stop_btn = QPushButton("Kill")
        self.stop_btn.setToolTip("Hard-kill the process now — no final eval; a later "
                                 "launch with AUTO_RESUME=1 continues from the last "
                                 "periodic checkpoint (all backbones).")
        self.stop_btn.clicked.connect(self._stop)
        run_row.addWidget(self.stop_btn)
        run_row.addStretch()

        self.summary_lbl = QLabel("")
        self.summary_lbl.setWordWrap(True)
        theme.set_accent(self.summary_lbl, "muted")
        self.summary_lbl.setStyleSheet(
            'font-family: "Cascadia Code", Consolas, monospace; font-size: 12px;')

        config_col = QVBoxLayout()
        config_col.addWidget(form_box)
        config_col.addWidget(self._tuning_box())
        config_col.addWidget(self._advanced_box())
        config_col.addWidget(self.warn_label)
        config_col.addWidget(self.summary_lbl)
        config_col.addLayout(run_row)
        config_col.addStretch()

        self.log = LogConsole()
        self.log.setPlaceholderText("Training logs…")
        self.progress_lbl = QLabel("")
        theme.set_accent(self.progress_lbl, "muted")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        prow = QHBoxLayout()
        prow.addWidget(self.progress_lbl)
        prow.addWidget(self.progress_bar, 1)
        self.progress_row = ui.wrap(prow)
        log_col = QVBoxLayout()
        log_col.addWidget(self.log, 1)
        log_col.addWidget(self.progress_row)

        metrics_col = QVBoxLayout()
        metrics_col.addWidget(QLabel("Live epoch metrics"))
        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(["Epoch", "Loss", "Acc", "mIoU"])
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        metrics_col.addWidget(self.metrics_table, 1)

        body = ui.hsplit(ui.wrap(log_col), ui.wrap(metrics_col), sizes=[680, 340])
        root.addWidget(ui.vsplit(ui.wrap(config_col), body,
                                 sizes=[400, 360]), 1)

        self.runner.output.connect(self._on_output)
        self.runner.finished.connect(self._on_finished)
        self.runner.failed.connect(self._on_runner_failed)
        self.pull_runner.output.connect(lambda s: self._append(s, newline=False))
        self.pull_runner.finished.connect(self._on_pull_finished)
        self.pull_runner.failed.connect(self._on_pull_failed)
        self.status_worker.done.connect(self._apply_statuses)
        self.parser.epoch.connect(self._on_epoch)
        self.parser.val_metrics.connect(self._on_val)
        self.parser.run_id.connect(self._on_run_id)
        self.modal_worker.done.connect(self._on_modal_preflight)
        self.modal_worker.error.connect(self._on_modal_preflight_error)

        self.apply_exec_mode(appstate.get_exec_mode() == "local")
        self._rebuild_params()
        self._restore_last_config()
        self.refresh_images()

    def apply_exec_mode(self, local: bool):
        """Re-scheme for the chosen backend: hide local-only / Modal-only rows."""
        self.sub.setText(
            "Pick a dataset and model. Parameters are pre-filled from density analysis; "
            "edit before launching. "
            + ("Runs locally in the model's pixi env on your GPU."
               if local else
               "Runs on a Modal cloud GPU — upload the dataset from the Datasets page "
               "first; the finished run lands on the model's outputs volume."))
        self.cfg_btn.setVisible(local)
        # graceful stop is local-only: the sentinel rides the /outputs bind mount
        self._set_run_live(self._run_live)
        self.form.setRowVisible(self.gpu_combo, not local)
        self.form.setRowVisible(self.detach_chk, not local)
        self.reload_datasets()

    def _sync_gpu_default(self):
        b = self._backbone()
        if b is not None:
            i = self.gpu_combo.findText(b.rec_gpu)
            if i >= 0:
                self.gpu_combo.setCurrentIndex(i)

    # --------------------------------------------- per-model pixi-env popup
    def _open_model_config(self):
        b = self._backbone()
        if b is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Model configuration")
        lay = QVBoxLayout(dlg)
        self._cfg_model_lbl = QLabel()
        self._cfg_model_lbl.setObjectName("pageTitle")
        lay.addWidget(self._cfg_model_lbl)
        self._cfg_tag = QLabel()
        theme.set_accent(self._cfg_tag, "muted")
        lay.addWidget(self._cfg_tag)
        self._cfg_status = QLabel()
        lay.addWidget(self._cfg_status)

        btn_row = QHBoxLayout()
        self._cfg_pull_btn = QPushButton("Install / update env")
        self._cfg_pull_btn.clicked.connect(self._install_current)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_images)
        close = QPushButton("Close")
        close.clicked.connect(dlg.close)
        btn_row.addWidget(self._cfg_pull_btn)
        btn_row.addWidget(refresh)
        btn_row.addStretch()
        btn_row.addWidget(close)
        lay.addLayout(btn_row)

        self._cfg_dialog = dlg
        dlg.finished.connect(lambda *_: setattr(self, "_cfg_dialog", None))
        self._update_cfg_dialog()
        self.refresh_images()
        dlg.show()   # non-modal: install progress shows in the main log

    @staticmethod
    def _status_text(s: dict | None):
        """(text, accent-role, install-enabled) for an env-status dict."""
        if s is None:
            return "checking…", "muted", False
        if not s["pixi"]:
            return "pixi not found", "muted", False
        if s["installed"]:
            return "✓ installed", "ok", True
        return "✗ not installed", "warn", True

    def _update_cfg_dialog(self):
        if self._cfg_dialog is None:
            return
        b = self._backbone()
        if b is None:
            return
        self._cfg_model_lbl.setText(b.label)
        self._cfg_tag.setText(f"pixi env: {local_cli.env_name(b)}  (envs/pixi.toml)")
        text, role, can_install = self._status_text(self._last_statuses.get(b.key))
        self._cfg_status.setText(f"status: {text}")
        theme.set_accent(self._cfg_status, role)
        self._cfg_pull_btn.setEnabled(can_install and not self.pull_runner.running)

    def refresh_images(self):
        """Re-check env presence off the GUI thread; updates the popup when done."""
        if self.status_worker.running:
            return
        self.status_worker.start(local_cli.all_statuses)

    def _apply_statuses(self, statuses: list):
        self._last_statuses = {s["key"]: s for s in statuses}
        self._update_cfg_dialog()

    def _install_current(self):
        b = self._backbone()
        if b is None:
            return
        s = self._last_statuses.get(b.key)
        if not (s and s["pixi"]):
            self._append("[local] pixi not found on PATH - install pixi "
                         "(https://pixi.sh), then retry.")
            return
        if self.pull_runner.running:
            return
        self._cfg_pull_btn.setEnabled(False)
        prog, args = local_cli.install(b, self.repo_root)
        self._append(f"\n[local] $ {local_cli.preview(prog, args)}\n")
        self.pull_runner.start(prog, args, cwd=self.repo_root)

    def _on_pull_finished(self, code: int):
        self._append("[local] ✓ env installed." if code == 0
                     else f"[local] ✗ env install failed (exit {code}).")
        self.refresh_images()

    def _on_pull_failed(self, err: str):
        self._append(f"\n[local] ✗ pixi install failed to start: {err}")
        self._update_cfg_dialog()

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
        if not name:
            self._set_ds_status("")
            self._ds_ready = False
        else:
            text, role, ready = self._local_split_status(info)
            self._set_ds_status(text, role)
            self._ds_ready = ready
        self._reload_backbones()

    def _local_split_status(self, info: dict):
        """(text, role, ready). Verify the train/val/test standard locally; val and
        test are both required to launch."""
        staged = info.get("staged_dir", "")
        if not staged or not os.path.isdir(staged):
            return ("No local copy - convert it on the Datasets page.",
                    "warn", False)
        root = Path(staged)
        tr = list((root / "train").glob("*.npz")) if (root / "train").is_dir() else []
        va = list((root / "val").glob("*.npz")) if (root / "val").is_dir() else []
        te = list((root / "test").glob("*.npz")) if (root / "test").is_dir() else []
        if not tr or not va or not te:
            return (f"train/val/test standard not met - train {len(tr)}, val {len(va)}, "
                    f"test {len(te)} scene(s). Re-build on the Datasets page.", "warn", False)
        return (f"✓ train/val/test met - {len(tr)} train / {len(va)} val / "
                f"{len(te)} test scene(s).", "ok", True)

    def _set_ds_status(self, text: str, role: str = "muted"):
        theme.set_accent(self.ds_status, role)
        self.ds_status.setText(text)

    def _reload_backbones(self):
        """Populate the model dropdown with every backbone."""
        prev = self.backbone_combo.currentData()
        self.backbone_combo.blockSignals(True)
        self.backbone_combo.clear()
        for key, b in BACKBONES.items():
            self.backbone_combo.addItem(b.label, key)
        i = self.backbone_combo.findData(prev)
        if i >= 0:
            self.backbone_combo.setCurrentIndex(i)
        self.backbone_combo.blockSignals(False)
        self._rebuild_params()

    # ------------------------------------------------------------- params form
    def _backbone(self):
        key = self.backbone_combo.currentData()
        return BACKBONES.get(key) if key else None

    def _rebuild_params(self):
        b = self._backbone()
        sig = (b.key if b else None, self.dataset_combo.currentText())
        if sig == self._built_sig and self._param_widgets:
            # same backbone + dataset: never clobber user-edited values
            self._update_cfg_dialog()
            self._refresh_summaries()
            return
        prev = {f: self._wvalue(w) for f, w in self._param_widgets.items()}
        same_bb = self._built_sig is not None and self._built_sig[0] == sig[0]
        self._built_sig = sig
        for w in self._key_rows:
            self.form.removeRow(w)
        self._key_rows.clear()
        while self.tuning_form.rowCount():
            self.tuning_form.removeRow(0)
        self._param_widgets.clear()
        if b is None:
            self.warn_label.setText("")
            self.form.setRowVisible(self.star_hint, False)
            self._rebuild_feat_list()
            self._update_cfg_dialog()
            self._refresh_summaries()
            return
        recs = (self._meta or {}).get("recommendations", {}).get(b.key, {})
        row = 3   # Job-form insert point: right after the Model row
        for spec in b.params:
            rec = recs.get(spec.recommend_key) if spec.recommend_key and \
                spec.recommend_key in recs else None
            # keep the user's value unless a fresh dataset rec supersedes it
            if spec.flag in prev and ((same_bb and rec is None) or spec.flag == "epochs"):
                value = prev[spec.flag]
            elif rec is not None:
                value = rec
            else:
                value = spec.default
            if spec.flag == "freeze-encoder":
                w = QCheckBox("Freeze encoder (linear probe)")
                w.setChecked(bool(int(value)))
                w.toggled.connect(self._refresh_summaries)
            elif spec.kind == "float":
                w = QDoubleSpinBox()
                w.setDecimals(spec.decimals)
                w.setSingleStep(spec.step)
                w.setRange(spec.lo, 1_000_000.0)   # spec.hi is a reco band, not a cap
                w.setValue(float(value))
                w.valueChanged.connect(self._refresh_summaries)
            else:
                w = QSpinBox()
                w.setRange(int(spec.lo), 100_000_000)
                w.setValue(int(value))
                w.valueChanged.connect(self._refresh_summaries)
            label = spec.label + ("  ★" if rec is not None else "")
            if spec.flag in _TUNING_FLAGS:
                self.tuning_form.addRow(label, w)
            elif spec.flag == "freeze-encoder":
                self.form.insertRow(row, "", w)
                self._key_rows.append(w)
                row += 1
            else:
                self.form.insertRow(row, label, w)
                self._key_rows.append(w)
                row += 1
            self._param_widgets[spec.flag] = w
        self.form.setRowVisible(
            self.star_hint,
            any(s.recommend_key and s.recommend_key in recs for s in b.params))
        if self._meta:
            warns = analysis.warnings_for(self._meta)
            self.warn_label.setText("\n".join("⚠ " + w for w in warns))
        else:
            self.warn_label.setText("")
        self._rebuild_feat_list()
        self._update_cfg_dialog()
        self._refresh_summaries()

    @staticmethod
    def _wvalue(w):
        """Param widget -> value (freeze-encoder checkbox maps to 0/1)."""
        return int(w.isChecked()) if isinstance(w, QCheckBox) else w.value()

    @staticmethod
    def _wset(w, v):
        if isinstance(w, QCheckBox):
            w.setChecked(bool(int(v)))
        elif isinstance(w, QSpinBox):
            w.setValue(int(v))
        else:
            w.setValue(float(v))

    # ------------------------------------------- tuning fold (collapsed default)
    def _tuning_box(self) -> QGroupBox:
        box = QGroupBox("Tuning")
        box.setCheckable(True)
        box.setChecked(False)
        outer = QVBoxLayout(box)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(inner)
        box.toggled.connect(inner.setVisible)
        inner.setVisible(False)
        self.tuning_box = box
        self.tuning_form = QFormLayout()
        lay.addLayout(self.tuning_form)
        return box

    def _refresh_summaries(self):
        """Echo current values into the Tuning fold title + launch summary bar."""
        if not hasattr(self, "summary_lbl"):   # during early construction
            return
        parts = []
        w = self._param_widgets.get("batch")
        if w is not None:
            parts.append(f"batch {w.value()}")
        w = (self._param_widgets.get("grid")
             or self._param_widgets.get("sub-grid"))
        if w is not None:
            parts.append(f"grid {w.value():g} m")
        w = self._param_widgets.get("chunk-xy")
        if w is not None:
            parts.append(f"tile {w.value():g} m")
        self.tuning_box.setTitle("Tuning — " + " · ".join(parts))
        b = self._backbone()
        name = self.dataset_combo.currentText()
        if b is None or not name:
            self.summary_lbl.setText("")
            return
        ep_w = self._param_widgets.get("epochs")
        ep = ep_w.value() if ep_w is not None else 0
        sw = self._param_widgets.get("steps-per-epoch")
        steps = sw.value() if sw is not None else 0
        mode = ("pixi (local)" if appstate.get_exec_mode() == "local"
                else f"Modal ({self.gpu_combo.currentText()})")
        self.summary_lbl.setText(
            f"▶ {b.label} · {name} · {ep} ep × {steps} steps · {mode}")

    # --------------- advanced (loss / class balance / density, one box per run)
    def _advanced_box(self) -> QGroupBox:
        box = QGroupBox("Advanced")
        box.setCheckable(True)
        box.setChecked(False)
        outer = QVBoxLayout(box)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(inner)
        box.toggled.connect(inner.setVisible)
        inner.setVisible(False)
        self.adv_box = box

        hint = QLabel("Loss & class balance — defaults already handle imbalance "
                      "(rare classes weighted up + oversampled). Everything set "
                      "here is recorded in run.json.")
        hint.setWordWrap(True)
        theme.set_accent(hint, "muted")
        lay.addWidget(hint)

        self.loss_focal = QCheckBox("Use focal loss instead of weighted cross-entropy")
        self.loss_focal.setToolTip("Down-weights easy points to focus on hard/rare ones. "
                                   "Off by default (weighted CE + Lovász).\n"
                                   "↑ (on): focuses training on hard/rare points; can "
                                   "starve easy classes. Off = weighted CE + Lovász "
                                   "(default).")
        self.loss_gamma = QDoubleSpinBox()
        self.loss_gamma.setRange(0.0, 5.0)
        self.loss_gamma.setSingleStep(0.5)
        self.loss_gamma.setValue(2.0)
        self.loss_gamma.setToolTip("↑ γ = harder focus on misclassified points, risk of "
                                   "instability; ↓ γ = closer to plain CE. 2.0 is the "
                                   "standard default.")
        self.loss_gamma.setEnabled(False)
        self.loss_focal.toggled.connect(self.loss_gamma.setEnabled)
        r1 = QHBoxLayout()
        r1.addWidget(self.loss_focal)
        r1.addWidget(QLabel("γ (focus)"))
        r1.addWidget(self.loss_gamma)
        r1.addStretch(1)
        lay.addLayout(r1)

        self.loss_cw = QCheckBox("Weight classes by inverse frequency")
        self.loss_cw.setChecked(True)
        self.loss_cw.setToolTip("On = rare classes count more in the loss; off = every "
                                "point equal (majority classes dominate).")
        self.loss_beta = QDoubleSpinBox()
        self.loss_beta.setRange(0.0, 1.0)
        self.loss_beta.setSingleStep(0.05)
        self.loss_beta.setValue(0.5)
        self.loss_beta.setToolTip("0 = none, 0.5 = inverse-sqrt (default), "
                                  "1 = full inverse-frequency (most aggressive).\n"
                                  "↑ β = stronger boost for rare classes (risk: noisy "
                                  "rare labels get amplified); ↓ β = closer to "
                                  "unweighted.")
        self.loss_cw.toggled.connect(self.loss_beta.setEnabled)
        r2 = QHBoxLayout()
        r2.addWidget(self.loss_cw)
        r2.addWidget(QLabel("strength β"))
        r2.addWidget(self.loss_beta)
        r2.addStretch(1)
        lay.addLayout(r2)

        self.loss_rare = QCheckBox("Oversample rare-class tiles (auto-detected)")
        self.loss_rare.setChecked(True)
        self.loss_rare.setToolTip("On = tiles containing rare classes are drawn more "
                                  "often; off = uniform tile sampling.")
        lay.addWidget(self.loss_rare)

        dg_hint = QLabel("Density robustness — train the model to tolerate point "
                         "clouds sparser than the training data. Costs a little "
                         "accuracy at native density; inference reads these "
                         "settings back from run.json automatically.")
        dg_hint.setWordWrap(True)
        theme.set_accent(dg_hint, "muted")
        lay.addWidget(dg_hint)
        self._dg_rows(lay)
        return box

    def _loss_collect(self) -> dict:
        if not self.adv_box.isChecked():        # off -> script defaults
            return {}
        return {"focal": self.loss_focal.isChecked(),
                "focal_gamma": round(self.loss_gamma.value(), 2),
                "class_weighting": self.loss_cw.isChecked(),
                "weight_beta": round(self.loss_beta.value(), 2),
                "rare_oversample": self.loss_rare.isChecked()}

    def _dg_rows(self, lay):
        """Density-robustness rows appended inside the Advanced box."""
        self.dg_aug = QCheckBox("Density augmentation (random coarsen per tile)")
        self.dg_aug.setToolTip(
            "Re-subsamples a share of training tiles to a coarser random grid "
            "(DG_DENSITY_AUG) so the model also sees sparse versions of the data.\n"
            "Coarsen-only: helps when inference is SPARSER than training; the "
            "voxel grid already canonicalizes denser inputs for free.")
        self.dg_coarsen = QDoubleSpinBox()
        self.dg_coarsen.setRange(1.5, 6.0)
        self.dg_coarsen.setSingleStep(0.5)
        self.dg_coarsen.setValue(2.5)
        self.dg_coarsen.setToolTip("Max coarsening factor over the model grid "
                                   "(DG_COARSEN_MAX). Size it to the sparsest "
                                   "density you expect at inference.")
        self.dg_coarsen.setEnabled(False)
        self.dg_pnative = QDoubleSpinBox()
        self.dg_pnative.setRange(0.0, 1.0)
        self.dg_pnative.setSingleStep(0.05)
        self.dg_pnative.setValue(0.5)
        self.dg_pnative.setToolTip("Share of tiles kept at native density "
                                   "(DG_P_NATIVE). Lower = more mass on the "
                                   "coarse end; 0.5 is the default.")
        self.dg_pnative.setEnabled(False)
        self.dg_aug.toggled.connect(self.dg_coarsen.setEnabled)
        self.dg_aug.toggled.connect(self.dg_pnative.setEnabled)
        r1 = QHBoxLayout()
        r1.addWidget(self.dg_aug)
        r1.addWidget(QLabel("max coarsen ×"))
        r1.addWidget(self.dg_coarsen)
        r1.addWidget(QLabel("p(native)"))
        r1.addWidget(self.dg_pnative)
        r1.addStretch(1)
        lay.addLayout(r1)

        self.dg_logdk = QCheckBox("log dₖ local-density input channel")
        self.dg_logdk.setToolTip(
            "Appends log of the k-th-neighbour distance as an input feature "
            "(DG_LOGDK_FEAT) so the model can condition on local density.\n"
            "Changes the input width: baked into the weights, retrain-only, and "
            "re-initializes the pretrained stem on Concerto/Sonata/Utonia. Only "
            "meaningful together with density augmentation.")
        self.dg_k = QSpinBox()
        self.dg_k.setRange(4, 32)
        self.dg_k.setValue(8)
        self.dg_k.setToolTip("k for the k-th-neighbour distance (DG_LOGDK_K).")
        self.dg_k.setEnabled(False)
        self.dg_logdk.toggled.connect(self.dg_k.setEnabled)
        r2 = QHBoxLayout()
        r2.addWidget(self.dg_logdk)
        r2.addWidget(QLabel("k"))
        r2.addWidget(self.dg_k)
        r2.addStretch(1)
        lay.addLayout(r2)

    def _dg_collect(self) -> dict:
        if not self.adv_box.isChecked():
            return {}
        return {"density_aug": self.dg_aug.isChecked(),
                "coarsen_max": round(self.dg_coarsen.value(), 2),
                "p_native": round(self.dg_pnative.value(), 2),
                "logdk": self.dg_logdk.isChecked(),
                "logdk_k": self.dg_k.value()}

    # -------------------------------------------- input features (per run)
    def _features_row(self) -> QWidget:
        """Wrapping chip list: the checked chips are the run's feature spec."""
        self.feat_list = QListWidget()
        self.feat_list.setFlow(QListWidget.LeftToRight)
        self.feat_list.setWrapping(True)
        self.feat_list.setResizeMode(QListWidget.Adjust)
        self.feat_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.feat_list.setDefaultDropAction(Qt.MoveAction)
        self.feat_list.setMaximumHeight(58)   # ~2 chip rows, then scrolls
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.feat_list, 1)
        return ui.wrap(row)

    def _rebuild_feat_list(self):
        """Repopulate for the selected backbone + dataset; standard channels
        pre-checked as a reversible default (default_feature_checks)."""
        if not hasattr(self, "feat_list"):
            return
        self.feat_list.clear()
        b = self._backbone()
        if b is None:
            return
        base = b.key
        meta = self._meta or {}
        # offered = arch capability ∩ channels the dataset was built with
        std = [n for n in _FEAT_STANDARD.get(base, []) if meta.get(f"has_{n}")]
        extra = ((meta.get("source") or {}).get("feature_channels")
                 if isinstance(meta.get("source"), dict) else None) or []
        extra_names = [c.get("name") if isinstance(c, dict) else str(c)
                       for c in extra]
        # older datasets may carry feat_ duplicates of canonical channels — drop them
        extra_names = [n for n in extra_names if n
                       and (dataset.canonical_channel(
                            n[5:] if n.startswith("feat_") else n) or n) not in std]
        names = std + [n if n.startswith("feat_") else f"feat_{n}"
                       for n in extra_names]
        # feat_hag never appears in feature_channels; offered via has_hag
        if meta.get("has_hag") and "feat_hag" not in names:
            names.append("feat_hag")
        checked = default_feature_checks(base, std)
        for n in names:
            # display strips the feat_ prefix; the real channel name rides in UserRole
            it = QListWidgetItem(n[5:] if n.startswith("feat_") else str(n))
            it.setData(Qt.UserRole, str(n))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if n in checked else Qt.Unchecked)
            self.feat_list.addItem(it)
        self.feat_list.setToolTip(
            "Checked features are this run's input channels, left-to-right "
            "(drag to reorder). Point coordinates are always fed to the model "
            "and aren't listed. Recorded in run.json.")

    def _feat_collect(self) -> str:
        """Checked names in list order -> FEAT_CHANNELS csv (xyz prepended
        where the trainer's contract expects it)."""
        names = [self.feat_list.item(i).data(Qt.UserRole)
                 for i in range(self.feat_list.count())
                 if self.feat_list.item(i).checkState() == Qt.Checked]
        b = self._backbone()
        if b is not None and b.key in _XYZ_IMPLICIT:
            names = ["x", "y", "z"] + names   # coords: always in, never a chip
        return ",".join(names)

    # ------------------------------------------------------------- launch
    def _launch(self):
        b = self._backbone()
        if b is None:
            self._append("Pick a model first.")
            return
        name = self.dataset_combo.currentText()
        if not name:
            self._append("Create a dataset on the Datasets page first.")
            return
        if not self._ds_ready:
            self._append("Dataset doesn't meet the train/val/test standard - fix it on "
                         "the Datasets page.")
            return
        if self.runner.running:
            self._append("A training process is already running.")
            return

        info = appstate.known_datasets().get(name, {})
        flags = {"dataset": name}
        for spec in b.params:
            flags[spec.flag] = self._wvalue(self._param_widgets[spec.flag])

        loss_env = analysis.loss_config_to_env(self._loss_collect())
        dg_env = analysis.dg_config_to_env(self._dg_collect())
        env = dict(loss_env)
        env.update(dg_env)
        if self.val_every.value() != 10:   # default emits nothing (script default)
            env["VAL_EVERY"] = str(self.val_every.value())
        feat_csv = self._feat_collect()
        if not feat_csv:
            # never silently fall back to the trainers' built-in specs
            self._append("No input features checked — check at least one on "
                         "the Input features row.")
            return
        if b.key in _PTV3_LIKE and {"intensity", "rgb"} <= set(feat_csv.split(",")):
            # ptv3 has one 3-wide color slot; both set = server-side ValueError
            self._append("ptv3-family models have a single color slot — check "
                         "intensity OR rgb, not both.")
            return
        env["FEAT_CHANNELS"] = feat_csv
        # forward only when actually set — a default would freeze the Modal
        # container's own defaults; AUTO_RESUME rides along for deliberate resumes
        for k in ("TT_TRAIN_STRIDE", "TT_AMP", "TT_PREFETCH", "AUTO_RESUME",
                  "EVAL_BATCH"):
            if os.environ.get(k):
                env[k] = os.environ[k]
        params = {f: self._wvalue(w) for f, w in self._param_widgets.items()}
        appstate.put("train_last_config", {
            "dataset": name, "backbone": b.key,
            "params": params,
            "val_every": self.val_every.value(),
            "loss": self._loss_collect(),
            "dg": self._dg_collect(),
            "features": feat_csv,
            "gpu": self.gpu_combo.currentText(),
            "detach": self.detach_chk.isChecked(),
        })
        self.log.begin_run(f"{b.label} · {name}")
        self._run_t0 = time.time()
        self._run_epochs = int(flags.get("epochs", 0))
        self.metrics_table.setRowCount(0)
        self._last_run_id = None
        self.launch_btn.setEnabled(False)
        if loss_env:
            self._append("[loss] overrides: "
                         + " ".join(f"{k}={v}" for k, v in sorted(loss_env.items())))
        if dg_env:
            self._append("[dg] overrides: "
                         + " ".join(f"{k}={v}" for k, v in sorted(dg_env.items())))
        if feat_csv:
            self._append(f"[features] FEAT_CHANNELS={feat_csv}")
        payload = {"backbone": b, "flags": flags, "dataset": name,
                   "env": env, "info": info}
        if appstate.get_exec_mode() == "local":
            self._start_local_run(payload)
        else:
            self._preflight_modal_run(payload)

    def _start_local_run(self, p):
        b, flags, name, info = p["backbone"], p["flags"], p["dataset"], p["info"]
        staged = info.get("staged_dir", "")
        # runs nest per dataset: TT_OUTPUTS_ROOT = <workspace>/<dataset>
        base = str(appstate.workspace_dir())
        out_root = str(Path(base) / name)
        os.makedirs(out_root, exist_ok=True)
        dataset_dir = ""
        if not (staged and os.path.isdir(staged)):
            self._append(f"[local] ⚠ No staged copy of '{name}' - "
                         f"the trainer won't find the dataset.")
        elif Path(staged).parent != Path(appstate.local_config()["datasets_root"]):
            # dataset outside the workspace: point TT_DATASET_DIR straight at it
            dataset_dir = staged
        prog, args, run_env = local_cli.run_script(
            b.script, flags, b, repo_root=self.repo_root,
            outputs_root=out_root, dataset_dir=dataset_dir, env=p.get("env", {}))
        self._append(f"\n[local] $ {local_cli.preview(prog, args, run_env)}\n")
        ok_gpu, msg_gpu = local_cli.gpu_preflight()
        if msg_gpu:
            self._append(msg_gpu)
        if not ok_gpu:
            self.launch_btn.setEnabled(True)
            return
        ok, msg = local_cli.env_preflight(b, self.repo_root)
        if msg:
            self._append(msg)
        if not ok:
            self.launch_btn.setEnabled(True)
            return
        self._out_root = out_root   # anchor for the graceful-stop sentinel
        self.runner.start(prog, args, cwd=self.repo_root, extra_env=run_env)
        self._set_run_live(True)

    # ------------------------------------------------------------- modal path
    def _preflight_modal_run(self, p):
        """Verify the dataset is on the volume (off-thread) before paying for
        a GPU container."""
        import shutil as _sh
        if _sh.which("modal") is None:
            self._append("[modal] 'modal' CLI not found on PATH — `pip install modal`, "
                         "then `modal setup` to authenticate, and launch again.")
            self.launch_btn.setEnabled(True)
            return
        self._pending = p
        name = p["dataset"]
        self._append(f"[modal] checking '{name}' on the "
                     f"'{modal_cli.DATASETS_VOLUME}' volume…")
        self.modal_worker.start(
            lambda progress=None: modal_cli.list_volume_entries(
                modal_cli.DATASETS_VOLUME, f"/{name}"))

    def _on_modal_preflight(self, entries):
        p, self._pending = self._pending, None
        if p is None:
            return
        if not entries:
            self._append(f"✗ Dataset '{p['dataset']}' isn't on the "
                         f"'{modal_cli.DATASETS_VOLUME}' volume. Upload it from the "
                         "Datasets page (Upload to Modal) and launch again.")
            self.launch_btn.setEnabled(True)
            return
        self._start_modal_run(p)

    def _on_modal_preflight_error(self, tb: str):
        p, self._pending = self._pending, None
        if p is None:
            return
        self._append("[modal] (couldn't verify the dataset on the volume — "
                     "proceeding anyway.)")
        self._start_modal_run(p)

    def _start_modal_run(self, p):
        b, flags = p["backbone"], p["flags"]
        gpu = self.gpu_combo.currentText()
        detach = self.detach_chk.isChecked()
        prog, args = modal_cli.run_script(b.script, flags, detach=detach,
                                          env=p.get("env") or None)
        self._append(f"\n[modal] $ TT_GPU={gpu} modal {' '.join(args)}\n")
        self._append(f"[modal] Training on {gpu}; the run is written to the "
                     f"'{b.outputs_volume}' volume as runs/<id> — pick it on the "
                     "Inference page (Training run) when it finishes.")
        if detach:
            self._append("[modal] Detached: this returns once the cloud app starts — "
                         "no logs/metrics stream here. Reattach with "
                         f"`modal app logs {b.app_name}`; then launch the next model.")
        self.runner.start(prog, args, cwd=self.repo_root, extra_env={"TT_GPU": gpu})
        self._set_run_live(True)

    def _set_run_live(self, live: bool):
        """Stop/kill buttons + progress strip only exist while a run is live."""
        self._run_live = live
        self.stop_btn.setVisible(live)
        self.stop_ckpt_btn.setVisible(live and appstate.get_exec_mode() == "local")
        self.progress_row.setVisible(live)
        if not live:
            self.progress_lbl.setText("")
            self.progress_bar.setValue(0)

    def _run_duration(self) -> str:
        dur = int(time.time() - self._run_t0) if self._run_t0 else 0
        return f"{dur // 60}m{dur % 60:02d}s"

    def _on_runner_failed(self, err: str):
        # FailedToStart fires failed, not finished; re-enable here
        self.launch_btn.setEnabled(True)
        self._clear_stop_sentinel()
        if self._run_live:
            self.log.end_run(f"{err} · {self._run_duration()}")
            self._set_run_live(False)
        self._append(f"\n✗ Failed to start: {err}")

    def _stop(self):
        if self.runner.running:
            self.runner.terminate()
            self._append("\n[stopped]")
            if appstate.get_exec_mode() != "local":
                self._append("[modal] note: killing the local client can leave the "
                             "cloud app running — `modal app list` to check, "
                             "`modal app stop <app>` to stop it.")
        else:
            self._append("\n[no process running]")

    def _stop_graceful(self):
        """Drop the STOP sentinel at the run's /outputs root; the trainer consumes
        it at epoch end, breaks, and runs its normal final eval + finalize."""
        if not self.runner.running:
            self._append("\n[no process running]")
            return
        if appstate.get_exec_mode() != "local" or not self._out_root:
            self._append("\n[stop] graceful stop works for local runs only — use Kill "
                         "(and `modal app stop <app>` for a cloud run).")
            return
        try:
            (Path(self._out_root) / "STOP").touch()
        except OSError as e:
            self._append(f"\n[stop] couldn't write the STOP sentinel: {e} — use Kill.")
            return
        self._append("\n[stop] Stopping after the current epoch… the run finishes its "
                     "final evaluation and checkpoint finalize, then exits. "
                     "(Kill remains available if you can't wait.)")

    def _clear_stop_sentinel(self):
        """Remove a leftover sentinel (killed after a graceful request, or the
        trainer crashed before consuming it) so it can't stop the next run."""
        if self._out_root:
            try:
                (Path(self._out_root) / "STOP").unlink()
            except OSError:
                pass
            self._out_root = None

    # ------------------------------------------------------------- stream
    def _on_output(self, text: str):
        self._append(text, newline=False)
        self.parser.feed(text)

    def _on_finished(self, code: int):
        self.launch_btn.setEnabled(True)
        self._clear_stop_sentinel()
        if self._run_live:
            self.log.end_run(f"exit {code} · {self._run_duration()}")
            self._set_run_live(False)
        if code == 0:
            extra = (f" Run id: {self._last_run_id}." if self._last_run_id else "")
            self._append(f"\n✓ Done.{extra} See the Runs/Plotting page for artifacts.")
        else:
            self._append(f"\n✗ Exited with code {code}.")

    def _on_epoch(self, m: dict):
        total = self._run_epochs
        if total:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(m["epoch"], total))
        self.progress_lbl.setText(f"epoch {m['epoch']}/{total or '?'} · "
                                  f"loss {m['loss']:.2f} · miou {m['miou']:.2f}")
        r = self.metrics_table.rowCount()
        self.metrics_table.insertRow(r)
        for col, val in enumerate((str(m["epoch"]), f"{m['loss']:.4f}",
                                   f"{m['acc']:.4f}", f"{m['miou']:.4f}")):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.metrics_table.setItem(r, col, item)
        self.metrics_table.scrollToBottom()

    def _on_val(self, m: dict):
        """Grey starred row for a held-out val pass: no train loss, val acc/mIoU
        (present-classes). Colored at insert time from the current theme."""
        grey = QBrush(QColor(theme.colors(appstate.get("ui_theme", "system"))["muted"]))
        r = self.metrics_table.rowCount()
        self.metrics_table.insertRow(r)
        for col, val in enumerate((f"{m['epoch']}★", "—",
                                   f"{m['acc']:.4f}", f"{m['miou']:.4f}")):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(grey)
            self.metrics_table.setItem(r, col, item)
        self.metrics_table.scrollToBottom()

    def _on_run_id(self, run_id: str):
        self._last_run_id = run_id
        history = appstate.get("run_history", [])
        b = self._backbone()
        history.append({"run_id": run_id, "backbone": b.key if b else "",
                        "dataset": self.dataset_combo.currentText()})
        appstate.put("run_history", history[-200:])

    def _restore_last_config(self):
        """Repopulate the page from the last launched config; stale dataset/
        backbone/param names fall back silently per-field."""
        cfg = appstate.get("train_last_config")
        if not isinstance(cfg, dict):
            return
        i = self.dataset_combo.findText(cfg.get("dataset", ""))
        if i >= 0:
            self.dataset_combo.setCurrentIndex(i)
        i = self.backbone_combo.findData(cfg.get("backbone"))
        if i >= 0:
            self.backbone_combo.setCurrentIndex(i)
        for flag, val in (cfg.get("params") or {}).items():
            w = self._param_widgets.get(flag)
            if w is not None:
                try:
                    self._wset(w, val)
                except (TypeError, ValueError):
                    pass
        try:
            self.val_every.setValue(int(cfg.get("val_every", 10)))
        except (TypeError, ValueError):
            pass
        i = self.gpu_combo.findText(cfg.get("gpu", ""))
        if i >= 0:
            self.gpu_combo.setCurrentIndex(i)
        self.detach_chk.setChecked(bool(cfg.get("detach")))
        loss = cfg.get("loss") if isinstance(cfg.get("loss"), dict) else {}
        dg = cfg.get("dg") if isinstance(cfg.get("dg"), dict) else {}
        self.adv_box.setChecked(bool(loss) or bool(dg))  # {}+{} = box off (defaults)
        if loss:
            try:
                self.loss_focal.setChecked(bool(loss.get("focal", False)))
                self.loss_gamma.setValue(float(loss.get("focal_gamma", 2.0)))
                self.loss_cw.setChecked(bool(loss.get("class_weighting", True)))
                self.loss_beta.setValue(float(loss.get("weight_beta", 0.5)))
                self.loss_rare.setChecked(bool(loss.get("rare_oversample", True)))
            except (TypeError, ValueError):
                pass
        if dg:
            try:
                self.dg_aug.setChecked(bool(dg.get("density_aug", False)))
                self.dg_coarsen.setValue(float(dg.get("coarsen_max", 2.5)))
                self.dg_pnative.setValue(float(dg.get("p_native", 0.5)))
                self.dg_logdk.setChecked(bool(dg.get("logdk", False)))
                self.dg_k.setValue(int(dg.get("logdk_k", 8)))
            except (TypeError, ValueError):
                pass
        # features not restored from last config — chips seed a per-arch default

    def _append(self, text: str, newline: bool = True):
        ui.append_log(self.log, text, newline)
