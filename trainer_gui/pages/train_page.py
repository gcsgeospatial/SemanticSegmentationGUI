"""Train page (local Docker): dataset + model + params -> a local docker run,
with live logs and epoch metrics. Dataset check verifies the train/val/test
standard; per-model image status/pull live in a popup; DG training knobs are
per run (inference-time DG is on the Infer page).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from .. import analysis, appstate, local_cli, theme, ui
from ..backbones import BACKBONES
from ..jobs import FuncWorker, JobRunner, LogParser


class TrainPage(QWidget):
    models_changed = Signal()   # kept for main.py; no longer emitted

    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.runner = JobRunner(self)          # training docker process
        self.pull_runner = JobRunner(self)     # docker pull, streamed to log
        self.status_worker = FuncWorker(self)  # off-thread image-presence check
        self.parser = LogParser(self)
        self._param_widgets: dict[str, QWidget] = {}
        self._meta: dict | None = None
        self._last_run_id: str | None = None
        self._pending: dict | None = None
        self._last_statuses: dict = {}         # key -> status dict from all_statuses
        self._cfg_dialog: QDialog | None = None  # per-model popup when open
        self._ds_ready = False                 # train/val/test standard met

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
        self.backbone_combo = QComboBox()      # populated in _reload_backbones
        self.backbone_combo.currentIndexChanged.connect(self._rebuild_params)
        model_row = QHBoxLayout()
        model_row.addWidget(self.backbone_combo, 1)
        self.cfg_btn = QPushButton("Configure model…")
        self.cfg_btn.setToolTip("Docker image status + pull.")
        self.cfg_btn.clicked.connect(self._open_model_config)
        model_row.addWidget(self.cfg_btn)
        form.addRow("Model", _wrap(model_row))
        self.smoke_chk = QCheckBox("Smoke run (2 epochs × 50 steps)")
        self.smoke_chk.toggled.connect(self._apply_smoke)
        form.addRow("Options", self.smoke_chk)
        # Base folder for output; a run lands at <base>/<dataset>/runs/<id> on the host.
        self.out_edit = QLineEdit()
        self.out_edit.setText(appstate.get("local_train_out") or str(appstate.workspace_dir()))
        self.out_edit.setPlaceholderText("default: workspace folder")
        self.out_edit.setToolTip("Base folder for training output. Each run is written "
                                 "to <this>/<dataset>/runs/<id>.")
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit)
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._pick_out)
        out_row.addWidget(out_btn)
        form.addRow("Output folder", _wrap(out_row))

        self.params_box = QGroupBox("Parameters (pre-filled)")
        self.params_form = QFormLayout(self.params_box)
        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        theme.set_accent(self.warn_label, "warn")

        run_row = QHBoxLayout()
        self.launch_btn = QPushButton("Launch training")
        self.launch_btn.setObjectName("primary")
        self.launch_btn.clicked.connect(self._launch)
        run_row.addWidget(self.launch_btn)
        self.stop_btn = QPushButton("Stop process")
        self.stop_btn.clicked.connect(self._stop)
        run_row.addWidget(self.stop_btn)
        run_row.addStretch()

        config_col = QVBoxLayout()
        config_col.addWidget(form_box)
        config_col.addWidget(self.params_box)
        # TODO(not ready): domain-generalization UI hidden until reviewed. The
        # _dg_* methods stay defined but unused; no DG env is sent (see _launch).
        # config_col.addWidget(self._dg_box())
        config_col.addWidget(self._loss_box())
        config_col.addWidget(self.warn_label)
        config_col.addLayout(run_row)
        config_col.addStretch()

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("log")
        self.log.setPlaceholderText("Training logs…")

        metrics_col = QVBoxLayout()
        metrics_col.addWidget(QLabel("Live epoch metrics"))
        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(["Epoch", "Loss", "Acc", "mIoU"])
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        metrics_col.addWidget(self.metrics_table, 1)

        body = ui.hsplit(self.log, ui.wrap(metrics_col), sizes=[680, 340])
        root.addWidget(ui.vsplit(ui.scrollable(ui.wrap(config_col)), body,
                                 sizes=[400, 360]), 1)

        self.runner.output.connect(self._on_output)
        self.runner.finished.connect(self._on_finished)
        self.runner.failed.connect(self._on_runner_failed)
        self.pull_runner.output.connect(lambda s: self._append(s, newline=False))
        self.pull_runner.finished.connect(self._on_pull_finished)
        self.pull_runner.failed.connect(self._on_pull_failed)
        self.status_worker.done.connect(self._apply_statuses)
        self.parser.epoch.connect(self._on_epoch)
        self.parser.run_id.connect(self._on_run_id)

        self.apply_exec_mode(True)   # local-only page
        self._rebuild_params()
        self.refresh_images()

    def apply_exec_mode(self, local: bool):
        """Local-only page, kept for main.py's call. Refresh copy and lists."""
        self.sub.setText(
            "Pick a dataset and model. Parameters are pre-filled from density analysis; "
            "edit before launching. Runs locally in Docker on your GPU.")
        self.reload_datasets()

    # ------------------------------------------------- per-model Docker popup
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

        reg_row = QHBoxLayout()
        reg_row.addWidget(QLabel("Registry"))
        self.registry_edit = QLineEdit()
        self.registry_edit.setPlaceholderText("e.g. ghcr.io/gcsgeospatial  (clear = local builds only)")
        self.registry_edit.setText(appstate.local_config().get("registry", ""))
        self.registry_edit.editingFinished.connect(self._on_registry_change)
        reg_row.addWidget(self.registry_edit, 1)
        lay.addLayout(reg_row)

        btn_row = QHBoxLayout()
        self._cfg_pull_btn = QPushButton("Pull")
        self._cfg_pull_btn.clicked.connect(self._pull_current)
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
        self.refresh_images()        # async; _apply_statuses refreshes the dialog
        dlg.show()                   # non-modal: pull progress shows in the main log

    @staticmethod
    def _status_text(s: dict | None):
        """(text, accent-role, pull-enabled) for an image-status dict."""
        if s is None:
            return "checking…", "muted", False
        if not s["docker"]:
            return "docker not found", "muted", False
        if s["present"]:
            return "✓ present", "ok", False
        if s["pullable"]:
            return "✗ not pulled", "warn", True
        return "✗ build it (set a registry to pull)", "warn", False

    def _update_cfg_dialog(self):
        if self._cfg_dialog is None:
            return
        b = self._backbone()
        if b is None:
            return
        self._cfg_model_lbl.setText(b.label)
        self._cfg_tag.setText(f"image: {local_cli.image_for(b)}")
        text, role, can_pull = self._status_text(self._last_statuses.get(b.key))
        self._cfg_status.setText(f"status: {text}")
        theme.set_accent(self._cfg_status, role)
        self._cfg_pull_btn.setEnabled(can_pull and not self.pull_runner.running)

    def refresh_images(self):
        """Re-check image presence off the GUI thread; updates the popup when done."""
        if self.status_worker.running:
            return
        self.status_worker.start(local_cli.all_statuses)

    def _apply_statuses(self, statuses: list):
        self._last_statuses = {s["key"]: s for s in statuses}
        self._update_cfg_dialog()

    def _on_registry_change(self):
        cfg = {**appstate.get("local_config", {}), "registry": self.registry_edit.text().strip()}
        appstate.set_local_config(cfg)
        self.refresh_images()        # registry change affects pullability

    def _pull_current(self):
        b = self._backbone()
        if b is None:
            return
        s = self._last_statuses.get(b.key)
        if not (s and s["docker"] and s["pullable"] and not s["present"]):
            self._append("[local] Nothing to pull - image present or not pullable "
                         "(set a registry; `docker login` if private).")
            return
        if self.pull_runner.running:
            return
        self._cfg_pull_btn.setEnabled(False)
        prog, args = local_cli.pull(b)
        self._append(f"\n[local] $ {local_cli.preview(prog, args)}\n")
        self.pull_runner.start(prog, args, cwd=self.repo_root)

    def _on_pull_finished(self, code: int):
        self._append("[local] ✓ pulled." if code == 0
                     else f"[local] ✗ pull failed (exit {code}). `docker login` if private.")
        self.refresh_images()

    def _on_pull_failed(self, err: str):
        self._append(f"\n[local] ✗ docker pull failed to start: {err}")
        self._update_cfg_dialog()

    # ------------------------------------------------------------- datasets
    def reload_datasets(self):
        current = self.dataset_combo.currentText()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for name in sorted(appstate.selectable_datasets()):
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
        # self._dg_bind_density()   # TODO(not ready): DG UI hidden
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

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(
            self, "Base output folder for runs",
            self.out_edit.text() or str(appstate.workspace_dir()))
        if d:
            self.out_edit.setText(d)

    def _apply_smoke(self):
        """Lock epochs=2, steps=50 while the smoke box is checked."""
        on = self.smoke_chk.isChecked()
        for flag, val in (("epochs", 2), ("steps-per-epoch", 50)):
            w = self._param_widgets.get(flag)
            if w is not None:
                if on:
                    w.setValue(val)
                w.setEnabled(not on)

    def _reload_backbones(self):
        """Populate the model dropdown with every backbone."""
        prev = self.backbone_combo.currentData()
        self.backbone_combo.blockSignals(True)
        self.backbone_combo.clear()
        for key, b in BACKBONES.items():
            self.backbone_combo.addItem(
                b.label + ("" if b.ready else "  (not wired yet)"), key)
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
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_widgets.clear()
        b = self._backbone()
        if b is None:
            self.warn_label.setText("")
            self._update_cfg_dialog()
            return
        recs = (self._meta or {}).get("recommendations", {}).get(b.key, {})
        for spec in b.params:
            value = recs.get(spec.recommend_key, spec.default) if spec.recommend_key else spec.default
            if spec.kind == "float":
                w = ui.NoWheelDoubleSpinBox()
                w.setDecimals(spec.decimals)
                w.setSingleStep(spec.step)
                w.setRange(spec.lo, 1_000_000.0)   # spec.hi is a reco band, not a cap
                w.setValue(float(value))
            else:
                w = ui.NoWheelSpinBox()
                w.setRange(int(spec.lo), 100_000_000)
                w.setValue(int(value))
            label = spec.label + ("  ★" if spec.recommend_key and spec.recommend_key in recs else "")
            self.params_form.addRow(label, w)
            self._param_widgets[spec.flag] = w
        if self._meta:
            warns = analysis.warnings_for(self._meta)
            self.warn_label.setText("\n".join("⚠ " + w for w in warns))
        else:
            self.warn_label.setText("")
        self._apply_smoke()          # re-lock epochs/steps for smoke runs
        self._update_cfg_dialog()    # sync popup with model

    # ------------------------------------------------- domain generalization (per run)
    def _dg_box(self) -> QGroupBox:
        box = QGroupBox("Domain generalization (training) - this run")
        box.setToolTip("Robustness to a different inference point density than trained on. "
                       "Set per run; not saved.")
        lay = QVBoxLayout(box)
        self.dg_train_lbl = QLabel("Pick a dataset to see training density.")
        theme.set_accent(self.dg_train_lbl, "muted")
        lay.addWidget(self.dg_train_lbl)

        row = QHBoxLayout()
        row.addWidget(QLabel("Target inference density (pts/m²):"))
        self.dg_infer = ui.NoWheelDoubleSpinBox()
        self.dg_infer.setRange(0.01, 100000.0)
        self.dg_infer.setDecimals(2)
        self.dg_infer.setValue(2.0)
        row.addWidget(self.dg_infer)
        self.dg_reco_btn = QPushButton("Recommend")
        self.dg_reco_btn.clicked.connect(self._dg_recommend)
        row.addWidget(self.dg_reco_btn)
        row.addStretch(1)
        lay.addLayout(row)

        self.dg_rationale = QLabel("")
        self.dg_rationale.setWordWrap(True)
        theme.set_accent(self.dg_rationale, "muted")
        lay.addWidget(self.dg_rationale)

        self.dg_aug = QCheckBox("Density augmentation - train across the range")
        self.dg_coarsen = ui.NoWheelDoubleSpinBox()
        self.dg_coarsen.setRange(1.0, 6.0)
        self.dg_coarsen.setSingleStep(0.1)
        self.dg_coarsen.setValue(2.5)
        self.dg_pnative = ui.NoWheelDoubleSpinBox()
        self.dg_pnative.setRange(0.0, 1.0)
        self.dg_pnative.setSingleStep(0.05)
        self.dg_pnative.setValue(0.5)
        r1 = QHBoxLayout()
        r1.addWidget(self.dg_aug)
        r1.addWidget(QLabel("coarsen ×"))
        r1.addWidget(self.dg_coarsen)
        r1.addWidget(QLabel("native P"))
        r1.addWidget(self.dg_pnative)
        r1.addStretch(1)
        lay.addLayout(r1)

        self.dg_logdk = QCheckBox("log d_k density channel - changes input dim")
        self.dg_k = ui.NoWheelSpinBox()
        self.dg_k.setRange(1, 64)
        self.dg_k.setValue(8)
        r2 = QHBoxLayout()
        r2.addWidget(self.dg_logdk)
        r2.addWidget(QLabel("k"))
        r2.addWidget(self.dg_k)
        r2.addStretch(1)
        lay.addLayout(r2)

        hint = QLabel("AdaBN & density-TTA are inference-time (no retrain) - set them "
                      "on the Inference page.")
        hint.setWordWrap(True)
        theme.set_accent(hint, "muted")
        lay.addWidget(hint)
        return box

    def _dg_bind_density(self):
        d = float((self._meta or {}).get("stats", {}).get("mean_pts_per_m2") or 0) or None
        if d:
            self.dg_train_lbl.setText(f"Training density: {d:.1f} pts/m²")
        else:
            self.dg_train_lbl.setText("No stored density - set DG features manually "
                                      "(Recommend needs a density).")
        self.dg_reco_btn.setEnabled(bool(d))
        self.dg_rationale.setText("")

    def _dg_recommend(self):
        d = float((self._meta or {}).get("stats", {}).get("mean_pts_per_m2") or 0) or None
        if not d:
            self.dg_rationale.setText("No stored density to recommend from.")
            return
        rec = analysis.dg_recommend(d, self.dg_infer.value())
        self.dg_aug.setChecked(rec["density_aug"])
        self.dg_coarsen.setValue(rec["coarsen_max"])
        self.dg_pnative.setValue(rec["p_native"])
        self.dg_logdk.setChecked(rec["logdk"])
        self.dg_k.setValue(rec["logdk_k"])
        self.dg_rationale.setText(rec["rationale"])

    def _dg_collect(self) -> dict:
        return {"infer_density": round(self.dg_infer.value(), 2),
                "density_aug": self.dg_aug.isChecked(),
                "coarsen_max": round(self.dg_coarsen.value(), 2),
                "p_native": round(self.dg_pnative.value(), 2),
                "logdk": self.dg_logdk.isChecked(),
                "logdk_k": self.dg_k.value()}

    # ------------------------------------------- loss / class balance (per run)
    def _loss_box(self) -> QGroupBox:
        box = QGroupBox("Loss & class balance (advanced)")
        box.setCheckable(True)
        box.setChecked(False)
        outer = QVBoxLayout(box)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(inner)
        box.toggled.connect(inner.setVisible)
        inner.setVisible(False)
        self.loss_box = box      # unchecked = off, use script defaults

        hint = QLabel("Defaults handle class imbalance (inverse-sqrt weighting + "
                      "Lovász-Softmax + rare-class oversampling). Tweak per run; "
                      "recorded in run.json.")
        hint.setWordWrap(True)
        theme.set_accent(hint, "muted")
        lay.addWidget(hint)

        self.loss_focal = QCheckBox("Use focal loss instead of weighted cross-entropy")
        self.loss_focal.setToolTip("Down-weights easy points to focus on hard/rare ones. "
                                   "Off by default (weighted CE + Lovász).")
        self.loss_gamma = ui.NoWheelDoubleSpinBox()
        self.loss_gamma.setRange(0.0, 5.0)
        self.loss_gamma.setSingleStep(0.5)
        self.loss_gamma.setValue(2.0)
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
        self.loss_beta = ui.NoWheelDoubleSpinBox()
        self.loss_beta.setRange(0.0, 1.0)
        self.loss_beta.setSingleStep(0.05)
        self.loss_beta.setValue(0.5)
        self.loss_beta.setToolTip("0 = none, 0.5 = inverse-sqrt (default), "
                                  "1 = full inverse-frequency (most aggressive).")
        self.loss_cw.toggled.connect(self.loss_beta.setEnabled)
        r2 = QHBoxLayout()
        r2.addWidget(self.loss_cw)
        r2.addWidget(QLabel("strength β"))
        r2.addWidget(self.loss_beta)
        r2.addStretch(1)
        lay.addLayout(r2)

        self.loss_rare = QCheckBox("Oversample rare-class tiles (auto-detected)")
        self.loss_rare.setChecked(True)
        lay.addWidget(self.loss_rare)
        return box

    def _loss_collect(self) -> dict:
        if not self.loss_box.isChecked():       # off -> script defaults
            return {}
        return {"focal": self.loss_focal.isChecked(),
                "focal_gamma": round(self.loss_gamma.value(), 2),
                "class_weighting": self.loss_cw.isChecked(),
                "weight_beta": round(self.loss_beta.value(), 2),
                "rare_oversample": self.loss_rare.isChecked()}

    # ------------------------------------------------------------- launch
    def _launch(self):
        b = self._backbone()
        if b is None:
            self._append("Pick a model first.")
            return
        if not b.ready:
            self._append(f"{b.label} isn't wired yet - pick a ready model.")
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
            flags[spec.flag] = self._param_widgets[spec.flag].value()
        if self.smoke_chk.isChecked():
            flags["epochs"] = 2
            flags["steps-per-epoch"] = 50

        dg_env = {}   # TODO(not ready): DG UI hidden; was analysis.dg_config_to_env(self._dg_collect())
        loss_env = analysis.loss_config_to_env(self._loss_collect())
        env = {**dg_env, **loss_env}
        self.log.clear()
        self.metrics_table.setRowCount(0)
        self._last_run_id = None
        self.launch_btn.setEnabled(False)
        if dg_env:
            self._append("[dg] on: "
                         + " ".join(f"{k}={v}" for k, v in sorted(dg_env.items())))
        if loss_env:
            self._append("[loss] overrides: "
                         + " ".join(f"{k}={v}" for k, v in sorted(loss_env.items())))
        self._start_local_run({"backbone": b, "flags": flags, "dataset": name,
                               "env": env, "info": info})

    def _start_local_run(self, p):
        b, flags, name, info = p["backbone"], p["flags"], p["dataset"], p["info"]
        staged = info.get("staged_dir", "")
        # Runs nest per dataset: bind <base>/<dataset> -> /outputs so the container's
        # /outputs/runs/<id> lands at <base>/<dataset>/runs/<id>. Base defaults to the
        # workspace (shown in the field), so for a workspace dataset that's
        # <workspace>/<dataset>/runs, right beside its data.
        base = self.out_edit.text().strip() or str(appstate.workspace_dir())
        out_root = str(Path(base) / name)
        os.makedirs(out_root, exist_ok=True)
        appstate.put("local_train_out", self.out_edit.text().strip())
        extra_mounts = []
        if not (staged and os.path.isdir(staged)):
            self._append(f"[local] ⚠ No staged copy of '{name}' - "
                         f"container won't find /datasets/{name}.")
        elif Path(staged).parent != Path(appstate.local_config()["datasets_root"]):
            # Dataset lives outside the workspace (pre-existing/relocated); the base
            # /datasets mount won't expose it, so bind it explicitly. A nested dataset
            # needs no extra mount — base /datasets = workspace already covers it.
            extra_mounts.append((staged, f"/datasets/{name}"))
        prog, args = local_cli.run_script(b.script, flags, b, repo_root=self.repo_root,
                                          extra_mounts=extra_mounts, outputs_root=out_root,
                                          env=p.get("env", {}))
        self._append(f"\n[local] $ {local_cli.preview(prog, args)}\n")
        if not local_cli.have_docker():
            self._append(
                "[local] docker not found on PATH - printed the command instead of running "
                "it. On a Docker+GPU host, build the images (docker/build_all), then launch: "
                f"training writes to {out_root}/runs/<id>.")
            self.launch_btn.setEnabled(True)
            return
        ok_gpu, msg_gpu = local_cli.gpu_preflight()
        if msg_gpu:
            self._append(msg_gpu)
        if not ok_gpu:
            self.launch_btn.setEnabled(True)
            return
        ok, msg = local_cli.image_preflight(b)
        if msg:
            self._append(msg)
        if not ok:
            self.launch_btn.setEnabled(True)
            return
        self.runner.start(prog, args, cwd=self.repo_root)

    def _on_runner_failed(self, err: str):
        # FailedToStart fires failed not finished; re-enable here.
        self.launch_btn.setEnabled(True)
        self._append(f"\n✗ Failed to start: {err}")

    def _stop(self):
        if self.runner.running:
            self.runner.terminate()
            self._append("\n[stopped]")
        else:
            self._append("\n[no process running]")

    # ------------------------------------------------------------- stream
    def _on_output(self, text: str):
        self._append(text, newline=False)
        self.parser.feed(text)

    def _on_finished(self, code: int):
        self.launch_btn.setEnabled(True)
        if code == 0:
            extra = (f" Run id: {self._last_run_id}." if self._last_run_id else "")
            self._append(f"\n✓ Done.{extra} See the Runs/Plotting page for artifacts.")
        else:
            self._append(f"\n✗ Exited with code {code}.")

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
        b = self._backbone()
        history.append({"run_id": run_id, "backbone": b.key if b else "",
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
