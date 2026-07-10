"""Train page: dataset + model + params -> a training run, with live logs and
epoch metrics. Two backends, chosen by the sidebar's Execution backend switch:
local (docker run on your GPU; output folder on the host) and Modal (modal run
on a cloud GPU; dataset read from / run written to Modal volumes). Dataset
check verifies the train/val/test standard; per-model image status/pull live
in a popup. Loss / class-balance knobs reach the trainer as env either way
(docker -e locally, --env-json through the modal shell in the cloud).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPlainTextEdit, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import analysis, appstate, local_cli, modal_cli, theme, ui
from ..backbones import BACKBONES, GPU_CHOICES
from ..jobs import FuncWorker, JobRunner, LogParser

# Input-feature picker: the arch-appropriate standard channel names per base
# backbone (offered in the list) and each trainer's exact legacy default
# (pre-checked). Mirrors the local_train_* scripts' FEAT_LEGACY lists; the
# PTv3 default color entry follows the dataset (intensity-first), see
# _rebuild_feat_list. Dataset feat_* channels are appended unchecked.
_FEAT_STANDARD = {
    "randlanet":    ["x", "y", "z", "intensity", "return_number"],
    "kpconvx_cold": ["intensity", "return_number", "height", "x", "y", "z"],
    "kpconv":       ["intensity", "return_number", "height", "x", "y", "z"],
    "ptv3":         ["x", "y", "z", "intensity", "rgb"],
}
_FEAT_DEFAULTS = {
    "randlanet":    ["x", "y", "z", "intensity", "return_number"],
    "kpconvx_cold": ["intensity", "return_number", "height"],
    "kpconv":       ["intensity", "return_number", "height"],
    "ptv3":         ["x", "y", "z", "intensity"],   # color slot swapped per dataset
}


class TrainPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.runner = JobRunner(self)          # training process (docker or modal run)
        self.pull_runner = JobRunner(self)     # docker pull, streamed to log
        self.status_worker = FuncWorker(self)  # off-thread image-presence check
        self.modal_worker = FuncWorker(self)   # off-thread modal volume preflight
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
        form.addRow("Model", ui.wrap(model_row))
        self.smoke_chk = QCheckBox("Smoke run (2 epochs × 50 steps)")
        self.smoke_chk.toggled.connect(self._apply_smoke)
        form.addRow("Options", self.smoke_chk)
        # Modal-only: detach = `modal run --detach`, returns as soon as the cloud
        # app starts, freeing this page to launch the next model — the way to
        # train several models on one dataset in parallel.
        self.detach_chk = QCheckBox("Detach (return immediately — launch several models in parallel)")
        self.detach_chk.setToolTip("Runs in the cloud without streaming logs here. Reattach with "
                                   "`modal app logs <app>`; the run id appears under runs/ on the "
                                   "model's outputs volume (the Inference page Run field is editable).")
        form.addRow("", self.detach_chk)
        # Modal-only: cloud GPU type (TT_GPU env, read by the modal shell at launch).
        self.gpu_combo = QComboBox()
        for g in GPU_CHOICES:
            self.gpu_combo.addItem(g)
        self.gpu_combo.setToolTip("Cloud GPU for this run (Modal only). Defaults to the "
                                  "model's recommendation when you switch models.")
        form.addRow("GPU (Modal)", self.gpu_combo)
        self.backbone_combo.currentIndexChanged.connect(self._sync_gpu_default)

        self.params_box = QGroupBox("Parameters (pre-filled)")
        self.params_form = QFormLayout(self.params_box)
        # Validation cadence (VAL_EVERY env; trainer default 10 emits nothing).
        self.val_every = QSpinBox()
        self.val_every.setRange(1, 100)
        self.val_every.setValue(10)
        self.val_every.setToolTip(
            "Run the held-out validation pass every N epochs. Lower N = slower "
            "training but finer best-checkpoint selection (the best model is "
            "picked among validated epochs only); higher N = faster, coarser.")
        val_row = QHBoxLayout()
        val_row.addWidget(QLabel("Validate every N epochs"))
        val_row.addWidget(self.val_every)
        val_row.addStretch(1)
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
        config_col.addLayout(val_row)
        # TODO(not ready): train-time DG UI disabled pending review; no DG env is
        # sent. analysis.dg_recommend/dg_config_to_env remain the backend hooks.
        config_col.addWidget(self._loss_box())
        config_col.addWidget(self._features_box())
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
        self.refresh_images()

    def apply_exec_mode(self, local: bool):
        """Re-scheme for the chosen backend: hide Docker-only / Modal-only rows."""
        self.sub.setText(
            "Pick a dataset and model. Parameters are pre-filled from density analysis; "
            "edit before launching. "
            + ("Runs locally in Docker on your GPU."
               if local else
               "Runs on a Modal cloud GPU — upload the dataset from the Datasets page "
               "first; the finished run lands on the model's outputs volume."))
        self.cfg_btn.setVisible(local)                      # docker image mgmt
        self.form.setRowVisible(self.gpu_combo, not local)  # cloud GPU pick
        self.form.setRowVisible(self.detach_chk, not local) # parallel cloud launches
        self.reload_datasets()

    def _sync_gpu_default(self):
        b = self._backbone()
        if b is not None:
            i = self.gpu_combo.findText(b.rec_gpu)
            if i >= 0:
                self.gpu_combo.setCurrentIndex(i)

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
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_widgets.clear()
        b = self._backbone()
        if b is None:
            self.warn_label.setText("")
            self._rebuild_feat_list()
            self._update_cfg_dialog()
            return
        recs = (self._meta or {}).get("recommendations", {}).get(b.key, {})
        for spec in b.params:
            value = recs.get(spec.recommend_key, spec.default) if spec.recommend_key else spec.default
            if spec.kind == "float":
                w = QDoubleSpinBox()
                w.setDecimals(spec.decimals)
                w.setSingleStep(spec.step)
                w.setRange(spec.lo, 1_000_000.0)   # spec.hi is a reco band, not a cap
                w.setValue(float(value))
            else:
                w = QSpinBox()
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
        self._rebuild_feat_list()    # feature picker follows model + dataset
        self._update_cfg_dialog()    # sync popup with model

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
        return box

    def _loss_collect(self) -> dict:
        if not self.loss_box.isChecked():       # off -> script defaults
            return {}
        return {"focal": self.loss_focal.isChecked(),
                "focal_gamma": round(self.loss_gamma.value(), 2),
                "class_weighting": self.loss_cw.isChecked(),
                "weight_beta": round(self.loss_beta.value(), 2),
                "rare_oversample": self.loss_rare.isChecked()}

    # -------------------------------------------- input features (per run)
    def _features_box(self) -> QGroupBox:
        box = QGroupBox("Input features (advanced)")
        box.setCheckable(True)
        box.setChecked(False)
        outer = QVBoxLayout(box)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(inner)
        box.toggled.connect(inner.setVisible)
        inner.setVisible(False)
        self.feat_box = box      # unchecked = off, use the trainer's legacy defaults

        hint = QLabel("Check the channels the model eats, drag to reorder "
                      "(top-to-bottom = channel order). Unchecked box = the "
                      "model's built-in defaults. Sent as FEAT_CHANNELS; "
                      "recorded in run.json.")
        hint.setWordWrap(True)
        theme.set_accent(hint, "muted")
        lay.addWidget(hint)

        self.feat_list = QListWidget()
        self.feat_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.feat_list.setDefaultDropAction(Qt.MoveAction)
        lay.addWidget(self.feat_list)
        return box

    def _rebuild_feat_list(self):
        """Repopulate the checklist for the selected backbone + dataset: the
        arch's standard names (legacy defaults pre-checked) plus the dataset's
        meta feature_channels as feat_* entries (unchecked)."""
        if not hasattr(self, "feat_list"):
            return
        self.feat_list.clear()
        b = self._backbone()
        if b is None:
            return
        base = b.key[:-4] if b.key.endswith("_hag") else b.key
        std = _FEAT_STANDARD.get(base, [])
        defaults = list(_FEAT_DEFAULTS.get(base, std))
        meta = self._meta or {}
        if base == "ptv3" and not meta.get("has_intensity", True) \
                and meta.get("has_rgb"):
            defaults = ["x", "y", "z", "rgb"]   # the trainer's rgb-dataset legacy
        extra = ((meta.get("source") or {}).get("feature_channels")
                 if isinstance(meta.get("source"), dict) else None) or []
        names = std + [n if str(n).startswith("feat_") else f"feat_{n}"
                       for n in extra]
        for n in names:
            it = QListWidgetItem(str(n))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if n in defaults else Qt.Unchecked)
            self.feat_list.addItem(it)
        tip = ("Channel spec for this run (FEAT_CHANNELS). feat_* entries are "
               "the dataset's extra feature channels.")
        if b.key.endswith("_hag"):
            tip += ("\nHAG is not in this list: it rides separately via the "
                    "_hag model choice and is appended after these channels.")
        self.feat_box.setToolTip(tip)

    def _feat_collect(self) -> str:
        """Checked names in list order -> FEAT_CHANNELS csv; '' = don't emit
        (group off, or nothing checked -> trainer legacy defaults)."""
        if not self.feat_box.isChecked():
            return ""
        names = [self.feat_list.item(i).text()
                 for i in range(self.feat_list.count())
                 if self.feat_list.item(i).checkState() == Qt.Checked]
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
            flags[spec.flag] = self._param_widgets[spec.flag].value()
        if self.smoke_chk.isChecked():
            flags["epochs"] = 2
            flags["steps-per-epoch"] = 50

        loss_env = analysis.loss_config_to_env(self._loss_collect())
        env = dict(loss_env)
        if self.val_every.value() != 10:   # default emits nothing (script default)
            env["VAL_EVERY"] = str(self.val_every.value())
        feat_csv = self._feat_collect()    # '' = no env (trainer legacy defaults)
        if feat_csv:
            env["FEAT_CHANNELS"] = feat_csv
        self.log.clear()
        self.metrics_table.setRowCount(0)
        self._last_run_id = None
        self.launch_btn.setEnabled(False)
        if loss_env:
            self._append("[loss] overrides: "
                         + " ".join(f"{k}={v}" for k, v in sorted(loss_env.items())))
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
        # Runs nest per dataset in the workspace: bind <workspace>/<dataset> ->
        # /outputs so the container's /outputs/runs/<id> lands at
        # <workspace>/<dataset>/runs/<id>, right beside the dataset's data.
        base = str(appstate.workspace_dir())
        out_root = str(Path(base) / name)
        os.makedirs(out_root, exist_ok=True)
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

    # ------------------------------------------------------------- modal path
    def _preflight_modal_run(self, p):
        """Check the dataset is on the datasets volume before paying for a GPU
        container that would just print 'No training tiles'. Off-thread — the
        `modal volume ls` call can take seconds."""
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
        # Can't list (auth hiccup, flaky network) — warn but let the run proceed;
        # the trainer itself fails fast and clearly if the dataset is missing.
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

    def _on_runner_failed(self, err: str):
        # FailedToStart fires failed not finished; re-enable here.
        self.launch_btn.setEnabled(True)
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

    def _append(self, text: str, newline: bool = True):
        ui.append_log(self.log, text, newline)
