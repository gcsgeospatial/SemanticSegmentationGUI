"""Inference page: pick weights + input folder, run --mode infer.

Stages (one JobRunner):
  Modal: convert -> upload scenes -> [upload weights] -> run -> download.
  Local: convert -> pixi run (TT_* env points scenes + predictions at host dirs).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QPushButton, QRadioButton,
                               QSpinBox, QVBoxLayout, QWidget)

from .. import analysis, appstate, dataset, local_cli, modal_cli, plots, pretrain, ui
from ..backbones import BACKBONES
from ..jobs import FuncWorker, JobRunner
from ..logconsole import LogConsole


class InferPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.converter = FuncWorker(self)
        self.preflight = FuncWorker(self)
        self.exporter = FuncWorker(self)   # npz -> chosen prediction format (host-side)
        self.runner = JobRunner(self)
        self._stage = ""
        self._job_id = ""
        self._staged: Path | None = None
        self._weights_remote = ""
        self._dl_dest: Path | None = None
        self._pred_dir: Path | None = None   # where local predictions land (host)
        self._manifest: dict | None = None         # the picked run.json (local runs)
        self._manifest_path: Path | None = None
        self._local_weights: Path | None = None    # weights = the run.json's sibling
        self._dg: dict = {}                         # DG settings baked into the weights (run.json["dg"])
        self._run_class_names: list | None = None   # the loaded run's own classes (run.json)
        self._manifest_features: list | None = None  # run.json "features" (None = legacy run)
        self._hag_ground_value: int | None = None   # parsed in _check_hag, used by the converter
        self._modal_cfg_run = ""                    # run id whose fetched config is applied
        self._run_tag = ""                          # weights' run id — prediction parent dir
        self._pending_cfg_run = ""                  # run id the cfg_fetcher is out for
        # Ensemble (Phase 3): captured members + the run-time member queue state.
        self._ens_members: list[dict] = []
        self._ens_running = False                   # True from launch until export/failure
        self._ens_idx = -1                          # index of the member currently running
        self._ens_dirs: list[Path] = []             # completed members' prediction dirs
        self._last_pred_dir: Path | None = None     # last finished predictions dir ('Plot this run')
        self._run_open = False                      # a begin_run header awaits its end_run

        root = QVBoxLayout(self)
        title = QLabel("Inference")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.sub.setObjectName("pageSub")
        root.addWidget(self.sub)

        wbox = QGroupBox("Weights")
        wf = self.wf = QFormLayout(wbox)
        # Fields fill width so Browse… buttons aren't clipped.
        wf.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        radio_row = QHBoxLayout()
        self.from_run_radio = QRadioButton("Training run")
        self.from_run_radio.setChecked(True)
        self.from_file_radio = QRadioButton("Local .pth file")
        self.from_run_radio.toggled.connect(self._on_source_toggle)
        radio_row.addWidget(self.from_run_radio)
        radio_row.addWidget(self.from_file_radio)
        radio_row.addStretch()
        wf.addRow("Source", ui.wrap(radio_row))
        # LOCAL: pick run.json — arch, grid, tile, intensity, HAG, weights path come from it.
        self.runjson_edit = QLineEdit()
        self.runjson_edit.setPlaceholderText("…/local_runs/runs/<id>/run.json")
        # Load on Enter/focus-out; any edit drops the stale load.
        self.runjson_edit.editingFinished.connect(self._load_run_manifest)
        self.runjson_edit.textChanged.connect(self._invalidate_manifest)
        rj_row = QHBoxLayout()
        rj_row.addWidget(self.runjson_edit, 1)   # edit absorbs slack; button keeps its size
        rj_btn = QPushButton("Browse…")
        rj_btn.clicked.connect(self._pick_runjson)
        rj_row.addWidget(rj_btn)
        self.runjson_row_w = ui.wrap(rj_row)
        wf.addRow("Run file (run.json)", self.runjson_row_w)
        # Weights default to the .pth named in run.json; can point anywhere.
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("default: the .pth named in run.json, beside it")
        w_row = QHBoxLayout()
        w_row.addWidget(self.weights_edit, 1)
        w_btn = QPushButton("Browse…")
        w_btn.clicked.connect(self._pick_weights)
        w_row.addWidget(w_btn)
        self.weights_row_w = ui.wrap(w_row)
        wf.addRow("Weights file", self.weights_row_w)
        # MODAL: pick/paste a run id (weights live on the cloud outputs volume).
        self.run_combo = QComboBox()
        self.run_combo.setEditable(True)   # run ids can also be typed/pasted
        self.run_combo.lineEdit().setPlaceholderText(
            "run id — or paste <volume>/runs/<id> straight from the train log")
        self.run_combo.currentIndexChanged.connect(self._on_run_pick)
        # A typed/pasted id pulls the run's config off the outputs volume.
        self.run_combo.lineEdit().editingFinished.connect(self._on_run_id_typed)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_combo, 1)
        self.dl_run_btn = QPushButton("Download run…")
        self.dl_run_btn.setToolTip("Fetch runs/<id> (weights + run.json + metrics) from the "
                                   "model's outputs volume to <workspace>/inference/<id> — "
                                   "for backup, local inference later, or the class names.")
        self.dl_run_btn.clicked.connect(self._download_run)
        run_row.addWidget(self.dl_run_btn)
        self.run_row_w = ui.wrap(run_row)
        wf.addRow("Run", self.run_row_w)
        self.pth_edit = QLineEdit()
        pth_row = QHBoxLayout()
        pth_row.addWidget(self.pth_edit, 1)
        pth_btn = QPushButton("Browse…")
        pth_btn.clicked.connect(self._pick_pth)
        pth_row.addWidget(pth_btn)
        inst_btn = QPushButton("Installed…")
        inst_btn.setToolTip("Weights installed as trainer-weights-* conda packages "
                            "in this model's pixi env")
        inst_btn.clicked.connect(self._pick_installed_weights)
        pth_row.addWidget(inst_btn)
        self.pth_row_w = ui.wrap(pth_row)
        wf.addRow("File", self.pth_row_w)
        # Folded manifest echo: while a run manifest is applied, the locked
        # Architecture / Grid / Tile rows hide and this one muted line stands in
        # (see _apply_manifest_lock).
        self.manifest_summary = QLabel("")
        self.manifest_summary.setObjectName("pageSub")
        self.manifest_summary.setWordWrap(True)
        wf.addRow(self.manifest_summary)
        self.backbone_combo = QComboBox()
        self.backbone_combo.currentIndexChanged.connect(self._sync_controls)
        # populated at the end of __init__ (reload_backbones needs grid_spin etc.)
        wf.addRow("Architecture", self.backbone_combo)
        # Ensemble (Phase 3): vote over several runs. Members are captured from the
        # normal weights inputs above ('Add current selection'), then run
        # sequentially over the once-staged scenes and soft-voted.
        # ponytail: LOCAL backend only — the per-member Modal download loop isn't
        # wired through the stage machine; the group is disabled in modal mode
        # (tooltip says so) rather than half-supported.
        self.ens_box = QGroupBox("Ensemble (vote over several runs)")
        self.ens_box.setCheckable(True)
        self.ens_box.setChecked(False)
        ecol = QVBoxLayout(self.ens_box)
        self.ens_list = QListWidget()
        self.ens_list.setMaximumHeight(96)
        ecol.addWidget(self.ens_list)
        erow = QHBoxLayout()
        ens_add = QPushButton("Add current selection")
        ens_add.setToolTip("Capture the model configured above (backbone + weights + "
                           "run.json) as an ensemble member.")
        ens_add.clicked.connect(self._add_ens_member)
        ens_del = QPushButton("Remove selected")
        ens_del.clicked.connect(self._remove_ens_member)
        erow.addWidget(ens_add)
        erow.addWidget(ens_del)
        erow.addStretch()
        ecol.addLayout(erow)
        ens_hint = QLabel("3+ models recommended; runtime scales with model count.")
        ens_hint.setObjectName("pageSub")   # muted, same style as the page subtitle
        ecol.addWidget(ens_hint)
        wf.addRow(self.ens_box)
        # Modal: the group above stays visible-but-disabled; this row lives
        # OUTSIDE the disabled group so its button remains clickable.
        ens_modal_hint = QLabel("Ensembles run locally — switch to Local mode to enable")
        ens_modal_hint.setObjectName("pageSub")
        self.ens_switch_btn = QPushButton("Switch to Local")
        self.ens_switch_btn.clicked.connect(self._switch_to_local)
        eh_row = QHBoxLayout()
        eh_row.addWidget(ens_modal_hint)
        eh_row.addWidget(self.ens_switch_btn)
        eh_row.addStretch()
        self.ens_hint_w = ui.wrap(eh_row)
        wf.addRow(self.ens_hint_w)

        ibox = QGroupBox("Input")
        iform = self.iform = QFormLayout(ibox)
        iform.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.input_edit = QLineEdit()
        in_row = QHBoxLayout()
        in_row.addWidget(self.input_edit, 1)   # edit absorbs slack; both buttons keep their size
        fold_btn = QPushButton("Folder…")
        fold_btn.clicked.connect(self._pick_input)
        file_btn = QPushButton("File…")
        file_btn.clicked.connect(self._pick_input_file)
        in_row.addWidget(fold_btn)
        in_row.addWidget(file_btn)
        iform.addRow("Point clouds (folder or file)", ui.wrap(in_row))
        self.grid_spin = QDoubleSpinBox()
        self.grid_spin.setRange(0.02, 1_000_000.0)
        self.grid_spin.setSingleStep(0.05)
        self.grid_spin.setDecimals(2)
        self.grid_spin.setValue(0.30)
        iform.addRow("Grid size (m) - from the run", self.grid_spin)
        # Intensity is p95-normalized end-to-end; nothing to match here.
        self.chunk_spin = QDoubleSpinBox()
        self.chunk_spin.setRange(10.0, 1_000_000.0)
        self.chunk_spin.setSingleStep(5.0)
        self.chunk_spin.setDecimals(0)
        self.chunk_spin.setValue(50.0)
        iform.addRow("Tile size (m)", self.chunk_spin)
        # HAG is opt-in: computing it costs a full ground-detection pass. Loading
        # a run whose features include feat_hag ticks this for you.
        self.hag_chk = QCheckBox("Compute Height-Above-Ground (HAG)")
        self.hag_chk.setToolTip("Bakes a per-point feat_hag channel into each converted "
                                "scene — required by runs trained with feat_hag. Ground "
                                "comes from the class named below when you set one, else "
                                "it's detected.")
        self.hag_chk.toggled.connect(lambda on: self.hag_opts_w.setVisible(on))
        iform.addRow("Height-Above-Ground", self.hag_chk)
        self.hag_filter = QComboBox()
        self.hag_filter.addItems(list(pretrain.HAG_METHODS))
        self.hag_filter.setToolTip("How HAG is interpolated from the ground points. "
                                   "grid: fast raster approximation, no PDAL needed. "
                                   "hag_nn / hag_delaunay: accurate PDAL filters.")
        self.hag_ground = QLineEdit()
        self.hag_ground.setPlaceholderText("blank = detect (SMRF)")
        self.hag_ground.setMaximumWidth(90)
        self.hag_ground.setToolTip("Classification value in the input clouds that means "
                                   "ground (e.g. 2). When set, those points are the ONLY "
                                   "ground source — never mixed with detection. Blank = "
                                   "SMRF detects ground instead (needs PDAL; without it "
                                   "the grid method's own heuristic is the fallback).")
        hag_row = QHBoxLayout()
        hag_row.addWidget(QLabel("method"))
        hag_row.addWidget(self.hag_filter)
        hag_row.addWidget(QLabel("ground class"))
        hag_row.addWidget(self.hag_ground)
        hag_row.addStretch()
        self.hag_opts_w = ui.wrap(hag_row)
        self.hag_opts_w.setVisible(False)
        iform.addRow("", self.hag_opts_w)
        if not pretrain.pdal_available():
            # grid still works without PDAL; convert_infer_job falls back to grid
            # if a PDAL filter is picked anyway.
            self.hag_chk.setText("Compute Height-Above-Ground (HAG) - grid only, "
                                 "PDAL not installed")
        # Label-free domain-adaptation knobs + probability export. Env-driven
        # (DG_INFER_ADABN / DG_INFER_TTA / TT_SAVE_PROBS); backend details in
        # scripts/DENSITY_DG.md.
        self.adabn_chk = QCheckBox("AdaBN - recalibrate BatchNorm on these scenes")
        self.adabn_chk.setToolTip(
            "Recomputes BatchNorm statistics on the target tiles (label-free) before "
            "predicting. KPConvX / RandLA only - the PTv3 family ignores it.\n"
            "Output depends on this job's density and class mix, so the same model "
            "can score differently per area; note it when comparing runs.")
        self.tta_spin = QSpinBox()
        self.tta_spin.setRange(0, 5)
        self.tta_spin.setSuffix(" views")
        self.tta_spin.setToolTip(
            "Extra density/scale views averaged into each tile's prediction "
            "(DG_INFER_TTA). 0 = off. Each view adds a full pass over the tile, "
            "so inference time scales with the count.")
        self.probs_chk = QCheckBox("Save class probabilities")
        self.probs_chk.setToolTip(
            "Store the full per-point class distribution (float16) in each "
            "prediction npz (TT_SAVE_PROBS). Needed for soft ensemble voting and "
            "offline confidence/mask analysis; costs ~2 bytes x classes per point.")
        dg_row = QHBoxLayout()
        dg_row.addWidget(self.adabn_chk)
        dg_row.addWidget(QLabel("TTA"))
        dg_row.addWidget(self.tta_spin)
        dg_row.addWidget(self.probs_chk)
        dg_row.addStretch()
        iform.addRow("Domain adaptation", ui.wrap(dg_row))

        # Per-job class mask: untick classes absent from these scenes (e.g. water).
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(96)
        self.class_list.setToolTip(
            "Untick classes that don't exist in these scenes. Their probability is "
            "zeroed after vote accumulation and the rest renormalized, so each point "
            "falls to its next-best class; exported confidence (and the low-confidence "
            "gate above) are post-mask. Recorded in infer_run.json.")
        self.class_list.itemChanged.connect(self._sync_class_mask_label)
        self.class_mask_lbl = QLabel("")
        self.class_mask_lbl.setObjectName("pageSub")
        ccol = QVBoxLayout()
        ccol.addWidget(self.class_list)
        ccol.addWidget(self.class_mask_lbl)
        iform.addRow("Classes (mask at launch)", ui.wrap(ccol))
        self._rebuild_class_list()   # starts disabled with a 'load a run' placeholder

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run inference")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        run_row.addWidget(self.run_btn)
        run_row.addStretch()

        forms_col = QVBoxLayout()
        forms_col.addWidget(wbox)
        forms_col.addWidget(ibox)
        forms_col.addLayout(run_row)
        forms_col.addStretch()

        self.log = LogConsole()   # \r-aware, colored console (drop-in for the old QPlainTextEdit)
        self.log.setPlaceholderText("Conversion and run logs appear here…")

        # After-the-run knobs: format + the confidence gate only shape the EXPORT
        # step (the .npz keeps raw predictions — re-export never re-runs
        # inference), so they live with the post-run actions, not the launch
        # decisions above.
        abox = QGroupBox("After the run")
        aform = QFormLayout(abox)
        aform.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        # Predictions always land under <workspace>/inference — the same spot
        # 'Download run…' uses, so runs + their predictions co-locate.
        # Prediction file format — every option is xyz + classification only
        # (no RGB colour columns: the deliverable carries classes, not colours).
        self.fmt_combo = QComboBox()
        for label, key in (("LAS (.las)", "las"), ("LAZ (.laz)", "laz"),
                           ("PLY (.ply)", "ply"), ("Text (.txt)", "txt"),
                           ("CSV (.csv)", "csv")):
            self.fmt_combo.addItem(label, key)
        i = self.fmt_combo.findData(appstate.get("infer_format", "las"))
        self.fmt_combo.setCurrentIndex(i if i >= 0 else 0)
        self.fmt_combo.setToolTip("Predictions are written as xyz + classification "
                                  "(no colour columns).")
        aform.addRow("Prediction format", self.fmt_combo)
        # Confidence gate at export: low-confidence points become ASPRS class 1.
        self.unclass_chk = QCheckBox("Mark low-confidence points Unclassified")
        self.unclass_chk.setChecked(True)
        self.unclass_spin = QDoubleSpinBox()
        self.unclass_spin.setRange(0.0, 1.0)
        self.unclass_spin.setSingleStep(0.05)
        self.unclass_spin.setDecimals(2)
        self.unclass_spin.setValue(0.50)
        tip = ("Raw max-softmax confidence: points below the cut export as ASPRS "
               "class 1 (Unclassified — processed, no class assigned). The .npz "
               "keeps the raw prediction, so re-exporting at a new threshold "
               "never re-runs inference.")
        self.unclass_chk.setToolTip(tip)
        self.unclass_spin.setToolTip(tip)
        self.unclass_chk.toggled.connect(self.unclass_spin.setEnabled)
        uc_row = QHBoxLayout()
        uc_row.addWidget(self.unclass_chk)
        uc_row.addWidget(self.unclass_spin)
        uc_row.addStretch()
        aform.addRow("Confidence", ui.wrap(uc_row))

        # Action bar; comparison metrics print to the log.
        self.compare_btn = QPushButton("Compare to ground truth…")
        self.compare_btn.setToolTip("Pick a prediction + its ground truth; accuracy, "
                                    "mIoU and per-class IoU print to the log.")
        self.compare_btn.clicked.connect(self._compare_gt)
        self.plot_btn = QPushButton("Plot this run")
        self.plot_btn.setEnabled(False)   # enabled once a run exports / compares
        self.plot_btn.setToolTip("Open the Plotting page for the last finished "
                                 "inference or comparison.")
        self.plot_btn.clicked.connect(self._plot_run)
        actions = QHBoxLayout()
        actions.addWidget(self.compare_btn)
        actions.addWidget(self.plot_btn)
        actions.addStretch(1)
        out_box = QVBoxLayout()
        out_box.addWidget(abox)
        out_box.addLayout(actions)

        root.addWidget(ui.vsplit(ui.wrap(forms_col), self.log,
                                 ui.wrap(out_box), sizes=[340, 340, 84]), 1)

        self.converter.output.connect(self._append)
        self.converter.done.connect(self._on_converted)
        self.converter.error.connect(self._on_error)
        self.preflight.output.connect(self._append)
        self.preflight.done.connect(self._on_preflight)
        self.preflight.error.connect(self._on_preflight_error)
        self.exporter.output.connect(self._append)
        self.exporter.done.connect(self._on_exported)
        self.exporter.error.connect(self._on_export_error)
        self.voter = FuncWorker(self)   # ensemble vote over the member prediction dirs
        self.voter.output.connect(self._append)
        self.voter.done.connect(self._on_voted)
        self.voter.error.connect(self._on_vote_error)
        self.cfg_fetcher = FuncWorker(self)   # modal run.json fetch for a typed run id
        self.cfg_fetcher.output.connect(self._append)
        self.cfg_fetcher.done.connect(self._on_cfg_fetched)
        self.cfg_fetcher.error.connect(
            lambda tb: self._append(f"✗ Run-config fetch failed:\n{tb}"))
        self.runner.output.connect(self._on_output)
        self.runner.finished.connect(self._on_stage_done)
        self.runner.failed.connect(self._on_runner_failed)
        self.dl_runner = JobRunner(self)   # run download, independent of the infer stages
        self.dl_runner.output.connect(lambda s: self._append(s, newline=False))
        self.dl_runner.finished.connect(self._on_run_downloaded)
        self.dl_runner.failed.connect(
            lambda e: self._append(f"✗ Run download failed to start: {e}"))

        self.reload_backbones()
        self.reload_runs()
        self._on_source_toggle()
        self.apply_exec_mode(appstate.get_exec_mode() == "local")

    def apply_exec_mode(self, local: bool):
        """Reword copy for the backend, refresh the env-install marks."""
        if self._manifest:
            self._invalidate_manifest()   # a run's config doesn't survive a backend switch
        self.sub.setText(
            "Label point clouds with a trained model. "
            + ("Pick a run.json (or a local .pth), a folder of clouds, and run locally."
               if local else
               "Pick a run (or a local .pth), a folder of clouds, and run on Modal."))
        # ponytail: ensemble ships local-only — modal would need a per-member
        # download loop through the stage machine; disabled with a tooltip instead.
        self.ens_box.setEnabled(local)
        self.ens_box.setToolTip(
            "" if local else "Ensemble runs on the LOCAL backend only — the "
            "per-member Modal download loop isn't wired. Switch execution to "
            "local on the Settings page to use it.")
        if not local:
            self.ens_box.setChecked(False)
        self.wf.setRowVisible(self.ens_hint_w, not local)   # why it's greyed + the fix
        self._sync_source_rows()
        self.reload_backbones()

    def _switch_to_local(self):
        """Flip the execution backend to local. Prefer the sidebar combo (keeps
        it in sync and re-schemes every page via main's _on_mode_change); a
        standalone page falls back to the appstate setter + its own refresh."""
        combo = getattr(self.window(), "mode_combo", None)
        if isinstance(combo, QComboBox):
            i = combo.findData("local")
            if i >= 0:
                combo.setCurrentIndex(i)
                return
        appstate.set_exec_mode("local")
        self.apply_exec_mode(True)

    def _sync_source_rows(self):
        """Show only the weights inputs that match the source radio + backend:
        Training run -> the run box (Modal) or run.json + weights rows (local);
        Local .pth file -> just the File box."""
        local = appstate.get_exec_mode() == "local"
        from_run = self.from_run_radio.isChecked()
        self.wf.setRowVisible(self.runjson_row_w, from_run and local)
        self.wf.setRowVisible(self.weights_row_w, from_run and local)
        self.wf.setRowVisible(self.run_row_w, from_run and not local)
        self.wf.setRowVisible(self.pth_row_w, not from_run)

    def reload_backbones(self):
        """Populate the model dropdown (every backbone) with env-install marks."""
        prev = self.backbone_combo.currentData()
        self.backbone_combo.blockSignals(True)
        self.backbone_combo.clear()
        for key, b in BACKBONES.items():
            self.backbone_combo.addItem(b.label, key)
        i = self.backbone_combo.findData(prev)
        if i >= 0:
            self.backbone_combo.setCurrentIndex(i)
        self.backbone_combo.blockSignals(False)
        self._refresh_env_marks()
        self._sync_controls()

    def _refresh_env_marks(self):
        """Local mode: mark models whose pixi env isn't installed yet, so the
        block at Run time isn't a surprise. Pure directory scan per backbone."""
        local = appstate.get_exec_mode() == "local"
        for i in range(self.backbone_combo.count()):
            b = BACKBONES.get(self.backbone_combo.itemData(i))
            if b is None:
                continue
            missing = local and not local_cli.installed(b, self.repo_root)
            self.backbone_combo.setItemText(
                i, b.label + ("  — env not installed" if missing else ""))

    def showEvent(self, ev):
        super().showEvent(ev)
        self._refresh_env_marks()   # env may have been installed from the Train page

    # ------------------------------------------------------------- class names
    def _set_run_classes(self, names):
        """Adopt the run's class names (they label the per-class IoU stats)."""
        self._run_class_names = list(names) if names else None
        self._rebuild_class_list()

    def _rebuild_class_list(self):
        """All-checked list of the loaded run's classes; disabled placeholder when
        no run manifest is loaded. Deliberately NOT persisted via appstate —
        exclusions are per-scene-batch judgments; all-on is the safe default."""
        self.class_list.blockSignals(True)
        self.class_list.clear()
        names = self._run_class_names or []
        for n in names:
            it = QListWidgetItem(str(n))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            self.class_list.addItem(it)
        if not names:
            self.class_list.addItem("(load a run to list its classes)")
        self.class_list.setEnabled(bool(names))
        self.class_list.blockSignals(False)
        self._sync_class_mask_label()

    def _excluded_classes(self) -> list[str]:
        """Unticked class names — the run's EXCLUDE_CLASSES env value."""
        if not self.class_list.isEnabled():
            return []
        return [self.class_list.item(i).text() for i in range(self.class_list.count())
                if self.class_list.item(i).checkState() != Qt.Checked]

    def _sync_class_mask_label(self, _item=None):
        exc, total = self._excluded_classes(), self.class_list.count()
        if not exc:
            self.class_mask_lbl.setText("")
        elif total - len(exc) < 2:
            self.class_mask_lbl.setText("⚠ keep at least 2 classes enabled")
        else:
            self.class_mask_lbl.setText(
                f"masking {len(exc)} of {total}: {', '.join(exc)} — next-best class wins")

    @staticmethod
    def _names_from_manifest(m: dict) -> list | None:
        """Class names from a manifest: class_names, else 'class 0..n-1', else None."""
        names = m.get("class_names")
        if names:
            return list(names)
        n = m.get("num_classes")
        return [f"class {i}" for i in range(int(n))] if n else None

    # ------------------------------------------------------------- inputs
    def reload_runs(self):
        prev = self.run_combo.currentText()   # keep a typed/pasted ref across reloads
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        seen = set()
        for h in reversed(appstate.get("run_history", [])):
            if h["run_id"] not in seen:
                seen.add(h["run_id"])
                self.run_combo.addItem(f"{h['run_id']}  ({h['backbone']})", h)
        # Local/downloaded runs: the SAME discovery source as the Plotting page
        # (appstate.run_roots + plots.discover_runs), filtered to dirs carrying
        # a run.json manifest — this picker needs backbone/classes, not CSVs.
        for root in appstate.run_roots(self.repo_root):
            for rdir in plots.discover_runs(root):
                if rdir.name in seen:
                    continue
                m = _manifest_in(rdir)
                if m is not None:
                    seen.add(rdir.name)
                    self.run_combo.addItem(f"{rdir.name}  ({m.get('backbone', '?')})",
                                           {"run_id": rdir.name, "backbone": m.get("backbone")})
        # No implicit default: an old history entry preselected here reads as "the"
        # run and may not even exist on Modal anymore. Restore what was typed, else blank.
        self.run_combo.setCurrentIndex(-1)
        self.run_combo.setEditText(prev)
        self.run_combo.blockSignals(False)
        self._on_run_pick()

    def _on_run_pick(self):
        """Sync architecture from the picked run; adopt its classes if downloaded."""
        h = self.run_combo.currentData()
        if self._modal_cfg_run and self._combo_run_ref()[1] != self._modal_cfg_run:
            self._invalidate_manifest()   # a different run than the fetched config
        if isinstance(h, dict) and h.get("backbone") in BACKBONES:
            i = self.backbone_combo.findData(h["backbone"])
            if i >= 0:
                self.backbone_combo.setCurrentIndex(i)
        # Typed/pasted text has no item data — leave any fetched classes in place.
        if appstate.get_exec_mode() != "local" and isinstance(h, dict):
            self._set_run_classes(self._run_pick_class_names(h))

    def _run_pick_class_names(self, h) -> list | None:
        """Class names for a Modal run if downloaded locally, else None."""
        if not isinstance(h, dict):
            return None
        rid = str(h.get("run_id", ""))
        for rdir in (appstate.workspace_dir() / "inference" / rid,
                     appstate.runs_dir() / str(h.get("backbone", "")) / rid):
            m = _manifest_in(rdir)
            if m is not None:
                return self._names_from_manifest(m)
        return None

    def _combo_run_ref(self) -> tuple:
        """(volume, run_id) from the run combo — typed text wins over currentData().
        Accepts a pasted '<volume>/runs/<id>' (the train log's copy string), a bare
        id, or the combo's own 'id  (backbone)' items. Volume is '' unless pasted."""
        return _parse_run_ref(self.run_combo.currentText())

    def _download_run(self):
        """Modal: fetch runs/<id> (weights + run.json + metrics) from the model's
        outputs volume to <workspace>/inference/<id> — the folder reload_runs and
        the class-name lookup also read, and where local inference can pick the
        run.json straight back up."""
        b = self._backbone()
        vol, run_id = self._combo_run_ref()
        if not (b and run_id):
            self._append("Pick (or type) a run id to download.")
            return
        if self.dl_runner.running:
            self._append("A run download is already in progress.")
            return
        volume = vol or b.outputs_volume   # a pasted <volume>/runs/<id> names it exactly
        dest_base = appstate.workspace_dir() / "inference"
        dest_base.mkdir(parents=True, exist_ok=True)
        self._dl_run_dest = dest_base / run_id
        self._append(f"\nDownloading {volume}:/runs/{run_id} -> {self._dl_run_dest} …")
        prog, args = modal_cli.volume_get(volume, f"runs/{run_id}", str(dest_base))
        self.dl_run_btn.setEnabled(False)
        self.dl_runner.start(prog, args, cwd=self.repo_root)

    def _on_run_downloaded(self, code: int):
        self.dl_run_btn.setEnabled(True)
        dest = getattr(self, "_dl_run_dest", None)
        if code != 0:
            self._append(f"\n✗ Run download failed (exit {code}).")
            return
        self._append(f"\n✓ Run downloaded -> {dest} (weights: final_model.pth). "
                     "It now also appears under locally-known runs.")
        self.reload_runs()          # adopt run.json classes + list the local copy
        self._on_run_pick()

    def _pick_runjson(self):
        start = self.runjson_edit.text().strip() or str(appstate.workspace_dir())
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose run.json", start, "Run manifest (run.json *.json)")
        if path:
            self.runjson_edit.setText(path)
            self._load_run_manifest()

    def _invalidate_manifest(self):
        """Drop the loaded manifest so a run can't proceed stale."""
        self._manifest = self._manifest_path = self._local_weights = None
        self._manifest_features = None
        self._dg = {}
        self._modal_cfg_run = ""
        self._apply_manifest_lock(False)   # run-derived inputs editable again
        # The class mask belongs to the dropped run — a stale list would send
        # its EXCLUDE_CLASSES names with unrelated weights.
        self._set_run_classes(None)

    def _infer_dg_env(self, dg: dict | None = None) -> dict:
        """DG_* env for inference. logdk recovered from run.json (it changed the input
        width, so must be recomputed or the load fails); AdaBN / TTA / save-probs come
        from the Domain-adaptation row. `dg` overrides the loaded run's block (ensemble
        members carry their own)."""
        dg = self._dg if dg is None else dg
        env: dict[str, str] = {}
        if dg.get("logdk"):
            env["DG_LOGDK_FEAT"] = "1"
            env["DG_LOGDK_K"] = str(int(dg.get("logdk_k", 8)))
        if self.adabn_chk.isChecked():
            env["DG_INFER_ADABN"] = "1"
        if self.tta_spin.value() > 0:
            env["DG_INFER_TTA"] = str(self.tta_spin.value())
        if self.probs_chk.isChecked():
            env["TT_SAVE_PROBS"] = "1"
        if env:
            self._append("[dg] inference: " + " ".join(f"{k}={v}" for k, v in sorted(env.items())))
        # Class mask rides the same env dict (local -e / modal --env-json), so it
        # covers single runs, modal runs, and every ensemble member unchanged.
        exc = self._excluded_classes()
        if exc:
            env["EXCLUDE_CLASSES"] = ",".join(exc)
            self._append("[mask] excluding: " + ", ".join(exc)
                         + " — masked points fall to their next-best class")
        return env

    def _load_run_manifest(self):
        """Apply the picked run.json: arch/grid/tile/intensity from it, weights = sibling.
        Refuses if the run's backbone isn't selectable here."""
        self._invalidate_manifest()
        text = self.runjson_edit.text().strip()
        if not text:
            return
        p = Path(text)
        if not p.is_file():
            self._append(f"✗ run.json not found: {p}")
            return
        try:
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self._append(f"✗ couldn't read {p.name}: {e}")
            return
        if not self._apply_manifest_fields(m):
            return
        self._manifest, self._manifest_path = m, p
        self._local_weights = p.parent / m.get("weights", "final_model.pth")
        self.weights_edit.setText(str(self._local_weights))   # default
        self._apply_manifest_lock(True)   # arch/grid/tile come from the run — grey them out
        ok = "✓" if self._local_weights.is_file() else "✗ weights missing -"
        self._append(f"Loaded {p.name}: {m.get('backbone')}, grid={m.get('grid')}, "
                     f"chunk={m.get('chunk_xy')}, intensity={m.get('intensity_norm')}. "
                     f"{ok} {self._local_weights}")

    def _apply_manifest_fields(self, m: dict) -> bool:
        """Arch/grid/tile/HAG/DG/classes from a run manifest — shared by the local
        run.json loader and the Modal fetch-by-run-id path. False (with a log
        line) when the run's model can't be used here."""
        bkey = m.get("backbone")
        legacy_hag = bool(m.get("hag_source"))
        if isinstance(bkey, str) and bkey.endswith("_hag"):
            # ponytail: TEMPORARY shim (remove with the trainers') — the deleted
            # *_hag variants run on their base trainer, which maps the legacy
            # 'height' channel to feat_hag when run.json carries hag_source.
            bkey = bkey[: -len("_hag")]
            self._append(f"[legacy-hag] run from the removed --hag variant — "
                         f"using '{bkey}'; HAG conversion forced on (temporary "
                         f"support for old weights).")
        i = self.backbone_combo.findData(bkey)
        if i < 0:   # the run's model is hidden/unknown — don't keep the wrong one
            self._append(f"✗ Model '{bkey}' isn't available here. Enable it on the "
                         f"Train page, then reload this run.")
            return False
        self.backbone_combo.setCurrentIndex(i)       # fires _sync_controls (sets defaults)
        if m.get("grid") is not None:                # then the manifest overrides them
            self.grid_spin.setValue(float(m["grid"]))
        if m.get("chunk_xy") is not None and self.chunk_spin.isEnabled():
            self.chunk_spin.setValue(float(m["chunk_xy"]))
        # logdk changes input width, so it MUST be re-fed at inference (DG_* env).
        self._dg = m.get("dg") or {}
        # The run's ordered input spec (run.json "features"); None = legacy run.
        # Legacy --hag runs listed the HAG channel as 'height' — translate it so
        # the channel gating below requires the baked feat_hag, like the trainer.
        feats = m.get("features")
        if legacy_hag and feats:
            feats = ["feat_hag" if n == "height" else n for n in feats]
        # Prefill HAG from the run's feature spec: tick + preselect the method the
        # dataset baked it with (meta "@hag:<method>[+source]"). The user can still
        # untick. Ground class isn't prefilled — run.json doesn't carry one, and it
        # describes the inference input, not the training set.
        need_hag = "feat_hag" in (feats or []) or legacy_hag
        self.hag_chk.setChecked(need_hag)
        if need_hag:
            chans = ((self._dataset_meta(m) or {}).get("source") or {}) \
                .get("feature_channels") or []
            src = next((str(c.get("source_field") or "") for c in chans
                        if isinstance(c, dict) and c.get("name") == "hag"), "")
            method = src[len("@hag:"):].split("+", 1)[0] if src.startswith("@hag:") else ""
            if not method and legacy_hag:
                # old runs carry the method in hag_source itself (e.g. "grid+smrf")
                method = str(m["hag_source"]).split("+", 1)[0]
            j = self.hag_filter.findText(method) if method else -1
            if j >= 0:
                self.hag_filter.setCurrentIndex(j)
            elif method not in ("", "source_dimension"):
                self._append(f"[hag] unknown train-time HAG method '{method}' — "
                             f"using {self.hag_filter.currentText()}.")
            self._append(f"[hag] run trained with feat_hag; HAG enabled, "
                         f"method={self.hag_filter.currentText()}.")
        if self._dg.get("logdk"):
            self._append(f"[dg] trained with the log-d_k density channel "
                         f"(k={self._dg.get('logdk_k', 8)}); recomputed at inference.")
        # Custom feat_* channels must be baked into the converted scenes.
        self._manifest_features = feats
        custom = [n for n in (self._manifest_features or []) if n.startswith("feat_")]
        if custom:
            self._append(f"[feat] run trained with custom channel(s): {', '.join(custom)} "
                         f"— the input clouds must carry the matching source field(s).")
        # Label the comparison stats with the model's own classes.
        self._set_run_classes(self._names_from_manifest(m))
        return True

    # ------------------------------------------- modal config-by-run-id fetch
    def _on_run_id_typed(self):
        """Modal: a typed/pasted run id — pull its run.json (locally downloaded
        copy first, else off the outputs volumes) so arch/grid/tile/HAG/classes
        match the run, same as picking a run.json does for local inference."""
        if appstate.get_exec_mode() == "local" or not self.from_run_radio.isChecked():
            return
        vol, rid = self._combo_run_ref()
        if not rid or rid == self._modal_cfg_run or self.cfg_fetcher.running:
            return
        if self._manifest and self._manifest_path is None:
            self._invalidate_manifest()   # a different run's fetched config — drop it
        # Already downloaded ('Download run…' / Runs page)? No network needed.
        rdirs = appstate.runs_dir()
        cands = [appstate.workspace_dir() / "inference" / rid]
        cands += [bdir / rid for bdir in (rdirs.iterdir() if rdirs.exists() else [])]
        for d in cands:
            m = _manifest_in(d)
            if m is not None:
                self._apply_modal_manifest(m, rid)
                return
        if vol:
            vols = [vol]   # pasted <volume>/runs/<id> — go straight there, no search
        else:
            # Bare id: search the current model's outputs volume first, then the others.
            cur = BACKBONES.get(self.backbone_combo.currentData())
            vols = [cur.outputs_volume] if cur else []
            for i in range(self.backbone_combo.count()):
                b = BACKBONES.get(self.backbone_combo.itemData(i))
                if b and b.outputs_volume not in vols:
                    vols.append(b.outputs_volume)
        self._pending_cfg_run = rid
        self._append(f"Fetching run config for '{rid}' from Modal…")
        self.cfg_fetcher.start(_fetch_run_config, vols, rid)

    def _on_cfg_fetched(self, m):
        rid = self._pending_cfg_run
        if not m:
            self._append(f"✗ No run.json for '{rid}' on any outputs volume — set "
                         f"Architecture / grid / tile to match the run manually.")
            self._forget_run(rid)   # a stale history entry stops being offered
            return
        if self._combo_run_ref()[1] != rid:
            return   # the user moved on to another run while we fetched
        self._apply_modal_manifest(m, rid)

    def _forget_run(self, rid: str):
        """A history run that's gone from Modal: purge it so it stops being offered.
        A pasted id that was never in history is left alone (the run may simply
        predate run.json manifests)."""
        hist = appstate.get("run_history", [])
        kept = [h for h in hist if h.get("run_id") != rid]
        if len(kept) == len(hist):
            return
        appstate.put("run_history", kept)
        if self._combo_run_ref()[1] == rid:
            self.run_combo.clearEditText()
        self.reload_runs()
        self._append(f"  (dropped '{rid}' from the run list — stale history entry.)")

    def _apply_modal_manifest(self, m: dict, rid: str):
        """A run manifest resolved from a Modal run id: apply + lock the run-derived
        inputs and keep it as self._manifest (no path — there's no local run.json)
        so intensity_norm and dataset-nested output work as in local mode."""
        if not self._apply_manifest_fields(m):
            return
        self._manifest, self._manifest_path = m, None
        self._modal_cfg_run = rid
        self._apply_manifest_lock(True)
        self._append(f"✓ Run '{rid}': {m.get('backbone')}, grid={m.get('grid')}, "
                     f"chunk={m.get('chunk_xy')}, intensity={m.get('intensity_norm')}.")

    def _on_source_toggle(self):
        self._sync_source_rows()   # hide the rows the other source owns
        # A loaded run.json dictates arch/grid/tile; the manual .pth source frees them.
        use_run = self.from_run_radio.isChecked() and self._manifest is not None
        self._apply_manifest_lock(use_run)
        # The class mask follows the active source: a bare .pth has no class set,
        # so a stale list would send the run's EXCLUDE_CLASSES with other weights.
        self._set_run_classes(
            self._names_from_manifest(self._manifest) if use_run else None)

    def _apply_manifest_lock(self, locked: bool):
        """Grey out the inputs a run.json dictates (architecture, grid, tile size) so
        they can't drift from the loaded run. Unlocking restores tile-size enablement
        to whether the backbone even has a tile param (RandLA has none).
        While a manifest is applied the three locked rows also FOLD AWAY into the
        single muted summary line under the run selector (manifest_summary);
        unlocking brings the editable rows back."""
        self.backbone_combo.setEnabled(not locked)
        self.grid_spin.setEnabled(not locked)
        if locked:
            self.chunk_spin.setEnabled(False)
        else:
            key = self.backbone_combo.currentData()
            b = BACKBONES.get(key) if key else None
            self.chunk_spin.setEnabled(bool(b) and b.has_chunk)
        folded = locked and self._manifest is not None
        self.wf.setRowVisible(self.backbone_combo, not folded)
        self.iform.setRowVisible(self.grid_spin, not folded)
        self.iform.setRowVisible(self.chunk_spin, not folded)
        if folded:
            m = self._manifest
            b = BACKBONES.get(m.get("backbone"))
            arch = b.label if b else str(m.get("backbone", "?"))
            grid = m.get("grid")
            tile = m.get("chunk_xy")
            grid = self.grid_spin.value() if grid is None else float(grid)
            tile = self.chunk_spin.value() if tile is None else float(tile)
            src = "run.json" if self._manifest_path else f"run '{self._modal_cfg_run}'"
            self.manifest_summary.setText(
                f"{arch} · grid {grid:g} · tile {tile:g} · from {src}")
        self.wf.setRowVisible(self.manifest_summary, folded)

    def _pick_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose weights (.pth)",
            self.weights_edit.text().strip() or str(appstate.workspace_dir()),
            "PyTorch checkpoints (*.pth *.pt)")
        if path:
            self.weights_edit.setText(path)

    def _resolved_weights(self):
        """Local weights: the explicit override if set, else the run.json's sibling.
        The two need not be co-located."""
        t = self.weights_edit.text().strip()
        return Path(t) if t else self._local_weights

    def _weights_run_tag(self, member: dict | None = None) -> str:
        """Parent folder for this job's prediction dirs: the run id of the weights
        used (the runs/<id> from run.json), so every prediction traces back to its
        model — and lands beside a 'Download run…' copy of the same run. A loose
        .pth has no run id; its filename stem stands in. Ensemble members resolve
        their own run; the voted output goes under 'ensemble'."""
        if member:
            mp = member.get("manifest_path")
            return Path(mp).parent.name if mp else Path(member["weights"]).stem
        return self._run_tag or "adhoc"

    def _sync_controls(self):
        """Auto-fill grid + tile from the backbone's defaults; disable tile for RandLA."""
        if self.backbone_combo.currentData() is None:
            return
        if self._manifest is not None and self.from_run_radio.isChecked():
            return   # the run's grid/tile are authoritative (and folded into the
                     # summary line) — defaults here would silently undercut them.
                     # Safe on load: _apply_manifest_fields sets the combo BEFORE
                     # assigning self._manifest, so the defaults still get applied.
        b = self._backbone()
        gp = next((p for p in b.params if p.recommend_key == "grid"), None)
        if gp:
            self.grid_spin.setRange(gp.lo, 1_000_000.0)
            self.grid_spin.setDecimals(gp.decimals)
            self.grid_spin.setSingleStep(gp.step)
            self.grid_spin.setValue(gp.default)
        cp = next((p for p in b.params if p.flag == "chunk-xy"), None)
        if cp:
            self.chunk_spin.setValue(cp.default)
        self.chunk_spin.setEnabled(b.has_chunk)

    def _pick_pth(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose weights", "",
                                              "PyTorch checkpoints (*.pth *.pt)")
        if path:
            self.pth_edit.setText(path)

    def _pick_installed_weights(self):
        """Menu of trainer-weights-* conda packages installed in the selected
        backbone's pixi env; picking one fills the .pth path like Browse…"""
        b = self._backbone()
        items = local_cli.installed_weights(b, self.repo_root) if b else []
        if not items:
            self._append("[local] no trainer-weights-* packages installed"
                         + (f" in the '{local_cli.env_name(b)}' env" if b else "")
                         + " — add one with `pixi add`, or use Browse…")
            return
        menu = QMenu(self)
        for name, path in items:
            menu.addAction(name, lambda p=path: (self.from_file_radio.setChecked(True),
                                                 self.pth_edit.setText(p)))
        menu.exec(QCursor.pos())

    def _pick_input(self):
        d = QFileDialog.getExistingDirectory(self, "Folder of point clouds to label")
        if d:
            self.input_edit.setText(d)

    def _pick_input_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Point-cloud file to label", appstate.get("last_view_dir", ""),
            "Point clouds (*.ply *.npz *.las *.laz *.txt *.csv *.pcd *.xyz *.pts);;All files (*)")
        if f:
            self.input_edit.setText(f)

    def _backbone(self):
        # The run.json's backbone is authoritative for LOCAL inference so no deprecated backbones
        if (self._manifest and self.from_run_radio.isChecked()
                and appstate.get_exec_mode() == "local"
                and self._manifest.get("backbone") in BACKBONES):
            return BACKBONES[self._manifest["backbone"]]
        return BACKBONES[self.backbone_combo.currentData()]

    def _check_hag(self) -> bool:
        """Parse the ground class and reconcile the HAG box with the run's
        feature spec before we convert. A run whose "features" include feat_hag
        needs the channel baked (auto-tick); a run without it just wastes
        conversion time when the box is on (warn, don't block). Sets
        self._hag_ground_value."""
        self._hag_ground_value = None
        need = "feat_hag" in (self._run_features() or [])
        if need and not self.hag_chk.isChecked():
            self.hag_chk.setChecked(True)
            self._append("[hag] run trained with feat_hag — enabling "
                         "'Compute Height-Above-Ground' for the conversion.")
        elif self.hag_chk.isChecked() and not need:
            self._append("[hag] note: this run doesn't use feat_hag — computing "
                         "HAG only costs conversion time.")
        gtxt = self.hag_ground.text().strip()
        if self.hag_chk.isChecked() and gtxt:
            try:
                self._hag_ground_value = int(gtxt)
            except ValueError:
                self._append(f"Ground class '{gtxt}' isn't an integer - clear it to "
                             f"detect ground, or enter a classification value.")
                return False
        return True

    # ------------------------------------------------------------- ensemble
    def _add_ens_member(self):
        """Capture the currently configured model as an ensemble member — the same
        resolution _run's local branch uses (backbone + resolved weights +
        run.json), snapshotted so later UI edits don't drift the member."""
        if self.from_file_radio.isChecked():
            # A bare .pth has no run.json: no class names to clamp on (its trainer
            # writes placeholder class_0… names that would fail the voter's exact
            # clamp only AFTER every member burned a full inference pass) and no
            # dataset to resolve feature channels from. Refuse early and loudly.
            self._append("✗ ensemble: members must come from training runs "
                         "(run.json) — a bare .pth carries no class/dataset info "
                         "to check compatibility against.")
            return
        if not (self._manifest and self._manifest_path):
            self._append("✗ ensemble: pick a run.json first.")
            return
        w = self._resolved_weights()
        if not (w and w.is_file()):
            self._append(f"✗ ensemble: weights not found ({w}).")
            return
        bkey = self._manifest.get("backbone")
        manifest, mpath = dict(self._manifest), str(self._manifest_path)
        # CLAMP: members must share one class set AND one dataset — feature
        # channels and intensity_norm are resolved through the first member's
        # dataset meta, so a different dataset would silently stage feat_*
        # columns from the wrong source field for the other members.
        new_n = manifest.get("num_classes")
        new_names = self._names_from_manifest(manifest)
        new_ds = manifest.get("dataset")
        for m in self._ens_members:
            mm = m.get("manifest") or {}
            old_n = mm.get("num_classes")
            old_names = self._names_from_manifest(mm)
            if (old_n and new_n and int(old_n) != int(new_n)) or \
                    (old_names and new_names and old_names != new_names):
                self._append(f"✗ ensemble: class mismatch — '{Path(m['weights']).name}' "
                             f"predicts {old_n} classes {old_names}, this run {new_n} "
                             f"{new_names}. Members must share one class set.")
                return
            old_ds = mm.get("dataset")
            if old_ds and new_ds and old_ds != new_ds:
                self._append(f"✗ ensemble: dataset mismatch — '{Path(m['weights']).name}' "
                             f"was trained on '{old_ds}', this run on '{new_ds}'. "
                             f"Members must share one dataset (feature channels and "
                             f"intensity normalization come from its meta).")
                return
        self._ens_members.append({
            "backbone": bkey, "weights": str(w), "manifest_path": mpath,
            "manifest": manifest, "dg": (manifest or {}).get("dg") or {},
            # grid/tile as currently configured (a loaded run.json sets both)
            "grid": self.grid_spin.value(), "chunk": self.chunk_spin.value()})
        src = Path(mpath).parent.name if mpath else w.name
        self.ens_list.addItem(f"{bkey} — {src}")
        self._append(f"[ensemble] member {len(self._ens_members)}: {bkey} ({w})")

    def _remove_ens_member(self):
        row = self.ens_list.currentRow()
        if row < 0:
            return
        self.ens_list.takeItem(row)
        self._ens_members.pop(row)

    def _run_ensemble(self, input_dir: str):
        """Ensemble launch: validate members, stage scenes ONCE, then run the
        members sequentially through the normal local stage machine."""
        n = len(self._ens_members)
        if n < 2:
            self._append("✗ ensemble needs at least 2 members — 'Add current "
                         "selection' for each model, or untick the group.")
            return
        for m in self._ens_members:
            if not Path(m["weights"]).is_file():
                self._append(f"✗ ensemble: weights missing for {m['backbone']}: "
                             f"{m['weights']}")
                return
        self._job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._begin_run(f"ensemble inference · {n} members · job {self._job_id}")
        self.run_btn.setEnabled(False)
        self._ens_running, self._ens_idx, self._ens_dirs = True, -1, []
        # _ens_running is set, so _check_hag sees the members' feature UNION —
        # any member with feat_hag turns HAG conversion on for the shared stage.
        if not self._check_hag():
            self._ens_running = False
            self.run_btn.setEnabled(True)
            self._end_run("✗ aborted before launch (HAG settings)")
            return
        if n == 2:
            self._append("[ensemble] ⚠ only 2 members — disagreements fall to the "
                         "more confident (then earlier) model; 3+ recommended.")
        self._append(f"[ensemble] {n} members; scenes staged once, models run "
                     f"sequentially (runtime scales with model count).")
        self._start_conversion(input_dir)

    def _start_next_member(self):
        self._ens_idx += 1
        m = self._ens_members[self._ens_idx]
        self._append(f"\n[ensemble] member {self._ens_idx + 1}/{len(self._ens_members)}: "
                     f"{m['backbone']} ({Path(m['weights']).name})")
        self._start_local_infer(member=m)

    def _on_member_done(self):
        """A member's local run exited 0: keep its infer_run.json beside its
        predictions (the vote's class clamp reads it), then next member or vote."""
        if not list(self._pred_dir.glob("*_pred.npz")):
            self._ens_running = False
            self.run_btn.setEnabled(True)
            self._append(f"\n✗ ensemble: member {self._ens_idx + 1} wrote no "
                         f"predictions in {self._pred_dir}. Check the log above.")
            self._end_run(f"✗ ensemble member {self._ens_idx + 1} wrote no predictions")
            return
        # The trainers write infer_run.json to the job dir (= the staged mount),
        # one level above the predictions mount — copy the member's snapshot in.
        src = Path(self._staged) / "infer_run.json"
        if src.is_file():
            try:
                shutil.copy2(src, self._pred_dir / "infer_run.json")
            except OSError as e:
                self._append(f"[ensemble] (couldn't copy infer_run.json: {e})")
        self._ens_dirs.append(self._pred_dir)
        if self._ens_idx + 1 < len(self._ens_members):
            self._start_next_member()
            return
        ens_dir = (appstate.workspace_dir() / "inference" / "ensemble"
                   / f"predictions_{self._job_id}_ensemble")
        self._append(f"\n[ensemble] voting over {len(self._ens_dirs)} member "
                     f"run(s) -> {ens_dir}…")
        self.voter.start(_vote_members, [str(d) for d in self._ens_dirs], str(ens_dir))

    def _on_voted(self, ens_dir):
        # _ens_running stays True through the export so the class map comes from
        # the members' manifest; _on_exported/_on_export_error reset it.
        self._report_predictions(Path(ens_dir))

    def _on_vote_error(self, tb: str):
        self._ens_running = False
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ ensemble vote failed — the per-member predictions "
                     f"remain in their predictions_{self._job_id}_m<k> folders.\n{tb}")
        self._end_run("✗ ensemble vote failed")

    # ------------------------------------------------------------- run chain
    def _run(self):
        input_dir = self.input_edit.text().strip()
        if not os.path.exists(input_dir):
            self._append("Choose an input folder or file first.")
            return
        exc = self._excluded_classes()
        if exc and self.class_list.count() - len(exc) < 2:
            self._append("✗ Class mask: keep at least 2 classes enabled — a one-class "
                         "prediction is meaningless.")
            return
        modal = appstate.get_exec_mode() != "local"
        if self.ens_box.isChecked() and not modal:   # modal keeps the group unchecked
            self._run_ensemble(input_dir)
            return
        weights_run_id = ""
        weights_vol = ""   # a pasted <volume>/runs/<id> names it; else the backbone's
        if self.from_file_radio.isChecked():
            if not os.path.isfile(self.pth_edit.text().strip()):
                self._append("Choose a .pth file.")
                return
            self._weights_remote = f"uploads/{Path(self.pth_edit.text()).name}"
            bkey = self.backbone_combo.currentData()   # a loose .pth has no manifest
            self._run_tag = Path(self.pth_edit.text().strip()).stem  # no run id: pth stem
        elif not modal:
            # LOCAL: run.json is the explicit input; weights = its sibling.
            if not (self._manifest and self._manifest_path):
                self._append("Pick a run.json first.")
                return
            w = self._resolved_weights()
            if not (w and w.is_file()):
                self._append(f"✗ Weights not found ({w}). Set the 'Weights file' box.")
                return
            bkey = self._manifest.get("backbone")
            self._run_tag = self._manifest_path.parent.name   # runs/<id>/run.json
        else:
            # MODAL: weights live on the cloud volume, keyed by run id. The typed
            # text wins — currentData() keeps returning the last picked item even
            # after the user types/pastes a different id over it.
            pasted_vol, run_id = self._combo_run_ref()
            if not run_id:
                self._append("Pick or type a run id.")
                return
            h = self.run_combo.currentData()
            bkey = (h.get("backbone") if isinstance(h, dict)
                    and h.get("run_id") == run_id else None) \
                or self.backbone_combo.currentData()
            self._weights_remote = f"runs/{run_id}/final_model.pth"
            weights_run_id = run_id
            weights_vol = pasted_vol
            self._run_tag = run_id

        if not self._check_hag():
            return

        self._job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._begin_run(f"inference · {bkey} · job {self._job_id}")
        self.run_btn.setEnabled(False)
        # Runs enter history at train START, so may lack final_model.pth. Check weights
        # exist before converting/uploading/paying for GPU. Runs in a worker.
        # A run already fetched with 'Download run…' was pulled OFF the volume, so
        # its weights provably exist there — no network round-trip needed.
        if modal and weights_run_id:
            if self._has_local_run_copy(weights_run_id):
                self._append(f"[0/4] Weights verified from the local download of "
                             f"'{weights_run_id}' — skipping the Modal check.")
                self._start_conversion(input_dir)
                return
            self._pending_input = input_dir
            self._pending_run_id = weights_run_id
            wvol = weights_vol or self._backbone().outputs_volume
            self._append(f"[0/4] Checking weights on Modal ({wvol})…")
            self.preflight.start(_check_weights_present, wvol, weights_run_id)
            return
        self._start_conversion(input_dir)

    def _has_local_run_copy(self, rid: str) -> bool:
        """final_model.pth from a 'Download run…' / Runs-page fetch of this run?"""
        dirs = [appstate.workspace_dir() / "inference" / rid]
        rd = appstate.runs_dir()
        dirs += [b / rid for b in (rd.iterdir() if rd.exists() else [])]
        return any((d / "final_model.pth").is_file() for d in dirs)

    def _start_conversion(self, input_dir: str):
        # p95 end-to-end; honor a legacy norm recorded in run.json.
        if self._ens_running:
            # ponytail: the FIRST member's manifest speaks for the ensemble
            # (norm/dataset) — members are same-dataset runs in practice.
            norm = (self._ens_members[0].get("manifest") or {}).get(
                "intensity_norm") or "p95"
        else:
            norm = (self._manifest or {}).get("intensity_norm", "p95") \
                if (self.from_run_radio.isChecked() and self._manifest) else "p95"
        # HAG comes from the checkbox — _check_hag() already reconciled it with
        # the run's feature spec and parsed the ground value.
        hag_on = self.hag_chk.isChecked()
        hag_filter = self.hag_filter.currentText() if hag_on else "grid"
        if hag_on:
            src = (f"ground = class {self._hag_ground_value}"
                   if self._hag_ground_value is not None else "ground detected (SMRF)")
            self._append(f"[1/4] Computing HeightAboveGround ({hag_filter}, {src}) "
                         "for the input scenes.")
        job_root = self._infer_out_dir()
        fields, geo, geo_r = self._infer_feature_fields()
        if geo:
            self._append(f"[1/4] Recomputing geometric feature(s) "
                         f"{', '.join(geo)} (r={geo_r:g} m) per scene.")
        self._append(f"[1/4] Converting {input_dir} to scenes (job {self._job_id}; "
                     f"intensity={norm}) -> {job_root}…")
        self.converter.start(dataset.convert_infer_job, self._job_id, input_dir,
                             appstate.workspace_dir(), intensity_norm=norm,
                             hag=hag_on, hag_filter=hag_filter,
                             ground_value=self._hag_ground_value,
                             feature_fields=fields, geo_features=geo,
                             geo_radius=geo_r, out_dir=job_root)

    def _run_features(self) -> list | None:
        """The loaded run's ordered input spec (run.json "features"), or None
        (loose .pth / legacy run -> legacy channel handling). During an ensemble
        run: the UNION of every member's features, so the once-staged scenes
        carry (and the channel report checks) what any member needs."""
        if self._ens_running:
            feats: list = []
            for m in self._ens_members:
                for f in (m.get("manifest") or {}).get("features") or []:
                    if f not in feats:
                        feats.append(f)
            return feats or None
        return self._manifest_features \
            if (self.from_run_radio.isChecked() and self._manifest) else None

    def _infer_feature_fields(self) -> tuple:
        """(raw source fields, jakteristics names, radius) for the run's custom
        feat_* channels, resolved through the dataset meta's
        source.feature_channels. Computed geo entries carry source_field
        "@geo:<jak name>" + "radius" — the meta radius is the train-time truth.
        A geo channel with no meta (loose .pth) is recovered from its
        feat_geo_<nm> name at the DEFAULT radius 1.0 with a warning; other
        unresolved names fall back to a same-named raw field (conversion fails
        loud on a truly missing one)."""
        # feat_hag is produced by the conversion's hag=/hag_filter= args, not a
        # raw source field — routing its "@hag:" source_field into feature_fields
        # would fail as a missing raw field.
        custom = [n for n in (self._run_features() or [])
                  if n.startswith("feat_") and n != "feat_hag"]
        if not custom:
            return None, None, 1.0
        jak_by_lower = {n.lower(): n for n in pretrain.GEO_FEATURES}
        chans = ((self._dataset_meta() or {}).get("source") or {}).get("feature_channels") or []
        by_name = {c.get("name"): c for c in chans if isinstance(c, dict)}
        fields, geo, radius = [], [], None
        for n in custom:
            c = by_name.get(n[len("feat_"):]) or {}
            src = c.get("source_field") or ""
            if src.startswith("@geo:"):
                geo.append(src[len("@geo:"):])
                r = c.get("radius")
                if r is not None and radius is not None and float(r) != radius:
                    self._append(f"[feat] ⚠ mixed geo radii in meta ({radius:g} vs "
                                 f"{float(r):g}) — using {radius:g}.")
                elif r is not None:
                    radius = float(r)
            elif src:
                fields.append(src)
            elif n.startswith("feat_geo_") and n[len("feat_geo_"):] in jak_by_lower:
                jak = jak_by_lower[n[len("feat_geo_"):]]
                self._append(f"[feat] no dataset meta for '{n}' — recomputing "
                             f"jakteristics '{jak}' at the DEFAULT radius 1.0 m "
                             f"(train radius unknown).")
                geo.append(jak)
            else:
                fields.append(n[len("feat_"):])
                self._append(f"[feat] no dataset meta maps '{n}' to a raw field — "
                             f"assuming the inputs carry a field named "
                             f"'{n[len('feat_'):]}'.")
        return fields or None, geo or None, (radius if radius is not None else 1.0)

    def _infer_out_dir(self) -> Path:
        """Nest this infer job under its owning dataset (<dataset>/infer/<job>) when
        the run names a known, on-disk dataset; else a findable workspace scratch
        spot (loose .pth has no linked dataset). The container still mounts it at
        /datasets/_infer/<job> regardless."""
        name = self._owning_dataset()
        staged = appstate.known_datasets().get(name or "", {}).get("staged_dir", "")
        if staged and os.path.isdir(staged):
            return appstate.dataset_root(name) / "infer" / self._job_id
        return appstate.scratch_infer_dir() / self._job_id

    def _on_preflight(self, present):
        """Weights check: True=found, False=missing (block), None=couldn't list (proceed)."""
        if present is False:
            self._append(f"✗ Run '{self._pending_run_id}' has no final_model.pth on the "
                         f"outputs volume. Pick a completed run, or use 'Local .pth file'.")
            self.run_btn.setEnabled(True)
            self._end_run("✗ weights missing on Modal")
            return
        if present is None:
            self._append("[0/4] (couldn't verify weights on Modal - proceeding.)")
        self._start_conversion(self._pending_input)

    def _on_preflight_error(self, tb: str):
        # A failed check shouldn't block; fall through to the in-container backstop.
        self._append(f"[0/4] (weights check errored, proceeding anyway)\n{tb}")
        self._start_conversion(self._pending_input)

    def _on_converted(self, staged: Path):
        self._staged = staged
        lines, blocked = _scene_channel_report(staged,
                                               features=self._run_features())
        for line in lines:
            self._append(line)
        if blocked:
            # Same shape as the HAG refusal: a clear log line, button back, stop.
            self._append("✗ Aborting — the input clouds lack channel(s) this run was "
                         "trained on (see above). Re-export the inputs with those "
                         "fields, or pick a run that doesn't need them.")
            self._ens_running = False
            self.run_btn.setEnabled(True)
            self._end_run("✗ inputs lack required channel(s)")
            return
        if appstate.get_exec_mode() == "local":
            if self._ens_running:
                self._start_next_member()
            else:
                self._start_local_infer()
            return
        self._append(f"[2/4] Uploading scenes -> {modal_cli.DATASETS_VOLUME}:/_infer/{self._job_id}… "
                     "(a 'volume already exists' message here is expected and harmless)")
        self._stage = "upload_scenes"
        prog, args = modal_cli.volume_put(modal_cli.DATASETS_VOLUME, str(staged),
                                          f"/_infer/{self._job_id}")
        # pre-create so inference works even on an account that has never
        # uploaded a dataset (put errors on a missing volume; create's
        # already-exists error is ignored by JobRunner's pre stage).
        self.runner.start(prog, args, cwd=self.repo_root,
                          pre=modal_cli.volume_create(modal_cli.DATASETS_VOLUME))

    def _on_stage_done(self, code: int):
        if code != 0:
            if self._ens_running:
                self._append(f"\n✗ ensemble member {self._ens_idx + 1} failed "
                             f"(exit {code}) — ensemble aborted.")
                self._ens_running = False
            self._append(f"\n✗ Stage '{self._stage}' failed (exit {code}).")
            self.run_btn.setEnabled(True)
            self._end_run(f"✗ stage '{self._stage}' failed (exit {code})")
            return
        b = self._backbone()
        if self._stage == "upload_scenes":
            if self.from_file_radio.isChecked():
                local = self.pth_edit.text().strip()
                self._append(f"[2b] Uploading weights -> {b.outputs_volume}:/{self._weights_remote}…")
                self._stage = "upload_weights"
                prog, args = modal_cli.volume_put(b.outputs_volume, local,
                                                  f"/{self._weights_remote}")
                self.runner.start(prog, args, cwd=self.repo_root)
                return
            self._start_modal_run()
        elif self._stage == "upload_weights":
            self._start_modal_run()
        elif self._stage == "run":
            # Predictions on the datasets volume at _infer/<job_id>/predictions.
            # Download under <workspace>/inference/<run_id>: grouped per model
            # (beside a 'Download run…' copy); job-id subfolder avoids collisions.
            self._dl_dest = (appstate.workspace_dir() / "inference"
                             / self._weights_run_tag()
                             / f"predictions_{self._job_id}")
            self._dl_dest.mkdir(parents=True, exist_ok=True)
            self._append(f"[4/4] Downloading predictions -> {self._dl_dest}…")
            self._stage = "download"
            prog, args = modal_cli.volume_get(modal_cli.DATASETS_VOLUME,
                                              f"_infer/{self._job_id}/predictions",
                                              str(self._dl_dest))
            self.runner.start(prog, args, cwd=self.repo_root)
        elif self._stage == "run_local":
            # Local: predictions already on the host, no download.
            if self._ens_running:
                self._on_member_done()
            else:
                self._report_predictions(self._pred_dir)
        elif self._stage == "download":
            self._report_predictions(self._dl_dest / "predictions")

    def _report_predictions(self, pred_dir):
        """Predictions landed (a stage can exit 0 yet write nothing) -> write the
        scripts' npz predictions as the chosen format (xyz + classification) on a
        worker thread; _on_exported prints the final green report."""
        pred_dir = Path(pred_dir) if pred_dir else None
        if not (pred_dir and pred_dir.is_dir()):
            self._ens_running = False
            self.run_btn.setEnabled(True)
            self._append(f"\n✗ No predictions folder at {pred_dir}.")
            self._end_run("✗ no predictions folder")
            return
        preds = [p for p in sorted(pred_dir.iterdir())
                 if p.suffix.lower() in (".ply", ".npz")]
        if not preds:
            self._ens_running = False
            self.run_btn.setEnabled(True)
            self._append(f"\n✗ No prediction files in {pred_dir}. Check the log above.")
            self._end_run("✗ no prediction files written")
            return
        appstate.put("last_view_dir", str(pred_dir))
        fmt = self.fmt_combo.currentData()
        appstate.put("infer_format", fmt)
        thr = self.unclass_spin.value() if self.unclass_chk.isChecked() else None
        self._append(f"\n[export] writing predictions as {fmt} (xyz + classification)…")
        self.exporter.start(dataset.export_predictions, pred_dir, fmt,
                            class_map=self._class_map(), unclass_threshold=thr)

    def _owning_dataset(self) -> str | None:
        """Dataset name the active weights belong to: the loaded run.json's, or —
        during an ensemble run — the first member's (members share a class set,
        so one dataset meta speaks for all)."""
        if self._ens_running:
            return (self._ens_members[0].get("manifest") or {}).get("dataset")
        return (self._manifest or {}).get("dataset") \
            if (self.from_run_radio.isChecked() and self._manifest) else None

    def _dataset_meta(self, manifest: dict | None = None) -> dict | None:
        """The run's dataset_meta.json as a dict, or None when it can't be
        resolved (loose .pth / unknown dataset). `manifest` overrides
        self._manifest — _apply_manifest_fields runs BEFORE the manifest is
        adopted, so it must resolve against the one being applied."""
        name = manifest.get("dataset") if manifest else self._owning_dataset()
        mp = appstate.known_datasets().get(name or "", {}).get("meta_path", "")
        try:
            with open(mp, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _class_map(self) -> dict | None:
        """{model index: source classification value} from the run's dataset meta,
        so exports carry the input's own codes. None (identity) when the dataset
        can't be resolved (loose .pth / unknown dataset) — logged once."""
        try:
            # meta collapses combined classes to "source_values" (list); the
            # FIRST value is the primary export code. Singular = very old metas.
            cmap = {int(c["index"]): int((c.get("source_values") or
                                          [c["source_value"]])[0])
                    for c in (self._dataset_meta() or {}).get("classes", [])}
        except (ValueError, KeyError, TypeError, IndexError):
            cmap = None
        if not cmap:
            self._append("[export] no dataset meta for these weights — exported "
                         "codes are raw model indices.")
            return None
        return cmap

    def _on_exported(self, written):
        self._ens_running = False   # an ensemble job ends at its export
        self.run_btn.setEnabled(True)
        if not written:
            self._append("✗ Nothing exported (no *_pred.npz in the predictions folder).")
            self._end_run("✗ nothing exported")
            return
        self._append(f"\n✓ Done - {len(written)} prediction file(s) in {written[0].parent}.\n"
                     f"  'Compare to ground truth…' for accuracy + mIoU.")
        self._last_pred_dir = written[0].parent
        self.plot_btn.setEnabled(True)
        self._end_run(f"✓ exported {len(written)} prediction file(s)")

    def _on_export_error(self, tb: str):
        # Predictions still exist as the scripts' raw .npz; only the rewrite failed.
        self._ens_running = False
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Format conversion failed — predictions remain as raw "
                     f".npz files.\n{tb}")
        self._end_run("✗ export failed (raw .npz kept)")

    def _start_modal_run(self):
        b = self._backbone()
        flags = {
            "mode": "infer",
            "weights": self._weights_remote,
            "infer-input": self._job_id,
            b.grid_flag: self.grid_spin.value(),   # --grid or --sub-grid per backbone
        }
        if b.has_chunk:
            flags["chunk-xy"] = self.chunk_spin.value()
        self._append(f"[3/4] Running inference on Modal ({b.label})…")
        self._stage = "run"
        # Same DG_LOGDK_* / EXCLUDE_CLASSES env the local branch passes — the
        # shell forwards it to the trainer subprocess via --env-json.
        prog, args = modal_cli.run_script(b.script, flags, detach=False,
                                          env=self._infer_dg_env())
        self._append(f"$ modal {' '.join(args)}\n")
        self.runner.start(prog, args, cwd=self.repo_root)

    def _start_local_infer(self, member: dict | None = None):
        """Local pixi inference: the trainer reads the staged scenes and writes
        predictions straight to host dirs via the TT_* env contract. No
        up/download. `member` (an ensemble member dict) overrides the UI-derived
        backbone/weights/grid and lands predictions in its own _m<k> dir."""
        b = BACKBONES[member["backbone"]] if member else self._backbone()
        # Predictions: <workspace>/inference/<run_tag>/predictions_<job> (same
        # spot as the Modal download), via TT_PRED_DIR — grouped per model so a
        # run's outputs live together. Scenes come from self._staged
        # (TT_INFER_DIR).
        suffix = f"_m{self._ens_idx + 1}" if member else ""
        self._pred_dir = (appstate.workspace_dir() / "inference"
                          / self._weights_run_tag(member)
                          / f"predictions_{self._job_id}{suffix}")
        self._pred_dir.mkdir(parents=True, exist_ok=True)
        # Weights: the picked .pth or run.json's sibling — passed absolute.
        if member:
            wpath = Path(member["weights"])
        else:
            wpath = Path(self.pth_edit.text().strip()) \
                if self.from_file_radio.isChecked() else self._resolved_weights()
        flags = {
            "mode": "infer",
            "weights": str(wpath),
            "infer-input": self._job_id,
            b.grid_flag: member["grid"] if member else self.grid_spin.value(),
        }
        if b.has_chunk:
            flags["chunk-xy"] = member["chunk"] if member else self.chunk_spin.value()
        env = self._infer_dg_env(member["dg"]) if member else self._infer_dg_env()
        if member:
            env["TT_SAVE_PROBS"] = "1"   # the soft vote averages the saved dists
        self._stage = "run_local"
        prog, args, run_env = local_cli.run_script(
            b.script, flags, b, repo_root=self.repo_root,
            infer_dir=str(self._staged), pred_dir=str(self._pred_dir), env=env)
        self._append(f"[local] Running inference in the pixi env ({b.label})…")
        self._append(f"[local] $ {local_cli.preview(prog, args, run_env)}\n")
        if not local_cli.runnable():
            self._append("[local] pixi not found (or not a Linux/CUDA host); printed "
                         "the command only. On the GPU box predictions land in "
                         f"{self._pred_dir.as_posix()}.")
            self._ens_running = False
            self.run_btn.setEnabled(True)
            return
        gok, gmsg = local_cli.gpu_preflight()   # CUDA-only — fail clearly, not cryptically
        if gmsg:
            self._append(gmsg)
        if not gok:
            self._ens_running = False
            self.run_btn.setEnabled(True)
            return
        ok, msg = local_cli.env_preflight(b, self.repo_root)
        if msg:
            self._append(msg)
        if not ok:
            self._ens_running = False
            self.run_btn.setEnabled(True)
            return
        self.runner.start(prog, args, cwd=self.repo_root, extra_env=run_env)

    def _on_output(self, text: str):
        # Legacy container-style paths (/datasets/_infer/<job>/…) in trainer logs
        # that mean nothing on the host; show the real output folder the files land in.
        disp = (_localize_paths(text, self._job_id, self._pred_dir, self._staged)
                if self._stage == "run_local" else text)
        self._append(disp, newline=False)

    def _on_error(self, tb: str):
        self._ens_running = False
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Conversion error:\n{tb}")
        self._end_run("✗ conversion error")

    def _on_runner_failed(self, err: str):
        # FailedToStart fires `failed` not `finished`; re-enable the button.
        self._ens_running = False
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Failed to start: {err}")
        self._end_run("✗ failed to start")

    # ------------------------------------------------------------- compare
    def _pick_pred_gt(self):
        """Prompt for a prediction cloud then its ground-truth labels.
        Returns (pred, gt) or None if cancelled."""
        flt = "Labeled clouds (*.npz *.las *.laz *.ply *.txt *.csv);;All files (*)"
        pred, _ = QFileDialog.getOpenFileName(
            self, "Prediction cloud to compare", appstate.get("last_view_dir", ""), flt)
        if not pred:
            return None
        appstate.put("last_view_dir", str(Path(pred).parent))
        gt, _ = QFileDialog.getOpenFileName(
            self, "Ground truth for this scene", appstate.get("truth_file", ""), flt)
        if not gt:
            return None
        appstate.put("truth_file", gt)
        return pred, gt

    def _compare_gt(self):
        """Prompt for a prediction + ground truth (both must carry explicit
        per-point classes) and print accuracy + mIoU to the log."""
        picked = self._pick_pred_gt()
        if not picked:
            return
        pred, gt = picked
        self._append(f"\nComparing {Path(pred).name} to {Path(gt).name} — "
                     f"computing accuracy + mIoU…")
        try:
            m = analysis.prediction_metrics(pred, gt)
        except Exception as e:  # noqa: BLE001
            self._append(f"  ✗ couldn't compute stats: {e}")
            return
        names = self._run_class_names or []
        nm = lambda c: names[c] if 0 <= c < len(names) else f"class {c}"
        lines = [f"── {m['scene'] or Path(pred).stem} vs ground truth ──",
                 f"  accuracy : {m['accuracy']:.4f}",
                 f"  mIoU     : {m['miou']:.4f}   (over {len(m['per_class_iou'])} present classes)",
                 f"  labeled  : {m['labeled']:,} pts",
                 "  per-class IoU:"]
        lines += [f"    {nm(c)}: {iou:.4f}" for c, iou in sorted(m["per_class_iou"].items())]
        # Persist beside the prediction so the numbers aren't lost with the log.
        mpath = Path(pred).with_suffix(".metrics.json")
        try:
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump({"prediction": str(pred), "ground_truth": str(gt),
                           "class_names": names, **m}, f, indent=2)
            lines.append(f"  saved -> {mpath}")
        except OSError as e:
            lines.append(f"  (couldn't save metrics json: {e})")
        # Accumulate one row per comparison in <workspace>/gt_metrics.csv so
        # one-at-a-time scoring builds the experiment table by itself. The file
        # is rewritten with the union of all headers, so a later run with a
        # different class set widens the table instead of losing columns.
        row = {"when": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "prediction": str(pred), "ground_truth": str(gt),
               "scene": m["scene"] or Path(pred).stem,
               "accuracy": f"{m['accuracy']:.4f}", "miou": f"{m['miou']:.4f}",
               "labeled": str(m["labeled"])}
        row.update({f"iou_{nm(c)}": f"{iou:.4f}"
                    for c, iou in m["per_class_iou"].items()})
        cpath = appstate.workspace_dir() / "gt_metrics.csv"
        try:
            rows = []
            if cpath.exists():
                with open(cpath, newline="", encoding="utf-8") as f:
                    rows = [r for r in csv.DictReader(f) if any(r.values())]
            rows.append(row)
            core = ["when", "prediction", "ground_truth", "scene",
                    "accuracy", "miou", "labeled"]
            extra = sorted({k for r in rows for k in r if k} - set(core))
            with open(cpath, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=core + extra, restval="")
                w.writeheader()
                w.writerows(rows)
            lines.append(f"  csv   -> {cpath}")
        except (OSError, csv.Error) as e:
            lines.append(f"  (couldn't update stats csv: {e})")
        self._append("\n".join(lines))
        self._last_pred_dir = Path(pred).parent
        self.plot_btn.setEnabled(True)

    def _plot_run(self):
        """Jump to the Plotting page focused on the run behind these weights —
        training metrics live under the run id, not the predictions folder.
        PlottingPage.receive_nav(run=...) preselects it (matches by dir or id)."""
        rid = (self._manifest_path.parent.name if self._manifest_path
               else self._modal_cfg_run or "")
        ui.navigate("Plotting", **({"run": rid} if rid else {}))

    # ------------------------------------------------------------- helpers
    def _begin_run(self, title: str):
        """Run header instead of the old log.clear(): earlier runs stay
        scrollable above the divider."""
        self._run_open = True
        self.log.begin_run(title)

    def _end_run(self, summary: str):
        """Close the current run header exactly once (terminal points can
        overlap, e.g. export error after a stage failure)."""
        if self._run_open:
            self._run_open = False
            self.log.end_run(summary)

    def _append(self, text: str, newline: bool = True):
        ui.append_log(self.log, text, newline)


def _vote_members(member_dirs: list, out_dir: str, progress=None):
    """FuncWorker body: run scripts/local/ensemble_vote.py over the member
    prediction dirs, then strip the big float16 'probs' payload from each member
    npz (the classification/confidence npz is kept for re-export/compare)."""
    import numpy as np
    scripts_local = str(Path(__file__).resolve().parents[2] / "scripts" / "local")
    if scripts_local not in sys.path:
        sys.path.insert(0, scripts_local)
    import ensemble_vote
    say = progress or print
    ensemble_vote.ensemble(member_dirs, out_dir, log=say)
    for d in member_dirs:
        for p in Path(d).glob("*_pred.npz"):
            with np.load(p) as z:
                if "probs" not in z.files:
                    continue
                slim = {k: z[k] for k in z.files if k != "probs"}
            np.savez(p, **slim)
    say("  (dropped the per-member probs payloads; classification npz kept — "
        "re-running ensemble_vote.py over these dirs will HARD-vote, which can "
        "give different labels than the soft vote above)")
    return Path(out_dir)


def _localize_paths(text: str, job_id: str, pred_dir, staged) -> str:
    """Rewrite container bind-mount paths to the host folders they map to (see the
    mounts in _start_local_infer), so a local run's log shows where files really go.
    Predictions first (the more specific mount), then the staging root."""
    if pred_dir:
        for p in (f"/datasets/_infer/{job_id}/predictions", f"_infer/{job_id}/predictions"):
            text = text.replace(p, str(pred_dir))
    if staged:
        for p in (f"/datasets/_infer/{job_id}", f"_infer/{job_id}"):
            text = text.replace(p, str(staged))
    return text


def _scene_channel_report(staged,
                          features: list | None = None) -> tuple[list[str], bool]:
    """(log lines, blocking) — verifies the converted scenes carry what the
    trainers read. The scene npz IS the model's input (fixed contract), so what's
    in it is exactly what inference sees. `features` is the run's ordered spec
    (run.json "features"): x/y/z/height cost nothing (xyz always present),
    intensity / return_number missing = constant filler + a warning (the
    trainers' own fallback), rgb / feat_* (incl. feat_hag) are HARD
    requirements -> blocking=True names which scenes lack what. None = legacy
    behavior (xyz + intensity), never blocking."""
    import numpy as np
    scenes = sorted(Path(staged).glob("scenes/*.npz"))
    if not scenes:
        return [f"⚠ no converted scenes under {staged} to check."], False
    hard = [n for n in (features or [])
            if n == "rgb" or n.startswith("feat_")]
    want_i = features is None or "intensity" in features
    want_r = features is not None and "return_number" in features
    missing_i, missing_r, keys0 = [], [], []
    missing_hard: dict[str, list[str]] = {}
    for p in scenes:
        with np.load(p) as z:
            names = set(z.files)
        if not keys0:
            keys0 = sorted(names)
        if want_i and "intensity" not in names:
            missing_i.append(p.stem)
        if want_r and "return_number" not in names:
            missing_r.append(p.stem)
        for ch in hard:
            if ch not in names:
                missing_hard.setdefault(ch, []).append(p.stem)
    lines = [f"[check] {len(scenes)} scene(s) carry: {', '.join(keys0)} — this npz "
             f"is exactly what the model reads."]
    if want_i and missing_i:
        lines.append(f"⚠ no intensity channel in: {', '.join(missing_i)} — the source "
                     f"file(s) had no intensity field. The trainer substitutes a "
                     f"constant filler, so models trained with real intensity see "
                     f"an unfamiliar value and accuracy will suffer.")
    elif want_i:
        # ponytail: value sanity from scene 0 only — full scan if this ever misleads
        with np.load(scenes[0]) as z:
            i = z["intensity"]
        lines.append(f"✓ intensity in all scenes ({scenes[0].stem}: "
                     f"min {float(i.min()):.2f}, max {float(i.max()):.2f})")
        if float(i.max()) <= 0.0:
            lines.append("⚠ intensity is all zeros — the source's intensity field is "
                         "empty; expect degraded accuracy.")
    if missing_r:
        lines.append(f"⚠ no return_number channel in: {', '.join(missing_r)} — the "
                     f"trainer feeds zeros there. Models trained with real return "
                     f"numbers see an unfamiliar constant and accuracy will suffer.")
    for ch, lost in missing_hard.items():
        hint = (" Tick 'Compute Height-Above-Ground' in the conversion box and "
                "run again." if ch == "feat_hag" else "")
        lines.append(f"✗ required channel '{ch}' missing in: {', '.join(lost)} — "
                     f"this run was trained with it; predicting without it would "
                     f"be garbage.{hint}")
    return lines, bool(missing_hard)


def _manifest_in(rdir: Path) -> dict | None:
    """run.json (legacy run_config.json) in a run folder, or None."""
    for fn in ("run.json", "run_config.json"):
        p = Path(rdir) / fn
        if p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
    return None


def _parse_run_ref(text: str) -> tuple:
    """('volume', 'run_id') out of whatever landed in the run box: a pasted
    '<volume>/runs/<id>' (the train log's copy string), 'runs/<id>', a bare id,
    or the combo's own 'id  (backbone)' items. Volume is '' when not named."""
    parts = text.split()
    tok = (parts[0] if parts else "").strip("/")
    if "/runs/" in tok:
        vol, rid = tok.split("/runs/", 1)
        return vol, rid.strip("/")
    if tok.startswith("runs/"):
        return "", tok[len("runs/"):].strip("/")
    return "", tok


def _entry_name(entry: dict) -> str:
    """Basename of a `modal volume ls --json` entry (key name varies by CLI ver)."""
    for k in ("path", "Filename", "filename", "name", "Name"):
        v = entry.get(k)
        if v:
            return str(v).rstrip("/").rsplit("/", 1)[-1]
    return ""


def _fetch_run_config(volumes: list, run_id: str, progress=None):
    """runs/<run_id>/run.json from the first outputs volume that has it, or None.
    Runs in a FuncWorker thread (each try blocks on a `modal volume get`)."""
    for vol in volumes:
        if progress:
            progress(f"  checking {vol}…")
        m = modal_cli.fetch_run_manifest(vol, run_id)
        if m:
            return m
    return None


def _check_weights_present(volume: str, run_id: str, progress=None):
    """runs/<run_id>/final_model.pth on the outputs volume?
    True=yes, False=missing (block), None=couldn't list. Runs in a FuncWorker thread."""
    entries = modal_cli.list_volume_entries(volume, f"/runs/{run_id}")
    if not entries:
        return None
    return any(_entry_name(e) == "final_model.pth" for e in entries)
