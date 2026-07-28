"""Inference page: pick weights + input folder, run --mode infer.
Modal: convert -> upload scenes -> [upload weights] -> run -> download.
Local: convert -> pixi run (TT_* env)."""

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
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu, QPushButton, QRadioButton,
                               QSpinBox, QVBoxLayout, QWidget)

from .. import analysis, appstate, dataset, local_cli, modal_cli, plots, pretrain, ui
from ..backbones import ALS_GRID_M, ALS_TILE_M, BACKBONES
from ..jobs import FuncWorker, JobRunner
from ..logconsole import LogConsole


class InferPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.converter = FuncWorker(self)
        self.preflight = FuncWorker(self)
        self.exporter = FuncWorker(self)
        self.runner = JobRunner(self)
        self._stage = ""
        self._job_id = ""
        self._staged: Path | None = None
        self._weights_remote = ""
        self._dl_dest: Path | None = None
        self._pred_dir: Path | None = None
        self._manifest: dict | None = None
        self._manifest_path: Path | None = None
        self._local_weights: Path | None = None
        self._dg: dict = {}
        self._run_class_names: list | None = None
        self._manifest_features: list | None = None
        self._hag_ground_value: int | None = None
        self._modal_cfg_run = ""
        self._run_tag = ""
        self._pending_cfg_run = ""
        self._ens_members: list[dict] = []
        self._ens_running = False
        self._ens_idx = -1
        self._ens_dirs: list[Path] = []
        self._run_open = False
        self._crs_probe = None
        self._crs_probe_name = ""
        self._crs_probe_path = ""
        self._declared_crs_epsg: int | None = None
        self._input_fields: list[str] = []   # extra columns in the probed input cloud
        self._input_std: set = set()         # standard channels it carries
        self._chan_combos: dict[str, QComboBox] = {}
        self._active_zeroed: list | None = None   # launch snapshots (frozen per run)
        self._active_cols: dict | None = None

        root = QVBoxLayout(self)
        title = QLabel("Inference")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.sub.setObjectName("pageSub")
        root.addWidget(self.sub)

        wbox = QGroupBox("Inputs")
        wf = self.wf = QFormLayout(wbox)
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
        self.runjson_edit = QLineEdit()
        self.runjson_edit.setPlaceholderText("…/local_runs/runs/<id>/run.json")
        self.runjson_edit.editingFinished.connect(self._load_run_manifest)
        self.runjson_edit.textChanged.connect(self._invalidate_manifest)
        rj_row = QHBoxLayout()
        rj_row.addWidget(self.runjson_edit, 1)
        rj_btn = QPushButton("Browse…")
        rj_btn.clicked.connect(self._pick_runjson)
        rj_row.addWidget(rj_btn)
        self.runjson_row_w = ui.wrap(rj_row)
        wf.addRow("Run file (run.json)", self.runjson_row_w)
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("default: the .pth named in run.json, beside it")
        w_row = QHBoxLayout()
        w_row.addWidget(self.weights_edit, 1)
        w_btn = QPushButton("Browse…")
        w_btn.clicked.connect(self._pick_weights)
        w_row.addWidget(w_btn)
        self.weights_row_w = ui.wrap(w_row)
        wf.addRow("Weights file", self.weights_row_w)
        self.run_combo = QComboBox()
        self.run_combo.setEditable(True)
        self.run_combo.lineEdit().setPlaceholderText(
            "run id - or paste it from the train log's 'run complete -> …' line")
        self.run_combo.currentIndexChanged.connect(self._on_run_pick)
        self.run_combo.lineEdit().editingFinished.connect(self._on_run_id_typed)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_combo, 1)
        self.dl_run_btn = QPushButton("Download run…")
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
        self.manifest_summary = QLabel("")
        self.manifest_summary.setObjectName("pageSub")
        self.manifest_summary.setWordWrap(True)
        wf.addRow(self.manifest_summary)
        self.backbone_combo = QComboBox()
        self.backbone_combo.currentIndexChanged.connect(self._sync_controls)
        wf.addRow("Architecture", self.backbone_combo)
        self.ens_box = QGroupBox("Ensemble (vote over several runs)")
        self.ens_box.setCheckable(True)
        self.ens_box.setChecked(False)
        ens_outer = QVBoxLayout(self.ens_box)
        ens_content = QWidget()
        self.ens_box.toggled.connect(ens_content.setVisible)
        ens_content.setVisible(False)
        ens_outer.addWidget(ens_content)
        ecol = QVBoxLayout(ens_content)
        ecol.setContentsMargins(0, 0, 0, 0)
        self.ens_list = QListWidget()
        self.ens_list.setMaximumHeight(96)
        ecol.addWidget(self.ens_list)
        erow = QHBoxLayout()
        ens_add = QPushButton("Add current selection")
        ens_add.clicked.connect(self._add_ens_member)
        ens_del = QPushButton("Remove selected")
        ens_del.clicked.connect(self._remove_ens_member)
        erow.addWidget(ens_add)
        erow.addWidget(ens_del)
        erow.addStretch()
        ecol.addLayout(erow)
        wf.addRow(self.ens_box)
        self.input_edit = QLineEdit()
        self.input_edit.editingFinished.connect(self._probe_crs)
        in_row = QHBoxLayout()
        in_row.addWidget(self.input_edit, 1)
        fold_btn = QPushButton("Folder…")
        fold_btn.clicked.connect(self._pick_input)
        file_btn = QPushButton("File…")
        file_btn.clicked.connect(self._pick_input_file)
        in_row.addWidget(fold_btn)
        in_row.addWidget(file_btn)
        wf.addRow("Point clouds (folder or file)", ui.wrap(in_row))
        wf.addRow("CRS", ui.crs_row(self, self._render_crs))

        ibox = QGroupBox("Inference settings")
        iform = self.iform = QFormLayout(ibox)
        iform.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.grid_spin = QDoubleSpinBox()
        self.grid_spin.setRange(0.02, 1_000_000.0)
        self.grid_spin.setSingleStep(0.05)
        self.grid_spin.setDecimals(2)
        self.grid_spin.setValue(ALS_GRID_M)
        iform.addRow("Grid size (m) - from the run", self.grid_spin)
        self.chunk_spin = QDoubleSpinBox()
        self.chunk_spin.setRange(10.0, 1_000_000.0)
        self.chunk_spin.setSingleStep(5.0)
        self.chunk_spin.setDecimals(0)
        self.chunk_spin.setValue(ALS_TILE_M)
        iform.addRow("Tile size (m)", self.chunk_spin)
        self.hag_chk = QCheckBox(ui.hag_title("Compute Height-Above-Ground (HAG)"))
        self.hag_chk.setToolTip("Bakes a per-point feat_hag channel into each converted "
                                "scene - required by runs trained with feat_hag. Pick "
                                "the ground source and interpolation below.")
        self.hag_chk.toggled.connect(lambda on: self.hag_opts_w.setVisible(on))
        iform.addRow("Height-Above-Ground", self.hag_chk)
        # same source × interpolation axes as the Datasets page; default CSF
        # because inference clouds are usually unlabeled
        self.hag_opts_w = ui.build_hag_options(
            self, filter_first=True, default_method="csf",
            ground_method_tip=(
                "Where ground comes from. Base off ground layer: a classification "
                "value already in the input clouds. CSF / SMRF: PDAL ground "
                "detection (needs PDAL). Z-min proxy: percentile-Z raster, no PDAL."),
            ground_tip=("Classification value in the input clouds that means "
                        "ground (e.g. 2). Those points are the ONLY ground "
                        "source - never mixed with detection."))
        self.hag_opts_w.setVisible(False)
        iform.addRow("", self.hag_opts_w)
        self._on_hag_method()
        self.adapt_combo = QComboBox()
        self.adapt_combo.addItems(["Off", "AdaBN", "APCoTTA"])
        self.adapt_combo.setToolTip(
            "Label-free adaptation to these scenes before predicting. "
            "KPConvX / RandLA only - the PTv3 family ignores it.\n"
            "AdaBN: recomputes BatchNorm statistics on the target tiles.\n"
            "APCoTTA (arXiv:2505.09971): AdaBN plus entropy-minimization steps "
            "on the BN scale/shift params, with stochastic restore toward the "
            "source weights. Slightly slower; can help on larger domain gaps.\n"
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
        dg_row.addWidget(QLabel("Adapt"))
        dg_row.addWidget(self.adapt_combo)
        dg_row.addWidget(QLabel("TTA"))
        dg_row.addWidget(self.tta_spin)
        dg_row.addWidget(self.probs_chk)
        dg_row.addStretch()
        iform.addRow("Domain adaptation", ui.wrap(dg_row))

        # per-channel source policy: calculated channels (feat_hag / feat_geo_*)
        # recompute or zero; found channels bind to a probed input column or zero
        self.chan_grid_host = QWidget()
        cg = QGridLayout(self.chan_grid_host)
        cg.setContentsMargins(0, 0, 0, 0)
        cg.setColumnStretch(1, 1)
        self.chan_hint = QLabel("")
        self.chan_hint.setObjectName("pageSub")
        self.chan_hint.setWordWrap(True)
        chan_col = QVBoxLayout()
        chan_col.addWidget(self.chan_grid_host)
        chan_col.addWidget(self.chan_hint)
        iform.addRow("Input channels", ui.wrap(chan_col))
        self._rebuild_chan_table()

        self.class_btn_host = QWidget()
        self.class_btn_row = QHBoxLayout(self.class_btn_host)
        self.class_btn_row.setContentsMargins(0, 0, 0, 0)
        self.class_btn_host.setToolTip(
            "Toggle off classes that don't exist in these scenes. Their probability is "
            "zeroed after vote accumulation and the rest renormalized, so each point "
            "falls to its next-best class; exported confidence (and the low-confidence "
            "gate above) are post-mask. Recorded in infer_run.json.")
        self._class_btns: list[QPushButton] = []
        self.class_mask_lbl = QLabel("")
        self.class_mask_lbl.setObjectName("pageSub")
        ccol = QVBoxLayout()
        ccol.addWidget(self.class_btn_host)
        ccol.addWidget(self.class_mask_lbl)
        iform.addRow("Classes (mask at launch)", ui.wrap(ccol))
        self._rebuild_class_list()

        # export-only knobs: the .npz keeps raw predictions, so re-export never re-runs inference
        self.fmt_combo = QComboBox()
        for label, key in (("LAS (.las)", "las"), ("LAZ (.laz)", "laz"),
                           ("PLY (.ply)", "ply"), ("Text (.txt)", "txt"),
                           ("CSV (.csv)", "csv")):
            self.fmt_combo.addItem(label, key)
        i = self.fmt_combo.findData(appstate.get("infer_format", "las"))
        self.fmt_combo.setCurrentIndex(i if i >= 0 else 0)
        self.fmt_combo.setToolTip("Predictions are written as xyz + classification "
                                  "(no colour columns).")
        iform.addRow("Prediction format", self.fmt_combo)
        self.unclass_chk = QCheckBox("Mark low-confidence points Unclassified")
        self.unclass_chk.setChecked(True)
        self.unclass_spin = QDoubleSpinBox()
        self.unclass_spin.setRange(0.0, 1.0)
        self.unclass_spin.setSingleStep(0.05)
        self.unclass_spin.setDecimals(2)
        self.unclass_spin.setValue(0.50)
        tip = ("Raw max-softmax confidence: points below the cut export as ASPRS "
               "class 1 (Unclassified - processed, no class assigned). The .npz "
               "keeps the raw prediction, so re-exporting at a new threshold "
               "never re-runs inference.")
        self.unclass_chk.setToolTip(tip)
        self.unclass_spin.setToolTip(tip)
        self.unclass_chk.toggled.connect(self.unclass_spin.setEnabled)
        uc_row = QHBoxLayout()
        uc_row.addWidget(self.unclass_chk)
        uc_row.addWidget(self.unclass_spin)
        uc_row.addStretch()
        iform.addRow("Confidence", ui.wrap(uc_row))

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run inference")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        run_row.addWidget(self.run_btn)
        self.kill_btn = QPushButton("Kill")
        self.kill_btn.setVisible(False)
        self.kill_btn.clicked.connect(self._kill)
        run_row.addWidget(self.kill_btn)
        self.compare_btn = QPushButton("Compare to ground truth…")
        self.compare_btn.setToolTip("Pick a prediction + its ground truth; accuracy, "
                                    "mIoU and per-class IoU print to the log.")
        self.compare_btn.clicked.connect(self._compare_gt)
        run_row.addWidget(self.compare_btn)
        run_row.addStretch()

        forms_col = QVBoxLayout()
        forms_col.addWidget(wbox)
        forms_col.addWidget(ibox)
        forms_col.addLayout(run_row)

        self.log = LogConsole()
        self.log.setPlaceholderText("Conversion and run logs appear here…")

        root.addWidget(ui.vsplit(ui.wrap(forms_col), self.log,
                                 sizes=[440, 324]), 1)

        self.converter.output.connect(self._append)
        self.converter.done.connect(self._on_converted)
        self.converter.error.connect(self._on_error)
        self.preflight.output.connect(self._append)
        self.preflight.done.connect(self._on_preflight)
        self.preflight.error.connect(self._on_preflight_error)
        self.exporter.output.connect(self._append)
        self.exporter.done.connect(self._on_exported)
        self.exporter.error.connect(self._on_export_error)
        self.voter = FuncWorker(self)
        self.voter.output.connect(self._append)
        self.voter.done.connect(self._on_voted)
        self.voter.error.connect(self._on_vote_error)
        self.cfg_fetcher = FuncWorker(self)
        self.cfg_fetcher.output.connect(self._append)
        self.cfg_fetcher.done.connect(self._on_cfg_fetched)
        self.cfg_fetcher.error.connect(
            lambda tb: self._append(f"✗ Run-config fetch failed:\n{tb}"))
        self.runner.output.connect(self._on_output)
        self.runner.finished.connect(self._on_stage_done)
        self.runner.failed.connect(self._on_runner_failed)
        self.dl_runner = JobRunner(self)
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
            self._invalidate_manifest()
        self.sub.setText(
            "Label point clouds with a trained model. "
            + ("Pick a run.json (or a local .pth), a folder of clouds, and run locally."
               if local else
               "Pick a run (or a local .pth), a folder of clouds, and run on Modal."))
        # ensembles are local-only; on Modal the box simply doesn't exist
        if not local:
            self.ens_box.setChecked(False)
        self.wf.setRowVisible(self.ens_box, local)
        self._sync_source_rows()
        self.reload_backbones()

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
        ui.repopulate_combo(self.backbone_combo,
                            [(b.label, key) for key, b in BACKBONES.items()])
        self._refresh_env_marks()
        self._sync_controls()

    def _refresh_env_marks(self):
        """Local mode: mark models whose pixi env isn't installed yet."""
        local = appstate.get_exec_mode() == "local"
        for i in range(self.backbone_combo.count()):
            b = BACKBONES.get(self.backbone_combo.itemData(i))
            if b is None:
                continue
            missing = local and not local_cli.installed(b, self.repo_root)
            self.backbone_combo.setItemText(
                i, b.label + ("  - env not installed" if missing else ""))

    def showEvent(self, ev):
        super().showEvent(ev)
        self._refresh_env_marks()

    def _set_run_classes(self, names):
        """Adopt the run's class names (they label the per-class IoU stats)."""
        self._run_class_names = list(names) if names else None
        self._rebuild_class_list()

    def _rebuild_class_list(self):
        """One toggled-on button per run class. Deliberately not persisted -
        all-on is the safe default."""
        while (it := self.class_btn_row.takeAt(0)) is not None:
            if it.widget() is not None:
                it.widget().deleteLater()
        self._class_btns = []
        names = self._run_class_names or []
        # ponytail: one flat row; add wrapping if a run ever ships dozens of classes
        for n in names:
            b = QPushButton(str(n))
            b.setCheckable(True)
            b.setChecked(True)
            b.toggled.connect(self._sync_class_mask_label)
            self.class_btn_row.addWidget(b)
            self._class_btns.append(b)
        if not names:
            ph = QLabel("(load a run to list its classes)")
            ph.setObjectName("pageSub")
            self.class_btn_row.addWidget(ph)
        self.class_btn_row.addStretch(1)
        self._sync_class_mask_label()

    def _excluded_classes(self) -> list[str]:
        """Toggled-off class names - the run's EXCLUDE_CLASSES env value."""
        return [b.text() for b in self._class_btns if not b.isChecked()]

    def _sync_class_mask_label(self, _item=None):
        exc, total = self._excluded_classes(), len(self._class_btns)
        if not exc:
            self.class_mask_lbl.setText("")
        elif total - len(exc) < 2:
            self.class_mask_lbl.setText("⚠ keep at least 2 classes enabled")
        else:
            self.class_mask_lbl.setText(
                f"masking {len(exc)} of {total}: {', '.join(exc)} - next-best class wins")

    @staticmethod
    def _chan_kind(name: str) -> str:
        """'found' = a custom feat_* column read from the input data; 'std' = a
        format-standard field readers fill. Calculated channels (feat_hag /
        feat_geo_*) are always recomputed at convert time and never listed."""
        return "found" if name.startswith("feat_") else "std"

    def _table_channels(self) -> list:
        """The run's DATA channels - the only inputs this table governs."""
        return [n for n in (self._run_features() or [])
                if n in ("intensity", "return_number", "rgb")
                or (n.startswith("feat_") and n != "feat_hag"
                    and not n.startswith("feat_geo_"))]

    def _meta_source_fields(self) -> dict:
        """channel name -> train-time source column, from the dataset meta."""
        chans = ((self._dataset_meta() or {}).get("source") or {}).get("feature_channels") or []
        return {f"feat_{c.get('name')}": str(c.get("source_field") or "")
                for c in chans if isinstance(c, dict) and c.get("name")}

    def _rebuild_chan_table(self):
        """One combo per DATA input: where it comes from in THESE clouds.
        auto = the standard/train-time source when present, zeros when the
        clouds don't carry it - missing data never blocks a run. Calculated
        channels are always recomputed and don't appear here."""
        if not hasattr(self, "chan_grid_host"):
            return
        prev = {n: c.currentData() for n, c in self._chan_combos.items()}
        grid = self.chan_grid_host.layout()
        while grid.count():
            it = grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._chan_combos = {}
        feats = self._table_channels()
        self.chan_grid_host.setVisible(bool(feats))
        if not feats:
            self.chan_hint.setText("(load a run to list its input channels)")
            return
        meta_src = self._meta_source_fields()
        probed = bool(self._crs_probe_name)
        cols_lower = {f.lower() for f in self._input_fields}
        for row, name in enumerate(feats):
            grid.addWidget(QLabel(name), row, 0)
            combo = QComboBox()
            if self._chan_kind(name) == "std":
                if probed and name not in self._input_std:
                    fb = {"intensity": "rgb-derived gray, else zeros",
                          "rgb": "a neutral mid-gray constant",
                          "return_number": "zeros"}[name]
                    combo.addItem(f"auto - missing here; model sees {fb}",
                                  ("auto", None))
                else:
                    combo.addItem("from the file's standard field", ("auto", None))
                if name != "rgb":   # rgb is 3-channel; column binding is 1-D only
                    for f in self._input_fields:
                        combo.addItem(f"column '{f}'", ("col", f))
            else:
                src = meta_src.get(name, "")
                lbl = (f"auto (train-time source: '{src}')" if src
                       else "auto (input column of the same name)")
                want = (src or name[len("feat_"):]).lower()
                if probed and not want.startswith("@") and want not in cols_lower:
                    lbl += " - missing here; model receives zeros"
                combo.addItem(lbl, ("auto", None))
                for f in self._input_fields:
                    combo.addItem(f"column '{f}'", ("col", f))
            combo.addItem("zeros (disable this input)", ("zeros", None))
            i = combo.findData(prev.get(name))
            combo.setCurrentIndex(i if i >= 0 else 0)
            grid.addWidget(combo, row, 1)
            self._chan_combos[name] = combo
        probe = (f"columns found in {self._crs_probe_name}: "
                 + (", ".join(self._input_fields) or "none")
                 if probed else "pick the input clouds to list their columns")
        self.chan_hint.setText(
            f"Where each data input comes from in these clouds ({probe}). "
            f"Missing data is fed as zeros; calculated channels (HAG, geometric "
            f"features) are always recomputed and not listed.")

    def _zeroed_channels(self) -> list[str]:
        """Channels the trainer must feed as zeros: user-disabled combos plus
        auto rows whose source column is absent from the probed input (missing
        data passes through as a zero field instead of blocking). Snapshot
        while a run is live: conversion, the post-convert report, the
        (possibly minutes-later) Modal env and every ensemble member must see
        ONE channel configuration, not live combo state."""
        if self._active_zeroed is not None:
            return self._active_zeroed
        zeroed = {n for n, c in self._chan_combos.items()
                  if (c.currentData() or ("auto",))[0] == "zeros"}
        meta_src = self._meta_source_fields()
        if self._crs_probe_name:
            cols = {f.lower() for f in self._input_fields}
            for n, c in self._chan_combos.items():
                if (not n.startswith("feat_")
                        or (c.currentData() or ("auto",))[0] != "auto"):
                    continue
                src = (meta_src.get(n) or n[len("feat_"):]).lower()
                if not src.startswith("@") and src not in cols:
                    zeroed.add(n)
        # geo channels the engine can't compute run as zeros
        geo_ok = {g.lower() for g in pretrain.GEO_FEATURES}
        for n in (self._run_features() or []):
            if not n.startswith("feat_geo_"):
                continue
            src = meta_src.get(n) or ""
            nm = src[len("@geo:"):] if src.startswith("@geo:") else n[len("feat_geo_"):]
            if nm.lower() not in geo_ok:
                zeroed.add(n)
        return sorted(zeroed)

    def _mapped_columns(self) -> dict:
        """channel -> user-bound input column (combos on 'column ...')."""
        if self._active_cols is not None:
            return self._active_cols
        return {n: c.currentData()[1] for n, c in self._chan_combos.items()
                if (c.currentData() or ("auto",))[0] == "col"}

    @staticmethod
    def _names_from_manifest(m: dict) -> list | None:
        """Class names from a manifest: class_names, else 'class 0..n-1', else None."""
        names = m.get("class_names")
        if names:
            return list(names)
        n = m.get("num_classes")
        return [f"class {i}" for i in range(int(n))] if n else None

    def reload_runs(self):
        prev = self.run_combo.currentText()
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        seen = set()
        for h in reversed(appstate.get("run_history", [])):
            if h["run_id"] not in seen:
                seen.add(h["run_id"])
                self.run_combo.addItem(f"{h['run_id']}  ({h['backbone']})", h)
        for root in appstate.run_roots(self.repo_root):
            for rdir in plots.discover_runs(root):
                if rdir.name in seen:
                    continue
                m = _manifest_in(rdir)
                if m is not None:
                    seen.add(rdir.name)
                    self.run_combo.addItem(f"{rdir.name}  ({m.get('backbone', '?')})",
                                           {"run_id": rdir.name, "backbone": m.get("backbone")})
        self.run_combo.setCurrentIndex(-1)
        self.run_combo.setEditText(prev)
        self.run_combo.blockSignals(False)
        self._on_run_pick()

    def _on_run_pick(self):
        """Sync architecture from the picked run; adopt its classes if downloaded."""
        h = self.run_combo.currentData()
        if self._modal_cfg_run and self._combo_run_ref()[1] != self._modal_cfg_run:
            self._invalidate_manifest()
        if isinstance(h, dict) and h.get("backbone") in BACKBONES:
            i = self.backbone_combo.findData(h["backbone"])
            if i >= 0:
                self.backbone_combo.setCurrentIndex(i)
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
        """(volume, run_id) from the run combo - typed text wins over
        currentData(); volume is '' unless pasted."""
        return _parse_run_ref(self.run_combo.currentText())

    def _download_run(self):
        """Fetch runs/<id> from the outputs volume to <workspace>/inference/<id>."""
        b = self._backbone()
        vol, run_id = self._combo_run_ref()
        if not (b and run_id):
            self._append("Pick (or type) a run id to download.")
            return
        if self.dl_runner.running:
            self._append("A run download is already in progress.")
            return
        volume = vol or b.outputs_volume
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
        self.reload_runs()
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
        self._apply_manifest_lock(False)
        self.hag_chk.setEnabled(True)
        self._set_run_classes(None)
        self._rebuild_chan_table()

    def _infer_dg_env(self, dg: dict | None = None) -> dict:
        """DG_* env for inference; logdk must be recomputed (it changed the
        input width). `dg` overrides the loaded run's block."""
        dg = self._dg if dg is None else dg
        env: dict[str, str] = {}
        if dg.get("logdk"):
            env["DG_LOGDK_FEAT"] = "1"
            env["DG_LOGDK_K"] = str(int(dg.get("logdk_k", 8)))
        adapt = self.adapt_combo.currentText()
        if adapt == "AdaBN":
            env["DG_INFER_ADABN"] = "1"
        elif adapt == "APCoTTA":
            env["DG_INFER_APCOTTA"] = "1"
        if self.tta_spin.value() > 0:
            env["DG_INFER_TTA"] = str(self.tta_spin.value())
        if self.probs_chk.isChecked():
            env["TT_SAVE_PROBS"] = "1"
        if env:
            self._append("[dg] inference: " + " ".join(f"{k}={v}" for k, v in sorted(env.items())))
        # class mask rides the same env dict, so it covers local, modal and ensemble alike
        exc = self._excluded_classes()
        if exc:
            env["EXCLUDE_CLASSES"] = ",".join(exc)
            self._append("[mask] excluding: " + ", ".join(exc)
                         + "; masked points fall to their next-best class")
        # channel kills ride the env like the class mask: local, modal, ensemble alike
        zc = self._zeroed_channels()
        if zc:
            env["TT_ZERO_CHANNELS"] = ",".join(zc)
            self._append("[channels] zero-filled inputs (missing or disabled): "
                         + ", ".join(zc) + " - the model runs without these signals")
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
        self.weights_edit.setText(str(self._local_weights))
        self._apply_manifest_lock(True)
        ok = "✓" if self._local_weights.is_file() else "✗ weights missing -"
        self._append(f"Loaded {p.name}: {m.get('backbone')}, grid={m.get('grid')}, "
                     f"chunk={m.get('chunk_xy')}, intensity={m.get('intensity_norm')}. "
                     f"{ok} {self._local_weights}")

    def _apply_manifest_fields(self, m: dict) -> bool:
        """Arch/grid/tile/HAG/DG/classes from a run manifest; False when the
        run's model can't be used here."""
        bkey = m.get("backbone")
        i = self.backbone_combo.findData(bkey)
        if i < 0:
            self._append(f"✗ Model '{bkey}' isn't available here. Enable it on the "
                         f"Train page, then reload this run.")
            return False
        self.backbone_combo.setCurrentIndex(i)
        if m.get("grid") is not None:
            self.grid_spin.setValue(float(m["grid"]))
        if m.get("chunk_xy") is not None and self.chunk_spin.isEnabled():
            self.chunk_spin.setValue(float(m["chunk_xy"]))
        self._dg = m.get("dg") or {}
        feats = m.get("features")
        need_hag = "feat_hag" in (feats or [])
        self.hag_chk.setChecked(need_hag)
        # a feat_hag run ALWAYS recomputes HAG; the user owns only ground/method
        self.hag_chk.setEnabled(not need_hag)
        if need_hag:
            chans = ((self._dataset_meta(m) or {}).get("source") or {}) \
                .get("feature_channels") or []
            src = next((str(c.get("source_field") or "") for c in chans
                        if isinstance(c, dict) and c.get("name") == "hag"), "")
            method = src[len("@hag:"):].split("+", 1)[0] if src.startswith("@hag:") else ""
            j = self.hag_filter.findText(method) if method else -1
            if j >= 0:
                self.hag_filter.setCurrentIndex(j)
            elif method not in ("", "source_dimension"):
                self._append(f"[hag] unknown train-time HAG method '{method}'; "
                             f"using {self.hag_filter.currentText()}.")
            self._append(f"[hag] run trained with feat_hag; HAG enabled, "
                         f"method={self.hag_filter.currentText()}.")
            # ground source is the user's call - train-time source doesn't bind it
        if self._dg.get("logdk"):
            self._append(f"[dg] trained with the log-d_k density channel "
                         f"(k={self._dg.get('logdk_k', 8)}); recomputed at inference.")
        self._manifest_features = feats
        # data channels only: alias feat_* collapse to canonical, geo/hag recompute
        custom = [n for n in (feats or [])
                  if n.startswith("feat_") and n != "feat_hag"
                  and not n.startswith("feat_geo_")
                  and dataset.canonical_channel(n) is None]
        if custom:
            self._append(f"[feat] run trained with data channel(s): {', '.join(custom)}. "
                         f"Bind each to an input column under 'Input channels' - "
                         f"or it rides as zeros when these clouds don't carry it.")
        self._set_run_classes(self._names_from_manifest(m))
        self._rebuild_chan_table()
        return True

    def _on_run_id_typed(self):
        """Modal: pull a typed/pasted run id's run.json (local copy first, else
        off the outputs volumes) so the inputs match the run."""
        if appstate.get_exec_mode() == "local" or not self.from_run_radio.isChecked():
            return
        vol, rid = self._combo_run_ref()
        if not rid or rid == self._modal_cfg_run or self.cfg_fetcher.running:
            return
        if self._manifest and self._manifest_path is None:
            self._invalidate_manifest()
        rdirs = appstate.runs_dir()
        cands = [appstate.workspace_dir() / "inference" / rid]
        cands += [bdir / rid for bdir in (rdirs.iterdir() if rdirs.exists() else [])]
        for d in cands:
            m = _manifest_in(d)
            if m is not None:
                self._apply_modal_manifest(m, rid)
                return
        if vol:
            vols = [vol]
        else:
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
            self._append(f"✗ No run.json for '{rid}' on any outputs volume. Set "
                         f"Architecture / grid / tile manually.")
            self._forget_run(rid)
            return
        if self._combo_run_ref()[1] != rid:
            return
        self._apply_modal_manifest(m, rid)

    def _forget_run(self, rid: str):
        """Purge a history run that's gone from Modal; a pasted id that was
        never in history is left alone."""
        hist = appstate.get("run_history", [])
        kept = [h for h in hist if h.get("run_id") != rid]
        if len(kept) == len(hist):
            return
        appstate.put("run_history", kept)
        if self._combo_run_ref()[1] == rid:
            self.run_combo.clearEditText()
        self.reload_runs()
        self._append(f"  (dropped '{rid}' from the run list; stale history entry.)")

    def _apply_modal_manifest(self, m: dict, rid: str):
        """Apply + lock a manifest resolved from a Modal run id (no local path)."""
        if not self._apply_manifest_fields(m):
            return
        self._manifest, self._manifest_path = m, None
        self._modal_cfg_run = rid
        self._apply_manifest_lock(True)
        self._append(f"✓ Run '{rid}': {m.get('backbone')}, grid={m.get('grid')}, "
                     f"chunk={m.get('chunk_xy')}, intensity={m.get('intensity_norm')}.")

    def _on_source_toggle(self):
        self._sync_source_rows()
        use_run = self.from_run_radio.isChecked() and self._manifest is not None
        self._apply_manifest_lock(use_run)
        self._set_run_classes(
            self._names_from_manifest(self._manifest) if use_run else None)
        # a stale table must not keep exporting kills for a model it wasn't built for
        self._rebuild_chan_table()

    def _apply_manifest_lock(self, locked: bool):
        """Grey out what a run.json dictates (arch, grid, tile); while a manifest
        is applied the three rows fold into the manifest_summary line."""
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
        """The explicit override if set, else the run.json's sibling."""
        t = self.weights_edit.text().strip()
        return Path(t) if t else self._local_weights

    def _weights_run_tag(self, member: dict | None = None) -> str:
        """Parent folder for prediction dirs: the weights' run id (loose .pth ->
        filename stem; the voted ensemble output goes under 'ensemble')."""
        if member:
            mp = member.get("manifest_path")
            return Path(mp).parent.name if mp else Path(member["weights"]).stem
        return self._run_tag or "adhoc"

    def _sync_controls(self):
        """Auto-fill grid + tile from the backbone's defaults; disable tile for RandLA."""
        if self.backbone_combo.currentData() is None:
            return
        if self._manifest is not None and self.from_run_radio.isChecked():
            return
        b = self._backbone()
        gp = next((p for p in b.params if p.flag == b.grid_flag), None)
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
                         + ". Add one with `pixi add`, or use Browse…")
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
            self._probe_crs()

    def _pick_input_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Point-cloud file to label", appstate.get("last_view_dir", ""),
            "Point clouds (*.ply *.npz *.las *.laz *.txt *.csv *.pcd *.xyz *.pts);;All files (*)")
        if f:
            self.input_edit.setText(f)
            self._probe_crs()

    def _probe_crs(self):
        """Read the first input cloud's CRS for the surface + D1 preflight (one
        file, like the scene-channel check; reuses readers, no reprojection here)."""
        path = self.input_edit.text().strip()
        self._crs_probe, self._crs_probe_name, self._crs_probe_path = None, "", path
        self._input_fields, self._input_std = [], set()
        files = dataset.expand_inputs(path) if path and os.path.exists(path) else []
        cloud = (ui.stamp_crs_probe(self, files, "Could not read CRS from {name}: {err}")
                 if files else None)
        if cloud is not None:
            self._input_fields = sorted(cloud.fields.keys())
            self._input_std = {c for c, a in (("intensity", cloud.intensity),
                                              ("rgb", cloud.rgb),
                                              ("return_number", cloud.return_number))
                               if a is not None}
        self._render_crs()
        self._rebuild_chan_table()

    def _render_crs(self, *_):
        ui.render_crs_status(self, "Run")

    def _backbone(self):
        if (self._manifest and self.from_run_radio.isChecked()
                and appstate.get_exec_mode() == "local"
                and self._manifest.get("backbone") in BACKBONES):
            return BACKBONES[self._manifest["backbone"]]
        return BACKBONES[self.backbone_combo.currentData()]

    def _on_hag_method(self):
        ui.sync_hag_method(self)

    def _check_hag(self) -> bool:
        """Reconcile the HAG box with the run's feature spec and parse the
        ground class; sets self._hag_ground_value."""
        self._hag_ground_value = None
        need = "feat_hag" in (self._run_features() or [])
        if need and not self.hag_chk.isChecked():
            self.hag_chk.setChecked(True)
            self._append("[hag] run trained with feat_hag; enabling "
                         "'Compute Height-Above-Ground' for the conversion.")
        elif self.hag_chk.isChecked() and not need:
            self._append("[hag] note: this run doesn't use feat_hag; computing "
                         "HAG only costs conversion time.")
        if self.hag_chk.isChecked() \
                and self.hag_ground_method.currentData() == "labels":
            gtxt = self.hag_ground.text().strip()
            try:
                self._hag_ground_value = int(gtxt)
            except ValueError:
                self._append("HAG 'Base off ground layer' needs a ground class - "
                             "enter the classification value that means ground "
                             "(e.g. 2), or pick CSF / SMRF / Z-min proxy.")
                return False
        return True

    def _add_ens_member(self):
        """Snapshot the currently configured model as an ensemble member so
        later UI edits don't drift it."""
        if self.from_file_radio.isChecked():
            self._append("✗ ensemble: members must come from training runs "
                         "(run.json). A bare .pth carries no class/dataset info "
                         "to check against.")
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
        # members must share one class set AND one dataset: feature channels and intensity_norm resolve through the first member's dataset meta
        new_n = manifest.get("num_classes")
        new_names = self._names_from_manifest(manifest)
        new_ds = manifest.get("dataset")
        for m in self._ens_members:
            mm = m.get("manifest") or {}
            old_n = mm.get("num_classes")
            old_names = self._names_from_manifest(mm)
            if (old_n and new_n and int(old_n) != int(new_n)) or \
                    (old_names and new_names and old_names != new_names):
                self._append(f"✗ ensemble: class mismatch. '{Path(m['weights']).name}' "
                             f"predicts {old_n} classes {old_names}, this run {new_n} "
                             f"{new_names}. Members must share one class set.")
                return
            old_ds = mm.get("dataset")
            if old_ds and new_ds and old_ds != new_ds:
                self._append(f"✗ ensemble: dataset mismatch. '{Path(m['weights']).name}' "
                             f"was trained on '{old_ds}', this run on '{new_ds}'. "
                             f"Members must share one dataset (feature channels and "
                             f"intensity normalization come from its meta).")
                return
        self._ens_members.append({
            "backbone": bkey, "weights": str(w), "manifest_path": mpath,
            "manifest": manifest, "dg": (manifest or {}).get("dg") or {},
            "grid": self.grid_spin.value(), "chunk": self.chunk_spin.value()})
        src = Path(mpath).parent.name if mpath else w.name
        self.ens_list.addItem(f"{bkey} - {src}")
        self._append(f"[ensemble] member {len(self._ens_members)}: {bkey} ({w})")

    def _remove_ens_member(self):
        row = self.ens_list.currentRow()
        if row < 0:
            return
        self.ens_list.takeItem(row)
        self._ens_members.pop(row)

    def _run_ensemble(self, input_dir: str):
        """Validate members, stage scenes once, run members sequentially."""
        n = len(self._ens_members)
        if n < 2:
            self._append("✗ ensemble needs at least 2 members. 'Add current "
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
        if not self._check_hag():
            self._ens_running = False
            self.run_btn.setEnabled(True)
            self._end_run("✗ aborted before launch (HAG settings)")
            return
        if n == 2:
            self._append("[ensemble] ⚠ only 2 members; disagreements fall to the "
                         "more confident (then earlier) model. 3+ recommended.")
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
        """Keep the member's infer_run.json beside its predictions (the vote's
        class clamp reads it), then next member or vote."""
        if not list(self._pred_dir.glob("*_pred.npz")):
            self._ens_running = False
            self.run_btn.setEnabled(True)
            self._append(f"\n✗ ensemble: member {self._ens_idx + 1} wrote no "
                         f"predictions in {self._pred_dir}. Check the log above.")
            self._end_run(f"✗ ensemble member {self._ens_idx + 1} wrote no predictions")
            return
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
        self._report_predictions(Path(ens_dir))

    def _on_vote_error(self, tb: str):
        self._ens_running = False
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ ensemble vote failed. The per-member predictions "
                     f"remain in their predictions_{self._job_id}_m<k> folders.\n{tb}")
        self._end_run("✗ ensemble vote failed")

    def _run(self):
        input_dir = self.input_edit.text().strip()
        if not os.path.exists(input_dir):
            self._append("Choose an input folder or file first.")
            return
        def reprobe_stale():
            if self._crs_probe_path != input_dir:
                self._probe_crs()
        ok, declared = ui.crs_launch_gate(self, "Run", err_prefix="✗ ",
                                          before_block=reprobe_stale)
        if not ok:
            return
        self._declared_crs_epsg = declared
        exc = self._excluded_classes()
        if exc and len(self._class_btns) - len(exc) < 2:
            self._append("✗ Class mask: keep at least 2 classes enabled; a one-class "
                         "prediction is meaningless.")
            return
        modal = appstate.get_exec_mode() != "local"
        if self.ens_box.isChecked() and not modal:
            self._run_ensemble(input_dir)
            return
        weights_run_id = ""
        weights_vol = ""
        if self.from_file_radio.isChecked():
            if not os.path.isfile(self.pth_edit.text().strip()):
                self._append("Choose a .pth file.")
                return
            self._weights_remote = f"uploads/{Path(self.pth_edit.text()).name}"
            bkey = self.backbone_combo.currentData()
            self._run_tag = Path(self.pth_edit.text().strip()).stem
        elif not modal:
            if not (self._manifest and self._manifest_path):
                self._append("Pick a run.json first.")
                return
            w = self._resolved_weights()
            if not (w and w.is_file()):
                self._append(f"✗ Weights not found ({w}). Set the 'Weights file' box.")
                return
            bkey = self._manifest.get("backbone")
            self._run_tag = self._manifest_path.parent.name
        else:
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
        # runs enter history at train start and may lack final_model.pth, so check before paying (a downloaded run provably has weights)
        if modal and weights_run_id:
            if self._has_local_run_copy(weights_run_id):
                self._append(f"[0/4] Weights verified from the local download of "
                             f"'{weights_run_id}'; skipping the Modal check.")
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
        if self._ens_running:
            norm = (self._ens_members[0].get("manifest") or {}).get(
                "intensity_norm") or "p95"
        else:
            norm = (self._manifest or {}).get("intensity_norm", "p95") \
                if (self.from_run_radio.isChecked() and self._manifest) else "p95"
        hag_on = self.hag_chk.isChecked()
        if self._ens_running and not hag_on:
            hag_on = any("feat_hag" in ((m.get("manifest") or {}).get("features") or [])
                         for m in self._ens_members)
            if hag_on:
                self._append("[hag] re-enabled for the ensemble: a member's spec "
                             "includes feat_hag (the last-loaded run didn't).")
        hag_filter = self.hag_filter.currentText() if hag_on else "grid"
        ground_method = self.hag_ground_method.currentData()
        if ground_method == "labels" and self._hag_ground_value is None:
            # ensemble re-enable path skipped _check_hag's parse; don't launch
            # 'labels' without a class
            ground_method = "csf"
            self._append("[hag] no ground class set - using CSF detection.")
        if hag_on:
            src = (f"ground = class {self._hag_ground_value}"
                   if ground_method == "labels"
                   else f"{pretrain.GROUND_LABELS.get(ground_method, ground_method)} detection")
            self._append(f"[1/4] Computing HeightAboveGround ({hag_filter}, {src}) "
                         "for the input scenes.")
        job_root = self._infer_out_dir()
        fields, geo, geo_k = self._infer_feature_fields()
        mapped = self._mapped_columns()
        if mapped:
            self._append("[channels] bound inputs: "
                         + ", ".join(f"{k} <- '{v}'" for k, v in sorted(mapped.items())))
        if geo:
            self._append(f"[1/4] Recomputing geometric feature(s) "
                         f"{', '.join(geo)} (pgeof optimal, k≤{geo_k}) per scene.")
        self._append(f"[1/4] Converting {input_dir} to scenes (job {self._job_id}; "
                     f"intensity={norm}) -> {job_root}…")
        crs_kw = ({"declared_crs_epsg": self._declared_crs_epsg}
                  if self._declared_crs_epsg is not None else {})
        self.converter.start(dataset.convert_infer_job, self._job_id, input_dir,
                             appstate.workspace_dir(), intensity_norm=norm,
                             hag=hag_on, hag_filter=hag_filter,
                             ground_value=self._hag_ground_value,
                             ground_method=ground_method,
                             feature_fields=fields, geo_features=geo,
                             geo_k=geo_k, out_dir=job_root, **crs_kw)

    def _run_features(self) -> list | None:
        """The run's ordered input spec (run.json "features"), or None (loose
        .pth). Ensemble: the union of every member's features.
        feat_intensity/feat_return_number aliases collapse to the canonical
        channel (mirrors parse_feat_spec trainer-side)."""
        if self._ens_running:
            feats: list = []
            for m in self._ens_members:
                for f in (m.get("manifest") or {}).get("features") or []:
                    if f not in feats:
                        feats.append(f)
        else:
            feats = list(self._manifest_features or []) \
                if (self.from_run_radio.isChecked() and self._manifest) else []
        feats = list(dict.fromkeys(dataset.canonical_channel(n) or n
                                   for n in feats))
        return feats or None

    def _infer_feature_fields(self) -> tuple:
        """(raw source fields, pgeof geo names, geo_k) for the run's custom
        feat_* channels, resolved through the dataset meta; unresolved names
        fall back with a logged warning. Channel-table overrides win: zeroed
        channels are skipped entirely (nothing computed for a dead input) and
        user-bound columns ride as 'target=Column' entries."""
        zeroed = set(self._zeroed_channels())
        mapped = self._mapped_columns()
        custom = [n for n in (self._run_features() or [])
                  if n.startswith("feat_") and n != "feat_hag" and n not in zeroed]
        fields = [f"{c}={mapped[c]}" for c in ("intensity", "return_number")
                  if c in mapped]
        if not custom:
            return fields or None, None, 100
        geo_by_lower = {n.lower(): n for n in pretrain.GEO_FEATURES}
        chans = ((self._dataset_meta() or {}).get("source") or {}).get("feature_channels") or []
        by_name = {c.get("name"): c for c in chans if isinstance(c, dict)}
        geo, geo_k = [], None
        for n in custom:
            if n in mapped:
                fields.append(f"{n[len('feat_'):]}={mapped[n]}")
                continue
            c = by_name.get(n[len("feat_"):]) or {}
            src = c.get("source_field") or ""
            if src.startswith("@geo:"):
                nm = src[len("@geo:"):]
                if nm not in pretrain.GEO_FEATURES:
                    self._append(f"[feat] ⚠ '{nm}' isn't a supported geometric "
                                 f"feature; feeding zeros. Rebuild this dataset "
                                 f"to recompute it.")
                    continue
                geo.append(nm)
                k = c.get("k")
                if k is not None and geo_k is not None and int(k) != geo_k:
                    self._append(f"[feat] ⚠ mixed geo k in meta ({geo_k} vs "
                                 f"{int(k)}); using {geo_k}.")
                elif k is not None:
                    geo_k = int(k)
            elif src:
                fields.append(src)
            elif n.startswith("feat_geo_") and n[len("feat_geo_"):] in geo_by_lower:
                nm = geo_by_lower[n[len("feat_geo_"):]]
                self._append(f"[feat] no dataset meta for '{n}'; recomputing "
                             f"pgeof '{nm}' at the DEFAULT k=100 (train k unknown).")
                geo.append(nm)
            else:
                fields.append(n[len("feat_"):])
                self._append(f"[feat] no dataset meta maps '{n}' to a raw field; "
                             f"assuming the inputs carry a field named "
                             f"'{n[len('feat_'):]}'.")
        return fields or None, geo or None, (geo_k if geo_k is not None else 100)

    def _infer_out_dir(self) -> Path:
        """<dataset>/infer/<job> when the run names a known on-disk dataset,
        else a workspace scratch spot."""
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
        self._append(f"[0/4] (weights check errored, proceeding anyway)\n{tb}")
        self._start_conversion(self._pending_input)

    def _on_converted(self, staged: Path):
        self._staged = staged
        lines, blocked = _scene_channel_report(staged,
                                               features=self._run_features(),
                                               zeroed=self._zeroed_channels())
        for line in lines:
            self._append(line)
        if blocked:
            self._append("✗ Aborting: calculated channel(s) this run needs failed to "
                         "appear during conversion (see above). Check the HAG / "
                         "geometric-feature settings and the conversion log, then "
                         "run again.")
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
        self._ds_vol = appstate.modal_datasets_volume()
        self._append(f"[2/4] Uploading scenes -> {self._ds_vol}:/_infer/{self._job_id}… "
                     "(a 'volume already exists' message here is expected and harmless)")
        self._stage = "upload_scenes"
        prog, args = modal_cli.volume_put(self._ds_vol, str(staged),
                                          f"/_infer/{self._job_id}")
        self.runner.start(prog, args, cwd=self.repo_root,
                          pre=modal_cli.volume_create(self._ds_vol))

    def _on_stage_done(self, code: int):
        if code != 0:
            if self._ens_running:
                self._append(f"\n✗ ensemble member {self._ens_idx + 1} failed "
                             f"(exit {code}); ensemble aborted.")
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
            self._dl_dest = (appstate.workspace_dir() / "inference"
                             / self._weights_run_tag()
                             / f"predictions_{self._job_id}")
            self._dl_dest.mkdir(parents=True, exist_ok=True)
            self._append(f"[4/4] Downloading predictions -> {self._dl_dest}…")
            self._stage = "download"
            prog, args = modal_cli.volume_get(self._ds_vol,
                                              f"_infer/{self._job_id}/predictions",
                                              str(self._dl_dest))
            self.runner.start(prog, args, cwd=self.repo_root)
        elif self._stage == "run_local":
            if self._ens_running:
                self._on_member_done()
            else:
                self._report_predictions(self._pred_dir)
        elif self._stage == "download":
            self._report_predictions(self._dl_dest / "predictions")

    def _report_predictions(self, pred_dir):
        """Verify predictions landed (a stage can exit 0 yet write nothing),
        then export them as the chosen format on a worker thread."""
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
        """Dataset the active weights belong to (ensemble: the first member's)."""
        if self._ens_running:
            return (self._ens_members[0].get("manifest") or {}).get("dataset")
        return (self._manifest or {}).get("dataset") \
            if (self.from_run_radio.isChecked() and self._manifest) else None

    def _dataset_meta(self, manifest: dict | None = None) -> dict | None:
        """The run's dataset_meta.json, or None. `manifest` overrides
        self._manifest (_apply_manifest_fields runs before it's adopted)."""
        name = manifest.get("dataset") if manifest else self._owning_dataset()
        mp = appstate.known_datasets().get(name or "", {}).get("meta_path", "")
        try:
            with open(mp, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _class_map(self) -> dict | None:
        """{model index: source classification value} from the run's dataset
        meta; None (identity) when the dataset can't be resolved."""
        try:
            cmap = {int(c["index"]): int((c.get("source_values") or
                                          [c["source_value"]])[0])
                    for c in (self._dataset_meta() or {}).get("classes", [])}
        except (ValueError, KeyError, TypeError, IndexError):
            cmap = None
        if not cmap:
            self._append("[export] no dataset meta for these weights; exported "
                         "codes are raw model indices.")
            return None
        return cmap

    def _on_exported(self, written):
        self._ens_running = False
        self.run_btn.setEnabled(True)
        if not written:
            self._append("✗ Nothing exported (no *_pred.npz in the predictions folder).")
            self._end_run("✗ nothing exported")
            return
        self._append(f"\n✓ Done - {len(written)} prediction file(s) in {written[0].parent}.\n"
                     f"  'Compare to ground truth…' for accuracy + mIoU.")
        self._end_run(f"✓ exported {len(written)} prediction file(s)")

    def _on_export_error(self, tb: str):
        self._ens_running = False
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Format conversion failed; predictions remain as raw "
                     f".npz files.\n{tb}")
        self._end_run("✗ export failed (raw .npz kept)")

    def _start_modal_run(self):
        b = self._backbone()
        flags = {
            "mode": "infer",
            "weights": self._weights_remote,
            "infer-input": self._job_id,
            b.grid_flag: self.grid_spin.value(),
        }
        if b.has_chunk:
            flags["chunk-xy"] = self.chunk_spin.value()
        self._append(f"[3/4] Running inference on Modal ({b.label})…")
        self._stage = "run"
        prog, args = modal_cli.run_script(b.script, flags, detach=False,
                                          env=self._infer_dg_env())
        self._append(f"$ TT_DATASET_VOLUME={self._ds_vol} modal {' '.join(args)}\n")
        self.runner.start(prog, args, cwd=self.repo_root,
                          extra_env={"TT_DATASET_VOLUME": self._ds_vol})

    def _start_local_infer(self, member: dict | None = None):
        """Local pixi inference via the TT_* env contract; `member` overrides
        the UI-derived backbone/weights/grid (ensemble)."""
        b = BACKBONES[member["backbone"]] if member else self._backbone()
        suffix = f"_m{self._ens_idx + 1}" if member else ""
        self._pred_dir = (appstate.workspace_dir() / "inference"
                          / self._weights_run_tag(member)
                          / f"predictions_{self._job_id}{suffix}")
        self._pred_dir.mkdir(parents=True, exist_ok=True)
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
            env["TT_SAVE_PROBS"] = "1"
        self._stage = "run_local"
        prog, args, run_env = local_cli.run_script(
            b.script, flags, b, repo_root=self.repo_root,
            infer_dir=str(self._staged), pred_dir=str(self._pred_dir), env=env)
        self._append(f"[local] Running inference in the pixi env ({b.label})…")
        self._append(f"[local] $ {local_cli.preview(prog, args, run_env)}\n")
        gok, gmsg = local_cli.gpu_preflight()
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
        disp = (_localize_paths(text, self._job_id, self._pred_dir, self._staged)
                if self._stage == "run_local" else text)
        self._append(disp, newline=False)

    def _on_error(self, tb: str):
        self._ens_running = False
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Conversion error:\n{tb}")
        self._end_run("✗ conversion error")

    def _on_runner_failed(self, err: str):
        self._ens_running = False
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Failed to start: {err}")
        self._end_run("✗ failed to start")

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
        self._append(f"\nComparing {Path(pred).name} to {Path(gt).name}; "
                     f"computing accuracy + mIoU…")
        try:
            m = analysis.prediction_metrics(pred, gt)
        except Exception as e:
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
        mpath = Path(pred).with_suffix(".metrics.json")
        try:
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump({"prediction": str(pred), "ground_truth": str(gt),
                           "class_names": names, **m}, f, indent=2)
            lines.append(f"  saved -> {mpath}")
        except OSError as e:
            lines.append(f"  (couldn't save metrics json: {e})")
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

    def _kill(self):
        """Hard-kill the live job: the subprocess tree plus any worker stage
        (conversion / vote / export), local and Modal alike."""
        killed = False
        if self.runner.running:
            self.runner.terminate()
            killed = True
        for w in (self.converter, self.preflight, self.voter, self.exporter):
            if w.running:
                w.cancel()
                killed = True
        if not killed:
            self._append("\n[no process running]")
            return
        self._append("\n[killed]")
        if appstate.get_exec_mode() != "local":
            self._append("[modal] note: killing the local client can leave the "
                         "cloud app running. `modal app list` to check, "
                         "`modal app stop <app>` to stop it.")
        self._end_run("✗ killed")

    def _begin_run(self, title: str):
        """Run header; earlier runs stay scrollable above the divider. Also
        freezes the channel table: the run uses this snapshot everywhere."""
        self._run_open = True
        self._active_zeroed = self._zeroed_channels()
        self._active_cols = self._mapped_columns()
        self.chan_grid_host.setEnabled(False)
        self.kill_btn.setVisible(True)
        self.log.begin_run(title)

    def _end_run(self, summary: str):
        """Close the current run header exactly once (terminal points can
        overlap, e.g. export error after a stage failure)."""
        self._active_zeroed = self._active_cols = None
        self.chan_grid_host.setEnabled(True)
        self.kill_btn.setVisible(False)
        if self._run_open:
            self._run_open = False
            self.log.end_run(summary)

    def _append(self, text: str, newline: bool = True):
        ui.append_log(self.log, text, newline)


def _vote_members(member_dirs: list, out_dir: str, progress=None):
    """Run ensemble_vote.py over the member prediction dirs, then strip the
    big 'probs' payload from each member npz."""
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
    say("  (dropped the per-member probs payloads; classification npz kept. "
        "Re-running ensemble_vote.py over these dirs will HARD-vote, which can "
        "give different labels than the soft vote above)")
    return Path(out_dir)


def _localize_paths(text: str, job_id: str, pred_dir, staged) -> str:
    """Rewrite container bind-mount paths to the host folders they map to;
    predictions first (the more specific mount), then the staging root."""
    if pred_dir:
        for p in (f"/datasets/_infer/{job_id}/predictions", f"_infer/{job_id}/predictions"):
            text = text.replace(p, str(pred_dir))
    if staged:
        for p in (f"/datasets/_infer/{job_id}", f"_infer/{job_id}"):
            text = text.replace(p, str(staged))
    return text


def _scene_channel_report(staged, features: list | None = None,
                          zeroed=()) -> tuple[list[str], bool]:
    """(log lines, blocking) - verify the converted scenes carry what the run
    needs. Only CALCULATED channels (feat_hag / feat_geo_*) can block: they are
    always recomputed, so their absence means the conversion itself failed.
    Data channels (intensity / return_number / rgb / custom feat_*) pass
    through as zeros or a documented fallback when missing; `zeroed` lists the
    ones already riding TT_ZERO_CHANNELS."""
    import numpy as np
    zeroed = set(zeroed)
    scenes = sorted(Path(staged).glob("scenes/*.npz"))
    if not scenes:
        return [f"⚠ no converted scenes under {staged} to check."], False
    hard = [n for n in (features or [])
            if (n == "feat_hag" or n.startswith("feat_geo_")) and n not in zeroed]
    soft = [n for n in (features or [])
            if n.startswith("feat_") and n not in hard and n not in zeroed]
    want_i = "intensity" not in zeroed and (features is None or "intensity" in features)
    want_r = (features is not None and "return_number" in features
              and "return_number" not in zeroed)
    missing_i, missing_r, keys0 = [], [], []
    missing_hard: dict[str, list[str]] = {}
    missing_soft: dict[str, list[str]] = {}
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
        for ch in soft:
            if ch not in names:
                missing_soft.setdefault(ch, []).append(p.stem)
    lines = [f"[check] {len(scenes)} scene(s) carry: {', '.join(keys0)}; this npz "
             f"is exactly what the model reads."]
    if zeroed:
        lines.append(f"○ fed as zeros (missing or disabled): "
                     f"{', '.join(sorted(zeroed))} - the model runs without "
                     f"these signals.")
    for ch, lost in missing_soft.items():
        lines.append(f"⚠ '{ch}' missing in: {', '.join(lost)} and not set to "
                     f"zeros - the run will abort naming it. Set the channel to "
                     f"'zeros' under Input channels, or re-probe the inputs.")
    if want_i and missing_i:
        lines.append(f"⚠ no intensity channel in: {', '.join(missing_i)}. The source "
                     f"file(s) had no intensity field. The trainer substitutes a "
                     f"constant filler, so models trained with real intensity see "
                     f"an unfamiliar value and accuracy will suffer.")
    elif want_i:
        with np.load(scenes[0]) as z:
            i = z["intensity"]
        lines.append(f"✓ intensity in all scenes ({scenes[0].stem}: "
                     f"min {float(i.min()):.2f}, max {float(i.max()):.2f})")
        if float(i.max()) <= 0.0:
            lines.append("⚠ intensity is all zeros; the source's intensity field is "
                         "empty. Expect degraded accuracy.")
    if missing_r:
        lines.append(f"⚠ no return_number channel in: {', '.join(missing_r)}. The "
                     f"trainer feeds zeros there. Models trained with real return "
                     f"numbers see an unfamiliar constant and accuracy will suffer.")
    for ch, lost in missing_hard.items():
        hint = (" Tick 'Compute Height-Above-Ground' in the conversion box and "
                "run again." if ch == "feat_hag" else "")
        lines.append(f"✗ calculated channel '{ch}' missing in: {', '.join(lost)} - "
                     f"its computation failed during conversion; check the log "
                     f"above, then run again.{hint}")
    return lines, bool(missing_hard)


def _manifest_in(rdir: Path) -> dict | None:
    """run.json in a run folder, or None."""
    p = Path(rdir) / "run.json"
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
