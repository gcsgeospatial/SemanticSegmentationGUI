"""Datasets page: point at clouds, name classes, split train/val/test, Build.
Intensity is p95-normalized (i/p95 clipped to 0..2) — the single norm across
build + train + inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import analysis, appstate, dataset, modal_cli, pretrain, theme, ui
from ..dataset import LabelSpec, SplitConfig
from ..jobs import FuncWorker, JobRunner
from ..logconsole import LogConsole
from ..readers import read_points


class DatasetsPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.worker = FuncWorker(self)
        self.uploader = JobRunner(self)
        self._staged_dir: Path | None = None
        self._uploading: Path | None = None
        self._label_values: dict[int, int] = {}
        self._done_cb = None
        self._scanned_for: str | None = None
        self._copied_classes: dict[int, tuple[str, bool]] = {}
        self._run_open = False
        # (source_wkt, proc_wkt, looks_degrees) of the probed first input, or None
        self._crs_probe = None
        self._crs_probe_name = ""

        root = QVBoxLayout(self)
        title = QLabel("Datasets")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.sub.setObjectName("pageSub")
        root.addWidget(self.sub)

        root.addWidget(self._new_dataset_box())    # 1
        root.addWidget(self._classes_box())        # 2
        root.addWidget(self._features_box())       # 3  input features
        root.addWidget(self._calculated_box())     # 4  computed channels: HAG + geometric
        root.addWidget(self._tiling_box())         # 5  split + build
        root.addLayout(self._status_block())

        # ---- saved datasets: bottom layer ----
        root.addWidget(QLabel("Saved Datasets"))
        sd_row = QHBoxLayout()
        sd_col = QVBoxLayout()
        self.known_list = QListWidget()
        self.known_list.setMaximumHeight(120)
        self.known_list.setMaximumWidth(360)
        self.known_list.itemSelectionChanged.connect(self._show_known)
        sd_col.addWidget(self.known_list)
        self.add_existing_btn = QPushButton("Add existing dataset…")
        self.add_existing_btn.clicked.connect(self._add_existing)
        sd_col.addWidget(self.add_existing_btn)
        self.upload_saved_btn = QPushButton("Upload selected to Modal")
        self.upload_saved_btn.clicked.connect(self._upload_saved)
        sd_col.addWidget(self.upload_saved_btn)
        self.delete_saved_btn = QPushButton("Delete selected")
        self.delete_saved_btn.clicked.connect(self._delete_saved)
        sd_col.addWidget(self.delete_saved_btn)
        sd_row.addLayout(sd_col)
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setAlignment(Qt.AlignTop)
        theme.set_accent(self.stats_label, "muted")
        sd_row.addWidget(self.stats_label, 1)
        root.addLayout(sd_row)
        self._reload_known()

        self._on_split_changed()
        # classes/build stay greyed until a label scan has run for the picked input
        self.input_edit.textChanged.connect(self._update_scan_gate)
        self._update_scan_gate()

        self.worker.output.connect(self._append)
        self.worker.done.connect(self._dispatch_done)
        self.worker.error.connect(self._on_worker_error)
        self.worker.stopped.connect(self._on_stopped)
        self.uploader.output.connect(lambda s: self._append(s, newline=False))
        self.uploader.finished.connect(self._on_upload_done)
        self.uploader.failed.connect(self._on_upload_failed)
        self.apply_exec_mode(appstate.get_exec_mode() == "local")

    def apply_exec_mode(self, local: bool):
        """Local mode never uploads to Modal; hide the upload button, reword the copy."""
        self.upload_saved_btn.setVisible(not local)
        self.sub.setText(
            "Point at clouds (las/laz, ply, txt/csv, pcd, npy/npz), name classes, "
            "split train/val, Build."
            + (" Staged on disk, ready to train locally."
               if local else
               " Then upload to a per-dataset Modal volume."))
        self._reload_known()
        self._refresh_next_btn()

    # ============================================================= 1. New dataset
    def _new_dataset_box(self) -> QWidget:
        box = QGroupBox("1 · New dataset")
        form = QFormLayout(box)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("my_city_lidar")
        form.addRow("Name", self.name_edit)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText(".laz file or a folder of clouds")
        in_row = QHBoxLayout()
        in_row.addWidget(self.input_edit)
        for text, slot in (("Folder…", self._pick_input_folder), ("File…", self._pick_input_file)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            in_row.addWidget(b)
        form.addRow("Input", ui.wrap(in_row))
        self.crs_status = QLabel("Pick an input — its CRS and the reprojection action appear here.")
        self.crs_status.setWordWrap(True)
        theme.set_accent(self.crs_status, "muted")
        form.addRow("CRS", self.crs_status)
        self.declare_epsg = QLineEdit()
        self.declare_epsg.setPlaceholderText("blank = auto-detect from the file")
        self.declare_epsg.setMaximumWidth(150)
        self.declare_epsg.setToolTip("EPSG code to assume for clouds that carry no CRS. Ignored "
                                     "for files that declare their own CRS. Required when a "
                                     "no-CRS cloud's coordinates look like lat/lon degrees.")
        self.declare_epsg.textChanged.connect(self._render_crs)
        form.addRow("Declare CRS (EPSG)", self.declare_epsg)
        self.field_combo = QComboBox()
        self.field_combo.setEditable(True)
        form.addRow("Label field", self.field_combo)
        self.copy_btn = QPushButton("Copy settings from…")
        self.copy_btn.setToolTip("Repopulate class names, ignored values, split config and "
                                 "feature selections from an existing dataset's "
                                 "dataset_meta.json. Fields absent from the meta are left "
                                 "untouched.")
        self.copy_btn.clicked.connect(self._copy_settings_menu)
        copy_row = QHBoxLayout()
        copy_row.addWidget(self.copy_btn)
        copy_row.addStretch()
        form.addRow("", ui.wrap(copy_row))
        # Datasets always build into <workspace>/<name> — no per-build output pick.
        return box

    # ============================================================= 2. Input features
    def _features_box(self) -> QWidget:
        """Only checked fields are baked into scenes — no implicit channels.
        intensity starts checked as a default, not a requirement."""
        box = QGroupBox("3 · Feature channels — input fields the model trains on")
        self.feat_group = box
        box.setToolTip("Per-point fields baked into every scene. ONLY what's "
                       "checked here ends up in the dataset — nothing is "
                       "implicit. intensity starts checked as a default; "
                       "uncheck it and the dataset carries none. Every scene "
                       "must have a checked field - a missing one fails the "
                       "build.")
        lay = QVBoxLayout(box)
        self.feat_hint = QLabel("Pick an input above — its per-point fields appear here.")
        theme.set_accent(self.feat_hint, "muted")
        lay.addWidget(self.feat_hint)
        self.feat_list = QListWidget()
        self.feat_list.setMaximumHeight(110)
        lay.addWidget(self.feat_list)
        # explicit mapping only — color is never auto-detected
        self.rgb_box = QGroupBox("RGB color (rare)")
        self.rgb_box.setCheckable(True)
        self.rgb_box.setChecked(False)
        self.rgb_box.setToolTip(
            "Map source columns to color channels. All three mapped = the "
            "dataset carries color; off or any 'none' = no color. Values are "
            "scaled to 8-bit automatically (16-bit and 0-1 sources included).")
        self.rgb_r = QComboBox()
        self.rgb_g = QComboBox()
        self.rgb_b = QComboBox()
        rgb_row = QHBoxLayout()
        for lbl, c in (("R", self.rgb_r), ("G", self.rgb_g), ("B", self.rgb_b)):
            c.addItem("none")
            rgb_row.addWidget(QLabel(lbl))
            rgb_row.addWidget(c, 1)
        rgb_row.addStretch()
        self.rgb_opts_w = ui.wrap(rgb_row)
        rgb_lay = QVBoxLayout(self.rgb_box)
        rgb_lay.addWidget(self.rgb_opts_w)
        self.rgb_box.toggled.connect(self.rgb_opts_w.setVisible)
        self.rgb_opts_w.setVisible(False)
        lay.addWidget(self.rgb_box)
        return box

    # ============================================================= 3. Calculated features
    def _calculated_box(self) -> QWidget:
        """Channels computed from xyz at build time — not fields of the input."""
        box = QGroupBox("4 · Calculated features — computed at build time")
        lay = QVBoxLayout(box)
        self.hag_box = QGroupBox("Compute Height-Above-Ground (HAG)"
                                 + ("" if pretrain.pdal_available()
                                    else " - grid only, PDAL not installed"))
        self.hag_box.setCheckable(True)
        self.hag_box.setChecked(False)
        self.hag_box.setToolTip("Bakes a per-point feat_hag channel into every scene. Pick "
                                "the ground source and interpolation below. Select feat_hag "
                                "in the Train page's feature list to feed it to any model.")
        # ground SOURCE — orthogonal to interpolation (any source × any filter)
        self.hag_ground_method = QComboBox()
        for _k in pretrain.GROUND_METHODS:
            self.hag_ground_method.addItem(pretrain.GROUND_LABELS[_k], _k)
        self.hag_ground_method.setToolTip(
            "Where ground comes from. Base off ground layer: your labeled ground "
            "class. CSF / SMRF: PDAL ground detection (needs PDAL). Z-min proxy: "
            "percentile-Z raster HAG, no classification (ignores interpolation).")
        self.hag_ground_method.currentIndexChanged.connect(self._on_hag_method)
        self.hag_filter = QComboBox()
        self.hag_filter.addItems(list(pretrain.HAG_METHODS))
        self.hag_filter.setToolTip("How HAG is interpolated from the ground points. "
                                   "grid: fast raster approximation, no PDAL needed. "
                                   "hag_nn / hag_delaunay: accurate PDAL filters.")
        self.hag_ground = QLineEdit()
        self.hag_ground.setMaximumWidth(90)
        self.hag_ground.setToolTip("Source value that means ground (from the Classes table, "
                                   "e.g. 2). Required for 'Base off ground layer'; those "
                                   "labels are the ground source (gaps are nearest-filled).")
        self._hag_ground_lbl = QLabel("ground class")
        hag_row = QHBoxLayout()
        hag_row.addWidget(QLabel("ground source"))
        hag_row.addWidget(self.hag_ground_method)
        hag_row.addWidget(self._hag_ground_lbl)
        hag_row.addWidget(self.hag_ground)
        hag_row.addWidget(QLabel("interpolation"))
        hag_row.addWidget(self.hag_filter)
        hag_row.addStretch()
        self.hag_opts_w = ui.wrap(hag_row)
        hag_lay = QVBoxLayout(self.hag_box)
        hag_lay.addWidget(self.hag_opts_w)
        self.hag_box.toggled.connect(self.hag_opts_w.setVisible)
        self.hag_opts_w.setVisible(False)
        self._on_hag_method()          # ground-class field only for 'Base off ground layer'
        lay.addWidget(self.hag_box)
        # stored raw as feat_geo_<name>
        geo_lbl = QLabel("Geometric features (pgeof) — max neighbors (k)"
                         + ("" if pretrain.pgeof_available()
                            else "  (pgeof not installed — the build will fail)"))
        self.geo_k = QSpinBox()
        self.geo_k.setRange(10, 500)
        self.geo_k.setSingleStep(10)
        self.geo_k.setValue(100)
        self.geo_k.setToolTip("Search ceiling for Weinmann optimal-neighborhood selection: "
                              "pgeof fetches this many neighbors, then picks the "
                              "eigenentropy-minimizing sub-neighborhood per point. Higher = "
                              "wider adaptive range, slower. The chosen size is the "
                              "'optimal_nn' channel.")
        geo_row = QHBoxLayout()
        geo_row.addWidget(geo_lbl)
        geo_row.addWidget(self.geo_k)
        geo_row.addStretch()
        self.geo_row_w = ui.wrap(geo_row)
        self.geo_list = QListWidget()
        self.geo_list.setMaximumHeight(110)
        for nm in pretrain.GEO_FEATURES:
            it = QListWidgetItem(nm)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.geo_list.addItem(it)
        lay.addWidget(self.geo_row_w)
        lay.addWidget(self.geo_list)
        return box

    def _on_hag_method(self):
        """Ground-class field only for 'Base off ground layer'; zmin (percentile-Z
        raster) needs no interpolation, so grey the filter out."""
        key = self.hag_ground_method.currentData()
        self._hag_ground_lbl.setVisible(key == "labels")
        self.hag_ground.setVisible(key == "labels")
        self.hag_filter.setEnabled(key != "zmin")

    # ============================================================= 4. Classes
    def _classes_box(self) -> QWidget:
        box = QGroupBox("2 · Classes - uncheck 'Train' to ignore a value; select rows + "
                        "Combine to merge into one class")
        cl = QVBoxLayout(box)
        btn_row = QHBoxLayout()
        self.scan_hint = QLabel("Scan labels first →")
        theme.set_accent(self.scan_hint, "muted")
        btn_row.addWidget(self.scan_hint)
        self.scan_btn = QPushButton("Scan label values")
        self.scan_btn.clicked.connect(self._scan_labels)
        btn_row.addWidget(self.scan_btn)
        self.analyze_btn = QPushButton("Analyze density")
        self.analyze_btn.clicked.connect(self._analyze)
        btn_row.addWidget(self.analyze_btn)
        self.combine_btn = QPushButton("Combine selected")
        self.combine_btn.clicked.connect(self._combine_selected)
        btn_row.addWidget(self.combine_btn)
        btn_row.addStretch()
        cl.addLayout(btn_row)
        self.class_table = QTableWidget(0, 4)
        self.class_table.setHorizontalHeaderLabels(["Train", "Source value(s)", "Points seen", "Class name"])
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.class_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.class_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.class_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.class_table.setMinimumHeight(160)
        cl.addWidget(self.class_table)
        self.analyze_label = QLabel("")
        self.analyze_label.setWordWrap(True)
        theme.set_accent(self.analyze_label, "muted")
        cl.addWidget(self.analyze_label)
        return box

    # ============================================================= 5. Split
    def _tiling_box(self) -> QWidget:
        box = QGroupBox("5 · Train / val / test split")
        self.tile_box = box
        self.split_form = form = QFormLayout(box)
        # whole-scene split only; scripts tile per model
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.05, 0.90)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.15)
        self.val_spin.valueChanged.connect(self._on_split_changed)
        form.addRow("Validation fraction", self.val_spin)
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.05, 0.90)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.15)
        self.test_spin.valueChanged.connect(self._on_split_changed)
        form.addRow("Test fraction", self.test_spin)
        self.train_label = QLabel("Train: 70%")
        form.addRow("", self.train_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Balanced (mirror class mix)", "Random"])
        form.addRow("Split mode", self.mode_combo)
        # TODO(not ready): split-seed UI hidden; seed fixed at 42 in _split_config
        self.split_provided_chk = QCheckBox("Separate train/val/test folders (use as-is)")
        self.split_provided_chk.toggled.connect(self._on_split_changed)
        form.addRow("", self.split_provided_chk)
        self.val_edit, self.val_row_w = self._dir_row(self._pick_val)
        form.addRow("Validation folder", self.val_row_w)
        self.test_edit, self.test_row_w = self._dir_row(self._pick_test)
        form.addRow("Test folder", self.test_row_w)
        # TODO(not ready): parallel-worker UI hidden; max_workers=1 forced in _conversion_plan
        self.tile_btn = QPushButton("Build dataset")
        self.tile_btn.setObjectName("primary")
        self.tile_btn.clicked.connect(self._start_tiling)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.next_btn = QPushButton()
        self.next_btn.setVisible(False)
        self.next_btn.clicked.connect(self._next_step)
        row = QHBoxLayout()
        row.addWidget(self.tile_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.next_btn)
        row.addStretch()
        form.addRow("", ui.wrap(row))
        return box

    # ============================================================= shared console
    def _status_block(self) -> QVBoxLayout:
        self.log = LogConsole()
        self.log.setMinimumHeight(140)
        self.log.setPlaceholderText("Progress and messages appear here…")
        lay = QVBoxLayout()
        lay.addWidget(self.log)
        return lay

    # ------------------------------------------------------------- widgets
    def _set_busy(self, on: bool):
        """One worker at a time; Stop lights up only while something runs."""
        for b in (self.scan_btn, self.analyze_btn, self.tile_btn):
            b.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        if on:
            # a running scan must stay stoppable, so lift the scan gate while busy
            self.tile_box.setEnabled(True)
        else:
            self._update_scan_gate()

    def _update_scan_gate(self, *_):
        """Classes + build stay greyed until the current input has been scanned."""
        if self.worker.running:
            return   # _set_busy owns the buttons while a job runs
        ready = bool(self._label_values) and self._scanned_for == self.input_edit.text().strip()
        for w in (self.class_table, self.combine_btn, self.analyze_btn):
            w.setEnabled(ready)
        self.tile_box.setEnabled(ready)
        self.scan_hint.setVisible(not ready)

    def _stop(self):
        if self.worker.running:
            self._append("Stopping after the current step…")
            self.worker.cancel()
            self.stop_btn.setEnabled(False)

    def _on_stopped(self):
        self._done_cb = None
        self._set_busy(False)
        self._append("⏹ Stopped.")
        self._end_run("stopped")

    # one begin_run per long op, closed exactly once via the _run_open latch
    def _begin_run(self, title: str):
        self.log.begin_run(title)
        self._run_open = True

    def _end_run(self, summary: str):
        if self._run_open:
            self._run_open = False
            self.log.end_run(summary)

    def _dispatch_done(self, result):
        cb, self._done_cb = self._done_cb, None
        if cb:
            cb(result)

    def _dir_row(self, slot):
        edit = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(edit)
        btn = QPushButton("Browse…")
        btn.clicked.connect(slot)
        row.addWidget(btn)
        return edit, ui.wrap(row)

    def _on_split_changed(self):
        provided = self.split_provided_chk.isChecked()
        self.split_form.setRowVisible(self.val_row_w, provided)
        self.split_form.setRowVisible(self.test_row_w, provided)
        if self.val_spin.value() + self.test_spin.value() > 0.90:
            if self.sender() is self.val_spin:
                spin, val = self.test_spin, 0.90 - self.val_spin.value()
            else:
                spin, val = self.val_spin, 0.90 - self.test_spin.value()
            spin.blockSignals(True)
            spin.setValue(round(val, 2))
            spin.blockSignals(False)
        train = max(0.0, 1.0 - self.val_spin.value() - self.test_spin.value())
        self.train_label.setText(f"Train: {train:.0%}")

    # ------------------------------------------------------------- pickers
    def _pick_input_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Input data folder")
        if d:
            self._set_input(d)

    def _pick_input_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Input point-cloud file")
        if f:
            self._set_input(f)

    def _set_input(self, path: str):
        self.input_edit.setText(path)
        self._populate_fields(path)
        if not self.name_edit.text():
            self.name_edit.setText(Path(path).stem)

    def _pick_val(self):
        d = QFileDialog.getExistingDirectory(self, "Validation data folder")
        if d:
            self.val_edit.setText(d)

    def _pick_test(self):
        d = QFileDialog.getExistingDirectory(self, "Test data folder")
        if d:
            self.test_edit.setText(d)

    def _populate_fields(self, path: str):
        self.field_combo.clear()
        self._crs_probe = None
        files = dataset.expand_inputs(path)
        if not files:
            self._append(f"No supported point clouds in {path}")
            self._render_crs()
            return
        try:
            cloud = read_points(files[0])   # one read feeds both fields and CRS
        except Exception as e:  # noqa: BLE001
            self._append(f"Could not probe {files[0].name}: {e}")
            self._render_crs()
            return
        fields = sorted(cloud.fields.keys())
        self.field_combo.addItems(fields)
        for preferred in ("classification", "Classification", "scalar_label", "label", "class"):
            i = self.field_combo.findText(preferred)
            if i >= 0:
                self.field_combo.setCurrentIndex(i)
                break
        # intensity starts checked — a default, not a requirement; nothing else is
        self.feat_list.clear()
        for f in fields:
            if f == self.field_combo.currentText():
                continue
            if f.lower() in ("x", "y", "z"):   # geometry, never a feature channel
                continue
            it = QListWidgetItem(f)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if dataset.canonical_channel(f) == "intensity"
                             else Qt.Unchecked)
            self.feat_list.addItem(it)
        self.feat_hint.setVisible(self.feat_list.count() == 0)
        for combo in (self.rgb_r, self.rgb_g, self.rgb_b):
            combo.clear()
            combo.addItem("none")
            combo.addItems(fields)
        self._crs_probe = crs_probe(cloud)
        self._crs_probe_name = (files[0].name if len(files) == 1
                                else f"{files[0].name} (+{len(files) - 1} more)")
        self._render_crs()

    def _render_crs(self, *_):
        """Show the probed input's detected CRS + the auto action (or the D1 block)."""
        if not self._crs_probe:
            self.crs_status.setText("Pick an input — its CRS and the reprojection "
                                    "action appear here.")
            return
        declared = parse_epsg(self.declare_epsg.text())
        detected, action, block = crs_story(
            *self._crs_probe, declared if type(declared) is int else None)
        if block:
            self.crs_status.setText(f"⚠ {self._crs_probe_name}: {detected}. "
                                    f"Blocks Build — {block}.")
        else:
            self.crs_status.setText(f"{self._crs_probe_name}: detected {detected} · {action}.")

    # ------------------------------------------------------------- config
    def _spec(self) -> LabelSpec:
        return LabelSpec(kind="field", field=self.field_combo.currentText().strip())

    def _split_config(self) -> SplitConfig:
        mode = "balanced" if self.mode_combo.currentIndex() == 0 else "random"
        return SplitConfig(
            val_frac=float(self.val_spin.value()),
            test_frac=float(self.test_spin.value()),
            mode=mode, seed=42,   # TODO(not ready): split-seed UI hidden; fixed at 42
            strategy="provided" if self.split_provided_chk.isChecked() else "auto")

    # ------------------------------------------------------------- copy settings
    def _copy_settings_menu(self):
        names = sorted(appstate.known_datasets())
        if not names:
            self._append("No saved datasets to copy settings from.")
            return
        menu = QMenu(self)
        for n in names:
            menu.addAction(n, lambda n=n: self._copy_settings_from(n))
        menu.exec(self.copy_btn.mapToGlobal(self.copy_btn.rect().bottomLeft()))

    def _copy_settings_from(self, name: str):
        """Repopulate classes, split and feature selections from a dataset's
        meta; fields missing from the meta are skipped silently."""
        meta_path = appstate.known_datasets().get(name, {}).get("meta_path", "")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._append(f"✗ Couldn't read dataset_meta.json for '{name}'.")
            return
        src = meta.get("source", {})
        copied = []
        # -- classes: stashed so a later scan re-applies names instead of wiping them
        if meta.get("classes"):
            groups: dict[int, tuple[str, list[int]]] = {}
            for cl in meta["classes"]:
                # written metas carry "source_values" (list); in-memory rows singular
                vals = cl.get("source_values") or [cl.get("source_value", cl["index"])]
                groups.setdefault(int(cl["index"]), (str(cl["name"]), []))[1].extend(
                    int(v) for v in vals)
            rows = [(vals, nm, True) for nm, vals in groups.values()]
            rows += [([int(v)], f"class_{v}", False) for v in src.get("ignore_values", [])]
            self.class_table.setRowCount(len(rows))
            for r, (vals, nm, train) in enumerate(rows):
                chk = QCheckBox()
                chk.setChecked(train)
                cell = QWidget()
                lay = QHBoxLayout(cell)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.setAlignment(Qt.AlignCenter)
                lay.addWidget(chk)
                self.class_table.setCellWidget(r, 0, cell)
                cnt = sum(self._label_values.get(v, 0) for v in vals)
                for col, text in ((1, ",".join(str(v) for v in vals)),
                                  (2, f"{cnt:,}" if cnt else "—")):
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.class_table.setItem(r, col, item)
                self.class_table.setItem(r, 3, QTableWidgetItem(nm))
            self._copied_classes = {v: (nm, train) for vals, nm, train in rows for v in vals}
            copied.append(f"{len(groups)} classes (+{len(rows) - len(groups)} ignored)")
        # -- split config
        sp = meta.get("split", {})
        req = sp.get("requested", {})
        if "val" in req:
            self.val_spin.setValue(float(req["val"]))
        if "test" in req:
            self.test_spin.setValue(float(req["test"]))
        if sp.get("mode") in ("balanced", "random"):
            self.mode_combo.setCurrentIndex(0 if sp["mode"] == "balanced" else 1)
        if req or sp.get("mode"):
            copied.append("split")
        # -- features: canonical intensity/return_number ride the has_* flags
        fields, geo, geo_k = set(), set(), None
        for fc in src.get("feature_channels") or []:
            sf = fc.get("source_field", "")
            if sf.startswith("@geo:"):
                geo.add(sf[5:])
                geo_k = fc.get("k", geo_k)
            elif sf and not sf.startswith("@"):
                fields.add(sf)
        canon = {c for c, k in (("intensity", "has_intensity"),
                                ("return_number", "has_return_number"))
                 if meta.get(k)}
        if fields or geo or canon:
            for i in range(self.feat_list.count()):
                it = self.feat_list.item(i)
                on = (it.text() in fields
                      or dataset.canonical_channel(it.text()) in canon)
                it.setCheckState(Qt.Checked if on else Qt.Unchecked)
            for i in range(self.geo_list.count()):
                it = self.geo_list.item(i)
                it.setCheckState(Qt.Checked if it.text() in geo else Qt.Unchecked)
            if geo_k is not None:
                self.geo_k.setValue(int(geo_k))
            copied.append("features")
        # -- HAG ("source_dimension" means HAG came pre-baked, nothing to compute)
        hag_src = src.get("hag_source")
        if hag_src is not None and hag_src != "source_dimension":
            self.hag_box.setChecked(bool(hag_src))
            if hag_src:
                gm = src.get("hag_ground_method")
                if not gm:          # legacy: parse the ground source from hag_source
                    gm = (hag_src.split("+", 1)[1] if "+" in hag_src
                          else (hag_src if hag_src == "zmin" else ""))
                k = self.hag_ground_method.findData(gm)
                if k >= 0:
                    self.hag_ground_method.setCurrentIndex(k)
                if hag_src != "zmin":
                    i = self.hag_filter.findText(hag_src.split("+")[0])
                    if i >= 0:
                        self.hag_filter.setCurrentIndex(i)
                gv = src.get("hag_ground_value")
                self.hag_ground.setText("" if gv is None else str(gv))
                self._on_hag_method()
                copied.append("HAG")
        if not copied:
            self._append(f"'{name}' has no copyable settings in its meta.")
            return
        self._append(f"Copied from '{name}': {', '.join(copied)}. Scan labels on the "
                     f"new input to verify the values before Build.")

    # ------------------------------------------------------------- scan / analyze
    def _scan_labels(self):
        in_path = self.input_edit.text().strip()
        if not os.path.exists(in_path):
            self._append("Pick an input file or folder first.")
            return
        spec = self._spec()
        self._scan_in_path = in_path
        self._append("Scanning label values…")
        self._set_busy(True)
        def job(progress):
            files = dataset.expand_inputs(in_path)
            progress(f"  sampling {min(len(files), 8)} of {len(files)} file(s)")
            return dataset.scan_label_values(files, spec)

        self._done_cb = self._on_scanned
        self.worker.start(job)

    def _on_scanned(self, counts):
        self._set_busy(False)
        self._label_values = counts
        # default-ignore 0 only for ASPRS-style classification fields
        ignore_zero = "class" in self._spec().field.lower()
        self.class_table.setRowCount(len(counts))
        for r, (val, cnt) in enumerate(counts.items()):
            chk = QCheckBox()
            chk.setChecked(not (ignore_zero and val == 0))
            cell = QWidget()
            lay = QHBoxLayout(cell)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setAlignment(Qt.AlignCenter)
            lay.addWidget(chk)
            self.class_table.setCellWidget(r, 0, cell)
            for col, text in ((1, str(val)), (2, f"{cnt:,}")):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(r, col, item)
            self.class_table.setItem(r, 3, QTableWidgetItem(f"class_{val}"))
        # re-apply "Copy settings from…" names to the fresh scan
        if self._copied_classes:
            for r in range(self.class_table.rowCount()):
                for v in _parse_values(self.class_table.item(r, 1).text()):
                    if v in self._copied_classes:
                        nm, train = self._copied_classes[v]
                        self.class_table.item(r, 3).setText(nm)
                        self.class_table.cellWidget(r, 0).findChild(QCheckBox).setChecked(train)
                        break
        self._scanned_for = self._scan_in_path   # unlocks classes + build
        self._update_scan_gate()
        self._append(f"Found {len(counts)} label values. Name them, uncheck any "
                     f"'unknown', then Build.")

    def _analyze(self):
        in_path = self.input_edit.text().strip()
        if not os.path.exists(in_path):
            self._append("Pick an input file or folder first.")
            return
        if not self._label_values:
            self._append("Run 'Scan label values' first.")
            return
        self._set_busy(True)
        self._append("Analyzing density…")

        def job(progress):
            files = dataset.expand_inputs(in_path)
            progress(f"  scanning up to {analysis.MAX_FILES_PER_SPLIT} of {len(files)} file(s)")
            return analysis.scan_folder(files)

        self._done_cb = self._on_analyzed
        self.worker.start(job)

    def _on_analyzed(self, stats):
        self._set_busy(False)
        recs = analysis.recommend(stats)
        chunk = next(iter(recs.values())).get("chunk_xy", 0.0)
        self.analyze_label.setText(
            f"Density: {stats['mean_pts_per_m2']:.2f} pts/m²  ·  "
            f"spacing {stats['mean_spacing_m']:.2f} m  ·  "
            f"largest scene {stats['max_scene_points']:,} pts  ·  "
            f"suggested tile {chunk:.0f} m (set per model on the Train page).")

    # ------------------------------------------------------------- convert/upload
    def _combine_selected(self):
        """Collapse selected rows into one whose Source value lists every merged
        value ("5,6") and Points seen is their total, under a shared name."""
        rows = sorted({i.row() for i in self.class_table.selectedItems()})
        if len(rows) < 2:
            self._append("Combine: select 2+ rows first (Ctrl/Shift for many).")
            return
        first = self.class_table.item(rows[0], 3)
        base = first.text().strip() if first else ""
        name, ok = QInputDialog.getText(self, "Combine classes",
                                        "Name for the combined class:", text=base)
        name = name.strip()
        if not ok or not name:
            return
        vals: list[int] = []
        for r in rows:
            vals += _parse_values(self.class_table.item(r, 1).text())
        vals = sorted(dict.fromkeys(vals))
        total = sum(self._label_values.get(v, 0) for v in vals)
        keep = rows[0]
        self.class_table.item(keep, 1).setText(",".join(str(v) for v in vals))
        self.class_table.item(keep, 2).setText(f"{total:,}")
        self.class_table.setItem(keep, 3, QTableWidgetItem(name))
        self.class_table.cellWidget(keep, 0).findChild(QCheckBox).setChecked(True)
        for r in reversed(rows[1:]):
            self.class_table.removeRow(r)
        self._append(f"Combined source values [{', '.join(map(str, vals))}] into class '{name}'.")

    def _classes_from_table(self):
        """One class per row. A row's Source value may list several values ("5,6");
        each maps to the same class index/name."""
        name_to_index: dict[str, int] = {}
        classes, ignored = [], []
        for r in range(self.class_table.rowCount()):
            vals = _parse_values(self.class_table.item(r, 1).text())
            chk = self.class_table.cellWidget(r, 0).findChild(QCheckBox)
            if not chk.isChecked():
                ignored.extend(vals)
                continue
            name = self.class_table.item(r, 3).text().strip() or f"class_{vals[0]}"
            if name not in name_to_index:
                name_to_index[name] = len(name_to_index)
            for v in vals:
                classes.append({"index": name_to_index[name], "source_value": v, "name": name})
        return classes, ignored

    def _conversion_plan(self):
        name = self.name_edit.text().strip()
        in_path = self.input_edit.text().strip()
        if not name or not os.path.exists(in_path):
            self._append("Need a name and an input file or folder.")
            return None
        declared = parse_epsg(self.declare_epsg.text())
        if declared is False:
            self._append("Declare CRS: enter an EPSG integer (e.g. 6539), or leave blank "
                         "to auto-detect from the file.")
            return None
        if self._crs_probe:
            _, _, block = crs_story(*self._crs_probe,
                                    declared if type(declared) is int else None)
            if block:
                self._append(f"✗ '{self._crs_probe_name}' carries no CRS and its "
                             f"coordinates look like lat/lon degrees — {block} to "
                             f"reproject it, then Build.")
                return None
        split = self._split_config()
        val_inputs = test_inputs = None
        if split.strategy == "provided":
            val_dir = self.val_edit.text().strip()
            test_dir = self.test_edit.text().strip()
            if not os.path.isdir(val_dir) or not os.path.isdir(test_dir):
                self._append("Pick both the val and test folders.")
                return None
            val_inputs, test_inputs = [val_dir], [test_dir]
        if self.class_table.rowCount() == 0:
            self._append("Run 'Scan label values' and name classes first.")
            return None
        classes, ignored = self._classes_from_table()
        if not classes:
            self._append("All values unchecked - nothing to train on.")
            return None
        method = self.hag_ground_method.currentData()
        gtxt = self.hag_ground.text().strip()
        try:
            gv = int(gtxt) if gtxt else None
        except ValueError:
            gv = None
        if gtxt and gv is None:
            self._append(f"Ground class '{gtxt}' isn't an integer - clear it or enter a "
                         f"Source value from the Classes table.")
            return None
        if self.hag_box.isChecked() and method == "labels" and gv is None:
            self._append("HAG 'Base off ground layer' needs a ground class - set a Source "
                         "value from the Classes table, or pick CSF / SMRF / Z-min proxy.")
            return None
        label_field = self.field_combo.currentText().strip()
        feats = [self.feat_list.item(i).text() for i in range(self.feat_list.count())
                 if self.feat_list.item(i).checkState() == Qt.Checked
                 and self.feat_list.item(i).text() != label_field]
        geo = [self.geo_list.item(i).text() for i in range(self.geo_list.count())
               if self.geo_list.item(i).checkState() == Qt.Checked]
        rgb_sel = ([c.currentText() for c in (self.rgb_r, self.rgb_g, self.rgb_b)]
                   if self.rgb_box.isChecked() else [])
        mapped = [s for s in rgb_sel if s and s != "none"]
        if self.rgb_box.isChecked() and len(mapped) < 3:
            self._append("RGB mapping needs all three channels set (or untick "
                         "the RGB box for no color).")
            return None
        return {
            "name": name, "in_path": in_path, "split": split,
            "val_inputs": val_inputs, "test_inputs": test_inputs,
            "classes": classes, "ignored": ignored, "spec": self._spec(),
            "out_root": appstate.workspace_dir(),
            "compute_hag": self.hag_box.isChecked(),
            "ground_value": gv,
            "hag_filter": self.hag_filter.currentText(),
            "ground_method": method,
            "feature_fields": feats or None,
            "geo_features": geo or None,
            "geo_k": int(self.geo_k.value()),
            "rgb_fields": rgb_sel if len(mapped) == 3 else None,
            "declared_crs_epsg": declared if type(declared) is int else None,
            "max_workers": 1,   # TODO(not ready): parallel UI hidden; force single-process
        }

    def _start_tiling(self):
        plan = self._conversion_plan()
        if plan is None:
            return
        name, classes, ignored = plan["name"], plan["classes"], plan["ignored"]
        split, out_root = plan["split"], plan["out_root"]
        self._begin_run(f"build '{name}'")
        self._set_busy(True)
        hag = "  + HAG" if plan["compute_hag"] else ""
        if plan["geo_features"]:
            hag += f"  + geo({len(plan['geo_features'])} @ k≤{plan['geo_k']})"
        self._append(f"Building '{name}'{hag} ({len(classes)} classes, "
                     f"val={split.val_frac:.0%} test={split.test_frac:.0%} {split.mode} "
                     f"seed={split.seed}, ignored values: {ignored}) -> {out_root}…")

        # only passed when the user declared one — keeps the meter-UTM path's call
        # byte-identical until dataset.convert_dataset gains the param
        crs_kw = ({"declared_crs_epsg": plan["declared_crs_epsg"]}
                  if plan["declared_crs_epsg"] is not None else {})

        def job(progress):
            return dataset.convert_dataset(
                name, [plan["in_path"]], plan["spec"], classes, ignored,
                out_root, val_inputs=plan["val_inputs"],
                test_inputs=plan["test_inputs"], split=split,
                intensity_norm="p95", compute_hag=plan["compute_hag"],
                ground_value=plan["ground_value"],
                hag_filter=plan["hag_filter"],
                ground_method=plan["ground_method"],
                feature_fields=plan["feature_fields"],
                geo_features=plan["geo_features"],
                geo_k=plan["geo_k"],
                rgb_fields=plan["rgb_fields"],
                max_workers=plan["max_workers"], progress=progress, **crs_kw)

        self._done_cb = self._on_converted
        self.worker.start(job)

    def _on_converted(self, staged: Path):
        self._staged_dir = staged
        self._set_busy(False)
        appstate.remember_dataset(staged.name, {
            "staged_dir": str(staged),
            "meta_path": str(staged / "dataset_meta.json"),
            "uploaded": False,
        })
        self._reload_known()
        hag = " (with HAG)" if self.hag_box.isChecked() else ""
        if appstate.get_exec_mode() == "local":
            self._append(f"✓ Built{hag} -> {staged}. Pick '{staged.name}' on the Train "
                         f"page (bind-mounted at /datasets/{staged.name}).")
        else:
            self._append(f"✓ Built{hag} -> {staged}. Select it under Saved Datasets to upload.")
        self._end_run(f"built '{staged.name}'")
        self._refresh_next_btn()

    def _refresh_next_btn(self):
        """Context-aware next step after a successful build this session: modal
        mode uploads the fresh build, local mode jumps to the Train page."""
        if self._staged_dir is None:
            self.next_btn.setVisible(False)
            return
        if appstate.get_exec_mode() == "local":
            self.next_btn.setText(f"Next: Train with {self._staged_dir.name} →")
        else:
            self.next_btn.setText("Upload to Modal")
        self.next_btn.setVisible(True)

    def _next_step(self):
        if self._staged_dir is None:
            return
        if appstate.get_exec_mode() == "local":
            # plain page switch; the Train page reloads its dataset list on entry
            ui.navigate("Train")
        else:
            self._start_upload(self._staged_dir)

    @staticmethod
    def _register_dataset(staged: Path) -> bool:
        """Validate + remember a converted dataset folder. It must hold a readable
        dataset_meta.json; False (nothing recorded) when it doesn't."""
        try:
            with open(staged / "dataset_meta.json", "r", encoding="utf-8") as f:
                json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        appstate.remember_dataset(staged.name, {
            "staged_dir": str(staged),
            "meta_path": str(staged / "dataset_meta.json"),
            "uploaded": False,
        })
        return True

    def _add_existing(self):
        """Register an already-converted dataset folder under Saved Datasets."""
        picked = QFileDialog.getExistingDirectory(
            self, "Converted dataset folder (must contain dataset_meta.json)",
            str(appstate.workspace_dir()))
        if not picked:
            return
        staged = Path(picked)
        if not self._register_dataset(staged):
            QMessageBox.warning(
                self, "Not a dataset",
                f"{staged} has no readable dataset_meta.json — not a converted "
                f"dataset. Build it with 'Build dataset' first.")
            return
        self._reload_known()
        self._append(f"✓ Added '{staged.name}' from {staged}.")

    def _upload_saved(self):
        """Upload a saved dataset from its remembered staged_dir, no re-conversion.
        If that folder moved or was deleted, ask where it is now (must hold
        dataset_meta.json)."""
        name = self._selected_known_name()
        if name is None:
            self._append("Select a saved dataset to upload.")
            return
        info = appstate.known_datasets().get(name, {})
        staged = Path(info.get("staged_dir", ""))
        if not (str(staged) and staged.exists() and (staged / "dataset_meta.json").exists()):
            picked = QFileDialog.getExistingDirectory(
                self, f"Locate the converted '{name}' folder (must contain dataset_meta.json)",
                str(staged.parent) if str(staged) else "")
            if not picked:
                return
            staged = Path(picked)
            if not self._register_dataset(staged):
                self._append(f"✗ {staged} has no dataset_meta.json - not a dataset.")
                return
        self._start_upload(staged)

    def _delete_saved(self):
        """Forget the selected dataset and delete its DATA on disk, keeping any
        runs/ + infer/ for record keeping, after confirm."""
        name = self._selected_known_name()
        if name is None:
            self._append("Select a saved dataset to delete.")
            return
        resp = QMessageBox.question(
            self, "Delete dataset",
            f"Delete '{name}'? Removes it from the list and deletes its data "
            f"(train/val/test + cache). Any training runs (runs/) and inference jobs "
            f"(infer/) are KEPT for your records. Can't be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if resp != QMessageBox.Yes:
            return
        staged, err = appstate.delete_dataset(name)
        self._reload_known()
        self.stats_label.setText("")
        if err:
            self._append(f"Removed '{name}' from the list, but some data at {staged} "
                         f"couldn't be deleted:\n  {err}\n  Close anything using them "
                         f"(point-cloud viewer, training run) and delete manually.")
        else:
            self._append(f"Deleted '{name}' data - removed from the list. Kept any "
                         f"runs/ + infer/ under {staged or 'nothing on disk'} for records.")

    def _start_upload(self, staged: Path):
        # one shared datasets volume (TT_DATASET_VOLUME), each dataset under /<name>
        self._uploading = staged
        name = staged.name
        self.upload_saved_btn.setEnabled(False)
        self._begin_run(f"upload '{name}'")
        self._append(f"\nUploading -> {modal_cli.DATASETS_VOLUME}:/{name} …\n"
                     f"(first ensuring the volume exists — a \"Volume "
                     f"'{modal_cli.DATASETS_VOLUME}' already exists\" error here is "
                     f"EXPECTED and harmless; the upload continues right after)")
        prog, args = modal_cli.volume_put(modal_cli.DATASETS_VOLUME, str(staged), f"/{name}")
        self.uploader.start(prog, args, cwd=self.repo_root,
                            pre=modal_cli.volume_create(modal_cli.DATASETS_VOLUME))

    def _on_upload_failed(self, err: str):
        self.upload_saved_btn.setEnabled(True)
        self._append(f"✗ Upload failed to run: {err}. Modal CLI on PATH and "
                     f"authenticated? (modal token new)")
        self._end_run("upload failed to start")

    def _on_upload_done(self, code: int):
        self.upload_saved_btn.setEnabled(True)
        staged = self._uploading
        if code != 0:
            self._append(f"\n✗ Upload failed (exit {code}). Modal CLI installed and "
                         f"authenticated? (modal token new)")
            self._end_run(f"upload failed (exit {code})")
            return
        name = staged.name
        appstate.remember_dataset(name, {
            "staged_dir": str(staged),
            "meta_path": str(staged / "dataset_meta.json"),
            "uploaded": True,
            "volume": modal_cli.DATASETS_VOLUME,
        })
        self._reload_known()
        self._append(f"\n✓ Uploaded '{name}' -> {modal_cli.DATASETS_VOLUME}:/{name}. "
                     "Go to the Train page.")
        self._end_run(f"uploaded '{name}'")

    def _on_worker_error(self, tb: str):
        self._done_cb = None
        self._set_busy(False)
        self._append("✗ Error - see the dialog.")
        self._end_run("failed")
        QMessageBox.critical(self, "Dataset error", tb)

    # ------------------------------------------------------------- known list
    def _reload_known(self):
        """Rows are '<name>   <badge>'; the bare name rides in UserRole."""
        c = theme.colors(appstate.get("ui_theme", "system"))
        self.known_list.clear()
        for name, info in sorted(appstate.known_datasets().items()):
            staged = info.get("staged_dir", "")
            if not (staged and os.path.isdir(staged)):
                suffix, color = "⚠ missing", c["warn"]
            elif info.get("uploaded"):
                suffix, color = "✓ uploaded", c["ok"]
            else:
                suffix, color = "● staged", None   # default text color
            it = QListWidgetItem(f"{name}   {suffix}")
            it.setData(Qt.UserRole, name)
            if color:
                it.setForeground(QColor(color))
            self.known_list.addItem(it)

    def _selected_known_name(self) -> str | None:
        items = self.known_list.selectedItems()
        if not items:
            return None
        return items[0].data(Qt.UserRole) or items[0].text()

    def _show_known(self):
        name = self._selected_known_name()
        if name is None:
            return
        info = appstate.known_datasets().get(name, {})
        staged = info.get("staged_dir", "")
        on_disk = bool(staged) and os.path.isdir(staged)
        if appstate.get_exec_mode() == "local":
            status = ("staged ✓ - ready to train" if on_disk
                      else "local copy missing - re-convert on this machine")
        else:
            status = ("uploaded ✓ (re-upload to refresh)" if info.get("uploaded")
                      else "not uploaded - click “Upload selected to Modal”")
            if not on_disk:
                status += "  ·  local copy missing (upload will ask where it is)"
        meta_path = info.get("meta_path", "")
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            s = meta.get("stats", {})
            spl = meta.get("splits", {})
            n_tr = len(spl.get("train", {}).get("scenes", []))
            n_va = len(spl.get("val", {}).get("scenes", []))
            n_te = len(spl.get("test", {}).get("scenes", []))
            hag = "  ·  HAG ✓" if meta.get("has_hag") else ""
            fc = meta.get("source", {}).get("feature_channels") or []
            extra = ("\nextra channels: " + ", ".join(c["name"] for c in fc)) if fc else ""
            self.stats_label.setText(
                f"{meta['name']}: {meta['num_classes']} classes "
                f"({', '.join(meta['class_names'])})  ·  "
                f"{s.get('mean_pts_per_m2', 0):.2f} pts/m²  ·  "
                f"train {n_tr}, val {n_va}, test {n_te} scenes{hag}{extra}\n{status}")
        else:
            self.stats_label.setText(f"{name}\n{status}")

    # ------------------------------------------------------------- helpers
    def _append(self, text: str, newline: bool = True):
        ui.append_log(self.log, text, newline)


def _parse_values(text: str) -> list[int]:
    """Source-value cell -> ints. Handles one value ("5") or a list ("5,6" / "5 6")."""
    return [int(t) for t in text.replace(",", " ").split() if t]


# --- CRS surface: labels readers' reprojection outcome; no reprojection here.
# infer_page imports crs_probe/crs_story/parse_epsg so the two pages share one impl.

def parse_epsg(text):
    """Declare-CRS field -> int, None (blank = auto-detect), or False (not an integer)."""
    t = (text or "").strip().upper()
    if t.startswith("EPSG:"):
        t = t[len("EPSG:"):].strip()
    if not t:
        return None
    try:
        return int(t)
    except ValueError:
        return False


def _looks_like_degrees(xyz) -> bool:
    """No-CRS coords sitting in lon/lat bounds with a small span — the D1 trigger.
    The span guard keeps small projected-metre local clouds from false-blocking."""
    import numpy as np
    if len(xyz) == 0:
        return False
    x, y = xyz[:, 0], xyz[:, 1]
    if not (x.min() >= -180 and x.max() <= 180 and y.min() >= -90 and y.max() <= 90):
        return False
    return max(float(x.max() - x.min()), float(y.max() - y.min())) <= 10.0


def _crs_name(wkt) -> str:
    try:
        from pyproj import CRS
        return CRS.from_wkt(wkt).name
    except Exception:
        return "custom CRS"


def crs_probe(cloud):
    """(source_wkt, proc_wkt, looks_degrees) pulled from an already-read Cloud."""
    return cloud.source_crs_wkt, cloud.crs_wkt, _looks_like_degrees(cloud.xyz)


def crs_story(source_wkt, proc_wkt, looks_degrees, declared_epsg):
    """(detected, action, block) for the CRS surface + D1 preflight. block is a
    remedy string (degree-looking no-CRS input with no declared EPSG) or None."""
    if source_wkt:                     # ingest reprojected it
        return _crs_name(source_wkt), f"reproject → {_crs_name(proc_wkt)}", None
    if proc_wkt:                       # identity fast-path: already metre-projected
        return _crs_name(proc_wkt), "keep as-is (already metre-projected)", None
    if declared_epsg is not None:      # no CRS, but the user declared one
        return "none in file", f"declared EPSG:{declared_epsg} → reproject", None
    if looks_degrees:
        return ("none — coordinates look like lat/lon degrees", None,
                "declare its EPSG in the 'Declare CRS (EPSG)' box")
    return "none", "keep as-is (assumed projected metres)", None
