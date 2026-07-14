"""Datasets page workflow, top to bottom:

  1. New dataset   point at a file/folder, name it, pick the label field
  2. Classes       scan label values, name them, mark ignored, check density
  3. Split         train/val/test split (whole scenes; scripts tile per model) +
                   optional per-scene Height-Above-Ground channel (grid raster or PDAL)

Labels come from a field in the cloud. Intensity is p95-normalized (i/p95
clipped to 0..2); the single norm used across build + train + inference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import analysis, appstate, dataset, modal_cli, pretrain, theme, ui
from ..dataset import LabelSpec, SplitConfig
from ..jobs import FuncWorker, JobRunner
from ..logconsole import LogConsole
from ..readers import list_label_fields


class DatasetsPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.worker = FuncWorker(self)
        self.uploader = JobRunner(self)
        self._staged_dir: Path | None = None
        self._uploading: Path | None = None   # dir being uploaded
        self._label_values: dict[int, int] = {}
        self._done_cb = None
        self._scanned_for: str | None = None   # input path the last label scan covered
        self._copied_classes: dict[int, tuple[str, bool]] = {}  # value -> (name, train) from "Copy settings"
        self._run_open = False                  # a begin_run header awaits its end_run

        root = QVBoxLayout(self)
        title = QLabel("Datasets")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.sub.setObjectName("pageSub")
        root.addWidget(self.sub)

        # Page scrolls, so each section keeps its natural height.
        root.addWidget(self._new_dataset_box())   # 1
        root.addWidget(self._classes_box())        # 2
        root.addWidget(self._tiling_box())         # 3  (incl. optional HAG)
        root.addLayout(self._status_block())       # shared console

        # ---- saved datasets: bottom layer ----
        root.addWidget(QLabel("Saved Datasets"))
        sd_row = QHBoxLayout()
        sd_col = QVBoxLayout()
        self.known_list = QListWidget()
        self.known_list.setMaximumHeight(120)
        self.known_list.setMaximumWidth(360)
        self.known_list.itemSelectionChanged.connect(self._show_known)
        sd_col.addWidget(self.known_list)
        # Register an already-converted folder (moved from another box, restored
        # backup, …) without re-converting; must hold a valid dataset_meta.json.
        self.add_existing_btn = QPushButton("Add existing dataset…")
        self.add_existing_btn.clicked.connect(self._add_existing)
        sd_col.addWidget(self.add_existing_btn)
        # Re-upload a converted dataset without re-converting; survives restart
        # (staged_dir is remembered in state.json).
        self.upload_saved_btn = QPushButton("Upload selected to Modal")
        self.upload_saved_btn.clicked.connect(self._upload_saved)
        sd_col.addWidget(self.upload_saved_btn)
        # Forget the dataset + delete its staged copy on disk.
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
        # Spine order Input -> Scan -> Classes -> Build: the classes/build widgets
        # stay greyed until a label scan has run for the picked input.
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
            + (" Staged on disk, ready to train in Docker."
               if local else
               " Then upload to a per-dataset Modal volume."))
        self._reload_known()
        self._refresh_next_btn()   # next-step wording follows the backend

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
        self.field_combo = QComboBox()
        self.field_combo.setEditable(True)
        form.addRow("Label field", self.field_combo)
        # Reuse a previous dataset's setup: class names/ignores, split, features.
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

    # ============================================================= 2. Classes
    def _classes_box(self) -> QWidget:
        box = QGroupBox("2 · Classes - uncheck 'Train' to ignore a value; select rows + "
                        "Combine to merge into one class")
        cl = QVBoxLayout(box)
        btn_row = QHBoxLayout()
        # Spine hint: classes + build stay greyed until the scan has run.
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

    # ============================================================= 3. Split
    def _tiling_box(self) -> QWidget:
        box = QGroupBox("3 · Train / val / test split")
        self.tile_box = box   # greyed by _update_scan_gate until labels are scanned
        self.split_form = form = QFormLayout(box)
        # Two point-count fractions (train = remainder); carve three whole-scene
        # folders once, scripts read them verbatim. No tiling here.
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
        # balanced mirrors the global class mix in every split; random fills by
        # point count alone.
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Balanced (mirror class mix)", "Random"])
        form.addRow("Split mode", self.mode_combo)
        # TODO(not ready): split-seed UI hidden until reviewed; seed is fixed at 42
        # in _split_config for now.
        # Explicit val + test folders bypass allocation (train = inputs).
        self.split_provided_chk = QCheckBox("Separate train/val/test folders (use as-is)")
        self.split_provided_chk.toggled.connect(self._on_split_changed)
        form.addRow("", self.split_provided_chk)
        self.val_edit, self.val_row_w = self._dir_row(self._pick_val)
        form.addRow("Validation folder", self.val_row_w)
        self.test_edit, self.test_row_w = self._dir_row(self._pick_test)
        form.addRow("Test folder", self.test_row_w)
        # Optional: compute HeightAboveGround per scene in the same pass. Checkable
        # group box, collapsed by default — same fold as the Train page's Loss box —
        # so the landing view is just the split spinners, not the expert knobs.
        self.hag_box = QGroupBox("Compute Height-Above-Ground (HAG)"
                                 + ("" if pretrain.pdal_available()
                                    else " - grid only, PDAL not installed"))
        self.hag_box.setCheckable(True)
        self.hag_box.setChecked(False)
        self.hag_box.setToolTip("Bakes a per-point feat_hag channel into every scene. Ground "
                                "comes from the labeled ground class when one is set, else "
                                "it's detected. Select feat_hag in the Train page's feature "
                                "list to feed it to any model.")
        # Interpolation only — the ground SOURCE is the labeled class when set,
        # else SMRF detection (any method; grid heuristic = the PDAL-less fallback).
        self.hag_filter = QComboBox()
        self.hag_filter.addItems(list(pretrain.HAG_METHODS))
        self.hag_filter.setToolTip("How HAG is interpolated from the ground points. "
                                   "grid: fast raster approximation, no PDAL needed. "
                                   "hag_nn / hag_delaunay: accurate PDAL filters.")
        # Which class is ground (raw Source value from the Classes table). When set,
        # the labels are the ONLY ground source — never mixed with detection.
        self.hag_ground = QLineEdit()
        self.hag_ground.setPlaceholderText("blank = detect (SMRF)")
        self.hag_ground.setMaximumWidth(90)
        self.hag_ground.setToolTip("Source value that means ground (from the Classes table, "
                                   "e.g. 2). When set, those labels are the only ground source "
                                   "(gaps are nearest-filled). Blank = SMRF detects ground "
                                   "instead (needs PDAL; without it the grid method's own "
                                   "heuristic is the fallback).")
        hag_row = QHBoxLayout()
        hag_row.addWidget(QLabel("method"))
        hag_row.addWidget(self.hag_filter)
        hag_row.addWidget(QLabel("ground class"))
        hag_row.addWidget(self.hag_ground)
        hag_row.addStretch()
        self.hag_opts_w = ui.wrap(hag_row)
        hag_lay = QVBoxLayout(self.hag_box)
        hag_lay.addWidget(self.hag_opts_w)
        self.hag_box.toggled.connect(self.hag_opts_w.setVisible)
        self.hag_opts_w.setVisible(False)
        form.addRow(self.hag_box)
        # Optional: bake extra per-point source fields into every scene as
        # feat_<name> (p95(|v|)-normalized, like intensity). Hidden until the
        # input carries candidate fields; collapsed (unchecked) by default.
        self.feat_group = QGroupBox("Extra feature channels")
        self.feat_group.setCheckable(True)
        self.feat_group.setChecked(False)
        self.feat_group.setToolTip("Carry extra per-point fields (e.g. eigenvalue features) "
                                   "into every scene as feat_<name>. Every scene must have "
                                   "the field - a missing one fails the build. Computed "
                                   "geometric features (jakteristics) need no source field: "
                                   "they come from xyz within the search radius and are "
                                   "stored raw as feat_geo_<name>.")
        self.feat_list = QListWidget()
        self.feat_list.setMaximumHeight(110)
        feat_lay = QVBoxLayout(self.feat_group)
        feat_lay.addWidget(self.feat_list)
        # Computed geometric channels (jakteristics, from xyz — no source field).
        geo_lbl = QLabel("Computed geometric features (jakteristics) — search radius"
                         + ("" if pretrain.jakteristics_available()
                            else "  (jakteristics not installed — the build will fail)"))
        self.geo_radius = QDoubleSpinBox()
        self.geo_radius.setRange(0.1, 50.0)
        self.geo_radius.setSingleStep(0.1)
        self.geo_radius.setValue(1.0)
        self.geo_radius.setSuffix(" m")
        self.geo_radius.setToolTip("Neighborhood radius for the local PCA. Most features "
                                   "are dimensionless ratios; eigenvalue_sum and "
                                   "omnivariance scale with radius² — keep it small.")
        geo_row = QHBoxLayout()
        geo_row.addWidget(geo_lbl)
        geo_row.addWidget(self.geo_radius)
        geo_row.addStretch()
        self.geo_row_w = ui.wrap(geo_row)
        self.geo_list = QListWidget()
        self.geo_list.setMaximumHeight(110)
        for nm in pretrain.GEO_FEATURES:
            it = QListWidgetItem(nm)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.geo_list.addItem(it)
        feat_lay.addWidget(self.geo_row_w)
        feat_lay.addWidget(self.geo_list)
        for w in (self.feat_list, self.geo_row_w, self.geo_list):
            self.feat_group.toggled.connect(w.setVisible)  # collapse w/ the checkbox
            w.setVisible(False)
        self.feat_group.setVisible(False)
        form.addRow("", self.feat_group)
        # TODO(not ready): parallel-worker UI hidden until reviewed; conversion runs
        # single-process (max_workers=1 forced in _conversion_plan).
        self.tile_btn = QPushButton("Build dataset")
        self.tile_btn.setObjectName("primary")
        self.tile_btn.clicked.connect(self._start_tiling)
        # One Stop for the shared worker: cancels whichever of scan/analyze/build
        # is running (a build stops between scenes).
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        # Context-aware next step; hidden until a build succeeds this session
        # (modal: upload the fresh build; local: jump to the Train page).
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
        # Scrolling console log; errors also pop a dialog (_on_worker_error).
        self.log = LogConsole()   # \r-aware, colored console (drop-in for the old QPlainTextEdit)
        self.log.setMinimumHeight(140)
        self.log.setPlaceholderText("Progress and messages appear here…")
        lay = QVBoxLayout()
        lay.addWidget(self.log)
        return lay

    # ------------------------------------------------------------- widgets
    def _set_busy(self, on: bool):
        """One worker at a time: disable every action that would start a second
        job while one runs (this is the guard that stops scan+analyze colliding),
        and light up Stop only while there's something to stop."""
        for b in (self.scan_btn, self.analyze_btn, self.tile_btn):
            b.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        if on:
            # Stop lives in the (possibly scan-gated) split box: a running scan
            # must stay stoppable, so lift the gate while busy.
            self.tile_box.setEnabled(True)
        else:
            self._update_scan_gate()

    def _update_scan_gate(self, *_):
        """Spine order: classes + build stay greyed until 'Scan label values' has
        run for the CURRENTLY picked input (changing the input re-locks them)."""
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
            self.stop_btn.setEnabled(False)   # buttons reset when the job unwinds

    def _on_stopped(self):
        self._done_cb = None
        self._set_busy(False)
        self._append("⏹ Stopped.")
        self._end_run("stopped")

    # LogConsole run headers: one begin_run per long operation (build, upload),
    # closed exactly once on success/failure/stop via the _run_open latch.
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
        # Provided mode reveals the val + test folder rows; else fractions drive
        # allocation. Keep val% + test% <= 0.90 (train keeps >= 10%).
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
        files = dataset.expand_inputs(path)
        if not files:
            self._append(f"No supported point clouds in {path}")
            return
        try:
            fields = list_label_fields(files[0])
        except Exception as e:  # noqa: BLE001
            self._append(f"Could not probe {files[0].name}: {e}")
            return
        self.field_combo.addItems(fields)
        for preferred in ("classification", "Classification", "scalar_label", "label", "class"):
            i = self.field_combo.findText(preferred)
            if i >= 0:
                self.field_combo.setCurrentIndex(i)
                break
        # Extra-channel candidates: every field except the label field (a late
        # label-field change is re-filtered at build time in _conversion_plan).
        self.feat_list.clear()
        for f in fields:
            if f == self.field_combo.currentText():
                continue
            if f.lower() in ("x", "y", "z"):   # geometry, never an extra channel
                continue
            it = QListWidgetItem(f)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.feat_list.addItem(it)
        # Always offerable once an input is probed: computed geometric channels
        # need no source field, so the group no longer hides on empty feat_list.
        self.feat_group.setVisible(True)

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
        """Repopulate class names/ignores, split config and feature selections from
        an existing dataset's dataset_meta.json. Fields missing from the meta are
        skipped silently; the scan gate still applies to the new input."""
        meta_path = appstate.known_datasets().get(name, {}).get("meta_path", "")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._append(f"✗ Couldn't read dataset_meta.json for '{name}'.")
            return
        src = meta.get("source", {})
        copied = []
        # -- classes: one row per class index (shared names merge to "5,6"), plus
        #    the ignored values as unchecked rows. Also stashed so a later scan of
        #    the new input re-applies the names instead of wiping them.
        if meta.get("classes"):
            groups: dict[int, tuple[str, list[int]]] = {}
            for cl in meta["classes"]:
                groups.setdefault(int(cl["index"]), (str(cl["name"]), []))[1].append(
                    int(cl["source_value"]))
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
        # -- feature selections (raw source fields + computed geometric channels)
        fields, geo, radius = set(), set(), None
        for fc in src.get("feature_channels") or []:
            sf = fc.get("source_field", "")
            if sf.startswith("@geo:"):
                geo.add(sf[5:])
                radius = fc.get("radius", radius)
            elif sf and not sf.startswith("@"):
                fields.add(sf)
        if fields or geo:
            self.feat_group.setChecked(True)
            for i in range(self.feat_list.count()):
                it = self.feat_list.item(i)
                it.setCheckState(Qt.Checked if it.text() in fields else Qt.Unchecked)
            for i in range(self.geo_list.count()):
                it = self.geo_list.item(i)
                it.setCheckState(Qt.Checked if it.text() in geo else Qt.Unchecked)
            if radius is not None:
                self.geo_radius.setValue(float(radius))
            copied.append("features")
        # -- HAG ("source_dimension" means HAG came pre-baked, nothing to compute)
        hag_src = src.get("hag_source")
        if hag_src is not None and hag_src != "source_dimension":
            self.hag_box.setChecked(bool(hag_src))
            if hag_src:
                i = self.hag_filter.findText(hag_src.split("+")[0])
                if i >= 0:
                    self.hag_filter.setCurrentIndex(i)
                gv = src.get("hag_ground_value")
                self.hag_ground.setText("" if gv is None else str(gv))
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
        self._scan_in_path = in_path   # what _on_scanned marks as scanned
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
        # Default-ignore value 0 only for ASPRS-style classification fields; else
        # class 0 may be real.
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
        # Re-apply names/train flags taken from "Copy settings from…" to the
        # freshly scanned values (the scan rebuild would otherwise wipe them).
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
        # Gather every source value across the rows (may already be a "5,6"),
        # dedupe + sort, sum point counts.
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
        for r in reversed(rows[1:]):        # drop the rows now folded into `keep`
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
        gtxt = self.hag_ground.text().strip()
        try:
            gv = int(gtxt) if gtxt else None
        except ValueError:
            gv = None
        if gtxt and gv is None:
            self._append(f"Ground class '{gtxt}' isn't an integer - clear it or enter a "
                         f"Source value from the Classes table.")
            return None
        label_field = self.field_combo.currentText().strip()
        feats = [self.feat_list.item(i).text() for i in range(self.feat_list.count())
                 if self.feat_list.item(i).checkState() == Qt.Checked
                 and self.feat_list.item(i).text() != label_field
                 ] if self.feat_group.isChecked() else []
        geo = [self.geo_list.item(i).text() for i in range(self.geo_list.count())
               if self.geo_list.item(i).checkState() == Qt.Checked
               ] if self.feat_group.isChecked() else []
        return {
            "name": name, "in_path": in_path, "split": split,
            "val_inputs": val_inputs, "test_inputs": test_inputs,
            "classes": classes, "ignored": ignored, "spec": self._spec(),
            "out_root": appstate.workspace_dir(),
            "compute_hag": self.hag_box.isChecked(),
            "ground_value": gv,
            "hag_filter": self.hag_filter.currentText(),
            "feature_fields": feats or None,
            "geo_features": geo or None,
            "geo_radius": float(self.geo_radius.value()),
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
            hag += f"  + geo({len(plan['geo_features'])} @ {plan['geo_radius']:g} m)"
        self._append(f"Building '{name}'{hag} ({len(classes)} classes, "
                     f"val={split.val_frac:.0%} test={split.test_frac:.0%} {split.mode} "
                     f"seed={split.seed}, ignored values: {ignored}) -> {out_root}…")

        def job(progress):
            return dataset.convert_dataset(
                name, [plan["in_path"]], plan["spec"], classes, ignored,
                out_root, val_inputs=plan["val_inputs"],
                test_inputs=plan["test_inputs"], split=split,
                intensity_norm="p95", compute_hag=plan["compute_hag"],
                ground_value=plan["ground_value"],
                hag_filter=plan["hag_filter"],
                feature_fields=plan["feature_fields"],
                geo_features=plan["geo_features"],
                geo_radius=plan["geo_radius"],
                max_workers=plan["max_workers"], progress=progress)

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
            # main.py's switcher only forwards kwargs to pages with receive_nav;
            # the Train page has none, so a plain page switch is the whole hop
            # (it reloads its dataset list on entry).
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
        # All datasets live on the ONE datasets volume the modal shells mount at
        # /datasets (TT_DATASET_VOLUME, default 'terminal-datasets'), each under
        # /<name> — the same layout the local path bind-mounts. (Was: one volume
        # per dataset, which the trainers never mounted — invisible in the cloud.)
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
        """Each row is '<name>   <badge>' (the bare name rides in UserRole):
        ✓ uploaded / ● staged / ⚠ missing (staged dir gone), colored with the
        theme's ok/warn tokens."""
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
        # Stream into the console. newline=False for chunked subprocess output
        # (uploader); True for one-shot status messages.
        ui.append_log(self.log, text, newline)


def _parse_values(text: str) -> list[int]:
    """Source-value cell -> ints. Handles one value ("5") or a list ("5,6" / "5 6")."""
    return [int(t) for t in text.replace(",", " ").split() if t]
