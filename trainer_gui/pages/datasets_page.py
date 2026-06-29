"""Datasets page — the full dataset workflow in one place, top to bottom:

  1. New dataset       point at a file/folder, name it, say where labels live
  2. Classes           scan label values, name them, mark ignored, check density
  3. Split & tiling     train/val split; data is tiled by point density by default
  4. Pre-process       (optional) add Height-Above-Ground with PDAL
  5. Convert + Upload  write the canonical npz locally, then push to a Modal volume

Tiling is on by default (size auto-derived from point density); the standalone
manual tiling step is gone — the trainer re-tiles anyway. Companion-file labels
and intensity normalization live under Advanced.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QLineEdit, QListWidget, QMessageBox, QProgressBar, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import analysis, appstate, dataset, modal_cli, pretrain, theme, ui
from ..dataset import LabelSpec, SplitConfig
from ..jobs import FuncWorker, JobRunner
from ..readers import list_label_fields

# split combo index -> SplitConfig.strategy
_SPLIT_STRATEGIES = ["auto", "scene", "tile", "provided"]


class DatasetsPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.worker = FuncWorker(self)
        self.uploader = JobRunner(self)
        self._staged_dir: Path | None = None
        self._uploading: Path | None = None   # dir currently being uploaded
        self._label_values: dict[int, int] = {}
        self._done_cb = None

        root = QVBoxLayout(self)
        title = QLabel("Datasets")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.sub.setObjectName("pageSub")
        root.addWidget(self.sub)

        # ---- the workflow: a plain top-to-bottom stack (the whole page scrolls),
        # so each section keeps its natural height and the collapsible Advanced
        # section grows/shrinks instead of being pinned to a fixed-height slot ----
        root.addWidget(self._new_dataset_box())   # 1
        root.addWidget(self._classes_box())       # 2
        root.addWidget(self._split_box())         # 3
        root.addWidget(self._advanced_box())      # 3b
        root.addWidget(self._prep_box())          # 4  (HAG — optional, below split)
        root.addLayout(self._go_layout())         # 5

        # ---- saved datasets: bottom layer ----
        root.addWidget(QLabel("Saved Datasets"))
        sd_row = QHBoxLayout()
        sd_col = QVBoxLayout()
        self.known_list = QListWidget()
        self.known_list.setMaximumHeight(120)
        self.known_list.setMaximumWidth(360)
        self.known_list.itemSelectionChanged.connect(self._show_known)
        sd_col.addWidget(self.known_list)
        # Re-upload a previously converted dataset without re-converting — works
        # after a restart (the staged_dir is remembered in state.json).
        self.upload_saved_btn = QPushButton("Upload selected to Modal")
        self.upload_saved_btn.clicked.connect(self._upload_saved)
        sd_col.addWidget(self.upload_saved_btn)
        sd_row.addLayout(sd_col)
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setAlignment(Qt.AlignTop)
        theme.set_accent(self.stats_label, "muted")
        sd_row.addWidget(self.stats_label, 1)
        root.addLayout(sd_row)
        self._reload_known()

        self._on_companion_toggled(False)
        self._on_split_changed()

        self.worker.output.connect(self._append)
        self.worker.done.connect(self._dispatch_done)
        self.worker.error.connect(self._on_worker_error)
        self.uploader.output.connect(lambda s: self._append(s, newline=False))
        self.uploader.finished.connect(self._on_upload_done)
        self.uploader.failed.connect(self._on_upload_failed)
        self.apply_exec_mode(appstate.get_exec_mode() == "local")

    def apply_exec_mode(self, local: bool):
        """Local mode never uploads to Modal — hide the upload buttons, drop the
        built-ins from the saved list, and reword the workflow copy."""
        self.upload_btn.setVisible(not local)
        self.upload_saved_btn.setVisible(not local)
        self.sub.setText(
            "Build a trainable dataset, step by step: point at point clouds "
            "(las/laz, ply, txt/csv, pcd, npy/npz), optionally add Height-Above-Ground, "
            "name the classes, split train/val (tiled by point density automatically), "
            + ("then convert — it's staged on disk and ready to train in Docker."
               if local else
               "then convert locally and upload to a per-dataset Modal volume."))
        self._reload_known()

    # ============================================================= 1. New dataset
    def _new_dataset_box(self) -> QWidget:
        box = QGroupBox("1 · New dataset")
        form = QFormLayout(box)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("my_city_lidar")
        form.addRow("Name", self.name_edit)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("a .laz file, or a folder of clouds")
        in_row = QHBoxLayout()
        in_row.addWidget(self.input_edit)
        for text, slot in (("Folder…", self._pick_input_folder), ("File…", self._pick_input_file)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            in_row.addWidget(b)
        form.addRow("Input", _wrap(in_row))
        self.field_combo = QComboBox()
        self.field_combo.setEditable(True)
        form.addRow("Label field", self.field_combo)
        self.out_edit, out_row = self._dir_row(self._pick_out)
        self.out_edit.setPlaceholderText("default: app staging folder")
        form.addRow("Output folder", out_row)
        return box

    # ============================================================= 4. Pre-process
    def _prep_box(self) -> QWidget:
        box = QGroupBox("4 · Pre-process raw clouds (optional) — add Height-Above-Ground (PDAL)")
        form = QFormLayout(box)
        self.hag_in = QLineEdit()
        self.hag_in.setPlaceholderText("a file, or a folder of clouds")
        hag_in_row = QHBoxLayout()
        hag_in_row.addWidget(self.hag_in)
        fbtn = QPushButton("Folder…")
        fbtn.clicked.connect(lambda: self._pick_into(self.hag_in, "Input clouds for HAG"))
        hag_in_row.addWidget(fbtn)
        flbtn = QPushButton("File…")
        flbtn.clicked.connect(self._pick_hag_file)
        hag_in_row.addWidget(flbtn)
        form.addRow("Input folder or file", _wrap(hag_in_row))
        self.hag_filter = QComboBox()
        self.hag_filter.addItems(list(pretrain.HAG_FILTERS))
        form.addRow("HAG filter", self.hag_filter)
        self.hag_btn = QPushButton("Add HAG")
        self.hag_btn.clicked.connect(self._run_hag)
        row = QHBoxLayout()
        row.addWidget(self.hag_btn)
        row.addStretch()
        form.addRow("", _wrap(row))
        # Loading bar for the (slow) HAG run — busy/indeterminate, shown only while
        # it runs, sitting right beneath the Add HAG button.
        self.hag_busy = QProgressBar()
        self.hag_busy.setRange(0, 0)
        self.hag_busy.setTextVisible(False)
        self.hag_busy.setVisible(False)
        form.addRow("", self.hag_busy)
        if not pretrain.pdal_available():
            self.hag_btn.setEnabled(False)
            warn = QLabel("PDAL not found — install python-pdal to enable this step.")
            warn.setWordWrap(True)
            theme.set_accent(warn, "error")
            form.addRow("", warn)
        return box

    # ============================================================= 3. Classes
    def _classes_box(self) -> QWidget:
        box = QGroupBox("2 · Classes — uncheck 'Train' to ignore a value; rows that share a "
                        "Class name are merged into one class")
        cl = QVBoxLayout(box)
        btn_row = QHBoxLayout()
        self.scan_btn = QPushButton("Scan label values")
        self.scan_btn.clicked.connect(self._scan_labels)
        btn_row.addWidget(self.scan_btn)
        self.combine_btn = QPushButton("Combine selected")
        self.combine_btn.clicked.connect(self._combine_selected)
        btn_row.addWidget(self.combine_btn)
        self.analyze_btn = QPushButton("Analyze density")
        self.analyze_btn.clicked.connect(self._analyze)
        btn_row.addWidget(self.analyze_btn)
        btn_row.addStretch()
        cl.addLayout(btn_row)
        self.class_table = QTableWidget(0, 4)
        self.class_table.setHorizontalHeaderLabels(["Train", "Source value", "Points seen", "Class name"])
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

    # ============================================================= 4. Split & tiling
    def _split_box(self) -> QWidget:
        box = QGroupBox("3 · Split & tiling")
        form = QFormLayout(box)
        self.split_combo = QComboBox()
        self.split_combo.addItems([
            "Auto — tile by point density (recommended)",
            "One scene per file (no tiling)",
            "Tile a single cloud & split its tiles",
            "Separate validation folder",
        ])
        self.split_combo.currentIndexChanged.connect(self._on_split_changed)
        form.addRow("Train / val split", self.split_combo)
        self.tile_m = QDoubleSpinBox()
        self.tile_m.setRange(0.0, 100000.0)
        self.tile_m.setValue(0.0)
        self.tile_m.setSpecialValueText("auto (from density)")
        form.addRow("Tile size (m)", self.tile_m)
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0.05, 0.5)
        self.val_ratio.setSingleStep(0.05)
        self.val_ratio.setValue(0.20)
        form.addRow("Validation fraction", self.val_ratio)
        self.val_edit, self.val_row_w = self._dir_row(self._pick_val)
        form.addRow("Validation folder", self.val_row_w)
        return box

    def _advanced_box(self) -> QWidget:
        box = QGroupBox("Advanced")
        box.setCheckable(True)
        box.setChecked(False)
        outer = QVBoxLayout(box)
        inner = QWidget()
        adv = QFormLayout(inner)
        adv.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(inner)
        box.toggled.connect(inner.setVisible)
        inner.setVisible(False)

        self.companion_chk = QCheckBox("Labels are in companion files (one label per point)")
        self.companion_chk.toggled.connect(self._on_companion_toggled)
        adv.addRow("", self.companion_chk)
        self.truth_edit, self.truth_row_w = self._dir_row(self._pick_truth)
        adv.addRow("Truth folder", self.truth_row_w)
        suffix_row = QHBoxLayout()
        self.src_suffix = QLineEdit("_PC3.txt")
        self.dst_suffix = QLineEdit("_CLS.txt")
        suffix_row.addWidget(QLabel("cloud suffix"))
        suffix_row.addWidget(self.src_suffix)
        suffix_row.addWidget(QLabel("label suffix"))
        suffix_row.addWidget(self.dst_suffix)
        self.suffix_row_w = _wrap(suffix_row)
        adv.addRow("", self.suffix_row_w)
        self.split_seed = QSpinBox()
        self.split_seed.setRange(0, 99999)
        self.split_seed.setValue(42)
        adv.addRow("Split seed", self.split_seed)
        self.intensity_norm = QComboBox()
        self.intensity_norm.addItems(["max (i / max → 0..1)", "p95 (IEEE training scripts)"])
        adv.addRow("Intensity norm", self.intensity_norm)
        return box

    # ============================================================= 5. Convert/Upload
    def _go_layout(self) -> QVBoxLayout:
        go_row = QHBoxLayout()
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.setObjectName("primary")
        self.convert_btn.clicked.connect(self._convert)
        go_row.addWidget(self.convert_btn)
        self.upload_btn = QPushButton("Upload to Modal")
        self.upload_btn.setEnabled(False)
        self.upload_btn.clicked.connect(self._upload)
        go_row.addWidget(self.upload_btn)
        go_row.addStretch()
        # Replaced the scrolling console with a loading bar + one-line status: a
        # busy/indeterminate bar shown only while an operation runs, plus the
        # latest message. Errors pop up a dialog (see _on_worker_error).
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setTextVisible(False)
        self.busy.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        theme.set_accent(self.status, "muted")
        lay = QVBoxLayout()
        lay.addLayout(go_row)
        lay.addWidget(self.busy)
        lay.addWidget(self.status)
        return lay

    def _busy_off(self):
        self.busy.setVisible(False)
        self.hag_busy.setVisible(False)

    # ------------------------------------------------------------- widgets
    def _dispatch_done(self, result):
        self._busy_off()
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
        return edit, _wrap(row)

    def _on_companion_toggled(self, on: bool):
        self.field_combo.setEnabled(not on)
        self.truth_row_w.setEnabled(on)
        self.suffix_row_w.setEnabled(on)

    def _on_split_changed(self):
        provided = self.split_combo.currentIndex() == 3
        self.val_row_w.setEnabled(provided)

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

    def _pick_truth(self):
        d = QFileDialog.getExistingDirectory(self, "Ground-truth label folder")
        if d:
            self.truth_edit.setText(d)

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Local output folder")
        if d:
            self.out_edit.setText(d)

    def _output_root(self) -> Path:
        txt = self.out_edit.text().strip()
        return Path(txt) if txt else appstate.staging_dir()

    def _pick_into(self, edit, caption):
        d = QFileDialog.getExistingDirectory(self, caption)
        if d:
            edit.setText(d)

    def _pick_hag_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Input point-cloud file for HAG")
        if f:
            self.hag_in.setText(f)

    def _populate_fields(self, path: str):
        self.field_combo.clear()
        files = dataset.expand_inputs(path)
        if not files:
            self._append(f"No supported point-cloud files in {path}")
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

    # ------------------------------------------------------------- config
    def _spec(self) -> LabelSpec:
        if self.companion_chk.isChecked():
            return LabelSpec(kind="file", truth_dir=self.truth_edit.text().strip(),
                             src_suffix=self.src_suffix.text(), dst_suffix=self.dst_suffix.text())
        return LabelSpec(kind="field", field=self.field_combo.currentText().strip())

    def _split_config(self) -> SplitConfig:
        return SplitConfig(strategy=_SPLIT_STRATEGIES[self.split_combo.currentIndex()],
                           val_ratio=float(self.val_ratio.value()),
                           seed=int(self.split_seed.value()),
                           tile_m=float(self.tile_m.value()))

    def _intensity_norm(self) -> str:
        return "p95" if self.intensity_norm.currentIndex() == 1 else "max"

    # ------------------------------------------------------------- pre-process (HAG)
    def _run_hag(self):
        in_dir = self.hag_in.text().strip()
        if not os.path.exists(in_dir):
            self._append("HAG: choose an input file or folder of point clouds.")
            return
        if self.worker.running:
            self._append("A local job is already running — wait for it to finish.")
            return
        # No output field: HAG writes to a sibling "<name>_hag" folder under the
        # staging/output root and auto-feeds it in as the dataset input.
        out_dir = str(self._output_root() / f"{Path(in_dir).stem}_hag")
        flt = self.hag_filter.currentText()
        self.hag_btn.setEnabled(False)
        self.hag_busy.setVisible(True)
        self._append(f"Adding HAG ({flt}, SMRF ground) …")

        def job(progress):
            return pretrain.add_hag(in_dir, out_dir, hag_filter=flt, progress=progress)

        self._done_cb = self._on_hag_done
        self.worker.start(job)

    def _on_hag_done(self, summary):
        self.hag_btn.setEnabled(pretrain.pdal_available())
        out = summary["output_dir"]
        if not self.input_edit.text().strip():   # streamline: feed straight into the dataset
            self._set_input(out)
        self._append(f"\n✓ HAG done — {summary['n_files']} cloud(s), "
                     f"{summary['total_points']:,} pts -> {out}\n"
                     f"  (set as the dataset input above — scan labels and convert).")

    # ------------------------------------------------------------- scan / analyze
    def _scan_labels(self):
        in_path = self.input_edit.text().strip()
        if not os.path.exists(in_path):
            self._append("Choose an input file or folder first.")
            return
        spec = self._spec()
        self._append("Scanning label values…")
        self.scan_btn.setEnabled(False)
        self.busy.setVisible(True)

        def job(progress):
            files = dataset.expand_inputs(in_path)
            progress(f"  sampling {min(len(files), 8)} of {len(files)} file(s)")
            return dataset.scan_label_values(files, spec)

        self._done_cb = self._on_scanned
        self.worker.start(job)

    def _on_scanned(self, counts):
        self.scan_btn.setEnabled(True)
        self._label_values = counts
        # Only default-ignore value 0 when the label clearly follows the ASPRS
        # classification convention; otherwise class 0 may be a real class.
        field = self._spec().field.lower()
        ignore_zero = self.companion_chk.isChecked() or "class" in field
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
        self._append(f"Found {len(counts)} distinct label values. Name the classes, "
                     f"uncheck any that mean 'unknown', then Convert.")

    def _analyze(self):
        in_path = self.input_edit.text().strip()
        if not os.path.exists(in_path):
            self._append("Choose an input file or folder first.")
            return
        self.analyze_btn.setEnabled(False)
        self.busy.setVisible(True)
        self._append("Analyzing density…")

        def job(progress):
            files = dataset.expand_inputs(in_path)
            progress(f"  scanning up to {analysis.MAX_FILES_PER_SPLIT} of {len(files)} file(s)")
            return analysis.scan_folder(files)

        self._done_cb = self._on_analyzed
        self.worker.start(job)

    def _on_analyzed(self, stats):
        self.analyze_btn.setEnabled(True)
        recs = analysis.recommend(stats)
        chunk = next(iter(recs.values())).get("chunk_xy", 0.0)
        if chunk:                                   # density preset -> fill the tile size
            self.tile_m.setValue(float(chunk))
        self.analyze_label.setText(
            f"Density: {stats['mean_pts_per_m2']:.2f} pts/m²  ·  "
            f"spacing {stats['mean_spacing_m']:.2f} m  ·  "
            f"largest scene {stats['max_scene_points']:,} pts  ·  "
            f"recommended tile {chunk:.0f} m (set in Split & tiling).")

    # ------------------------------------------------------------- convert/upload
    def _combine_selected(self):
        """Merge the selected class rows into one class by giving them a shared
        name — rows with the same Class name map to the same index on convert."""
        rows = sorted({i.row() for i in self.class_table.selectedItems()})
        if len(rows) < 2:
            self._append("Combine: select 2+ class rows first (click rows; Ctrl/Shift for many).")
            return
        first = self.class_table.item(rows[0], 3)
        base = (first.text().strip() if first else "") or \
            f"class_{self.class_table.item(rows[0], 1).text()}"
        name, ok = QInputDialog.getText(self, "Combine classes",
                                        "Name for the combined class:", text=base)
        name = name.strip()
        if not ok or not name:
            return
        for r in rows:
            self.class_table.setItem(r, 3, QTableWidgetItem(name))
            chk = self.class_table.cellWidget(r, 0).findChild(QCheckBox)
            chk.setChecked(True)            # combining implies training on it
        self._refresh_group_counts()        # show the merged classes added together
        vals = ", ".join(self.class_table.item(r, 1).text() for r in rows)
        self._append(f"Combined source values [{vals}] into one class '{name}'.")

    def _refresh_group_counts(self):
        """Display each row's 'Points seen' as the total over all rows sharing its
        class name, so combined classes read as one added-together class. Col 2 is
        display-only — conversion recounts class totals from the actual data."""
        totals: dict[str, int] = {}
        for r in range(self.class_table.rowCount()):
            val = int(self.class_table.item(r, 1).text())
            name = self.class_table.item(r, 3).text().strip() or f"class_{val}"
            totals[name] = totals.get(name, 0) + self._label_values.get(val, 0)
        for r in range(self.class_table.rowCount()):
            val = int(self.class_table.item(r, 1).text())
            name = self.class_table.item(r, 3).text().strip() or f"class_{val}"
            cell = self.class_table.item(r, 2)
            if cell:
                cell.setText(f"{totals[name]:,}")

    def _classes_from_table(self):
        """Rows sharing a Class name collapse to one class index (combine); each
        unique name gets the next contiguous index, in first-seen order."""
        name_to_index: dict[str, int] = {}
        classes, ignored = [], []
        for r in range(self.class_table.rowCount()):
            val = int(self.class_table.item(r, 1).text())
            chk = self.class_table.cellWidget(r, 0).findChild(QCheckBox)
            if not chk.isChecked():
                ignored.append(val)
                continue
            name = self.class_table.item(r, 3).text().strip() or f"class_{val}"
            if name not in name_to_index:
                name_to_index[name] = len(name_to_index)
            classes.append({"index": name_to_index[name], "source_value": val, "name": name})
        return classes, ignored

    def _convert(self):
        name = self.name_edit.text().strip()
        in_path = self.input_edit.text().strip()
        if not (name and os.path.exists(in_path)):
            self._append("Need a name and an input file or folder.")
            return
        split = self._split_config()
        val_inputs = None
        if split.strategy == "provided":
            val_dir = self.val_edit.text().strip()
            if not os.path.isdir(val_dir):
                self._append("'Separate validation folder' is selected — choose that folder.")
                return
            val_inputs = [val_dir]
        if self.class_table.rowCount() == 0:
            self._append("Run 'Scan label values' and name your classes first.")
            return
        classes, ignored = self._classes_from_table()
        if not classes:
            self._append("All label values are unchecked — nothing to train on.")
            return
        spec = self._spec()
        norm = self._intensity_norm()
        out_root = self._output_root()
        self.convert_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.busy.setVisible(True)
        self._append(f"Converting '{name}' ({len(classes)} classes, split={split.strategy}, "
                     f"ignored values: {ignored}) -> {out_root}…")

        def job(progress):
            return dataset.convert_dataset(name, [in_path], spec, classes, ignored,
                                           out_root, val_inputs=val_inputs,
                                           split=split, intensity_norm=norm, progress=progress)

        self._done_cb = self._on_converted
        self.worker.start(job)

    def _on_converted(self, staged: Path):
        self._staged_dir = staged
        self.convert_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        appstate.remember_dataset(staged.name, {
            "staged_dir": str(staged),
            "meta_path": str(staged / "dataset_meta.json"),
            "uploaded": False,
        })
        self._reload_known()
        if appstate.get_exec_mode() == "local":
            self._append(f"\n✓ Converted -> {staged}\nReady — pick '{staged.name}' on the Train "
                         f"page (it's bind-mounted into the container at /datasets/{staged.name}).")
        else:
            self._append(f"\n✓ Converted -> {staged}\nClick 'Upload to Modal' to push it to its "
                         f"per-dataset volume.")

    def _upload(self):
        if not (self._staged_dir and self._staged_dir.exists()):
            self._append("Convert a dataset first.")
            return
        self._start_upload(self._staged_dir)

    def _upload_saved(self):
        """Upload a dataset already listed under Saved Datasets, using its
        remembered staged_dir — no re-conversion. If that folder has moved or was
        deleted, ask where the converted folder is now (must hold dataset_meta.json)."""
        items = self.known_list.selectedItems()
        if not items:
            self._append("Select a saved dataset to upload.")
            return
        name = items[0].text()
        info = appstate.known_datasets().get(name, {})
        if info.get("builtin"):
            self._append(f"'{name}' is built-in — it already lives on the ieee-data volume.")
            return
        staged = Path(info.get("staged_dir", ""))
        if not (str(staged) and staged.exists() and (staged / "dataset_meta.json").exists()):
            picked = QFileDialog.getExistingDirectory(
                self, f"Locate the converted '{name}' folder (must contain dataset_meta.json)",
                str(staged.parent) if str(staged) else "")
            if not picked:
                return
            staged = Path(picked)
            if not (staged / "dataset_meta.json").exists():
                self._append(f"✗ {staged} has no dataset_meta.json — not a converted dataset.")
                return
            appstate.remember_dataset(staged.name, {
                "staged_dir": str(staged),
                "meta_path": str(staged / "dataset_meta.json"),
                "uploaded": False,
            })
        self._start_upload(staged)

    def _start_upload(self, staged: Path):
        # Each dataset gets its own auto-created Modal volume named after it; the
        # training script mounts it at /datasets, so the remote path stays /<name>.
        self._uploading = staged
        name = staged.name
        self.upload_btn.setEnabled(False)
        self.upload_saved_btn.setEnabled(False)
        self.busy.setVisible(True)
        self._append(f"\nCreating + uploading volume '{name}' (-> /{name}) …")
        prog, args = modal_cli.volume_put(name, str(staged), f"/{name}")
        self.uploader.start(prog, args, cwd=self.repo_root,
                            pre=modal_cli.volume_create(name))

    def _on_upload_failed(self, err: str):
        self._busy_off()
        self.upload_btn.setEnabled(True)
        self.upload_saved_btn.setEnabled(True)
        self._append(f"✗ Upload process failed to run: {err}. Is the Modal CLI on PATH "
                     f"and authenticated? (modal token new)")

    def _on_upload_done(self, code: int):
        self._busy_off()
        self.upload_btn.setEnabled(True)
        self.upload_saved_btn.setEnabled(True)
        staged = self._uploading
        if code != 0:
            self._append(f"\n✗ Upload failed (exit {code}). Is the Modal CLI installed and "
                         f"authenticated? (modal token new)")
            return
        name = staged.name
        appstate.remember_dataset(name, {
            "staged_dir": str(staged),
            "meta_path": str(staged / "dataset_meta.json"),
            "uploaded": True,
            "volume": name,
        })
        self._reload_known()
        self._append(f"\n✓ Dataset '{name}' uploaded to volume '{name}'. Head to the Train page.")

    def _on_worker_error(self, tb: str):
        self._done_cb = None
        self._busy_off()
        self.scan_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.convert_btn.setEnabled(True)
        self.hag_btn.setEnabled(pretrain.pdal_available())
        self.status.setText("✗ Error — see the dialog for details.")
        QMessageBox.critical(self, "Dataset error", tb)

    # ------------------------------------------------------------- known list
    def _reload_known(self):
        # selectable_datasets() drops built-ins in local mode (no /data pipeline).
        self.known_list.clear()
        for name in sorted(appstate.selectable_datasets()):
            self.known_list.addItem(name)

    def _show_known(self):
        items = self.known_list.selectedItems()
        if not items:
            return
        info = appstate.known_datasets().get(items[0].text(), {})
        if info.get("builtin"):
            self.stats_label.setText(f"{items[0].text()} (built-in): {info.get('note', '')}")
            return
        staged = info.get("staged_dir", "")
        on_disk = bool(staged) and os.path.isdir(staged)
        if appstate.get_exec_mode() == "local":
            status = ("staged on disk ✓ — ready to train" if on_disk
                      else "local copy missing — re-convert it on this machine")
        else:
            status = ("uploaded ✓ (re-upload to refresh)" if info.get("uploaded")
                      else "not uploaded — click “Upload selected to Modal”")
            if not on_disk:
                status += "  ·  local copy missing (upload will ask where it is)"
        meta_path = info.get("meta_path", "")
        if meta_path and os.path.exists(meta_path):
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            s = meta.get("stats", {})
            self.stats_label.setText(
                f"{meta['name']}: {meta['num_classes']} classes "
                f"({', '.join(meta['class_names'])})  ·  "
                f"{s.get('mean_pts_per_m2', 0):.2f} pts/m²  ·  "
                f"train {len(meta['splits']['train']['scenes'])}, "
                f"val {len(meta['splits']['val']['scenes'])} scenes\n{status}")
        else:
            self.stats_label.setText(f"{items[0].text()}\n{status}")

    # ------------------------------------------------------------- helpers
    def _append(self, text: str, newline: bool = True):
        # The console is gone — show the latest message on one line (whitespace +
        # newlines collapsed) in the status label beside the loading bar.
        msg = " ".join(text.split())
        if msg:
            self.status.setText(msg)


def _wrap(layout) -> QWidget:
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w
