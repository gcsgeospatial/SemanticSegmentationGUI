"""Datasets page — the dataset workflow, top to bottom:

  1. New dataset   point at a file/folder, name it, say which field holds labels
  2. Classes       scan label values, name them, mark ignored, check density
  3. Tiling        train/val split + tile size; Start Tiling stages the dataset
  4. HAG           add a per-tile Height-Above-Ground channel -> a sibling
                   <name>_hag dataset (PDAL SMRF -> hag_nn over each tile)

Labels come from a field in the cloud (companion/sidecar label files are no
longer offered here). Intensity is normalized by max (i/max -> 0..1) by default.
Density-generalization controls have moved to the Train and Inference pages.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QLineEdit, QListWidget, QMessageBox, QProgressBar, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import analysis, appstate, dataset, modal_cli, pretrain, theme
from ..dataset import LabelSpec, SplitConfig
from ..jobs import FuncWorker, JobRunner
from ..readers import list_label_fields

# split combo index -> SplitConfig.strategy
_SPLIT_STRATEGIES = ["scene", "tile", "provided"]


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

        # The whole page scrolls, so each section keeps its natural height.
        root.addWidget(self._new_dataset_box())   # 1
        root.addWidget(self._classes_box())        # 2
        root.addWidget(self._tiling_box())         # 3  (incl. Start Tiling)
        root.addWidget(self._prep_box())           # 4  (HAG — a sibling dataset)
        root.addLayout(self._status_block())       # shared busy bar + status line

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

        self._on_split_changed()

        self.worker.output.connect(self._append)
        self.worker.done.connect(self._dispatch_done)
        self.worker.error.connect(self._on_worker_error)
        self.uploader.output.connect(lambda s: self._append(s, newline=False))
        self.uploader.finished.connect(self._on_upload_done)
        self.uploader.failed.connect(self._on_upload_failed)
        self.apply_exec_mode(appstate.get_exec_mode() == "local")

    def apply_exec_mode(self, local: bool):
        """Local mode never uploads to Modal — hide the upload button, drop the
        built-ins from the saved list, and reword the workflow copy."""
        self.upload_saved_btn.setVisible(not local)
        self.sub.setText(
            "Build a trainable dataset, step by step: point at point clouds "
            "(las/laz, ply, txt/csv, pcd, npy/npz), name the classes, split "
            "train/val and Start Tiling"
            + (" — it's staged on disk and ready to train in Docker. "
               if local else
               ", then upload it to a per-dataset Modal volume. ")
            + "Add a Height-Above-Ground channel last to get a sibling HAG dataset.")
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

    # ============================================================= 2. Classes
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

    # ============================================================= 3. Tiling
    def _tiling_box(self) -> QWidget:
        box = QGroupBox("3 · Tiling")
        form = QFormLayout(box)
        self.split_combo = QComboBox()
        self.split_combo.addItems([
            "Folder of clouds → split by val fraction",
            "Single cloud → tile & split by val fraction",
            "Separate train + val folders (use as-is)",
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
        self.tile_btn = QPushButton("Start Tiling")
        self.tile_btn.setObjectName("primary")
        self.tile_btn.clicked.connect(self._start_tiling)
        row = QHBoxLayout()
        row.addWidget(self.tile_btn)
        row.addStretch()
        form.addRow("", _wrap(row))
        return box

    # ============================================================= 4. HAG
    def _prep_box(self) -> QWidget:
        box = QGroupBox("4 · Height-Above-Ground — adds a per-tile HAG channel "
                        "(saved as a separate <name>_hag dataset)")
        form = QFormLayout(box)
        self.hag_in = QLineEdit()
        self.hag_in.setPlaceholderText("a tiled dataset folder (has train/ and val/)")
        hag_in_row = QHBoxLayout()
        hag_in_row.addWidget(self.hag_in)
        fbtn = QPushButton("Folder…")
        fbtn.clicked.connect(self._pick_hag_in)
        hag_in_row.addWidget(fbtn)
        form.addRow("Tiled dataset", _wrap(hag_in_row))
        self.hag_filter = QComboBox()
        self.hag_filter.addItems(list(pretrain.HAG_FILTERS))
        form.addRow("HAG filter", self.hag_filter)
        # Skip SMRF only when the tiles already carry ground class 2 (rare for
        # label-remapped tiles — hag_for_cloud re-runs SMRF when no ground exists).
        self.hag_skip_ground = QCheckBox("Tiles already carry ground class 2 — skip SMRF")
        self.hag_skip_ground.setToolTip("Most tiled datasets store remapped training labels, "
                                        "not ASPRS ground; SMRF still runs unless real ground "
                                        "(class 2) is present in the tile.")
        form.addRow("", self.hag_skip_ground)
        self.hag_btn = QPushButton("Start HAG")
        self.hag_btn.clicked.connect(self._run_hag)
        row = QHBoxLayout()
        row.addWidget(self.hag_btn)
        row.addStretch()
        form.addRow("", _wrap(row))
        # Loading bar for the (slow) HAG run — busy/indeterminate, shown only while
        # it runs, sitting right beneath the Start HAG button.
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

    # ============================================================= shared status
    def _status_block(self) -> QVBoxLayout:
        # A busy/indeterminate bar shown only while an operation runs, plus the
        # latest one-line message. Errors pop up a dialog (see _on_worker_error).
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setTextVisible(False)
        self.busy.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        theme.set_accent(self.status, "muted")
        lay = QVBoxLayout()
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

    def _on_split_changed(self):
        provided = self.split_combo.currentIndex() == 2
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

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Local output folder")
        if d:
            self.out_edit.setText(d)

    def _output_root(self) -> Path:
        txt = self.out_edit.text().strip()
        return Path(txt) if txt else appstate.staging_dir()

    def _pick_hag_in(self):
        d = QFileDialog.getExistingDirectory(
            self, "Converted dataset folder (must contain train/ and val/)",
            self.hag_in.text().strip() or str(appstate.staging_dir()))
        if d:
            self.hag_in.setText(d)

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
        return LabelSpec(kind="field", field=self.field_combo.currentText().strip())

    def _split_config(self) -> SplitConfig:
        return SplitConfig(strategy=_SPLIT_STRATEGIES[self.split_combo.currentIndex()],
                           val_ratio=float(self.val_ratio.value()),
                           tile_m=float(self.tile_m.value()))

    # ------------------------------------------------------------- HAG (per-tile)
    def _run_hag(self):
        in_dir = self.hag_in.text().strip() or (str(self._staged_dir) if self._staged_dir else "")
        if not in_dir or not os.path.isdir(in_dir):
            self._append("HAG: choose a converted dataset folder (one with train/ and val/).")
            return
        if not os.path.isfile(os.path.join(in_dir, "dataset_meta.json")):
            self._append("HAG: that folder isn't a converted dataset — run Start Tiling first, or "
                         "pick a folder containing dataset_meta.json.")
            return
        if self.worker.running:
            self._append("A local job is already running — wait for it to finish.")
            return
        src = Path(in_dir)
        out_dir = src.parent / f"{src.name}_hag"
        skip_ground = self.hag_skip_ground.isChecked()
        flt = self.hag_filter.currentText()
        self.hag_btn.setEnabled(False)
        self.tile_btn.setEnabled(False)
        self.hag_busy.setVisible(True)
        self._append(f"Adding per-tile HAG ({flt}) to '{src.name}' -> '{out_dir.name}' …")

        def job(progress):
            return dataset.add_hag_to_dataset(src, out_dir, skip_ground=skip_ground,
                                              hag_filter=flt, progress=progress)

        self._done_cb = self._on_hag_done
        self.worker.start(job)

    def _on_hag_done(self, out_dir):
        self.tile_btn.setEnabled(True)
        self.hag_btn.setEnabled(pretrain.pdal_available())
        out = Path(out_dir)
        self._staged_dir = out      # a following Upload targets the HAG set
        appstate.remember_dataset(out.name, {
            "staged_dir": str(out),
            "meta_path": str(out / "dataset_meta.json"),
            "uploaded": False,
        })
        self._reload_known()
        if appstate.get_exec_mode() == "local":
            self._append(f"✓ HAG dataset '{out.name}' ready — pick it (or a *_hag backbone) "
                         f"on the Train page.")
        else:
            self._append(f"✓ HAG dataset '{out.name}' staged — select it under Saved Datasets "
                         f"to upload to Modal.")

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
        self._append(f"Found {len(counts)} distinct label values. Name the classes, "
                     f"uncheck any that mean 'unknown', then Start Tiling.")

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
            f"recommended tile {chunk:.0f} m (set in Tiling).")

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

    def _conversion_plan(self):
        name = self.name_edit.text().strip()
        in_path = self.input_edit.text().strip()
        if not name or not os.path.exists(in_path):
            self._append("Need a name and an input file or folder.")
            return None
        split = self._split_config()
        val_inputs = None
        if split.strategy == "provided":
            val_dir = self.val_edit.text().strip()
            if not os.path.isdir(val_dir):
                self._append("'Separate train + val folders' is selected — choose the val folder.")
                return None
            val_inputs = [val_dir]
        if self.class_table.rowCount() == 0:
            self._append("Run 'Scan label values' and name your classes first.")
            return None
        classes, ignored = self._classes_from_table()
        if not classes:
            self._append("All label values are unchecked — nothing to train on.")
            return None
        return {
            "name": name, "in_path": in_path, "split": split, "val_inputs": val_inputs,
            "classes": classes, "ignored": ignored, "spec": self._spec(),
            "out_root": self._output_root(),
        }

    def _start_tiling(self):
        plan = self._conversion_plan()
        if plan is None:
            return
        name, classes, ignored = plan["name"], plan["classes"], plan["ignored"]
        split, out_root = plan["split"], plan["out_root"]
        self.tile_btn.setEnabled(False)
        self.hag_btn.setEnabled(False)
        self.busy.setVisible(True)
        self._append(f"Tiling '{name}' ({len(classes)} classes, split={split.strategy}, "
                     f"ignored values: {ignored}) -> {out_root}…")

        def job(progress):
            return dataset.convert_dataset(
                name, [plan["in_path"]], plan["spec"], classes, ignored,
                out_root, val_inputs=plan["val_inputs"], split=split,
                intensity_norm="max", progress=progress)

        self._done_cb = self._on_converted
        self.worker.start(job)

    def _on_converted(self, staged: Path):
        self._staged_dir = staged
        self.tile_btn.setEnabled(True)
        self.hag_btn.setEnabled(pretrain.pdal_available())
        self.hag_in.setText(str(staged))   # default the HAG step at the dataset just tiled
        appstate.remember_dataset(staged.name, {
            "staged_dir": str(staged),
            "meta_path": str(staged / "dataset_meta.json"),
            "uploaded": False,
        })
        self._reload_known()
        if appstate.get_exec_mode() == "local":
            self._append(f"✓ Tiled -> {staged}. Ready — pick '{staged.name}' on the Train page "
                         f"(bind-mounted at /datasets/{staged.name}). Add HAG below for a "
                         f"sibling HAG dataset.")
        else:
            self._append(f"✓ Tiled -> {staged}. Select it under Saved Datasets to upload, or "
                         f"add HAG below for a sibling HAG dataset.")

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
        self.upload_saved_btn.setEnabled(False)
        self.busy.setVisible(True)
        self._append(f"\nCreating + uploading volume '{name}' (-> /{name}) …")
        prog, args = modal_cli.volume_put(name, str(staged), f"/{name}")
        self.uploader.start(prog, args, cwd=self.repo_root,
                            pre=modal_cli.volume_create(name))

    def _on_upload_failed(self, err: str):
        self._busy_off()
        self.upload_saved_btn.setEnabled(True)
        self._append(f"✗ Upload process failed to run: {err}. Is the Modal CLI on PATH "
                     f"and authenticated? (modal token new)")

    def _on_upload_done(self, code: int):
        self._busy_off()
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
        self.tile_btn.setEnabled(True)
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
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            s = meta.get("stats", {})
            hag = "  ·  HAG ✓" if meta.get("has_hag") else ""
            self.stats_label.setText(
                f"{meta['name']}: {meta['num_classes']} classes "
                f"({', '.join(meta['class_names'])})  ·  "
                f"{s.get('mean_pts_per_m2', 0):.2f} pts/m²  ·  "
                f"train {len(meta['splits']['train']['scenes'])}, "
                f"val {len(meta['splits']['val']['scenes'])} scenes{hag}\n{status}")
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
