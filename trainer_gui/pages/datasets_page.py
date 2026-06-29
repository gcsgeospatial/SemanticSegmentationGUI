"""Datasets page — the dataset workflow, top to bottom:

  1. New dataset   point at a file/folder, name it, say which field holds labels
  2. Classes       scan label values, name them, mark ignored, check density
  3. Split         train/val split (whole scenes; the dataset layer does NOT tile —
                   each training script tiles for its own model) + optionally compute
                   a per-scene Height-Above-Ground channel (PDAL SMRF -> hag)

Labels come from a field in the cloud (companion/sidecar label files are no
longer offered here). Intensity is normalized by max (i/max -> 0..1) by default.
Density-generalization controls have moved to the Train and Inference pages.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import analysis, appstate, dataset, modal_cli, pretrain, theme
from ..dataset import LabelSpec, SplitConfig
from ..jobs import FuncWorker, JobRunner
from ..readers import list_label_fields

# split combo index -> SplitConfig.strategy
_SPLIT_STRATEGIES = ["scene", "spatial", "provided"]


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
        root.addWidget(self._tiling_box())         # 3  (incl. optional HAG + Start Tiling)
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
            "train/val and Build dataset"
            + (" — it's staged on disk and ready to train in Docker."
               if local else
               ", then upload it to a per-dataset Modal volume."))
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
        box = QGroupBox("2 · Classes — uncheck 'Train' to ignore a value; select rows + "
                        "Combine to merge them into one class")
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
        box = QGroupBox("3 · Train / val split")
        form = QFormLayout(box)
        self.split_combo = QComboBox()
        self.split_combo.addItems([
            "Folder of clouds → split scenes by val fraction",
            "Single cloud → spatial train/val split by val fraction",
            "Separate train + val folders (use as-is)",
        ])
        self.split_combo.currentIndexChanged.connect(self._on_split_changed)
        form.addRow("Train / val split", self.split_combo)
        # No tiling here — each training script tiles for its own model; the dataset
        # layer only decides which whole scenes (or scene regions) go to each split.
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0.05, 0.5)
        self.val_ratio.setSingleStep(0.05)
        self.val_ratio.setValue(0.20)
        form.addRow("Validation fraction", self.val_ratio)
        self.val_edit, self.val_row_w = self._dir_row(self._pick_val)
        form.addRow("Validation folder", self.val_row_w)
        # Optional: compute HeightAboveGround per scene in the same pass (one read/
        # write per scene). Whole-scene SMRF -> better ground than per-tile.
        self.hag_chk = QCheckBox("Compute Height-Above-Ground (HAG)")
        self.hag_chk.setToolTip("Bakes a per-point HAG channel (PDAL SMRF -> hag) into every "
                                "scene as it's written. The *_hag models use it; the others "
                                "ignore the extra channel.")
        self.hag_chk.toggled.connect(lambda on: self.hag_opts_w.setVisible(on))
        form.addRow("Height-Above-Ground", self.hag_chk)
        self.hag_filter = QComboBox()
        self.hag_filter.addItems(list(pretrain.HAG_FILTERS))
        self.hag_skip_ground = QCheckBox("ground already classified (class 2) — skip SMRF")
        hag_row = QHBoxLayout()
        hag_row.addWidget(QLabel("filter"))
        hag_row.addWidget(self.hag_filter)
        hag_row.addWidget(self.hag_skip_ground)
        hag_row.addStretch()
        self.hag_opts_w = _wrap(hag_row)
        self.hag_opts_w.setVisible(False)
        form.addRow("", self.hag_opts_w)
        if not pretrain.pdal_available():
            self.hag_chk.setEnabled(False)
            self.hag_chk.setText("Compute Height-Above-Ground (HAG) — PDAL not installed")
        self.tile_btn = QPushButton("Build dataset")
        self.tile_btn.setObjectName("primary")
        self.tile_btn.clicked.connect(self._start_tiling)
        row = QHBoxLayout()
        row.addWidget(self.tile_btn)
        row.addStretch()
        form.addRow("", _wrap(row))
        return box

    # ============================================================= shared console
    def _status_block(self) -> QVBoxLayout:
        # A scrolling console log — progress streams here line by line (clearer
        # than an indeterminate bar). Errors also pop up a dialog (_on_worker_error).
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("log")
        self.log.setMinimumHeight(140)
        self.log.setPlaceholderText("Progress and messages appear here…")
        lay = QVBoxLayout()
        lay.addWidget(self.log)
        return lay

    # ------------------------------------------------------------- widgets
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
                           val_ratio=float(self.val_ratio.value()))

    # ------------------------------------------------------------- scan / analyze
    def _scan_labels(self):
        in_path = self.input_edit.text().strip()
        if not os.path.exists(in_path):
            self._append("Choose an input file or folder first.")
            return
        spec = self._spec()
        self._append("Scanning label values…")
        self.scan_btn.setEnabled(False)
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
                     f"uncheck any that mean 'unknown', then Build dataset.")

    def _analyze(self):
        in_path = self.input_edit.text().strip()
        if not os.path.exists(in_path):
            self._append("Choose an input file or folder first.")
            return
        self.analyze_btn.setEnabled(False)
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
        self.analyze_label.setText(
            f"Density: {stats['mean_pts_per_m2']:.2f} pts/m²  ·  "
            f"spacing {stats['mean_spacing_m']:.2f} m  ·  "
            f"largest scene {stats['max_scene_points']:,} pts  ·  "
            f"suggested training tile {chunk:.0f} m (set per model on the Train page).")

    # ------------------------------------------------------------- convert/upload
    def _combine_selected(self):
        """Collapse the selected rows into ONE row whose Source value lists every
        merged value (e.g. "5,6") and whose Points seen is their total, under a
        shared name — so a combine reads as one class, not duplicated rows."""
        rows = sorted({i.row() for i in self.class_table.selectedItems()})
        if len(rows) < 2:
            self._append("Combine: select 2+ class rows first (click rows; Ctrl/Shift for many).")
            return
        first = self.class_table.item(rows[0], 3)
        base = first.text().strip() if first else ""
        name, ok = QInputDialog.getText(self, "Combine classes",
                                        "Name for the combined class:", text=base)
        name = name.strip()
        if not ok or not name:
            return
        # Gather every source value across the selected rows (a row may already be
        # a combined "5,6"), dedupe + sort, sum their point counts.
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
        """One class per row. A row's Source value may list several values (a
        combine, e.g. "5,6") — each maps to the SAME class index/name."""
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
            "compute_hag": pretrain.pdal_available() and self.hag_chk.isChecked(),
            "skip_ground": self.hag_skip_ground.isChecked(),
            "hag_filter": self.hag_filter.currentText(),
        }

    def _start_tiling(self):
        plan = self._conversion_plan()
        if plan is None:
            return
        name, classes, ignored = plan["name"], plan["classes"], plan["ignored"]
        split, out_root = plan["split"], plan["out_root"]
        self.tile_btn.setEnabled(False)
        hag = "  + HAG" if plan["compute_hag"] else ""
        self._append(f"Building '{name}'{hag} ({len(classes)} classes, split={split.strategy}, "
                     f"ignored values: {ignored}) -> {out_root}…")

        def job(progress):
            return dataset.convert_dataset(
                name, [plan["in_path"]], plan["spec"], classes, ignored,
                out_root, val_inputs=plan["val_inputs"], split=split,
                intensity_norm="max", compute_hag=plan["compute_hag"],
                skip_ground=plan["skip_ground"], hag_filter=plan["hag_filter"],
                progress=progress)

        self._done_cb = self._on_converted
        self.worker.start(job)

    def _on_converted(self, staged: Path):
        self._staged_dir = staged
        self.tile_btn.setEnabled(True)
        appstate.remember_dataset(staged.name, {
            "staged_dir": str(staged),
            "meta_path": str(staged / "dataset_meta.json"),
            "uploaded": False,
        })
        self._reload_known()
        hag = " (with HAG)" if self.hag_chk.isChecked() and pretrain.pdal_available() else ""
        if appstate.get_exec_mode() == "local":
            self._append(f"✓ Built{hag} -> {staged}. Ready — pick '{staged.name}' on the Train "
                         f"page (bind-mounted at /datasets/{staged.name}).")
        else:
            self._append(f"✓ Built{hag} -> {staged}. Select it under Saved Datasets to upload.")

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
        self._append(f"\nCreating + uploading volume '{name}' (-> /{name}) …")
        prog, args = modal_cli.volume_put(name, str(staged), f"/{name}")
        self.uploader.start(prog, args, cwd=self.repo_root,
                            pre=modal_cli.volume_create(name))

    def _on_upload_failed(self, err: str):
        self.upload_saved_btn.setEnabled(True)
        self._append(f"✗ Upload process failed to run: {err}. Is the Modal CLI on PATH "
                     f"and authenticated? (modal token new)")

    def _on_upload_done(self, code: int):
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
        self.scan_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.tile_btn.setEnabled(True)
        self._append("✗ Error — see the dialog for details.")
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
        # Stream into the scrolling console. newline=False for chunked subprocess
        # output (the uploader); True for one-shot status messages.
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text + ("\n" if newline else ""))
        self.log.moveCursor(QTextCursor.End)


def _parse_values(text: str) -> list[int]:
    """Source-value cell -> ints. A cell may hold one value ("5") or a combined
    list ("5,6" / "5 6"); both parse to a list of ints."""
    return [int(t) for t in text.replace(",", " ").split() if t]


def _wrap(layout) -> QWidget:
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w
