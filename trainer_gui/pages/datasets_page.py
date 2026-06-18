"""Datasets page: pick train/val folders + label source, scan classes, analyze
density, convert to the canonical npz layout and upload to terminal-datasets."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QPlainTextEdit, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import analysis, appstate, dataset, modal_cli, ui
from ..dataset import LabelSpec
from ..jobs import FuncWorker, JobRunner
from ..readers import list_label_fields


class DatasetsPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.worker = FuncWorker(self)
        self.uploader = JobRunner(self)
        self._staged_dir: Path | None = None
        self._label_values: dict[int, int] = {}

        root = QVBoxLayout(self)
        title = QLabel("Datasets")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        sub = QLabel("Point at a training folder and a validation folder of point clouds "
                     "(las/laz, ply, txt/csv, pcd, npy/npz). Choose where the ground-truth "
                     "labels live and which value means 'unknown'; the app converts everything "
                     "to a canonical format and uploads it to the terminal-datasets volume.")
        sub.setWordWrap(True)
        sub.setObjectName("pageSub")
        root.addWidget(sub)

        # ---- left: known datasets ----
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.addWidget(QLabel("Known datasets"))
        self.known_list = QListWidget()
        self.known_list.setMinimumHeight(420)
        self.known_list.itemSelectionChanged.connect(self._show_known)
        ll.addWidget(self.known_list, 1)
        self._reload_known()

        # ---- right: create a dataset ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)

        form_box = QGroupBox("New dataset")
        form = QFormLayout(form_box)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("my_city_lidar")
        form.addRow("Name", self.name_edit)
        self.train_edit, train_row = self._dir_row(self._pick_train)
        form.addRow("Training folder", train_row)
        self.val_edit, val_row = self._dir_row(self._pick_val)
        form.addRow("Validation folder", val_row)

        self.label_kind = QComboBox()
        self.label_kind.addItems(["Field inside each file", "Companion label file (one per point)"])
        self.label_kind.currentIndexChanged.connect(self._on_label_kind)
        form.addRow("Labels come from", self.label_kind)

        self.field_combo = QComboBox()
        self.field_combo.setEditable(True)
        form.addRow("Label field", self.field_combo)

        self.truth_edit, truth_row = self._dir_row(self._pick_truth)
        self.truth_row_w = truth_row
        form.addRow("Truth folder", truth_row)
        suffix_row = QHBoxLayout()
        self.src_suffix = QLineEdit("_PC3.txt")
        self.dst_suffix = QLineEdit("_CLS.txt")
        suffix_row.addWidget(QLabel("cloud suffix"))
        suffix_row.addWidget(self.src_suffix)
        suffix_row.addWidget(QLabel("label suffix"))
        suffix_row.addWidget(self.dst_suffix)
        self.suffix_row_w = _wrap(suffix_row)
        form.addRow("", self.suffix_row_w)
        self._on_label_kind()

        # ---- classes ----
        cls_box = QGroupBox("Classes — uncheck 'Train' to treat a value as unknown/ignored")
        cl = QVBoxLayout(cls_box)
        btn_row = QHBoxLayout()
        self.scan_btn = QPushButton("Scan label values")
        self.scan_btn.clicked.connect(self._scan_labels)
        btn_row.addWidget(self.scan_btn)
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
        cl.addWidget(self.class_table)
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        cl.addWidget(self.stats_label)

        # ---- convert + upload ----
        go_row = QHBoxLayout()
        self.convert_btn = QPushButton("Convert + Upload")
        self.convert_btn.setObjectName("primary")
        self.convert_btn.clicked.connect(self._convert_upload)
        go_row.addWidget(self.convert_btn)
        go_row.addStretch()

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("log")
        self.log.setPlaceholderText("Conversion + upload progress appears here…")

        bottom = QVBoxLayout()
        bottom.addLayout(go_row)
        bottom.addWidget(self.log, 1)
        rl.addWidget(ui.vsplit(form_box, cls_box, ui.wrap(bottom),
                               sizes=[280, 300, 220]), 1)
        root.addWidget(ui.hsplit(left, right, sizes=[260, 820]), 1)

        self._done_cb = None
        self.worker.output.connect(self._append)
        self.worker.done.connect(self._dispatch_done)
        self.worker.error.connect(self._on_worker_error)
        self.uploader.output.connect(lambda s: self._append(s, newline=False))
        self.uploader.finished.connect(self._on_upload_done)

    def _dispatch_done(self, result):
        cb, self._done_cb = self._done_cb, None
        if cb:
            cb(result)

    # ------------------------------------------------------------- widgets
    def _dir_row(self, slot):
        edit = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(edit)
        btn = QPushButton("Browse…")
        btn.clicked.connect(slot)
        row.addWidget(btn)
        return edit, _wrap(row)

    def _on_label_kind(self):
        companion = self.label_kind.currentIndex() == 1
        self.field_combo.setEnabled(not companion)
        self.truth_row_w.setEnabled(companion)
        self.suffix_row_w.setEnabled(companion)

    # ------------------------------------------------------------- pickers
    def _pick_train(self):
        d = QFileDialog.getExistingDirectory(self, "Training data folder")
        if d:
            self.train_edit.setText(d)
            self._populate_fields(d)
            if not self.name_edit.text():
                self.name_edit.setText(Path(d).name)

    def _pick_val(self):
        d = QFileDialog.getExistingDirectory(self, "Validation data folder")
        if d:
            self.val_edit.setText(d)

    def _pick_truth(self):
        d = QFileDialog.getExistingDirectory(self, "Ground-truth label folder")
        if d:
            self.truth_edit.setText(d)

    def _populate_fields(self, folder: str):
        self.field_combo.clear()
        files = dataset.discover_scenes(folder)
        if not files:
            self._append(f"No supported point-cloud files in {folder}")
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

    # ------------------------------------------------------------- label spec
    def _spec(self) -> LabelSpec:
        if self.label_kind.currentIndex() == 1:
            return LabelSpec(kind="file", truth_dir=self.truth_edit.text().strip(),
                             src_suffix=self.src_suffix.text(), dst_suffix=self.dst_suffix.text())
        return LabelSpec(kind="field", field=self.field_combo.currentText().strip())

    # ------------------------------------------------------------- scan / analyze
    def _scan_labels(self):
        train_dir = self.train_edit.text().strip()
        if not os.path.isdir(train_dir):
            self._append("Choose a training folder first.")
            return
        spec = self._spec()
        self._append("Scanning label values…")
        self.scan_btn.setEnabled(False)

        def job(progress):
            files = dataset.discover_scenes(train_dir)
            progress(f"  sampling {min(len(files), 8)} of {len(files)} scenes")
            return dataset.scan_label_values(files, spec)

        self._done_cb = self._on_scanned
        self.worker.start(job)

    def _on_scanned(self, counts):
        self.scan_btn.setEnabled(True)
        self._label_values = counts
        self.class_table.setRowCount(len(counts))
        for r, (val, cnt) in enumerate(counts.items()):
            chk = QCheckBox()
            chk.setChecked(val != 0)   # ASPRS 0 = unclassified — default to ignored
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
                     f"uncheck any that mean 'unknown', then Convert + Upload.")

    def _analyze(self):
        train_dir = self.train_edit.text().strip()
        if not os.path.isdir(train_dir):
            self._append("Choose a training folder first.")
            return
        self.analyze_btn.setEnabled(False)
        self._append("Analyzing density…")

        def job(progress):
            files = dataset.discover_scenes(train_dir)
            progress(f"  scanning up to {analysis.MAX_FILES_PER_SPLIT} of {len(files)} scenes")
            return analysis.scan_folder(files)

        self._done_cb = self._on_analyzed
        self.worker.start(job)

    def _on_analyzed(self, stats):
        self.analyze_btn.setEnabled(True)
        recs = analysis.recommend(stats)
        self.stats_label.setText(
            f"Density: {stats['mean_pts_per_m2']:.2f} pts/m²   ·   "
            f"mean spacing: {stats['mean_spacing_m']:.2f} m   ·   "
            f"largest scene: {stats['max_scene_points']:,} pts   ·   "
            f"rgb: {'yes' if stats['has_rgb'] else 'no'}   ·   "
            f"intensity: {'yes' if stats['has_intensity'] else 'no'}\n"
            f"Recommended grid (PTv3 warm): {recs['ptv3_warm']['grid']} m   ·   "
            f"tile: {recs['ptv3_warm']['chunk_xy']:.0f} m")
        self._append("Analysis done — recommendations will be saved with the dataset "
                     "and pre-fill the Train page.")

    # ------------------------------------------------------------- convert/upload
    def _classes_from_table(self):
        classes, ignored = [], []
        idx = 0
        for r in range(self.class_table.rowCount()):
            val = int(self.class_table.item(r, 1).text())
            chk = self.class_table.cellWidget(r, 0).findChild(QCheckBox)
            if not chk.isChecked():
                ignored.append(val)
                continue
            classes.append({"index": idx, "source_value": val,
                            "name": self.class_table.item(r, 3).text().strip() or f"class_{val}"})
            idx += 1
        return classes, ignored

    def _convert_upload(self):
        name = self.name_edit.text().strip()
        train_dir = self.train_edit.text().strip()
        val_dir = self.val_edit.text().strip()
        if not (name and os.path.isdir(train_dir) and os.path.isdir(val_dir)):
            self._append("Need a name, a training folder and a validation folder.")
            return
        if self.class_table.rowCount() == 0:
            self._append("Run 'Scan label values' and name your classes first.")
            return
        classes, ignored = self._classes_from_table()
        if not classes:
            self._append("All label values are unchecked — nothing to train on.")
            return
        spec = self._spec()
        self.convert_btn.setEnabled(False)
        self.log.clear()
        self._append(f"Converting '{name}' ({len(classes)} classes, ignored values: {ignored})…")

        def job(progress):
            return dataset.convert_dataset(name, train_dir, val_dir, spec, classes,
                                           ignored, appstate.staging_dir(), progress=progress)

        self._done_cb = self._on_converted
        self.worker.start(job)

    def _on_converted(self, staged: Path):
        self._staged_dir = staged
        name = staged.name
        # The volume is mounted at /datasets in the containers, so the
        # volume-relative remote path is just /<name>.
        self._append(f"\nUploading to volume {modal_cli.DATASETS_VOLUME} -> /{name} …")
        prog, args = modal_cli.volume_put(modal_cli.DATASETS_VOLUME, str(staged), f"/{name}")
        self.uploader.start(prog, args, cwd=self.repo_root)

    def _on_upload_done(self, code: int):
        self.convert_btn.setEnabled(True)
        if code != 0:
            self._append(f"\n✗ Upload failed (exit {code}). Is the Modal CLI installed and "
                         f"authenticated? (modal token new)")
            return
        name = self._staged_dir.name
        appstate.remember_dataset(name, {
            "staged_dir": str(self._staged_dir),
            "meta_path": str(self._staged_dir / "dataset_meta.json"),
            "uploaded": True,
        })
        self._reload_known()
        self._append(f"\n✓ Dataset '{name}' uploaded. Head to the Train page.")

    def _on_worker_error(self, tb: str):
        self._done_cb = None
        self.scan_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.convert_btn.setEnabled(True)
        self._append(f"\n✗ Error:\n{tb}")

    # ------------------------------------------------------------- known list
    def _reload_known(self):
        self.known_list.clear()
        for name in sorted(appstate.known_datasets()):
            self.known_list.addItem(name)

    def _show_known(self):
        items = self.known_list.selectedItems()
        if not items:
            return
        info = appstate.known_datasets().get(items[0].text(), {})
        if info.get("builtin"):
            self.stats_label.setText(f"{items[0].text()} (built-in): {info.get('note', '')}")
            return
        meta_path = info.get("meta_path", "")
        if meta_path and os.path.exists(meta_path):
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            s = meta.get("stats", {})
            self.stats_label.setText(
                f"{meta['name']}: {meta['num_classes']} classes "
                f"({', '.join(meta['class_names'])})   ·   "
                f"{s.get('mean_pts_per_m2', 0):.2f} pts/m²   ·   "
                f"train scenes: {len(meta['splits']['train']['scenes'])}, "
                f"val scenes: {len(meta['splits']['val']['scenes'])}")

    # ------------------------------------------------------------- helpers
    def _append(self, text: str, newline: bool = True):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text + ("\n" if newline else ""))
        self.log.moveCursor(QTextCursor.End)


def _wrap(layout) -> QWidget:
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w
