"""Pretraining page: PDAL-augment a folder of clouds with HeightAboveGround,
then tile a folder into train-ready tiles for a chosen backbone.

A standalone pre-step before the Datasets tab — writes to a local folder you
then point Datasets at. Both stages run off the GUI thread via FuncWorker.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from .. import backbones, prep, pretrain, ui
from ..jobs import FuncWorker


def _size_spec(b):
    """The tile/grid ParamSpec prep needs for this backbone (sub-grid for
    RandLA-Net, tile size otherwise)."""
    by_flag = {p.flag: p for p in b.params}
    return by_flag.get("sub-grid") or by_flag.get("chunk-xy")


class PretrainPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.worker = FuncWorker(self)
        self._done_cb = None

        root = QVBoxLayout(self)
        title = QLabel("Pretraining")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        sub = QLabel("Prepare raw clouds before converting a dataset: add a "
                     "Height-Above-Ground column with PDAL, and tile a folder into "
                     "train-ready tiles for a specific model. Outputs go to a local "
                     "folder you then point the Datasets tab at.")
        sub.setWordWrap(True)
        sub.setObjectName("pageSub")
        root.addWidget(sub)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("log")
        self.log.setPlaceholderText("Progress appears here…")

        root.addWidget(ui.vsplit(ui.scrollable(self._hag_box()),
                                 ui.scrollable(self._tile_box()),
                                 ui.wrap(self._log_layout()),
                                 sizes=[230, 260, 240]), 1)

        self.worker.output.connect(self._append)
        self.worker.done.connect(self._dispatch_done)
        self.worker.error.connect(self._on_error)

    # --------------------------------------------------------------- Stage A
    def _hag_box(self) -> QWidget:
        box = QGroupBox("Add Height-Above-Ground (PDAL)")
        form = QFormLayout(box)
        self.hag_in, in_row = self._dir_row(self._pick_hag_in)
        form.addRow("Input folder (LAS/LAZ/TXT…)", in_row)
        self.hag_out, out_row = self._dir_row(self._pick_hag_out)
        form.addRow("Output folder", out_row)

        self.skip_ground = QCheckBox("ground already classified (skip SMRF)")
        form.addRow("", self.skip_ground)
        self.hag_filter = QComboBox()
        self.hag_filter.addItems(list(pretrain.HAG_FILTERS))
        form.addRow("HAG filter", self.hag_filter)

        self.hag_btn = QPushButton("Add HAG")
        self.hag_btn.setObjectName("primary")
        self.hag_btn.clicked.connect(self._run_hag)
        row = QHBoxLayout()
        row.addWidget(self.hag_btn)
        row.addStretch()
        form.addRow("", _wrap(row))

        if not pretrain.pdal_available():
            self.hag_btn.setEnabled(False)
            warn = QLabel("PDAL not found — install python-pdal (it ships in the pixi "
                          "env) to enable this step.")
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #b03030;")
            form.addRow("", warn)
        return box

    # --------------------------------------------------------------- Stage B
    def _tile_box(self) -> QWidget:
        box = QGroupBox("Tile for a model")
        form = QFormLayout(box)
        self.tile_in, in_row = self._dir_row(self._pick_tile_in)
        form.addRow("Input folder", in_row)
        self.tile_out, out_row = self._dir_row(self._pick_tile_out)
        form.addRow("Output folder", out_row)

        self.backbone = QComboBox()
        self._tile_backbones = [b for b in backbones.ready_backbones().values()
                                if prep.supports_local_prep(b.key)]
        for b in self._tile_backbones:
            self.backbone.addItem(b.label, b.key)
        self.backbone.currentIndexChanged.connect(self._sync_size)
        form.addRow("Model", self.backbone)

        self.size_label = QLabel("Tile size (m)")
        self.size_spin = QDoubleSpinBox()
        form.addRow(self.size_label, self.size_spin)
        self._sync_size()

        self.tile_btn = QPushButton("Tile dataset")
        self.tile_btn.setObjectName("primary")
        self.tile_btn.clicked.connect(self._run_tile)
        row = QHBoxLayout()
        row.addWidget(self.tile_btn)
        row.addStretch()
        form.addRow("", _wrap(row))
        return box

    def _sync_size(self):
        if not self._tile_backbones:
            return
        b = self._tile_backbones[self.backbone.currentIndex()]
        spec = _size_spec(b)
        self.size_label.setText("Sub-grid (m)" if spec.flag == "sub-grid" else "Tile size (m)")
        self.size_spin.setDecimals(spec.decimals)
        self.size_spin.setRange(spec.lo, spec.hi)
        self.size_spin.setSingleStep(spec.step)
        self.size_spin.setValue(spec.default)

    def _log_layout(self) -> QVBoxLayout:
        lay = QVBoxLayout()
        lay.addWidget(self.log, 1)
        return lay

    # --------------------------------------------------------------- runners
    def _run_hag(self):
        in_dir = self.hag_in.text().strip()
        out_dir = self.hag_out.text().strip()
        if not os.path.isdir(in_dir):
            self._append("Choose an input folder of point clouds first.")
            return
        if not out_dir:
            self._append("Choose an output folder first.")
            return
        skip = self.skip_ground.isChecked()
        flt = self.hag_filter.currentText()
        self._busy(True)
        self.log.clear()
        self._append(f"Adding HAG ({flt}{'' if skip else ', SMRF ground'}) …")

        def job(progress):
            return pretrain.add_hag(in_dir, out_dir, skip_ground=skip,
                                    hag_filter=flt, progress=progress)

        self._done_cb = self._on_hag_done
        self.worker.start(job)

    def _on_hag_done(self, summary):
        self._busy(False)
        # Convenience: prefill Stage B input with the HAG output.
        if not self.tile_in.text().strip():
            self.tile_in.setText(summary["output_dir"])
        self._append(f"\n✓ Done — {summary['n_files']} cloud(s), "
                     f"{summary['total_points']:,} pts. Tile them below or point the "
                     f"Datasets tab at:\n  {summary['output_dir']}")

    def _run_tile(self):
        in_dir = self.tile_in.text().strip()
        out_dir = self.tile_out.text().strip()
        if not os.path.isdir(in_dir):
            self._append("Choose an input folder first.")
            return
        if not out_dir:
            self._append("Choose an output folder first.")
            return
        b = self._tile_backbones[self.backbone.currentIndex()]
        params = {_size_spec(b).flag: float(self.size_spin.value())}
        self._busy(True)
        self.log.clear()
        self._append(f"Tiling for {b.label} ({_size_spec(b).flag}={self.size_spin.value()}) …")

        def job(progress):
            return pretrain.tile_for_model(in_dir, out_dir, b.key, params,
                                           progress=progress)

        self._done_cb = self._on_tile_done
        self.worker.start(job)

    def _on_tile_done(self, prep_dir: Path):
        self._busy(False)
        self._append(f"\n✓ Tiles written to:\n  {prep_dir}")

    # --------------------------------------------------------------- helpers
    def _busy(self, on: bool):
        self.hag_btn.setEnabled(not on and pretrain.pdal_available())
        self.tile_btn.setEnabled(not on)

    def _dispatch_done(self, result):
        cb, self._done_cb = self._done_cb, None
        if cb:
            cb(result)

    def _on_error(self, tb: str):
        self._done_cb = None
        self._busy(False)
        self._append(f"\n✗ Error:\n{tb}")

    def _dir_row(self, slot):
        edit = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(edit)
        btn = QPushButton("Browse…")
        btn.clicked.connect(slot)
        row.addWidget(btn)
        return edit, _wrap(row)

    def _pick(self, edit, caption):
        d = QFileDialog.getExistingDirectory(self, caption)
        if d:
            edit.setText(d)

    def _pick_hag_in(self):
        self._pick(self.hag_in, "Input folder of clouds (LAS/LAZ/TXT/…)")

    def _pick_hag_out(self):
        self._pick(self.hag_out, "Output folder for HAG clouds")

    def _pick_tile_in(self):
        self._pick(self.tile_in, "Folder of clouds to tile")

    def _pick_tile_out(self):
        self._pick(self.tile_out, "Output folder for tiles")

    def _append(self, text: str, newline: bool = True):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text + ("\n" if newline else ""))
        self.log.moveCursor(QTextCursor.End)


def _wrap(layout) -> QWidget:
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w
