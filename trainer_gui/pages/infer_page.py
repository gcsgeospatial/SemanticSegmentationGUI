"""Inference page: pick trained weights + an input folder, run --mode infer on
Modal, auto-download the predictions and open them in the 3D viewer.

The run is a chain of stages handled by one JobRunner:
  convert (local) -> upload scenes -> [upload local weights] -> modal run
  -> download predictions -> view.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QPlainTextEdit,
                               QPushButton, QRadioButton, QVBoxLayout, QWidget)

from .. import appstate, dataset, modal_cli, ui
from ..backbones import BACKBONES, infer_backbones
from ..jobs import FuncWorker, JobRunner, LogParser

PROJECT_DIR = str(Path(__file__).resolve().parents[2])


class InferPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.converter = FuncWorker(self)
        self.runner = JobRunner(self)
        self.parser = LogParser(self)
        self._stage = ""
        self._job_id = ""
        self._staged: Path | None = None
        self._weights_remote = ""
        self._run_id = ""

        root = QVBoxLayout(self)
        title = QLabel("Inference")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        sub = QLabel("Label new point clouds with an already-trained model. Pick the weights "
                     "(a finished run, or a local .pth), point at a folder of clouds, and run.")
        sub.setWordWrap(True)
        sub.setObjectName("pageSub")
        root.addWidget(sub)

        wbox = QGroupBox("Weights")
        wf = QFormLayout(wbox)
        radio_row = QHBoxLayout()
        self.from_run_radio = QRadioButton("From a training run")
        self.from_run_radio.setChecked(True)
        self.from_file_radio = QRadioButton("Local .pth file")
        self.from_run_radio.toggled.connect(self._on_source_toggle)
        radio_row.addWidget(self.from_run_radio)
        radio_row.addWidget(self.from_file_radio)
        radio_row.addStretch()
        wf.addRow("Source", _wrap(radio_row))
        self.run_combo = QComboBox()
        self.run_combo.setEditable(True)   # run ids can also be typed/pasted
        self.run_combo.currentIndexChanged.connect(self._on_run_pick)
        wf.addRow("Run", self.run_combo)
        self.pth_edit = QLineEdit()
        pth_row = QHBoxLayout()
        pth_row.addWidget(self.pth_edit)
        pth_btn = QPushButton("Browse…")
        pth_btn.clicked.connect(self._pick_pth)
        pth_row.addWidget(pth_btn)
        self.pth_row_w = _wrap(pth_row)
        wf.addRow("File", self.pth_row_w)
        self.backbone_combo = QComboBox()
        for key, b in infer_backbones().items():
            self.backbone_combo.addItem(b.label, key)
        self.backbone_combo.currentIndexChanged.connect(self._sync_controls)
        wf.addRow("Model architecture", self.backbone_combo)

        ibox = QGroupBox("Input")
        iform = QFormLayout(ibox)
        self.input_edit = QLineEdit()
        in_row = QHBoxLayout()
        in_row.addWidget(self.input_edit)
        in_btn = QPushButton("Browse…")
        in_btn.clicked.connect(self._pick_input)
        in_row.addWidget(in_btn)
        iform.addRow("Folder of point clouds", _wrap(in_row))
        self.grid_spin = QDoubleSpinBox()
        self.grid_spin.setRange(0.02, 5.0)
        self.grid_spin.setSingleStep(0.05)
        self.grid_spin.setDecimals(2)
        self.grid_spin.setValue(0.30)
        iform.addRow("Grid size (m) — match the training run", self.grid_spin)
        self.chunk_spin = QDoubleSpinBox()
        self.chunk_spin.setRange(10.0, 200.0)
        self.chunk_spin.setSingleStep(5.0)
        self.chunk_spin.setDecimals(0)
        self.chunk_spin.setValue(50.0)
        iform.addRow("Tile size (m)", self.chunk_spin)

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

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("log")
        self.log.setPlaceholderText("Conversion, upload and Modal logs appear here…")

        out_row = QHBoxLayout()
        out_col = QVBoxLayout()
        out_col.addWidget(QLabel("Predictions from the last run  (double-click to view)"))
        self.pred_list = QListWidget()
        self.pred_list.itemDoubleClicked.connect(self._view_item)
        out_col.addWidget(self.pred_list, 1)
        out_row.addLayout(out_col, 1)
        btn_col = QVBoxLayout()
        self.view_btn = QPushButton("View a point cloud…")
        self.view_btn.clicked.connect(self._view_file)
        self.compare_btn = QPushButton("Compare to ground truth…")
        self.compare_btn.clicked.connect(self._compare_gt)
        self.export_btn = QPushButton("Export comparison PLY…")
        self.export_btn.clicked.connect(self._export_gt)
        btn_col.addWidget(self.view_btn)
        btn_col.addWidget(self.compare_btn)
        btn_col.addWidget(self.export_btn)
        btn_col.addStretch()
        out_row.addLayout(btn_col)

        root.addWidget(ui.vsplit(ui.scrollable(ui.wrap(forms_col)), self.log,
                                 ui.wrap(out_row), sizes=[330, 300, 160]), 1)

        self.converter.output.connect(self._append)
        self.converter.done.connect(self._on_converted)
        self.converter.error.connect(self._on_error)
        self.runner.output.connect(self._on_output)
        self.runner.finished.connect(self._on_stage_done)
        self.parser.run_id.connect(self._on_run_id)

        self.reload_runs()
        self._on_source_toggle()
        self._sync_controls()

    # ------------------------------------------------------------- inputs
    def reload_runs(self):
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        seen = set()
        for h in reversed(appstate.get("run_history", [])):
            if h["run_id"] not in seen:
                seen.add(h["run_id"])
                self.run_combo.addItem(f"{h['run_id']}  ({h['backbone']})", h)
        # also offer runs already downloaded via the Runs page
        for bdir in appstate.runs_dir().iterdir() if appstate.runs_dir().exists() else []:
            if bdir.is_dir():
                for rdir in bdir.iterdir():
                    if rdir.name not in seen and (rdir / "run_config.json").exists():
                        seen.add(rdir.name)
                        self.run_combo.addItem(f"{rdir.name}  ({bdir.name})",
                                               {"run_id": rdir.name, "backbone": bdir.name})
        self.run_combo.blockSignals(False)
        self._on_run_pick()

    def _on_run_pick(self):
        h = self.run_combo.currentData()
        if isinstance(h, dict) and h.get("backbone") in BACKBONES:
            i = self.backbone_combo.findData(h["backbone"])
            if i >= 0:
                self.backbone_combo.setCurrentIndex(i)

    def _on_source_toggle(self):
        from_run = self.from_run_radio.isChecked()
        self.run_combo.setEnabled(from_run)
        self.pth_row_w.setEnabled(not from_run)

    def _sync_controls(self):
        """Auto-fill grid + tile size from the selected script's own defaults
        (RandLA uses sub-grid; KPConvX a 2 m grid / 100 m tiles), and disable
        tile size for RandLA (it samples spheres, so tiling is meaningless)."""
        b = self._backbone()
        gp = next((p for p in b.params if p.recommend_key == "grid"), None)
        if gp:
            self.grid_spin.setRange(gp.lo, gp.hi)
            self.grid_spin.setDecimals(gp.decimals)
            self.grid_spin.setSingleStep(gp.step)
            self.grid_spin.setValue(gp.default)
        cp = next((p for p in b.params if p.flag == "chunk-xy"), None)
        if cp:
            self.chunk_spin.setValue(cp.default)
        self.chunk_spin.setEnabled(b.has_chunk)

    def _pick_pth(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose model weights", "",
                                              "PyTorch checkpoints (*.pth *.pt)")
        if path:
            self.pth_edit.setText(path)

    def _pick_input(self):
        d = QFileDialog.getExistingDirectory(self, "Folder of point clouds to label")
        if d:
            self.input_edit.setText(d)

    def _backbone(self):
        return BACKBONES[self.backbone_combo.currentData()]

    # ------------------------------------------------------------- run chain
    def _run(self):
        input_dir = self.input_edit.text().strip()
        if not os.path.isdir(input_dir):
            self._append("Choose an input folder first.")
            return
        if self.from_run_radio.isChecked():
            h = self.run_combo.currentData()
            run_id = h["run_id"] if isinstance(h, dict) else self.run_combo.currentText().split()[0]
            if not run_id:
                self._append("Pick or type a run id.")
                return
            bkey = h.get("backbone") if isinstance(h, dict) else None
            if bkey in BACKBONES and not BACKBONES[bkey].folder_infer:
                self._append(f"✗ {BACKBONES[bkey].label} doesn't support folder inference "
                             f"(its script has no --infer-input mode). Open that run's "
                             f"predictions on the Runs page instead.")
                return
            self._weights_remote = f"runs/{run_id}/final_model.pth"
        else:
            if not os.path.isfile(self.pth_edit.text().strip()):
                self._append("Choose a .pth file.")
                return
            self._weights_remote = f"uploads/{Path(self.pth_edit.text()).name}"

        self._job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = ""
        self.pred_list.clear()
        self.log.clear()
        self.run_btn.setEnabled(False)
        self._append(f"[1/4] Converting {input_dir} to canonical scenes (job {self._job_id})…")
        self.converter.start(dataset.convert_infer_job, self._job_id, input_dir,
                             appstate.staging_dir())

    def _on_converted(self, staged: Path):
        self._staged = staged
        self._append(f"[2/4] Uploading scenes -> {modal_cli.DATASETS_VOLUME}:/_infer/{self._job_id} …")
        self._stage = "upload_scenes"
        prog, args = modal_cli.volume_put(modal_cli.DATASETS_VOLUME, str(staged),
                                          f"/_infer/{self._job_id}")
        self.runner.start(prog, args, cwd=self.repo_root)

    def _on_stage_done(self, code: int):
        if code != 0:
            self._append(f"\n✗ Stage '{self._stage}' failed (exit {code}).")
            self.run_btn.setEnabled(True)
            return
        b = self._backbone()
        if self._stage == "upload_scenes":
            if self.from_file_radio.isChecked():
                local = self.pth_edit.text().strip()
                self._append(f"[2b] Uploading weights -> {b.outputs_volume}:/{self._weights_remote} …")
                self._stage = "upload_weights"
                prog, args = modal_cli.volume_put(b.outputs_volume, local,
                                                  f"/{self._weights_remote}")
                self.runner.start(prog, args, cwd=self.repo_root)
                return
            self._start_modal_run()
        elif self._stage == "upload_weights":
            self._start_modal_run()
        elif self._stage == "run":
            if not self._run_id:
                self._append("\n✗ Could not detect the run id from the logs — "
                             "check the Runs page for a *_infer run and download it there.")
                self.run_btn.setEnabled(True)
                return
            dest = appstate.runs_dir() / b.key
            dest.mkdir(parents=True, exist_ok=True)
            self._append(f"[4/4] Downloading predictions runs/{self._run_id} -> {dest} …")
            self._stage = "download"
            prog, args = modal_cli.volume_get(b.outputs_volume, f"runs/{self._run_id}", str(dest))
            self.runner.start(prog, args, cwd=self.repo_root)
        elif self._stage == "download":
            self.run_btn.setEnabled(True)
            pred_dir = appstate.runs_dir() / b.key / self._run_id / "predictions"
            if pred_dir.is_dir():
                for p in sorted(pred_dir.iterdir()):
                    if p.suffix.lower() in (".ply", ".npz"):
                        self.pred_list.addItem(str(p))
                self._append(f"\n✓ Done — {self.pred_list.count()} prediction files. "
                             f"Double-click one to view it.")
            else:
                self._append(f"\n✗ No predictions folder at {pred_dir}.")

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
        self._append(f"[3/4] Running inference on Modal ({b.label}) …")
        self._stage = "run"
        prog, args = modal_cli.run_script(b.script, flags, detach=False)
        self._append(f"$ modal {' '.join(args)}\n")
        self.runner.start(prog, args, cwd=self.repo_root)

    def _on_output(self, text: str):
        self._append(text, newline=False)
        if self._stage == "run":
            self.parser.feed(text)

    def _on_run_id(self, run_id: str):
        self._run_id = run_id

    def _on_error(self, tb: str):
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Conversion error:\n{tb}")

    # ------------------------------------------------------------- view
    def _open_viewer(self, *args: str):
        QProcess.startDetached(sys.executable, ["-m", "trainer_gui.viewer", *args], PROJECT_DIR)

    def _view_item(self, item):
        self._open_viewer(item.text())
        self._append(f"Opened viewer for {Path(item.text()).name}")

    def _view_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Point cloud to view (coloured by class)", appstate.get("last_view_dir", ""),
            "Point clouds (*.ply *.npz *.las *.laz *.txt *.pcd);;All files (*)")
        if not path:
            return
        appstate.put("last_view_dir", str(Path(path).parent))
        self._open_viewer(path)
        self._append(f"Opened viewer for {Path(path).name}")

    def _pick_pred_gt(self):
        """Prompt for a prediction cloud then its ground-truth labels.
        Returns (pred, gt) or None if cancelled."""
        pred, _ = QFileDialog.getOpenFileName(
            self, "Prediction cloud to compare", appstate.get("last_view_dir", ""),
            "Prediction cloud (*.ply *.npz);;All files (*)")
        if not pred:
            return None
        appstate.put("last_view_dir", str(Path(pred).parent))
        gt, _ = QFileDialog.getOpenFileName(
            self, "Ground truth for this scene (.ply, .npz, or <scene>_CLS.txt)",
            appstate.get("ieee_truth_file", ""),
            "Ground truth (*.ply *.npz *.txt *.csv);;All files (*)")
        if not gt:
            return None
        appstate.put("ieee_truth_file", gt)
        return pred, gt

    def _compare_gt(self):
        """Prompt for a prediction + ground truth and open the error map
        (yellow where the predicted class differs)."""
        picked = self._pick_pred_gt()
        if not picked:
            return
        pred, gt = picked
        self._open_viewer(pred, "--gt", gt)
        self._append(f"Comparing {Path(pred).name} to {Path(gt).name} "
                     f"(yellow = predicted class differs from the ground truth).")

    def _export_gt(self):
        """Prompt for a prediction + ground truth and write the error-map cloud
        to a .ply (yellow = wrong, intensity-shaded grey = correct)."""
        picked = self._pick_pred_gt()
        if not picked:
            return
        pred, gt = picked
        default = str(Path(pred).with_name(Path(pred).stem + "_vs_gt.ply"))
        out, _ = QFileDialog.getSaveFileName(
            self, "Save comparison PLY", default, "PLY (*.ply)")
        if not out:
            return
        self._open_viewer(pred, "--gt", gt, "--save", out)
        self._append(f"Exporting comparison of {Path(pred).name} vs {Path(gt).name} "
                     f"-> {out}")

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
