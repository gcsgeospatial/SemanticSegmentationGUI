"""Inference page: pick trained weights + an input folder, run --mode infer on
Modal, auto-download the predictions and open them in the 3D viewer.

The run is a chain of stages handled by one JobRunner:
  convert (local) -> upload scenes -> [upload local weights] -> modal run
  -> download predictions -> view.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPlainTextEdit, QPushButton, QRadioButton, QVBoxLayout,
                               QWidget)

from .. import appstate, dataset, local_cli, modal_cli, ui
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
        self._dl_dest: Path | None = None
        self._pred_dir: Path | None = None   # where local predictions land (host)

        root = QVBoxLayout(self)
        title = QLabel("Inference")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.sub.setObjectName("pageSub")
        root.addWidget(self.sub)

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
        self.backbone_combo.currentIndexChanged.connect(self._sync_controls)
        # populated at the end of __init__ (reload_backbones needs grid_spin etc.)
        wf.addRow("Model architecture", self.backbone_combo)

        ibox = QGroupBox("Input")
        iform = self.iform = QFormLayout(ibox)
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
        # Local mode: where prediction files land on the host (bind-mounted to the
        # container's predictions dir). Empty = the app staging folder. No upload.
        self.out_edit = QLineEdit()
        self.out_edit.setText(appstate.get("infer_out", ""))
        self.out_edit.setPlaceholderText("default: app staging folder")
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit)
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._pick_out)
        out_row.addWidget(out_btn)
        self.out_row_w = _wrap(out_row)
        iform.addRow("Output folder (predictions)", self.out_row_w)

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
        out_col.addWidget(QLabel("Prediction stats (vs ground truth)"))
        self.stats_box = QPlainTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setMaximumHeight(150)
        self.stats_box.setPlaceholderText(
            "Run 'Compare to ground truth…' on a prediction cloud to compute its "
            "overall accuracy, mIoU and per-class IoU here.")
        out_col.addWidget(self.stats_box)
        out_row.addLayout(out_col, 1)
        btn_col = QVBoxLayout()
        btn_col.addWidget(QLabel("Class palette (legend)"))
        self.palette_combo = QComboBox()
        self.palette_combo.setToolTip(
            "Pick the class names for the legend. Auto = names embedded in the file, "
            "else IEEE. Use 'Configure Palette…' to set the colour of each class.")
        self.palette_combo.currentIndexChanged.connect(self._on_palette_change)
        btn_col.addWidget(self.palette_combo)
        self.configure_palette_btn = QPushButton("Configure Palette…")
        self.configure_palette_btn.setToolTip(
            "Edit the colour shown for each class in the 3D viewer (saved per palette).")
        self.configure_palette_btn.clicked.connect(self._configure_palette)
        btn_col.addWidget(self.configure_palette_btn)
        # Live swatch legend: how each class is coloured in the 3D viewer, for the
        # selected palette (a Configure-Palette override, else the baked default).
        self.legend_label = QLabel()
        self.legend_label.setToolTip("How each class is coloured in the 3D viewer.")
        btn_col.addWidget(self.legend_label)
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
        self.runner.failed.connect(self._on_runner_failed)
        self.parser.run_id.connect(self._on_run_id)

        self.reload_backbones()
        self.reload_runs()
        self._on_source_toggle()
        self.apply_exec_mode(appstate.get_exec_mode() == "local")

    def apply_exec_mode(self, local: bool):
        """Reword copy for the backend + apply the local backbone filter. Inference
        has no other Modal-only controls (just weights + a folder)."""
        self.sub.setText(
            "Label new point clouds with an already-trained model. Pick the weights "
            "(a finished run, or a local .pth), point at a folder of clouds, and run"
            + (" — inference runs locally in Docker." if local else " on Modal."))
        self.iform.setRowVisible(self.out_row_w, local)   # output folder is a local pick
        self.reload_backbones()

    def reload_backbones(self):
        """Populate the model dropdown, honoring the local-mode backbone filter."""
        prev = self.backbone_combo.currentData()
        self.backbone_combo.blockSignals(True)
        self.backbone_combo.clear()
        for key, b in infer_backbones().items():
            if appstate.backbone_enabled(key):
                self.backbone_combo.addItem(b.label, key)
        i = self.backbone_combo.findData(prev)
        if i >= 0:
            self.backbone_combo.setCurrentIndex(i)
        self.backbone_combo.blockSignals(False)
        self._sync_controls()

    # ------------------------------------------------------- class palette
    def reload_palettes(self):
        """Offer 'Auto' + every known dataset whose class names we can resolve
        (built-ins, or a converted dataset with a readable dataset_meta.json)."""
        self.palette_combo.blockSignals(True)
        self.palette_combo.clear()
        self.palette_combo.addItem("Auto (from file / IEEE)", None)
        for nm, info in sorted(appstate.known_datasets().items()):
            if info.get("builtin") or os.path.exists(info.get("meta_path", "") or ""):
                self.palette_combo.addItem(nm, nm)
        want = appstate.get("infer_palette") or None
        i = self.palette_combo.findData(want)
        self.palette_combo.setCurrentIndex(i if i >= 0 else 0)
        self.palette_combo.blockSignals(False)
        self._refresh_legend()

    def _effective_names(self) -> list:
        """Class names labelling the palette: the selected dataset's, else IEEE."""
        from ..palette import IEEE_CLASS_NAMES
        return list(self._selected_class_names() or IEEE_CLASS_NAMES)

    def _palette_key(self) -> str:
        """Override-storage key for the current selection ('__auto__' for Auto)."""
        return self.palette_combo.currentData() or "__auto__"

    def _default_colors(self, n: int) -> list:
        from ..palette import palette_for
        return palette_for(n).tolist()

    def _palette_colors(self) -> list:
        """Effective per-class [r,g,b] for the selected palette: a saved override
        (Configure Palette…) when it matches the class count, else the default
        categorical palette the training scripts bake into predictions."""
        names = self._effective_names()
        ov = appstate.get("palette_overrides", {}).get(self._palette_key())
        if isinstance(ov, list) and len(ov) == len(names):
            return [[int(x) for x in c] for c in ov]
        return self._default_colors(len(names))

    def _refresh_legend(self):
        """Swatch-per-class legend for the selected palette — the exact colours the
        3D viewer paints (a Configure-Palette override, else the baked default)."""
        names = self._effective_names()
        rows = "".join(
            f'<tr><td><span style="font-size:15px; color:#{r:02x}{g:02x}{b:02x}">'
            f'■</span></td><td>&nbsp;{name}</td></tr>'
            for name, (r, g, b) in zip(names, self._palette_colors()))
        self.legend_label.setText(f"<table cellspacing='2'>{rows}</table>")

    def _configure_palette(self):
        """Popup to edit the per-class colour for the selected palette. Saves an
        override (per palette) and refreshes the legend; the viewer uses it too."""
        names = self._effective_names()
        dlg = PaletteDialog(names, self._palette_colors(),
                            self._default_colors(len(names)), self)
        if dlg.exec():
            overrides = dict(appstate.get("palette_overrides", {}))
            overrides[self._palette_key()] = dlg.colors()
            appstate.put("palette_overrides", overrides)
            self._refresh_legend()

    def _on_palette_change(self):
        appstate.put("infer_palette", self.palette_combo.currentData() or "")
        self._refresh_legend()

    def _selected_class_names(self) -> list | None:
        """Class names for the chosen dataset's palette, or None for 'Auto'."""
        name = self.palette_combo.currentData()
        if not name:
            return None
        info = appstate.known_datasets().get(name, {})
        if info.get("builtin"):
            from ..palette import IEEE_CLASS_NAMES
            return IEEE_CLASS_NAMES
        mp = info.get("meta_path", "")
        if mp and os.path.exists(mp):
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    return json.load(f).get("class_names")
            except (OSError, json.JSONDecodeError):
                return None
        return None

    # ------------------------------------------------------------- inputs
    def reload_runs(self):
        self.reload_palettes()
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
        if self.backbone_combo.currentData() is None:   # all backbones unticked
            return
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

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(
            self, "Output folder for predictions",
            self.out_edit.text() or str(appstate.staging_dir()))
        if d:
            self.out_edit.setText(d)

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
        self.stats_box.clear()
        self.log.clear()
        self.run_btn.setEnabled(False)
        self._append(f"[1/4] Converting {input_dir} to canonical scenes (job {self._job_id})…")
        self.converter.start(dataset.convert_infer_job, self._job_id, input_dir,
                             appstate.staging_dir())

    def _on_converted(self, staged: Path):
        self._staged = staged
        if appstate.get_exec_mode() == "local":
            self._start_local_infer()
            return
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
            # Predictions now live on the shared terminal-datasets volume next to the
            # input scenes (_infer/<job_id>/predictions), keyed on the job id we
            # generated — no need to parse a per-backbone run id from the logs.
            self._dl_dest = appstate.runs_dir() / "_infer" / self._job_id
            self._dl_dest.mkdir(parents=True, exist_ok=True)
            self._append(f"[4/4] Downloading predictions -> {self._dl_dest} …")
            self._stage = "download"
            prog, args = modal_cli.volume_get(modal_cli.DATASETS_VOLUME,
                                              f"_infer/{self._job_id}/predictions",
                                              str(self._dl_dest))
            self.runner.start(prog, args, cwd=self.repo_root)
        elif self._stage == "run_local":
            # Local Docker run: predictions were written straight to the chosen
            # output folder on the host (no download stage).
            self.run_btn.setEnabled(True)
            pred_dir = self._pred_dir
            if pred_dir and pred_dir.is_dir():
                preds = [p for p in sorted(pred_dir.iterdir())
                         if p.suffix.lower() in (".ply", ".npz")]
                appstate.put("last_view_dir", str(pred_dir))
                self._append(f"\n✓ Done — {len(preds)} prediction file(s) in {pred_dir}.\n"
                             f"  'View a point cloud…' to open one, or 'Compare to ground "
                             f"truth…' for accuracy + mIoU.")
            else:
                self._append(f"\n✗ No predictions folder at {pred_dir}.")
        elif self._stage == "download":
            self.run_btn.setEnabled(True)
            pred_dir = self._dl_dest / "predictions"
            if pred_dir.is_dir():
                preds = [p for p in sorted(pred_dir.iterdir())
                         if p.suffix.lower() in (".ply", ".npz")]
                appstate.put("last_view_dir", str(pred_dir))   # View dialog opens here
                self._append(f"\n✓ Done — {len(preds)} prediction file(s) in {pred_dir}.\n"
                             f"  'View a point cloud…' to open one, or 'Compare to ground "
                             f"truth…' for accuracy + mIoU.")
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

    def _start_local_infer(self):
        """Local (Docker) inference: the staged scenes are bind-mounted into the
        container at /datasets/_infer/<job>, predictions land straight back in the
        host staging dir — no upload, no download."""
        b = self._backbone()
        # Scenes (and where predictions get written) are self._staged on the host.
        extra_mounts = [(str(self._staged), f"/datasets/_infer/{self._job_id}")]
        # Predictions: a user-picked folder bind-mounted over the container's
        # predictions dir (so they land straight there, no copy), else staging.
        out = self.out_edit.text().strip()
        appstate.put("infer_out", out)
        if out:
            self._pred_dir = Path(out)
            self._pred_dir.mkdir(parents=True, exist_ok=True)
            extra_mounts.append(
                (str(self._pred_dir), f"/datasets/_infer/{self._job_id}/predictions"))
        else:
            self._pred_dir = self._staged / "predictions"
        if self.from_file_radio.isChecked():
            wpath = Path(self.pth_edit.text().strip())
            extra_mounts.append((str(wpath.parent), "/outputs/_local_weights"))
            weights = f"_local_weights/{wpath.name}"
        else:
            # Resolve the run's weights on THIS host. A locally-trained run lands at
            # local_runs/runs/<id> (already under the /outputs mount); a Modal run
            # downloaded via the Runs page lives under runs/<backbone>/<id> — bind-mount
            # whichever exists. Refuse (don't fail in-container) if neither does.
            run_id = self._weights_remote.split("/")[1]   # runs/<id>/final_model.pth
            local_w = appstate.local_runs_dir() / "runs" / run_id / "final_model.pth"
            dl_w = next(iter(appstate.runs_dir().glob(f"*/{run_id}/final_model.pth")), None)
            if local_w.exists():
                weights = self._weights_remote
            elif dl_w is not None:
                extra_mounts.append((str(dl_w.parent), "/outputs/_local_weights"))
                weights = "_local_weights/final_model.pth"
            else:
                self._append(f"✗ Run '{run_id}' has no local weights (looked in "
                             f"{local_w.parent} and {appstate.runs_dir()}). Train it locally, "
                             f"download it on the Runs page, or use the 'Local .pth file' option.")
                self.run_btn.setEnabled(True)
                return
        flags = {
            "mode": "infer",
            "weights": weights,
            "infer-input": self._job_id,
            b.grid_flag: self.grid_spin.value(),
        }
        if b.has_chunk:
            flags["chunk-xy"] = self.chunk_spin.value()
        self._stage = "run_local"
        prog, args = local_cli.run_script(b.script, flags, b, repo_root=self.repo_root,
                                          extra_mounts=extra_mounts)
        self._append(f"[local] Running inference in Docker ({b.label}) …")
        self._append(f"[local] $ {local_cli.preview(prog, args)}\n")
        if not local_cli.have_docker():
            self._append("[local] docker not found on PATH — printed the exact command "
                         "(design-now mode). On a Docker+GPU host the predictions land in "
                         f"{self._pred_dir.as_posix()}.")
            self.run_btn.setEnabled(True)
            return
        ok, msg = local_cli.image_preflight(b)
        if msg:
            self._append(msg)
        if not ok:
            self.run_btn.setEnabled(True)
            return
        self.runner.start(prog, args, cwd=self.repo_root)

    def _on_output(self, text: str):
        self._append(text, newline=False)
        if self._stage in ("run", "run_local"):
            self.parser.feed(text)

    def _on_run_id(self, run_id: str):
        self._run_id = run_id

    def _on_error(self, tb: str):
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Conversion error:\n{tb}")

    def _on_runner_failed(self, err: str):
        # QProcess FailedToStart fires `failed`, not `finished` — re-enable the
        # button so a bad docker/modal exec doesn't wedge the UI.
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Failed to start process: {err}")

    # ------------------------------------------------------------- view
    def _open_viewer(self, *args: str):
        QProcess.startDetached(sys.executable, ["-m", "trainer_gui.viewer", *args], PROJECT_DIR)

    def _view_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Point cloud to view (coloured by class)", appstate.get("last_view_dir", ""),
            "Point clouds (*.ply *.npz *.las *.laz *.txt *.pcd);;All files (*)")
        if not path:
            return
        appstate.put("last_view_dir", str(Path(path).parent))
        names = self._effective_names()
        pal = ";".join(f"{r},{g},{b}" for r, g, b in self._palette_colors())
        self._open_viewer(path, "--class-names", ",".join(names), "--palette", pal)
        self._append(f"Opened viewer for {Path(path).name}  "
                     f"({self.palette_combo.currentText()} palette)")

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
        self._show_stats(pred, gt)
        self._append(f"Comparing {Path(pred).name} to {Path(gt).name} "
                     f"(yellow = predicted class differs from the ground truth).")

    def _show_stats(self, pred: str, gt: str):
        """Compute accuracy + mIoU of the prediction cloud vs ground truth and
        list them in the stats box (alongside opening the error-map viewer)."""
        from .. import viewer
        self.stats_box.setPlainText("Computing accuracy + mIoU …")
        try:
            m = viewer.prediction_metrics(pred, gt)
        except Exception as e:  # noqa: BLE001
            self.stats_box.setPlainText(f"Could not compute stats:\n{e}")
            return
        lines = [m["scene"] or Path(pred).stem,
                 f"accuracy : {m['accuracy']:.4f}",
                 f"mIoU     : {m['miou']:.4f}   (over {len(m['per_class_iou'])} present classes)",
                 f"labeled  : {m['labeled']:,} pts",
                 "per-class IoU:"]
        lines += [f"  class {c}: {iou:.4f}" for c, iou in sorted(m["per_class_iou"].items())]
        self.stats_box.setPlainText("\n".join(lines))

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


class PaletteDialog(QDialog):
    """Per-class colour editor. A swatch button per class opens a colour picker;
    `colors()` returns the chosen [[r,g,b], …] aligned to `names`."""

    def __init__(self, names, colors, defaults, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure palette")
        self._colors = [[int(x) for x in c] for c in colors]
        self._defaults = [[int(x) for x in c] for c in defaults]
        self._btns: list[QPushButton] = []

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Click a swatch to set how that class is coloured in the viewer:"))
        grid = QGridLayout()
        for i, name in enumerate(names):
            grid.addWidget(QLabel(name), i, 0)
            b = QPushButton()
            b.setFixedSize(72, 22)
            b.clicked.connect(lambda _=False, idx=i: self._pick(idx))
            self._btns.append(b)
            grid.addWidget(b, i, 1)
        lay.addLayout(grid)

        foot = QHBoxLayout()
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self._reset)
        foot.addWidget(reset)
        foot.addStretch()
        lay.addLayout(foot)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)
        self._refresh()

    def _refresh(self):
        for b, c in zip(self._btns, self._colors):
            b.setStyleSheet(f"background-color: rgb({c[0]},{c[1]},{c[2]}); border: 1px solid #888;")

    def _pick(self, idx: int):
        col = QColorDialog.getColor(QColor(*self._colors[idx]), self, "Pick a class colour")
        if col.isValid():
            self._colors[idx] = [col.red(), col.green(), col.blue()]
            self._refresh()

    def _reset(self):
        self._colors = [list(c) for c in self._defaults]
        self._refresh()

    def colors(self) -> list:
        return [list(c) for c in self._colors]


def _wrap(layout) -> QWidget:
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w
