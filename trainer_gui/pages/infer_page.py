"""Inference page: pick weights + input folder, run --mode infer, view predictions.

Stages (one JobRunner):
  Modal: convert -> upload scenes -> [upload weights] -> run -> download -> view.
  Local: convert -> docker run (scenes mounted, predictions to host) -> view.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                               QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from .. import appstate, dataset, local_cli, modal_cli, ui
from ..backbones import BACKBONES, infer_backbones
from ..jobs import FuncWorker, JobRunner, LogParser

PROJECT_DIR = str(Path(__file__).resolve().parents[2])

# Generic 'class i' count when no run/dataset supplies class names.
_FALLBACK_NUM_CLASSES = 5


class InferPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.converter = FuncWorker(self)
        self.preflight = FuncWorker(self)
        self.exporter = FuncWorker(self)   # npz -> chosen prediction format (host-side)
        self.runner = JobRunner(self)
        self.parser = LogParser(self)
        self._stage = ""
        self._job_id = ""
        self._staged: Path | None = None
        self._weights_remote = ""
        self._run_id = ""
        self._dl_dest: Path | None = None
        self._pred_dir: Path | None = None   # where local predictions land (host)
        self._manifest: dict | None = None         # the picked run.json (local runs)
        self._manifest_path: Path | None = None
        self._local_weights: Path | None = None    # weights = the run.json's sibling
        self._dg: dict = {}                         # DG settings baked into the weights (run.json["dg"])
        self._run_class_names: list | None = None   # the loaded run's own classes (run.json)

        root = QVBoxLayout(self)
        title = QLabel("Inference")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.sub.setObjectName("pageSub")
        root.addWidget(self.sub)

        wbox = QGroupBox("Weights")
        wf = self.wf = QFormLayout(wbox)
        # Fields fill width so Browse… buttons aren't clipped.
        wf.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        radio_row = QHBoxLayout()
        self.from_run_radio = QRadioButton("Training run")
        self.from_run_radio.setChecked(True)
        self.from_file_radio = QRadioButton("Local .pth file")
        self.from_run_radio.toggled.connect(self._on_source_toggle)
        radio_row.addWidget(self.from_run_radio)
        radio_row.addWidget(self.from_file_radio)
        radio_row.addStretch()
        wf.addRow("Source", _wrap(radio_row))
        # LOCAL: pick run.json — arch, grid, tile, intensity, HAG, weights path come from it.
        self.runjson_edit = QLineEdit()
        self.runjson_edit.setPlaceholderText("…/local_runs/runs/<id>/run.json")
        # Load on Enter/focus-out; any edit drops the stale load.
        self.runjson_edit.editingFinished.connect(self._load_run_manifest)
        self.runjson_edit.textChanged.connect(self._invalidate_manifest)
        rj_row = QHBoxLayout()
        rj_row.addWidget(self.runjson_edit, 1)   # edit absorbs slack; button keeps its size
        rj_btn = QPushButton("Browse…")
        rj_btn.clicked.connect(self._pick_runjson)
        rj_row.addWidget(rj_btn)
        self.runjson_row_w = _wrap(rj_row)
        wf.addRow("Run file (run.json)", self.runjson_row_w)
        # Weights default to the .pth named in run.json; can point anywhere.
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("default: the .pth named in run.json, beside it")
        w_row = QHBoxLayout()
        w_row.addWidget(self.weights_edit, 1)
        w_btn = QPushButton("Browse…")
        w_btn.clicked.connect(self._pick_weights)
        w_row.addWidget(w_btn)
        self.weights_row_w = _wrap(w_row)
        wf.addRow("Weights file", self.weights_row_w)
        # MODAL: pick/paste a run id (weights live on the cloud outputs volume).
        self.run_combo = QComboBox()
        self.run_combo.setEditable(True)   # run ids can also be typed/pasted
        self.run_combo.currentIndexChanged.connect(self._on_run_pick)
        wf.addRow("Run", self.run_combo)
        self.pth_edit = QLineEdit()
        pth_row = QHBoxLayout()
        pth_row.addWidget(self.pth_edit, 1)
        pth_btn = QPushButton("Browse…")
        pth_btn.clicked.connect(self._pick_pth)
        pth_row.addWidget(pth_btn)
        self.pth_row_w = _wrap(pth_row)
        wf.addRow("File", self.pth_row_w)
        self.backbone_combo = QComboBox()
        self.backbone_combo.currentIndexChanged.connect(self._sync_controls)
        # populated at the end of __init__ (reload_backbones needs grid_spin etc.)
        wf.addRow("Architecture", self.backbone_combo)

        ibox = QGroupBox("Input")
        iform = self.iform = QFormLayout(ibox)
        iform.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.input_edit = QLineEdit()
        in_row = QHBoxLayout()
        in_row.addWidget(self.input_edit, 1)   # edit absorbs slack; both buttons keep their size
        fold_btn = QPushButton("Folder…")
        fold_btn.clicked.connect(self._pick_input)
        file_btn = QPushButton("File…")
        file_btn.clicked.connect(self._pick_input_file)
        in_row.addWidget(fold_btn)
        in_row.addWidget(file_btn)
        iform.addRow("Point clouds (folder or file)", _wrap(in_row))
        self.grid_spin = QDoubleSpinBox()
        self.grid_spin.setRange(0.02, 1_000_000.0)
        self.grid_spin.setSingleStep(0.05)
        self.grid_spin.setDecimals(2)
        self.grid_spin.setValue(0.30)
        iform.addRow("Grid size (m) - from the run", self.grid_spin)
        # Intensity is p95-normalized end-to-end; nothing to match here.
        self.chunk_spin = QDoubleSpinBox()
        self.chunk_spin.setRange(10.0, 1_000_000.0)
        self.chunk_spin.setSingleStep(5.0)
        self.chunk_spin.setDecimals(0)
        self.chunk_spin.setValue(50.0)
        iform.addRow("Tile size (m)", self.chunk_spin)
        # TODO(not ready): inference-time density-adapt UI (AdaBN / density TTA)
        # hidden until reviewed; _infer_dg_env sends no AdaBN/TTA env for now.
        # Density generalization: inference-time, label-free, no retrain. Any model.
        # self.dg_adabn_chk = QCheckBox("AdaBN - re-fit norm stats to this cloud (KPConvX / RandLA)")
        # self.dg_adabn_chk.setToolTip(
        #     "Re-fit BatchNorm stats to the target tiles before predicting. "
        #     "Label-free, no retrain. No-op for PTv3.")
        # iform.addRow("Density adapt", self.dg_adabn_chk)
        # self.dg_tta_chk = QCheckBox("Density TTA - average over")
        # self.dg_tta_chk.setToolTip(
        #     "Average softmax over several density/scale resamplings of each tile. "
        #     "Label-free, no retrain. More views = slower.")
        # self.dg_tta_spin = QSpinBox()
        # self.dg_tta_spin.setRange(1, 9)
        # self.dg_tta_spin.setValue(3)
        # tta_row = QHBoxLayout()
        # tta_row.addWidget(self.dg_tta_chk)
        # tta_row.addWidget(self.dg_tta_spin)
        # tta_row.addWidget(QLabel("extra views"))
        # tta_row.addStretch(1)
        # iform.addRow("", _wrap(tta_row))
        # Prediction output folder; empty falls back to Downloads.
        self.out_edit = QLineEdit()
        self.out_edit.setText(appstate.get("infer_out", ""))
        self.out_edit.setPlaceholderText(
            f"default: {appstate.default_download_dir().as_posix()}")
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit, 1)
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._pick_out)
        out_row.addWidget(out_btn)
        self.out_row_w = _wrap(out_row)
        iform.addRow("Output folder (predictions)", self.out_row_w)
        # Prediction file format — every option is xyz + classification only
        # (no RGB: colour/palette is a viewer concern, not the deliverable's).
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
        self.log.setPlaceholderText("Conversion and run logs appear here…")

        # Action bar + one-line legend; comparison metrics print to the log.
        self.view_btn = QPushButton("View a point cloud…")
        self.view_btn.clicked.connect(self._view_file)
        self.compare_btn = QPushButton("Compare to ground truth…")
        self.compare_btn.clicked.connect(self._compare_gt)
        self.export_btn = QPushButton("Export comparison PLY…")
        self.export_btn.clicked.connect(self._export_gt)
        self.palette_btn = QPushButton("Class colours & names…")
        self.palette_btn.setToolTip("Pick the name source (run / dataset / Auto), "
                                    "rename classes, and set colours.")
        self.palette_btn.clicked.connect(self._configure_palette)
        actions = QHBoxLayout()
        actions.addWidget(self.view_btn)
        actions.addWidget(self.compare_btn)
        actions.addWidget(self.export_btn)
        actions.addStretch(1)
        actions.addWidget(self.palette_btn)
        # Swatch legend — the colours the viewer paints.
        self.legend_label = QLabel()
        self.legend_label.setWordWrap(True)
        self.legend_label.setToolTip("Class colours used by the viewer.")
        out_box = QVBoxLayout()
        out_box.addLayout(actions)
        out_box.addWidget(self.legend_label)

        root.addWidget(ui.vsplit(ui.scrollable(ui.wrap(forms_col)), self.log,
                                 ui.wrap(out_box), sizes=[340, 340, 84]), 1)

        self.converter.output.connect(self._append)
        self.converter.done.connect(self._on_converted)
        self.converter.error.connect(self._on_error)
        self.preflight.output.connect(self._append)
        self.preflight.done.connect(self._on_preflight)
        self.preflight.error.connect(self._on_preflight_error)
        self.exporter.output.connect(self._append)
        self.exporter.done.connect(self._on_exported)
        self.exporter.error.connect(self._on_export_error)
        self.runner.output.connect(self._on_output)
        self.runner.finished.connect(self._on_stage_done)
        self.runner.failed.connect(self._on_runner_failed)
        self.parser.run_id.connect(self._on_run_id)

        self.reload_backbones()
        self.reload_runs()
        self._on_source_toggle()
        self.apply_exec_mode(appstate.get_exec_mode() == "local")

    def apply_exec_mode(self, local: bool):
        """Reword copy for the backend, apply the local backbone filter."""
        self.sub.setText(
            "Label point clouds with a trained model. "
            + ("Pick a run.json (or a local .pth), a folder of clouds, and run in Docker."
               if local else
               "Pick a run (or a local .pth), a folder of clouds, and run on Modal."))
        # Output folder shown in both modes (Modal: download target).
        self.iform.setRowVisible(self.out_row_w, True)
        self.wf.setRowVisible(self.runjson_row_w, local)  # run.json picker = local only
        self.wf.setRowVisible(self.weights_row_w, local)  # weights override = local only
        self.wf.setRowVisible(self.run_combo, not local)  # run-id combo = modal only
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

    # ------------------------------------------------- class palette & names
    # Per-class table (source + name + colour). appstate keys, all keyed by source:
    #   infer_palette          -> chosen name source key
    #   palette_name_overrides -> {source_key: [name, …]}
    #   palette_overrides      -> {source_key: [[r,g,b], …]}
    def reload_palettes(self):
        """Refresh the legend (called from reload_runs / apply_exec_mode)."""
        self._refresh_legend()

    def _set_run_classes(self, names):
        """Adopt the run's class names and select that source."""
        self._run_class_names = list(names) if names else None
        if self._run_class_names:
            appstate.put("infer_palette", "__run__")
        self._refresh_legend()

    @staticmethod
    def _names_from_manifest(m: dict) -> list | None:
        """Class names from a manifest: class_names, else 'class 0..n-1', else None."""
        names = m.get("class_names")
        if names:
            return list(names)
        n = m.get("num_classes")
        return [f"class {i}" for i in range(int(n))] if n else None

    def _source_options(self) -> list:
        """(label, key) name-source choices: run, Auto, then datasets with class names."""
        opts = []
        if self._run_class_names:
            opts.append((f"Loaded run ({len(self._run_class_names)} classes)", "__run__"))
        opts.append(("Auto (names in file, else class i)", "__auto__"))
        for nm, info in sorted(appstate.known_datasets().items()):
            if os.path.exists(info.get("meta_path", "") or ""):
                opts.append((nm, nm))
        return opts

    def _current_source_key(self) -> str:
        k = appstate.get("infer_palette") or ""
        if k == "__run__" and not self._run_class_names:
            k = ""
        return k or ("__run__" if self._run_class_names else "__auto__")

    def _names_for_key(self, key: str) -> list:
        """Base class names for a source key; generic 'class i' if none."""
        from ..palette import generic_names
        if key == "__run__":
            return list(self._run_class_names or generic_names(_FALLBACK_NUM_CLASSES))
        if key in (None, "", "__auto__"):
            return generic_names(_FALLBACK_NUM_CLASSES)
        info = appstate.known_datasets().get(key, {})
        mp = info.get("meta_path", "")
        if mp and os.path.exists(mp):
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("class_names"):
                    return list(meta["class_names"])
                if meta.get("num_classes"):
                    return generic_names(int(meta["num_classes"]))
            except (OSError, json.JSONDecodeError):
                pass
        return generic_names(_FALLBACK_NUM_CLASSES)

    def _apply_name_overrides(self, key: str, base: list) -> list:
        ov = appstate.get("palette_name_overrides", {}).get(key)
        if isinstance(ov, list):
            return [(ov[i] if i < len(ov) and ov[i] else base[i]) for i in range(len(base))]
        return list(base)

    def _effective_names(self) -> list:
        key = self._current_source_key()
        return self._apply_name_overrides(key, self._names_for_key(key))

    def _default_colors(self, n: int) -> list:
        from ..palette import palette_for
        return palette_for(n).tolist()

    def _colors_for(self, key: str, names: list) -> list:
        ov = appstate.get("palette_overrides", {}).get(key)
        if isinstance(ov, list) and len(ov) == len(names):
            return [[int(x) for x in c] for c in ov]
        return self._default_colors(len(names))

    def _palette_colors(self) -> list:
        return self._colors_for(self._current_source_key(), self._effective_names())

    def _refresh_legend(self):
        """One-line swatch legend."""
        parts = "".join(
            f'<span style="font-size:14px;color:#{r:02x}{g:02x}{b:02x}">■</span>'
            f'<span>&nbsp;{name}</span>&nbsp;&nbsp;&nbsp;'
            for name, (r, g, b) in zip(self._effective_names(), self._palette_colors()))
        self.legend_label.setText(parts)

    def _configure_palette(self):
        """Open the class menu; persist source, renames and colours."""
        dlg = ClassPaletteDialog(self, self)
        if not dlg.exec():
            return
        key = dlg.source_key()
        appstate.put("infer_palette", "" if key == "__auto__" else key)
        base = self._names_for_key(key)
        edited = dlg.names()
        names_ov = dict(appstate.get("palette_name_overrides", {}))
        names_ov[key] = [edited[i] if i < len(edited) and edited[i] != base[i] else ""
                         for i in range(len(base))]
        appstate.put("palette_name_overrides", names_ov)
        cols_ov = dict(appstate.get("palette_overrides", {}))
        cols_ov[key] = dlg.colors()
        appstate.put("palette_overrides", cols_ov)
        self._refresh_legend()

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
                    if rdir.name not in seen and any(
                            (rdir / fn).exists() for fn in ("run.json", "run_config.json")):
                        seen.add(rdir.name)
                        self.run_combo.addItem(f"{rdir.name}  ({bdir.name})",
                                               {"run_id": rdir.name, "backbone": bdir.name})
        self.run_combo.blockSignals(False)
        self._on_run_pick()

    def _on_run_pick(self):
        """Sync architecture from the picked run; adopt its classes if downloaded."""
        h = self.run_combo.currentData()
        if isinstance(h, dict) and h.get("backbone") in BACKBONES:
            i = self.backbone_combo.findData(h["backbone"])
            if i >= 0:
                self.backbone_combo.setCurrentIndex(i)
        if appstate.get_exec_mode() != "local":
            self._set_run_classes(self._run_pick_class_names(h))

    def _run_pick_class_names(self, h) -> list | None:
        """Class names for a Modal run if downloaded locally, else None."""
        if not isinstance(h, dict):
            return None
        rdir = appstate.runs_dir() / str(h.get("backbone", "")) / str(h.get("run_id", ""))
        for fn in ("run.json", "run_config.json"):
            p = rdir / fn
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as f:
                        return self._names_from_manifest(json.load(f))
                except (OSError, json.JSONDecodeError):
                    pass
        return None

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
        self._dg = {}
        self._apply_manifest_lock(False)   # run-derived inputs editable again

    def _infer_dg_env(self) -> dict:
        """DG_* env for inference. logdk recovered from run.json (it changed the input
        width, so must be recomputed or the load fails). AdaBN/TTA toggles are hidden
        for now (not ready) — see the commented-out controls in __init__."""
        env: dict[str, str] = {}
        if self._dg.get("logdk"):
            env["DG_LOGDK_FEAT"] = "1"
            env["DG_LOGDK_K"] = str(int(self._dg.get("logdk_k", 8)))
        # TODO(not ready): density-adapt UI hidden; no AdaBN/TTA env for now.
        # if self.dg_adabn_chk.isChecked():
        #     env["DG_INFER_ADABN"] = "1"
        # if self.dg_tta_chk.isChecked():
        #     env["DG_INFER_TTA"] = str(self.dg_tta_spin.value())
        if env:
            self._append("[dg] inference: " + " ".join(f"{k}={v}" for k, v in sorted(env.items())))
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
        bkey = m.get("backbone")
        i = self.backbone_combo.findData(bkey)
        if i < 0:   # the run's model is hidden/unknown — don't keep the wrong one
            self._append(f"✗ Model '{bkey}' isn't available here. Enable it on the "
                         f"Train page, then reload this run.json.")
            return
        self._manifest, self._manifest_path = m, p
        self.backbone_combo.setCurrentIndex(i)       # fires _sync_controls (sets defaults)
        if m.get("grid") is not None:                # then the manifest overrides them
            self.grid_spin.setValue(float(m["grid"]))
        if m.get("chunk_xy") is not None and self.chunk_spin.isEnabled():
            self.chunk_spin.setValue(float(m["chunk_xy"]))
        self._local_weights = p.parent / m.get("weights", "final_model.pth")
        self.weights_edit.setText(str(self._local_weights))   # default
        # logdk changes input width, so it MUST be re-fed at inference (DG_* env below).
        self._dg = m.get("dg") or {}
        # Label legend + viewer with the model's own classes.
        self._set_run_classes(self._names_from_manifest(m))
        self._apply_manifest_lock(True)   # arch/grid/tile come from the run — grey them out
        ok = "✓" if self._local_weights.is_file() else "✗ weights missing -"
        self._append(f"Loaded {p.name}: {bkey}, grid={m.get('grid')}, "
                     f"chunk={m.get('chunk_xy')}, intensity={m.get('intensity_norm')}. "
                     f"{ok} {self._local_weights}")
        if self._dg.get("logdk"):
            self._append(f"[dg] trained with the log-d_k density channel "
                         f"(k={self._dg.get('logdk_k', 8)}); recomputed at inference.")

    def _on_source_toggle(self):
        from_run = self.from_run_radio.isChecked()
        self.run_combo.setEnabled(from_run)
        self.runjson_row_w.setEnabled(from_run)
        self.weights_row_w.setEnabled(from_run)
        self.pth_row_w.setEnabled(not from_run)
        # A loaded run.json dictates arch/grid/tile; the manual .pth source frees them.
        self._apply_manifest_lock(from_run and self._manifest is not None)

    def _apply_manifest_lock(self, locked: bool):
        """Grey out the inputs a run.json dictates (architecture, grid, tile size) so
        they can't drift from the loaded run. Unlocking restores tile-size enablement
        to whether the backbone even has a tile param (RandLA has none)."""
        self.backbone_combo.setEnabled(not locked)
        self.grid_spin.setEnabled(not locked)
        if locked:
            self.chunk_spin.setEnabled(False)
        else:
            key = self.backbone_combo.currentData()
            b = BACKBONES.get(key) if key else None
            self.chunk_spin.setEnabled(bool(b) and b.has_chunk)

    def _pick_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose weights (.pth)",
            self.weights_edit.text().strip() or str(appstate.workspace_dir()),
            "PyTorch checkpoints (*.pth *.pt)")
        if path:
            self.weights_edit.setText(path)

    def _resolved_weights(self):
        """Local weights: the explicit override if set, else the run.json's sibling.
        The two need not be co-located."""
        t = self.weights_edit.text().strip()
        return Path(t) if t else self._local_weights

    def _sync_controls(self):
        """Auto-fill grid + tile from the backbone's defaults; disable tile for RandLA."""
        if self.backbone_combo.currentData() is None: 
            return
        b = self._backbone()
        gp = next((p for p in b.params if p.recommend_key == "grid"), None)
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

    def _pick_input(self):
        d = QFileDialog.getExistingDirectory(self, "Folder of point clouds to label")
        if d:
            self.input_edit.setText(d)

    def _pick_input_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Point-cloud file to label", appstate.get("last_view_dir", ""),
            "Point clouds (*.ply *.npz *.las *.laz *.txt *.csv *.pcd *.xyz *.pts);;All files (*)")
        if f:
            self.input_edit.setText(f)

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(
            self, "Output folder for predictions",
            self.out_edit.text() or str(appstate.default_download_dir()))
        if d:
            self.out_edit.setText(d)

    def _backbone(self):
        # The run.json's backbone is authoritative for LOCAL inference so no deprecated backbones
        if (self._manifest and self.from_run_radio.isChecked()
                and appstate.get_exec_mode() == "local"
                and self._manifest.get("backbone") in BACKBONES):
            return BACKBONES[self._manifest["backbone"]]
        return BACKBONES[self.backbone_combo.currentData()]

    # ------------------------------------------------------------- run chain
    def _run(self):
        input_dir = self.input_edit.text().strip()
        if not os.path.exists(input_dir):
            self._append("Choose an input folder or file first.")
            return
        modal = appstate.get_exec_mode() != "local"
        weights_run_id = ""
        if self.from_file_radio.isChecked():
            if not os.path.isfile(self.pth_edit.text().strip()):
                self._append("Choose a .pth file.")
                return
            self._weights_remote = f"uploads/{Path(self.pth_edit.text()).name}"
        elif not modal:
            # LOCAL: run.json is the explicit input; weights = its sibling.
            if not (self._manifest and self._manifest_path):
                self._append("Pick a run.json first.")
                return
            w = self._resolved_weights()
            if not (w and w.is_file()):
                self._append(f"✗ Weights not found ({w}). Set the 'Weights file' box.")
                return
            bkey = self._manifest.get("backbone")
            if bkey in BACKBONES and not BACKBONES[bkey].folder_infer:
                self._append(f"✗ {BACKBONES[bkey].label} doesn't support folder inference.")
                return
        else:
            # MODAL: weights live on the cloud volume, keyed by run id.
            h = self.run_combo.currentData()
            if isinstance(h, dict):
                run_id = h["run_id"]
            else:
                parts = self.run_combo.currentText().split()   # empty combo -> no crash
                run_id = parts[0] if parts else ""
            if not run_id:
                self._append("Pick or type a run id.")
                return
            bkey = h.get("backbone") if isinstance(h, dict) else None
            if bkey in BACKBONES and not BACKBONES[bkey].folder_infer:
                self._append(f"✗ {BACKBONES[bkey].label} doesn't support folder inference.")
                return
            self._weights_remote = f"runs/{run_id}/final_model.pth"
            weights_run_id = run_id

        self._job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = ""
        self.log.clear()
        self.run_btn.setEnabled(False)
        # Runs enter history at train START, so may lack final_model.pth. Check weights
        # exist before converting/uploading/paying for GPU. Runs in a worker.
        if modal and weights_run_id:
            self._pending_input = input_dir
            self._pending_run_id = weights_run_id
            self._append(f"[0/4] Checking weights on Modal "
                         f"({self._backbone().outputs_volume})…")
            self.preflight.start(_check_weights_present,
                                 self._backbone().outputs_volume, weights_run_id)
            return
        self._start_conversion(input_dir)

    def _start_conversion(self, input_dir: str):
        # p95 end-to-end; honor a legacy norm recorded in run.json.
        norm = (self._manifest or {}).get("intensity_norm", "p95") \
            if (self.from_run_radio.isChecked() and self._manifest) else "p95"
        hag_filter = self._run_hag_filter()
        if hag_filter:
            self._append(f"[1/4] Run trained on real HAG ({hag_filter}) - reproducing "
                         "it for the input scenes.")
        job_root = self._infer_out_dir()
        self._append(f"[1/4] Converting {input_dir} to scenes (job {self._job_id}; "
                     f"intensity={norm}) -> {job_root}…")
        self.converter.start(dataset.convert_infer_job, self._job_id, input_dir,
                             appstate.workspace_dir(), intensity_norm=norm,
                             hag=bool(hag_filter), hag_filter=hag_filter or "grid",
                             out_dir=job_root)

    def _infer_out_dir(self) -> Path:
        """Nest this infer job under its owning dataset (<dataset>/infer/<job>) when
        the run names a known, on-disk dataset; else a findable workspace scratch
        spot (loose .pth has no linked dataset). The container still mounts it at
        /datasets/_infer/<job> regardless."""
        name = (self._manifest or {}).get("dataset") \
            if (self.from_run_radio.isChecked() and self._manifest) else None
        staged = appstate.known_datasets().get(name or "", {}).get("staged_dir", "")
        if staged and os.path.isdir(staged):
            return appstate.dataset_root(name) / "infer" / self._job_id
        return appstate.scratch_infer_dir() / self._job_id

    def _run_hag_filter(self) -> str | None:
        """The HAG method the run trained with (to reproduce at inference), or
        None when it trained on the z-min proxy / no HAG. Legacy hag_source
        strings (pdal_hag_nn, smrf, labeled+smrf, per_tile_smrf, ...) were all
        PDAL-based -> hag_nn."""
        m = self._manifest if (self.from_run_radio.isChecked() and self._manifest) else None
        if not m:
            return None
        s = str(m.get("hag_source", "")).lower()
        if not s or "proxy" in s or "z_minus" in s:
            return None
        if "grid" in s:
            return "grid"
        if "delaunay" in s:
            return "hag_delaunay"
        return "hag_nn"

    def _on_preflight(self, present):
        """Weights check: True=found, False=missing (block), None=couldn't list (proceed)."""
        if present is False:
            self._append(f"✗ Run '{self._pending_run_id}' has no final_model.pth on the "
                         f"outputs volume. Pick a completed run, or use 'Local .pth file'.")
            self.run_btn.setEnabled(True)
            return
        if present is None:
            self._append("[0/4] (couldn't verify weights on Modal - proceeding.)")
        self._start_conversion(self._pending_input)

    def _on_preflight_error(self, tb: str):
        # A failed check shouldn't block; fall through to the in-container backstop.
        self._append(f"[0/4] (weights check errored, proceeding anyway)\n{tb}")
        self._start_conversion(self._pending_input)

    def _on_converted(self, staged: Path):
        self._staged = staged
        if appstate.get_exec_mode() == "local":
            self._start_local_infer()
            return
        self._append(f"[2/4] Uploading scenes -> {modal_cli.DATASETS_VOLUME}:/_infer/{self._job_id}…")
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
            # Predictions on the datasets volume at _infer/<job_id>/predictions.
            # Download to the chosen folder; job-id subfolder avoids collisions.
            base = self.out_edit.text().strip() or str(appstate.default_download_dir())
            appstate.put("infer_out", self.out_edit.text().strip())
            self._dl_dest = Path(base) / f"predictions_{self._job_id}"
            self._dl_dest.mkdir(parents=True, exist_ok=True)
            self._append(f"[4/4] Downloading predictions -> {self._dl_dest}…")
            self._stage = "download"
            prog, args = modal_cli.volume_get(modal_cli.DATASETS_VOLUME,
                                              f"_infer/{self._job_id}/predictions",
                                              str(self._dl_dest))
            self.runner.start(prog, args, cwd=self.repo_root)
        elif self._stage == "run_local":
            # Local: predictions already on the host, no download.
            self._report_predictions(self._pred_dir)
        elif self._stage == "download":
            self._report_predictions(self._dl_dest / "predictions")

    def _report_predictions(self, pred_dir):
        """Predictions landed (a stage can exit 0 yet write nothing) -> write the
        scripts' npz predictions as the chosen format (xyz + classification) on a
        worker thread; _on_exported prints the final green report."""
        pred_dir = Path(pred_dir) if pred_dir else None
        if not (pred_dir and pred_dir.is_dir()):
            self.run_btn.setEnabled(True)
            self._append(f"\n✗ No predictions folder at {pred_dir}.")
            return
        preds = [p for p in sorted(pred_dir.iterdir())
                 if p.suffix.lower() in (".ply", ".npz")]
        if not preds:
            self.run_btn.setEnabled(True)
            self._append(f"\n✗ No prediction files in {pred_dir}. Check the log above.")
            return
        appstate.put("last_view_dir", str(pred_dir))
        fmt = self.fmt_combo.currentData()
        appstate.put("infer_format", fmt)
        self._append(f"\n[export] writing predictions as {fmt} (xyz + classification)…")
        self.exporter.start(dataset.export_predictions, pred_dir, fmt)

    def _on_exported(self, written):
        self.run_btn.setEnabled(True)
        if not written:
            self._append("✗ Nothing exported (no *_pred.npz in the predictions folder).")
            return
        self._append(f"\n✓ Done - {len(written)} prediction file(s) in {written[0].parent}.\n"
                     f"  'View a point cloud…' to open one, or 'Compare to ground "
                     f"truth…' for accuracy + mIoU.")

    def _on_export_error(self, tb: str):
        # Predictions still exist as the scripts' raw .npz; only the rewrite failed.
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Format conversion failed — predictions remain as raw "
                     f".npz files.\n{tb}")

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
        self._append(f"[3/4] Running inference on Modal ({b.label})…")
        self._stage = "run"
        prog, args = modal_cli.run_script(b.script, flags, detach=False)
        self._append(f"$ modal {' '.join(args)}\n")
        self.runner.start(prog, args, cwd=self.repo_root)

    def _start_local_infer(self):
        """Local Docker inference: scenes bind-mounted, predictions to host. No up/download."""
        b = self._backbone()
        # Scenes (and where predictions get written) are self._staged on the host.
        extra_mounts = [(str(self._staged), f"/datasets/_infer/{self._job_id}")]
        # Predictions: user folder mounted over the container's predictions dir, else staging.
        out = self.out_edit.text().strip()
        appstate.put("infer_out", out)
        if out:
            self._pred_dir = Path(out)
            self._pred_dir.mkdir(parents=True, exist_ok=True)
            extra_mounts.append(
                (str(self._pred_dir), f"/datasets/_infer/{self._job_id}/predictions"))
        else:
            self._pred_dir = self._staged / "predictions"
        # Weights: the picked .pth or run.json's sibling; mount its dir.
        wpath = Path(self.pth_edit.text().strip()) if self.from_file_radio.isChecked() \
            else self._resolved_weights()
        extra_mounts.append((str(wpath.parent), "/outputs/_local_weights"))
        weights = f"_local_weights/{wpath.name}"
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
                                          extra_mounts=extra_mounts, env=self._infer_dg_env())
        self._append(f"[local] Running inference in Docker ({b.label})…")
        self._append(f"[local] $ {local_cli.preview(prog, args)}\n")
        if not local_cli.have_docker():
            self._append("[local] docker not found on PATH; printed the command only. "
                         "On a Docker+GPU host predictions land in "
                         f"{self._pred_dir.as_posix()}.")
            self.run_btn.setEnabled(True)
            return
        gok, gmsg = local_cli.gpu_preflight()   # CUDA-only — fail clearly, not cryptically
        if gmsg:
            self._append(gmsg)
        if not gok:
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
        # Local Docker logs print container bind-mount paths (/datasets/_infer/<job>/…)
        # that mean nothing on the host; show the real output folder the files land in.
        disp = (_localize_paths(text, self._job_id, self._pred_dir, self._staged)
                if self._stage == "run_local" else text)
        self._append(disp, newline=False)
        if self._stage in ("run", "run_local"):
            self.parser.feed(text)   # parser sees the raw text

    def _on_run_id(self, run_id: str):
        self._run_id = run_id

    def _on_error(self, tb: str):
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Conversion error:\n{tb}")

    def _on_runner_failed(self, err: str):
        # FailedToStart fires `failed` not `finished`; re-enable the button.
        self.run_btn.setEnabled(True)
        self._append(f"\n✗ Failed to start: {err}")

    # ------------------------------------------------------------- view
    def _open_viewer(self, *args: str):
        QProcess.startDetached(sys.executable, ["-m", "trainer_gui.viewer", *args], PROJECT_DIR)

    def _view_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Point cloud to view", appstate.get("last_view_dir", ""),
            "Point clouds (*.ply *.npz *.las *.laz *.txt *.pcd);;All files (*)")
        if not path:
            return
        appstate.put("last_view_dir", str(Path(path).parent))
        names = self._effective_names()
        pal = ";".join(f"{r},{g},{b}" for r, g, b in self._palette_colors())
        self._open_viewer(path, "--class-names", ",".join(names), "--palette", pal)
        self._append(f"Opened viewer for {Path(path).name}.")

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
            self, "Ground truth for this scene",
            appstate.get("truth_file", ""),
            "Ground truth (*.ply *.npz);;All files (*)")
        if not gt:
            return None
        appstate.put("truth_file", gt)
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
                     f"(yellow = predicted class differs).")

    def _show_stats(self, pred: str, gt: str):
        """Print accuracy + mIoU of prediction vs ground truth to the log."""
        from .. import viewer
        self._append("  computing accuracy + mIoU…")
        try:
            m = viewer.prediction_metrics(pred, gt)
        except Exception as e:  # noqa: BLE001
            self._append(f"  ✗ couldn't compute stats: {e}")
            return
        names = self._effective_names()
        nm = lambda c: names[c] if 0 <= c < len(names) else f"class {c}"
        lines = [f"\n── {m['scene'] or Path(pred).stem} vs ground truth ──",
                 f"  accuracy : {m['accuracy']:.4f}",
                 f"  mIoU     : {m['miou']:.4f}   (over {len(m['per_class_iou'])} present classes)",
                 f"  labeled  : {m['labeled']:,} pts",
                 "  per-class IoU:"]
        lines += [f"    {nm(c)}: {iou:.4f}" for c, iou in sorted(m["per_class_iou"].items())]
        # Persist beside the prediction so the numbers aren't lost with the log.
        mpath = Path(pred).with_suffix(".metrics.json")
        try:
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump({"prediction": str(pred), "ground_truth": str(gt),
                           "class_names": names, **m}, f, indent=2)
            lines.append(f"  saved -> {mpath}")
        except OSError as e:
            lines.append(f"  (couldn't save metrics json: {e})")
        self._append("\n".join(lines))

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
        self._append(f"Exporting {Path(pred).name} vs {Path(gt).name} -> {out}")

    # ------------------------------------------------------------- helpers
    def _append(self, text: str, newline: bool = True):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text + ("\n" if newline else ""))
        self.log.moveCursor(QTextCursor.End)


class ClassPaletteDialog(QDialog):
    """Class menu: pick the name source, rename classes, set colours.
    source_key()/names()/colors() are read back on accept."""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self.setWindowTitle("Class colours & names")
        self.resize(440, 440)

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Names from:"))
        self.src = QComboBox()
        for label, key in page._source_options():
            self.src.addItem(label, key)
        i = self.src.findData(page._current_source_key())
        self.src.setCurrentIndex(i if i >= 0 else 0)
        self.src.currentIndexChanged.connect(self._rebuild)
        top.addWidget(self.src, 1)
        lay.addLayout(top)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Class", "Name", "Colour"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        lay.addWidget(self.table)

        foot = QHBoxLayout()
        reset = QPushButton("Reset colours")
        reset.clicked.connect(self._reset_colors)
        foot.addWidget(reset)
        foot.addStretch()
        lay.addLayout(foot)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

        self._colors: list = []
        self._name_edits: list = []
        self._color_btns: list = []
        self._rebuild()

    def _rebuild(self):
        key = self.src.currentData()
        names = self.page._apply_name_overrides(key, self.page._names_for_key(key))
        self._colors = self.page._colors_for(key, names)
        self._name_edits, self._color_btns = [], []
        self.table.setRowCount(len(names))
        for r, nm in enumerate(names):
            idx = QTableWidgetItem(str(r))
            idx.setFlags(idx.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, idx)
            edit = QLineEdit(nm)
            self._name_edits.append(edit)
            self.table.setCellWidget(r, 1, edit)
            btn = QPushButton()
            btn.setFixedHeight(20)
            btn.clicked.connect(lambda _=False, i=r: self._pick(i))
            self._color_btns.append(btn)
            self.table.setCellWidget(r, 2, btn)
        self.table.resizeColumnToContents(0)
        self._refresh_swatches()

    def _refresh_swatches(self):
        for b, c in zip(self._color_btns, self._colors):
            b.setStyleSheet(f"background-color: rgb({c[0]},{c[1]},{c[2]}); border: 1px solid #888;")

    def _pick(self, idx: int):
        col = QColorDialog.getColor(QColor(*self._colors[idx]), self, "Pick a class colour")
        if col.isValid():
            self._colors[idx] = [col.red(), col.green(), col.blue()]
            self._refresh_swatches()

    def _reset_colors(self):
        self._colors = self.page._default_colors(len(self._colors))
        self._refresh_swatches()

    def source_key(self) -> str:
        return self.src.currentData()

    def names(self) -> list:
        return [e.text().strip() or f"class {i}" for i, e in enumerate(self._name_edits)]

    def colors(self) -> list:
        return [list(c) for c in self._colors]


def _localize_paths(text: str, job_id: str, pred_dir, staged) -> str:
    """Rewrite container bind-mount paths to the host folders they map to (see the
    mounts in _start_local_infer), so a local run's log shows where files really go.
    Predictions first (the more specific mount), then the staging root."""
    if pred_dir:
        for p in (f"/datasets/_infer/{job_id}/predictions", f"_infer/{job_id}/predictions"):
            text = text.replace(p, str(pred_dir))
    if staged:
        for p in (f"/datasets/_infer/{job_id}", f"_infer/{job_id}"):
            text = text.replace(p, str(staged))
    return text


def _entry_name(entry: dict) -> str:
    """Basename of a `modal volume ls --json` entry (key name varies by CLI ver)."""
    for k in ("path", "Filename", "filename", "name", "Name"):
        v = entry.get(k)
        if v:
            return str(v).rstrip("/").rsplit("/", 1)[-1]
    return ""


def _check_weights_present(volume: str, run_id: str, progress=None):
    """runs/<run_id>/final_model.pth on the outputs volume?
    True=yes, False=missing (block), None=couldn't list. Runs in a FuncWorker thread."""
    entries = modal_cli.list_volume_entries(volume, f"/runs/{run_id}")
    if not entries:
        return None
    return any(_entry_name(e) == "final_model.pth" for e in entries)


def _wrap(layout) -> QWidget:
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w
