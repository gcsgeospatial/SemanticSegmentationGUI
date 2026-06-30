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

# How many generic 'class i' names to show when neither the loaded run nor a
# dataset supplies class names (the Auto source before a labelled file is opened).
_FALLBACK_NUM_CLASSES = 5


class InferPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.converter = FuncWorker(self)
        self.preflight = FuncWorker(self)   # Modal weights existence check (H4)
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
        # Let each row's field fill the width so the trailing Browse… button always
        # sits flush-right with room (never clipped/occluded by a too-narrow field).
        wf.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        radio_row = QHBoxLayout()
        self.from_run_radio = QRadioButton("From a training run")
        self.from_run_radio.setChecked(True)
        self.from_file_radio = QRadioButton("Local .pth file")
        self.from_run_radio.toggled.connect(self._on_source_toggle)
        radio_row.addWidget(self.from_run_radio)
        radio_row.addWidget(self.from_file_radio)
        radio_row.addStretch()
        wf.addRow("Source", _wrap(radio_row))
        # LOCAL: pick the run's run.json — the self-contained manifest training writes
        # next to the weights. Architecture, grid, tile, intensity norm, HAG, and the
        # weights path (its sibling) all come from it. No searching, no conventions.
        self.runjson_edit = QLineEdit()
        self.runjson_edit.setPlaceholderText("…/local_runs/runs/<id>/run.json")
        # #8: a typed/pasted path loads on Enter/focus-out; any edit drops the stale
        # load so a run can't proceed with a manifest that no longer matches the text.
        self.runjson_edit.editingFinished.connect(self._load_run_manifest)
        self.runjson_edit.textChanged.connect(self._invalidate_manifest)
        rj_row = QHBoxLayout()
        rj_row.addWidget(self.runjson_edit)
        rj_btn = QPushButton("Browse…")
        rj_btn.clicked.connect(self._pick_runjson)
        rj_row.addWidget(rj_btn)
        self.runjson_row_w = _wrap(rj_row)
        wf.addRow("Run file (run.json)", self.runjson_row_w)
        # Weights default to the .pth named in run.json (beside it), but can point
        # anywhere — run.json and weights need NOT be co-located.
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText("default: the weights named in run.json, beside it")
        w_row = QHBoxLayout()
        w_row.addWidget(self.weights_edit)
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
        iform.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.input_edit = QLineEdit()
        in_row = QHBoxLayout()
        in_row.addWidget(self.input_edit)
        fold_btn = QPushButton("Folder…")
        fold_btn.clicked.connect(self._pick_input)
        file_btn = QPushButton("File…")
        file_btn.clicked.connect(self._pick_input_file)
        in_row.addWidget(fold_btn)
        in_row.addWidget(file_btn)
        iform.addRow("Point clouds (folder or file)", _wrap(in_row))
        self.grid_spin = QDoubleSpinBox()
        self.grid_spin.setRange(0.02, 5.0)
        self.grid_spin.setSingleStep(0.05)
        self.grid_spin.setDecimals(2)
        self.grid_spin.setValue(0.30)
        iform.addRow("Grid size (m) — auto-filled from the run", self.grid_spin)
        # Intensity is p95-normalized end-to-end (build + train + inference), so there
        # is nothing to match here — convert_infer_job always uses p95.
        self.chunk_spin = QDoubleSpinBox()
        self.chunk_spin.setRange(10.0, 200.0)
        self.chunk_spin.setSingleStep(5.0)
        self.chunk_spin.setDecimals(0)
        self.chunk_spin.setValue(50.0)
        iform.addRow("Tile size (m)", self.chunk_spin)
        # Density generalization — INFERENCE-time, label-free, no retrain. Applies to ANY
        # model, so it lives here (a serving choice) not on the per-dataset Train panel.
        # Worth turning on when this cloud's density differs from the training density.
        self.dg_adabn_chk = QCheckBox("AdaBN — re-fit norm stats to this cloud (KPConvX / RandLA)")
        self.dg_adabn_chk.setToolTip(
            "Recompute BatchNorm running stats on the target tiles before predicting, so "
            "source-density stats stop mis-normalizing at a different inference density. "
            "Label-free, no retrain. No-op for PTv3 (its BN is stem/pooling only).")
        iform.addRow("Density adapt", self.dg_adabn_chk)
        self.dg_tta_chk = QCheckBox("Density TTA — average over")
        self.dg_tta_chk.setToolTip(
            "Average softmax over several density/scale resamplings of each tile to lower "
            "boundary-point variance. Label-free, no retrain. More views = slower.")
        self.dg_tta_spin = QSpinBox()
        self.dg_tta_spin.setRange(1, 9)
        self.dg_tta_spin.setValue(3)
        tta_row = QHBoxLayout()
        tta_row.addWidget(self.dg_tta_chk)
        tta_row.addWidget(self.dg_tta_spin)
        tta_row.addWidget(QLabel("extra views"))
        tta_row.addStretch(1)
        iform.addRow("", _wrap(tta_row))
        # Where prediction files land: on the host directly in local mode, or where
        # Modal predictions get downloaded. Always the user's express choice — empty
        # just falls back to a findable Downloads folder, never a hidden app dir.
        self.out_edit = QLineEdit()
        self.out_edit.setText(appstate.get("infer_out", ""))
        self.out_edit.setPlaceholderText(
            f"default: {appstate.default_download_dir().as_posix()}")
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

        # Output / view: a single action bar + a compact one-line legend (no stat
        # box, no crammed side column). Comparison metrics print to the log.
        self.view_btn = QPushButton("View a point cloud…")
        self.view_btn.clicked.connect(self._view_file)
        self.compare_btn = QPushButton("Compare to ground truth…")
        self.compare_btn.clicked.connect(self._compare_gt)
        self.export_btn = QPushButton("Export comparison PLY…")
        self.export_btn.clicked.connect(self._export_gt)
        self.palette_btn = QPushButton("Class colours & names…")
        self.palette_btn.setToolTip("Open the class menu: pick the name source (the run / a "
                                    "dataset / Auto), rename classes, and set each colour.")
        self.palette_btn.clicked.connect(self._configure_palette)
        actions = QHBoxLayout()
        actions.addWidget(self.view_btn)
        actions.addWidget(self.compare_btn)
        actions.addWidget(self.export_btn)
        actions.addStretch(1)
        actions.addWidget(self.palette_btn)
        # Live one-line swatch legend (the exact colours the 3D viewer paints).
        self.legend_label = QLabel()
        self.legend_label.setWordWrap(True)
        self.legend_label.setToolTip("How each class is coloured in the 3D viewer.")
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
            "Label new point clouds with an already-trained model. "
            + ("Pick the run's run.json (or a local .pth), a folder of clouds, and run "
               "— inference runs locally in Docker."
               if local else
               "Pick a training run (or a local .pth), a folder of clouds, and run on Modal."))
        # Output folder is shown in BOTH modes: in modal mode it's where predictions
        # download to, so the user always chooses a findable location (not %APPDATA%).
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
    # The class system is a per-class table (name source + editable name + colour)
    # opened from the "Class colours & names…" button. State lives in appstate:
    #   infer_palette          -> the chosen name SOURCE key
    #   palette_name_overrides -> {source_key: [name, …]}  (renames)
    #   palette_overrides      -> {source_key: [[r,g,b], …]}  (colours)
    # All keyed by source so a run, a dataset, and Auto keep independent edits.
    def reload_palettes(self):
        """Kept for callers (reload_runs / apply_exec_mode) — just refresh legend."""
        self._refresh_legend()

    def _set_run_classes(self, names):
        """Adopt the loaded run's class names + select that source, so predictions
        are labelled with the model's OWN trained classes (matched to run config)."""
        self._run_class_names = list(names) if names else None
        if self._run_class_names:
            appstate.put("infer_palette", "__run__")
        self._refresh_legend()

    @staticmethod
    def _names_from_manifest(m: dict) -> list | None:
        """Class names from a run manifest: explicit class_names, else synthesized
        'class 0..n-1' from num_classes, else None."""
        names = m.get("class_names")
        if names:
            return list(names)
        n = m.get("num_classes")
        return [f"class {i}" for i in range(int(n))] if n else None

    def _source_options(self) -> list:
        """(label, key) name-source choices for the class menu: the loaded run, Auto,
        then every dataset with resolvable class names."""
        opts = []
        if self._run_class_names:
            opts.append((f"The loaded run ({len(self._run_class_names)} classes)", "__run__"))
        opts.append(("Auto (names in the file, else class i)", "__auto__"))
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
        """Base (un-renamed) class names for a source key. Falls back to generic
        'class i' names when neither the run nor the dataset supplies any."""
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
        """Compact one-line swatch legend — the exact colours the 3D viewer paints."""
        parts = "".join(
            f'<span style="font-size:14px;color:#{r:02x}{g:02x}{b:02x}">■</span>'
            f'<span>&nbsp;{name}</span>&nbsp;&nbsp;&nbsp;'
            for name, (r, g, b) in zip(self._effective_names(), self._palette_colors()))
        self.legend_label.setText(parts)

    def _configure_palette(self):
        """Open the class menu (source + per-class name + colour). Persists the
        chosen source, renames and colours per source; the viewer uses them too."""
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
                    if rdir.name not in seen and (rdir / "run_config.json").exists():
                        seen.add(rdir.name)
                        self.run_combo.addItem(f"{rdir.name}  ({bdir.name})",
                                               {"run_id": rdir.name, "backbone": bdir.name})
        self.run_combo.blockSignals(False)
        self._on_run_pick()

    def _on_run_pick(self):
        """Modal run-id combo: sync the architecture from the picked run's backbone,
        and (when the run was downloaded locally) adopt its classes for the palette."""
        h = self.run_combo.currentData()
        if isinstance(h, dict) and h.get("backbone") in BACKBONES:
            i = self.backbone_combo.findData(h["backbone"])
            if i >= 0:
                self.backbone_combo.setCurrentIndex(i)
        if appstate.get_exec_mode() != "local":
            self._set_run_classes(self._run_pick_class_names(h))

    def _run_pick_class_names(self, h) -> list | None:
        """Class names for a Modal run, if it was downloaded locally (its run.json /
        run_config.json sits under the Runs-page download dir). None otherwise."""
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
        start = self.runjson_edit.text().strip() or str(appstate.local_runs_dir())
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose the run's run.json", start, "Run manifest (run.json *.json)")
        if path:
            self.runjson_edit.setText(path)
            self._load_run_manifest()

    def _invalidate_manifest(self):
        """#8: any edit to the path text drops the previously-loaded manifest, so a
        run can't proceed with stale weights/params (editingFinished reloads it)."""
        self._manifest = self._manifest_path = self._local_weights = None
        self._dg = {}

    def _infer_dg_env(self) -> dict:
        """DG_* env for the inference container. Two sources, both inference-time:
        logdk is RECOVERED from the weights' run.json (it changed the input width, so
        the channel must be recomputed or the load fails); AdaBN/TTA are the live
        toggles above (label-free, no retrain, applicable to any model)."""
        env: dict[str, str] = {}
        if self._dg.get("logdk"):
            env["DG_LOGDK_FEAT"] = "1"
            env["DG_LOGDK_K"] = str(int(self._dg.get("logdk_k", 8)))
        if self.dg_adabn_chk.isChecked():
            env["DG_INFER_ADABN"] = "1"
        if self.dg_tta_chk.isChecked():
            env["DG_INFER_TTA"] = str(self.dg_tta_spin.value())
        if env:
            self._append("[dg] inference: " + " ".join(f"{k}={v}" for k, v in sorted(env.items())))
        return env

    def _load_run_manifest(self):
        """Apply the picked run.json — the SINGLE explicit input for local inference.
        Architecture, grid, tile and intensity norm are read straight from it (no
        GUI-default guessing); weights resolve as its sibling. Refuses (rather than
        silently keeping the wrong architecture) if the run's backbone isn't
        selectable here (#7)."""
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
        if i < 0:   # #7: the run's model is hidden/unknown — don't keep the wrong one
            self._append(f"✗ This run's model '{bkey}' isn't available here. Enable it on "
                         f"the Train page (backbone checkboxes), then reload this run.json.")
            return
        self._manifest, self._manifest_path = m, p
        self.backbone_combo.setCurrentIndex(i)       # fires _sync_controls (sets defaults)
        if m.get("grid") is not None:                # then the manifest overrides them
            self.grid_spin.setValue(float(m["grid"]))
        if m.get("chunk_xy") is not None and self.chunk_spin.isEnabled():
            self.chunk_spin.setValue(float(m["chunk_xy"]))
        self._local_weights = p.parent / m.get("weights", "final_model.pth")
        self.weights_edit.setText(str(self._local_weights))   # default; user may override
        # DG settings baked into the weights: logdk changes the input width, so it MUST
        # be re-fed at inference (re-injected as DG_* env below). AdaBN/TTA are separate
        # inference-time toggles in the Input box.
        self._dg = m.get("dg") or {}
        # Label the legend + viewer with the model's OWN classes (the uploaded
        # dataset's classes, as recorded in the run config), combined with the palette.
        self._set_run_classes(self._names_from_manifest(m))
        ok = "✓" if self._local_weights.is_file() else "✗ weights missing —"
        self._append(f"Loaded {p.name}: {bkey}, grid={m.get('grid')}, "
                     f"chunk={m.get('chunk_xy')}, intensity={m.get('intensity_norm')}. "
                     f"{ok} {self._local_weights}")
        if self._dg.get("logdk"):
            self._append(f"[dg] this model was trained with the log-d_k density channel "
                         f"(k={self._dg.get('logdk_k', 8)}) — it will be recomputed at inference.")

    def _on_source_toggle(self):
        from_run = self.from_run_radio.isChecked()
        self.run_combo.setEnabled(from_run)
        self.runjson_row_w.setEnabled(from_run)
        self.weights_row_w.setEnabled(from_run)
        self.pth_row_w.setEnabled(not from_run)

    def _pick_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose model weights (.pth)",
            self.weights_edit.text().strip() or str(appstate.local_runs_dir()),
            "PyTorch checkpoints (*.pth *.pt)")
        if path:
            self.weights_edit.setText(path)

    def _resolved_weights(self):
        """Weights for a local from-a-run: the explicit override if set, else the
        run.json's sibling. The two need NOT be co-located — the GUI passes the
        run.json's grid/tile/intensity/HAG/log-d_k to the run as flags + env, so
        accuracy doesn't depend on the .pth sitting next to its run.json."""
        t = self.weights_edit.text().strip()
        return Path(t) if t else self._local_weights

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
        # The run.json's backbone is authoritative for a from-a-run LOCAL inference,
        # so execution never uses a stale/changed architecture dropdown (#7).
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
        weights_run_id = ""          # modal from-a-run only; drives the H4 preflight
        if self.from_file_radio.isChecked():
            if not os.path.isfile(self.pth_edit.text().strip()):
                self._append("Choose a .pth file.")
                return
            self._weights_remote = f"uploads/{Path(self.pth_edit.text()).name}"
        elif not modal:
            # LOCAL from-a-run: the run.json is the single explicit input; weights are
            # its sibling (resolved on load). No run-id parsing, no searching.
            if not (self._manifest and self._manifest_path):
                self._append("Pick the run's run.json first (Browse…).")
                return
            w = self._resolved_weights()
            if not (w and w.is_file()):
                self._append(f"✗ weights file not found ({w}). Set it in the 'Weights "
                             f"file' box (defaults to the .pth named in run.json, beside it).")
                return
            bkey = self._manifest.get("backbone")
            if bkey in BACKBONES and not BACKBONES[bkey].folder_infer:
                self._append(f"✗ {BACKBONES[bkey].label} doesn't support folder inference.")
                return
        else:
            # MODAL from-a-run: weights live on the cloud volume, keyed by run id.
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
                self._append(f"✗ {BACKBONES[bkey].label} doesn't support folder inference "
                             f"(its script has no --infer-input mode).")
                return
            self._weights_remote = f"runs/{run_id}/final_model.pth"
            weights_run_id = run_id

        self._job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = ""
        self.log.clear()
        self.run_btn.setEnabled(False)
        # H4: on Modal, a run is added to history at train START, so a crashed/
        # unfinished run shows in the picker with no final_model.pth. Confirm the
        # weights exist on the outputs volume BEFORE converting + uploading + paying
        # for a GPU spin-up. Runs in a worker (network call) so the UI never freezes.
        if modal and weights_run_id:
            self._pending_input = input_dir
            self._pending_run_id = weights_run_id
            self._append(f"[0/4] Checking weights on Modal "
                         f"({self._backbone().outputs_volume}) …")
            self.preflight.start(_check_weights_present,
                                 self._backbone().outputs_volume, weights_run_id)
            return
        self._start_conversion(input_dir)

    def _start_conversion(self, input_dir: str):
        # Intensity is p95 end-to-end; honor a run.json that recorded a legacy norm
        # so an old max-trained model still infers in-distribution.
        norm = (self._manifest or {}).get("intensity_norm", "p95") \
            if (self.from_run_radio.isChecked() and self._manifest) else "p95"
        want_hag = self._run_wants_real_hag()
        if want_hag:
            from .. import pretrain
            if pretrain.pdal_available():
                self._append("[1/4] Run trained on real PDAL HAG — computing it for the "
                             "input scenes (matches training).")
            else:
                self._append("⚠ This run trained on real PDAL HAG, but PDAL isn't installed "
                             "here — inference will use a z-min proxy (degraded results).")
        self._append(f"[1/4] Converting {input_dir} to canonical scenes (job {self._job_id}; "
                     f"intensity={norm})…")
        self.converter.start(dataset.convert_infer_job, self._job_id, input_dir,
                             appstate.staging_dir(), intensity_norm=norm, hag=want_hag)

    def _run_wants_real_hag(self) -> bool:
        """True if the picked run trained on real PDAL HAG (read from its run.json),
        so convert_infer_job reproduces it instead of a z-min proxy. A bare .pth, or a
        run.json that records the proxy, -> False."""
        m = self._manifest if (self.from_run_radio.isChecked() and self._manifest) else None
        if not m:
            return False
        s = str(m.get("hag_source", "")).lower()
        return ("pdal" in s or "hag_nn" in s) and "proxy" not in s and "z_minus" not in s

    def _on_preflight(self, present):
        """Result of the H4 Modal weights check: True=found, False=dir exists but no
        final_model.pth (block), None=couldn't list (proceed; in-container backstop)."""
        if present is False:
            self._append(f"✗ Run '{self._pending_run_id}' has no final_model.pth on the "
                         f"outputs volume — it likely crashed before a best epoch or hasn't "
                         f"finished. Pick a completed run, or use the 'Local .pth file' option.")
            self.run_btn.setEnabled(True)
            return
        if present is None:
            self._append("[0/4] (couldn't verify weights on Modal — proceeding; the run "
                         "will fail in-container if they're truly missing.)")
        self._start_conversion(self._pending_input)

    def _on_preflight_error(self, tb: str):
        # A failed CHECK shouldn't block a launch — fall through to the in-container
        # backstop rather than stranding the user on a transient modal/CLI error.
        self._append(f"[0/4] (weights check errored, proceeding anyway)\n{tb}")
        self._start_conversion(self._pending_input)

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
            # Download into the user's chosen folder (findable Downloads by default),
            # not a hidden app dir. job-id subfolder keeps repeat runs from colliding.
            base = self.out_edit.text().strip() or str(appstate.default_download_dir())
            appstate.put("infer_out", self.out_edit.text().strip())
            self._dl_dest = Path(base) / f"predictions_{self._job_id}"
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
            self._report_predictions(self._pred_dir)
        elif self._stage == "download":
            self._report_predictions(self._dl_dest / "predictions")

    def _report_predictions(self, pred_dir):
        """Final-stage report — green ONLY when predictions actually landed (L5): a
        stage can exit 0 yet write nothing, and a green '0 prediction file(s)' reads
        as success when it isn't."""
        self.run_btn.setEnabled(True)
        pred_dir = Path(pred_dir) if pred_dir else None
        if not (pred_dir and pred_dir.is_dir()):
            self._append(f"\n✗ No predictions folder at {pred_dir}.")
            return
        preds = [p for p in sorted(pred_dir.iterdir())
                 if p.suffix.lower() in (".ply", ".npz")]
        if not preds:
            self._append(f"\n✗ Produced no prediction files in {pred_dir} — check the log "
                         f"above for errors.")
            return
        appstate.put("last_view_dir", str(pred_dir))
        self._append(f"\n✓ Done — {len(preds)} prediction file(s) in {pred_dir}.\n"
                     f"  'View a point cloud…' to open one, or 'Compare to ground "
                     f"truth…' for accuracy + mIoU.")

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
        # Weights = an explicit host file: the picked .pth, or the run.json's sibling
        # (resolved on load). Mount its dir into the container; no searching.
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
        self._append(f"[local] Running inference in Docker ({b.label}) …")
        self._append(f"[local] $ {local_cli.preview(prog, args)}\n")
        if not local_cli.have_docker():
            self._append("[local] docker not found on PATH — printed the exact command "
                         "(design-now mode). On a Docker+GPU host the predictions land in "
                         f"{self._pred_dir.as_posix()}.")
            self.run_btn.setEnabled(True)
            return
        gok, gmsg = local_cli.gpu_preflight()   # M4: CUDA-only — fail clearly, not cryptically
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
            self, "Ground truth for this scene (.ply or .npz)",
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
                     f"(yellow = predicted class differs from the ground truth).")

    def _show_stats(self, pred: str, gt: str):
        """Compute accuracy + mIoU of the prediction vs ground truth and print them
        to the log (the side stats box is gone)."""
        from .. import viewer
        self._append("  computing accuracy + mIoU …")
        try:
            m = viewer.prediction_metrics(pred, gt)
        except Exception as e:  # noqa: BLE001
            self._append(f"  ✗ could not compute stats: {e}")
            return
        names = self._effective_names()
        nm = lambda c: names[c] if 0 <= c < len(names) else f"class {c}"
        lines = [f"\n── {m['scene'] or Path(pred).stem} vs ground truth ──",
                 f"  accuracy : {m['accuracy']:.4f}",
                 f"  mIoU     : {m['miou']:.4f}   (over {len(m['per_class_iou'])} present classes)",
                 f"  labeled  : {m['labeled']:,} pts",
                 "  per-class IoU:"]
        lines += [f"    {nm(c)}: {iou:.4f}" for c, iou in sorted(m["per_class_iou"].items())]
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
        self._append(f"Exporting comparison of {Path(pred).name} vs {Path(gt).name} "
                     f"-> {out}")

    # ------------------------------------------------------------- helpers
    def _append(self, text: str, newline: bool = True):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text + ("\n" if newline else ""))
        self.log.moveCursor(QTextCursor.End)


class ClassPaletteDialog(QDialog):
    """The class-matching menu (like the Datasets classes table, plus colour): pick
    the name SOURCE (the loaded run / a dataset / Auto), rename each class, and set
    its viewer colour. `source_key()/names()/colors()` are read back on accept.

    Driven by the InferPage (`page`) so it can resolve names per source + read the
    saved name/colour overrides. Changing the source rebuilds the table."""

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


def _entry_name(entry: dict) -> str:
    """Basename of a `modal volume ls --json` entry (key name varies by CLI ver)."""
    for k in ("path", "Filename", "filename", "name", "Name"):
        v = entry.get(k)
        if v:
            return str(v).rstrip("/").rsplit("/", 1)[-1]
    return ""


def _check_weights_present(volume: str, run_id: str, progress=None):
    """H4: does runs/<run_id>/final_model.pth exist on the Modal outputs volume?
    True=yes, False=the dir lists but lacks it (block the launch), None=empty/
    unreachable listing (couldn't tell — caller proceeds to the in-container check).
    Blocking `modal volume ls`; runs in a FuncWorker thread."""
    entries = modal_cli.list_volume_entries(volume, f"/runs/{run_id}")
    if not entries:
        return None
    return any(_entry_name(e) == "final_model.pth" for e in entries)


def _wrap(layout) -> QWidget:
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w
