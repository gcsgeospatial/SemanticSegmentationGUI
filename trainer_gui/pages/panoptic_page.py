"""Panoptic page: ALPINE clustering (arXiv 2503.13203) over an inference
predictions folder - per-point instance ids from the semantics alone, no
training. Instances ride back into the *_pred.npz and export as las/laz
"instance_id"."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from .. import appstate, dataset, panoptic, ui
from ..jobs import FuncWorker
from ..logconsole import LogConsole


def _job_id(pred_dir: Path) -> str:
    """predictions_<job>[_m<k>|_ensemble][/predictions] -> <job>."""
    name = pred_dir.parent.name if pred_dir.name == "predictions" else pred_dir.name
    name = re.sub(r"(_m\d+|_ensemble)$", "", name)
    return name[len("predictions_"):] if name.startswith("predictions_") else name


def _class_names_for(pred_dir: Path) -> tuple[list | None, str]:
    """(class_names, source label): infer_run.json beside the npz (ensemble),
    run.json a level up (downloaded runs), else the staged infer job's
    infer_run.json (local runs). None when nothing names the classes."""
    for p in (pred_dir / "infer_run.json", pred_dir.parent / "run.json",
              pred_dir.parent.parent / "run.json"):
        m = appstate.read_json(p)
        if m and m.get("class_names"):
            return list(m["class_names"]), p.name
    roots = [appstate.scratch_infer_dir()]
    for name in appstate.known_datasets():
        try:
            roots.append(appstate.dataset_root(name) / "infer")
        except (KeyError, OSError):
            pass
    for r in roots:
        m = appstate.read_json(r / _job_id(pred_dir) / "infer_run.json")
        if m and m.get("class_names"):
            return list(m["class_names"]), "infer_run.json"
    return None, ""


class PanopticPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.clusterer = FuncWorker(self)
        self.exporter = FuncWorker(self)
        self._pred_dir: Path | None = None
        self._active_dir: Path | None = None   # frozen at launch; callbacks use this
        self._rows: list[tuple] = []   # (class index, name, chk, len_spin, wid_spin)
        self._run_open = False

        root = QVBoxLayout(self)
        title = QLabel("Panoptic")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        sub = QLabel("Turn semantic predictions into instances with ALPINE "
                     "clustering (training-free, arXiv 2503.13203). Pick a "
                     "predictions folder, check the classes that are countable "
                     "objects, set their typical footprint, and run. Instance "
                     "ids are saved into the npz and exported as 'instance_id'.")
        sub.setWordWrap(True)
        sub.setObjectName("pageSub")
        root.addWidget(sub)

        ibox = QGroupBox("Input")
        form = QFormLayout(ibox)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.pred_combo = QComboBox()
        self.pred_combo.currentIndexChanged.connect(self._on_pick)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse)
        self.rescan_btn = QPushButton("Rescan")
        self.rescan_btn.clicked.connect(self.reload_preds)
        prow = QHBoxLayout()
        prow.addWidget(self.pred_combo, 1)
        prow.addWidget(self.browse_btn)
        prow.addWidget(self.rescan_btn)
        form.addRow("Predictions", ui.wrap(prow))
        self.src_label = QLabel("")
        self.src_label.setObjectName("pageSub")
        self.src_label.setWordWrap(True)
        form.addRow("", self.src_label)

        self.cbox = QGroupBox("Thing classes - checked classes get split into instances")
        self.cgrid = QGridLayout(self.cbox)
        self.cgrid.setColumnStretch(0, 1)

        pbox = QGroupBox("Clustering")
        prow2 = QHBoxLayout(pbox)
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 256)
        self.k_spin.setValue(32)
        self.k_spin.setToolTip("kNN graph neighbors per point (paper default 32).")
        self.split_chk = QCheckBox("split oversized clusters")
        self.split_chk.setChecked(True)
        self.split_chk.setToolTip("Split merged neighbors whose fitted 2D box "
                                  "exceeds the class footprint (paper's box-splitting).")
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(1.0, 5.0)
        self.margin_spin.setSingleStep(0.1)
        self.margin_spin.setValue(1.3)
        self.margin_spin.setToolTip("Footprint tolerance before a cluster is split.")
        self.fmt_combo = QComboBox()
        for f in dataset.PRED_EXPORT_FORMATS:
            if f != "ply":   # ply is classification-only; instances would vanish
                self.fmt_combo.addItem(f, f)
        self.fmt_combo.setCurrentIndex(max(0, self.fmt_combo.findData(
            appstate.get("infer_format", "las"))))
        prow2.addWidget(QLabel("neighbors k"))
        prow2.addWidget(self.k_spin)
        prow2.addWidget(self.split_chk)
        prow2.addWidget(QLabel("margin"))
        prow2.addWidget(self.margin_spin)
        prow2.addStretch()
        prow2.addWidget(QLabel("export as"))
        prow2.addWidget(self.fmt_combo)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run panoptic + export")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        self.kill_btn = QPushButton("Kill")
        self.kill_btn.setVisible(False)
        self.kill_btn.clicked.connect(self._kill)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.kill_btn)
        run_row.addStretch()

        forms_col = QVBoxLayout()
        forms_col.addWidget(ibox)
        forms_col.addWidget(self.cbox)
        forms_col.addWidget(pbox)
        forms_col.addLayout(run_row)

        self.log = LogConsole()
        self.log.setPlaceholderText("Clustering and export logs appear here…")
        root.addWidget(ui.vsplit(ui.wrap(forms_col), self.log, sizes=[380, 300]), 1)

        self.clusterer.output.connect(self._append)
        self.clusterer.done.connect(self._on_clustered)
        self.clusterer.error.connect(self._on_error)
        self.clusterer.stopped.connect(lambda: self._finish("✗ stopped"))
        self.exporter.output.connect(self._append)
        self.exporter.done.connect(self._on_exported)
        self.exporter.error.connect(self._on_error)
        self.exporter.stopped.connect(lambda: self._finish("✗ stopped"))

        self.reload_preds()

    # --------------------------------------------------------------- discovery
    def _scan_preds(self) -> list[Path]:
        """Every predictions folder under <workspace>/inference (modal downloads
        nest the npz one level deeper)."""
        out = []
        root = appstate.workspace_dir() / "inference"
        if root.is_dir():
            for d in sorted(root.glob("*/predictions_*")):
                for p in (d, d / "predictions"):
                    if p.is_dir() and any(p.glob("*_pred.npz")):
                        out.append(p)
                        break
        return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)

    def reload_preds(self):
        if self._run_open:   # a live worker owns the npz files; don't touch them
            return
        root = appstate.workspace_dir() / "inference"
        items = []
        for p in self._scan_preds():
            try:
                label = str(p.relative_to(root))
            except ValueError:
                label = str(p)
            items.append((label, str(p)))
        ui.repopulate_combo(self.pred_combo, items)
        if self.pred_combo.currentIndex() < 0 and items:
            self.pred_combo.setCurrentIndex(0)
        self._on_pick()

    def _browse(self):
        d = QFileDialog.getExistingDirectory(
            self, "Predictions folder (contains *_pred.npz)",
            appstate.get("last_view_dir", str(appstate.workspace_dir())))
        if not d:
            return
        p = Path(d)
        if not any(p.glob("*_pred.npz")):
            self._append(f"✗ No *_pred.npz files in {p} - pick the predictions "
                         "folder an inference run wrote, or run Inference first.")
            return
        if self.pred_combo.findData(str(p)) < 0:
            self.pred_combo.addItem(str(p), str(p))
        self.pred_combo.setCurrentIndex(self.pred_combo.findData(str(p)))

    # ------------------------------------------------------------- class table
    def _on_pick(self):
        d = self.pred_combo.currentData()
        self._pred_dir = Path(d) if d else None
        self._build_class_table()

    def _classes(self) -> tuple[list[tuple[int, str]], str]:
        """[(model index, name)] for the picked folder + where the names came
        from; falls back to the first npz's label values."""
        names, src = _class_names_for(self._pred_dir)
        if names:
            return list(enumerate(names)), f"class names from {src}"
        first = next(iter(sorted(self._pred_dir.glob("*_pred.npz"))), None)
        if first is None:
            return [], ""
        with np.load(first) as npz:
            vals = np.unique(np.asarray(npz["classification"]))
        return [(int(v), f"class {v}") for v in vals], \
            f"no run manifest found - raw model indices from {first.name}"

    def _build_class_table(self):
        while self.cgrid.count():
            it = self.cgrid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._rows = []
        if not self._pred_dir:
            self.src_label.setText("No predictions found - run Inference first, "
                                   "or Browse… to a predictions folder.")
            return
        classes, src = self._classes()
        n_scenes = len(list(self._pred_dir.glob("*_pred.npz")))
        self.src_label.setText(f"{n_scenes} scene(s) · {src}")
        saved = appstate.get("panoptic_boxes", {})
        self.cgrid.addWidget(QLabel("length m"), 0, 1)
        self.cgrid.addWidget(QLabel("width m"), 0, 2)
        for row, (idx, name) in enumerate(classes, start=1):
            prefs = saved.get(name, {})
            chk = QCheckBox(name)
            # ground-ish classes are terrain, not countable objects
            chk.setChecked(bool(prefs.get("thing", "ground" not in name.lower())))
            spins = []
            for col, key in ((1, "l"), (2, "w")):
                s = QDoubleSpinBox()
                s.setRange(0.1, 500.0)
                s.setSingleStep(0.5)
                s.setValue(float(prefs.get(key, 2.0)))
                s.setToolTip("Typical object footprint; sets the clustering "
                             "distance and the split threshold.")
                self.cgrid.addWidget(s, row, col)
                spins.append(s)
            self.cgrid.addWidget(chk, row, 0)
            self._rows.append((idx, name, chk, spins[0], spins[1]))

    def _save_prefs(self):
        saved = appstate.get("panoptic_boxes", {})
        for _, name, chk, ls, ws in self._rows:
            saved[name] = {"thing": chk.isChecked(), "l": ls.value(), "w": ws.value()}
        appstate.put("panoptic_boxes", saved)

    # -------------------------------------------------------------------- run
    def _run(self):
        if self.clusterer.running or self.exporter.running:
            self._append("A panoptic run is already in progress.")
            return
        if not self._pred_dir:
            self._append("✗ Pick a predictions folder first - run Inference, "
                         "then come back to this page.")
            return
        things = {idx: (ls.value(), ws.value())
                  for idx, _, chk, ls, ws in self._rows if chk.isChecked()}
        if not things:
            self._append("✗ No thing classes checked - check at least one class "
                         "in the table to cluster it into instances.")
            return
        self._save_prefs()
        self._active_dir = self._pred_dir
        self.run_btn.setEnabled(False)
        self._begin_run(f"Panoptic - {self.pred_combo.currentText()}")
        names = {i: n for i, n, *_ in self._rows}
        self._append("ALPINE clustering (k=%d, split=%s, margin=%.2f) over: %s"
                     % (self.k_spin.value(), self.split_chk.isChecked(),
                        self.margin_spin.value(),
                        ", ".join(f"{names[i]} ({l:g}x{w:g} m)"
                                  for i, (l, w) in sorted(things.items()))))
        self.clusterer.start(panoptic.run_panoptic, str(self._active_dir), things,
                             k=self.k_spin.value(), split=self.split_chk.isChecked(),
                             margin=self.margin_spin.value())

    def _on_clustered(self, result):
        self._append(f"\n✓ {result['instances']:,} instances across "
                     f"{result['files']} scene(s).")
        fmt = self.fmt_combo.currentData()
        appstate.put("last_view_dir", str(self._active_dir))
        self._append(f"[export] rewriting predictions as {fmt} with instance ids…")
        self.exporter.start(dataset.export_predictions, self._active_dir, fmt,
                            class_map=self._class_map())

    def _run_manifest(self) -> dict | None:
        """The clustered folder's run.json: beside/above it (Modal downloads),
        in any known run root under the folder's run tag (local runs), or a
        sibling ensemble member's run dir (ensemble votes)."""
        pd = self._active_dir
        parent = pd.parent.parent if pd.name == "predictions" else pd.parent
        cands = [pd.parent, pd.parent.parent,
                 appstate.local_runs_dir() / "runs" / parent.name]
        cands += [Path(r) / parent.name for r in appstate.run_roots(self.repo_root)]
        cands += [d.parent for d in (appstate.workspace_dir() / "inference")
                  .glob(f"*/predictions_{_job_id(pd)}_m*")]
        for c in cands:
            m = appstate.read_json(c / "run.json")
            if m and m.get("dataset"):
                return m
        return None

    def _class_map(self) -> dict | None:
        """{model index: exported LAS code} via the run's dataset meta, using
        the Inference page's Output codes choice; None = raw indices."""
        m = self._run_manifest()
        mp = appstate.known_datasets().get((m or {}).get("dataset") or "",
                                           {}).get("meta_path", "")
        meta = appstate.read_json(Path(mp)) if mp else None
        mode = appstate.get("infer_codes", "asprs")
        return dataset.class_map_for_export(meta, mode, self._append, "this run")

    def _on_exported(self, written):
        if not written:
            self._finish("✗ nothing exported")
            return
        fmt = self.fmt_combo.currentData()
        how = ("'instance_id' extra dim" if fmt in ("las", "laz")
               else "an 'instance' column")
        self._append(f"\n✓ Done - {len(written)} file(s) in {written[0].parent} "
                     f"with {how} (0 = stuff).")
        self._finish(f"✓ {len(written)} panoptic file(s) exported")

    def _on_error(self, tb: str):
        self._append(f"\n✗ Panoptic step failed:\n{tb}")
        self._finish("✗ failed")

    def _kill(self):
        killed = False
        for w in (self.clusterer, self.exporter):
            if w.running:
                w.cancel()
                killed = True
        if not killed:
            self._append("\n[no process running]")

    # ------------------------------------------------------------ log framing
    def _begin_run(self, title: str):
        """Freeze the input surface: the worker owns the npz files, and the
        export callback must see the folder the run started on."""
        ui.begin_log_run(self, title)
        for w in (self.pred_combo, self.browse_btn, self.rescan_btn, self.cbox):
            w.setEnabled(False)
        self.kill_btn.setVisible(True)

    def _finish(self, summary: str):
        for w in (self.pred_combo, self.browse_btn, self.rescan_btn, self.cbox):
            w.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.kill_btn.setVisible(False)
        ui.end_log_run(self, summary)

    def _append(self, text: str, newline: bool = True):
        ui.append_log(self.log, text, newline)
