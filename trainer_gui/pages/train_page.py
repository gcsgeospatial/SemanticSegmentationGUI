"""Train page: dataset + backbone + recommended params -> modal run, live logs."""

from __future__ import annotations

import json
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import analysis, appstate, local_cli, modal_cli, prep, ui
from ..backbones import BACKBONES, GPU_CHOICES
from ..jobs import FuncWorker, JobRunner, LogParser


class TrainPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root
        self.runner = JobRunner(self)
        self.log_runner = JobRunner(self)   # for re-attaching to detached runs
        self.prep_worker = FuncWorker(self)  # local tiling/subsampling
        self.prep_uploader = JobRunner(self)
        self.verify_worker = FuncWorker(self)  # `modal volume ls` presence check
        self.parser = LogParser(self)
        self._param_widgets: dict[str, QWidget] = {}
        self._meta: dict | None = None
        self._last_run_id: str | None = None
        self._pending: dict | None = None   # launch args while prep/upload run

        root = QVBoxLayout(self)
        title = QLabel("Train")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        self.sub.setObjectName("pageSub")
        root.addWidget(self.sub)

        form_box = QGroupBox("Job")
        form = self.form = QFormLayout(form_box)
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_change)
        ds_row = QHBoxLayout()
        ds_row.addWidget(self.dataset_combo, 1)
        self.verify_btn = QPushButton("Check on Modal")
        self.verify_btn.clicked.connect(self._verify_dataset)
        ds_row.addWidget(self.verify_btn)
        form.addRow("Dataset", _wrap(ds_row))
        self.ds_status = QLabel("")
        self.ds_status.setWordWrap(True)
        self.ds_status.setStyleSheet("color: #6a6a6a;")
        form.addRow("", self.ds_status)
        self.backbone_combo = QComboBox()  # populated per-dataset in _reload_backbones
        self.backbone_combo.currentIndexChanged.connect(self._rebuild_params)
        form.addRow("Model", self.backbone_combo)
        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(GPU_CHOICES)
        self.gpu_combo.setCurrentText("A100")
        form.addRow("GPU", self.gpu_combo)
        opts_row = QHBoxLayout()
        self.detach_chk = QCheckBox("Detached (survives closing the app)")
        self.detach_chk.setChecked(True)
        self.smoke_chk = QCheckBox("Smoke run (2 epochs × 50 steps on A10G)")
        self.smoke_chk.toggled.connect(self._apply_smoke)
        opts_row.addWidget(self.detach_chk)
        opts_row.addWidget(self.smoke_chk)
        opts_row.addStretch()
        form.addRow("Options", _wrap(opts_row))
        self.prep_chk = QCheckBox("Prep tiles locally + upload (no Modal CPU time "
                                  "spent on preprocessing)")
        self.prep_chk.setChecked(True)
        form.addRow("", self.prep_chk)

        self.params_box = QGroupBox("Parameters (recommended values pre-filled)")
        self.params_form = QFormLayout(self.params_box)
        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color: #b25f00;")

        run_row = QHBoxLayout()
        self.launch_btn = QPushButton("Launch training")
        self.launch_btn.setObjectName("primary")
        self.launch_btn.clicked.connect(self._launch)
        run_row.addWidget(self.launch_btn)
        self.attach_btn = QPushButton("Re-attach to logs")
        self.attach_btn.clicked.connect(self._reattach)
        run_row.addWidget(self.attach_btn)
        self.stop_btn = QPushButton("Stop local process")
        self.stop_btn.clicked.connect(self._stop)
        run_row.addWidget(self.stop_btn)
        run_row.addStretch()

        # Config column (scrolls when squeezed so the log can take the room).
        config_col = QVBoxLayout()
        config_col.addWidget(form_box)
        config_col.addWidget(self.params_box)
        config_col.addWidget(self.warn_label)
        config_col.addLayout(run_row)
        config_col.addStretch()

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("log")
        self.log.setPlaceholderText("Modal logs appear here…")

        metrics_col = QVBoxLayout()
        metrics_col.addWidget(QLabel("Live epoch metrics"))
        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(["Epoch", "Loss", "Acc", "mIoU"])
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        metrics_col.addWidget(self.metrics_table, 1)

        # Drag the handles: config | (log | metrics) are all resizable.
        body = ui.hsplit(self.log, ui.wrap(metrics_col), sizes=[680, 340])
        root.addWidget(ui.vsplit(ui.scrollable(ui.wrap(config_col)), body,
                                 sizes=[400, 360]), 1)

        self.runner.output.connect(self._on_output)
        self.runner.finished.connect(self._on_finished)
        self.runner.failed.connect(self._on_runner_failed)
        self.log_runner.output.connect(self._on_output)
        self.prep_worker.output.connect(self._append)
        self.prep_worker.done.connect(self._on_prepped)
        self.prep_worker.error.connect(self._on_prep_error)
        self.prep_uploader.output.connect(lambda s: self._append(s, newline=False))
        self.prep_uploader.finished.connect(self._on_prep_uploaded)
        self.verify_worker.done.connect(self._on_verified)
        self.verify_worker.error.connect(lambda tb: self._set_ds_status(
            "Could not reach Modal — is the CLI authenticated? (modal token new)", "#b25f00"))
        self.parser.epoch.connect(self._on_epoch)
        self.parser.run_id.connect(self._on_run_id)

        self.apply_exec_mode(appstate.get_exec_mode() == "local")
        self._rebuild_params()

    def apply_exec_mode(self, local: bool):
        """Hide Modal-only controls + reword copy for the local (Docker) backend."""
        self.form.setRowVisible(self.gpu_combo, not local)   # GPU type is a Modal pick
        self.form.setRowVisible(self.prep_chk, not local)    # prep+upload is Modal-only
        self.detach_chk.setVisible(not local)                # Modal detach
        self.attach_btn.setVisible(not local)                # modal app logs
        self.verify_btn.setVisible(not local)                # Check on Modal volume
        self.smoke_chk.setText("Smoke run (2 epochs × 50 steps)" if local
                               else "Smoke run (2 epochs × 50 steps on A10G)")
        self.sub.setText(
            "Pick a dataset and a model. Parameters are pre-filled from the dataset's "
            "density analysis — edit anything before launching. "
            + ("Runs execute locally in Docker on your GPU."
               if local else
               "Runs execute on Modal; detached runs keep going if you close this app."))
        self.reload_datasets()

    # ------------------------------------------------------------- datasets
    def reload_datasets(self):
        current = self.dataset_combo.currentText()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for name in sorted(appstate.selectable_datasets()):
            self.dataset_combo.addItem(name)
        self.dataset_combo.blockSignals(False)
        if current:
            i = self.dataset_combo.findText(current)
            if i >= 0:
                self.dataset_combo.setCurrentIndex(i)
        self._on_dataset_change()

    def _on_dataset_change(self):
        self._meta = None
        name = self.dataset_combo.currentText()
        info = appstate.known_datasets().get(name, {})
        meta_path = info.get("meta_path", "")
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
        # Show what we know locally; the actual Modal volume is only confirmed by
        # the "Check on Modal" button (the local flag can be stale).
        if not name:
            self._set_ds_status("")
        elif info.get("builtin"):
            self._set_ds_status("Built-in dataset (lives on the ieee-data volume).")
        elif info.get("uploaded"):
            self._set_ds_status("Marked uploaded locally — click “Check on Modal” to confirm.")
        else:
            self._set_ds_status("Not uploaded yet — upload it on the Datasets page first.",
                                "#b25f00")
        self.verify_btn.setEnabled(bool(name) and not info.get("builtin"))
        self._reload_backbones()

    def _set_ds_status(self, text: str, color: str = "#6a6a6a"):
        self.ds_status.setStyleSheet(f"color: {color};")
        self.ds_status.setText(text)

    def _verify_dataset(self):
        """Actually list the dataset's Modal volume so the user can confirm the
        upload is present rather than trusting the local 'uploaded' flag."""
        name = self.dataset_combo.currentText()
        info = appstate.known_datasets().get(name, {})
        if not name or info.get("builtin"):
            return
        if self.verify_worker.running:
            return
        vol = info.get("volume", name)
        self.verify_btn.setEnabled(False)
        self._set_ds_status(f"Checking volume “{vol}” on Modal …")
        self.verify_worker.start(_check_dataset_present, vol, name)

    def _on_verified(self, res: dict):
        self.verify_btn.setEnabled(True)
        vol, n = res["volume"], res["scenes"]
        if res["has_meta"]:
            self._set_ds_status(
                f"✓ Present on Modal volume “{vol}”: dataset_meta.json"
                + (f" + {n} train scene(s)." if res["has_train"] else
                   " but no train/ folder — re-upload on the Datasets page."),
                "#2e7d32" if res["has_train"] else "#b25f00")
        else:
            self._set_ds_status(
                f"✗ Not found on Modal volume “{vol}” — upload it on the Datasets page "
                f"before training.", "#b25f00")

    def _apply_smoke(self):
        """Reflect the smoke override in the form so it's visible, not silent:
        lock epochs=2 / steps=50 / GPU=A10G while the box is checked."""
        on = self.smoke_chk.isChecked()
        self.gpu_combo.setEnabled(not on)
        if on:
            self.gpu_combo.setCurrentText("A10G")
        for flag, val in (("epochs", 2), ("steps-per-epoch", 50)):
            w = self._param_widgets.get(flag)
            if w is not None:
                if on:
                    w.setValue(val)
                w.setEnabled(not on)

    def _reload_backbones(self):
        """Populate the model dropdown. Built-in IEEE datasets restrict it to the
        scripts whose no-`--dataset` default trains on that data."""
        allowed = appstate.known_datasets().get(
            self.dataset_combo.currentText(), {}).get("backbones")
        prev = self.backbone_combo.currentData()
        self.backbone_combo.blockSignals(True)
        self.backbone_combo.clear()
        for key, b in BACKBONES.items():
            if allowed and key not in allowed:
                continue
            self.backbone_combo.addItem(
                b.label + ("" if b.ready else "  (script not wired yet)"), key)
        i = self.backbone_combo.findData(prev)
        if i >= 0:
            self.backbone_combo.setCurrentIndex(i)
        self.backbone_combo.blockSignals(False)
        self._rebuild_params()

    # ------------------------------------------------------------- params form
    def _backbone(self):
        return BACKBONES[self.backbone_combo.currentData()]

    def _rebuild_params(self):
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_widgets.clear()
        b = self._backbone()
        recs = (self._meta or {}).get("recommendations", {}).get(b.key, {})
        for spec in b.params:
            value = recs.get(spec.recommend_key, spec.default) if spec.recommend_key else spec.default
            if spec.kind == "float":
                w = QDoubleSpinBox()
                w.setDecimals(spec.decimals)
                w.setSingleStep(spec.step)
                w.setRange(spec.lo, spec.hi)
                w.setValue(float(value))
            else:
                w = QSpinBox()
                w.setRange(int(spec.lo), int(spec.hi))
                w.setValue(int(value))
            label = spec.label + ("  ★" if spec.recommend_key and spec.recommend_key in recs else "")
            self.params_form.addRow(label, w)
            self._param_widgets[spec.flag] = w
        if self._meta:
            warns = analysis.warnings_for(self._meta)
            self.warn_label.setText("\n".join("⚠ " + w for w in warns))
        else:
            self.warn_label.setText("")
        self._apply_smoke()   # re-lock epochs/steps if a smoke run is selected

    # ------------------------------------------------------------- launch
    def _launch(self):
        b = self._backbone()
        if not b.ready:
            self._append(f"{b.label} isn't wired for CLI args yet — pick a ready model.")
            return
        name = self.dataset_combo.currentText()
        if not name:
            self._append("Create + upload a dataset on the Datasets page first.")
            return
        if self.runner.running or self.prep_worker.running or self.prep_uploader.running:
            self._append("A local process is already running.")
            return

        info = appstate.known_datasets().get(name, {})
        # Built-in IEEE datasets run the script's no-`--dataset` default (real data
        # already on the ieee-data volume — real HAG for the HAG variants).
        flags = {} if info.get("builtin") else {"dataset": name}
        for spec in b.params:
            w = self._param_widgets[spec.flag]
            flags[spec.flag] = w.value()
        gpu = self.gpu_combo.currentText()
        if self.smoke_chk.isChecked():
            flags["epochs"] = 2
            flags["steps-per-epoch"] = 50
            gpu = "A10G"

        self.log.clear()
        self.metrics_table.setRowCount(0)
        self._last_run_id = None
        self.launch_btn.setEnabled(False)
        self._pending = {"backbone": b, "flags": flags, "gpu": gpu, "dataset": name}

        # Optional local prep: tile/subsample here and upload the cache so the
        # GPU container finds everything already preprocessed and skips it.
        # Built-ins already have their prep on the ieee-data volume — skip it.
        # Local mode skips prep+upload entirely: the staged dataset IS the data,
        # bind-mounted straight into the container at /datasets.
        if (appstate.get_exec_mode() == "modal" and self.prep_chk.isChecked()
                and not info.get("builtin")):
            staged = appstate.known_datasets().get(name, {}).get("staged_dir", "")
            if not prep.supports_local_prep(b.key):
                self._append(f"[prep] {b.label} has no local prep path — the script "
                             f"will preprocess remotely.")
            elif not staged or not os.path.isdir(staged):
                self._append("[prep] No local staged copy of this dataset on this "
                             "machine — the script will preprocess remotely.")
            else:
                self._append(f"[prep] Building {b.label} cache locally from {staged} …")
                self.prep_worker.start(prep.prep_dataset, b.key, staged, dict(flags))
                return

        self._start_run()

    def _on_prepped(self, prep_dir):
        p = self._pending
        if p is None:
            return
        tag = prep_dir.name
        vol = p["dataset"]              # per-dataset volume named after the dataset
        remote = f"/{vol}/prep/{tag}"
        self._append(f"[prep] Uploading cache -> {vol}:{remote} …")
        prog, args = modal_cli.volume_put(vol, str(prep_dir), remote)
        self.prep_uploader.start(prog, args, cwd=self.repo_root,
                                 pre=modal_cli.volume_create(vol))

    def _on_prep_error(self, tb: str):
        self._append(f"\n[prep] Local prep failed — falling back to remote "
                     f"preprocessing.\n{tb}")
        self._start_run()

    def _on_prep_uploaded(self, code: int):
        if code != 0:
            self._append(f"[prep] Upload failed (exit {code}) — the script will "
                         f"preprocess remotely instead.")
        else:
            self._append("[prep] ✓ Cache uploaded — remote preprocessing will be skipped.")
        self._start_run()

    def _start_run(self):
        """Dispatch the pending launch to Modal (cloud) or Docker (local)."""
        p, self._pending = self._pending, None
        if p is None:
            self.launch_btn.setEnabled(True)
            return
        if appstate.get_exec_mode() == "local":
            self._start_local_run(p)
        else:
            self._start_modal_run(p)

    def _start_modal_run(self, p):
        prog, args = modal_cli.run_script(p["backbone"].script, p["flags"],
                                          detach=self.detach_chk.isChecked())
        extra_env = {"TT_GPU": p["gpu"]}
        # User datasets live in their own volume (= dataset name); tell the script
        # to mount it at /datasets. Built-ins (no "dataset" flag) leave it unset, so
        # the script falls back to terminal-datasets (also used by inference).
        ds_vol = p["flags"].get("dataset")
        if ds_vol:
            extra_env["TT_DATASET_VOLUME"] = ds_vol
        self._append(f"\n$ modal {' '.join(args)}   [TT_GPU={p['gpu']}"
                     f"{', TT_DATASET_VOLUME=' + ds_vol if ds_vol else ''}]\n")
        self.runner.start(prog, args, cwd=self.repo_root, extra_env=extra_env)

    def _start_local_run(self, p):
        b, flags, gpu, name = p["backbone"], p["flags"], p["gpu"], p["dataset"]
        info = appstate.known_datasets().get(name, {})
        extra_mounts = []
        if info.get("builtin"):
            # Built-ins read raw data from /data/IEEE — only the user can supply it.
            if not appstate.local_config().get("data_root"):
                self._append("[local] ⚠ Built-in IEEE training reads /data/IEEE — set "
                             "local_config['data_root'] (host → /data) or the run will fail.")
        else:
            # Mount the dataset's real staged dir at /datasets/<name> so local training
            # works wherever it was converted (independent of datasets_root).
            staged = info.get("staged_dir", "")
            if staged and os.path.isdir(staged):
                extra_mounts.append((staged, f"/datasets/{name}"))
            else:
                self._append(f"[local] ⚠ No local staged copy of '{name}' on this machine — "
                             f"the container won't find /datasets/{name}.")
        prog, args = local_cli.run_script(b.script, flags, b, repo_root=self.repo_root,
                                          gpu=gpu, extra_mounts=extra_mounts)
        self._append(f"\n[local] $ {local_cli.preview(prog, args)}\n")
        if not local_cli.have_docker():
            self._append(
                "[local] docker not found on PATH — printed the exact command instead of "
                "running it (design-now mode). On a Docker+GPU host, build the images with "
                "docker/build_all script, then launch: training writes to "
                f"{appstate.local_runs_dir().as_posix()}/runs/<id> (no upload/download).")
            self.launch_btn.setEnabled(True)
            return
        ok, msg = local_cli.image_preflight(b)
        if msg:
            self._append(msg)
        if not ok:
            self.launch_btn.setEnabled(True)
            return
        self.runner.start(prog, args, cwd=self.repo_root)

    def _on_runner_failed(self, err: str):
        # QProcess FailedToStart fires `failed`, not `finished`, so re-enable the
        # button here (otherwise a bad docker/modal exec wedges the UI).
        self.launch_btn.setEnabled(True)
        self._append(f"\n✗ Failed to start process: {err}")

    def _reattach(self):
        b = self._backbone()
        if self.log_runner.running:
            self.log_runner.terminate()
            return
        self._append(f"\n$ modal app logs {b.app_name}\n")
        prog, args = modal_cli.app_logs(b.app_name)
        self.log_runner.start(prog, args, cwd=self.repo_root)

    def _stop(self):
        stopped = False
        for r in (self.runner, self.log_runner):
            if r.running:
                r.terminate()
                stopped = True
        self._append("\n[stopped local process — a detached Modal run keeps going; "
                     "use `modal app stop <app>` to kill it remotely]" if stopped
                     else "\n[no local process running]")

    # ------------------------------------------------------------- stream
    def _on_output(self, text: str):
        self._append(text, newline=False)
        self.parser.feed(text)

    def _on_finished(self, code: int):
        self.launch_btn.setEnabled(True)
        if code == 0:
            extra = (f" Run id: {self._last_run_id}." if self._last_run_id else "")
            self._append(f"\n✓ Process finished.{extra} See the Runs page for artifacts.")
        else:
            self._append(f"\n✗ Process exited with code {code}.")

    def _on_epoch(self, m: dict):
        r = self.metrics_table.rowCount()
        self.metrics_table.insertRow(r)
        for col, val in enumerate((str(m["epoch"]), f"{m['loss']:.4f}",
                                   f"{m['acc']:.4f}", f"{m['miou']:.4f}")):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.metrics_table.setItem(r, col, item)
        self.metrics_table.scrollToBottom()

    def _on_run_id(self, run_id: str):
        self._last_run_id = run_id
        history = appstate.get("run_history", [])
        history.append({"run_id": run_id, "backbone": self._backbone().key,
                        "dataset": self.dataset_combo.currentText()})
        appstate.put("run_history", history[-200:])

    def _append(self, text: str, newline: bool = True):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text + ("\n" if newline else ""))
        self.log.moveCursor(QTextCursor.End)


def _wrap(layout) -> QWidget:
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w


def _entry_name(entry: dict) -> str:
    """Basename of a `modal volume ls --json` entry (key name varies by CLI ver)."""
    for k in ("path", "Filename", "filename", "name", "Name"):
        v = entry.get(k)
        if v:
            return str(v).rstrip("/").rsplit("/", 1)[-1]
    return ""


def _check_dataset_present(volume: str, name: str, progress=None) -> dict:
    """List the dataset's Modal volume to confirm the upload is really there.
    The training script reads <volume>:/<name>/dataset_meta.json + /train, so
    those are what we look for. Runs in a worker thread (blocking subprocess)."""
    root = modal_cli.list_volume_entries(volume, f"/{name}")
    names = {_entry_name(e) for e in root}
    train = modal_cli.list_volume_entries(volume, f"/{name}/train") if "train" in names else []
    return {"volume": volume, "name": name,
            "has_meta": "dataset_meta.json" in names,
            "has_train": "train" in names,
            "scenes": sum(1 for e in train if _entry_name(e).endswith(".npz"))}
