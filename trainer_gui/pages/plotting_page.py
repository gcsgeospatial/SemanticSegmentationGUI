"""Plotting page: chart per-epoch validation metrics from run folders.

Select runs, pick a metric (val mIoU / accuracy / class IoU); the chart overlays
each run plus their mean ± std band. The toolbar saves a PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout,
                               QWidget)

from .. import appstate, plots, ui


class PlottingPage(QWidget):
    def __init__(self, repo_root: str):
        super().__init__()
        self.repo_root = repo_root

        root = QVBoxLayout(self)
        title = QLabel("Plotting")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        sub = QLabel("Chart per-epoch validation metrics across runs. The bold line is their "
                     "average (± std). Use the toolbar to zoom or save a PNG.")
        sub.setWordWrap(True)
        sub.setObjectName("pageSub")
        root.addWidget(sub)

        # ---- left: run list + controls
        left = QVBoxLayout()
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add runs folder…")
        add_btn.clicked.connect(self._add_folder)
        refresh_btn = QPushButton("Rescan")
        refresh_btn.clicked.connect(self._rescan)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(refresh_btn)
        left.addLayout(btn_row)

        left.addWidget(QLabel("Runs (ctrl/shift-click to compare)"))
        self.run_list = QListWidget()
        self.run_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.run_list.setMinimumHeight(220)
        self.run_list.itemSelectionChanged.connect(self._on_selection)
        left.addWidget(self.run_list, 1)

        self.metric_combo = QComboBox()
        self.metric_combo.currentIndexChanged.connect(self._redraw)
        left.addWidget(QLabel("Metric"))
        left.addWidget(self.metric_combo)

        self.show_runs_chk = QCheckBox("Show individual runs")
        self.show_runs_chk.setChecked(True)
        self.show_runs_chk.toggled.connect(self._redraw)
        self.show_avg_chk = QCheckBox("Show average (mean ± std)")
        self.show_avg_chk.setChecked(True)
        self.show_avg_chk.toggled.connect(self._redraw)
        left.addWidget(self.show_runs_chk)
        left.addWidget(self.show_avg_chk)

        # Final test metrics (test_metrics.json) for a single selected run.
        self.test_label = QLabel("")
        self.test_label.setWordWrap(True)
        self.test_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.test_label.setVisible(False)
        left.addWidget(self.test_label)

        # ---- right: embedded chart
        self.fig = plots.Figure(figsize=(9, 5.5))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(480)
        right = QVBoxLayout()
        right.addWidget(NavToolbar(self.canvas, self))
        right.addWidget(self.canvas, 1)

        root.addWidget(ui.hsplit(ui.wrap(left), ui.wrap(right), sizes=[340, 760]), 1)

        self._rescan()

    # ------------------------------------------------------------- run discovery
    def _default_roots(self) -> list[Path]:
        """The ONE run-discovery source shared with the Inference picker:
        appstate.run_roots (dataset-nested runs/, downloaded runs, Runs-page
        layout, repo runs/, legacy local_train_out)."""
        return appstate.run_roots(self.repo_root)

    def _rescan(self):
        """Reload the list from the default roots, keeping any user-added folders."""
        extra = appstate.get("plot_extra_roots", [])
        self._populate([*self._default_roots(), *(Path(p) for p in extra)])

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder containing run(s)")
        if not d:
            return
        extra = appstate.get("plot_extra_roots", [])
        if d not in extra:
            extra.append(d)
            appstate.put("plot_extra_roots", extra)
        self._rescan()

    def _populate(self, roots):
        selected = {i.data(Qt.UserRole) for i in self.run_list.selectedItems()}
        self.run_list.blockSignals(True)
        self.run_list.clear()
        seen: set[str] = set()
        for rootp in roots:
            for run_dir in plots.discover_runs(rootp):
                key = str(run_dir)
                if key in seen:
                    continue
                seen.add(key)
                item = QListWidgetItem(plots.run_label(run_dir))
                item.setData(Qt.UserRole, key)
                self.run_list.addItem(item)
                if key in selected:
                    item.setSelected(True)
        self.run_list.blockSignals(False)
        self._refresh_metrics()
        self._update_test_summary()
        self._redraw()

    # ------------------------------------------------------------- metric choices
    def _selected_dirs(self) -> list[Path]:
        return [Path(i.data(Qt.UserRole)) for i in self.run_list.selectedItems()]

    def _refresh_metrics(self):
        """Union of metrics across the selected runs (or all, if none selected)."""
        dirs = self._selected_dirs() or [Path(self.run_list.item(i).data(Qt.UserRole))
                                          for i in range(self.run_list.count())]
        keys: list[str] = []
        for d in dirs:
            for m in plots.available_metrics(d):
                if m not in keys:
                    keys.append(m)
        keys = keys or ["val_miou"]
        current = self.metric_combo.currentData()
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for k in keys:
            self.metric_combo.addItem(plots.metric_label(k), k)
        i = self.metric_combo.findData(current)
        self.metric_combo.setCurrentIndex(i if i >= 0 else 0)
        self.metric_combo.blockSignals(False)

    def _on_selection(self):
        self._refresh_metrics()
        self._update_test_summary()
        self._redraw()

    # ------------------------------------------------------------- final test metrics
    def _update_test_summary(self):
        """Compact read-only summary of test_metrics.json when one run is selected."""
        dirs = self._selected_dirs()
        text = ""
        if len(dirs) == 1:
            tm = plots.read_test_metrics(dirs[0])
            parts = []
            for split in ("val", "test"):
                m = tm.get(split) or {}
                bits = [f"acc {m['overall_acc']:.3f}" if m.get("overall_acc") is not None else "",
                        f"mIoU {m['overall_mIoU']:.3f}" if m.get("overall_mIoU") is not None else ""]
                pc = m.get("per_class_iou") or {}
                if pc:
                    bits.append(", ".join(f"{c} {v:.2f}" for c, v in pc.items()))
                bits = [b for b in bits if b]
                if bits:
                    parts.append(f"final {split}: " + " · ".join(bits))
            text = "\n".join(parts)
        self.test_label.setText(text)
        self.test_label.setVisible(bool(text))

    # ------------------------------------------------------------- navigation target
    def receive_nav(self, run=None, **_):
        """navigate("Plotting", run=<run dir or id>) preselects that run."""
        if not run:
            return
        want = str(run)
        want_name = Path(want).name
        self.run_list.clearSelection()
        for i in range(self.run_list.count()):
            item = self.run_list.item(i)
            key = item.data(Qt.UserRole)
            if key == want or Path(key).name == want_name:
                item.setSelected(True)
                self.run_list.scrollToItem(item)
                break

    # ------------------------------------------------------------- draw
    def _redraw(self):
        metric = self.metric_combo.currentData() or "val_miou"
        plots.multi_run_figure(self._selected_dirs(), metric,
                               show_runs=self.show_runs_chk.isChecked(),
                               show_avg=self.show_avg_chk.isChecked(),
                               fig=self.fig)
        self.canvas.draw_idle()
