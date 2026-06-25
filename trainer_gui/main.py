"""Training terminal — entry point + main window (sidebar nav over stacked pages)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QStackedWidget, QVBoxLayout, QWidget)

# Repo root = the project dir holding the modal_train_*.py scripts (one level up
# from this trainer_gui/ package). modal runs are launched with cwd=REPO_ROOT, so
# this must be where `modal run modal_train_*.py` can find the script.
#   .../<repo>/trainer_gui/main.py  -> parents[0]=trainer_gui pkg, parents[1]=<repo>
REPO_ROOT = str(Path(__file__).resolve().parents[1])

STYLE = """
QWidget { font-size: 14px; }
#sidebar { background: #1f2430; }
#sidebar QListWidget { background: #1f2430; color: #c8cdd6; border: none; outline: none; }
#sidebar QListWidget::item { padding: 12px 18px; }
#sidebar QListWidget::item:selected { background: #323a4d; color: #ffffff; }
#sidebar QListWidget::item:disabled { color: #5b6273; }
#brand { color: #ffffff; font-size: 18px; font-weight: 600; padding: 18px 18px 6px 18px; }
#brandSub { color: #7f8696; padding: 0 18px 14px 18px; }
#modeLabel { color: #7f8696; padding: 4px 18px 2px 18px; font-size: 12px; }
#sidebar QComboBox { background: #2a3040; color: #d6dae3; border: 1px solid #3a4252;
                     border-radius: 4px; padding: 4px 8px; margin: 0 18px 12px 18px; }
#sidebar QComboBox QAbstractItemView { background: #2a3040; color: #d6dae3;
                                       selection-background-color: #3b6cf6; }
#pageTitle { font-size: 22px; font-weight: 600; }
#pageSub { color: #5b6273; margin-bottom: 8px; }
#log { font-family: Consolas, "Courier New", monospace; font-size: 12px;
       background: #11141b; color: #d6dae3; }
QPushButton { padding: 7px 14px; border-radius: 5px; border: 1px solid #c2c8d2; background: #fff; }
QPushButton:hover { background: #f0f2f6; }
QPushButton#primary { background: #3b6cf6; color: #fff; border: none; font-weight: 600; }
QPushButton#primary:hover { background: #2f59d6; }
QPushButton#primary:disabled { background: #9bb0ee; }
QGroupBox { font-weight: 600; margin-top: 10px; border: 1px solid #e1e4ea;
            border-radius: 6px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QSplitter::handle { background: #e7eaf0; border-radius: 3px; }
QSplitter::handle:hover { background: #b9c2d4; }
QSplitter::handle:vertical { height: 8px; margin: 1px 60px; }
QSplitter::handle:horizontal { width: 8px; margin: 60px 1px; }
"""

PAGES = ["Datasets", "Train", "Inference", "Plotting"]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Terminal")
        self.resize(1180, 800)
        self._restore_geometry()

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(220)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        brand = QLabel("Training Terminal")
        brand.setObjectName("brand")
        sl.addWidget(brand)
        self.tag = QLabel()
        self.tag.setObjectName("brandSub")
        self.tag.setWordWrap(True)
        sl.addWidget(self.tag)

        # Execution backend toggle — the seamless Modal <-> Local switch. Pages
        # read appstate.get_exec_mode() when they launch, so flipping this just
        # changes where the next Train / Inference run goes.
        from . import appstate
        mode_label = QLabel("Execution backend")
        mode_label.setObjectName("modeLabel")
        sl.addWidget(mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Modal (cloud)", "modal")
        self.mode_combo.addItem("Local (Docker)", "local")
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(appstate.get_exec_mode())))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        sl.addWidget(self.mode_combo)
        self._apply_mode_tag(appstate.get_exec_mode())

        self.nav = QListWidget()
        for name in PAGES:
            self.nav.addItem(QListWidgetItem(name))
        self.nav.currentRowChanged.connect(self._go)
        sl.addWidget(self.nav, 1)
        row.addWidget(side)

        self.stack = QStackedWidget()
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 18, 24, 18)
        cl.addWidget(self.stack)
        row.addWidget(content, 1)

        from . import ui
        from .pages.datasets_page import DatasetsPage
        from .pages.infer_page import InferPage
        from .pages.plotting_page import PlottingPage
        from .pages.train_page import TrainPage

        self.datasets_page = DatasetsPage(REPO_ROOT)
        self.train_page = TrainPage(REPO_ROOT)
        self.plotting_page = PlottingPage(REPO_ROOT)
        self.infer_page = InferPage(REPO_ROOT)
        # Local backbone selection (Train page checkboxes) also filters Inference.
        self.train_page.models_changed.connect(self.infer_page.reload_backbones)
        # Each page scrolls vertically instead of being crammed into the window.
        for page in (self.datasets_page, self.train_page,
                     self.infer_page, self.plotting_page):
            self.stack.addWidget(ui.scroll_v(page))

        self.nav.setCurrentRow(0)

    def _on_mode_change(self):
        from . import appstate
        mode = self.mode_combo.currentData()
        appstate.set_exec_mode(mode)
        self._apply_mode_tag(mode)
        # Re-scheme every page for the chosen backend (hide Modal-only controls,
        # drop built-ins, reword copy).
        local = mode == "local"
        for page in (self.datasets_page, self.train_page, self.infer_page):
            page.apply_exec_mode(local)

    def _apply_mode_tag(self, mode: str):
        self.tag.setText("point-cloud training — local (Docker)" if mode == "local"
                         else "point-cloud training on Modal")

    def _go(self, row: int):
        # PAGES = [Datasets, Train, Inference, Plotting]
        if row == 1:
            self.train_page.reload_datasets()
        elif row == 2:
            self.infer_page.reload_runs()
        elif row == 3:
            self.plotting_page._rescan()
        self.stack.setCurrentIndex(row)

    # Window size/position persists across sessions (stored in appstate JSON).
    def _restore_geometry(self):
        from . import appstate
        geo = appstate.get("window_geometry")
        if geo:
            try:
                self.restoreGeometry(QByteArray.fromBase64(geo.encode("ascii")))
            except Exception:
                pass

    def closeEvent(self, event):
        from . import appstate
        appstate.put("window_geometry",
                     bytes(self.saveGeometry().toBase64()).decode("ascii"))
        super().closeEvent(event)


def _check_modal_cli(parent=None) -> bool:
    if shutil.which("modal"):
        return True
    QMessageBox.warning(
        parent, "Modal CLI not found",
        "The `modal` command was not found on PATH.\n\n"
        "Install it with:  pip install modal\n"
        "then authenticate:  modal token new\n\n"
        "The app will open, but launching jobs will fail until Modal is installed.")
    return False


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("trainer_gui")
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    _check_modal_cli(win)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
