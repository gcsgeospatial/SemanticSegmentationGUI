"""Training terminal - entry point + main window (top tab bar over stacked pages)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QObject
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QFileDialog, QHBoxLayout,
                               QLabel, QMessageBox, QStackedWidget, QTabBar,
                               QVBoxLayout, QWidget)

# modal runs launch with cwd=REPO_ROOT so `modal run scripts/...` resolves
REPO_ROOT = str(Path(__file__).resolve().parents[1])

PAGES = ["Datasets", "Train", "Inference"]


class _NoWheelEdit(QObject):
    """Eat wheel events on spin boxes/combos - focused or not - so the wheel
    never mutates a value; type or use the arrows instead. ponytail: add
    QSlider if one becomes a scroll victim."""

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Wheel
                and isinstance(obj, (QAbstractSpinBox, QComboBox))):
            event.ignore()
            return True
        return super().eventFilter(obj, event)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Training Terminal")
        self.resize(1180, 800)
        self._restore_geometry()

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        top = QWidget()
        top.setObjectName("topbar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(14, 0, 14, 0)
        tl.setSpacing(10)
        self.nav = QTabBar()
        self.nav.setExpanding(False)
        self.nav.setUsesScrollButtons(False)
        for name in PAGES:
            self.nav.addTab(name)
        self.nav.currentChanged.connect(self._go)
        tl.addWidget(self.nav)
        tl.addStretch(1)

        # pages read appstate.get_exec_mode() at launch
        from . import appstate
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Modal (cloud)", "modal")
        self.mode_combo.addItem("Local (pixi)", "local")
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(appstate.get_exec_mode())))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        tl.addWidget(self.mode_combo)

        from . import theme
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.setCurrentIndex(
            max(0, self.theme_combo.findData(theme.current_mode())))
        self.theme_combo.currentIndexChanged.connect(self._on_theme_change)
        tl.addWidget(self.theme_combo)
        col.addWidget(top)

        self.stack = QStackedWidget()
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 12, 24, 12)
        cl.addWidget(self.stack)
        col.addWidget(content, 1)

        from . import ui
        from .pages.datasets_page import DatasetsPage
        from .pages.infer_page import InferPage
        from .pages.train_page import TrainPage

        self.datasets_page = DatasetsPage(REPO_ROOT)
        self.train_page = TrainPage(REPO_ROOT)
        self.infer_page = InferPage(REPO_ROOT)
        for page in (self.datasets_page, self.train_page, self.infer_page):
            ui.polish_forms(page)
            self.stack.addWidget(ui.scroll_v(page))

        ui.set_navigator(self._navigate)
        self._go(0)

    def _on_mode_change(self):
        from . import appstate
        mode = self.mode_combo.currentData()
        appstate.set_exec_mode(mode)
        local = mode == "local"
        for page in (self.datasets_page, self.train_page, self.infer_page):
            page.apply_exec_mode(local)

    def _on_theme_change(self):
        from . import appstate, theme
        mode = self.theme_combo.currentData()
        appstate.put("ui_theme", mode)
        theme.apply(QApplication.instance(), mode)

    def _navigate(self, page_name: str):
        """ui.navigate target: switch pages."""
        self.nav.setCurrentIndex(PAGES.index(page_name))

    def _go(self, row: int):
        # name-keyed so inserting a page never silently shifts the refreshes
        refresh = {"Train": lambda: self.train_page.reload_datasets(),
                   "Inference": lambda: self.infer_page.reload_runs()}.get(PAGES[row])
        if refresh:
            refresh()
        self.stack.setCurrentIndex(row)

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


def _app_icon() -> QIcon:
    """icon.png from the repo root or package."""
    here = Path(__file__).resolve()
    for base in (here.parent, here.parents[1], here.parents[2]):
        p = base / "icon.png"
        if p.exists():
            return QIcon(str(p))
    return QIcon()


def _ensure_workspace(parent=None) -> None:
    """First launch only: ask for the workspace root; cancel falls back to staging."""
    from . import appstate
    if appstate.get("workspace"):
        return
    d = QFileDialog.getExistingDirectory(
        parent, "Choose a workspace folder (datasets, training runs, and inference live here)",
        str(appstate.staging_dir()))
    appstate.set_workspace(d or str(appstate.staging_dir()))


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
    # cores-2 before any numpy/pgeof/PDAL import: native libs otherwise grab
    # every core at normal priority and starve the UI event loop
    cores = os.cpu_count() or 4
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(max(cores - 2, 1)))
    from . import appstate, theme
    if sys.platform == "win32":
        # own taskbar identity so Windows shows icon.png, not python.exe's
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("trainer_gui")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.installEventFilter(_NoWheelEdit(app))
    app.setApplicationName("trainer_gui")
    app.setWindowIcon(_app_icon())
    theme.apply(app)
    _ensure_workspace()   # set the workspace BEFORE pages read it for their defaults
    win = MainWindow()
    win.show()
    bad = appstate.state_quarantine()
    if bad:
        QMessageBox.warning(
            win, "Settings were corrupt",
            f"state.json could not be read and was renamed to:\n{bad}\n\n"
            "The app started with empty settings: saved datasets, the workspace "
            "and preferences are gone.\n\n"
            "Re-add each dataset folder on the Datasets page (Add existing…), "
            "or copy values back out of the renamed file by hand.")
    _check_modal_cli(win)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
