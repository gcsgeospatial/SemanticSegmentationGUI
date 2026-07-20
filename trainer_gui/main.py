"""Training terminal — entry point + main window (sidebar nav over stacked pages)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QObject, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QFileDialog, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMessageBox, QStackedWidget,
                               QVBoxLayout, QWidget)

# Repo root = the project dir holding scripts/ (one level up from this trainer_gui/
# package). modal runs are launched with cwd=REPO_ROOT, so this must be where
# `modal run scripts/modal/modal_train_*.py` can find the script.
#   .../<repo>/trainer_gui/main.py  -> parents[0]=trainer_gui pkg, parents[1]=<repo>
REPO_ROOT = str(Path(__file__).resolve().parents[1])

PAGES = ["Datasets", "Train", "Inference", "Plotting"]


class _NoWheelEdit(QObject):
    """App-wide guard: a mouse wheel must never change a spin box / combo value.
    Scrolling a page used to silently bump whatever number/dropdown the cursor
    passed over (fractions, local↔modal, appearance, …). We eat wheel events on
    spin boxes and combos *unless* they have keyboard focus, so deliberate
    scroll-to-adjust still works once you click in, but casual page scrolling
    never mutates a value. Open combo popups (a separate list view) are untouched.
    ponytail: covers QAbstractSpinBox + QComboBox; add QSlider here if any slider
    becomes a scroll victim too."""

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Wheel
                and isinstance(obj, (QAbstractSpinBox, QComboBox))
                and not obj.hasFocus()):
            event.ignore()
            return True
        return super().eventFilter(obj, event)


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
        self.mode_combo.addItem("Local (pixi)", "local")
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(appstate.get_exec_mode())))
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        sl.addWidget(self.mode_combo)
        self._apply_mode_tag(appstate.get_exec_mode())

        # Appearance: System (follows the OS), Light or Dark — persisted.
        theme_label = QLabel("Appearance")
        theme_label.setObjectName("modeLabel")
        sl.addWidget(theme_label)
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("System", "system")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.setCurrentIndex(
            max(0, self.theme_combo.findData(appstate.get("ui_theme", "system"))))
        self.theme_combo.currentIndexChanged.connect(self._on_theme_change)
        sl.addWidget(self.theme_combo)

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
        # Each page scrolls vertically instead of being crammed into the window,
        # and gets HIG form spacing (Fusion's 6px default clumps the text).
        for page in (self.datasets_page, self.train_page,
                     self.infer_page, self.plotting_page):
            ui.polish_forms(page)
            self.stack.addWidget(ui.scroll_v(page))

        ui.set_navigator(self._navigate)   # pages jump via ui.navigate("Train", …)
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
        self.tag.setText("point-cloud training - local (pixi)" if mode == "local"
                         else "point-cloud training on Modal")

    def _on_theme_change(self):
        from . import appstate, theme
        mode = self.theme_combo.currentData()
        appstate.put("ui_theme", mode)
        theme.apply(QApplication.instance(), mode)

    def _navigate(self, page_name: str, **kwargs):
        """ui.navigate target: switch pages (setCurrentRow fires _go's reload
        hooks), then hand any payload to the page's receive_nav if it has one."""
        self.nav.setCurrentRow(PAGES.index(page_name))
        page = {"Datasets": self.datasets_page, "Train": self.train_page,
                "Inference": self.infer_page, "Plotting": self.plotting_page}[page_name]
        if kwargs and hasattr(page, "receive_nav"):
            page.receive_nav(**kwargs)

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


def _app_icon() -> QIcon:
    """Window/taskbar icon — icon.png shipped at the repo root (or the package)."""
    here = Path(__file__).resolve()
    for base in (here.parent, here.parents[1], here.parents[2]):
        p = base / "icon.png"
        if p.exists():
            return QIcon(str(p))
    return QIcon()


def _ensure_workspace(parent=None) -> None:
    """First launch only: ask where the workspace root should live — datasets and,
    nested inside each, their runs/ and infer/ output. Seeded with the current
    staging dir so accepting it moves nothing; cancel falls back to staging too.
    Once stored, never prompts again."""
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
    from . import appstate, theme
    if sys.platform == "win32":
        # Give the app its OWN taskbar identity so Windows shows icon.png there
        # instead of the generic python.exe icon.
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("trainer_gui")
        except Exception:  # noqa: BLE001
            pass
    app = QApplication(sys.argv)
    app.installEventFilter(_NoWheelEdit(app))  # wheel never changes a value (see class)
    app.setApplicationName("trainer_gui")
    app.setWindowIcon(_app_icon())
    theme.apply(app, appstate.get("ui_theme", "system"))
    try:   # live-follow the OS light/dark switch while in System mode
        app.styleHints().colorSchemeChanged.connect(
            lambda *_: (appstate.get("ui_theme", "system") == "system")
            and theme.apply(app, appstate.get("ui_theme", "system")))
    except (AttributeError, TypeError):
        pass
    _ensure_workspace()   # set the workspace BEFORE pages read it for their defaults
    win = MainWindow()
    win.show()
    _check_modal_cli(win)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
