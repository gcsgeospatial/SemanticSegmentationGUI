"""Shared UI helpers: vsplit/hsplit stacked layouts (sizes = minimum heights /
width weights, not splitters) and the page-level scroll_v wrapper."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QScrollArea, QVBoxLayout, QWidget)


def append_log(log: QPlainTextEdit, text: str, newline: bool = True):
    """Append with tail-f autoscroll: only follows if already at the bottom."""
    if hasattr(log, "begin_run"):   # logconsole.LogConsole: it owns scroll/color
        log.append(text, newline)
        return
    bar = log.verticalScrollBar()
    at_bottom = bar.value() >= bar.maximum() - log.fontMetrics().height()
    log.moveCursor(QTextCursor.End)
    log.insertPlainText(text + ("\n" if newline else ""))
    if at_bottom:
        bar.setValue(bar.maximum())


# cross-page navigation: MainWindow registers its switcher; pages call navigate()
_navigator = None


def set_navigator(fn) -> None:
    global _navigator
    _navigator = fn


def navigate(page_name: str, **kwargs) -> None:
    if _navigator is not None:
        _navigator(page_name, **kwargs)


def vsplit(*widgets: QWidget, sizes: list[int] | None = None) -> QWidget:
    """Stack widgets top-to-bottom. `sizes` = each one's min height + grow weight."""
    return _stack(QVBoxLayout, widgets, sizes, vertical=True)


def hsplit(*widgets: QWidget, sizes: list[int] | None = None) -> QWidget:
    """Place widgets side by side. `sizes` = width weights."""
    return _stack(QHBoxLayout, widgets, sizes, vertical=False)


def _stack(layout_cls, widgets, sizes, vertical: bool) -> QWidget:
    host = QWidget()
    lay = layout_cls(host)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(10)
    sizes = sizes or [1] * len(widgets)
    for child, size in zip(widgets, sizes):
        if vertical:
            # clamp to the content-derived minimum: setMinimumHeight replaces it, and a bare `size` let dense forms squash until rows overlapped
            polish_forms(child)
            child.setMinimumHeight(max(int(size), child.minimumSizeHint().height()))
        else:
            child.setMinimumWidth(160)          # keep side panes usable
        lay.addWidget(child, int(size))
    return host


def wrap(layout) -> QWidget:
    """Layout -> widget (zero margins), e.g. to feed a layout into a stack."""
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w


def polish_forms(root: QWidget) -> None:
    """Open up every form under `root`. Fusion's 6px default layout spacing reads
    as clumped text at our 14px font; 12px between rows and 14px between a label
    and its field are the usual desktop-HIG numbers. Called once per page from
    main.py so every page (and any future one) gets the same rhythm."""
    from PySide6.QtWidgets import QFormLayout
    for f in root.findChildren(QFormLayout):
        f.setVerticalSpacing(12)
        f.setHorizontalSpacing(14)


def scroll_v(widget: QWidget) -> QScrollArea:
    """Wrap a page so it scrolls vertically when its content exceeds the window."""
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return area


def repopulate_combo(combo: QComboBox, items) -> None:
    """Refill a combo from (label, data) pairs, keeping the selected data."""
    prev = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    for label, data in items:
        combo.addItem(label, data)
    i = combo.findData(prev)
    if i >= 0:
        combo.setCurrentIndex(i)
    combo.blockSignals(False)


# --- CRS surface: labels readers' reprojection outcome; Datasets and Infer share one impl.

def parse_epsg(text):
    """Declare-CRS field -> int, None (blank = auto-detect), or False (not an integer)."""
    t = (text or "").strip().upper()
    if t.startswith("EPSG:"):
        t = t[len("EPSG:"):].strip()
    if not t:
        return None
    try:
        return int(t)
    except ValueError:
        return False


def _looks_like_degrees(xyz) -> bool:
    """No-CRS coords sitting in lon/lat bounds with a small span - the D1 trigger.
    The span guard keeps small projected-metre local clouds from false-blocking."""
    if len(xyz) == 0:
        return False
    x, y = xyz[:, 0], xyz[:, 1]
    if not (x.min() >= -180 and x.max() <= 180 and y.min() >= -90 and y.max() <= 90):
        return False
    return max(float(x.max() - x.min()), float(y.max() - y.min())) <= 10.0


def _crs_name(wkt) -> str:
    try:
        from pyproj import CRS
        return CRS.from_wkt(wkt).name
    except Exception:
        return "custom CRS"


def crs_probe(cloud):
    """(source_wkt, proc_wkt, looks_degrees) pulled from an already-read Cloud."""
    return cloud.source_crs_wkt, cloud.crs_wkt, _looks_like_degrees(cloud.xyz)


def crs_story(source_wkt, proc_wkt, looks_degrees, declared_epsg):
    """(detected, action, block) for the CRS surface + D1 preflight. block is a
    remedy string (degree-looking no-CRS input with no declared EPSG) or None."""
    if source_wkt:
        return _crs_name(source_wkt), f"reproject → {_crs_name(proc_wkt)}", None
    if proc_wkt:
        return _crs_name(proc_wkt), "keep as-is (already metre-projected)", None
    if declared_epsg is not None:
        return "none in file", f"declared EPSG:{declared_epsg} → reproject", None
    if looks_degrees:
        return ("none - coordinates look like lat/lon degrees", None,
                "declare its EPSG in the 'Declare CRS (EPSG)' box")
    return "none", "keep as-is (assumed projected metres)", None


def declare_epsg_edit(on_change, max_width: int | None = None) -> QLineEdit:
    """The 'Declare CRS (EPSG)' box; every edit re-renders the CRS line."""
    e = QLineEdit()
    e.setPlaceholderText("blank = auto-detect from the file")
    if max_width is not None:
        e.setMaximumWidth(max_width)
    e.setToolTip("EPSG code to assume for clouds that carry no CRS. Ignored "
                 "for files that declare their own CRS. Required when a "
                 "no-CRS cloud's coordinates look like lat/lon degrees.")
    e.textChanged.connect(on_change)
    return e


def stamp_crs_probe(page, files, error_fmt: str):
    """Probe files[0] (header-only for las/laz) and stamp page._crs_probe/
    _crs_probe_name for the CRS surface. Returns the probe, or None after
    printing error_fmt. Callers may only use field names + channel presence."""
    from .readers import probe_points
    try:
        cloud = probe_points(files[0])
    except Exception as e:
        page._append(error_fmt.format(name=files[0].name, err=e))
        return None
    page._crs_probe = crs_probe(cloud)
    page._crs_probe_name = (files[0].name if len(files) == 1
                            else f"{files[0].name} (+{len(files) - 1} more)")
    return cloud


def render_crs_status(page, verb: str, empty_text: str = "") -> None:
    """CRS status line: probed name + detected CRS + auto action, or the D1 block."""
    if not page._crs_probe:
        page.crs_status.setText(empty_text)
        return
    declared = parse_epsg(page.declare_epsg.text())
    detected, action, block = crs_story(
        *page._crs_probe, declared if type(declared) is int else None)
    if block:
        page.crs_status.setText(f"⚠ {page._crs_probe_name}: {detected}. "
                                f"Blocks {verb} - {block}.")
    else:
        page.crs_status.setText(f"{page._crs_probe_name}: detected {detected} · {action}.")


def crs_launch_gate(page, verb: str, err_prefix: str = "", degrees_sep: str = ".",
                    before_block=None) -> tuple:
    """Pre-launch D1 gate -> (ok, declared_epsg_int_or_None); a False ok already
    printed its remedy. before_block runs between the parse and block checks
    (Infer re-probes a changed input there)."""
    declared = parse_epsg(page.declare_epsg.text())
    if declared is False:
        page._append(err_prefix + "Declare CRS: enter an EPSG integer (e.g. 6539), "
                     "or leave blank to auto-detect from the file.")
        return False, None
    declared = declared if type(declared) is int else None
    if before_block is not None:
        before_block()
    if page._crs_probe:
        _, _, block = crs_story(*page._crs_probe, declared)
        if block:
            page._append(f"✗ '{page._crs_probe_name}' carries no CRS and its "
                         f"coordinates look like lat/lon degrees{degrees_sep} "
                         f"{block} to reproject it, then {verb}.")
            return False, None
    return True, declared


# --- HAG options: ground source × interpolation controls shared by Datasets and Infer.

def hag_title(text: str) -> str:
    """Append the PDAL-missing suffix to a HAG box/checkbox title."""
    from . import pretrain
    return text + ("" if pretrain.pdal_available()
                   else " - grid only, PDAL not installed")


def build_hag_options(page, ground_method_tip: str, ground_tip: str,
                      filter_first: bool = False,
                      default_method: str | None = None) -> QWidget:
    """Create page.hag_ground_method/.hag_filter/.hag_ground/._hag_ground_lbl
    and return the wrapped options row; the ground-source combo drives
    page._on_hag_method (= sync_hag_method)."""
    from . import pretrain
    page.hag_ground_method = QComboBox()
    for k in pretrain.GROUND_METHODS:
        page.hag_ground_method.addItem(pretrain.GROUND_LABELS[k], k)
    if default_method is not None:
        page.hag_ground_method.setCurrentIndex(
            page.hag_ground_method.findData(default_method))
    page.hag_ground_method.setToolTip(ground_method_tip)
    page.hag_ground_method.currentIndexChanged.connect(page._on_hag_method)
    page.hag_filter = QComboBox()
    page.hag_filter.addItems(list(pretrain.HAG_METHODS))
    page.hag_filter.setToolTip("How HAG is interpolated from the ground points. "
                               "grid: fast raster approximation, no PDAL needed. "
                               "hag_nn / hag_delaunay: accurate PDAL filters.")
    page.hag_ground = QLineEdit()
    page.hag_ground.setMaximumWidth(90)
    page.hag_ground.setToolTip(ground_tip)
    page._hag_ground_lbl = QLabel("ground class")
    row = QHBoxLayout()
    row.addWidget(QLabel("ground source"))
    row.addWidget(page.hag_ground_method)
    if filter_first:
        row.addWidget(QLabel("method"))
        row.addWidget(page.hag_filter)
        row.addWidget(page._hag_ground_lbl)
        row.addWidget(page.hag_ground)
    else:
        row.addWidget(page._hag_ground_lbl)
        row.addWidget(page.hag_ground)
        row.addWidget(QLabel("interpolation"))
        row.addWidget(page.hag_filter)
    row.addStretch()
    return wrap(row)


def sync_hag_method(page) -> None:
    """Ground-class field only for 'Base off ground layer'; zmin is
    self-contained and ignores the interpolation method."""
    key = page.hag_ground_method.currentData()
    page._hag_ground_lbl.setVisible(key == "labels")
    page.hag_ground.setVisible(key == "labels")
    page.hag_filter.setEnabled(key != "zmin")
