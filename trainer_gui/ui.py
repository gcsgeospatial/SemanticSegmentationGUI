"""Shared UI helpers: vsplit/hsplit stacked layouts (sizes = minimum heights /
width weights, not splitters) and the page-level scroll_v wrapper."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QPlainTextEdit, QScrollArea, QVBoxLayout, QWidget


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
            # clamp to the content-derived minimum: setMinimumHeight replaces it,
            # and a bare `size` let dense forms squash until rows overlapped
            polish_forms(child)
            child.setMinimumHeight(max(int(size), child.minimumSizeHint().height()))
        else:
            child.setMinimumWidth(160)          # keep side panes usable
        lay.addWidget(child, int(size))         # weight: grow proportionally
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
