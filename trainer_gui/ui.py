"""Small shared UI helpers — stacked sections and a page-level vertical scroll.

`vsplit`/`hsplit` used to return draggable QSplitters with fixed pixel sizes that
cropped content. They now return plain stacked layouts: for `vsplit` the `sizes`
become each section's *minimum* height plus a growth weight (so nothing is
clipped and sections grow proportionally when there's room); for `hsplit` they're
width weights. Pages are wrapped in `scroll_v`, so anything taller than the
window scrolls instead of being squeezed.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QPlainTextEdit, QScrollArea, QVBoxLayout, QWidget


def append_log(log: QPlainTextEdit, text: str, newline: bool = True):
    """Append to a log console without yanking a scrolled-up user back to the
    bottom. Autoscrolls only if the view was already at (or within a line of)
    the bottom before this append — the standard `tail -f` behavior — so
    reading earlier output stays put while new lines keep arriving below."""
    bar = log.verticalScrollBar()
    at_bottom = bar.value() >= bar.maximum() - log.fontMetrics().height()
    log.moveCursor(QTextCursor.End)
    log.insertPlainText(text + ("\n" if newline else ""))
    if at_bottom:
        bar.setValue(bar.maximum())


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
            # Floor: never crop a section. An explicit setMinimumHeight REPLACES
            # the content-derived minimum, so clamp to whichever is larger — a
            # bare `size` let dense forms be squashed until their rows overlapped.
            # Spacing must be final before measuring, hence polish_forms here.
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
