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
from PySide6.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QScrollArea, QSpinBox, QVBoxLayout,
                               QWidget)


class NoWheelSpinBox(QSpinBox):
    """A QSpinBox that ignores the scroll wheel — values change by typing or the
    arrows only, so scrolling the page never silently nudges a number."""
    def wheelEvent(self, e):  # noqa: N802 (Qt signature)
        e.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox twin of NoWheelSpinBox — wheel scrolling is ignored."""
    def wheelEvent(self, e):  # noqa: N802 (Qt signature)
        e.ignore()


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
            child.setMinimumHeight(int(size))   # floor: never crop a section
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


def scrollable(widget: QWidget) -> QWidget:
    """No-op: the whole page now scrolls (see scroll_v), so inner scroll areas
    would just nest. Kept so existing call sites need no change."""
    return widget


def scroll_v(widget: QWidget) -> QScrollArea:
    """Wrap a page so it scrolls vertically when its content exceeds the window."""
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return area
