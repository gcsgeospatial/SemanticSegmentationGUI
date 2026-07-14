"""LogConsole — drop-in console widget for streaming process output.

Wraps a read-only QPlainTextEdit (objectName "log", so the theme QSS still
applies) behind a chunk-safe line pipeline: QProcess delivers arbitrary
fragments, so text is buffered until a full line lands; `\\r` progress frames
(tqdm) REPLACE the previous line instead of piling up hundreds of fragments;
ANSI SGR colors map to real text colors and every other escape (cursor
movement, erase) is stripped; committed lines get a muted HH:MM:SS prefix and
content-based severity coloring. A slim toolbar (objectName "logToolbar") adds
Clear / Copy all / Wrap / autoscroll-pin / errors-only. The console core is
deliberately theme-INVARIANT — a terminal stays dark in both app themes —
hence the module color constants below instead of theme.py tokens. Foreground
colors go through QTextCharFormat, never QSS.
"""

from __future__ import annotations

import re
import time
from collections import deque

from PySide6.QtGui import QColor, QGuiApplication, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QPlainTextEdit, QToolButton, QVBoxLayout, QWidget

# ---- theme-invariant console colors (see module docstring) ----------------------
CONSOLE_BG = "#0e1116"
CONSOLE_TEXT = "#d6dae3"
CONSOLE_MUTED = "#6b7383"
CONSOLE_OK = "#5fd07a"
CONSOLE_WARN = "#f0a85e"
CONSOLE_ERROR = "#ff7a7a"
CONSOLE_ACCENT = "#7f9cff"

_SEV_COLOR = {"ok": CONSOLE_OK, "warn": CONSOLE_WARN, "error": CONSOLE_ERROR}
_ANSI_FG = {30: CONSOLE_MUTED, 31: CONSOLE_ERROR, 32: CONSOLE_OK, 33: CONSOLE_WARN,
            34: CONSOLE_ACCENT, 35: "#d29ae0", 36: "#7ecfd4", 37: CONSOLE_TEXT}
_ANSI_FG.update({k + 60: v for k, v in list(_ANSI_FG.items())})  # bright 90-97

_SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")
# every non-SGR escape: other CSI (cursor/erase), OSC titles, single-char escapes
_OTHER_ESC_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b.")
_TAG_RE = re.compile(r"^\[[\w@:./ -]+\]")
_MAX_LINES = 10000


class LineAssembler:
    """Raw chunk stream -> committed-line events, testable without Qt.

    feed() returns [("append"|"replace", line), ...]: "replace" means the line
    rewrites the previously committed one (a `\\r` progress frame). `\\r\\n` is a
    plain newline — the trailing `\\r` of a chunk is carried over so a split
    `\\r\\n` never fakes a rewrite."""

    def __init__(self):
        self._buf = ""
        self._soft = False      # last commit was \r-open: next commit replaces it
        self._carry_cr = False  # chunk ended in \r; join with the next chunk

    def feed(self, text: str) -> list[tuple[str, str]]:
        if self._carry_cr:
            text = "\r" + text
            self._carry_cr = False
        if text.endswith("\r"):
            text = text[:-1]
            self._carry_cr = True
        events = []
        for part in re.split(r"(\r\n|\n|\r)", text):
            if part == "\r":
                if self._buf:
                    events.append(("replace" if self._soft else "append", self._buf))
                    self._buf = ""
                    self._soft = True
            elif part in ("\n", "\r\n"):
                events.append(("replace" if self._soft else "append", self._buf))
                self._buf = ""
                self._soft = False
            elif part:
                self._buf += part
        return events

    def flush(self) -> list[tuple[str, str]]:
        """Commit any partial line (called before run headers so order holds)."""
        if not self._buf:
            self._soft = False
            return []
        ev = [("replace" if self._soft else "append", self._buf)]
        self._buf, self._soft = "", False
        return ev


def _ansi_segments(raw: str) -> list[tuple[str, str | None]]:
    """Line -> [(text, fg hex | None)]: SGR 30-37/90-97 become colors, SGR 0
    resets, extended colors (38/48;5;n) and all non-SGR escapes are stripped."""
    segs, color, pos = [], None, 0
    for m in _SGR_RE.finditer(raw):
        if m.start() > pos:
            segs.append((_OTHER_ESC_RE.sub("", raw[pos:m.start()]), color))
        codes = [int(x) for x in m.group(1).split(";") if x] or [0]
        for code in codes:
            if code in (38, 48):     # extended color — args follow; ignore the rest
                break
            if code == 0:
                color = None
            elif code in _ANSI_FG:
                color = _ANSI_FG[code]
        pos = m.end()
    segs.append((_OTHER_ESC_RE.sub("", raw[pos:]), color))
    return [(t, c) for t, c in segs if t]


def _classify(line: str) -> str | None:
    """Severity by content; error wins over warn wins over ok."""
    low = line.lower()
    if "traceback" in low or "✗" in line or "error" in low or "exited with code" in low:
        return "error"
    if "⚠" in line or "warn" in low:
        return "warn"
    if "✓" in line:
        return "ok"
    return None


class LogConsole(QWidget):
    """Toolbar + colored, \\r-aware, bounded console. Pages treat it like the old
    bare QPlainTextEdit via append()/clear()/setPlaceholderText()."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._asm = LineAssembler()
        # (ts, segments, severity, is_header) — kept so the errors-only filter
        # can re-render; bounded to match setMaximumBlockCount below.
        self._entries: deque = deque(maxlen=_MAX_LINES)
        self._fmt_cache: dict[str, QTextCharFormat] = {}
        self._doc_has_content = False

        bar = QWidget()
        bar.setObjectName("logToolbar")
        hb = QHBoxLayout(bar)
        hb.setContentsMargins(6, 3, 6, 3)
        hb.setSpacing(4)
        clear_btn = self._tool("Clear", tip="Clear the console")
        copy_btn = self._tool("Copy all", tip="Copy the full console text")
        self._wrap_btn = self._tool("Wrap", checkable=True, checked=True,
                                    tip="Wrap long lines")
        self._pin_btn = self._tool("Autoscroll", checkable=True, checked=True,
                                   tip="Follow new output (tail)")
        self._err_btn = self._tool("Errors only", checkable=True,
                                   tip="Show only error lines")
        hb.addWidget(clear_btn)
        hb.addWidget(copy_btn)
        hb.addStretch(1)
        hb.addWidget(self._wrap_btn)
        hb.addWidget(self._pin_btn)
        hb.addWidget(self._err_btn)

        self._edit = QPlainTextEdit()
        self._edit.setReadOnly(True)
        self._edit.setObjectName("log")
        self._edit.setMaximumBlockCount(_MAX_LINES)
        # declaration-only sheet = this widget alone; theme #log rule (font,
        # border) still applies, but the core stays dark in both themes.
        self._edit.setStyleSheet(f"background: {CONSOLE_BG}; color: {CONSOLE_TEXT};")

        clear_btn.clicked.connect(self.clear)
        copy_btn.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self._edit.toPlainText()))
        self._wrap_btn.toggled.connect(
            lambda on: self._edit.setLineWrapMode(
                QPlainTextEdit.WidgetWidth if on else QPlainTextEdit.NoWrap))
        self._err_btn.toggled.connect(self._re_render)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(bar)
        lay.addWidget(self._edit, 1)

    # ---- public API (pages code against exactly these) --------------------------

    def append(self, text: str, newline: bool = True) -> None:
        for kind, line in self._asm.feed(text + ("\n" if newline else "")):
            self._commit(kind, line)

    def begin_run(self, title: str) -> None:
        self._header(f"── run started {time.strftime('%H:%M:%S')} · {title} ──")

    def end_run(self, summary: str) -> None:
        self._header(f"── run finished · {summary} ──")

    def clear(self) -> None:
        self._edit.clear()
        self._entries.clear()
        self._asm = LineAssembler()
        self._doc_has_content = False

    def setPlaceholderText(self, text: str) -> None:  # noqa: N802 — Qt casing
        self._edit.setPlaceholderText(text)

    def plain_widget(self) -> QPlainTextEdit:
        return self._edit

    # ---- internals ---------------------------------------------------------------

    @staticmethod
    def _tool(text: str, checkable: bool = False, checked: bool = False,
              tip: str = "") -> QToolButton:
        b = QToolButton()
        b.setText(text)
        b.setCheckable(checkable)
        b.setChecked(checked)
        b.setToolTip(tip)
        return b

    def _fmt(self, color: str) -> QTextCharFormat:
        f = self._fmt_cache.get(color)
        if f is None:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            self._fmt_cache[color] = f
        return f

    def _header(self, text: str) -> None:
        for kind, line in self._asm.flush():   # keep a split partial line in order
            self._commit(kind, line)
        entry = (None, [(text, None)], None, True)
        self._entries.append(entry)
        self._render_append(entry)

    def _commit(self, kind: str, line: str) -> None:
        segs = _ansi_segments(line)
        entry = (time.strftime("%H:%M:%S"), segs,
                 _classify("".join(t for t, _ in segs)), False)
        if kind == "replace" and self._entries and not self._entries[-1][3]:
            was_visible = self._visible(self._entries[-1])
            self._entries[-1] = entry
            if was_visible:
                self._replace_last_block(entry)
            elif self._visible(entry):
                self._render_append(entry)
        else:
            self._entries.append(entry)
            if self._visible(entry):
                self._render_append(entry)

    def _visible(self, entry) -> bool:
        return entry[3] or not self._err_btn.isChecked() or entry[2] == "error"

    def _at_bottom(self) -> bool:
        bar = self._edit.verticalScrollBar()
        return bar.value() >= bar.maximum() - self._edit.fontMetrics().height()

    def _autoscroll(self, was_at_bottom: bool) -> None:
        # same tail-follow heuristic as ui.append_log, gated by the pin toggle
        if self._pin_btn.isChecked() and was_at_bottom:
            bar = self._edit.verticalScrollBar()
            bar.setValue(bar.maximum())

    def _render_append(self, entry) -> None:
        at_bottom = self._at_bottom()
        cur = QTextCursor(self._edit.document())
        cur.movePosition(QTextCursor.End)
        if self._doc_has_content:
            cur.insertText("\n")
        self._insert_entry(cur, entry)
        self._doc_has_content = True
        self._autoscroll(at_bottom)

    def _replace_last_block(self, entry) -> None:
        at_bottom = self._at_bottom()
        cur = QTextCursor(self._edit.document())
        cur.movePosition(QTextCursor.End)
        cur.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        cur.removeSelectedText()
        if self._visible(entry):
            self._insert_entry(cur, entry)
        elif cur.atStart():                    # replacement is filtered out
            self._doc_has_content = False
        else:
            cur.deletePreviousChar()           # drop the now-empty block
        self._autoscroll(at_bottom)

    def _insert_entry(self, cur: QTextCursor, entry) -> None:
        ts, segs, sev, header = entry
        if header:                              # muted, never severity-colored
            cur.insertText("".join(t for t, _ in segs), self._fmt(CONSOLE_MUTED))
            return
        if not segs:                            # blank line: no lone timestamp
            return
        cur.insertText(ts + " ", self._fmt(CONSOLE_MUTED))
        sev_color = _SEV_COLOR.get(sev)
        first = True
        for text, ansi in segs:
            if first:
                first = False
                m = _TAG_RE.match(text)
                if m:                           # [local]/[modal]/[loss]/… tag
                    cur.insertText(m.group(0), self._fmt(CONSOLE_ACCENT))
                    text = text[m.end():]
                    if not text:
                        continue
            # severity verdict beats per-char ANSI: one glance = one meaning
            cur.insertText(text, self._fmt(sev_color or ansi or CONSOLE_TEXT))

    def _re_render(self) -> None:
        self._edit.clear()
        self._doc_has_content = False
        for entry in self._entries:
            if self._visible(entry):
                self._render_append(entry)
        if self._pin_btn.isChecked():
            bar = self._edit.verticalScrollBar()
            bar.setValue(bar.maximum())


if __name__ == "__main__":
    # logic-level self-check (no QApplication): line assembly, ANSI, severity
    asm = LineAssembler()
    assert asm.feed("hel") == []
    assert asm.feed("lo\nwor") == [("append", "hello")]
    assert asm.feed("ld\n") == [("append", "world")]
    # \r progress frames: each frame replaces the previous one
    assert asm.feed("\r 10%|#") == []
    assert asm.feed("\r 20%|##") == [("append", " 10%|#")]
    assert asm.feed("\r100%|###\ndone\n") == [
        ("replace", " 20%|##"), ("replace", "100%|###"), ("append", "done")]
    # \r\n is a plain newline — even when split across chunks
    assert asm.feed("a\r") == []
    assert asm.feed("\nb\n") == [("append", "a"), ("append", "b")]
    # ANSI: SGR maps to colors; erase/cursor escapes are stripped
    assert _ansi_segments("\x1b[31mfail\x1b[0m ok\x1b[2K\x1b[1A") == [
        ("fail", CONSOLE_ERROR), (" ok", None)]
    assert _ansi_segments("\x1b[92mbright\x1b[m") == [("bright", CONSOLE_OK)]
    assert _classify("Traceback (most recent call last):") == "error"
    assert _classify("Exited with code 1") == "error"
    assert _classify("⚠ low disk") == "warn"
    assert _classify("✓ saved checkpoint") == "ok"
    assert _classify("ep  12: loss=0.4321 acc=0.9123") is None
    assert _TAG_RE.match("[local] starting docker").group(0) == "[local]"
    print("ok — logconsole line pipeline")
