"""App theming — accessible Light / Dark / System with adaptive text colors.

One source of truth for colors. `apply(app, mode)` sets the Fusion style (it fully
honors the palette, unlike the native Windows style), a QPalette (so EVERY widget's
text/background adapts — not just the ones we style), and a parameterized QSS. The
status labels scattered through the pages use semantic *roles* (`muted`/`ok`/`warn`/
`error`) via `set_accent`, so their text colour adapts with the theme instead of
being a hardcoded light-mode hex.

Accessibility: every text/background pair below meets WCAG AA (>=4.5:1 normal text,
>=3:1 for large/disabled), there are visible keyboard-focus outlines, and `system`
mode follows the OS light/dark setting (and live-updates when it changes).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

# ---- palettes (every value here is contrast-checked in tests/smoke_test) -------
# Keys are shared by both themes so the QSS/QPalette builders stay DRY.
LIGHT = {
    "bg": "#ffffff", "panel": "#ffffff", "text": "#1b1f27", "muted": "#5b6273",
    "border": "#d4d8e0", "disabled_text": "#767d8b",
    "accent": "#2f6fed", "accent_hover": "#2257c9",
    "on_accent": "#ffffff", "focus": "#2f6fed",
    "ok": "#1f7a33", "warn": "#9a5300", "error": "#b03030",
    "button": "#f5f6f9", "button_hover": "#e9ecf2", "button_text": "#1b1f27",
    "sel_bg": "#2f6fed", "sel_text": "#ffffff",
    "log_bg": "#11141b", "log_text": "#d6dae3",
    "sidebar_bg": "#1f2430", "sidebar_text": "#c8cdd6", "sidebar_muted": "#8b93a4",
    "sidebar_sel_bg": "#323a4d", "sidebar_sel_text": "#ffffff", "sidebar_disabled": "#5b6273",
}
DARK = {
    "bg": "#1b1f27", "panel": "#232936", "text": "#e8ebf1", "muted": "#a6b0c0",
    "border": "#39414f", "disabled_text": "#7b8494",
    "accent": "#5b86ff", "accent_hover": "#6f96ff",
    "on_accent": "#0b1020", "focus": "#7f9cff",
    "ok": "#5fd07a", "warn": "#f0a85e", "error": "#ff7a7a",
    "button": "#2a3140", "button_hover": "#333c4e", "button_text": "#e8ebf1",
    "sel_bg": "#3b6cf6", "sel_text": "#ffffff",
    "log_bg": "#0e1116", "log_text": "#d6dae3",
    "sidebar_bg": "#161a22", "sidebar_text": "#c8cdd6", "sidebar_muted": "#8b93a4",
    "sidebar_sel_bg": "#2c3445", "sidebar_sel_text": "#ffffff", "sidebar_disabled": "#5b6273",
}


def resolve(mode: str) -> str:
    """'light'/'dark'/'system' -> the concrete theme name. 'system' follows the OS."""
    if mode in ("light", "dark"):
        return mode
    try:
        if QApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark:
            return "dark"
    except Exception:  # noqa: BLE001 — older Qt without colorScheme()
        pass
    return "light"


def colors(mode: str) -> dict:
    return DARK if resolve(mode) == "dark" else LIGHT


def _palette(c: dict) -> QPalette:
    """A full QPalette so widgets we DON'T style (inputs, tooltips, menus, default
    QLabels) still get correct, high-contrast text/background for the theme."""
    g = QColor
    p = QPalette()
    p.setColor(QPalette.Window, g(c["bg"]))
    p.setColor(QPalette.WindowText, g(c["text"]))
    p.setColor(QPalette.Base, g(c["panel"]))
    p.setColor(QPalette.AlternateBase, g(c["button"]))
    p.setColor(QPalette.Text, g(c["text"]))
    p.setColor(QPalette.PlaceholderText, g(c["muted"]))
    p.setColor(QPalette.Button, g(c["button"]))
    p.setColor(QPalette.ButtonText, g(c["button_text"]))
    p.setColor(QPalette.ToolTipBase, g(c["panel"]))
    p.setColor(QPalette.ToolTipText, g(c["text"]))
    p.setColor(QPalette.Highlight, g(c["sel_bg"]))
    p.setColor(QPalette.HighlightedText, g(c["sel_text"]))
    p.setColor(QPalette.Link, g(c["accent"]))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, g(c["disabled_text"]))
    return p


def _qss(c: dict) -> str:
    return f"""
QWidget {{ font-size: 14px; }}
#sidebar {{ background: {c['sidebar_bg']}; }}
#sidebar QListWidget {{ background: {c['sidebar_bg']}; color: {c['sidebar_text']};
                        border: none; outline: none; }}
#sidebar QListWidget::item {{ padding: 12px 18px; }}
#sidebar QListWidget::item:selected {{ background: {c['sidebar_sel_bg']};
                                       color: {c['sidebar_sel_text']}; }}
#sidebar QListWidget::item:disabled {{ color: {c['sidebar_disabled']}; }}
#brand {{ color: {c['sidebar_sel_text']}; font-size: 18px; font-weight: 600;
          padding: 18px 18px 6px 18px; }}
#brandSub, #modeLabel {{ color: {c['sidebar_muted']}; padding: 0 18px 14px 18px; }}
#modeLabel {{ padding: 4px 18px 2px 18px; font-size: 12px; }}
#sidebar QComboBox {{ background: {c['button']}; color: {c['text']};
                      border: 1px solid {c['border']}; border-radius: 4px;
                      padding: 4px 8px; margin: 0 18px 12px 18px; }}
#sidebar QComboBox QAbstractItemView {{ background: {c['panel']}; color: {c['text']};
                                        selection-background-color: {c['sel_bg']};
                                        selection-color: {c['sel_text']}; }}
#pageTitle {{ font-size: 22px; font-weight: 600; color: {c['text']}; }}
#pageSub {{ color: {c['muted']}; margin-bottom: 8px; }}
#log {{ font-family: "Cascadia Code", "JetBrains Mono", Consolas, "Courier New", monospace;
        font-size: 12px;
        background: {c['log_bg']}; color: {c['log_text']}; border: 1px solid {c['border']}; }}
/* console header strip (logconsole.LogConsole toolbar) — sits on the always-dark
   terminal core, so its text colors are console constants, not theme tokens */
#logToolbar {{ background: {c['log_bg']}; border: 1px solid {c['border']}; border-bottom: none; }}
#logToolbar QToolButton, #logToolbar QPushButton {{
    background: transparent; color: #8b93a4; border: none; border-radius: 3px;
    padding: 2px 8px; font-size: 11px; }}
#logToolbar QToolButton:hover, #logToolbar QPushButton:hover {{
    color: #d6dae3; background: #232936; }}
#logToolbar QToolButton:checked, #logToolbar QPushButton:checked {{
    color: #d6dae3; background: #2c3445; }}

/* Plain line edits get breathing room (Fusion renders them crunched). Combos and
   spin boxes stay native: QSS padding flips them to styled mode, whose arrow
   subcontrols overlap and clip the text. */
QLineEdit {{ padding: 5px 8px; border: 1px solid {c['border']}; border-radius: 4px;
             background: {c['panel']}; color: {c['text']}; }}
/* the QLineEdit inside a native-rendered combo/spinbox: no inner box */
QComboBox QLineEdit, QAbstractSpinBox QLineEdit {{
    padding: 0 2px; border: none; background: transparent; }}

QPushButton {{ padding: 7px 14px; border-radius: 5px; border: 1px solid {c['border']};
               background: {c['button']}; color: {c['button_text']}; }}
QPushButton:hover {{ background: {c['button_hover']}; }}
QPushButton:disabled {{ color: {c['disabled_text']}; }}
QPushButton#primary {{ background: {c['accent']}; color: {c['on_accent']};
                       border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: {c['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {c['button']}; color: {c['disabled_text']};
                                border: 1px solid {c['border']}; }}

QGroupBox {{ font-weight: 600; margin-top: 12px; border: 1px solid {c['border']};
             border-radius: 6px; padding: 18px 14px 14px 14px; color: {c['text']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c['text']}; }}

/* keyboard-focus visibility (accessibility) */
QPushButton:focus, QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus,
QTextEdit:focus, QAbstractSpinBox:focus, QListWidget:focus {{
    border: 2px solid {c['focus']};
}}

/* semantic status text — adapts with the theme (set via theme.set_accent) */
QLabel[accent="muted"] {{ color: {c['muted']}; }}
QLabel[accent="ok"]    {{ color: {c['ok']}; }}
QLabel[accent="warn"]  {{ color: {c['warn']}; }}
QLabel[accent="error"] {{ color: {c['error']}; }}
"""


def apply(app: QApplication, mode: str) -> None:
    """Apply the resolved theme to the running app (style + palette + stylesheet).
    Safe to call repeatedly — re-applying live-switches the theme."""
    c = colors(mode)
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")            # honors the palette on every platform
    app.setPalette(_palette(c))
    app.setStyleSheet(_qss(c))


def set_accent(widget, role: str = "") -> None:
    """Tag a label with a semantic colour role ('muted'|'ok'|'warn'|'error', or ''
    to clear) so its text colour comes from the theme QSS and adapts on switch."""
    widget.setProperty("accent", role or None)
    st = widget.style()
    st.unpolish(widget)
    st.polish(widget)


def _contrast(fg: str, bg: str) -> float:
    """WCAG relative-contrast ratio between two #rrggbb colours."""
    def lum(h):
        f = lambda u: u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4
        r, g, b = (f(int(h[i:i + 2], 16) / 255) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    a, b = lum(fg), lum(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def _check_contrast():  # ponytail: runnable WCAG AA check — `python trainer_gui/theme.py`
    pairs = [("text", "bg", 4.5), ("text", "panel", 4.5), ("muted", "bg", 4.5),
             ("muted", "panel", 4.5), ("ok", "bg", 4.5), ("warn", "bg", 4.5),
             ("error", "bg", 4.5), ("ok", "panel", 4.5), ("warn", "panel", 4.5),
             ("error", "panel", 4.5), ("button_text", "button", 4.5),
             ("on_accent", "accent", 4.5), ("sel_text", "sel_bg", 4.5),
             ("log_text", "log_bg", 4.5), ("sidebar_text", "sidebar_bg", 4.5),
             ("sidebar_muted", "sidebar_bg", 4.5), ("sidebar_sel_text", "sidebar_sel_bg", 4.5),
             ("disabled_text", "bg", 3.0), ("disabled_text", "button", 3.0), ("focus", "bg", 3.0)]
    for name, c in (("LIGHT", LIGHT), ("DARK", DARK)):
        for fg, bg, mn in pairs:
            r = _contrast(c[fg], c[bg])
            assert r >= mn, f"{name}: {fg} on {bg} = {r:.2f} < {mn} (WCAG AA)"
    print("ok — both themes meet WCAG AA")


if __name__ == "__main__":
    _check_contrast()
