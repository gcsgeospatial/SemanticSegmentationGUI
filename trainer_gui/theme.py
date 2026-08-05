"""App theming - Light/Dark (Vista-simple: flat chrome, aero blue accent).
The default is whatever the OS says, resolved once at startup; the user's
explicit pick persists. apply() sets Fusion + QPalette + QSS; status labels
use semantic roles via set_accent. Every pair meets WCAG AA."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

LIGHT = {
    "bg": "#f0f3f8", "panel": "#ffffff", "text": "#1b1f27", "muted": "#5b6273",
    "border": "#c5cbd6", "disabled_text": "#767d8b",
    "accent": "#0066cc", "accent_hover": "#0052a3",
    "on_accent": "#ffffff", "focus": "#0066cc",
    "ok": "#1f7a33", "warn": "#9a5300", "error": "#b03030",
    "button": "#f4f6fa", "button_hover": "#e4ebf5", "button_text": "#1b1f27",
    "sel_bg": "#0066cc", "sel_text": "#ffffff",
    "log_bg": "#11141b", "log_text": "#d6dae3",
    "sidebar_bg": "#dfe7f2", "sidebar_text": "#2a3140", "sidebar_muted": "#5b6273",
    "sidebar_sel_bg": "#0066cc", "sidebar_sel_text": "#ffffff",
    "sidebar_disabled": "#8b93a4",
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
    "sidebar_sel_bg": "#3b6cf6", "sidebar_sel_text": "#ffffff",
    "sidebar_disabled": "#5b6273",
}


def system_default() -> str:
    """What the OS says, resolved once - there is no live 'System' mode."""
    try:
        if QApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark:
            return "dark"
    except Exception:  # noqa: BLE001 - older Qt without colorScheme()
        pass
    return "light"


def current_mode() -> str:
    from . import appstate
    return appstate.get("ui_theme") or system_default()


def colors(mode: str = "") -> dict:
    return DARK if (mode or current_mode()) == "dark" else LIGHT


def _palette(c: dict) -> QPalette:
    """Full QPalette so unstyled widgets still adapt to the theme."""
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
#topbar {{ background: {c['sidebar_bg']}; border-bottom: 1px solid {c['border']};
           min-height: 44px; }}
#topbar QTabBar {{ background: transparent; }}
#topbar QTabBar::tab {{ background: transparent; color: {c['sidebar_text']};
                        padding: 12px 18px; border: none; }}
#topbar QTabBar::tab:selected {{ color: {c['accent']}; font-weight: 600;
                                 border-bottom: 3px solid {c['accent']}; }}
#topbar QTabBar::tab:hover:!selected {{ color: {c['text']};
                                        border-bottom: 3px solid {c['border']}; }}
#topbar QComboBox {{ background: {c['panel']}; color: {c['text']};
                     border: 1px solid {c['border']}; border-radius: 4px;
                     padding: 4px 8px; }}
#topbar QComboBox QAbstractItemView {{ background: {c['panel']}; color: {c['text']};
                                       selection-background-color: {c['sel_bg']};
                                       selection-color: {c['sel_text']}; }}
#pageTitle {{ font-size: 22px; font-weight: 600; color: {c['text']}; }}
#pageSub {{ color: {c['muted']}; margin-bottom: 8px; }}
#log {{ font-family: "Cascadia Code", "JetBrains Mono", Consolas, "Courier New", monospace;
        font-size: 12px;
        background: {c['log_bg']}; color: {c['log_text']}; border: 1px solid {c['border']}; }}
/* console toolbar sits on the always-dark terminal core: console constants, not theme tokens */
#logToolbar {{ background: {c['log_bg']}; border: 1px solid {c['border']}; border-bottom: none; }}
#logToolbar QToolButton, #logToolbar QPushButton {{
    background: transparent; color: #8b93a4; border: none; border-radius: 3px;
    padding: 2px 8px; font-size: 11px; }}
#logToolbar QToolButton:hover, #logToolbar QPushButton:hover {{
    color: #d6dae3; background: #232936; }}
#logToolbar QToolButton:checked, #logToolbar QPushButton:checked {{
    color: #d6dae3; background: #2c3445; }}

/* combos/spinboxes must stay native: QSS padding flips them to styled mode, which clips text */
QLineEdit {{ padding: 5px 8px; border: 1px solid {c['border']}; border-radius: 4px;
             background: {c['panel']}; color: {c['text']}; }}
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
             border-radius: 6px; padding: 18px 14px 14px 14px; color: {c['text']};
             background: {c['panel']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c['text']}; }}

QPushButton:focus, QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus,
QTextEdit:focus, QAbstractSpinBox:focus, QListWidget:focus {{
    border: 2px solid {c['focus']};
}}

/* semantic status text (theme.set_accent) */
QLabel[accent="muted"] {{ color: {c['muted']}; }}
QLabel[accent="ok"]    {{ color: {c['ok']}; }}
QLabel[accent="warn"]  {{ color: {c['warn']}; }}
QLabel[accent="error"] {{ color: {c['error']}; }}
"""


def apply(app: QApplication, mode: str = "") -> None:
    """Apply light or dark; safe to call repeatedly (live-switches)."""
    c = colors(mode)
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    app.setPalette(_palette(c))
    app.setStyleSheet(_qss(c))


def set_accent(widget, role: str = "") -> None:
    """Tag a label with a semantic colour role ('muted'|'ok'|'warn'|'error', '' clears)."""
    widget.setProperty("accent", role or None)
    st = widget.style()
    st.unpolish(widget)
    st.polish(widget)
