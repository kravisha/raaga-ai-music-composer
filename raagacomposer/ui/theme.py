"""Application palette and stylesheet."""
from __future__ import annotations

from PySide6.QtGui import QColor

DARK = {
    "bg": "#14161b",
    "panel": "#1b1e25",
    "panel_alt": "#22262f",
    "border": "#2e333d",
    "text": "#e6e9ef",
    "muted": "#9aa3b2",
    "accent": "#e0a458",
    "accent_dim": "#7d5b2e",
    "ok": "#5fb87a",
    "warn": "#d4864b",
    "error": "#d46a6a",
    "playhead": "#f2f4f8",
    "selection": "#3a4a63",
}

ROLE_COLORS = {
    "lead": "#c98a4b",
    "counter": "#5f8fb8",
    "pad": "#6a6fa8",
    "bass": "#4f7a5c",
    "rhythm": "#a35a63",
    "fill": "#8b7ab8",
    "drone": "#57666f",
}

SECTION_COLORS = {
    "prelude": "#2a323d",
    "pallavi": "#33303c",
    "anupallavi": "#2f3440",
    "verse": "#2b3239",
    "chorus": "#3a3340",
    "charanam": "#2d333c",
    "interlude": "#26303a",
    "bridge": "#312f38",
    "outro": "#282f36",
}


def color(name: str) -> QColor:
    return QColor(DARK.get(name, "#ffffff"))


def role_color(role: str) -> QColor:
    return QColor(ROLE_COLORS.get(role, "#7a7f8a"))


def section_color(kind: str) -> QColor:
    return QColor(SECTION_COLORS.get(kind, "#2a2f38"))


STYLESHEET = f"""
QWidget {{
    background: {DARK['bg']};
    color: {DARK['text']};
    font-size: 12px;
}}
QGroupBox {{
    background: {DARK['panel']};
    border: 1px solid {DARK['border']};
    border-radius: 6px;
    margin-top: 14px;
    padding: 8px 6px 6px 6px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {DARK['accent']};
}}
QPushButton {{
    background: {DARK['panel_alt']};
    border: 1px solid {DARK['border']};
    border-radius: 4px;
    padding: 5px 10px;
}}
QPushButton:hover {{ border-color: {DARK['accent_dim']}; }}
QPushButton:pressed {{ background: {DARK['accent_dim']}; }}
QPushButton:disabled {{ color: {DARK['muted']}; border-color: #262a32; }}
QPushButton#primary {{
    background: {DARK['accent_dim']};
    border-color: {DARK['accent']};
    font-weight: 600;
}}
QPushButton#record[listening="true"] {{
    background: #7a3540;
    border-color: {DARK['error']};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget,
QTableWidget, QTreeWidget {{
    background: {DARK['panel_alt']};
    border: 1px solid {DARK['border']};
    border-radius: 4px;
    padding: 3px;
    selection-background-color: {DARK['accent_dim']};
}}
QTabWidget::pane {{
    border: 1px solid {DARK['border']};
    border-radius: 6px;
    background: {DARK['panel']};
}}
QTabBar::tab {{
    background: {DARK['panel']};
    border: 1px solid {DARK['border']};
    border-bottom: none;
    padding: 6px 14px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    color: {DARK['muted']};
}}
QTabBar::tab:selected {{ color: {DARK['text']}; background: {DARK['panel_alt']}; }}
QHeaderView::section {{
    background: {DARK['panel_alt']};
    border: none;
    border-right: 1px solid {DARK['border']};
    padding: 4px;
}}
QSlider::groove:horizontal {{
    height: 4px; background: {DARK['border']}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {DARK['accent']}; width: 12px; margin: -5px 0; border-radius: 6px;
}}
QProgressBar {{
    background: {DARK['panel_alt']};
    border: 1px solid {DARK['border']};
    border-radius: 3px;
    text-align: center;
    height: 14px;
}}
QProgressBar::chunk {{ background: {DARK['accent_dim']}; border-radius: 2px; }}
QStatusBar {{ background: {DARK['panel']}; border-top: 1px solid {DARK['border']}; }}
QMenuBar {{ background: {DARK['panel']}; }}
QMenuBar::item:selected {{ background: {DARK['panel_alt']}; }}
QMenu {{ background: {DARK['panel']}; border: 1px solid {DARK['border']}; }}
QMenu::item:selected {{ background: {DARK['accent_dim']}; }}
QToolBar {{ background: {DARK['panel']}; border-bottom: 1px solid {DARK['border']};
            spacing: 4px; padding: 4px; }}
QSplitter::handle {{ background: {DARK['border']}; }}
QScrollBar:vertical {{ background: {DARK['bg']}; width: 10px; }}
QScrollBar:horizontal {{ background: {DARK['bg']}; height: 10px; }}
QScrollBar::handle {{ background: {DARK['border']}; border-radius: 5px; }}
QLabel#hint {{ color: {DARK['muted']}; }}
QLabel#title {{ font-size: 15px; font-weight: 600; }}
QCheckBox::indicator {{ width: 14px; height: 14px; }}
"""
