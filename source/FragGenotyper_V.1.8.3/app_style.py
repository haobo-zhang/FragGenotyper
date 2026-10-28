"""Coherent Qt themes, independent of scientific figure/export styling."""
from __future__ import annotations

import os

from PyQt6.QtCore import QObject, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


THEMES = {
    "light": {
        "window": "#F3F6FA", "surface": "#FFFFFF", "alternate": "#EAF0F6",
        "button": "#E8EFF6", "hover": "#DDEAF4", "pressed": "#D0E1EE",
        "text": "#1B293A", "muted": "#48596D", "placeholder": "#536277",
        "disabled_bg": "#E4EAF1", "disabled_text": "#4B596B",
        "disabled_selection": "#D2DDE9", "border": "#B8C6D5", "border_strong": "#687D94",
        "accent": "#1E6085", "accent_hover": "#174D6B", "accent_text": "#FFFFFF",
        "tooltip": "#19334B", "tooltip_text": "#FFFFFF", "brand": "#153F5C",
        "link": "#075E92", "visited": "#6F3695", "progress_chunk": "#B6D4E6",
        "danger": "#A71D35", "shadow": "#17283C",
    },
    "dark": {
        "window": "#17212D", "surface": "#202D3C", "alternate": "#273749",
        "button": "#2B3C4F", "hover": "#344C63", "pressed": "#3D5871",
        "text": "#EDF3FA", "muted": "#C3D0DF", "placeholder": "#B7C6D8",
        "disabled_bg": "#293746", "disabled_text": "#B8C5D5",
        "disabled_selection": "#354A60", "border": "#52677E", "border_strong": "#7C95AE",
        "accent": "#9AD2FF", "accent_hover": "#BCE2FF", "accent_text": "#142434",
        "tooltip": "#E8F2FD", "tooltip_text": "#182B40", "brand": "#B7E0FF",
        "link": "#A8D7FF", "visited": "#DFBDFF", "progress_chunk": "#345875",
        "danger": "#FFBCC8", "shadow": "#080F18",
    },
}


def build_palette(theme):
    """Set all usable color roles for active, inactive and disabled widgets."""
    colors = THEMES[theme]
    palette = QPalette()
    normal = {
        "Window": "window", "WindowText": "text", "Base": "surface",
        "AlternateBase": "alternate", "ToolTipBase": "tooltip", "ToolTipText": "tooltip_text",
        "Text": "text", "Button": "button", "ButtonText": "text", "BrightText": "danger",
        "Light": "surface", "Midlight": "hover", "Dark": "border_strong", "Mid": "border",
        "Shadow": "shadow", "Highlight": "accent", "HighlightedText": "accent_text",
        "Link": "link", "LinkVisited": "visited", "PlaceholderText": "placeholder", "Accent": "accent",
    }
    disabled = dict(normal, WindowText="disabled_text", Base="disabled_bg", AlternateBase="disabled_bg",
                    Text="disabled_text", Button="disabled_bg", ButtonText="disabled_text",
                    BrightText="disabled_text", Highlight="disabled_selection", HighlightedText="disabled_text",
                    Link="disabled_text", LinkVisited="disabled_text", PlaceholderText="disabled_text", Accent="disabled_text")
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled):
        for role_name, token in (disabled if group == QPalette.ColorGroup.Disabled else normal).items():
            # Accent is absent from Qt 6.5, the minimum supported version.
            role = getattr(QPalette.ColorRole, role_name, None)
            if role is not None:
                palette.setColor(group, role, QColor(colors[token]))
    return palette


def build_stylesheet(theme):
    """Complete QSS pairs every foreground with an appropriate background."""
    c = THEMES[theme]
    sheet = """
QWidget { color: @text; font-size: 13px; }
QWidget:disabled { color: @disabled_text; }
QMainWindow, QDialog, QMessageBox, QWidget#appRoot { background-color: @window; }
QLabel { background-color: transparent; }
QLabel#brand { font-size: 23px; font-weight: 700; color: @brand; }
QLabel#subtitle { color: @muted; }
QLabel#sectionTitle { font-size: 17px; font-weight: 600; color: @brand; padding: 4px 0; }
QStatusBar { background-color: @window; color: @muted; }
QStatusBar::item { border: none; }
QGroupBox { background-color: @surface; color: @text; border: 1px solid @border; border-radius: 8px; margin-top: 18px; padding: 16px 12px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0 5px; color: @text; }
QGroupBox:disabled, QGroupBox::title:disabled { color: @disabled_text; }
QLineEdit, QAbstractSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: @surface; color: @text; border: 1px solid @border;
    selection-background-color: @accent; selection-color: @accent_text;
    placeholder-text-color: @placeholder; border-radius: 5px; padding: 6px; min-height: 20px;
}
QLineEdit:read-only, QLineEdit[readOnly="true"], QTextEdit[readOnly="true"], QPlainTextEdit[readOnly="true"] {
    background-color: @surface; color: @text; selection-background-color: @accent; selection-color: @accent_text;
}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus { border-color: @accent; }
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
    background-color: @disabled_bg; color: @disabled_text; border-color: @border;
    placeholder-text-color: @disabled_text; selection-background-color: @disabled_selection; selection-color: @disabled_text;
}
QComboBox { padding-right: 27px; }
QComboBox QAbstractItemView { background-color: @surface; color: @text; alternate-background-color: @alternate; selection-background-color: @accent; selection-color: @accent_text; border: 1px solid @border_strong; outline: 0; }
QAbstractSpinBox { padding-right: 23px; }
QPushButton { background-color: @button; color: @text; border: 1px solid @border; border-radius: 5px; padding: 8px 13px; min-height: 20px; }
QPushButton:hover { background-color: @hover; border-color: @border_strong; }
QPushButton:pressed, QPushButton:checked { background-color: @pressed; border-color: @accent; }
QPushButton:focus { border-color: @accent; }
QPushButton#primary { background-color: @accent; color: @accent_text; border-color: @accent; font-weight: 600; }
QPushButton#primary:hover, QPushButton#primary:pressed { background-color: @accent_hover; color: @accent_text; border-color: @accent_hover; }
QPushButton:disabled, QPushButton#primary:disabled { background-color: @disabled_bg; color: @disabled_text; border-color: @border; }
QToolBar { background-color: @window; color: @text; border: none; spacing: 3px; }
QToolButton { background-color: transparent; color: @text; border: 1px solid transparent; border-radius: 4px; padding: 4px; }
QToolButton:hover { background-color: @hover; color: @text; border-color: @border; }
QToolButton:checked, QToolButton:pressed { background-color: @hover; color: @text; border-color: @accent; }
QToolButton:disabled { background-color: @disabled_bg; color: @disabled_text; border-color: transparent; }
QTabWidget::pane { background-color: @surface; border: 1px solid @border; border-radius: 6px; }
QTabBar::tab { color: @muted; background-color: @alternate; padding: 9px 14px; border: none; }
QTabBar::tab:hover { background-color: @hover; color: @text; }
QTabBar::tab:selected { color: @text; background-color: @surface; border-bottom: 2px solid @accent; }
QTabBar::tab:disabled { color: @disabled_text; background-color: @disabled_bg; }
QAbstractItemView { background-color: @surface; color: @text; alternate-background-color: @alternate; selection-background-color: @accent; selection-color: @accent_text; border: 1px solid @border; border-radius: 4px; outline: 0; show-decoration-selected: 1; }
QAbstractItemView::item:selected, QAbstractItemView::item:selected:active, QAbstractItemView::item:selected:!active { background-color: @accent; color: @accent_text; }
QAbstractItemView:disabled { background-color: @disabled_bg; color: @disabled_text; }
QAbstractItemView::item:selected:disabled { background-color: @disabled_selection; color: @disabled_text; }
QTableView { gridline-color: @border; }
QListWidget#navigation { background-color: @alternate; color: @text; border: none; padding: 10px 4px; }
QListWidget#navigation::item { padding: 15px 10px; margin: 3px 5px; border-radius: 7px; }
QListWidget#navigation::item:selected, QListWidget#navigation::item:selected:!active { background-color: @accent; color: @accent_text; }
QListWidget#navigation::item:selected:disabled { background-color: @disabled_selection; color: @disabled_text; }
QHeaderView { background-color: @alternate; color: @text; }
QHeaderView::section, QTableCornerButton::section { background-color: @alternate; color: @text; border: none; border-bottom: 1px solid @border; padding: 7px; font-weight: 600; }
QHeaderView::section:disabled, QTableCornerButton::section:disabled { background-color: @disabled_bg; color: @disabled_text; }
QProgressBar { color: @text; border: 1px solid @border; border-radius: 4px; text-align: center; min-height: 16px; background-color: @button; }
QProgressBar::chunk { background-color: @progress_chunk; border-radius: 3px; }
QProgressBar:disabled { color: @disabled_text; background-color: @disabled_bg; }
QProgressBar::chunk:disabled { background-color: @disabled_selection; }
QCheckBox, QRadioButton { color: @text; background-color: transparent; spacing: 7px; }
QCheckBox:disabled, QRadioButton:disabled { color: @disabled_text; }
QCheckBox::indicator, QRadioButton::indicator { width: 15px; height: 15px; }
QScrollArea { border: none; background-color: @surface; }
QScrollArea > QWidget > QWidget { background-color: @surface; }
QScrollBar:vertical { background-color: @window; width: 12px; margin: 0; }
QScrollBar:horizontal { background-color: @window; height: 12px; margin: 0; }
QScrollBar::handle { background-color: @border_strong; border-radius: 5px; min-width: 22px; min-height: 22px; }
QScrollBar::handle:hover { background-color: @muted; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: none; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
QSplitter::handle { background-color: @border; width: 5px; }
QMenuBar, QMenu { background-color: @surface; color: @text; }
QMenu { border: 1px solid @border_strong; padding: 4px; }
QMenu::item { padding: 6px 24px 6px 12px; }
QMenu::item:selected, QMenuBar::item:selected { background-color: @accent; color: @accent_text; }
QMenu::item:disabled, QMenuBar::item:disabled { background-color: @surface; color: @disabled_text; }
QMenu::separator { height: 1px; background-color: @border; margin: 4px 6px; }
QToolTip { background-color: @tooltip; color: @tooltip_text; padding: 5px; border: 1px solid @border_strong; }
"""
    for token in sorted(c, key=len, reverse=True):
        sheet = sheet.replace("@" + token, c[token])
    return sheet


STYLE = build_stylesheet("light")


class ThemeController(QObject):
    """Select an explicit theme or follow Qt's live system color scheme."""
    theme_changed = pyqtSignal(str)
    MODES = frozenset(("system", "light", "dark"))
    SETTINGS_KEY = "appearance/theme"

    def __init__(self, app, settings=None, system_theme=None):
        super().__init__(app)
        self.app = app
        self.settings = settings if settings is not None else QSettings("FragGenotyper", "Desktop")
        self._system_theme = system_theme
        self._fallback_system_theme = "dark" if app.palette().color(QPalette.ColorRole.Window).lightnessF() < .5 else "light"
        self.mode, self.theme = "system", ""
        self._applying = False
        self._scheme_signal = getattr(app.styleHints(), "colorSchemeChanged", None)
        if self._scheme_signal is not None:
            self._scheme_signal.connect(self._system_changed)
        saved = str(self.settings.value(self.SETTINGS_KEY, "system")).strip().lower()
        self.set_mode(saved if saved in self.MODES else "system", persist=False)

    def _read_system_theme(self):
        try:
            if self._system_theme is not None:
                value = self._system_theme()
            else:
                getter = getattr(self.app.styleHints(), "colorScheme", None)
                value = getter() if getter is not None else None
            schemes = getattr(Qt, "ColorScheme", None)
            if value == "dark" or value == getattr(schemes, "Dark", object()):
                return "dark"
            if value == "light" or value == getattr(schemes, "Light", object()):
                return "light"
        except Exception:
            pass
        return self._fallback_system_theme

    def set_mode(self, mode, persist=True):
        mode = str(mode).strip().lower()
        if mode not in self.MODES:
            raise ValueError("Theme mode must be 'system', 'light', or 'dark'.")
        previous_mode, previous_theme = self.mode, self.theme
        self.mode = mode
        if persist:
            self.settings.setValue(self.SETTINGS_KEY, mode)
            sync = getattr(self.settings, "sync", None)
            if sync is not None:
                sync()
        self._apply(self._read_system_theme() if mode == "system" else mode)
        if previous_mode != mode and previous_theme == self.theme:
            # Preference controls must update even if system and explicit modes
            # currently resolve to the same colors. No palette rebuild is needed.
            self.theme_changed.emit(self.theme)
        return self.theme

    def refresh_system_theme(self):
        if self.mode == "system" and not self._applying:
            self._apply(self._read_system_theme())
        return self.theme

    def _system_changed(self, *_):
        self.refresh_system_theme()

    def _apply(self, theme):
        if self._applying or self.theme == theme:
            return
        self._applying = True
        try:
            self.app.setStyle("Fusion")
            self.app.setPalette(build_palette(theme))
            self.app.setStyleSheet(build_stylesheet(theme))
            self.theme = theme
            self.app.setProperty("appearanceTheme", theme)
        finally:
            self._applying = False
        self.theme_changed.emit(theme)


def get_theme_controller(app=None):
    """Install/reuse the application controller without persisting diagnostic overrides."""
    app = app if app is not None else QApplication.instance()
    if app is None:
        raise RuntimeError("Create QApplication before installing the theme controller.")
    controller = getattr(app, "_theme_controller", None)
    if not isinstance(controller, ThemeController):
        controller = ThemeController(app)
        app._theme_controller = controller
        override = os.environ.get("FRAGGENOTYPER_DIAGNOSTIC_THEME", "").strip().lower()
        if override in ThemeController.MODES:
            controller.set_mode(override, persist=False)
    return controller
