"""Theme preferences, live system changes and readable Qt widget palettes."""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fraggenotyper-mpl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QLineEdit,
    QListWidget, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, QTextEdit, QToolButton,
    QVBoxLayout, QWidget,
)

import app_style


def contrast(first, second):
    """Independent sRGB contrast calculation for actual palette colors."""
    def luminance(value):
        color = QColor(value)
        components = [color.redF(), color.greenF(), color.blueF()]
        linear = [part / 12.92 if part <= .04045 else ((part + .055) / 1.055) ** 2.4
                  for part in components]
        return sum(part * weight for part, weight in zip(linear, (.2126, .7152, .0722)))
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + .05) / (low + .05)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def make_theme(app, tmp_path):
    old_palette, old_style, old_sheet = QPalette(app.palette()), app.style().objectName(), app.styleSheet()
    old_property = app.property("appearanceTheme")
    controllers = []

    def make(*, source=None, settings=None):
        if settings is None:
            settings = QSettings(str(tmp_path / f"theme-{len(controllers)}.ini"), QSettings.Format.IniFormat)
        controller = app_style.ThemeController(app, settings=settings,
                                                system_theme=source or (lambda: "light"))
        controllers.append(controller)
        return controller

    make.controllers = controllers
    yield make
    for controller in controllers:
        if controller._scheme_signal is not None:
            controller._scheme_signal.disconnect(controller._system_changed)
        controller.deleteLater()
    app.setStyle(old_style)
    app.setPalette(old_palette)
    app.setStyleSheet(old_sheet)
    app.setProperty("appearanceTheme", old_property)
    app.processEvents()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_theme_text_pairs_have_readable_contrast(theme):
    colors = app_style.THEMES[theme]
    pairs = (
        [("text", background) for background in ("window", "surface", "alternate", "button", "hover", "pressed", "progress_chunk")]
        + [("muted", background) for background in ("window", "surface", "alternate", "button", "hover")]
        + [("placeholder", background) for background in ("surface", "disabled_bg")]
        + [("disabled_text", background) for background in ("disabled_bg", "disabled_selection", "window", "surface")]
        + [("accent_text", background) for background in ("accent", "accent_hover")]
        + [("tooltip_text", "tooltip"), ("brand", "window"), ("link", "surface"),
           ("visited", "surface"), ("danger", "surface")]
    )
    for foreground, background in pairs:
        assert contrast(colors[foreground], colors[background]) >= 4.5, (theme, foreground, background)
    assert "@" not in app_style.build_stylesheet(theme)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_all_palette_states_cover_fields_selection_tooltips_and_disabled_text(app, theme):
    palette = app_style.build_palette(theme)
    roles = QPalette.ColorRole
    pairs = [(roles.WindowText, roles.Window), (roles.Text, roles.Base),
             (roles.Text, roles.AlternateBase), (roles.ButtonText, roles.Button),
             (roles.ToolTipText, roles.ToolTipBase), (roles.HighlightedText, roles.Highlight),
             (roles.PlaceholderText, roles.Base)]
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled):
        for foreground, background in pairs:
            assert contrast(palette.color(group, foreground), palette.color(group, background)) >= 4.5
        for role in roles:
            if role.name not in {"NoRole", "NColorRoles"}:
                assert palette.isBrushSet(group, role), (theme, group, role)
    assert palette.color(QPalette.ColorGroup.Inactive, roles.HighlightedText) == palette.color(QPalette.ColorGroup.Active, roles.HighlightedText)


def test_default_follows_system_without_writing_preferences_and_explicit_modes_persist(make_theme):
    system = {"theme": "dark"}
    controller = make_theme(source=lambda: system["theme"])
    assert controller.mode == "system" and controller.theme == "dark"
    assert not controller.settings.contains("appearance/theme")
    changes = []
    controller.theme_changed.connect(changes.append)
    controller.set_mode("light")
    assert controller.mode == "light" and controller.theme == "light"
    assert controller.settings.value("appearance/theme") == "light"
    system["theme"] = "dark"
    controller.refresh_system_theme()
    assert controller.theme == "light"
    controller.set_mode("system")
    assert controller.theme == "dark"
    assert controller.settings.value("appearance/theme") == "system"
    system["theme"] = "light"
    controller.refresh_system_theme()
    assert changes == ["light", "dark", "light"]
    assert controller.settings.value("appearance/theme") == "system"


def test_saved_preference_is_restored_and_nonpersistent_switch_does_not_overwrite_it(make_theme, tmp_path):
    settings = QSettings(str(tmp_path / "saved.ini"), QSettings.Format.IniFormat)
    settings.setValue("appearance/theme", "dark")
    settings.sync()
    controller = make_theme(settings=settings)
    assert controller.mode == "dark" and controller.theme == "dark"
    controller.set_mode("light", persist=False)
    assert settings.value("appearance/theme") == "dark"
    restored = make_theme(settings=QSettings(str(tmp_path / "saved.ini"), QSettings.Format.IniFormat))
    assert restored.mode == "dark" and restored.theme == "dark"


def test_mode_change_emits_even_when_resolved_theme_does_not_change(app, make_theme, monkeypatch):
    controller = make_theme(source=lambda: "light")
    events, palettes = [], []
    controller.theme_changed.connect(events.append)
    original = app.setPalette
    monkeypatch.setattr(app, "setPalette", lambda palette: (palettes.append(palette), original(palette))[-1])
    controller.set_mode("light", persist=False)
    controller.set_mode("system", persist=False)

    assert events == ["light", "light"]
    assert not palettes


def test_controller_applies_fusion_before_palette_and_stylesheet(app, make_theme, monkeypatch):
    calls = []
    for name in ("setStyle", "setPalette", "setStyleSheet"):
        original = getattr(app, name)
        def capture(value, name=name, original=original):
            calls.append((name, value))
            return original(value)
        monkeypatch.setattr(app, name, capture)
    make_theme()

    assert calls[0] == ("setStyle", "Fusion")
    assert [name for name, _ in calls[:3]] == ["setStyle", "setPalette", "setStyleSheet"]


def test_invalid_saved_mode_falls_back_to_system_without_rewriting_preferences(make_theme, tmp_path):
    settings = QSettings(str(tmp_path / "invalid.ini"), QSettings.Format.IniFormat)
    settings.setValue("appearance/theme", "old-invalid-value")
    controller = make_theme(source=lambda: "dark", settings=settings)
    assert controller.mode == "system" and controller.theme == "dark"
    assert settings.value("appearance/theme") == "old-invalid-value"
    with pytest.raises(ValueError):
        controller.set_mode("sepia")
    assert controller.mode == "system"


def test_unknown_system_mode_uses_original_palette_not_the_last_manual_theme(app, make_theme):
    app.setPalette(app_style.build_palette("light"))
    controller = make_theme(source=lambda: None)
    assert controller.theme == "light"
    controller.set_mode("dark", persist=False)
    assert controller.theme == "dark"
    controller.set_mode("system", persist=False)
    assert controller.theme == "light"


def test_live_color_scheme_signal_only_changes_system_mode(app, make_theme):
    signal = getattr(app.styleHints(), "colorSchemeChanged", None)
    if signal is None:
        pytest.skip("Qt does not expose live color-scheme changes.")
    system = {"theme": "light"}
    controller = make_theme(source=lambda: system["theme"])
    system["theme"] = "dark"
    signal.emit(Qt.ColorScheme.Dark)
    assert controller.mode == "system" and controller.theme == "dark"
    controller.set_mode("light")
    signal.emit(Qt.ColorScheme.Dark)
    assert controller.theme == "light"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_polished_widgets_have_readable_text_popup_readonly_and_disabled_controls(app, make_theme, theme):
    controller = make_theme()
    controller.set_mode(theme, persist=False)
    parent = QWidget()
    layout = QVBoxLayout(parent)
    line, plain, rich = QLineEdit(), QPlainTextEdit(), QTextEdit()
    line.setPlaceholderText("Sample ID")
    plain.setReadOnly(True)
    plain.setPlainText("Read-only diagnostic output")
    rich.setReadOnly(True)
    rich.setPlainText("Read-only details")
    combo = QComboBox()
    combo.addItems(["Option one", "Option two"])
    table = QTableWidget(2, 2)
    table.setAlternatingRowColors(True)
    table.setItem(0, 0, QTableWidgetItem("Selected value"))
    table.selectRow(0)
    choices = QListWidget()
    choices.addItems(["Selected group", "Another group"])
    choices.setCurrentRow(0)
    primary = QPushButton("Run")
    primary.setObjectName("primary")
    primary.setEnabled(False)
    disabled = QLineEdit("Disabled value")
    disabled.setEnabled(False)
    controls = [line, plain, rich, combo, table, choices, primary, disabled,
                QSpinBox(), QDoubleSpinBox(), QCheckBox("Check"), QRadioButton("Radio"),
                QToolButton(), QProgressBar(), QLabel("Help text")]
    for control in controls:
        layout.addWidget(control)
        control.ensurePolished()
    parent.show()
    app.processEvents()
    roles, active, inactive, off = QPalette.ColorRole, QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled
    try:
        for control in (line, plain, rich, combo.view(), table, choices):
            palette = control.palette()
            assert contrast(palette.color(active, roles.Text), palette.color(active, roles.Base)) >= 4.5
            assert contrast(palette.color(inactive, roles.HighlightedText), palette.color(inactive, roles.Highlight)) >= 4.5
        for control, foreground, background in [(primary, roles.ButtonText, roles.Button), (disabled, roles.Text, roles.Base)]:
            palette = control.palette()
            assert contrast(palette.color(off, foreground), palette.color(off, background)) >= 4.5
        assert plain.isReadOnly() and plain.isEnabled()
        assert rich.isReadOnly() and rich.isEnabled()
        assert app.property("appearanceTheme") == theme
        # Non-native Qt dialogs and menus inherit the same complete palette.
        for popup in (QMessageBox(parent), QMenu(parent)):
            popup.ensurePolished()
            assert contrast(popup.palette().color(roles.WindowText), popup.palette().color(roles.Window)) >= 4.5
    finally:
        parent.close()
        parent.deleteLater()
        app.processEvents()


def test_singleton_and_diagnostic_override_do_not_write_user_settings(app, make_theme, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "diagnostic.ini"), QSettings.Format.IniFormat)
    settings.setValue("appearance/theme", "light")
    monkeypatch.setattr(app_style, "QSettings", lambda *args: settings)
    monkeypatch.setattr(app, "_theme_controller", None, raising=False)
    monkeypatch.setenv("FRAGGENOTYPER_DIAGNOSTIC_THEME", "dark")
    controller = app_style.get_theme_controller(app)
    make_theme.controllers.append(controller)

    assert controller.theme == "dark" and controller.mode == "dark"
    assert app_style.get_theme_controller() is controller
    assert settings.value("appearance/theme") == "light"


def test_legacy_style_is_light_and_theme_does_not_modify_matplotlib(make_theme):
    import matplotlib as mpl
    from matplotlib.figure import Figure
    keys = ("figure.facecolor", "axes.facecolor", "text.color", "axes.labelcolor", "savefig.facecolor")
    before = {key: mpl.rcParams[key] for key in keys}
    before_face = Figure().get_facecolor()
    controller = make_theme()
    controller.set_mode("dark", persist=False)
    controller.set_mode("light", persist=False)

    assert app_style.STYLE == app_style.build_stylesheet("light")
    assert {key: mpl.rcParams[key] for key in keys} == before
    assert Figure().get_facecolor() == before_face
