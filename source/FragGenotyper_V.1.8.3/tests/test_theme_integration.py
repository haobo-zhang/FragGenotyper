"""Live appearance changes preserve analysis and keep plotting controls visible."""
import copy
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import QCoreApplication,QEvent,QSettings,Qt
from PyQt6.QtGui import QIcon,QPalette
from PyQt6.QtWidgets import QApplication
from app_style import ThemeController
from app_window import MainWindow
from genotype_core import VariantData
from plot_toolbar import PlotNavigationToolbar
from startup import LoadingWindow


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app,tmp_path):
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    previous=getattr(app,'_theme_controller',None)
    palette,stylesheet=QPalette(app.palette()),app.styleSheet()
    settings=QSettings(str(tmp_path/'appearance.ini'),QSettings.Format.IniFormat)
    controller=ThemeController(app,settings=settings,system_theme=lambda:'light')
    app._theme_controller=controller
    result=MainWindow()
    result._test_settings=settings
    yield result
    for panel in (result.genotype_panel,result.phenotype_panel,result.filter_panel):
        panel.begin_shutdown()
        deadline=time.monotonic()+10
        while panel.worker is not None:
            assert time.monotonic()<deadline
            app.processEvents()
            time.sleep(.005)
    result.close()
    result.deleteLater()
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    app.processEvents()
    if previous is None:
        del app._theme_controller
        app.setPalette(palette)
        app.setStyleSheet(stylesheet)
    else:
        app._theme_controller=previous
        previous.set_mode(previous.mode,persist=False)
    controller.deleteLater()
    app.processEvents()


def icon_brightness(toolbar):
    icon=toolbar._actions['home'].icon()
    image=icon.pixmap(32,32,QIcon.Mode.Normal,QIcon.State.Off).toImage()
    pixels=[image.pixelColor(x,y) for x in range(image.width()) for y in range(image.height()) if image.pixelColor(x,y).alpha()>200]
    assert pixels
    return sum((p.redF()+p.greenF()+p.blueF())/3 for p in pixels)/len(pixels)


def test_header_saves_choice_and_language_keeps_mode(window,app):
    window.genotype_panel.k_spin.setValue(7)
    window.theme_combo.setCurrentIndex(window.theme_combo.findData('dark'))
    app.processEvents()
    assert window.theme_controller.theme=='dark'
    assert window._test_settings.value('appearance/theme')=='dark'
    window.language_combo.setCurrentIndex(1)
    assert window.theme_combo.currentData()=='dark'
    assert window.theme_combo.currentText()=='Dark mode'
    assert window.genotype_panel.k_spin.value()==7
    window.theme_controller.set_mode('light',persist=False)
    app.processEvents()
    assert window.theme_combo.currentData()=='light'


def test_existing_and_new_toolbars_recolor_when_theme_changes(window,app):
    toolbar=window.findChildren(PlotNavigationToolbar)[0]
    window.theme_controller.set_mode('light',persist=False)
    app.processEvents()
    light=icon_brightness(toolbar)
    window.theme_controller.set_mode('dark',persist=False)
    app.processEvents()
    dark=icon_brightness(toolbar)
    assert light<.3 and dark>.65,(light,dark)
    # A toolbar created after the switch must start with readable dark icons.
    fresh=PlotNavigationToolbar(window.genotype_panel.heatmap_canvas,window)
    assert icon_brightness(fresh)>.65
    fresh.close()
    fresh.deleteLater()


def test_theme_switch_preserves_plot_data_selection_and_white_export_surface(window,app):
    gen=window.genotype_panel
    data=VariantData(['001','002','003','004'],[('1',10),('1',20)],np.array([[0,0],[0,np.nan],[2,2],[2,1.]]),{})
    gen._analysis_ready({'data':data,'groups':[[0,1],[2,3]],'parameters':{},'source':'fixture.vcf'})
    gen.group_table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
    frame=gen.genotype_result.copy(deep=True)
    pixels=gen.heatmap_figure.axes[0].images[0].get_array().copy()
    face=gen.heatmap_figure.get_facecolor()
    window.theme_controller.set_mode('dark',persist=False)
    app.processEvents()
    assert gen.selected_plot_groups()==['Genotype001']
    pd.testing.assert_frame_equal(frame,gen.genotype_result)
    np.testing.assert_equal(pixels,gen.heatmap_figure.axes[0].images[0].get_array())
    assert gen.heatmap_figure.get_facecolor()==face==(1.,1.,1.,1.)
    assert window.phenotype_panel.genotype_df.Genotype.nunique()==2


def test_loading_header_and_error_text_follow_both_palettes(window,app):
    loading=LoadingWindow()
    try:
        assert not loading.heading.styleSheet()
        loading.show_failure('Readable diagnostic text')
        loading.show()
        for mode in ['dark','light']:
            window.theme_controller.set_mode(mode,persist=False)
            app.processEvents()
            palette=loading.error_box.palette()
            assert palette.color(QPalette.ColorRole.Text)!=palette.color(QPalette.ColorRole.Base)
            assert palette.color(QPalette.ColorGroup.Inactive,QPalette.ColorRole.Text)!=palette.color(QPalette.ColorGroup.Inactive,QPalette.ColorRole.Base)
    finally:
        loading.finish_close()
        loading.deleteLater()
        app.processEvents()
