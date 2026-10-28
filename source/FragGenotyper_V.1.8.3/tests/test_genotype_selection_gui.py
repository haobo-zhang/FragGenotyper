"""Plot selection must preserve complete genotyping and phenotype inputs."""
import json
from pathlib import Path
import threading
import time
import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from genotype_core import VariantData
from genotype_panel import GenotypePanel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def wait(app,panel):
    deadline=time.monotonic()+10
    while panel.worker is not None:
        assert time.monotonic()<deadline,'Export worker timed out'
        app.processEvents()
        time.sleep(.005)
    app.processEvents()


@pytest.fixture
def panel(app):
    widget=GenotypePanel(language='en')
    data=VariantData(samples=[f'{i:03d}' for i in range(6)],positions=[('A01',10),('A01',20)],
        matrix=np.array([[0,0],[0,np.nan],[1,1],[1,2],[2,2],[2,1.]]),report={'source':'fixture.vcf'})
    widget._analysis_ready({'data':data,'groups':[[0,1],[2,3],[4,5]],'parameters':{},'source':'fixture.vcf'})
    yield widget
    if widget.worker is not None:
        widget.cancel_analysis()
        wait(app,widget)
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_subset_preview_exports_and_full_csv_remain_independent(panel,tmp_path):
    frame=panel.genotype_result.copy(deep=True)
    before=panel.data.matrix.copy()
    signals=[]
    panel.genotypes_ready.connect(lambda *args:signals.append(args))
    panel.group_table.item(1,0).setCheckState(Qt.CheckState.Unchecked)
    assert panel.selected_plot_groups()==['Genotype000','Genotype002']
    axis=panel.heatmap_figure.axes[0]
    assert [t.get_text() for t in axis.get_yticklabels()]==['Genotype000','Genotype002']
    np.testing.assert_equal(axis.images[0].get_array().filled(np.nan),before[[0,1,4,5]])
    for extension,magic in [('png',b'\x89PNG'),('pdf',b'%PDF')]:
        target=tmp_path/f'selected.{extension}'
        assert panel.export_image(str(target))==str(target)
        assert target.read_bytes().startswith(magic)
    panel.export_groups(str(tmp_path/'groups.csv'))
    assert len(pd.read_csv(tmp_path/'groups.csv'))==6
    pd.testing.assert_frame_equal(panel.genotype_result,frame)
    np.testing.assert_equal(panel.data.matrix,before)
    assert panel.groups==[[0,1],[2,3],[4,5]] and signals==[]


def test_empty_selection_has_no_stale_plot_and_k_plot_still_exports(panel,tmp_path):
    panel.select_plot_groups(False)
    assert panel.selected_plot_groups()==[] and panel.heatmap_figure.axes==[]
    assert not panel.export_image_button.isEnabled() and not panel.export_selected_button.isEnabled()
    assert panel.export_image(str(tmp_path/'empty.png')) is None
    assert panel.export_selected_images(tmp_path) is None
    assert not (tmp_path/'empty.png').exists()
    panel.k_figure.subplots().plot([1,2],[2,1])
    panel.tabs.setCurrentIndex(1)
    assert panel.export_image_button.isEnabled()
    assert panel.export_image(str(tmp_path/'k.png'))
    panel.tabs.setCurrentIndex(0)
    assert not panel.export_image_button.isEnabled()
    panel.select_plot_groups(True)
    assert len(panel.selected_plot_groups())==3


def test_language_side_and_same_data_k_keep_selection_new_result_resets_it(panel):
    panel.group_table.item(0,0).setCheckState(Qt.CheckState.Unchecked)
    panel.label_side_combo.setCurrentIndex(panel.label_side_combo.findData('left'))
    panel.set_language('zh')
    assert panel.selected_plot_groups()==['Genotype001','Genotype002']
    assert panel.label_side_combo.currentData()=='left'
    assert all(t.label1.get_visible() and not t.label2.get_visible() for t in panel.heatmap_figure.axes[0].yaxis.get_major_ticks())
    panel._k_ready({'source':panel.source_path,'data':panel.data,'result':{'recommended_k':2,
        'table':pd.DataFrame({'K':[1,2],'SSE':[9.,3.],'silhouette':[np.nan,.6],'clusters':[1,2]})}})
    assert panel.selected_plot_groups()==['Genotype001','Genotype002']
    panel.tabs.setCurrentIndex(0)
    emitted=[]
    panel.genotypes_ready.connect(lambda frame,source:emitted.append(frame))
    panel._analysis_ready({'data':panel.data,'groups':panel.groups,'parameters':{},'source':'fixture.vcf'})
    assert len(panel.selected_plot_groups())==3 and len(emitted[0])==6
    panel.set_source('other.vcf')
    assert panel.group_table.rowCount()==0 and panel.data is None
    assert not panel.export_selected_button.isEnabled()


def test_individual_export_runs_in_worker_with_frozen_controls(panel,app,tmp_path,monkeypatch):
    import genotype_panel
    entered,release=threading.Event(),threading.Event()
    original=genotype_panel.export_genotype_plots
    threads=[]
    def held(*args,**kwargs):
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return original(*args,**kwargs)
    monkeypatch.setattr(genotype_panel,'export_genotype_plots',held)
    panel.group_table.item(1,0).setCheckState(Qt.CheckState.Unchecked)
    panel.export_selected_images(tmp_path,file_format='png')
    try:
        deadline=time.monotonic()+5
        while not entered.is_set() and time.monotonic()<deadline:
            app.processEvents()
            time.sleep(.005)
        assert entered.is_set() and threads[0]!=threading.get_ident()
        worker=panel.worker
        assert panel.is_busy() and not panel.group_table.isEnabled() and not panel.label_side_combo.isEnabled()
        panel.export_selected_images(tmp_path)
        assert panel.worker is worker
    finally:
        release.set()
        wait(app,panel)
    manifest=json.loads((Path(panel.last_plot_export)/'manifest.json').read_text())
    assert manifest['selected_labels']==['Genotype000','Genotype002']
    assert [p['samples'] for p in manifest['plots']]==[2,2]
    assert panel.selected_plot_groups()==['Genotype000','Genotype002']
    assert panel.group_table.isEnabled() and panel.export_selected_button.isEnabled()
    assert len(panel.genotype_result)==6
