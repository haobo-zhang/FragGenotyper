"""Individual images contain selected original groups and publish safely."""
import json
import threading
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import genotype_export as exporter


@pytest.fixture
def inputs():
    data=SimpleNamespace(samples=['a1','a2','b1','c1'],positions=[('A01',10),('A01',20)],
        matrix=np.array([[0,1],[0,np.nan],[1,1],[2,2.]]),report={'source':'original.vcf'})
    frame=pd.DataFrame({'Sample':data.samples,'Genotype':['Hap001','Hap001','Hap005','Hap009']})
    return data,[[0,1],[2],[3]],frame


@pytest.mark.parametrize('file_format,magic',[('png',b'\x89PNG'),('pdf',b'%PDF')])
def test_images_preserve_selected_labels_and_existing_output(inputs,tmp_path,file_format,magic,monkeypatch):
    existing=tmp_path/'FragGenotyper_haplotype_plots'
    existing.mkdir()
    (existing/'keep.txt').write_text('original')
    captured=[]
    original=exporter.draw_genotype_heatmap
    def observe(*args,**kwargs):
        axis=original(*args,**kwargs)
        captured.append((kwargs['selected_labels'],axis.images[0].get_array().shape,kwargs['label_side']))
        return axis
    monkeypatch.setattr(exporter,'draw_genotype_heatmap',observe)
    folder=exporter.export_genotype_plots(tmp_path,*inputs,['Hap009','Hap001'],label_side='left',file_format=file_format)
    assert folder==tmp_path/'FragGenotyper_haplotype_plots_2'
    assert captured==[(['Hap001'],(2,2),'left'),(['Hap009'],(1,2),'left')]
    report=json.loads((folder/'manifest.json').read_text())
    assert report['selected_labels']==['Hap001','Hap009'] and report['source']=='original.vcf'
    assert [e['samples'] for e in report['plots']]==[2,1]
    for entry in report['plots']:
        assert (folder/entry['file']).read_bytes().startswith(magic)
    assert (existing/'keep.txt').read_text()=='original'
    assert len(list(folder.iterdir()))==3 and not list(tmp_path.glob('.FragGenotyper*'))


@pytest.mark.parametrize('labels,message',[([], 'at least one'),(['missing'],'Unknown'),(['Hap001','Hap001'],'unique')])
def test_invalid_selection_creates_no_output(inputs,tmp_path,labels,message):
    with pytest.raises(ValueError,match=message):
        exporter.export_genotype_plots(tmp_path,*inputs,labels)
    assert not list(tmp_path.iterdir())


def test_cancel_and_failure_clean_staging(inputs,tmp_path,monkeypatch):
    cancel=threading.Event()
    with pytest.raises(InterruptedError):
        exporter.export_genotype_plots(tmp_path,*inputs,['Hap001','Hap009'],
            progress_callback=lambda *args:cancel.set(),cancel_event=cancel)
    assert not list(tmp_path.iterdir())
    def fail(*args,**kwargs):
        raise OSError('disk error')
    monkeypatch.setattr(exporter,'draw_genotype_heatmap',fail)
    with pytest.raises(OSError,match='disk error'):
        exporter.export_genotype_plots(tmp_path,*inputs,['Hap001'])
    assert not list(tmp_path.iterdir())
