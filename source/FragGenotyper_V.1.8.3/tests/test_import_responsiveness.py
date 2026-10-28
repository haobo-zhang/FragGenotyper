import threading
import time
import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
import haplotype_phenotype as hp
from phenotype_tab import PhenotypeComparisonTab, MAX_VISIBLE_PAIRS


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait(app, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("Import worker timed out")
        app.processEvents()
        time.sleep(.003)
    app.processEvents()


@pytest.fixture
def tab(app):
    widget = PhenotypeComparisonTab(language="en")
    yield widget
    if widget.worker is not None:
        widget.cancel_task()
        wait(app, lambda: widget.worker is None)
    widget.begin_shutdown()
    widget.close()
    widget.deleteLater()
    app.processEvents()


def frames(groups):
    ids = [f"ID{i:05d}" for i in range(groups * 3)]
    return (pd.DataFrame({"Sample": ids, "Genotype": [f"G{i // 3:05d}" for i in range(len(ids))]}),
            pd.DataFrame({"Sample": ids, "Phenotype": np.arange(len(ids))}))


@pytest.mark.parametrize("phase", ["read", "prepare"])
def test_import_read_and_matching_do_not_block_qt(tab, app, monkeypatch, tmp_path, phase):
    genotype, phenotype = frames(3)
    tab.genotype_df = genotype
    tab.populate_columns()
    source = tmp_path / "phenotype.csv"
    phenotype.to_csv(source, index=False)
    original = hp.read_table if phase == "read" else hp.prepare_data
    entered, release = threading.Event(), threading.Event()
    threads = []
    def held(*args, **kwargs):
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(hp, "read_table" if phase == "read" else "prepare_data", held)
    heartbeats = []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeats.append(1))
    timer.start()
    try:
        tab.start_table_import(source, kind="phenotype")
        worker = tab.worker
        assert worker is not None
        wait(app, entered.is_set)
        wait(app, lambda: len(heartbeats) >= 8)
        tab.start_table_import(source, kind="phenotype")
        assert tab.worker is worker
        assert not tab.controls.isEnabled()
        assert threads == [threads[0]] and threads[0] != threading.get_ident()
    finally:
        release.set()
        wait(app, lambda: tab.worker is None)
        timer.stop()
    assert tab.phenotype_path == str(source)
    assert len(tab.selected_groups()) == 3
    assert tab.import_diagnostics["rows"] == 9
    assert tab.run_button.isEnabled()


def test_thousands_of_groups_do_not_create_hidden_pair_widgets(tab):
    tab.genotype_df, tab.phenotype_df = frames(3840)
    tab.populate_columns()
    assert len(tab.selected_groups()) == 3840
    assert tab.pair_list.count() == tab.annotations_list.count() == 0
    tab.comparison_combo.setCurrentIndex(tab.comparison_combo.findData("custom"))
    assert tab.pair_list.count() == 0
    assert "limit" in tab.pair_notice.text()
    assert 3840 * 3839 // 2 > MAX_VISIBLE_PAIRS
    tab.comparison_combo.setCurrentIndex(tab.comparison_combo.findData("reference"))
    assert tab.pair_list.count() == tab.annotations_list.count() == 0
    tab.annotation_filter.setCurrentIndex(tab.annotation_filter.findData("custom"))
    assert tab.annotations_list.count() == 3839
    tab.annotation_filter.setCurrentIndex(tab.annotation_filter.findData("all"))
    assert tab.annotations_list.count() == 0


def test_cancel_import_keeps_previous_frames_and_results(tab, app, monkeypatch, tmp_path):
    tab.genotype_df, tab.phenotype_df = frames(2)
    tab.populate_columns()
    previous_frame = tab.phenotype_df
    tab.run_analysis()
    wait(app, lambda: tab.worker is None)
    previous_analysis = tab.analysis
    entered, release = threading.Event(), threading.Event()
    def held(*args, **kwargs):
        entered.set()
        release.wait(5)
        return pd.DataFrame({"Sample": ["replacement"], "Phenotype": [999]})
    monkeypatch.setattr(hp, "read_table", held)
    tab.start_table_import(tmp_path / "replacement.csv")
    wait(app, entered.is_set)
    tab.cancel_task()
    release.set()
    wait(app, lambda: tab.worker is None)
    assert tab.phenotype_df is previous_frame
    assert tab.analysis is previous_analysis
    assert tab.export_button.isEnabled()


def test_failed_import_keeps_previous_table(tab, app, tmp_path):
    tab.genotype_df, tab.phenotype_df = frames(2)
    tab.populate_columns()
    previous = tab.phenotype_df
    tab.start_table_import(tmp_path / "not_present.csv")
    wait(app, lambda: tab.worker is None)
    assert tab.phenotype_df is previous
    assert tab.controls.isEnabled()


def test_changed_columns_refresh_in_background_after_import(tab, app, monkeypatch, tmp_path):
    genotype, phenotype = frames(2)
    phenotype["Length"] = phenotype.Phenotype + 10
    tab.genotype_df = genotype
    tab.populate_columns()
    source = tmp_path / "table.csv"
    phenotype.to_csv(source, index=False)
    tab.start_table_import(source)
    wait(app, lambda: tab.worker is None)
    calls = []
    original = hp.prepare_data
    def record(*args, **kwargs):
        calls.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(hp, "prepare_data", record)
    tab.check_all(tab.trait_list, True)
    wait(app, lambda: bool(calls) and tab.worker is None and not tab._metadata_pending)
    assert all(thread != threading.get_ident() for thread in calls)
