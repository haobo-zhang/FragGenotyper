"""Exercise project restoration, worker lifecycle and real phenotype exports."""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFileDialog
from scipy import stats

import haplotype_phenotype as hp
from phenotype_tab import PhenotypeComparisonTab


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_for_worker(tab, app, timeout=30):
    deadline = time.monotonic() + timeout
    while tab.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert tab.worker is None, "The background worker did not finish."


@pytest.fixture
def tab(app, monkeypatch):
    widget = PhenotypeComparisonTab(language="en")
    widget.test_errors = []
    monkeypatch.setattr(widget, "show_error", lambda exc: widget.test_errors.append(str(exc)))
    yield widget
    if widget.worker is not None:
        widget.cancel_task()
        wait_for_worker(widget, app)
    widget.clear_plot()
    widget.close()
    widget.deleteLater()
    app.processEvents()


def combined_table(include_sparse=False):
    frame = pd.DataFrame({
        "Hap": ["001"] * 3 + ["010"] * 3 + ["100"] * 3,
        "Yolo": [1, 2, 3, 2, 4, 8, 10, 11, 15],
        "Weight": [20, 21, 25, 5, 9, 10, 7, 8, 9],
    })
    if include_sparse:
        frame["Sparse"] = [1, np.nan, np.nan, 2, 3, 4, 5, 6, 7]
    return frame


def load_combined(tab, frame=None):
    tab.input_mode.setCurrentIndex(tab.input_mode.findData("combined"))
    tab.genotype_df = combined_table() if frame is None else frame.copy()
    tab.genotype_path = None
    tab.populate_columns()
    assert tab.group_combo.currentText() == "Hap"
    assert tab.selected_groups() == ["001", "010", "100"]


def select_values(tab, widget, values):
    values = set(values)
    widget.blockSignals(True)
    for index in range(widget.count()):
        item = widget.item(index)
        item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in values
                           else Qt.CheckState.Unchecked)
    widget.blockSignals(False)
    if widget is tab.trait_list:
        tab.traits_changed()
    elif widget is tab.group_list:
        tab.groups_changed()


def select_pairs(widget, pairs):
    desired = {frozenset(pair) for pair in pairs}
    for index in range(widget.count()):
        item = widget.item(index)
        item.setCheckState(Qt.CheckState.Checked if frozenset(item.data(Qt.ItemDataRole.UserRole)) in desired
                           else Qt.CheckState.Unchecked)


def set_display_style(tab):
    tab.title_edit.setText("Selected haplotypes")
    tab.y_label_edit.setText("Length (mm)")
    tab.font_spin.setValue(15)
    tab.point_spin.setValue(37)
    tab.points_check.setChecked(False)
    tab.means_check.setChecked(False)
    tab.palette_combo.setCurrentText("Set2")
    tab.annotation_style.setCurrentIndex(tab.annotation_style.findData("p"))
    tab.annotation_filter.setCurrentIndex(tab.annotation_filter.findData("custom"))
    select_pairs(tab.annotations_list, [("100", "001")])
    for row in range(tab.labels_table.rowCount()):
        group = tab.labels_table.item(row, 0).text()
        if group == "001":
            tab.labels_table.item(row, 1).setText("Hap alpha")
            tab.labels_table.item(row, 2).setText("#123456")
        elif group == "100":
            tab.labels_table.item(row, 1).setText("Reference line")
            tab.labels_table.item(row, 2).setText("#654321")


@pytest.mark.parametrize("mode", ["reference", "custom"])
def test_project_restores_traits_order_comparisons_and_all_plot_controls(tab, app, mode):
    load_combined(tab)
    select_values(tab, tab.trait_list, ["Yolo", "Weight"])
    tab.group_list.setCurrentRow(2)
    tab.move_group(-1)
    tab.move_group(-1)
    if mode == "reference":
        select_values(tab, tab.group_list, ["001", "100"])
    tab.comparison_combo.setCurrentIndex(tab.comparison_combo.findData(mode))
    tab.reference_combo.setCurrentText("100")
    if mode == "custom":
        select_pairs(tab.pair_list, [("001", "010"), ("100", "001")])
    tab.correction_combo.setCurrentIndex(tab.correction_combo.findData("none"))
    tab.scope_combo.setCurrentIndex(tab.scope_combo.findData("global"))
    set_display_style(tab)
    saved = tab.project_settings()
    untouched = copy.deepcopy(saved)
    restored = PhenotypeComparisonTab(language="en")
    try:
        restored.apply_project(saved)

        actual = restored.project_settings()
        assert saved == untouched
        assert actual["columns"] == saved["columns"]
        assert actual["analysis"] == saved["analysis"]
        assert actual["group_order"] == ["100", "001", "010"]
        assert actual["plot"] == saved["plot"]
        assert restored.checked(restored.trait_list) == ["Yolo", "Weight"]
        assert restored.analysis is None and restored.canvas is None
        assert restored.worker is None
        assert not restored.export_button.isEnabled()
        assert restored._error is None
    finally:
        restored.close()
        restored.deleteLater()
        app.processEvents()


def test_multitrait_worker_results_and_trait_switch_use_the_correct_columns(tab, app):
    frame = combined_table(include_sparse=True)
    load_combined(tab, frame)
    select_values(tab, tab.trait_list, ["Yolo", "Weight", "Sparse"])
    tab.scope_combo.setCurrentIndex(tab.scope_combo.findData("global"))

    tab.run_analysis()
    wait_for_worker(tab, app)

    assert not tab.test_errors
    assert set(tab.analysis["analyses"]) == {"Yolo", "Weight"}
    assert set(tab.analysis["failures"]) == {"Sparse"}
    all_results = pd.concat([value["results"] for value in tab.analysis["analyses"].values()], ignore_index=True)
    np.testing.assert_allclose(all_results.p_adjusted, stats.false_discovery_control(all_results.p_value.to_numpy()))
    for trait in ["Yolo", "Weight"]:
        tab.result_trait_combo.setCurrentText(trait)
        app.processEvents()
        assert tab.canvas is not None
        assert tab.canvas.figure.axes[0].get_ylabel() == trait
        frame_results = tab.analysis["analyses"][trait]["results"]
        headers = [tab.test_table.horizontalHeaderItem(i).text() for i in range(tab.test_table.columnCount())]
        assert headers == list(frame_results.columns)
        assert tab.test_table.rowCount() == 3
        p_index, first_index, second_index = map(headers.index, ["p_value", "Group1", "Group2"])
        for row in range(tab.test_table.rowCount()):
            first = tab.test_table.item(row, first_index).text()
            second = tab.test_table.item(row, second_index).text()
            oracle = stats.ttest_ind(frame.loc[frame.Hap == first, trait], frame.loc[frame.Hap == second, trait], equal_var=False)
            assert float(tab.test_table.item(row, p_index).text()) == pytest.approx(oracle.pvalue, rel=1e-5)
        summary_headers = [tab.summary_table.horizontalHeaderItem(i).text() for i in range(tab.summary_table.columnCount())]
        mean_index = summary_headers.index("mean")
        group_index = summary_headers.index("Genotype")
        for row in range(tab.summary_table.rowCount()):
            group = tab.summary_table.item(row, group_index).text()
            assert float(tab.summary_table.item(row, mean_index).text()) == pytest.approx(frame.loc[frame.Hap == group, trait].mean(), rel=1e-5)
    analysis = tab.analysis
    tab.title_edit.setText("Reformatted result")
    tab.title_edit.editingFinished.emit()
    tab.font_spin.setValue(16)
    assert tab.analysis is analysis
    assert tab.worker is None
    assert tab.canvas.figure.axes[0].get_title() == "Reformatted result"
    assert tab.export_button.isEnabled()

    select_values(tab, tab.group_list, ["001", "100"])
    assert tab.analysis is None and tab.canvas is None
    assert tab.test_table.rowCount() == 0
    assert not tab.export_button.isEnabled()


def test_second_run_does_not_replace_a_running_worker(tab, app, monkeypatch):
    load_combined(tab)
    real_analysis = hp.analyze_traits
    gate = threading.Event()
    calls = []

    def held_analysis(*args, **kwargs):
        calls.append(1)
        assert gate.wait(10)
        return real_analysis(*args, **kwargs)

    monkeypatch.setattr(hp, "analyze_traits", held_analysis)
    try:
        tab.run_analysis()
        first_worker = tab.worker
        assert first_worker is not None
        assert not tab.controls.isEnabled()
        tab.run_analysis()
        assert tab.worker is first_worker
    finally:
        gate.set()
        wait_for_worker(tab, app)
    assert len(calls) == 1
    assert tab.analysis is not None
    assert tab.controls.isEnabled()
    assert tab.export_button.isEnabled()


def test_cancelled_worker_discards_result_and_reenables_controls(tab, app, monkeypatch):
    load_combined(tab)
    real_analysis = hp.analyze_traits
    gate = threading.Event()

    def held_analysis(*args, **kwargs):
        assert gate.wait(10)
        return real_analysis(*args, **kwargs)

    monkeypatch.setattr(hp, "analyze_traits", held_analysis)
    try:
        tab.run_analysis()
        tab.cancel_task()
        assert not tab.cancel_button.isEnabled()
    finally:
        gate.set()
        wait_for_worker(tab, app)
    assert tab.analysis is None and tab.canvas is None
    assert tab.controls.isEnabled() and tab.run_button.isEnabled()
    assert not tab.export_button.isEnabled()
    assert "Cancelled" in tab.status_label.text()


def test_missing_source_does_not_replace_the_open_project_or_its_results(tab, app, tmp_path):
    load_combined(tab)
    tab.run_analysis()
    wait_for_worker(tab, app)
    analysis, canvas = tab.analysis, tab.canvas
    before = tab.project_settings()
    invalid = copy.deepcopy(before)
    invalid["inputs"]["genotype_file"] = str(tmp_path / "not_here.csv")

    with pytest.raises((OSError, ValueError)):
        tab.apply_project(invalid)

    assert tab.analysis is analysis
    assert tab.canvas is canvas
    assert tab.project_settings() == before
    assert tab.export_button.isEnabled()


def test_real_background_export_contains_images_statistics_and_matching_metadata(tab, app, tmp_path, monkeypatch):
    load_combined(tab)
    select_values(tab, tab.trait_list, ["Yolo", "Weight"])
    select_values(tab, tab.group_list, ["001", "100"])
    tab.annotation_filter.setCurrentIndex(tab.annotation_filter.findData("none"))
    tab.run_analysis()
    wait_for_worker(tab, app)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(tmp_path))

    tab.export_analysis()
    wait_for_worker(tab, app)

    output = tmp_path / "FragGenotyper_analysis"
    assert output.is_dir()
    assert not list(tmp_path.glob(".fraggenotyper-export-*"))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["analysis"]["selected_groups"] == ["001", "100"]
    assert metadata["plot"]["annotation_filter"] == "none"
    assert set(metadata["trait_files"]) == {"Yolo", "Weight"}
    assert not metadata["failures"]
    all_tests = pd.read_csv(output / "all_tests.csv", dtype={"Group1": str, "Group2": str})
    assert set(all_tests.Trait) == {"Yolo", "Weight"}
    assert set(all_tests.Group1) | set(all_tests.Group2) == {"001", "100"}
    for trait, stem in metadata["trait_files"].items():
        assert (output / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (output / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
        assert (output / f"{stem}_tests.csv").is_file()
        assert (output / f"{stem}_summary.csv").is_file()
        observations = pd.read_csv(output / f"{stem}_data.csv", dtype={"Genotype": str})
        assert set(observations.Genotype) == {"001", "100"}
        assert len(observations) == 6
    assert tab.export_button.isEnabled()


def test_zero_valid_observations_in_first_trait_does_not_hide_a_group_for_other_traits(tab, app):
    frame = combined_table()
    frame.loc[frame.Hap == "001", "Yolo"] = np.nan
    load_combined(tab, frame)
    select_values(tab, tab.trait_list, ["Yolo", "Weight"])
    assert "001" in tab.selected_groups()
    tab.run_analysis()
    wait_for_worker(tab, app)
    assert set(tab.analysis["analyses"]) == {"Weight"}
    assert set(tab.analysis["failures"]) == {"Yolo"}


def test_custom_annotations_only_offer_comparisons_actually_tested(tab, app):
    load_combined(tab)
    tab.comparison_combo.setCurrentIndex(tab.comparison_combo.findData("custom"))
    select_pairs(tab.pair_list, [("001", "100")])
    assert tab.annotations_list.count() == 0
    tab.annotation_filter.setCurrentIndex(tab.annotation_filter.findData("custom"))
    assert tab.checked(tab.annotations_list) == [("001", "100")]
    tab.run_analysis()
    wait_for_worker(tab, app)
    assert len(tab.analysis["analyses"]["Yolo"]["results"]) == 1
    assert tab.canvas is not None
    tab.annotations_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert tab.canvas is not None
    assert tab.plot_options()["annotation_filter"] == "none"


def test_preview_keeps_axis_label_clear_of_footer_when_resized(tab, app):
    load_combined(tab)
    tab.resize(1080, 700)
    tab.show()
    tab.run_analysis()
    wait_for_worker(tab, app)
    app.processEvents()
    tab.canvas.draw()
    renderer = tab.canvas.get_renderer()
    figure = tab.canvas.figure
    footer_top = max(text.get_window_extent(renderer).y1 for text in figure.texts)
    assert figure.axes[0].xaxis.label.get_window_extent(renderer).y0 > footer_top + 3
