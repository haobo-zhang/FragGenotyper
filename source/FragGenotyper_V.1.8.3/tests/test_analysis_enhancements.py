"""Regression checks for Welch intervals, batch families and plot controls."""
from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from scipy import stats


MODULE_PATH = Path(__file__).resolve().parents[1] / "haplotype_phenotype.py"
SPEC = importlib.util.spec_from_file_location("enhanced_haplotype_phenotype", MODULE_PATH)
hp = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = hp
SPEC.loader.exec_module(hp)


def observations(groups):
    return pd.DataFrame([
        {"Sample": f"{group}_{index}", "Genotype": group, "Phenotype": value}
        for group, values in groups.items() for index, value in enumerate(values)
    ])


def multi_trait_data():
    return pd.DataFrame({
        "Sample": [f"s{index}" for index in range(12)],
        "Hap": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        "height": [1, 2, 3, 5, 2, 3, 4, 7, 8, 10, 11, 15],
        "weight": [10, 11, 12, 16, 11, 13, 17, 20, 12, 14, 20, 25],
        "too_sparse": [1, None, None, None, 2, 3, 4, 5, 6, 7, 8, 9],
    })


@pytest.mark.parametrize("first,second", [
    ([1, 2, 2, 3], [0, 3, 10, 20, 40, 50]),
    ([4, 4, 4], [1, 2, 3, 6, 8]),
    ([-6, -2, 4, 9], [0, 1]),
])
def test_welch_confidence_interval_and_df_match_scipy(first, second):
    data = observations({"A": first, "B": second})
    result = hp.compare_groups(data, ["A", "B"], correction="none")[0].iloc[0]
    oracle = stats.ttest_ind(first, second, equal_var=False)
    interval = oracle.confidence_interval(confidence_level=0.95)

    assert result["df"] == pytest.approx(oracle.df)
    assert result["ci_lower"] == pytest.approx(interval.low)
    assert result["ci_upper"] == pytest.approx(interval.high)
    assert result["standard_error"] == pytest.approx(
        np.sqrt(np.var(first, ddof=1) / len(first) + np.var(second, ddof=1) / len(second))
    )


@pytest.mark.parametrize("second,expected_status", [([4, 4, 4], "undefined"),
                                                     ([7, 7, 7], "constant_different")])
def test_constant_groups_do_not_claim_zero_width_confidence_intervals(second, expected_status):
    result = hp.compare_groups(observations({"A": [4, 4, 4], "B": second}), ["A", "B"])[0].iloc[0]

    assert result["status"] == expected_status
    assert result["standard_error"] == 0
    assert pd.isna(result["df"])
    assert pd.isna(result["ci_lower"]) and pd.isna(result["ci_upper"])


def test_custom_pairs_are_exact_deduplicated_and_keep_first_direction():
    data = observations({"A": [1, 2, 3], "B": [4, 5, 6], "C": [8, 9, 12]})
    results, summary = hp.compare_groups(data, ["A", "B", "C"], mode="custom",
                                         pairs=[("C", "A"), ("A", "C"), ("B", "C")])

    assert list(zip(results.Group1, results.Group2)) == [("C", "A"), ("B", "C")]
    assert len(summary) == 3
    assert results.iloc[0].mean_difference > 0
    np.testing.assert_allclose(results.p_adjusted,
                               stats.false_discovery_control(results.p_value.to_numpy()))


@pytest.mark.parametrize("pairs", [None, [], [("A", "A")], [("A", "D")],
                                  [("A",)], ["AB"]])
def test_custom_pairs_reject_empty_self_unknown_and_malformed_pairs(pairs):
    data = observations({"A": [1, 2, 3], "B": [4, 5, 6]})
    with pytest.raises(ValueError):
        hp.compare_groups(data, ["A", "B"], mode="custom", pairs=pairs)


def test_reference_mode_is_not_changed_by_hidden_custom_pairs():
    data = observations({"A": [1, 2, 3], "B": [4, 5, 6], "C": [8, 9, 12]})
    for pairs in ([], [("A", "C")]):
        result = hp.compare_groups(data, ["A", "B", "C"], mode="reference",
                                   reference="B", pairs=pairs)[0]
        assert list(zip(result.Group1, result.Group2)) == [("B", "A"), ("B", "C")]


@pytest.mark.parametrize("scope", ["per_trait", "global"])
def test_batch_bh_matches_scipy_for_each_defined_family_and_reports_failures(scope):
    table = multi_trait_data()
    callbacks = []
    outcome = hp.analyze_traits(
        table, traits=["height", "too_sparse", "missing", "weight"],
        selected_groups=["A", "B", "C"], group_col="Hap", correction_scope=scope,
        progress_callback=lambda done, total, trait: callbacks.append((done, total, trait)),
    )
    assert set(outcome["analyses"]) == {"height", "weight"}
    assert set(outcome["failures"]) == {"too_sparse", "missing"}
    assert callbacks == [(1, 4, "height"), (2, 4, "too_sparse"),
                         (3, 4, "missing"), (4, 4, "weight")]
    frames = [outcome["analyses"][trait]["results"] for trait in ("height", "weight")]
    for trait, frame in zip(("height", "weight"), frames):
        assert frame.Trait.eq(trait).all()
        assert frame.correction.eq("bh").all()
        assert frame.correction_scope.eq(scope).all()
    if scope == "global":
        combined = pd.concat(frames, ignore_index=True)
        np.testing.assert_allclose(combined.p_adjusted,
                                   stats.false_discovery_control(combined.p_value.to_numpy()))
    else:
        for frame in frames:
            np.testing.assert_allclose(frame.p_adjusted,
                                       stats.false_discovery_control(frame.p_value.to_numpy()))


def test_batch_separate_tables_join_ids_and_filter_selected_groups():
    table = multi_trait_data()
    genotypes = table[["Sample", "Hap"]].rename(columns={"Sample": "Plant"})
    phenotypes = table[["Sample", "height", "weight"]].iloc[::-1]
    result = hp.analyze_traits(
        genotypes, phenotypes, traits=["height", "weight"], selected_groups=["B", "A"],
        sample_col="Plant", phenotype_sample_col="Sample", group_col="Hap",
        mode="custom", pairs=iter([("B", "A")]), correction="none", correction_scope="global",
    )
    assert not result["failures"]
    for analysis in result["analyses"].values():
        assert set(analysis["data"].Genotype) == {"A", "B"}
        assert analysis["report"]["selected_valid_rows"] == 8
        assert list(zip(analysis["results"].Group1, analysis["results"].Group2)) == [("B", "A")]
        np.testing.assert_allclose(analysis["results"].p_adjusted, analysis["results"].p_value)
    height = result["analyses"]["height"]["data"].set_index("Sample")
    assert height.loc["s0", "Phenotype"] == 1


def test_global_bh_excludes_only_undefined_p_values_and_failed_traits():
    table = pd.DataFrame({"Hap": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
                          "first": [4, 4, 4, 4, 4, 4, 1, 2, 3],
                          "second": [1, 2, 3, 5, 6, 7, 8, 10, 12]})
    outcome = hp.analyze_traits(table, traits=["first", "second"], group_col="Hap",
                                selected_groups=["A", "B", "C"], correction_scope="global")
    combined = pd.concat([analysis["results"] for analysis in outcome["analyses"].values()])
    finite = combined[np.isfinite(combined.p_value)]
    undefined = combined[~np.isfinite(combined.p_value)]
    assert len(undefined) == 1
    assert undefined.p_adjusted.isna().all()
    np.testing.assert_allclose(finite.p_adjusted,
                               stats.false_discovery_control(finite.p_value.to_numpy()))


def test_batch_cancellation_before_and_between_traits_never_returns_partial_global_bh():
    table = multi_trait_data()
    event = threading.Event()
    event.set()
    with pytest.raises(hp.AnalysisCancelled):
        hp.analyze_traits(table, traits=["height"], selected_groups=["A", "B"],
                          group_col="Hap", cancel_event=event)
    event.clear()
    callbacks = []

    def cancel_after_first(done, total, trait):
        callbacks.append(trait)
        event.set()

    with pytest.raises(InterruptedError):
        hp.analyze_traits(table, traits=["height", "weight"], selected_groups=["A", "B"],
                          group_col="Hap", correction_scope="global", cancel_event=event,
                          progress_callback=cancel_after_first)
    assert callbacks == ["height"]


def test_plot_controls_reorder_rename_and_color_without_changing_results():
    data = observations({"A": [1, 2, 3], "B": [2, 3, 4], "C": [8, 9, 12]})
    results = hp.compare_groups(data, ["A", "B", "C"])[0]
    before = results.copy(deep=True)
    figure = hp.plot_comparison(
        data, ["A", "B", "C"], results, group_order=["C", "A", "B"],
        group_labels={"C": "Target haplotype"}, group_colors={"C": "red", "A": "blue", "B": "green"},
        y_label="Height (cm)", title="Selected comparison", font_size=14,
        point_size=38, annotation_style="stars", annotation_filter="custom",
        annotation_pairs=[("A", "C")],
    )
    FigureCanvasAgg(figure).draw()
    axis = figure.axes[0]
    assert [label.get_text().split("\n")[0] for label in axis.get_xticklabels()] == ["Target haplotype", "A", "B"]
    assert axis.get_ylabel() == "Height (cm)"
    assert axis.get_title() == "Selected comparison"
    assert len(axis.texts) == 1
    assert "P" not in axis.texts[0].get_text()
    np.testing.assert_allclose(axis.collections[0].get_facecolors()[0], to_rgba("red", alpha=0.72))
    assert axis.collections[0].get_sizes()[0] == 38
    pd.testing.assert_frame_equal(results, before)
    figure.clear()


def test_plot_significant_filter_changes_only_annotations_and_keeps_all_points():
    data = observations({"A": [1, 2, 3], "B": [2, 3, 4], "C": [8, 9, 12]})
    results = hp.compare_groups(data, ["A", "B", "C"])[0]
    before = results.copy(deep=True)
    figure = hp.plot_comparison(data, ["A", "B", "C"], results,
                                annotation_filter="significant", annotation_style="p",
                                group_colors="Set2")
    FigureCanvasAgg(figure).draw()
    axis = figure.axes[0]
    assert len(axis.texts) == int(results.p_adjusted.lt(0.05).sum())
    assert all("P_adj =" in text.get_text() and "\n" not in text.get_text() for text in axis.texts)
    point_collections = [collection for collection in axis.collections if len(collection.get_offsets()) > 1]
    assert sum(len(collection.get_offsets()) for collection in point_collections) == len(data)
    pd.testing.assert_frame_equal(results, before)
    figure.clear()


def test_plot_can_hide_points_means_and_annotations_but_keeps_outlier_limits():
    data = observations({"A": [-10000, 1, 2, 3], "B": [4, 5, 6, 10000]})
    results = hp.compare_groups(data, ["A", "B"])[0]
    figure = hp.plot_comparison(data, ["A", "B"], results, show_points=False,
                                show_means=False, annotation_filter="none")
    FigureCanvasAgg(figure).draw()
    axis = figure.axes[0]
    assert not axis.collections
    assert not axis.texts
    lower, upper = axis.get_ylim()
    assert lower < -10000 and upper > 10000
    assert "diamonds" not in "\n".join(text.get_text() for text in figure.texts)
    figure.clear()


@pytest.mark.parametrize("kwargs", [
    {"group_order": ["A", "A"]}, {"group_order": ["A"]},
    {"group_colors": "not_a_palette"}, {"group_colors": {"A": "not_a_color"}},
    {"annotation_style": "nonsense"}, {"annotation_filter": "custom", "annotation_pairs": [("A", "A")]},
    {"font_size": 0}, {"point_size": -5},
])
def test_invalid_plot_options_are_rejected(kwargs):
    data = observations({"A": [1, 2, 3], "B": [4, 5, 6]})
    results = hp.compare_groups(data, ["A", "B"])[0]
    with pytest.raises(ValueError):
        hp.plot_comparison(data, ["A", "B"], results, **kwargs)


def test_many_group_brackets_never_share_overlapping_intervals_at_one_height():
    groups = {f"Hap{index}": [index, index + 0.2, index + 1] for index in range(7)}
    data = observations(groups)
    results = hp.compare_groups(data, list(groups))[0]
    figure = hp.plot_comparison(data, list(groups), results, font_size=14)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    axis = figure.axes[0]
    text_boxes = [text.get_window_extent(canvas.get_renderer()) for text in axis.texts]
    assert not [(index, other_index) for index, box in enumerate(text_boxes)
                for other_index, other in enumerate(text_boxes[index + 1:], start=index + 1)
                if box.overlaps(other)]
    bracket_lines = [line for line in axis.lines if len(line.get_xdata()) == 4]
    assert len(bracket_lines) == len(results)
    for index, first in enumerate(bracket_lines):
        x_first, y_first = np.asarray(first.get_xdata()), np.asarray(first.get_ydata())
        for second in bracket_lines[index + 1:]:
            x_second, y_second = np.asarray(second.get_xdata()), np.asarray(second.get_ydata())
            if y_first[0] == y_second[0]:
                assert x_first.max() < x_second.min() or x_first.min() > x_second.max()
    low, high = axis.get_ylim()
    for line in axis.lines:
        y_values = np.asarray(line.get_ydata(), dtype=float)
        assert np.all(y_values[np.isfinite(y_values)] >= low)
        assert np.all(y_values[np.isfinite(y_values)] <= high)
    figure.clear()
