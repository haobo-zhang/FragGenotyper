"""Selection and appearance regressions for the standalone heatmap renderer."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure


SPEC = importlib.util.spec_from_file_location("genotype_plot_under_test", Path(__file__).resolve().parents[1] / "genotype_plot.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


@pytest.fixture
def observations():
    matrix = np.array([[0, np.nan, 2], [1, 0, 2], [2, 2, 0], [0, 1, 1], [1, 2, np.nan]], dtype=float)
    data = SimpleNamespace(samples=["001", "002", "003", "004", "005"],
                           positions=[("A01", 10), ("A01", 20), ("A02", 30)], matrix=matrix)
    groups = [[0, 1], [2], [3, 4]]
    # Deliberately shuffle result rows: labels must be looked up by Sample.
    result = pd.DataFrame({"Sample": ["005", "001", "003", "002", "004"],
                           "Genotype": ["Hap009", "Hap001", "Hap005", "Hap001", "Hap009"]})
    return data, groups, result


def figure():
    result = Figure(figsize=(8, 5), constrained_layout=True)
    FigureCanvasAgg(result)
    return result


def test_default_draws_all_members_and_preserves_nan(observations):
    data, groups, result = observations
    before = data.matrix.copy()
    axis = plot.draw_genotype_heatmap(figure(), data, groups, result)
    rendered = axis.images[0].get_array()
    np.testing.assert_equal(rendered.filled(np.nan), data.matrix)
    np.testing.assert_equal(np.ma.getmaskarray(rendered), np.isnan(data.matrix))
    np.testing.assert_equal(data.matrix, before)
    assert [tick.get_text() for tick in axis.get_yticklabels()] == ["Hap001", "Hap005", "Hap009"]
    np.testing.assert_allclose(axis.images[0].cmap.get_bad(), to_rgba("#D9D9D9"))
    for value, color in [(0, "#e7ddb8"), (1, "#b186a3"), (2, "#684e94")]:
        np.testing.assert_allclose(axis.images[0].cmap(axis.images[0].norm(value)), to_rgba(color))


def test_filter_uses_original_group_order_and_does_not_renumber(observations):
    data, groups, result = observations
    chosen = ["Hap009", "Hap001"]
    groups_before = [list(members) for members in groups]
    axis = plot.draw_genotype_heatmap(figure(), data, groups, result, selected_labels=chosen)
    np.testing.assert_equal(axis.images[0].get_array().filled(np.nan), data.matrix[[0, 1, 3, 4]])
    assert [tick.get_text() for tick in axis.get_yticklabels()] == ["Hap001", "Hap009"]
    np.testing.assert_equal(axis.get_yticks(), [0.5, 2.5])
    assert groups == groups_before
    assert chosen == ["Hap009", "Hap001"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_labels_are_adjacent_on_requested_side_and_all_frames_are_removed(observations, side):
    data, groups, result = observations
    fig = figure()
    axis = plot.draw_genotype_heatmap(fig, data, groups, result, label_side=side)
    fig.canvas.draw()
    for tick in axis.yaxis.get_major_ticks():
        assert tick.label1.get_visible() == (side == "left")
        assert tick.label2.get_visible() == (side == "right")
    assert axis.yaxis.get_ticks_position() in {side, "unknown"}
    for chart_axis in fig.axes:
        assert all(not spine.get_visible() for spine in chart_axis.spines.values())
        for tick in chart_axis.xaxis.get_major_ticks() + chart_axis.yaxis.get_major_ticks():
            assert tick.tick1line.get_markersize() == 0
            assert tick.tick2line.get_markersize() == 0
    colorbar = axis.images[0].colorbar
    assert not colorbar.outline.get_visible()
    assert not colorbar.dividers.get_visible()
    heatmap_box = axis.get_position()
    colorbar_box = colorbar.ax.get_position()
    if side == "right":
        assert colorbar_box.x1 < heatmap_box.x0
    else:
        assert colorbar_box.x0 > heatmap_box.x1


def test_empty_selection_clears_previous_plot_without_reverting_to_all(observations):
    fig = figure()
    plot.draw_genotype_heatmap(fig, *observations)
    assert len(fig.axes) == 2
    assert plot.draw_genotype_heatmap(fig, *observations, selected_labels=[]) is None
    assert not fig.axes
    assert any("Select at least one" in text.get_text() for text in fig.texts)
    fig.canvas.draw()


def test_unknown_label_and_invalid_side_are_explicit_errors(observations):
    fig = figure()
    with pytest.raises(ValueError, match="Unknown genotype.*Hap404"):
        plot.draw_genotype_heatmap(fig, *observations, selected_labels=["Hap404"])
    with pytest.raises(ValueError, match="label_side"):
        plot.draw_genotype_heatmap(fig, *observations, label_side="center")


def test_single_group_keeps_its_label_and_members(observations):
    data, groups, result = observations
    fig = figure()
    axis = plot.draw_genotype_heatmap(fig, data, groups, result, selected_labels=["Hap005"])
    assert axis.images[0].get_array().shape == (1, 3)
    assert [tick.get_text() for tick in axis.get_yticklabels()] == ["Hap005"]
    np.testing.assert_equal(axis.images[0].get_array(), data.matrix[[2]])
    fig.canvas.draw()


def test_more_than_fifty_groups_all_keep_visible_original_labels():
    count = 61
    samples = [f"s{i}" for i in range(count)]
    labels = [f"OriginalHap{i * 3:03d}" for i in range(count)]
    data = SimpleNamespace(samples=samples, positions=[("1", 1)], matrix=(np.arange(count) % 3).reshape(-1, 1).astype(float))
    result = pd.DataFrame({"Sample": samples, "Genotype": labels})
    fig = figure()
    fig.set_size_inches(8, 16)
    axis = plot.draw_genotype_heatmap(fig, data, [[i] for i in range(count)], result)
    fig.canvas.draw()
    ticks = axis.get_yticklabels()
    assert [tick.get_text() for tick in ticks] == labels
    assert all(tick.get_visible() for tick in ticks)


def test_all_nan_group_renders_gray_with_fixed_dosage_colorbar(tmp_path):
    data = SimpleNamespace(samples=["missing"], positions=[("1", 12), ("1", 24)], matrix=np.full((1, 2), np.nan))
    result = pd.DataFrame({"Sample": ["missing"], "Genotype": ["Hap777"]})
    fig = figure()
    axis = plot.draw_genotype_heatmap(fig, data, [[0]], result)
    assert np.ma.getmaskarray(axis.images[0].get_array()).all()
    assert axis.images[0].colorbar.get_ticks().tolist() == [0, 1, 2]
    for suffix in ["png", "pdf"]:
        path = tmp_path / f"all_missing.{suffix}"
        fig.savefig(path)
        assert path.stat().st_size > 0


def test_inconsistent_group_members_are_not_silently_mislabelled(observations):
    data, _, result = observations
    with pytest.raises(ValueError, match="same original genotype label"):
        plot.draw_genotype_heatmap(figure(), data, [[0, 2]], result)
