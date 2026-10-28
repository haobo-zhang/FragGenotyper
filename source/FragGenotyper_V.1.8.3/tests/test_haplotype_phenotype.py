"""Behavioral checks for the selected-haplotype phenotype workflow.

Run from the repository root with::

    python3 -m pytest fragGenotyper/V.1.7.4/tests/test_haplotype_phenotype.py

These tests use SciPy as an independent statistical oracle and save actual
figures through the noninteractive Matplotlib canvas; no Qt session is needed.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from scipy import stats


MODULE_PATH = Path(__file__).resolve().parents[1] / "haplotype_phenotype.py"
SPEC = importlib.util.spec_from_file_location("haplotype_phenotype", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
hp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hp
SPEC.loader.exec_module(hp)

RESULT_COLUMNS = {
    "Group1", "Group2", "n1", "n2", "mean1", "mean2", "mean_difference",
    "t_stat", "p_value", "p_adjusted", "significance", "status",
}


def phenotype_data(groups: dict[str, list[float]]) -> pd.DataFrame:
    """Construct observations, without implementing any production logic."""
    rows = [
        {"Sample": f"{group}_{i}", "Genotype": group, "Phenotype": value}
        for group, values in groups.items()
        for i, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


def plotted_text(figure: Figure) -> str:
    return "\n".join(item.get_text() for item in figure.findobj(Text))


@pytest.mark.parametrize("separator", [",", "\t", "   "])
def test_read_table_preserves_identifiers_for_supported_delimiters(tmp_path, separator):
    path = tmp_path / "input.txt"
    path.write_text(
        separator.join(["Sample", "Genotype", "Phenotype"]) + "\n"
        + separator.join(["001", "001", "1.5"]) + "\n"
        + separator.join(["002", "010", "2.5"]) + "\n",
        encoding="utf-8",
    )

    table = hp.read_table(path)

    assert table["Sample"].tolist() == ["001", "002"]
    assert table["Genotype"].tolist() == ["001", "010"]
    assert pd.to_numeric(table["Phenotype"]).tolist() == [1.5, 2.5]


def test_prepare_two_tables_matches_by_sample_and_supports_custom_columns():
    genotypes = pd.DataFrame({"ID": ["001", "002", "003"], "Hap": ["01", "02", "03"]})
    phenotypes = pd.DataFrame({"plant": ["002", "001", "004"], "height": [20, 10, 40]})

    data, report = hp.prepare_data(
        genotypes, phenotypes, sample_col="ID", group_col="Hap",
        trait_col="height", phenotype_sample_col="plant",
    )

    assert {"Sample", "Genotype", "Phenotype"}.issubset(data.columns)
    actual = data.set_index("Sample")
    assert set(actual.index) == {"001", "002"}
    assert actual.loc["001", "Genotype"] == "01"
    assert actual.loc["002", "Genotype"] == "02"
    assert actual.loc["001", "Phenotype"] == 10
    assert actual.loc["002", "Phenotype"] == 20
    assert report


@pytest.mark.parametrize("duplicate_side", ["genotypes", "phenotypes"])
def test_duplicate_sample_ids_are_rejected_in_either_input(duplicate_side):
    genotypes = pd.DataFrame({"Sample": ["a", "b"], "Genotype": ["A", "B"]})
    phenotypes = pd.DataFrame({"Sample": ["a", "b"], "Phenotype": [1, 2]})
    if duplicate_side == "genotypes":
        genotypes = pd.concat([genotypes, genotypes.iloc[[0]]], ignore_index=True)
    else:
        phenotypes = pd.concat([phenotypes, phenotypes.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError):
        hp.prepare_data(genotypes, phenotypes)


def test_merged_table_needs_no_sample_and_removes_invalid_observations():
    merged = pd.DataFrame({
        "Genotype": ["001", "010", "", "   ", None, "A", "A", "A", "A"],
        "Phenotype": ["1.5", 2.5, 3, 4, 5, np.nan, np.inf, "-inf", "not a number"],
    })

    data, report = hp.prepare_data(merged)

    assert {"Sample", "Genotype", "Phenotype"}.issubset(data.columns)
    assert data["Genotype"].tolist() == ["001", "010"]
    assert data["Phenotype"].tolist() == [1.5, 2.5]
    assert np.isfinite(data["Phenotype"]).all()
    assert report


def test_welch_statistics_match_scipy_with_unequal_variance_and_sample_size():
    # Pooled-variance Student's t gives a materially different answer here.
    groups = {"001": [1, 2, 2, 3], "010": [0, 3, 10, 20, 40, 50]}
    data = phenotype_data(groups)

    results, summary = hp.compare_groups(data, ["001", "010"], correction="none")

    assert RESULT_COLUMNS.issubset(results.columns)
    assert len(results) == 1
    row = results.iloc[0]
    left, right = groups[row["Group1"]], groups[row["Group2"]]
    oracle = stats.ttest_ind(left, right, equal_var=False)
    assert row["n1"] == len(left)
    assert row["n2"] == len(right)
    assert row["mean1"] == pytest.approx(np.mean(left))
    assert row["mean2"] == pytest.approx(np.mean(right))
    assert row["mean_difference"] == pytest.approx(np.mean(left) - np.mean(right))
    assert row["t_stat"] == pytest.approx(oracle.statistic)
    assert row["p_value"] == pytest.approx(oracle.pvalue)
    assert row["p_adjusted"] == pytest.approx(oracle.pvalue)
    assert not summary.empty


def test_bh_adjustment_matches_known_values_and_independent_scipy_oracle():
    data = phenotype_data({"A": [1, 2, 3], "B": [2, 3, 4], "C": [5, 6, 7]})

    results, _ = hp.compare_groups(data, ["A", "B", "C"], correction="bh")

    # Fixed expected values for three real Welch comparisons, not synthetic p-values.
    expected = {
        frozenset(("A", "B")): 0.2878641347266908,
        frozenset(("A", "C")): 0.02414967930251316,
        frozenset(("B", "C")): 0.03196746169313509,
    }
    assert len(results) == 3
    for _, row in results.iterrows():
        pair = frozenset((row["Group1"], row["Group2"]))
        assert row["p_adjusted"] == pytest.approx(expected[pair])
    np.testing.assert_allclose(
        results["p_adjusted"],
        stats.false_discovery_control(results["p_value"].to_numpy(), method="bh"),
    )
    ordered = results.sort_values("p_value")["p_adjusted"].to_numpy()
    assert np.all(np.diff(ordered) >= 0)
    assert np.all(results["p_adjusted"] >= results["p_value"])


def test_pairwise_comparison_uses_only_the_selected_haplotypes():
    data = phenotype_data({"A": [1, 2, 3], "B": [4, 5, 6], "C": [10, 11, 12], "unused": [7]})

    results, _ = hp.compare_groups(data, ["C", "A"], correction="none")

    assert len(results) == 1
    assert set(results.iloc[0][["Group1", "Group2"]]) == {"A", "C"}
    assert results.iloc[0]["p_value"] == pytest.approx(
        stats.ttest_ind([10, 11, 12], [1, 2, 3], equal_var=False).pvalue
    )


def test_reference_comparison_includes_only_the_reference_pairs():
    data = phenotype_data({"A": [1, 2, 3], "B": [4, 5, 6], "C": [10, 11, 12], "unused": [7]})

    results, _ = hp.compare_groups(
        data, ["A", "B", "C"], mode="reference", reference="B", correction="bh",
    )

    pairs = {frozenset((row["Group1"], row["Group2"])) for _, row in results.iterrows()}
    assert pairs == {frozenset(("A", "B")), frozenset(("B", "C"))}
    np.testing.assert_allclose(
        results["p_adjusted"], stats.false_discovery_control(results["p_value"].to_numpy()),
    )


@pytest.mark.parametrize("selection", [["A"], ["A", "missing"], ["A", "small"]])
def test_invalid_selection_or_insufficient_group_size_is_rejected(selection):
    data = phenotype_data({"A": [1, 2, 3], "B": [4, 5, 6], "small": [1]})

    with pytest.raises(ValueError):
        hp.compare_groups(data, selection)


def test_reference_must_belong_to_the_selected_groups():
    data = phenotype_data({"A": [1, 2], "B": [4, 5], "C": [10, 11]})

    with pytest.raises(ValueError):
        hp.compare_groups(data, ["A", "B"], mode="reference", reference="C")


def test_identical_constant_groups_are_undefined_and_never_significant():
    data = phenotype_data({"A": [4, 4, 4], "B": [4, 4, 4]})

    results, _ = hp.compare_groups(data, ["A", "B"])

    row = results.iloc[0]
    assert row["status"] == "undefined"
    assert pd.isna(row["p_value"])
    assert pd.isna(row["p_adjusted"])
    assert "*" not in str(row["significance"])
    figure = hp.plot_comparison(data, ["A", "B"], results)
    FigureCanvasAgg(figure).draw()
    assert "n/a" in plotted_text(figure).lower()
    figure.clear()


def test_undefined_pair_does_not_break_bh_for_other_comparisons():
    data = phenotype_data({"A": [4, 4, 4], "B": [4, 4, 4], "C": [1, 2, 3]})

    results, _ = hp.compare_groups(data, ["A", "B", "C"], correction="bh")

    undefined = results[results["status"] == "undefined"]
    assert len(undefined) == 1
    assert undefined["p_adjusted"].isna().all()
    finite = results[np.isfinite(results["p_value"])]
    assert len(finite) == 2
    np.testing.assert_allclose(
        finite["p_adjusted"], stats.false_discovery_control(finite["p_value"].to_numpy()),
    )


def test_plot_annotates_p_values_and_exports_real_png_and_pdf(tmp_path):
    data = phenotype_data({"A": [1, 2, 3], "B": [2, 3, 4], "C": [5, 6, 7]})
    results, _ = hp.compare_groups(data, ["A", "B", "C"], correction="bh")

    figure = hp.plot_comparison(
        data, ["A", "B", "C"], results, trait_label="Plant height (cm)", correction="bh",
    )

    assert isinstance(figure, Figure)
    FigureCanvasAgg(figure).draw()
    assert figure.axes[0].get_ylabel() == "Plant height (cm)"
    text = plotted_text(figure)
    assert re.search(r"[pPqQ].*\d", text)
    assert "*" in text
    assert re.search(r"\bns\b|not significant|不显著", text, flags=re.IGNORECASE)
    png_path, pdf_path = tmp_path / "comparison.png", tmp_path / "comparison.pdf"
    figure.savefig(png_path, dpi=100, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.read_bytes().startswith(b"%PDF-")
    assert png_path.stat().st_size > 1000
    assert pdf_path.stat().st_size > 1000
    figure.clear()


def test_plot_keeps_extreme_observations_and_comparison_brackets_inside_axes():
    data = phenotype_data({"001": [-1_000_000, 0, 1, 2], "010": [1, 2, 3, 1_000_000], "omit": [0, 1]})
    results, _ = hp.compare_groups(data, ["001", "010"], correction="none")

    figure = hp.plot_comparison(data, ["001", "010"], results, correction="none")

    FigureCanvasAgg(figure).draw()
    axis = figure.axes[0]
    low, high = axis.get_ylim()
    assert low < -1_000_000
    assert high > 1_000_000
    labels = "\n".join(label.get_text() for label in axis.get_xticklabels())
    assert "001" in labels and "010" in labels
    assert "omit" not in labels
    # Every bracket/summary line remains visible, including above extreme values.
    for line in axis.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        finite_y = y[np.isfinite(y)]
        if finite_y.size:
            assert finite_y.min() >= low
            assert finite_y.max() <= high
    figure.clear()
