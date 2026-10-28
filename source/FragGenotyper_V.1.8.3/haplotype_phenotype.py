"""Welch comparisons and publication-style plots for genotype groups.

This module has no Qt or pyplot dependency and does not select a Matplotlib
backend.  ``prepare_data`` is the boundary between input tables and analysis;
the remaining functions consume its Sample/Genotype/Phenotype table.
"""

from __future__ import annotations

import csv
import textwrap
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib import colormaps, font_manager
from matplotlib.colors import is_color_like
from matplotlib.text import Text
from scipy import stats


RESULT_COLUMNS = [
    "Group1", "Group2", "n1", "n2", "mean1", "mean2",
    "mean_difference", "standard_error", "df", "ci_lower", "ci_upper",
    "t_stat", "p_value", "p_adjusted", "significance", "status",
]
SUMMARY_COLUMNS = ["Genotype", "n", "mean", "std", "median", "min", "max"]


class AnalysisCancelled(InterruptedError):
    """Raised when a caller cancels a multi-trait analysis."""


from table_io import read_table


def _labels(series):
    """Use nullable strings: real missing values must never become 'nan'."""
    result = series.astype("string").str.strip()
    return result.mask(result.eq(""))


def _require_columns(table, columns, table_name):
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing column(s): {', '.join(map(str, missing))}")


def _reject_duplicate_samples(samples, table_name):
    duplicate_mask = samples.notna() & samples.duplicated(keep=False)
    if duplicate_mask.any():
        examples = samples[duplicate_mask].drop_duplicates().tolist()[:5]
        raise ValueError(f"{table_name} has duplicate nonempty Sample IDs: {', '.join(examples)}. "
                         "Each sample must occur once to avoid pseudoreplication.")


def _row_ids(count, occupied=()):
    used = set(occupied)
    result = []
    next_id = 1
    while len(result) < count:
        candidate = f"Row{next_id:06d}"
        next_id += 1
        if candidate not in used:
            used.add(candidate)
            result.append(candidate)
    return result


def prepare_data(genotype_df, phenotype_df=None, *, sample_col="Sample",
                 group_col="Genotype", trait_col="Phenotype", phenotype_sample_col=None):
    """Return a clean canonical table and row accounting.

    With two tables, nonempty sample IDs must be unique in each table and are
    joined one-to-one. ``matched_rows`` counts that inner join before filtering;
    ``dropped_rows`` counts matched rows with missing groups or invalid traits.
    Unmatched sample counts exclude empty IDs. With one table, every input row
    is initially matched, and absent/empty sample IDs receive generated row IDs.
    No input DataFrame is modified.
    """
    _require_columns(genotype_df, [group_col], "Genotype table")
    report = {
        "input_genotype_rows": int(len(genotype_df)),
        "input_phenotype_rows": int(len(phenotype_df)) if phenotype_df is not None else 0,
        "matched_rows": 0, "valid_rows": 0, "dropped_rows": 0,
        "unmatched_genotype_samples": 0, "unmatched_phenotype_samples": 0,
    }
    groups = _labels(genotype_df[group_col]).reset_index(drop=True)
    if phenotype_df is None:
        _require_columns(genotype_df, [trait_col], "Input table")
        if sample_col in genotype_df.columns:
            samples = _labels(genotype_df[sample_col]).reset_index(drop=True)
            _reject_duplicate_samples(samples, "Input table")
            empty = samples.isna()
            samples.loc[empty] = _row_ids(int(empty.sum()), samples.dropna())
        else:
            samples = pd.Series(_row_ids(len(genotype_df)), dtype="string")
        merged = pd.DataFrame({"Sample": samples, "Genotype": groups,
                               "Phenotype": genotype_df[trait_col].reset_index(drop=True)})
    else:
        phenotype_sample_col = phenotype_sample_col or sample_col
        _require_columns(genotype_df, [sample_col], "Genotype table")
        _require_columns(phenotype_df, [phenotype_sample_col, trait_col], "Phenotype table")
        genotype_samples = _labels(genotype_df[sample_col]).reset_index(drop=True)
        phenotype_samples = _labels(phenotype_df[phenotype_sample_col]).reset_index(drop=True)
        _reject_duplicate_samples(genotype_samples, "Genotype table")
        _reject_duplicate_samples(phenotype_samples, "Phenotype table")
        genotype_ids, phenotype_ids = set(genotype_samples.dropna()), set(phenotype_samples.dropna())
        report["unmatched_genotype_samples"] = len(genotype_ids - phenotype_ids)
        report["unmatched_phenotype_samples"] = len(phenotype_ids - genotype_ids)
        left = pd.DataFrame({"Sample": genotype_samples, "Genotype": groups})
        right = pd.DataFrame({"Sample": phenotype_samples,
                              "Phenotype": phenotype_df[trait_col].reset_index(drop=True)})
        # Pandas otherwise matches null keys with each other, unlike SQL.
        merged = left[left.Sample.notna()].merge(right[right.Sample.notna()], on="Sample",
                                                how="inner", validate="one_to_one", sort=False)
    report["matched_rows"] = int(len(merged))
    numeric = pd.to_numeric(merged["Phenotype"], errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    valid = merged["Genotype"].notna().to_numpy() & np.isfinite(values)
    data = merged.loc[valid, ["Sample", "Genotype", "Phenotype"]].copy()
    data["Sample"] = data["Sample"].astype(str)
    data["Genotype"] = data["Genotype"].astype(str)
    data["Phenotype"] = values[valid]
    data = data.reset_index(drop=True)
    report["valid_rows"] = int(len(data))
    report["dropped_rows"] = report["matched_rows"] - report["valid_rows"]
    return data, report


def _selected_values(data, selected_groups):
    _require_columns(data, ["Genotype", "Phenotype"], "Analysis table")
    if isinstance(selected_groups, (str, bytes)):
        raise ValueError("selected_groups must be a sequence of group labels.")
    selected = [str(group) for group in selected_groups]
    if len(selected) < 2:
        raise ValueError("Select at least two groups for comparison.")
    if len(set(selected)) != len(selected):
        raise ValueError("Selected group labels must be unique.")
    labels = _labels(data["Genotype"])
    values = {}
    for group in selected:
        series = pd.to_numeric(data.loc[labels.eq(group).fillna(False), "Phenotype"], errors="coerce")
        array = series.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(array).all():
            raise ValueError(f"Group {group!r} contains invalid traits; call prepare_data first.")
        if len(array) < 2:
            raise ValueError(f"Group {group!r} has {len(array)} valid values; at least two are required.")
        values[group] = array
    return selected, values


def _significance(p_value):
    if not np.isfinite(p_value):
        return "n/a"
    for threshold, stars in [(0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p_value < threshold:
            return stars
    return "ns"


def _adjust_bh(p_values):
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted
    order = np.argsort(values[finite_indices], kind="stable")
    ranked_indices = finite_indices[order]
    ranked = values[ranked_indices]
    corrected = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    adjusted[ranked_indices] = np.clip(corrected, 0, 1)
    return adjusted


def _validate_pairs(pairs, selected_groups, *, name="pairs"):
    """Deduplicate undirected pairs, retaining their first orientation."""
    if pairs is None or isinstance(pairs, (str, bytes)):
        raise ValueError(f"{name} must contain at least one pair of selected groups.")
    selected = set(selected_groups)
    unique, seen = [], set()
    try:
        iterator = iter(pairs)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of group pairs.") from exc
    for pair in iterator:
        if isinstance(pair, (str, bytes)):
            raise ValueError(f"Each item in {name} must contain exactly two group labels.")
        try:
            items = list(pair)
        except TypeError as exc:
            raise ValueError(f"Each item in {name} must contain two group labels.") from exc
        if len(items) != 2:
            raise ValueError(f"Each item in {name} must contain exactly two group labels.")
        first, second = map(str, items)
        if first == second:
            raise ValueError(f"A group cannot be compared with itself: {first!r}.")
        if first not in selected or second not in selected:
            raise ValueError(f"Every group in {name} must be among the selected groups.")
        key = frozenset((first, second))
        if key not in seen:
            seen.add(key)
            unique.append((first, second))
    if not unique:
        raise ValueError(f"{name} must contain at least one pair of selected groups.")
    return unique


def _comparison_pairs(selected, mode, reference=None, pairs=None):
    if mode == "custom":
        return _validate_pairs(pairs, selected)
    if mode == "reference":
        reference = str(reference) if reference is not None else None
        if reference not in selected:
            raise ValueError("The reference group must be among the selected groups.")
        return [(reference, group) for group in selected if group != reference]
    return [(group, other) for i, group in enumerate(selected) for other in selected[i + 1:]]


def _welch_interval(first, second, difference):
    """Return SE, Welch-Satterthwaite df and an unadjusted 95% interval.

    The weighted form of the df equation avoids squaring large variances.
    Zero variance in both groups supplies no estimable sampling uncertainty;
    it must not produce a reassuring zero-width confidence interval.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        components = np.array([np.var(first, ddof=1) / len(first),
                               np.var(second, ddof=1) / len(second)], dtype=float)
        variance = float(components.sum())
        standard_error = float(np.sqrt(variance))
        if not np.isfinite(variance) or variance <= 0:
            return standard_error, np.nan, np.nan, np.nan
        weights = components / variance
        degrees = float(1 / (weights[0] ** 2 / (len(first) - 1)
                             + weights[1] ** 2 / (len(second) - 1)))
        half_width = float(stats.t.ppf(0.975, degrees) * standard_error)
    if not np.isfinite(degrees) or not np.isfinite(half_width) or not np.isfinite(difference):
        return standard_error, degrees, np.nan, np.nan
    return standard_error, degrees, difference - half_width, difference + half_width


def compare_groups(data, selected_groups, *, mode="pairwise", reference=None,
                   pairs=None, correction="bh"):
    """Perform two-sided Welch tests; mean_difference is mean1 minus mean2.

    ``mode='custom'`` tests only the explicit, deduplicated ``pairs``. In other
    modes ``pairs`` is ignored, so switching modes cannot change a reference
    comparison through hidden custom selections. BH uses the finite P values
    in the tested family. CI limits are individual, unadjusted 95% Welch
    intervals; BH adjustment of P values does not adjust these intervals.

    Identical constant groups have no defined test (``status='undefined'``).
    Different constant groups retain SciPy's limiting P value and status
    ``constant_different``, but their df and confidence interval are NaN.
    """
    if mode not in {"pairwise", "reference", "custom"}:
        raise ValueError("mode must be 'pairwise', 'reference', or 'custom'.")
    if correction not in {"bh", "none"}:
        raise ValueError("correction must be 'bh' or 'none'.")
    selected, values = _selected_values(data, selected_groups)
    comparisons = _comparison_pairs(selected, mode, reference, pairs)
    rows = []
    for group1, group2 in comparisons:
        first, second = values[group1], values[group2]
        first_constant = bool(np.all(first == first[0]))
        second_constant = bool(np.all(second == second[0]))
        difference = float(np.mean(first) - np.mean(second))
        standard_error, degrees, ci_lower, ci_upper = _welch_interval(first, second, difference)
        if first_constant and second_constant and first[0] == second[0]:
            t_stat, p_value, status = np.nan, np.nan, "undefined"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                test = stats.ttest_ind(first, second, equal_var=False, alternative="two-sided")
            t_stat, p_value = float(test.statistic), float(test.pvalue)
            if np.isnan(t_stat) or not np.isfinite(p_value):
                status, t_stat, p_value = "undefined", np.nan, np.nan
                degrees, ci_lower, ci_upper = np.nan, np.nan, np.nan
            elif first_constant and second_constant:
                status = "constant_different"
            elif first_constant or second_constant:
                status = "one_constant_group"
            else:
                status = "ok"
        rows.append({"Group1": group1, "Group2": group2,
                     "n1": len(first), "n2": len(second),
                     "mean1": float(np.mean(first)), "mean2": float(np.mean(second)),
                     "mean_difference": difference, "standard_error": standard_error,
                     "df": degrees, "ci_lower": ci_lower, "ci_upper": ci_upper,
                     "t_stat": t_stat, "p_value": p_value, "status": status})
    results = pd.DataFrame(rows)
    results["p_adjusted"] = (_adjust_bh(results["p_value"]) if correction == "bh"
                             else results["p_value"].to_numpy(copy=True))
    results["significance"] = results["p_adjusted"].map(_significance)
    summary = pd.DataFrame([
        {"Genotype": group, "n": len(values[group]), "mean": np.mean(values[group]),
         "std": np.std(values[group], ddof=1), "median": np.median(values[group]),
         "min": np.min(values[group]), "max": np.max(values[group])}
        for group in selected
    ], columns=SUMMARY_COLUMNS)
    return results[RESULT_COLUMNS], summary


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled("Analysis cancelled.")


def analyze_traits(genotype_df, phenotype_df=None, *, traits, selected_groups,
                   sample_col="Sample", group_col="Genotype", phenotype_sample_col=None,
                   mode="pairwise", reference=None, pairs=None, correction="bh",
                   correction_scope="per_trait", progress_callback=None, cancel_event=None):
    """Analyze traits independently, then optionally adjust their shared family.

    Returns ``{'analyses': {trait: {'data', 'results', 'summary', 'report'}},
    'failures': {trait: message}, 'correction_scope': correction_scope}``.
    Each successful ``data`` contains only selected groups. A missing trait or
    insufficient group size fails only that trait; it contributes no fabricated
    test to global BH. Results include Trait/correction/correction_scope columns.
    ``progress_callback(completed, total, trait)`` runs after every trait,
    including failures. Cancellation between stages raises AnalysisCancelled
    and never returns an incomplete global correction as a finished analysis.
    """
    _check_cancelled(cancel_event)
    if isinstance(traits, (str, bytes)):
        raise ValueError("traits must be a sequence of column names.")
    trait_names = list(traits)
    if not trait_names or len(set(trait_names)) != len(trait_names):
        raise ValueError("traits must contain at least one unique column name.")
    if isinstance(selected_groups, (str, bytes)):
        raise ValueError("selected_groups must be a sequence of group labels.")
    selected = [str(group) for group in selected_groups]
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise ValueError("Select at least two unique groups for comparison.")
    if mode not in {"pairwise", "reference", "custom"}:
        raise ValueError("mode must be 'pairwise', 'reference', or 'custom'.")
    if correction not in {"bh", "none"}:
        raise ValueError("correction must be 'bh' or 'none'.")
    if correction_scope not in {"per_trait", "global"}:
        raise ValueError("correction_scope must be 'per_trait' or 'global'.")
    # Materialize custom iterators once so all traits test the same pairs.
    normalized_pairs = _comparison_pairs(selected, mode, reference, pairs)
    analyses, failures = {}, {}
    for completed, trait in enumerate(trait_names, start=1):
        _check_cancelled(cancel_event)
        try:
            data, report = prepare_data(
                genotype_df, phenotype_df, sample_col=sample_col, group_col=group_col,
                trait_col=trait, phenotype_sample_col=phenotype_sample_col,
            )
            _check_cancelled(cancel_event)
            results, summary = compare_groups(
                data, selected, mode=mode, reference=reference, pairs=normalized_pairs,
                correction=correction if correction_scope == "per_trait" else "none",
            )
            _check_cancelled(cancel_event)
            selected_data = data.loc[data["Genotype"].isin(selected)].copy()
            report["selected_valid_rows"] = int(len(selected_data))
            results.insert(0, "Trait", str(trait))
            results["correction"] = correction
            results["correction_scope"] = correction_scope
            analyses[trait] = {"data": selected_data, "results": results,
                               "summary": summary, "report": report}
        except AnalysisCancelled:
            raise
        except Exception as exc:
            failures[trait] = str(exc)
        if progress_callback is not None:
            progress_callback(completed, len(trait_names), trait)
        _check_cancelled(cancel_event)
    _check_cancelled(cancel_event)
    if correction_scope == "global" and analyses:
        frames = [analysis["results"] for analysis in analyses.values()]
        raw_p = np.concatenate([frame["p_value"].to_numpy(dtype=float) for frame in frames])
        adjusted = _adjust_bh(raw_p) if correction == "bh" else raw_p.copy()
        _check_cancelled(cancel_event)
        offset = 0
        for frame in frames:
            count = len(frame)
            frame["p_adjusted"] = adjusted[offset:offset + count]
            frame["significance"] = frame["p_adjusted"].map(_significance)
            offset += count
    _check_cancelled(cancel_event)
    return {"analyses": analyses, "failures": failures, "correction_scope": correction_scope}


def _bracket_levels(results, positions):
    """Place brackets sharing any x interval on different vertical levels."""
    pending = []
    for _, row in results.iterrows():
        if row["Group1"] not in positions or row["Group2"] not in positions:
            raise ValueError("Comparison results contain a group outside selected_groups.")
        left, right = sorted((positions[row["Group1"]], positions[row["Group2"]]))
        pending.append((left, right, row))
    pending.sort(key=lambda item: (item[1] - item[0], item[0], item[1]))
    occupied = []
    placed = []
    for left, right, row in pending:
        level = 0
        while level < len(occupied) and any(not (right < a or left > b) for a, b in occupied[level]):
            level += 1
        if level == len(occupied):
            occupied.append([])
        occupied[level].append((left, right))
        placed.append((left, right, level, row))
    return placed, len(occupied)


def _plot_colors(selected, group_colors):
    default = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
    if group_colors is None:
        return [default[i % len(default)] for i in range(len(selected))]
    if isinstance(group_colors, str):
        try:
            palette = colormaps[group_colors]
        except KeyError as exc:
            raise ValueError(f"Unknown Matplotlib color palette: {group_colors!r}.") from exc
        return list(palette(np.linspace(0, 1, len(selected))))
    if not isinstance(group_colors, Mapping):
        raise ValueError("group_colors must be a group-to-color mapping or Matplotlib palette name.")
    mapping = {str(group): color for group, color in group_colors.items()}
    colors = [mapping.get(group, default[i % len(default)]) for i, group in enumerate(selected)]
    if any(not is_color_like(color) for color in colors):
        raise ValueError("group_colors contains an invalid Matplotlib color.")
    return colors


def _annotation_results(results, selected, annotation_filter, annotation_pairs):
    if annotation_filter == "none":
        return results.iloc[:0].copy()
    if annotation_filter == "significant":
        p_values = pd.to_numeric(results["p_adjusted"], errors="coerce")
        return results.loc[np.isfinite(p_values) & p_values.lt(0.05)
                           & results["status"].ne("undefined")].copy()
    if annotation_filter == "custom":
        pairs = _validate_pairs(annotation_pairs, selected, name="annotation_pairs")
        wanted = {frozenset(pair) for pair in pairs}
        actual = [frozenset((str(row["Group1"]), str(row["Group2"])))
                  for _, row in results.iterrows()]
        if not wanted.issubset(actual):
            raise ValueError("Every annotation pair must have an existing comparison result.")
        return results.loc[[pair in wanted for pair in actual]].copy()
    return results.copy()


def plot_comparison(data, selected_groups, results, *, trait_label="Phenotype", correction="bh",
                    group_labels=None, group_colors=None, group_order=None, y_label=None,
                    title=None, font_size=11, point_size=25, show_points=True, show_means=True,
                    annotation_style="both", annotation_filter="all", annotation_pairs=None):
    """Create hollow boxes and configurable points, means and P brackets.

    ``group_labels`` maps original group IDs to display names. ``group_colors``
    is a group-to-color mapping (missing entries use default colors), or a
    Matplotlib palette name such as 'tab10'/'Set2'. Colors affect observations.
    ``group_order`` must be a permutation of all selected IDs, so arranging a
    plot never silently removes tested groups. Annotation style is 'stars',
    'p', or 'both'; filter is 'all', 'significant', 'custom', or 'none'. Custom
    annotation pairs only filter displayed brackets, never the results table.

    Stars and numeric P values use ``p_adjusted``; with no correction that is
    the raw P value. All observations determine axis limits, including points
    hidden by ``show_points=False``. The Figure supports Qt and headless export.
    """
    if correction not in {"bh", "none"}:
        raise ValueError("correction must be 'bh' or 'none'.")
    if annotation_style not in {"stars", "p", "both"}:
        raise ValueError("annotation_style must be 'stars', 'p', or 'both'.")
    if annotation_filter not in {"all", "significant", "custom", "none"}:
        raise ValueError("annotation_filter must be 'all', 'significant', 'custom', or 'none'.")
    try:
        font_size, point_size = float(font_size), float(point_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("font_size and point_size must be numeric.") from exc
    if not np.isfinite(font_size) or font_size <= 0:
        raise ValueError("font_size must be positive and finite.")
    if not np.isfinite(point_size) or point_size < 0:
        raise ValueError("point_size must be nonnegative and finite.")
    _require_columns(results, ["Group1", "Group2", "p_adjusted", "status"], "Comparison results")
    selected, values = _selected_values(data, selected_groups)
    if group_order is not None:
        if isinstance(group_order, (str, bytes)):
            raise ValueError("group_order must be a permutation of selected groups.")
        ordered = [str(group) for group in group_order]
        if len(ordered) != len(selected) or set(ordered) != set(selected):
            raise ValueError("group_order must contain every selected group exactly once.")
        selected = ordered
    if group_labels is not None and not isinstance(group_labels, Mapping):
        raise ValueError("group_labels must be a group-to-display-name mapping.")
    display_labels = {str(group): str(label) for group, label in (group_labels or {}).items()}
    colors = _plot_colors(selected, group_colors)
    # Validate all tested groups even when annotation filtering hides a row.
    positions = {group: i + 1 for i, group in enumerate(selected)}
    if not (results["Group1"].isin(positions).all() and results["Group2"].isin(positions).all()):
        raise ValueError("Comparison results contain a group outside selected_groups.")
    annotations = _annotation_results(results, selected, annotation_filter, annotation_pairs)
    brackets, level_count = _bracket_levels(annotations, positions)
    font_scale = font_size / 11
    width = max(7.0, 1.65 * len(selected) + 2.0) * max(1, font_scale)
    level_inches = 0.40 if annotation_style == "both" else 0.28
    height = (5.6 + level_inches * level_count) * max(1, font_scale)
    figure = Figure(figsize=(width, height), facecolor="white")
    axis = figure.subplots()
    arrays = [values[group] for group in selected]
    all_values = np.concatenate(arrays)
    # Rescale only near floating point limits so axis padding remains finite.
    max_abs = float(np.max(np.abs(all_values)))
    exponent = int(np.floor(np.log10(max_abs))) if max_abs > 0 else 0
    exponent = max(-300, min(300, exponent))
    scale = 10.0 ** exponent if abs(exponent) > 150 else 1.0
    arrays = [array / scale for array in arrays]
    axis.boxplot(arrays, positions=list(positions.values()), widths=0.52,
                 patch_artist=True, showfliers=not show_points,
                 boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.3},
                 whiskerprops={"color": "black", "linewidth": 1.1},
                 capprops={"color": "black", "linewidth": 1.1},
                 medianprops={"color": "black", "linewidth": 1.4})
    random = np.random.default_rng(20261023)
    for i, array in enumerate(arrays):
        if show_points:
            jitter = random.uniform(-0.17, 0.17, len(array))
            axis.scatter(i + 1 + jitter, array, s=point_size, color=colors[i],
                         alpha=0.72, edgecolors="none", zorder=3)
        if show_means:
            axis.scatter([i + 1], [float(np.mean(array))], marker="D", s=46,
                         facecolor="white", edgecolor="black", linewidth=1.2, zorder=4)
    plotted = np.concatenate(arrays)
    lower, upper = float(plotted.min()), float(plotted.max())
    span = upper - lower
    if span == 0:
        span = max(abs(lower) * 0.2, 1.0)
    base, step, cap = upper + span * 0.14, span * 0.18, span * 0.026
    p_name = "P_adj" if correction == "bh" else "P"
    annotation_font = max(7, font_size - 2)
    for left, right, level, row in brackets:
        y = base + level * step
        axis.plot([left, left, right, right], [y, y + cap, y + cap, y],
                  color="black", linewidth=1, clip_on=False)
        p_value = float(row["p_adjusted"])
        defined = row["status"] != "undefined" and np.isfinite(p_value)
        if defined:
            numeric = f"{p_value:.2e}" if 0 < p_value < 0.001 else f"{p_value:.3g}"
            p_text, stars = f"{p_name} = {numeric}", _significance(p_value)
        else:
            p_text, stars = f"{p_name} = n/a", "n/a"
        if annotation_style == "stars":
            annotation = stars
        elif annotation_style == "p" or not defined:
            annotation = p_text
        else:
            annotation = f"{p_text}\n{stars}"
        axis.text((left + right) / 2, y + cap + span * 0.012, annotation,
                  ha="center", va="bottom", fontsize=annotation_font, linespacing=1.05)
    y_top = (base + max(level_count - 1, 0) * step + span * 0.18
             if level_count else upper + span * 0.15)
    axis.set_ylim(lower - span * 0.10, y_top)
    axis.set_xlim(0.45, len(selected) + 0.55)
    labels = [textwrap.fill(display_labels.get(group, group), width=18)
              + f"\n(n={len(values[group])})" for group in selected]
    axis.set_xticks(list(positions.values()), labels=labels, fontsize=max(7, font_size - 1))
    axis.set_xlabel("Haplotype", labelpad=10, fontsize=font_size)
    actual_y_label = str(trait_label) if y_label is None else str(y_label)
    if scale != 1:
        actual_y_label += f" (×10^{exponent})"
    axis.set_ylabel(actual_y_label, labelpad=10, fontsize=font_size)
    axis.set_title(f"{trait_label} among haplotype groups" if title is None else str(title),
                   pad=16, fontsize=font_size + 1)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", direction="out")
    axis.tick_params(axis="y", labelsize=max(7, font_size - 1))
    axis.grid(axis="y", alpha=0.15, zorder=0)
    correction_note = "BH-adjusted P" if correction == "bh" else "raw P (no correction)"
    if correction == "bh" and "correction_scope" in results and results["correction_scope"].eq("global").all():
        correction_note += " across all successful traits"
    mean_note = "; diamonds: group means" if show_means else ""
    figure.text(0.5, 0.025, f"Two-sided Welch t-test; {correction_note}{mean_note}.\n"
                "* P < 0.05; ** P < 0.01; *** P < 0.001; **** P < 0.0001; ns: P ≥ 0.05; n/a: undefined.",
                ha="center", va="bottom", fontsize=annotation_font)
    max_lines = max(label.count("\n") + 1 for label in labels)
    bottom = min(0.44, max(0.25, (1.1 + 0.22 * max_lines) * max(1, font_scale) / height))
    figure.subplots_adjust(left=0.13, right=0.98, bottom=bottom, top=0.91)
    # Use an installed CJK font without changing global Matplotlib settings.
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("Arial Unicode MS", "PingFang SC", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
        if family in installed:
            for text in figure.findobj(Text):
                text.set_fontfamily([family, "DejaVu Sans"])
            break
    return figure
