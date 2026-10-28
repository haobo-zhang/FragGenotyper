"""Figure-controlled genotype heatmaps, without Qt or pyplot dependencies."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def draw_genotype_heatmap(figure, data, groups, genotype_result,
                          selected_labels=None, label_side="right"):
    """Draw selected original groups and return their heatmap axis.

    Selection changes only which complete groups are displayed. Samples retain
    their original group/member order, dosage values and group labels. An empty
    selection clears the figure and returns None; it never falls back to all
    groups. The caller owns figure sizing, canvas updates and file export.
    """
    if label_side not in {"left", "right"}:
        raise ValueError("label_side must be 'left' or 'right'.")
    matrix = np.asarray(data.matrix)
    if matrix.ndim != 2 or matrix.shape != (len(data.samples), len(data.positions)):
        raise ValueError("The genotype matrix must have shape samples × loci.")
    if not {"Sample", "Genotype"}.issubset(genotype_result.columns):
        raise ValueError("genotype_result must contain Sample and Genotype columns.")
    if genotype_result[["Sample", "Genotype"]].isna().any().any():
        raise ValueError("Sample and genotype labels must not be missing.")
    result_samples = genotype_result["Sample"].astype(str)
    if result_samples.duplicated().any():
        raise ValueError("genotype_result must contain unique Sample IDs.")
    sample_labels = dict(zip(result_samples, genotype_result["Genotype"].astype(str)))
    labelled_groups, seen_indices = [], set()
    for members in groups:
        if not len(members):
            raise ValueError("Genotype groups must not contain empty groups.")
        indices = list(members)
        for index in indices:
            if not isinstance(index, (int, np.integer)) or not 0 <= index < len(data.samples):
                raise ValueError("A group contains an invalid sample index.")
            if index in seen_indices:
                raise ValueError("A sample index occurs in more than one group.")
            seen_indices.add(index)
        names = [str(data.samples[index]) for index in indices]
        missing = [name for name in names if name not in sample_labels]
        if missing:
            raise ValueError("No genotype label was found for sample(s): " + ", ".join(missing[:5]))
        labels = {sample_labels[name] for name in names}
        if len(labels) != 1:
            raise ValueError("Members of one group must share the same original genotype label.")
        labelled_groups.append((sample_labels[names[0]], indices))
    available = {label for label, _ in labelled_groups}
    if selected_labels is None:
        selected = available
    else:
        if isinstance(selected_labels, (str, bytes)):
            raise ValueError("selected_labels must be a sequence of genotype labels.")
        selected = {str(label) for label in selected_labels}
        unknown = selected - available
        if unknown:
            raise ValueError("Unknown genotype label(s): " + ", ".join(sorted(unknown)))
    displayed = [(label, indices) for label, indices in labelled_groups if label in selected]
    if displayed and not len(data.positions):
        raise ValueError("At least one locus is required to draw a genotype heatmap.")

    figure.clear()
    if not displayed:
        message = ("Select at least one genotype group to display."
                   if labelled_groups else "No genotype groups are available to display.")
        figure.text(0.5, 0.5, message, ha="center", va="center", color="#6B7280", fontsize=11)
        return None

    axis = figure.subplots()
    order = [index for _, indices in displayed for index in indices]
    cmap = ListedColormap(["#e7ddb8", "#b186a3", "#684e94"])
    cmap.set_bad("#D9D9D9")
    image = axis.imshow(np.ma.masked_invalid(matrix[order]), aspect="auto", interpolation="nearest",
                        cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
    midpoints, labels, position = [], [], 0
    for label, indices in displayed:
        midpoints.append(position + (len(indices) - 1) / 2)
        labels.append(label)
        position += len(indices)
        if position < len(order):
            axis.axhline(position - 0.5, color="white", linewidth=1.5)
    axis.set_yticks(midpoints, labels=labels, fontsize=8)
    if label_side == "right":
        axis.yaxis.tick_right()
    else:
        axis.yaxis.tick_left()
    axis.tick_params(axis="y", which="both", labelleft=label_side == "left",
                     labelright=label_side == "right", pad=4)
    step = max(1, int(np.ceil(len(data.positions) / 8)))
    ticks = list(range(0, len(data.positions), step))
    axis.set_xticks(ticks, labels=[f"{data.positions[index][0]}:{data.positions[index][1]}" for index in ticks],
                      rotation=30, ha="right", fontsize=8)
    axis.set_xlabel("SNP position")
    axis.set_title("Unphased genotype patterns · missing = gray")
    axis.grid(False)
    axis.minorticks_off()
    axis.tick_params(axis="both", which="both", length=0, top=False, bottom=False, left=False, right=False)
    axis.set_frame_on(False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    # Put the colorbar opposite the group names so those names stay directly
    # beside the heatmap, rather than beyond the legend or separated by it.
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2], label="ALT dosage (0 / 1 / 2)",
                              location="left" if label_side == "right" else "right", pad=0.035)
    colorbar.outline.set_visible(False)
    colorbar.dividers.set_visible(False)
    colorbar.ax.set_frame_on(False)
    colorbar.ax.minorticks_off()
    colorbar.ax.tick_params(axis="both", which="both", length=0)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)
    return axis
