"""Cancellable individual genotype plots, rendered without Qt widgets."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from genotype_plot import draw_genotype_heatmap


def _check(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Genotype plot export cancelled.")


def export_genotype_plots(parent_directory, data, groups, genotype_result, selected_labels, *,
                          label_side="right", file_format="png", progress_callback=None, cancel_event=None):
    """Export one complete, original group per image into a new result folder.

    All rendering happens in a private staging directory. Cancellation or an
    error removes that directory; existing output folders are never reused.
    """
    parent = Path(parent_directory)
    if not parent.is_dir():
        raise NotADirectoryError(f"Output parent directory does not exist: {parent}")
    if file_format not in {"png", "pdf"}:
        raise ValueError("file_format must be 'png' or 'pdf'.")
    if label_side not in {"left", "right"}:
        raise ValueError("label_side must be 'left' or 'right'.")
    if isinstance(selected_labels, (str, bytes)):
        raise ValueError("Select a sequence of genotype labels.")
    selected = [str(label) for label in selected_labels]
    if not selected:
        raise ValueError("Select at least one genotype group to export.")
    if len(selected) != len(set(selected)):
        raise ValueError("Selected genotype labels must be unique.")
    if not {"Sample", "Genotype"}.issubset(genotype_result.columns):
        raise ValueError("genotype_result must contain Sample and Genotype columns.")
    if genotype_result.Sample.astype(str).duplicated().any():
        raise ValueError("Sample IDs must be unique.")
    labels = dict(zip(genotype_result.Sample.astype(str), genotype_result.Genotype.astype(str)))
    group_info = []
    for members in groups:
        if not len(members):
            raise ValueError("Genotype groups must not be empty.")
        sample = str(data.samples[members[0]])
        if sample not in labels:
            raise ValueError(f"No genotype label for sample: {sample}")
        group_info.append((labels[sample], len(members)))
    available = {label for label, _ in group_info}
    unknown = set(selected) - available
    if unknown:
        raise ValueError("Unknown genotype label(s): " + ", ".join(sorted(unknown)))
    chosen = [(label, count) for label, count in group_info if label in set(selected)]
    if len(chosen) != len(selected):
        raise ValueError("Each genotype label must identify exactly one group.")
    _check(cancel_event)
    temporary = Path(tempfile.mkdtemp(prefix=".FragGenotyper_haplotype_plots-", dir=parent))
    try:
        outputs = []
        for index, (label, count) in enumerate(chosen, start=1):
            _check(cancel_event)
            if progress_callback:
                progress_callback(int((index - 1) * 95 / len(chosen)), f"Exporting {label} ({index}/{len(chosen)})")
            safe_name = re.sub(r"[^\w.-]+", "_", label).strip("._")[:80] or "group"
            filename = f"{index:03d}_{safe_name}.{file_format}"
            figure = Figure(figsize=(8, 4), constrained_layout=True)
            FigureCanvasAgg(figure)
            try:
                draw_genotype_heatmap(figure, data, groups, genotype_result,
                                      selected_labels=[label], label_side=label_side)
                figure.savefig(temporary / filename, dpi=300, bbox_inches="tight", format=file_format)
            finally:
                figure.clear()
            outputs.append({"genotype": label, "samples": count, "file": filename})
        _check(cancel_event)
        manifest = {"schema_version": 1, "source": (getattr(data, "report", None) or {}).get("source"),
                    "selected_labels": [label for label, _ in chosen], "label_side": label_side,
                    "file_format": file_format, "png_dpi": 300 if file_format == "png" else None,
                    "plots": outputs}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        _check(cancel_event)
        number = 1
        while True:
            destination = parent / ("FragGenotyper_haplotype_plots" + (f"_{number}" if number > 1 else ""))
            if destination.exists():
                number += 1
                continue
            try:
                temporary.rename(destination)
                break
            except FileExistsError:
                number += 1
        if progress_callback:
            progress_callback(100, f"Exported {len(outputs)} genotype plots")
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
