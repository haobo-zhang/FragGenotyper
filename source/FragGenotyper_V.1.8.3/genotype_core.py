"""Headless, cancellable genotype analysis for diploid biallelic SNPs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import os
import re
import uuid

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class AnalysisCancelled(RuntimeError):
    pass


@dataclass
class VariantData:
    samples: list[str]
    positions: list[tuple[str, int]]
    matrix: np.ndarray
    report: dict


@dataclass
class KMeansInput:
    matrix: np.ndarray
    kept_loci: np.ndarray
    imputed_calls: int
    dropped_loci: int


def _check(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled("Analysis cancelled.")


def _progress(callback, value, message):
    if callback is not None:
        callback(int(value), message)


def _opener(path):
    return gzip.open if str(path).lower().endswith(".gz") else open


def _is_snp(ref, alt):
    return len(ref) == 1 and len(alt) == 1 and ref.upper() in "ACGT" and alt.upper() in "ACGT" and ref.upper() != alt.upper()


def _decode_gt(gt):
    """Return dosage and QC category; partial calls are entirely missing."""
    if gt is None or pd.isna(gt) or str(gt).strip() in {"", ".", "./.", ".|."}:
        return np.nan, "missing_calls"
    alleles = re.split(r"[/|]", str(gt).strip())
    if len(alleles) != 2:
        return np.nan, "non_diploid_calls"
    if "." in alleles:
        return np.nan, "partial_missing_calls"
    if any(allele not in {"0", "1"} for allele in alleles):
        return np.nan, "unsupported_calls"
    return float(int(alleles[0]) + int(alleles[1])), "called_calls"


def _report(path):
    return {"source": str(Path(path).resolve()), "input_variants": 0, "retained_loci": 0,
            "unsupported_variants": 0, "malformed_rows": 0, "missing_gt_fields": 0,
            "called_calls": 0, "missing_calls": 0, "partial_missing_calls": 0,
            "non_diploid_calls": 0, "unsupported_calls": 0,
            "interpretation": "Unphased diploid biallelic SNP dosage; unknown calls are NaN, never reference.",
            "missing_policy": "Partial missing, non-diploid, multi-allelic and non-SNP calls are missing."}


def _finish(samples, positions, rows, report):
    if not samples:
        raise ValueError("No sample columns were found.")
    if len(set(samples)) != len(samples) or any(not sample.strip() for sample in samples):
        raise ValueError("Sample names must be nonempty and unique.")
    if not rows:
        raise ValueError("No readable variant rows were found.")
    matrix = np.asarray(rows, dtype=float).T
    report.update(sample_count=len(samples), retained_loci=len(positions),
                  all_missing_loci=int(np.isnan(matrix).all(axis=0).sum()),
                  all_missing_samples=[sample for sample, missing in zip(samples, np.isnan(matrix).all(axis=1)) if missing],
                  missing_fraction=float(np.isnan(matrix).mean()))
    return VariantData(samples, positions, matrix, report)


def read_variants(path, progress_callback=None, cancel_event=None):
    """Stream VCF/VCF.GZ or read the project's CHROM/POS/REF/ALT + GT table.

    FORMAT determines the GT field position. Unsupported loci remain in the
    matrix as NaN, with explicit QC counts, so no allele is fabricated.
    """
    path = Path(path)
    report, positions, rows = _report(path), [], []
    if ".vcf" not in path.name.lower():
        table = pd.read_csv(path, sep=r"\s+", dtype=str, encoding="utf-8-sig")
        required = {"CHROM", "POS", "REF", "ALT"}
        if not required.issubset(table.columns):
            raise ValueError("HapMap input needs CHROM, POS, REF and ALT columns plus sample GT columns.")
        samples = [str(column) for column in table.columns if column not in required]
        for row_number, (_, row) in enumerate(table.iterrows()):
            _check(cancel_event)
            report["input_variants"] += 1
            try:
                position = int(row["POS"])
                if position < 1:
                    raise ValueError
            except (TypeError, ValueError):
                report["malformed_rows"] += 1
                continue
            supported = _is_snp(str(row["REF"]), str(row["ALT"]))
            if not supported:
                report["unsupported_variants"] += 1
            dosages = []
            for sample in samples:
                dosage, category = _decode_gt(row[sample]) if supported else (np.nan, "unsupported_calls")
                report[category] += 1
                dosages.append(dosage)
            positions.append((str(row["CHROM"]), position))
            rows.append(dosages)
            if row_number % 500 == 0:
                _progress(progress_callback, min(95, 1 + row_number * 94 // max(len(table), 1)), f"Read {row_number + 1} loci")
    else:
        samples = None
        with _opener(path)(path, "rt", encoding="utf-8-sig") as handle:
            for line in handle:
                _check(cancel_event)
                if line.startswith("##") or not line.strip():
                    continue
                if line.startswith("#CHROM"):
                    columns = line.rstrip("\r\n").split("\t")
                    samples = columns[9:]
                    continue
                if line.startswith("#"):
                    continue
                if samples is None:
                    raise ValueError("VCF #CHROM header was not found before data rows.")
                report["input_variants"] += 1
                fields = line.rstrip("\r\n").split("\t")
                try:
                    position = int(fields[1])
                    if len(fields) < 9 or position < 1:
                        raise ValueError
                except (IndexError, ValueError):
                    report["malformed_rows"] += 1
                    continue
                supported = _is_snp(fields[3], fields[4])
                if not supported:
                    report["unsupported_variants"] += 1
                formats = fields[8].split(":")
                gt_index = formats.index("GT") if "GT" in formats else None
                if gt_index is None:
                    report["missing_gt_fields"] += 1
                dosages = []
                for i, _ in enumerate(samples):
                    values = fields[i + 9].split(":") if i + 9 < len(fields) else []
                    gt = values[gt_index] if gt_index is not None and gt_index < len(values) else None
                    dosage, category = _decode_gt(gt) if supported else (np.nan, "unsupported_calls")
                    report[category] += 1
                    dosages.append(dosage)
                positions.append((fields[0], position))
                rows.append(dosages)
                if len(rows) % 500 == 0:
                    _progress(progress_callback, min(95, 1 + len(rows) // 500), f"Read {len(rows)} loci")
        if samples is None:
            raise ValueError("VCF #CHROM header was not found.")
    result = _finish(samples, positions, rows, report)
    _progress(progress_callback, 100, f"Read {len(samples)} samples × {len(positions)} loci")
    return result


def _validate_samples(data):
    matrix = np.asarray(data.matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(data.samples), len(data.positions)):
        raise ValueError("Variant matrix must have shape samples × loci.")
    if not len(data.samples) or not len(data.positions):
        raise ValueError("Analysis requires at least one sample and one locus.")
    if np.isinf(matrix).any():
        raise ValueError("Variant data must contain finite dosages or NaN, not infinity.")
    missing = np.isnan(matrix).all(axis=1)
    if missing.any():
        names = [sample for sample, absent in zip(data.samples, missing) if absent]
        raise ValueError("All-missing samples cannot be classified: " + ", ".join(names[:12]) +
                         (f" … ({len(names)} total)" if len(names) > 12 else ""))
    return matrix


def similarity_groups(data, threshold=0.9, mode="matrix", min_common_snps=1,
                      progress_callback=None, cancel_event=None):
    """Greedy seed grouping or sequential any-member grouping.

    Both methods retain input-order dependence. No comparison with fewer than
    min_common_snps shared calls can match, including zero shared calls.
    """
    matrix = _validate_samples(data)
    if not 0 <= threshold <= 1:
        raise ValueError("Similarity threshold must be between zero and one.")
    if not isinstance(min_common_snps, (int, np.integer)) or min_common_snps < 1:
        raise ValueError("min_common_snps must be a positive integer.")
    if mode not in {"matrix", "onebyone"}:
        raise ValueError("Similarity mode must be matrix or onebyone.")
    count = len(matrix)

    def matches(first, others):
        common = np.isfinite(first) & np.isfinite(others)
        denominators = common.sum(axis=1)
        numerators = ((first == others) & common).sum(axis=1)
        similarity = np.divide(numerators, denominators, out=np.zeros(len(others)), where=denominators > 0)
        return (denominators >= min_common_snps) & (similarity >= threshold)

    groups = []
    if mode == "matrix":
        assigned = np.zeros(count, dtype=bool)
        for i in range(count):
            _check(cancel_event)
            if not assigned[i]:
                remaining = np.flatnonzero(~assigned)
                members = remaining[matches(matrix[i], matrix[remaining])].tolist()
                if i not in members:
                    members.insert(0, i)
                groups.append(members)
                assigned[members] = True
            _progress(progress_callback, (i + 1) * 100 // count, f"Grouped {i + 1}/{count} samples")
    else:
        for i in range(count):
            _check(cancel_event)
            for group in groups:
                _check(cancel_event)
                if matches(matrix[i], matrix[group]).any():
                    group.append(i)
                    break
            else:
                groups.append([i])
            _progress(progress_callback, (i + 1) * 100 // count, f"Grouped {i + 1}/{count} samples")
    return groups


def groups_frame(data, groups):
    labels = {}
    width = max(3, len(str(max(0, len(groups) - 1))))
    for number, members in enumerate(groups):
        for index in members:
            if index in labels or not 0 <= index < len(data.samples):
                raise ValueError("Groups must contain each sample index exactly once.")
            labels[index] = f"Genotype{number:0{width}d}"
    if len(labels) != len(data.samples):
        raise ValueError("Some samples were not assigned to a group.")
    return pd.DataFrame({"Sample": data.samples, "Genotype": [labels[i] for i in range(len(data.samples))]})


def prepare_kmeans(data):
    matrix = _validate_samples(data)
    kept = np.flatnonzero(np.isfinite(matrix).any(axis=0))
    if not len(kept):
        raise ValueError("K-means requires at least one observed SNP.")
    work = matrix[:, kept].copy()
    missing = np.isnan(work)
    means = np.nanmean(work, axis=0)
    work[missing] = np.broadcast_to(means, work.shape)[missing]
    return KMeansInput(work, kept, int(missing.sum()), matrix.shape[1] - len(kept))


def search_k(data, max_k=20, progress_callback=None, cancel_event=None):
    """Evaluate the same samples × loci preprocessing used by final K-means."""
    prepared = prepare_kmeans(data)
    matrix = prepared.matrix
    if not isinstance(max_k, (int, np.integer)) or max_k < 1:
        raise ValueError("max_k must be a positive integer.")
    upper = min(max_k, len(matrix), len(np.unique(matrix, axis=0)))
    rows = []
    for k in range(1, upper + 1):
        _check(cancel_event)
        fit = KMeans(n_clusters=k, random_state=0, n_init=10).fit(matrix)
        actual = len(np.unique(fit.labels_))
        score = np.nan
        if 2 <= actual < len(matrix):
            score = float(silhouette_score(matrix, fit.labels_))
        rows.append({"K": k, "SSE": float(fit.inertia_), "silhouette": score, "clusters": actual})
        _progress(progress_callback, k * 100 // upper, f"Evaluated K={k}/{upper}")
    table = pd.DataFrame(rows)
    eligible = table[table["silhouette"].notna()]
    recommended = int(eligible.loc[eligible["silhouette"].idxmax(), "K"]) if len(eligible) else 1
    return {"table": table, "recommended_k": recommended,
            "preprocessing": {"samples": matrix.shape[0], "used_loci": matrix.shape[1],
                              "dropped_all_missing_loci": prepared.dropped_loci,
                              "mean_imputed_calls": prepared.imputed_calls},
            "recommendation_rule": "Maximum valid silhouette; ties prefer smaller K. Without a valid score, conservatively recommend K=1."}


def kmeans_groups(data, k, progress_callback=None, cancel_event=None):
    prepared = prepare_kmeans(data)
    matrix = prepared.matrix
    maximum = min(len(matrix), len(np.unique(matrix, axis=0)))
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= maximum:
        raise ValueError(f"K must be between 1 and {maximum} (sample/distinct-vector limit).")
    _check(cancel_event)
    _progress(progress_callback, 10, "Fitting K-means after per-SNP mean imputation")
    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(matrix)
    _check(cancel_event)
    # Stable group labels follow first occurrence in the original sample order.
    groups = [np.flatnonzero(labels == label).tolist() for label in dict.fromkeys(labels)]
    _progress(progress_callback, 100, f"Found {len(groups)} groups")
    return groups


def matrix_frame(data):
    columns = [f"{chrom}_{position}" for chrom, position in data.positions]
    frame = pd.DataFrame(data.matrix, index=data.samples, columns=columns)
    frame.index.name = "Sample"
    return frame


def newick_tree(data):
    """Average-linkage Euclidean similarity tree, using K-means imputation.

    This is a clustering dendrogram, not a fitted evolutionary phylogeny.
    """
    matrix = prepare_kmeans(data).matrix
    labels = ["'" + sample.replace("'", "''") + "'" for sample in data.samples]
    if len(matrix) == 1:
        return labels[0] + ";\n"
    root = to_tree(linkage(matrix, method="average", metric="euclidean"))
    def emit(node, parent_height):
        length = max(0.0, (parent_height - node.dist) / 2)
        body = labels[node.id] if node.is_leaf() else f"({emit(node.left, node.dist)},{emit(node.right, node.dist)})"
        return body + f":{length:.10g}"
    return f"({emit(root.left, root.dist)},{emit(root.right, root.dist)});\n"


def biallelic_snp_stats(genotypes):
    """Diploid sample missingness; ALT frequency is folded to minor frequency."""
    values = np.asarray([_decode_gt(gt)[0] for gt in genotypes], dtype=float)
    called = np.isfinite(values)
    if not len(values):
        raise ValueError("At least one selected sample is required.")
    allele_count = int(values[called].sum())
    allele_number = int(called.sum() * 2)
    af = allele_count / allele_number if allele_number else np.nan
    return {"maf": min(af, 1 - af) if allele_number else np.nan,
            "missing_rate": float((~called).mean()), "AC": allele_count,
            "AN": allele_number, "AF": af, "NS": int(called.sum())}


def filter_vcf(input_path, output_path, *, chrom=None, start=None, end=None,
               samples=None, min_maf=0.05, max_missing=0.1,
               progress_callback=None, cancel_event=None):
    """Stream biallelic SNP filtering; .gz output is real ordinary gzip.

    Region coordinates are one-based and inclusive. Statistics use only the
    selected samples. Existing AC/AN/AF/NS INFO values are recalculated; stale
    MLEAC/MLEAF values are removed. Output is atomic, and ordinary gzip is not
    BGZF/tabix compatible.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output VCF paths must differ.")
    if not 0 <= min_maf <= 0.5 or not 0 <= max_missing <= 1:
        raise ValueError("MAF must be 0–0.5 and missingness must be 0–1.")
    if start is not None and start < 1 or end is not None and end < 1:
        raise ValueError("Region positions must be positive.")
    if start is not None and end is not None and start > end:
        raise ValueError("Region start must not exceed end.")
    requested = None if samples is None else [str(sample).strip() for sample in samples if str(sample).strip()]
    if requested is not None and (not requested or len(set(requested)) != len(requested)):
        raise ValueError("Select at least one sample, with no duplicate sample IDs.")
    report = {"source": str(input_path.resolve()), "output": str(output_path.resolve()),
              "input_variants": 0, "retained_variants": 0, "outside_region": 0,
              "unsupported_variants": 0, "failed_maf": 0, "failed_missingness": 0,
              "malformed_rows": 0, "compression": "ordinary gzip; not BGZF/tabix" if str(output_path).endswith(".gz") else "plain VCF",
              "min_maf": min_maf, "max_missing_rate": max_missing,
              "coordinate_convention": "one-based inclusive"}
    temporary = output_path.with_name("." + output_path.name + ".tmp-" + uuid.uuid4().hex)
    writer = gzip.open if str(output_path).lower().endswith(".gz") else open
    indices, header_seen = None, False
    try:
        with _opener(input_path)(input_path, "rt", encoding="utf-8-sig") as source, writer(temporary, "wt", encoding="utf-8") as target:
            for line in source:
                _check(cancel_event)
                if line.startswith("##"):
                    target.write(line)
                    continue
                if line.startswith("#CHROM"):
                    fields = line.rstrip("\r\n").split("\t")
                    names = fields[9:]
                    if len(names) != len(set(names)):
                        raise ValueError("VCF sample IDs must be unique.")
                    if requested is not None and set(requested) - set(names):
                        raise ValueError("Unknown selected samples: " + ", ".join(sorted(set(requested) - set(names))))
                    selected = names if requested is None else [name for name in names if name in set(requested)]
                    if not selected:
                        raise ValueError("No sample columns were selected.")
                    indices = [9 + names.index(name) for name in selected]
                    report["selected_samples"] = selected
                    target.write("\t".join(fields[:9] + selected) + "\n")
                    header_seen = True
                    continue
                if line.startswith("#") or not line.strip():
                    continue
                if not header_seen:
                    raise ValueError("VCF #CHROM header was not found before data rows.")
                report["input_variants"] += 1
                fields = line.rstrip("\r\n").split("\t")
                try:
                    position = int(fields[1])
                    if len(fields) < 9 or position < 1:
                        raise ValueError
                except (IndexError, ValueError):
                    report["malformed_rows"] += 1
                    continue
                if (chrom is not None and fields[0] != str(chrom)) or (start is not None and position < start) or (end is not None and position > end):
                    report["outside_region"] += 1
                    continue
                if not _is_snp(fields[3], fields[4]):
                    report["unsupported_variants"] += 1
                    continue
                selected_values = [fields[index] if index < len(fields) else "./." for index in indices]
                formats = fields[8].split(":")
                gt_index = formats.index("GT") if "GT" in formats else None
                gts = []
                for value in selected_values:
                    parts = value.split(":")
                    gts.append(parts[gt_index] if gt_index is not None and gt_index < len(parts) else None)
                statistics = biallelic_snp_stats(gts)
                if statistics["missing_rate"] > max_missing:
                    report["failed_missingness"] += 1
                    continue
                if not np.isfinite(statistics["maf"]) or statistics["maf"] < min_maf:
                    report["failed_maf"] += 1
                    continue
                info = []
                for item in fields[7].split(";"):
                    key = item.split("=", 1)[0]
                    if key in {"MLEAC", "MLEAF"}:
                        continue
                    if key in {"AC", "AN", "AF", "NS"}:
                        info.append(f"{key}={statistics[key]:.10g}")
                    elif item and item != ".":
                        info.append(item)
                fields[7] = ";".join(info) or "."
                target.write("\t".join(fields[:9] + selected_values) + "\n")
                report["retained_variants"] += 1
                if report["input_variants"] % 500 == 0:
                    _progress(progress_callback, min(95, 1 + report["input_variants"] // 500), f"Filtered {report['input_variants']} variants")
            if not header_seen:
                raise ValueError("VCF #CHROM header was not found.")
        _check(cancel_event)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    _progress(progress_callback, 100, f"Retained {report['retained_variants']} variants")
    return report
