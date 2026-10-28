"""Regression checks for real GT parsing, missingness and sample clustering."""
import gzip
import importlib.util
from pathlib import Path
import sys
import threading

import numpy as np
import pytest
from sklearn.cluster import KMeans


SPEC = importlib.util.spec_from_file_location("genotype_core_v180", Path(__file__).resolve().parents[1] / "genotype_core.py")
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def data(matrix, samples=None):
    matrix = np.asarray(matrix, dtype=float)
    return core.VariantData(samples or [f"s{i}" for i in range(len(matrix))],
                            [("1", i + 1) for i in range(matrix.shape[1])], matrix, {})


def vcf_text(samples, records):
    return "##fileformat=VCFv4.2\n" + "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples) + "\n" + "\n".join(records) + "\n"


def test_gt_is_read_from_format_and_unsupported_calls_are_nan(tmp_path):
    path = tmp_path / "calls.vcf"
    names = ["001", "002", "003", "partial", "haploid", "triploid", "allele2", "missing"]
    path.write_text(vcf_text(names, [
        "1\t10\t.\tA\tC\t.\tPASS\t.\tDP:GT\t4:0/0\t9:0|1\t2:1/1\t1:0/.\t1:1\t1:0/1/1\t1:0/2\t7:.|.",
        "1\t20\t.\tA\tC,G\t.\tPASS\t.\tGT\t" + "\t".join(["0/1"] * 8),
        "1\t30\t.\tA\tG\t.\tPASS\t.\tDP\t" + "\t".join(["8"] * 8),
        "1\t40\t.\tA\tAT\t.\tPASS\t.\tGT\t" + "\t".join(["0/1"] * 8),
    ]))
    result = core.read_variants(path)
    assert result.samples[:3] == ["001", "002", "003"]
    assert result.matrix.shape == (8, 4)
    np.testing.assert_equal(result.matrix[:3, 0], [0, 1, 2])
    assert np.isnan(result.matrix[3:, 0]).all()
    assert np.isnan(result.matrix[:, 1:]).all()
    assert result.report["partial_missing_calls"] == 1
    assert result.report["non_diploid_calls"] == 2
    assert result.report["missing_gt_fields"] == 1
    assert result.report["unsupported_variants"] == 2


def test_gz_vcf_and_project_hapmap_agree(tmp_path):
    compressed = tmp_path / "input.vcf.gz"
    with gzip.open(compressed, "wt") as handle:
        handle.write(vcf_text(["001", "010"], ["A01\t12\t.\tC\tT\t.\tPASS\t.\tGT:DP\t1|0:8\t0/0:3"]))
    hapmap = tmp_path / "input.hmp"
    hapmap.write_text("CHROM\tPOS\tREF\tALT\t001\t010\nA01\t12\tC\tT\t1|0\t0/0\n")
    first, second = core.read_variants(compressed), core.read_variants(hapmap)
    assert first.samples == second.samples == ["001", "010"]
    assert first.positions == second.positions == [("A01", 12)]
    np.testing.assert_equal(first.matrix, second.matrix)


@pytest.mark.parametrize("mode", ["matrix", "onebyone"])
def test_shared_missingness_never_creates_similarity(mode):
    variants = data([[0, np.nan], [2, np.nan]])
    assert core.similarity_groups(variants, threshold=0.5, mode=mode) == [[0], [1]]
    # Even threshold zero cannot match samples with no jointly called SNP.
    variants = data([[0, np.nan], [np.nan, 0]])
    assert core.similarity_groups(variants, threshold=0, mode=mode) == [[0], [1]]


@pytest.mark.parametrize("mode", ["matrix", "onebyone"])
def test_minimum_shared_call_requirement(mode):
    variants = data([[0, np.nan], [0, 2]])
    assert core.similarity_groups(variants, mode=mode, min_common_snps=1) == [[0, 1]]
    assert core.similarity_groups(variants, mode=mode, min_common_snps=2) == [[0], [1]]


def test_two_greedy_strategies_retain_their_distinction():
    variants = data([[0, 0, 0], [0, 0, 1], [0, 1, 1]])
    assert core.similarity_groups(variants, threshold=0.66, mode="matrix") == [[0, 1], [2]]
    assert core.similarity_groups(variants, threshold=0.66, mode="onebyone") == [[0, 1, 2]]
    frame = core.groups_frame(variants, [[0, 1], [2]])
    assert frame.columns.tolist() == ["Sample", "Genotype"]
    assert frame.Genotype.tolist() == ["Genotype000", "Genotype000", "Genotype001"]


def test_kmeans_mean_imputation_and_all_missing_locus_removal():
    variants = data([[0, np.nan, np.nan, 2], [2, 2, np.nan, 2], [0, 0, np.nan, np.nan]])
    prepared = core.prepare_kmeans(variants)
    np.testing.assert_allclose(prepared.matrix, [[0, 1, 2], [2, 2, 2], [0, 0, 2]])
    assert prepared.kept_loci.tolist() == [0, 1, 3]
    assert prepared.imputed_calls == 2
    assert prepared.dropped_loci == 1


def test_search_k_and_final_fit_cluster_samples_not_snps():
    variants = data([[0, 0, 0, np.nan, 0], [0, 0, 0, 0, 0], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]])
    result = core.search_k(variants, max_k=20)
    prepared = core.prepare_kmeans(variants).matrix
    assert result["preprocessing"]["samples"] == 4
    assert result["preprocessing"]["used_loci"] == 5
    # The imputed first sample is a third distinct vector, despite its otherwise
    # matching the second sample. The cap must use the actual fitted vectors.
    assert result["table"].K.tolist() == [1, 2, 3]
    for _, row in result["table"].iterrows():
        oracle = KMeans(n_clusters=int(row.K), n_init=10, random_state=0).fit(prepared)
        assert row.SSE == pytest.approx(oracle.inertia_)
    assert result["recommended_k"] == 2
    assert core.kmeans_groups(variants, 2) == [[0, 1], [2, 3]]


@pytest.mark.parametrize("matrix,expected_candidates", [([[0, 1]], [1]), ([[0], [2]], [1, 2]), ([[1], [1], [1]], [1])])
def test_small_or_identical_samples_do_not_request_impossible_k(matrix, expected_candidates):
    result = core.search_k(data(matrix), max_k=20)
    assert result["table"].K.tolist() == expected_candidates
    assert result["recommended_k"] == 1
    assert result["table"]["silhouette"].isna().all()


@pytest.mark.parametrize("function", [lambda d: core.similarity_groups(d), lambda d: core.search_k(d), lambda d: core.kmeans_groups(d, 1)])
def test_all_missing_samples_are_rejected_with_their_names(function):
    with pytest.raises(ValueError, match="All-missing samples.*empty"):
        function(data([[np.nan, np.nan], [0, 1]], samples=["empty", "called"]))


def test_maf_counts_missing_samples_and_valid_diploid_alleles():
    statistics = core.biallelic_snp_stats(["0/0", "0|1", "1/1", "0/.", "1", "0/2"])
    assert statistics["maf"] == pytest.approx(0.5)
    assert statistics["missing_rate"] == pytest.approx(0.5)
    assert statistics["AC"] == 3 and statistics["AN"] == 6 and statistics["NS"] == 3


def test_filter_is_gzip_streaming_subset_aware_and_updates_info(tmp_path):
    source, output = tmp_path / "original.vcf", tmp_path / "subset.vcf.gz"
    source.write_text(vcf_text(["a", "b", "c"], [
        "1\t10\t.\tA\tC\t.\tPASS\tAC=3;AN=6;AF=0.5;NS=3;MLEAC=3\tDP:GT\t9:0/0\t8:0/1\t4:1/1",
        "1\t20\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0\t./.",
        "1\t30\t.\tA\tC,G\t.\tPASS\t.\tGT\t0/1\t0/0\t1/1",
        "2\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0\t1/1",
    ]))
    report = core.filter_vcf(source, output, chrom="1", start=10, end=30, samples=["a", "c"], min_maf=0.1, max_missing=0.1)
    assert output.read_bytes()[:2] == b"\x1f\x8b"
    with gzip.open(output, "rt") as handle:
        lines = handle.read().splitlines()
    header = next(line for line in lines if line.startswith("#CHROM"))
    retained = [line.split("\t") for line in lines if not line.startswith("#")]
    assert header.split("\t")[9:] == ["a", "c"]
    assert len(retained) == 1 and retained[0][1] == "10"
    assert retained[0][7] == "AC=2;AN=4;AF=0.5;NS=2"
    assert report["retained_variants"] == 1
    assert report["unsupported_variants"] == 1
    assert report["failed_missingness"] == 1
    assert report["outside_region"] == 1
    np.testing.assert_equal(core.read_variants(output).matrix[:, 0], [0, 2])


def test_cancellation_does_not_replace_existing_filter_output(tmp_path):
    source, output = tmp_path / "input.vcf", tmp_path / "result.vcf.gz"
    source.write_text(vcf_text(["a"], ["1\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0/1"]))
    output.write_bytes(b"previous result")
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(core.AnalysisCancelled):
        core.filter_vcf(source, output, cancel_event=cancelled)
    assert output.read_bytes() == b"previous result"
    assert not list(tmp_path.glob(".*.tmp-*"))
