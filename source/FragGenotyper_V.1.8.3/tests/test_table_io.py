import threading
import pandas as pd
import pytest
from table_io import read_table


@pytest.mark.parametrize("delimiter", [",", "\t", "   "])
def test_headerless_family_sample_phenotype_retains_first_sample(tmp_path, delimiter):
    path = tmp_path / "phenotype.txt"
    path.write_text(delimiter.join(["MAGIC001", "MAGIC001", "29.30372272"]) + "\n" +
                    delimiter.join(["MAGIC002", "MAGIC002", "28.5"]) + "\n")
    table = read_table(path)
    assert table.columns.tolist() == ["FID", "Sample", "Phenotype"]
    assert table.Sample.tolist() == ["MAGIC001", "MAGIC002"]
    assert len(table) == 2
    assert table.attrs["table_import"]["header"] == "none"


def test_bom_csv_and_leading_zero_ids_are_preserved(tmp_path):
    path = tmp_path / "phenotype.csv"
    path.write_text("\ufeffSample,FibLen\n001,29.3\n002,28.5\n")
    table = read_table(path)
    assert table.Sample.tolist() == ["001", "002"]
    assert table.columns.tolist() == ["Sample", "FibLen"]
    assert table.attrs["table_import"]["engine"] == "c"


def test_quoted_csv_headers_and_values_are_not_split(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text('Sample,"Length, mm"\n"sample,001",30.2\n"sample,002",29.8\n')
    table = read_table(path)
    assert table.columns.tolist() == ["Sample", "Length, mm"]
    assert table.Sample.tolist() == ["sample,001", "sample,002"]


def test_duplicate_headers_are_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("Sample, Sample\n001,002\n")
    with pytest.raises(ValueError, match="unique"):
        read_table(path)


def test_progress_and_cancel_between_chunks(tmp_path):
    path = tmp_path / "large.csv"
    pd.DataFrame({"Sample": [f"ID{i:06d}" for i in range(6000)], "Phenotype": range(6000)}).to_csv(path, index=False)
    cancel = threading.Event()
    progress = []
    def callback(value, message):
        progress.append((value, message))
        if "Read" in message:
            cancel.set()
    with pytest.raises(InterruptedError):
        read_table(path, callback, cancel)
    assert len(progress) >= 2


def test_explicit_header_override_and_minus_nine_are_preserved(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("A -9\nB 3\n")
    table = read_table(path, header="none")
    assert table.Phenotype.tolist() == ["-9", "3"]
    assert table.Sample.tolist() == ["A", "B"]
