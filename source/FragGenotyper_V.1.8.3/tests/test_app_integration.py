"""Full connected workflows through actual Qt workers and file exports."""
import gzip
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox
from app_window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait(panel, app, timeout=20):
    deadline = time.monotonic() + timeout
    while panel.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert panel.worker is None


def write_vcf(path):
    path.write_text("##fileformat=VCFv4.2\n"
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t001\t002\t003\t004\n"
                    "1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/0\t1/1\t1/1\n"
                    "1\t20\t.\tG\tT\t.\tPASS\t.\tGT\t0/0\t0/0\t1/1\t1/1\n")


def test_genotyping_to_phenotypes_and_filtering_exports(app, monkeypatch, tmp_path):
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(str(args)))
    source = tmp_path / "small.vcf"
    write_vcf(source)
    window = MainWindow()
    try:
        gen = window.genotype_panel
        gen.set_source(source)
        gen.run_analysis()
        wait(gen, app)
        assert not errors
        assert gen.genotype_result.Genotype.nunique() == 2
        pheno = window.phenotype_panel
        assert pheno.genotype_df.Sample.tolist() == ["001", "002", "003", "004"]
        pheno.phenotype_df = pd.DataFrame({"Sample": ["004", "003", "002", "001"], "Phenotype": [7, 6, 2, 1]})
        pheno.populate_columns()
        pheno.run_analysis()
        wait(pheno, app)
        assert pheno.analysis["analyses"]["Phenotype"]["data"].shape[0] == 4
        assert pheno.canvas is not None
        gen.export_groups(str(tmp_path / "groups.csv"))
        gen.export_matrix(str(tmp_path / "matrix.csv"))
        gen.export_image(str(tmp_path / "heatmap.png"))
        gen.export_qc(str(tmp_path / "qc.json"))
        gen.export_tree(str(tmp_path / "groups.tre"))
        wait(gen, app)
        assert pd.read_csv(tmp_path / "groups.csv").shape == (4, 2)
        assert (tmp_path / "heatmap.png").read_bytes().startswith(b"\x89PNG")
        assert json.loads((tmp_path / "qc.json").read_text())["input_qc"]["source"]
        assert (tmp_path / "groups.tre").read_text().strip().endswith(";")
        gen.search_k()
        wait(gen, app)
        assert gen.k_result["recommended_k"] == 2
        assert gen.k_result["table"].K.tolist() == [1, 2]
        filter_panel = window.filter_panel
        filter_panel.set_source(source)
        output = tmp_path / "filtered.vcf.gz"
        filter_panel.output_edit.setText(str(output))
        filter_panel.maf_spin.setValue(0)
        filter_panel.run_filter()
        wait(filter_panel, app)
        assert not errors
        assert output.read_bytes()[:2] == b"\x1f\x8b"
        with gzip.open(output, "rt") as handle:
            assert len([line for line in handle if not line.startswith("#")]) == 2
        window.language_combo.setCurrentIndex(1)
        assert window.phenotype_panel.language == "en"
        gen.set_source(tmp_path / "another.vcf")
        assert gen.data is None and gen.genotype_result is None
        assert not gen.export_group_button.isEnabled()
    finally:
        for panel in (window.genotype_panel, window.phenotype_panel, window.filter_panel):
            if panel.worker is not None:
                panel.begin_shutdown()
                wait(panel, app)
        window.close()
        window.deleteLater()
        app.processEvents()


def test_closing_main_window_requests_cancellation_and_waits_for_worker(app):
    window = MainWindow()
    window.show()
    def computation(progress, cancel_event):
        cancel_event.wait(5)
        if cancel_event.is_set():
            raise InterruptedError()
        return None
    window.genotype_panel._submit(computation, lambda result: None)
    window.close()
    wait(window.genotype_panel, app)
    deadline = time.monotonic() + 2
    while window.isVisible() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert not window.isVisible()
    window.deleteLater()
    app.processEvents()


def test_result_delivery_errors_are_reported_without_crashing_qt(app, monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(str(args)))
    window = MainWindow()
    def failed_delivery(result):
        raise OSError("Output folder is not writable")
    window.genotype_panel._submit(lambda progress, cancel_event: None, failed_delivery)
    wait(window.genotype_panel, app)
    assert any("not writable" in message for message in messages)
    window.close()
    window.deleteLater()
    app.processEvents()


def test_export_button_reports_io_error_without_crashing_qt(app, monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(str(args)))
    window = MainWindow()
    def failed_export():
        raise OSError("Disk full")
    window.genotype_panel._export_clicked(failed_export)
    assert any("Disk full" in message for message in messages)
    window.close()
    window.deleteLater()
    app.processEvents()
