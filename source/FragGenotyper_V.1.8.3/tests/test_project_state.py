"""Portable project files use plain JSON and preserve the caller's state."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "project_state.py"
SPEC = importlib.util.spec_from_file_location("project_state_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
project_state = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(project_state)


def valid_document(**changes):
    document = {
        "schema_version": 1,
        "application": "FragGenotyper",
        "inputs": {},
    }
    document.update(changes)
    return document


def test_save_uses_relative_paths_and_preserves_nested_caller_state(tmp_path):
    source_dir = tmp_path / "输入数据"
    source_dir.mkdir()
    genotype = source_dir / "基因型.csv"
    phenotype = source_dir / "表型.csv"
    genotype.write_text("Sample,Genotype\n001,Hap1\n", encoding="utf-8")
    phenotype.write_text("Sample,Phenotype\n001,3.5\n", encoding="utf-8")
    project = tmp_path / "experiment.fgproject.json"
    state = {
        "schema_version": 900,
        "application": "other application",
        "inputs": {"genotype_file": str(genotype), "phenotype_file": str(phenotype)},
        "analysis": {"groups": ["001", "Hap2"], "reference": "001", "correction": "bh"},
        "ui": {"language": "zh", "trait": "纤维长度"},
    }
    original = copy.deepcopy(state)

    project_state.save_project(project, state)

    assert state == original
    saved = json.loads(project.read_text(encoding="utf-8"))
    assert saved["schema_version"] == project_state.SCHEMA_VERSION == 1
    assert saved["application"] == "FragGenotyper"
    for key, source in (("genotype_file", genotype), ("phenotype_file", phenotype)):
        stored_path = Path(saved["inputs"][key])
        assert not stored_path.is_absolute()
        assert (project.parent / stored_path).resolve() == source.resolve()
    assert saved["analysis"] == original["analysis"]
    assert "纤维长度" in project.read_text(encoding="utf-8")
    loaded = project_state.load_project(project)
    assert loaded["inputs"] == original["inputs"]
    assert loaded["analysis"] == original["analysis"]


def test_project_and_relative_inputs_can_move_together(tmp_path):
    original_dir = tmp_path / "original"
    (original_dir / "tables").mkdir(parents=True)
    (original_dir / "sessions").mkdir()
    genotype = original_dir / "tables" / "genotypes.csv"
    genotype.write_text("Sample,Genotype\n001,A\n", encoding="utf-8")
    project = original_dir / "sessions" / "trial.json"
    project_state.save_project(project, {"inputs": {"genotype_file": str(genotype)}})
    moved_dir = tmp_path / "moved"
    shutil.copytree(original_dir, moved_dir)

    loaded = project_state.load_project(moved_dir / "sessions" / "trial.json")

    assert loaded["inputs"]["genotype_file"] == str((moved_dir / "tables" / "genotypes.csv").resolve())
    assert Path(loaded["inputs"]["genotype_file"]).is_file()


def test_load_resolves_missing_files_without_trying_to_read_them(tmp_path):
    project = tmp_path / "missing_sources.json"
    project.write_text(json.dumps(valid_document(inputs={
        "genotype_file": "data/not_here.csv", "phenotype_file": None,
    })), encoding="utf-8")

    loaded = project_state.load_project(project)

    assert loaded["inputs"]["genotype_file"] == str((tmp_path / "data" / "not_here.csv").resolve())
    assert not Path(loaded["inputs"]["genotype_file"]).exists()
    assert loaded["inputs"]["phenotype_file"] is None


def test_empty_optional_paths_survive_round_trip(tmp_path):
    project = tmp_path / "no_inputs.json"
    project_state.save_project(project, {"inputs": {"genotype_file": "", "phenotype_file": None}})

    loaded = project_state.load_project(project)

    assert loaded["inputs"] == {"genotype_file": "", "phenotype_file": None}


@pytest.mark.parametrize("version", [None, 0, 2, "1", True, 1.0])
def test_unknown_or_wrongly_typed_schema_is_rejected(tmp_path, version):
    project = tmp_path / "wrong_schema.json"
    project.write_text(json.dumps(valid_document(schema_version=version)), encoding="utf-8")

    with pytest.raises(ValueError):
        project_state.load_project(project)


@pytest.mark.parametrize("document", [
    [],
    None,
    "a project is not a JSON string",
    {"application": "FragGenotyper"},
    valid_document(application="different application"),
    valid_document(application=None),
    valid_document(inputs=[]),
    valid_document(inputs=None),
    valid_document(inputs={"genotype_file": ["a.csv"]}),
    valid_document(inputs={"phenotype_file": 123}),
    valid_document(inputs={"phenotype_file": True}),
])
def test_malformed_project_documents_are_rejected(tmp_path, document):
    project = tmp_path / "invalid.json"
    project.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        project_state.load_project(project)


def test_non_json_text_is_rejected_without_evaluating_it(tmp_path):
    project = tmp_path / "not_json.json"
    sentinel = tmp_path / "must_not_be_created"
    project.write_text(f"__import__('pathlib').Path({str(sentinel)!r}).touch()", encoding="utf-8")

    with pytest.raises(ValueError):
        project_state.load_project(project)

    assert not sentinel.exists()


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_json_numeric_constants_are_rejected_on_load(tmp_path, constant):
    project = tmp_path / "nonfinite.json"
    project.write_text(
        '{"schema_version": 1, "application": "FragGenotyper", "extra": '
        + constant + "}", encoding="utf-8",
    )

    with pytest.raises(ValueError):
        project_state.load_project(project)


def test_json_strings_and_pickle_shaped_keys_remain_ordinary_data(tmp_path):
    project = tmp_path / "literal_text.json"
    sentinel = tmp_path / "must_not_be_created"
    payload = {
        "__reduce__": ["os.system", "touch must_not_be_created"],
        "expression": f"__import__('pathlib').Path({str(sentinel)!r}).touch()",
    }
    project_state.save_project(project, {"notes": payload})

    loaded = project_state.load_project(project)

    assert loaded["notes"] == payload
    assert not sentinel.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_values_do_not_overwrite_existing_project(tmp_path, value):
    project = tmp_path / "existing.json"
    original = json.dumps(valid_document(ui={"language": "zh"}))
    project.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError):
        project_state.save_project(project, {"analysis": {"threshold": value}})

    assert project.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".fraggenotyper-*.tmp"))


def test_successful_save_replaces_from_the_same_directory(tmp_path, monkeypatch):
    project = tmp_path / "atomic.json"
    real_replace = project_state.os.replace
    replacements = []

    def observe_replace(source, destination):
        source, destination = Path(source), Path(destination)
        assert source.parent == destination.parent == tmp_path
        assert source.is_file()
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(project_state.os, "replace", observe_replace)

    project_state.save_project(project, {"ui": {"language": "en"}})

    assert len(replacements) == 1
    assert project_state.load_project(project)["ui"]["language"] == "en"
    assert not list(tmp_path.glob(".fraggenotyper-*.tmp"))


def test_failed_atomic_replace_preserves_existing_file_and_cleans_temporary(tmp_path, monkeypatch):
    project = tmp_path / "existing.json"
    original = json.dumps(valid_document(ui={"language": "zh"}))
    project.write_text(original, encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(project_state.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated filesystem failure"):
        project_state.save_project(project, {"ui": {"language": "en"}})

    assert project.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".fraggenotyper-*.tmp"))
