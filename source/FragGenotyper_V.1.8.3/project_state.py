"""Portable JSON project settings; source tables remain ordinary files."""
import copy
import json
import os
from pathlib import Path
import tempfile

SCHEMA_VERSION = 1
PATH_KEYS = ("genotype_file", "phenotype_file")


def save_project(path, state):
    path = Path(path).resolve()
    document = copy.deepcopy(state)
    document["schema_version"] = SCHEMA_VERSION
    document["application"] = "FragGenotyper"
    inputs = document.setdefault("inputs", {})
    for key in PATH_KEYS:
        if inputs.get(key):
            source = Path(inputs[key]).expanduser().resolve()
            try:
                inputs[key] = os.path.relpath(source, path.parent)
            except ValueError:  # Different Windows drive letters.
                inputs[key] = str(source)
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".fraggenotyper-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_project(path):
    path = Path(path).resolve()
    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON number: {value}")
    document = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported project schema; expected schema_version=1.")
    if document.get("application") != "FragGenotyper":
        raise ValueError("This JSON file is not a FragGenotyper project.")
    inputs = document.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("Project inputs must be a JSON object.")
    for key in PATH_KEYS:
        value = inputs.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Invalid input path: {key}")
        if value:
            source = Path(value).expanduser()
            inputs[key] = str((path.parent / source).resolve() if not source.is_absolute() else source.resolve())
    return document

