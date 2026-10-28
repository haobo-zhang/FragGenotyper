"""Fast text-table imports with conservative phenotype-header detection.

Only a few initial lines are inspected. Pandas' C tokenizer performs the full
read in cancellable chunks; sample and group labels always remain strings.
"""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
import re
import shlex
import warnings

import pandas as pd


CHUNK_ROWS = 2048
_HEADER_NAMES = {
    "sample", "samples", "sampleid", "sample_id", "fid", "iid", "id",
    "genotype", "hap", "haplotype", "group", "phenotype", "phenotypes",
    "trait", "traits", "accession", "accessionid", "accession_id", "taxa",
    "taxon", "name", "样本", "样本名", "样本编号", "分型", "单倍型", "表型", "性状",
}
_MISSING_TEXT = {"", "na", "nan", "n/a", "null", "none", ".", "<na>"}


def _check(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Table import cancelled.")


def _emit(callback, percent, message):
    if callback is not None:
        callback(int(percent), message)


def _peek(path):
    lines = []
    truncated = False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for _ in range(24):
            line = handle.readline(8192)
            if not line:
                break
            if line.strip():
                lines.append(line)
            if len(line) == 8192 and not line.endswith(("\n", "\r")):
                truncated = True
                break
            if len(lines) >= 8:
                break
    if not lines:
        raise ValueError("The input table is empty.")
    return lines, truncated


def _records(lines, delimiter):
    if delimiter == r"\s+":
        records = []
        for line in lines:
            lexer = shlex.shlex(line, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            records.append(list(lexer))
        return records
    return list(csv.reader(io.StringIO("".join(lines)), delimiter=delimiter))


def _detect_delimiter(lines):
    candidates = []
    for delimiter in (",", "\t"):
        try:
            records = _records(lines, delimiter)
        except csv.Error:
            continue
        widths = [len(row) for row in records if row]
        if widths and widths[0] > 1:
            consistent = sum(width == widths[0] for width in widths)
            candidates.append((consistent / len(widths), widths[0], delimiter, records))
    if candidates:
        # Quoted commas/tabs are interpreted by csv.reader, not counted as
        # delimiters. Consistency takes precedence over the number of fields.
        _, _, delimiter, records = max(candidates, key=lambda item: (item[0], item[1]))
        return delimiter, records
    try:
        return r"\s+", _records(lines, r"\s+")
    except ValueError:
        # An unusually long/incomplete quoted first line is still readable by
        # the C tokenizer; it is never used to infer a missing header.
        return r"\s+", [lines[0].strip().split()]


def _numeric(value):
    try:
        return math.isfinite(float(value.strip()))
    except (ValueError, TypeError):
        return False


def _known_header(value):
    return value.strip().casefold().replace(" ", "_") in _HEADER_NAMES


def _identifier(value):
    value = value.strip()
    if not value or _known_header(value):
        return False
    # Pure ordinary numbers are ambiguous (a numeric trait title or sample
    # ID). Leading-zero IDs and IDs containing letters plus digits are clear.
    if re.fullmatch(r"\d+", value):
        return len(value) > 1 and value.startswith("0")
    return any(character.isalpha() for character in value) and any(character.isdigit() for character in value)


def _id_signature(value):
    return re.sub(r"\d+", "#", value.strip().casefold())


def _auto_header(records, truncated):
    first = records[0]
    if any(_known_header(value) for value in first):
        return "present", "Conventional column name in the first row."
    if truncated or len(first) not in {2, 3} or not _numeric(first[-1]):
        return "present", "Ambiguous input; conservatively retained the first row as column names."
    width = len(first)
    if any(len(row) != width for row in records):
        return "present", "Inconsistent initial row widths; header was not inferred."
    sample_index = 0 if width == 2 else 1
    if not _identifier(first[sample_index]):
        return "present", "First-row sample ID is ambiguous; use header='none' to override."
    signature = _id_signature(first[sample_index])
    for row in records:
        if (not _identifier(row[sample_index]) or _id_signature(row[sample_index]) != signature
                or not (_numeric(row[-1]) or row[-1].strip().casefold() in _MISSING_TEXT)):
            return "present", "Initial rows do not establish a consistent ID-and-value phenotype table."
    if width == 3:
        family_values = [row[0].strip() for row in records]
        if not all(family_values):
            return "present", "Family IDs are empty; header was not inferred."
        if not (all(_identifier(value) for value in family_values)
                or len(set(family_values)) == 1
                or all(row[0].strip() == row[1].strip() for row in records)):
            return "present", "Family/sample identifiers are ambiguous; header was not inferred."
    return "none", "Consistent sample-ID plus numeric phenotype rows; the first sample was retained."


def _column_names(first, header):
    if header == "none":
        if len(first) == 2:
            return ["Sample", "Phenotype"]
        if len(first) == 3:
            return ["FID", "Sample", "Phenotype"]
        return [f"Column{index + 1}" for index in range(len(first))]
    names = [str(name).strip() for name in first]
    if len(set(names)) != len(names):
        raise ValueError("Column names must be unique after trimming whitespace.")
    names = [name if name else f"Unnamed: {index}" for index, name in enumerate(names)]
    if len(set(names)) != len(names):
        raise ValueError("Column names must be unique after normalizing empty names.")
    return names


def read_table(path, progress_callback=None, cancel_event=None, header="auto"):
    """Read CSV, TSV or whitespace-delimited UTF-8 tables with string labels.

    ``header='auto'`` recognizes conventional headers first. Only clear two-
    column ID/value or three-column FID/IID/value rows are inferred headerless;
    ambiguous cases retain a header. Explicit ``present``/``none`` overrides
    detection. Missing values keep pandas' NA semantics; -9 stays the string
    '-9' until the user chooses a phenotype missing-value policy.
    """
    if header not in {"auto", "present", "none"}:
        raise ValueError("header must be 'auto', 'present' or 'none'.")
    _check(cancel_event)
    path = Path(path)
    _emit(progress_callback, 0, "Inspecting table format")
    _check(cancel_event)
    lines, truncated = _peek(path)
    delimiter, records = _detect_delimiter(lines)
    if not records or not records[0]:
        raise ValueError("No readable columns were found.")
    detected, reason = _auto_header(records, truncated) if header == "auto" else (header, "Explicit header override.")
    names = _column_names(records[0], detected)
    chunks, rows = [], 0
    file_bytes = max(path.stat().st_size, 1)
    with path.open("rb") as handle, warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        reader = pd.read_csv(handle, sep=delimiter, engine="c", dtype=str,
                             encoding="utf-8-sig", header=0 if detected == "present" else None,
                             names=names, chunksize=CHUNK_ROWS, index_col=False)
        try:
            while True:
                _check(cancel_event)
                try:
                    chunk = next(reader)
                except StopIteration:
                    break
                _check(cancel_event)
                chunks.append(chunk)
                rows += len(chunk)
                percent = min(99, max(1, int(handle.tell() * 100 / file_bytes)))
                _emit(progress_callback, percent, f"Read {rows:,} rows")
        finally:
            reader.close()
    _check(cancel_event)
    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=names)
    table.attrs["table_import"] = {
        "source": str(path.resolve()),
        "format": {",": "csv", "\t": "tsv", r"\s+": "whitespace"}[delimiter],
        "delimiter": delimiter, "header": detected, "header_mode": header,
        "header_detection": reason, "rows": int(len(table)), "columns": len(table.columns),
        "engine": "c", "chunk_rows": CHUNK_ROWS,
    }
    _emit(progress_callback, 100, f"Loaded {len(table):,} rows × {len(table.columns)} columns")
    _check(cancel_event)
    return table
