"""
Shared metric extractors for evaluation.parsing.source in {stdout, json, csv}.

Both autoresearch Step 6 (Verify) and dataset hunter Phase 6 Step 2 need to
extract metrics from either stdout (regex over run.log), a json file with
dotted paths, or the last row of a csv. Implementing all three in both places
meant that the B1 bug (json/csv branches were `...` stubs) existed twice over.

This module implements them once. Usage:

    from shared.parse_metrics import extract

    # evaluation.parsing from research_config.yaml
    results = extract(source="stdout", parsing_cfg={"patterns": {...}})
    results = extract(source="json",   parsing_cfg={"json_file": "...", "json_paths": {...}})
    results = extract(source="csv",    parsing_cfg={"csv_file": "...",  "csv_columns": {...}})

All extractors return dict[str, float | None]. None means the metric was not
found (regex no match / dotted path miss / missing column). Callers decide
whether None counts as crash.
"""
from __future__ import annotations

import csv as _csv
import json as _json
import pathlib
import re
from typing import Optional

Num = Optional[float]


# ---------------------------------------------------------------------------
# Dotted-path resolver for json source
# ---------------------------------------------------------------------------

def _pluck(obj, path: str):
    """Resolve 'a.b[0].c' style path on obj. Returns None on any miss.

    Examples:
        _pluck({"a": {"b": [10, 20]}}, "a.b[0]")   -> 10
        _pluck({"a": {"b": [10, 20]}}, "a.missing") -> None
        _pluck({"a": [1, 2, 3]},       "a[10]")    -> None
    """
    cur = obj
    for key, idx in re.findall(r"([^.\[\]]+)|\[(\d+)\]", path):
        try:
            if key:
                cur = cur[key]
            elif idx:
                cur = cur[int(idx)]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_from_stdout(log_path: str, patterns: dict[str, str]) -> dict[str, Num]:
    """Match regex patterns against log file text. Each pattern must have one
    capture group — the metric value."""
    log = pathlib.Path(log_path).read_text(errors="ignore")
    out: dict[str, Num] = {}
    for name, pat in patterns.items():
        m = re.search(pat, log)
        if m:
            try:
                out[name] = float(m.group(1))
            except (TypeError, ValueError):
                out[name] = None
        else:
            out[name] = None
    return out


def extract_from_json(json_file: str, json_paths: dict[str, str]) -> dict[str, Num]:
    """Load json file and pluck values by dotted paths."""
    try:
        data = _json.loads(pathlib.Path(json_file).read_text())
    except (OSError, _json.JSONDecodeError):
        return {name: None for name in json_paths}
    out: dict[str, Num] = {}
    for name, dotted in json_paths.items():
        v = _pluck(data, dotted)
        try:
            out[name] = float(v) if v is not None else None
        except (TypeError, ValueError):
            out[name] = None
    return out


def extract_from_csv(csv_file: str, cols: dict[str, str]) -> dict[str, Num]:
    """Read csv and pluck last non-empty row's columns by header name."""
    try:
        with open(csv_file, newline="") as f:
            rows = [r for r in _csv.DictReader(f)
                    if any((v or "").strip() for v in r.values())]
    except OSError:
        return {name: None for name in cols}
    last = rows[-1] if rows else {}
    out: dict[str, Num] = {}
    for name, col in cols.items():
        raw = (last.get(col) or "").strip()
        if not raw:
            out[name] = None
        else:
            try:
                out[name] = float(raw)
            except ValueError:
                out[name] = None
    return out


def extract(source: str, parsing_cfg: dict) -> dict[str, Num]:
    """Dispatch on source and return {metric_name: value | None}.

    parsing_cfg is evaluation.parsing from research_config.yaml. Required
    keys depend on source:

      source=stdout:
        - patterns: {metric_name: regex_with_one_capture_group}
        - log_file: optional, default "run.log"

      source=json:
        - json_file: path to json output
        - json_paths: {metric_name: dotted_path}

      source=csv:
        - csv_file: path to csv output
        - csv_columns: {metric_name: column_header}

    Raises RuntimeError for unknown source.
    """
    if source == "stdout":
        return extract_from_stdout(
            parsing_cfg.get("log_file", "run.log"),
            parsing_cfg["patterns"],
        )
    if source == "json":
        return extract_from_json(
            parsing_cfg["json_file"],
            parsing_cfg["json_paths"],
        )
    if source == "csv":
        return extract_from_csv(
            parsing_cfg["csv_file"],
            parsing_cfg["csv_columns"],
        )
    raise RuntimeError(
        f"Unsupported evaluation.parsing.source: {source!r}. "
        f"Supported: stdout / json / csv."
    )
