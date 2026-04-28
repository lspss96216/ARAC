"""test_tuning_history.py — v1.13"""
import pathlib
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import tuning_history as th


def _make_attempt(module_name, attempt_n, shape, final_map, hp, loop_count=1, diag="diag"):
    return th.Attempt(
        timestamp=datetime.utcnow().isoformat(),
        loop_count=loop_count,
        module_name=module_name,
        attempt_n=attempt_n,
        shape=shape,
        final_map=final_map,
        peak_map=final_map,
        peak_epoch=10,
        final_epoch=15,
        hyperparams=hp,
        diagnosis=diag,
    )


def test_append_creates_file_with_header():
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "th.tsv")
        a = _make_attempt("CBAM", 1, "oscillating", 0.26, {"LR0": 0.01})
        th.append_attempt(path, a)
        text = pathlib.Path(path).read_text()
        assert "timestamp\tloop_count\t" in text
        assert "CBAM" in text
    print("✓ test_append_creates_file_with_header")


def test_append_multiple_does_not_duplicate_header():
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "th.tsv")
        for i, shape in enumerate(["oscillating", "monotonic_climbing", "converged_above_baseline"]):
            th.append_attempt(path, _make_attempt("CBAM", i + 1, shape, 0.26 + 0.01*i, {"LR0": 0.01 / (i+1)}))
        text = pathlib.Path(path).read_text()
        assert text.count("timestamp\tloop_count") == 1   # header once
        assert text.count("CBAM") == 3                    # 3 data rows
    print("✓ test_append_multiple_does_not_duplicate_header")


def test_read_all_round_trip():
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "th.tsv")
        a = _make_attempt("CBAM", 1, "oscillating", 0.26, {"LR0": 0.01, "OPTIMIZER": "SGD"})
        th.append_attempt(path, a)
        loaded = th.read_all(path)
    assert len(loaded) == 1
    got = loaded[0]
    assert got.module_name == "CBAM"
    assert got.shape == "oscillating"
    assert abs(got.final_map - 0.26) < 1e-6
    assert got.hyperparams["LR0"] == 0.01
    assert got.hyperparams["OPTIMIZER"] == "SGD"
    print("✓ test_read_all_round_trip")


def test_read_missing_returns_empty():
    assert th.read_all("/tmp/v113_does_not_exist.tsv") == []
    print("✓ test_read_missing_returns_empty")


def test_attempts_for_module_filters():
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "th.tsv")
        th.append_attempt(path, _make_attempt("CBAM", 1, "oscillating", 0.26, {"LR0": 0.01}))
        th.append_attempt(path, _make_attempt("EMA",  1, "monotonic_climbing", 0.27, {"LR0": 0.01}))
        th.append_attempt(path, _make_attempt("CBAM", 2, "converged_above_baseline", 0.29, {"LR0": 0.005}))
        cbam = th.attempts_for_module(path, "CBAM")
    assert len(cbam) == 2
    assert all(a.module_name == "CBAM" for a in cbam)
    assert cbam[0].attempt_n == 1
    assert cbam[1].attempt_n == 2
    print("✓ test_attempts_for_module_filters")


def test_latest_attempt_returns_most_recent():
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "th.tsv")
        th.append_attempt(path, _make_attempt("CBAM", 1, "oscillating",        0.26, {"LR0": 0.01}))
        th.append_attempt(path, _make_attempt("CBAM", 2, "monotonic_climbing", 0.27, {"LR0": 0.005}))
        th.append_attempt(path, _make_attempt("CBAM", 3, "converged_below_baseline", 0.28, {"LR0": 0.005}))
        latest = th.latest_attempt_for_module(path, "CBAM")
    assert latest.attempt_n == 3
    assert latest.shape == "converged_below_baseline"
    print("✓ test_latest_attempt_returns_most_recent")


def test_latest_attempt_none_for_unknown_module():
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "th.tsv")
        th.append_attempt(path, _make_attempt("CBAM", 1, "oscillating", 0.26, {"LR0": 0.01}))
        assert th.latest_attempt_for_module(path, "DoesNotExist") is None
    print("✓ test_latest_attempt_none_for_unknown_module")


def test_attempt_count_for_module():
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "th.tsv")
        for i in range(3):
            th.append_attempt(path, _make_attempt("CBAM", i + 1, "monotonic_climbing", 0.26 + 0.01*i, {"LR0": 0.01}))
        th.append_attempt(path, _make_attempt("EMA", 1, "oscillating", 0.25, {"LR0": 0.01}))
        assert th.attempt_count_for_module(path, "CBAM") == 3
        assert th.attempt_count_for_module(path, "EMA") == 1
        assert th.attempt_count_for_module(path, "Other") == 0
    print("✓ test_attempt_count_for_module")


def test_kept_attempts_filters_by_shape():
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "th.tsv")
        th.append_attempt(path, _make_attempt("CBAM", 1, "oscillating",                 0.26, {"LR0": 0.01}))
        th.append_attempt(path, _make_attempt("EMA",  1, "converged_above_baseline",    0.30, {"LR0": 0.001}))
        th.append_attempt(path, _make_attempt("WIoU", 1, "converged_below_baseline",    0.27, {"LR0": 0.01}))
        th.append_attempt(path, _make_attempt("MPDIoU", 1, "converged_above_baseline",  0.29, {"LR0": 0.005}))
        kept = th.kept_attempts(path)
    names = [a.module_name for a in kept]
    assert "EMA" in names
    assert "MPDIoU" in names
    assert "CBAM" not in names
    assert "WIoU" not in names
    print("✓ test_kept_attempts_filters_by_shape")


def test_diagnosis_truncated_to_500():
    """Long diagnoses must be truncated so TSV parser doesn't choke."""
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "th.tsv")
        long_diag = "x" * 800 + "\nembedded newline\nshould be removed"
        th.append_attempt(path, _make_attempt("CBAM", 1, "oscillating", 0.26, {"LR0": 0.01}, diag=long_diag))
        loaded = th.read_all(path)[0]
    assert len(loaded.diagnosis) <= 500
    assert "\n" not in loaded.diagnosis   # newlines were stripped
    print("✓ test_diagnosis_truncated_to_500")


def test_format_module_history_includes_all_attempts():
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "th.tsv")
        th.append_attempt(path, _make_attempt("CBAM", 1, "oscillating", 0.26, {"LR0": 0.01}))
        th.append_attempt(path, _make_attempt("CBAM", 2, "converged_below_baseline", 0.28, {"LR0": 0.005}))
        text = th.format_module_history_for_agent(path, "CBAM")
    assert "CBAM" in text
    assert "Attempt 1" in text
    assert "Attempt 2" in text
    assert "oscillating" in text
    assert "LR0=0.01" in text
    print("✓ test_format_module_history_includes_all_attempts")


def test_format_module_history_empty_returns_empty():
    with tempfile.TemporaryDirectory() as d:
        path = str(pathlib.Path(d) / "th.tsv")
        text = th.format_module_history_for_agent(path, "DoesNotExist")
    assert text == ""
    print("✓ test_format_module_history_empty_returns_empty")


TESTS = [
    test_append_creates_file_with_header,
    test_append_multiple_does_not_duplicate_header,
    test_read_all_round_trip,
    test_read_missing_returns_empty,
    test_attempts_for_module_filters,
    test_latest_attempt_returns_most_recent,
    test_latest_attempt_none_for_unknown_module,
    test_attempt_count_for_module,
    test_kept_attempts_filters_by_shape,
    test_diagnosis_truncated_to_500,
    test_format_module_history_includes_all_attempts,
    test_format_module_history_empty_returns_empty,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\nall {len(TESTS)} tests passed")
