"""test_invariants.py — v1.8

Pure-python tests for invariants.py. No torch / ultralytics dependencies;
runs in the same fast suite as test_modules_md / test_weight_transfer /
test_hook_utils.
"""
import sys
import pathlib
import tempfile
import os

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import invariants as inv


# ── fixture ────────────────────────────────────────────────────────────────

CLEAN_SRC = """\
# Section ① — Imports
import numpy as np

# Section ② — Tunables
TIME_BUDGET = 7200
SEED        = 42
IMGSZ       = 1920
OPTIMIZER   = "SGD"
LR0         = 0.01
BATCH_SIZE  = 16

# Section ③ — Modules

# Section ④ — Main
def main(): pass
"""

CANONICAL_STATE = {
    "loop_time_budget": 7200,
    "seed": 42,
    "imgsz": 1920,
}


# ── individual check tests ────────────────────────────────────────────────

def test_locked_variables_all_match():
    assert inv.check_locked_variables(CLEAN_SRC, CANONICAL_STATE) == []
    print("✓ test_locked_variables_all_match")


def test_locked_variables_imgsz_changed():
    state = {**CANONICAL_STATE, "imgsz": 640}
    violations = inv.check_locked_variables(CLEAN_SRC, state)
    assert len(violations) == 1
    assert violations[0].rule == "IMGSZ_locked"
    assert violations[0].expected == 640
    assert violations[0].observed == 1920
    print("✓ test_locked_variables_imgsz_changed")


def test_locked_variables_seed_changed():
    src = CLEAN_SRC.replace("SEED        = 42", "SEED        = 7")
    violations = inv.check_locked_variables(src, CANONICAL_STATE)
    assert len(violations) == 1
    assert violations[0].rule == "SEED_locked"
    assert violations[0].observed == 7
    print("✓ test_locked_variables_seed_changed")


def test_locked_variables_time_budget_halved():
    """Common agent mistake — halve TIME_BUDGET to test faster."""
    src = CLEAN_SRC.replace("TIME_BUDGET = 7200", "TIME_BUDGET = 3600")
    violations = inv.check_locked_variables(src, CANONICAL_STATE)
    assert len(violations) == 1
    assert violations[0].rule == "TIME_BUDGET_locked"
    assert violations[0].observed == 3600
    print("✓ test_locked_variables_time_budget_halved")


def test_locked_variables_multiple_violations():
    """All three locked vars wrong → all three reported."""
    src = (CLEAN_SRC
        .replace("TIME_BUDGET = 7200", "TIME_BUDGET = 1200")
        .replace("SEED        = 42", "SEED        = 99")
        .replace("IMGSZ       = 1920", "IMGSZ       = 640"))
    violations = inv.check_locked_variables(src, CANONICAL_STATE)
    rules = {v.rule for v in violations}
    assert rules == {"TIME_BUDGET_locked", "SEED_locked", "IMGSZ_locked"}, rules
    print("✓ test_locked_variables_multiple_violations")


def test_locked_variables_missing_state_key_skipped():
    """If state lacks a key (e.g. resume from pre-v1.7.2), don't false-positive."""
    state_partial = {"loop_time_budget": 7200, "seed": 42}   # no imgsz
    violations = inv.check_locked_variables(CLEAN_SRC, state_partial)
    rules = {v.rule for v in violations}
    assert "IMGSZ_locked" not in rules
    print("✓ test_locked_variables_missing_state_key_skipped")


def test_locked_variables_missing_var_in_script_skipped():
    """If train.py is missing the variable entirely, that's a spec compliance
    issue (caught elsewhere), not a lock violation."""
    src = CLEAN_SRC.replace("IMGSZ       = 1920\n", "")
    violations = inv.check_locked_variables(src, CANONICAL_STATE)
    rules = {v.rule for v in violations}
    assert "IMGSZ_locked" not in rules
    print("✓ test_locked_variables_missing_var_in_script_skipped")


def test_locked_variables_with_trailing_comment():
    """Variable with comment on same line should still be parsed."""
    src = CLEAN_SRC.replace(
        "TIME_BUDGET = 7200",
        "TIME_BUDGET = 7200  # 2 hours per run, locked"
    )
    violations = inv.check_locked_variables(src, CANONICAL_STATE)
    assert violations == []
    print("✓ test_locked_variables_with_trailing_comment")


# ── optimizer auto check ───────────────────────────────────────────────────

def test_optimizer_sgd_ok():
    assert inv.check_optimizer_not_auto(CLEAN_SRC) == []
    print("✓ test_optimizer_sgd_ok")


def test_optimizer_adamw_ok():
    src = CLEAN_SRC.replace('OPTIMIZER   = "SGD"', 'OPTIMIZER   = "AdamW"')
    assert inv.check_optimizer_not_auto(src) == []
    print("✓ test_optimizer_adamw_ok")


def test_optimizer_auto_detected():
    src = CLEAN_SRC.replace('OPTIMIZER   = "SGD"', 'OPTIMIZER   = "auto"')
    violations = inv.check_optimizer_not_auto(src)
    assert len(violations) == 1
    assert violations[0].rule == "OPTIMIZER_not_auto"
    print("✓ test_optimizer_auto_detected")


def test_optimizer_auto_case_insensitive():
    """'AUTO', 'Auto', etc. all trigger."""
    for variant in ["Auto", "AUTO", "aUtO"]:
        src = CLEAN_SRC.replace('OPTIMIZER   = "SGD"', f'OPTIMIZER   = "{variant}"')
        violations = inv.check_optimizer_not_auto(src)
        assert len(violations) == 1, f"{variant} should be detected"
    print("✓ test_optimizer_auto_case_insensitive")


def test_optimizer_single_quotes_ok():
    """Single quotes accepted for OPTIMIZER."""
    src = CLEAN_SRC.replace('OPTIMIZER   = "SGD"', "OPTIMIZER   = 'SGD'")
    assert inv.check_optimizer_not_auto(src) == []
    print("✓ test_optimizer_single_quotes_ok")


def test_optimizer_missing_var_skipped():
    """Missing OPTIMIZER → spec compliance handles, not this check."""
    src = CLEAN_SRC.replace('OPTIMIZER   = "SGD"\n', "")
    assert inv.check_optimizer_not_auto(src) == []
    print("✓ test_optimizer_missing_var_skipped")


# ── section markers ────────────────────────────────────────────────────────

def test_section_markers_all_present_unicode():
    assert inv.check_section_markers_present(CLEAN_SRC) == []
    print("✓ test_section_markers_all_present_unicode")


def test_section_markers_all_present_ascii():
    """v1.7.5 #11 — ASCII section markers also accepted."""
    src = (CLEAN_SRC
        .replace("Section ①", "Section 1")
        .replace("Section ②", "Section 2")
        .replace("Section ③", "Section 3")
        .replace("Section ④", "Section 4"))
    assert inv.check_section_markers_present(src) == []
    print("✓ test_section_markers_all_present_ascii")


def test_section_marker_3_missing():
    src = CLEAN_SRC.replace("# Section ③ — Modules\n", "")
    violations = inv.check_section_markers_present(src)
    assert len(violations) == 1
    assert violations[0].rule == "section_3_marker"
    print("✓ test_section_marker_3_missing")


# ── aggregator ─────────────────────────────────────────────────────────────

def test_run_all_checks_clean():
    """Write a clean train.py to a tmpdir; run_all_checks returns []."""
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        pathlib.Path("train.py").write_text(CLEAN_SRC)
        violations = inv.run_all_checks(CANONICAL_STATE)
        assert violations == []
    print("✓ test_run_all_checks_clean")


def test_run_all_checks_aggregates_multiple():
    """Both train.py and track.py exist, both have violations → all reported."""
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        # train.py: IMGSZ wrong
        pathlib.Path("train.py").write_text(
            CLEAN_SRC.replace("IMGSZ       = 1920", "IMGSZ       = 640"))
        # track.py: OPTIMIZER auto
        pathlib.Path("track.py").write_text(
            CLEAN_SRC.replace('OPTIMIZER   = "SGD"', 'OPTIMIZER   = "auto"'))
        violations = inv.run_all_checks(CANONICAL_STATE)
        scripts = {v.script for v in violations}
        assert scripts == {"train.py", "track.py"}, scripts
        rules = {(v.script, v.rule) for v in violations}
        assert ("train.py", "IMGSZ_locked") in rules
        assert ("track.py", "OPTIMIZER_not_auto") in rules
    print("✓ test_run_all_checks_aggregates_multiple")


def test_run_all_checks_missing_track_py_ok():
    """Default behaviour skips missing scripts gracefully."""
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        pathlib.Path("train.py").write_text(CLEAN_SRC)
        violations = inv.run_all_checks(CANONICAL_STATE)
        assert violations == []   # track.py not present, only train.py checked
    print("✓ test_run_all_checks_missing_track_py_ok")


def test_format_violations_empty():
    assert inv.format_violations([]) == "no violations"
    print("✓ test_format_violations_empty")


def test_format_violations_pretty():
    violations = [
        inv.Violation(rule="X", script="train.py", expected=1, observed=2, hint="fix it"),
        inv.Violation(rule="Y", script="track.py", expected=3, observed=4, hint="fix it too"),
    ]
    s = inv.format_violations(violations)
    assert "2 contract violation(s)" in s
    assert "[X]" in s
    assert "[Y]" in s
    print("✓ test_format_violations_pretty")


def test_contract_violation_exception():
    """Raising ContractViolation works and carries the violations."""
    violations = [
        inv.Violation(rule="Z", script="train.py", expected=1, observed=2, hint="x"),
    ]
    try:
        raise inv.ContractViolation(violations)
    except inv.ContractViolation as e:
        assert e.violations == violations
        assert "[Z]" in str(e)
        print("✓ test_contract_violation_exception")
        return
    raise AssertionError("expected ContractViolation to raise")


# ── v1.9.2 — run.log freshness check ──────────────────────────────────────

def test_run_log_fresh_missing_file():
    with tempfile.TemporaryDirectory() as d:
        violations = inv.check_run_log_fresh(
            {"step5_started_at": "2026-04-27T10:00:00"},
            run_log_path=str(pathlib.Path(d) / "nonexistent.log"),
        )
    assert len(violations) == 1
    assert violations[0].rule == "run_log_exists"
    print("✓ test_run_log_fresh_missing_file")


def test_run_log_fresh_no_sentinel():
    """run.log without __RUN_START__ on first line → start_sentinel violation."""
    with tempfile.TemporaryDirectory() as d:
        log = pathlib.Path(d) / "run.log"
        log.write_text("Loading model...\nepoch 1/100\n")
        violations = inv.check_run_log_fresh(
            {"step5_started_at": "2026-04-27T10:00:00"},
            run_log_path=str(log),
        )
    assert len(violations) == 1
    assert violations[0].rule == "run_log_start_sentinel"
    print("✓ test_run_log_fresh_no_sentinel")


def test_run_log_fresh_empty_file():
    with tempfile.TemporaryDirectory() as d:
        log = pathlib.Path(d) / "run.log"
        log.write_text("")
        violations = inv.check_run_log_fresh(
            {"step5_started_at": "2026-04-27T10:00:00"},
            run_log_path=str(log),
        )
    assert len(violations) == 1
    assert violations[0].rule == "run_log_start_sentinel"
    print("✓ test_run_log_fresh_empty_file")


def test_run_log_fresh_stale():
    """RUN_START iso older than step5_started_at → freshness violation."""
    with tempfile.TemporaryDirectory() as d:
        log = pathlib.Path(d) / "run.log"
        log.write_text(
            "__RUN_START__: 2026-04-26T10:00:00 abcd1234 12345\n"
            "epoch 1/100\n"
        )
        violations = inv.check_run_log_fresh(
            {"step5_started_at": "2026-04-27T10:00:00"},
            run_log_path=str(log),
        )
    assert len(violations) == 1
    assert violations[0].rule == "run_log_freshness"
    print("✓ test_run_log_fresh_stale")


def test_run_log_fresh_clean():
    """Valid sentinel + fresh timestamp → no violations."""
    with tempfile.TemporaryDirectory() as d:
        log = pathlib.Path(d) / "run.log"
        log.write_text(
            "__RUN_START__: 2026-04-27T10:30:00 abcd1234 12345\n"
            "epoch 1/100\n"
            "__RUN_END__: 2026-04-27T12:30:00 0\n"
        )
        violations = inv.check_run_log_fresh(
            {"step5_started_at": "2026-04-27T10:00:00"},
            run_log_path=str(log),
        )
    assert violations == []
    print("✓ test_run_log_fresh_clean")


def test_run_log_fresh_no_step5_timestamp_skips_freshness():
    """If state lacks step5_started_at, skip freshness check (resume from
    pre-v1.9.2 state). Sentinel format is still checked."""
    with tempfile.TemporaryDirectory() as d:
        log = pathlib.Path(d) / "run.log"
        log.write_text(
            "__RUN_START__: 2026-04-27T10:30:00 abcd1234 12345\n"
            "epoch 1/100\n"
        )
        violations = inv.check_run_log_fresh({}, run_log_path=str(log))
    assert violations == []
    print("✓ test_run_log_fresh_no_step5_timestamp_skips_freshness")


TESTS = [
    test_locked_variables_all_match,
    test_locked_variables_imgsz_changed,
    test_locked_variables_seed_changed,
    test_locked_variables_time_budget_halved,
    test_locked_variables_multiple_violations,
    test_locked_variables_missing_state_key_skipped,
    test_locked_variables_missing_var_in_script_skipped,
    test_locked_variables_with_trailing_comment,
    test_optimizer_sgd_ok,
    test_optimizer_adamw_ok,
    test_optimizer_auto_detected,
    test_optimizer_auto_case_insensitive,
    test_optimizer_single_quotes_ok,
    test_optimizer_missing_var_skipped,
    test_section_markers_all_present_unicode,
    test_section_markers_all_present_ascii,
    test_section_marker_3_missing,
    test_run_all_checks_clean,
    test_run_all_checks_aggregates_multiple,
    test_run_all_checks_missing_track_py_ok,
    test_format_violations_empty,
    test_format_violations_pretty,
    test_contract_violation_exception,
    # v1.9.2 — freshness checks
    test_run_log_fresh_missing_file,
    test_run_log_fresh_no_sentinel,
    test_run_log_fresh_empty_file,
    test_run_log_fresh_stale,
    test_run_log_fresh_clean,
    test_run_log_fresh_no_step5_timestamp_skips_freshness,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\nall {len(TESTS)} tests passed")
