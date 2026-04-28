"""invariants.py — v1.8

Runtime contract checks called by autoresearch's Step 4 (Commit) before
each `git commit`. The goal is to catch contract violations as crashes
rather than letting them silently corrupt the experimental record.

Background. Across v1.7.x development we accumulated several rules that
the agent is supposed to follow:

  - Locked variables (TIME_BUDGET, SEED, IMGSZ) must match orchestrator's
    canonical values in pipeline_state.json
  - OPTIMIZER must never be 'auto' (silently overrides LR0/MOMENTUM)
  - Section ② structure must remain intact (orchestrator scaffold contract)

Until v1.8 these rules lived only in SKILL.md prose. v1.7.x agents mostly
followed them, but `discoveries.md` from real runs shows occasional
violations (different IMGSZ between iterations because agent halved it
during OOM, OPTIMIZER set to 'auto' once during an "AdamW vs SGD" test
where the agent forgot the rule). Each violation invalidates that
iteration's keep/discard verdict relative to the baseline.

This module makes those rules enforceable. Each check returns either
None (OK) or a Violation describing the specific breach. Step 4 runs
all checks; any violation aborts the commit, logs to discoveries.md as
`agent_violation` category, increments consecutive_crashes (so 3 in a
row triggers crash-pause), and discards the iteration.

Design principles:

  1. Read-only. Never modify train.py to "fix" a violation — the agent's
     deviation IS the data point. Rolling back loses the signal.
  2. Cheap. Every check is regex on already-loaded source. Whole sweep
     should take <100 ms even on a 1000-line train.py.
  3. Explicit. Each Violation includes the rule name, the expected value,
     the observed value, and a remediation hint.
"""

from __future__ import annotations
from dataclasses import dataclass
import re
import pathlib
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Violation type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Violation:
    """A single contract violation. Returned from check_* functions."""
    rule: str               # e.g. "IMGSZ_locked"
    script: str             # e.g. "train.py"
    expected: Any           # e.g. 1920
    observed: Any           # e.g. 640
    hint: str               # actionable remediation message

    def __str__(self) -> str:
        return (
            f"[{self.rule}] in {self.script}: "
            f"expected={self.expected!r}, observed={self.observed!r}. "
            f"{self.hint}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────────────

# Variables that orchestrator locks at Stage 3 Step 3. These must match
# canonical values from pipeline_state.json across the entire run.
LOCKED_VARS = {
    # state_key            → train.py variable name
    "loop_time_budget":    "TIME_BUDGET",
    "seed":                "SEED",
    "imgsz":               "IMGSZ",
    # v1.12 — BATCH_SIZE locked. Previously (v1.6 - v1.11.1) BATCH_SIZE was
    # initial-set-but-not-locked: yaml provided the starting value and
    # autoresearch was free to halve dynamically (resource_impact auto-halve,
    # crash-pause halve, OOM detection). Real-world session 2026-04-27
    # demonstrated this conflicted with fair comparison: half the loops ran
    # at BATCH=64 (baseline), the other half at BATCH=32 (auto-halved or
    # OOM-recovered), and Loop 9 had to manually re-run a vanilla baseline
    # at BATCH=32 to interpret results. v1.12 locks BATCH_SIZE the same way
    # as IMGSZ: yaml sets one value, every experiment uses that value, and
    # OOM is treated as the experiment being infeasible (discard) rather
    # than a signal to dynamically re-size the experimental setup.
    "batch_size":          "BATCH_SIZE",
}


def _read_int_var(src: str, var: str) -> int | None:
    """Find `^VAR = <int>` and return the int, or None if not found / not int."""
    # Tolerate trailing comments and whitespace, e.g. "TIME_BUDGET = 7200  # 2h"
    m = re.search(rf"(?m)^{re.escape(var)}\s*=\s*(\d+)\b", src)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _read_string_var(src: str, var: str) -> str | None:
    """Find `^VAR = "value"` (double or single quoted) → value string."""
    m = re.search(rf"""(?m)^{re.escape(var)}\s*=\s*['"]([^'"]+)['"]""", src)
    return m.group(1) if m else None


def check_locked_variables(src: str, state: dict, script: str = "train.py") -> list[Violation]:
    """Check TIME_BUDGET / SEED / IMGSZ match state. Returns list of violations."""
    violations: list[Violation] = []
    for state_key, var_name in LOCKED_VARS.items():
        if state_key not in state:
            # State doesn't have this key (e.g. resume from pre-v1.7.2).
            # Don't false-positive — just skip this check.
            continue
        expected = state[state_key]
        observed = _read_int_var(src, var_name)
        if observed is None:
            # Variable missing entirely from script. That's a different
            # problem (spec compliance) caught elsewhere; don't double-flag.
            continue
        if observed != expected:
            violations.append(Violation(
                rule=f"{var_name}_locked",
                script=script,
                expected=expected,
                observed=observed,
                hint=(
                    f"{var_name} is locked by orchestrator (Critical Rules "
                    f"#9/#10/#11). Restore it to {expected!r}. If you "
                    f"genuinely need to change it, that requires changing "
                    f"research_config.yaml and restarting the pipeline."
                ),
            ))
    return violations


def check_optimizer_not_auto(src: str, script: str = "train.py") -> list[Violation]:
    """v1.7.7 Critical Rule #15 — OPTIMIZER must never be 'auto'."""
    observed = _read_string_var(src, "OPTIMIZER")
    if observed is None:
        return []   # variable missing → spec compliance issue, not this check
    if observed.lower() == "auto":
        return [Violation(
            rule="OPTIMIZER_not_auto",
            script=script,
            expected="any of: SGD, AdamW, Adam, RMSProp, NAdam, RAdam",
            observed=observed,
            hint=(
                "ultralytics 'optimizer=auto' silently overrides LR0 and "
                "MOMENTUM, making any LR experiment a no-op. Set OPTIMIZER "
                "to a concrete optimizer. See train-script-spec.md "
                "§ Tunable contract."
            ),
        )]
    return []


def check_section_markers_present(src: str, script: str = "train.py") -> list[Violation]:
    """All four section markers must be present (orchestrator scaffold contract).

    Uses the same regex as check_spec_compliance but as a runtime check —
    catches the case where agent edited Section ② and accidentally deleted
    the Section ③ marker.
    """
    violations: list[Violation] = []
    for n, marker_name in enumerate(["Section 1", "Section 2", "Section 3", "Section 4"], 1):
        # Match circled or ASCII digit, with or without leading # / ═
        char_class = ["①1", "②2", "③3", "④4"][n - 1]
        pat = re.compile(rf"(?mi)^#?\s*(?:═+\s*\n#\s*)?Section\s*[{char_class}]\b")
        if not pat.search(src):
            violations.append(Violation(
                rule=f"section_{n}_marker",
                script=script,
                expected=f"a line matching `Section {n}` or `Section {['①','②','③','④'][n-1]}`",
                observed="MISSING",
                hint=(
                    f"Section {n} marker was deleted from {script}. "
                    f"Re-scaffold via `rm {script}` and let orchestrator "
                    f"Stage 0 Step 6 regenerate. Manual edits should never "
                    f"remove section markers — they are the contract surface "
                    f"that lets autoresearch find what to edit."
                ),
            ))
    return violations


# v1.9.2 — run.log freshness sentinel. train.py writes:
#   __RUN_START__: <iso> <commit> <pid>     (first stdout line)
#   __RUN_END__:   <iso> <exit_code>        (last stdout line, atexit)
# Step 6 verify calls check_run_log_fresh(state) FIRST. If run.log was
# written by a different project / earlier loop / never started, the
# check raises before any metrics are parsed.
RUN_START_RE = re.compile(r"^__RUN_START__:\s+(\S+)\s+(\S+)\s+(\d+)\s*$")
RUN_END_RE   = re.compile(r"^__RUN_END__:\s+(\S+)\s+(-?\d+)\s*$")


def check_run_log_fresh(state: dict, run_log_path: str = "run.log") -> list[Violation]:
    """Verify run.log was written by THIS Step 5 invocation, not stale.

    Three checks, any failure → Violation:

      1. File exists. (Step 5 must have produced output.)
      2. First line is `__RUN_START__: <iso> <commit> <pid>`. Absence
         means train.py was killed before it could write the sentinel
         (process never reached Section ④'s entry point) OR the file
         was produced by a non-pipeline tool (i.e. wrong file).
      3. RUN_START's iso timestamp >= state["step5_started_at"]. If
         RUN_START is older than Step 5's recorded start, the file was
         left over from a previous loop OR a neighbour project's run.

    The presence of __RUN_END__ is NOT required here — a missing
    __RUN_END__ may indicate a kill or genuine crash that Step 6's
    metrics parser handles separately. We only require RUN_END exists
    if state['step5_completed'] is True (meaning Step 5 returned cleanly).
    """
    violations: list[Violation] = []
    p = pathlib.Path(run_log_path)
    if not p.exists():
        violations.append(Violation(
            rule="run_log_exists",
            script=run_log_path,
            expected="file exists after Step 5",
            observed="MISSING",
            hint=(
                "run.log not present. Step 5 may have failed before train.py "
                "produced any output (e.g. immediate import error in train.py "
                "before the runner could redirect stdout). Treat as crash."
            ),
        ))
        return violations

    try:
        text = p.read_text(errors="replace")
    except Exception as e:
        violations.append(Violation(
            rule="run_log_readable",
            script=run_log_path,
            expected="readable text file",
            observed=f"read failed: {e!r}",
            hint="run.log is not a regular text file. Inspect manually.",
        ))
        return violations

    # First non-empty line must be RUN_START. ultralytics' progressbar lib
    # may emit ANSI escape codes; the print() call writes our sentinel
    # before any ultralytics import, so it should be on line 1.
    lines = text.splitlines()
    first_nonempty = next((ln for ln in lines if ln.strip()), "")
    m_start = RUN_START_RE.match(first_nonempty)
    if not m_start:
        violations.append(Violation(
            rule="run_log_start_sentinel",
            script=run_log_path,
            expected="first line: __RUN_START__: <iso> <commit> <pid>",
            observed=first_nonempty[:80] if first_nonempty else "<empty file>",
            hint=(
                "run.log first line is not __RUN_START__. Either (a) train.py "
                "is from before v1.9.2 and lacks the sentinel — re-scaffold, "
                "or (b) the file was written by a different process (cross-"
                "project pollution). Verify project_root matches cwd."
            ),
        ))
        return violations

    # Check freshness: RUN_START iso >= state["step5_started_at"]
    expected_start = state.get("step5_started_at")
    observed_start = m_start.group(1)
    if expected_start and observed_start < expected_start:
        violations.append(Violation(
            rule="run_log_freshness",
            script=run_log_path,
            expected=f"RUN_START >= {expected_start} (Step 5 start time)",
            observed=f"RUN_START = {observed_start}",
            hint=(
                "run.log RUN_START is older than the timestamp Step 5 recorded "
                "before launching train.py. The log file is stale, possibly "
                "from a previous loop or from another project sharing this "
                "directory. Re-run Step 5 and verify cwd == project_root."
            ),
        ))

    return violations



    """All four section markers must be present (orchestrator scaffold contract).

    Uses the same regex as check_spec_compliance but as a runtime check —
    catches the case where agent edited Section ② and accidentally deleted
    the Section ③ marker.
    """
    violations: list[Violation] = []
    for n, marker_name in enumerate(["Section 1", "Section 2", "Section 3", "Section 4"], 1):
        # Match circled or ASCII digit, with or without leading # / ═
        char_class = ["①1", "②2", "③3", "④4"][n - 1]
        pat = re.compile(rf"(?mi)^#?\s*(?:═+\s*\n#\s*)?Section\s*[{char_class}]\b")
        if not pat.search(src):
            violations.append(Violation(
                rule=f"section_{n}_marker",
                script=script,
                expected=f"a line matching `Section {n}` or `Section {['①','②','③','④'][n-1]}`",
                observed="MISSING",
                hint=(
                    f"Section {n} marker was deleted from {script}. "
                    f"Re-scaffold via `rm {script}` and let orchestrator "
                    f"Stage 0 Step 6 regenerate. Manual edits should never "
                    f"remove section markers — they are the contract surface "
                    f"that lets autoresearch find what to edit."
                ),
            ))
    return violations


# ──────────────────────────────────────────────────────────────────────────────
# Aggregator
# ──────────────────────────────────────────────────────────────────────────────

    return violations


# v1.11.1 #3 — `inject_modules()` must not replace the model's ModuleList.
# Real-world (Loop 2 of session 2026-04-27): user wrote
#     model.model.model = nn.ModuleList(new_layers)
# inside inject_modules() to insert attention modules. This shifts every
# layer's index by +1, but ultralytics' Detect head uses HARDCODED from-
# indices (e.g. Concat[-1, 6]) baked into the YAML. Index shift breaks
# Concat shape resolution → mAP crashes to 0.0644. The bug is silent at
# import/scaffold time and only manifests as "experiment looks broken"
# during training.
#
# Correct alternatives:
#   - hook mode: PicklableHook.attach(layer, HookCls, ...) — wraps forward
#     output without changing layer count.
#   - yaml_inject mode: weight_transfer.apply_yaml_spec — handles index
#     shifting via update_head_refs.
#   - full_yaml mode: replace the entire architecture YAML; ultralytics
#     parses the new from-refs from scratch.

_INJECT_MODULES_BODY_RE = re.compile(
    r"def\s+inject_modules\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\s*\n(.*?)(?=^def\s+|\Z)",
    re.DOTALL | re.MULTILINE,
)
# Anti-pattern: `<obj>.model.model = ...` assignment.
# The user code from real-world Loop 2 was literally:
#     model.model.model = nn.ModuleList(new_layers)
# Where outer `model` is the YOLO wrapper, `.model` is DetectionModel,
# `.model.model` is the ModuleList. Replacing this is the bug.
# We match the full chain `\b\w+\.model\.model\s*=` (assignment to a
# variable's .model.model attribute).
_MODULELIST_ASSIGN_RE = re.compile(r"\b\w+\.model\.model\s*=(?!=)")


def check_no_modulelist_replacement_in_inject_modules(
    src: str, script: str = "train.py"
) -> list[Violation]:
    """v1.11.1 #3 — flag `model.model.model = ...` assignment inside
    inject_modules() function body.

    Catches the silent-corruption anti-pattern surfaced by real-world
    Loop 2 of 2026-04-27: replacing the layer ModuleList shifts Detect
    head from-indices and crashes mAP to ~0 with no error message.
    """
    m = _INJECT_MODULES_BODY_RE.search(src)
    if not m:
        # No inject_modules() defined — nothing to check. Many scripts
        # don't have one (vanilla baseline, etc.); silent OK.
        return []
    body = m.group(1)
    if _MODULELIST_ASSIGN_RE.search(body):
        return [Violation(
            rule="inject_modules_no_modulelist_replacement",
            script=script,
            expected=(
                "hook mode (PicklableHook.attach) or yaml_inject mode "
                "(weight_transfer.apply_yaml_spec) for structural changes"
            ),
            observed=(
                "model.model.model = ... assignment inside inject_modules() body"
            ),
            hint=(
                "v1.11.1 #3 anti-pattern. Replacing the layer ModuleList shifts "
                "Detect head from-indices (e.g. Concat[-1, 6]) and silently "
                "crashes mAP. Use one of:\n"
                "  • PicklableHook.attach(layer, HookCls, ...) — wraps forward "
                "output, no index change.\n"
                "  • weight_transfer.apply_yaml_spec(...) — full custom YAML "
                "with index re-resolution.\n"
                "  • full_yaml mode in modules.md — entire architecture replaced."
            ),
        )]
    return []


# v1.12 C — ultralytics auto-batch-reduce is a contract violation now.
# Real-world Loop 10 of session 2026-04-27: MPDIoU was set up at BATCH=64
# but OOMed; ultralytics' built-in handler emitted
#     WARNING ⚠️ CUDA out of memory with batch=64. Reducing to batch=32 and retrying (1/3).
# and silently re-ran at BATCH=32. Loop 11 hit the same path with AdamW.
# Both experiments produced metrics that LOOKED comparable to the BATCH=64
# baseline but were trained on a different effective configuration. Fair
# comparison demands BATCH_SIZE be locked; mid-run reduction breaks that.
#
# v1.12 catches this by scanning run.log for ultralytics' specific warning
# string. If found: ContractViolation, autoresearch discards the iteration,
# and the module gets marked blocked (this BATCH+module combination is
# infeasible on this GPU; user can rerun with smaller yaml batch_size or
# disable the module).

_ULTRALYTICS_AUTO_BATCH_REDUCE_RE = re.compile(
    r"WARNING.*?CUDA out of memory with batch=\d+\..*?Reducing to batch=\d+",
    re.IGNORECASE | re.DOTALL,
)


def check_no_ultralytics_auto_batch_reduce(
    run_log_path: str = "run.log",
) -> list[Violation]:
    """v1.12 — flag ultralytics' auto-batch-reduce warning in run.log.

    Why this is an invariant: BATCH_SIZE is locked from yaml. ultralytics'
    OOM handler tries to "help" by halving batch and retrying — but that
    produces results trained at a smaller batch than every other
    experiment, breaking fair comparison. Either:
      - Run finishes at the original locked batch (success, fair compare)
      - Run truly OOMs (discard, mark module blocked, user knows)

    The middle path — silently retraining at half batch — is forbidden.

    Detects by scanning run.log for ultralytics' specific warning string.
    Caller (autoresearch Step 6) reads the result and treats any match
    as ContractViolation → discard iteration, mark module blocked.
    """
    p = pathlib.Path(run_log_path)
    if not p.exists():
        # No run.log to check — pass through (other checks catch this case).
        return []
    try:
        log_text = p.read_text(errors="ignore")
    except Exception:
        return []
    m = _ULTRALYTICS_AUTO_BATCH_REDUCE_RE.search(log_text)
    if m is None:
        return []
    return [Violation(
        rule="ultralytics_auto_batch_reduce_detected",
        script="run.log",
        expected="run completes at the locked BATCH_SIZE, or fails with OOM (no auto-reduce)",
        observed=f"ultralytics auto-batch-reduce warning: {m.group(0)[:200]!r}",
        hint=(
            "v1.12 — BATCH_SIZE is locked. ultralytics' built-in OOM handler "
            "tried to halve batch and retrain, but that breaks fair comparison "
            "across iterations (every other run uses the locked batch). "
            "Treat this experiment as DISCARD and mark the module blocked.\n"
            "Resolution options:\n"
            "  • Lower yaml `autoresearch.loop.batch_size` so all experiments "
            "(including baseline) run at the smaller value. Re-run from Loop 0.\n"
            "  • Tag the module with higher resource_impact in modules.md so "
            "autoresearch can predict the OOM (it'll still discard, but at "
            "least the discard happens before the run instead of mid-run)."
        ),
    )]


def run_all_checks(state: dict, script_paths: list[str] | None = None) -> list[Violation]:
    """Run every invariant check across given scripts. Returns flat list of
    violations (empty = all OK).

    Default scripts: train.py and track.py if it exists.
    """
    if script_paths is None:
        script_paths = []
        for p in ("train.py", "track.py"):
            if pathlib.Path(p).exists():
                script_paths.append(p)

    all_violations: list[Violation] = []
    for path in script_paths:
        try:
            src = pathlib.Path(path).read_text()
        except FileNotFoundError:
            continue
        all_violations.extend(check_locked_variables(src, state, script=path))
        all_violations.extend(check_optimizer_not_auto(src, script=path))
        all_violations.extend(check_section_markers_present(src, script=path))
        # v1.11.1 #3 — silent-corruption anti-pattern guard
        all_violations.extend(check_no_modulelist_replacement_in_inject_modules(src, script=path))
    return all_violations


def format_violations(violations: list[Violation]) -> str:
    """Pretty-print violations for discoveries.md / stderr."""
    if not violations:
        return "no violations"
    lines = [f"{len(violations)} contract violation(s):"]
    for v in violations:
        lines.append(f"  • {v}")
    return "\n".join(lines)


class ContractViolation(Exception):
    """Raised by autoresearch Step 4 when invariants fail.

    Caller should catch this, log to discoveries.md as agent_violation
    category, increment consecutive_crashes, and discard the iteration.
    Do NOT catch and ignore — silent acceptance of contract violation
    silently invalidates the experimental record.
    """
    def __init__(self, violations: list[Violation]):
        super().__init__(format_violations(violations))
        self.violations = violations
