# Changelog — v1.9.2

Release type: **Patch — cross-project run.log pollution fix.** Adds 4
layers of defence so an agent in the wrong cwd cannot read a neighbour
project's run.log and parse metrics from a different experiment. Drop-in
on top of v1.9.1.

## Background

Real-world report (2026-04): a user running multiple ARAC projects on
the same machine reported that "Step 5 還沒跑完 agent 就以為跑完了" —
Step 5 hadn't finished but the agent thought it had. Investigation
revealed the failure mode:

1. Agent in project A is at the wrong cwd (started in project B's dir, or
   a subdir, or `/tmp`)
2. Agent reads `run.log` (relative path) → gets project B's last run
3. project B's run is already complete → parser succeeds, returns metrics
4. **project A's results.tsv gets written with project B's metrics**
5. project A's keep/discard decision runs on someone else's data
6. Symptom looks like "Step 5 finished early" because project B's
   results were already there

This is worse than v1.7.5 #9's `ULTRALYTICS_RUNS_DIR` issue (where ckpt
goes to neighbour `runs/`). That one polluted artefacts; this one
**poisons the experimental record itself** with foreign metrics. v1.7.5
#9 was caught because subsequent loops failed to find checkpoints.
This one is silent corruption.

## The fix — 4 layers of defence

All four layers are independent and required. Any single one prevents
the pollution; all four together also catch the corner cases (race
conditions, partial writes, pre-v1.9.2 train.py without sentinels,
malformed timestamps).

### Layer 1 — `state["project_root"]` absolute path (orchestrator Stage 0)

Orchestrator's state init now resolves `paths.project_root` to an
absolute path at Stage 0:

```python
"project_root":          str(pathlib.Path(
                             cfg.get("paths", {}).get("project_root", ".")
                         ).resolve()),
```

This is the **source of truth** for every file operation across the
pipeline. SKILL code that previously wrote `pathlib.Path("run.log")`
or `cd <project>` now uses
`pathlib.Path(state["project_root"]) / "run.log"`. cwd is no longer
trusted.

### Layer 2 — Run sentinels in `run.log` (train.py / track.py templates)

`train.py.detection` and `train.py.tracking` now write boundary
sentinels around `main()`:

```
__RUN_START__: <iso timestamp> <git_commit_short> <pid>     ← line 1
... normal output ...
__RUN_END__:   <iso timestamp> <exit_code>                   ← last line
```

For tracking task_type, `track.py.tracking` overwrites only `__RUN_END__`
since it appends to run.log after train.py — train.py's `__RUN_START__`
remains as the first line.

The sentinel is the freshness fingerprint. A stale run.log either lacks
the sentinel (different file format) or has an older timestamp than
this loop's Step 5.

### Layer 3 — Step 5 prelude (autoresearch SKILL § Step 5)

Step 5 now performs FOUR pre-run actions before launching train.py:

```python
# 1. cwd verification — refuse to run if cwd != project_root
if pathlib.Path(os.getcwd()).resolve() != pathlib.Path(state["project_root"]):
    raise RuntimeError("cwd-lock violated: ...")

# 2. unlink stale run.log + the per-loop archive (defence in depth)
(project_root / "run.log").unlink(missing_ok=True)
(project_root / "logs" / f"loop_{LOOP_N}.log").unlink(missing_ok=True)

# 3. record Step 5 start time (Layer 4 will compare against this)
state["step5_started_at"] = datetime.now().isoformat(timespec="seconds")
state["step5_started_at_unix"] = datetime.now().timestamp()

# 4. cd into project_root in the bash run command (belt-and-braces)
cd "$PROJECT_ROOT"
$RUNNER train.py > run.log 2>&1
```

### Layer 4 — Pre-Step-6 freshness check (`shared/invariants.py`)

New invariant `check_run_log_fresh(state)` runs as the FIRST thing in
Step 6, before any metrics parsing. Three sub-checks:

1. **File exists.** Step 5 must have produced output. If `run.log`
   doesn't exist, raise.
2. **First line is `__RUN_START__`.** If the file's first non-empty
   line doesn't match the sentinel regex, the file is either pre-v1.9.2
   (train.py wasn't writing sentinels) or was produced by a non-pipeline
   process (cross-project pollution). Raise.
3. **RUN_START timestamp ≥ `state["step5_started_at"]`.** If the
   sentinel's iso timestamp is older than when Step 5 started, the
   file is stale (left over from a previous loop OR a different
   project sharing the directory). Raise.

If any check fails, autoresearch:
- Logs a `ContractViolation` to `discoveries.md` as `agent_violation`
  category
- Increments `consecutive_crashes` (so 3 in a row triggers v1.7.6
  crash-pause)
- Raises `ContractViolation` — Step 6 skips metric parsing entirely;
  iteration treated as crash; results.tsv NOT written with stale
  metrics

This is the contract: Step 6 verifies "the run.log I'm about to read
is mine" before parsing it.

## Files changed

### Modified

| File | Change |
|---|---|
| `shared/invariants.py` | New `check_run_log_fresh(state, run_log_path)` (~80 lines), 2 new regex constants for sentinels |
| `shared/test_invariants.py` | +6 tests (23 → 29) covering missing/no-sentinel/empty/stale/clean/no-step5-timestamp |
| `shared/templates/train.py.detection` | New `_write_run_start_sentinel()` + `_write_run_end_sentinel()` helpers; `if __name__` block uses try/finally to guarantee RUN_END writes even on crash |
| `shared/templates/train.py.tracking` | Same |
| `shared/templates/track.py.tracking` | New `_write_run_end_sentinel()` only (does not overwrite train.py's RUN_START); same try/finally pattern |
| `shared/test_templates.py` | New REQUIRED_SENTINELS + REQUIRED_SENTINELS_TRACK_ONLY checks |
| `shared/state_migrate.py` | New keys `project_root`, `step5_started_at`, `step5_started_at_unix`; post-default hook fills `project_root` from cwd if None (best-effort migration) |
| `shared/file-contracts.md` | New § "v1.9.2 — Run sentinels" documenting RUN_START / RUN_END format |
| `research-orchestrator/SKILL.md` | State init `project_root` resolves to absolute path |
| `autoresearch/SKILL.md` | Step 5 § "v1.9.2 — Pre-run cleanup and cwd lock" subsection (Layer 3); Step 5 archive `cp` uses absolute paths; Step 6 § "v1.9.2 — Freshness check (FIRST thing in Step 6)" subsection (Layer 4) |
| `examples/research_config.visdrone-detection.yaml` | `project_root` comment expanded |
| `examples/research_config.visdrone-mot.yaml` | Same |
| `CHANGELOG_v1.9.2.md` | New (this file) |
| `README.md` | Banner + Versions list |

### Unchanged

`paper-finder/SKILL.md`, `dataset-hunter/SKILL.md`, `shared/modules_md.py`,
`shared/parse_metrics.py`, `shared/weight_transfer.py`,
`shared/hook_utils.py`, `shared/results-tsv-guide.md`, all earlier
CHANGELOGs, `templates/arch_spec.schema.json`.

## Test coverage

| Suite | v1.9.1 | v1.9.2 |
|---|---|---|
| `shared/test_modules_md.py` | 23 | 23 |
| `shared/test_templates.py` | 3 | 3 (REQUIRED_SENTINELS now enforced) |
| `shared/test_weight_transfer.py` | 61 | 61 |
| `shared/test_hook_utils.py` | 13 | 13 |
| `shared/test_invariants.py` | 23 | **29** (+6) |
| **Total python tests** | **123** | **129** (+6) |
| SKILL.md python snippets | 98 | **100** (+2) |
| YAML examples | 2 | 2 |

The 6 new invariants tests cover all check_run_log_fresh failure modes
plus the clean-pass case plus the resume-from-pre-v1.9.2 graceful path
(state without `step5_started_at` skips freshness check rather than
false-positive).

## Upgrade path

Drop-in. State migration adds 3 new keys with sensible defaults:
- `project_root`: filled in from `pathlib.Path(".").resolve()` if
  state file lacks it (safe fallback; Step 5's cwd-lock catches the
  rare case where this is wrong)
- `step5_started_at`: None — Step 5 sets on first invocation post-upgrade
- `step5_started_at_unix`: same

For users with v1.9.1 in-progress runs:
- Resume continues normally
- The first post-upgrade Step 5 records `step5_started_at` and creates
  fresh sentinels
- Step 6's freshness check skips when `step5_started_at` is None
  (graceful resume path), so pre-v1.9.2 in-flight loops don't
  false-positive
- All subsequent loops have full v1.9.2 protection

For brand-new pipelines: nothing user-facing changes. Yaml's `project_root`
field was already there; v1.9.2 just uses it more rigorously.

**Important behaviour note**: if an agent had been silently reading
neighbour project run.logs before v1.9.2, that pollution path is now
hard-failed. The first iteration after upgrade may raise
`ContractViolation` if the agent was previously parsing stale logs;
this is the correct behaviour. Verify cwd matches project_root and
the run completes cleanly.

## v1.10 outlook

Unchanged from v1.9 outlook. v1.9.2 is a single-issue patch; v1.10
candidates remain the speculative list (per-machine resource
calibration, yaml_inject REPLACE semantics, real-env integration tests).
The agent-contract surface keeps growing — invariants.py is now 4
checks (3 from v1.8, 1 from v1.9.2), 29 tests. Future contract
violations follow the same pattern: detect via invariants, raise
`ContractViolation`, iteration → crash, no silent corruption.
