# Changelog — v1.7.6

Release type: **Bug fix.** Four real bugs found during code review of
v1.7.5, plus removal of v1.7.5's auto-write `pyproject.toml` feature.
No new features. No schema changes. Drop-in on top of v1.7.5.

## Background — why this release exists

v1.7.5 shipped 11 onboarding fixes based on user-reported friction. A
post-release code review found **four real bugs** in the existing crash
handling and repair flow that had been latent across multiple versions
(v1.6, v1.7.1, and the new v1.7.5 pyproject.toml feature). Three of the
four would only surface in long-running pipelines under specific
conditions (>3 consecutive crashes, successful Step 5.5 short-test,
pre-v1.7.5 state files), which is why they hadn't been hit during
earlier testing.

v1.7.6 is purely defensive — fix the four bugs, retreat from one
feature whose design was wrong. v1.8 work (full_yaml mode) starts
immediately after this release.

## Bug fixes

### Bug 1 — Crash-pause sequencing (autoresearch Step 9)

**Severity**: deadlock under sustained crash conditions.

**Symptom**: After 3 consecutive crashes, the handler would:
1. Halve `BATCH_SIZE` in train.py's working tree
2. `git reset --hard HEAD~1` to revert the broken experiment commit

The `git reset` reverts both the broken commit AND the BATCH_SIZE
change in the working tree (since the halve was never committed). Next
loop iteration reads the pre-crash BATCH_SIZE from train.py, crashes
again the same way. Counter resets to 0 every 3 crashes, halve runs but
gets reset, infinite loop with no observable forward progress.

**Why it was latent**: requires actually hitting 3 consecutive
real-OOM-style crashes in production. Most v1.7 testing used clean
yaml_inject experiments where Step 5.5 caught failures earlier.

**Fix**: reorder to revert FIRST, then halve in clean working tree,
then commit the halve so it survives any future reset:

```python
subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=False)  # 1. revert
# 2. read post-revert BATCH_SIZE, halve, write back to train.py
# ...
subprocess.run(["git", "commit", "-m", "crash-pause: halve BATCH_SIZE..."])  # 3. commit
```

### Bug 2 — BATCH_SIZE halve floor

**Severity**: silent infinite no-op.

**Symptom**: `new_bs = max(1, int(m.group(1)) // 2)` — once BATCH_SIZE
hits 1, halving stays at 1 forever (1 // 2 = 0, max with 1 = 1). Every
3 crashes after that triggers the same no-op halve + revert dance, with
nothing changing and nothing flagged.

**Fix**: detect `current <= 1` and skip the halve entirely. Log a
`limitation`-category discovery instead so the user sees that BATCH_SIZE
can't be reduced further and intervention is needed (smaller IMGSZ,
fewer epochs, different model, etc.). Counter still resets to avoid
filling discoveries.md with duplicate entries every 3 loops.

### Bug 3 — Step 5.5 short-test restore reads wrong state key

**Severity**: KeyError on every successful Step 5.5 short-test.

**Symptom**: After Step 5.5 short-test (120s) passes, the SKILL example
restored TIME_BUDGET via `state['time_budget_sec']`. But orchestrator
stores this value as `state['loop_time_budget']`. Every successful
short-test that progressed to full-budget rerun would crash with
`KeyError: 'time_budget_sec'`.

**Why it was latent since v1.7.1**: no real Step 5.5 short-test had
completed end-to-end. Every actual Step 5.5 invocation in testing had
either hit unfixable crashes or stopped after Tier 1 (which doesn't go
through this restore code).

**Fix**: read `state['loop_time_budget']` to match orchestrator's
canonical key.

### Bug 4 — state_migrate `python_runner` default contradicts v1.7.5

**Severity**: stale state forces wrong runner across upgrade.

**Symptom**: `state_migrate.CURRENT_DEFAULTS["python_runner"] = "uv run"`
predates v1.7.5's local-first detection logic. When a pre-v1.7.5 state
file is migrated and is missing the `python_runner` key, state_migrate
fills in `"uv run"` regardless of whether uv works on the machine. The
loop then runs everything under uv, hitting the same
`ModuleNotFoundError: numpy` symptom that v1.7.5 was supposed to fix.

**Fix**:
- Default in state_migrate changed to `"python3"` (matches v1.7.5
  philosophy)
- Stage 0 Step 2 resume branch now **overwrites** state's
  `python_runner` with this Stage 0's detection result every time.
  Stale state can no longer lock the pipeline into a broken runner.

## Feature removed — auto-write pyproject.toml (v1.7.5 #5)

v1.7.5 added Stage 0 Step 6.5 to scaffold a minimal `pyproject.toml`
when the user wanted uv-managed envs. Three problems with the
implementation:

1. **Hardcoded `requires-python = ">=3.10"`** — Python 3.9 systems
   would fail `uv sync`, fall back to system python3 (which was also
   3.9), then crash later on PEP 604 `int | None` syntax.
2. **`[tool.uv] override-dependencies = []` was a no-op** — empty list
   doesn't override anything. The CHANGELOG claimed it would prevent
   re-downloading torch over the user's CUDA install. It didn't.
3. **`name = "{project_name}"` had no PEP 503 normalisation** — project
   names with spaces, uppercase, or non-ASCII characters made `uv sync`
   crash with cryptic errors before the pipeline even reached Stage 1.

The scope was wrong. Torch / CUDA / cuDNN / Python version combinations
on research machines vary too much for any auto-generated pyproject.toml
to work reliably.

**v1.7.6 replacement**: Step 6.5 now does the import verification only —
no scaffolding. If the chosen runner can't `import ultralytics`, it
fails with a clear remediation message offering two paths:
- (a) `pip install ultralytics` for system python
- (b) Write your own `pyproject.toml`, run `uv sync`, then re-run

The pipeline's job is to pick the right runner and verify it works; the
user's job is to manage the environment. Trying to do both was a
category mistake.

## Files changed in v1.7.6

| File | Change |
|---|---|
| `autoresearch/SKILL.md` | Crash-pause sequence reordered (Bug 1) + BATCH_SIZE=1 floor (Bug 2). Step 5.5 short-test restore key fixed (Bug 3). |
| `shared/state_migrate.py` | `python_runner` default changed to `"python3"` (Bug 4). |
| `research-orchestrator/SKILL.md` | Resume branch overwrites stale `python_runner` (Bug 4). Step 6.5 rewritten as verify-only, no auto-scaffold (feature removal). |
| `CHANGELOG_v1.7.6.md` | New (this file) |
| `README.md` | Version bumped, layout tree, Versions list |

## Files NOT changed

Byte-identical to v1.7.5: `paper-finder/SKILL.md`, `dataset-hunter/SKILL.md`,
all templates, all examples, `weight_transfer.py`, `modules_md.py`,
`parse_metrics.py`, all test files, `arch_spec.schema.json`,
`train-script-spec.md`, `file-contracts.md`.

## Test coverage

Unchanged from v1.7.5:

| Suite | Count |
|---|---|
| `shared/test_modules_md.py` | 18 |
| `shared/test_templates.py` | 3 |
| `shared/test_weight_transfer.py` | 36 |
| **Total python tests** | **57** |
| SKILL.md python snippets | 88 |
| YAML examples | 2 |

The fixes are sequencing/key-name corrections + feature removal — none
of them require new test infrastructure. The infinite-loop bugs (1, 2)
would need a long-running integration test to exercise; that's beyond
v1.7.6 scope and has been flagged for v1.8.

## Upgrade path

Drop-in. State files from any prior version migrate transparently:
- `python_runner` either gets re-detected at Stage 0 (existing key
  overwritten) or filled in as `"python3"` (missing key default updated)
- All other state keys unchanged

For users who had a working v1.7.5 setup with `pyproject.toml`
auto-scaffolded:
- The existing `pyproject.toml` is not deleted or modified by upgrade
- Step 6.5 will check it works and proceed; if it doesn't work, the
  remediation message points to the two options
- v1.7.5's `orchestrator.use_uv_project: true` yaml flag is now ignored
  (no harm, no warning)

## v1.7.x line — closing remarks

v1.7.6 is the final v1.7.x patch release planned. v1.8 work begins
immediately on `apply_yaml_spec` for `full_yaml` integration mode.

| Version | Theme |
|---|---|
| v1.7   | yaml_inject feature + weight_transfer.py |
| v1.7.1 | Step 5.5 repair loop + Q2/Q3 patches |
| v1.7.2 | Cross-skill IMGSZ handoff |
| v1.7.3 | preferred_locations secondary sort |
| v1.7.4 | Tie-breaking rule (don't ask, pick one) |
| v1.7.5 | Onboarding hardening (11 fixes from real run) |
| v1.7.6 | 4 latent crash-handling bugs + pyproject.toml retreat |

The line is now stable. Next release: **v1.8 — `full_yaml` mode**.
