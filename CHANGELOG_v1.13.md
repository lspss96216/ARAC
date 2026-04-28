# Changelog — v1.13

Release type: **Major — multi-attempt per-module hyperparameter tuning.**
Loop semantics changed: a "loop" was 1 (module, default_hp) experiment in
v1.6-v1.12.1; in v1.13 a loop is 1 (module, attempt_N) experiment, with up
to 3 attempts per module (extensible to 5 if attempt-to-attempt mAP improves
≥3%). The agent reads trajectory shape + diagnosis between attempts and
chooses next-attempt hyperparams using paper recipes + cross-loop history.

## Background

Real-world session 2026-04-27 (v1.11 production run on YOLO26x + VisDrone
@ IMGSZ=640) discarded all 11 loops. Root-cause analysis showed three of
the discards (FlexSimAM, CBAM-backbone, IEMA-neck — all yaml_inject) had
mAP -0.025 to -0.035 vs baseline 0.2850. The trajectory pattern was
consistent: monotonic climbing through training, never reaching baseline
within the 1200s TIME_BUDGET. Random-init layers needed more time or
different hyperparams to converge.

v1.12 / v1.12.1 addressed orthogonal issues (BATCH_SIZE locked, dead
config wired up). v1.13 attacks the convergence problem directly by
giving each module multiple attempts with adaptive hyperparams informed
by trajectory analysis.

The user's design constraints (clarified in design discussion):

1. **Same module = up to 3 attempts** with attempt-to-attempt
   improvement ≥2% NOT counting as no-improvement
2. **OPTIMIZER becomes attempt-changeable** (was v1.9.3 "initial-set" —
   one-shot per pipeline). Still never `'auto'`.
3. **Trajectory shape + diagnosis as SIGNALS, not rules**. SKILL emits
   shape (`monotonic_climbing` / `oscillating` / etc.) and a free-form
   diagnosis string; agent reads + paper recipe + cross-loop history
   and decides hyperparams.
4. **Extension allowed**: if attempt 2 → 3 improvement ≥3%, autoresearch
   can grant +1/+2 extension up to a hard cap of 5 attempts.

## What v1.13 introduces

### New module: `shared/trajectory.py` (~280 lines, 18 tests)

Per-epoch trajectory parsing + 6-shape classification. Public API:

```python
from shared.trajectory import parse_results_csv, classify_shape

points = parse_results_csv("runs/detect/exp/results.csv")
diag = classify_shape(points, baseline_final_map=0.2850)
# diag.shape ∈ CANONICAL_SHAPES, diag.diagnosis is free-form text,
# diag.{final,peak}_map, diag.{peak,final}_epoch for tsv recording
```

The 6 canonical shapes (priority order — first match wins):

| Shape | Trigger | Implication for next attempt |
|---|---|---|
| `flat_no_learning` | Whole-run swing < 1% of mean | LR way too low / model frozen |
| `early_collapse` | Peak before epoch 0.7n + drop ≥5% | LR too high / NaN gradients |
| `oscillating` | Late-half swing ≥5% AND non-monotonic | LR too high or optimizer unstable |
| `train_val_diverge` | Second-half train_loss ≥10%↓ but val plateau | Overfit — weight_decay↑, augmentation↑ |
| `monotonic_climbing` | Last-third still rising | Budget-limited — try LR↑+warmup↑ or more epochs |
| `converged_above_baseline` | Plateau at end, final ≥ baseline | KEEP candidate |
| `converged_below_baseline` | Plateau at end, final < baseline | Hyperparam tweaks unlikely to recover; consider discard |

Thresholds tunable via `_SHAPE_THRESHOLDS` dict. Calibrated from real-run
trajectories so misclassification is rare on YOLO+VisDrone-shape curves.

### New module: `shared/tuning_history.py` (~200 lines, 12 tests)

TSV-based append-only attempt history. One row per (loop, module,
attempt_n) with hyperparams used + trajectory shape + diagnosis. Public API:

```python
from shared.tuning_history import (
    Attempt, append_attempt, attempts_for_module,
    latest_attempt_for_module, attempt_count_for_module,
    kept_attempts, format_module_history_for_agent,
)
```

The `format_module_history_for_agent` returns human-readable text for
inclusion in the agent's reasoning context — this is how cross-attempt
learning is exposed to the agent.

### `shared/modules_md.py` — 2 new statuses

```python
VALID_STATUSES = {"pending", "tuning", "injected", "tested", "discarded", "blocked"}
```

- `tuning`: between `pending` and `tested`. autoresearch is mid-attempt
  sequence (1 < attempt_n ≤ max_extended_attempts). `find_pending`
  excludes these so dispatch doesn't re-pick during tuning.
- `blocked`: was state["blocked_modules"] in v1.12; promoted to first-class
  status. Module can't run on this GPU at locked BATCH_SIZE.

### autoresearch SKILL — Step 2/3/7/8 changes

**Step 2 (Pick) — tuning continuation prelude**

Before walking the priority ladder, check `state.current_tuning_module`.
If set with `attempt_n > 0`: bump `current_tuning_attempt+=1`, jump
straight to Step 3 with the same module locked in. Priority ladder is
bypassed for tuning continuations.

When a NEW pending module is picked from priority ladder, set:
```python
state["current_tuning_module"]   = chosen.name
state["current_tuning_attempt"]  = 1
state["tuning_attempt_extended"] = False
state["last_attempt_final_map"]  = None
```

**Step 3 (Modify) — hyperparam patch helper**

New function `patch_train_var(name, value)` for modifying Section ② vars.
Refuses to touch locked vars (raises RuntimeError). For attempt 2..N,
the architectural change persists from attempt 1; only hyperparam vars
are patched. Allowed: `LR0`, `MOMENTUM`, `WEIGHT_DECAY`, `WARMUP_EPOCHS`,
`OPTIMIZER`. Forbidden: `BATCH_SIZE`, `IMGSZ`, `SEED`, `TIME_BUDGET`,
`OPTIMIZER='auto'`.

The agent's reasoning workflow (illustrated in SKILL prose):

```
Read tuning_history.tsv for this module → attempt 1: shape=oscillating, LR=0.01
Read paper recipe                       → "SGD lr=0.001 momentum=0.9"
Read trajectory diagnosis               → "swing 0.04 suggests LR too high"
Reason                                  → halve LR toward paper's value
Patch                                   → patch_train_var("LR0", 0.005)
```

**Step 7 (Decide) — tuning-aware verdict overlay**

Existing keep/discard logic is preserved. v1.13 adds an overlay:

| Verdict | At final attempt? | Action |
|---|---|---|
| keep | any | Finalize → status=`tested`, clear tuning state |
| discard | yes (cap reached) | Finalize → status=`discarded`, reason=`tuning_failed` |
| discard | no (more attempts allowed) | Status=`tuning` (provisional), keep state for next loop |

Extension granted when `tuning_attempt_extended == False` AND
`attempt == max_attempts` AND attempt-to-attempt improvement ≥
`attempt_extension_threshold` (default 3%). One extension per module
max; extension cap is `max_extended_attempts` (default 5).

**Step 8 (Log) — trajectory recording + tuning-aware no_improvement counter**

Every attempt records to `tuning_history.tsv` regardless of verdict. The
`no_improvement_loops` counter is now tuning-aware: when in mid-tuning
sequence AND attempt-to-attempt improvement ≥ `no_improvement_skip_threshold`
(default 2%), the counter does NOT increment. This prevents stop trigger
from firing during productive tuning runs.

### yaml — new `autoresearch.tuning` block

```yaml
autoresearch:
  tuning:
    enabled: true
    max_attempts: 3
    attempt_extension_threshold: 0.03      # 3% attempt-to-attempt → extension allowed
    max_extended_attempts: 5               # absolute cap with extensions
    no_improvement_skip_threshold: 0.02    # 2% attempt-to-attempt → don't count toward stall
```

`enabled: false` reverts to v1.12.1 single-attempt-per-module behaviour.

### state schema — 4 new fields

```python
"current_tuning_module":   None,    # name of module being tuned (None when between)
"current_tuning_attempt":  0,       # 1-indexed attempt within current module
"tuning_attempt_extended": False,   # whether +1/+2 extension already granted
"last_attempt_final_map":  None,    # previous attempt's mAP for delta check
```

state_migrate adds these with safe defaults so pre-v1.13 state files
load without KeyError.

## Files changed

| File | Change |
|---|---|
| `shared/trajectory.py` | NEW (~280 lines) — parse_results_csv + classify_shape (6 shapes) + Diagnosis dataclass |
| `shared/test_trajectory.py` | NEW (18 tests) — parser + shape classification + priority + edge cases |
| `shared/tuning_history.py` | NEW (~200 lines) — Attempt dataclass + append/read/format helpers |
| `shared/test_tuning_history.py` | NEW (12 tests) — round-trip + filter + format |
| `shared/modules_md.py` | VALID_STATUSES adds `tuning` + `blocked` (~15 lines doc) |
| `shared/test_modules_md.py` | +3 tests for tuning + blocked status semantics |
| `shared/state_migrate.py` | +4 fields with safe defaults |
| `research-orchestrator/SKILL.md` | State init mirrors 4 new fields |
| `autoresearch/SKILL.md` | Step 2 tuning-continuation prelude (~80 lines), Step 3 patch helper (~80 lines), Step 7 tuning verdict overlay (~70 lines), Step 8 trajectory record + tuning-aware no_improvement counter (~80 lines) |
| `examples/research_config.visdrone-detection.yaml` | New `autoresearch.tuning` block (~25 lines) |
| `examples/research_config.visdrone-mot.yaml` | Same block, abbreviated |
| `CHANGELOG_v1.13.md` | NEW (this file) |
| `README.md` | Banner + Versions list |

### Unchanged

`shared/invariants.py`, `shared/hook_utils.py`, `shared/weight_transfer.py`,
`shared/templates/`, `paper-finder/SKILL.md`, `dataset-hunter/SKILL.md`,
`papers2code/SKILL.md`. v1.13 is a SKILL-level + new shared-module change;
existing infrastructure modules unchanged.

## Test coverage

| Suite | v1.12.1 | v1.13 |
|---|---|---|
| `shared/test_modules_md.py` | 34 | **37** (+3 v1.13 status) |
| `shared/test_templates.py` | 3 | 3 |
| `shared/test_weight_transfer.py` | 61 | 61 |
| `shared/test_hook_utils.py` | 23 | 23 |
| `shared/test_invariants.py` | 40 | 40 |
| `shared/test_trajectory.py` | — | **18** NEW |
| `shared/test_tuning_history.py` | — | **12** NEW |
| **Total python tests** | **161** | **191** (+30) |
| SKILL.md python snippets | 103 | 107 |
| YAML examples | 2 | 2 |

## Upgrade path

**Drop-in for state schema** — v1.12.1 state files load fine. New 4
fields default to None / 0 / False; first iteration after upgrade picks
a fresh module starting at `attempt=1`.

**Drop-in for yaml** — v1.12.1 yamls load fine; v1.13's `autoresearch.tuning`
block defaults to `enabled: true` if the block is missing entirely.

**BREAKING for autoresearch behaviour** — pipeline behaviour changes
when picked modules don't beat baseline immediately:
- v1.12.1: 1 attempt → discard → next module
- v1.13: up to 3 attempts → keep / discard verdict

Concretely: a v1.11-style 11-loop run that discarded 11 distinct modules
might now be 11 loops covering ~4 modules (each with 2-3 attempts). The
final module count is lower but the verdict for each tested module is
much more reliable.

To revert to v1.12.1 behaviour:

```yaml
autoresearch:
  tuning:
    enabled: false
```

This bypasses all v1.13 logic and falls back to single-attempt verdicts.

## Operational expectations

For real runs at TIME_BUDGET=7200s (2 hours, your current setting):

- Loop 0: vanilla baseline (no tuning state)
- Loop 1: pick CBAM (priority A token), attempt 1, default hp from paper
  - If keep → Loop 2 picks new module
  - If discard → status=`tuning`, Loop 2 continues CBAM at attempt 2
- Loop 2 (if Loop 1 discarded): CBAM attempt 2, agent reads Loop 1's
  trajectory shape + paper hp, picks adjusted hp, runs
  - If keep → Loop 3 picks new module
  - If discard, attempt < 3 → status stays `tuning`, Loop 3 continues
  - If discard at attempt 3 with extension threshold not met →
    status=`discarded`, reason=`tuning_failed`, Loop 3 picks new module
- Continues until `max_no_improvement_loops` triggers stop

Expected first-run behavior with v1.13: 5-7 loops cover 2-3 modules
deeply. If those modules really don't fit (architectural mismatch),
they reach `tuning_failed` discard quickly and dispatch moves on.

## v1.14 outlook

Outstanding from prior CHANGELOG (deferred):

1. **paper-finder yaml_inject `after_class:` translation** (v1.11 self-corrected,
   not yet codified — see Problem E)
2. **SPD-Conv weight reshaping** (v1.11.1 deferred — needs REPLACE
   semantics + new helper, ~300 lines)
3. **Generic-task abstraction** (currently OD/tracking only — would let
   pipeline handle classification, segmentation, NLP)

None have implementation pressure. v1.13 is the major release; v1.14+
informed by what v1.13 real-run feedback exposes.
