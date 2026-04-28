# Changelog — v1.12.1

Release type: **Patch — wire up dead-config + fix stale yaml comment.**
Two small fixes addressing dead-config debt accumulated since v1.7.

## Background

User noticed two yaml fields that looked operational but weren't:

1. **`orchestrator.optional_pretrain_trigger`** — present in yaml since
   v1.7, with documenting prose added to research-orchestrator/SKILL.md
   in v1.8. Three sub-fields (`enabled`, `min_autoresearch_loops`,
   `stalled_loops_required`) explicitly described as the trigger for
   mid-run pretrain when autoresearch stalls. **No SKILL code ever
   evaluated these fields.** The user's v1.11 production run hit 11
   consecutive discards with no trigger firing — confirming the dead
   config experimentally.

2. **`error_policy.autoresearch_crash_pause_after`** — comment in
   detection yaml still described v1.7.6 - v1.11.1 behaviour
   ("halve BATCH_SIZE in train.py, revert the last commit") that was
   replaced in v1.12 with log-only behaviour (BATCH_SIZE is locked
   under v1.12, can't be halved). The yaml comment was missed during
   v1.12 sweep.

This patch wires up the trigger and corrects the comment.

## What v1.12.1 fixes

### Fix #1 — `optional_pretrain_trigger` actually evaluates now

`autoresearch/SKILL.md § Step 8 (Log)` adds an evaluation block after
the existing `no_improvement_loops` update. Trigger condition (all four
must hold):

1. `orchestrator.optional_pretrain_trigger.enabled == true` (yaml)
2. `state.pretrain_offer_declined != true` (no recent decline)
3. `state.loop_count >= min_autoresearch_loops` (don't trigger early)
4. `state.no_improvement_loops >= stalled_loops_required` (genuine stall)

When fired:

| state.pretrain_done | Behaviour |
|---|---|
| `false` (dataset-hunter never produced corpus) | Log one-time warning, set `pretrain_offer_declined: true` to self-disable, increment `pretrain_trigger_fired_count`. Loop continues; trigger does NOT fire again this run. |
| `true` (corpus exists) | Set `request_repretrain: true` + `repretrain_reason`. Reset `no_improvement_loops` to 0. orchestrator Step 6.5 picks up next iteration and runs dataset-hunter Phase 5+6 to re-pretrain. |

This **doesn't change behaviour for the user's current setup** (where
`dataset_hunter.enabled: false` means corpus never exists). With
v1.12.1's default `enabled: false`, the trigger stays dormant; with
`enabled: true`, it warns once and self-disables. Either way, no
spurious behaviour change.

### Fix #2 — yaml default `enabled: false`

v1.7 - v1.12 yaml example had `enabled: true` (because the trigger
evaluation didn't actually exist, so setting `true` was harmless). Now
that the evaluation is real, the default flips to `false` to preserve
the behaviour users currently observe (no triggering). Users who want
the trigger explicitly opt in.

This is a no-op for current users (their pipelines weren't being
triggered before, and won't be now).

### Fix #3 — `autoresearch_crash_pause_after` comment corrected

```yaml
# Before (incorrect):
autoresearch_crash_pause_after: 3  # After N consecutive crashes, halve BATCH_SIZE in train.py
                                   # (and track.py if present), revert the last commit, reset
                                   # the crash counter, log to discoveries.md, and continue
                                   # looping. Does NOT pause for user input. Set to a very large
                                   # number to effectively disable. (A3/C8 — now actually wired up.)

# After (matches v1.12 actual behaviour):
autoresearch_crash_pause_after: 3  # [v1.12+] After N consecutive crashes, log to discoveries.md
                                   # and reset the crash counter. v1.12 NO LONGER halves BATCH_SIZE
                                   # or reverts commits — BATCH_SIZE is locked. User intervenes manually
                                   # if 3+ consecutive crashes happen. Set to a very large number to
                                   # effectively disable. (A3/C8.)
```

## Files changed

| File | Change |
|---|---|
| `autoresearch/SKILL.md § Step 8` | New evaluation block ~50 lines: reads yaml `optional_pretrain_trigger`, checks four-gate condition, branches on `state.pretrain_done`. Sets `request_repretrain: true` (corpus available) or warns + self-disables (no corpus). |
| `research-orchestrator/SKILL.md § state init` | Adds two new state fields: `pretrain_trigger_fired_count` (informative), `pretrain_trigger_no_corpus_warned` (one-time latch). |
| `shared/state_migrate.py` | Two new keys with safe defaults (`0`, `False`). |
| `examples/research_config.visdrone-detection.yaml` | `optional_pretrain_trigger.enabled` flipped `true` → `false`. Block comments updated to reference v1.12.1 implementation status. `autoresearch_crash_pause_after` comment corrected to v1.12 log-only behaviour. |
| `examples/research_config.visdrone-mot.yaml` | Same yaml changes, abbreviated. |
| `CHANGELOG_v1.12.1.md` | New (this file). |
| `README.md` | Banner + Versions list. |

### Unchanged

`shared/modules_md.py`, `shared/invariants.py`, `shared/hook_utils.py`,
`shared/weight_transfer.py`, `shared/templates/`, `paper-finder/SKILL.md`,
`dataset-hunter/SKILL.md`. Pure SKILL+yaml change.

## Test coverage

| Suite | v1.12 | v1.12.1 |
|---|---|---|
| `shared/test_modules_md.py` | 34 | 34 |
| `shared/test_templates.py` | 3 | 3 |
| `shared/test_weight_transfer.py` | 61 | 61 |
| `shared/test_hook_utils.py` | 23 | 23 |
| `shared/test_invariants.py` | 40 | 40 |
| **Total python tests** | **161** | **161** |
| SKILL.md python snippets | 103 | 103 |
| YAML examples | 2 | 2 |

No new tests because v1.12.1's logic lives in `autoresearch/SKILL.md`
prose (executed by the agent at runtime, not by deterministic Python
modules). The `shared/` modules unchanged → existing tests cover what
they cover.

The one place a unit test would fit is the trigger condition
evaluation itself (the four-gate AND check), but it's literally three
`state.get(...)` reads + comparisons in SKILL prose — testing it
would require a mock orchestrator runtime, which we don't have
infrastructure for. The existing test_invariants.py + test_modules_md.py
catch the failure modes that matter (state schema integrity, locked
variables drift, etc.).

## Upgrade path

**Drop-in**. State migration adds 2 new keys with safe defaults. yaml
default flip from `enabled: true` → `enabled: false` is a no-op for
current users (the `true` value was being silently ignored).

For users who WANT mid-run pretrain triggering:

```yaml
orchestrator:
  optional_pretrain_trigger:
    enabled: true                 # opt-in
    min_autoresearch_loops: 5
    stalled_loops_required: 5
```

This requires `dataset_hunter.enabled: true` and a successful corpus
build in Stage 2 (`state.pretrain_done == true`). Without those, the
trigger fires once, warns, and self-disables.

## Operational expectations

For your current pipeline (dataset_hunter disabled): the v1.12.1
upgrade is invisible. Loops 1-N with all-discard outcomes will NOT
trigger the pretrain reset, same as v1.11/v1.12 behaviour.

If you want to BE NOTIFIED when stalls happen (without acting on
them), the existing `orchestrator.stopping.max_no_improvement_loops`
yaml field is the right tool. v1.12.1 doesn't change that path.

If you eventually enable `dataset_hunter`:
1. Set `dataset_hunter.enabled: true` and configure pretrain settings
2. Set `orchestrator.optional_pretrain_trigger.enabled: true`
3. The trigger fires when `min_autoresearch_loops` AND
   `stalled_loops_required` both reach their thresholds
4. orchestrator Step 6.5 runs dataset-hunter Phase 5+6 (re-pretrain
   on existing corpus); train.py WEIGHTS gets patched to new
   checkpoint; results.tsv gets a `rebase` row marker

## v1.13 outlook

Unchanged from v1.12. Three candidates standing:

1. yaml_inject random-init convergence (problem A from session 2026-04-27)
2. paper-finder yaml_inject `after_class:` translation (problem E)
3. SPD-Conv weight reshaping (deferred from v1.11.1)

v1.12.1 is purely a config-debt clearout. Scope minimal.
