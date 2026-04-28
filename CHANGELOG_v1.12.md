# Changelog — v1.12

Release type: **Minor — strict fair-comparison mode.** Three fixes
addressing experimental-fairness gaps surfaced during a v1.11
production run on YOLO26x + VisDrone @ IMGSZ=640 (11 loops over ~12
hours, session 2026-04-27).

**BREAKING change**: `BATCH_SIZE` is now LOCKED across all iterations
(same enforcement as IMGSZ / SEED / TIME_BUDGET). Auto-halve paths
removed from autoresearch. yaml field renamed `initial_batch_size` →
`batch_size` (alias kept for backward compat).

## Background

The v1.11 production run produced an 11-loop campaign against YOLO26x
(baseline `mAP50_95 = 0.2850`). Key problems surfaced in the
discoveries log:

1. **Loop 1 — FlexSimAM tagged `resource_impact: none` but scope=all
   triggered CPU TaskAlignedAssigner fallback.** The module had zero
   parameters but inserted at every C3k2 block in the model;
   activation memory blew through 81.5GB on H100. The
   `resource_impact: none` tag was technically correct (no extra
   parameters) but didn't capture the scope multiplier on memory.
2. **Loop 9 — User had to re-establish a BATCH=32 vanilla baseline**
   because half the loops ran at BATCH=64 (baseline) and half at
   BATCH=32 (auto-halved by v1.9 `resource_impact` logic, ultralytics
   OOM auto-reduce, etc.). MPDIoU at BATCH=32 was -0.003 vs BATCH=64
   baseline but +0.002 vs the BATCH=32 baseline — a sign-flipped
   comparison that pushed it across the keep/discard line.
3. **Loop 10 + Loop 11 — ultralytics auto-batch-reduce silently
   trained at a smaller batch than baseline.** MPDIoU at BATCH=64
   OOMed; ultralytics emitted `WARNING ⚠️ CUDA out of memory with
   batch=64. Reducing to batch=32 and retrying.` Run completed at
   BATCH=32 but logged metrics that LOOKED like BATCH=64 results.
   AdamW hit the same path. Both runs DISCARDED. Worse than that:
   they wasted the time-budget slot they consumed.

The common thread: **dynamic mutation of BATCH_SIZE** (autoresearch
auto-halve, ultralytics auto-reduce, crash-pause halve) traded
experimental fairness for coverage. v1.12 inverts the trade-off.

## What v1.12 fixes

### Fix B — `resource_impact` scope-aware escalation

**Problem**: real-world Loop 1 — FlexSimAM with `resource_impact: none`
+ `scope: all` consumed enough VRAM to trigger CPU assigner fallback.
The `none` tag was technically accurate (zero parameters) but didn't
account for activation memory at every insertion point.

**Fix**: new Module property `effective_resource_impact` in
`shared/modules_md.py`. For yaml_inject modules with `scope: all`,
escalates the base tag by one tier:

| Base tag | scope=backbone/neck/head | scope=all |
|---|---|---|
| `none` | `none` | **`vram_2x`** |
| `vram_2x` | `vram_2x` | **`vram_4x`** |
| `vram_4x` | `vram_4x` | `vram_4x` (ceiling) |
| `cpu_fallback_risk` | `cpu_fallback_risk` | `cpu_fallback_risk` (orthogonal) |

Hook mode and full_yaml mode use base resource_impact unchanged
(scope concept doesn't apply). New helper `Module.yaml_inject_scope`
extracts `scope:` from Integration notes regex.

autoresearch's predictive-skip (formerly auto-halve) reads
`effective_resource_impact`, not raw `resource_impact`.

### Fix C+D — BATCH_SIZE locked

**Problem**: BATCH_SIZE was initial-set in v1.9.3-v1.11.1 (yaml
provided starting value, autoresearch could halve dynamically). Real
runs showed three independent paths that mutated BATCH_SIZE
mid-pipeline:

1. **v1.9 resource_impact auto-halve** (Step 3 — vram_4x → ÷4, vram_2x
   → ÷2 before launching experiment)
2. **v1.7.6 crash-pause halve** (after 3 consecutive crashes)
3. **ultralytics built-in OOM handler** (the
   `WARNING ⚠️ CUDA out of memory ... Reducing to batch=` path)

All three made experiments incomparable across iterations, requiring
manual re-baselining (Loop 9 of session 2026-04-27).

**Fix**: BATCH_SIZE now locked. `invariants.LOCKED_VARS` includes
`"batch_size" → "BATCH_SIZE"`. Stage 0 patches the value once from
yaml; subsequent commits with a different BATCH_SIZE raise
ContractViolation.

What was REMOVED:

- v1.9 § "Resource-impact auto-halve" block — replaced with predictive
  skip-and-block. Modules predicted to OOM (effective_resource_impact
  == `vram_4x` or `cpu_fallback_risk`) are skipped before launch, marked
  `blocked` in modules.md, logged to discoveries.md as
  `resource_constraint`.
- v1.7.6 § "Crash-pause halve" — replaced with crash-pause LOG only.
  Counter still resets, but no `git reset --hard HEAD~1`, no
  BATCH_SIZE halve, no commit. User intervenes manually if 3+
  consecutive crashes happen.
- v1.6 OOM repair tier — `oom` → discard + block module instead of
  halve.

What was ADDED:

- New invariant `check_no_ultralytics_auto_batch_reduce`
  (`shared/invariants.py`) scans run.log for ultralytics' specific
  warning string. Match → ContractViolation in Step 6 → discard +
  block module + log as `resource_constraint`.
- New OOM repair tier `auto_batch_reduce` documented in repair table.
- New state field `blocked_modules` (list) tracking modules
  unilaterally skipped or post-run blocked.
- yaml field renamed `autoresearch.loop.initial_batch_size` →
  `autoresearch.loop.batch_size`. Old name kept as backward-compat
  alias (state init reads `batch_size` first, then
  `initial_batch_size`).

### What user does when a module is blocked

The "blocked" status is sticky for this pipeline run. To enable the
module:

1. Lower yaml `autoresearch.loop.batch_size` to a value at which the
   module fits.
2. Re-run from Loop 0 (the new baseline must use the same lowered
   batch — v1.12 enforces this via lock).
3. Previously-blocked modules return to `pending` automatically when
   Loop 0 re-baseline triggers state reset.

This is by design: v1.12 prioritises experimental fairness over
coverage. A module that needs half the baseline batch is comparing
apples to oranges; v1.12 forces the user to pick a configuration that
fits all candidates, or accept that some won't run.

## Files changed

| File | Change |
|---|---|
| `shared/modules_md.py` | New `Module.yaml_inject_scope` + `Module.effective_resource_impact` properties (~80 lines). New `_RESOURCE_ESCALATION` table. Existing `resource_impact` property unchanged for backward compat. |
| `shared/test_modules_md.py` | +11 tests for scope extraction + escalation logic. Total 23 → 34 |
| `shared/invariants.py` | `LOCKED_VARS` adds `batch_size → BATCH_SIZE`. New `check_no_ultralytics_auto_batch_reduce` (~50 lines). |
| `shared/test_invariants.py` | +7 tests (3 BATCH_SIZE lock + 4 auto-batch-reduce detection). Total 33 → 40 |
| `shared/state_migrate.py` | New `batch_size` key with None default. `initial_batch_size` kept as deprecated alias (ensures pre-v1.12 state files still load). |
| `research-orchestrator/SKILL.md` | State init reads `batch_size` (with `initial_batch_size` fallback). Stage 0 Step 3 patches BATCH_SIZE + locks. Locked-variables prose updated. |
| `autoresearch/SKILL.md` | v1.9 resource_impact auto-halve replaced with v1.12 predictive skip. v1.7.6 crash-pause halve replaced with crash-pause log-only. OOM repair table updated. New `auto_batch_reduce` tier. Step 6 calls `check_no_ultralytics_auto_batch_reduce` after freshness check. |
| `examples/research_config.visdrone-detection.yaml` | `batch_size` field with sizing guidance for H100 80GB at IMGSZ=640. `initial_batch_size` marked DEPRECATED. |
| `examples/research_config.visdrone-mot.yaml` | Same, abbreviated. |
| `CHANGELOG_v1.12.md` | New (this file). |
| `README.md` | Banner + Versions list. |

### Unchanged

`shared/hook_utils.py`, `shared/weight_transfer.py`,
`shared/templates/`, `paper-finder/SKILL.md`, `dataset-hunter/SKILL.md`,
`papers2code/SKILL.md`. v1.12 doesn't touch the architecture-injection
or paper-search machinery.

## Test coverage

| Suite | v1.11.1 | v1.12 |
|---|---|---|
| `shared/test_modules_md.py` | 23 | **34** (+11) |
| `shared/test_templates.py` | 3 | 3 |
| `shared/test_weight_transfer.py` | 61 | 61 |
| `shared/test_hook_utils.py` | 23 | 23 |
| `shared/test_invariants.py` | 33 | **40** (+7) |
| **Total python tests** | **143** | **161** (+18) |
| SKILL.md python snippets | 103 | 103 |
| YAML examples | 2 | 2 |

## Upgrade path

**Drop-in for state schema** — pre-v1.12 state files load fine. Old
`initial_batch_size` key is preserved + read as fallback if new
`batch_size` key is None.

**Drop-in for existing yaml** — old `initial_batch_size` value is
honoured. State init reads new key first, falls back to old.
Recommend migrating to `batch_size` at next yaml edit.

**BREAKING for autoresearch behaviour** — runs that relied on
auto-halve will now hit DISCARD + block instead. The pipeline will
still progress (next pending module), but coverage of memory-heavy
modules is reduced. To restore old behaviour temporarily, the user
needs to:
- Edit yaml `batch_size` to the smallest value the heaviest module
  needs (e.g. 16 if any module is `vram_4x`-after-escalation)
- Re-run from Loop 0 (new baseline at smaller batch)
- All experiments now run at the smaller batch — fewer epochs per
  TIME_BUDGET, but every result comparable

This is a deliberate cost. Real-world session 2026-04-27 demonstrated
that the previous "let autoresearch halve dynamically" approach
created sign-flipped keep/discard decisions (Loop 9 / MPDIoU). The
v1.12 trade-off is fewer modules tested but every result trustworthy.

## Operational expectations

After upgrade:
- Modules with `resource_impact: vram_4x` (or `vram_2x` + `scope: all`,
  which now escalates to vram_4x) will be skipped before launch and
  marked blocked. Real-world examples from your modules.md: P2 head,
  any dense-attention scope=all module.
- Runs that would have OOMed and auto-reduced now DISCARD with a
  clear discoveries.md entry. No more silent mid-run batch changes.
- The crash-pause counter still runs but doesn't take corrective
  action. After 3 consecutive crashes you'll see a discoveries.md
  log telling you to investigate manually.

If you want to test a module flagged as `blocked`, edit yaml's
`batch_size` to a smaller value and re-run from Loop 0. There's no
in-pipeline way to override the lock — that's the point.

## v1.13 outlook

Outstanding from real-run feedback (not in v1.12 scope):

1. **yaml_inject random-init convergence** (problem A from session
   2026-04-27). All yaml_inject backbone/neck modules at 1200s budget
   discarded with mAP -0.025 to -0.035 because random-init layers
   couldn't converge in 17 epochs. Possible fix: per-module
   `requires_extended_budget: true` flag, or warm-up freeze schedule
   (freeze pretrained layers for N epochs while new layers converge
   to identity-like state, then unfreeze).
2. **paper-finder yaml_inject `after_class:` translation** (problem E).
   v1.11 paper-finder self-corrected from C2f → C3k2 for YOLO26x; v1.13
   could codify by reading base_model.md's block class explicitly.
3. **SPD-Conv weight reshaping** (deferred from v1.11.1). Requires
   yaml_inject REPLACE semantics + `weight_transfer.reshape_for_spd_conv`
   helper.

None of these have implementation pressure yet. v1.12 ships first,
real-run feedback informs v1.13+ scope.
