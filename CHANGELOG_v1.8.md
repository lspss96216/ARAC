# Changelog — v1.8

Release type: **Production-usability + agent contract enforcement.**
Combines real-world bug fixes from a 31-loop production run, a new
runtime invariants module, and several pieces of design polish that were
deferred from earlier v1.7.x releases. Drop-in on top of v1.7.7. No
schema changes that break existing pipelines.

This is the first release that **does not** add any new architectural
capability — `full_yaml` mode and `apply_yaml_spec` remain reserved for
v1.9. v1.8's job is to make v1.7.7's existing capabilities reliable in
agent-hostile environments and provide observable contract enforcement.

## Background — what motivated this release

A 42-hour, 31-loop production run on VisDrone2019 with YOLO26X
(documented in user's `thinking_log.md` + `discoveries.md`) achieved
val_mAP50_95 0.2847 → 0.2971 (+4.4% relative). Along the way it
surfaced:

- **Three real bugs** in `weight_transfer.py` and `paper-finder/SKILL.md`
  that the user patched locally — never made it back to the codebase
- **One implicit-but-important step** (vanilla baseline as Loop 0) that
  the agent decided to add by judgment call rather than because SKILL
  said so
- **Two semantic ambiguities** (description format, pretrain trigger
  config) that produced different verdicts between runs
- **Zero autonomous stop triggers** beyond the manual sentinel — the
  user had to Ctrl+C to terminate even after convergence was clear

Plus several deferred items from v1.7.x reviews that finally have a
home: `discoveries.md` categories guide, per-loop log archive,
results.tsv mental model doc, stall-with-blocked semantic, and a runtime
invariants module that catches agent contract violations.

## Sections of this release

| Section | What it covers |
|---|---|
| **B. Real-run regression fixes** | B1-B5 — bugs / missing steps surfaced by the production run |
| **C. Pipeline polish** | C1-C6 — autonomous stop, semantic clarifications, log archive |
| **D. Agent contract enforcement** | D1 — new `invariants.py` module |
| **E. User-facing docs** | E2-E3 — results.tsv guide, blocked-vs-pending text |

(Numbering preserves the v1.8 working scope from `v1.8 statement` —
A and C4 deferred to v1.9.)

## B. Real-run regression fixes

### B1 — `compute_layer_map` off-by-one

**Severity**: silent weight-transfer corruption.

**Symptom**: an insertion at base position `p` was incorrectly counted
as shifting the original layer at orig_idx `p` itself. The user's
LazySimAM injection saw pretrained C3k2 weights routed into the inserted
LazySimAM wrapper instead of the preserved original C3k2 — symptomless
during training (no crash) but the experiment was running with a
half-broken weight initialisation.

**Why it was latent through v1.7.7**: existing test
`test_layer_map_with_insertions` expected the buggy formula. v1.7.7's
`update_head_refs` (Fix #13) had the correct `p < ref` semantics by
coincidence — the two functions used inconsistent semantics for the
same conceptual question.

**Fix**: `weight_transfer.compute_layer_map` changed `p <= orig_idx` →
`p < orig_idx`. Existing test's expected values updated to match
correct semantics, plus new test
`test_layer_map_insertion_at_p_does_not_shift_orig_p` directly verifies
the fixed property.

### B2 — yaml output filename scale suffix

**Severity**: yaml_inject silently builds wrong-scale model.

**Symptom**: `weight_transfer` wrote `exp_arch.yaml` regardless of base
model. ultralytics' `guess_model_scale` reads scale from filename via
regex (looking for `n`, `s`, `m`, `l`, `x` near the start), and
defaulted to `n` when no match. Subsequent `YOLO("exp_arch.yaml")`
built a YOLO_N variant; weight transfer then mostly failed because
YOLO_N has different layer counts than YOLO_X (the actual base).

**Why it was latent**: tests for `generate_custom_yaml` checked the
content of the YAML, not the filename. tests for `transfer_weights`
constructed the model directly without going through `YOLO(yaml_path)`.

**Fix**: `build_custom_model_with_injection` derives filename from base
weights stem: `weights/yolo26x.pt` → `yolo26x_arch.yaml`. ultralytics'
filename regex now matches `x` and builds the right scale.

### B3 — paper-finder base-aware `after_class`

**Severity**: yaml_inject can't find layers, returns "no matches"
errors or inserts at wrong positions.

**Symptom**: paper-finder's prompts default to YOLOv8-era class names
(`C2f`). YOLO26X, YOLO11, and YOLOv9 use different backbone block
classes. modules.md entries with `after_class: C2f` raise `no matches
in scope` when run against a YOLO26X base — `C2f` simply doesn't appear
in YOLO26X's YAML.

**Why it was latent**: paper-finder didn't read `base_model.md`'s actual
class names; it wrote `after_class` based on prompt patterns alone.

**Fix**: `paper-finder/SKILL.md` Phase 5 yaml_inject Integration notes
adds a 6-family table (YOLOv5 = `C3`, YOLOv8/v10 = `C2f`, YOLO11/26 =
`C3k2`, YOLOv9 = `RepNCSPELAN4`) and helper code that reads
`base_model.md` to detect the family. Documented preference: if unsure,
use `at_index` (unambiguous) over `after_class` (family-dependent).

### B4 — train.py uses absolute path for `project=`

**Severity**: ultralytics save_dir resolves to wrong location across
some env configurations.

**Symptom** (from real run's discoveries.md Loop 1): v1.7.5's
environment shield (`SETTINGS.update(runs_dir=...)`) updates
`settings.json` on disk, but ultralytics' Trainer captures `runs_dir`
at module-import time. Result: actual checkpoints landed at
`/<project>/runs/detect/runs/train/...` instead of expected
`/<project>/runs/train/...` — discoverable only by manual `find`.

**Fix**: `train.py.detection` and `train.py.tracking` templates pass
`project=str(CKPT_DIR.parent.resolve())` to `model.train()` — absolute
path overrides any settings.json / env-var ambiguity. v1.7.5's
SETTINGS shield kept as second line of defence (it now matters less
because `project=` wins).

### B5 — Loop 0 vanilla baseline as explicit step

**Severity**: comparison baseline can be a sub-optimal first module.

**Symptom**: `autoresearch/SKILL.md` Step 7 rule 1 says "first
iteration → keep unconditionally" — but the first iteration was
whatever module sits at `pending[0]`. If that module slightly
underperforms a clean fine-tune, every later keep/discard decision
is anchored against an already-bad baseline.

**Why it was latent**: the user's agent in the real run noticed this
on judgment call (`thinking_log.md` 17:14) and added a vanilla
baseline as iteration 1. The next user might not.

**Fix**: New explicit "Loop 0 — Vanilla baseline" subsection in
`autoresearch/SKILL.md` before "## The Loop". Mandatory unless
`state.vanilla_baseline_done == True`. Runs train.py with all USE_*
flags False, ARCH_INJECTION_ENABLED False, writes row at loop=0 with
description `vanilla baseline`, marks `state.vanilla_baseline_done =
True`, then enters the iterative loop at Loop 1.

## C. Pipeline polish

### C1 — Autonomous stop triggers

**Motivation**: real run had to be Ctrl+C'd at Loop 31 because the
only stop trigger was the sentinel flag file. For unattended runs the
loop runs forever.

**Fix**: orchestrator Stage 4's stop check now has six triggers:

| # | Trigger | Old in v1.7.x | New in v1.8 |
|---|---|---|---|
| 1 | sentinel flag file | ✓ | unchanged |
| 2 | bounded `iterations` count | ✓ | unchanged |
| 3 | paper-finder expansions exhausted | ✓ | unchanged |
| 4 | N consecutive no-improvement loops | — | **new** |
| 5 | hard cap on total iterations | — | **new** |
| 6 | wallclock cap | — | **new** |

All three new triggers are optional yaml fields (`null` = no limit):

```yaml
orchestrator:
  stopping:
    max_no_improvement_loops: 10   # convergence detection
    max_total_loops: 100            # hard ceiling
    max_wallclock_hours: 72         # batch-job deadline
```

`no_improvement_loops` counter maintained in autoresearch Step 9 (reset
to 0 on keep, +1 on discard).

### C2 — `pretrain_offer_declined` / `optional_pretrain_trigger` semantics

**Motivation**: real run's agent set `pretrain_offer_declined: true`
defensively (to prevent mid-loop trigger), then user wondered why the
yaml-configured `optional_pretrain_trigger.enabled: true` never fired.
Fields' interaction was undocumented.

**Fix**: `research-orchestrator/SKILL.md` Step 7 prefixed with
"v1.8 — Semantic clarification" subsection that defines:
- `pretrain_offer_declined`: runtime state, sticky, "skip auto-trigger
  next time"
- `optional_pretrain_trigger.enabled`: yaml config, immutable, "is the
  trigger enabled at all"
- Trigger fires iff: `enabled == true` AND `declined == false` AND
  stall counter at threshold

Also adds one-time runtime warning when both are set inconsistently
(triggered=true with declined=true → warning that auto-trigger is dead
config).

Stage 2's two skip paths now distinguish:
- `dataset_hunter.enabled: false` → sets `declined: true` (intentional)
- `dataset_hunter.pretrain.time_budget_sec: 0` → leaves `declined:
  false` (skip-but-trigger-allowed)

### C3 — `description` format contract

**Motivation**: real run's results.tsv had descriptions like
`"copy_paste/mixup tweaks"` that lost which exact values were tested.
Reconstructing the cumulative path required reading git log.

**Fix**: `autoresearch/SKILL.md` Step 8 prefixed with "v1.8 — Description
format contract" specifying 5 patterns (vanilla / hook / yaml_inject /
hyperparameter / combination) plus the `[tiebreak]` suffix from v1.7.7's
Step 7 rule 4. Includes example formats for each kind.

### C5 — `discoveries.md` categories guide

**Motivation**: SKILL listed 4 categories
(`observation/limitation/strategy_shift/bug_workaround`) without
guidance on when to use each. Real run's `discoveries.md` used all 4
appropriately, but partly by happenstance.

**Fix**: § Discoveries adds a 7-category table with usage guidance.
v1.8 adds three new categories: `agent_violation` (auto-logged by D1's
invariants check), `misclassification` (paper-finder mode wrong),
`resource_constraint` (VRAM / compute limited what could run). The
canonical `log_discovery` helper's VALID set updated.

### C6 — Per-loop log archive

**Motivation**: real run had only `run.log` overwritten each loop, so
debugging a Loop 8 issue at Loop 30 required `git log --grep` and
checkout dance. No way to compare two iterations' run logs side by side.

**Fix**: After Step 5 completes, `cp run.log logs/loop_<N>.log`. Archive
keeps indefinitely (small, ~100 KB per archive). Documented in Step 5.
`results.tsv` does NOT add a `log_tail` column — full archive is the
canonical inspection target via `tail -50 logs/loop_N.log`.

## D. Agent contract enforcement

### D1 — `shared/invariants.py`

**Motivation**: v1.7.x relied on SKILL prose to keep agents from changing
locked variables (`TIME_BUDGET`, `SEED`, `IMGSZ`) or setting `OPTIMIZER
= "auto"`. The real run's agent followed all rules, but the surface for
violation was wide and entirely by-convention.

**Fix**: New module `shared/invariants.py` (~200 lines) with 23 unit
tests. Three checks:

| Check | What it catches |
|---|---|
| `check_locked_variables` | TIME_BUDGET / SEED / IMGSZ in train.py vs canonical state values |
| `check_optimizer_not_auto` | OPTIMIZER set to 'auto' (case-insensitive) |
| `check_section_markers_present` | Section ① / ② / ③ / ④ markers (ASCII or Unicode) deleted |

Aggregator `run_all_checks(state)` returns a list of `Violation`
objects. `format_violations(...)` pretty-prints. `ContractViolation`
exception lets autoresearch's Step 4 raise on violation.

`autoresearch/SKILL.md` Step 4 Commit prefixed with "v1.8 — Pre-commit
invariant check" subsection. On violation:
1. Log to `discoveries.md` with category `agent_violation` (D1's new
   category)
2. Increment `consecutive_crashes` (3 in a row → crash-pause halves
   BATCH_SIZE)
3. `git checkout -- train.py track.py` to revert working tree
4. Raise `ContractViolation`, skip rest of iteration, return to Step 1

This makes contract violations into an observable signal in
discoveries.md instead of silent corruption of the experimental record.

## E. User-facing docs

### E2 — `results-tsv-guide.md`

**Motivation**: Final summary in real run's discoveries.md required the
user to read TSV by hand and reconstruct cumulative paths. No reference
doc existed for "how do I read this file".

**Fix**: New `shared/results-tsv-guide.md` (~180 lines). Aimed at users,
not agents. Covers:
- Why TSV vs CSV / JSON
- Per-column semantics (what `commit` empty means, how `0.0000` differs
  from a real zero, etc.)
- Common analysis patterns with `awk` / Python snippets
- Spotting OOM-fallback runs (low memory_gb + low it/s in archived log)
- Cross-references to spec docs

### E3 — Stall-with-blocked semantics

**Motivation**: real run's Loop 8 saw 4 modules marked
`Status: blocked` (needing full_yaml). `count_pending` correctly returned
0 and stall fired, but the relationship between "pending" and "blocked"
was implicit.

**Fix**: `autoresearch/SKILL.md` Step 2 Priority A adds explicit
"v1.8 — `pending > 0 but all blocked` semantics" subsection:
- `find_pending` returns only `Status: pending` (not blocked)
- `count_pending` likewise — blocked entries don't count for stall
- Agents marking modules `blocked` MUST also write
  `discoveries.md` entry explaining why (so post-run review sees the
  reason, not just the dead end)

## Files changed

### New files

| File | Purpose |
|---|---|
| `shared/invariants.py` | Runtime contract checks (~200 lines) |
| `shared/test_invariants.py` | 23 tests, no torch dependency |
| `shared/results-tsv-guide.md` | User-facing TSV reading guide |
| `CHANGELOG_v1.8.md` | This file |

### Modified

| File | Changes |
|---|---|
| `shared/weight_transfer.py` | B1 (compute_layer_map off-by-one) + B2 (filename scale suffix) |
| `shared/test_weight_transfer.py` | Updated test_layer_map_with_insertions; added test_layer_map_insertion_at_p_does_not_shift_orig_p (+1 test) |
| `shared/state_migrate.py` | +3 v1.8 keys (vanilla_baseline_done, no_improvement_loops, pretrain_dead_config_warned) |
| `shared/templates/train.py.detection` | B4: project= uses .resolve() |
| `shared/templates/train.py.tracking` | B4: same |
| `paper-finder/SKILL.md` | B3: base-aware after_class table + helper code |
| `autoresearch/SKILL.md` | B5 (Loop 0 step), C3 (description format), C5 (categories), C6 (per-loop archive), D1 (invariants check in Step 4), E3 (blocked semantics), Step 9 maintains no_improvement_loops |
| `research-orchestrator/SKILL.md` | C1 (3 new stop triggers), C2 (pretrain semantics + dead-config warning), state init adds 3 v1.8 keys |
| `examples/research_config.visdrone-detection.yaml` | C1: 3 stop trigger fields (commented) |
| `examples/research_config.visdrone-mot.yaml` | Same |
| `README.md` | v1.8 banner, Versions list, layout tree |

### Unchanged

`paper-finder/SKILL.md` (only B3 added, no other changes), `dataset-hunter/SKILL.md`
(zero changes), `shared/hook_utils.py`, `shared/modules_md.py`, `shared/parse_metrics.py`,
`shared/file-contracts.md`, `shared/templates/track.py.tracking`, `shared/templates/arch_spec.schema.json`,
`shared/test_modules_md.py`, `shared/test_hook_utils.py`, all earlier CHANGELOGs.

## Test coverage

| Suite | v1.7.7 | v1.8 |
|---|---|---|
| `shared/test_modules_md.py` | 18 | 18 |
| `shared/test_templates.py` | 3 | 3 |
| `shared/test_weight_transfer.py` | 48 | **49** (+1) |
| `shared/test_hook_utils.py` | 13 | 13 |
| `shared/test_invariants.py` | 0 | **23** (NEW) |
| **Total python tests** | **82** | **106** (+24) |
| SKILL.md python snippets | 89 | **93** (+4) |
| YAML examples | 2 | 2 |

The 23 new invariants tests cover all three checks plus aggregator,
pretty-printing, and exception behaviour. No torch dependency — runs
in the same fast suite as the rest of `shared/`.

## Upgrade path

Drop-in. State migration covers the 3 new v1.8 keys
(`vanilla_baseline_done`, `no_improvement_loops`,
`pretrain_dead_config_warned`) — pre-v1.8 state files resume with these
defaulted to safe values.

`research_config.yaml` files from earlier versions work unchanged. The
3 new stop trigger fields are optional and `null` = no limit. The
existing `iterations: null` + `stop_pipeline.flag` workflow remains
fully supported.

For users with v1.7.x in-progress runs:
- Resume continues without Loop 0 vanilla (existing baseline-via-iter-1
  remains the comparison anchor; flagged in state via the existing keep
  row)
- Invariants check fires on the next commit; if your existing train.py
  was edited to use `OPTIMIZER='auto'` or change a locked variable, the
  next commit will fail. Restore the canonical values.

For new pipelines: Loop 0 vanilla baseline runs automatically as part of
autoresearch entry; no user action needed.

## Operational tip

After upgrade, run a single iteration and confirm:
- `logs/loop_0.log` exists (per-loop archive working)
- `results.tsv` Loop 0 row has description `vanilla baseline`
- `pipeline_state.json` has `vanilla_baseline_done: true` after Loop 0
  completes
- If you intentionally edit train.py to test an experiment, Step 4
  invariants check passes (no locked variables changed)

If invariants check fires falsely (e.g. flagging a value you genuinely
need to change), the violation is logged to `discoveries.md` and the
iteration is reverted — there's no quiet-mode for this. The right path
is: stop the pipeline, edit `research_config.yaml` to update the canonical
value, rerun Stage 0 (it overwrites `pipeline_state` from yaml), resume.

## v1.9 outlook

v1.8 closes the production-usability gap; v1.9 returns to feature work:

- **A1-A5**: full_yaml mode (apply_yaml_spec, autoresearch dispatch
  full_yaml branch, paper-finder full_yaml heuristic, modules.md
  full_yaml schema, train-script-spec.md full_yaml chapter)
- **C4**: modules.md `resource_impact` field + autoresearch automatic
  BATCH_SIZE halve when `vram_4x` flagged

These two together unlock the 5 modules from real run that were marked
`blocked` (BiFPN, P2 head, TPH, QueryDet, SPD-Conv) — currently an
upper bound on what an unmodified pipeline can test.
