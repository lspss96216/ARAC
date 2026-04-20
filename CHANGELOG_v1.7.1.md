# Changelog — v1.7.1

Release type: **Bug fixes + adaptive repair feature.** Patch-level release on
top of v1.7. No schema changes, no breaking behaviour. Existing v1.6 / v1.7
state files, modules.md, and commits resume transparently.

## The big addition — Repair loop (Step 5.5)

v1.6 and v1.7 had a binary stance on crashes: Step 5 runs, it either
succeeds or fails, Step 7 keeps or discards. But in practice, many crashes
are **glue-code bugs** — a missing `register_custom_modules` entry, a
`TypeError` because `yaml_args` doesn't match the class signature, a shape
mismatch because the inserted module's channel count doesn't line up with
its insertion point. Discarding for these is wasteful: the paper's idea
wasn't tested at all, the glue code was wrong.

v1.7.1 introduces **Step 5.5 — Repair check** between Step 5 (Run) and Step
6 (Verify). When `train.py` crashes, autoresearch:

1. **Classifies** the crash by stderr pattern (Tier 1 code bug / Tier 2
   shape mismatch / OOM / unfixable / unknown).
2. **Tries to repair** Tier 1 (edit custom_modules.py, fix imports,
   adjust `yaml_args`) and Tier 2 (synthesize 1×1 Conv adapters around the
   inserted module by probing runtime shapes).
3. **Short-tests** with a 120s budget — success = first-epoch loss line
   appears in `run.log` and all three losses are finite positive.
4. **Re-runs** with full `TIME_BUDGET` when the short test passes,
   producing the real experiment; at most 3 repair attempts total before
   giving up.

This turns "loop wasted" into "architecture adapted and tested", with the
`description` column in `results.tsv` marked `[adapted: ...]` so a human
reviewer can tell apart paper-faithful from adapted experiments.

## New primitives in `weight_transfer.py`

Added at the bottom of the file (~350 lines total), fully testable without
torch / ultralytics:

| Name | Purpose |
|---|---|
| `classify_crash(stderr_tail)` | Regex-match stderr against known crash patterns; returns one of 10 categories |
| `loss_first_value_is_valid(run_log)` | Short-test success check: scan for first-epoch loss line, require finite positive box/cls/dfl |
| `ShapeInfo` / `ShapeInfo.from_tensor_shape` | Typed shape at a point in the model |
| `get_shape_at_index(model, idx, imgsz)` | Runtime probe via PyTorch forward-hook. No dependence on ultralytics internal attrs |
| `probe_module_io(class_name, in_shape, yaml_args, kwargs)` | Instantiate module, dummy-forward, return (in, out) shapes |
| `AdapterPlan`, `plan_adapter(up, mod_in, mod_out, down)` | Decide what adapters are needed given 4 shape observations |
| `_make_1x1_conv_line(out_channels)` | Synthesize ultralytics `Conv` YAML line for adapter |
| `extend_spec_with_adapters(spec, plan, idx)` | Layer pre/post adapter insertions onto an existing `ARCH_INJECTION_SPEC` |

Only **channel mismatches** are auto-repaired. Spatial-dimension changes
(H×W mismatches) are flagged as un-adaptable — they imply the module
fundamentally changes resolution, which is Tier 3+ territory and requires
a different insertion plan, not an adapter.

## New tests

`test_weight_transfer.py`: 19 → **36 tests** (17 new). Covers:

- Crash classification (Tier 1, Tier 2, OOM, unfixable, unknown)
- Loss validity check (valid first epoch, NaN, Inf, no line at all)
- Adapter planning (no-op, pre-only, post-only, both, spatial refused)
- Spec extension ordering (pre, target, post)
- `ShapeInfo` rejection of non-4D shapes

Total test count across all shared modules: **51** (was 34).

## Q2 patch — Hook-mode silent-failure warning

`autoresearch/SKILL.md § Dispatch on Integration mode` now has a
"Hook-mode scope limit (v1.7+)" preamble warning that:

- Hook mode can't insert a parameterised layer between existing layers
- Doing so causes a silent failure — training completes, metrics match
  baseline, whole TIME_BUDGET wasted — with no crash
- This is the exact failure mode Loop-36 CBAM hit in v1.6
- v1.7's `yaml_inject` is the fix; v1.7.1 adds the detection (Step 5.5f
  logs to `discoveries.md` on shape-mismatch crash in hook mode)

## Q2 reinforcement — paper-finder asymmetric-cost warning

`paper-finder/SKILL.md § Integration mode` now explains that
misclassification is NOT symmetric:

- Hook-when-should-be-yaml_inject = silent failure (worst case)
- yaml_inject-when-should-be-hook = loud crash that autoresearch handles

Previous wording "when unsure, default to hook" is replaced with "when
unsure, tag `yaml_inject`". Loud failures are cheaper.

## Q3 patch — Misleading stall_count history removed

`autoresearch/SKILL.md § Update counters` previously had a "Previous
versions had autoresearch reset stall_count..." paragraph that was
helpful in v1.6 (explaining the v1.5→v1.6 change) but by v1.7 became
stale history that actively misled readers (Claude Code's review flagged
it). Removed, and the replacement paragraph explicitly states that Step
2 Priority A picks pending modules automatically — no force-reset
mechanism is needed.

## Files changed in v1.7.1

| File | Change |
|---|---|
| `shared/weight_transfer.py` | +~350 lines of repair primitives |
| `shared/test_weight_transfer.py` | +17 tests for repair primitives |
| `autoresearch/SKILL.md` | New § Step 5.5 (7 subsections: 5.5a–5.5g). Q2 preamble in § Dispatch. Q3 cleanup in § Update counters. |
| `paper-finder/SKILL.md` | Asymmetric-cost warning replaces the old "default to hook" line |
| `CHANGELOG_v1.7.1.md` | New (this file) |
| `README.md` | Version bumped; repair loop summarised |

## Files NOT changed

Byte-identical to v1.7 (and, except for the ones v1.7 touched, to v1.6):

- `shared/modules_md.py` / `test_modules_md.py`
- `shared/state_migrate.py` / `parse_metrics.py` / `file-contracts.md` /
  `train-script-spec.md` / `test_templates.py`
- `shared/templates/*` (train.py.detection, train.py.tracking,
  track.py.tracking, arch_spec.schema.json)
- `research-orchestrator/SKILL.md`
- `dataset-hunter/SKILL.md`
- Both `examples/*.yaml`

## Known limitations (v1.7.1 scope boundaries)

- **Tier 2 only handles channel mismatches.** Spatial (H×W) mismatches
  abort the plan — repair can't adapt them. This is deliberate:
  auto-inserting AdaptiveAvgPool / Upsample changes the model's
  resolution in unpredictable ways, and the fix is usually "insert the
  module somewhere else" which belongs in paper-finder's spec, not in
  runtime adaptation.
- **Tier 3 (downsize module config — smaller channels, smaller kernels)
  NOT implemented.** Deferred. OOM still goes through the v1.6 path
  (halve BATCH_SIZE).
- **Tier 4 (swap the module entirely) NOT implemented.** This is a
  different experiment, not a repair.
- **Repair doesn't count toward re-pretrain trigger.** Kept adapted
  experiments contribute to `architecture_keeps` counter, but the
  adaptation itself is not a distinct architectural change.
- **Adapted experiments look paper-faithful in git log** (fix commits
  are cleaned up as unique commits; `description` shows `[adapted: ...]`
  but git log messages just say "exp: CBAM yaml_inject"). Human review
  should read `results.tsv`, not just `git log`.

## Install

```bash
unzip pipeline_v1.7.1.zip
cp -r pipeline_v1.7.1/skills/* ~/.claude/skills/
cp pipeline_v1.7.1/examples/* ~/.claude/skills/examples/

# Verify
cd ~/.claude/skills/shared
python3 test_modules_md.py       # 12/12
python3 test_templates.py        # 3/3
python3 test_weight_transfer.py  # 36/36
```

Drop-in on top of v1.7 — no state migration needed.
