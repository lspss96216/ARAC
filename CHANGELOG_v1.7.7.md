# Changelog — v1.7.7

Release type: **Production-readiness for hook + yaml_inject.** Six fixes
that, together, make the v1.7 architecture-injection feature actually
work end-to-end. No schema changes. Drop-in on top of v1.7.6.

## Background — what was broken

v1.7 introduced two architectural-injection paths:
- **hook mode** — `inject_modules()` adds forward hooks
- **yaml_inject mode** — `weight_transfer.py` builds a custom YAML
  with new layers and transfers pretrained weights

Both modes had test suites that passed. Both modes shipped through
v1.7.1 → v1.7.6. **Neither mode actually worked end-to-end.** Real
runs (Loops 4-21 documented in `skills_integration_notes.md` Part II)
revealed:

- **yaml_inject** built the model fine, transferred weights fine, then
  crashed on the first forward pass because head's `Concat [-1, 6]`
  references pointed at the wrong layer (#13).
- **hook mode** had hooks attached to the original model, but
  ultralytics' `Trainer.setup_model()` rebuilds the model in a fresh
  instance — all hooks silently lost (#14). Hooks that survived
  rebuild then crashed at checkpoint pickle time because they were
  closures (#15), or at val phase because of AMP/FP32 dtype mismatch
  (#17), or because `layer.forward = wrapper` bypassed _call_impl
  (#18).
- **LR experiments** silently no-op'd because ultralytics' default
  `optimizer='auto'` overrides any user-supplied `lr0/momentum` while
  printing only one log line (#16).
- **tiebreak rule** was vague enough that two agents would make
  different keep/discard decisions on the same data (#19).

The unit tests passed because they tested individual primitives in
isolation. None of them exercised the full pipeline:
```
inject_modules → ultralytics.train() → trainer.setup_model →
trainer.optimizer.step() → val_phase → checkpoint.save → resume
```
Every step in that chain has a way to silently break the previous
step's work.

v1.7.7 adds the contracts that make each step survive the next.

## The 6 fixes

### Fix #13 — yaml_inject head reference shifting

**Severity**: yaml_inject was effectively unusable in v1.7 → v1.7.6.

**Symptom**: Insertion in backbone shifted downstream layer indices by
+1, but head's absolute `Concat [-1, 6]` references still pointed at
base index 6 (which was now a different layer in the modified model).
First forward pass crashed:
```
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 16 but got size 32 for tensor number 1 in the list.
```

**Why it was latent**: `test_weight_transfer.py` tested
`generate_custom_yaml` for YAML correctness (right insertion in right
section) and tested `transfer_weights` for tensor-shape correctness.
Neither test built the custom model and ran a forward pass through
it. The bug surfaced only in real Loop 4 attempt 1 of the user's
VisDrone run.

**Fix**: New `update_head_refs(section_lines, inserted_base_positions)`
function in `weight_transfer.py`. Called twice from
`generate_custom_yaml`: once on head, once on neck. Shifts every
absolute (`>= 0`) from-reference by the count of insertions whose
base index is strictly less than the reference. Negative refs (`-1`,
`-2`) are layer-relative and untouched. List-of-refs (`[-1, 6]`)
shifts each absolute element independently.

12 new tests cover edge cases (empty, no-op, single shift, compound
shift, insertion AT ref position vs before, three-way Concat,
defensive copy, malformed lines) plus 2 integration tests verifying
`generate_custom_yaml` produces correctly-shifted head refs after
an insertion.

### Fix #14 — Hooks survive trainer rebuild

**Severity**: hook mode silently produced baseline metrics; whole
TIME_BUDGET wasted per iteration.

**Symptom**: `inject_modules(model)` attaches hooks. Then
`model.train()` runs ultralytics' Trainer, which calls
`setup_model() → self.model = self.get_model(...)` constructing a
fresh DetectionModel and loading weights into it. Hooks bound to the
original model are silently lost. Training completes; mAP equals
baseline; model summary line shows the baseline parameter count
(e.g. 55.6M) instead of the augmented count (e.g. 60.4M with CBAM).

**Fix**: New `reapply_on_rebuild(model, reapply_fn)` helper in
`shared/hook_utils.py`. Registers an `on_pretrain_routine_start`
callback that monkey-patches `trainer.get_model` so `reapply_fn` is
called on the freshly-rebuilt model. If `reapply_fn` raises, training
proceeds anyway — the agent will see the symptom (baseline mAP) and
investigate, rather than masking the underlying problem with a
crash inside an ultralytics callback.

Template `inject_modules()` docstring updated with mandatory contract:
"Branches that attach hooks MUST also call reapply_on_rebuild."

### Fix #15 — Hooks must be picklable

**Severity**: training succeeds for epoch 1 train+val, then crashes at
checkpoint write. Lost work proportional to TIME_BUDGET / epoch_count.

**Symptom**:
```
Can't pickle local object 'apply_X.<locals>.make_hook.<locals>.hook'
```
ultralytics calls `torch.save(model)` after every epoch, pickling the
entire nn.Module including all `_forward_hooks`. Closure-based hooks
contain references to enclosing scope and cannot be pickled.

**Fix**: `PicklableHook` base class in `shared/hook_utils.py`.
Subclasses must be defined at module top level. Documented in
`train-script-spec.md § Hook-mode contract`. Critical Rule #14 in
autoresearch SKILL forbids `layer.forward = wrapper` (related, see #18).

### Fix #17 — Hooks must be dtype-aware

**Severity**: training succeeds for ~all train batches, then crashes
at val batch 0.

**Symptom**:
```
RuntimeError: Input type (HalfTensor) and weight type (FloatTensor)
should be the same
```
Training uses AMP autocast (FP16). Val phase doesn't (FP32). A hook
with internal `nn.Module` gets FP16 inputs during train and FP32
during val — its FP32 weights cannot accept FP16 inputs.

**Fix**: `PicklableHook` provides `_param_dtype(submodule)` and
`_dtype_cast(tensor, target)` helpers. Subclasses use them in
`__call__` to align input dtype with the inner module's parameter
dtype. Pattern documented in `train-script-spec.md § Hook-mode
contract Rule 2`.

### Fix #18 — Use `register_forward_hook`, never `.forward = wrapper`

**Severity**: works in train phase, breaks unpredictably in val/eval
phase or fused conv path.

**Symptom**: dtype mismatch in val phase even with dtype-aware hook
(because direct `.forward = wrapper` bypasses PyTorch's `_call_impl`
machinery, including `_global_forward_pre_hooks` and ultralytics'
fused-conv path).

**Fix**: `PicklableHook.attach(layer, cls, *args, **kwargs)`
classmethod is the only sanctioned way to attach a hook —
internally always uses `register_forward_hook`. Critical Rule #14
in autoresearch SKILL forbids direct method override.

### Fix #16 — Optimizer cannot be 'auto' but can be swapped

**Severity**: every LR experiment under default settings silently
no-ops, looking identical to baseline.

**Symptom**: User sets `lr0=0.005`. ultralytics's default
`optimizer='auto'` picks SGD/AdamW by model size and **silently
overrides lr0 and momentum**, printing one log line that's easy to
miss:
```
'optimizer=auto' found, ignoring 'lr0=0.005' and 'momentum=0.937' ...
```

**Fix**: Three-layer defence:

1. Both `train.py.detection` and `train.py.tracking` templates have
   explicit `OPTIMIZER = "SGD"`, `LR0 = 0.01`, `MOMENTUM = 0.937` in
   Section ②, wired into `model.train(optimizer=OPTIMIZER, lr0=LR0,
   momentum=MOMENTUM)`.
2. The `train()` function raises `ValueError` if it detects
   `OPTIMIZER == 'auto'` at runtime — fail-fast rather than waste an
   iteration.
3. autoresearch Critical Rule #15 forbids setting `OPTIMIZER` to
   `'auto'`. Concrete optimizer swaps (SGD → AdamW) remain a
   legitimate experiment and are encouraged when an LR test plateaus.

`research_config.yaml → autoresearch.optimizer` lets the user pin
the initial optimizer (default `SGD`). orchestrator Stage 3 Step 3
writes it into both scripts at startup and rejects 'auto' with a
clear error.

`shared/test_templates.py` `REQUIRED_VARIABLES` now requires
`OPTIMIZER`, `LR0`, `MOMENTUM` in Section ②.

### Fix #19 — Tiebreak decision rule made explicit

**Severity**: ambiguous spec; different agents reached different
verdicts on identical results.

**Symptom**: research_config.yaml sets `primary: val_mAP50_95,
tiebreak: val_mAP50, min_improvement: 0.001`. SKILL Step 7 said
"if PRIMARY tied within MIN_IMPROVE AND tiebreak improves → keep"
but the word "tied" was undefined. Does primary = +0.0007 (below
threshold but still positive) count as "tied"? What about -0.0003
(slightly negative)? User's Loop 11/12 had +0.0007 with +0.0017
tiebreak — kept by user's interpretation, would have been discarded
under stricter reading.

**Fix**: Step 7 rewritten as a pseudo-code ladder with a new
`signed_improvement(new, old, name)` helper returning a directional
delta (positive = better in the metric's preferred direction). The
ladder's rule 4 (tiebreak rescue) is now:
```
primary_delta >= -EPSILON  AND  tiebreak_delta >= MIN_IMPROVE
```
where `EPSILON` defaults to `MIN_IMPROVE / 2` and is configurable
via the new optional yaml field `evaluation.metrics.regression_epsilon`.

Description suffix `[tiebreak]` added to `results.tsv` row when
rule 4 fires, so the human-readable distinction between "PRIMARY
clearly improved" and "PRIMARY borderline + TIEBREAK rescued" is
preserved.

## Files changed

| File | Change |
|---|---|
| `shared/weight_transfer.py` | New `update_head_refs()` function (~70 lines); `generate_custom_yaml` calls it for head + neck before `_reassemble` |
| `shared/test_weight_transfer.py` | +12 tests (10 unit + 2 integration). Total 36 → 48. |
| `shared/hook_utils.py` | **NEW** ~190 lines. `PicklableHook` base class, `_dtype_cast`, `_param_dtype`, `attach` classmethod, `reapply_on_rebuild` helper. |
| `shared/test_hook_utils.py` | **NEW** ~210 lines, 13 tests, no torch dependency. |
| `shared/templates/train.py.detection` | Section ② adds OPTIMIZER/LR0/MOMENTUM with locked OPTIMIZER. `train()` adds 'auto' guard + optimizer/lr0/momentum kwargs. `inject_modules()` docstring rewritten with 4-point contract + CBAMHook example. |
| `shared/templates/train.py.tracking` | Same OPTIMIZER/LR0/MOMENTUM additions. inject_modules docstring updated to reference Lazy-wrapper contract. |
| `shared/test_templates.py` | `REQUIRED_VARIABLES` adds OPTIMIZER/LR0/MOMENTUM. |
| `shared/train-script-spec.md` | New `§ Hook-mode contract` (3 rules with examples). New `§ Trainer rebuild — reapply_on_rebuild contract`. New `§ Tunable contract — OPTIMIZER must never be 'auto'`. |
| `autoresearch/SKILL.md` | New Critical Rules #14 (no `layer.forward = wrapper`) + #15 (no `OPTIMIZER = "auto"`). Step 7 Decide rewritten as pseudo-code ladder. § Crash Diagnosis text synced to v1.7.6 actual sequence (was stale). |
| `paper-finder/SKILL.md` | § Integration mode adds v1.7.7 announcement that yaml_inject head-ref bug is fixed; existing hook-workaround entries can revert to yaml_inject. |
| `examples/research_config.visdrone-detection.yaml` | Adds `autoresearch.optimizer: SGD`. |
| `examples/research_config.visdrone-mot.yaml` | Adds `autoresearch.optimizer: SGD`. |
| `research-orchestrator/SKILL.md` | Stage 3 Step 3 writes initial OPTIMIZER from yaml; rejects 'auto'. |
| `CHANGELOG_v1.7.7.md` | New (this file) |
| `README.md` | Version bumped, layout tree, Versions list. |

## Files NOT changed

`shared/modules_md.py`, `shared/state_migrate.py`, `shared/parse_metrics.py`,
`shared/file-contracts.md`, `shared/templates/track.py.tracking` (no model
training in track.py), `shared/templates/arch_spec.schema.json`,
`shared/test_modules_md.py`, `dataset-hunter/SKILL.md`.

## Test coverage

| Suite | v1.7.6 | v1.7.7 |
|---|---|---|
| `shared/test_modules_md.py` | 18 | 18 |
| `shared/test_templates.py` | 3 | 3 |
| `shared/test_weight_transfer.py` | 36 | **48** (+12) |
| `shared/test_hook_utils.py` | 0 | **13** (NEW) |
| **Total python tests** | **57** | **82** (+25) |
| SKILL.md python snippets | 88 | 89 (+1) |

The 25 new python tests are the largest single addition since v1.7
itself. Most cover behaviour that previously only surfaced in real
training runs (head ref shifting, hook pickling, dtype casting,
trainer rebuild). With these tests in place, v1.7.7-onward releases
have a much stronger guarantee that latent integration bugs surface
before deployment.

## Upgrade path

Drop-in. State files migrate transparently. Existing
`research_config.yaml` files work; `autoresearch.optimizer` defaults
to SGD if missing.

**Templates with the old Section ② (no OPTIMIZER) will fail
`test_templates.py`** — this is intentional. The orchestrator's
Stage 0 Step 6 check would fail to find the new required variables.
Re-scaffold via:
```bash
rm train.py
# Then resume the pipeline; orchestrator regenerates train.py from
# the current template at Stage 0 Step 6.
```

For users who modified inject_modules in train.py to use closures or
direct `.forward = wrapper`: read `train-script-spec.md § Hook-mode
contract` and refactor to `PicklableHook` subclasses. Existing closures
will start failing at checkpoint write in v1.7.7-scaffolded runs.

For users whose modules.md tagged yaml_inject modules as `hook` to
work around the v1.7 head-ref bug: restore the original `yaml_inject`
tagging. The bug is fixed.

## Operational tip

After upgrade, run one short experiment with hook mode and confirm:
- ultralytics' model summary line shows the augmented param count
  (not the baseline count) — this confirms the hook survived trainer
  rebuild
- epoch 1 ckpt write succeeds — confirms hooks are picklable
- val phase completes without dtype errors — confirms dtype handling
- mAP differs from baseline — confirms the hook actually fires in the
  training data path

If any of these fail, the hook contract was violated somewhere; check
`train-script-spec.md § Hook-mode contract` line by line.

## v1.8 outlook

v1.7.7 closes the hook + yaml_inject correctness gap. v1.8 work
proceeds as planned:

- `weight_transfer.apply_yaml_spec()` for `full_yaml` mode (agent
  writes complete custom YAML, autoresearch supplies layer_map
  override or trusts class-name+structural-position heuristic)
- autoresearch SKILL § Dispatch full_yaml branch
- paper-finder Phase 5 full_yaml heuristic
- Part II remaining issues #20-#24 (BATCH_SIZE auto-halve for
  resource-impact tags, discoveries.md categories guide, per-loop
  log archive, stall-with-blocked handling, base_model.md
  user-override formalisation)

The v1.7.x line ends here for real this time. Next release: **v1.8**.
