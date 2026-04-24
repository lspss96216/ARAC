# train.py Specification

Every skill in this pipeline touches `train.py`. Before this document existed, each
skill assumed a slightly different structure: autoresearch referenced
`Section ②` and `inject_modules()`, dataset hunter assumed `FREEZE_LAYERS` and
`CKPT_DIR` existed as top-level variables, orchestrator patched `TIME_BUDGET` and
`SEED` with regex. None of these assumptions were written down.

This file is where they are written down. Skills do not reinvent the contract; they
read this spec, then obey it. Templates that follow this spec live under
`<skills_dir>/shared/templates/`.

---

## What `train.py` is

A single Python script that:
1. Loads a model from `WEIGHTS`.
2. Optionally injects experimental modules declared by `USE_<MODULE>` flags.
3. Trains on `DATA_YAML` for `TIME_BUDGET` seconds with `SEED` fixed.
4. For `task_type: object_detection`, also runs validation and prints metrics.
5. Exits 0 on success, non-zero on failure.

## What `train.py` is not

- It is not a CLI tool. Skills modify it by editing variables, not by passing flags.
- It is not a library. Skills run it as `uv run train.py` and parse its stdout.
- It is not the tracker. For `task_type: object_tracking`, a separate `track.py`
  runs inference + TrackEval after `train.py` completes (see § Task-type variants).

---

## File layout — four sections, in order

Every `train.py` has four labelled sections. The labels are **load-bearing**:
autoresearch `sed`s into Section ②, orchestrator's regex patch expects
`TIME_BUDGET` and `SEED` at column 0, and dataset hunter copies the whole file to
`pretrain.py` and overrides specific variables. Breaking the labels breaks every
skill.

```python
# train.py  (minimal shape — real template at shared/templates/train.py.detection)

# ═══════════════════════════════════════════════════════════════════════════════
# Section ① — Imports and constants
# ═══════════════════════════════════════════════════════════════════════════════
from pathlib import Path
import random, numpy as np, torch
from ultralytics import YOLO
from custom_modules import *   # noqa: F401,F403 — populated by autoresearch

# ═══════════════════════════════════════════════════════════════════════════════
# Section ② — Tunables (the contract surface)
# ═══════════════════════════════════════════════════════════════════════════════
# Locked by orchestrator — never edit from autoresearch
TIME_BUDGET = 1200      # seconds per training run
SEED        = 42        # random seed

# Modifiable by autoresearch and dataset hunter
BATCH_SIZE     = 16
WEIGHTS        = "weights/yolo26x.pt"
DATA_YAML      = "data/visdrone.yaml"
NUM_CLASSES    = 10
FREEZE_LAYERS  = 0
CKPT_DIR       = Path("runs/train")

# Architecture injection (v1.7) — flips on for experiments that modify the
# model's layer graph via weight_transfer.py. See § Architecture injection.
ARCH_INJECTION_ENABLED   = False
ARCH_INJECTION_SPEC_FILE = "arch_spec.json"

# Module toggles — autoresearch flips these one at a time
USE_CMC              = False
USE_NSA_KALMAN       = False
USE_SMALL_OBJECT_FPN = False
# (autoresearch appends new USE_* flags here as it pulls modules from modules.md)

# ═══════════════════════════════════════════════════════════════════════════════
# Section ③ — inject_modules hook
# ═══════════════════════════════════════════════════════════════════════════════
def inject_modules(model):
    """Apply experimental modules based on USE_* flags.

    autoresearch adds new branches here when it pulls a module from modules.md.
    Each branch must be idempotent and survive being run even when its flag
    is False (i.e. no side effects when off).
    """
    if USE_SMALL_OBJECT_FPN:
        from custom_modules import SmallObjectFPN
        # If a branch returns a replaced model, the caller MUST rebind
        # to the return value (see § inject_modules() — Contract).
        return SmallObjectFPN.replace_neck(model)
    # Tracker-layer modules (CMC, NSA_KALMAN, ...) are applied in track.py,
    # not here — see Task-type variants.
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# Section ④ — Main (train + eval)
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # v1.7 — architectural modifications go via weight_transfer (see
    # § Architecture injection). Hook-style modules still go through
    # inject_modules() below.
    if ARCH_INJECTION_ENABLED:
        import json
        from weight_transfer import build_custom_model_with_injection
        spec = json.loads(Path(ARCH_INJECTION_SPEC_FILE).read_text())
        model = build_custom_model_with_injection(WEIGHTS, spec, imgsz=IMGSZ)
    else:
        model = YOLO(WEIGHTS)

    model = inject_modules(model)   # ← REBIND REQUIRED; see § Contract

    model.train(
        data=DATA_YAML, epochs=1000, batch=BATCH_SIZE,
        time=TIME_BUDGET / 3600,         # ultralytics wants hours
        freeze=FREEZE_LAYERS, project=str(CKPT_DIR), seed=SEED,
        pretrained=False,                # weight_transfer handles transfer manually
    )

    # For object_detection, evaluate and print metrics here.
    # For object_tracking, main() ends after training — track.py handles eval.
    metrics = model.val(data=DATA_YAML, imgsz=1920, batch=BATCH_SIZE)
    print_metrics(metrics, model)

def print_metrics(metrics, model):
    """Emit metrics in a format matchable by evaluation.parsing.patterns."""
    box = metrics.box
    print(f"val_precision:    {box.mp:.4f}")
    print(f"val_recall:       {box.mr:.4f}")
    print(f"val_mAP50:        {box.map50:.4f}")
    print(f"val_mAP50_95:     {box.map:.4f}")
    # Floor at 0.1M to prevent degenerate guard tolerance on tiny test models (D7)
    n_params = max(sum(p.numel() for p in model.parameters()) / 1e6, 0.1)
    print(f"Model Summary: {n_params:.1f}M params")

if __name__ == "__main__":
    main()
```

---

## The contract surface (Section ②)

These top-level variables are the **only** surface skills touch. Anything else
lives in Section ③ or ④ and is not part of the contract — skills must not edit it.

### Locked by orchestrator

| Variable | Who sets it | Why locked |
|---|---|---|
| `TIME_BUDGET` | `research_config.yaml → autoresearch.loop.time_budget_sec` | Fair comparison across experiments requires identical wall-clock budget |
| `SEED` | `research_config.yaml → autoresearch.loop.seed` | Reproducibility; keep/discard decisions must reflect the change being tested, not RNG |
| `IMGSZ` | `research_config.yaml → evaluation.ultralytics_val.imgsz` (or template default) | Changing resolution between experiments invalidates metric comparisons — a model at 1920 vs 640 is a different operating point. If memory is tight, halve `BATCH_SIZE` instead. |

Orchestrator patches all three with regex at Stage 3 and never unlocks them. Skills
that edit any of these values corrupt the experimental log. The regex assumes column 0:

```python
# Each line must start at column 0, one assignment per line
TIME_BUDGET = 1200
SEED        = 42
IMGSZ       = 1920
```

### Modifiable by autoresearch

| Variable | Typical reason to change |
|---|---|
| `BATCH_SIZE` | OOM mitigation (halve on OOM), or deliberate ablation |
| `WEIGHTS` | Orchestrator sets this to pretrain checkpoint if pretrain won |
| `DATA_YAML` | Rarely; usually only dataset hunter when it writes `pretrain.py` |
| `NUM_CLASSES` | Rarely; only when changing `DATA_YAML` |
| `FREEZE_LAYERS` | Transfer-learning experiments, or `0` in pretrain |
| `CKPT_DIR` | Avoid checkpoint-dir collisions (e.g. pretrain vs finetune) |
| `USE_<MODULE>` | Every autoresearch hook-mode loop flips exactly one of these |
| `ARCH_INJECTION_ENABLED` | **v1.7** — flipped True when autoresearch picks a module with `Integration mode: yaml_inject` (see § Architecture injection) |
| `ARCH_INJECTION_SPEC_FILE` | **v1.7** — rarely changed; path to the JSON spec file (default `arch_spec.json`). autoresearch writes the spec file, not this variable. |

### Appendable by autoresearch

`USE_<MODULE>` flags are the extension point. When autoresearch pulls a module
from `modules.md`, it **appends** a new `USE_<MODULE> = False` line at the end of
Section ②, then adds the matching branch to `inject_modules()`. Never reorder or
delete existing `USE_*` flags — `results.tsv` references them by name in the
`description` column for past experiments.

---

## `inject_modules()` — the hook contract

### Signature

```python
def inject_modules(model):
    ...
    return model
```

**Note:** earlier drafts wrote `def inject_modules(model) -> model` as the
signature. That is not valid Python — `model` is a parameter name, not a type.
The actual object returned is whatever `WEIGHTS` loads (for ultralytics YOLO
this is a `YOLO` instance; for other frameworks, the framework's model type).

### Contract — what callers and branch authors must guarantee

1. **Caller rebind (C6).** Section ④ must call `inject_modules` with its
   return value rebound:

   ```python
   model = YOLO(WEIGHTS)
   model = inject_modules(model)   # ← MUST rebind; do not just call
   model.train(...)
   ```

   Never write `inject_modules(model)` without the rebind. Some branches
   return a replaced instance (e.g. `SmallObjectFPN.replace_neck(model)`
   builds a new model around a new neck) — if the caller doesn't rebind,
   the replacement is silently lost and `model.train(...)` runs on the old
   model. Bug appears as "the module does nothing" and is nearly impossible
   to diagnose from run.log alone.

2. **Branch idempotency when flag is False.** Each branch inside
   `inject_modules` must be a no-op (no mutation, no side effects) when its
   `USE_<MODULE>` flag is False. A branch that does
   `globals()["YOLO"] = MonkeyPatched` at import time is NOT idempotent;
   flipping the flag back to False does not undo the monkey-patch.

3. **Call site.** `inject_modules` is called exactly once, in Section ④,
   between `YOLO(WEIGHTS)` and `model.train(...)`. It is not called from
   pretrain scripts or from within the training loop.

### Injection technique: replace, don't wrap

**Never monkey-patch `model.forward()` or wrap layers with decorators.**
Ultralytics saves checkpoints by serializing model state dicts. Monkey-patched
forward methods and wrapper objects are not part of the state dict, so their
weights are lost on checkpoint save/load. An experiment that "works" during
training but whose weights vanish on reload is a false positive.

Instead, **replace the target layer** with a new `nn.Module` subclass:

```python
# WRONG — monkey-patch (weights not saved)
original_forward = model.model[6].forward
def patched_forward(x):
    return my_attention(original_forward(x))
model.model[6].forward = patched_forward

# RIGHT — subclass replacement (weights saved with checkpoint)
class EnhancedBlock(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.block = original_block
        self.attention = MyAttention(channels=original_block.out_channels)
    def forward(self, x):
        return self.attention(self.block(x))

model.model[6] = EnhancedBlock(model.model[6])
```

The subclass approach makes the new parameters part of `model.parameters()`,
so they appear in the state dict and survive checkpoint round-trips.

Tracker-layer modules (CMC, NSA Kalman, re-ID) do **not** go through
`inject_modules()` — they live in the tracker config and are applied by
`track.py`. `inject_modules` is strictly for detector surgery (backbone / neck /
head / loss).

### Anti-patterns (will not survive ablation)

**Non-idempotent: module-level monkey-patch**
```python
# WRONG — side effect runs at import, survives flag being False
if USE_CMC:
    import ultralytics
    ultralytics.YOLO = lambda *a, **k: MonkeyPatchedYOLO(*a, **k)
```

**Non-idempotent: forward-wrapping**
```python
# WRONG — wrapped method not serialized in checkpoint; false positive
if USE_ATTENTION:
    orig_forward = model.forward
    model.forward = lambda x: attention(orig_forward(x))
```

**Correct: subclass-based replacement**
```python
if USE_SMALL_OBJECT_FPN:
    from custom_modules import SmallObjectFPN
    return SmallObjectFPN.replace_neck(model)   # returns new instance
return model
```

### Helper: `assert_idempotent()`

When writing a new branch, verify idempotency during development. Call with
the flag forced to False — the state dict should not change.

```python
def assert_idempotent(model, fn):
    """Dev-only: verify fn does not mutate model when its flag is False."""
    before = {k: v.clone() for k, v in model.state_dict().items()}
    _ = fn(model)
    after = model.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), \
            f"inject_modules mutated {k} when flag was False"
```

---

## Architecture injection (v1.7)

`inject_modules()` covers **hook-mode** changes: swap a layer, replace the
neck, wrap a forward pass. It cannot do **structural** changes — inserting a
new layer mid-backbone shifts every downstream layer index, and pretrained
weights stop aligning. That's what `weight_transfer.py` handles.

### Two injection modes

Autoresearch picks one per iteration based on the module's `Integration mode`
field in `modules.md`:

| Integration mode | Mechanism | What changes |
|---|---|---|
| `hook` (default, v1.6) | `inject_modules(model)` branch | Layer swap, forward wrap, block replacement (same index layout) |
| `yaml_inject` (v1.7) | `weight_transfer.build_custom_model_with_injection` | Insert new layer(s) into YAML, rebuild model, transfer pretrained weights according to computed layer_map |
| `full_yaml` (reserved v1.8+) | Agent writes full custom YAML | Arbitrary structural changes — NotImplementedError in v1.7 |

Unknown `Integration mode` values fall back to `hook` with a stderr warning
(warn-not-reject policy in `modules_md.py`).

### How `yaml_inject` works

When autoresearch picks a pending module with `Integration mode: yaml_inject`:

1. autoresearch writes `ARCH_INJECTION_SPEC_FILE` (default `arch_spec.json`)
   containing the module's insertion spec — see
   `shared/templates/arch_spec.schema.json` for the schema.
2. autoresearch flips `ARCH_INJECTION_ENABLED = True` in Section ②.
3. autoresearch runs `train.py`. Section ④ branches on
   `ARCH_INJECTION_ENABLED` and calls
   `weight_transfer.build_custom_model_with_injection(WEIGHTS, spec, imgsz=IMGSZ)`.
4. The helper:
   - Reads the base model's YAML from `WEIGHTS` (a .pt file).
   - Generates a new YAML by applying the spec's insertions (position + scope).
   - Builds a fresh `YOLO(new_yaml)` and computes `layer_map` (base-index →
     custom-index) using the insertion bookkeeping.
   - Stage 1: transfers matching tensors per layer_map with per-entry strict
     mode. Any layer_map entry that transfers zero tensors raises — this is
     how misaligned maps get caught instead of silently producing a
     random-init layer.
   - Forces every `Lazy*` wrapper to build its inner module (see § Lazy-wrapper
     contract below) so the optimizer sees their parameters.
   - Registers a Stage 2 `on_train_epoch_start` callback that re-transfers
     before epoch 0 begins, in case the trainer's own init re-initialised
     any layers during Stage 1.
5. The returned `YOLO` instance is then passed through `inject_modules(model)`
   for any hook-mode modules that coexist in the same run.

### `ARCH_INJECTION_SPEC_FILE` format

Pointed to by Section ② `ARCH_INJECTION_SPEC_FILE`. JSON, validated by
`shared/templates/arch_spec.schema.json`. Minimal example (insertion mode):

```json
{
  "mode": "insertions",
  "insertions": [
    {
      "module_class": "LazyCBAM",
      "position": {"kind": "after_class", "class_name": "Conv"},
      "scope": "backbone",
      "yaml_args": [64],
      "module_kwargs": {"kernel_size": 7}
    }
  ],
  "strict": true
}
```

Positions:
- `{"kind": "after_class", "class_name": "Conv"}` — inserts after **every**
  layer of that class within `scope`.
- `{"kind": "at_index", "index": 5}` — inserts after base-yaml index 5. The
  index is relative to the **base** YAML, not the custom YAML being built;
  the helper handles offsets. `at_index` must fall within `scope` or raises.

Scopes: `backbone`, `neck`, `head`, `all`. For `at_index` mode, `scope` is
enforced as an assertion (out-of-scope index raises).

### Lazy-wrapper contract

`yaml_inject` requires the inserted class to be a **lazy wrapper** whose
`__init__` is side-effect-free and whose inner module is built on the first
`forward`. Reasons:

1. Ultralytics' `parse_model` resolves unknown modules by calling
   `cls(*yaml_args)` — **not** prepending the input channel count. The lazy
   wrapper reads the real channel count from `x.shape[1]` at forward time.
2. `build_custom_model_with_injection` calls `force_lazy_build(model, imgsz)`
   after Stage 1 transfer. This runs one forward pass so the inner module is
   built and its parameters appear in `model.parameters()` before the
   trainer's `build_optimizer()` captures the parameter list. Without this
   step, the new module is silently excluded from training — loss is
   identical to baseline, no crash.

The helper moves the model to GPU before the dummy forward if available, so
the inner module is built on the right device.

Minimal lazy wrapper:

```python
import torch.nn as nn

class LazyCBAM(nn.Module):
    def __init__(self, _c_hint=None, kernel_size=7):
        # Side-effect-free: no nn.Conv2d, no nn.Linear.
        # _c_hint is the YAML's first positional arg; ignored here, the real
        # channel count comes from x.shape[1] below.
        super().__init__()
        self.kernel_size = kernel_size
        self.cbam = None

    def forward(self, x):
        if self.cbam is None:
            from ultralytics.nn.modules.conv import CBAM as _CBAM
            self.cbam = _CBAM(x.shape[1], self.kernel_size).to(x.device)
        return self.cbam(x)
```

Register in `custom_modules.py` the same way as hook-mode modules (see
`paper-finder/SKILL.md § register_custom_modules`). The name used in the
YAML insertion (`LazyCBAM` above) must match the registered class name
exactly.

### Hook-mode contract (v1.7.7)

When you write a hook-mode module (the `inject_modules()` branch), the
hook callable that you pass to `register_forward_hook` must obey three
hard rules. Violating any of them produces a silent or delayed failure
that wastes a full TIME_BUDGET and is hard to attribute back to the hook.

**Rule 1 — hooks must be picklable.** ultralytics calls `torch.save(model)`
every epoch end to write `last.pt` and `best.pt`. This pickles the entire
nn.Module including all forward hooks. Any closure-based hook fails:

```python
# ✗ BAD — closure captures `cbam` from enclosing scope; not picklable
def make_hook(cbam):
    def hook(module, inputs, output):
        return cbam(output)
    return hook

# ✓ GOOD — top-level class; pickle-safe
class CBAMHook(PicklableHook):
    def __init__(self, channels):
        super().__init__()
        self.cbam = CBAM(channels)
    def __call__(self, module, inputs, output):
        return self.cbam(output)
```

Symptom of violation: `Can't pickle local object` at epoch 1 ckpt write.
Train + val of epoch 1 both succeed; the pickle error appears only when
ultralytics tries to write the checkpoint file.

**Rule 2 — hooks must be dtype-aware.** Training uses AMP autocast (FP16);
val phase doesn't (FP32). A hook that owns its own nn.Module gets FP32
input during val, FP16 input during train, and crashes:

```python
# ✓ GOOD — _dtype_cast handles AMP/val mismatch
class CBAMHook(PicklableHook):
    def __init__(self, channels):
        super().__init__()
        self.cbam = CBAM(channels)
    def __call__(self, module, inputs, output):
        target = self._param_dtype(self.cbam)
        x = self._dtype_cast(output, target)
        y = self.cbam(x)
        return y.to(output.dtype)
```

Symptom: `RuntimeError: Input type (HalfTensor) and weight type
(FloatTensor) should be the same`. Train all batches succeed, val batch 0
crashes.

**Rule 3 — never assign `layer.forward = wrapper`.** Always use
`layer.register_forward_hook(callable)`. Direct method assignment bypasses
PyTorch's `_call_impl` machinery (global hooks, fused-conv eval paths,
ultralytics' `Detect` postprocessing) and breaks unpredictably:

```python
# ✗ BAD — bypasses _call_impl
layer.forward = my_wrapper

# ✓ GOOD — uses the official hook system
PicklableHook.attach(layer, MyHook, *args)
```

`PicklableHook.attach()` is a classmethod that instantiates the hook and
calls `register_forward_hook`. Use it as the only way to attach a hook in
this pipeline.

### Trainer rebuild — reapply_on_rebuild contract (v1.7.7)

ultralytics' `Trainer.setup_model()` runs `self.model = self.get_model(...)`,
which **constructs a fresh DetectionModel and loads weights into it**.
Any hooks you registered on the model *before* `model.train()` are bound
to the OLD object. The fresh object has no hooks. `inject_modules()`'s
work is silently lost.

Symptom of violation: training completes, mAP equals baseline within
floating-point noise, model summary line printed by ultralytics shows the
**baseline parameter count** (e.g. 55.6M) rather than the augmented count
(e.g. 60.4M including CBAM blocks).

**Required fix**: every `inject_modules()` branch that attaches a hook
MUST also register `reapply_on_rebuild` so the hook gets re-attached on
the rebuilt model:

```python
from hook_utils import PicklableHook, reapply_on_rebuild

def inject_modules(model):
    if USE_CBAM:
        from custom_modules import CBAMHook
        # Initial attach to the model that exists right now
        for idx in (4, 6, 8):
            CBAMHook.attach(model.model.model[idx], channels=256)
        # Re-attach after trainer rebuilds the model in setup_model
        def _reapply(rebuilt):
            for idx in (4, 6, 8):
                CBAMHook.attach(rebuilt.model[idx], channels=256)
        reapply_on_rebuild(model, _reapply)
    return model
```

The reapply function takes the **rebuilt nn.Module** (not the YOLO wrapper)
as its argument. Index into it the same way you'd index `model.model`,
omitting the outer `.model` wrapper level.

If `reapply` raises an exception, training proceeds anyway — the hooks
are simply missing on the rebuilt model. This is intentional: a crash
inside reapply would mask the underlying problem (the agent should see
the symptom mAP-equals-baseline and inspect, rather than have the entire
training fail with a confusing traceback from inside an ultralytics
callback).

### Tunable contract — OPTIMIZER must never be 'auto' (v1.7.7)

The Section ② variable `OPTIMIZER` is a tunable that autoresearch may
change between experiments (e.g. SGD → AdamW for an adaptive-LR test).
It must NEVER be set to the literal string `'auto'`. ultralytics' 'auto'
mode silently overrides any user-supplied `LR0` and `MOMENTUM`, only
printing one log line:

```
'optimizer=auto' found, ignoring 'lr0=0.005' and 'momentum=0.937' and ...
```

Any LR-tuning experiment under 'auto' would silently run with ultralytics'
internally-chosen LR instead of the LR being tested. Whole iterations of
TIME_BUDGET wasted producing data that doesn't answer the experimental
question.

Permitted values: `SGD`, `AdamW`, `Adam`, `RMSProp`, `NAdam`, `RAdam`.
The `train()` function in the template raises `ValueError` if it detects
`'auto'` at runtime, but autoresearch must not even try.

### Coexistence with `inject_modules()`

`yaml_inject` runs first, producing a structurally modified `model`. Then
`inject_modules(model)` runs on the result — hook modules apply on top of
the new structure. Most experiments use one mode or the other, but both
paths are live every iteration.

### Failure modes and how they surface

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: transfer_weights strict mode: N layer_map entries transferred 0 tensors` | Insertion spec caused layer misalignment (wrong class, wrong scope, or multiple insertions with overlapping indices) | Autoresearch records crash → discard → revert. Paper-finder may need to refine the module's insertion spec. |
| `NotImplementedError: mode='full_yaml' is reserved for v1.8+` | Someone wrote `"mode": "full_yaml"` in the JSON spec | v1.7 only supports `insertions`. Either downgrade the spec or wait for v1.8. |
| Training runs but loss identical to baseline | `force_lazy_build` skipped or lazy wrapper's `__init__` has side effects that pre-built params | Ensure the module class is a spec-compliant lazy wrapper (see above). `build_custom_model_with_injection` calls `force_lazy_build` automatically. |

---


`train.py`'s stdout is the canonical source of metrics. Whatever it prints must be
parseable by the regex / JSONPath / CSV rules defined in
`research_config.yaml → evaluation.parsing`.

The parser pattern and the print statement are **two sides of the same contract**.
If you change one you must change the other. When a skill's parser returns
`None` for a metric, the first thing to check is not the model — it is whether
`train.py`'s print matches `evaluation.parsing.patterns[<metric>]`.

Example (detection, `tool: ultralytics_val`):

```
# train.py prints                       # evaluation.parsing.patterns matches
val_mAP50_95:     0.2451                val_mAP50_95: 'val_mAP50_95:\s+([\d.]+)'
Model Summary: 7.2M params              num_params_M: 'Model Summary:.*?([\d.]+)M params'
```

### Templates always reformat tool output

Tool-native output (ultralytics val tables, TrackEval column reports) is
**unstable and position-dependent** — a single column added upstream shifts
every regex. Templates therefore convert tool output into canonical
`<key>: <value>` lines before exiting. Each template is responsible for this
reformatting:

- `train.py.detection` — wraps `model.val()` output via `print_metrics()`
- `track.py.tracking` — parses TrackEval's table and re-emits
  `HOTA:`, `MOTA:`, `IDF1:`, `IDSW:`, and `FPS:` lines (see the template for
  the extraction)

This means `evaluation.parsing.patterns` only ever needs to match one stable
format: the template's canonical prints. Consumers of `run.log` do not parse
raw tool output.

---

## Task-type variants

### `object_detection`

`train.py` does train + val in one script, Section ④ ends with
`model.val(...); print_metrics(...)`. Run command: `uv run train.py > run.log`.
This is the default template at `templates/train.py.detection`.

### `object_tracking`

`train.py` does **training only**. It trains the detector on labelled video
frames and exits after printing `num_params_M`. Tracking evaluation requires:
1. The trained detector from `train.py`
2. A tracker (configured in `tracker.yaml`, with CMC / Kalman flags)
3. TrackEval comparing tracker output to GT

That pipeline lives in `track.py` (template: `templates/track.py.tracking`).
`track.py` is also in-scope for autoresearch — tracker-layer modules
(Camera Motion Compensation, NSA Kalman, re-ID) are toggled via `USE_*` flags at
the top of `track.py`, not `train.py`.

Run command when `task_type: object_tracking`:
```bash
uv run train.py > run.log 2>&1      # train detector
uv run track.py >> run.log 2>&1     # run tracker + TrackEval, append to same log
```

Both scripts write to the same `run.log`, so a single `evaluation.parsing`
config can extract detection and tracking metrics from one file.

### Other task types

Not yet templated. When support is added, each new `task_type` gets its own
template under `templates/` and a subsection in this spec. The four contract
surfaces (Section ② variables, `inject_modules()`, metric print format, run
command) apply universally.

---

## How to verify a `train.py` is spec-compliant

Quick checklist a skill can run before modifying:

```python
import pathlib, re
src = pathlib.Path("train.py").read_text()

# v1.7.5 — Section markers matched with regex tolerating both ASCII digits
# (Section 2) and circled digits (Section ②). Editors or IDEs that rewrite
# Unicode don't break the spec check.
required_section_pats = [
    ("Section 1", re.compile(r"(?mi)^#?\s*Section\s*[①1]\b")),
    ("Section 2", re.compile(r"(?mi)^#?\s*Section\s*[②2]\b")),
    ("Section 3", re.compile(r"(?mi)^#?\s*Section\s*[③3]\b")),
    ("Section 4", re.compile(r"(?mi)^#?\s*Section\s*[④4]\b")),
]
required_variables = ["TIME_BUDGET", "SEED", "BATCH_SIZE", "WEIGHTS",
                      "DATA_YAML", "NUM_CLASSES", "CKPT_DIR",
                      # v1.7 — architecture injection surface
                      "ARCH_INJECTION_ENABLED", "ARCH_INJECTION_SPEC_FILE"]
required_functions = ["def inject_modules", "def main"]
required_branches  = ["if ARCH_INJECTION_ENABLED",
                      "build_custom_model_with_injection"]

missing = []
missing += [name for name, pat in required_section_pats if not pat.search(src)]
missing += [v for v in required_variables if not re.search(rf"(?m)^{v}\s*=", src)]
missing += [f for f in required_functions if f not in src]
missing += [b for b in required_branches  if b not in src]

if missing:
    raise RuntimeError(f"train.py is not spec-compliant. Missing: {missing}")
```

Skills that mutate `train.py` must run this check before their first edit and
abort with a clear error message if anything is missing, rather than silently
producing broken patches.

The orchestrator scaffolds a fresh `train.py` by copying the template matching
`task.task_type` and patching `TIME_BUDGET` / `SEED`. Autoresearch and dataset
hunter assume the resulting file conforms to this spec and do not re-validate
the layout.

---

## Deployment

```
<skills_dir>/
├── shared/
│   ├── modules_md.py               # canonical modules.md parser
│   ├── state_migrate.py            # pipeline_state.json schema migration
│   ├── parse_metrics.py            # shared stdout/json/csv metric extraction
│   ├── weight_transfer.py          # v1.7 — yaml_inject + pretrained transfer
│   ├── test_weight_transfer.py     # v1.7 — weight_transfer unit tests
│   ├── train-script-spec.md        # this file
│   ├── file-contracts.md           # schemas for all cross-skill files
│   └── templates/
│       ├── train.py.detection      # default for task_type: object_detection
│       ├── train.py.tracking       # detector-training half of tracking
│       ├── track.py.tracking       # tracker + TrackEval half
│       └── arch_spec.schema.json   # v1.7 — JSON schema for ARCH_INJECTION_SPEC
├── paper-finder/
├── autoresearch/
├── research-orchestrator/
└── dataset-hunter/
```

Each skill reads this spec when it touches `train.py`. The orchestrator
scaffolds a fresh `train.py` by copying the template matching `task.task_type`
and patching `TIME_BUDGET` / `SEED` / `IMGSZ`. Autoresearch adds `USE_<MODULE>`
flags (hook mode) or writes `arch_spec.json` + flips `ARCH_INJECTION_ENABLED`
(yaml_inject mode).
