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

# Module toggles — autoresearch flips these one at a time
USE_CMC              = False
USE_NSA_KALMAN       = False
USE_SMALL_OBJECT_FPN = False
# (autoresearch appends new USE_* flags here as it pulls modules from modules.md)

# ═══════════════════════════════════════════════════════════════════════════════
# Section ③ — inject_modules hook
# ═══════════════════════════════════════════════════════════════════════════════
def inject_modules(model):
    """Apply experimental modules based on USE_* flags. Mutates model in place.

    autoresearch adds new branches here when it pulls a module from modules.md.
    Each branch must be idempotent and survive being run even when its flag
    is False (i.e. no side effects when off).
    """
    if USE_SMALL_OBJECT_FPN:
        from custom_modules import SmallObjectFPN
        SmallObjectFPN.replace_neck(model)
    # Tracker-layer modules (CMC, NSA_KALMAN, ...) are applied in track.py,
    # not here — see Task-type variants.
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# Section ④ — Main (train + eval)
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    model = YOLO(WEIGHTS)
    model = inject_modules(model)

    model.train(
        data=DATA_YAML, epochs=1000, batch=BATCH_SIZE,
        time=TIME_BUDGET / 3600,         # ultralytics wants hours
        freeze=FREEZE_LAYERS, project=str(CKPT_DIR), seed=SEED,
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
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
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
| `USE_<MODULE>` | Every autoresearch loop flips exactly one of these |

### Appendable by autoresearch

`USE_<MODULE>` flags are the extension point. When autoresearch pulls a module
from `modules.md`, it **appends** a new `USE_<MODULE> = False` line at the end of
Section ②, then adds the matching branch to `inject_modules()`. Never reorder or
delete existing `USE_*` flags — `results.tsv` references them by name in the
`description` column for past experiments.

---

## `inject_modules()` — the hook signature

```python
def inject_modules(model) -> model
```

- Input: `model` — whatever type `WEIGHTS` loads. For ultralytics YOLO, this is a
  `YOLO` instance; for other frameworks, it is the framework's model type.
- Output: `model` (return the same or a replaced instance — both acceptable).
- Called: once in Section ④ between `YOLO(WEIGHTS)` and `model.train(...)`.
- Contract: **each branch inside is idempotent when its flag is False.** A branch
  with an always-True side effect (e.g. monkey-patching a global) breaks the
  ablation logic because flipping the flag off no longer actually turns the
  module off.

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

---

## Metric output contract

`train.py`'s stdout is the canonical source of metrics. Whatever it prints must be
parseable by the regex / JSONPath / CSV rules defined in
`research_config.yaml → evaluation.parsing`.

The parser pattern and the print statement are **two sides of the same contract**.
If you change one you must change the other. When a skill's parser returns
`None` for a metric, the first thing to check is not the model — it is whether
`train.py`'s print matches `evaluation.parsing.patterns[<metric>]`.

Example (detection, `tool: ultralytics_val`):

```python
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
  `HOTA:`, `MOTA:`, `IDF1:`, `IDSW:` lines (see the template for the extraction)

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

## Deployment

```
<skills_dir>/
├── shared/
│   ├── modules_md.py               # P1 Theme 3 output
│   ├── train-script-spec.md        # this file
│   └── templates/
│       ├── train.py.detection      # default for task_type: object_detection
│       ├── train.py.tracking       # detector-training half of tracking
│       └── track.py.tracking       # tracker + TrackEval half
├── paper-finder/
├── autoresearch/
├── research-orchestrator/
└── dataset-hunter/
```

Each skill reads this spec when it touches `train.py`. The orchestrator
scaffolds a fresh `train.py` by copying the template matching `task.task_type`
and patching `TIME_BUDGET` / `SEED`. Autoresearch and dataset hunter assume the
resulting file conforms to this spec and do not re-validate the layout.

---

## How to verify a `train.py` is spec-compliant

Quick checklist a skill can run before modifying:

```python
import pathlib, re
src = pathlib.Path("train.py").read_text()

required_sections  = ["Section ①", "Section ②", "Section ③", "Section ④"]
required_variables = ["TIME_BUDGET", "SEED", "BATCH_SIZE", "WEIGHTS",
                      "DATA_YAML", "NUM_CLASSES", "CKPT_DIR"]
required_functions = ["def inject_modules", "def main"]

missing = []
missing += [s for s in required_sections  if s not in src]
missing += [v for v in required_variables if not re.search(rf"(?m)^{v}\s*=", src)]
missing += [f for f in required_functions if f not in src]

if missing:
    raise RuntimeError(f"train.py is not spec-compliant. Missing: {missing}")
```

Skills that mutate `train.py` must run this check before their first edit and
abort with a clear error message if anything is missing, rather than silently
producing broken patches.
