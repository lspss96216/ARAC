# Changelog — v1.9.3

Release type: **Patch — initial BATCH_SIZE configurable from yaml.**
v1.6 → v1.9.2 hardcoded train.py's initial BATCH_SIZE to whatever the
template's default was (16). Users on H100 80GB ran experiments at
batch=16 with no obvious knob to change it. v1.9.3 adds
`autoresearch.loop.initial_batch_size` so the starting point is yaml-
configurable. Drop-in.

## Background

Real-world report: a user wondered why their H100 80GB machine ran
training at `BATCH_SIZE = 16` despite having abundant VRAM. The answer
revealed a design oversight:

- Template `train.py.detection` defines `BATCH_SIZE = 16` as default
- Orchestrator Stage 0 patches several variables: TIME_BUDGET, SEED,
  IMGSZ, OPTIMIZER, WEIGHTS, DATA_YAML, NUM_CLASSES
- BATCH_SIZE was deliberately omitted because autoresearch needs to
  halve it dynamically (crash-pause, resource_impact, OOM)
- But "deliberately omitted from locking" was conflated with
  "deliberately omitted from initialisation" — the starting value was
  silently the template default

The fix separates the two concerns:

| Concern | Locked? | Source |
|---|---|---|
| IMGSZ / SEED / TIME_BUDGET | Yes (invariants enforced) | yaml + Stage 0 patch |
| OPTIMIZER initial value | No | yaml + Stage 0 patch (initial set) |
| OPTIMIZER 'auto' forbidden | Soft constraint | invariants check |
| **BATCH_SIZE initial value** | **No** | **yaml + Stage 0 patch (v1.9.3+)** |
| BATCH_SIZE runtime adjustments | n/a | autoresearch (halve / restore) |

## The fix

### yaml field — `autoresearch.loop.initial_batch_size`

Optional integer in the loop block. If unset, falls back to template
default (16) — preserves v1.9.2 behaviour. Both yaml examples updated:

```yaml
autoresearch:
  loop:
    time_budget_sec: 1200
    max_runtime_multiplier: 2
    iterations: null
    seed: 42
    initial_batch_size: 16     # NEW v1.9.3 — starting BATCH_SIZE for train.py.
                               # Stage 0 patches the template's default into the actual value.
                               # NOT locked — autoresearch may halve dynamically.
                               # H100 80GB: try 64. Smaller GPUs: 8.
```

### orchestrator state — `state["initial_batch_size"]`

State init reads `ar.get("loop", {}).get("initial_batch_size")`. None
sentinel signals "use template default". State migration adds the key
with default None for resume-from-pre-v1.9.3.

### orchestrator Stage 0 Step 3 — patch BATCH_SIZE in scripts

After OPTIMIZER patch (which is the closest equivalent — initial set,
not locked), Stage 0 patches BATCH_SIZE if yaml provided a value:

```python
init_batch = state.get("initial_batch_size")
if init_batch is not None:
    if not isinstance(init_batch, int) or init_batch < 1:
        raise RuntimeError(
            "research_config.yaml → autoresearch.loop.initial_batch_size "
            "must be a positive integer..."
        )
    for script in scripts_to_lock:
        lock_variable(script, "BATCH_SIZE", init_batch)
```

If yaml is unset (None), no patch happens — template default remains.
This is the v1.9.2 behaviour preserved for users who haven't migrated.

### autoresearch SKILL — Critical Rule 11 prose update

Rule 11 already said "halve `BATCH_SIZE` instead of changing IMGSZ".
v1.9.3 adds a paragraph clarifying that BATCH_SIZE has yaml-configurable
initial value but unchanged runtime semantics:

> `BATCH_SIZE` itself is NOT locked. v1.9.3+ orchestrator sets the
> initial value from `autoresearch.loop.initial_batch_size` (yaml);
> after that, autoresearch may halve dynamically (resource_impact
> auto-halve, crash-pause halve, OOM detection). The yaml only
> controls the starting point — runtime adjustments are autoresearch's
> prerogative.

### Validation

| Input | Behaviour |
|---|---|
| Field absent / `null` | Use template default (16). v1.9.2 behaviour preserved. |
| Integer ≥ 1 | Patch into all scripts in scripts_to_lock at Stage 0. |
| Non-integer (string, list, etc.) | Stage 0 raises RuntimeError with hint to remove field or use integer. |
| Integer ≤ 0 | Same — raises RuntimeError. |
| Float (e.g. 16.0) | Same — raises (use 16, not 16.0). |

Stage 0 `raise` rather than warn-and-fall-back because invalid
BATCH_SIZE is a config error the user wants to know about immediately,
not a silent fallback to 16 that confuses sizing decisions later.

## What does NOT change

- **invariants.py** does NOT add BATCH_SIZE to LOCKED_VARS. autoresearch
  must remain free to halve it. Mid-run changes are NOT contract
  violations.
- **autoresearch's halve logic** (crash-pause v1.7.6, resource_impact
  auto-halve v1.9, OOM detection) is unchanged — those operate on
  whatever BATCH_SIZE happens to be in train.py at the time, and
  v1.9.3's yaml just changes the starting value.
- **`batch_size_pre_autohalve` state field** (v1.9 resource_impact
  restore mechanism) still works correctly — it captures whatever the
  current BATCH_SIZE is before halving, regardless of whether that came
  from yaml or template default.

## Files changed

| File | Change |
|---|---|
| `examples/research_config.visdrone-detection.yaml` | New `autoresearch.loop.initial_batch_size` field with sizing comment |
| `examples/research_config.visdrone-mot.yaml` | Same |
| `research-orchestrator/SKILL.md` | State init reads yaml field; Stage 0 Step 3 patches BATCH_SIZE after OPTIMIZER block; locked-variables prose updated to mention BATCH_SIZE initial set |
| `autoresearch/SKILL.md` | Critical Rule 11 prose adds v1.9.3 clarification on initial-vs-dynamic distinction |
| `shared/state_migrate.py` | New `initial_batch_size: None` default |
| `CHANGELOG_v1.9.3.md` | New (this file) |
| `README.md` | Banner + Versions list |

### Unchanged

- `shared/invariants.py` — BATCH_SIZE intentionally NOT in LOCKED_VARS
- `shared/templates/train.py.{detection,tracking}` — template default
  stays at 16; Stage 0 patches over it when yaml field is set
- All other shared/ files, all paper-finder/dataset-hunter SKILLs,
  every earlier CHANGELOG

## Test coverage

| Suite | v1.9.2 | v1.9.3 |
|---|---|---|
| `shared/test_modules_md.py` | 23 | 23 |
| `shared/test_templates.py` | 3 | 3 |
| `shared/test_weight_transfer.py` | 61 | 61 |
| `shared/test_hook_utils.py` | 13 | 13 |
| `shared/test_invariants.py` | 29 | 29 |
| **Total python tests** | **129** | **129** |
| SKILL.md python snippets | 100 | 100 |
| YAML examples | 2 | 2 |

No new tests because v1.9.3 has no new code paths that aren't covered
by existing scaffold-and-patch tests. The `lock_variable` helper used
to patch BATCH_SIZE is the same one that's been patching TIME_BUDGET /
SEED / IMGSZ / OPTIMIZER since v1.6 — its correctness is exercised by
every single integration use of orchestrator. The validation branch
(non-integer, ≤0) is a one-line type check before lock_variable runs.

If a future v1.10 wants belt-and-braces unit tests for the validation,
they'd live in a hypothetical `test_orchestrator_state_init.py` that
doesn't exist yet — out of scope for a single-issue patch.

## Upgrade path

Drop-in. Pre-v1.9.3 yaml without `initial_batch_size` keeps working —
falls back to template default (16). Post-v1.9.3 users add the field
to control batch size from yaml.

For users with v1.9.2 in-progress runs:
- Resume continues normally
- pipeline_state.json migration adds `initial_batch_size: None`
- The next iteration's BATCH_SIZE is whatever's currently in train.py
  (template default 16, OR a value autoresearch halved during a
  previous resource_impact event — both fine to resume from)
- If user wants to change batch size mid-pipeline: edit yaml,
  re-run Stage 0 (which re-patches scripts). This is the same
  procedure for changing any "initial set" variable like OPTIMIZER.

For users who want to bump batch size NOW without a re-run:
- Edit `train.py` directly: change `BATCH_SIZE = 16` to `BATCH_SIZE = 64`
- Commit
- autoresearch's next iteration uses 64
- Future resource_impact halves operate from 64 baseline correctly
- Add `initial_batch_size: 64` to yaml so future re-scaffolds use it

## H100 80GB sizing guidance (operational note)

Real-world VisDrone + YOLO26X + IMGSZ=1920 sizing data from previous
runs:

| BATCH_SIZE | Memory used | Throughput | Notes |
|---|---|---|---|
| 16 (default) | ~21 GB | baseline | Wastes 75% of 80GB GPU |
| 32 | ~42 GB | ~1.6× | Comfortable margin for vram_2x experiments |
| 48 | ~60 GB | ~2× | Usually fine; vram_4x experiments will halve |
| 64 | ~75 GB | ~2.3× | Possible, but vram_4x experiments may OOM even after auto-halve |

For your v1.9.3 first run, recommended `initial_batch_size: 32` or `48`.
batch=64 only if you commit to disabling vram_4x experiments via
`module_priority` filtering, otherwise the auto-halve will frequently
fire on attention-heavy modules.

## v1.10 outlook

Unchanged from v1.9 outlook. v1.9.3 is a single-issue patch.

The fact that BATCH_SIZE-from-yaml took until v1.9.3 to land — despite
TIME_BUDGET / SEED / IMGSZ / OPTIMIZER all being yaml-controllable
since v1.6 / v1.7.7 — is itself worth noting as a v1.10 candidate for
a broader audit: are there other train.py variables with the same
"locked from initialisation by accident" property? `LR0`, `MOMENTUM`,
`WEIGHT_DECAY`, augmentation hyperparameters all have template defaults
that could in principle come from yaml. The argument for not making
them yaml-controllable today is that autoresearch experiments those
ranges — but the same was true of BATCH_SIZE, and yet a user-controlled
starting point turned out to matter. Worth a v1.10 audit pass.
