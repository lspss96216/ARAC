# Changelog — v1.7.2

Release type: **Bug fix.** Two-location IMGSZ sync fix in cross-skill state
handoff. No schema changes, no new features. Drop-in on top of v1.7.1 /
v1.7 / v1.6 — state files resume transparently.

## The bug

`IMGSZ` in `research_config.yaml → evaluation.ultralytics_val.imgsz` was
**correctly locked into `train.py` by orchestrator Stage 3 Step 3**, but
silently lost in two downstream places:

1. **`dataset-hunter`'s `pretrain.py`** did not patch `IMGSZ` when deriving
   from the detection template. The template's default (1920) applied
   regardless of what the user configured. If the user set
   `imgsz: 1280`, the pretrain run used 1920, and the self-eval
   comparison (pretrain vs finetune) measured performance at two
   different resolutions, making the pretrain-improves-or-not signal
   unreliable.

2. **`autoresearch`'s Step 5.5c (yaml_inject repair, v1.7.1)** read
   `state["imgsz"]`, but **nobody wrote it**. Orchestrator had resolved
   `imgsz` locally at Step 3, locked it into the script, then let the
   local variable go out of scope. This latent `KeyError` only surfaced
   if v1.7.1's Step 5.5 shape probing actually triggered — yaml_inject
   + shape-mismatch crash. So far the repair loop had never actually run
   against a real crash, so the bug stayed hidden.

Both symptoms trace to the same root cause: **no canonical place to
read `imgsz` outside orchestrator Stage 3**. Each downstream skill
either hardcoded a default, skipped patching, or hallucinated a
`state["imgsz"]` that didn't exist.

## The fix

**One location now writes, everyone else reads from there.**

`research-orchestrator/SKILL.md § Stage 3 Step 3` — after locking
TIME_BUDGET / SEED / IMGSZ into the scripts, also persist:

```python
state["imgsz"] = imgsz
pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
```

`dataset-hunter/SKILL.md § Write pretrain.py / Apply it` — read imgsz
from state and patch it into `pretrain.py` Section ②:

```python
imgsz = state.get("imgsz", 1920)   # fallback for pre-v1.7.2 resume
patch_section_2(SELF_EVAL_SOURCE, "pretrain.py", {
    ...
    "IMGSZ": str(imgsz),
    ...
})
```

`autoresearch/SKILL.md § Step 5.5c` — the `state["imgsz"]` access
becomes `state.get("imgsz")` with a research_config.yaml re-read
fallback, so pre-v1.7.2 state files resumed after upgrade don't
KeyError before the next orchestrator Stage 3 pass refreshes state.

## Why not bump `state_migrate.py`?

`imgsz` has no universally-correct default. Writing `1920` (template
default) at migrate time would silently fix half the cases and break
the other half where the user configured something else. The correct
behaviour on resume is to re-run orchestrator Stage 3, which
re-resolves from yaml and writes the current value — which is what
already happens.

`CURRENT_DEFAULTS` in `state_migrate.py` unchanged.

## Files changed in v1.7.2

| File | Change |
|---|---|
| `research-orchestrator/SKILL.md` | +2 lines at end of Stage 3 Step 3: persist `imgsz` into state |
| `dataset-hunter/SKILL.md` | Read `state.get("imgsz", 1920)`, add to `patch_section_2` call |
| `autoresearch/SKILL.md` | Step 5.5c reads `state.get("imgsz")` with yaml fallback |
| `CHANGELOG_v1.7.2.md` | New (this file) |
| `README.md` | Version bumped |

## Files NOT changed

Everything else is byte-identical to v1.7.1.

Notably: `shared/weight_transfer.py`, `shared/modules_md.py`, every
`shared/templates/*`, `state_migrate.py`, all test files, all example
configs, `paper-finder/SKILL.md`. No behavioural change to any
primitive or helper.

## Test coverage

Unchanged from v1.7.1 — the fix is cross-skill state handoff, not
library code. Existing tests still green:

| Suite | Count |
|---|---|
| `shared/test_modules_md.py` | 12 |
| `shared/test_templates.py`  |  3 |
| `shared/test_weight_transfer.py` | 36 |
| **Total** | **51** |

SKILL.md python snippets: 83/83 parse.

## Upgrade path

Drop-in replacement. No manual migration:

```bash
unzip pipeline_v1.7.2.zip
cp -r pipeline_v1.7.2/skills/* ~/.claude/skills/
cp pipeline_v1.7.2/examples/* ~/.claude/skills/examples/
```

For in-flight projects resumed after upgrade:

- First orchestrator Stage 3 pass after upgrade writes `state["imgsz"]`
  for the first time. Before that pass, autoresearch's fallback re-reads
  from yaml — same result, slightly slower.
- Dataset-hunter runs started pre-v1.7.2 have already used the wrong
  IMGSZ; they won't be retroactively re-run. Their `pretrain_eval.json`
  decision is whatever it was. Next pretrain-eligible trigger will use
  the correct IMGSZ.
- If a previous pretrain decision was marginal (pretrain improved by
  ~2% but you suspect IMGSZ mismatch helped rather than hurt), consider
  forcing a re-pretrain by clearing `pretrain_attempt_failed` and
  `pretrain_eval.json`, then resuming. Judgement call — most drops or
  clear wins won't change sign.
