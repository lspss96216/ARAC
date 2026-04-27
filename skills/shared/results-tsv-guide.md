# Reading `results.tsv` — a mental model

This doc is for **users** reviewing a pipeline run, not for agents. If
you're Claude reading this: see `autoresearch/SKILL.md § Step 8 — Log`
instead.

## Why TSV, not CSV or JSON

Tab-separated keeps the file usable both as a database (`csv.DictReader`
in any analysis script) and as a terminal-friendly view (`column -t -s
$'\t' results.tsv | less -S`). Commas occur inside values (description
fields like `"lr0=0.005, momentum=0.937"`); tabs don't.

## Schema overview

The header is **dynamic** — each pipeline run defines columns in this
order, fixed at Stage 0:

```
loop  commit  <PRIMARY>  <TIEBREAK>  <SECONDARY...>  num_params_M  memory_gb  status  description
```

For a typical detection run:

```
loop  commit    val_mAP50_95  val_mAP50  val_precision  val_recall  num_params_M  memory_gb  status   description
0     7c2d4f9   0.2847        0.4523     0.6612         0.3924      55.6312       21.34      keep     vanilla baseline
1     a8b3e21   0.2729        0.4341     0.6502         0.3811      55.7244       21.44      discard  LazySimAM (4× backbone)
```

For a tracking run, primary metric is `HOTA` and secondary metrics may
include `MOTA`, `IDF1`, `AssA`. Same shape, different metric names.

## Per-column semantics

### `loop`

Iteration number, integer. Loop 0 is the **vanilla baseline** (v1.8 —
written by autoresearch's Loop 0 step before the iterative loop starts).
Every later loop is one experiment.

### `commit`

8-character git short hash of the commit that produced this row. Useful
for `git checkout <commit>` to inspect exactly what train.py looked like
during this experiment.

For Loop 0 vanilla baseline: this is the post-scaffold initial commit.
For agent-violation rows: this commit may not exist (Step 4 reverts on
violation); the column will be empty.

### Primary / tiebreak / secondary metric columns

The exact names come from `research_config.yaml → evaluation.metrics`.
Typical examples:

| Column | Type | Range | Direction |
|---|---|---|---|
| `val_mAP50_95` | float | 0.0–1.0 | higher is better |
| `val_mAP50` | float | 0.0–1.0 | higher is better |
| `HOTA` | float | 0.0–100.0 | higher is better |
| `IDF1` | float | 0.0–100.0 | higher is better |
| `loss` | float | 0.0+ | LOWER is better — listed in `evaluation.metrics.minimize` |

All values printed with 4 decimal places. Missing values (e.g. metric
not produced because the run crashed) are `0.0000` — distinguishable
from a real zero only by the `status` column.

### `num_params_M`

Total parameter count in millions. Useful for spotting:
- Architecture experiments that didn't actually inject (param count
  unchanged from baseline → likely hook-mode bug, see v1.7.7 #14)
- Resource accounting (a +5M params change may explain a memory_gb spike)

### `memory_gb`

Peak GPU memory observed during the run, parsed from ultralytics' run
log. Useful for spotting OOM-fallback (memory_gb stays low because
TaskAlignedAssigner moved to CPU, but training is 3-7× slower —
discoverable as low memory + low it/s in the run log).

### `status`

One of:

| Value | Meaning |
|---|---|
| `keep` | passed Step 7's decision rule; this experiment becomes the new baseline |
| `discard` | failed Step 7's rule; reverted via `git reset --hard HEAD~1` |
| `crash` | run did not complete; some metric columns may be 0.0000 |

Loop 0 vanilla baseline always has `status=keep` by definition (it IS
the baseline; nothing to compare against).

`discard` rows still get written to `results.tsv` so the experimental
record is complete — the git revert removes the code change, not the row.

### `description`

Structured per the format contract in `autoresearch/SKILL.md § Step 8`:

| Pattern | Meaning |
|---|---|
| `vanilla baseline` | Loop 0 only |
| `USE_<MODULE> (var=val, ...)` | hook-mode experiment |
| `<ModuleClass> [adapted: ...]` | yaml_inject experiment |
| `var=val (was old_val), ...` | hyperparameter sweep |
| `<exp_A> + <exp_B>` | combination experiment |
| `... [tiebreak]` | kept via Step 7 rule 4 (primary borderline + tiebreak rescued) |

## Common analysis patterns

### Find the cumulative path of keeps

```bash
awk -F'\t' '$NF~/keep|"keep"/ {print}' results.tsv
# Or with column for readable view:
grep -E "^[0-9]+\s+[a-f0-9]+.*\skeep\s" results.tsv | column -t -s $'\t'
```

### Diff two experiments by checkout

```bash
git diff <loop_5_commit>..<loop_15_commit> train.py
```

### Confirm Loop 0 was actually vanilla

The first row should have description `vanilla baseline` and
`num_params_M` matching the base model's published count (e.g. YOLO26X
≈ 55.6M). If `num_params_M` is higher than baseline, your "vanilla"
row already had injection enabled — re-scaffold and start fresh.

### Spot OOM-fallback runs

Low `memory_gb` (e.g. 12 GB on H100 80GB) AND `num_params_M` matching
an architecture experiment usually means TaskAlignedAssigner went to
CPU. The metric reported is from an under-trained run; comparison may
be unfair. Cross-check with the per-loop archive in `logs/loop_<N>.log`
for an `it/s` collapse.

### Count keep rate by category

```python
import csv
from collections import Counter
keeps = Counter()
totals = Counter()
for row in csv.DictReader(open("results.tsv"), delimiter="\t"):
    desc = row["description"]
    cat = ("hyperparameter" if "(was " in desc
           else "yaml_inject"   if " [adapted" in desc or "Lazy" in desc
           else "hook"           if desc.startswith("USE_")
           else "baseline"       if desc == "vanilla baseline"
           else "other")
    totals[cat] += 1
    if row["status"] == "keep":
        keeps[cat] += 1
for cat in totals:
    print(f"{cat}: {keeps[cat]}/{totals[cat]}")
```

## What to ignore

- `agent_violation` rows that have empty `commit` — these are bookkeeping,
  not real experiments
- Rows with all-zero metrics AND `status=crash` — same, the run never
  produced numbers
- Tiny absolute differences (< `evaluation.metrics.min_improvement`) —
  Step 7 already filtered these; if a `discard` row's primary is within
  noise of baseline that's expected behaviour

## Cross-references

- Format-level spec: `shared/file-contracts.md § results.tsv`
- Decision rule (when keep vs discard fires): `autoresearch/SKILL.md § Step 7 — Decide`
- Description format: `autoresearch/SKILL.md § Step 8 — Log`
