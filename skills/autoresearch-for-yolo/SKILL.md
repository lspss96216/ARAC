---
name: autoresearch for yolo
description: Autonomous ML experiment loop for YOLO models. Iteratively modifies a training script, runs experiments, and keeps only changes that improve the primary metric defined in research_config.yaml while respecting guard metrics. Reads modules.md to find paper-backed modules to test, calls paper2code to generate code, and triggers paper finder when experiments stall. Trigger on phrases like "start the experiment loop", "run autoresearch", "keep trying things", or "optimise my model".
---

# Autoresearch for YOLO — Autonomous Experiment Loop

Autonomous experiment loop for YOLO. Modify `train.py` → Run → Keep if better → Revert if not → Repeat forever.

**Objective:** Maximise the **primary metric** declared in `research_config.yaml → evaluation.metrics.primary`, while respecting all **guard metrics** declared in `evaluation.metrics.guard`.

The skill never hardcodes specific metric names. Whether the task is detection (`val_mAP50_95`), tracking (`HOTA`), or anything else, the loop reads the metric definitions at startup and uses them throughout.

Shared files read/written by this skill (`pipeline_state.json`, `modules.md`,
`results.tsv`, `run.log`) have their schemas documented in
`<skills_dir>/shared/file-contracts.md` — read it once at setup.

---

## When to Activate

- User invokes the skill directly
- User says "start the experiment loop", "run autoresearch", "keep trying things", "optimise my model"
- Any task requiring repeated training iterations with measurable outcomes

---

## Bounded Iterations

By default, loops until manually interrupted. To run exactly N iterations, specify:
```
Iterations: N
```
Each loop takes ~20 minutes (TIME_BUDGET) plus a few seconds for startup and eval.

---

## Setup Phase (do once)

0. **Read the train.py contract once.** Every edit autoresearch makes to
   `train.py` assumes the four-section layout, locked variables, and
   `inject_modules()` hook documented in the spec:
   ```bash
   cat "$SKILLS_DIR/shared/train-script-spec.md"
   ```
   Orchestrator already scaffolded `train.py` from a compliant template at
   Stage 0 Step 6, so the file is spec-compliant at entry.

1. **Load the evaluation config.** Read `research_config.yaml` and pull out the `evaluation` block:
   ```python
   import yaml, pathlib
   cfg = yaml.safe_load(pathlib.Path("research_config.yaml").read_text())
   ev = cfg["evaluation"]

   PRIMARY      = ev["metrics"]["primary"]                 # e.g. "val_mAP50_95" or "HOTA"
   TIEBREAK     = ev["metrics"].get("tiebreak")            # may be None
   SECONDARY    = ev["metrics"].get("secondary", [])
   GUARD        = ev["metrics"].get("guard", {})           # {name: tolerance_pct}
   MIN_IMPROVE  = ev["metrics"].get("min_improvement", 0.001)
   MINIMIZE     = set(ev["metrics"].get("minimize", []))   # metrics where smaller is better

   PARSE_SOURCE   = ev["parsing"]["source"]                # stdout / json / csv
   PARSE_PATTERNS = ev["parsing"].get("patterns", {})      # for stdout regex
   ```
   These variables are referenced throughout the loop. **Never hardcode `val_mAP50_95` or any other metric name** — always go through these variables.

2. **Propose a run tag** from today's date. Create branch:
   ```bash
   git checkout -b autoresearch/<tag>
   ```
   If no git: `git init && git add -A && git commit -m "init"`

3. **Read all in-scope files** end-to-end before touching anything.
   Identify: TIME_BUDGET, files not to edit, forbidden packages.

4. **Check for modules.md** — if it exists, read it now and note all entries with `Status: pending`.
   These are paper-backed modules available to test. If it does not exist, that is fine — proceed without it.

5. **Initialise `results.tsv`** — untracked, never `git add`. Generate the header dynamically from the evaluation config:
   ```python
   tracked_metrics = [PRIMARY] + ([TIEBREAK] if TIEBREAK else []) + list(SECONDARY) + list(GUARD.keys())
   tracked_metrics = list(dict.fromkeys(tracked_metrics))   # dedupe, preserve order
   header = ["loop", "commit"] + tracked_metrics + ["memory_gb", "status", "description"]
   pathlib.Path("results.tsv").write_text("\t".join(header) + "\n")
   ```
   `loop` is a sequential integer starting from 1, incremented each iteration.

6. **Confirm and go** — say "Setup complete. Tracking: primary=`{PRIMARY}`, guard=`{list(GUARD)}`" and immediately begin The Loop.

---

## The Loop

```
LOOP (FOREVER or N times):
  1. Review
  2. Ideate
  3. Modify train.py
  4. Commit
  5. Run
  6. Verify
  7. Decide: keep or discard
  8. Log
  9. Check stall → trigger paper finder if needed
  10. Repeat
```

---

### Step 1 — Review

Read `results.tsv` and current `train.py` state.
Read `stall_count` from `pipeline_state.json` — do not recompute it here.
Note the current best **PRIMARY** value from `results.tsv` for use in Step 7 and Step 9.

---

### Step 2 — Ideate

Pick the next change using this priority order:

**Priority A — modules.md pending entries (if file exists)**

Read `modules.md` via the canonical parser — never grep for `Status: pending`, and
never parse the pipe-table format manually:

```python
import sys, pathlib, json
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
sys.path.insert(0, str(pathlib.Path(state["skills_dir"]) / "shared"))
import modules_md as mm

pending = mm.find_pending("modules.md")   # already sorted low → medium → high
```

If `pending` is empty, fall through to Priority B. Otherwise take `pending[0]`.

#### Determine which file to edit

The module's `Location` field decides whether it goes in `train.py` (detector
modifications) or `track.py` (tracker modifications):

```python
chosen = pending[0]
DETECTOR_LOCATIONS = {"backbone", "neck", "head", "loss", "label_assignment"}
TRACKER_LOCATIONS  = {"tracker", "post_processing", "association", "reid"}

location = chosen.fields.get("Location", "").strip().lower()
if location in DETECTOR_LOCATIONS:
    target_file   = "train.py"
    hook_function = "inject_modules"
elif location in TRACKER_LOCATIONS:
    target_file   = "track.py"
    hook_function = "apply_tracker_modules"
    if not pathlib.Path("track.py").exists():
        # Detection-only projects have no track.py — such a module cannot be
        # applied. Mark discarded with a clear reason.
        mm.update_status("modules.md", chosen.name, "discarded")
        print(f"Skipping {chosen.name}: tracker module on detection-only project")
        # fall through to Priority B in this iteration
        chosen = None
else:
    # Unknown location — default to detector and log a warning
    print(f"WARN: unknown Location {location!r} for {chosen.name}; "
          f"defaulting to train.py + inject_modules")
    target_file   = "train.py"
    hook_function = "inject_modules"
```

`target_file` and `hook_function` are used below when generating the injection.
Both files share the Section ② / Section ③ structure from
`train-script-spec.md`, so the patching logic is identical — only the file path
and hook name differ.

#### Apply the change

Process the chosen module using the branch that matches its `paper2code` field value:

Before touching `train.py`, the layout rules in
`<skills_dir>/shared/train-script-spec.md` apply. Section ② is the only part of
the file autoresearch may edit; `TIME_BUDGET`, `SEED`, and `IMGSZ` inside
Section ② are locked by orchestrator (see § Contract surface).

##### Paper-faithful module application

A paper's "one module" typically involves a bundle of settings: a main toggle
plus supporting hyperparameters the paper specifies as a package. All of these
belong to **one experiment** per Critical Rule #3 — apply them together in one
commit, not spread across multiple loop iterations.

**Extraction procedure.** Before writing the patch, read both `modules.md` and
the generated code, and collect every parameter the paper recommends:

1. **From `modules.md`** — the `Integration notes` section lists named
   hyperparameters the paper author specified (see paper-finder's template for
   the expected prose). Parse lines that look like `<NAME>: <value> (§<section>)`.

2. **From the generated code** — if paper2code produced a class, scan its
   `__init__` for default values the author chose:
   ```bash
   grep -A 20 "class <ClassName>" <paper_slug>/src/model.py | head -40
   ```
   Defaults in `__init__` are the paper's recommended configuration — use them,
   not arbitrary values.

3. **From the paper's tables** — if `modules.md` cites a results table (e.g.
   "best config in Table 2"), those are the values to use. If uncertain, use
   the code defaults from step 2.

Build a single **module patch bundle** — a dict mapping Section ② variable
names to values:

```python
module_patch = {
    "USE_RFLA_NECK":        True,                     # main toggle
    "RFLA_PROPOSAL_SIZE":   8,                        # §3.2
    "RFLA_SCALE_FACTORS":   "[0.5, 1.0, 2.0]",        # Table 2
    "ANCHOR_MATCH_METHOD":  "'rfla'",                 # §3.3
}
```

All four variables go into **one commit**. `results.tsv` row's `description`
column should name the module (`"RFLA_neck"`), not list each parameter — the
git diff has the exact values.

**What still counts as "one idea" vs "multiple ideas":**

| Scenario | Verdict |
|---|---|
| RFLA module (toggle + 3 hyperparams from paper) | ONE idea — apply together |
| RFLA module + unrelated LR tuning | TWO ideas — do RFLA this iteration, LR next |
| CMC + NSA Kalman (two separate modules from `modules.md`) | TWO ideas — one per iteration |
| Hyperparameter sweep: LR ∈ {1e-3, 3e-4, 1e-4} | THREE ideas — each is one iteration |
| Priority E combination of 2 previously-kept modules | Explicit exception (Rule #3) |

If paper finder wrote the `Integration notes` incompletely and you only find
the toggle (no supporting hyperparams), use the module's code defaults as-is
— don't invent values, don't leave parameters at the template default when the
paper specified otherwise.

##### Branches

1. If `paper2code: yes` → run paper2code to generate the module code:
   ```
   /paper2code https://arxiv.org/abs/<id>
   ```
   From the generated `<paper_slug>/src/model.py`, extract the target class:
   ```bash
   grep -n "class <ClassName>" <paper_slug>/src/model.py
   ```
   Copy the class into `custom_modules.py` (create if not present). The template's
   `from custom_modules import *` in Section ① means you do not need to add an
   import to `target_file` — just ensure `ClassName` is defined in `custom_modules.py`.
   Append a new `USE_<MODULE> = False` line **plus all supporting hyperparameters**
   at the end of `target_file`'s Section ②, and add a matching branch inside its
   `hook_function` in Section ③. The branch must be idempotent when `USE_<MODULE>`
   is False — see the spec's § `inject_modules()` section for the full contract.

2. If `paper2code: yes (GitHub repo available)` → clone the repo and extract the class:
   ```bash
   git clone <code_url> /tmp/<repo_name>
   grep -rn "class <ClassName>" /tmp/<repo_name>/
   ```
   Copy the located class into `custom_modules.py`.
   Follow the same injection pattern as branch 1 (append flag in `target_file`'s
   Section ②, add branch in its `hook_function`).

3. If `paper2code: no` → write the module class manually in `custom_modules.py` based on the
   paper description in modules.md, using citation-anchoring comments (`# §3.2 — ...`).
   Follow the same injection pattern as branch 1.

4. Regardless of which branch above was taken:
   - Set `USE_<MODULE> = True` in `target_file` Section ② as the experiment
     change for this iteration.
   - Update the module's status via the parser (do not sed/grep/hand-edit
     `modules.md`):
     ```python
     mm.update_status("modules.md", chosen.name, "injected")
     ```
     Valid statuses: `pending / injected / tested / discarded`. The parser rejects
     typos at runtime, which is how we keep the four skills agreeing on status values.

After Step 5 (Run) completes, Step 6 (Verify) is responsible for flipping
`injected → tested` on success, or `injected → discarded` on crash/discard. The
three status-transition callsites for a given iteration are:
- Step 2 (Ideate, Priority A): `pending → injected`
- Step 6 (Verify): `injected → tested` OR `injected → discarded`

**Priority B — zero-param changes (if no pending modules, or as an interleave)**

Hyperparameters, LR schedule, loss weights, augmentation, freeze strategy. These cost nothing
and should be tried before adding any module.

**Priority C — replacement changes**

Swap one existing block for another of similar parameter count (net delta ≈ 0).

**Priority D — additive changes**

Only if A, B, C are exhausted. Prefer the lightest option that could plausibly help.

**Priority E — combinations**

After finding what works individually, combine compatible wins (max 2 things at once).

#### Forced architecture exploration

Track the type of each experiment (param vs architecture) in `pipeline_state.json`:

```python
# At the end of Step 2, after choosing the change:
is_architecture_change = chosen is not None  # modules.md module = architecture
                         or change_type in ("replacement", "additive", "combination")
is_param_change = not is_architecture_change

state = json.loads(pathlib.Path("pipeline_state.json").read_text())
streak = state.get("param_only_streak", 0)
```

**Rule:** If `param_only_streak >= 5` (five consecutive param-only experiments
with no improvement to PRIMARY), **force the next experiment to be architecture**:

- Skip Priority B entirely
- If modules.md has pending entries → pick one (Priority A)
- If no pending modules → try Priority C or D
- If all exhausted → trigger paper finder expand (same as stall)
- Reset `param_only_streak = 0` after any architecture experiment, regardless
  of whether it was kept

This prevents the loop from endlessly tweaking LR/augmentation when the
architecture is the actual bottleneck.

#### Architecture combination + re-pretrain cycle

When autoresearch has kept **3 or more architecture changes** (counted from
`results.tsv` rows with `status=keep` and architecture-type descriptions),
trigger a re-combination + re-pretrain cycle:

1. **Combine** the best-performing architecture changes into one experiment
   (Priority E, but forced — up to 3 changes at once)
2. If the combination is kept, **signal orchestrator** to re-run pretrain
   with the new architecture:
   ```python
   state["request_repretrain"] = True
   state["repretrain_reason"] = "architecture_combination"
   pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
   ```
3. Orchestrator picks up `request_repretrain` and re-runs dataset hunter
   Phase 5–6 (pretrain + self-eval) with the updated `train.py`. See
   orchestrator's Stage 3 Step 6.5.

This cycle surfaces architecture improvements that only show with proper
pretraining — a module that looks marginal on random init may be significant
with pretrained features.

---

### Step 3 — Modify

Make exactly the change decided in Step 2. One idea per experiment.

**Never modify `TIME_BUDGET`** — this value is set by the user in `research_config.yaml`
and locked into `pipeline_state.loop_time_budget`. If an experiment is slow, halve
`BATCH_SIZE` instead. Changing `TIME_BUDGET` invalidates fair comparison between runs.

**Never modify `SEED`** — the random seed must be identical across all runs so that
differences in the primary metric reflect the change being tested, not randomness.

---

### Step 4 — Commit
```bash
git add train.py custom_modules.py
[ -f track.py ] && git add track.py      # only exists for task_type: object_tracking
git commit -m "experiment: <description>"
```

---

### Step 5 — Run

The run command depends on `task_type` (read from `pipeline_state.json`). See
`<skills_dir>/shared/train-script-spec.md` § Task-type variants for why the split
exists.

```bash
TASK_TYPE=$(python3 -c "import json; print(json.load(open('pipeline_state.json'))['task_type'])")

case "$TASK_TYPE" in
  object_detection)
    uv run train.py > run.log 2>&1
    ;;
  object_tracking)
    uv run train.py  > run.log 2>&1
    # train.py writes `trained_weights: <path>` sentinel; track.py reads it back
    uv run track.py >> run.log 2>&1
    ;;
  *)
    echo "Unsupported task_type: $TASK_TYPE" >&2
    exit 1
    ;;
esac
```

If `uv` is unavailable, substitute `python`. Never pipe through `tee` — the
parser in Step 6 reads `run.log` directly and `tee` corrupts exit codes.

Each loop takes ~20 minutes (TIME_BUDGET) plus a few seconds for startup and eval.
Tracking adds maybe another minute for `track.py` inference + TrackEval.
If total runtime exceeds `TIME_BUDGET × 2`: kill and treat as crash.

---

### Step 6 — Verify

Extract metrics using the parsing rules from `evaluation.parsing`. The exact extraction
depends on `PARSE_SOURCE`:

```python
import re, json, csv

results = {}

if PARSE_SOURCE == "stdout":
    log = pathlib.Path("run.log").read_text()
    for metric_name, pattern in PARSE_PATTERNS.items():
        m = re.search(pattern, log)
        results[metric_name] = float(m.group(1)) if m else None

elif PARSE_SOURCE == "json":
    # Use ev["parsing"]["json_file"] and ev["parsing"]["json_paths"]
    ...

elif PARSE_SOURCE == "csv":
    # Use ev["parsing"]["csv_file"] and ev["parsing"]["csv_columns"]
    ...
```

If `results[PRIMARY]` is `None` → crash. Diagnose: `tail -n 50 run.log`.
- Trivial fix (typo, import): fix and rerun.
- Broken idea: log as `crash`, revert, move on.

If the experiment used a module from modules.md, update its status via the parser:
```python
# (mm already imported in Step 2; re-import if this is a fresh process)
mm.update_status("modules.md", "<Module Name>",
                 "tested" if experiment_ran_ok else "discarded")
```
- Run succeeded → `injected` → `tested`
- Run crashed or discarded → `injected` → `discarded`

---

### Step 7 — Decide

Build helper that respects each metric's direction:
```python
def is_better(new, old, name):
    if old is None: return True
    return (new < old) if name in MINIMIZE else (new > old)

def guard_violated(new, baseline):
    for name, tol_pct in GUARD.items():
        if name not in new or baseline.get(name) is None:
            continue
        if name in MINIMIZE:
            # smaller is better → violation if grew by more than tolerance
            if new[name] > baseline[name] * (1 + tol_pct / 100):
                return name
        else:
            # larger is better → violation if dropped by more than tolerance
            if new[name] < baseline[name] * (1 - tol_pct / 100):
                return name
    return None
```

**Decision rule:**

1. **First iteration** (`best_PRIMARY is None`) → **keep** unconditionally. There
   is no baseline to compare against; this run establishes it.
2. Else if `guard_violated(results, best_so_far)` returns a metric name → **discard**
   (log the violating metric name as the reason).
3. Else if `is_better(results[PRIMARY], best_PRIMARY, PRIMARY)` AND
   improvement magnitude `>= MIN_IMPROVE` → **keep**.
4. Else if PRIMARY tied (within `MIN_IMPROVE`) AND `TIEBREAK` exists AND
   `is_better(results[TIEBREAK], best_TIEBREAK, TIEBREAK)` → **keep**.
5. Otherwise → **discard**.

Discard = revert:
```bash
git reset --hard HEAD~1
```

---

### Step 8 — Log

```bash
git log --oneline -1
```

Use the `append_result` helper to write exactly one row. This helper reads the
header from `results.tsv` and fills columns **in header order**, so the output
is always aligned regardless of how `values` dict is constructed:

```python
import pathlib, json

def append_result(path: str, values: dict) -> None:
    """Append one row to results.tsv, columns aligned to the existing header.
    Missing metric columns get '0.0000'. Missing text columns get empty string.
    Extra keys in values are ignored."""
    p = pathlib.Path(path)
    header = p.read_text().splitlines()[0].split("\t")
    # Columns that are always text, not metrics
    TEXT_COLS = {"loop", "commit", "status", "description"}
    row = []
    for col in header:
        v = values.get(col)
        if v is None or v == "":
            # Missing value: zero-fill metrics, empty for text
            row.append("" if col in TEXT_COLS else "0.0000")
        elif isinstance(v, float):
            row.append(f"{v:.4f}")
        elif isinstance(v, int):
            row.append(str(v))
        else:
            # Sanitize: strip tabs and newlines from free-text fields
            row.append(str(v).replace("\t", " ").replace("\n", " "))
    with p.open("a") as f:
        f.write("\t".join(row) + "\n")

# Read loop count, increment
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
loop = state.get("loop_count", 0) + 1
state["loop_count"] = loop

# Get commit hash
import subprocess
commit = subprocess.check_output(["git", "log", "--oneline", "-1"],
                                 text=True).strip()[:8]

# Build values dict — keys must match the header column names exactly
append_result("results.tsv", {
    "loop":        loop,
    "commit":      commit,
    **results,                           # metric name → value from Step 6
    "memory_gb":   peak_vram_mb / 1024,  # convert MB to GB
    "status":      status,               # "keep" / "discard" / "crash"
    "description": experiment_description,
})
```

**Never construct TSV rows by hand.** Always go through `append_result` — it
guarantees column count and order match the header, and sanitizes free-text
fields. This prevents the misalignment that occurs when Claude builds a
tab-separated string from memory.

The `results.tsv` schema (dynamic header, value formats, sanitization rules)
is documented in `<skills_dir>/shared/file-contracts.md § results.tsv`.

---

### Step 9 — Update stall count and write to pipeline state

After logging, update `stall_count` and `param_only_streak`, then write to `pipeline_state.json`:
- If this round improved PRIMARY by `>= MIN_IMPROVE` (in the right direction): reset `stall_count = 0`
- Otherwise: `stall_count += 1`
- If this round was a param-only experiment that did not improve: `param_only_streak += 1`
- If this round was an architecture experiment (regardless of keep/discard): `param_only_streak = 0`
- Count architecture keeps: scan `results.tsv` for `keep` rows where description
  does not start with "tune_" / "lr_" / "aug_" / "freeze_" → `architecture_keeps`

```python
import json, pathlib
from datetime import datetime
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
state["stall_count"] = stall_count

# param_only_streak
if is_architecture_change:
    state["param_only_streak"] = 0
elif status != "keep":
    state["param_only_streak"] = state.get("param_only_streak", 0) + 1

# best primary
prev_best = state.get("best_primary_value")    # None on first iteration
if is_better(results[PRIMARY], prev_best, PRIMARY):
    state["best_primary_value"] = results[PRIMARY]
state["primary_metric_name"] = PRIMARY

# Count architecture keeps for combination trigger
import csv
arch_keeps = 0
if pathlib.Path("results.tsv").exists():
    reader = csv.DictReader(open("results.tsv"), delimiter="\t")
    for row in reader:
        if row.get("status") == "keep":
            desc = row.get("description", "")
            if not any(desc.startswith(p) for p in ["tune_", "lr_", "aug_", "freeze_"]):
                arch_keeps += 1
state["architecture_keeps"] = arch_keeps

state["last_updated"] = datetime.now().isoformat()
pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
```

**Do NOT call paper finder directly from here.**
Orchestrator monitors `pipeline_state.json` and handles the stall → paper finder handoff.
Autoresearch only writes the count; orchestrator decides when to act.

If `stall_count >= 10` AND `modules.md` has pending entries:
- Reset `stall_count = 5` (force-test a pending module next iteration)
- Update `pipeline_state.json` with the new count
- Continue the loop normally

---

## Critical Rules

1. **Loop until done** — Unbounded: loop until interrupted. Bounded: loop N times then summarise.
2. **Read before write** — Always read `results.tsv` and current `train.py` before modifying.
3. **One coherent idea per iteration** — Atomic at the level of *experimental
   intent*, not lines of code. A single paper-proposed module typically includes
   an on/off flag plus 2–4 supporting parameters that the paper specifies as a
   package (e.g. RFLA = USE_RFLA + proposal_size + scale_factors + label_assignment).
   Those belong to **one experiment**, applied together. The rule forbids mixing
   *unrelated* ideas in one run — do not combine "tune LR" with "try new attention
   module" in the same loop. Changes that originate from the same paper's
   recommended configuration are one idea, not multiple.

   See § Paper-faithful module application for how to extract the full parameter
   package from modules.md and apply it in one commit.
4. **Mechanical verification only** — `PRIMARY` is the only judge for keep/discard direction;
   `GUARD` metrics can override with a discard. No subjective calls.
5. **Automatic rollback** — Failed changes revert instantly.
6. **Simplicity wins** — Equal PRIMARY + lower guard = KEEP. Tiny PRIMARY improvement + guard
   violation = DISCARD.
7. **Zero-param first, but not forever** — Try B before C before D. Modules from modules.md (A)
   are tried early because they are paper-backed. However, if `param_only_streak >= 5`,
   skip Priority B entirely and force an architecture experiment — see § Forced architecture
   exploration. The pipeline must not get stuck endlessly tuning hyperparameters.
8. **When stuck, trigger paper finder** — After 10 stalled rounds with no pending modules, expand
   the module pool. Never stop to ask unless truly blocked.
9. **Never touch `TIME_BUDGET`** — it is fixed by the user. Every run must use the same
   time budget so results are comparable. This value is read-only for autoresearch.
10. **Never touch `SEED`** — random seed must be identical across all runs.
11. **Never touch `IMGSZ`** — image size is fixed by the user in `research_config.yaml` and
    locked in the template. Changing `IMGSZ` between runs makes metric comparisons invalid
    (a model at 1920 vs 640 is a completely different operating point). If you need to reduce
    memory, halve `BATCH_SIZE` instead — never change resolution.
12. **Never hardcode metric names in the skill body** — always reference `PRIMARY`, `TIEBREAK`,
    `SECONDARY`, `GUARD` loaded from `research_config.yaml` at setup time. The same skill must
    work unchanged for detection (`val_mAP50_95`), tracking (`HOTA`), segmentation, etc.
13. **Never ask the user for decisions.** The loop is fully autonomous. Every error has a
    predefined resolution (see § Crash Diagnosis). Every stall triggers automatic expansion.
    If you find yourself about to ask the user a question, there is a default — use it.

---

## Crash Diagnosis

- `loss = nan` → LR too high, reduce 5–10×
- Flat loss from step 1 → wrong `num_classes` or bad head init
- OOM → halve `BATCH_SIZE` or revert the last change
- Slow convergence → LR too low or warmup too long
- `ImportError` from custom_modules → check class name and import statement in train.py

---

## Output Format Reference

`train.py` must print metrics in a format matched by `evaluation.parsing.patterns`.

Example for `tool: ultralytics_val` (regex on stdout):
```
all      548     38759      0.42   0.38   0.41   0.245
Model Summary: 295 layers, 3.2M params, 8.7 GFLOPs
```

Example for `tool: trackeval`:
```
VisDrone COMBINED 0.412
MOTA  0.587
IDF1  0.612
IDSW  342
```

If `train.py` does not emit metrics in the format your `evaluation.parsing.patterns` expect,
adjust either the script's print statements or the regex — they are the contract between
`train.py` and this skill.
