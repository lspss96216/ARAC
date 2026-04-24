---
name: autoresearch
description: Autonomous ML experiment loop. Iteratively modifies a training script, runs experiments, and keeps only changes that improve the primary metric defined in research_config.yaml while respecting guard metrics. Reads modules.md to find paper-backed modules to test, calls paper2code to generate code, and triggers paper finder when experiments stall. Trigger on phrases like "start the experiment loop", "run autoresearch", "keep trying things", or "optimise my model".
---

# Autoresearch — Autonomous Experiment Loop

Autonomous experiment loop. Modify `train.py` → Run → Keep if better → Revert if not → Repeat forever.

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
   PARSING_CFG    = ev["parsing"]                          # full dict for B1 helper
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

7. **Initialise `discoveries.md`** — a log for observations that arise during
   experiments. This is the replacement for stopping to talk to the user:

   ```python
   if not pathlib.Path("discoveries.md").exists():
       pathlib.Path("discoveries.md").write_text(
           "# Discoveries\n\n"
           "Observations from the experiment loop. User reads these after the run.\n\n"
           "---\n\n"
       )
   ```

8. **Define `load_best_row()` helper (A2).** Step 1 (Review) and Step 7
   (Decide) need the current best PRIMARY, TIEBREAK, and full baseline row
   from a single consistent read:

   ```python
   import csv

   def load_best_row(results_tsv: str, PRIMARY: str, MINIMIZE: set) -> dict | None:
       """Read results.tsv and return the keep-row with the best PRIMARY.
       Ignores crash and discard rows. Returns None if no keep rows yet."""
       p = pathlib.Path(results_tsv)
       if not p.exists():
           return None
       rows = [r for r in csv.DictReader(open(p), delimiter="\t")
               if r.get("status") == "keep"]
       if not rows:
           return None
       def keyfn(r):
           try: return float(r[PRIMARY])
           except (KeyError, ValueError): return float("-inf")
       reverse = PRIMARY not in MINIMIZE
       rows.sort(key=keyfn, reverse=reverse)
       return rows[0]
   ```

9. **Define `log_discovery()` helper.** The loop calls this from multiple
   places to append to `discoveries.md` without stopping (D9 — atomic write):

   ```python
   from datetime import datetime

   def log_discovery(message: str, loop: int = 0, category: str = "observation") -> None:
       """Append to discoveries.md atomically. Never blocks on user."""
       VALID = {"observation", "limitation", "strategy_shift", "bug_workaround"}
       if category not in VALID:
           category = "observation"
       entry = (
           f"\n## Loop {loop} — {category}\n\n"
           f"{datetime.now().isoformat(timespec='seconds')}\n\n"
           f"{message.strip()}\n"
       )
       p = pathlib.Path("discoveries.md")
       existing = p.read_text() if p.exists() else (
           "# Discoveries\n\n"
           "Observations from the experiment loop. User reads these after the run.\n\n"
           "---\n"
       )
       tmp = p.with_suffix(".md.tmp")
       tmp.write_text(existing + entry)
       tmp.replace(p)   # atomic rename on POSIX
   ```

---

## Discoveries — the "don't stop, write it down" mechanism

During the loop you will notice things: a module injection technique doesn't
persist to checkpoint, time budget is too short for convergence, a particular
combination is promising, a paper's method doesn't apply to this architecture.

**These are valuable observations. They are NOT a reason to stop the loop.**

When you notice something, call `log_discovery(...)` (Setup Step 9) and continue.

Categories: `observation` / `limitation` / `strategy_shift` / `bug_workaround`

### Examples of what goes into discoveries.md (NOT into chat)

| You notice... | Write to discoveries.md | Then... |
|---|---|---|
| "Monkey-patch forward wrapping doesn't save weights to checkpoint" | `log_discovery("Monkey-patching model.forward() causes injected module weights to not persist in checkpoint. Switching to subclass-based injection for future modules.", loop, "bug_workaround")` | Switch injection method, continue loop |
| "TIME_BUDGET=1200 only allows 5 epochs at imgsz=1280" | `log_discovery("At imgsz=1280, 1200s budget yields ~5 epochs. Short runs may not converge fully, but relative comparison between experiments is still valid.", loop, "limitation")` | Continue — TIME_BUDGET is locked, you can't change it. The comparison is still fair because every run gets the same budget |
| "Freeze backbone + cosine LR is the best combo so far" | `log_discovery("freeze_backbone + cosine_lr combination gave best PRIMARY so far. Will try adding modules on top of this configuration.", loop, "strategy_shift")` | Adopt as new baseline, continue loop |
| "This module clearly needs more epochs to converge" | `log_discovery("Module X shows improving trend but didn't converge in TIME_BUDGET. Relative comparison still valid — if it can't beat baseline in equal time, discard.", loop, "observation")` | Discard if it didn't beat baseline. Equal time is fair |
| "I found 3 useful insights the user should know about" | Write all three to `discoveries.md` | **Continue the loop.** User reads `discoveries.md` when they check back |

### What NEVER goes into chat mid-loop

- Summaries of what you learned
- Questions about strategy direction
- Suggestions to change locked parameters
- Requests for user input on which approach to try next
- "Important findings" reports
- "Do you want me to continue?" — YES, always continue

The user will read `discoveries.md` when they stop the pipeline. Until then,
your job is to loop. If you find yourself composing a message to the user that
isn't a git commit message or a one-line "Loop N: keep/discard" status — stop,
write it to `discoveries.md` instead, and move to the next iteration.

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
  9. Update stall / crash / streak counters in pipeline_state
  10. Repeat
```

---

### Step 1 — Review

Read `results.tsv` and current `train.py` state.
Read `stall_count` from `pipeline_state.json` — do not recompute it here.

**Load the current best row once (A2)** and cache its values for use in
Step 7 (decide) and Step 9 (counter updates):

```python
best_row = load_best_row("results.tsv", PRIMARY, MINIMIZE)
best_PRIMARY  = float(best_row[PRIMARY])  if best_row else None
best_TIEBREAK = float(best_row[TIEBREAK]) if (best_row and TIEBREAK) else None
baseline_row  = best_row   # passed to guard_violated() in Step 7
```

`best_TIEBREAK is None` means either no keep row exists yet, or the yaml did
not declare a tiebreak metric. Both cases are valid; Step 7's tiebreak branch
skips when `best_TIEBREAK is None`.

---

### Step 2 — Ideate

Pick the next change using this priority order:

**Priority A — modules.md pending entries (if file exists)**

Read `modules.md` via the canonical parser — never grep for `Status: pending`, and
never parse the pipe-table format manually:

```python
import sys, pathlib, json, yaml
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
sys.path.insert(0, str(pathlib.Path(state["skills_dir"]) / "shared"))
import modules_md as mm

# v1.7.3 — pull preferred_locations from research_config.yaml so the
# yaml field actually influences iteration order. Without this, the yaml
# key was read only by paper-finder (for Stage 1 search filtering) and
# had zero effect on which pending module autoresearch picks first —
# making `preferred_locations: [backbone, neck, head, loss]` look
# meaningful but actually ignored.
_cfg = yaml.safe_load(pathlib.Path("research_config.yaml").read_text()) or {}
preferred = (_cfg.get("paper_finder", {})
                 .get("modules", {})
                 .get("preferred_locations")) or None

pending = mm.find_pending("modules.md", preferred_locations=preferred)
# Sort keys applied, in order:
#   1. Complexity (low < medium < high)
#   2. Location rank (index in preferred_locations; unlisted → after listed)
#   3. Write order in modules.md (stable tiebreak)
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
    # A5 — Unknown Location. DO NOT default to detector.
    # Defaulting would inject a tracker-shaped module into inject_modules(model),
    # which accepts a detector model — the resulting crash is opaque and
    # hard to diagnose from run.log alone. Discarding here is strictly safer
    # and tells the user via discoveries.md where the fix is needed.
    print(f"Unknown Location {location!r} for {chosen.name}; discarding "
          f"(paper finder should set Location to one of: "
          f"{sorted(DETECTOR_LOCATIONS | TRACKER_LOCATIONS)})")
    mm.update_status("modules.md", chosen.name, "discarded")
    log_discovery(
        f"Module {chosen.name!r} had unknown Location {location!r}. "
        f"Fix paper finder's Location vocabulary or edit modules.md by hand. "
        f"Valid locations: {sorted(DETECTOR_LOCATIONS | TRACKER_LOCATIONS)}",
        loop=state.get("loop_count", 0),
        category="observation",
    )
    chosen = None      # fall through to Priority B
```

`target_file` and `hook_function` are used below when generating the injection.
Both files share the Section ② / Section ③ structure from
`train-script-spec.md`, so the patching logic is identical — only the file path
and hook name differ.

#### Dispatch on Integration mode (v1.7)

Before applying any change, check the module's `Integration mode` field. This
decides whether autoresearch edits Section ②/③ code (hook mode) or writes an
`arch_spec.json` and flips `ARCH_INJECTION_ENABLED` (yaml_inject mode).

**Hook-mode scope limit (v1.7+).** Hook mode applies changes via
`inject_modules(model)`, which runs after the model is built — it can wrap a
layer's `forward`, replace a block with a subclass, or swap a loss. It
**cannot** insert a new parameterised layer between existing layers, because
any layer index shift invalidates pretrained weight alignment AND the
monkey-patched forward of an inserted layer is not serialised into the
checkpoint (weights vanish on save/reload). Symptom of mis-using hook this
way: training runs, loss looks normal, but metrics are identical to
baseline and the saved checkpoint is the baseline. Nothing crashes — the
loop completes, spends the full TIME_BUDGET, and discards as "no
improvement". This is the failure that motivated v1.7's `yaml_inject` mode.

If paper-finder's heuristic was wrong and CBAM-style "insert between layers"
modules end up as `hook`, the iteration will silently fail as described. v1.7's
Step 5.5f detects shape mismatches in hook mode and logs a `discoveries.md`
note prompting manual re-tagging to `yaml_inject`.

```python
mode = chosen.integration_mode   # parser returns "hook" for missing/blank

KNOWN_HOOK       = {"hook"}
KNOWN_YAML_INJECT = {"yaml_inject"}
KNOWN_FULL_YAML   = {"full_yaml"}   # v1.8+ — not implemented yet

if mode in KNOWN_HOOK:
    pass   # fall through to § Apply the change (hook path)
elif mode in KNOWN_YAML_INJECT:
    if target_file != "train.py":
        # yaml_inject only makes sense for detector modules. Tracker-location
        # modules with yaml_inject set were misconfigured by paper-finder —
        # discard with a clear reason.
        mm.update_status("modules.md", chosen.name, "discarded")
        log_discovery(
            f"Module {chosen.name!r} has Integration mode 'yaml_inject' "
            f"but Location {location!r} routes to {target_file}. "
            f"yaml_inject requires a detector Location (backbone/neck/head/loss).",
            loop=state.get("loop_count", 0),
            category="observation",
        )
        chosen = None   # fall through to Priority B
    # else: proceed to § Apply the change (yaml_inject path) below
elif mode in KNOWN_FULL_YAML:
    # Reserved for v1.8. weight_transfer.build_custom_model_with_injection
    # will raise NotImplementedError at train.py run time, but it's cleaner
    # to discard now and log the mismatch than to crash the run.
    mm.update_status("modules.md", chosen.name, "discarded")
    log_discovery(
        f"Module {chosen.name!r} has Integration mode 'full_yaml' — reserved "
        f"for v1.8+. Not supported in this release. Discarded.",
        loop=state.get("loop_count", 0),
        category="limitation",
    )
    chosen = None   # fall through to Priority B
else:
    # Unknown mode — parser already warned. Fall back to hook (the v1.6 default)
    # rather than skip, so forward-compat mode values don't block work.
    log_discovery(
        f"Module {chosen.name!r} has unknown Integration mode {mode!r}; "
        f"falling back to hook injection.",
        loop=state.get("loop_count", 0),
        category="observation",
    )
    mode = "hook"
```

#### Apply the change

Process the chosen module using the branch that matches its `paper2code` field
value (for code generation) and `Integration mode` (for code placement):

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
   the expected prose). Parse lines that look like `<name>: <value> (§<section>)`.

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

##### Branches (hook mode)

These branches apply when `mode == "hook"`. For `yaml_inject`, skip past
them to § Branches (yaml_inject mode) below.

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

4. Regardless of which hook branch above was taken:
   - Set `USE_<MODULE> = True` in `target_file` Section ② as the experiment
     change for this iteration.
   - Update the module's status via the parser (do not sed/grep/hand-edit
     `modules.md`):
     ```python
     mm.update_status("modules.md", chosen.name, "injected")
     ```
     Valid statuses: `pending / injected / tested / discarded`. The parser rejects
     typos at runtime, which is how we keep the four skills agreeing on status values.

##### Branches (yaml_inject mode, v1.7)

When `mode == "yaml_inject"`, autoresearch does **not** edit `custom_modules.py`
new classes or append `USE_*` flags. Instead:

1. **Get the lazy wrapper class** into `custom_modules.py` using whichever of
   paper2code / GitHub clone / manual-write applies — same code-acquisition
   logic as hook branches 1–3 above. The class must be a lazy wrapper
   conforming to `train-script-spec.md § Lazy-wrapper contract`:
   - Side-effect-free `__init__`
   - Inner module built on first `forward` using `x.shape[1]` as channel count
   - `.to(x.device)` applied at build time

   Register it in `custom_modules.py`'s `register_custom_modules()` function
   the same way any other module is registered.

2. **Extract the injection spec** from the module's `Integration notes` in
   `modules.md`. Paper-finder Phase 5 is responsible for writing these fields;
   autoresearch parses them here. Expected shape:

   ```
   ### Integration notes
   module_class: LazyCBAM
   position:     after_class: Conv
   scope:        backbone
   yaml_args:    [64]
   module_kwargs: {"kernel_size": 7}
   ```

   If any field is missing, mark the module `discarded` and
   `log_discovery` — don't invent values. Paper-finder should be fixed to
   write complete specs.

3. **Write `arch_spec.json`** (path per `ARCH_INJECTION_SPEC_FILE` in
   `train.py`'s Section ②; default `arch_spec.json`):

   ```python
   import json, pathlib
   spec = {
       "mode": "insertions",
       "insertions": [{
           "module_class": extracted_class_name,
           "position": extracted_position_dict,   # {"kind": ..., ...}
           "scope":    extracted_scope,
           "yaml_args":      extracted_yaml_args,
           "module_kwargs":  extracted_module_kwargs,
       }],
       "strict": True,
   }
   spec_path = pathlib.Path("arch_spec.json")   # matches ARCH_INJECTION_SPEC_FILE default
   spec_path.write_text(json.dumps(spec, indent=2))
   ```

4. **Flip `ARCH_INJECTION_ENABLED = True`** in `train.py` Section ② using the
   standard patch_variable helper from orchestrator (reuse the same regex).
   Leave all hook-mode `USE_*` flags at their current value — they still
   take effect after `build_custom_model_with_injection` returns the
   structurally modified model.

5. **Commit files together**: `arch_spec.json`, `train.py`, `custom_modules.py`.
   The spec file is git-tracked so `git reset --hard HEAD~1` on discard
   rewinds everything, including the spec. Previous spec content is restored.

6. **Update module status**:
   ```python
   mm.update_status("modules.md", chosen.name, "injected")
   ```

**Strict-mode crash handling.** If `weight_transfer` raises at train.py run
time ("transfer_weights strict mode: N layer_map entries transferred 0
tensors"), that counts as a crash in Step 7. autoresearch's standard crash
flow applies: increment `consecutive_crashes`, log to `run.log`, Step 6
verify extracts no metrics, Step 7 discards, Step 9 bumps counter. If the
paper-finder spec was genuinely wrong (e.g. `after_class: "Conv"` but the
base model has no Conv under that scope — also a crash path), the module
ends up `discarded` and won't be retried.

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

#### Tie-breaking — pick one, never ask

After walking Priority A → E, you may still end up with multiple equally
valid candidates (two modules at the same complexity+location rank, two
hyperparameters of similar plausibility, two combinations worth trying).
**You must pick one deterministically and continue.** Never stop to ask
the user which to try first.

Why: asking the user breaks autonomy (Rule #13). Worse, it trades a
definite ~20-minute iteration for an indefinite wait — the iteration
would almost certainly complete before the user gets back. Picking
"wrong" costs at most one iteration; picking "ask" costs an unbounded
amount of wall-clock time. **A deterministic wrong-seeming choice beats
the right choice obtained by asking.**

Apply these tie-breakers in order, stopping as soon as one resolves the
tie:

| # | Tie-breaker | Rationale |
|---|---|---|
| 1 | modules.md **write order** (earliest line wins, within matching complexity and location rank) | Matches `find_pending`'s stable sort; reproducible; paper-finder's discovery order is already a prioritised signal |
| 2 | **Lower complexity** first | Cheaper to test, so if wrong, discard sooner |
| 3 | `preferred_locations` order from `research_config.yaml` (already enforced by `find_pending`, listed here as reminder) | User's declared architectural priority |
| 4 | Alphabetical by module / description string | Deterministic across runs — same modules.md always yields same tie-break |

For non-modules.md ties (e.g. "which hyperparameter to tune first"):

| # | Tie-breaker | Rationale |
|---|---|---|
| 1 | **Lightest change** (smallest parameter delta, no architecture change) first | Cheap experiments come first |
| 2 | Change touching **more basic** settings (LR → augmentation → loss weight → freeze) | Gradient of invasiveness |
| 3 | Alphabetical by variable name | Deterministic fallback |

Whichever candidate you pick, **commit to it for this iteration**. If it
doesn't improve PRIMARY, Step 7 discards it and Step 2 of the next
iteration picks the next candidate. You never owe the user a
justification for which tied option you chose — the git log and
`results.tsv` record what was tried.

Things that are NOT tie-breakers — don't use these to justify stopping
to ask:

- "Which approach does the user think is more promising?" — user doesn't
  know; that's why they're running autoresearch
- "This is a strategic choice that needs human input" — no iteration is
  strategic; each is one data point
- "These two options test fundamentally different hypotheses" — good,
  pick one, the other will come up next iteration
- "VRAM is tight, should I use a smaller module first?" — yes, apply
  tie-breaker #2 (lower complexity), continue
- "I want to confirm the user agrees with my priority" — do not confirm,
  apply the rules above, continue

#### Forced architecture exploration

Track the type of each experiment (param vs architecture) in `pipeline_state.json`:

```python
# At the end of Step 2, after choosing the change:
# A1 — parenthesised to avoid stray `or` on a continuation line (SyntaxError).
is_architecture_change = (
    chosen is not None                                  # modules.md module = architecture
    or change_type in ("replacement", "additive", "combination")
)
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

**D5 — runner auto-detection.** Orchestrator Stage 0 Step 0 picks either
`uv run` or `python3` and stores it in `state["python_runner"]`. Use that
value rather than hardcoding `uv`:

```bash
TASK_TYPE=$(python3 -c "import json; print(json.load(open('pipeline_state.json'))['task_type'])")
RUNNER=$(python3 -c "import json; print(json.load(open('pipeline_state.json')).get('python_runner', 'uv run'))")

case "$TASK_TYPE" in
  object_detection)
    $RUNNER train.py > run.log 2>&1
    ;;
  object_tracking)
    $RUNNER train.py  > run.log 2>&1
    # train.py writes `trained_weights: <path>` sentinel; track.py reads it back
    $RUNNER track.py >> run.log 2>&1
    ;;
  *)
    echo "Unsupported task_type: $TASK_TYPE" >&2
    exit 1
    ;;
esac
```

Never pipe through `tee` — the parser in Step 6 reads `run.log` directly and
`tee` corrupts exit codes.

Each loop takes ~20 minutes (TIME_BUDGET) plus a few seconds for startup and eval.
Tracking adds maybe another minute for `track.py` inference + TrackEval.
If total runtime exceeds `TIME_BUDGET × 2`: kill and treat as crash.

---

### Step 5.5 — Repair check (v1.7.1)

**Purpose.** When `train.py` crashes, not every crash means "the experiment
was a bad idea". Some are code bugs (missing import, typo in lazy wrapper)
that have nothing to do with the paper's method. Others are shape / channel
mismatches that can be resolved by inserting small adapter layers around
the experimental module — the paper's core idea is kept, the glue code is
adjusted. v1.7.1 formalises the repair flow so one crash doesn't necessarily
mean discarding the whole iteration.

**When this step runs.** Only when Step 5 exited non-zero AND the current
experiment involves a module injection (hook or yaml_inject). If a plain
hyperparameter change crashes, skip this step and go straight to Step 7
discard — there's nothing to repair.

**Attempts.** Up to **3** repair attempts. Each uses a short test budget
(`REPAIR_TEST_BUDGET = 120s` — patches `TIME_BUDGET` for the retry only,
restored after). Success = run.log contains a first-epoch loss triple where
all three losses are finite positive (not NaN, not Inf). Failure to produce
valid loss after 3 attempts → treat as architectural incompatibility,
discard via Step 7.

If repair succeeds within 3 attempts, **re-run with the full TIME_BUDGET**
to produce the real experiment. The short-test run is throwaway — its
purpose is only to confirm "this architecture can run".

#### Step 5.5a — Classify the crash

```python
import sys, pathlib, json
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
sys.path.insert(0, str(pathlib.Path(state["skills_dir"]) / "shared"))
import weight_transfer as wt

stderr_tail = pathlib.Path("run.log").read_text()[-8000:]   # last ~8 KB is plenty
category = wt.classify_crash(stderr_tail)
```

Categories the classifier returns:

| Category | Meaning | Repair action |
|---|---|---|
| `tier1_missing_register` | `NameError: name 'X' is not defined` — lazy wrapper class not registered in `parse_model.__globals__` | Append class to `register_custom_modules()` in `custom_modules.py` |
| `tier1_missing_import`   | `ModuleNotFoundError` / `ImportError` | Add `import` line or `pip install` |
| `tier1_init_signature`   | `TypeError: ...__init__() got unexpected kwarg` / `missing required arg` | Read stderr's class name, inspect the class, adjust `yaml_args` or `module_kwargs` in `arch_spec.json` (yaml_inject) or the `USE_*` branch's call site (hook) |
| `tier1_syntax`           | `SyntaxError` in agent-written file | Read file, fix syntax |
| `tier2_shape_mismatch`   | `RuntimeError: Given groups=1, weight of size [...]` / tensor-size-mismatch / mat1/mat2 shape | Tier-2 adapter repair — see Step 5.5c |
| `oom`                    | `CUDA out of memory` | Existing v1.6 OOM path — halve `BATCH_SIZE`, not counted against repair attempts. |
| `unfixable_layer_map`    | `transfer_weights strict mode: N entries 0 tensors` | Go straight to Step 7 discard. The insertion spec is architecturally wrong (scope too wide, wrong class name, position disagreeing with the base model's layer layout). |
| `unfixable_weight_transfer` | `size mismatch for model.N.*` during state_dict load | Same — discard. |
| `unfixable_dtype_device` | 2D vs 3D conv confusion, CPU/GPU mixing | Discard. |
| `unknown`                | Didn't match any pattern | Discard (conservative default). |

Rule of thumb for the split: **if the fix would change the paper's method
(swap the module, change its hyperparameters, pick a different position),
it's unfixable. If the fix is glue code around the method, it's repairable.**

#### Step 5.5b — Tier 1 repair (code bugs)

Cheap fixes that don't touch the experimental variable. Examples:

```python
# tier1_missing_register — append class to custom_modules.py's registry.
# Do NOT change the class itself, only add it to the registry dict.
cls_name = re.search(r"name '(\w+)' is not defined", stderr_tail).group(1)
# Edit custom_modules.py: add cls_name to register_custom_modules()'s registry

# tier1_missing_import — add the import.
# Read ModuleNotFoundError's message, add "import X" at top of the file
# that imported it, or add X to the pip install step.

# tier1_init_signature — inspect the class signature and adjust the call site.
# For yaml_inject: edit arch_spec.json's yaml_args / module_kwargs.
# For hook: edit the USE_* branch in inject_modules() to match the signature.
```

**Fix commits go in their own commit**, separate from the experiment commit:

```bash
git add custom_modules.py
git commit -m "fix: register LazyX in custom_modules (repair attempt 1)"
```

This way, if the experiment is ultimately discarded, `git reset --hard HEAD~1`
rewinds only the experiment commit and the fix stays — future experiments
benefit from it. The experiment commit itself comes later (Step 8) after
the real full-budget run.

#### Step 5.5c — Tier 2 repair (shape / channel mismatch, yaml_inject only)

For `tier2_shape_mismatch`, probe the upstream / module / downstream shapes
and synthesize an adapter spec:

```python
from ultralytics import YOLO
# 1. Load the model WITHOUT the experimental insertion, just to read shapes
base = YOLO(state["base_weights_local"])

# 2. Find the insertion point. For after_class position, pick the first
#    match (or whichever one crashed — run.log's traceback tells you).
spec_path = pathlib.Path("arch_spec.json")
spec = json.loads(spec_path.read_text())
target_ins = spec["insertions"][0]        # usually only one

# 3. Probe upstream shape at the insertion point
#    (upstream_idx is where the module would be inserted AFTER)
upstream_idx = ...   # determined from the spec's position/scope
# v1.7.2 — orchestrator Stage 3 Step 3 writes state["imgsz"]. Fall back to
# re-reading research_config.yaml for pre-v1.7.2 state files on resume.
imgsz = state.get("imgsz")
if imgsz is None:
    _cfg = yaml.safe_load(pathlib.Path("research_config.yaml").read_text())
    _ev = _cfg.get("evaluation", {})
    imgsz = (_ev.get("ultralytics_val", {}).get("imgsz")
             or _ev.get("trackeval", {}).get("imgsz")
             or 1920)
upstream = wt.get_shape_at_index(base.model, upstream_idx, imgsz)

# 4. Probe the module in isolation to find its in/out shape
from custom_modules import *   # ensure class is registered
mod_in, mod_out = wt.probe_module_io(
    target_ins["module_class"],
    upstream,                             # assume it accepts upstream's shape
    target_ins["yaml_args"],
    target_ins["module_kwargs"],
)

# 5. Downstream expected shape is whatever the base layer right AFTER the
#    insertion point expects — probe at upstream_idx + 1 to get its OUTPUT
#    shape (which on an unmodified model equals its input shape topologically
#    when layers are channel-preserving; for non-preserving, use the layer's
#    input dimension directly).
downstream = wt.get_shape_at_index(base.model, upstream_idx + 1, imgsz)

# 6. Plan the adapter
plan = wt.plan_adapter(upstream, mod_in, mod_out, downstream)

if plan.needs_adaptation:
    new_spec = wt.extend_spec_with_adapters(spec, plan, insertion_idx=0)
    spec_path.write_text(json.dumps(new_spec, indent=2))
    repair_note = f"adapter: {plan.reason}"
else:
    # probe said no adapter needed but the run still crashed — likely
    # a deeper architectural issue. Treat as unfixable.
    repair_note = None
```

If `plan.needs_adaptation` is False **and** Step 5 crashed with shape
mismatch, Tier 2 can't help — fall through to Step 7 discard.

#### Step 5.5d — Short-test retry

After the repair, re-run `train.py` with a reduced budget to verify the
architecture can run at all:

```bash
# Patch TIME_BUDGET in train.py's Section ②, run, restore.
python3 -c "
import pathlib, re
src = pathlib.Path('train.py').read_text()
src = re.sub(r'(?m)^TIME_BUDGET\s*=.*', 'TIME_BUDGET = 120', src, count=1)
pathlib.Path('train.py').write_text(src)
"
$RUNNER train.py > run.log 2>&1

# Restore TIME_BUDGET — orchestrator's value lives in state["loop_time_budget"]
# (v1.7.6 fix: previous version read state["time_budget_sec"] which never
# existed; KeyError on every successful short-test that progressed to
# full-budget rerun. Latent since v1.7.1 because no real Step 5.5 short-test
# completed end-to-end before now.)
python3 -c "
import pathlib, re, json
state = json.load(open('pipeline_state.json'))
tb = state['loop_time_budget']
src = pathlib.Path('train.py').read_text()
src = re.sub(r'(?m)^TIME_BUDGET\s*=.*', f'TIME_BUDGET = {tb}', src, count=1)
pathlib.Path('train.py').write_text(src)
"
```

Check success with `wt.loss_first_value_is_valid`:

```python
ok, msg = wt.loss_first_value_is_valid(pathlib.Path("run.log").read_text())
if ok:
    # Architecture can run. Proceed to the REAL run with full TIME_BUDGET.
    print(f"[repair] short test passed: {msg}")
    # Run again with the restored TIME_BUDGET
    subprocess.run(["bash", "-c", f"{RUNNER} train.py > run.log 2>&1"], check=False)
    # Then fall through to Step 6 Verify
else:
    # Short test failed — record, retry (up to 3 attempts), or give up
    attempts += 1
    if attempts >= 3:
        print(f"[repair] 3 attempts exhausted; discarding")
        # fall through to Step 7 discard
    else:
        # back to Step 5.5a to classify this new crash
        ...
```

#### Step 5.5e — Record what was repaired

Successful repair runs append to the eventual `description` in `results.tsv`
(written in Step 8). Track repair notes through the attempts:

```python
# Track across attempts (local var, not persisted until Step 8)
repair_notes = []
# On each successful repair tier:
repair_notes.append("tier1: registered LazyCBAM")
repair_notes.append("adapter: pre-Conv 256→64, post-Conv 32→256")

# In Step 8:
description = "CBAM (yaml_inject)"
if repair_notes:
    description += f" [adapted: {'; '.join(repair_notes)}]"
# results.tsv row gets "CBAM (yaml_inject) [adapted: ...]"
```

This flags the keep/discard decision as **not paper-faithful** — the
experiment tested a module close to the paper's but not identical. Priority
E combinations in later iterations should prefer kept experiments without
`[adapted:` in description when there's a choice.

#### Step 5.5f — Hook-mode repair is Tier 1 only

Hook mode can't use Tier 2 (no YAML to inject adapters into). When a hook
crash is Tier 2 category, the real problem is usually that paper-finder
mis-classified the module — it should have been `yaml_inject`. Log to
`discoveries.md` and discard:

```python
log_discovery(
    f"Module {chosen.name!r} crashed with shape mismatch under hook mode. "
    f"This usually means the module inserts a new layer (shifting downstream "
    f"indices) and should be tagged Integration mode: yaml_inject in "
    f"modules.md. Manual fix: edit modules.md, set Status back to pending, "
    f"set Integration mode to yaml_inject, resume.",
    loop=state.get("loop_count", 0),
    category="observation",
)
# → Step 7 discard
```

#### Step 5.5g — When NOT to enter repair

- Plain hyperparameter changes (Priority B/C) that crash → straight to Step 7
- Combination experiments (Priority E) — discard whole combination rather
  than trying to fix; it's cheaper to test components individually
- OOM — goes through v1.6 OOM path (halve BATCH_SIZE in orchestrator's
  re-queue), not through repair. OOM is not a code bug.

---

### Step 6 — Verify

Extract metrics using the parsing rules from `evaluation.parsing`. The exact
extraction depends on `PARSE_SOURCE`. All three branches are implemented
(B1 fix — prior versions had json/csv as `...` stubs and silently crashed).

Use the shared helper:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(state["skills_dir"]) / "shared"))
import parse_metrics

# PARSING_CFG is the full evaluation.parsing dict loaded at Setup Step 1.
results = parse_metrics.extract(PARSE_SOURCE, PARSING_CFG)
```

If you prefer inline implementation (not using the shared helper), the logic
is equivalent to:

```python
import re, json as _json, csv as _csv

results = {}

if PARSE_SOURCE == "stdout":
    log = pathlib.Path("run.log").read_text()
    for metric_name, pattern in PARSE_PATTERNS.items():
        m = re.search(pattern, log)
        results[metric_name] = float(m.group(1)) if m else None

elif PARSE_SOURCE == "json":
    # B1 — full implementation (was `...` stub in v1.5)
    def _pluck(obj, path):
        cur = obj
        for key, idx in re.findall(r"([^.\[\]]+)|\[(\d+)\]", path):
            try:
                if key:   cur = cur[key]
                elif idx: cur = cur[int(idx)]
            except (KeyError, IndexError, TypeError):
                return None
        return cur
    data = _json.loads(pathlib.Path(PARSING_CFG["json_file"]).read_text())
    for name, dotted in PARSING_CFG["json_paths"].items():
        v = _pluck(data, dotted)
        try: results[name] = float(v) if v is not None else None
        except (TypeError, ValueError): results[name] = None

elif PARSE_SOURCE == "csv":
    # B1 — full implementation (was `...` stub in v1.5)
    with open(PARSING_CFG["csv_file"], newline="") as f:
        rows = [r for r in _csv.DictReader(f)
                if any((v or "").strip() for v in r.values())]
    last = rows[-1] if rows else {}
    for name, col in PARSING_CFG["csv_columns"].items():
        raw = (last.get(col) or "").strip()
        try: results[name] = float(raw) if raw else None
        except ValueError: results[name] = None

else:
    raise RuntimeError(f"Unsupported parsing.source: {PARSE_SOURCE!r}")
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

**Consecutive-crash handling (A3/C8).** After the decision for this iteration
is made (Step 7) and before returning to Step 1, update crash counters and
possibly halve `BATCH_SIZE`:

```python
import re, subprocess

state = json.loads(pathlib.Path("pipeline_state.json").read_text())
crash_pause_after = state.get("crash_pause_after", 3)

if status == "crash":
    state["consecutive_crashes"] = state.get("consecutive_crashes", 0) + 1
else:
    state["consecutive_crashes"] = 0

if state["consecutive_crashes"] >= crash_pause_after:
    # v1.7.6 — fixed crash-pause sequencing.
    #
    # Previous order: halve BATCH_SIZE in working tree → git reset --hard HEAD~1
    # Bug: `git reset --hard` reverts both the broken commit AND the halve in
    #      the working tree. Next loop reads the unchanged BATCH_SIZE from
    #      train.py, crashes again the same way, counter resets to 0 after 3,
    #      halve runs but is reset again — infinite loop with no observable
    #      forward progress.
    # Fix order: revert first → halve in clean working tree → commit halve so
    #            the smaller BATCH_SIZE survives any later reset.
    #
    # Also: v1.7.6 detects BATCH_SIZE=1 and stops further halving with a
    # discoveries log (Bug 5 from review). Previous behaviour halved 1 → 1
    # forever and did the revert dance every 3 crashes with no real change.

    # Step 1 — revert the broken commit FIRST (clean working tree)
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=False)

    # Step 2 — read current BATCH_SIZE (after revert) and decide what to do
    halved_any = False
    floored_any = False
    for script in ("train.py", "track.py"):
        p = pathlib.Path(script)
        if not p.exists():
            continue
        src = p.read_text()
        m = re.search(r"(?m)^BATCH_SIZE\s*=\s*(\d+)", src)
        if not m:
            continue
        current = int(m.group(1))
        if current <= 1:
            # Already at floor — halving would do nothing (1//2 = 0, max=1).
            # Don't write the file, don't commit a no-op. Log and let
            # autoresearch keep retrying — but the next 3 crashes won't
            # trigger another silent halve; they'll trigger another log.
            floored_any = True
            continue
        new_bs = max(1, current // 2)
        p.write_text(re.sub(
            r"(?m)^BATCH_SIZE\s*=.*$",
            f"BATCH_SIZE     = {new_bs}",
            src, count=1))
        halved_any = True

    # Step 3 — commit the halve so it survives future resets
    if halved_any:
        subprocess.run(["git", "add", "train.py"], check=False)
        subprocess.run(["git", "add", "track.py"], check=False)   # no-op if absent
        subprocess.run(
            ["git", "commit", "-m",
             f"crash-pause: halve BATCH_SIZE after {crash_pause_after} crashes"],
            check=False,
        )
        log_discovery(
            f"{crash_pause_after} consecutive crashes. Reverted last "
            f"experiment, halved BATCH_SIZE, committed. Continuing loop.",
            loop=state.get("loop_count", 0),
            category="bug_workaround",
        )
    elif floored_any:
        # Already at floor. Don't pretend to fix anything — flag clearly
        # so the user can intervene (smaller image size, fewer epochs,
        # different model). Counter still resets to avoid loop spam.
        log_discovery(
            f"{crash_pause_after} consecutive crashes but BATCH_SIZE "
            f"already at 1. Cannot reduce further. Possible causes: model "
            f"too large, IMGSZ too high, NaN gradients, broken module. "
            f"Manual intervention recommended; loop continues anyway.",
            loop=state.get("loop_count", 0),
            category="limitation",
        )

    state["consecutive_crashes"] = 0

pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
```

This is the sole implementation of `autoresearch_crash_pause_after`. Without
this block the yaml setting is advertised but inert.

---

### Step 7 — Decide

Build helper that respects each metric's direction:
```python
def is_better(new, old, name):
    if old is None: return True
    return (new < old) if name in MINIMIZE else (new > old)

def guard_violated(new, baseline):
    """baseline is the full row dict returned by load_best_row(). When baseline
    is None (first iteration) there are no guards to violate."""
    if baseline is None:
        return None
    for name, tol_pct in GUARD.items():
        if name not in new or name not in baseline:
            continue
        try:
            b = float(baseline[name])
        except (TypeError, ValueError):
            continue
        if name in MINIMIZE:
            # smaller is better → violation if grew by more than tolerance
            if new[name] > b * (1 + tol_pct / 100):
                return name
        else:
            # larger is better → violation if dropped by more than tolerance
            if new[name] < b * (1 - tol_pct / 100):
                return name
    return None
```

**Decision rule (v1.7.7 explicit pseudo-code; C7 — crash is checked FIRST):**

The decision is a strict ladder: each rule fires in order, the first one
that matches wins. Earlier vague wording let agents reach different
keep/discard verdicts on identical results. v1.7.7 fixes the formula.

```
Inputs (per iteration):
  primary_delta  = signed_improvement(results[PRIMARY], best_PRIMARY, PRIMARY)
                   # positive = better; negative = worse
                   # for a MAXIMIZE metric: results - best
                   # for a MINIMIZE metric: best - results
  tiebreak_delta = signed_improvement(results[TIEBREAK], best_TIEBREAK, TIEBREAK)
                   # None when TIEBREAK undefined or best_TIEBREAK None
  guard_failed   = guard_violated(results, baseline_row)  # metric name or None
  EPSILON        = ev["metrics"].get("regression_epsilon", MIN_IMPROVE / 2)
                   # default = half of MIN_IMPROVE

Decision (first matching rule wins):

  0. crash:                         status == "crash" OR results[PRIMARY] is None
                                    → DISCARD (revert)
  1. first iteration:               best_PRIMARY is None
                                    → KEEP (establishes baseline)
  2. guard violation:               guard_failed is not None
                                    → DISCARD (reason = guard_failed)
  3. clear primary improvement:     primary_delta >= MIN_IMPROVE
                                    → KEEP
  4. tiebreak rescue:               primary_delta >= -EPSILON
                                    AND tiebreak_delta is not None
                                    AND tiebreak_delta >= MIN_IMPROVE
                                    → KEEP (description gets " [tiebreak]" suffix)
  5. otherwise:                     → DISCARD
```

The key clarification (rule 4): a tiebreak rescue is allowed when the
primary did **not regress significantly** (delta within `-EPSILON`) and
the tiebreak metric did **clearly improve** (delta beyond `MIN_IMPROVE`).
The two thresholds use different scales because a primary slightly
worse + tiebreak clearly better is the "evidence is mixed but leans
slightly positive" case — typical of a borderline architecture change
that improved fine-grained quality without hurting coarse quality much.

```python
def signed_improvement(new, old, name):
    """Positive = better in the direction this metric prefers."""
    if old is None or new is None:
        return None
    return (old - new) if name in MINIMIZE else (new - old)
```

Use this helper in Step 7's evaluation. Returning `None` means
"undefined" (e.g. TIEBREAK metric not declared in yaml) — rule 4 then
cannot fire and we fall through to rule 5.

**EPSILON yaml field (v1.7.7 new, optional)**:

```yaml
evaluation:
  metrics:
    primary: val_mAP50_95
    tiebreak: val_mAP50
    min_improvement: 0.001
    regression_epsilon: 0.0005   # [v1.7.7 optional] default = min_improvement / 2
                                 # primary may regress by up to this and still
                                 # be eligible for tiebreak rescue (rule 4).
```

Older yamls without this field fall through to the default
(`MIN_IMPROVE / 2`), so existing pipelines retain the same behaviour as
the previous looser rule.

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

### Step 9 — Update counters and write to pipeline state

After logging, update `stall_count`, `param_only_streak`, `best_primary_value`,
`best_tiebreak_value`, then write to `pipeline_state.json`:

- If this round improved PRIMARY by `>= MIN_IMPROVE` (in the right direction): reset `stall_count = 0`
- Otherwise: `stall_count += 1`
- If this round was a param-only experiment that did not improve: `param_only_streak += 1`
- If this round was an architecture experiment (regardless of keep/discard): `param_only_streak = 0`
- Count architecture keeps: scan `results.tsv` for `keep` rows where description
  does not start with "tune_" / "lr_" / "aug_" / "freeze_" → `architecture_keeps`

```python
import json, pathlib, csv
from datetime import datetime

state = json.loads(pathlib.Path("pipeline_state.json").read_text())
state["stall_count"] = stall_count

# param_only_streak
if is_architecture_change:
    state["param_only_streak"] = 0
elif status != "keep":
    state["param_only_streak"] = state.get("param_only_streak", 0) + 1

# best primary / tiebreak (A2 — keep both in sync)
if status == "keep":
    prev_best = state.get("best_primary_value")
    if is_better(results[PRIMARY], prev_best, PRIMARY):
        state["best_primary_value"] = results[PRIMARY]
    if TIEBREAK and results.get(TIEBREAK) is not None:
        prev_tb = state.get("best_tiebreak_value")
        if is_better(results[TIEBREAK], prev_tb, TIEBREAK):
            state["best_tiebreak_value"] = results[TIEBREAK]
state["primary_metric_name"] = PRIMARY

# Count architecture keeps for combination trigger
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

**Stall book-keeping is delegated to orchestrator (C1/C2).** Autoresearch
writes `stall_count` here and only here; it never decides whether to reset
it or trigger a paper-finder expand. The orchestrator owns that state
machine — see orchestrator's Stage 3 Step 6 for the reset / expand logic
configured via `autoresearch.stall.force_test_reset` in
`research_config.yaml`. Step 2 Priority A automatically picks a pending
module on the next iteration if `modules.md` has one, so no "force test"
mechanism is needed from autoresearch's side.

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
8. **When stuck, trigger paper finder** — Orchestrator watches `stall_count` and
   triggers paper-finder expand. Autoresearch just increments the counter.
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
13. **Never talk to the user mid-loop.** The loop is fully autonomous. Every error has a
    predefined resolution (see § Crash Diagnosis). Every stall triggers automatic expansion.
    Every observation goes into `discoveries.md` — see § Discoveries.
    If you find yourself composing a multi-sentence message to the user, or asking a question,
    or reporting findings, or saying "do you want me to continue" — **STOP**. Write it to
    `discoveries.md` instead. Then start the next iteration. The user reads `discoveries.md`
    after the pipeline stops. Your only chat output during the loop is one-line status per
    iteration: `"Loop N: keep/discard — <description>"`.
14. **Never directly assign `layer.forward = wrapper`** — always use
    `layer.register_forward_hook(...)`. Direct `.forward` assignment bypasses
    PyTorch's `_call_impl` machinery (global hooks, `__call__` dispatch,
    fused/eval phase variations) and breaks during val phase under AMP.
    See `train-script-spec.md § Lazy-wrapper contract` and v1.7.7 #18.
15. **Never set `OPTIMIZER = "auto"`** (v1.7.7). ultralytics' 'auto' silently
    overrides any user-supplied `LR0` and `MOMENTUM`, only printing one log
    line. Any LR-tuning experiment under 'auto' silently no-ops and looks
    identical to baseline. The template's `train()` function raises
    `ValueError` if it detects 'auto' at runtime, but autoresearch must not
    even try to set it. Permitted values: `SGD`, `AdamW`, `Adam`, `RMSProp`,
    `NAdam`, `RAdam`. Switching between concrete optimizers is fine and is a
    legitimate experiment (e.g. SGD → AdamW for a momentum-vs-adaptive
    comparison) — just never `'auto'`.

---

## Crash Diagnosis

### How consecutive crashes are handled

Every crash increments `pipeline_state.consecutive_crashes`. Every
non-crash outcome resets it to 0. When the counter reaches
`crash_pause_after` (default 3, from `research_config.yaml →
orchestrator.error_policy.autoresearch_crash_pause_after`), the loop
(v1.7.6 sequence — order matters):

1. **Reverts the last broken commit FIRST** (`git reset --hard HEAD~1`),
   so the working tree starts clean
2. Reads post-revert `BATCH_SIZE`. If already at 1, logs a `limitation`
   discovery and skips the halve (Bug 2 fix); otherwise halves and writes
   back to `train.py` (and `track.py` if present)
3. **Commits the halve** with message `"crash-pause: halve BATCH_SIZE..."`
   so the new value survives any future reset (Bug 1 fix)
4. Resets `consecutive_crashes` to 0
5. Continues looping — never pauses for user input

This is the sole implementation of `autoresearch_crash_pause_after`. Step 6
contains the actual code block.

### Actual crashes (fix and rerun or revert)

- `loss = nan` → LR too high, reduce 5–10×
- Flat loss from step 1 → wrong `num_classes` or bad head init
- OOM → halve `BATCH_SIZE` or revert the last change
- Slow convergence → LR too low or warmup too long
- `ImportError` from custom_modules → check class name and import statement in train.py

### Common observations that are NOT crashes (log to discoveries.md, continue)

These are real technical insights that feel important enough to report to the
user. They are not. Write them to `discoveries.md` and keep looping.

**"Monkey-patch forward wrapping doesn't save weights to checkpoint"**
→ This is a known ultralytics limitation. Wrapping `model.forward()` with a
  decorator works for the current run but the weights aren't saved by
  `model.save()`. Solution: use subclass-based injection (create a new
  `nn.Module` subclass that replaces the target layer) instead of
  monkey-patching. Log the discovery, switch technique, continue.

**"TIME_BUDGET only allows N epochs, results may not converge"**
→ TIME_BUDGET is locked. You cannot change it. The key insight is: **relative
  comparison is still valid**. Both the baseline and the experiment get the
  same wall-clock. If a module can't beat baseline in equal time, it's either
  not helpful or too expensive — both are valid discard reasons. Log to
  discoveries.md, discard if didn't improve, continue.

**"Hyperparameter tuning has limited benefit at this time budget"**
→ Correct observation. This is why the pipeline has `param_only_streak` and
  forced architecture exploration — if param changes aren't helping after 5
  rounds, the loop automatically switches to architecture changes. You don't
  need to report this. The mechanism handles it.
