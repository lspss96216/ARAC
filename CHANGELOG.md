# Changelog

## v1.2 — Paper-faithful module application + annotated configs

### One coherent idea per iteration (refined Critical Rule #3)

Previously the rule said "one change per iteration", which Claude interpreted as
"one variable assignment per loop". This caused paper-recommended modules
(e.g. RFLA needs `USE_RFLA + proposal_size + scale_factors + label_assignment`)
to be split across 4 loops, each tested in isolation, all judged "no effect"
because the paper's actual configuration was never applied as a unit.

The rule now reads: **one coherent idea per iteration, where atomicity is at
the level of experimental intent, not lines of code**. A paper's recommended
configuration is one idea; combining it with unrelated hyperparameter tuning is
two ideas.

Implementation:
- Autoresearch Step 2 has a new § Paper-faithful module application subsection
  describing how to extract the full hyperparameter package from `modules.md`
  Integration notes, paper2code-generated `__init__` defaults, or paper tables.
- Paper finder Phase 5 now includes a Hyperparameters extraction step.
- Paper finder's `Integration notes` template requires listing every
  paper-specified parameter as `<NAME>: <value> (§<section>)`.
- Decision matrix added clarifying what counts as ONE vs TWO ideas.

### Annotated example yamls

Both `research_config.visdrone-detection.yaml` and `.visdrone-mot.yaml` now
have inline comments on every field:
- `[required]` / `[optional]` / `[tracking-only]` tags
- One-line purpose
- Allowed values or default

This is the easiest way to discover what each knob does without reading the
SKILL.md files.

---

## v1.1 — Five dynamic-run bug fixes

These five bugs only appeared during real pipeline execution. Each had a
specific root cause in a SKILL.md instruction (or its absence).

### Bug #1: Dataset downloads abandoned after ~10 minutes

Cause: dataset hunter SKILL.md never told Claude that long downloads are
expected. Claude treated 10-minute network silence as failure.

Fix: explicit patience directive ("10–60 minutes per dataset is normal") +
`timeout 3600` per individual download (1 hour ceiling). Orchestrator Stage 2
also got a "patience" header acknowledging Stage 2 is the longest stage.

### Bug #2: Loop preferred parameter tweaks over architecture changes

Cause: Priority B (zero-param) ran before C/D (architecture) and there was no
mechanism to escape the local optimum.

Fix:
- New `param_only_streak` counter in `pipeline_state.json`.
- After 5 consecutive param-only experiments without improvement, autoresearch
  skips Priority B and forces an architecture experiment.
- New "architecture combination + re-pretrain cycle": after 3+ architecture
  keeps accumulate, the loop combines them and signals orchestrator to re-run
  pretrain. This surfaces module benefits that only show with proper pretraining.

### Bug #3: Pipeline paused asking for user decisions

Cause: orchestrator had three "ask user" code paths (yaml-missing, optional
pretrain trigger, crash 3x). All broke unattended runs.

Fix:
- yaml-missing → stop with clear error, no interactive prompt
- optional pretrain → auto-trigger after 5 stalled loops, set `pretrain_offer_declined`
  to prevent infinite loops
- crash 3x → revert to last known good commit, halve `BATCH_SIZE`, continue
- New Critical Rule (orchestrator + autoresearch): "Never ask the user for
  decisions during a run."

### Bug #4: Agent silently changed `IMGSZ` between experiments

Cause: `IMGSZ` was in Section ② "Modifiable" — even with verbal "don't change",
Claude would lower it to fix OOM.

Fix:
- Spec moved `IMGSZ` from Modifiable to Locked, with explicit reason
  ("changing resolution invalidates metric comparisons")
- Orchestrator Stage 3 Step 3 lock loop now patches `IMGSZ` alongside
  `TIME_BUDGET` and `SEED`
- Yaml resolves `IMGSZ` from `evaluation.ultralytics_val.imgsz` (or trackeval)
- New autoresearch Critical Rule: "Never touch IMGSZ — halve BATCH_SIZE for OOM"
- All three templates moved `IMGSZ` to their locked block

### Bug #5: `results.tsv` columns misaligned

Cause: Step 8 said "append a tab-separated row matching the header" but gave
no concrete writer. Claude built rows from a dict in arbitrary key order, and
the values landed under wrong columns.

Fix: new `append_result(path, values)` helper that:
- reads the header row to get authoritative column order
- iterates header columns in order, looking up each in the values dict
- formats numbers consistently (`{:.4f}` for floats, str for ints)
- zero-fills missing metric columns (`0.0000`), empty for text columns
- sanitizes tabs/newlines from free-text fields

Step 8 now mandates using this helper and explicitly forbids hand-built TSV strings.

---

## v1.0 — Initial unification pass

Made four pre-existing skills agree on common contracts. Before this:
- Each skill assumed a different `train.py` structure
- Metric names were hardcoded in three places
- `modules.md` format was written one way, read another, grepped a third

### Shared infrastructure introduced

- **`shared/modules_md.py`** — canonical parser for `modules.md`. All read/write
  goes through this module. 8 unit tests.
- **`shared/train-script-spec.md`** — formal contract for `train.py` /
  `track.py`: four sections, locked vs modifiable variables, `inject_modules()`
  hook signature with idempotency requirement, metric output contract.
- **`shared/templates/`** — three reference implementations
  (`train.py.detection`, `train.py.tracking`, `track.py.tracking`).
- **`shared/file-contracts.md`** — schemas for `pipeline_state.json`,
  `pretrain_eval.json`, `base_model.md`, `modules.md`, `results.tsv`, `run.log`.

### Configuration: single source of truth

`autoresearch.metrics` was deleted; metric config now lives only in
`evaluation.metrics`:

```yaml
evaluation:
  metrics:
    primary:     HOTA
    tiebreak:    MOTA
    secondary:   [IDF1, IDSW]
    guard:       {num_params_M: 5, FPS: 20}
    min_improvement: 0.005
    minimize:    [IDSW]
```

Pipeline state uses generic field names: `primary_metric_name` records what
the metric is (string), `best_primary_value` records the running best (number).
No skill hardcodes `val_mAP50_95`.

### Task-type awareness

`task.task_type` (new field): `object_detection` or `object_tracking`.
- Stage 0 scaffolds the right template(s)
- Stage 3 locks `TIME_BUDGET`/`SEED` across all experiment scripts (train.py
  alone for detection; train.py + track.py for tracking)
- Autoresearch routes module changes to `train.py` (detector modules) or
  `track.py` (tracker modules) based on `Module.Location` field

### Bugs fixed at v1.0

- `json.dump(float("-inf"))` crash → state uses `None` sentinel
- TrackEval column-table output unmatchable → `track.py` reformats into
  canonical key:value lines
- Spec contradicted templates on "reformat vs forward" → spec updated, templates
  always reformat
- Dataset hunter self-eval broken for tracking → always derives from
  `train.py.detection` template, always measures detection mAP regardless of
  `task_type` (independent of `evaluation.metrics.primary`)
- Stage 3 Step 3 silently no-opped on missing variables → `lock_variable()`
  helper raises `RuntimeError` on 0 matches
- Autoresearch file routing was implicit → explicit
  `DETECTOR_LOCATIONS` / `TRACKER_LOCATIONS` mapping in Priority A

---

## Considered but not done

These would be significant architectural changes deferred for now:

- **Theme 5: source search broadened to non-detection benchmarks.** Paper
  finder and dataset hunter source lists are detection-biased. For tracking
  tasks, paper finder's user-supplied queries cover the gap; dataset hunter's
  detection-only sources are acceptable per Bug #5's design (it always
  measures detector quality).

- **Theme 7: stage-substep resume.** Each stage is currently atomic on resume.
  A crash inside paper finder restarts paper finder from scratch.

- **`base_model.md` YAML frontmatter refactor.** Currently parsed by regex.
  A structured format defined in `file-contracts.md` would make changes
  safer.

- **LLM / segmentation / diffusion task support.** Would require either
  forking the bundle (Route A: complete LLM-specific bundle) or rebuilding
  the architecture as framework-agnostic with plugins (Route B). Both have
  significant engineering cost; deferred until concrete LLM training
  requirements are defined.

- **Bundle install via `npx skills add` or similar package manager.**
  Currently install is `unzip + cp -r`.
