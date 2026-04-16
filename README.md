# YOLO Research Pipeline Skills

Autonomous ML experiment pipeline for YOLO-family models. Four Claude skills
(orchestrator, paper finder, dataset hunter, autoresearch) coordinate via
shared files to: find the best base model + improvement modules, build a
pretrain corpus, run pretrain, then iterate `train.py` through experiments
keeping what improves the primary metric.

Supports `task_type: object_detection` and `object_tracking`, driven entirely
by `research_config.yaml`. No skill hardcodes a specific metric.

---

## Layout

```
skills/                                     → copy this to ~/.claude/skills/
├── shared/                                 ← shared infrastructure
│   ├── modules_md.py                       · canonical modules.md parser
│   ├── test_modules_md.py                  · parser unit tests (8 cases)
│   ├── train-script-spec.md                · train.py / track.py contract
│   ├── file-contracts.md                   · schemas for all cross-skill files
│   ├── test_templates.py                   · template compliance checker
│   └── templates/
│       ├── train.py.detection              · default for task_type: object_detection
│       ├── train.py.tracking               · detector-training half of tracking
│       └── track.py.tracking               · tracker + TrackEval half
├── yolo-research-orchestrator/SKILL.md     ← Stage 0–4 master controller
├── paper-finder/SKILL.md                   ← Stage 1: base_model.md + modules.md
├── dataset-hunter-for-yolo/SKILL.md        ← Stage 2: pretrain corpus + self-eval
└── autoresearch-for-yolo/SKILL.md          ← Stage 3: keep/discard experiment loop

examples/
├── research_config.visdrone-detection.yaml · annotated detection task config
└── research_config.visdrone-mot.yaml       · annotated tracking task config

CHANGELOG.md                                · summary of all changes by version
```

---

## Quick start

### 1. Install

```bash
unzip yolo-skills-bundle-v1.2.zip
cp -r bundle/skills/* ~/.claude/skills/
```

### 2. Verify

```bash
cd ~/.claude/skills/shared
python3 test_modules_md.py    # expect: 8/8 passed
python3 test_templates.py     # expect: 3/3 passed
```

### 3. New project

```bash
mkdir my-project && cd my-project
cp ~/.claude/skills/examples/research_config.visdrone-detection.yaml research_config.yaml
# (or .visdrone-mot.yaml for tracking)
```

Edit at minimum:
- `paths.skills_dir` → absolute path to `~/.claude/skills`
- `paths.dataset_root` → your dataset's location
- `task.description` → what you're trying to do (paper finder uses this)
- `dataset_hunter.disk.budget_gb` → match your free disk

### 4. Start

In Claude (Code, desktop, or wherever), open the project folder and say:

```
start the pipeline
```

Orchestrator handles everything from there.

---

## How the skills talk

All cross-skill communication goes through files. No skill calls another directly.

```
                            ┌─────────────────────────────────┐
                            │        research_config.yaml     │
                            │  (user-authored, all settings)  │
                            └─────────────────┬───────────────┘
                                              ▼
     ┌──────────────────── yolo-research-orchestrator ────────────────────┐
     │                                                                    │
     │  Stage 0: read yaml → pipeline_state.json + scaffold train.py      │
     │                                                                    │
     │  ┌─ Stage 1 ──┐    ┌─ Stage 2 ────┐    ┌─ Stage 3 ─────────┐       │
     │  │ paper      │    │ dataset       │    │ autoresearch      │       │
     │  │ finder     │    │ hunter        │    │ (loops forever)   │       │
     │  │            │    │               │    │                   │       │
     │  │ writes:    │    │ writes:       │    │ reads:            │       │
     │  │ base_model │───▶│ pretrain_eval │───▶│ modules.md        │       │
     │  │ modules.md │    │ pretrain_ckpt │    │ writes:           │       │
     │  │            │    │               │    │ results.tsv       │       │
     │  │            │    │               │    │ run.log           │       │
     │  └────────────┘    └───────────────┘    └───────────────────┘       │
     │                                                                    │
     │  Stage 4: summary from pipeline_state.json (on Ctrl+C)             │
     └────────────────────────────────────────────────────────────────────┘
```

Schemas for every cross-skill file are in `skills/shared/file-contracts.md`.
Contract for `train.py` (and `track.py` for tracking) is in
`skills/shared/train-script-spec.md`. Skills fail loudly if either is violated.

---

## Switching task types

Change one field in `research_config.yaml`:

```yaml
task:
  task_type: object_detection     # or object_tracking
```

and update `evaluation.metrics` / `evaluation.parsing` to match the metrics
the chosen tool reports. Template scaffolding picks the right `train.py`
(and `track.py` for tracking).

Dataset hunter's self-eval **always uses detection mAP** (`val_mAP50_95` by
default) regardless of `task_type` — pretrain quality is fundamentally a
detector question, not a downstream-task question. Override via
`dataset_hunter.pretrain.eval_metric` if needed.

---

## Adding a new task type

1. Add a template at `skills/shared/templates/train.py.<newtype>` following
   `train-script-spec.md` (four sections, locked variables, `inject_modules`
   hook, canonical metric print format).
2. Extend Stage 0 Step 6 in `yolo-research-orchestrator/SKILL.md` with a
   branch for the new `task_type`.
3. Add the matching `evaluation.parsing.patterns` entries to your yaml.
4. Run `test_templates.py` — it picks up the new template and checks compliance.

---

## What the pipeline guarantees

These are the guarantees the v1.2 design enforces:

- **Reproducibility**: `TIME_BUDGET`, `SEED`, and `IMGSZ` are locked across
  all experiments. Changing any of them invalidates fair keep/discard
  comparison, so orchestrator patches them once and rejects edits.
- **Atomicity**: each experiment is one coherent idea (a paper-recommended
  module's full hyperparameter package, or one hyperparameter change), not
  one line of code. The pipeline doesn't mix unrelated changes in one run.
- **Autonomy**: no decision point pauses for user input. Every error has a
  predefined resolution (crash → revert + retry, stall → expand papers,
  missing weights → fallback). Designed for unattended overnight runs.
- **Metric-agnosticism**: skills never hardcode `val_mAP50_95` or any other
  specific metric. Whatever you put in `evaluation.metrics.primary` is what
  drives keep/discard, what `pipeline_state.best_primary_value` tracks, what
  the Stage 4 summary prints.
- **Architecture exploration**: when 5 consecutive param-only experiments
  fail to improve PRIMARY, the loop forces an architecture experiment. After
  3+ architecture changes are kept, the loop combines them and re-runs
  pretrain so module benefits show under proper pretrained features.
- **Format stability**: `train.py` writes canonical `<key>: <value>` lines at
  column 0; tools' raw output is reformatted inside the templates so
  `evaluation.parsing.patterns` only ever sees a stable contract.

---

## Known limitations

- **Source search is detection-biased.** For tracking tasks, paper finder
  still works (queries are user-supplied), but dataset hunter only knows
  detection dataset sources. This is acceptable per Bug #5's design:
  dataset hunter measures detector quality regardless of `task_type`.

- **Resume is stage-level, not stage-substep.** A crash inside paper finder
  restarts paper finder from scratch (idempotent). Dataset hunter doesn't
  resume from a specific failed dataset.

- **`base_model.md` parsing is regex-based.** Paper finder writes
  hand-formatted markdown that orchestrator scrapes with regex. A YAML
  frontmatter refactor (described in `file-contracts.md`) would be more
  robust.

- **No LLM/segmentation/diffusion support.** All four skills assume
  ultralytics-YOLO. To support other framework families, see CHANGELOG's
  "Considered but not done" section.

---

## Versions

See `CHANGELOG.md` for the full history.

- **v1.2** (current): paper-faithful module application — agent now applies
  all hyperparameters from a paper as one experiment, not split across loops.
  Annotated example yamls.
- **v1.1**: 5 dynamic-run bugs fixed — patience for downloads, forced
  architecture exploration, full autonomy (no user prompts), `IMGSZ` locked,
  `results.tsv` column alignment.
- **v1.0**: unification pass — modules.md parser, train-script spec, file
  contracts, metric-agnostic state, task-type aware templates.
