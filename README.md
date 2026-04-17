# Research Pipeline Skills

Autonomous ML experiment pipeline. Four Claude skills
(orchestrator, paper finder, dataset hunter, autoresearch) coordinate via
shared files to: find the best base model + improvement modules, build a
pretrain corpus, run pretrain, then iterate `train.py` through experiments
keeping what improves the primary metric.

Supports `task_type: object_detection` and `object_tracking`, driven entirely
by `research_config.yaml`. No skill hardcodes a specific metric.

---

## Layout

```
skills/                                     → copy to ~/.claude/skills/
├── shared/
│   ├── modules_md.py                       · modules.md parser (8 unit tests)
│   ├── train-script-spec.md                · train.py / track.py contract
│   ├── file-contracts.md                   · schemas for all cross-skill files
│   ├── test_modules_md.py                  · parser unit tests
│   ├── test_templates.py                   · template compliance checker
│   └── templates/
│       ├── train.py.detection              · scaffold for object_detection
│       ├── train.py.tracking               · detector-training half of tracking
│       └── track.py.tracking               · tracker + TrackEval half
├── research-orchestrator/SKILL.md     ← Stage 0–4 master controller
├── paper-finder/SKILL.md                   ← Stage 1: base_model.md + modules.md
├── dataset-hunter/SKILL.md        ← Stage 2: pretrain corpus + self-eval
└── autoresearch/SKILL.md          ← Stage 3: keep/discard experiment loop

examples/
├── research_config.visdrone-detection.yaml · annotated detection config
└── research_config.visdrone-mot.yaml       · annotated tracking config

.env.example                                · API key template (copy to project as .env)
CHANGELOG.md
```

---

## Quick start

### 1. Install

```bash
unzip research-skills-bundle-v1.5.zip
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
cp ~/.claude/skills/.env.example .env
```

Edit at minimum:
- `paths.skills_dir` → absolute path to your skills install
- `paths.dataset_root` → your YOLO-format dataset (must contain `data.yaml`)
- `task.description` → what you're trying to do (paper finder uses this)
- `.env` → fill in any API keys you have (all optional)

### 4. Start

In Claude Code (or desktop), open the project folder and say:

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
     ┌──────────────────── research-orchestrator ────────────────────┐
     │                                                                    │
     │  Stage 0: read yaml → .env → pipeline_state.json → scaffold       │
     │                                                                    │
     │  ┌─ Stage 1 ──┐    ┌─ Stage 2 ────┐    ┌─ Stage 3 ─────────┐      │
     │  │ paper      │    │ dataset       │    │ autoresearch      │      │
     │  │ finder     │    │ hunter        │    │ (loops forever)   │      │
     │  │            │    │               │    │                   │      │
     │  │ writes:    │    │ writes:       │    │ reads:            │      │
     │  │ base_model │───▶│ pretrain_eval │───▶│ modules.md        │      │
     │  │ modules.md │    │ pretrain_ckpt │    │ writes:           │      │
     │  │            │    │               │    │ results.tsv       │      │
     │  │            │    │               │    │ discoveries.md    │      │
     │  └────────────┘    └───────────────┘    └───────────────────┘      │
     │                                                                    │
     │  Stage 4: summary + discoveries.md (on Ctrl+C)                    │
     └────────────────────────────────────────────────────────────────────┘
```

Schemas for every cross-skill file are in `shared/file-contracts.md`.
Contract for `train.py` is in `shared/train-script-spec.md`.

---

## API keys

API keys live in `.env` in the project root, **not** in `research_config.yaml`
(which may be committed to git). Orchestrator loads `.env` at Stage 0 Step 1.5.

```bash
# .env
FIRECRAWL_API_KEY=fc-...   # deep scrape of PwC pages (paper finder + dataset hunter)
ROBOFLOW_API_KEY=...       # Roboflow Universe dataset downloads
HF_TOKEN=hf_...            # gated HuggingFace datasets
```

All keys are optional. Without them, the corresponding features degrade
gracefully (paper finder uses JSON API only, dataset hunter skips Roboflow, etc).

---

## Optional dependencies

| Tool | What it enables | Install |
|---|---|---|
| paper2code | Generates module code from arXiv papers | `npx skills add PrathamLearnsToCode/paper2code/skills/paper2code` |
| Firecrawl CLI | Deep scrape of PwC pages, weights URL validation, dataset URL discovery | `npm install -g firecrawl` + set `FIRECRAWL_API_KEY` in `.env` |

Without paper2code, autoresearch falls back to cloning GitHub repos or having
Claude write module code from the paper description. Without Firecrawl,
paper finder uses PwC's JSON API only and dataset hunter skips non-direct-download
datasets.

---

## What the pipeline guarantees

- **Reproducibility**: `TIME_BUDGET`, `SEED`, and `IMGSZ` are locked across
  all experiments. Changing any of them invalidates fair keep/discard comparison.
- **Correctness at first run**: orchestrator patches `WEIGHTS`, `DATA_YAML`,
  and `NUM_CLASSES` to match the user's actual paths before any experiment runs.
  Template placeholders are never left in place.
- **Atomicity**: each experiment is one coherent idea (a paper-recommended
  module's full hyperparameter package, or one hyperparameter change), not
  one line of code.
- **Autonomy**: no decision point pauses for user input. Every error has a
  predefined resolution. Observations go to `discoveries.md`, not to chat.
- **Metric-agnosticism**: whatever you put in `evaluation.metrics.primary`
  drives keep/discard. No skill hardcodes `val_mAP50_95`.
- **Architecture exploration**: after 5 stalled param-only experiments, the
  loop forces an architecture change. After 3+ architecture keeps, it
  combines them and re-runs pretrain.
- **Format stability**: `train.py` always reformats tool output into
  canonical `key: value` lines for stable regex parsing.

---

## Switching task types

Change `task.task_type` in the yaml and update `evaluation` to match:

```yaml
task:
  task_type: object_tracking    # scaffolds train.py + track.py
evaluation:
  tool: trackeval
  metrics:
    primary: HOTA
    minimize: [IDSW]
```

Dataset hunter's self-eval always uses detection mAP regardless of task type.

---

## Known limitations

- **Source search is detection-biased.** For tracking tasks, paper finder
  works (queries are user-supplied) but dataset hunter only knows detection
  dataset sources.
- **Resume is stage-level, not substep.** A crash inside paper finder
  restarts it from scratch.
- **`base_model.md` is regex-parsed.** A YAML frontmatter refactor would be
  more robust.
- **No LLM/segmentation support.** All skills assume ultralytics-YOLO.

---

## Versions

See `CHANGELOG.md` for full history.

- **v1.5** (current): dry-run bug fixes — `DATA_YAML` and `WEIGHTS` patching,
  `imgsz` ordering, `skills_dir` docs, `paper2code` fallback.
- **v1.4**: external skill integration — HF Dataset Viewer preflight,
  Firecrawl deep scrape, API keys moved to `.env`.
- **v1.3**: `discoveries.md` mechanism, monkey-patch prohibition, expanded
  crash diagnosis.
- **v1.2**: paper-faithful module application, annotated yaml examples.
- **v1.1**: 5 runtime bug fixes — download patience, forced architecture
  exploration, full autonomy, `IMGSZ` locked, `results.tsv` alignment.
- **v1.0**: initial unification — parser, spec, contracts, metric-agnostic state.
