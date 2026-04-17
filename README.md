# Research Pipeline Skills

Autonomous ML experiment pipeline. Four Claude skills
(orchestrator, paper finder, dataset hunter, autoresearch) coordinate via
shared files to: find the best base model + improvement modules, build a
pretrain corpus, run pretrain, then iterate `train.py` through experiments
keeping what improves the primary metric.

Supports `task_type: object_detection` and `object_tracking`, driven entirely
by `research_config.yaml`. No skill hardcodes a specific metric.

**This is v1.6** — see `CHANGELOG_v1.6.md` for the ~26 bug fixes applied on
top of v1.5 (crash-level, silent-failure, design/consistency, and details).
State files from v1.5 and earlier resume transparently via
`shared/state_migrate.py`.

---

## Layout

```
skills/                                     → copy to ~/.claude/skills/
├── shared/
│   ├── modules_md.py                       · modules.md parser (8 unit tests)
│   ├── state_migrate.py                    · pipeline_state.json schema migration [v1.6]
│   ├── parse_metrics.py                    · shared stdout/json/csv metric extractor [v1.6]
│   ├── train-script-spec.md                · train.py / track.py contract
│   ├── file-contracts.md                   · schemas for all cross-skill files
│   ├── test_modules_md.py                  · parser unit tests
│   ├── test_templates.py                   · template compliance checker
│   └── templates/
│       ├── train.py.detection              · scaffold for object_detection
│       ├── train.py.tracking               · detector-training half of tracking
│       └── track.py.tracking               · tracker + TrackEval half
├── research-orchestrator/SKILL.md          ← Stage 0–4 master controller
├── paper-finder/SKILL.md                   ← Stage 1: base_model.md + modules.md
├── dataset-hunter/SKILL.md                 ← Stage 2: pretrain corpus + self-eval
└── autoresearch/SKILL.md                   ← Stage 3: keep/discard experiment loop

examples/
├── research_config.visdrone-detection.yaml · annotated detection config
└── research_config.visdrone-mot.yaml       · annotated tracking config

CHANGELOG_v1.6.md                           · what changed in v1.6
```

---

## Quick start

### 1. Install

```bash
unzip pipeline_v1.6.zip
cp -r pipeline_v1.6/skills/*    ~/.claude/skills/
cp -r pipeline_v1.6/examples    ~/.claude/skills/
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

# .env — create from this template (all keys optional; features degrade gracefully)
cat > .env <<'EOF'
FIRECRAWL_API_KEY=
ROBOFLOW_API_KEY=
HF_TOKEN=
EOF
```

Edit `research_config.yaml` at minimum:
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

### 5. Stopping the run

Three ways to stop cleanly (all documented in orchestrator Stage 4 and yaml
`orchestrator.stopping`):

1. **Sentinel file** — drop a file at the path given by
   `orchestrator.stopping.stop_flag_file` (default `stop_pipeline.flag`).
   Orchestrator stops after the current loop and prints the summary. Best for
   unattended / cron-driven runs.
   ```bash
   touch stop_pipeline.flag     # from any shell, any time
   ```
2. **Bounded iterations** — set `autoresearch.loop.iterations: N` in yaml.
3. **Ctrl+C** — interactive interrupt from the terminal.

After any of the above, Stage 4 prints the summary (respecting re-pretrain
generations via `status=rebase` markers in results.tsv), dumps
`discoveries.md`, and marks state as `done`.

---

## How the skills talk

All cross-skill communication goes through files. No skill calls another directly.

```
                            ┌─────────────────────────────────┐
                            │        research_config.yaml     │
                            │  (user-authored, all settings)  │
                            └─────────────────┬───────────────┘
                                              ▼
     ┌──────────────────── research-orchestrator ────────────────────────┐
     │                                                                    │
     │  Stage 0: read yaml → .env → migrate state → scaffold train.py    │
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
     │  Stage 4: summary + discoveries.md                                │
     │  (triggered by Ctrl+C, stop flag, iteration cap,                  │
     │   or exhausted paper-finder expansions)                           │
     └────────────────────────────────────────────────────────────────────┘
```

Schemas for every cross-skill file are in `shared/file-contracts.md`.
Contract for `train.py` is in `shared/train-script-spec.md`.
Stage 4 trigger matrix is in `orchestrator.stopping` and documented in
orchestrator Stage 3 Step 6.75.

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
gracefully (paper finder uses JSON API only, dataset hunter skips Roboflow,
etc).

Note on Firecrawl (v1.6 change): there is no official `firecrawl` CLI binary.
Weights URL validation now uses curl HEAD + range-GET fallback. The
Firecrawl-based deep-scrape path in paper finder remains best-effort and is
skipped when the CLI is not installed. If you want Firecrawl, use the Python
SDK (`pip install firecrawl-py`) per the caveat in paper-finder Phase 2.

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
- **Guard-metric baseline can drift** across re-pretrain cycles (known issue
  B2, intentionally preserved in v1.6). The state schema reserves a
  `baseline_snapshot` field for the future fix.
- **modules.md section bodies** must not start lines with `## ` at column 0
  (parser limitation, documented in `modules_md.py`).

---

## Versions

See `CHANGELOG_v1.6.md` for the latest round of fixes.

- **v1.6** (current): 26 bug fixes across crash-level, silent-failure,
  design, and detail categories. Notable: `safe_wget()` everywhere, full
  json/csv parsing implementations, autonomous Stage 4 stop triggers,
  pinned `inject_modules()` rebind contract, `consecutive_crashes` counter
  actually wired up, `dataset` → `dataset_root` rename with auto-migration.
- **v1.5**: dry-run bug fixes — `DATA_YAML` and `WEIGHTS` patching,
  `imgsz` ordering, `skills_dir` docs, `paper2code` fallback.
- **v1.4**: external skill integration — HF Dataset Viewer preflight,
  Firecrawl deep scrape, API keys moved to `.env`.
- **v1.3**: `discoveries.md` mechanism, monkey-patch prohibition, expanded
  crash diagnosis.
- **v1.2**: paper-faithful module application, annotated yaml examples.
- **v1.1**: 5 runtime bug fixes — download patience, forced architecture
  exploration, full autonomy, `IMGSZ` locked, `results.tsv` alignment.
- **v1.0**: initial unification — parser, spec, contracts, metric-agnostic state.
