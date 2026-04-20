# Research Pipeline Skills

Autonomous ML experiment pipeline. Four Claude skills
(orchestrator, paper finder, dataset hunter, autoresearch) coordinate via
shared files to: find the best base model + improvement modules, build a
pretrain corpus, run pretrain, then iterate `train.py` through experiments
keeping what improves the primary metric.

Supports `task_type: object_detection` and `object_tracking`, driven entirely
by `research_config.yaml`. No skill hardcodes a specific metric.

**This is v1.7.1** — adds an adaptive repair loop (Step 5.5). When a
yaml_inject or hook experiment crashes, autoresearch now classifies the
error: code bugs (Tier 1) are patched and retried, shape/channel
mismatches (Tier 2) are adapted by auto-inserting 1×1 Conv adapters
around the experimental module, and architectural impossibilities
(transfer_weights strict, dtype mismatch) go straight to discard. Up to
3 repair attempts with a 120s short-test budget each; success requires
a first-epoch loss triple that's finite and positive (not NaN / not
Inf). Adapted experiments are marked `[adapted: ...]` in `results.tsv`
so paper-faithful and auto-adapted runs stay distinguishable. See
`CHANGELOG_v1.7.1.md`.

v1.7.1 also patches two v1.7 issues flagged in review: (Q2) hook mode's
silent-failure mode when used on a layer-insertion module is now called
out explicitly with reciprocal "asymmetric cost" guidance in
paper-finder; (Q3) a stale historical note about stall_count in
autoresearch has been removed.

Drop-in on top of v1.7: no state schema migration, no breaking
changes. Builds on v1.7's architectural injection (yaml_inject mode):
modules that insert a new layer mid-backbone work end-to-end via
`weight_transfer.py`, which generates a modified YAML, builds the
model, transfers weights from the base `.pt` per a computed layer_map,
and forces lazy modules to build before the optimizer captures
parameters. See `CHANGELOG_v1.7.md`.

v1.6 → v1.7 → v1.7.1 is a pure-addition chain: every existing code
path, contract, and state file is preserved. Modules without the
`Integration mode` field default to `hook` (the v1.6 path). State
files from v1.5 / v1.6 / v1.7 resume transparently via
`shared/state_migrate.py`.

---

## Layout

```
skills/                                     → copy to ~/.claude/skills/
├── shared/
│   ├── modules_md.py                       · modules.md parser (12 unit tests) [v1.7]
│   ├── state_migrate.py                    · pipeline_state.json schema migration [v1.6]
│   ├── parse_metrics.py                    · shared stdout/json/csv metric extractor [v1.6]
│   ├── weight_transfer.py                  · yaml_inject + pretrained transfer [v1.7]
│   │                                         + Step 5.5 repair primitives  [v1.7.1]
│   ├── train-script-spec.md                · train.py / track.py contract
│   ├── file-contracts.md                   · schemas for all cross-skill files
│   ├── test_modules_md.py                  · parser unit tests (12)
│   ├── test_templates.py                   · template compliance checker
│   ├── test_weight_transfer.py             · weight_transfer unit tests (36) [v1.7.1]
│   └── templates/
│       ├── train.py.detection              · scaffold for object_detection
│       ├── train.py.tracking               · detector-training half of tracking
│       ├── track.py.tracking               · tracker + TrackEval half
│       └── arch_spec.schema.json           · JSON schema for ARCH_INJECTION_SPEC [v1.7]
├── research-orchestrator/SKILL.md          ← Stage 0–4 master controller
├── paper-finder/SKILL.md                   ← Stage 1: base_model.md + modules.md
├── dataset-hunter/SKILL.md                 ← Stage 2: pretrain corpus + self-eval
└── autoresearch/SKILL.md                   ← Stage 3: keep/discard experiment loop
                                               + Step 5.5 Repair check     [v1.7.1]

examples/
├── research_config.visdrone-detection.yaml · annotated detection config
└── research_config.visdrone-mot.yaml       · annotated tracking config

CHANGELOG_v1.6.md                           · 26 bug fixes on top of v1.5
CHANGELOG_v1.7.md                           · yaml_inject feature addition
CHANGELOG_v1.7.1.md                         · Step 5.5 repair loop + Q2/Q3 patches
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
  B2, intentionally preserved in v1.6 → v1.7.1). The state schema reserves
  a `baseline_snapshot` field for the future fix.
- **modules.md section bodies** must not start lines with `## ` at column 0
  (parser limitation, documented in `modules_md.py`).
- **v1.7 — `yaml_inject` is detector-only.** No `track.py` equivalent. Tracker
  modules tagged `yaml_inject` are discarded with a log_discovery at dispatch.
- **v1.7 — `full_yaml` mode is reserved, not implemented.** Papers requiring
  full YAML replacement (BiFPN-style structural redesign) are discarded in
  Priority A dispatch. Wait for v1.8.
- **v1.7 — re-pretrain trigger doesn't count `yaml_inject` keeps.** The
  `architecture_keeps ≥ 3` counter in orchestrator Stage 3 Step 6.5 still
  only counts hook-mode keeps. Unlocking this is a v1.8 scope item.
- **v1.7 — Priority E combinations of two kept `yaml_inject` experiments are
  not merged into a single run.** E combines hook-mode keeps only.
- **v1.7.1 — repair Tier 2 only handles channel mismatches.** Spatial (H×W)
  mismatches are flagged un-adaptable and discarded; auto-inserting
  pooling/upsample layers would change the model's resolution in
  unpredictable ways. If a module needs a different insertion position,
  that should come from paper-finder's spec, not runtime adaptation.
- **v1.7.1 — repair Tier 3 (downsize module config) and Tier 4 (swap
  module) not implemented.** OOM still uses v1.6's halve-BATCH_SIZE path.

---

## Versions

See `CHANGELOG_v1.7.1.md`, `CHANGELOG_v1.7.md`, and `CHANGELOG_v1.6.md`
for recent release notes.

- **v1.7.1** (current): adaptive repair loop — Step 5.5 in autoresearch.
  `weight_transfer.py` gains ~350 lines of repair primitives (crash
  classification, shape probing, adapter planning, loss validation). Test
  count 19 → 36. Also patches hook-mode silent-failure warning (Q2) and
  removes misleading stall_count history (Q3) flagged in review of v1.7.
  Drop-in on top of v1.7.
- **v1.7**: architectural injection (`yaml_inject` mode) with pretrained
  weight transfer. New `weight_transfer.py` + 19 unit tests, new
  `arch_spec.json` contract, new `Integration mode` field in `modules.md`
  (warn-not-reject). Pure addition on top of v1.6 — no state or contract
  migration.
- **v1.6**: 26 bug fixes across crash-level, silent-failure, design, and
  detail categories. Notable: `safe_wget()` everywhere, full json/csv
  parsing implementations, autonomous Stage 4 stop triggers, pinned
  `inject_modules()` rebind contract, `consecutive_crashes` counter
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
