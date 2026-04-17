# Changelog

## v1.5 — Dry-run fixes + skill renaming

### Skill names generalized

All skills renamed to drop "yolo" from their identifiers, making the
pipeline framework-agnostic at the naming level:

| Old | New |
|---|---|
| `yolo-research-orchestrator` | `research-orchestrator` |
| `autoresearch-for-yolo` | `autoresearch` |
| `dataset-hunter-for-yolo` | `dataset-hunter` |
| `paper-finder` | `paper-finder` (unchanged) |

Frontmatter `name` and `description` fields updated accordingly.
All cross-skill path references (`$SKILLS_DIR/<name>/SKILL.md`) updated.

### Dry-run bug fixes

A "first-contact" dry run (simulating a fresh Claude that has never seen these
skills) uncovered 5 issues. Two were crash-level:

**[CRASH] DATA_YAML not patched (Issue #4)**

The `train.py` template ships with `DATA_YAML = "data/dataset.yaml"` — a
placeholder that doesn't match the user's actual dataset. Nothing in the
pipeline patched this to the real path. Every first run crashed with
`FileNotFoundError`.

Fix: new orchestrator Stage 3 **Step 2.5** searches `paths.dataset_root` for
`data.yaml` / `data.yml`, patches `DATA_YAML` and reads `nc` to patch
`NUM_CLASSES`. If no `data.yaml` is found, prints a warning with the exact
path that needs manual setup.

**[CRASH] WEIGHTS not downloaded (Issue #5)**

When `pretrain_skipped = true` (Stage 2 skipped), nobody downloaded the base
model weights from `base_model.md`'s URL. The template default
`weights/yolo26x.pt` doesn't exist. First run crashed with `FileNotFoundError`.

Fix: orchestrator Stage 3 **Step 2** rewritten. Now resolves weights from
three sources in priority order: `pretrain_weights` → `base_weights_local` →
`base_model.md` Weights URL (downloaded via `wget`). Patches `WEIGHTS` in
`train.py` to the resolved local path.

**[CODE] imgsz ordering (Issue #3)**

The `imgsz` variable was referenced in the lock loop before its resolve block.
A Claude reading top-to-bottom would hit `NameError`. Fix: moved resolve above
the lock loop in the same code block.

**[DOCS] skills_dir chicken-and-egg (Issue #1)**

Line 29 said "Read `skills_dir` from `pipeline_state.json`" but at startup
that file doesn't exist yet. Fix: clarified that first run reads from
`research_config.yaml`, resume reads from `pipeline_state.json`.

**[DOCS] paper2code made optional (Issue #2)**

Step 4 previously printed "MISSING" as if paper2code were required. Fix:
changed to "OPTIONAL" with explicit fallback strategy (GitHub clone or Claude
writes code from paper description).

---

## v1.4 — External skill integrations + API keys to .env

### HF Dataset Viewer preflight (dataset hunter)

Before downloading any HuggingFace dataset candidate, query the Dataset Viewer
API (`/info` + `/first-rows`) to check if the dataset has bounding box
columns. Filters out 60–80% of false-hit candidates in seconds, saving hours
of wasted download time. Falls back gracefully if the API is unavailable.

### Firecrawl deep scrape (paper finder + dataset hunter)

Three integration points, all optional (require `FIRECRAWL_API_KEY` in `.env`):

- Paper finder Phase 2: scrape PwC paper pages to extract GitHub repo URLs,
  benchmark tables, and pretrained weights links not available via JSON API.
- Paper finder Phase 4: validate base model weights URL is alive before
  writing to `base_model.md`.
- Dataset hunter Source 5: scrape PwC dataset pages to find real download
  URLs for datasets that would otherwise be skipped as "no direct download".

### API keys moved to .env

Secrets removed from `research_config.yaml` (which may be committed to git).
New `.env.example` template with three keys: `FIRECRAWL_API_KEY`,
`ROBOFLOW_API_KEY`, `HF_TOKEN`. Orchestrator loads `.env` at Stage 0 Step 1.5
via line-by-line parser into `os.environ`. Skills read via
`os.environ.get()`.

---

## v1.3 — Discoveries mechanism + crash diagnosis expansion

### discoveries.md

Replaces the pattern where Claude stops mid-loop to report findings to the
user. Now: observations go to `discoveries.md` (append-only markdown log),
the loop continues, and orchestrator prints discoveries at Stage 4 summary.

New `log_discovery(message, loop, category)` helper. Categories:
`observation` / `limitation` / `strategy_shift` / `bug_workaround`.

Added to `file-contracts.md` as a documented cross-skill file.

### Monkey-patch prohibition

`train-script-spec.md` § inject_modules() now explicitly forbids wrapping
`model.forward()` with decorators. Ultralytics doesn't save monkey-patched
weights to checkpoints. Required technique: subclass-based layer replacement.
Includes correct/incorrect code examples.

### Crash Diagnosis expanded

Split into "Actual crashes" (fix and rerun) and "Common observations that are
NOT crashes" (log to discoveries.md and continue). Six specific scenarios
documented with predefined resolutions, including "TIME_BUDGET only allows
N epochs" and "module didn't converge".

Critical Rule #13 strengthened: "Never talk to the user mid-loop."

---

## v1.2 — Paper-faithful module application + annotated configs

### One coherent idea per iteration

Critical Rule #3 redefined from "one code change" to "one coherent idea".
A paper's recommended module configuration (toggle + supporting hyperparameters)
is one experiment applied together, not split across multiple iterations.

New § Paper-faithful module application in autoresearch: three-step extraction
procedure (modules.md Integration notes → paper2code `__init__` defaults →
paper tables). Decision matrix for what counts as one vs two ideas.

Paper finder Phase 5 updated: must extract hyperparameters from papers.
Integration notes template now requires listing all paper-specified parameters.

### Annotated yaml examples

Both detection and tracking yaml examples have inline comments on every field
with `[required]` / `[optional]` / `[tracking-only]` tags.

---

## v1.1 — Five runtime bug fixes

### Bug #1: Downloads abandoned after ~10 minutes

Dataset hunter had no patience directive. Added explicit timing expectations
("10–60 min per dataset is normal") + `timeout 3600` per download.

### Bug #2: Loop preferred param tweaks over architecture

Added `param_only_streak` counter. After 5 consecutive param-only experiments
without improvement, skip Priority B and force architecture experiment.
Architecture combination + re-pretrain cycle after 3+ architecture keeps.

### Bug #3: Pipeline paused asking for user decisions

Three "ask user" paths removed. All error handlers now have predefined
resolutions. Critical Rule: "Never ask the user for decisions during a run."

### Bug #4: IMGSZ silently changed between experiments

`IMGSZ` moved from Modifiable to Locked in spec. Orchestrator locks it
alongside `TIME_BUDGET` and `SEED`. Autoresearch Critical Rule: "halve
BATCH_SIZE for OOM, never change resolution."

### Bug #5: results.tsv columns misaligned

New `append_result(path, values)` helper reads header order, fills columns
in header sequence, zero-fills missing metrics, sanitizes text fields.

---

## v1.0 — Initial unification

### Shared infrastructure

- `modules_md.py`: canonical parser for modules.md (8 unit tests)
- `train-script-spec.md`: four-section contract for train.py / track.py
- `file-contracts.md`: schemas for all cross-skill files
- Three reference templates in `shared/templates/`

### Single source of truth for metrics

`evaluation.metrics` in yaml is the only place metrics are defined.
Pipeline state uses generic `primary_metric_name` + `best_primary_value`.
No skill hardcodes any specific metric name.

### Task-type awareness

`object_detection` and `object_tracking` supported. Correct templates
scaffolded automatically. Module routing to `train.py` vs `track.py`
based on Location field.

---

## Considered but not done

- **LLM / segmentation / diffusion support.** Skills assume ultralytics-YOLO
  at the framework level. Naming is now generic; framework-specific content
  would need task.md plugins per the extensibility design discussed.
- **Stage-substep resume.** Currently stage-level only.
- **`base_model.md` YAML frontmatter refactor.** Still regex-parsed.
- **Package manager install.** Currently `unzip + cp -r`.
