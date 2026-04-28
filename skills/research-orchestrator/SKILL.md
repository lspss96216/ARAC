---
name: research orchestrator
description: >
  Master controller for the autonomous research pipeline.
  Coordinates paper finder, dataset hunter, and autoresearch in sequence,
  passing outputs between them automatically. Trigger when the user wants to start the full
  research pipeline end-to-end, or says "start the pipeline", "run the full experiment",
  "start from scratch", or provides a task description and dataset.
---

# Research Orchestrator

Single entry point for the full autonomous research pipeline. You read this skill once,
then drive the entire workflow — invoking each sub-skill in order, passing their outputs
forward, and looping back when needed. Never stop or ask for confirmation between stages
unless a blocking error occurs.

## Skill precedence (v1.7.5) — read this first

**Always prefer local skills over any same-named skill loaded from elsewhere.**

This pipeline ships as a self-contained repo with four sub-skills in
`<skills_dir>/` (resolved from `research_config.yaml → paths.skills_dir`,
default `./skills`). The sub-skills are named exactly:

- `research-orchestrator/`
- `paper-finder/`
- `dataset-hunter/`
- `autoresearch/`

If you have access to other skills with similar names (e.g.
`yolo-research-orchestrator`, `dataset-hunter-for-yolo`,
`autoresearch-for-yolo`) from any other source, **ignore them for this
pipeline.** The local `<skills_dir>/` is the authoritative source. Before
invoking any sub-skill:

1. Read the SKILL.md from `<skills_dir>/<skill-name>/SKILL.md` directly —
   do not rely on skill name resolution
2. If a sub-skill with the expected name is missing from `<skills_dir>/`,
   stop and report — do not silently fall back to a similarly-named skill
   from a different source

This avoids the failure mode where a cloud-hosted skill with an older or
differently-scoped version of the pipeline contract gets invoked, causing
schema mismatches, path assumptions, and handoff contracts to disagree
between skills.

---

Shared files consumed/produced by this skill (`pipeline_state.json`,
`base_model.md`, `modules.md`, `pretrain_eval.json`, `results.tsv`) have their
schemas documented in `<skills_dir>/shared/file-contracts.md`. Read that file
once at startup so you know exactly what to write and what each downstream
skill expects to read.

---

## How sub-skills are invoked

Each sub-skill lives in `<skills_dir>/<n>/SKILL.md`.
On first run, `skills_dir` comes from `research_config.yaml → paths.skills_dir`.
On resume, read it from `pipeline_state.json` (which was populated from the yaml
at Stage 0 Step 2). Fallback: `~/.claude/skills`.

To invoke a sub-skill:
1. Read its SKILL.md with the `view` tool or `cat`
2. Follow its instructions as if it were your active skill
3. When it completes, return here and advance to the next stage

```bash
SKILLS_DIR=$(python3 -c "import json; print(json.load(open('pipeline_state.json'))['skills_dir'])" 2>/dev/null || echo "./skills")
cat $SKILLS_DIR/paper-finder/SKILL.md
cat $SKILLS_DIR/dataset-hunter/SKILL.md
cat $SKILLS_DIR/autoresearch/SKILL.md
```

---

## Helpers used throughout this skill

### `safe_wget` (A4)

The previous version called `subprocess.run(["wget", ...], check=True)`. A 404
on a dead base-model URL would raise `CalledProcessError` and crash the
pipeline mid-stage — worse, sometimes dropping a zero-byte file that future
resumes would mistake for success. Use this helper everywhere we download:

```python
def safe_wget(url: str, dest: str, timeout_sec: int = 1800) -> bool:
    """Download url to dest with wget -c. Return True on success, False on any
    failure. Removes partial/zero-byte outputs so callers can retry or fall
    back cleanly. Never raises — callers decide what to do on False."""
    import subprocess, pathlib
    dest_p = pathlib.Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["wget", "-q", "-c", "--tries=2", "--timeout=60",
             url, "-O", dest],
            timeout=timeout_sec,
        )
        if r.returncode != 0:
            print(f"wget failed for {url} (rc={r.returncode})")
            if dest_p.exists() and dest_p.stat().st_size == 0:
                dest_p.unlink()
            return False
    except subprocess.TimeoutExpired:
        print(f"wget timed out for {url}")
        if dest_p.exists() and dest_p.stat().st_size == 0:
            dest_p.unlink()
        return False
    return dest_p.exists() and dest_p.stat().st_size > 0
```

---

## Pipeline state file

Maintain `pipeline_state.json` in the project root throughout the run.
Never git add this file. Update it after every stage transition.

The full JSON schema lives in `<skills_dir>/shared/file-contracts.md` —
refer to it when adding fields or investigating serialization issues. Key
invariant: **only JSON-spec types** (no `float("inf")`, `datetime`, `Path`
objects). Sentinels for "no measurement yet" are `null`, not infinities.

C3 rename: the `"dataset"` key from pre-v1.6 state files is renamed to
`"dataset_root"` during Stage 0 resume via
`shared.state_migrate.migrate()`. All downstream references use
`state["dataset_root"]`.

**Metric-agnostic by design:** the state schema contains **no hardcoded metric names**.
Whatever the user declared in `research_config.yaml → evaluation.metrics.primary` gets
stored in `primary_metric_name`, and the running best value goes in `best_primary_value`.
All sub-skills read these two fields when they need the current metric.

Read this file at startup to resume a partially completed pipeline.

---

## Stage 0 — Startup

### Step 0 — Choose python runner (v1.7.5 reworked)

Some environments have `uv` installed, some only have `python3`. Prefer
the **local system python** unless the project already has a
`pyproject.toml` (which strongly implies the user wants uv-managed deps).
This avoids the failure mode where `uv run python3 train.py` launches in
a fresh empty venv and immediately crashes on `import numpy` because the
project never configured uv properly.

**Default preference**: system `python3` (most research machines already
have ultralytics / torch / etc. installed globally).

**Use `uv run` only when**:
1. `shutil.which("uv")` returns a path, AND
2. `pyproject.toml` exists in cwd, AND
3. `uv run python3 -c "import ultralytics"` actually succeeds (the uv env
   has the deps the pipeline needs)

Otherwise fall back to `python3` and either install missing packages
system-wide, or if the user explicitly wants uv and a `pyproject.toml`
is absent, generate a minimal one (see Step 6.5 below).

```python
import shutil, subprocess, pathlib, json

def choose_python_runner() -> tuple[str, str]:
    """Return (runner, reason) tuple.
    runner is 'python3' or 'uv run'. reason explains the choice for logging.
    """
    # Fast path 1: system python already has ultralytics → prefer it
    rc_sys = subprocess.run(
        ["python3", "-c", "import ultralytics"],
        capture_output=True,
    ).returncode
    if rc_sys == 0:
        return "python3", "system python3 has ultralytics installed"

    # Fast path 2: uv + pyproject.toml + uv env has ultralytics
    if shutil.which("uv") and pathlib.Path("pyproject.toml").exists():
        rc_uv = subprocess.run(
            ["uv", "run", "python3", "-c", "import ultralytics"],
            capture_output=True,
        ).returncode
        if rc_uv == 0:
            return "uv run", "uv-managed env has ultralytics installed"

    # Neither works out of the box — default to python3 and let Step 6.5
    # either install globally or scaffold a uv project (user's choice via
    # yaml flag `orchestrator.use_uv_project: true`).
    return "python3", (
        "neither system python3 nor uv env has ultralytics; "
        "defaulting to python3 — see Step 6.5 for remediation"
    )

python_runner, reason = choose_python_runner()
print(f"[orchestrator] python_runner = {python_runner!r} ({reason})")
# Written to state in Step 2. Autoresearch Step 5 reads state["python_runner"].
#
# v1.7.6 — On resume, this overwrites whatever value state had previously.
# Rationale: machine state may have changed between runs (uv installed,
# ultralytics installed in system python, venv broken, etc.). Re-detection
# every Stage 0 means stale `python_runner` in state files (e.g. pre-v1.7.5
# defaults of "uv run") cannot lock the pipeline into a broken runner.
```

### Step 1 — Check for `research_config.yaml`

```bash
[ -f "research_config.yaml" ] && echo "Config found" || echo "Config missing"
```

- **Found** → read it with:
  ```python
  import yaml
  cfg = yaml.safe_load(open("research_config.yaml"))
  ```
  All values from the yaml are the authoritative source — they override any defaults
  in the individual skills.
- **Not found** → stop and report:
  ```
  ERROR: research_config.yaml not found in project root.
  Copy an example from <skills_dir>/shared/templates/examples/ and edit it.
  ```
  Do not ask the user for fields interactively — the yaml is the single source
  of configuration. Without it, the pipeline cannot start.

### Step 1.5 — Load `.env` for API keys

Secrets (API keys, tokens) live in a `.env` file in the project root, NOT in
`research_config.yaml` (which may be committed to git). Load it before state init:

```python
import pathlib, os

env_path = pathlib.Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")   # strip optional quotes
        os.environ.setdefault(key, value)     # don't override existing env vars
```

Expected `.env` fields (all optional — features degrade gracefully without them):

```bash
# .env — API keys for the research pipeline
# This file is gitignored. Never commit secrets to the repo.

FIRECRAWL_API_KEY=fc-...       # Enables deep scrape of PwC pages + weights URL validation.
                                # Free 500 credits/month at firecrawl.dev. Without it,
                                # paper finder uses JSON API only; dataset hunter Source 5
                                # skips non-direct-download datasets.

ROBOFLOW_API_KEY=...           # Enables Roboflow Universe dataset downloads.
                                # Free at roboflow.com. Without it, Source 2 is skipped.

HF_TOKEN=hf_...                # Enables access to gated HuggingFace datasets.
                                # Without it, only public datasets are downloadable.
```

Skills access keys via `os.environ.get("FIRECRAWL_API_KEY")` — never from
`pipeline_state.json` or `research_config.yaml`. This keeps secrets out of
state files that might be shared or accidentally committed.

### Step 2 — Initialise or resume `pipeline_state.json`

Check if `pipeline_state.json` exists:

#### `skills_dir` resolution helper (v1.7.5)

Before either branch below, resolve `skills_dir` once. The yaml default is
`./skills` (relative to `research_config.yaml`) but users may pass `~/...`
or absolute paths. Do **not** assume the caller has already
`expanduser`'d the value.

```python
def resolve_skills_dir(raw: str, yaml_path: str = "research_config.yaml") -> str:
    """Resolve skills_dir from research_config.yaml into an absolute path.

    Handles:
      - absolute paths (returned as-is after normalisation)
      - ~ / $HOME expansion
      - relative paths (resolved against the yaml's directory, NOT cwd)

    Fails loudly if the resolved path does not exist OR does not contain
    a `shared/` subdirectory — a silent misconfiguration here causes every
    subsequent skill to fail with confusing ImportErrors.
    """
    import os, pathlib
    expanded = os.path.expanduser(os.path.expandvars(raw))
    p = pathlib.Path(expanded)
    if not p.is_absolute():
        # Resolve relative to the yaml's directory, not cwd — user may run
        # the pipeline from anywhere
        yaml_dir = pathlib.Path(yaml_path).resolve().parent
        p = (yaml_dir / p).resolve()
    if not p.exists():
        raise RuntimeError(
            f"skills_dir does not exist: {p}\n"
            f"(from research_config.yaml → paths.skills_dir = {raw!r})\n"
            f"Fix: set it to an absolute path or a path relative to the yaml."
        )
    if not (p / "shared").is_dir():
        raise RuntimeError(
            f"skills_dir resolved to {p}, but no `shared/` subdirectory found.\n"
            f"Expected layout: {p}/shared/ , {p}/research-orchestrator/ , etc."
        )
    return str(p)
```

Apply it in both branches:

- **Exists** → run migration (C3 rename + backfill missing keys) and resume:
  ```python
  import sys, pathlib, json
  SKILLS_DIR = resolve_skills_dir(
      cfg["paths"].get("skills_dir", "./skills"),
      yaml_path="research_config.yaml",
  )
  sys.path.insert(0, str(pathlib.Path(SKILLS_DIR) / "shared"))
  import state_migrate
  state = state_migrate.migrate("pipeline_state.json")
  # state now has dataset_root (not dataset), all v1.6 keys present, no float('inf')

  # v1.7.6 — overwrite python_runner with this Stage 0's detection.
  # Stale state from earlier versions (pre-v1.7.5 default "uv run") would
  # otherwise lock the loop into a broken runner even after the user set up
  # ultralytics in system python.
  state["python_runner"] = python_runner
  pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))

  print(f"Resuming from stage: {state.get('stage')} (runner: {python_runner})")
  ```

- **Does not exist** → create from yaml values:
  ```python
  import json, os
  from datetime import datetime

  ev  = cfg["evaluation"]
  dh  = cfg.get("dataset_hunter", {})
  ar  = cfg.get("autoresearch", {})
  orc = cfg.get("orchestrator", {})
  adv = cfg.get("advanced", {})

  # Pull metric config from evaluation (single source of truth)
  metrics      = ev["metrics"]
  PRIMARY      = metrics["primary"]
  TIEBREAK     = metrics.get("tiebreak")
  MIN_IMPROVE  = metrics.get("min_improvement", 0.001)
  MINIMIZE     = metrics.get("minimize", [])
  # Sentinel for "no result yet" — None survives JSON round-trip where
  # float('inf') does not. Autoresearch's is_better() treats None as
  # "no baseline, always better than None", so the first keep always wins.

  stopping = orc.get("stopping", {})
  state = {
      # identity
      "project_name":          cfg["meta"]["project_name"],
      "run_tag":               cfg["meta"].get("run_tag"),
      "task":                  cfg["task"]["description"],
      "task_type":             cfg["task"].get("task_type", "object_detection"),
      "dataset_root":          cfg["paths"]["dataset_root"],   # C3 — renamed
      # v1.9.2 — absolute project root, so file ops never depend on cwd.
      # Bug this fixes: agent in multi-project working dir reads neighbour's
      # run.log and parses metrics from the wrong experiment. cwd cannot be
      # trusted; this canonical absolute path is the SOURCE OF TRUTH for
      # every file operation in the pipeline.
      "project_root":          str(pathlib.Path(
                                   cfg.get("paths", {}).get("project_root", ".")
                               ).resolve()),
      "stage":                 "init",
      # skill outputs
      "paper_finder_done":     False,
      "base_model_md_ready":   False,
      "modules_md_ready":      False,
      "pretrain_done":         False,
      "pretrain_skipped":      False,
      "pretrain_offer_declined": False,
      "pretrain_attempt_failed": False,            # B3 — distinguish failed-attempt from decline
      "pretrain_weights":      None,
      "base_weights_local":    None,
      "autoresearch_running":  False,
      # metric tracking (metric-agnostic field names)
      "primary_metric_name":   PRIMARY,
      "best_primary_value":    None,        # set by autoresearch on first keep
      "best_tiebreak_value":   None,        # A2
      "stall_count":           0,
      "loop_count":            0,
      "vanilla_baseline_done": False,           # v1.8 — Loop 0 vanilla baseline
      "no_improvement_loops":  0,               # v1.8 — for autonomous stop trigger AND v1.12.1 pretrain trigger
      "pretrain_dead_config_warned": False,     # v1.8 — one-time warning latch
      "batch_size_pre_autohalve": None,         # v1.9 — (deprecated v1.12; field kept for state migration)
      # v1.12.1 — optional_pretrain_trigger state (was dead config v1.7-v1.12; now wired up)
      # trigger_fired_count: how many times the auto-trigger has fired this run (informative)
      # trigger_no_corpus_warned: one-time latch for "trigger fired but no pretrain corpus available"
      "pretrain_trigger_fired_count": 0,
      "pretrain_trigger_no_corpus_warned": False,
      # v1.13 — per-module tuning state (multi-attempt loop)
      # current_tuning_module: name of module being tuned (None when between modules)
      # current_tuning_attempt: 1-indexed attempt within current module (0 = not tuning)
      # tuning_attempt_extended: whether attempt-extension already granted (1 extension max per module)
      # last_attempt_final_map: previous attempt's final mAP, for improvement checks
      "current_tuning_module":   None,
      "current_tuning_attempt":  0,
      "tuning_attempt_extended": False,
      "last_attempt_final_map":  None,
      # v1.11 — concurrent paper-finder state
      # spawned: did we Task() spawn the subagent for this pipeline run? Resume-safe.
      # done:    has subagent written paper_finder.done sentinel? Mirrored from disk for state queries.
      # fallback_used: did sequential fallback fire? For discoveries.md narrative.
      "concurrent_paper_finder_spawned":      False,
      "concurrent_paper_finder_done":         False,
      "concurrent_paper_finder_fallback_used": False,
      # v1.11.1 — subagent provenance (informative only; populated by Loop 0 spawn)
      "concurrent_paper_finder_subagent_type":  None,
      "concurrent_paper_finder_subagent_model": None,
      "seed":                  ar.get("loop", {}).get("seed", 42),
      # v1.12 — BATCH_SIZE is now LOCKED (invariants.LOCKED_VARS includes
      # "batch_size"). yaml field renamed: `batch_size` (v1.12+) preferred,
      # `initial_batch_size` (v1.9.3) still accepted for backward compat.
      # Lookup priority:
      #   1. yaml's `autoresearch.loop.batch_size` (v1.12)
      #   2. yaml's `autoresearch.loop.initial_batch_size` (deprecated alias)
      #   3. None → template default (16)
      # Stored ONLY as state["batch_size"] regardless of which yaml key
      # was read; downstream invariants check this canonical key.
      "batch_size":            (ar.get("loop", {}).get("batch_size") or
                                ar.get("loop", {}).get("initial_batch_size")),
      "paper_finder_expansions": 0,
      "consecutive_crashes":   0,           # A3/C8
      "param_only_streak":     0,
      "architecture_keeps":    0,
      "rebase_marker_loops":   [],          # B4
      # repretrain signaling
      "request_repretrain":    False,
      "repretrain_reason":     None,
      # paths
      "local_papers_dir":      cfg["paths"].get("local_papers_dir"),
      "local_datasets_dir":    cfg["paths"].get("local_datasets_dir"),
      "skills_dir":            resolve_skills_dir(
          cfg["paths"].get("skills_dir", "./skills"),
          yaml_path="research_config.yaml",
      ),
      "train_script":          cfg["paths"].get("train_script", "train.py"),
      "custom_modules_file":   cfg["paths"].get("custom_modules_file", "custom_modules.py"),
      "weights_dir":           cfg["paths"].get("weights_dir", "weights"),
      # time budgets
      "pretrain_time_budget":  dh.get("pretrain", {}).get("time_budget_sec", 21600),
      "self_eval_time_budget": dh.get("pretrain", {}).get("eval_time_budget_sec", 1800),
      "loop_time_budget":      ar.get("loop", {}).get("time_budget_sec", 1200),
      "pretrain_improvement_threshold": dh.get("pretrain", {}).get("improvement_threshold", 0.002),
      # dataset hunter
      "dataset_hunter_enabled": dh.get("enabled", True),
      "disk_budget_gb":        dh.get("disk", {}).get("budget_gb", 100),
      # API keys are read from os.environ at runtime, NOT stored in state.
      # See Stage 0 Step 1.5 for .env loading. Skills access them via
      # os.environ.get("ROBOFLOW_API_KEY") etc.
      # autoresearch behaviour (metric-related values now come from evaluation)
      "stall_threshold":       ar.get("stall", {}).get("threshold", 10),
      "stall_force_test_reset": ar.get("stall", {}).get("force_test_reset", 5),   # C1
      "improvement_threshold": MIN_IMPROVE,              # from evaluation.metrics.min_improvement
      "loop_iterations":       ar.get("loop", {}).get("iterations"),
      # orchestrator
      "crash_pause_after":     orc.get("error_policy", {}).get("autoresearch_crash_pause_after", 3),
      "python_runner":         python_runner,                                        # D5
      "stop_flag_file":        stopping.get("stop_flag_file", "stop_pipeline.flag"), # C9
      "max_paper_finder_expansions": stopping.get("max_paper_finder_expansions", 3), # C9
      "stop_requested":        False,                                                # C9
      "stop_reason":           None,                                                 # C9
      # v1.8 — autonomous stop triggers (all optional; null = no limit)
      # Loaded from research_config.yaml → orchestrator.stopping
      "max_no_improvement_loops":  stopping.get("max_no_improvement_loops"),
      "max_total_loops":           stopping.get("max_total_loops"),
      "max_wallclock_hours":       stopping.get("max_wallclock_hours"),
      "baseline_snapshot":     None,                                                 # reserved for future B2
      # advanced
      "results_tsv":           adv.get("results_tsv", "results.tsv"),
      "modules_md_path":       adv.get("modules_md", "modules.md"),
      "base_model_md_path":    adv.get("base_model_md", "base_model.md"),
      "pretrain_ckpt_dir":     adv.get("pretrain_ckpt_dir", "pretrain_ckpt"),
      # timestamps
      "started_at":            datetime.now().isoformat(),
      "last_updated":          datetime.now().isoformat(),
  }
  state_path = orc.get("pipeline_state", cfg["paths"].get("pipeline_state", "pipeline_state.json"))
  json.dump(state, open(state_path, "w"), indent=2)
  ```

  **Note on metric field names:** `best_primary_value` replaces the old `best_map50_95`.
  All stages reference `state["primary_metric_name"]` when they need to know what
  the metric actually is (e.g. for printing summaries). No skill hardcodes
  `val_mAP50_95` or any other metric name.

### Step 3 — Verify skill files exist

(using `skills_dir` from pipeline state, default: `./skills` (resolved relative to the yaml)):
```bash
SKILLS_DIR=$(python3 -c "import json; print(json.load(open('pipeline_state.json'))['skills_dir'])")
for s in paper-finder dataset-hunter autoresearch; do
  [ -f "$SKILLS_DIR/$s/SKILL.md" ] && echo "OK: $s" || echo "MISSING: $s"
done
```
If any skill is missing, print the install command and stop:
```bash
unzip <skill>.skill -d $SKILLS_DIR/<skill-name>
# e.g. unzip paper-finder.skill -d ~/.claude/skills/paper-finder
```

### Step 4 — Verify paper2code is available (optional but recommended)

```bash
[ -f "$SKILLS_DIR/paper2code/SKILL.md" ] && echo "OK: paper2code" \
  || echo "OPTIONAL: paper2code not installed. Modules with paper2code: yes will use GitHub clone fallback."
```
paper2code converts arXiv papers to code. Without it, autoresearch handles
modules by:
- `paper2code: yes (GitHub repo available)` → clone the repo directly
- `paper2code: yes` (no repo) → Claude writes the module code from the paper's
  description in `modules.md`. Quality varies — log any concerns to `discoveries.md`
- `paper2code: no` → same as above

Install if desired: `npx skills add PrathamLearnsToCode/paper2code/skills/paper2code`

### Step 5 — Initialise git if needed

```bash
# Initialise repo if not already one
git rev-parse --git-dir 2>/dev/null || git init

# v1.7.5 — ensure git identity exists BEFORE any commit. Default git config
# in fresh installs has no user.email, and the first `git commit` fails with
#   fatal: unable to auto-detect email address (got '<user>@<host>.(none)')
# We set a --local (per-repo) identity derived from the project name so the
# global git config is not touched. User can override at any time with
# `git config --local user.email / user.name` of their own.
if [ -z "$(git config --local --get user.email 2>/dev/null)" ]; then
    PROJECT=$(python3 -c "import json; print(json.load(open('pipeline_state.json')).get('project_name', 'autoresearch'))" 2>/dev/null || echo "autoresearch")
    git config --local user.email "pipeline@${PROJECT}.local"
    git config --local user.name  "autoresearch pipeline"
    echo "[orchestrator] Set git identity (--local): pipeline@${PROJECT}.local"
fi

# First commit (if repo is brand new with no commits yet)
git rev-parse --verify HEAD 2>/dev/null || (git add -A && git commit -m "init")
```

### Step 6 — Check for `train.py` and friends

All skills in this pipeline assume `train.py` follows the contract documented
in `<skills_dir>/shared/train-script-spec.md`. Read that file once now so
later stages can rely on its guarantees.

```bash
cat "$SKILLS_DIR/shared/train-script-spec.md"
```

#### (v1.7.5) Ensure `weights/` exists before any model probing

Any test call like `YOLO("yolo26x.pt")` that an agent makes *outside*
train.py will download the file to **cwd** — which may be project root,
not `weights/`. Create the directory early so at least the structure is
in place, and always use an explicit path in any diagnostic probe:

```python
import pathlib, json
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
pathlib.Path(state.get("weights_dir", "weights")).mkdir(exist_ok=True)
```

**Rule**: if at any point a diagnostic call needs to load a model file
(to confirm ultralytics recognises it, to read its layer layout, etc.),
always prefix the path: `YOLO("weights/yolo26x.pt")`, never
`YOLO("yolo26x.pt")`. The latter writes to cwd if the file is missing.

Then verify and scaffold:

```python
import shutil, pathlib, re, json
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
skills_dir   = pathlib.Path(state["skills_dir"])
templates    = skills_dir / "shared" / "templates"
task_type    = state["task_type"]
train_script = pathlib.Path(state["train_script"])

# Pick templates based on task_type
if task_type == "object_detection":
    needed = {"train.py": templates / "train.py.detection"}
elif task_type == "object_tracking":
    needed = {
        "train.py": templates / "train.py.tracking",
        "track.py": templates / "track.py.tracking",
    }
else:
    raise SystemExit(f"No template for task_type={task_type!r}. "
                     f"Add one under {templates}/ and update the spec.")

for dst_name, src in needed.items():
    dst = pathlib.Path(dst_name)
    if not dst.exists():
        shutil.copy(src, dst)
        print(f"scaffolded {dst} from {src.name}")

# Spec compliance check — run before any patch
# v1.7.5 — Use regex for Section markers so the check tolerates ASCII
# digits (Section 1), circled digits (Section ①), and mixed styles.
SECTION_RE = [
    (1, re.compile(r"(?m)^#\s*(?:═+\s*\n#\s*)?Section\s*[①1]\b", re.IGNORECASE)),
    (2, re.compile(r"(?m)^#\s*(?:═+\s*\n#\s*)?Section\s*[②2]\b", re.IGNORECASE)),
    (3, re.compile(r"(?m)^#\s*(?:═+\s*\n#\s*)?Section\s*[③3]\b", re.IGNORECASE)),
    (4, re.compile(r"(?m)^#\s*(?:═+\s*\n#\s*)?Section\s*[④4]\b", re.IGNORECASE)),
]

def check_spec_compliance(path):
    src = pathlib.Path(path).read_text()
    missing = []
    for n, pat in SECTION_RE:
        if not pat.search(src):
            missing.append(f"Section {n}")
    if path.name == "train.py":
        for var in ["TIME_BUDGET", "SEED", "BATCH_SIZE", "WEIGHTS",
                    "DATA_YAML", "NUM_CLASSES", "CKPT_DIR"]:
            if not re.search(rf"(?m)^{var}\s*=", src):
                missing.append(var)
        for fn in ["def inject_modules", "def main"]:
            if fn not in src:
                missing.append(fn)
    return missing

for name in needed:
    missing = check_spec_compliance(pathlib.Path(name))
    if missing:
        raise SystemExit(
            f"{name} is not spec-compliant: missing {missing}. "
            f"Fix it or delete it and let orchestrator re-scaffold from "
            f"{templates}/."
        )
```

If the user already had a `train.py` that fails the compliance check,
stop and report — do not try to auto-fix. Regex patches at Stage 3 assume
the spec shape.

After this step, `git add` the new/existing script(s) and commit:
```bash
git add train.py
[ -f track.py ] && git add track.py
git diff --cached --quiet || git commit -m "scaffold: train.py per spec"
```

### Step 6.5 — (v1.7.6) Verify the chosen runner can import ultralytics

v1.7.5 attempted to auto-write a `pyproject.toml` for users who wanted
uv-managed environments. This was removed in v1.7.6 because:

1. The hardcoded `requires-python = ">=3.10"` excluded Python 3.9 systems
2. `[tool.uv] override-dependencies = []` did not actually prevent uv
   from re-downloading torch over the user's CUDA-specific install
3. `name = "<project_name>"` failed PEP 503 validation when project
   names contained spaces, uppercase, or non-ASCII characters

The right separation of concerns: **the pipeline picks the runner; the
user manages the environment.** Step 0's `choose_python_runner()`
defaults to system `python3` and only falls through to `uv run` when
both `pyproject.toml` exists AND uv's env can already `import
ultralytics`. If neither path works, the pipeline now fails loudly at
startup with a clear remediation message rather than scaffolding a
partially-working project.

```python
import shutil, subprocess, json, pathlib

state = json.loads(pathlib.Path("pipeline_state.json").read_text())
runner = state["python_runner"]

# Re-verify import works under the chosen runner. This guards against
# rare cases where Step 0's detection succeeded but the env then changed
# (e.g. apt/uv tampered with the env between Step 0 and now).
if runner == "uv run":
    rc = subprocess.run(
        ["uv", "run", "python3", "-c", "import ultralytics"],
        capture_output=True,
    ).returncode
else:
    rc = subprocess.run(
        ["python3", "-c", "import ultralytics"],
        capture_output=True,
    ).returncode

if rc != 0:
    raise RuntimeError(
        f"ultralytics import failed under python_runner={runner!r}.\n"
        f"\n"
        f"Remediation options:\n"
        f"  (a) System install: pip install ultralytics\n"
        f"  (b) uv-managed: write your own pyproject.toml with the deps\n"
        f"      your CUDA setup needs, then run `uv sync` before retrying.\n"
        f"      The pipeline will pick up uv automatically once `uv run\n"
        f"      python3 -c 'import ultralytics'` succeeds.\n"
        f"\n"
        f"Why no auto-scaffold: torch / CUDA / cuDNN / Python version\n"
        f"combinations vary too much across research machines for a\n"
        f"generic pyproject.toml to be safe. v1.7.5's auto-write was\n"
        f"removed because it routinely re-downloaded a CPU-only torch\n"
        f"over a working CUDA install."
    )
```

Update `pipeline_state.json` → `stage: "paper_finder"` and proceed immediately.
All subsequent stages read their parameters from `pipeline_state.json` — never hardcode
values that should come from `research_config.yaml`.

---

## Stage 1 — Paper Finder (initial)

**Entry condition:** `paper_finder_done == false`

### v1.11 — Two execution modes

The paper-finder runs in one of two modes depending on
`orchestrator.concurrent_paper_finder.enabled`:

| Mode | When | What runs in Stage 1 | What runs later |
|---|---|---|---|
| **Sequential** (v1.10 behaviour) | `concurrent_paper_finder.enabled: false` OR yaml block missing | All phases 1-6: base_model.md + modules.md both produced before Stage 2 | Nothing |
| **Concurrent** (v1.11+ default) | `concurrent_paper_finder.enabled: true` | Phase 1 only — produces base_model.md (orchestrator needs WEIGHTS for train.py scaffold). Phase 5-6 deferred. | Phase 5-6 spawned as subagent during Loop 0 vanilla baseline |

Why Phase 1 stays sequential: orchestrator Stage 0 Step 6 scaffolds
`train.py` with `WEIGHTS = "<path>"` — that path comes from
`base_model.md`. The vanilla baseline needs to actually run train.py;
it can't start until WEIGHTS is resolved. So Phase 1 (~5 min) blocks
Stage 2; Phase 5-6 (~2h) is the concurrency target.

Why dataset-hunter doesn't get the same treatment: dataset-hunter
writes `pretrain_weights` into pipeline_state, which the train.py
scaffold also depends on. Running it in parallel with anything that
edits pipeline_state creates race conditions. Keep sequential.

### Mode selection

```python
import yaml, pathlib, json
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
cfg = yaml.safe_load(pathlib.Path("research_config.yaml").read_text()) or {}
cpf_cfg = (cfg.get("orchestrator", {}) or {}).get("concurrent_paper_finder", {}) or {}
concurrent_enabled = bool(cpf_cfg.get("enabled", True))   # default True (v1.11 aggressive)
```

### Sequential mode (concurrent_enabled == False)

1. Read `$SKILLS_DIR/paper-finder/SKILL.md`
2. Run paper finder with `mode: initial` and the task description from pipeline state.
   Run all phases 1-6 in a single invocation.
3. Verify outputs exist before advancing:
   ```bash
   [ -f "base_model.md" ] && echo "base_model.md OK" || echo "ERROR: base_model.md missing"
   [ -f "modules.md" ]    && echo "modules.md OK"    || echo "ERROR: modules.md missing"
   ```
4. If either file is missing → retry paper finder once. If still missing → stop and report.

5. Update `pipeline_state.json`:
   ```json
   { "stage": "dataset_hunter", "paper_finder_done": true,
     "base_model_md_ready": true, "modules_md_ready": true }
   ```

Proceed immediately to Stage 2.

### Concurrent mode (concurrent_enabled == True, v1.11+ default)

1. Read `$SKILLS_DIR/paper-finder/SKILL.md`
2. Run paper finder with `mode: initial_phase1_only` and the task
   description. The subagent prompt below documents the exact phase
   restriction; for Stage 1 here, only Phase 1 is requested.
3. Verify Phase 1 output:
   ```bash
   [ -f "base_model.md" ] && echo "base_model.md OK" || echo "ERROR: base_model.md missing"
   ```
4. If missing → retry once. If still missing → fall back to full
   sequential paper-finder (don't proceed without base_model.md).

5. Update `pipeline_state.json`:
   ```json
   { "stage": "dataset_hunter",
     "base_model_md_ready": true,
     "paper_finder_done": false,         // Phase 5-6 still pending
     "modules_md_ready": false }
   ```

Note: `paper_finder_done` stays False because Phase 5-6 hasn't run.
The flag flips to True after the concurrent subagent (spawned during
Loop 0) finishes OR after the post-baseline fallback completes.

Proceed immediately to Stage 2. The subagent for Phase 5-6 is spawned
later, in autoresearch's Loop 0 prelude (see § Loop 0 spawn point).

---

## Stage 2 — Dataset Hunter (pretrain optional)

**Entry condition:** `dataset_hunter_enabled == true` AND `pretrain_done == false` AND `pretrain_skipped == false` AND `pretrain_time_budget > 0`

### C10 fast-path — dataset_hunter_enabled: false

If the user set `dataset_hunter.enabled: false` in the yaml, skip this stage
entirely. Pretrain never runs, and autoresearch starts from the base model
weights resolved from `base_model.md`:

```python
import json, pathlib
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
if not state["dataset_hunter_enabled"]:
    print("dataset_hunter disabled — skipping Stage 2 entirely.")
    state["stage"] = "autoresearch"
    state["pretrain_skipped"] = True
    state["pretrain_offer_declined"] = True   # prevent auto-trigger later
    pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
    # Fall through directly to Stage 3.
```

### v1.7.5 fast-path — pretrain.time_budget_sec: 0

If the user set `dataset_hunter.pretrain.time_budget_sec: 0` in the yaml,
dataset search **may still run** (so you get modules.md coverage and
downloaded datasets for later re-pretrain), but **pretrain is skipped**.
This is the "I want to start experimenting fast, pretrain can wait"
workflow — common when a user wants Loop 1 baseline within the hour,
not tomorrow.

```python
if state.get("pretrain_time_budget", 21600) == 0:
    print("pretrain_time_budget=0 — skipping pretrain, dataset search still runs.")
    state["pretrain_skipped"] = True
    state["pretrain_offer_declined"] = True   # prevent auto-trigger later
    # Don't set stage=autoresearch yet — Phase 1-4 of dataset-hunter still run
    # if dataset_hunter_enabled. Only pretrain (Phase 5-6) is skipped.
    pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
    # After dataset-hunter Phase 1-4 finishes, fall through to Stage 3.
```

To skip **both** dataset search AND pretrain, use the stronger form
`dataset_hunter.enabled: false` above.

**Patience:** Stage 2 is the longest stage. Dataset downloads alone may take 1–3
hours depending on how many sources are available. Pretrain adds another
`pretrain_time_budget` seconds (default 6 hours). Self-eval adds
`self_eval_time_budget` × 2 (two finetune jobs). **Do not skip, shortcut,
or abort Stage 2 because it is taking a long time.** The only valid reasons
to skip are:
(a) corpus is empty after all downloads, or
(b) the user set `dataset_hunter.enabled: false` in the yaml, or
(c) the user set `dataset_hunter.pretrain.time_budget_sec: 0` in the yaml
    (pretrain skipped but dataset search still runs).

1. Read `$SKILLS_DIR/dataset-hunter/SKILL.md`
2. Pass from pipeline state: disk budget, Roboflow API key, pretrain TIME_BUDGET, `local_datasets_dir`

3. **Resolve base model weights to a local path before anything else (A4):**
   ```python
   import re, pathlib, subprocess, json

   state = json.loads(pathlib.Path("pipeline_state.json").read_text())
   text = pathlib.Path("base_model.md").read_text()
   m = re.search(r"Weights URL.*?:\s*(.+)", text)
   weights_url = m.group(1).strip() if m else None

   BASE_WEIGHTS = None

   if weights_url and weights_url.startswith("http"):
       local_path = f"weights/{pathlib.Path(weights_url).name}"
       if pathlib.Path(local_path).exists() and pathlib.Path(local_path).stat().st_size > 0:
           BASE_WEIGHTS = local_path
       elif safe_wget(weights_url, local_path):
           BASE_WEIGHTS = local_path
       else:
           print(f"Base model weights download failed from {weights_url}; "
                 f"falling back to yolo26x.pt")
           BASE_WEIGHTS = "weights/yolo26x.pt"

   elif weights_url == "reconstruct via paper2code":
       # Extract arXiv ID from base_model.md and run paper2code now
       arxiv_m = re.search(r"arxiv\.org/abs/([\w.]+)", text)
       if arxiv_m:
           arxiv_id = arxiv_m.group(1)
           print(f"Running paper2code for base model: {arxiv_id}")
           # Invoke paper2code skill
           # After paper2code completes, find the generated checkpoint
           ckpts = (sorted(pathlib.Path(".").glob("*/checkpoints/*.pt"))
                    + sorted(pathlib.Path(".").glob("*/src/*.pt")))
           if ckpts:
               BASE_WEIGHTS = str(ckpts[-1])
               print(f"Base model reconstructed: {BASE_WEIGHTS}")
           else:
               print("WARNING: paper2code ran but no .pt found — falling back to yolo26x.pt")
               BASE_WEIGHTS = "weights/yolo26x.pt"
       else:
           print("WARNING: no arXiv ID in base_model.md — cannot reconstruct, falling back")
           BASE_WEIGHTS = "weights/yolo26x.pt"
   else:
       BASE_WEIGHTS = "weights/yolo26x.pt"  # fallback

   print(f"Base weights resolved: {BASE_WEIGHTS}")
   state["base_weights_local"] = BASE_WEIGHTS
   pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
   ```

4. Run dataset hunter **download + convert + merge phases only** (stop before pretrain).

5. **Check corpus size — decide whether to pretrain:**
   ```bash
   TRAIN_COUNT=$(ls pretrain_data/merged/images/train 2>/dev/null | wc -l)
   echo "Corpus train images: $TRAIN_COUNT"
   ```
   - If `TRAIN_COUNT == 0` (no data at all — neither external downloads nor local datasets):
     - Log: `"Corpus empty — skipping pretrain, proceeding directly to autoresearch"`
     - Update `pipeline_state.json`:
       ```json
       { "stage": "autoresearch", "pretrain_skipped": true,
         "pretrain_weights": null }
       ```
     - **Proceed immediately to Stage 3** — do not run pretrain.

   - If `TRAIN_COUNT > 0`:
     - Run pretrain + self-eval with TIME_BUDGET from pipeline state.
     - Dataset hunter's self-eval compares the primary metric (from pipeline state
       `primary_metric_name`) of pretrained vs original weights, and writes its
       recommendation. Orchestrator reads the recommendation — does not re-evaluate.
     - **B3 — distinguish decline from failure.** If pretrain or self-eval crashed:
       set `pretrain_attempt_failed = true` (NOT `pretrain_offer_declined`).
       `pretrain_offer_declined = true` is reserved for "we tried and the metric
       didn't clear the threshold" — using it for crashes suppresses future
       auto-trigger retries that might succeed once e.g. RAM is freed.
     - Find the pretrain checkpoint:
       ```bash
       PRETRAIN_CKPT=$(ls -t pretrain_ckpt/*.pt 2>/dev/null | head -1)
       ```
     - Read dataset hunter's recommendation (written to `pretrain_eval.json` in Phase 6).
       - recommendation == "use_pretrained" → `pretrain_weights = $PRETRAIN_CKPT`
       - recommendation == "use_original" or "inconclusive" → `pretrain_weights = null`,
         `pretrain_offer_declined = true` (the attempt completed; it just wasn't better)
     - Update `pipeline_state.json`:
       ```json
       { "stage": "autoresearch", "pretrain_done": true,
         "pretrain_weights": "<resolved path or null>" }
       ```
     - Proceed immediately to Stage 3.

---

## Stage 3 — Autoresearch Loop

**Entry condition:** `autoresearch_running == false` or resuming

### Step 1 — Read autoresearch SKILL.md

```bash
cat $SKILLS_DIR/autoresearch/SKILL.md
```

### Step 2 — Resolve and patch `WEIGHTS` in `train.py` (A4)

The template ships with a placeholder (`weights/yolo26x.pt`). Orchestrator
must set this to the actual weights path before the first run, or `train.py`
will crash with `FileNotFoundError`.

```python
import re, pathlib, json, yaml

state = json.loads(pathlib.Path("pipeline_state.json").read_text())
train_py = pathlib.Path(state["train_script"])

def patch_variable(path, name, value):
    """Replace ^<n>\s*=.*$ with <n> = <value>. Not locked."""
    src = path.read_text()
    pattern = re.compile(rf"(?m)^{re.escape(name)}\s*=.*$")
    new_src, n = pattern.subn(f'{name} = {value}', src, count=1)
    if n == 0:
        raise RuntimeError(f"{path}: {name} not found at column 0.")
    path.write_text(new_src)

# Resolve weights to a local file
if state.get("pretrain_weights"):
    weights_local = state["pretrain_weights"]
elif state.get("base_weights_local"):
    weights_local = state["base_weights_local"]
else:
    # Download from base_model.md via safe_wget (A4)
    bm = pathlib.Path(state["base_model_md_path"])
    weights_local = None
    if bm.exists():
        bm_text = bm.read_text()
        m = re.search(r"Weights URL.*?:\s*(.+)", bm_text)
        url = m.group(1).strip() if m else None
        if url and url.startswith("http"):
            pathlib.Path(state["weights_dir"]).mkdir(exist_ok=True)
            local = f"{state['weights_dir']}/{pathlib.Path(url).name}"
            if pathlib.Path(local).exists() and pathlib.Path(local).stat().st_size > 0:
                weights_local = local
            elif safe_wget(url, local):
                weights_local = local
            # If download failed, weights_local stays None; WARNING below fires.

if weights_local and pathlib.Path(weights_local).exists():
    patch_variable(train_py, "WEIGHTS", f'"{weights_local}"')
    state["base_weights_local"] = weights_local
else:
    print("WARNING: no weights file resolved. train.py WEIGHTS must point to "
          "an existing .pt file or the first run will crash.")
```

### Step 2.5 — Patch `DATA_YAML` and `NUM_CLASSES` in `train.py` (B6/C3)

The template ships with `DATA_YAML = "data/dataset.yaml"` — a placeholder.
Orchestrator must patch this to match `paths.dataset_root` from the yaml,
or `train.py` will crash with `FileNotFoundError`.

```python
dataset_root = pathlib.Path(state["dataset_root"])     # C3 — renamed from "dataset"

# Find the data.yaml in the dataset directory
data_yaml = None
for candidate in [dataset_root / "data.yaml",
                  dataset_root / "data.yml",
                  dataset_root / "dataset.yaml",
                  dataset_root.parent / "data.yaml"]:
    if candidate.exists():
        data_yaml = str(candidate.resolve())
        break

if data_yaml:
    patch_variable(train_py, "DATA_YAML", f'"{data_yaml}"')
    # B6 — warn clearly if NUM_CLASSES could not be read. Previously this
    # block silently swallowed all exceptions, leaving NUM_CLASSES = 10
    # (template default). The resulting model learned 10 classes when the
    # dataset had e.g. 3, producing silently wrong metrics.
    try:
        ds_cfg = yaml.safe_load(open(data_yaml))
        nc = ds_cfg.get("nc") or len(ds_cfg.get("names") or [])
        if nc:
            patch_variable(train_py, "NUM_CLASSES", nc)
        else:
            print(f"WARNING: {data_yaml} has neither 'nc' nor 'names'. "
                  f"train.py NUM_CLASSES stays at template default — "
                  f"fix data.yaml or patch NUM_CLASSES manually.")
    except Exception as e:
        print(f"WARNING: could not read {data_yaml}: {e}. "
              f"train.py NUM_CLASSES stays at template default — "
              f"fix data.yaml or patch NUM_CLASSES manually.")
else:
    print(f"WARNING: no data.yaml found under {dataset_root}. "
          f"Patch DATA_YAML in train.py manually before running.")

# Save updated state
pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
```

### Step 3 — Lock `TIME_BUDGET`, `SEED`, and `IMGSZ` across all experiment scripts

For `task_type: object_detection` this is just `train.py`; for
`object_tracking` it is `train.py` + `track.py` (both have Section ②
per the spec). Helper function below fails loudly if any variable is
missing — a silent no-op here would let experiments run with stale
values and silently corrupt results.

```python
def lock_variable(path, name, value):
    """Replace ^<n>\s*=.*$ with <n> = <value>  # locked by orchestrator.
    Raises RuntimeError if no line matches."""
    src = path.read_text()
    pattern = re.compile(rf"(?m)^{re.escape(name)}\s*=.*$")
    new_src, n = pattern.subn(
        f"{name} = {value}  # locked by orchestrator", src, count=1)
    if n == 0:
        raise RuntimeError(f"{path}: {name} not found at column 0.")
    path.write_text(new_src)

# Resolve IMGSZ from yaml BEFORE the lock loop (not after)
cfg = yaml.safe_load(pathlib.Path("research_config.yaml").read_text())
ev = cfg.get("evaluation", {})
imgsz = (ev.get("ultralytics_val", {}).get("imgsz")
         or ev.get("trackeval", {}).get("imgsz")
         or 1920)

scripts_to_lock = [train_py]
if state["task_type"] == "object_tracking":
    scripts_to_lock.append(pathlib.Path("track.py"))

for script in scripts_to_lock:
    if not script.exists():
        raise RuntimeError(f"{script} not found.")
    lock_variable(script, "TIME_BUDGET", state["loop_time_budget"])
    lock_variable(script, "SEED",        state.get("seed", 42))
    lock_variable(script, "IMGSZ",       imgsz)

# v1.7.7 — OPTIMIZER initial set. Not locked — autoresearch may swap it
# (e.g. SGD → AdamW for an LR schedule experiment). But the value 'auto'
# is forbidden because ultralytics silently overrides LR0 and MOMENTUM under
# 'auto', so any LR experiment under 'auto' would silently no-op.
#
# Read the user's preference from yaml (default SGD); reject 'auto' here.
# autoresearch's contract (Critical Rule #15) also forbids setting it back.
init_optimizer = ar.get("optimizer", "SGD")
if str(init_optimizer).lower() == "auto":
    raise RuntimeError(
        "research_config.yaml → autoresearch.optimizer cannot be 'auto'. "
        "ultralytics 'auto' silently overrides LR0/MOMENTUM. "
        "Use a concrete optimizer: SGD / AdamW / Adam / RMSProp / NAdam / RAdam."
    )
for script in scripts_to_lock:
    # Initial set only — not locked. Written once via the same helper for
    # consistency, but autoresearch may overwrite later (without using 'auto').
    lock_variable(script, "OPTIMIZER", repr(init_optimizer))
state["optimizer"] = init_optimizer

# v1.12 — BATCH_SIZE LOCKED. Was initial-set in v1.9.3-v1.11.1; v1.12
# promotes it to the same lock semantics as IMGSZ / SEED / TIME_BUDGET.
# The patch happens once at Stage 0 Step 3, and invariants.LOCKED_VARS
# enforces no drift across iterations.
#
# Why locked: real-world session 2026-04-27 demonstrated that allowing
# autoresearch to dynamically halve BATCH_SIZE (resource_impact auto-halve,
# crash-pause, OOM detection) made experiments incomparable across
# iterations. Loop 9 had to re-establish a vanilla baseline at BATCH=32
# because half the loops ran at 64 and half at 32. v1.12 forbids the
# divergence: BATCH_SIZE is one value for the entire run; OOM = discard +
# block module rather than auto-shrink.
#
# yaml field lookup priority (handled at state init): batch_size > initial_batch_size > None.
init_batch = state.get("batch_size")
if init_batch is not None:
    if not isinstance(init_batch, int) or init_batch < 1:
        raise RuntimeError(
            f"research_config.yaml → autoresearch.loop.batch_size "
            f"must be a positive integer, got {init_batch!r}. Set to a "
            f"concrete batch size with ~20GB headroom on baseline so "
            f"resource-heavy modules can still run. H100 80GB users at "
            f"IMGSZ=640 should try batch_size: 32."
        )
    for script in scripts_to_lock:
        lock_variable(script, "BATCH_SIZE", init_batch)
    print(f"[orchestrator] BATCH_SIZE locked to {init_batch} from yaml (v1.12)")
else:
    # Fall back to template default 16. Lock it anyway so invariants
    # know to enforce no drift; user can edit yaml + re-run Stage 0 to
    # change it.
    print(f"[orchestrator] BATCH_SIZE: yaml unset, locking template default (16)")
    state["batch_size"] = 16
    for script in scripts_to_lock:
        lock_variable(script, "BATCH_SIZE", 16)

# v1.7.2 — persist the resolved IMGSZ in pipeline_state so downstream skills
# (dataset-hunter's pretrain.py scaffold, autoresearch's Step 5.5 repair
# probes) don't each have to re-read research_config.yaml and re-implement
# the same fallback chain. One canonical resolution lives here.
state["imgsz"] = imgsz
pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
```

These four values (TIME_BUDGET, SEED, IMGSZ, BATCH_SIZE) are fixed for
the entire pipeline run — identical across every experiment so that
keep/discard decisions reflect the change being tested, not randomness,
resolution differences, or batch-size differences.

OPTIMIZER is **initialised** from yaml but NOT locked.
- OPTIMIZER: default `SGD` from `autoresearch.optimizer`. autoresearch may
  swap it to test other optimizers (e.g. AdamW) but **must never** set it
  to `'auto'` (see Critical Rule #15 in autoresearch SKILL).

BATCH_SIZE is **locked** as of v1.12 (was initial-set in v1.9.3-v1.11.1).
- BATCH_SIZE: from `autoresearch.loop.batch_size` (or backward-compat
  alias `initial_batch_size`). Locked across iterations.
- autoresearch may NOT halve it dynamically — modules that OOM at the
  locked batch are marked DISCARD + blocked. resource_impact auto-halve
  (v1.9), crash-pause halve (v1.7.6), and OOM-halve paths are all
  REMOVED in v1.12.
- Why: session 2026-04-27 (Loop 9) demonstrated that mixing BATCH=64
  and BATCH=32 across iterations made experiments incomparable. The
  user manually re-baselined to disambiguate. v1.12 forbids the
  divergence.

### Step 4 — Mark autoresearch running

Update pipeline state: `autoresearch_running: true`

### Step 5 — Run autoresearch loop

This runs indefinitely — do not stop it.
The loop internally handles:
- Reading modules.md for pending entries
- Calling paper2code for module code generation
- Injecting into train.py
- keep/discard decisions
- stall detection

**stall_count authority (C2).** `pipeline_state.json` is the single source of
truth, and **orchestrator owns the state-machine transitions**: autoresearch
only writes `stall_count` at its Step 9, never resets it. Step 6 below is
where stall resets happen, not inside the autoresearch loop. This eliminates
the C2 race where both skills touched `stall_count`.

### Step 6 — Stall → paper finder expand handoff (C1/C2)

Orchestrator is the sole owner of this handoff. After each autoresearch loop
iteration, orchestrator checks `pipeline_state.json` and `modules.md` via the
canonical parser:

```python
import sys, pathlib, json
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
sys.path.insert(0, str(pathlib.Path(state["skills_dir"]) / "shared"))
import modules_md as mm

pending_count = mm.count_pending("modules.md")
force_test_reset = state.get("stall_force_test_reset", 5)   # C1 — from yaml

if state["stall_count"] >= state["stall_threshold"]:
    if pending_count > 0:
        # Force-test the next pending module rather than expanding the pool.
        # Reset to the yaml-configured value (C1 — no longer hardcoded 5).
        print(f"Stall at {state['stall_count']}, but {pending_count} pending "
              f"modules remain. Resetting stall_count to {force_test_reset} "
              f"to force-test next pending.")
        state["stall_count"] = force_test_reset
        pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
    else:
        # No pending modules — trigger paper finder expand.
        print(f"Stall at {state['stall_count']}, no pending modules — "
              f"triggering paper finder expand.")
        state["paper_finder_expansions"] = state.get("paper_finder_expansions", 0) + 1
        pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
        # Read and run paper finder in Mode B
        # ... cat $SKILLS_DIR/paper-finder/SKILL.md and follow its Phase 8 ...
        # After paper finder completes:
        state = json.loads(pathlib.Path("pipeline_state.json").read_text())
        state["stall_count"] = 0
        pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
```

### Step 6.5 — Architecture combination → re-pretrain cycle (B4)

After each autoresearch loop iteration, also check for the re-pretrain signal:
```python
if state.get("request_repretrain") and state.get("pretrain_done"):
    # Architecture combination triggered re-pretrain
    ...
```

When triggered:

a. Log: `"Re-pretrain requested after architecture combination"`
b. Read `$SKILLS_DIR/dataset-hunter/SKILL.md`
c. Run dataset hunter **Phase 5 + Phase 6 only** (pretrain + self-eval).
   Use current `train.py` (which now has the combined architecture modules).
   Corpus at `pretrain_data/merged/` already exists from Stage 2.
d. If recommendation == `use_pretrained`:
   - Patch `WEIGHTS` in `train.py` to the new pretrain checkpoint
   - Update `pipeline_state.pretrain_weights`
e. **B4 — write a rebase marker to results.tsv.** Stage 4 summary segments
   the file at these rows so "top 5" is computed per pretrain generation,
   not pooled across different baselines:
   ```python
   import csv, subprocess
   loop = state.get("loop_count", 0)
   # Write a rebase row with zeros for all metric cols
   header = pathlib.Path(state["results_tsv"]).read_text().splitlines()[0].split("\t")
   TEXT_COLS = {"loop", "commit", "status", "description"}
   commit = subprocess.check_output(["git", "log", "--oneline", "-1"],
                                    text=True).strip()[:8]
   row = []
   for col in header:
       if col == "loop":        row.append(str(loop))
       elif col == "commit":    row.append(commit)
       elif col == "status":    row.append("rebase")
       elif col == "description":
           row.append(f"re-pretrain: {state.get('repretrain_reason','unspecified')}")
       elif col in TEXT_COLS:   row.append("")
       else:                    row.append("0.0000")
   with open(state["results_tsv"], "a") as f:
       f.write("\t".join(row) + "\n")
   state.setdefault("rebase_marker_loops", []).append(loop)
   ```
f. Clear the signal: `state["request_repretrain"] = False`;
   `state["repretrain_reason"] = None`
g. Resume autoresearch loop from Step 1

This cycle typically fires after 3+ architecture changes are combined. It
lets modules that only shine with pretrained features show their real value.

### Step 6.75 — Check stop triggers (C9)

Stage 4 previously only fired on Ctrl+C. For unattended runs (cron, systemd),
that's unworkable. Check every loop for the three autonomous triggers:

```python
state = json.loads(pathlib.Path("pipeline_state.json").read_text())

# Trigger 1: sentinel flag file
flag = pathlib.Path(state["stop_flag_file"])
if flag.exists():
    state["stop_requested"] = True
    state["stop_reason"] = f"stop flag {flag} present"

# Trigger 2: bounded iteration count
iters = state.get("loop_iterations")
if iters is not None and state.get("loop_count", 0) >= iters:
    state["stop_requested"] = True
    state["stop_reason"] = f"completed {iters} iterations"

# Trigger 3: too many paper-finder expansions with nothing new
max_exp = state.get("max_paper_finder_expansions", 3)
if state.get("paper_finder_expansions", 0) >= max_exp:
    # Count pending after last expansion — if still zero, we've exhausted ideas
    pending = mm.count_pending("modules.md")
    if pending == 0:
        state["stop_requested"] = True
        state["stop_reason"] = (f"{max_exp} paper-finder expansions produced "
                                f"no new pending modules")

# v1.8 — Trigger 4: no improvement for N consecutive loops
#
# Tracks when autoresearch has been generating only discards. Useful for
# unattended runs where the user wants to stop once the loop has clearly
# converged. Counter is incremented in autoresearch Step 9 on every
# discard and reset to 0 on every keep.
max_no_imp = state.get("max_no_improvement_loops")
if max_no_imp is not None and state.get("no_improvement_loops", 0) >= max_no_imp:
    state["stop_requested"] = True
    state["stop_reason"] = (
        f"{max_no_imp} consecutive iterations without a keep — convergence"
    )

# v1.8 — Trigger 5: hard cap on total iterations
#
# Distinct from `loop_iterations` (Trigger 2): loop_iterations is the
# user's "I want exactly N iterations and then summary" intent. This is
# the safety brake: "regardless of intent, never go above this". Useful
# when iterations: null is set but you still want a sanity ceiling.
max_total = state.get("max_total_loops")
if max_total is not None and state.get("loop_count", 0) >= max_total:
    state["stop_requested"] = True
    state["stop_reason"] = f"hit max_total_loops cap ({max_total})"

# v1.8 — Trigger 6: wallclock cap
#
# For long-running unattended jobs. started_at lives in state as ISO8601;
# compute elapsed hours. Imprecise to ~ minutes (only checked once per
# loop after a TIME_BUDGET-long run).
max_hours = state.get("max_wallclock_hours")
if max_hours is not None and state.get("started_at"):
    from datetime import datetime
    started = datetime.fromisoformat(state["started_at"])
    elapsed_hours = (datetime.now() - started).total_seconds() / 3600
    if elapsed_hours >= max_hours:
        state["stop_requested"] = True
        state["stop_reason"] = (
            f"wallclock {elapsed_hours:.1f}h >= {max_hours}h cap"
        )

pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))

if state["stop_requested"]:
    print(f"Stop requested: {state['stop_reason']}")
    # Jump to Stage 4
```

Users can also stop with Ctrl+C or a "stop" command; those paths remain
unchanged. Stage 4 reads `stop_reason` for the summary.

### Step 7 — Automatic pretrain trigger (no user interaction)

#### v1.8 — Semantic clarification of the two pretrain control fields

Two fields in pipeline_state interact here, and v1.7.x docs left their
relationship implicit. From `thinking_log.md` of a real run, an agent
interpreted `pretrain_offer_declined: true` as "permanently disable the
auto-trigger" while the user's yaml had `optional_pretrain_trigger.enabled:
true`, expecting the trigger to fire after stalls. The trigger never
fired. v1.8 makes the contract explicit:

| Field | Type | Set by | Meaning |
|---|---|---|---|
| `pretrain_offer_declined` | runtime state (boolean, mutable) | Stage 2 path or this Step 7 | "Skip the auto-trigger NEXT time it would fire". Reset to `false` after a successful pretrain or by user-edit; otherwise stays sticky. |
| `optional_pretrain_trigger.enabled` | yaml config (boolean, immutable in run) | User in research_config.yaml | "If `enabled: true`, the auto-trigger MAY fire. If `enabled: false`, this whole Step 7 is dead code — pretrain only runs at Stage 2." |
| `optional_pretrain_trigger.stalled_loops_required` | yaml config (int) | User | How many consecutive no-improvement loops before the trigger considers firing. Only meaningful when `enabled: true`. |

The trigger fires iff **all three** hold:
1. `optional_pretrain_trigger.enabled == true` (yaml says trigger is allowed)
2. `pretrain_offer_declined == false` (no recent decline)
3. Stall counter `>= stalled_loops_required`

A common user mistake (also documented in real runs):

```yaml
# Ambiguous configuration the agent will interpret defensively:
optional_pretrain_trigger:
  enabled: true
  stalled_loops_required: 5
# But Stage 2 sets pretrain_offer_declined = true, making the trigger
# permanently dormant.
```

If the user wants the trigger to genuinely fire on stall, Stage 2 must
NOT set `pretrain_offer_declined: true`. v1.8 Stage 2's "skip pretrain"
path now distinguishes:

- `dataset_hunter.enabled: false` → never pretrain → `pretrain_offer_declined: true`
  (correct: agent declined; auto-trigger should not override)
- `dataset_hunter.pretrain.time_budget_sec: 0` → skip but allow auto-trigger →
  `pretrain_offer_declined: false`
  (the v1.7.5 fast-path already does this; documenting the rationale)

If `optional_pretrain_trigger.enabled` is true AND
`pretrain_offer_declined` is true, orchestrator logs a one-time warning at
loop start so the user notices the dead config:

```python
if (cfg.get("orchestrator", {})
       .get("optional_pretrain_trigger", {})
       .get("enabled") is True
    and state.get("pretrain_offer_declined") is True
    and not state.get("pretrain_dead_config_warned")):
    print(
        "[orchestrator] WARNING: optional_pretrain_trigger.enabled=true "
        "but pretrain_offer_declined=true — auto-trigger will never fire. "
        "Set dataset_hunter.pretrain.time_budget_sec: 0 (instead of "
        "dataset_hunter.enabled: false) if you want the trigger to fire on stall."
    )
    state["pretrain_dead_config_warned"] = True
    pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
```

#### Trigger conditions

If `pretrain_skipped == true` AND `pretrain_offer_declined == false`
AND `pretrain_attempt_failed == false` (B3)
AND autoresearch has run ≥ 5 loops
AND `best_primary_value` has not improved in last 5 loops:

- **Automatically** run pretrain (do not ask the user):
  1. Log: `"Auto-triggering pretrain after 5 stalled loops"`
  2. Read `$SKILLS_DIR/dataset-hunter/SKILL.md`
  3. Skip directly to **Phase 5 — Pretrain** (dataset hunter's phases 1–4 already done)
     using `BASE_WEIGHTS` from `pipeline_state.base_weights_local`
     and corpus already in `pretrain_data/merged/`
  4. Run pretrain + self-eval (Phase 5 + Phase 6 of dataset hunter only)
  5. Resolve new `pretrain_weights` and update `pipeline_state.json`
  6. Patch `train.py`: `WEIGHTS = "<new pretrain_weights>"`
  7. If the attempt completed and the metric cleared threshold →
     `pretrain_offer_declined = false`, `pretrain_weights = <new ckpt>`.
     If the attempt completed but metric didn't clear →
     `pretrain_offer_declined = true` (legitimate decline).
     If the attempt crashed → `pretrain_attempt_failed = true` (B3)
     without setting `pretrain_offer_declined` so the next genuine trigger can
     try again once whatever caused the crash is resolved.
  8. Resume autoresearch loop from Step 1

- If corpus in `pretrain_data/merged/` is empty (nothing was downloaded in
  Stage 2), set `pretrain_offer_declined = true` and skip — cannot pretrain
  with no data.

### Step 8 — best_primary_value

Update `best_primary_value` in pipeline state whenever autoresearch keeps a result.
Autoresearch already writes this field at Step 9 of its loop; orchestrator only
needs to read it for summary printing.

---

## Stage 4 — Done

Triggered by any of:
- User interruption (Ctrl+C) or explicit user command "stop"
- `state["stop_requested"] == true` set by Step 6.75 (C9)

### Step 1 — Print final summary

The primary metric name is read from pipeline state — do not hardcode
`val_mAP50_95` or any specific metric:

```python
import json, pathlib, csv
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
primary = state["primary_metric_name"]
best    = state["best_primary_value"]
reason  = state.get("stop_reason") or "user interrupt"
```

```
=== Pipeline Summary ===
Task:                  <task>
Task type:             <task_type>
Stop reason:           <reason>
Best <primary>:        <best>
Total autoresearch loops: <pipeline_state.loop_count>
Paper finder expansions:  <paper_finder_expansions>
Pretrain used:         <yes/no>
Best weights:          <path from git log>
```

**B4 — print top experiments per rebase segment, not pooled.** When re-pretrain
ran mid-loop, rows before and after the `status=rebase` marker are on
different baselines; mixing them in "top 5 overall" is apples-to-oranges.

```python
cfg = yaml.safe_load(pathlib.Path("research_config.yaml").read_text())
MINIMIZE = set(cfg["evaluation"]["metrics"].get("minimize", []))

rows = list(csv.DictReader(open(state["results_tsv"]), delimiter="\t"))
# Segment at rebase markers
segments = [[]]
for r in rows:
    if r.get("status") == "rebase":
        segments.append([])   # start new segment after marker
    elif r.get("status") == "keep":
        segments[-1].append(r)

reverse = primary not in MINIMIZE
for i, seg in enumerate(segments):
    if not seg:
        continue
    seg.sort(key=lambda r: float(r.get(primary, 0) or 0), reverse=reverse)
    label = f"Top 5 (generation {i + 1}, {len(seg)} keeps)" if len(segments) > 1 \
            else f"Top 5 ({len(seg)} keeps)"
    print(f"\n{label}:")
    for r in seg[:5]:
        print(f"  loop {r['loop']}  {primary}={r.get(primary,'')}  "
              f"{r.get('description','')}")
```

The "sort by primary metric" step must respect `evaluation.metrics.minimize` —
if the primary metric is in `minimize`, sort ascending; otherwise descending.

### Step 2 — Print discoveries

If `discoveries.md` exists, print its contents. This is where autoresearch
logged technical insights during the run instead of stopping to ask the user:

```python
disc = pathlib.Path("discoveries.md")
if disc.exists() and disc.stat().st_size > 100:  # more than just the header
    print("\n=== Discoveries (observations from the loop) ===")
    print(disc.read_text())
```

### Step 3 — Mark done

Update pipeline state: `stage: "done"`. Clear the stop flag file if it
exists, so the next pipeline run isn't pre-stopped:

```python
flag = pathlib.Path(state["stop_flag_file"])
if flag.exists():
    flag.unlink()
state["stage"] = "done"
pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))
```

---

## Resume logic

If `pipeline_state.json` exists and `stage` is not `"init"`:

| stage value      | Resume action |
|------------------|---------------|
| `paper_finder`   | Re-run Stage 1 from scratch (paper finder is idempotent) |
| `dataset_hunter` | Check `pretrain_skipped` flag — if true skip to Stage 3 directly; else check if `pretrain_ckpt/` exists — if yes skip to Stage 3, else re-run Stage 2 from pretrain step |
| `autoresearch`   | Skip to Stage 3, autoresearch reads existing results.tsv and continues |
| `done`           | Print summary and exit |

State migration runs automatically at Stage 0 Step 2 — a resume of an older
pipeline_state.json picks up the renamed `dataset_root` key and any missing
v1.6 fields without manual intervention.

---

## Error handling

**Fully autonomous.** Every error case below has a predefined resolution. The
pipeline never pauses to ask the user for a decision — that breaks
unattended runs. If you genuinely cannot proceed, write a clear error to
stdout and `pipeline_state.json`, then stop. The user reads the state on
their next session.

| Error | Action |
|-------|--------|
| Sub-skill SKILL.md missing | Print install command, stop pipeline |
| paper finder produces no base_model.md | Retry once. If still missing, fall back to default base model (`weights/yolo26x.pt`), log warning, continue to Stage 2 |
| Dataset download all fail | Set `pretrain_skipped: true`, skip pretrain entirely, go to Stage 3 with original weights |
| Pretrain OOM | Halve `BATCH_SIZE` in `pretrain.py`, retry once |
| base model URL dead (A4) | `safe_wget` returns False; fall back to `yolo26x.pt`, log warning, continue. Never crashes the stage. |
| Autoresearch crash 3x in a row | Autoresearch Step 6 handles this autonomously: halve `BATCH_SIZE`, revert last commit, reset counter, log, continue. Orchestrator does nothing — the mechanism is inside autoresearch (A3/C8). |
| git conflict on resume | `git stash && git checkout autoresearch/<tag>` — auto-resolve by stashing uncommitted changes |

---

## Critical rules

1. **Never stop between stages** — advance automatically unless a blocking error occurs.
2. **Always update pipeline_state.json** before and after each stage transition.
3. **Never modify prepare.py or program.md** — these are read-only across all sub-skills.
4. **Sub-skill outputs are inputs to the next stage** — always verify files exist before advancing.
5. **On resume, read pipeline_state.json first** — never re-run completed stages.
6. **Never ask the user for decisions during a run.** The pipeline is designed for
   unattended execution. Every decision point has a predefined default: crash →
   retry then continue, missing file → fallback, stall → expand.
7. **Orchestrator owns the stall state machine** (C2). Autoresearch writes
   `stall_count`; only orchestrator resets it or triggers expansion.
8. **Every download goes through `safe_wget`** (A4). Never call `wget ... check=True`
   directly — a dead URL mid-pipeline should trigger a fallback, not a crash.
