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
SKILLS_DIR=$(python3 -c "import json; print(json.load(open('pipeline_state.json'))['skills_dir'])" 2>/dev/null || echo "$HOME/.claude/skills")
cat $SKILLS_DIR/paper-finder/SKILL.md
cat $SKILLS_DIR/dataset-hunter/SKILL.md
cat $SKILLS_DIR/autoresearch/SKILL.md
```

---

## Pipeline state file

Maintain `pipeline_state.json` in the project root throughout the run.
Never git add this file. Update it after every stage transition.

The full JSON schema lives in `<skills_dir>/shared/file-contracts.md` —
refer to it when adding fields or investigating serialization issues. Key
invariant: **only JSON-spec types** (no `float("inf")`, `datetime`, `Path`
objects). Sentinels for "no measurement yet" are `null`, not infinities.

```json
{
  "task": "<user task description>",
  "task_type": "object_detection | object_tracking | segmentation | classification",
  "dataset": "<dataset name>",
  "stage": "init | paper_finder | dataset_hunter | autoresearch | done",
  "paper_finder_done": false,
  "base_model_md_ready": false,
  "modules_md_ready": false,
  "pretrain_done": false,
  "pretrain_skipped": false,
  "pretrain_offer_declined": false,
  "pretrain_weights": null,
  "base_weights_local": null,
  "autoresearch_running": false,
  "primary_metric_name": "<name from evaluation.metrics.primary>",
  "best_primary_value": null,
  "stall_count": 0,
  "loop_count": 0,
  "paper_finder_expansions": 0,
  "local_papers_dir": null,
  "local_datasets_dir": null,
  "started_at": "<iso datetime>",
  "last_updated": "<iso datetime>"
}
```

**Metric-agnostic by design:** the state schema contains **no hardcoded metric names**.
Whatever the user declared in `research_config.yaml → evaluation.metrics.primary` gets
stored in `primary_metric_name`, and the running best value goes in `best_primary_value`.
All sub-skills read these two fields when they need the current metric.

Read this file at startup to resume a partially completed pipeline.

---

## Stage 0 — Startup

1. **Check for `research_config.yaml`** in the project root:
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

1.5. **Load `.env` for API keys.** Secrets (API keys, tokens) live in a `.env`
   file in the project root, NOT in `research_config.yaml` (which may be
   committed to git). Load it before state init:

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

2. Check if `pipeline_state.json` exists:
   - **Exists** → read it, print current stage, resume from that stage
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

     state = {
       # identity
       "project_name":          cfg["meta"]["project_name"],
       "run_tag":               cfg["meta"].get("run_tag"),
       "task":                  cfg["task"]["description"],
       "task_type":             cfg["task"].get("task_type", "object_detection"),
       "dataset":               cfg["paths"]["dataset_root"],
       "stage":                 "init",
       # skill outputs
       "paper_finder_done":     False,
       "base_model_md_ready":   False,
       "modules_md_ready":      False,
       "pretrain_done":         False,
       "pretrain_skipped":      False,
       "pretrain_offer_declined": False,
       "pretrain_weights":      None,
       "base_weights_local":    None,
       "autoresearch_running":  False,
       # metric tracking (metric-agnostic field names)
       "primary_metric_name":   PRIMARY,
       "best_primary_value":    None,        # set by autoresearch on first keep
       "stall_count":           0,
       "loop_count":            0,
       "seed":                  ar.get("loop", {}).get("seed", 42),
       "paper_finder_expansions": 0,
       # paths
       "local_papers_dir":      cfg["paths"].get("local_papers_dir"),
       "local_datasets_dir":    cfg["paths"].get("local_datasets_dir"),
       "skills_dir":            os.path.expanduser(cfg["paths"].get("skills_dir", "~/.claude/skills")),
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
       "improvement_threshold": MIN_IMPROVE,              # from evaluation.metrics.min_improvement
       "loop_iterations":       ar.get("loop", {}).get("iterations"),
       # orchestrator
       "crash_pause_after":     orc.get("error_policy", {}).get("autoresearch_crash_pause_after", 3),
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

3. Verify skill files exist (using `skills_dir` from pipeline state, default: `~/.claude/skills`):
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

4. Verify paper2code is available (optional but recommended):
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

5. Initialise git if needed:
   ```bash
   git rev-parse --git-dir 2>/dev/null || (git init && git add -A && git commit -m "init")
   ```

6. **Check for `train.py` and friends:**

   All skills in this pipeline assume `train.py` follows the contract documented
   in `<skills_dir>/shared/train-script-spec.md`. Read that file once now so
   later stages can rely on its guarantees.

   ```bash
   cat "$SKILLS_DIR/shared/train-script-spec.md"
   ```

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
   def check_spec_compliance(path):
       src = pathlib.Path(path).read_text()
       missing = []
       for section in ["Section ①", "Section ②", "Section ③", "Section ④"]:
           if section not in src:
               missing.append(section)
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

Update `pipeline_state.json` → `stage: "paper_finder"` and proceed immediately.
All subsequent stages read their parameters from `pipeline_state.json` — never hardcode
values that should come from `research_config.yaml`.

---

## Stage 1 — Paper Finder (initial)

**Entry condition:** `paper_finder_done == false`

1. Read `$SKILLS_DIR/paper-finder/SKILL.md`
2. Run paper finder with `mode: initial` and the task description from pipeline state
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

---

## Stage 2 — Dataset Hunter (pretrain optional)

**Entry condition:** `dataset_hunter_enabled == true` AND `pretrain_done == false` AND `pretrain_skipped == false`

**Patience:** Stage 2 is the longest stage. Dataset downloads alone may take 1–3
hours depending on how many sources are available. Pretrain adds another
`pretrain_time_budget` seconds (default 6 hours). Self-eval adds
`self_eval_time_budget` × 2 (two finetune jobs). **Do not skip, shortcut,
or abort Stage 2 because it is taking a long time.** The only valid reasons
to skip pretrain are: (a) corpus is empty after all downloads, or (b) the
user set `dataset_hunter.enabled: false` in the yaml.

1. Read `$SKILLS_DIR/dataset-hunter/SKILL.md`
2. Pass from pipeline state: disk budget, Roboflow API key, pretrain TIME_BUDGET, `local_datasets_dir`
3. **Resolve base model weights to a local path before anything else:**
   ```python
   import re, pathlib, subprocess
   text = pathlib.Path("base_model.md").read_text()
   m = re.search(r"Weights URL.*?:\s*(.+)", text)
   weights_url = m.group(1).strip() if m else None

   if weights_url and weights_url.startswith("http"):
       local_path = f"weights/{pathlib.Path(weights_url).name}"
       pathlib.Path("weights").mkdir(exist_ok=True)
       subprocess.run(["wget", "-c", weights_url, "-O", local_path], check=True)
       BASE_WEIGHTS = local_path
   elif weights_url == "reconstruct via paper2code":
       # Extract arXiv ID from base_model.md and run paper2code now
       arxiv_m = re.search(r"arxiv\.org/abs/([\w.]+)", text)
       if arxiv_m:
           arxiv_id = arxiv_m.group(1)
           print(f"Running paper2code for base model: {arxiv_id}")
           # Invoke paper2code skill
           # After paper2code completes, find the generated checkpoint
           ckpts = sorted(pathlib.Path(".").glob("*/checkpoints/*.pt")) +                    sorted(pathlib.Path(".").glob("*/src/*.pt"))
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
   # Update pipeline state
   # pipeline_state["base_weights_local"] = BASE_WEIGHTS
   ```
   Store `BASE_WEIGHTS` in `pipeline_state.json` as `"base_weights_local"`.

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
     - Find the pretrain checkpoint:
       ```bash
       PRETRAIN_CKPT=$(ls -t pretrain_ckpt/*.pt 2>/dev/null | head -1)
       ```
     - Read dataset hunter's recommendation (written to `pretrain_eval.json` in Phase 6).
       - recommendation == "use_pretrained" → `pretrain_weights = $PRETRAIN_CKPT`
       - recommendation == "use_original" or "inconclusive" → `pretrain_weights = null`
     - Update `pipeline_state.json`:
       ```json
       { "stage": "autoresearch", "pretrain_done": true,
         "pretrain_weights": "<resolved path or null>" }
       ```
     - Proceed immediately to Stage 3.

---

## Stage 3 — Autoresearch Loop

**Entry condition:** `autoresearch_running == false` or resuming

1. Read `$SKILLS_DIR/autoresearch/SKILL.md`

2. **Resolve and patch `WEIGHTS` in `train.py`.**

   The template ships with a placeholder (`weights/yolo26x.pt`). Orchestrator
   must set this to the actual weights path before the first run, or `train.py`
   will crash with `FileNotFoundError`.

   ```python
   import re, pathlib, subprocess, json, yaml

   state = json.loads(pathlib.Path("pipeline_state.json").read_text())
   train_py = pathlib.Path(state["train_script"])

   def patch_variable(path, name, value):
       """Replace ^<name>\s*=.*$ with <name> = <value>. Not locked."""
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
       # Download from base_model.md
       bm = pathlib.Path(state["base_model_md_path"])
       weights_local = None
       if bm.exists():
           bm_text = bm.read_text()
           m = re.search(r"Weights URL.*?:\s*(.+)", bm_text)
           url = m.group(1).strip() if m else None
           if url and url.startswith("http"):
               pathlib.Path(state["weights_dir"]).mkdir(exist_ok=True)
               local = f"{state['weights_dir']}/{pathlib.Path(url).name}"
               if not pathlib.Path(local).exists():
                   subprocess.run(["wget", "-q", "-c", url, "-O", local], check=True)
               weights_local = local

   if weights_local and pathlib.Path(weights_local).exists():
       patch_variable(train_py, "WEIGHTS", f'"{weights_local}"')
       state["base_weights_local"] = weights_local
   else:
       print("WARNING: no weights file resolved. train.py WEIGHTS must point to "
             "an existing .pt file or the first run will crash.")
   ```

2.5. **Patch `DATA_YAML` and `NUM_CLASSES` in `train.py`.**

   The template ships with `DATA_YAML = "data/dataset.yaml"` — a placeholder.
   Orchestrator must patch this to match `paths.dataset_root` from the yaml,
   or `train.py` will crash with `FileNotFoundError`.

   ```python
   dataset_root = pathlib.Path(state["dataset"])

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
       # Read NUM_CLASSES from data.yaml
       try:
           ds_cfg = yaml.safe_load(open(data_yaml))
           nc = ds_cfg.get("nc") or len(ds_cfg.get("names", []))
           if nc:
               patch_variable(train_py, "NUM_CLASSES", nc)
       except Exception:
           pass
   else:
       print(f"WARNING: no data.yaml found under {dataset_root}. "
             f"Patch DATA_YAML in train.py manually before running.")

   # Save updated state
   json.dump(state, open("pipeline_state.json", "w"), indent=2)
   ```

3. **Lock `TIME_BUDGET`, `SEED`, and `IMGSZ` across all experiment scripts.** For
   `task_type: object_detection` this is just `train.py`; for
   `object_tracking` it is `train.py` + `track.py` (both have Section ②
   per the spec). Helper function below fails loudly if any variable is
   missing — a silent no-op here would let experiments run with stale
   values and silently corrupt results.

   ```python
   def lock_variable(path, name, value):
       """Replace ^<name>\s*=.*$ with <name> = <value>  # locked by orchestrator.
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
   ```

   These three values (TIME_BUDGET, SEED, IMGSZ) are fixed for the entire pipeline
   run — identical across every experiment so that keep/discard decisions reflect
   the change being tested, not randomness or resolution differences.

4. Update pipeline state: `autoresearch_running: true`

5. Run autoresearch loop (this runs indefinitely — do not stop it).
   The loop internally handles:
   - Reading modules.md for pending entries
   - Calling paper2code for module code generation
   - Injecting into train.py
   - keep/discard decisions
   - stall detection

   **stall_count authority:** `pipeline_state.json` is the single source of truth.
   Autoresearch writes its stall count to `pipeline_state.json` at every Step 9.
   Orchestrator reads from there. Never maintain a separate counter — always sync to file.

6. **Stall → paper finder expand handoff:**
   Orchestrator is the sole owner of this handoff. Autoresearch only writes `stall_count`
   to `pipeline_state.json` — it never calls paper finder itself.

   After each autoresearch loop iteration, orchestrator checks `pipeline_state.json`
   and `modules.md` via the canonical parser:
   ```python
   import sys, pathlib, json
   state = json.loads(pathlib.Path("pipeline_state.json").read_text())
   sys.path.insert(0, str(pathlib.Path(state["skills_dir"]) / "shared"))
   import modules_md as mm

   pending_count = mm.count_pending("modules.md")
   if state["stall_count"] >= state["stall_threshold"] and pending_count == 0:
       # trigger paper finder expand
   ```
   When condition is met:
   a. Update pipeline state: `paper_finder_expansions += 1`
   b. Read `$SKILLS_DIR/paper-finder/SKILL.md`
   c. Run paper finder with `mode: expand`, passing current `modules.md`
      Pass `local_papers_dir` from pipeline state so paper finder re-scans local PDFs too
   d. After paper finder completes:
      Update pipeline state: `stall_count: 0`
   e. Return to autoresearch loop — read the skill again and continue

6.5. **Architecture combination → re-pretrain cycle:**
   After each autoresearch loop iteration, also check for the re-pretrain signal:
   ```python
   if state.get("request_repretrain") and state.get("pretrain_done"):
       # Architecture combination triggered re-pretrain
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
   e. Clear the signal: `state["request_repretrain"] = False`
   f. Resume autoresearch loop from Step 1

   This cycle typically fires after 3+ architecture changes are combined. It
   lets modules that only shine with pretrained features show their real value.

7. **Automatic pretrain trigger (no user interaction):**
   If `pretrain_skipped == true` AND `pretrain_offer_declined == false`
   AND autoresearch has run ≥ 5 loops AND `best_primary_value` has not improved in last 5 loops:
   - **Automatically** run pretrain (do not ask the user):
     1. Log: `"Auto-triggering pretrain after 5 stalled loops"`
     2. Read `$SKILLS_DIR/dataset-hunter/SKILL.md`
     3. Skip directly to **Phase 5 — Pretrain** (dataset hunter's phases 1–4 already done)
        using `BASE_WEIGHTS` from `pipeline_state.base_weights_local`
        and corpus already in `pretrain_data/merged/`
     4. Run pretrain + self-eval (Phase 5 + Phase 6 of dataset hunter only)
     5. Resolve new `pretrain_weights` and update `pipeline_state.json`
     6. Patch `train.py`: `WEIGHTS = "<new pretrain_weights>"`
     7. Set `pipeline_state.pretrain_offer_declined = true` (one-time only,
        prevents infinite pretrain loops)
     8. Resume autoresearch loop from Step 1
   - If corpus in `pretrain_data/merged/` is empty (nothing was downloaded in
     Stage 2), set `pretrain_offer_declined = true` and skip — cannot pretrain
     with no data.

8. Update `best_primary_value` in pipeline state whenever autoresearch keeps a result.
   Autoresearch already writes this field at Step 9 of its loop; orchestrator only
   needs to read it for summary printing.

---

## Stage 4 — Done

Triggered only by user interruption (Ctrl+C) or explicit user command "stop".

1. Print final summary. The primary metric name is read from pipeline state —
   do not hardcode `val_mAP50_95` or any specific metric:
   ```python
   import json, pathlib
   state = json.loads(pathlib.Path("pipeline_state.json").read_text())
   primary = state["primary_metric_name"]
   best    = state["best_primary_value"]
   ```
   ```
   === Pipeline Summary ===
   Task:                  <task>
   Task type:             <task_type>
   Best <primary>:        <best>
   Total autoresearch loops: <pipeline_state.loop_count>
   Paper finder expansions:  <paper_finder_expansions>
   Pretrain used:         <yes/no>
   Best weights:          <path from git log>

   Top 5 experiments:
   <top 5 rows from results.tsv sorted by the primary metric column>
   ```
   The "sort by primary metric" step must respect `evaluation.metrics.minimize` —
   if the primary metric is in `minimize`, sort ascending; otherwise descending.

2. **Print discoveries** — if `discoveries.md` exists, print its contents.
   This is where autoresearch logged technical insights during the run
   instead of stopping to ask the user:
   ```python
   disc = pathlib.Path("discoveries.md")
   if disc.exists() and disc.stat().st_size > 100:  # more than just the header
       print("\n=== Discoveries (observations from the loop) ===")
       print(disc.read_text())
   ```

3. Update pipeline state: `stage: "done"`

---

## Resume logic

If `pipeline_state.json` exists and `stage` is not `"init"`:

| stage value      | Resume action |
|------------------|---------------|
| `paper_finder`   | Re-run Stage 1 from scratch (paper finder is idempotent) |
| `dataset_hunter` | Check `pretrain_skipped` flag — if true skip to Stage 3 directly; else check if `pretrain_ckpt/` exists — if yes skip to Stage 3, else re-run Stage 2 from pretrain step |
| `autoresearch`   | Skip to Stage 3, autoresearch reads existing results.tsv and continues |
| `done`           | Print summary and exit |

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
| Autoresearch crash 3x in a row | Print last 50 lines of `run.log`, revert to last known good commit (`git log --oneline -5`), reduce `BATCH_SIZE` by half, reset crash counter, continue looping. Do **not** pause or ask the user. |
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
   retry then continue, missing file → fallback, stall → expand. If you find
   yourself about to type a question to the user, stop — there is a default in
   this document or in the sub-skill. Use it.
