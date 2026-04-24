# Changelog — v1.7.5

Release type: **Onboarding hardening.** 11 fixes across orchestrator,
templates, and cross-skill contracts to resolve friction points observed
when a real user ran v1.7.4 end-to-end on a fresh machine. No schema
changes, no new features, no behavioural changes to any algorithm.
Drop-in on top of v1.7.4.

This is the first release that consolidates observed-in-the-wild
friction rather than addressing theoretical issues. The v1.7.0 → v1.7.4
chain fixed bugs discovered during design review; v1.7.5 fixes bugs
discovered during actual pipeline runs.

## Categories

| Category | # |
|---|---|
| Onboarding / environment setup | #2, #3, #4, #5, #7 |
| Cross-skill coordination | #1, #6, #8 |
| Template hardening | #9, #10 |
| Spec robustness | #11 |

## Per-fix details

### Fix #1 — Local skill precedence (cross-skill)

**Symptom**: User ran `Skill: yolo-research-orchestrator` and loaded a
cloud-hosted skill that expected sub-skills named `dataset-hunter-for-yolo`
and `autoresearch-for-yolo`. Local repo has `dataset-hunter` and
`autoresearch` — invocation failed with "sub-skill not found".

**Fix**: `research-orchestrator/SKILL.md` opens with an explicit "Skill
precedence" block listing the four canonical sub-skill names, telling the
agent to ignore similarly-named cloud skills and always use the local
`<skills_dir>/` as authoritative.

### Fix #2 — `~/skills` default path

**Symptom**: yaml default was `"~/.claude/skills"`. The `~` was not
expanded by all callers; also the machine didn't have anything at
`~/.claude/skills`. Skills loaded from the wrong place or not at all.

**Fix**:
- Default changed to `"./skills"` (relative to the yaml's directory)
- New `resolve_skills_dir()` helper in orchestrator Stage 0 Step 2:
  handles `~`, `$HOME`, relative paths, and validates the `shared/`
  subdirectory exists before proceeding
- Two bash fallbacks (`echo "$HOME/.claude/skills"`) updated to
  `"./skills"`

### Fix #3 — Git identity auto-configuration

**Symptom**: `git init` succeeded, then first `git commit` failed with
`fatal: unable to auto-detect email address (got '<user>@<host>.(none)')`
because the machine had no `user.email` in global config.

**Fix**: Stage 0 Step 5 now checks `git config --local --get user.email`
and auto-sets a per-repo identity (`pipeline@<project>.local` /
`autoresearch pipeline`) if absent. Uses `--local` only, never touches
global config. User can override at any time.

### Fix #4 — python_runner local-first preference

**Symptom**: Fresh machine had `uv` installed but no `pyproject.toml`.
Previous logic `"uv run" if shutil.which("uv") else "python3"` picked
`uv run`, which created an empty venv, and `train.py` immediately
crashed with `ModuleNotFoundError: numpy`.

**Fix**: New `choose_python_runner()` helper tries system `python3`
first — if it can `import ultralytics`, use it. Only fall through to
`uv run` when:
1. `uv` is on PATH, AND
2. `pyproject.toml` exists, AND
3. `uv run python3 -c "import ultralytics"` returns 0

Otherwise default to `python3` and log a clear reason.

### Fix #5 — Optional `pyproject.toml` generation

**Symptom**: Users who genuinely want uv-managed envs had no scaffolding;
they had to write `pyproject.toml` by hand.

**Fix**: New Stage 0 Step 6.5 writes a minimal `pyproject.toml` when
`python_runner == "uv run"` OR yaml has
`orchestrator.use_uv_project: true`. Runs `uv sync` afterwards. If sync
fails (network, CUDA, etc.), reverts to `python3` with a logged reason
rather than leaving the project in a broken state.

### Fix #6 — `preferred_base_model` yaml field

**Symptom**: User wanted to pin YOLO26X (no formal arXiv paper,
ultralytics release). Paper-finder's Phase 4 scoring assumed every
candidate had an arxiv_id; the override workflow was undocumented, so
the user improvised with a yaml field that paper-finder didn't read.

**Fix**:
- `paper_finder/SKILL.md` Phase 1 new "User-specified base model"
  sub-phase: reads `task.preferred_base_model` +
  `task.preferred_base_weights_url` (+ optional
  `preferred_base_arxiv_id`). If both present, writes `base_model.md`
  directly with `Source: user override`, sets
  `state["base_model_user_override"] = True`, **skips Phase 2-4**,
  proceeds to Phase 5 (modules collection).
- Both example yamls document the three new fields (commented out).

### Fix #7 — `weights/` directory + probe safety

**Symptom**: Agent ran `YOLO("yolo26x.pt")` as a diagnostic probe
before pipeline setup. Ultralytics auto-downloaded 113 MB into **cwd**
(project root) instead of `weights/`.

**Fix**: Orchestrator Stage 0 Step 6 creates `weights/` early,
documents the rule: **any model-loading diagnostic call must use an
explicit path like `YOLO("weights/yolo26x.pt")`**, never bare
filename.

### Fix #8 — `pretrain.time_budget_sec: 0` skip path

**Symptom**: Stage 2's explicit rule said "only valid reasons to skip
pretrain are: (a) corpus empty, (b) `dataset_hunter.enabled: false`".
User wanted to start Loop 1 within the hour rather than waiting 7+
hours for pretrain; there was no legal skip route, forcing the agent
to take a discretionary decision it wasn't authorised to make.

**Fix**: New legal skip path — `dataset_hunter.pretrain.time_budget_sec: 0`:
- Dataset search still runs (modules.md coverage preserved for later
  re-pretrain)
- Pretrain (Phase 5-6) skipped with `pretrain_skipped = true`,
  `pretrain_offer_declined = true`
- Stage 2 entry condition now reads `pretrain_time_budget > 0`
- Both example yamls document this option inline

To skip both search AND pretrain, `dataset_hunter.enabled: false`
remains the stronger form.

### Fix #9 — `ULTRALYTICS_RUNS_DIR` environment shield

**Symptom**: User-wide ultralytics settings.json (or
`ULTRALYTICS_RUNS_DIR` env var) forced `runs/` to a directory outside
the project. Result: autoresearch's Step 6 looking for
`runs/train/weights/best.pt` got confused paths, and wandb run dirs
followed the rogue location.

**Fix**: All three templates (`train.py.detection`,
`train.py.tracking`, `track.py.tracking`) now reset the env var and
`SETTINGS.runs_dir` at import time, before any `YOLO()` call:

```python
import os as _os
_os.environ.pop("ULTRALYTICS_RUNS_DIR", None)
try:
    from ultralytics.utils import SETTINGS as _SETTINGS
    _SETTINGS.update({"runs_dir": str(Path("runs").resolve())})
except Exception:
    pass  # older ultralytics; env var reset alone still helps
```

Wrapped in try/except so older ultralytics versions without SETTINGS
don't crash the template.

### Fix #10 — WANDB_MODE=offline default

**Symptom**: If wandb was logged in globally (common on research
machines), ultralytics auto-attached its callback and created a new
run per iteration. After 30 iterations, dashboard was 30 identical
runs.

**Fix**: Same shield block in all three templates adds
`os.environ.setdefault("WANDB_MODE", "offline")`. `setdefault` means
the user can still override externally if they actually want online
wandb — set `WANDB_MODE=online` in shell or edit the template.

### Fix #11 — Section marker regex tolerance

**Symptom**: Spec compliance check at Stage 0 Step 6 did literal
substring search for `"Section ①"`, `"Section ②"`, etc. If any
editor / tool replaced the Unicode circled digits with `"Section 1"`,
`"Section 2"`, spec compliance would fail and templates would be
rejected as broken.

**Fix**: Four regex patterns replacing four literal lookups:
- `research-orchestrator/SKILL.md` `check_spec_compliance`
- `dataset-hunter/SKILL.md` `patch_section_2` (section boundary search)
- `shared/test_templates.py` `REQUIRED_SECTIONS` → `REQUIRED_SECTION_PATS`
- `shared/train-script-spec.md` example code

Pattern: `re.compile(r"(?mi)^#?\s*Section\s*[②2]\b")` accepts:
- `# Section ② — Tunables`
- `# Section 2 — Tunables`
- `#Section 2` (no space)
- Mixed case (`section`, `SECTION`)

Existing templates (Unicode) still pass; ASCII templates written by
editors that strip Unicode now also pass.

## Files changed in v1.7.5

| File | Change summary |
|---|---|
| `research-orchestrator/SKILL.md` | Skill precedence block, `resolve_skills_dir`, `choose_python_runner`, git identity, Step 6.5 pyproject.toml, `weights/` dir, Stage 2 `time_budget=0` skip, `check_spec_compliance` regex |
| `paper-finder/SKILL.md` | Phase 1 user-specified base model short-circuit |
| `dataset-hunter/SKILL.md` | `patch_section_2` regex |
| `autoresearch/SKILL.md` | **unchanged** |
| `shared/templates/train.py.detection` | Env shield block |
| `shared/templates/train.py.tracking` | Env shield block |
| `shared/templates/track.py.tracking` | Env shield block |
| `shared/test_templates.py` | `REQUIRED_SECTIONS` → `REQUIRED_SECTION_PATS` regex |
| `shared/train-script-spec.md` | Spec example updated to regex form |
| `examples/research_config.visdrone-detection.yaml` | `skills_dir` default, `preferred_base_model`, `pretrain.time_budget_sec: 0` docs |
| `examples/research_config.visdrone-mot.yaml` | Same 3 updates |
| `CHANGELOG_v1.7.5.md` | New (this file) |
| `README.md` | Version bumped, layout tree, Versions list |

## Files NOT changed

- `shared/weight_transfer.py` — unchanged
- `shared/modules_md.py` — unchanged
- `shared/state_migrate.py` — unchanged
- `shared/parse_metrics.py` — unchanged
- `shared/templates/arch_spec.schema.json` — unchanged
- `test_modules_md.py` — unchanged
- `test_weight_transfer.py` — unchanged
- `autoresearch/SKILL.md` — unchanged (v1.7.4 tie-breaking rule holds)

## Test coverage

| Suite | v1.7.4 | v1.7.5 | Notes |
|---|---|---|---|
| `shared/test_modules_md.py` | 18 | 18 | unchanged |
| `shared/test_templates.py` | 3 | 3 | regex-updated but still 3 templates |
| `shared/test_weight_transfer.py` | 36 | 36 | unchanged |
| SKILL.md python snippets | 83 | **88** | +5 new snippets from #5, #6, #8 |
| YAML examples | 2 | 2 | unchanged |
| **Total python tests** | **57** | **57** | |

No new test file. Fixes are either doc-level (SKILL.md regex changes
verified by the existing template parse tests) or environment-level
(subprocess / env var behaviour — covered by fresh install-from-zip
verification).

## Upgrade path

Drop-in. Existing `research_config.yaml` files keep working:
- `skills_dir: "~/.claude/skills"` still works via the new
  `resolve_skills_dir` helper (it expands `~`)
- Existing `pretrain.time_budget_sec: 21600` unchanged; only `: 0`
  activates the new skip path
- Missing `task.preferred_base_model` → normal Phase 2-4 search runs
- Existing templates (Unicode section markers) pass both old and new
  regex
- Existing `state["python_runner"]` is overwritten on next Stage 0 by
  the smarter detection

For fresh projects, new yaml examples document all new options inline.

## Operational notes

### Recommended: verify your environment shield works

After upgrade, run the first iteration and confirm:
- `runs/train/` exists in project root (not `/some/other/path/runs/...`)
- No new wandb run created (unless you set `WANDB_MODE=online`)

If either assumption is violated, your ultralytics installation has a
deeper user-wide override — check `~/.config/Ultralytics/settings.json`
and the shell environment.

### Recommended: set preferred_base_model if using ultralytics releases

If your base model is an ultralytics release (YOLO11x, YOLO26X, etc.)
that doesn't have a formal arXiv paper, set `task.preferred_base_model`
in your yaml. Paper-finder's Phase 2-4 search wastes several minutes
on models with no findable paper before giving up.
