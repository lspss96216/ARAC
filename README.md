# Research Pipeline Skills

Autonomous ML experiment pipeline. Four Claude skills
(orchestrator, paper finder, dataset hunter, autoresearch) coordinate via
shared files to: find the best base model + improvement modules, build a
pretrain corpus, run pretrain, then iterate `train.py` through experiments
keeping what improves the primary metric.

Supports `task_type: object_detection` and `object_tracking`, driven entirely
by `research_config.yaml`. No skill hardcodes a specific metric.

**This is v1.12.1** — patch wiring up dead config + correcting stale
yaml comment. `optional_pretrain_trigger` (yaml since v1.7, documented
in v1.8 SKILL prose, never evaluated in code) now actually fires —
default `enabled: false` so behaviour is preserved for current users.
`autoresearch_crash_pause_after` yaml comment corrected to v1.12's
log-only behaviour. Pure SKILL+yaml patch, no code or tests changed.
See `CHANGELOG_v1.12.1.md`.

**This is v1.12** — strict fair-comparison mode. **BREAKING**:
`BATCH_SIZE` is now LOCKED across all iterations (same enforcement as
IMGSZ/SEED/TIME_BUDGET). Auto-halve paths removed: v1.9
`resource_impact` halve replaced with predictive skip-and-block,
v1.7.6 crash-pause halve replaced with log-only, ultralytics auto-
batch-reduce now caught as ContractViolation. New `effective_resource_impact`
property: `scope: all` on yaml_inject escalates the tag (`none` →
`vram_2x`, `vram_2x` → `vram_4x`) — fixes real-world FlexSimAM
silent CPU fallback. yaml field renamed `initial_batch_size` →
`batch_size` (alias kept). +18 tests (143 → 161). See
`CHANGELOG_v1.12.md` for migration steps.

**This is v1.11.1** — patch release fixing 4 bugs surfaced by a
v1.11 production run. **BREAKING change**: `reapply_on_rebuild`
callback signature `cb(m) → cb(layers)`. Other 3 fixes are
non-breaking: PicklableHook auto device sync, inject_modules
ModuleList anti-pattern invariant, yaml-configurable subagent_type +
subagent_model. See `CHANGELOG_v1.11.1.md` for migration steps.

**This is v1.11** — concurrent paper-finder during baseline. Save
~2h wallclock per pipeline startup by spawning paper-finder Phase 5-6
as a subagent in parallel with Loop 0 vanilla baseline. Defaults to
enabled; user can disable via `orchestrator.concurrent_paper_finder.
enabled: false`. 30 min timeout post-baseline → fall back to
sequential. Stage 1 splits Phase 1-4 (sequential, blocks Stage 2)
from Phase 5-6 (concurrent with Loop 0). Drop-in. SKILL+yaml only.
See `CHANGELOG_v1.11.md`.

**This is v1.10** — paper-finder source diversification + dedup. Adds
HuggingFace Papers as Source 6 (ML-specialised ranking, fixes PwC
coverage gaps) and a new Phase 2.5 cross-source dedup step (collapses
the same paper appearing across arXiv / PwC / S2 / HF). SKILL-only
changes — no new Python modules, no new dependencies, no new tests.
Drop-in. See `CHANGELOG_v1.10.md` for what was deferred (Phase 3.5
read methodology, Phase 5 citation graph) pending v1.9.3 real-run
feedback.

**This is v1.9.3** — patch release. Adds
`autoresearch.loop.initial_batch_size` yaml field. v1.6 → v1.9.2
hardcoded train.py's initial BATCH_SIZE to template default (16) with
no yaml knob; users on H100 80GB ran experiments at batch=16 with no
obvious way to change it. v1.9.3 makes the starting value
yaml-configurable while preserving autoresearch's runtime halve
authority (resource_impact, crash-pause, OOM). Drop-in. See
`CHANGELOG_v1.9.3.md`.

**This is v1.9.2** — patch release. Fixes cross-project run.log
pollution: an agent in the wrong cwd was reading a neighbour project's
run.log and parsing metrics from a different experiment, silently
corrupting results.tsv. 4 layers of defence: (1) `state["project_root"]`
absolute path canonicalised at Stage 0, (2) `__RUN_START__` /
`__RUN_END__` sentinels in run.log, (3) Step 5 prelude with cwd-lock +
stale-file cleanup + start-time recording, (4) Step 6 pre-parse
freshness check via new `check_run_log_fresh` invariant. Total tests
123 → 129. Drop-in. See `CHANGELOG_v1.9.2.md`.

**This is v1.9.1** — patch release. Wires up
`autoresearch.module_priority` yaml field which had been dead config
since v1.6 (yaml example listed it; no skill ever read it). Same
class as v1.7.3's `preferred_locations` fix. Tokens omitted from the
list are now SKIPPED ENTIRELY, enabling architecture-only or
hyperparameter-only runs via yaml. Drop-in. See
`CHANGELOG_v1.9.1.md`.

**This is v1.9** — `full_yaml` mode + resource-impact auto-halve.
Implements `weight_transfer.apply_yaml_spec` end-to-end (the v1.7-v1.8
placeholder that raised `NotImplementedError`), unblocking the structural-
rewrite class of experiments (BiFPN replacing PAFPN, DETR-style decoders,
ConvNeXt backbones, P2 head architectures). Adds autoresearch dispatch
branch + paper-finder Phase 5 heuristic + modules.md schema +
train-script-spec.md chapter for full_yaml mode. Separately adds
`resource_impact` field on modules.md (vram_4x / vram_2x /
cpu_fallback_risk) and autoresearch Step 3 auto-halve to prevent the
silent CPU-fallback class of failures observed in real runs. Total
python tests **106 → 123**. Drop-in. See `CHANGELOG_v1.9.md`.

**This is v1.8** — production-usability + agent contract enforcement.
Combines real-world fixes from a 31-loop production run (3 weight_transfer
bugs, base-aware after_class, project= absolute path, mandatory Loop 0
vanilla baseline) with autonomous stop triggers (no_improvement_loops,
max_total_loops, max_wallclock_hours), pretrain-trigger semantic clarification,
description format contract, per-loop log archive, expanded discoveries.md
categories, and a new **`shared/invariants.py`** module (23 tests) that
catches agent contract violations at runtime. Total python tests **82 → 106**.
**No new architectural capability** — full_yaml mode reserved for v1.9.
Drop-in. See `CHANGELOG_v1.8.md`.

**This is v1.7.7** — production-readiness for hook + yaml_inject.
Six fixes that make architectural injection actually work end-to-end:
yaml_inject's silent forward-pass crash from unshifted head Concat
references (#13); hook-mode's silent failure where ultralytics' Trainer
rebuilds the model and loses all hooks (#14); checkpoint pickle failure
from closure-based hooks (#15); val-phase dtype mismatch under AMP (#17);
unpredictable breakage from `layer.forward = wrapper` bypassing
`_call_impl` (#18); silent LR override under `optimizer='auto'` (#16);
ambiguous tiebreak rule causing inconsistent keep/discard verdicts
(#19). New `shared/hook_utils.py` (`PicklableHook` base class,
`reapply_on_rebuild` helper) + 13 tests. New `update_head_refs` in
`weight_transfer.py` + 12 tests. Total python tests **57 → 82**.
Drop-in. v1.7.x line ends here. See `CHANGELOG_v1.7.7.md`.

**This is v1.7.6** — bug fix release. Four real bugs found during code
review of v1.7.5: crash-pause sequence reverted its own halve
(deadlock-prone under sustained crashes); BATCH_SIZE halving stayed
silently at 1 forever; Step 5.5 short-test restore read a state key
that didn't exist (`time_budget_sec` instead of `loop_time_budget`);
state_migrate's stale `python_runner` default could lock the pipeline
into a broken runner across upgrade. Plus removal of v1.7.5's
auto-write `pyproject.toml` feature, which had three independent
implementation problems and the wrong scope (env management is the
user's job; runner choice is the pipeline's). Step 6.5 now does
verify-only, with a clear two-option remediation message on failure.
All 57 tests + 88 SKILL snippets still green. Drop-in. See
`CHANGELOG_v1.7.6.md`.

The v1.7.x line ends here. v1.8 work (full_yaml mode + apply_yaml_spec)
begins immediately.

**This is v1.7.5** — onboarding hardening. 11 fixes addressing friction
points observed when a real user ran v1.7.4 end-to-end on a fresh
machine: local skill precedence over cloud-hosted similarly-named skills,
relative `skills_dir` default, git identity auto-config, smarter
python_runner detection (system python3 first), optional pyproject.toml
generation, user-specified base model short-circuit, weights/ directory
safety, `pretrain.time_budget_sec: 0` skip path, environment shield for
ULTRALYTICS_RUNS_DIR and WANDB, Section marker regex tolerance. No
schema changes, no new features, no behavioural changes to any
algorithm. Drop-in on top of v1.7.4. See `CHANGELOG_v1.7.5.md`.

**This is v1.7.4** — documentation bug fix. Adds a Tie-breaking rule
to `autoresearch/SKILL.md` Step 2 so the loop always has a
deterministic fallback when multiple pending modules look equally
valid. Before v1.7.4, autoresearch's agent sometimes paused mid-loop
to ask the user "which should I try first?" despite Critical Rule #13
saying "never talk to the user mid-loop" — because the rules didn't
spell out what to do *instead* when genuinely tied. The new rule
names the 4-step tie-breaker (write order → lower complexity →
preferred_locations → alphabetical) and lists the common
rationalisations that are NOT valid reasons to stop. Pure SKILL.md
prose change, all 57 tests still green. See `CHANGELOG_v1.7.4.md`.

**This is v1.7.3** — bug fix making `paper_finder.modules.preferred_locations`
actually affect autoresearch iteration order. The yaml field had been
read only by paper-finder (Stage 1 search filtering); autoresearch's
`find_pending` sorted purely on complexity and ignored the yaml
entirely, so `[backbone, neck, head, loss]` looked meaningful but
didn't influence which pending module got picked first. v1.7.3 adds
`preferred_locations` as a secondary sort key (complexity → location
rank → write order). Unlisted locations still get picked, they just
sort after listed ones. 57 tests green (+6). See `CHANGELOG_v1.7.3.md`.

**v1.7.2** — bug fix for a cross-skill `IMGSZ` handoff gap.
`research_config.yaml → evaluation.ultralytics_val.imgsz` was correctly
locked into `train.py` by orchestrator Stage 3, but was silently lost
when dataset-hunter derived `pretrain.py` from the detection template
(stuck at 1920 regardless of user config), and referenced as
`state["imgsz"]` in v1.7.1's yaml_inject repair loop without anyone
writing it. Both sites now read a canonical `state["imgsz"]` that
orchestrator persists. See `CHANGELOG_v1.7.2.md`.

This is purely a cross-skill state-handoff fix. No schema changes, no
new features, no library changes. 51 tests still green, drop-in on top
of v1.7.1.

**v1.7.1 features remain** — adaptive repair loop (Step 5.5). When a
yaml_inject or hook experiment crashes, autoresearch classifies the
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

**v1.7 features remain** — architectural injection (yaml_inject mode):
modules that insert a new layer mid-backbone work end-to-end via
`weight_transfer.py`, which generates a modified YAML, builds the
model, transfers weights from the base `.pt` per a computed layer_map,
and forces lazy modules to build before the optimizer captures
parameters. See `CHANGELOG_v1.7.md`.

v1.6 → v1.7 → v1.7.1 → v1.7.2 is a pure-addition-and-fix chain: every
existing code path, contract, and state file is preserved. Modules
without the `Integration mode` field default to `hook` (the v1.6
path). State files from v1.5 / v1.6 / v1.7 / v1.7.1 resume
transparently via `shared/state_migrate.py`.

---

## Layout

```
skills/                                     → copy to ~/.claude/skills/
├── shared/
│   ├── modules_md.py                       · modules.md parser (23 unit tests) [v1.7+v1.9]
│   ├── state_migrate.py                    · pipeline_state.json schema migration [v1.6]
│   ├── parse_metrics.py                    · shared stdout/json/csv metric extractor [v1.6]
│   ├── weight_transfer.py                  · yaml_inject + pretrained transfer [v1.7]
│   │                                         + Step 5.5 repair primitives  [v1.7.1]
│   │                                         + update_head_refs (head ref shifting) [v1.7.7]
│   │                                         + compute_layer_map fix + filename scale suffix [v1.8]
│   │                                         + apply_yaml_spec full_yaml mode [v1.9]
│   ├── hook_utils.py                       · PicklableHook + reapply_on_rebuild (13 tests) [v1.7.7]
│   ├── invariants.py                       · runtime contract checks (23 tests) [v1.8]
│   ├── train-script-spec.md                · train.py / track.py contract
│   ├── file-contracts.md                   · schemas for all cross-skill files
│   ├── results-tsv-guide.md                · user-facing TSV reading guide [v1.8]
│   ├── test_modules_md.py                  · parser unit tests (23) [v1.7+v1.9]
│   ├── test_templates.py                   · template compliance checker
│   ├── test_weight_transfer.py             · weight_transfer tests (61) [v1.7.1+v1.7.7+v1.8+v1.9]
│   ├── test_hook_utils.py                  · hook_utils tests (13) [v1.7.7]
│   ├── test_invariants.py                  · invariants tests (23) [v1.8]
│   └── templates/
│       ├── train.py.detection              · scaffold for object_detection
│       ├── train.py.tracking               · detector-training half of tracking
│       ├── track.py.tracking               · tracker + TrackEval half
│       └── arch_spec.schema.json           · JSON schema for ARCH_INJECTION_SPEC [v1.7+v1.9]
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
CHANGELOG_v1.7.2.md                         · IMGSZ state-handoff fix
CHANGELOG_v1.7.3.md                         · preferred_locations secondary sort
CHANGELOG_v1.7.4.md                         · tie-breaking rule (don't ask, pick one)
CHANGELOG_v1.7.5.md                         · onboarding hardening (11 fixes)
CHANGELOG_v1.7.6.md                         · 4 latent bugs + pyproject.toml retreat
CHANGELOG_v1.7.7.md                         · hook + yaml_inject production-readiness
CHANGELOG_v1.8.md                           · production-usability + agent contracts
CHANGELOG_v1.9.md                           · full_yaml mode + resource-impact auto-halve
CHANGELOG_v1.9.1.md                         · module_priority yaml dead-config fix
CHANGELOG_v1.9.2.md                         · cross-project run.log pollution fix
CHANGELOG_v1.9.3.md                         · initial BATCH_SIZE configurable from yaml
CHANGELOG_v1.10.md                          · paper-finder HF Papers + cross-source dedup
CHANGELOG_v1.11.md                          · concurrent paper-finder during baseline
CHANGELOG_v1.11.1.md                        · hook + invariant + subagent fixes (BREAKING reapply)
CHANGELOG_v1.12.md                          · BATCH_SIZE locked + scope-aware resource_impact (BREAKING)
CHANGELOG_v1.12.1.md                        · wire up optional_pretrain_trigger + crash-pause yaml comment fix
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

See `CHANGELOG_v1.12.1.md`, `CHANGELOG_v1.12.md`, `CHANGELOG_v1.11.1.md`, `CHANGELOG_v1.11.md`, `CHANGELOG_v1.10.md`, `CHANGELOG_v1.9.3.md`, `CHANGELOG_v1.9.2.md`, `CHANGELOG_v1.9.1.md`,
`CHANGELOG_v1.9.md`, `CHANGELOG_v1.8.md`, `CHANGELOG_v1.7.7.md`,
`CHANGELOG_v1.7.6.md`, `CHANGELOG_v1.7.5.md`, `CHANGELOG_v1.7.4.md`,
`CHANGELOG_v1.7.3.md`, `CHANGELOG_v1.7.2.md`, `CHANGELOG_v1.7.1.md`,
`CHANGELOG_v1.7.md`, and `CHANGELOG_v1.6.md` for recent release notes.

- **v1.12.1** (current): patch — wires up `optional_pretrain_trigger`
  (was dead config since v1.7). yaml fields now actually evaluated in
  autoresearch Step 8: triggers `request_repretrain` when stall
  threshold met AND corpus exists; warns + self-disables when no
  corpus. Default `enabled: false` preserves current behaviour.
  `autoresearch_crash_pause_after` yaml comment corrected to v1.12
  log-only semantics. Pure SKILL+yaml patch.
- **v1.12**: strict fair-comparison mode. **BREAKING**:
  `BATCH_SIZE` LOCKED across iterations. v1.9 `resource_impact` auto-
  halve replaced with predictive skip-and-block (modules predicted
  to OOM are marked `blocked`, never run). v1.7.6 crash-pause halve
  replaced with log-only. New invariant
  `check_no_ultralytics_auto_batch_reduce` catches ultralytics' own
  silent batch reduction. New `Module.effective_resource_impact`:
  scope=all escalates tag one tier (fixes FlexSimAM silent CPU
  fallback). yaml renamed `initial_batch_size` → `batch_size` (alias
  kept). +18 tests (143 → 161). State migration backward-compatible.
- **v1.11.1**: patch — 4 fixes from real-run feedback.
  **BREAKING**: `reapply_on_rebuild` callback signature changed
  `cb(m) → cb(layers)`; helper `_get_layers` resolves DetectionModel
  vs YOLO wrapper shapes; WARNING → RuntimeError. Plus: PicklableHook
  auto device sync (CPU sub-modules → output.device on first call),
  new invariant blocks `inject_modules() → ModuleList` anti-pattern
  at commit, yaml-configurable subagent_type + subagent_model. +14
  tests (129 → 143).
- **v1.11**: concurrent paper-finder. Spawns paper-finder
  Phase 5-6 as a subagent during Loop 0 vanilla baseline, saving ~2h
  per pipeline startup. Defaults to enabled. 30 min timeout →
  graceful fallback to sequential. Stage 1 splits into Phase 1-4
  (sequential, produces base_model.md) + Phase 5-6 (concurrent with
  Loop 0). New paper-finder modes `initial_phase1_only` and
  `phase56_only`. Drop-in. SKILL+yaml only.
- **v1.10**: paper-finder source diversification + dedup.
  Adds HuggingFace Papers as Source 6 (ML-specialised ranking, fills
  PwC coverage gaps) + Phase 2.5 cross-source dedup (collapses
  duplicate papers across arXiv / PwC / S2 / HF / Local, merges
  per-source metadata fields up). All SKILL-only changes — no new
  Python modules, no new dependencies, no new tests. Drop-in.
- **v1.9.3**: patch — initial BATCH_SIZE configurable from
  yaml. New `autoresearch.loop.initial_batch_size` field; Stage 0
  patches train.py with the value (NOT locked, so autoresearch may
  halve dynamically). Falls back to template default 16 if unset.
  Drop-in.
- **v1.9.2**: patch — cross-project run.log pollution fix.
  Real-world bug: agent in wrong cwd was reading neighbour project's
  run.log and parsing metrics from a different experiment, silently
  corrupting results.tsv. 4 layers of defence: project_root absolute
  path canonicalised at Stage 0, `__RUN_START__`/`__RUN_END__` sentinels
  in run.log, Step 5 prelude with cwd-lock + stale-file cleanup, Step 6
  pre-parse freshness check via new `check_run_log_fresh` invariant.
  +6 tests (123 → 129). Drop-in.
- **v1.9.1**: patch — wires up the `autoresearch.module_priority`
  yaml field. Dead config since v1.6; now actually controls priority
  order. Tokens omitted from the list are SKIPPED ENTIRELY, enabling
  architecture-only (`[modules_md_pending]`) or hyperparameter-only
  (`[zero_param_changes]`) runs. Drop-in.
- **v1.9**: full_yaml mode + resource-impact auto-halve.
  Implements `weight_transfer.apply_yaml_spec` end-to-end (was a
  placeholder in v1.7-v1.8), unblocking structural rewrites: BiFPN
  replacing PAFPN, DETR-style decoders, ConvNeXt backbones, P2 head
  architectures. New full_yaml dispatch in autoresearch + paper-finder
  Phase 5 heuristic + arch_spec schema + spec doc chapter. Separately
  adds `resource_impact` field on modules.md (vram_4x / vram_2x /
  cpu_fallback_risk) and Step 3 auto-halve to prevent silent
  CPU-fallback failures. 17 new tests; total 123. Drop-in.
- **v1.8**: production-usability + agent contract enforcement.
  Real-run regression fixes (B1-B5: compute_layer_map off-by-one, yaml
  filename scale suffix, base-aware after_class, project= absolute path,
  Loop 0 vanilla baseline). Pipeline polish (C1-C6: 3 autonomous stop
  triggers, pretrain-trigger semantic clarification, description format
  contract, expanded discoveries categories, per-loop log archive).
  New `shared/invariants.py` module with 23 tests for runtime contract
  enforcement (locked variables, OPTIMIZER not auto, section markers).
  User-facing `results-tsv-guide.md`. Total python tests **82 → 106**.
  No new architectural capability — full_yaml mode reserved for v1.9.
  Drop-in.
- **v1.7.7**: production-readiness for hook + yaml_inject. Six
  fixes — head ref shifting in `update_head_refs` (yaml_inject was
  silently broken from v1.7 to v1.7.6); `PicklableHook` + `reapply_on_rebuild`
  in new `shared/hook_utils.py` (hook mode was silently broken from
  v1.6 to v1.7.6); explicit OPTIMIZER never 'auto' (LR experiments
  silently no-op'd); pseudo-code tiebreak rule (ambiguous decisions
  resolved). 25 new python tests, total 82. Drop-in. v1.7.x line ends.
- **v1.7.6**: 4 latent crash-handling bugs fixed (crash-pause
  sequence, BATCH_SIZE=1 floor, Step 5.5 short-test state key,
  python_runner stale default). v1.7.5's auto-write `pyproject.toml`
  feature removed — replaced with verify-only check that fails fast
  with a clear remediation message. Final patch in v1.7.x line; v1.8
  starts next.
- **v1.7.5**: onboarding hardening. 11 fixes covering local
  skill precedence, `skills_dir` relative path resolution, git identity
  auto-config, smarter python_runner detection, optional pyproject.toml
  generation, user-specified base model, weights/ directory safety,
  `pretrain.time_budget_sec: 0` skip path, environment shield for
  ULTRALYTICS_RUNS_DIR and WANDB, Section marker regex tolerance.
  All 57 tests + 88 SKILL snippets still green.
- **v1.7.4**: autoresearch tie-breaking rule added to Step 2.
  When Priority A–E leaves multiple candidates at the same sort rank,
  apply write order → lower complexity → preferred_locations →
  alphabetical, and NEVER stop to ask the user. Pure SKILL.md prose,
  all 57 tests still green.
- **v1.7.3**: `preferred_locations` in `research_config.yaml`
  now actually sorts autoresearch's pending-module picks. Before
  v1.7.3, the yaml field was read only by paper-finder for Stage 1
  search filtering; autoresearch ignored it, so the yaml promise
  wasn't kept. Secondary sort key added after complexity. +6 tests,
  total 57 green.
- **v1.7.2**: cross-skill `IMGSZ` handoff fix. Orchestrator
  now persists the resolved `imgsz` into `pipeline_state.json`;
  dataset-hunter's `pretrain.py` and autoresearch's Step 5.5 read from
  there instead of the template default (1920) or a missing
  `state["imgsz"]`. No new features, no schema change, all 51 tests
  still green.
- **v1.7.1**: adaptive repair loop — Step 5.5 in autoresearch.
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
