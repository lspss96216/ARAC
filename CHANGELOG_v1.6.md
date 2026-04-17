# Changelog — v1.6 Bug Fixes

## Fixes applied

### A — Crash-level (blocking)

- **A1** `is_architecture_change` SyntaxError — continuation line had stray `or`.
  Parenthesised the expression in autoresearch Step 2 Forced architecture exploration.
- **A2** `best_TIEBREAK` was referenced in Step 7 decision rule but never assigned.
  Added Setup Step 8 `load_best_row()` helper; Step 1 now populates both
  `best_PRIMARY` and `best_TIEBREAK` from a single consistent read.
  New state field `best_tiebreak_value` tracked in Step 9.
- **A3 / C8** `consecutive_crashes` counter was documented in yaml but never
  implemented. Added full block in autoresearch Step 6: counts crashes, halves
  `BATCH_SIZE`, reverts last commit, logs, resets counter, never pauses for user.
- **A4** `wget` calls used `check=True` with no try/except — a dead URL would
  crash the pipeline. Replaced every download site with new `safe_wget()` helper
  in orchestrator Stage 2 Step 3, Stage 3 Step 2, and dataset-hunter Sources 2/3/4/5
  and Phase 5.
- **A5** Unknown `Location` field silently defaulted to detector. Now explicitly
  discards the module and logs to `discoveries.md` with the list of valid
  locations — paper-finder Phase 5 documents the vocabulary.

### B — Silent-failure

- **B1** `parsing.source: json` / `csv` branches were `...` stubs in both
  autoresearch Step 6 and dataset-hunter Phase 6 Step 2. Created shared
  `parse_metrics.py` helper; both callsites now have full implementations.
- **B2** Guard-metric baseline drift — **PRESERVED AS-IS per your instruction.**
  State schema reserves a `baseline_snapshot` field for the future fix.
- **B3** `pretrain_offer_declined` was set True even when pretrain crashed.
  Distinguished with new `pretrain_attempt_failed` field. Only genuine
  "we tried, metric didn't clear threshold" sets `offer_declined`; crashes
  set `attempt_failed` so a future trigger can retry.
- **B4** `results.tsv` was not segmented across re-pretrain cycles. Orchestrator
  Stage 3 Step 6.5 now writes a `status=rebase` marker row; Stage 4 summary
  segments the file at each marker and prints top 5 per generation instead
  of pooling across different baselines.
- **B5** `val_split` hash method could produce 0 val images on small datasets
  and the actual file-move code was missing. Dataset-hunter Phase 3 now has
  `make_val_split()` helper with fallback-to-10%-last and real `shutil.move`.
- **B6** `DATA_YAML nc` read failure silently preserved template default
  `NUM_CLASSES=10`. Orchestrator Stage 3 Step 2.5 now warns with the exact
  fix instruction when `nc` can't be read.

### C — Design/consistency

- **C1** `force_test_reset` was hardcoded 5 instead of read from yaml.
  Now stored in `state.stall_force_test_reset` from
  `autoresearch.stall.force_test_reset`.
- **C2** autoresearch + orchestrator both managed `stall_count` — race
  condition. Orchestrator now owns the state machine; autoresearch only
  increments. Reset / expand decisions all happen in orchestrator Stage 3 Step 6.
- **C3** `state["dataset"]` vs `dataset_root` name inconsistency. Renamed to
  `dataset_root` everywhere. `state_migrate.py` handles the rename for
  resumed pre-v1.6 state files.
- **C4** `_refresh_header` in `modules_md.py` used awkward list-mutation.
  Now a pure function that returns updated text.
- **C5** `str.replace(img_path.suffix, ".txt")` could match the suffix
  mid-string (e.g. `img.jpg.jpg` → `img..txt`). Replaced with
  `with_suffix(".txt")` in both dataset-hunter Phase 3 val-split and Phase 4 merge.
- **C6** `inject_modules` contract was ambiguous (in-place vs return). Pinned
  in `train-script-spec.md`: callers MUST rebind (`model = inject_modules(model)`).
  Templates got the rebind reminder comment and a docstring update. Spec
  added `assert_idempotent()` helper and anti-patterns section.
- **C7** First-iteration keep was before crash check — a crashed first run
  became the baseline. Promoted crash check to decision rule #0 in
  autoresearch Step 7.
- **C8** See A3 above.
- **C9** Stage 4 only triggered by Ctrl+C — no daemon-compatible trigger.
  Added `orchestrator.stopping` yaml section with `stop_flag_file` sentinel,
  `max_paper_finder_expansions` cap, plus the existing `loop_iterations`
  bound. New Step 6.75 checks all three triggers every loop.
- **C10** `dataset_hunter_enabled: false` path was undefined. Stage 2 now
  has a fast-path at the top that advances directly to Stage 3.

### D — Details

- **D1/D2** `validate_weights_url` used `firecrawl` CLI primary — the CLI
  doesn't exist as an official binary. Rewrote paper-finder Phase 4 to use
  curl HEAD as primary + range-GET as fallback. Firecrawl Python SDK noted
  as alternative. Same D1/D2 caveat propagated to dataset-hunter Source 5.
- **D3** `test_templates.py` hardcoded `/home/claude` path. Now resolves
  templates dir relative to this file, with `PIPELINE_TEMPLATES_DIR` env
  var override.
- **D4** `append_module` re-parsed whole file on every call (O(n²)). Now
  reads `Total modules` counter directly from header (O(1) per append).
- **D5** `uv run` fallback was inconsistent. Orchestrator Stage 0 Step 0
  detects runner once, stores in `state.python_runner`, all sub-skills read
  from there.
- **D6** `results.tsv` schema drift on yaml edit — same mechanism as C2
  handles this; header is generated dynamically from yaml at Setup Step 5.
- **D7** `num_params_M = 0.0` edge case. Floored at 0.1M in both
  train.py.detection and train.py.tracking.
- **D8** `modules_md.py` parser split on `## ` inside registry header. Now
  skips past the first `---` line before looking for module headers.
  Known residual limitation documented: section bodies must not start
  lines with `## ` at column 0.
- **D9** `discoveries.md` append was non-atomic. Autoresearch Setup Step 9
  has new `log_discovery()` helper using write-and-rename.
- **D10** `track.py` `avg_fps` calculation bug — cumulative frame count
  divided by per-sequence elapsed time inflated FPS. Now per-sequence
  (seq_frames / elapsed). Also changed emit format to `FPS: <value>` (prefix)
  so the documented regex `FPS:\s+([\d.]+)` matches — previous format put
  the number first and never matched.

## New files

- `skills/shared/state_migrate.py` — pipeline_state.json schema migration.
  Handles `dataset → dataset_root` rename + missing key backfill + inf/NaN scrub.
- `skills/shared/parse_metrics.py` — shared stdout/json/csv metric extraction
  used by autoresearch + dataset-hunter.

## New pipeline_state.json fields

All have safe defaults; resume of pre-v1.6 state auto-fills via `state_migrate.py`:

| Field | Default | Fix |
|---|---|---|
| `consecutive_crashes` | 0 | A3/C8 |
| `best_tiebreak_value` | null | A2 |
| `pretrain_attempt_failed` | false | B3 |
| `rebase_marker_loops` | [] | B4 |
| `stall_force_test_reset` | 5 | C1 |
| `dataset_root` (rename of `dataset`) | — | C3 |
| `python_runner` | "uv run" | D5 |
| `stop_flag_file` | "stop_pipeline.flag" | C9 |
| `max_paper_finder_expansions` | 3 | C9 |
| `stop_requested` | false | C9 |
| `stop_reason` | null | C9 |
| `baseline_snapshot` | null | reserved for future B2 |

## Validation

- `modules_md.py`: 8/8 unit tests pass
- `test_templates.py`: 3/3 templates comply
- `state_migrate.py`: rename + backfill + scrub verified
- `parse_metrics.py`: stdout + json + csv all tested
- `autoresearch/SKILL.md`: 19/19 Python snippets parse cleanly
- `research-orchestrator/SKILL.md`: 20/20 Python snippets parse cleanly
- Both yaml examples parse + contain `orchestrator.stopping` section
- Both yaml examples reference `stall.force_test_reset`

## Install

```bash
# Drop into your existing install location
cp -r pipeline_v1.6/skills/* ~/.claude/skills/
cp pipeline_v1.6/examples/* ~/.claude/skills/examples/

# Verify
cd ~/.claude/skills/shared
python3 test_modules_md.py     # expect: 8/8 passed
python3 test_templates.py      # expect: 3/3 passed
```

## Resume behaviour

Existing pipeline_state.json files from v1.5 and earlier work transparently —
orchestrator Stage 0 Step 2 runs `state_migrate.migrate()` which adds all new
keys with safe defaults and renames `dataset` → `dataset_root` in one pass.
