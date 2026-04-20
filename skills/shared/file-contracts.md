# File Contracts

Files that cross skill boundaries have schemas. Without this document, each skill
parses them with ad-hoc regex, schemas drift silently, and bugs surface only at
runtime (e.g. `str.count("| Status | pending |")` returning 0 because paper
finder switched from pipe tables to YAML frontmatter).

This file pins those schemas. When a skill wants to change a format, update this
document first — the other skills read it to know what to expect.

---

## `pipeline_state.json`

Authoritative state shared by all four skills. Lives in the project root.
Never committed to git.

### Allowed types

JSON spec only: `string`, `number` (int or finite float), `bool`, `null`, `array`,
`object`. No `Infinity`, `NaN`, `datetime`, `pathlib.Path`, or custom classes.

- Datetimes → ISO 8601 strings (`datetime.now().isoformat()`)
- Paths → forward-slash strings (or platform-native — readers must tolerate both)
- "Not yet measured" sentinels → `null`, never `float("-inf")` or `float("inf")`

### Schema

```jsonc
{
  // ---------------------------------------------------------------------
  // Identity
  // ---------------------------------------------------------------------
  "project_name":               "string",          // from meta.project_name
  "run_tag":                    "string | null",
  "task":                       "string",          // task.description
  "task_type":                  "string",          // object_detection | object_tracking | ...
  "dataset_root":               "string",          // C3 — renamed from "dataset"
  "stage":                      "string",          // init | paper_finder | dataset_hunter | autoresearch | done

  // ---------------------------------------------------------------------
  // Skill progress flags
  // ---------------------------------------------------------------------
  "paper_finder_done":          "bool",
  "base_model_md_ready":        "bool",
  "modules_md_ready":           "bool",
  "pretrain_done":              "bool",
  "pretrain_skipped":           "bool",
  "pretrain_offer_declined":    "bool",
  "pretrain_attempt_failed":    "bool",            // B3 — distinguishes "tried and crashed" from "tried and not worth it"
  "pretrain_weights":           "string | null",
  "base_weights_local":         "string | null",
  "autoresearch_running":       "bool",

  // ---------------------------------------------------------------------
  // Metric tracking (metric-agnostic by design)
  // ---------------------------------------------------------------------
  "primary_metric_name":        "string",          // from evaluation.metrics.primary
  "best_primary_value":         "number | null",   // null until first keep
  "best_tiebreak_value":        "number | null",   // A2 — null until first keep with a tiebreak metric
  "stall_count":                "int",
  "loop_count":                 "int",
  "seed":                       "int",
  "paper_finder_expansions":    "int",
  "param_only_streak":          "int",             // consecutive param-only experiments without improvement
  "architecture_keeps":         "int",             // count of kept architecture changes (for re-pretrain trigger)
  "consecutive_crashes":        "int",             // A3/C8 — reset on non-crash; triggers BATCH_SIZE halving at crash_pause_after
  "rebase_marker_loops":        "array[int]",      // B4 — loop numbers where a re-pretrain cycle occurred; Stage 4 segments results.tsv by these
  "baseline_snapshot":          "object | null",   // reserved for future B2 guard-metric baseline fix; currently unused

  // ---------------------------------------------------------------------
  // Paths
  // ---------------------------------------------------------------------
  "local_papers_dir":           "string | null",
  "local_datasets_dir":         "string | null",
  "skills_dir":                 "string",
  "train_script":               "string",
  "custom_modules_file":        "string",
  "weights_dir":                "string",

  // ---------------------------------------------------------------------
  // Time budgets (seconds)
  // ---------------------------------------------------------------------
  "pretrain_time_budget":       "int",
  "self_eval_time_budget":      "int",
  "loop_time_budget":           "int",
  "pretrain_improvement_threshold": "number",

  // ---------------------------------------------------------------------
  // Dataset hunter knobs
  // ---------------------------------------------------------------------
  "dataset_hunter_enabled":     "bool",
  "disk_budget_gb":             "int",
  "roboflow_api_key":           "string | null",

  // ---------------------------------------------------------------------
  // Autoresearch knobs
  // ---------------------------------------------------------------------
  "stall_threshold":            "int",
  "stall_force_test_reset":     "int",             // C1 — from autoresearch.stall.force_test_reset (default 5)
  "improvement_threshold":      "number",
  "loop_iterations":            "int | null",

  // ---------------------------------------------------------------------
  // Orchestrator knobs
  // ---------------------------------------------------------------------
  "crash_pause_after":          "int",             // consecutive crashes before BATCH_SIZE halve
  "python_runner":              "string",          // D5 — "uv run" or "python3"; chosen at Stage 0 Step 0
  "stop_flag_file":             "string",          // C9 — path to sentinel file that triggers Stage 4
  "max_paper_finder_expansions": "int",            // C9 — after this many expansions with no new modules, stop
  "stop_requested":             "bool",            // C9 — set when any Stage 4 trigger fires
  "stop_reason":                "string | null",   // C9 — human-readable reason printed in summary

  // ---------------------------------------------------------------------
  // Advanced paths
  // ---------------------------------------------------------------------
  "results_tsv":                "string",
  "modules_md_path":            "string",
  "base_model_md_path":         "string",
  "pretrain_ckpt_dir":          "string",

  // ---------------------------------------------------------------------
  // Re-pretrain signaling (autoresearch -> orchestrator)
  // ---------------------------------------------------------------------
  "request_repretrain":         "bool",            // autoresearch raises; orchestrator clears after Step 6.5
  "repretrain_reason":          "string | null",

  // ---------------------------------------------------------------------
  // Timestamps
  // ---------------------------------------------------------------------
  "started_at":                 "iso-8601 string",
  "last_updated":               "iso-8601 string"
}
```

### Write discipline

- Read → modify → write in one short block. Do not hold state open across
  long-running operations; other skills may read concurrently.
- Always update `last_updated` on every write.
- `json.dump(state, f, indent=2)` with default settings. If you need to serialize
  something the encoder cannot handle, convert to a native type first — never
  pass `default=str` (hides bugs like serializing `float("inf")`).

### State migration on resume

Schemas evolve. Orchestrator Stage 0 reads `pipeline_state.json` and runs the
`migrate()` helper at `<skills_dir>/shared/state_migrate.py` which:

1. Adds any missing keys from the current schema with the documented default.
2. Renames legacy keys — specifically `"dataset"` → `"dataset_root"`.
3. Coerces any `float("inf")` / `NaN` sentinels to `null`.
4. Writes the normalized state back in-place.

Callers obtain a schema-complete state with:

```python
from shared.state_migrate import migrate
state = migrate("pipeline_state.json")
```

New fields added in this version (all have safe defaults — resume of an older
state file will auto-fill them):

| Field | Default | Fix ID |
|---|---|---|
| `consecutive_crashes` | 0 | A3 / C8 |
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
| `baseline_snapshot` | null | (reserved for future B2) |
| `param_only_streak` | 0 | (existing, schema-documented now) |
| `architecture_keeps` | 0 | (existing, schema-documented now) |
| `request_repretrain` | false | (existing, schema-documented now) |
| `repretrain_reason` | null | (existing, schema-documented now) |

---

## `pretrain_eval.json`

Written by dataset hunter Phase 6 Step 3. Read by orchestrator Stage 2 to decide
whether to use the pretrained checkpoint or fall back to baseline weights.

```jsonc
{
  "eval_metric":       "string",          // dataset hunter's own metric, NOT evaluation.metrics.primary
  "baseline":          "number | null",   // null if baseline eval failed
  "pretrained":        "number | null",   // null if pretrained eval failed
  "delta":             "number | null",   // direction-corrected: positive means pretrained is better
  "threshold":         "number",          // from dataset_hunter.pretrain.improvement_threshold
  "recommendation":    "string"           // "use_pretrained" | "use_original" | "inconclusive"
}
```

Orchestrator treats `inconclusive` as `use_original` (safer default).

---

## `base_model.md`

Written by paper finder Phase 4. Read by orchestrator Stage 2 (weights URL) and
by dataset hunter Phase 5 (only if orchestrator did not already resolve).

Current format is hand-written markdown with `**Weights URL**: <value>` bold
lines, parsed with regex. Parsers should tolerate minor whitespace / casing
variation. The key fields readers need are `Weights URL` (download target) and
`arxiv.org/abs/<id>` (paper2code fallback).

A future refactor should move to YAML frontmatter for robustness.

---

## `modules.md`

Format is owned by `<skills_dir>/shared/modules_md.py`. All skills read and
write via that module — never parse by hand. See the parser's docstring for
the canonical format.

Read: `mm.parse(path)`, `mm.find_pending(path)`, `mm.count_pending(path)`,
`mm.list_pdf_paths(path)`.
Write: `mm.append_module(path, module_dict)`, `mm.update_status(path, name, new_status)`.

Valid statuses: `pending`, `injected`, `tested`, `discarded` (enforced by parser).
Valid complexities: `low`, `medium`, `high` (enforced by parser).

### Field conventions

| Field | Required? | Values | Notes |
|---|---|---|---|
| `Paper` | yes | free text | Paper title |
| `arXiv` | recommended | URL | `https://arxiv.org/abs/...` |
| `Published` | optional | year string | |
| `Location` | yes | `backbone` / `neck` / `head` / `loss` / `label_assignment` / `tracker` / `post_processing` / `association` / `reid` | Determines hook vs tracker dispatch, and scope for yaml_inject (A5) |
| `Complexity` | yes | `low` / `medium` / `high` | Affects `find_pending` sort order; `high` is tried last |
| `paper2code` | yes | `yes` / `yes (GitHub repo: <url>)` / `no (not on arXiv)` / `no (no public repo)` | How autoresearch gets the module's code |
| `pdf_path` | only for local PDFs | file path | Used by `list_pdf_paths` for dedup in Mode B |
| `Status` | yes | `pending` / `injected` / `tested` / `discarded` | Default `pending` on append |
| `Integration mode` | optional | `hook` / `yaml_inject` / `full_yaml` | **v1.7.** Omit or set `hook` for hook-style modules (v1.6 default). Use `yaml_inject` for modules that need YAML-level insertion + weight transfer (see `train-script-spec.md § Architecture injection`). `full_yaml` is reserved for v1.8+ and raises `NotImplementedError` in v1.7. Unknown values warn to stderr and fall back to `hook` (warn-not-reject). |

### yaml_inject module extras

When `Integration mode: yaml_inject`, the module's `Integration notes`
section should additionally specify the injection parameters — at minimum
`module_class` (the name of the lazy wrapper class in `custom_modules.py`),
`position` (`after_class: <class>` or `at_index: <int>`), and `scope`
(`backbone` / `neck` / `head` / `all`). Autoresearch extracts these and
writes `arch_spec.json` before running `train.py`.

---

## `arch_spec.json` (v1.7)

Written by autoresearch when it picks a module with
`Integration mode: yaml_inject`. Read by `weight_transfer.py` at
`train.py` run time.

Lives at the path `train.py`'s Section ② `ARCH_INJECTION_SPEC_FILE` points
to — default `arch_spec.json` in the project root. It IS git-tracked (unlike
`pipeline_state.json`) so the experiment is fully reproducible from the
commit alone.

Schema in `shared/templates/arch_spec.schema.json`. Two modes:

```jsonc
// v1.7 — insertions mode
{
  "mode": "insertions",
  "insertions": [
    {
      "module_class":  "string, class name in custom_modules.py",
      "position": {
        "kind": "after_class" | "at_index",
        "class_name": "string (if after_class)",
        "index":      "int (if at_index)"
      },
      "scope":         "backbone" | "neck" | "head" | "all",
      "yaml_args":     ["list of literals, first is channel hint"],
      "module_kwargs": { "optional": "dict of kwargs" }
    }
  ],
  "strict": true   // per-entry: every layer_map entry must transfer ≥1 tensor
}

// v1.8+ — full_yaml mode (NotImplementedError in v1.7)
{
  "mode": "full_yaml",
  "custom_yaml_path":   "exp_bifpn.yaml",
  "layer_map_override": null
}
```

Rules:
- `position.kind: at_index` uses **base-YAML indices**, not custom-YAML indices.
  `weight_transfer` handles offsets.
- `scope` is enforced as an assertion in `at_index` mode — out-of-scope raises.
- `strict: true` is the default; set false only if you deliberately expect
  some layers to transfer zero tensors.

---

## `results.tsv`

Written by autoresearch Step 8 (one row per experiment loop). Read by orchestrator
Stage 4 summary. Tab-separated, first row is header.

### Column naming rule

Header is built dynamically in autoresearch Setup Step 5 from the evaluation
config. There is **no fixed column set**. Consumers must read the header row
and index by column name, not by position.

### Value format

- Metrics → float, 4 decimal places for fractions (`0.4523`), integers for counts (`342`)
- Crashed experiments → `0.0000` / `0` in metric columns, `crash` in status
- `status` column → one of `keep`, `discard`, `crash`, or `rebase` (B4 — marks a re-pretrain boundary)
- `description` → free text, no tabs or newlines (sanitize before writing)

### Rebase markers (B4)

When orchestrator re-runs pretrain mid-loop (Stage 3 Step 6.5), it writes a
sentinel row to results.tsv:

```
<loop>\trebase\t<zeros for all metric cols>\t0.0000\trebase\tre-pretrain: <reason>
```

Stage 4 summary segments the file at each `status == "rebase"` row and prints
"Top 5 per pretrain generation" rather than pooling across different baselines.

---

## `run.log`

Written by `uv run train.py` (and `track.py` for tracking). Read by autoresearch
Step 6 verify and dataset hunter Phase 6 Step 2 extract.

`run.log` is stdout+stderr capture, exactly what the scripts printed. Skills
parse it using regex / JSONPath / CSV per `evaluation.parsing` in the yaml.

### The canonical-format invariant

Templates (`train.py.detection`, `track.py.tracking`) guarantee metrics appear as
lines matching `<key>: <value>` at column 0. Consumers rely on this. Raw tool
output (ultralytics val tables, TrackEval column reports) must be reformatted
inside the template before emission — see
`<skills_dir>/shared/train-script-spec.md` § Templates always reformat tool output.

---

## `hunt_log.tsv`

Written by dataset hunter, not currently read by any other skill. Internal log.
Format defined in dataset hunter SKILL.md § Phase 1 Step 4 — not part of the
cross-skill contract and may change.

---

## `discoveries.md`

Written by autoresearch during the loop. Read by orchestrator at Stage 4
(printed in the final summary). Designed to absorb observations that would
otherwise cause the agent to stop and talk to the user.

Format is append-only markdown:

```markdown
# Discoveries

Observations from the experiment loop. User reads these after the run.

---

## Loop 3 — observation

2026-04-15T14:23:00

At imgsz=1280, TIME_BUDGET=1200s yields ~5 epochs. Relative comparison is
still valid but convergence is limited.

## Loop 7 — bug_workaround

2026-04-15T15:10:00

Monkey-patching model.forward() causes injected module weights to not persist
in checkpoint. Switched to subclass-based injection.
```

Categories: `observation` / `limitation` / `strategy_shift` / `bug_workaround`

This file has no schema enforcement — it is human-readable notes, not
machine-parsed state. The only contract is: autoresearch appends to it (atomic
via write-and-rename to prevent partial reads during concurrent access — D9),
orchestrator reads and prints it.
