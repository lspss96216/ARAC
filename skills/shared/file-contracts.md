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
  // Identity
  "project_name":               "string",          // from meta.project_name
  "run_tag":                    "string | null",
  "task":                       "string",          // task.description
  "task_type":                  "string",          // object_detection | object_tracking | ...
  "dataset":                    "string",
  "stage":                      "string",          // init | paper_finder | dataset_hunter | autoresearch | done

  // Skill progress flags
  "paper_finder_done":          "bool",
  "base_model_md_ready":        "bool",
  "modules_md_ready":           "bool",
  "pretrain_done":              "bool",
  "pretrain_skipped":           "bool",
  "pretrain_offer_declined":    "bool",
  "pretrain_weights":           "string | null",
  "base_weights_local":         "string | null",
  "autoresearch_running":       "bool",

  // Metric tracking (metric-agnostic by design)
  "primary_metric_name":        "string",          // from evaluation.metrics.primary
  "best_primary_value":         "number | null",   // null until first keep
  "stall_count":                "int",
  "loop_count":                 "int",
  "seed":                       "int",
  "paper_finder_expansions":    "int",

  // Paths
  "local_papers_dir":           "string | null",
  "local_datasets_dir":         "string | null",
  "skills_dir":                 "string",
  "train_script":               "string",
  "custom_modules_file":        "string",
  "weights_dir":                "string",

  // Time budgets (seconds)
  "pretrain_time_budget":       "int",
  "self_eval_time_budget":      "int",
  "loop_time_budget":           "int",
  "pretrain_improvement_threshold": "number",

  // Dataset hunter knobs
  "dataset_hunter_enabled":     "bool",
  "disk_budget_gb":             "int",
  "roboflow_api_key":           "string | null",

  // Autoresearch knobs
  "stall_threshold":            "int",
  "improvement_threshold":      "number",
  "loop_iterations":            "int | null",

  // Orchestrator knobs
  "crash_pause_after":          "int",

  // Advanced paths
  "results_tsv":                "string",
  "modules_md_path":            "string",
  "base_model_md_path":         "string",
  "pretrain_ckpt_dir":          "string",

  // Timestamps
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
- `status` column → one of `keep`, `discard`, `crash`
- `description` → free text, no tabs or newlines (sanitize before writing)

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
