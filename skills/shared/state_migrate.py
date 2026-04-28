"""
Idempotent pipeline_state.json migration.

Old state files may be missing keys added in later versions, or use the
legacy `dataset` key instead of `dataset_root`. Run `migrate()` at
orchestrator startup to normalize.

All skills that read pipeline_state.json should call this first — it's
cheap (single file read/write) and makes downstream code safe to use
`state[key]` without KeyError defensive coding.
"""
from __future__ import annotations
import json
import math
import pathlib
from typing import Any

# Defaults for every key added in v1.6. Keys that existed before v1.6 with
# known defaults are also listed here so a missing-key resume still works.
CURRENT_DEFAULTS: dict[str, Any] = {
    # --- v1.6 additions ---
    "consecutive_crashes":         0,
    "best_tiebreak_value":         None,
    "pretrain_attempt_failed":     False,
    "rebase_marker_loops":         [],
    "stall_force_test_reset":      5,
    "stop_requested":              False,
    "stop_reason":                 None,
    "python_runner":               "python3",
    "stop_flag_file":              "stop_pipeline.flag",
    "max_paper_finder_expansions": 3,
    "baseline_snapshot":           None,

    # --- existed before v1.6 but not always initialised ---
    "param_only_streak":           0,
    "architecture_keeps":          0,
    "request_repretrain":          False,
    "repretrain_reason":           None,

    # --- v1.8 additions ---
    "vanilla_baseline_done":       False,
    "no_improvement_loops":        0,
    "pretrain_dead_config_warned": False,

    # --- v1.9 additions ---
    "batch_size_pre_autohalve":    None,

    # --- v1.9.2 additions ---
    # project_root: absolute path of project root. Migration falls back to
    # cwd; if that's wrong the next Step 5 cwd-lock check will catch it.
    # Best practice is to re-run orchestrator Stage 0 to pick up the
    # canonical absolute path from research_config.yaml.
    "project_root":                None,    # filled in by Stage 0 init or migration_post_hook below
    "step5_started_at":            None,    # ISO timestamp Step 5 sets before launching train.py
    "step5_started_at_unix":       None,    # same, unix epoch (for mtime comparisons)

    # --- v1.9.3 additions ---
    # initial_batch_size: starting BATCH_SIZE from yaml. None = use template
    # default (16). Stage 0 patches train.py once at scaffold time.
    # v1.12 — DEPRECATED in favour of `batch_size`. Kept here as a
    # backward-compat field so resumes from pre-v1.12 state files don't
    # raise KeyError. New code should read state["batch_size"].
    "initial_batch_size":          None,

    # --- v1.12 additions ---
    # batch_size: LOCKED across all iterations (invariants.LOCKED_VARS
    # enforces). Replaces v1.9.3's initial_batch_size with locked semantics.
    # State init reads yaml's batch_size first, then initial_batch_size as
    # backward-compat alias.
    "batch_size":                  None,

    # --- v1.11 additions ---
    # concurrent_paper_finder state. Resume-safe: spawned=True survives across
    # session breaks so we don't double-spawn the subagent on resume.
    "concurrent_paper_finder_spawned":       False,
    "concurrent_paper_finder_done":          False,
    "concurrent_paper_finder_fallback_used": False,

    # --- v1.11.1 additions ---
    # Record which subagent_type + model the spawn used. Informative only —
    # not used for control flow. Lets discoveries.md narrative show
    # "this run's paper-finder was on Sonnet" so cost / quality review is easy.
    "concurrent_paper_finder_subagent_type":  None,
    "concurrent_paper_finder_subagent_model": None,

    # --- v1.12.1 additions ---
    # optional_pretrain_trigger state. Was dead config v1.7-v1.12; v1.12.1
    # wires it up. fired_count is informative; no_corpus_warned is a one-time
    # latch so logs aren't spammed when dataset-hunter never ran.
    "pretrain_trigger_fired_count":      0,
    "pretrain_trigger_no_corpus_warned": False,

    # --- v1.13 additions ---
    # Per-module tuning state. autoresearch tracks the current module being
    # tuned (or None when between modules), the attempt number within that
    # module's sequence, and the previous attempt's final_map for the
    # attempt-to-attempt improvement check.
    #
    # current_tuning_module: name of module being tuned (None when not tuning)
    # current_tuning_attempt: 1-indexed attempt within current module (0 when not tuning)
    # tuning_attempt_extended: whether we've granted attempt-extension for current module
    # last_attempt_final_map: previous attempt's final mAP, for ≥2% / ≥3% improvement checks
    "current_tuning_module":   None,
    "current_tuning_attempt":  0,
    "tuning_attempt_extended": False,
    "last_attempt_final_map":  None,
}

# Legacy -> current key renames. Applied once; after migration the old key
# is gone.
LEGACY_RENAMES: dict[str, str] = {
    "dataset": "dataset_root",
}


def migrate(path: str | pathlib.Path = "pipeline_state.json") -> dict:
    """Read, normalize, write back, and return the state dict.

    Returns {} if the file does not exist (orchestrator Stage 0 should then
    initialise from research_config.yaml). Always returns a schema-complete
    dict when the file exists.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return {}

    state = json.loads(p.read_text())
    changed = False

    # 1. Legacy renames
    for old, new in LEGACY_RENAMES.items():
        if old in state and new not in state:
            state[new] = state.pop(old)
            changed = True

    # 2. Fill in missing defaults
    for key, default in CURRENT_DEFAULTS.items():
        if key not in state:
            state[key] = default
            changed = True

    # v1.9.2 — post-default hook: if project_root is still None after
    # migration (resume from pre-v1.9.2), fall back to cwd resolved.
    # This is best-effort; Step 5's cwd-lock will catch the case where
    # cwd is genuinely wrong.
    if state.get("project_root") is None:
        state["project_root"] = str(pathlib.Path(".").resolve())
        changed = True

    # 3. Sanitize non-JSON-spec sentinels (e.g. a buggy older version wrote
    #    float('inf')). Recursively walk the structure.
    def scrub(v):
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            return None
        if isinstance(v, dict):
            return {k: scrub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [scrub(x) for x in v]
        return v

    cleaned = scrub(state)
    if cleaned != state:
        state = cleaned
        changed = True

    if changed:
        p.write_text(json.dumps(state, indent=2))
    return state


if __name__ == "__main__":
    import sys
    s = migrate(sys.argv[1] if len(sys.argv) > 1 else "pipeline_state.json")
    print(json.dumps(s, indent=2))
