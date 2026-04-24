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
