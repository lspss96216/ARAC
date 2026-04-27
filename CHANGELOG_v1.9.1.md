# Changelog — v1.9.1

Release type: **Patch — single dead-config fix.** Wires up the
`autoresearch.module_priority` yaml field so it actually controls
priority order. Drop-in on top of v1.9.

## Background

The yaml field `autoresearch.module_priority` has shipped in
`research_config.yaml` example since v1.6 with the comment "Order
autoresearch picks experiment types. Lower index = tried first." It
was never read by any skill — autoresearch hardcoded the priority
ladder as A → B → C → D → E throughout v1.6 → v1.9. Users editing
the yaml expected reordering to take effect; in reality nothing
changed.

This is the same class of dead-config bug as v1.7.3's fix for
`paper_finder.modules.preferred_locations` (yaml-exposed, never read).

## The fix

### autoresearch SKILL Step 2 — read yaml priority order

`autoresearch/SKILL.md § Step 2 — Ideate` now opens with a v1.9.1
subsection that reads `cfg["autoresearch"]["module_priority"]` and
walks the priority ladder in that order.

The 5 valid tokens (matching the existing yaml example) map to the
existing priority blocks:

| Token | Block |
|---|---|
| `modules_md_pending` | Priority A |
| `zero_param_changes` | Priority B |
| `replacement_changes` | Priority C |
| `additive_changes` | Priority D |
| `combinations` | Priority E |

Each priority block heading now includes its token (e.g.
"Priority A — modules.md pending entries (token: `modules_md_pending`)")
so the SKILL ladder loop can dispatch by name.

### Semantics — omitted tokens are SKIPPED, not deferred

The most important semantic decision: tokens omitted from the user's
list are **not** silently fallen-through to default. They are
**skipped entirely**. Use cases this enables:

- **Architecture-only run** — `module_priority: [modules_md_pending]`
  → loop tries paper-backed modules; when modules.md exhausted, stalls
  rather than fall through to hyperparameter sweep
- **Hyperparameter-only run** — `module_priority: [zero_param_changes]`
  → loop never touches modules.md, never tries combinations, never
  attempts replacements or additions
- **Reorder for context-specific runs** — `[zero_param_changes,
  modules_md_pending]` → reverse default; try hyperparameters first,
  then fall through to paper-backed modules

This is the explicit user-control contract: the yaml says exactly
which experiments are allowed and in what order. The loop respects
that even when "respecting it" means stalling earlier than the
hardcoded ladder would have.

### Validation — warn-not-reject on bad input

| Input | Behaviour |
|---|---|
| Field absent or `null` | Use default 5-token order (v1.6+ behaviour preserved) |
| Empty list `[]` | Same — fall back to default |
| Not a list (e.g. dict) | Log to discoveries.md as `agent_violation`, fall back to default |
| Unknown token (e.g. `[modules_md_pending, telepathy]`) | Log warning, skip the unknown token, use the rest |
| All tokens unknown | Log warning, fall back to default |
| Duplicate tokens | Use first occurrence, log nothing (silent dedup) |

Same warn-not-reject philosophy as `integration_mode` parsing in
`modules_md.py` — the loop never crashes on bad config, but does
record the issue in `discoveries.md` so the user sees it post-run.

### `combinations` first is allowed

If `combinations` is the first token and no prior keeps exist (loop
just started), the Priority E branch is a no-op and the loop falls
through to the next configured token (or stalls if it's the only
token). The SKILL does NOT validate dependency order — we trust the
user's explicit choice.

This matters because the alternative ("validate that combinations
isn't first") would silently rewrite the user's configuration, which
is the same class of footgun the v1.9.1 fix is trying to eliminate.

## Files changed

| File | Change |
|---|---|
| `autoresearch/SKILL.md` | Step 2 prefixed with v1.9.1 yaml-reading subsection (~80 lines including walk-the-ladder dispatch). 5 priority blocks tagged with their token names. "fall through to Priority B" wording replaced with "fall through to the next enabled priority" globally (4 sites). |
| `examples/research_config.visdrone-detection.yaml` | `module_priority` comment expanded — semantic clarification + 4 example configurations |
| `examples/research_config.visdrone-mot.yaml` | Same comment expansion |
| `CHANGELOG_v1.9.1.md` | New (this file) |
| `README.md` | Banner + Versions list |

### Unchanged

Every other file. This is a SKILL/yaml-only fix; no Python code
changed.

## Test coverage

| Suite | v1.9 | v1.9.1 |
|---|---|---|
| `shared/test_modules_md.py` | 23 | 23 |
| `shared/test_templates.py` | 3 | 3 |
| `shared/test_weight_transfer.py` | 61 | 61 |
| `shared/test_hook_utils.py` | 13 | 13 |
| `shared/test_invariants.py` | 23 | 23 |
| **Total python tests** | **123** | **123** |
| SKILL.md python snippets | 96 | **98** (+2) |
| YAML examples | 2 | 2 |

The 2 new SKILL snippets are the v1.9.1 yaml-reading + walk-the-ladder
patterns. No new unit tests because the fix is at SKILL level (the
canonical regex-driven `_read_yaml_var`-style helper this would use
already exists for the `IMGSZ` v1.7.2 fix).

## Upgrade path

Drop-in. Existing yaml files keep working:

- Yaml without `module_priority` → uses default 5-token order (same as
  v1.9 hardcoded)
- Yaml with all 5 tokens listed in default order → identical behaviour
  to v1.9
- **Yaml with non-default order or omitted tokens → behaviour now
  matches what the yaml says.** This is a behaviour change for users
  who previously edited `module_priority` expecting it to work; the
  fix unsticks the dead config.

If you previously edited `module_priority` and saw "no effect" — your
new v1.9.1 run will respect the edit. Verify your yaml reflects what
you actually want.

## Operational note — interaction with stall expansion

`module_priority: [modules_md_pending]` (architecture-only) interacts
with the v1.6 stall expansion path:

1. Loop tries paper-backed modules
2. modules.md gets fully exhausted (all `injected` or `discarded`)
3. `pending` becomes 0; Priority A returns None
4. No other priority is enabled → loop stalls
5. Stall expansion fires (orchestrator's paper-finder retry)
6. paper-finder finds new modules → modules.md grows → loop resumes

This is the intended behaviour for the architecture-only configuration.
The user gets pure architecture exploration; when ideas run out the
pipeline auto-expands the search rather than fallback-pivoting to
hyperparameters.

If you want pure architecture exploration **without** stall expansion
(for time-bounded runs), set `orchestrator.stopping.max_paper_finder_expansions: 0`
in the same yaml.

## v1.10 outlook

Unchanged from v1.9 outlook. v1.9.1 is a single-issue patch and does
not affect v1.10 scope decisions.
