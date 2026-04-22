# Changelog — v1.7.3

Release type: **Bug fix.** `preferred_locations` in
`research_config.yaml` now actually affects autoresearch iteration
order. No schema changes, no new features. Drop-in on top of v1.7.2 /
v1.7.1 / v1.7 / v1.6.

## The bug

`paper_finder.modules.preferred_locations` (e.g.
`[backbone, neck, head, loss]`) was documented as "Module insertion
points to prioritize" in both example yamls. It worked for paper-finder
Stage 1 (filtering which paper locations get searched) but was
completely ignored by autoresearch's `find_pending` when picking which
module to inject next.

`modules_md.find_pending` sorted on a single key — complexity only —
and preserved `modules.md` write order within each complexity tier.
Write order is whatever sequence paper-finder discovered papers in,
which correlates with search-engine relevance, not architectural
preference. So:

- yaml says `preferred_locations: [backbone, neck, head, loss]`
- 18 pending modules, all `low` complexity
- autoresearch picks them in write order (e.g. CBAM backbone, NWD-Loss
  loss, P2-Small-Object-Head head, SPD-Conv backbone, ...)
- Any location order appears — including unlisted locations interleaved
  between listed ones
- Changing the yaml to `[loss, head, neck, backbone]` has **zero**
  observable effect

The yaml option made a promise the code never kept.

## The fix

`modules_md.find_pending` gains a `preferred_locations=None` parameter.
When non-empty, it becomes a secondary sort key after complexity:

```python
# v1.7.3 signature
def find_pending(path, sort_by_complexity=True, preferred_locations=None):
    ...
```

Sort key tuple, in order:
1. Complexity — low (0) < medium (1) < high (2) < missing (99)
2. Location rank — index in `preferred_locations` list (case-insensitive).
   Locations not in the list get rank `len(preferred_locations)`, so
   they sort after all listed locations.
3. Write order in modules.md — stable tiebreak, preserves paper-finder
   write sequence for modules tying on both above keys.

`autoresearch/SKILL.md § Step 2 Priority A` now reads
`research_config.yaml → paper_finder.modules.preferred_locations` and
passes it through:

```python
_cfg = yaml.safe_load(pathlib.Path("research_config.yaml").read_text()) or {}
preferred = (_cfg.get("paper_finder", {})
                 .get("modules", {})
                 .get("preferred_locations")) or None
pending = mm.find_pending("modules.md", preferred_locations=preferred)
```

**Unlisted locations are not excluded** — they sort last but still get
picked eventually. This is a `prioritize` semantic, not a filter. If a
paper-finder expand introduces a `Location: label_assignment` module
and the user's yaml lists only `[backbone, neck, head, loss]`, the new
module still runs after all listed-location modules.

## Backward compatibility

- `find_pending(path)` with no `preferred_locations` → identical
  behaviour to v1.7.2 (single key on complexity, write order preserved
  within tie).
- `find_pending(path, preferred_locations=[])` → same as None. Empty
  list is not treated as "exclude everything".
- Missing `preferred_locations` in yaml → autoresearch passes None.
- Module with missing `Location` field → treated as unlisted, sorts last
  (same rank as other unlisted).
- Case mismatch between yaml entries and module `Location` field → both
  are lower-cased before comparison (paper-finder sometimes writes
  `Backbone` with capital B).

## Files changed in v1.7.3

| File | Change |
|---|---|
| `shared/modules_md.py` | `find_pending` signature gains `preferred_locations`; secondary sort logic + docstring |
| `shared/test_modules_md.py` | +6 tests for new parameter behaviour |
| `autoresearch/SKILL.md` | `§ Step 2 Priority A` reads yaml, passes through to `find_pending` |
| `examples/research_config.visdrone-detection.yaml` | Comment clarifies autoresearch also uses this key |
| `examples/research_config.visdrone-mot.yaml` | Same |
| `CHANGELOG_v1.7.3.md` | New (this file) |
| `README.md` | Version bumped |

## Files NOT changed

Everything else is byte-identical to v1.7.2. Notably: `weight_transfer.py`,
`state_migrate.py`, `train-script-spec.md`, `file-contracts.md`,
`research-orchestrator/SKILL.md`, `dataset-hunter/SKILL.md`,
`paper-finder/SKILL.md`, all templates. No behavioural change to any
helper, contract, or cross-skill handoff.

## Test coverage

| Suite | v1.7.2 | v1.7.3 |
|---|---|---|
| `shared/test_modules_md.py` | 12 | **18** (+6) |
| `shared/test_templates.py` | 3 | 3 |
| `shared/test_weight_transfer.py` | 36 | 36 |
| **Total** | 51 | **57** |

New tests:
- Orders within complexity by preferred order
- Reversed preferred list reverses order
- Partial list — unlisted locations go after listed
- `preferred_locations=None` preserves v1.7.2 behaviour
- Case-insensitive matching
- Empty list behaves same as None

SKILL.md python snippets: 83/83 parse.

## Upgrade path

Drop-in. No manual migration. Existing `research_config.yaml` files
work as-is — if `preferred_locations` was already set (it was, in both
example yamls from v1.6 onward), it now starts doing what the comment
always claimed it did. If it wasn't set, behaviour is identical to
v1.7.2.

For in-flight projects: the change takes effect on the next
autoresearch Step 2 Priority A. Next pending module picked may differ
from what v1.7.2 would have picked. Past keep/discard decisions are
not affected.

## Notes

This bug had been latent since v1.0. The yaml field was introduced to
let paper-finder constrain its search and autoresearch never knew about
it. The comment in the yaml said "prioritize", which was true for
paper-finder (it only finds papers at those locations) but not for
autoresearch (which picked anything paper-finder left in modules.md, in
write order). v1.7.3 makes the two skills share the field consistently.
