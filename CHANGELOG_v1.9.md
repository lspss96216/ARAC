# Changelog — v1.9

Release type: **`full_yaml` mode + resource-impact auto-halve.** v1.9
unblocks the architectural-rewrite class of experiments that v1.7-v1.8
could not run (BiFPN replacing PAFPN, DETR-style decoders, ConvNeXt
backbones, P2 head architectures). Adds the resource-impact safety net
that prevents the silent CPU-fallback class of failures observed in
real runs. Drop-in on top of v1.8.

## Background

The 31-loop production run documented for v1.8 surfaced a clear
ceiling: 5 of 11 paper-finder modules (BiFPN, P2 head, TPH, QueryDet,
SPD-Conv) could not be tested at all under v1.7-v1.8 because
`weight_transfer.apply_yaml_spec` was a placeholder raising
`NotImplementedError`, and the autoresearch dispatch branch for
`full_yaml` mode discarded the modules without running them. The
result: 0/8 architectural keeps in that run; the entire +4.4% mAP
came from hyperparameter sweeps stacked atop a clean fine-tune.

v1.9 implements the missing piece — `apply_yaml_spec` end-to-end +
the autoresearch dispatch branch + paper-finder Phase 5 heuristic +
modules.md schema + train-script-spec.md chapter — so the same
production run today would have access to those 5 blocked modules
plus any future structural-change experiments.

Separately, v1.9 adds `resource_impact` tagging in modules.md and
auto-halve in autoresearch Step 3 to prevent the silent CPU-fallback
mode of failure (real run's Loop 7 — EMA at batch=64 silently moved
to CPU TaskAlignedAssigner, training 3-7× slower, run discarded as
"untrainable in equal-budget regime" with no signal that the assigner
moved).

## Sections

| Section | Theme |
|---|---|
| **A1-A5** | full_yaml mode (apply_yaml_spec + dispatch + paper-finder + schema + spec doc) |
| **C4** | resource_impact field + auto-halve BATCH_SIZE |

(A and C4 were the deferred-from-v1.8 items per v1.8's CHANGELOG outlook.)

## A. full_yaml mode

### A1 — `weight_transfer.apply_yaml_spec` implementation

**Severity in v1.7-v1.8**: hard `NotImplementedError`. 5 of 11 modules
in real run could not be tested.

**v1.9 fix**: ~140 lines of new logic in `weight_transfer.py`:

```python
def apply_yaml_spec(
    base_weights: str,
    custom_yaml_path: str,
    layer_map_override: list[dict] | None = None,
    layer_map_strategy: str = "auto",
    transfer_scope: str = "backbone",
    imgsz: int = 640,
    strict: bool = True,
) -> "YOLO":
```

8-step flow:
1. Validate args (before importing ultralytics so error paths are
   testable in environments without ultralytics)
2. Load base `.pt` and parse its YAML
3. Load agent's custom YAML from disk; verify `backbone` + `head`
   sections exist
4. Compute layer_map: `auto` → class+position monotonic-cursor
   heuristic; `override` → resolve list of `{base_idx, custom_idx}` to
   dict
5. Strict mode: verify every parameter-bearing base layer in
   transfer_scope was paired (raises with unpaired class list if not)
6. Build custom YOLO from agent's YAML
7. transfer_weights with computed layer_map (per-entry strict)
8. force_lazy_build + register_stage2_callback (same as yaml_inject's
   v1.7.7 hook-survival mechanism)

3 new helpers in `weight_transfer.py`:
- `_yaml_layer_class(line)` — get class name from a YAML layer line
- `_flatten_yaml_layers(yaml_dict)` — backbone + head as one ordered list
- `auto_compute_full_yaml_layer_map(base, custom, transfer_scope)` —
  the auto pairing heuristic
- `_resolve_layer_map_override(override_list)` — schema-list → dict

`build_custom_model_with_injection` no longer raises on
`mode='full_yaml'`; it dispatches to `apply_yaml_spec` with all spec
fields forwarded. Train.py templates remain unchanged — the dispatcher
inside `build_custom_model_with_injection` is the single entry point.

12 new tests:
- 4 dispatch / arg validation tests
- 6 `auto_compute_full_yaml_layer_map` tests covering identical YAMLs,
  backbone-only scope, neck replacement, class rename, cursor monotonic
  advance, extra custom layers
- 3 `_resolve_layer_map_override` tests (basic, partial entries skip,
  empty/None)

Replaced v1.7's `test_build_rejects_full_yaml_mode` with v1.9's
`test_build_dispatches_full_yaml_mode` (monkey-patches
`apply_yaml_spec` to verify dispatch and kwargs forwarding without
needing ultralytics + a real `.pt`).

### A2 — autoresearch SKILL Dispatch on Integration mode

**v1.9 fix**: replaced "discard with limitation" branch with full
implementation. New § "v1.9 — full_yaml branch" in § Apply the change
documents the 6-step apply flow:

1. Read `Integration notes` from modules.md (`custom_yaml_template`,
   `layer_map_strategy`, etc.)
2. Write the custom YAML to disk as `<base_stem>_<module>.yaml`
   (filename-encodes-scale matters, per v1.8 B2)
3. Write `arch_spec.json` with `mode: full_yaml` shape
4. Flip `ARCH_INJECTION_ENABLED = True` (same as yaml_inject)
5. Commit `arch_spec.json`, custom YAML file, `train.py` together
6. `mm.update_status(... "injected")`

Documents 3 full_yaml-specific failure modes:
- `auto` strategy unpaired layers → strict mode crash (with hint to
  use `override` or `transfer_scope=full`)
- Missing `backbone`/`head` section → ValueError
- Filename scale mismatch → ultralytics builds wrong scale silently

Defensive branch added for legacy modules.md entries with the v1.8-only
"reserved" sentinel (currently empty set, placeholder for future
migrations).

### A3 — paper-finder Phase 5 heuristic + Integration notes schema

**v1.9 fix**: § Integration mode block expanded. The 4-rule preference
ladder:

```
1. Loss / regularizer / scheduler change → hook
2. Same layer count, only forward behaviour differs → hook
3. New layers inserted but base structure preserved → yaml_inject
4. Backbone, neck, OR head substantively replaced → full_yaml
```

Heuristic table extended with full_yaml triggers:

| Paper language | Mode |
|---|---|
| "we replace the entire neck with BiFPN" | full_yaml |
| "our backbone uses ConvNeXt blocks instead of CSP-Darknet" | full_yaml |
| "we propose a new decoder with cross-attention" | full_yaml |
| "the head outputs N×M predictions instead of 4-bbox + class" | full_yaml |

New § "v1.9 — full_yaml modules need a custom YAML template" documents:
- Required `Integration notes` schema (`custom_yaml_template`,
  `layer_map_strategy`, `layer_map_override`, `transfer_scope`)
- When to choose `auto` vs `override` (decision table)
- When to choose `backbone` vs `backbone+neck` vs `full` (decision table)
- When NOT to use full_yaml (prefer yaml_inject when changes are purely
  insertions)
- Concrete example: BiFPN replacing PAFPN with auto strategy + backbone
  scope

### A4 — `arch_spec.schema.json` updated

**v1.9 fix**: full_yaml branch went from 1 placeholder field to 5
proper fields:

```json
{
  "mode": "full_yaml",
  "custom_yaml_path": "yolo26x_bifpn.yaml",
  "layer_map_strategy": "auto" | "override",
  "layer_map_override": [{"base_idx": ..., "custom_idx": ...}, ...],
  "transfer_scope": "backbone" | "backbone+neck" | "full",
  "strict": true
}
```

`mode` field description updated — v1.9 no longer says full_yaml is
"reserved".

### A5 — `train-script-spec.md § Architecture injection` chapter

**v1.9 fix**: § "Two injection modes" table updated to include full_yaml
row. New § "How `full_yaml` works (v1.9)" mirrors the existing yaml_inject
section's structure, documents the 7-step apply_yaml_spec flow,
provides decision tables for `transfer_scope` and `layer_map_strategy`,
and lists 4 full_yaml-specific failure modes. § ARCH_INJECTION_SPEC_FILE
format gets two new full_yaml examples (auto + override variants).

## C. resource_impact + auto-halve

### C4 — `modules.md` `resource_impact` field

**v1.9 fix**: New optional field on every module entry. Three
canonical values plus warn-not-reject for unknowns:

| Tag | Meaning | Auto-action |
|---|---|---|
| `vram_4x` | ~4× baseline VRAM (P2 head, dense attention) | autoresearch ×0.25 BATCH_SIZE |
| `vram_2x` | ~2× baseline VRAM (single attention layer added) | autoresearch ×0.5 BATCH_SIZE |
| `cpu_fallback_risk` | Triggers CPU assigner fallback at default batch | autoresearch ×0.5 BATCH_SIZE + log |

`Module.resource_impact` property added to `modules_md.py` (returns
None for absent/blank field, raw string otherwise — same warn-not-reject
policy as `integration_mode`).

`KNOWN_RESOURCE_IMPACTS` set exported for caller validation.

paper-finder SKILL gets new § "v1.9 — Resource impact tagging" with:
- Description of why this matters (real run's silent CPU-fallback
  example referenced)
- Decision table mapping paper language to recommended tag
- Guidance: prefer to omit when in doubt (loud failure beats silent
  over-correction)

autoresearch SKILL Step 3 Modify gets new § "v1.9 — Resource-impact
auto-halve" subsection:
- Read `chosen.resource_impact`
- Halve BATCH_SIZE in train.py per the table (vram_4x = halve twice)
- Log a `resource_constraint` discovery
- Track original via `state["batch_size_pre_autohalve"]`
- On NEXT iteration that does NOT touch a high-impact module, restore
  original BATCH_SIZE — auto-halve is per-iteration, not sticky
- Auto-halve is independent of crash-pause halve (v1.7.6 path); both
  reductions can stack under combined pressure

5 new tests in `test_modules_md.py` cover absent / blank / known
values / unknown / round-trip-through-append.

## Files changed

### New
None (this release adds only to existing files).

### Modified

| File | Change summary |
|---|---|
| `shared/weight_transfer.py` | A1: `apply_yaml_spec` (~140 lines) + 3 helpers; `build_custom_model_with_injection` dispatches full_yaml |
| `shared/test_weight_transfer.py` | A1: +12 tests (49 → 61) |
| `shared/templates/arch_spec.schema.json` | A4: full_yaml branch fully specified (5 fields) |
| `shared/modules_md.py` | C4: `resource_impact` property + `KNOWN_RESOURCE_IMPACTS` set |
| `shared/test_modules_md.py` | C4: +5 tests (18 → 23) |
| `shared/state_migrate.py` | C4: `batch_size_pre_autohalve` default added |
| `shared/train-script-spec.md` | A5: § "How full_yaml works", § ARCH_INJECTION_SPEC_FILE full_yaml examples, table updated |
| `paper-finder/SKILL.md` | A3: § "Integration mode" expanded with full_yaml ladder + heuristic table; new § "v1.9 — full_yaml modules need a custom YAML template"; new § "v1.9 — Resource impact tagging" |
| `autoresearch/SKILL.md` | A2: § Dispatch full_yaml branch implemented; § Apply the change full_yaml subbranch added; Step 3 § "v1.9 — Resource-impact auto-halve" |
| `research-orchestrator/SKILL.md` | C4: state init adds `batch_size_pre_autohalve` |
| `CHANGELOG_v1.9.md` | New (this file) |
| `README.md` | v1.9 banner, Versions list, layout tree |

### Unchanged

`dataset-hunter/SKILL.md`, `shared/hook_utils.py`, `shared/parse_metrics.py`,
`shared/file-contracts.md`, `shared/templates/train.py.detection`,
`shared/templates/train.py.tracking`, `shared/templates/track.py.tracking`,
`shared/results-tsv-guide.md`, `shared/test_hook_utils.py`,
`shared/test_invariants.py`, all earlier CHANGELOGs, both yaml examples.

## Test coverage

| Suite | v1.8 | v1.9 |
|---|---|---|
| `shared/test_modules_md.py` | 18 | **23** (+5) |
| `shared/test_templates.py` | 3 | 3 |
| `shared/test_weight_transfer.py` | 49 | **61** (+12) |
| `shared/test_hook_utils.py` | 13 | 13 |
| `shared/test_invariants.py` | 23 | 23 |
| **Total python tests** | **106** | **123** (+17) |
| SKILL.md python snippets | 93 | **96** (+3) |
| YAML examples | 2 | 2 |

The 17 new tests focus on `apply_yaml_spec` correctness (the v1.7-v1.8
silent-failure-class) and `resource_impact` parser round-tripping.

## Upgrade path

Drop-in. State migration covers the new `batch_size_pre_autohalve` key
(defaults to None). Existing `research_config.yaml` files work unchanged.
Existing `modules.md` files without `resource_impact` work unchanged
(absent field = no auto-halve, default behaviour).

For users with v1.8 in-progress runs:
- Resume continues normally
- Modules previously tagged `Status: blocked` because they needed
  full_yaml can be re-tagged `Status: pending` once the user verifies
  paper-finder's Integration notes have a `custom_yaml_template`
- Existing arch_spec.json files using `mode: insertions` are unchanged
  by v1.9 (full_yaml is purely additive to the schema)

For users running new pipelines with v1.9: paper-finder will start
emitting `full_yaml` mode for structural-change papers automatically.
Expect the first few full_yaml runs to need iteration on
`layer_map_strategy` (auto often suffices for neck-only changes; switch
to override when the strict mode raises with unpaired layers).

## Operational observations

**`auto` strategy reliability.** The class+position monotonic-cursor
heuristic was designed for the most common case (preserve backbone,
replace neck). It will under-pair when:
- Custom YAML inserts new layer types between matching base classes
  (the cursor advances past intervening positions correctly, but if
  the inserted classes happen to match a downstream base class, the
  cursor mis-aligns)
- Class names are renamed (Conv → SPDConv) — these layers won't pair
  at all, and strict mode raises (which is the correct behaviour;
  agent should use `override` or `transfer_scope=full`)

**`transfer_scope=backbone+neck` heuristic.** Detection of the
"first Detect-family layer" relies on substring match against
`Detect` and `RTDETRDecoder`. If a paper's custom decoder uses neither
class name, the heuristic treats the entire model as neck and may
over-transfer. Workaround: use `transfer_scope=backbone` (safer) or
explicitly set `override` with pairs only up to the desired boundary.

**Auto-halve calibration.** The vram_4x = ×0.25, vram_2x = ×0.5
mapping is conservative. For machines with very high BATCH_SIZE
defaults (batch=64+ on H100), this may over-halve and waste GPU. Future
v1.10 may add per-machine baseline calibration; for now, users with
abundant VRAM can manually edit train.py's BATCH_SIZE after the
auto-halve fires (autoresearch will restore the auto-halved value on
next iteration regardless).

## v1.10 outlook

v1.9 closes the architecture-injection capability gap. v1.10 candidates,
if real-world v1.9 use surfaces them:

- Per-machine resource calibration (replace fixed halve-counts with
  empirical VRAM measurement)
- yaml_inject REPLACE semantics (currently INSERT-only, see v1.7
  limitation in real run's discoveries.md)
- `apply_yaml_spec` adapter generation for known shape mismatches
  (parallel to Step 5.5's repair primitives for yaml_inject)
- Integration test that builds an actual PyTorch model + runs forward
  (currently all tests are mock; would catch the v1.7 → v1.7.6 class
  of "unit tests pass but ultralytics dispatch breaks")

These are speculative — actual v1.10 scope depends on which v1.9
limitations surface in real runs.
