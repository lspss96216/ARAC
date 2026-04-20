# Changelog — v1.7

Release type: **Feature addition.** v1.6 was 26 bug fixes; v1.7 introduces one
new capability (architectural injection with pretrained weight transfer) while
keeping every v1.6 code path and contract stable. If you don't use the new
feature, v1.7 behaves identically to v1.6.

v1.6 → v1.7 migration: no file rewrites, no state schema change, existing
`modules.md` and `pipeline_state.json` work as-is. Old modules without
`Integration mode` field default to `hook` (the v1.6 path).

---

## The one new thing — `yaml_inject` mode

Autoresearch can now modify the model's **layer graph**, not just wrap or
swap existing layers. A module with `Integration mode: yaml_inject` is
applied by:

1. Generating a new YAML that inserts the module at the specified
   position/scope in the base model's architecture.
2. Building a fresh `YOLO` from that YAML.
3. Transferring pretrained weights from the base `.pt` into the custom
   model per a computed `layer_map` (orig index → custom index, accounting
   for the shifted indices caused by insertion).
4. Forcing every `Lazy*` wrapper to build its inner module before the
   optimizer captures `model.parameters()`.
5. Registering an `on_train_epoch_start` callback that re-transfers
   weights before epoch 0, in case the trainer re-initialised layers
   during its own setup.

This is the v1.7 scope — **receiving** insertions into existing structure.
Arbitrary YAML replacement (`mode: full_yaml`) is deferred to v1.8.

Why this matters: most attention / SE / lightweight-module papers work by
inserting a new block at a specific point in the backbone. In v1.6,
autoresearch could only wrap existing layers' forward passes — which gave
the right gradient flow for some modules but not others, and couldn't
handle papers like CBAM that add a new parameterised block between
existing ones. v1.7 closes that gap.

---

## New files

| File | Purpose |
|---|---|
| `shared/weight_transfer.py` | Layered helper: parse base yaml → generate custom yaml → build model → transfer weights → force lazy build → register Stage 2 callback. ~550 lines, three abstraction levels (low/mid/high) so v1.8's `full_yaml` mode can reuse the low-level primitives. |
| `shared/test_weight_transfer.py` | 19 unit tests covering YAML generation, layer_map math, strict-mode failures, Insertion schema round-trip, and entry-point error paths. Runs without torch/ultralytics. |
| `shared/templates/arch_spec.schema.json` | JSON schema owning the format of `arch_spec.json` (the per-experiment injection spec). Tagged-dict `position` definition so v1.8 can extend with `after_named` / `before_class` without breaking old specs. |

## Modified files

### `shared/train-script-spec.md`

- Section ② example adds `ARCH_INJECTION_ENABLED` and `ARCH_INJECTION_SPEC_FILE`
- Section ④ example adds the `if ARCH_INJECTION_ENABLED` branch calling
  `build_custom_model_with_injection`, and sets `pretrained=False` in
  `model.train(...)` (weight_transfer handles transfer manually)
- § Modifiable variables table gains two rows for the new variables
- **New § Architecture injection (v1.7)** — full chapter covering the
  two-mode table, yaml_inject workflow, `arch_spec.json` format, Lazy-wrapper
  contract, coexistence with `inject_modules()`, and failure-mode table
- Compliance checker adds `ARCH_INJECTION_*` to required vars and introduces
  a `required_branches` check for the conditional substring presence
- Deployment tree updated with `weight_transfer.py` + `test_weight_transfer.py`
  + `arch_spec.schema.json`

### `shared/file-contracts.md`

- `modules.md` section gains a **Field conventions table** covering every
  documented field, including v1.6 fields and the new `Integration mode`
  with its warn-not-reject policy
- New section **`arch_spec.json`** with JSONC schemas for both v1.7
  (`insertions`) and v1.8+ (`full_yaml`) modes

### `shared/modules_md.py`

- Added `KNOWN_INTEGRATION_MODES = {"hook", "yaml_inject", "full_yaml"}`
- Added `DEFAULT_INTEGRATION_MODE = "hook"` — backward compat default
- Added `Module.integration_mode` property (returns default when missing,
  blank, or whitespace-only; returns the raw value for known & unknown)
- `append_module` warns on unknown values but does not reject — forward
  compatibility for future modes added in later skill versions

### `shared/test_modules_md.py`

Added 4 tests covering the new field: default when absent, parsing
`yaml_inject`, unknown-value warn-but-accept, whitespace treated as blank.
v1.6 → v1.7: 8 → 12 tests. All passing.

### `shared/test_templates.py`

Added `ARCH_INJECTION_ENABLED` / `ARCH_INJECTION_SPEC_FILE` to required
variables and a `required_branches` list checking for `if ARCH_INJECTION_ENABLED`
and `build_custom_model_with_injection` literal substrings in train
templates.

### `shared/templates/train.py.detection` and `train.py.tracking`

- Section ② adds the two new variables, both default-off
- Section ④ `main()` gets the `if ARCH_INJECTION_ENABLED:` conditional
  calling `build_custom_model_with_injection`; `else` branch falls through
  to the v1.6 `YOLO(WEIGHTS)` path

### `autoresearch/SKILL.md`

- New **§ Dispatch on Integration mode (v1.7)** subsection in Priority A,
  splitting the flow into hook / yaml_inject / full_yaml / unknown paths,
  with discard-and-log for unsupported or misconfigured modules
- Existing `§ Branches` renamed `§ Branches (hook mode)` to make the
  split explicit
- New **§ Branches (yaml_inject mode, v1.7)** with 6 concrete steps: acquire
  lazy wrapper, extract spec from `Integration notes`, write `arch_spec.json`,
  flip `ARCH_INJECTION_ENABLED`, commit three files together
  (`arch_spec.json` + `train.py` + `custom_modules.py`), update status
- Strict-mode crash handling connects back to v1.6 crash counter and
  `consecutive_crashes` stop trigger

### `paper-finder/SKILL.md`

- New **§ Integration mode (v1.7)** in Phase 5 with the three legal values
  and a **heuristic decision table** (paper language patterns → injection
  mode)
- New **§ yaml_inject modules need injection spec fields** section listing
  the required `Integration notes` fields (`module_class`, `position`,
  `scope`, `yaml_args`, `module_kwargs`)
- Phase 6 `append_module` example gains `Integration mode: hook` (explicit
  default) and a full **yaml_inject variant example** showing the CBAM case
  with complete injection spec prose

## Unchanged between v1.6 and v1.7

These files are byte-identical to v1.6 — no state machine or scaffold
changes required to adopt v1.7:

- `research-orchestrator/SKILL.md`
- `dataset-hunter/SKILL.md`
- `shared/state_migrate.py`
- `shared/parse_metrics.py`
- `shared/templates/track.py.tracking`
- `examples/research_config.visdrone-detection.yaml`
- `examples/research_config.visdrone-mot.yaml`

---

## Design decisions locked in v1.7 (informing v1.8)

These choices were made deliberately so v1.8's `full_yaml` mode doesn't
require breaking changes:

1. **Tagged-dict `position`** (`{"kind": "after_class", "class_name": "Conv"}`
   rather than `"after:Conv"`) — v1.8 can add `before_class`, `after_named`,
   `at_relative` without changing the schema's shape.
2. **Generic variable names** (`ARCH_INJECTION_ENABLED` / `ARCH_INJECTION_SPEC_FILE`
   rather than `CBAM_ENABLED` / `USE_CUSTOM_YAML`) — no rename needed when
   `full_yaml` mode ships. `train.py` Section ④'s conditional branches on
   the boolean flag; the dispatch on `spec["mode"]` happens inside
   `weight_transfer`, not in `train.py`.
3. **Warn-not-reject for `Integration mode`** — unknown future values don't
   break old modules.md files. Parser emits stderr warning and defaults to
   hook.
4. **Per-entry strict mode** in `transfer_weights` — rejects on any
   layer_map entry transferring 0 tensors, not aggregate. Catches silent
   class-mismatch errors that aggregate checks miss.
5. **`ARCH_INJECTION_SPEC_FILE` points to a JSON file, not an inline dict
   literal in Section ②** — keeps Section ② clean (short assignments only),
   and the JSON is git-tracked so `git reset --hard HEAD~1` on discard
   rewinds the spec alongside `train.py`.

v1.8's work is adding the `mode: full_yaml` branch inside
`weight_transfer.build_custom_model_with_injection` (currently
`NotImplementedError`) and teaching paper-finder to emit `full_yaml` specs
for papers that redesign whole substructures (BiFPN, DEtr decoder, etc.).

---

## Test coverage

Before release (all green):

| Suite | Count | Notes |
|---|---|---|
| `shared/test_modules_md.py` | 12 | 8 v1.6 + 4 v1.7 (integration_mode) |
| `shared/test_templates.py`  |  3 | detection + tracking + track templates |
| `shared/test_weight_transfer.py` | 19 | All v1.7; runs without torch/ultralytics |
| **Total** | **34** | |

No regressions in v1.6 tests. `modules_md.py` changes are additive (new
property, new constants, new warn path in validate) — existing callers
work unchanged.

---

## Known limitations (v1.7 scope boundaries)

- **`mode: full_yaml` raises `NotImplementedError`.** Reserved for v1.8.
  Modules tagged `full_yaml` by paper-finder are discarded by autoresearch
  at Step 2 dispatch time rather than crashing the run.
- **No `yaml_inject` for tracker modules.** yaml_inject only affects
  `train.py`'s detector. `track.py` has no equivalent machinery in v1.7.
  Autoresearch discards tracker modules tagged yaml_inject with a clear
  log_discovery message.
- **Priority E combinations of kept yaml_inject changes are not
  automatic.** v1.7 can keep multiple yaml_inject experiments, but merging
  two custom YAMLs into one "combination" experiment is not implemented.
  Priority E falls back to hook-mode combinations.
- **re-pretrain trigger doesn't count yaml_inject keeps.** Orchestrator
  Stage 3 Step 6.5's `architecture_keeps ≥ 3` counter continues to count
  hook-mode keeps only. yaml_inject kept experiments don't trigger
  re-pretrain in v1.7 (keeps state machine stable; v1.8 can unlock this).
- **B2 (guard-metric baseline drift) still preserved as-is** — same scope
  boundary as v1.6. `baseline_snapshot` field reserved for future.

---

## Install

```bash
unzip pipeline_v1.7.zip
cp -r pipeline_v1.7/skills/* ~/.claude/skills/
cp pipeline_v1.7/examples/* ~/.claude/skills/examples/

# Verify
cd ~/.claude/skills/shared
python3 test_modules_md.py      # 12/12
python3 test_templates.py       # 3/3
python3 test_weight_transfer.py # 19/19
```

Pre-v1.7 state files resume transparently (no schema migration needed;
`state_migrate.py` unchanged from v1.6).
