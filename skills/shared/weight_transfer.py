"""weight_transfer.py — Architectural injection + pretrained weight transfer for YOLO.

v1.7 (C): supports mode='insertions' for programmatic YAML generation.
v1.8+ (A): will add mode='full_yaml' for agent-written YAML specs.

Layered API:
  Low level (used by both C and A):
    parse_base_yaml(pt_path)            → base_yaml_dict
    split_sections(yaml_dict)           → (backbone_lines, neck_lines, head_lines)
    generate_custom_yaml(base, insertions)  → new_yaml_dict + insertion_record
    compute_layer_map(orig_yaml, custom_yaml, skip_head)  → dict[int, int]
    transfer_weights(orig_model, custom_model, layer_map, strict)  → count
    force_lazy_build(nn_module, imgsz)  → None
    register_stage2_callback(yolo, pretrained_pt, layer_map)  → None

  Mid level (C entry point):
    build_custom_model_with_injection(base_weights, spec, imgsz)  → YOLO

  High level (A entry point, v1.8+):
    apply_yaml_spec(...)  → YOLO      # NotImplementedError in v1.7

All helpers raise on error — no silent failures. transfer_weights enforces
per-entry strict mode: every layer_map entry must transfer ≥1 tensor or raise.

Usage from train.py (Section ④):
    if ARCH_INJECTION_ENABLED:
        import json, pathlib
        from weight_transfer import build_custom_model_with_injection
        spec = json.loads(pathlib.Path(ARCH_INJECTION_SPEC_FILE).read_text())
        model = build_custom_model_with_injection(WEIGHTS, spec, imgsz=IMGSZ)
    else:
        from ultralytics import YOLO
        model = YOLO(WEIGHTS)
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Insertion:
    """One architectural insertion. Flat dataclass for ease of json↔obj round-trip."""
    module_class: str
    position_kind: str           # "after_class" | "at_index"
    position_value: Any          # str for after_class, int for at_index
    scope: str                   # "backbone" | "neck" | "head" | "all"
    yaml_args: list
    module_kwargs: dict

    @classmethod
    def from_dict(cls, d: dict) -> "Insertion":
        pos = d["position"]
        if pos["kind"] == "after_class":
            pk, pv = "after_class", pos["class_name"]
        elif pos["kind"] == "at_index":
            pk, pv = "at_index", int(pos["index"])
        else:
            raise ValueError(f"Unknown position.kind: {pos['kind']!r}")
        return cls(
            module_class=d["module_class"],
            position_kind=pk,
            position_value=pv,
            scope=d["scope"],
            yaml_args=list(d.get("yaml_args", [])),
            module_kwargs=dict(d.get("module_kwargs", {})),
        )


@dataclass
class InsertionRecord:
    """Bookkeeping from generate_custom_yaml, consumed by compute_layer_map."""
    inserted_base_positions: list[int]   # 0-indexed positions in the BASE yaml
                                         # (not custom). Sorted ascending.


# ──────────────────────────────────────────────────────────────────────────────
# Low level — YAML parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_base_yaml(pt_path: str) -> dict:
    """Extract the YAML dict that Ultralytics used to build a .pt model.

    Ultralytics stores the parsed yaml in model.yaml (a dict with 'backbone'
    and 'head' keys — note: 'neck' is conventionally the top of 'head' in
    ultralytics yaml; we split it logically in split_sections).
    """
    # Delay ultralytics import so this module is testable without it
    from ultralytics import YOLO
    m = YOLO(pt_path)
    y = m.model.yaml
    if not isinstance(y, dict) or "backbone" not in y:
        raise RuntimeError(
            f"{pt_path} has no parseable yaml dict (got {type(y).__name__}). "
            f"Only standard Ultralytics models are supported."
        )
    # Deep copy — caller mutates
    return json.loads(json.dumps(y))


def split_sections(yaml_dict: dict) -> tuple[list, list, list]:
    """Split a YOLO yaml's layer list into (backbone, neck, head).

    Ultralytics stores layers in two keys: 'backbone' and 'head'. Within
    'head', the final Detect layer is the head proper; everything before
    Detect in the 'head' list is the neck.

    Returns three lists of YAML lines ([from, repeats, module, args]).
    """
    backbone = list(yaml_dict.get("backbone", []))
    head_raw = list(yaml_dict.get("head", []))

    # Find the last Detect-family layer. Everything at that position and
    # after is the head proper; everything before is the neck.
    DETECT_NAMES = {"Detect", "Segment", "Pose", "OBB", "Classify", "RTDETRDecoder"}
    detect_idx_in_head = None
    for i, line in enumerate(head_raw):
        module_name = line[2] if len(line) >= 3 else None
        if module_name in DETECT_NAMES:
            detect_idx_in_head = i
            break

    if detect_idx_in_head is None:
        # No Detect found — treat entire "head" as neck, head is empty
        neck, head = head_raw, []
    else:
        neck = head_raw[:detect_idx_in_head]
        head = head_raw[detect_idx_in_head:]

    return backbone, neck, head


def _reassemble(backbone: list, neck: list, head: list) -> dict:
    """Rebuild a yaml dict after section edits."""
    return {"backbone": backbone, "head": neck + head}


# ──────────────────────────────────────────────────────────────────────────────
# Low level — YAML generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_custom_yaml(
    base_yaml: dict,
    insertions: list[Insertion],
) -> tuple[dict, InsertionRecord]:
    """Apply a list of insertions to base_yaml, return (custom_yaml, record).

    Rules:
      - All indices in insertions (for kind='at_index') are BASE indices.
      - Processes insertions in base-index-descending order so earlier
        insertions don't shift later ones.
      - For kind='after_class', expands to every matching layer within scope.
      - Validates scope against position: kind='at_index' with out-of-scope
        index raises.
      - from=-1 is used for all inserted lines (channel-preserving assumption).

    Returns (custom_yaml_dict, InsertionRecord(sorted-asc list of base positions
    where an insertion happened — used by compute_layer_map to reconstruct
    the index mapping).
    """
    backbone, neck, head = split_sections(base_yaml)

    # Build a flat "scope-tagged" view of the base yaml so we can resolve
    # insertion targets uniformly. Each entry: (section, section_local_idx, line).
    flat: list[tuple[str, int, list]] = []
    for i, line in enumerate(backbone): flat.append(("backbone", i, line))
    for i, line in enumerate(neck):     flat.append(("neck",     i, line))
    for i, line in enumerate(head):     flat.append(("head",     i, line))

    # Global base index = position in flat
    # Scope boundaries in global base indices:
    n_b = len(backbone)
    n_bn = n_b + len(neck)
    # backbone: [0, n_b)
    # neck:     [n_b, n_bn)
    # head:     [n_bn, len(flat))

    def scope_range(scope: str) -> range:
        if scope == "backbone": return range(0, n_b)
        if scope == "neck":     return range(n_b, n_bn)
        if scope == "head":     return range(n_bn, len(flat))
        if scope == "all":      return range(0, len(flat))
        raise ValueError(f"Unknown scope: {scope!r}")

    def global_class_at(base_idx: int) -> str:
        return flat[base_idx][2][2] if len(flat[base_idx][2]) >= 3 else ""

    # Resolve each insertion to a list of base-global indices
    # AFTER which the new layer should be inserted.
    resolved: list[tuple[int, Insertion]] = []   # (after_idx, insertion)

    for ins in insertions:
        rng = scope_range(ins.scope)

        if ins.position_kind == "after_class":
            cname = ins.position_value
            matches = [i for i in rng if global_class_at(i) == cname]
            if not matches:
                raise ValueError(
                    f"Insertion {ins.module_class!r}: no {cname!r} layer found "
                    f"in scope {ins.scope!r}. Check the base model's layer list."
                )
            for i in matches:
                resolved.append((i, ins))

        elif ins.position_kind == "at_index":
            idx = ins.position_value
            if idx not in rng:
                raise ValueError(
                    f"Insertion {ins.module_class!r}: at_index={idx} is "
                    f"outside scope {ins.scope!r} (valid base indices: "
                    f"{rng.start}..{rng.stop - 1})."
                )
            resolved.append((idx, ins))

        else:
            raise ValueError(f"Unknown position_kind: {ins.position_kind!r}")

    # Process in DESCENDING order of after_idx so earlier edits don't shift
    # later base-index references. Ties (two insertions at same after_idx):
    # fall back to the definition order (first-defined ends up closer to target).
    # We enumerate BEFORE sorting to capture the stable original order — using
    # resolved.index(t) during sort is O(n²) and breaks when list mutates.
    indexed = list(enumerate(resolved))
    indexed.sort(key=lambda it: (-it[1][0], it[0]))
    resolved_ordered = [t for _, t in indexed]

    # Work on a mutable copy of flat
    flat_mut: list[tuple[str, int, list]] = list(flat)

    for after_idx, ins in resolved_ordered:
        section = flat_mut[after_idx][0]
        new_line = [-1, 1, ins.module_class,
                    list(ins.yaml_args) if not ins.module_kwargs
                    else [*ins.yaml_args, dict(ins.module_kwargs)]]
        # Insert AFTER after_idx → position after_idx + 1 in flat
        flat_mut.insert(after_idx + 1, (section, -1, new_line))

    # Split flat_mut back into sections. We preserved 'section' tags on each
    # tuple, including newly inserted lines.
    new_backbone = [line for sec, _, line in flat_mut if sec == "backbone"]
    new_neck     = [line for sec, _, line in flat_mut if sec == "neck"]
    new_head     = [line for sec, _, line in flat_mut if sec == "head"]

    custom = _reassemble(new_backbone, new_neck, new_head)

    # Preserve non-layer keys (nc, scales, depth_multiple, etc.) from base
    for k, v in base_yaml.items():
        if k not in ("backbone", "head"):
            custom[k] = v

    # Record base positions where insertions occurred, sorted ascending
    inserted_positions = sorted({after_idx for after_idx, _ in resolved})
    # Multiple insertions at same after_idx → multiple new lines, but same
    # base position. Need to count them for layer_map offsetting.
    base_pos_counts: dict[int, int] = {}
    for after_idx, _ in resolved:
        base_pos_counts[after_idx] = base_pos_counts.get(after_idx, 0) + 1

    # Flatten into a list where each base position appears once per insertion
    # at that position, sorted ascending.
    flat_positions: list[int] = []
    for pos in sorted(base_pos_counts):
        flat_positions.extend([pos] * base_pos_counts[pos])

    return custom, InsertionRecord(inserted_base_positions=flat_positions)


# ──────────────────────────────────────────────────────────────────────────────
# Low level — layer_map computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_layer_map(
    record: InsertionRecord,
    orig_total_layers: int,
    skip_head: bool = True,
    head_start_orig_idx: int | None = None,
    paramless_orig_indices: set[int] | None = None,
    detect_orig_idx: int | None = None,
) -> dict[int, int]:
    """Build orig_idx → custom_idx mapping from insertion bookkeeping.

    For every insertion at base position `p`, all custom indices > p shift
    by +1. So custom_idx(orig_idx=k) = k + (# insertions with pos ≤ k).

    Skips:
      - `detect_orig_idx` if skip_head is True (class-count-dependent shapes)
      - every index in `paramless_orig_indices` (Concat, Upsample, nn.Identity)
        — transferring them is a no-op but including them in strict mode
        causes spurious failures

    Arguments:
      record: from generate_custom_yaml
      orig_total_layers: len(orig_model.model.model)
      skip_head: whether to exclude the Detect layer from transfer
      head_start_orig_idx: currently unused (reserved for "skip entire head")
      paramless_orig_indices: layers with empty state_dict; usually autodiscovered
        by the caller, but can be passed explicitly
      detect_orig_idx: the Detect layer's index in orig; required iff skip_head
    """
    skip = set(paramless_orig_indices or ())
    if skip_head:
        if detect_orig_idx is None:
            # Convention: Detect is the last layer
            detect_orig_idx = orig_total_layers - 1
        skip.add(detect_orig_idx)

    positions = sorted(record.inserted_base_positions)

    def offset(orig_idx: int) -> int:
        # number of insertions with position ≤ orig_idx
        return sum(1 for p in positions if p <= orig_idx)

    return {
        orig_idx: orig_idx + offset(orig_idx)
        for orig_idx in range(orig_total_layers)
        if orig_idx not in skip
    }


def discover_paramless_layers(model) -> set[int]:
    """Return indices of layers with empty state_dict. Caller passes this to
    compute_layer_map to avoid false strict-mode failures on Concat/Upsample."""
    out = set()
    for i, layer in enumerate(model.model):
        if not layer.state_dict():
            out.add(i)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Low level — weight transfer
# ──────────────────────────────────────────────────────────────────────────────

def transfer_weights(
    orig_model,
    custom_model,
    layer_map: dict[int, int],
    strict: bool = True,
) -> int:
    """Copy matching tensors from orig_model to custom_model per layer_map.

    Returns total number of tensors transferred.

    When strict=True: every entry in layer_map MUST transfer ≥1 tensor.
    If any entry transfers 0 (shape mismatch, wrong class, etc.), raises.

    Rules:
      - strict=False on load_state_dict always (custom model has extra keys).
      - Shape guard — prevents mismatched tensors from loading silently.
      - strict mode here is about the layer_map integrity, not load_state_dict.
    """
    total = 0
    failures: list[str] = []

    for orig_idx, custom_idx in layer_map.items():
        orig_layer   = orig_model.model[orig_idx]
        custom_layer = custom_model.model[custom_idx]

        orig_sd   = orig_layer.state_dict()
        custom_sd = custom_layer.state_dict()

        matched = {
            k: v for k, v in orig_sd.items()
            if k in custom_sd and custom_sd[k].shape == v.shape
        }

        if matched:
            custom_layer.load_state_dict(matched, strict=False)
            total += len(matched)
        else:
            failures.append(
                f"orig[{orig_idx}]={orig_layer.__class__.__name__} → "
                f"custom[{custom_idx}]={custom_layer.__class__.__name__}: "
                f"no shape-matched tensors (orig keys: {list(orig_sd)[:3]}..., "
                f"custom keys: {list(custom_sd)[:3]}...)"
            )

    if strict and failures:
        raise RuntimeError(
            f"transfer_weights strict mode: {len(failures)} layer_map entries "
            f"transferred 0 tensors. This usually means the layer_map is "
            f"misaligned (wrong base index, wrong scope, or class mismatch).\n"
            f"Details:\n  " + "\n  ".join(failures)
        )

    return total


# ──────────────────────────────────────────────────────────────────────────────
# Low level — force_lazy_build + Stage 2 callback
# ──────────────────────────────────────────────────────────────────────────────

def force_lazy_build(model, imgsz: int = 640) -> None:
    """Run one forward pass so every Lazy* wrapper builds its inner module.

    Must be called BEFORE model.train(...) — Ultralytics' build_optimizer()
    captures model.parameters() before the first real forward, so lazy
    modules built during training won't have their params in the optimizer.

    Moves the model to GPU first if available and it's still on CPU, so
    the lazy inner modules land on the correct device (Lazy* wrappers call
    inner_module.to(x.device) at build time).
    """
    import torch

    if torch.cuda.is_available() and next(model.parameters()).device.type == "cpu":
        model.cuda()

    device = next(model.parameters()).device
    was_training = model.training

    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, imgsz, imgsz, device=device))
    finally:
        model.train(was_training)


def register_stage2_callback(
    yolo_instance,
    pretrained_pt: str,
    layer_map: dict[int, int],
    strict: bool = True,
) -> None:
    """Register an on_train_epoch_start hook that re-transfers weights before
    epoch 0 starts. Necessary because Ultralytics' trainer may re-initialise
    some layers during its own setup, overwriting Stage 1 transfers.

    layer_map and pretrained_pt are captured in the closure; trainer.model
    is resolved at call time (handles DDP unwrap).
    """
    from ultralytics import YOLO

    def on_train_epoch_start(trainer):
        # Ultralytics fires on_train_epoch_start at the BEGINNING of each epoch,
        # so trainer.epoch == 0 means "just before any training step runs".
        if trainer.epoch != 0:
            return
        the_model = trainer.model
        if hasattr(the_model, "module"):      # unwrap DDP
            the_model = the_model.module
        orig = YOLO(pretrained_pt)
        n = transfer_weights(orig.model, the_model, layer_map, strict=strict)
        print(f"[weight_transfer] Stage 2 (callback): transferred {n} tensors "
              f"from {pretrained_pt}")
        del orig

    yolo_instance.add_callback("on_train_epoch_start", on_train_epoch_start)


# ──────────────────────────────────────────────────────────────────────────────
# Mid level — the C entry point
# ──────────────────────────────────────────────────────────────────────────────

def build_custom_model_with_injection(
    base_weights: str,
    spec: dict,
    imgsz: int,
) -> "YOLO":
    """End-to-end: read spec → generate YAML → build model → Stage 1 transfer
    → force_lazy_build → register Stage 2 callback → return ready-to-train YOLO.

    Arguments:
      base_weights: path to the pretrained .pt to inject into and transfer from
      spec: loaded ARCH_INJECTION_SPEC dict (schema in arch_spec.schema.json)
      imgsz: image size used for force_lazy_build's dummy forward

    Returns a YOLO instance. Caller just needs to call .train(..., pretrained=False).

    Raises NotImplementedError for mode='full_yaml' (v1.8+).
    """
    # Validate mode BEFORE importing ultralytics, so environments without
    # the dependency can still test error paths.
    mode = spec.get("mode")
    if mode == "full_yaml":
        raise NotImplementedError(
            "mode='full_yaml' is reserved for v1.8+. Use mode='insertions' in v1.7."
        )
    if mode != "insertions":
        raise ValueError(f"Unknown spec.mode: {mode!r}")

    from ultralytics import YOLO

    strict = spec.get("strict", True)
    insertions = [Insertion.from_dict(d) for d in spec["insertions"]]

    # 1. Load base model and its yaml
    print(f"[weight_transfer] Loading base weights: {base_weights}")
    orig = YOLO(base_weights)
    base_yaml = parse_base_yaml(base_weights)

    # 2. Generate custom yaml
    custom_yaml_dict, record = generate_custom_yaml(base_yaml, insertions)

    # Write custom yaml to disk so YOLO(...) can read it and the file is
    # part of the git commit for reproducibility
    yaml_out = pathlib.Path("exp_arch.yaml")
    import yaml as _yaml
    yaml_out.write_text(_yaml.safe_dump(custom_yaml_dict, sort_keys=False))
    print(f"[weight_transfer] Generated custom YAML: {yaml_out} "
          f"(+{len(record.inserted_base_positions)} insertions)")

    # 3. Build custom model from yaml
    custom = YOLO(str(yaml_out))

    # 4. Compute layer_map
    paramless = discover_paramless_layers(orig.model)
    orig_total = len(orig.model.model)
    # Detect index: convention is the last layer. Could be more robust
    # by scanning for Detect-family class names.
    detect_idx = orig_total - 1

    layer_map = compute_layer_map(
        record,
        orig_total_layers=orig_total,
        skip_head=True,
        paramless_orig_indices=paramless,
        detect_orig_idx=detect_idx,
    )
    print(f"[weight_transfer] Computed layer_map: {len(layer_map)} pairs "
          f"(skipped {len(paramless) + 1}: Detect + param-less)")

    # 5. Stage 1 transfer
    n = transfer_weights(orig.model, custom.model, layer_map, strict=strict)
    print(f"[weight_transfer] Stage 1: transferred {n} tensors from {base_weights}")
    del orig

    # 6. Force lazy build (also moves model to GPU if available)
    force_lazy_build(custom.model, imgsz=imgsz)
    # Count Lazy* submodules to report
    n_lazy = sum(1 for m in custom.model.modules()
                 if m.__class__.__name__.startswith("Lazy"))
    print(f"[weight_transfer] Force-built {n_lazy} lazy modules")

    # 7. Register Stage 2 callback
    register_stage2_callback(custom, base_weights, layer_map, strict=strict)
    print(f"[weight_transfer] Stage 2 callback registered")

    return custom


# ──────────────────────────────────────────────────────────────────────────────
# High level — v1.8+ placeholder
# ──────────────────────────────────────────────────────────────────────────────

def apply_yaml_spec(*args, **kwargs):
    """Reserved for v1.8. Will handle mode='full_yaml' specs."""
    raise NotImplementedError(
        "apply_yaml_spec is reserved for v1.8. v1.7 supports "
        "build_custom_model_with_injection with mode='insertions' only."
    )


# ──────────────────────────────────────────────────────────────────────────────
# v1.7.1 — Repair primitives
#
# When yaml_inject crashes in a way that's plausibly shape/channel mismatch
# (not a logic error in the module itself), autoresearch's Step 5.5 can use
# these to diagnose and generate an adapter-augmented spec.
#
# Design principles:
#   1. Never mutate the user's Insertion (paper-faithful config). Always
#      produce a NEW spec with extra adapter insertions layered on.
#   2. Runtime probe — don't rely on ultralytics internal attrs like .c2.
#      Register forward hooks, run one dummy forward, read real shapes.
#   3. Report what was changed so autoresearch can write a truthful
#      description in results.tsv.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ShapeInfo:
    """Shape observed at a point in the model during a dummy forward."""
    channels: int          # C in (B, C, H, W)
    height: int            # H (or first spatial dim for non-2d)
    width: int             # W (or second spatial dim, may equal height)

    @classmethod
    def from_tensor_shape(cls, shape) -> "ShapeInfo":
        # Handle (B, C, H, W) — the standard YOLO activation shape.
        # Sequences like lists/tuples (e.g. FPN multi-scale) are represented
        # by their first element — callers handle the sequence case.
        if len(shape) != 4:
            raise ValueError(
                f"Expected a 4D (B, C, H, W) tensor shape, got {tuple(shape)}. "
                f"Adapter generation only handles standard 2D conv activations."
            )
        return cls(channels=int(shape[1]),
                   height=int(shape[2]),
                   width=int(shape[3]))


def get_shape_at_index(model, target_idx: int, imgsz: int):
    """Run one dummy forward and capture the output shape of layer
    model.model[target_idx]. Returns a ShapeInfo for standard 4D activations.

    Relies on register_forward_hook — stable public PyTorch API, unlike
    ultralytics .c2 which has shifted across versions.
    """
    import torch
    captured: dict = {}

    def hook(_module, _inp, out):
        # If layer emits a list/tuple (some necks), take the first tensor
        if isinstance(out, (list, tuple)):
            captured["shape"] = out[0].shape
        else:
            captured["shape"] = out.shape

    layer = model.model[target_idx]
    h = layer.register_forward_hook(hook)

    # Move model to GPU if available so the probe runs on realistic device
    if torch.cuda.is_available() and next(model.parameters()).device.type == "cpu":
        model.cuda()
    device = next(model.parameters()).device

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, imgsz, imgsz, device=device))
    finally:
        model.train(was_training)
        h.remove()

    if "shape" not in captured:
        raise RuntimeError(
            f"Forward hook at index {target_idx} was never called. "
            f"The layer may be dead code — check that the YAML graph "
            f"actually routes through model.model[{target_idx}]."
        )
    return ShapeInfo.from_tensor_shape(captured["shape"])


def probe_module_io(module_class: str, input_shape: ShapeInfo,
                    yaml_args: list, module_kwargs: dict):
    """Instantiate a module via register_custom_modules and measure its
    input/output shape on a dummy tensor. Returns (in_shape, out_shape).

    Used during Tier-2 repair to decide what adapter is needed around
    a mismatching module.
    """
    import torch
    from ultralytics.nn.tasks import parse_model

    # Resolve class via the same registry parse_model uses
    cls = parse_model.__globals__.get(module_class)
    if cls is None:
        raise RuntimeError(
            f"probe_module_io: {module_class!r} not in parse_model globals. "
            f"Make sure register_custom_modules() has been called."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Call with YAML-style args; lazy wrappers tolerate any positional set.
    # parse_model passes yaml_args only (no c1 prepended) for unknown modules.
    try:
        m = cls(*yaml_args, **module_kwargs).to(device)
    except TypeError as e:
        raise RuntimeError(
            f"probe_module_io: cannot instantiate {module_class}"
            f"(*{yaml_args}, **{module_kwargs}): {e}"
        ) from e

    x = torch.zeros(1, input_shape.channels,
                    input_shape.height, input_shape.width,
                    device=device)

    m.eval()
    with torch.no_grad():
        try:
            y = m(x)
        except Exception as e:
            raise RuntimeError(
                f"probe_module_io: {module_class} forward() failed on input "
                f"shape (1,{input_shape.channels},{input_shape.height},"
                f"{input_shape.width}): {e}"
            ) from e

    if isinstance(y, (list, tuple)):
        y = y[0]

    out_shape = ShapeInfo.from_tensor_shape(y.shape)
    return input_shape, out_shape


@dataclass
class AdapterPlan:
    """Result of analysing a mismatch. pre_adapter / post_adapter are None
    when unneeded. reason is human-readable for results.tsv description."""
    pre_adapter: dict | None     # YAML line ready for insertion (or None)
    post_adapter: dict | None
    reason: str                  # e.g. "added 256→64 projection before LazyX"

    @property
    def needs_adaptation(self) -> bool:
        return self.pre_adapter is not None or self.post_adapter is not None


def plan_adapter(
    upstream: ShapeInfo,
    module_in_observed: ShapeInfo,
    module_out: ShapeInfo,
    downstream_expected: ShapeInfo,
) -> AdapterPlan:
    """Decide what adapters (if any) are needed so a module fits at an
    insertion point.

    Inputs:
      upstream            : shape coming OUT of the layer right before insertion
      module_in_observed  : shape the probed module ACCEPTS (what it was called with)
      module_out          : shape the probed module RETURNS given module_in_observed
      downstream_expected : shape the layer right after insertion EXPECTS

    Output: AdapterPlan. Channels get 1×1 Conv adapters; spatial size changes
    are flagged as un-adaptable (Tier-3+ territory) — these indicate the
    module fundamentally changes resolution, which needs YAML-level rethink
    not an adapter.

    Only channel mismatches are auto-repaired in v1.7.1. Spatial mismatches
    abort the plan and signal unfixable to the caller.
    """
    # Only 2D conv activations are supported
    if (upstream.height != module_in_observed.height
        or upstream.width  != module_in_observed.width):
        return AdapterPlan(
            None, None,
            f"spatial mismatch at pre-insertion: upstream "
            f"{upstream.height}x{upstream.width} vs module input "
            f"{module_in_observed.height}x{module_in_observed.width} "
            f"— not auto-adaptable",
        )
    if (module_out.height != downstream_expected.height
        or module_out.width  != downstream_expected.width):
        return AdapterPlan(
            None, None,
            f"spatial mismatch at post-insertion: module output "
            f"{module_out.height}x{module_out.width} vs downstream "
            f"{downstream_expected.height}x{downstream_expected.width} "
            f"— not auto-adaptable",
        )

    pre  = None
    post = None
    notes = []

    # Pre-adapter: upstream channels ≠ module in channels
    if upstream.channels != module_in_observed.channels:
        pre = _make_1x1_conv_line(
            out_channels=module_in_observed.channels
        )
        notes.append(f"pre-Conv {upstream.channels}→{module_in_observed.channels}")

    # Post-adapter: module out channels ≠ downstream expected channels
    if module_out.channels != downstream_expected.channels:
        post = _make_1x1_conv_line(
            out_channels=downstream_expected.channels
        )
        notes.append(f"post-Conv {module_out.channels}→{downstream_expected.channels}")

    reason = ", ".join(notes) if notes else "no adapter needed"
    return AdapterPlan(pre_adapter=pre, post_adapter=post, reason=reason)


def _make_1x1_conv_line(out_channels: int) -> dict:
    """Produce a YAML insertion entry for a 1×1 Conv adapter.

    Emits a dict in the Insertion schema shape — the caller layers it into
    an ARCH_INJECTION_SPEC alongside the original insertion. We use ultralytics'
    `Conv` (known module, parse_model will prepend c1 automatically), so
    yaml_args is just [out_channels, kernel_size=1, stride=1].
    """
    return {
        "module_class": "Conv",          # ultralytics built-in
        "position":     None,            # caller fills in — relative to the
                                         # wrapped module's insertion point
        "scope":        None,            # caller fills in
        "yaml_args":    [out_channels, 1, 1],   # out_ch, kernel=1, stride=1
        "module_kwargs": {},
    }


def extend_spec_with_adapters(
    original_spec: dict,
    adapter_plan: AdapterPlan,
    insertion_idx: int = 0,
) -> dict:
    """Given an original ARCH_INJECTION_SPEC and a plan, produce a new spec
    that inserts pre/post adapters around the original[insertion_idx].

    The new spec keeps mode='insertions' and just adds up to 2 extra entries.
    Pre/post adapters inherit the original's scope and anchor on the same
    base position, so generate_custom_yaml processes them all in one pass.
    """
    if not adapter_plan.needs_adaptation:
        return original_spec

    target = original_spec["insertions"][insertion_idx]

    new_insertions = []
    pre  = adapter_plan.pre_adapter
    post = adapter_plan.post_adapter

    if pre is not None:
        # Pre-adapter goes at the SAME position as the target module; since
        # multiple insertions at the same after_idx stack in order, and
        # generate_custom_yaml processes descending-index then definition-order,
        # ordering here determines whether pre or target ends up closer to
        # the insertion point.
        pre = {**pre, "position": target["position"], "scope": target["scope"]}
        new_insertions.append(pre)

    new_insertions.append(target)

    if post is not None:
        post = {**post, "position": target["position"], "scope": target["scope"]}
        new_insertions.append(post)

    return {
        **original_spec,
        "insertions": (
            original_spec["insertions"][:insertion_idx]
            + new_insertions
            + original_spec["insertions"][insertion_idx + 1:]
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# v1.7.1 — Crash classification
# ──────────────────────────────────────────────────────────────────────────────

# Regex patterns matched against stderr tail. First match wins.
# Categories:
#   tier1      — code bug; repair by editing custom_modules.py or train.py
#   tier2      — shape/channel mismatch; repair by adding adapter via plan_adapter
#   unfixable  — architectural; discard immediately
#   oom        — out of memory; handled by v1.6 OOM policy (halve BATCH_SIZE)
#   unknown    — fall back to discard (conservative)

import re as _re

_CRASH_PATTERNS: list[tuple[str, str]] = [
    # Tier 1 — code bugs
    (r"NameError: name '\w+' is not defined",                     "tier1_missing_register"),
    (r"ModuleNotFoundError: No module named '[^']+'",             "tier1_missing_import"),
    (r"TypeError: \w+\.__init__\(\) got an unexpected keyword",   "tier1_init_signature"),
    (r"TypeError: \w+\.__init__\(\) missing \d+ required",        "tier1_init_signature"),
    (r"SyntaxError: ",                                            "tier1_syntax"),
    (r"ImportError: ",                                            "tier1_missing_import"),
    # Tier 2 — shape mismatch (channel-level)
    (r"RuntimeError: Given groups=1, weight of size \[",          "tier2_shape_mismatch"),
    (r"RuntimeError: Expected \d+D (?:input|tensor), got \dD",    "tier2_shape_mismatch"),
    (r"RuntimeError: The size of tensor a \(\d+\) must match",    "tier2_shape_mismatch"),
    (r"RuntimeError: mat1 and mat2 shapes cannot be multiplied",  "tier2_shape_mismatch"),
    # OOM — handled by v1.6 path
    (r"torch\.cuda\.OutOfMemoryError",                            "oom"),
    (r"RuntimeError: CUDA out of memory",                         "oom"),
    # Unfixable — architectural
    (r"RuntimeError: transfer_weights strict mode:",              "unfixable_layer_map"),
    (r"RuntimeError: size mismatch for model\.\d+\.",             "unfixable_weight_transfer"),
    # dtype / device mismatch — usually signals 2D/3D conv confusion or similar
    (r"RuntimeError: Input type .* and weight type .* should be the same", "unfixable_dtype_device"),
    (r"RuntimeError: Expected all tensors to be on the same device",       "unfixable_dtype_device"),
]


def classify_crash(stderr_tail: str) -> str:
    """Return a category string for the crash, or 'unknown' if no pattern matched.

    Caller (autoresearch Step 5.5) dispatches on the prefix:
       tier1_*    → attempt code repair, re-run (short test)
       tier2_*    → attempt adapter insertion, re-run (short test)
       oom        → existing v1.6 OOM path (halve BATCH_SIZE)
       unfixable_*→ go to Step 7 discard
       unknown    → go to Step 7 discard (conservative default)
    """
    for pattern, category in _CRASH_PATTERNS:
        if _re.search(pattern, stderr_tail):
            return category
    return "unknown"


def loss_first_value_is_valid(run_log_text: str) -> tuple[bool, str]:
    """v1.7.1 — Short-test success check.

    Scan run.log for the first per-epoch loss line emitted by ultralytics
    during training. Ultralytics prints something like:

        Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  ...
        1/10     2.31G     0.8231     1.9021     1.2133    ...

    We look for the first numeric triple on a line matching that shape
    AND require all values to be finite positive (not NaN / not Inf / > 0).

    Returns (is_valid, reason).
    """
    import math

    # Match: epoch_marker like "1/10" (or "1/1000"),  then GPU_mem, then 3 floats
    # that should be the box/cls/dfl losses. Tolerant to extra whitespace.
    # Regex accepts numeric literals, nan, and inf so we can detect NaN losses
    # (we'd rather classify "1/10 ... nan nan nan" as invalid than as "no line
    # found" — the training is broken, not missing).
    _NUM = r"(?:[\d.eE+\-]+|nan|inf|-inf|NaN|Inf|-Inf)"
    pat = _re.compile(
        rf"^\s*\d+/\d+\s+\S+\s+({_NUM})\s+({_NUM})\s+({_NUM})",
        _re.MULTILINE,
    )
    m = pat.search(run_log_text)
    if not m:
        return False, "no per-epoch loss line found in run.log"

    try:
        vals = [float(g) for g in m.groups()]
    except ValueError:
        return False, f"could not parse loss values: {m.groups()!r}"

    # NaN / Inf check
    for v in vals:
        if math.isnan(v):
            return False, f"loss is NaN: {vals}"
        if math.isinf(v):
            return False, f"loss is Inf: {vals}"
        if v <= 0:
            return False, f"loss is non-positive: {vals}"

    return True, f"first loss triple OK: box={vals[0]:.4f} cls={vals[1]:.4f} dfl={vals[2]:.4f}"

