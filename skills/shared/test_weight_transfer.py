"""test_weight_transfer.py — unit tests that don't require ultralytics/torch.

Covers:
  - split_sections: backbone/neck/head split
  - generate_custom_yaml: insertions in every scope/position combo
  - compute_layer_map: offset math + paramless/Detect skipping
  - transfer_weights: strict mode per-entry failure
  - Insertion.from_dict: schema validation round-trip

Does NOT cover (needs Ultralytics):
  - parse_base_yaml (reads from .pt)
  - force_lazy_build (needs torch + GPU)
  - register_stage2_callback (needs YOLO instance)
  - build_custom_model_with_injection end-to-end (needs all of above)

Run: python3 test_weight_transfer.py
"""

import sys
import pathlib
from unittest.mock import MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import weight_transfer as wt


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — synthetic yaml dicts
# ──────────────────────────────────────────────────────────────────────────────

def make_base_yaml():
    """A minimal yolo-like yaml: 5 backbone layers, 3 neck, 1 head (Detect)."""
    return {
        "nc": 80,
        "scales": {"x": [1.0, 1.0, 1024]},
        "backbone": [
            [-1, 1, "Conv",  [64, 3, 2]],    # 0
            [-1, 1, "Conv",  [128, 3, 2]],   # 1
            [-1, 3, "C3",    [128]],         # 2
            [-1, 1, "Conv",  [256, 3, 2]],   # 3
            [-1, 1, "SPPF",  [256, 5]],      # 4
        ],
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],   # neck 0 (global 5)
            [[-1, 2], 1, "Concat", [1]],                    # neck 1 (global 6)
            [-1, 3, "C3", [256]],                           # neck 2 (global 7)
            [[-1], 1, "Detect", [80]],                      # head   (global 8)
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# split_sections
# ──────────────────────────────────────────────────────────────────────────────

def test_split_sections_basic():
    y = make_base_yaml()
    b, n, h = wt.split_sections(y)
    assert len(b) == 5, f"backbone should be 5, got {len(b)}"
    assert len(n) == 3, f"neck should be 3, got {len(n)}"
    assert len(h) == 1, f"head should be 1, got {len(h)}"
    assert h[0][2] == "Detect"
    print("✓ test_split_sections_basic")


def test_split_sections_no_detect():
    """A yaml with no Detect-family layer — entire 'head' list is neck."""
    y = make_base_yaml()
    y["head"] = [[-1, 1, "Conv", [256]]]
    b, n, h = wt.split_sections(y)
    assert len(n) == 1 and len(h) == 0
    print("✓ test_split_sections_no_detect")


# ──────────────────────────────────────────────────────────────────────────────
# generate_custom_yaml — single insertion
# ──────────────────────────────────────────────────────────────────────────────

def test_generate_after_class_backbone():
    """Insert LazyCBAM after every Conv in backbone. Base Conv at [0,1,3]."""
    base = make_base_yaml()
    insertions = [wt.Insertion.from_dict({
        "module_class": "LazyCBAM",
        "position": {"kind": "after_class", "class_name": "Conv"},
        "scope": "backbone",
        "yaml_args": [64],
    })]
    custom, record = wt.generate_custom_yaml(base, insertions)

    # Expect 3 insertions, so backbone grows from 5 → 8
    assert len(custom["backbone"]) == 8, f"got {len(custom['backbone'])}"
    # Inserted at base positions 0, 1, 3 → custom backbone positions 1, 3, 6
    assert custom["backbone"][0][2] == "Conv"
    assert custom["backbone"][1][2] == "LazyCBAM"
    assert custom["backbone"][2][2] == "Conv"
    assert custom["backbone"][3][2] == "LazyCBAM"
    assert custom["backbone"][4][2] == "C3"
    assert custom["backbone"][5][2] == "Conv"
    assert custom["backbone"][6][2] == "LazyCBAM"
    assert custom["backbone"][7][2] == "SPPF"

    # Head unchanged
    assert len(custom["head"]) == 4
    # Record tracks BASE positions: 0, 1, 3
    assert record.inserted_base_positions == [0, 1, 3]
    print("✓ test_generate_after_class_backbone")


def test_generate_at_index_within_scope():
    """Insert at base index 2 with scope='backbone' — valid."""
    base = make_base_yaml()
    insertions = [wt.Insertion.from_dict({
        "module_class": "LazyX",
        "position": {"kind": "at_index", "index": 2},
        "scope": "backbone",
        "yaml_args": [128],
    })]
    custom, record = wt.generate_custom_yaml(base, insertions)

    assert len(custom["backbone"]) == 6
    # at_index=2 inserts AFTER base index 2 → custom backbone[3]
    assert custom["backbone"][2][2] == "C3"
    assert custom["backbone"][3][2] == "LazyX"
    assert record.inserted_base_positions == [2]
    print("✓ test_generate_at_index_within_scope")


def test_generate_at_index_out_of_scope_raises():
    """at_index=6 with scope='backbone' (which is 0..4) must raise."""
    base = make_base_yaml()
    insertions = [wt.Insertion.from_dict({
        "module_class": "LazyX",
        "position": {"kind": "at_index", "index": 6},
        "scope": "backbone",
        "yaml_args": [],
    })]
    try:
        wt.generate_custom_yaml(base, insertions)
    except ValueError as e:
        assert "outside scope" in str(e), str(e)
        print("✓ test_generate_at_index_out_of_scope_raises")
        return
    raise AssertionError("expected ValueError for out-of-scope at_index")


def test_generate_after_class_no_matches_raises():
    """after_class='NoSuchLayer' → no matches → raise."""
    base = make_base_yaml()
    insertions = [wt.Insertion.from_dict({
        "module_class": "LazyX",
        "position": {"kind": "after_class", "class_name": "Ghost"},
        "scope": "backbone",
        "yaml_args": [],
    })]
    try:
        wt.generate_custom_yaml(base, insertions)
    except ValueError as e:
        assert "no 'Ghost' layer found" in str(e), str(e)
        print("✓ test_generate_after_class_no_matches_raises")
        return
    raise AssertionError("expected ValueError for no matches")


# ──────────────────────────────────────────────────────────────────────────────
# generate_custom_yaml — multi insertion
# ──────────────────────────────────────────────────────────────────────────────

def test_generate_multi_insertion_descending():
    """Two insertions at different base indices — processed descending so
    earlier one's base index is unaffected."""
    base = make_base_yaml()
    insertions = [
        wt.Insertion.from_dict({
            "module_class": "LazyA",
            "position": {"kind": "at_index", "index": 0},
            "scope": "backbone",
            "yaml_args": [],
        }),
        wt.Insertion.from_dict({
            "module_class": "LazyB",
            "position": {"kind": "at_index", "index": 3},
            "scope": "backbone",
            "yaml_args": [],
        }),
    ]
    custom, record = wt.generate_custom_yaml(base, insertions)

    # Base:    [Conv, Conv, C3, Conv, SPPF]
    # After:   [Conv, LazyA, Conv, C3, Conv, LazyB, SPPF]
    #          at_idx=0 inserts after 0 → custom[1]
    #          at_idx=3 inserts after 3 → custom[4] originally, but shift by
    #                                   earlier insertion puts it at custom[5]
    names = [l[2] for l in custom["backbone"]]
    assert names == ["Conv", "LazyA", "Conv", "C3", "Conv", "LazyB", "SPPF"], names
    assert record.inserted_base_positions == [0, 3]
    print("✓ test_generate_multi_insertion_descending")


def test_generate_insertion_in_neck():
    """Insert in neck scope — backbone unchanged."""
    base = make_base_yaml()
    insertions = [wt.Insertion.from_dict({
        "module_class": "LazyN",
        "position": {"kind": "after_class", "class_name": "Concat"},
        "scope": "neck",
        "yaml_args": [],
    })]
    custom, record = wt.generate_custom_yaml(base, insertions)

    assert len(custom["backbone"]) == 5
    # Neck was: [Upsample, Concat, C3]; insert after Concat → [Upsample, Concat, LazyN, C3]
    # Head's raw list (in yaml) = neck + head_detect = 4 → 5
    assert len(custom["head"]) == 5
    assert custom["head"][2][2] == "LazyN"
    # Base global position of Concat in neck: backbone(5) + neck_local(1) = 6
    assert record.inserted_base_positions == [6]
    print("✓ test_generate_insertion_in_neck")


# ──────────────────────────────────────────────────────────────────────────────
# compute_layer_map
# ──────────────────────────────────────────────────────────────────────────────

def test_layer_map_no_insertions():
    """No insertions → identity map minus skipped."""
    record = wt.InsertionRecord(inserted_base_positions=[])
    # 9 layers: backbone 0-4, neck 5-7, Detect 8
    paramless = {5, 6}   # Upsample, Concat
    lm = wt.compute_layer_map(
        record, orig_total_layers=9,
        skip_head=True, detect_orig_idx=8,
        paramless_orig_indices=paramless,
    )
    # Expected: {0:0, 1:1, 2:2, 3:3, 4:4, 7:7}  (skip 5,6 paramless; skip 8 Detect)
    assert lm == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 7: 7}, lm
    print("✓ test_layer_map_no_insertions")


def test_layer_map_with_insertions():
    """3 insertions at base positions [0,1,3] — offset per orig_idx."""
    record = wt.InsertionRecord(inserted_base_positions=[0, 1, 3])
    # orig_total: 9 layers (backbone 5 + neck 3 + detect 1)
    paramless = {5, 6}
    lm = wt.compute_layer_map(
        record, orig_total_layers=9,
        skip_head=True, detect_orig_idx=8,
        paramless_orig_indices=paramless,
    )
    # Offset formula: custom_idx = orig_idx + count(p ≤ orig_idx)
    # orig 0 → count(≤0) = 1 → 1
    # orig 1 → count(≤1) = 2 → 3
    # orig 2 → count(≤2) = 2 → 4
    # orig 3 → count(≤3) = 3 → 6
    # orig 4 → count(≤4) = 3 → 7
    # orig 7 → count(≤7) = 3 → 10
    expected = {0: 1, 1: 3, 2: 4, 3: 6, 4: 7, 7: 10}
    assert lm == expected, f"got {lm}, want {expected}"
    print("✓ test_layer_map_with_insertions")


def test_layer_map_skip_head_default():
    """detect_orig_idx=None + skip_head=True → default to last layer."""
    record = wt.InsertionRecord(inserted_base_positions=[])
    lm = wt.compute_layer_map(
        record, orig_total_layers=3, skip_head=True,
        paramless_orig_indices=set(),
    )
    # Layer 2 (last) is implicit Detect, should be skipped
    assert lm == {0: 0, 1: 1}, lm
    print("✓ test_layer_map_skip_head_default")


# ──────────────────────────────────────────────────────────────────────────────
# transfer_weights — strict mode
# ──────────────────────────────────────────────────────────────────────────────

class FakeLayer:
    """A minimal stand-in for an nn.Module with a state_dict."""
    def __init__(self, state_dict, class_name="Fake"):
        self._sd = dict(state_dict)
        self.__class__.__name__ = class_name

    def state_dict(self):
        return dict(self._sd)

    def load_state_dict(self, sd, strict=False):
        # Simulate load: update internal dict
        self._sd.update(sd)


class FakeModel:
    """A FakeModel has .model = list-like of FakeLayer."""
    def __init__(self, layers):
        self.model = layers


def _tensor(shape):
    """Fake tensor: an object with .shape."""
    m = MagicMock()
    m.shape = shape
    return m


def test_transfer_weights_all_match():
    orig = FakeModel([
        FakeLayer({"w": _tensor((64, 3, 3, 3)), "b": _tensor((64,))}),
        FakeLayer({"w": _tensor((128, 64, 3, 3))}),
    ])
    custom = FakeModel([
        FakeLayer({"w": _tensor((64, 3, 3, 3)), "b": _tensor((64,))}),
        FakeLayer({"w": _tensor((128, 64, 3, 3))}),
    ])
    n = wt.transfer_weights(orig, custom, {0: 0, 1: 1}, strict=True)
    assert n == 3
    print("✓ test_transfer_weights_all_match")


def test_transfer_weights_strict_raises_on_zero():
    """One layer_map entry has shape-mismatch → 0 tensors transferred → raise."""
    orig = FakeModel([
        FakeLayer({"w": _tensor((64, 3, 3, 3))}, "Conv"),
        FakeLayer({"w": _tensor((128, 64, 3, 3))}, "Conv"),
    ])
    # custom[1] has wrong shape → won't match
    custom = FakeModel([
        FakeLayer({"w": _tensor((64, 3, 3, 3))}, "Conv"),
        FakeLayer({"w": _tensor((999, 1, 1, 1))}, "LazyX"),
    ])
    try:
        wt.transfer_weights(orig, custom, {0: 0, 1: 1}, strict=True)
    except RuntimeError as e:
        assert "strict mode" in str(e)
        assert "0 tensors" in str(e)
        print("✓ test_transfer_weights_strict_raises_on_zero")
        return
    raise AssertionError("expected RuntimeError on zero-transfer entry")


def test_transfer_weights_non_strict_permits_zero():
    """With strict=False, zero transfers are just recorded, no raise."""
    orig = FakeModel([FakeLayer({"w": _tensor((64,))})])
    custom = FakeModel([FakeLayer({"w": _tensor((32,))})])   # mismatch
    n = wt.transfer_weights(orig, custom, {0: 0}, strict=False)
    assert n == 0
    print("✓ test_transfer_weights_non_strict_permits_zero")


# ──────────────────────────────────────────────────────────────────────────────
# Insertion.from_dict
# ──────────────────────────────────────────────────────────────────────────────

def test_insertion_from_dict_after_class():
    d = {
        "module_class": "LazyCBAM",
        "position": {"kind": "after_class", "class_name": "Conv"},
        "scope": "backbone",
        "yaml_args": [64],
        "module_kwargs": {"kernel_size": 7},
    }
    ins = wt.Insertion.from_dict(d)
    assert ins.module_class == "LazyCBAM"
    assert ins.position_kind == "after_class"
    assert ins.position_value == "Conv"
    assert ins.scope == "backbone"
    assert ins.yaml_args == [64]
    assert ins.module_kwargs == {"kernel_size": 7}
    print("✓ test_insertion_from_dict_after_class")


def test_insertion_from_dict_at_index():
    d = {
        "module_class": "LazyX",
        "position": {"kind": "at_index", "index": 5},
        "scope": "neck",
    }
    ins = wt.Insertion.from_dict(d)
    assert ins.position_kind == "at_index"
    assert ins.position_value == 5
    assert ins.yaml_args == []
    assert ins.module_kwargs == {}
    print("✓ test_insertion_from_dict_at_index")


def test_insertion_from_dict_unknown_kind():
    d = {
        "module_class": "X",
        "position": {"kind": "telepathy"},
        "scope": "all",
    }
    try:
        wt.Insertion.from_dict(d)
    except ValueError as e:
        assert "telepathy" in str(e)
        print("✓ test_insertion_from_dict_unknown_kind")
        return
    raise AssertionError("expected ValueError for unknown kind")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry (mid-level) — smoke test via mock
# ──────────────────────────────────────────────────────────────────────────────

def test_build_rejects_full_yaml_mode():
    """v1.7 must refuse mode='full_yaml'."""
    try:
        wt.build_custom_model_with_injection(
            "fake.pt", {"mode": "full_yaml", "custom_yaml_path": "x.yaml"},
            imgsz=640,
        )
    except NotImplementedError as e:
        assert "full_yaml" in str(e)
        print("✓ test_build_rejects_full_yaml_mode")
        return
    raise AssertionError("expected NotImplementedError for full_yaml")


def test_build_rejects_unknown_mode():
    try:
        wt.build_custom_model_with_injection(
            "fake.pt", {"mode": "teleport"},
            imgsz=640,
        )
    except ValueError as e:
        assert "Unknown spec.mode" in str(e)
        print("✓ test_build_rejects_unknown_mode")
        return
    raise AssertionError("expected ValueError for unknown mode")


# ──────────────────────────────────────────────────────────────────────────────
# v1.7.1 — Repair primitive tests (no torch/ultralytics deps)
# ──────────────────────────────────────────────────────────────────────────────

def test_classify_crash_tier1():
    """Common Tier 1 errors map to tier1_* categories."""
    assert wt.classify_crash(
        "  File \"train.py\", line 3, in <module>\n"
        "NameError: name 'LazyCBAM' is not defined"
    ) == "tier1_missing_register"

    assert wt.classify_crash(
        "ModuleNotFoundError: No module named 'my_block'"
    ) == "tier1_missing_import"

    assert wt.classify_crash(
        "TypeError: LazyCBAM.__init__() got an unexpected keyword argument 'foo'"
    ) == "tier1_init_signature"

    assert wt.classify_crash(
        "TypeError: LazyCBAM.__init__() missing 2 required positional arguments"
    ) == "tier1_init_signature"

    assert wt.classify_crash(
        "  File \"x.py\", line 5\n    def broken(:\nSyntaxError: invalid syntax"
    ) == "tier1_syntax"
    print("✓ test_classify_crash_tier1")


def test_classify_crash_tier2():
    """Shape-mismatch RuntimeErrors map to tier2_shape_mismatch."""
    assert wt.classify_crash(
        "RuntimeError: Given groups=1, weight of size [128, 256, 3, 3]"
    ) == "tier2_shape_mismatch"

    assert wt.classify_crash(
        "RuntimeError: The size of tensor a (64) must match the size of tensor b (32)"
    ) == "tier2_shape_mismatch"

    assert wt.classify_crash(
        "RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x5 and 7x3)"
    ) == "tier2_shape_mismatch"
    print("✓ test_classify_crash_tier2")


def test_classify_crash_oom():
    assert wt.classify_crash(
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate"
    ) == "oom"
    # Pre-exception raw text also classifies OK
    assert wt.classify_crash(
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    ) == "oom"
    print("✓ test_classify_crash_oom")


def test_classify_crash_unfixable():
    """Architectural errors go straight to discard."""
    assert wt.classify_crash(
        "RuntimeError: transfer_weights strict mode: 3 layer_map entries "
        "transferred 0 tensors."
    ) == "unfixable_layer_map"

    assert wt.classify_crash(
        "RuntimeError: size mismatch for model.12.cv1.conv.weight: "
        "copying a param with shape torch.Size([64])"
    ) == "unfixable_weight_transfer"

    assert wt.classify_crash(
        "RuntimeError: Input type (torch.cuda.FloatTensor) and weight type "
        "(torch.FloatTensor) should be the same"
    ) == "unfixable_dtype_device"
    print("✓ test_classify_crash_unfixable")


def test_classify_crash_unknown():
    """Uncategorised errors fall back to 'unknown' (autoresearch discards)."""
    assert wt.classify_crash("ZeroDivisionError: division by zero") == "unknown"
    assert wt.classify_crash("") == "unknown"
    print("✓ test_classify_crash_unknown")


def test_loss_valid_first_epoch():
    log = (
        "Starting training for 10 epochs...\n"
        "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n"
        "       1/10     2.31G     0.8231     1.9021     1.2133        128       1920: 100%|\n"
    )
    ok, msg = wt.loss_first_value_is_valid(log)
    assert ok, msg
    assert "box=0.8231" in msg
    print("✓ test_loss_valid_first_epoch")


def test_loss_nan_rejected():
    log = (
        "       1/10     2.31G     nan        1.9021     1.2133        128       1920: 100%|\n"
    )
    ok, msg = wt.loss_first_value_is_valid(log)
    assert not ok
    assert "NaN" in msg
    print("✓ test_loss_nan_rejected")


def test_loss_inf_rejected():
    log = (
        "       1/10     2.31G     inf        1.9021     1.2133        128       1920: 100%|\n"
    )
    ok, msg = wt.loss_first_value_is_valid(log)
    assert not ok
    assert "Inf" in msg
    print("✓ test_loss_inf_rejected")


def test_loss_no_line_in_log():
    ok, msg = wt.loss_first_value_is_valid("some unrelated log content")
    assert not ok
    assert "no per-epoch" in msg
    print("✓ test_loss_no_line_in_log")


def test_plan_adapter_no_mismatch():
    """When all shapes line up, plan says no adapter needed."""
    same = wt.ShapeInfo(channels=64, height=80, width=80)
    plan = wt.plan_adapter(same, same, same, same)
    assert not plan.needs_adaptation
    assert "no adapter" in plan.reason
    print("✓ test_plan_adapter_no_mismatch")


def test_plan_adapter_channel_pre_only():
    """Upstream 256 → module expects 64. Need pre-adapter 256→64."""
    upstream   = wt.ShapeInfo(channels=256, height=40, width=40)
    mod_in     = wt.ShapeInfo(channels=64,  height=40, width=40)
    mod_out    = wt.ShapeInfo(channels=64,  height=40, width=40)
    downstream = wt.ShapeInfo(channels=64,  height=40, width=40)
    plan = wt.plan_adapter(upstream, mod_in, mod_out, downstream)
    assert plan.needs_adaptation
    assert plan.pre_adapter is not None
    assert plan.post_adapter is None
    assert plan.pre_adapter["module_class"] == "Conv"
    assert plan.pre_adapter["yaml_args"] == [64, 1, 1]
    assert "pre-Conv 256→64" in plan.reason
    print("✓ test_plan_adapter_channel_pre_only")


def test_plan_adapter_channel_post_only():
    """Module 64→32, downstream expects 64. Need post-adapter 32→64."""
    upstream   = wt.ShapeInfo(channels=64, height=40, width=40)
    mod_in     = wt.ShapeInfo(channels=64, height=40, width=40)
    mod_out    = wt.ShapeInfo(channels=32, height=40, width=40)
    downstream = wt.ShapeInfo(channels=64, height=40, width=40)
    plan = wt.plan_adapter(upstream, mod_in, mod_out, downstream)
    assert plan.needs_adaptation
    assert plan.pre_adapter is None
    assert plan.post_adapter is not None
    assert plan.post_adapter["yaml_args"] == [64, 1, 1]
    assert "post-Conv 32→64" in plan.reason
    print("✓ test_plan_adapter_channel_post_only")


def test_plan_adapter_both():
    upstream   = wt.ShapeInfo(channels=256, height=40, width=40)
    mod_in     = wt.ShapeInfo(channels=64,  height=40, width=40)
    mod_out    = wt.ShapeInfo(channels=32,  height=40, width=40)
    downstream = wt.ShapeInfo(channels=256, height=40, width=40)
    plan = wt.plan_adapter(upstream, mod_in, mod_out, downstream)
    assert plan.needs_adaptation
    assert plan.pre_adapter is not None
    assert plan.post_adapter is not None
    assert "pre-Conv" in plan.reason and "post-Conv" in plan.reason
    print("✓ test_plan_adapter_both")


def test_plan_adapter_spatial_mismatch_aborts():
    """Spatial size change is Tier-3+ territory — plan must refuse."""
    upstream   = wt.ShapeInfo(channels=64, height=40, width=40)
    mod_in     = wt.ShapeInfo(channels=64, height=20, width=20)   # half!
    mod_out    = wt.ShapeInfo(channels=64, height=20, width=20)
    downstream = wt.ShapeInfo(channels=64, height=40, width=40)
    plan = wt.plan_adapter(upstream, mod_in, mod_out, downstream)
    assert not plan.needs_adaptation
    assert "spatial mismatch" in plan.reason
    print("✓ test_plan_adapter_spatial_mismatch_aborts")


def test_extend_spec_with_adapters_pre_target_post_ordering():
    """extend_spec_with_adapters preserves order: pre, target, post."""
    original = {
        "mode": "insertions",
        "insertions": [{
            "module_class": "LazyCBAM",
            "position": {"kind": "after_class", "class_name": "Conv"},
            "scope": "backbone",
            "yaml_args": [64],
            "module_kwargs": {},
        }],
    }
    plan = wt.AdapterPlan(
        pre_adapter=wt._make_1x1_conv_line(out_channels=64),
        post_adapter=wt._make_1x1_conv_line(out_channels=256),
        reason="test",
    )
    new_spec = wt.extend_spec_with_adapters(original, plan, insertion_idx=0)
    classes = [ins["module_class"] for ins in new_spec["insertions"]]
    assert classes == ["Conv", "LazyCBAM", "Conv"], classes
    # Pre and post inherit the target's position/scope
    assert new_spec["insertions"][0]["scope"] == "backbone"
    assert new_spec["insertions"][2]["scope"] == "backbone"
    print("✓ test_extend_spec_with_adapters_pre_target_post_ordering")


def test_extend_spec_noop_when_plan_empty():
    """If AdapterPlan.needs_adaptation is False, spec comes back unchanged."""
    original = {
        "mode": "insertions",
        "insertions": [{
            "module_class": "LazyX",
            "position": {"kind": "at_index", "index": 2},
            "scope": "backbone",
            "yaml_args": [],
            "module_kwargs": {},
        }],
    }
    empty_plan = wt.AdapterPlan(pre_adapter=None, post_adapter=None,
                                reason="no adapter needed")
    out = wt.extend_spec_with_adapters(original, empty_plan, 0)
    assert out == original
    print("✓ test_extend_spec_noop_when_plan_empty")


def test_shapeinfo_rejects_non_4d():
    try:
        wt.ShapeInfo.from_tensor_shape((1, 64, 80))   # only 3 dims
    except ValueError as e:
        assert "4D" in str(e)
        print("✓ test_shapeinfo_rejects_non_4d")
        return
    raise AssertionError("expected ValueError on 3D shape")


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# v1.7.7 — update_head_refs (Fix #13: head/neck absolute-ref shifting)
# ──────────────────────────────────────────────────────────────────────────────

def test_update_head_refs_no_insertions_passthrough():
    """No insertions → returns input unchanged."""
    lines = [[-1, 1, "Conv", [64]], [6, 1, "Concat", [1]]]
    out = wt.update_head_refs(lines, [])
    assert out == lines
    print("✓ test_update_head_refs_no_insertions_passthrough")


def test_update_head_refs_empty_section():
    """Empty list returns empty list."""
    assert wt.update_head_refs([], [3]) == []
    print("✓ test_update_head_refs_empty_section")


def test_update_head_refs_negative_refs_untouched():
    """from=-1, -2, etc. are layer-relative — never shift."""
    lines = [[-1, 1, "Conv", [64]], [-2, 1, "Conv", [128]]]
    out = wt.update_head_refs(lines, [3, 5, 8])
    assert out == lines
    print("✓ test_update_head_refs_negative_refs_untouched")


def test_update_head_refs_single_absolute_shifts():
    """Concat [-1, 6] with insertion at base index 3 → [-1, 7]."""
    lines = [[[-1, 6], 1, "Concat", [1]]]
    out = wt.update_head_refs(lines, [3])
    assert out == [[[-1, 7], 1, "Concat", [1]]]
    print("✓ test_update_head_refs_single_absolute_shifts")


def test_update_head_refs_multiple_insertions_compound():
    """3 insertions all before ref 6 → 6 + 3 = 9."""
    lines = [[6, 1, "Conv", [64]]]
    out = wt.update_head_refs(lines, [2, 3, 5])
    assert out == [[9, 1, "Conv", [64]]]
    print("✓ test_update_head_refs_multiple_insertions_compound")


def test_update_head_refs_insertion_at_ref_does_not_shift():
    """Insertion AT base index 6 is inserted AFTER 6 → produces new layer 7.
    A ref to base 6 still points at base 6 (unchanged), so no shift."""
    lines = [[6, 1, "Conv", [64]]]
    out = wt.update_head_refs(lines, [6])
    assert out == [[6, 1, "Conv", [64]]]
    print("✓ test_update_head_refs_insertion_at_ref_does_not_shift")


def test_update_head_refs_mixed_before_and_after():
    """Insertions [3, 6, 8] applied to ref 6: only 3 < 6 shifts → 6 + 1 = 7."""
    lines = [[6, 1, "Conv", [64]]]
    out = wt.update_head_refs(lines, [3, 6, 8])
    assert out == [[7, 1, "Conv", [64]]]
    print("✓ test_update_head_refs_mixed_before_and_after")


def test_update_head_refs_concat_three_way_mixed():
    """Concat with multiple absolute refs all shift independently."""
    lines = [[[4, 6, 9], 1, "Concat", [1]]]
    out = wt.update_head_refs(lines, [5])
    # ref 4: 5 not < 4, no shift → 4
    # ref 6: 5 < 6, +1 → 7
    # ref 9: 5 < 9, +1 → 10
    assert out == [[[4, 7, 10], 1, "Concat", [1]]]
    print("✓ test_update_head_refs_concat_three_way_mixed")


def test_update_head_refs_does_not_mutate_input():
    """Caller's list should remain unchanged (defensive copy)."""
    lines = [[6, 1, "Conv", [64]]]
    original = [list(line) for line in lines]
    _ = wt.update_head_refs(lines, [3])
    assert lines == original
    print("✓ test_update_head_refs_does_not_mutate_input")


def test_update_head_refs_passthrough_malformed_lines():
    """Lines that don't look like [from, ...] are passed through."""
    lines = [None, [], "not a list", [6, 1, "Conv", [64]]]
    out = wt.update_head_refs(lines, [3])
    assert out[0] is None
    assert out[1] == []
    assert out[2] == "not a list"
    assert out[3] == [7, 1, "Conv", [64]]
    print("✓ test_update_head_refs_passthrough_malformed_lines")


def test_generate_custom_yaml_shifts_head_concat():
    """Integration: insertion in backbone shifts head's Concat refs.
    This is the v1.7 → v1.7.6 silent failure case (#13 from review)."""
    # Minimal base: 4 backbone layers (Conv x2, C2f x1, SPPF x1), 2 head layers
    # head Concat[-1, 1] should still reference the original base layer 1
    # after we insert a CBAM after backbone layer 1.
    base = {
        "nc": 80,
        "backbone": [
            [-1, 1, "Conv",     [64, 3, 2]],   # base idx 0
            [-1, 1, "Conv",     [128, 3, 2]],  # base idx 1
            [-1, 1, "C2f",      [128, True]],  # base idx 2
            [-1, 1, "SPPF",     [256, 5]],     # base idx 3
        ],
        "head": [
            [[-1, 1], 1, "Concat", [1]],       # base idx 4 — refs backbone Conv at 1
            [-1, 1, "Detect",  [80]],          # base idx 5
        ],
    }
    insertion = wt.Insertion(
        module_class="LazyCBAM",
        position_kind="at_index",
        position_value=2,             # insert AFTER base idx 2 (C2f)
        scope="backbone",
        yaml_args=[128],
        module_kwargs={},
    )
    custom, record = wt.generate_custom_yaml(base, [insertion])

    # Backbone now has 5 entries (4 originals + 1 CBAM)
    assert len(custom["backbone"]) == 5
    # CBAM is at new index 3 (one past the C2f at original idx 2)
    assert custom["backbone"][3][2] == "LazyCBAM"

    # Head Concat ref to base idx 1 stays as 1 (insertion at 2 doesn't shift base 1)
    head_concat = custom["head"][0]
    assert head_concat[0] == [-1, 1], (
        f"Concat refs should remain [-1, 1] because insertion at base idx 2 "
        f"does not shift refs <= 2; got {head_concat[0]}"
    )
    print("✓ test_generate_custom_yaml_shifts_head_concat")


def test_generate_custom_yaml_shifts_head_concat_when_ref_after_insertion():
    """Same but insertion BEFORE the head ref — head ref must shift."""
    base = {
        "nc": 80,
        "backbone": [
            [-1, 1, "Conv",     [64, 3, 2]],   # base idx 0
            [-1, 1, "Conv",     [128, 3, 2]],  # base idx 1
            [-1, 1, "C2f",      [128, True]],  # base idx 2
            [-1, 1, "SPPF",     [256, 5]],     # base idx 3
        ],
        "head": [
            [[-1, 3], 1, "Concat", [1]],       # refs SPPF at base idx 3
            [-1, 1, "Detect",  [80]],
        ],
    }
    # Insert AFTER base idx 1 (between Conv and C2f) — ref to base 3 must
    # become ref to new index 4
    insertion = wt.Insertion(
        module_class="LazyCBAM",
        position_kind="at_index",
        position_value=1,
        scope="backbone",
        yaml_args=[128],
        module_kwargs={},
    )
    custom, _ = wt.generate_custom_yaml(base, [insertion])
    head_concat = custom["head"][0]
    assert head_concat[0] == [-1, 4], (
        f"Concat ref to base 3 should shift to 4 because insertion at base 1 "
        f"is < 3; got {head_concat[0]}"
    )
    print("✓ test_generate_custom_yaml_shifts_head_concat_when_ref_after_insertion")



TESTS = [
    test_split_sections_basic,
    test_split_sections_no_detect,
    test_generate_after_class_backbone,
    test_generate_at_index_within_scope,
    test_generate_at_index_out_of_scope_raises,
    test_generate_after_class_no_matches_raises,
    test_generate_multi_insertion_descending,
    test_generate_insertion_in_neck,
    test_layer_map_no_insertions,
    test_layer_map_with_insertions,
    test_layer_map_skip_head_default,
    test_transfer_weights_all_match,
    test_transfer_weights_strict_raises_on_zero,
    test_transfer_weights_non_strict_permits_zero,
    test_insertion_from_dict_after_class,
    test_insertion_from_dict_at_index,
    test_insertion_from_dict_unknown_kind,
    test_build_rejects_full_yaml_mode,
    test_build_rejects_unknown_mode,
    # v1.7.1 — repair primitives
    test_classify_crash_tier1,
    test_classify_crash_tier2,
    test_classify_crash_oom,
    test_classify_crash_unfixable,
    test_classify_crash_unknown,
    test_loss_valid_first_epoch,
    test_loss_nan_rejected,
    test_loss_inf_rejected,
    test_loss_no_line_in_log,
    test_plan_adapter_no_mismatch,
    test_plan_adapter_channel_pre_only,
    test_plan_adapter_channel_post_only,
    test_plan_adapter_both,
    test_plan_adapter_spatial_mismatch_aborts,
    test_extend_spec_with_adapters_pre_target_post_ordering,
    test_extend_spec_noop_when_plan_empty,
    test_shapeinfo_rejects_non_4d,
    # v1.7.7 — update_head_refs (Fix #13)
    test_update_head_refs_no_insertions_passthrough,
    test_update_head_refs_empty_section,
    test_update_head_refs_negative_refs_untouched,
    test_update_head_refs_single_absolute_shifts,
    test_update_head_refs_multiple_insertions_compound,
    test_update_head_refs_insertion_at_ref_does_not_shift,
    test_update_head_refs_mixed_before_and_after,
    test_update_head_refs_concat_three_way_mixed,
    test_update_head_refs_does_not_mutate_input,
    test_update_head_refs_passthrough_malformed_lines,
    test_generate_custom_yaml_shifts_head_concat,
    test_generate_custom_yaml_shifts_head_concat_when_ref_after_insertion,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\nall {len(TESTS)} tests passed")
