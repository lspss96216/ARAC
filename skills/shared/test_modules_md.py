"""Quick sanity tests for modules_md parser."""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import modules_md as mm


# Sample modules.md in paper-finder's format
SAMPLE = """# Modules Registry

Task: Small object detection in drone imagery
Base model: YOLOv11-p2
Last updated: 2025-01-10
Total modules: 3

---

## Camera Motion Compensation

| Field | Value |
|-------|-------|
| Paper | BoT-SORT |
| arXiv | https://arxiv.org/abs/2206.14651 |
| Published | 2022 |
| Location | tracker |
| Complexity | medium |
| paper2code | yes |
| Status | pending |

### What it does
Uses ECC-based camera motion compensation to improve tracking under drone motion.

### Integration notes
Wrap the tracker config with CMC=True. Requires opencv-contrib-python.

## Adaptive Kalman Filter

| Field | Value |
|-------|-------|
| Paper | StrongSORT |
| arXiv | https://arxiv.org/abs/2202.13514 |
| Location | tracker |
| Complexity | low |
| paper2code | yes |
| Status | tested |

### What it does
NSA Kalman filter adjusts noise based on detection confidence.

## Small Object FPN

| Field | Value |
|-------|-------|
| Paper | RFLA |
| Location | neck |
| Complexity | high |
| paper2code | no (not on arXiv) |
| pdf_path | /data/papers/rfla.pdf |
| Status | pending |

### What it does
Receptive-field-based feature pyramid.
"""


def test_parse_basic():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(SAMPLE)
        path = f.name

    mods = mm.parse(path)
    assert len(mods) == 3, f"expected 3, got {len(mods)}"
    assert mods[0].name == "Camera Motion Compensation"
    assert mods[0].status == "pending"
    assert mods[0].complexity == "medium"
    assert mods[0].paper2code == "yes"
    assert mods[0].arxiv_url == "https://arxiv.org/abs/2206.14651"
    assert "ECC-based" in mods[0].sections["What it does"]
    assert "opencv-contrib-python" in mods[0].sections["Integration notes"]
    print("✓ test_parse_basic")


def test_count_pending():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(SAMPLE)
        path = f.name
    assert mm.count_pending(path) == 2, mm.count_pending(path)
    print("✓ test_count_pending")


def test_find_pending_sorted():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(SAMPLE)
        path = f.name
    pending = mm.find_pending(path)
    # medium should sort before high
    assert [m.name for m in pending] == [
        "Camera Motion Compensation",
        "Small Object FPN",
    ], [m.name for m in pending]
    print("✓ test_find_pending_sorted")


def test_list_pdf_paths():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(SAMPLE)
        path = f.name
    assert mm.list_pdf_paths(path) == {"/data/papers/rfla.pdf"}, mm.list_pdf_paths(path)
    print("✓ test_list_pdf_paths")


def test_update_status():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(SAMPLE)
        path = f.name
    assert mm.update_status(path, "Camera Motion Compensation", "injected")
    mods = mm.parse(path)
    cmc = next(m for m in mods if m.name == "Camera Motion Compensation")
    assert cmc.status == "injected", cmc.status
    # Other module unchanged
    other = next(m for m in mods if m.name == "Small Object FPN")
    assert other.status == "pending"
    # Returns False for unknown module
    assert not mm.update_status(path, "Nonexistent", "pending")
    print("✓ test_update_status")


def test_append_module():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(SAMPLE)
        path = f.name

    before = len(mm.parse(path))
    mm.append_module(path, {
        "name": "P2 Head",
        "fields": {
            "Paper": "YOLOv11-p2",
            "Location": "head",
            "Complexity": "low",
            "paper2code": "yes",
        },
        "sections": {
            "What it does": "Adds a p2 detection head for small objects.",
        },
    })
    after = mm.parse(path)
    assert len(after) == before + 1
    new = after[-1]
    assert new.name == "P2 Head"
    assert new.status == "pending"   # default
    assert new.complexity == "low"

    # Header "Total modules" should be updated
    text = pathlib.Path(path).read_text()
    assert f"Total modules: {before + 1}" in text, text[:300]
    print("✓ test_append_module")


def test_append_to_empty_file():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        path = f.name
    pathlib.Path(path).write_text("")   # empty

    mm.append_module(path, {
        "name": "First Module",
        "fields": {"Complexity": "low"},
    })
    mods = mm.parse(path)
    assert len(mods) == 1
    assert mods[0].name == "First Module"
    text = pathlib.Path(path).read_text()
    assert "# Modules Registry" in text
    assert "Total modules: 1" in text
    print("✓ test_append_to_empty_file")


def test_invalid_status_rejected():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(SAMPLE)
        path = f.name
    try:
        mm.update_status(path, "Camera Motion Compensation", "bogus")
        assert False, "should have raised"
    except ValueError:
        pass
    print("✓ test_invalid_status_rejected")


# ---------------------------------------------------------------------------
# v1.7 — integration_mode
# ---------------------------------------------------------------------------

def test_integration_mode_default():
    """Module without Integration mode field defaults to 'hook' (v1.6 behaviour)."""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(SAMPLE)
        path = f.name
    mods = mm.parse(path)
    assert mods[0].integration_mode == "hook", mods[0].integration_mode
    print("✓ test_integration_mode_default")


def test_integration_mode_yaml_inject():
    """Module with Integration mode: yaml_inject is parsed correctly."""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        path = f.name
    pathlib.Path(path).write_text("")
    mm.append_module(path, {
        "name": "CBAM Backbone",
        "fields": {
            "Location": "backbone",
            "Complexity": "medium",
            "paper2code": "yes",
            "Integration mode": "yaml_inject",
        },
    })
    mods = mm.parse(path)
    assert mods[0].integration_mode == "yaml_inject"
    print("✓ test_integration_mode_yaml_inject")


def test_integration_mode_unknown_warns_but_accepts():
    """Unknown Integration mode is warned about but not rejected.
    This is the warn-not-reject policy so forward-compatible.
    """
    import io, contextlib
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        path = f.name
    pathlib.Path(path).write_text("")

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        mm.append_module(path, {
            "name": "Future Module",
            "fields": {
                "Complexity": "low",
                "Integration mode": "telepathy",   # not in KNOWN_INTEGRATION_MODES
            },
        })
    stderr = buf.getvalue()
    assert "telepathy" in stderr, f"warning missing: {stderr!r}"
    assert "WARN" in stderr
    # Still appended
    mods = mm.parse(path)
    assert len(mods) == 1
    assert mods[0].integration_mode == "telepathy"   # returned as-is, not rewritten
    print("✓ test_integration_mode_unknown_warns_but_accepts")


def test_integration_mode_blank_is_default():
    """Empty string or whitespace-only → DEFAULT_INTEGRATION_MODE."""
    m = mm.Module(name="x", fields={"Integration mode": "   "})
    assert m.integration_mode == mm.DEFAULT_INTEGRATION_MODE
    m2 = mm.Module(name="y", fields={"Integration mode": ""})
    assert m2.integration_mode == mm.DEFAULT_INTEGRATION_MODE
    print("✓ test_integration_mode_blank_is_default")


# ---------------------------------------------------------------------------
# v1.9 — resource_impact field
# ---------------------------------------------------------------------------

def test_resource_impact_absent_returns_none():
    """No resource_impact field → property returns None (treated as 'unknown')."""
    m = mm.Module(name="x", fields={})
    assert m.resource_impact is None
    print("✓ test_resource_impact_absent_returns_none")


def test_resource_impact_blank_returns_none():
    """Blank or whitespace-only → None."""
    m1 = mm.Module(name="x", fields={"resource_impact": ""})
    m2 = mm.Module(name="y", fields={"resource_impact": "   "})
    assert m1.resource_impact is None
    assert m2.resource_impact is None
    print("✓ test_resource_impact_blank_returns_none")


def test_resource_impact_known_values():
    """Each KNOWN_RESOURCE_IMPACTS value parses cleanly."""
    for tag in mm.KNOWN_RESOURCE_IMPACTS:
        m = mm.Module(name="x", fields={"resource_impact": tag})
        assert m.resource_impact == tag, f"{tag} → {m.resource_impact}"
    print("✓ test_resource_impact_known_values")


def test_resource_impact_unknown_returned_as_is():
    """Unknown values returned as-is (warn-not-reject policy parallel to
    integration_mode). autoresearch decides what to do with them."""
    m = mm.Module(name="x", fields={"resource_impact": "vram_8x_unknown"})
    assert m.resource_impact == "vram_8x_unknown"
    assert m.resource_impact not in mm.KNOWN_RESOURCE_IMPACTS
    print("✓ test_resource_impact_unknown_returned_as_is")


def test_resource_impact_round_trip_through_append():
    """Field survives append + parse round-trip."""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        path = f.name
    pathlib.Path(path).write_text("")
    mm.append_module(path, {
        "name": "P2 Head",
        "fields": {
            "Location": "head",
            "Complexity": "high",
            "paper2code": "no",
            "Integration mode": "full_yaml",
            "resource_impact": "vram_4x",
        },
    })
    mods = mm.parse(path)
    assert mods[0].resource_impact == "vram_4x"
    print("✓ test_resource_impact_round_trip_through_append")


# ---------------------------------------------------------------------------
# v1.7.3 — preferred_locations secondary sort
# ---------------------------------------------------------------------------

# A fixture with several modules at the same complexity but different
# Location — so the secondary key actually has something to order.
_LOC_SAMPLE = """# Modules Registry

Task: test
Last updated: 2026-04-20
Total modules: 5

---

## M_head_low

| Field | Value |
|-------|-------|
| Location | head |
| Complexity | low |
| Status | pending |

## M_loss_low

| Field | Value |
|-------|-------|
| Location | loss |
| Complexity | low |
| Status | pending |

## M_backbone_low

| Field | Value |
|-------|-------|
| Location | backbone |
| Complexity | low |
| Status | pending |

## M_neck_low

| Field | Value |
|-------|-------|
| Location | neck |
| Complexity | low |
| Status | pending |

## M_backbone_medium

| Field | Value |
|-------|-------|
| Location | backbone |
| Complexity | medium |
| Status | pending |
"""


def _write_loc_sample():
    f = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
    f.write(_LOC_SAMPLE)
    f.close()
    return f.name


def test_preferred_locations_orders_within_complexity():
    """preferred_locations=[backbone, neck, head, loss] should order the
    four low-complexity modules accordingly, THEN put medium after all lows."""
    path = _write_loc_sample()
    pending = mm.find_pending(path, preferred_locations=["backbone", "neck", "head", "loss"])
    got = [m.name for m in pending]
    assert got == [
        "M_backbone_low",     # low, backbone (rank 0)
        "M_neck_low",         # low, neck (rank 1)
        "M_head_low",         # low, head (rank 2)
        "M_loss_low",         # low, loss (rank 3)
        "M_backbone_medium",  # medium, backbone — still after all lows
    ], got
    print("✓ test_preferred_locations_orders_within_complexity")


def test_preferred_locations_reversed():
    """Reversing preferred_locations reverses the secondary order."""
    path = _write_loc_sample()
    pending = mm.find_pending(path, preferred_locations=["loss", "head", "neck", "backbone"])
    got_low = [m.name for m in pending if m.complexity == "low"]
    assert got_low == ["M_loss_low", "M_head_low", "M_neck_low", "M_backbone_low"], got_low
    print("✓ test_preferred_locations_reversed")


def test_preferred_locations_partial_list_puts_unlisted_last():
    """Modules whose Location is not in preferred_locations sort after those
    that are, in write order (stable sort)."""
    path = _write_loc_sample()
    # Only rank backbone and neck; head and loss are unlisted → rank 2 each
    pending = mm.find_pending(path, preferred_locations=["backbone", "neck"])
    got_low = [m.name for m in pending if m.complexity == "low"]
    # Listed ones come first in preference order, then unlisted in write order
    # (write order: head, loss, backbone, neck → unlisted slice is head, loss)
    assert got_low[:2] == ["M_backbone_low", "M_neck_low"], got_low
    assert set(got_low[2:]) == {"M_head_low", "M_loss_low"}, got_low[2:]
    print("✓ test_preferred_locations_partial_list_puts_unlisted_last")


def test_preferred_locations_none_preserves_v172_behaviour():
    """preferred_locations=None → same as v1.7.2: single key on complexity,
    write order preserved within tie."""
    path = _write_loc_sample()
    pending = mm.find_pending(path)   # preferred_locations defaults to None
    got_low = [m.name for m in pending if m.complexity == "low"]
    # Write order of low modules: head, loss, backbone, neck
    assert got_low == ["M_head_low", "M_loss_low", "M_backbone_low", "M_neck_low"], got_low
    print("✓ test_preferred_locations_none_preserves_v172_behaviour")


def test_preferred_locations_case_insensitive():
    """Location field values and preferred_locations entries are compared
    case-insensitively (paper-finder sometimes writes 'Backbone')."""
    path = _write_loc_sample()
    pending = mm.find_pending(path, preferred_locations=["BACKBONE", "Neck"])
    got_low = [m.name for m in pending if m.complexity == "low"]
    assert got_low[0] == "M_backbone_low", got_low
    assert got_low[1] == "M_neck_low", got_low
    print("✓ test_preferred_locations_case_insensitive")


def test_preferred_locations_empty_list_same_as_none():
    """preferred_locations=[] behaves identically to None."""
    path = _write_loc_sample()
    a = [m.name for m in mm.find_pending(path, preferred_locations=[])]
    b = [m.name for m in mm.find_pending(path, preferred_locations=None)]
    assert a == b, (a, b)
    print("✓ test_preferred_locations_empty_list_same_as_none")


# ─── v1.12 B — effective_resource_impact + yaml_inject_scope ───────────────


def _module_with_notes(integration_mode: str, resource_impact: str, notes: str):
    """Helper — build a Module instance with given fields/sections."""
    return mm.Module(
        name="X",
        fields={"Integration mode": integration_mode, "resource_impact": resource_impact},
        sections={"Integration notes": notes},
    )


def test_yaml_inject_scope_extracts_backbone():
    """v1.12 B — scope: backbone parsed from yaml_inject Integration notes."""
    notes = """yaml_inject spec:
 - module_class: LazyCBAM
 - position: after_class: C3k2
 - scope: backbone
 - yaml_args: [256]"""
    m = _module_with_notes("yaml_inject", "vram_2x", notes)
    assert m.yaml_inject_scope == "backbone"
    print("✓ test_yaml_inject_scope_extracts_backbone")


def test_yaml_inject_scope_extracts_all_case_insensitive():
    """v1.12 B — scope value is lowercased; ALL → all."""
    notes = "yaml_inject spec:\n - scope: ALL\n - module_class: X"
    m = _module_with_notes("yaml_inject", "none", notes)
    assert m.yaml_inject_scope == "all"
    print("✓ test_yaml_inject_scope_extracts_all_case_insensitive")


def test_yaml_inject_scope_none_for_hook_mode():
    """v1.12 B — hook mode has no scope concept."""
    m = _module_with_notes("hook", "vram_2x", "scope: backbone")  # notes ignored
    assert m.yaml_inject_scope is None
    print("✓ test_yaml_inject_scope_none_for_hook_mode")


def test_yaml_inject_scope_none_when_missing():
    """v1.12 B — yaml_inject without scope: line returns None."""
    notes = "yaml_inject spec:\n - module_class: X\n - position: at_index: 5"
    m = _module_with_notes("yaml_inject", "vram_2x", notes)
    assert m.yaml_inject_scope is None
    print("✓ test_yaml_inject_scope_none_when_missing")


def test_effective_resource_impact_zero_param_scope_all_escalates():
    """v1.12 B core scenario — Loop 1 FlexSimAM bug.
    scope=all + resource_impact=none → effective=vram_2x."""
    notes = """yaml_inject spec:
 - module_class: LazySimAM
 - scope: all
 - position: after_class: C3k2"""
    m = _module_with_notes("yaml_inject", "none", notes)
    assert m.effective_resource_impact == "vram_2x"
    print("✓ test_effective_resource_impact_zero_param_scope_all_escalates")


def test_effective_resource_impact_vram_2x_scope_all_escalates_to_4x():
    """v1.12 B — vram_2x + scope=all → vram_4x."""
    notes = "yaml_inject spec:\n - scope: all\n - module_class: X"
    m = _module_with_notes("yaml_inject", "vram_2x", notes)
    assert m.effective_resource_impact == "vram_4x"
    print("✓ test_effective_resource_impact_vram_2x_scope_all_escalates_to_4x")


def test_effective_resource_impact_4x_scope_all_no_further_escalation():
    """v1.12 B — vram_4x is the ceiling; scope=all doesn't escalate further."""
    notes = "yaml_inject spec:\n - scope: all\n - module_class: X"
    m = _module_with_notes("yaml_inject", "vram_4x", notes)
    assert m.effective_resource_impact == "vram_4x"
    print("✓ test_effective_resource_impact_4x_scope_all_no_further_escalation")


def test_effective_resource_impact_backbone_scope_no_escalation():
    """v1.12 B — non-all scope (backbone/neck/head) leaves base unchanged."""
    notes = "yaml_inject spec:\n - scope: backbone\n - module_class: X"
    m = _module_with_notes("yaml_inject", "vram_2x", notes)
    assert m.effective_resource_impact == "vram_2x"
    print("✓ test_effective_resource_impact_backbone_scope_no_escalation")


def test_effective_resource_impact_hook_mode_uses_base():
    """v1.12 B — hook mode ignores scope concept entirely; effective == base."""
    m = _module_with_notes("hook", "vram_2x", "(some hook description)")
    assert m.effective_resource_impact == "vram_2x"
    print("✓ test_effective_resource_impact_hook_mode_uses_base")


def test_effective_resource_impact_returns_none_when_base_missing():
    """v1.12 B — no resource_impact field → effective is None."""
    m = mm.Module(
        name="X",
        fields={"Integration mode": "yaml_inject"},  # no resource_impact key
        sections={"Integration notes": "scope: all"},
    )
    assert m.effective_resource_impact is None
    print("✓ test_effective_resource_impact_returns_none_when_base_missing")


def test_effective_resource_impact_cpu_fallback_orthogonal_to_scope():
    """v1.12 B — cpu_fallback_risk is orthogonal axis; scope=all doesn't change it."""
    notes = "yaml_inject spec:\n - scope: all"
    m = _module_with_notes("yaml_inject", "cpu_fallback_risk", notes)
    assert m.effective_resource_impact == "cpu_fallback_risk"
    print("✓ test_effective_resource_impact_cpu_fallback_orthogonal_to_scope")


# ─── v1.13 — tuning + blocked statuses ─────────────────────────────────────


def test_tuning_status_accepted_by_update():
    """v1.13 — `tuning` is a valid status."""
    sample = """# Modules

## TestMod

| Field | Value |
|-------|-------|
| Paper | dummy |
| arXiv | 1234.56789 |
| Published | 2024 |
| Location | backbone |
| Complexity | low |
| paper2code | yes |
| Status | pending |

### What it does
test

### Integration notes
test
"""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(sample)
        path = f.name
    ok = mm.update_status(path, "TestMod", "tuning")
    assert ok
    mods = mm.parse(path)
    m = next(x for x in mods if x.name == "TestMod")
    assert m.status == "tuning"
    print("✓ test_tuning_status_accepted_by_update")


def test_blocked_status_accepted_by_update():
    """v1.13 — `blocked` is a valid status (was implicit state field in v1.12)."""
    sample = """# Modules

## TestMod

| Field | Value |
|-------|-------|
| Paper | dummy |
| arXiv | 1234.56789 |
| Published | 2024 |
| Location | backbone |
| Complexity | low |
| paper2code | yes |
| Status | pending |

### What it does
test

### Integration notes
test
"""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(sample)
        path = f.name
    ok = mm.update_status(path, "TestMod", "blocked")
    assert ok
    mods = mm.parse(path)
    m = next(x for x in mods if x.name == "TestMod")
    assert m.status == "blocked"
    print("✓ test_blocked_status_accepted_by_update")


def test_tuning_status_does_not_count_as_pending():
    """v1.13 — modules in `tuning` state should NOT appear in find_pending().
    Otherwise dispatch keeps re-picking the same module while it's mid-tuning."""
    sample = """# Modules

## TestPending

| Field | Value |
|-------|-------|
| Paper | dummy |
| arXiv | 1.1 |
| Published | 2024 |
| Location | backbone |
| Complexity | low |
| paper2code | yes |
| Status | pending |

### What it does
x

### Integration notes
x

## TestTuning

| Field | Value |
|-------|-------|
| Paper | dummy |
| arXiv | 1.2 |
| Published | 2024 |
| Location | backbone |
| Complexity | low |
| paper2code | yes |
| Status | tuning |

### What it does
x

### Integration notes
x

## TestBlocked

| Field | Value |
|-------|-------|
| Paper | dummy |
| arXiv | 1.3 |
| Published | 2024 |
| Location | backbone |
| Complexity | low |
| paper2code | yes |
| Status | blocked |

### What it does
x

### Integration notes
x
"""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(sample)
        path = f.name
    pending = mm.find_pending(path)
    names = [m.name for m in pending]
    assert "TestPending" in names
    assert "TestTuning" not in names
    assert "TestBlocked" not in names
    print("✓ test_tuning_status_does_not_count_as_pending")


if __name__ == "__main__":
    test_parse_basic()
    test_count_pending()
    test_find_pending_sorted()
    test_list_pdf_paths()
    test_update_status()
    test_append_module()
    test_append_to_empty_file()
    test_invalid_status_rejected()
    # v1.7
    test_integration_mode_default()
    test_integration_mode_yaml_inject()
    test_integration_mode_unknown_warns_but_accepts()
    test_integration_mode_blank_is_default()
    # v1.9 — resource_impact
    test_resource_impact_absent_returns_none()
    test_resource_impact_blank_returns_none()
    test_resource_impact_known_values()
    test_resource_impact_unknown_returned_as_is()
    test_resource_impact_round_trip_through_append()
    # v1.7.3
    test_preferred_locations_orders_within_complexity()
    test_preferred_locations_reversed()
    test_preferred_locations_partial_list_puts_unlisted_last()
    test_preferred_locations_none_preserves_v172_behaviour()
    test_preferred_locations_case_insensitive()
    test_preferred_locations_empty_list_same_as_none()
    # v1.12 — scope-dependent resource_impact
    test_yaml_inject_scope_extracts_backbone()
    test_yaml_inject_scope_extracts_all_case_insensitive()
    test_yaml_inject_scope_none_for_hook_mode()
    test_yaml_inject_scope_none_when_missing()
    test_effective_resource_impact_zero_param_scope_all_escalates()
    test_effective_resource_impact_vram_2x_scope_all_escalates_to_4x()
    test_effective_resource_impact_4x_scope_all_no_further_escalation()
    test_effective_resource_impact_backbone_scope_no_escalation()
    test_effective_resource_impact_hook_mode_uses_base()
    test_effective_resource_impact_returns_none_when_base_missing()
    test_effective_resource_impact_cpu_fallback_orthogonal_to_scope()
    # v1.13 — tuning + blocked statuses
    test_tuning_status_accepted_by_update()
    test_blocked_status_accepted_by_update()
    test_tuning_status_does_not_count_as_pending()
    print("\nall tests passed")
