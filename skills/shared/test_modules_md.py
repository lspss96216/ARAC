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
    # v1.7.3
    test_preferred_locations_orders_within_complexity()
    test_preferred_locations_reversed()
    test_preferred_locations_partial_list_puts_unlisted_last()
    test_preferred_locations_none_preserves_v172_behaviour()
    test_preferred_locations_case_insensitive()
    test_preferred_locations_empty_list_same_as_none()
    print("\nall tests passed")
