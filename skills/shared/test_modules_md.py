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


if __name__ == "__main__":
    test_parse_basic()
    test_count_pending()
    test_find_pending_sorted()
    test_list_pdf_paths()
    test_update_status()
    test_append_module()
    test_append_to_empty_file()
    test_invalid_status_rejected()
    print("\nall tests passed")
