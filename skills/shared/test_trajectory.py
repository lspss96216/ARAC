"""test_trajectory.py — v1.13"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import trajectory as T


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(path, rows):
    """Write a minimal ultralytics-style results.csv."""
    header = "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/mAP50-95(B)"
    lines = [header]
    for ep, val_map, box, cls, dfl in rows:
        lines.append(f"{ep},{box},{cls},{dfl},{val_map}")
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def _make_points(val_maps, train_losses=None):
    """Build TrajectoryPoint list from val_map sequence."""
    if train_losses is None:
        train_losses = [1.5 - 0.05 * i for i in range(len(val_maps))]
    return [
        T.TrajectoryPoint(epoch=i + 1, val_map=v, train_loss=t)
        for i, (v, t) in enumerate(zip(val_maps, train_losses))
    ]


# ──────────────────────────────────────────────────────────────────────────────
# parse_results_csv
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_csv_basic():
    with tempfile.TemporaryDirectory() as d:
        csv_path = pathlib.Path(d) / "r.csv"
        _write_csv(csv_path, [
            (1, 0.20, 1.5, 1.2, 0.8),
            (2, 0.25, 1.3, 1.0, 0.7),
            (3, 0.28, 1.1, 0.9, 0.6),
        ])
        points = T.parse_results_csv(str(csv_path))
    assert len(points) == 3
    assert points[0].epoch == 1
    assert abs(points[0].val_map - 0.20) < 1e-9
    # train_loss is sum of three: 1.5 + 1.2 + 0.8 = 3.5
    assert abs(points[0].train_loss - 3.5) < 1e-9
    print("✓ test_parse_csv_basic")


def test_parse_csv_missing_returns_empty():
    points = T.parse_results_csv("/tmp/does_not_exist_v1_13.csv")
    assert points == []
    print("✓ test_parse_csv_missing_returns_empty")


def test_parse_csv_handles_whitespace_in_headers():
    """ultralytics sometimes has trailing whitespace in column names."""
    with tempfile.TemporaryDirectory() as d:
        csv_path = pathlib.Path(d) / "r.csv"
        # Force trailing space on header
        csv_path.write_text(
            " epoch , train/box_loss , train/cls_loss , train/dfl_loss , metrics/mAP50-95(B) \n"
            "1,1.5,1.2,0.8,0.20\n2,1.3,1.0,0.7,0.25\n3,1.1,0.9,0.6,0.28\n"
        )
        points = T.parse_results_csv(str(csv_path))
    assert len(points) == 3
    assert abs(points[0].val_map - 0.20) < 1e-9
    print("✓ test_parse_csv_handles_whitespace_in_headers")


def test_parse_csv_skips_malformed_rows():
    with tempfile.TemporaryDirectory() as d:
        csv_path = pathlib.Path(d) / "r.csv"
        csv_path.write_text(
            "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/mAP50-95(B)\n"
            "1,1.5,1.2,0.8,0.20\n"
            "garbage,malformed,row\n"
            "2,1.3,1.0,0.7,0.25\n"
        )
        points = T.parse_results_csv(str(csv_path))
    # Malformed row skipped; 2 valid rows survive
    assert len(points) == 2
    print("✓ test_parse_csv_skips_malformed_rows")


# ──────────────────────────────────────────────────────────────────────────────
# classify_shape — 6 canonical shapes
# ──────────────────────────────────────────────────────────────────────────────

def test_classify_empty_returns_flat():
    diag = T.classify_shape([])
    assert diag.shape == "flat_no_learning"
    assert diag.final_map == 0.0
    print("✓ test_classify_empty_returns_flat")


def test_classify_too_few_epochs_returns_monotonic():
    """Below min_epochs threshold (3), can't classify — assume budget cut."""
    points = _make_points([0.20, 0.22])   # only 2 epochs
    diag = T.classify_shape(points)
    assert diag.shape == "monotonic_climbing"
    assert "Only 2 epoch" in diag.diagnosis
    print("✓ test_classify_too_few_epochs_returns_monotonic")


def test_classify_flat_no_learning():
    """val_mAP barely moves → not learning."""
    # 5 epochs all near 0.20 (swing < 1% of mean)
    points = _make_points([0.200, 0.201, 0.199, 0.200, 0.2005])
    diag = T.classify_shape(points)
    assert diag.shape == "flat_no_learning"
    assert "not learning" in diag.diagnosis
    print("✓ test_classify_flat_no_learning")


def test_classify_early_collapse():
    """Peak early then crash."""
    # Peak at epoch 4 (0.30), drop to 0.10 by end
    points = _make_points([0.20, 0.25, 0.28, 0.30, 0.20, 0.15, 0.12, 0.10])
    diag = T.classify_shape(points)
    assert diag.shape == "early_collapse"
    assert "collapsed" in diag.diagnosis or "collapse" in diag.diagnosis
    print("✓ test_classify_early_collapse")


def test_classify_oscillating():
    """Late-half swings >= 5% of mean."""
    # Mean ~0.27, late-half swings 0.22-0.30 (swing 0.08, > 5% of 0.27 = 0.0135)
    points = _make_points([0.20, 0.24, 0.27, 0.28, 0.22, 0.30, 0.23, 0.29])
    diag = T.classify_shape(points)
    assert diag.shape == "oscillating"
    assert "oscillat" in diag.diagnosis.lower()
    print("✓ test_classify_oscillating")


def test_classify_train_val_diverge():
    """train_loss drops a lot; val_mAP plateaus early."""
    # train_loss: 5.0 → 1.0 (80% drop). val_mAP: plateau at 0.27 in late half
    val_maps = [0.20, 0.25, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27]
    train_losses = [5.0, 4.0, 3.0, 2.0, 1.5, 1.2, 1.1, 1.0]
    points = _make_points(val_maps, train_losses)
    diag = T.classify_shape(points)
    assert diag.shape == "train_val_diverge"
    assert "overfit" in diag.diagnosis.lower()
    print("✓ test_classify_train_val_diverge")


def test_classify_monotonic_climbing():
    """Last third still climbing — budget-limited.
    Use 12 epochs so last-third (4 epochs) has a clear climb."""
    points = _make_points([0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.32])
    diag = T.classify_shape(points)
    assert diag.shape == "monotonic_climbing"
    assert "monotonic" in diag.diagnosis.lower()
    print("✓ test_classify_monotonic_climbing")


def test_classify_converged_above_baseline():
    """Plateau at end, final >= baseline."""
    # 0.20 → 0.29 with last third stable at 0.29; baseline=0.27.
    # train_loss exponential decay (real-world shape): 1.5, 1.0, 0.7, 0.55, 0.50, 0.49, 0.49, 0.49
    val_maps = [0.20, 0.24, 0.27, 0.28, 0.29, 0.29, 0.29, 0.29]
    train_losses = [1.5, 1.0, 0.7, 0.55, 0.50, 0.49, 0.49, 0.49]
    points = _make_points(val_maps, train_losses)
    diag = T.classify_shape(points, baseline_final_map=0.27)
    assert diag.shape == "converged_above_baseline"
    assert "keep candidate" in diag.diagnosis
    print("✓ test_classify_converged_above_baseline")


def test_classify_converged_below_baseline():
    """Plateau at end, final < baseline. Same exponential train_loss decay."""
    val_maps = [0.20, 0.22, 0.23, 0.24, 0.24, 0.24, 0.24, 0.24]
    train_losses = [1.5, 1.0, 0.8, 0.65, 0.55, 0.51, 0.50, 0.50]
    points = _make_points(val_maps, train_losses)
    diag = T.classify_shape(points, baseline_final_map=0.28)
    assert diag.shape == "converged_below_baseline"
    assert "under-performs" in diag.diagnosis
    print("✓ test_classify_converged_below_baseline")


def test_classify_no_baseline_defaults_to_below():
    """If baseline omitted (e.g. Loop 0 itself), converged shape lacks split — defaults to 'below'."""
    val_maps = [0.20, 0.22, 0.23, 0.24, 0.24, 0.24, 0.24, 0.24]
    train_losses = [1.5, 1.0, 0.8, 0.65, 0.55, 0.51, 0.50, 0.50]
    points = _make_points(val_maps, train_losses)
    diag = T.classify_shape(points, baseline_final_map=None)
    assert diag.shape == "converged_below_baseline"
    assert "no baseline available" in diag.diagnosis
    print("✓ test_classify_no_baseline_defaults_to_below")


# ──────────────────────────────────────────────────────────────────────────────
# Diagnosis fields
# ──────────────────────────────────────────────────────────────────────────────

def test_diagnosis_records_peak_and_final():
    points = _make_points([0.20, 0.30, 0.28, 0.25, 0.24, 0.23, 0.22, 0.22])
    diag = T.classify_shape(points)
    assert abs(diag.peak_map - 0.30) < 1e-9
    assert diag.peak_epoch == 2
    assert diag.final_epoch == 8
    assert abs(diag.final_map - 0.22) < 1e-9
    print("✓ test_diagnosis_records_peak_and_final")


def test_canonical_shapes_set_complete():
    """Every shape returned by classify_shape must be in CANONICAL_SHAPES."""
    expected = {
        "flat_no_learning", "early_collapse", "oscillating",
        "train_val_diverge", "monotonic_climbing",
        "converged_above_baseline", "converged_below_baseline",
    }
    assert T.CANONICAL_SHAPES == expected
    print("✓ test_canonical_shapes_set_complete")


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────

def test_priority_flat_beats_others():
    """Flat-no-learning wins over what would be 'monotonic' or 'converged' calls."""
    # Tiny variations all near 0.20 (swing 0.001)
    points = _make_points([0.2000, 0.2002, 0.2001, 0.2003, 0.2001, 0.2002])
    diag = T.classify_shape(points)
    assert diag.shape == "flat_no_learning"
    print("✓ test_priority_flat_beats_others")


def test_priority_collapse_beats_oscillating():
    """Early peak then sustained drop is collapse, not oscillation, even if there's some noise."""
    # Peak at 4, drops sustained — not bouncing back
    points = _make_points([0.20, 0.27, 0.30, 0.32, 0.28, 0.22, 0.18, 0.15])
    diag = T.classify_shape(points)
    assert diag.shape == "early_collapse"
    print("✓ test_priority_collapse_beats_oscillating")


TESTS = [
    test_parse_csv_basic,
    test_parse_csv_missing_returns_empty,
    test_parse_csv_handles_whitespace_in_headers,
    test_parse_csv_skips_malformed_rows,
    test_classify_empty_returns_flat,
    test_classify_too_few_epochs_returns_monotonic,
    test_classify_flat_no_learning,
    test_classify_early_collapse,
    test_classify_oscillating,
    test_classify_train_val_diverge,
    test_classify_monotonic_climbing,
    test_classify_converged_above_baseline,
    test_classify_converged_below_baseline,
    test_classify_no_baseline_defaults_to_below,
    test_diagnosis_records_peak_and_final,
    test_canonical_shapes_set_complete,
    test_priority_flat_beats_others,
    test_priority_collapse_beats_oscillating,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\nall {len(TESTS)} tests passed")
