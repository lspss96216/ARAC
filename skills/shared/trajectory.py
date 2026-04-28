"""trajectory.py — v1.13

Per-epoch trajectory parsing + shape classification for autoresearch.

Background: v1.6-v1.12.1 made keep/discard decisions on the LAST EPOCH's
val_mAP50_95 only. A module that's still climbing at TIME_BUDGET expiry
got the same verdict as a module that converged at epoch 3 — both judged
on their final value, no information about the curve shape. v1.13's
multi-attempt per-module loop needs trajectory shape so the next attempt
can adjust hyperparams meaningfully (e.g. "oscillating → halve LR",
"monotonic climbing → extend warmup").

This module is responsible for:
  1. Parsing ultralytics' results.csv into a list of (epoch, val_mAP, train_loss) tuples
  2. Classifying the curve into one of 6 canonical shapes
  3. Producing a human-readable diagnosis string for the agent to reason from

The agent does NOT decide hyperparam changes from rules in this module.
This module only emits SIGNALS (shape + diagnosis text). Agent reads
signal + paper recipe + tuning history and decides what to change.

Usage:

    from shared.trajectory import parse_results_csv, classify_shape

    points = parse_results_csv("runs/train/exp/results.csv")
    # points = [(1, 0.20, 1.5), (2, 0.24, 1.2), ...]

    diag = classify_shape(points, baseline_final_map=0.2850)
    # diag.shape       = "monotonic_climbing"
    # diag.diagnosis   = "val mAP rose monotonically from 0.20 → 0.27 ..."
    # diag.final_map   = 0.27
    # diag.peak_map    = 0.27
"""

from __future__ import annotations
import csv
import pathlib
import statistics
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory data
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    """One epoch's snapshot. ultralytics results.csv has many columns; we
    extract just the three we need for shape classification."""
    epoch: int
    val_map: float       # val_mAP50_95
    train_loss: float    # sum of box+cls+dfl train losses


@dataclass
class Diagnosis:
    """Output of classify_shape. Agent reads `shape` + `diagnosis` to
    reason about next attempt; `final_map`, `peak_map`, `peak_epoch`
    are for results.tsv recording.

    `shape` is one of:
      - monotonic_climbing      : val mAP rising at end → maybe budget-limited
      - converged_below_baseline: val mAP plateaued below baseline → real fail
      - converged_above_baseline: val mAP plateaued above baseline → success
      - oscillating             : val mAP swing >= 5% peak-to-peak in late epochs
      - early_collapse          : val mAP peaked early then dropped >= 5%
      - flat_no_learning        : val mAP variance < 1% over whole run
      - train_val_diverge       : train_loss drops, val mAP plateaus early
    """
    shape: str
    diagnosis: str
    final_map: float
    peak_map: float
    peak_epoch: int
    final_epoch: int


# ──────────────────────────────────────────────────────────────────────────────
# Parsing — ultralytics results.csv format
# ──────────────────────────────────────────────────────────────────────────────

def parse_results_csv(csv_path: str) -> list[TrajectoryPoint]:
    """Parse ultralytics' per-epoch results.csv into trajectory points.

    Expected columns include 'epoch', 'metrics/mAP50-95(B)', 'train/box_loss',
    'train/cls_loss', 'train/dfl_loss'. Sums the three losses for `train_loss`.

    Returns [] if file missing or unparseable. v1.13 callers handle empty list
    gracefully — typically by falling back to v1.12 last-epoch behaviour.
    """
    p = pathlib.Path(csv_path)
    if not p.exists():
        return []

    points: list[TrajectoryPoint] = []
    try:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip whitespace from keys (ultralytics has trailing spaces)
                row = {k.strip(): v for k, v in row.items()}
                try:
                    epoch = int(float(row["epoch"]))
                    val_map = float(row.get("metrics/mAP50-95(B)") or
                                     row.get("metrics/mAP50-95") or 0.0)
                    train_loss = (
                        float(row.get("train/box_loss") or 0.0)
                        + float(row.get("train/cls_loss") or 0.0)
                        + float(row.get("train/dfl_loss") or 0.0)
                    )
                except (KeyError, ValueError, TypeError):
                    # Malformed row — skip rather than crash. Caller can still
                    # work with the points we did get.
                    continue
                points.append(TrajectoryPoint(epoch=epoch, val_map=val_map,
                                              train_loss=train_loss))
    except Exception:
        return []   # corrupt file or csv error — treat as missing

    return points


# ──────────────────────────────────────────────────────────────────────────────
# Shape classification
# ──────────────────────────────────────────────────────────────────────────────

# Tunable thresholds — explicit so tests can verify and users can adjust if
# the defaults turn out wrong on their dataset. v1.13 picks conservative
# defaults; if real-run feedback shows misclassifications, raise/lower these
# rather than adding more rules.
_SHAPE_THRESHOLDS = {
    "monotonic_min_climb_pct":     0.05,   # last-1/3 epochs must climb >= 5% from start to count as monotonic
    "oscillating_swing_pct":       0.05,   # peak-to-peak swing in late half >= 5% of mean → oscillating
    "early_collapse_drop_pct":     0.05,   # peak followed by drop >= 5% → collapse
    "flat_total_swing_pct":        0.01,   # whole-run swing < 1% → no learning
    "diverge_train_drop_pct":      0.20,   # train_loss drops >= 20% from epoch 1 → counts as "training is happening"
    "diverge_val_plateau_pct":     0.02,   # ...but val_mAP swing in last half < 2% → val plateau
    "converged_late_swing_pct":    0.02,   # last-1/3 swing < 2% → "plateau" (converged)
    "min_epochs_for_classification": 3,    # need at least 3 epochs to classify; below that → "monotonic_climbing"
}


def classify_shape(points: list[TrajectoryPoint],
                   baseline_final_map: Optional[float] = None) -> Diagnosis:
    """Classify a trajectory into one of 6 shapes + emit a diagnosis string.

    The shapes are mutually exclusive and tested in priority order — first
    match wins. Priority (highest first):
        1. flat_no_learning      (whole-run variance tiny → useless run, save time)
        2. early_collapse        (peak then drop → instability is loud)
        3. oscillating           (late swings → LR likely too high)
        4. train_val_diverge     (train_loss falling but val_mAP flat → overfit)
        5. monotonic_climbing    (last-1/3 climbing → budget-limited)
        6. converged             (default — late plateau)
            ↳ then baseline-aware split:
                converged_above_baseline (≥ baseline → keep candidate)
                converged_below_baseline (< baseline → real fail, hyperparams won't help much)

    If `baseline_final_map` is None (e.g. Loop 0 baseline itself), the
    `converged` shape stays unsplit.
    """
    if not points:
        return Diagnosis(
            shape="flat_no_learning",
            diagnosis="No trajectory data available — results.csv missing or empty.",
            final_map=0.0, peak_map=0.0, peak_epoch=0, final_epoch=0,
        )

    n = len(points)
    final = points[-1]
    peak = max(points, key=lambda p: p.val_map)

    if n < _SHAPE_THRESHOLDS["min_epochs_for_classification"]:
        # Not enough data — call it monotonic_climbing (assume budget-limited).
        return Diagnosis(
            shape="monotonic_climbing",
            diagnosis=(f"Only {n} epoch(s) recorded — too few to classify shape. "
                       f"Final val_mAP={final.val_map:.4f}. Either training "
                       f"crashed early or TIME_BUDGET cut off before convergence."),
            final_map=final.val_map, peak_map=peak.val_map,
            peak_epoch=peak.epoch, final_epoch=final.epoch,
        )

    val_maps = [p.val_map for p in points]
    train_losses = [p.train_loss for p in points]

    overall_min, overall_max = min(val_maps), max(val_maps)
    overall_mean = statistics.mean(val_maps) if val_maps else 0.0
    overall_swing = overall_max - overall_min

    # Late half / last-third slices for plateau and climbing detection
    late_half = val_maps[n // 2:]
    last_third = val_maps[-(max(1, n // 3)):]
    first_third = val_maps[:max(1, n // 3)]

    late_swing = max(late_half) - min(late_half) if late_half else 0.0
    last_third_climb = (last_third[-1] - last_third[0]) if len(last_third) >= 2 else 0.0
    first_to_last_climb = (last_third[-1] - first_third[0]) if (last_third and first_third) else 0.0

    # ─── Priority 1: flat_no_learning ──────────────────────────────────────
    # Whole-run swing < 1% of mean → model isn't learning at all
    if overall_mean > 1e-6:
        flat_threshold = _SHAPE_THRESHOLDS["flat_total_swing_pct"] * overall_mean
        if overall_swing < flat_threshold:
            return Diagnosis(
                shape="flat_no_learning",
                diagnosis=(f"val mAP barely moved across {n} epochs "
                           f"(min={overall_min:.4f}, max={overall_max:.4f}, "
                           f"swing={overall_swing:.4f} < {flat_threshold:.4f}). "
                           f"Model is not learning — likely LR too low, "
                           f"frozen layers, or data pipeline broken."),
                final_map=final.val_map, peak_map=peak.val_map,
                peak_epoch=peak.epoch, final_epoch=final.epoch,
            )

    # ─── Priority 2: early_collapse ────────────────────────────────────────
    # Peak somewhere before epoch n*0.7, with subsequent drop >= 5% of peak
    if peak.epoch < points[int(n * 0.7)].epoch:
        post_peak_min = min(p.val_map for p in points if p.epoch >= peak.epoch)
        drop = peak.val_map - post_peak_min
        drop_threshold = _SHAPE_THRESHOLDS["early_collapse_drop_pct"] * peak.val_map
        if drop >= drop_threshold:
            return Diagnosis(
                shape="early_collapse",
                diagnosis=(f"val mAP peaked at {peak.val_map:.4f} (epoch {peak.epoch}) "
                           f"then collapsed to {post_peak_min:.4f} (drop of {drop:.4f} "
                           f">= {drop_threshold:.4f}). Final={final.val_map:.4f}. "
                           f"Indicates overfitting, NaN gradients, or LR schedule too aggressive."),
                final_map=final.val_map, peak_map=peak.val_map,
                peak_epoch=peak.epoch, final_epoch=final.epoch,
            )

    # ─── Priority 3: oscillating ───────────────────────────────────────────
    # Late-half swing >= 5% of mean AND non-monotonic (peak not near end).
    # The non-monotonic condition prevents misclassifying still-climbing
    # curves: a curve going 0.26 → 0.27 → 0.29 → 0.30 has late-swing 0.04
    # but it's monotonic, not oscillating. We check that the late-half max
    # appears at neither the first nor the last position — true oscillation
    # has the peak somewhere in the middle of the late half.
    if overall_mean > 1e-6:
        osc_threshold = _SHAPE_THRESHOLDS["oscillating_swing_pct"] * overall_mean
        if late_swing >= osc_threshold and len(late_half) >= 3:
            late_max_idx = late_half.index(max(late_half))
            late_min_idx = late_half.index(min(late_half))
            # True oscillation: max is interior OR (max-min sequence isn't monotonic)
            is_monotonic_climb = late_max_idx == len(late_half) - 1 and late_min_idx == 0
            is_monotonic_drop  = late_min_idx == len(late_half) - 1 and late_max_idx == 0
            if not (is_monotonic_climb or is_monotonic_drop):
                late_min, late_max = min(late_half), max(late_half)
                return Diagnosis(
                    shape="oscillating",
                    diagnosis=(f"val mAP oscillating in late half: range [{late_min:.4f}, "
                               f"{late_max:.4f}], swing {late_swing:.4f} >= "
                               f"{osc_threshold:.4f} ({_SHAPE_THRESHOLDS['oscillating_swing_pct']*100:.0f}% of mean), "
                               f"non-monotonic (peak at index {late_max_idx} of {len(late_half)} late epochs). "
                               f"Final={final.val_map:.4f}. Suggests LR too high or "
                               f"momentum/optimizer instability."),
                    final_map=final.val_map, peak_map=peak.val_map,
                    peak_epoch=peak.epoch, final_epoch=final.epoch,
                )

    # ─── Priority 4: train_val_diverge ─────────────────────────────────────
    # Classic overfitting signature:
    #   1. Train loss is STILL falling in second half (not just overall)
    #   2. Val mAP plateaued in second half (barely changed)
    # This separates true diverge from "model converged and train+val both
    # plateaued together" — that's healthy convergence, not overfit.
    if len(train_losses) >= 4 and len(val_maps) >= 4:
        sh_start = n // 2
        train_sh = train_losses[sh_start:]
        val_sh = val_maps[sh_start:]

        # Second-half train_loss still meaningfully dropping?
        if train_sh[0] > 1e-6:
            train_sh_drop_pct = (train_sh[0] - train_sh[-1]) / train_sh[0]
        else:
            train_sh_drop_pct = 0

        # Second-half val_mAP barely moving?
        val_sh_improvement = val_sh[-1] - val_sh[0]
        val_sh_swing = max(val_sh) - min(val_sh)
        relative_improvement = (val_sh_improvement / overall_mean
                                if overall_mean > 1e-6 else 0)

        # Both conditions must hold:
        if (train_sh_drop_pct >= 0.10                                              # train still falling >= 10% in second half
            and relative_improvement < _SHAPE_THRESHOLDS["diverge_val_plateau_pct"]   # val barely improving
            and val_sh_swing < _SHAPE_THRESHOLDS["diverge_val_plateau_pct"] * overall_mean * 2):
            return Diagnosis(
                shape="train_val_diverge",
                diagnosis=(f"Second-half train_loss kept falling {train_sh_drop_pct*100:.1f}% "
                           f"({train_sh[0]:.3f} → {train_sh[-1]:.3f}) but "
                           f"val mAP plateaued ({val_sh[0]:.4f} → {val_sh[-1]:.4f}, "
                           f"swing {val_sh_swing:.4f}). "
                           f"Final val={final.val_map:.4f}. Classic overfitting — "
                           f"consider weight_decay↑, dropout, or more augmentation."),
                final_map=final.val_map, peak_map=peak.val_map,
                peak_epoch=peak.epoch, final_epoch=final.epoch,
            )

    # ─── Priority 5: monotonic_climbing ────────────────────────────────────
    # Last third specifically still climbing — not just "overall higher than start".
    # The key difference vs converged: last_third_climb (within last third) must
    # itself be > threshold, indicating the curve hasn't plateaued yet.
    if (last_third_climb >= _SHAPE_THRESHOLDS["monotonic_min_climb_pct"] * overall_mean
        and first_to_last_climb >= _SHAPE_THRESHOLDS["monotonic_min_climb_pct"] * overall_mean):
        return Diagnosis(
            shape="monotonic_climbing",
            diagnosis=(f"val mAP rose monotonically: first-third start "
                       f"{first_third[0]:.4f} → last-third end {last_third[-1]:.4f} "
                       f"(overall climb {first_to_last_climb:.4f}, last-third "
                       f"still rising by {last_third_climb:.4f}). Final={final.val_map:.4f}. "
                       f"TIME_BUDGET likely cut off before convergence — module may "
                       f"need different LR / warmup."),
            final_map=final.val_map, peak_map=peak.val_map,
            peak_epoch=peak.epoch, final_epoch=final.epoch,
        )

    # ─── Priority 6: converged (default) ───────────────────────────────────
    # Baseline-aware split for final verdict context
    if baseline_final_map is not None and final.val_map >= baseline_final_map:
        shape = "converged_above_baseline"
        verdict_note = (f">= baseline ({baseline_final_map:.4f}) — keep candidate")
    elif baseline_final_map is not None:
        shape = "converged_below_baseline"
        verdict_note = (f"< baseline ({baseline_final_map:.4f}) — module under-performs "
                        f"on this setup; further hyperparam tuning unlikely to recover "
                        f"baseline (consider discarding)")
    else:
        shape = "converged_below_baseline"   # baseline-less defaults to "below"
        verdict_note = "no baseline available for comparison"

    return Diagnosis(
        shape=shape,
        diagnosis=(f"val mAP plateaued in last third (swing "
                   f"{last_third[-1] - last_third[0]:.4f}). "
                   f"Final={final.val_map:.4f}, peak={peak.val_map:.4f} "
                   f"(epoch {peak.epoch}). {verdict_note}."),
        final_map=final.val_map, peak_map=peak.val_map,
        peak_epoch=peak.epoch, final_epoch=final.epoch,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public canonical shape names — for callers that want the set
# ──────────────────────────────────────────────────────────────────────────────

CANONICAL_SHAPES = {
    "flat_no_learning",
    "early_collapse",
    "oscillating",
    "train_val_diverge",
    "monotonic_climbing",
    "converged_above_baseline",
    "converged_below_baseline",
}
