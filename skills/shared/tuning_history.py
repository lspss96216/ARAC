"""tuning_history.py — v1.13

Per-module attempt history. Records each (loop, module, attempt_n)
triple's trajectory shape + diagnosis + hyperparams used + final mAP,
so:

  1. Agent reading next-attempt context sees what was tried before
     (LR=0.01 oscillating → LR=0.005 next).
  2. Cross-iteration learning: pattern across modules ("attention
     modules consistently want LR=0.001 on this dataset").
  3. results.tsv stays clean (1 row per module, the BEST attempt) —
     details live here.

Schema (TSV, append-only, header on first write):

    timestamp ISO
    loop_count int
    module_name str
    attempt_n int (1, 2, 3, ...)
    shape str (one of CANONICAL_SHAPES)
    final_map float
    peak_map float
    peak_epoch int
    final_epoch int
    hyperparams_json str (JSON of {LR0, MOMENTUM, WEIGHT_DECAY, WARMUP_EPOCHS, OPTIMIZER, ...})
    diagnosis str (truncated to 500 chars to keep tsv parseable)

The file is append-only. Old runs are never modified — even a failed
mid-run pretrain triggers re-baseline, which restarts attempt counts
but doesn't erase prior tuning_history.
"""

from __future__ import annotations
import csv
import json
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# Header columns in canonical order. Writers emit, readers expect this.
COLUMNS = [
    "timestamp",
    "loop_count",
    "module_name",
    "attempt_n",
    "shape",
    "final_map",
    "peak_map",
    "peak_epoch",
    "final_epoch",
    "hyperparams_json",
    "diagnosis",
]

# Truncation cap for diagnosis field — TSV gets unwieldy past this and
# agent only needs the gist for cross-loop reasoning anyway.
_DIAGNOSIS_MAX_LEN = 500


@dataclass
class Attempt:
    """Single (module, attempt_n) record. Used by both writers and readers."""
    timestamp: str
    loop_count: int
    module_name: str
    attempt_n: int
    shape: str
    final_map: float
    peak_map: float
    peak_epoch: int
    final_epoch: int
    hyperparams: dict[str, Any] = field(default_factory=dict)
    diagnosis: str = ""

    def to_row(self) -> dict[str, str]:
        """Serialize to TSV row dict."""
        return {
            "timestamp":         self.timestamp,
            "loop_count":        str(self.loop_count),
            "module_name":       self.module_name,
            "attempt_n":         str(self.attempt_n),
            "shape":             self.shape,
            "final_map":         f"{self.final_map:.6f}",
            "peak_map":          f"{self.peak_map:.6f}",
            "peak_epoch":        str(self.peak_epoch),
            "final_epoch":       str(self.final_epoch),
            "hyperparams_json":  json.dumps(self.hyperparams, sort_keys=True, separators=(",", ":")),
            "diagnosis":         (self.diagnosis or "")[:_DIAGNOSIS_MAX_LEN].replace("\t", " ").replace("\n", " "),
        }

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Attempt":
        """Parse a TSV row dict back into an Attempt."""
        try:
            hp = json.loads(row.get("hyperparams_json", "") or "{}")
        except json.JSONDecodeError:
            hp = {}
        return cls(
            timestamp=row.get("timestamp", ""),
            loop_count=int(row.get("loop_count", "0") or "0"),
            module_name=row.get("module_name", ""),
            attempt_n=int(row.get("attempt_n", "0") or "0"),
            shape=row.get("shape", ""),
            final_map=float(row.get("final_map", "0") or "0"),
            peak_map=float(row.get("peak_map", "0") or "0"),
            peak_epoch=int(row.get("peak_epoch", "0") or "0"),
            final_epoch=int(row.get("final_epoch", "0") or "0"),
            hyperparams=hp,
            diagnosis=row.get("diagnosis", ""),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Writer — append-only
# ──────────────────────────────────────────────────────────────────────────────

def append_attempt(history_path: str, attempt: Attempt) -> None:
    """Append one attempt to the tuning_history.tsv. Creates file with header
    on first write. Never overwrites existing rows.

    Idempotency: NOT enforced at this level. Caller is responsible for not
    double-appending the same (loop_count, module_name, attempt_n). The
    autoresearch SKILL does this naturally — append happens once per Step 8.
    """
    p = pathlib.Path(history_path)
    file_exists = p.exists()

    # Open in append mode. csv.DictWriter handles the TSV details.
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(attempt.to_row())


# ──────────────────────────────────────────────────────────────────────────────
# Readers
# ──────────────────────────────────────────────────────────────────────────────

def read_all(history_path: str) -> list[Attempt]:
    """Read every attempt in chronological order. Returns [] if file missing."""
    p = pathlib.Path(history_path)
    if not p.exists():
        return []
    out: list[Attempt] = []
    try:
        with open(p, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    out.append(Attempt.from_row(row))
                except (ValueError, KeyError):
                    # Skip malformed rows — never crash the whole reader on
                    # one bad line. Agent gets the rest, can investigate
                    # discrepancy from raw file if needed.
                    continue
    except Exception:
        return []
    return out


def attempts_for_module(history_path: str, module_name: str) -> list[Attempt]:
    """All attempts for a given module, oldest first. Empty list if no record."""
    return [a for a in read_all(history_path) if a.module_name == module_name]


def latest_attempt_for_module(history_path: str, module_name: str) -> Optional[Attempt]:
    """Most recent attempt for a module, or None. Used by autoresearch to
    decide next attempt's hyperparams based on previous trajectory."""
    matches = attempts_for_module(history_path, module_name)
    return matches[-1] if matches else None


def attempt_count_for_module(history_path: str, module_name: str) -> int:
    """How many attempts have been made for this module? Used for the
    3-attempt cap (with possible 1-2 attempt extension under v1.13 rules)."""
    return len(attempts_for_module(history_path, module_name))


# ──────────────────────────────────────────────────────────────────────────────
# Cross-module pattern queries (for agent reasoning)
# ──────────────────────────────────────────────────────────────────────────────

def attempts_with_shape(history_path: str, shape: str) -> list[Attempt]:
    """All attempts that ended with the given trajectory shape. Useful for
    agent to ask 'what hyperparams worked when previous attempts oscillated?'."""
    return [a for a in read_all(history_path) if a.shape == shape]


def kept_attempts(history_path: str) -> list[Attempt]:
    """All attempts that ended with shape='converged_above_baseline'. The
    agent uses this to learn 'when CBAM kept on attempt 3, LR was 0.001;
    let me try LR=0.001 on this new attention module as starting point.'"""
    return attempts_with_shape(history_path, "converged_above_baseline")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for autoresearch agent context-building
# ──────────────────────────────────────────────────────────────────────────────

def format_module_history_for_agent(history_path: str, module_name: str) -> str:
    """Return a human-readable summary of all prior attempts for a module,
    suitable for inclusion in the agent's reasoning context. Returns empty
    string if no history."""
    attempts = attempts_for_module(history_path, module_name)
    if not attempts:
        return ""

    lines = [f"Tuning history for {module_name} ({len(attempts)} prior attempt(s)):"]
    for a in attempts:
        hp_str = ", ".join(f"{k}={v}" for k, v in sorted(a.hyperparams.items()))
        lines.append(
            f"  Attempt {a.attempt_n} (loop {a.loop_count}): "
            f"shape={a.shape}, final_mAP={a.final_map:.4f}, peak={a.peak_map:.4f}@ep{a.peak_epoch}, "
            f"hp=[{hp_str}]"
        )
        if a.diagnosis:
            lines.append(f"    diagnosis: {a.diagnosis[:200]}")
    return "\n".join(lines)
