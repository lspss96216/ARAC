"""
modules.md canonical parser.

All skills that read or write modules.md go through this module so the file format
lives in exactly one place. Do not parse modules.md with regex or str.count() in
the skills themselves.

## File format

Top of file: registry header
    # Modules Registry

    Task: <task description>
    Base model: <n>
    Last updated: <iso date>
    Total modules: <n>

    ---

Each module is a level-2 section:
    ## <Module Name>

    | Field | Value |
    |-------|-------|
    | Status | pending |
    | Complexity | low |
    | Location | neck |
    | paper2code | yes |
    | arXiv | https://arxiv.org/abs/... |
    ...

    ### What it does
    Free text.

    ### Integration notes
    Free text.

Field values are single-line. Subsection bodies may be multi-line.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


VALID_STATUSES = {"pending", "injected", "tested", "discarded"}
VALID_COMPLEXITIES = {"low", "medium", "high"}
_COMPLEXITY_ORDER = {"low": 0, "medium": 1, "high": 2}

# v1.7 — integration_mode declares how autoresearch should apply the module.
# WARN-NOT-REJECT policy: unknown values are tolerated (just logged) so adding
# new modes in future versions doesn't break old modules.md files.
#   "hook"       — existing v1.6 path: append USE_X flag + branch in inject_modules()
#   "yaml_inject" — v1.7: architectural insertion via weight_transfer.py
#   "full_yaml"  — reserved for v1.8+: agent writes complete custom YAML
KNOWN_INTEGRATION_MODES = {"hook", "yaml_inject", "full_yaml"}
DEFAULT_INTEGRATION_MODE = "hook"

# v1.9 — resource_impact tags. autoresearch Step 3 Modify uses these to
# preemptively halve BATCH_SIZE for memory-heavy experiments (avoiding the
# silent CPU TaskAlignedAssigner fallback that under-trains the run).
#
# v1.12 — semantics extended.
# Real-world Loop 1 of session 2026-04-27: FlexSimAM was tagged
# resource_impact=none (zero parameters), but scope=all in YOLO26x meant it
# inserted in dozens of C3k2 positions. Each insertion creates 3 forward-
# pass intermediate tensors (d, e_inv, sigmoid output) the size of the
# feature map. Backprop must keep all of them as activation buffer.
# Combined with TaskAlignedAssigner, the run hit 81.5GB VRAM ceiling on
# H100 80GB and degraded to CPU assigner (13× slowdown).
#
# Lesson: parameter count alone doesn't predict memory cost. Scope
# multiplier matters too. v1.12 introduces the concept of EFFECTIVE
# resource_impact = base × scope multiplier, with the rule:
#
#   - scope=backbone or scope=neck or scope=head:  base   (1× the tag)
#   - scope=all:                                    +1 step (none → 2x, 2x → 4x)
#
# Modules with no scope info (hook mode, full_yaml mode) use base only —
# their memory profile depends on different factors that are caller-side
# concerns.
KNOWN_RESOURCE_IMPACTS = {
    "vram_4x",            # ~4× baseline VRAM (P2 head, dense attention)
    "vram_2x",            # ~2× baseline VRAM (single attention layer added)
    "cpu_fallback_risk",  # known to trigger CPU assigner fallback at default batch
    "none",               # explicit "no extra cost" — for parser sanity, optional
}

# v1.12 — escalation order for scope=all multiplier.
_RESOURCE_ESCALATION = {
    "none":              "vram_2x",     # zero-param + scope=all → behaves like 2x
    "vram_2x":           "vram_4x",     # 2x + scope=all → 4x territory
    "vram_4x":           "vram_4x",     # already at ceiling; no further escalation
    "cpu_fallback_risk": "cpu_fallback_risk",  # orthogonal axis; don't escalate
}


@dataclass
class Module:
    """A single module entry."""
    name: str
    fields: dict = field(default_factory=dict)
    sections: dict = field(default_factory=dict)

    @property
    def status(self) -> Optional[str]:
        return self.fields.get("Status")

    @property
    def complexity(self) -> Optional[str]:
        return self.fields.get("Complexity")

    @property
    def paper2code(self) -> Optional[str]:
        return self.fields.get("paper2code")

    @property
    def arxiv_url(self) -> Optional[str]:
        return self.fields.get("arXiv")

    @property
    def pdf_path(self) -> Optional[str]:
        return self.fields.get("pdf_path")

    @property
    def integration_mode(self) -> str:
        """v1.7 — returns DEFAULT_INTEGRATION_MODE if field absent, blank,
        or whitespace-only. Unknown values are returned as-is (warn-not-reject);
        callers dispatch on known values and fall back to hook for unknown."""
        raw = self.fields.get("Integration mode") or ""
        stripped = raw.strip()
        return stripped if stripped else DEFAULT_INTEGRATION_MODE

    @property
    def resource_impact(self) -> Optional[str]:
        """v1.9 — autoresearch reads this to decide whether to halve BATCH_SIZE
        before running. Returns None when the field is absent or blank (treated
        as 'unknown', no auto-halve). Returns the raw value otherwise; callers
        check membership in KNOWN_RESOURCE_IMPACTS and fall back gracefully."""
        raw = self.fields.get("resource_impact") or ""
        stripped = raw.strip()
        return stripped if stripped else None

    @property
    def yaml_inject_scope(self) -> Optional[str]:
        """v1.12 — extract `scope:` from the Integration notes section
        (yaml_inject mode only). Returns "backbone", "neck", "head", "all",
        or None if not yaml_inject / not specified.

        Real-world Loop 1 evidence: FlexSimAM with scope=all consumed enough
        VRAM to trigger CPU assigner fallback despite zero parameters.
        Scope multiplier needs to feed into resource_impact decisions.
        """
        if self.integration_mode != "yaml_inject":
            return None
        notes = self.sections.get("Integration notes") or ""
        m = _SCOPE_LINE_RE.search(notes)
        return m.group(1).lower() if m else None

    @property
    def effective_resource_impact(self) -> Optional[str]:
        """v1.12 — resource_impact escalated by scope=all multiplier.

        For yaml_inject modules, scope=all means the module is inserted at
        every match point in the model (typically dozens of C3k2 / C2f
        blocks). The activation memory cost scales linearly with insertion
        count. A "none" or "vram_2x" tagged module with scope=all behaves
        like one tier higher in practice.

        Returns the escalated tag (still in KNOWN_RESOURCE_IMPACTS), or None
        if no resource_impact is set. autoresearch should use this property,
        not raw resource_impact, when deciding batch halving.
        """
        base = self.resource_impact
        if base is None:
            return None
        # Only escalate for yaml_inject modules — hook mode and full_yaml
        # mode have different memory profiles that scope doesn't capture.
        if self.integration_mode != "yaml_inject":
            return base
        scope = self.yaml_inject_scope
        if scope == "all":
            return _RESOURCE_ESCALATION.get(base, base)
        return base


# v1.12 — match `scope: <value>` in Integration notes. Spec format:
#   yaml_inject spec:
#    - module_class: LazyCBAM
#    - position: after_class: C3k2
#    - scope: backbone        # ← we extract this
#    - yaml_args: [256]
_SCOPE_LINE_RE = re.compile(
    r"^\s*-?\s*scope\s*:\s*([A-Za-z_+]+)",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse(path: str | pathlib.Path) -> list[Module]:
    """Parse modules.md. Returns [] if file missing or has no modules.

    D8 fix: only look for module headers AFTER the registry header's
    first `---` separator. Without this, any `## ` sequence appearing
    in the registry header block would be parsed as a module.

    Known limitation: if a module's free-text section body contains a
    line starting with `## ` at column 0, it WILL be parsed as a new
    module. Callers should write section bodies via `append_module` and
    avoid starting lines with `## ` at column 0 (indent or use a
    different heading level for body content).
    """
    p = pathlib.Path(path)
    if not p.exists():
        return []

    text = p.read_text()

    # D8 — skip past the registry header. The spec guarantees a `---`
    # line (horizontal rule) between the registry header and the first
    # module. If absent, fall back to parsing from the start.
    sep_m = re.search(r"(?m)^-{3,}\s*$", text)
    body = text[sep_m.end():] if sep_m else text

    # Split on level-2 headers. Require `## ` followed by a non-space
    # character so lines like `##` in code blocks (e.g. `##` in a bash
    # heredoc) aren't treated as headers.
    chunks = re.split(r"(?m)^## (?=\S)", body)
    modules: list[Module] = []
    for chunk in chunks[1:]:       # chunk 0 is whatever was before the first module
        lines = chunk.splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        mbody = "\n".join(lines[1:])
        modules.append(Module(
            name=name,
            fields=_parse_pipe_table(mbody),
            sections=_parse_sections(mbody),
        ))
    return modules


_SEP_CELL = re.compile(r"^:?-+:?$")   # table separator cell like `---` or `:--:`


def _parse_pipe_table(body: str) -> dict:
    out: dict = {}
    for line in body.splitlines():
        line = line.rstrip()
        if not line.startswith("|"):
            if out:
                break       # table ended once we start seeing non-pipe lines
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        k, v = cells
        # Skip header row and separator row
        if (k, v) == ("Field", "Value"):
            continue
        if _SEP_CELL.match(k) and _SEP_CELL.match(v):
            continue
        if k:
            out[k] = v
    return out


def _parse_sections(body: str) -> dict:
    sections: dict = {}
    current: str | None = None
    buf: list[str] = []
    for line in body.splitlines():
        m = re.match(r"^###\s+(.+)$", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def find_pending(
    path,
    sort_by_complexity: bool = True,
    preferred_locations: list[str] | None = None,
) -> list[Module]:
    """Return pending modules, optionally sorted.

    Sort keys (applied as a tuple in order):
      1. Complexity — low (0), medium (1), high (2), missing → 99
         Disable with sort_by_complexity=False.
      2. Location rank — index of Location in preferred_locations
         (case-insensitive). Locations not in the list rank after all
         listed ones. Pass None or [] to skip this key.
      3. Write order — stable tiebreak, preserves modules.md write order
         for modules that tie on the above keys.

    v1.7.3 — added preferred_locations secondary key. Pre-v1.7.3 behaviour
    is preserved when preferred_locations is None or empty (single-key
    sort by complexity).
    """
    pending = [m for m in parse(path) if m.status == "pending"]
    if not sort_by_complexity:
        return pending

    # Build case-insensitive location → rank map
    loc_rank: dict[str, int] = {}
    if preferred_locations:
        for i, loc in enumerate(preferred_locations):
            if isinstance(loc, str):
                loc_rank[loc.strip().lower()] = i
    unknown_rank = len(loc_rank)   # unlisted locations sort after listed

    def key(m: Module):
        c = _COMPLEXITY_ORDER.get(m.complexity or "high", 99)
        location = (m.fields.get("Location") or "").strip().lower()
        r = loc_rank.get(location, unknown_rank)
        return (c, r)

    pending.sort(key=key)
    return pending


def count_pending(path) -> int:
    return len(find_pending(path, sort_by_complexity=False))


def list_pdf_paths(path) -> set[str]:
    return {m.pdf_path for m in parse(path) if m.pdf_path}


def find_by_name(path, name: str) -> Optional[Module]:
    for m in parse(path):
        if m.name == name:
            return m
    return None


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def update_status(path, module_name: str, new_status: str) -> bool:
    """Update Status field for a named module.
    Returns True if a change was made."""
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status {new_status!r}; must be one of {sorted(VALID_STATUSES)}"
        )

    p = pathlib.Path(path)
    text = p.read_text()

    pattern = re.compile(
        rf"(^##\s+{re.escape(module_name)}\s*$[\s\S]*?)"
        rf"(^\|\s*Status\s*\|\s*)([^|\n]+?)(\s*\|)",
        re.MULTILINE,
    )
    new_text, n = pattern.subn(rf"\1\2{new_status}\4", text, count=1)
    if n == 0:
        return False
    # C4 — pure-function refresh, no list-mutation tricks.
    p.write_text(_refresh_header(new_text))
    return True


def append_module(path, module: Module | dict) -> None:
    if isinstance(module, dict):
        module = Module(
            name=module["name"],
            fields=dict(module.get("fields", {})),
            sections=dict(module.get("sections", {})),
        )

    module.fields.setdefault("Status", "pending")
    if module.fields["Status"] not in VALID_STATUSES:
        raise ValueError(f"Invalid Status {module.fields['Status']!r}")
    if "Complexity" in module.fields and module.fields["Complexity"] not in VALID_COMPLEXITIES:
        raise ValueError(f"Invalid Complexity {module.fields['Complexity']!r}")

    # v1.7 — integration_mode is optional; default "hook" keeps v1.6 behaviour.
    # Unknown values are WARNED but not rejected, so future values (added by
    # later skill versions) don't break old modules.md files.
    mode = module.fields.get("Integration mode")
    if mode is not None:
        mode = mode.strip()
        if mode and mode not in KNOWN_INTEGRATION_MODES:
            import sys as _sys
            print(
                f"WARN: Integration mode {mode!r} for module {module.name!r} "
                f"is not one of {sorted(KNOWN_INTEGRATION_MODES)}. "
                f"Autoresearch will fall back to {DEFAULT_INTEGRATION_MODE!r}.",
                file=_sys.stderr,
            )

    p = pathlib.Path(path)
    rendered = _render(module)

    if not p.exists() or not p.read_text().strip():
        header = (
            "# Modules Registry\n\n"
            f"Last updated: {date.today().isoformat()}\n"
            "Total modules: 1\n\n"
            "---\n\n"
        )
        p.write_text(header + rendered + "\n")
        return

    text = p.read_text()

    # D4 — read Total modules counter directly from the header instead of
    # re-parsing the whole file. Each append is now O(1) in existing module
    # count, not O(n).
    m = re.search(r"(?m)^Total modules:\s*(\d+)\s*$", text)
    current = int(m.group(1)) if m else 0
    new_count = current + 1
    text = re.sub(
        r"(?m)^Total modules:\s*\d+\s*$",
        f"Total modules: {new_count}",
        text,
    )
    text = re.sub(
        r"(?m)^Last updated:\s*\S+\s*$",
        f"Last updated: {date.today().isoformat()}",
        text,
    )
    if not text.endswith("\n"):
        text += "\n"
    text += "\n" + rendered + "\n"
    p.write_text(text)


def _render(module: Module) -> str:
    lines = [f"## {module.name}", "", "| Field | Value |", "|-------|-------|"]
    for k, v in module.fields.items():
        lines.append(f"| {k} | {v} |")
    for section_name, section_body in module.sections.items():
        lines.append("")
        lines.append(f"### {section_name}")
        lines.append(section_body)
    return "\n".join(lines)


def _refresh_header(text: str) -> str:
    """C4 — pure function: return text with 'Last updated' in the registry
    header set to today. Previous version mutated a list passed in as arg,
    which was an awkward workaround for Python not having pass-by-reference
    for strings. Pure is clearer."""
    return re.sub(
        r"(?m)^Last updated:\s*\S+\s*$",
        f"Last updated: {date.today().isoformat()}",
        text,
    )
