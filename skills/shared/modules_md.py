"""
modules.md canonical parser.

All skills that read or write modules.md go through this module so the file format
lives in exactly one place. Do not parse modules.md with regex or str.count() in
the skills themselves.

## File format

Top of file: registry header
    # Modules Registry

    Task: <task description>
    Base model: <name>
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


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse(path: str | pathlib.Path) -> list[Module]:
    """Parse modules.md. Returns [] if file missing or has no modules."""
    p = pathlib.Path(path)
    if not p.exists():
        return []

    text = p.read_text()
    # Split on level-2 headers. Leading \n prevents accidental mid-line matches.
    # First chunk (before any ##) is the registry header — skip it.
    chunks = re.split(r"(?m)^## ", text)
    modules: list[Module] = []
    for chunk in chunks[1:]:
        lines = chunk.splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        body = "\n".join(lines[1:])
        modules.append(Module(
            name=name,
            fields=_parse_pipe_table(body),
            sections=_parse_sections(body),
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

def find_pending(path, sort_by_complexity: bool = True) -> list[Module]:
    pending = [m for m in parse(path) if m.status == "pending"]
    if sort_by_complexity:
        pending.sort(key=lambda m: _COMPLEXITY_ORDER.get(m.complexity or "high", 99))
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
    """Update Status field for a named module. Returns True if a change was made."""
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status {new_status!r}; must be one of {sorted(VALID_STATUSES)}")

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
    _refresh_header(new_text_list := [new_text], p)   # refresh "Last updated"
    p.write_text(new_text_list[0])
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
    # Update header counters
    new_count = len(parse(p)) + 1
    text = re.sub(r"(?m)^Total modules:\s*\d+\s*$", f"Total modules: {new_count}", text)
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


def _refresh_header(text_list: list[str], path: pathlib.Path) -> None:
    """Refresh 'Last updated' in the registry header (in place on text_list[0])."""
    text_list[0] = re.sub(
        r"(?m)^Last updated:\s*\S+\s*$",
        f"Last updated: {date.today().isoformat()}",
        text_list[0],
    )
