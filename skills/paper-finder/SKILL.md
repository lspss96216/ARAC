---
name: paper finder
description: >
  Autonomous research skill for model improvement. Given a task
  description, searches academic sources (arXiv, Papers with Code, Semantic Scholar) to find
  the best base model and collect improvement modules from related papers. Writes findings to
  modules.md for use by autoresearch and paper2code. Also handles the feedback loop when
  autoresearch experiments stall — searches for additional papers to expand the module pool.
  Also scans a local folder of manually downloaded PDFs (for papers too new or not yet
  public on arXiv). Trigger on phrases like "find papers for my task", "search for base model",
  "look for relevant techniques", or when autoresearch reports no improvement for N consecutive rounds.
---

# Paper Finder

Searches academic sources to find (1) the best base model for the task, and (2) improvement
modules from related papers. Outputs are written to `modules.md` for downstream use by
`autoresearch` and `paper2code`.

Shared files produced by this skill (`base_model.md`, `modules.md`) have their
schemas documented in `<skills_dir>/shared/file-contracts.md` — read it once
before writing either file.

---

## When this skill runs

**Mode A — Initial search** (triggered by user task description):
Find base model + populate modules.md from scratch.

**Mode B — Expansion search** (triggered by autoresearch stall signal):
Keep existing modules.md, append newly found modules without overwriting.

The caller (autoresearch or user) passes one of:
- `mode: initial` + task description
- `mode: expand` + path to existing `modules.md` + reason for stall (e.g. "last 10 runs <0.001 improvement")

---

## Phase 1 — Parse the task

Read the task description and extract:
- **Detection target**: what objects need to be detected (e.g. small UAV objects, pedestrians)
- **Key challenges**: small objects, dense scenes, real-time constraint, low resolution, etc.
- **Dataset domain**: aerial / drone, medical, autonomous driving, satellite, general
- **Local papers directory**: path to a folder of manually downloaded PDFs (optional).
  If not provided by the user or pipeline state, check `pipeline_state.json` for
  `"local_papers_dir"`. If still not set, skip the local source silently.

Use these to build targeted search queries for each source.

---

## Phase 2 — Search sources

Search the following sources in parallel. For each source, collect paper titles, arxiv IDs,
and abstract snippets. Do not download full PDFs yet — that happens in Phase 3.

### arXiv

```bash
# Search via arXiv API (no key required)
curl "https://export.arxiv.org/api/query?search_query=all:<query>&max_results=30&sortBy=relevance" 
```

Use multiple targeted queries based on Phase 1 output. Examples for drone detection:
- `"small object detection UAV YOLO"`
- `"drone aerial object detection transformer"`
- `"object detection attention mechanism small target"`

Extract: `id`, `title`, `summary`, `published` date from the XML response.

### Papers with Code

```bash
curl "https://paperswithcode.com/api/v1/papers/?q=<query>&page=1" \
    -H "Accept: application/json"
```

Focus on papers with associated code repos and benchmark results on relevant datasets
(VisDrone, COCO, DOTA, etc.).

#### Deep scrape via Firecrawl (optional enhancement)

The PwC JSON API returns basic metadata (title, abstract, URL) but misses
repo README details, exact benchmark numbers, and pretrained weights links.
If `firecrawl` CLI is installed, scrape each paper's PwC page to extract richer
data:

```bash
# Check if firecrawl is available
if command -v firecrawl &>/dev/null; then
    for paper_url in $PwC_PAPER_URLS; do
        slug=$(echo "$paper_url" | grep -oE '[^/]+$')
        firecrawl scrape "$paper_url" -o ".firecrawl/pwc-${slug}.md" 2>/dev/null
    done
fi
```

From the scraped markdown, extract:
- **GitHub repo URL**: look for `github.com` links in the "Code" section
- **Benchmark table rows**: look for markdown tables with mAP / HOTA / MOTA columns
- **Pretrained weights URL**: look for links ending in `.pt` / `.pth` / `.ckpt`

```python
import pathlib, re

def extract_pwc_details(scraped_md: str) -> dict:
    """Extract structured data from a Firecrawl-scraped PwC page."""
    details = {"repo_url": None, "benchmarks": [], "weights_urls": []}
    # GitHub repo
    m = re.search(r'(https?://github\.com/[\w-]+/[\w.-]+)', scraped_md)
    if m: details["repo_url"] = m.group(1)
    # Weights links
    details["weights_urls"] = re.findall(
        r'(https?://\S+\.(?:pt|pth|ckpt|safetensors))', scraped_md)
    # Benchmark numbers (mAP, HOTA, etc.)
    for line in scraped_md.splitlines():
        if re.search(r'\b(?:mAP|HOTA|MOTA|AP50)\b', line) and '|' in line:
            details["benchmarks"].append(line.strip())
    return details
```

If Firecrawl is not installed, the pipeline falls back to the JSON API alone —
no data is lost, just less detail available for scoring. Install with:
`npm install -g firecrawl` + set `FIRECRAWL_API_KEY` in `.env` (see
orchestrator Stage 0 Step 1.5). The CLI reads the key from `os.environ`
automatically.

### Semantic Scholar

```bash
curl "https://api.semanticscholar.org/graph/v1/paper/search?query=<query>&fields=title,abstract,year,citationCount,externalIds&limit=20"
```

Prioritise by citation count as a quality signal.

### Local PDF folder

If `local_papers_dir` is set and the directory exists, scan it for PDF files:

```bash
find "<local_papers_dir>" -name "*.pdf" | sort
```

For each PDF:

1. **Extract text** using `pdfplumber` or `pymupdf`:
   ```python
   import pdfplumber, pathlib

   def extract_pdf_text(pdf_path: str, max_pages: int = 12) -> str:
       with pdfplumber.open(pdf_path) as pdf:
           pages = pdf.pages[:max_pages]
           return "\n".join(p.extract_text() or "" for p in pages)
   ```
   Read at most 12 pages (abstract + intro + method section is usually enough).

2. **Extract metadata** from the text:
   - Title: first non-empty line, or largest font text on page 1
   - Year: search for 4-digit year pattern in first 200 chars
   - arXiv ID: search for `arxiv.org/abs/` or `arXiv:XXXX.XXXXX` pattern
   - Authors: second non-empty line after title (best-effort)

3. **Check if paper is on arXiv** using the extracted ID or title:
   ```bash
   # If arXiv ID found:
   curl "https://export.arxiv.org/api/query?id_list=<arxiv_id>"
   # If no arXiv ID, search by title:
   curl "https://export.arxiv.org/api/query?search_query=ti:<title>&max_results=3"
   ```
   - If found on arXiv → use the arXiv metadata (abstract, year, ID) and mark
     `paper2code: yes`
   - If not found → use extracted text as abstract, mark `arxiv_id: null`,
     `paper2code: no (not on arXiv)`

4. **Search GitHub for a matching repo**:
   ```bash
   curl "https://api.github.com/search/repositories?q=<model_name>+object+detection&sort=stars&per_page=5"
   ```
   Use the model/method name extracted from the title as the search query.
   - If a repo with ≥ 10 stars and name matching the paper title is found →
     record as `code_url` and set `paper2code: yes (GitHub repo available)`
   - Otherwise → `paper2code: no (no public repo found)`

5. Add to the candidate pool with `source: local_pdf` and `pdf_path: <path>`.
   Local papers are **not** penalised in the Recency score for lacking an arXiv ID —
   they are assumed to be recent by virtue of being manually collected.

**If `pdfplumber` is not installed:**
```bash
pip install pdfplumber --quiet
```
If install fails (network restricted), fall back to `pdftotext` CLI:
```bash
pdftotext -l 12 "<pdf_path>" -
```

### Semantic Scholar citation expansion (fallback)

If the above sources return < 10 relevant results, expand via citation graph:
```bash
# Get papers that cite the most relevant paper found so far
curl "https://api.semanticscholar.org/graph/v1/paper/<s2_paper_id>/citations?fields=title,abstract,year,citationCount,externalIds&limit=50"
# Also search for survey papers in the domain
curl "https://api.semanticscholar.org/graph/v1/paper/search?query=survey+<domain>+object+detection&fields=title,abstract,year,citationCount,externalIds&limit=20"
```
Mine related-work sections of survey papers for additional technique references.

---

## Phase 3 — Score and select papers

For each collected paper, score on these criteria:

| Criterion | Weight | How to assess |
|-----------|--------|---------------|
| Task relevance | 40% | Does abstract mention the same detection domain / challenge? |
| Recency | 20% | Published 2022 or later scores higher. Local PDFs with no year → assume current year (full 20 pts) |
| Has official code | 20% | GitHub link in paper or Papers with Code entry |
| Benchmark result | 20% | Reports mAP on VisDrone, COCO, or relevant dataset |

Compute a weighted score (0–100) for each paper. Keep the top 20.

---

## Phase 4 — Select base model

From the top-scored papers, identify the **base model** candidates:

A paper qualifies as a base model candidate if:
- It proposes a complete detection architecture (not just a plug-in module)
- Pretrained weights are publicly available **OR** the architecture can be fully reconstructed
  from the paper + paper2code
- It outperforms plain YOLOv8/v9/v10/v11/v26 on at least one relevant benchmark

Select the single highest-scoring candidate as base model.

#### Validate weights URL via Firecrawl (optional)

Before writing the weights URL to `base_model.md`, verify the link is alive.
Dead links cause orchestrator Stage 2 to `wget` a 404 and silently fall back
to `yolo26x.pt`, wasting the entire base model selection.

```python
import subprocess, shutil

def validate_weights_url(url: str) -> bool:
    """Check if a weights URL is reachable. Uses firecrawl if available,
    falls back to curl HEAD request."""
    if not url or url == "reconstruct via paper2code":
        return True   # paper2code path — no URL to check

    if shutil.which("firecrawl"):
        # Firecrawl scrape returns non-zero on unreachable pages
        r = subprocess.run(["firecrawl", "scrape", url, "--format", "links"],
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    else:
        # Fallback: curl HEAD
        r = subprocess.run(["curl", "-sI", "-o", "/dev/null", "-w", "%{http_code}", url],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip().startswith("2")

# After selecting the base model, before writing base_model.md:
if not validate_weights_url(weights_url):
    log_discovery(f"Base model {model_name}: weights URL {weights_url} is dead. "
                  f"Falling back to runner-up.", loop=0, category="bug_workaround")
    # Try runner-up's weights URL; if also dead, use "reconstruct via paper2code"
```

Write to `base_model.md`. The file is read by orchestrator Stage 2 (extracts
`Weights URL`) and by dataset hunter Phase 5 (fallback path when orchestrator
did not pre-resolve). The contract — which fields those consumers need and how
they parse — is in `<skills_dir>/shared/file-contracts.md § base_model.md`.
```markdown
# Base Model

## Selected: <Model Name>
- **Paper**: <title>
- **arXiv**: https://arxiv.org/abs/<id>
- **Published**: <year>
- **Score**: <score>/100
- **Weights URL**: <url or "reconstruct via paper2code">
- **Benchmark**: <metric> on <dataset>
- **Reason**: <1–2 sentences why this was chosen>

## Runner-up: <Model Name>
...
```

---

## Phase 5 — Collect improvement modules

From the remaining top papers (excluding the base model paper), extract modular techniques
that can be plugged into an existing backbone, neck, head, or loss.

For each module, determine:
- **Location**: where it inserts (backbone / neck / head / loss / label assignment)
- **Improvement aspect**: what it claims to improve (small objects / classification accuracy /
  localisation / speed / etc.)
- **Integration complexity**: low (toggle flag) / medium (replace one block) / high (major surgery)
- **paper2code compatible**: one of:
  - `yes` — paper has an arXiv ID (paper2code can generate from it)
  - `yes (GitHub repo available)` — no arXiv ID but a matching public repo was found
  - `no (not on arXiv)` — local PDF only, no arXiv ID, no public repo found

- **Hyperparameters** (critical — autoresearch needs these to reproduce the
  paper's actual results, not just toggle the module). Extract from:
  - Abstract and method section: values the author mentions as "we set <x> = <y>"
  - Results tables: "best config" or "recommended" values
  - Official repo's default config / README, if `paper2code: yes (GitHub repo)`

  A toggle without supporting hyperparameters often reproduces as "module had
  no effect" because the default values of supporting parameters don't match
  what the paper tested.

  Capture at least these when the paper specifies them:
  - Internal scale factors, proposal sizes, ratios
  - Loss weights
  - Label assignment strategies tied to this module
  - Any numeric constants appearing in equations the paper labels "best"

Prioritise modules with:
- Integration complexity = low or medium
- Clear improvement aspect matching the task's key challenges
- paper2code compatible = yes

---

## Phase 6 — Write modules.md

All writes to `modules.md` go through the canonical parser at
`<skills_dir>/shared/modules_md.py`. Do not hand-write markdown or use string
concatenation — the parser owns the format so all three skills agree on it.

### Import the parser

```python
import sys, pathlib, json
state = json.loads(pathlib.Path("pipeline_state.json").read_text())
sys.path.insert(0, str(pathlib.Path(state["skills_dir"]) / "shared"))
import modules_md as mm
```

### Append each module

For each module collected in Phase 5, build a dict and call `mm.append_module`.
The parser handles headers, separators, counters, and "Last updated" timestamps.

```python
mm.append_module("modules.md", {
    "name": "<Module Name>",
    "fields": {
        "Paper":       "<full title>",
        "arXiv":       "https://arxiv.org/abs/<id>",     # or omit if not on arXiv
        "Published":   "<year>",
        "Authors":     "<first author et al.>",
        "Location":    "backbone",                        # backbone / neck / head / loss / tracker / post_processing
        "Improves":    "small objects",                   # free text; describes what it targets
        "Complexity":  "low",                             # low / medium / high
        "paper2code":  "yes",                             # yes / yes (GitHub repo: <url>) / no (not on arXiv) / no (no public repo)
        "pdf_path":    "<local pdf path>",                # only set for local PDFs
        "Status":      "pending",                         # default if omitted
    },
    "sections": {
        "What it does":     "<2–3 sentences from abstract, in your own words>",
        "Integration notes": (
            "<Where does this go in train.py? Use the vocabulary from "
            "train-script-spec.md § File layout:\n"
            " - Main toggle: USE_<MODULE> flag name to append\n"
            " - Target file: train.py (detector) or track.py (tracker)\n"
            " - Which Section ③ hook: inject_modules() or apply_tracker_modules()\n"
            " - Any gotchas: layer indices, required imports, ordering constraints.\n\n"
            "Hyperparameters the paper specifies (REQUIRED — autoresearch will "
            "apply these together with the toggle in one experiment):\n"
            " - <PARAM_NAME_1>: <value> (§<section> / Table <n>)\n"
            " - <PARAM_NAME_2>: <value> (§<section>)\n"
            " - ... list every parameter the paper lists as 'best config', "
            "'recommended', or uses in the main experiment.\n"
            "If the paper leaves a parameter unspecified, omit it — autoresearch "
            "will fall back to the code's __init__ default."
            ">"
        ),
        "paper2code command": "/paper2code https://arxiv.org/abs/<id>\nExtract: `<ClassName>` from `src/model.py`",
    },
})
```

**Validation handled by parser:**
- `Status` defaults to `"pending"` if omitted, and must be one of
  `pending / injected / tested / discarded`.
- `Complexity` must be one of `low / medium / high`.
- If the file does not exist, the parser writes the registry header automatically.
- If it exists, the parser updates `Total modules` and `Last updated` atomically.

### Mode B (expansion) — do not overwrite existing entries

In Mode B, only append **new** modules whose `pdf_path` (for local PDFs) or `arXiv`
URL is not already present. Dedupe using the parser:

```python
existing_pdfs    = mm.list_pdf_paths("modules.md")
existing_arxivs  = {m.arxiv_url for m in mm.parse("modules.md") if m.arxiv_url}

for candidate in new_candidates:
    key_pdf   = candidate.get("pdf_path")
    key_arxiv = candidate.get("arxiv_url")
    if key_pdf and key_pdf in existing_pdfs: continue
    if key_arxiv and key_arxiv in existing_arxivs: continue
    mm.append_module("modules.md", build_module_dict(candidate))
```

---

### Format reference (informational — parser owns this)

For readers who want to know what the file looks like on disk:

```markdown
# Modules Registry

Task: <task description>
Base model: <n>
Last updated: <iso date>
Total modules: <n>

---

## <Module Name>

| Field | Value |
|-------|-------|
| Paper | <full title> |
| Status | pending |
...

### What it does
...
```

Do not construct this format by hand. Always go through `mm.append_module`.

---

## Phase 7 — Report to user

Print a summary:
```
=== Paper Finder Summary ===
Mode: initial / expand

Sources searched: arXiv, Papers with Code, Semantic Scholar
Papers evaluated: XX
Papers kept (top 20): 20

Base model selected: <name>
  arXiv: <url>
  Weights: available / reconstruct via paper2code

Modules added to modules.md: XX
  low complexity:    X
  medium complexity: X
  high complexity:   X
  paper2code ready:  X

Next step:
  autoresearch will pick modules from modules.md ordered by complexity (low first).
  For each selected module, paper2code will generate the class, then inject into train.py.
```

---

## Phase 8 — Feedback loop trigger (Mode B)

When called from autoresearch in Mode B (stall detected):

1. Read existing `modules.md` — note which modules have `Status: pending`
2. If pending modules remain → report "X untested modules still in modules.md — no new search needed" and exit
3. If all pending modules have been tested → expand from all sources:

   **a. Re-scan local PDF folder** (if `local_papers_dir` is set):
   - List all PDF filenames in `local_papers_dir`
   - Use the parser to get pdf_paths already registered, then diff:
   ```python
   import sys, pathlib, json
   state = json.loads(pathlib.Path("pipeline_state.json").read_text())
   sys.path.insert(0, str(pathlib.Path(state["skills_dir"]) / "shared"))
   import modules_md as mm

   existing_paths = mm.list_pdf_paths("modules.md")
   all_pdfs = {str(p) for p in pathlib.Path(local_papers_dir).glob("*.pdf")}
   new_pdfs = all_pdfs - existing_paths
   ```
   Process each `new_pdf` through the Phase 2 local PDF pipeline (extract text,
   check arXiv, search GitHub repo), then append via `mm.append_module`.

   **b. Expand network queries:**
   - Add synonyms and adjacent techniques to the original queries
   - Search for papers that cite the base model paper (Semantic Scholar citations API)
   - Look for survey papers covering the task domain and mine their related-work sections

4. Append new modules to modules.md, mark them `Status: pending`
5. Report how many new modules were added (split by source: local / network)

---

## Coordination with dataset hunter

After writing `base_model.md`, notify the user that dataset hunter should use the selected
base model weights for pretrain. The relevant field is `Weights URL` in `base_model.md`.

When dataset hunter runs Phase 5 (pretrain), it must read `base_model.md` first:
- If `Weights URL` is a direct download → use that as `WEIGHTS` in `pretrain.py`
- If `Weights URL` is `"reconstruct via paper2code"` → run `/paper2code <arxiv_id>` on
  the base model paper first, build the weights, then set `WEIGHTS` accordingly
- If `base_model.md` does not exist → fall back to the default `weights/yolo26x.pt`

---

## Integration with paper2code

This section is the authoritative reference. Autoresearch Step 2 follows this priority order.

When autoresearch selects a module from modules.md to test, use the **first available** option:

**Priority 1 — arXiv ID available (`paper2code: yes`)**
```
/paper2code https://arxiv.org/abs/<id>
```
From the generated `<paper_slug>/src/model.py`, extract the target class:
```bash
grep -n "class <ClassName>" <paper_slug>/src/model.py
```

**Priority 2 — GitHub repo available (`paper2code: yes (GitHub repo available)`)**
Only if no arXiv ID exists:
```bash
git clone <code_url> /tmp/<repo_name>
grep -rn "class <ClassName>" /tmp/<repo_name>/
```
Read the located class from the repo.

**Priority 3 — Manual write (`paper2code: no`)**
Only if neither arXiv nor GitHub repo is available:
Write the module class manually in `custom_modules.py` based on the paper description
in modules.md, using citation-anchoring comments (`# §3.2 — ...`).

**After any of the above, autoresearch handles the injection** (not paper finder).
Autoresearch's Priority A flow picks the pending module from modules.md and does
the following, using the contract in `<skills_dir>/shared/train-script-spec.md`:

1. Copy or write the class into `custom_modules.py`
   (the spec's template already has `from custom_modules import *` in Section ①,
   so no explicit per-class import is needed in `train.py`)
2. Append `USE_<MODULE> = False` to the end of Section ②
3. Add an idempotent branch inside `inject_modules()` in Section ③ (detector
   modules) or `apply_tracker_modules()` in `track.py` (tracker modules)
4. Flip `USE_<MODULE> = True` as the experiment change
5. Mark the module `injected` via `modules_md.update_status(...)`

Paper finder's job stops at writing the module into `modules.md` with a clear
`Integration notes` section. The actual edits to `train.py` / `track.py` /
`custom_modules.py` happen in autoresearch.

---

## Error handling

| Error | Action |
|-------|--------|
| arXiv API rate limit (HTTP 429) | Wait 3s, retry once |
| Semantic Scholar 429 | Skip, continue with arXiv + PwC results |
| < 5 relevant papers found | Broaden queries, remove domain-specific terms, retry |
| Base model has no public weights and paper2code fails | Select runner-up |
| modules.md already has 20+ pending modules | Skip search, report to user |
