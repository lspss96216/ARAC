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

### v1.7.5 — User-specified base model (short-circuit Phase 2–4)

If `research_config.yaml → task` contains `preferred_base_model` and
`preferred_base_weights_url`, **skip Phase 2–4 entirely** and write
`base_model.md` directly. This handles cases like:
- Model exists but has no formal arXiv paper (ultralytics releases,
  private models)
- User wants to pin a specific model for reproducibility
- Scoring/benchmarking infrastructure unavailable (`.env` missing keys)

```python
import yaml, pathlib, json
cfg = yaml.safe_load(pathlib.Path("research_config.yaml").read_text())
task = cfg.get("task", {})
pref_name = task.get("preferred_base_model")
pref_url  = task.get("preferred_base_weights_url")

if pref_name and pref_url:
    pref_arxiv = task.get("preferred_base_arxiv_id")   # optional
    md = f"""# Base Model: {pref_name}

## Selected: {pref_name} (user-specified)

**Source**: user override via `research_config.yaml → task.preferred_base_model`

**Weights URL**: {pref_url}
"""
    if pref_arxiv:
        md += f"\n**arXiv**: https://arxiv.org/abs/{pref_arxiv}\n"
    md += """
**Score**: n/a (user override, Phases 2-4 skipped)
**Benchmark**: n/a
**Reason**: User specified this model directly in research_config.yaml.

Paper finder did not evaluate alternatives. To re-enable automatic
selection, remove `preferred_base_model` from the yaml and re-run.
"""
    pathlib.Path("base_model.md").write_text(md)

    # Mark state so orchestrator Stage 1 doesn't think paper-finder failed
    state = json.loads(pathlib.Path("pipeline_state.json").read_text())
    state["base_model_md_ready"] = True
    state["base_model_user_override"] = True
    pathlib.Path("pipeline_state.json").write_text(json.dumps(state, indent=2))

    print(f"[paper-finder] user-specified base: {pref_name}")
    print(f"[paper-finder] skipping Phase 2-4, proceeding to Phase 5")
    # Continue directly to Phase 5 (modules collection)
else:
    # Normal flow — proceed to Phase 2
    pass
```

---

## Phase 2 — Search sources

Search the following sources in parallel. For each source, collect paper titles, arxiv IDs,
and abstract snippets. Do not download full PDFs yet — that happens in Phase 3.

**v1.10 — `_source` tag required.** Every candidate added to the
collection MUST carry a `_source` field identifying which source it
came from. This drives Phase 2.5's cross-source dedup. Six valid values:

| `_source` value | Meaning |
|---|---|
| `arxiv` | arXiv API (this section) |
| `pwc` | Papers with Code |
| `s2` | Semantic Scholar primary search |
| `local` | Local PDF folder (user-provided) |
| `hf` | HuggingFace Papers (Source 6, v1.10+) |
| `s2_expansion` | Semantic Scholar citation expansion fallback |

Skipping `_source` makes the candidate invisible to dedup. Phase 2.5
will warn but cannot recover.

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
If the `firecrawl` CLI is installed, scrape each paper's PwC page to extract
richer data:

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

**Note on Firecrawl CLI (D1/D2):** Firecrawl is primarily a hosted API
service. There is no official `firecrawl` CLI binary shipped by the vendor,
so `command -v firecrawl` will normally return nothing and this block is
skipped. If you want Firecrawl's richer scraping here, install the Python
package and call it programmatically:

```python
# pip install firecrawl-py
from firecrawl import FirecrawlApp
app = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])
result = app.scrape_url(page_url, params={"formats": ["markdown"]})
scraped_md = result.get("markdown", "")
```

The current CLI-based path is best-effort and designed to degrade
silently when firecrawl is not installed.

If Firecrawl is not available, the pipeline falls back to the JSON API alone —
no data is lost, just less detail available for scoring.

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

### HuggingFace Papers (v1.10)

ML-specialised paper aggregator with daily-curated rankings. Useful when
PwC has coverage gaps for HF-ecosystem models or recent transformer
work that hasn't been benchmarked yet. Each paper carries `ai_summary`,
`ai_keywords`, and `githubRepo` fields, often saving Phase 3 scoring
work.

**Search by query** (1-3 task-relevant queries):
```bash
curl "https://huggingface.co/api/papers/search?q=<query>&limit=20"
```

Returns JSON list with shape (relevant fields only):
```json
[
  {
    "id": "2403.12345",
    "title": "...",
    "summary": "abstract text",
    "ai_summary": "auto-generated 1-line summary",
    "ai_keywords": ["object detection", "small objects", ...],
    "githubRepo": "username/repo-name",
    "publishedAt": "2024-03-15T..."
  }
]
```

**Trending fallback** if search comes back thin:
```bash
curl "https://huggingface.co/api/daily_papers?limit=30"
```

`daily_papers` is HF's manually-curated daily ML paper list — lower
recall but very high precision. Use when Source 1-5 are returning
mostly off-target.

**Integration with Phase 3 scoring**: HF Papers' `ai_keywords` field
can contribute to "Task relevance" scoring directly (keyword overlap
with task description). Don't treat `ai_summary` as authoritative — it
can be wrong about specific numbers — but as a triage signal it saves
LLM calls.

**Rate limit**: HF API doesn't enforce hard limits but be polite —
batch all queries in a single subagent invocation, don't spawn parallel
hammers.

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

## Phase 2.5 — Cross-source dedup (v1.10)

Source 1-6 + S2 citation expansion can return the same paper multiple
times. BiFPN paper appears in arXiv search, Papers with Code,
Semantic Scholar AND HF Papers — without dedup, Phase 3's "top 20"
ends up being 12 unique papers padded with 8 duplicate variants of 4
papers, weakening the candidate pool.

Run dedup BEFORE Phase 3 scoring.

### Dedup key

Primary key: `arxiv_id` (normalised — strip `arXiv:` prefix, strip
version suffix `v1`/`v2`/etc.). Each source provides this in a
slightly different field:

| Source | Field for arxiv_id |
|---|---|
| arXiv API | `id` (e.g., `http://arxiv.org/abs/2305.18290v1`) |
| Papers with Code | `arxiv_id` (when present; sometimes null) |
| Semantic Scholar | `externalIds.ArXiv` |
| HF Papers | `id` (already in `2305.18290` form) |
| Local PDF | derived from filename if pattern matches; else None |
| S2 citation expansion | `externalIds.ArXiv` |

Fallback key for arxiv_id-less candidates: lowercased title with
whitespace collapsed. Less reliable (typos, subtitle differences) but
catches the common cases.

### Dedup logic

```python
import re

def normalize_arxiv_id(raw):
    """Extract canonical arxiv_id from any of the source-specific shapes.
    Returns None if this paper doesn't have one (rare but possible — e.g.
    NeurIPS-only papers, local PDFs of preprints)."""
    if not raw:
        return None
    s = str(raw)
    # Strip URL prefixes
    s = re.sub(r'^https?://arxiv\.org/abs/', '', s)
    s = re.sub(r'^arXiv:', '', s, flags=re.I)
    # Strip version suffix v1, v2, etc.
    s = re.sub(r'v\d+$', '', s)
    # Verify shape — should be NNNN.NNNNN or older format
    if re.match(r'^(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})$', s):
        return s
    return None


def normalize_title(t):
    if not t:
        return None
    return re.sub(r'\s+', ' ', t.lower().strip())


def dedup_candidates(all_candidates):
    """all_candidates: list of dicts from various sources, each with at minimum
       title and source-specific id fields. Returns deduped list, preserving
       the highest-quality occurrence (preferred order: arXiv > S2 > PwC > HF
       Papers > local PDF) when same paper appears multiple times."""
    SOURCE_PRIORITY = {
        "arxiv": 5, "s2": 4, "pwc": 3, "hf": 2, "local": 1,
        "s2_expansion": 4,  # same as s2
    }
    by_arxiv_id = {}
    by_title = {}
    for cand in all_candidates:
        # Try arxiv_id first
        aid = normalize_arxiv_id(
            cand.get("arxiv_id") or cand.get("id") or
            (cand.get("externalIds") or {}).get("ArXiv")
        )
        title_key = normalize_title(cand.get("title"))
        cand_priority = SOURCE_PRIORITY.get(cand.get("_source"), 0)
        if aid:
            existing = by_arxiv_id.get(aid)
            if not existing or cand_priority > SOURCE_PRIORITY.get(existing.get("_source"), 0):
                # Merge fields from lower-priority duplicate (e.g. keep PwC's
                # github URL even when arXiv version wins as the "primary")
                if existing:
                    for k, v in existing.items():
                        if k not in cand or not cand[k]:
                            cand[k] = v
                by_arxiv_id[aid] = cand
            elif existing:
                # Lower priority version — merge our fields up
                for k, v in cand.items():
                    if k not in existing or not existing[k]:
                        existing[k] = v
        elif title_key:
            existing = by_title.get(title_key)
            if not existing or cand_priority > SOURCE_PRIORITY.get(existing.get("_source"), 0):
                by_title[title_key] = cand
    return list(by_arxiv_id.values()) + list(by_title.values())
```

Each source's collection step must tag entries with `_source`:

```python
# Example — when collecting from HF Papers Source 6
for p in hf_papers_response:
    p["_source"] = "hf"
    candidates.append(p)
```

### Field merging across duplicates

When the same paper appears in 3 sources, dedup keeps the highest-priority
version but **merges fields** from lower-priority versions. This is
important because each source carries different metadata:

- arXiv: clean abstract, structured authors, ID
- PwC: GitHub URL, benchmark results table
- S2: citation count, isInfluential flag
- HF Papers: ai_keywords, ai_summary, githubRepo (sometimes different from PwC's)

After dedup, each surviving candidate has the **union** of metadata
across its source occurrences. Phase 3 scoring then has fuller signal
than any single source provided.

### Sanity checks

After dedup, verify:
- Total candidate count went DOWN (else dedup didn't fire — likely a
  bug in `_source` tagging)
- arXiv-keyed dedup count > title-keyed dedup count (most papers should
  resolve via arxiv_id)
- No survivor has `_source: None` (unmissed-tag bug)

If any check fails, log the imbalance to discoveries.md (subagent
context) but proceed — false-positive dedup is much worse than
false-negative dedup, so when in doubt, **keep both copies** and let
Phase 3 scoring sort it.

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

#### Validate weights URL (D1/D2)

Before writing the weights URL to `base_model.md`, verify the link is alive.
Dead links cause orchestrator Stage 2 to download a 404 and silently fall back
to `yolo26x.pt`, wasting the entire base model selection.

```python
import subprocess

def validate_weights_url(url: str) -> bool:
    """Check if a weights URL is reachable.

    Strategy (D1/D2):
      1. Primary: HTTP HEAD via curl. Fast, no API cost, works for the vast
         majority of direct weights downloads (.pt / .pth / .safetensors).
      2. Fallback: if HEAD returns ambiguous status (e.g. 403 because the
         host doesn't allow HEAD, or a redirect that the default curl didn't
         follow), try a range GET for the first byte.
      3. Firecrawl is NOT used here. It is an HTML-scraping service and
         misclassifies binary downloads. Kept only for PwC deep scrape
         elsewhere in the pipeline.
    """
    if not url or url == "reconstruct via paper2code":
        return True   # paper2code path — no URL to check

    # Step 1: HEAD with redirect-follow
    try:
        r = subprocess.run(
            ["curl", "-sIL", "-o", "/dev/null",
             "-w", "%{http_code}", "--max-time", "15", url],
            capture_output=True, text=True, timeout=20,
        )
        code = r.stdout.strip()
        if code.startswith("2"):
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Step 2: GET first byte (some hosts reject HEAD with 403)
    try:
        r = subprocess.run(
            ["curl", "-sSL", "-o", "/dev/null",
             "-r", "0-0", "-w", "%{http_code}",
             "--max-time", "15", url],
            capture_output=True, text=True, timeout=20,
        )
        code = r.stdout.strip()
        return code.startswith("2")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

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

Location vocabulary for the `Location` field (A5 depends on this being one of
the recognised values — autoresearch will `discard` any module with an
unknown Location rather than guess a default):

- Detector locations: `backbone`, `neck`, `head`, `loss`, `label_assignment`
- Tracker locations: `tracker`, `post_processing`, `association`, `reid`

Use the exact lower-case spelling above. If uncertain, pick the closest match
and note the ambiguity in `Integration notes`.

### Integration mode (v1.7, full_yaml available v1.9)

A per-module field telling autoresearch **how** to apply the module. The
three legal values:

- `hook` (default, v1.6) — autoresearch adds a `USE_<MODULE>` flag in
  `train.py` Section ② and a branch in `inject_modules()`. Good for layer
  swaps, forward wraps, and block replacements that preserve the model's
  layer index layout.
- `yaml_inject` (v1.7) — autoresearch writes `arch_spec.json` and flips
  `ARCH_INJECTION_ENABLED = True`. At train time,
  `weight_transfer.build_custom_model_with_injection` programmatically
  generates a new YAML with inserted layers and transfers pretrained
  weights per-entry strict. Good for inserting attention blocks, SE
  modules, extra heads — anything that shifts downstream indices.
- `full_yaml` (v1.9) — autoresearch writes a complete custom YAML to disk
  (provided by paper-finder in `Integration notes` → `custom_yaml_template`)
  and `arch_spec.json` points at it. At train time,
  `weight_transfer.apply_yaml_spec` builds the model from the custom YAML
  and transfers weights via either an auto class+position heuristic or an
  explicit layer_map_override. Good for **structural rewrites** the
  insertion-based path can't express: BiFPN replacing PAFPN, DETR-style
  decoder replacing the YOLO Detect head, ConvNeXt backbone substitution.

Heuristic for picking the mode from paper text:

| Paper language suggests | Mode |
|---|---|
| "we wrap the existing X with ..." / "after X, apply ..." on forward pass | `hook` |
| "we modify X's forward to ..." | `hook` |
| "we replace the SE block with ..." (same layer, different implementation) | `hook` |
| "we insert <module> after every <class>" / "add <module> at stage N" | `yaml_inject` |
| "we add a new detection head at P2" | `yaml_inject` |
| "we add an auxiliary branch from <layer>" | `yaml_inject` (if the branch is a new layer) |
| "we replace the entire neck with BiFPN" / "we redesign the FPN" | `full_yaml` |
| "our backbone uses ConvNeXt blocks instead of CSP-Darknet" | `full_yaml` |
| "we propose a new decoder with cross-attention on top of the CNN backbone" | `full_yaml` |
| "the head outputs N×M predictions instead of 4-bbox + class" | `full_yaml` |

The mode-selection ladder, in order of preference (each rule fires the first
time it matches):

1. Loss / regularizer / scheduler change → `hook`
2. Same layer count, only forward behaviour differs → `hook`
3. New layers inserted but base structure preserved → `yaml_inject`
4. Backbone, neck, OR head substantively replaced → `full_yaml`

When both modes could apply, prefer `hook` over `yaml_inject` over
`full_yaml` — simpler is cheaper to debug. Reserve `full_yaml` for cases
where the paper's structural change cannot be expressed as a list of
insertions (i.e. the change deletes or restructures, not just adds).

**Asymmetric cost of misclassification (v1.7.1, refined v1.7.7).** If you
can't decide, **prefer `yaml_inject`**. The failure modes are not
symmetric:

- **Hook wrongly applied to a yaml_inject-needing module** (e.g. CBAM tagged
  hook, inserted via forward monkey-patch): training completes with no
  crash, loss looks reasonable, metrics match baseline — a whole
  `TIME_BUDGET` consumed for nothing. autoresearch discards it with no
  warning that this was a misclassification.
- **yaml_inject wrongly applied to a hook-only module** (e.g. a loss-swap
  module tagged yaml_inject with no layer to insert): the insertion spec
  typically fails either at YAML-parse time or at the per-entry strict
  weight transfer — autoresearch gets a clear crash, Step 5.5 classifies
  as `unfixable_*`, the experiment is discarded AND a `discoveries.md`
  entry records the misclassification for human review.

Loud failures are better than silent ones. When unsure, tag `yaml_inject`.

**v1.7.7 — yaml_inject head reference shifting (Fix #13).** Versions
v1.7 through v1.7.6 had a latent bug in `weight_transfer.generate_custom_yaml`
where head's absolute `Concat [-1, N]` references were NOT updated when an
insertion in backbone/neck shifted downstream layer indices. Symptom: the
inserted CBAM/SE/etc. would build successfully and pass weight transfer,
then crash on the FIRST forward pass with
`RuntimeError: Sizes of tensors must match except in dimension 1`.

**This is fixed in v1.7.7.** `update_head_refs` now runs as part of
`generate_custom_yaml` and shifts every absolute from-reference in head
and neck by the count of insertions before it. yaml_inject for
backbone/neck modules now works end-to-end. Existing modules.md entries
tagged yaml_inject that were previously failing (and were workaround-tagged
as `hook` in user's modules.md) can be returned to `yaml_inject`.

If you tagged a yaml_inject module as `hook` between v1.7 and v1.7.6 as
a workaround for this bug, the original `yaml_inject` choice is now the
correct one and should be restored.

### yaml_inject modules need injection spec fields

When you pick `yaml_inject`, the module's `Integration notes` in modules.md
**must** include the injection spec — otherwise autoresearch has nothing to
put in `arch_spec.json`. Required fields:

```
### Integration notes
module_class:  <ClassName matching a Lazy* class in custom_modules.py>
position:      <after_class: ClassName>   OR   <at_index: N>
scope:         <backbone | neck | head | all>
yaml_args:     [<first_is_channel_hint>, <optional_extras>]
module_kwargs: {<optional_kwargs_dict>}
```

- `module_class` — the name of the lazy wrapper you're writing (paper2code
  output + a Lazy* wrapper on top, per `train-script-spec.md § Lazy-wrapper
  contract`).
- `position.at_index` — always base-YAML index. `weight_transfer` handles
  offsets when multiple insertions apply.
- `scope` — enforces where in the model the module can go. For `at_index`,
  scope is an assertion (out-of-scope raises).

#### v1.8 — base-aware `after_class` selection

The class name in `after_class:` MUST match a real class in the actual
base model's YAML. Different YOLO families use different block class
names; using a YOLOv8-era default (`C2f`) when the base is YOLO11 / YOLO26
(which use `C3k2`) silently produces "no matches" failures or wrong
insertion positions.

**Before writing any insertion**, read `base_model.md` and check which
backbone family is in use. The mapping:

| Base model family | Backbone block class | Notes |
|---|---|---|
| YOLOv5 (v5/v5n/v5s/v5m/v5l/v5x) | `C3` | older, used residual connection inside |
| YOLOv8 (v8/v8n/v8s/v8m/v8l/v8x) | `C2f` | added second branch fusion |
| YOLOv9 (v9, v9c, etc.) | `RepNCSPELAN4` | ELAN-derived block |
| YOLOv10 (v10, v10n, v10s, etc.) | `C2f` | inherited from v8 |
| YOLO11 (yolo11, yolo11n, etc.) | `C3k2` | C3-derived with k2 kernel |
| YOLO26 (yolo26, yolo26x, etc.) | `C3k2` | inherited from YOLO11 |

For non-listed families (RT-DETR, YOLOv6, etc.), open the actual base
YAML and look at the backbone block class names directly. If unsure,
prefer `at_index` over `after_class` — `at_index` is unambiguous.

The `Conv` class is stable across all YOLO families and safe to use
without family-aware checks.

```python
# Helper to check the base model's backbone block class before writing
# insertions. Defensively reads base_model.md for the model name.
import re, pathlib
base_md = pathlib.Path("base_model.md").read_text()
# Extract model name from a line like "**Selected**: YOLO26X (user-specified)"
m = re.search(r"\*\*?Selected\*\*?:\s*([A-Za-z0-9_.-]+)", base_md)
base_name = m.group(1).lower() if m else ""

if "yolo26" in base_name or "yolo11" in base_name:
    backbone_block = "C3k2"
elif "yolov9" in base_name:
    backbone_block = "RepNCSPELAN4"
elif "yolov5" in base_name:
    backbone_block = "C3"
else:
    backbone_block = "C2f"   # YOLOv8 / v10 / fallback
```

Without these fields, paper-finder should leave `Integration mode` as `hook`
or mark the module `paper2code: no (incomplete)` so the user knows to fill
in the details manually before autoresearch picks it up.

### v1.9 — full_yaml modules need a custom YAML template

When you pick `full_yaml`, the module's `Integration notes` must include a
**complete YAML template** that the agent will write to disk during apply.
Required fields:

```
### Integration notes
custom_yaml_template: |
  nc: <num_classes — autoresearch will overwrite from train.py's NUM_CLASSES>
  scales:
    n: [...]    # paper's depth/width multipliers per scale
  backbone:
    - [-1, 1, Conv, [64, 3, 2]]
    - ...
  head:
    - [-1, 1, BiFPN, [256, 3]]
    - ...
layer_map_strategy: <auto | override>
layer_map_override: <only when strategy=override; list of {base_idx, custom_idx}>
transfer_scope:    <backbone | backbone+neck | full>
```

#### Choosing `layer_map_strategy`

| Use `auto` when | Use `override` when |
|---|---|
| Backbone preserved exactly; only neck or head replaced | Custom YAML renames classes (Conv → SPDConv) |
| Class names unchanged for the layers you want transferred | Custom YAML reorders or duplicates layer types |
| Number of each preserved class matches between base and custom | You want to skip specific base layers from transfer |

Default to `auto`. The auto heuristic is class-name + structural-position
matching with a monotonic cursor — it works for the common case of
"replace neck, keep backbone". Use `override` when you can predict auto
will mis-pair (e.g. base has 6 Conv but custom has 8 Conv at different
roles).

#### Choosing `transfer_scope`

| Scope | Transfer FROM these base layers | When to use |
|---|---|---|
| `backbone` | backbone only | Default. Head class counts may differ; neck topology may differ; safest. |
| `backbone+neck` | backbone + everything before first Detect | When ONLY the head is replaced (e.g. DETR decoder). |
| `full` | all layers | Rarely correct — equivalent to insertions mode minus head ref shifting. Almost always wrong for full_yaml; use insertions mode instead. |

#### When NOT to use full_yaml

Even when paper text suggests structural change, prefer `yaml_inject` if:

- The change is "insert N modules in the existing structure" (use yaml_inject
  with a list of insertions; v1.7.7's `update_head_refs` handles index
  shifting correctly)
- The change is "swap module class X for module class Y at the same
  position" (use yaml_inject with explicit `at_index` insertion of the
  new class — but accept that v1.9 doesn't yet support REPLACE semantics
  for yaml_inject, only INSERT; the workaround is full_yaml or a hook
  that wraps the existing layer)

Reserve `full_yaml` for cases where the structure substantively changes:
neck topology, head architecture, or backbone family.

#### Example full_yaml Integration notes (BiFPN replacing PAFPN)

```
### Integration notes
custom_yaml_template: |
  nc: 80
  scales:
    n: [0.50, 0.25, 1024]
    s: [0.50, 0.50, 1024]
    m: [0.67, 0.75, 768]
    l: [1.00, 1.00, 512]
    x: [1.00, 1.25, 512]
  backbone:
    - [-1, 1, Conv, [64, 3, 2]]
    - [-1, 1, Conv, [128, 3, 2]]
    - [-1, 3, C3k2, [128, True]]
    - [-1, 1, Conv, [256, 3, 2]]
    - [-1, 6, C3k2, [256, True]]
    - [-1, 1, Conv, [512, 3, 2]]
    - [-1, 6, C3k2, [512, True]]
    - [-1, 1, Conv, [1024, 3, 2]]
    - [-1, 3, C3k2, [1024, True]]
    - [-1, 1, SPPF, [1024, 5]]
  head:
    - [[4, 6, 9], 1, BiFPN, [256, 3]]      # paper's weighted bidirectional fusion
    - [[-1], 1, BiFPN, [256, 3]]
    - [[-1], 1, BiFPN, [256, 3]]
    - [[-3, -2, -1], 1, Detect, [nc]]
layer_map_strategy: auto
transfer_scope: backbone
```

The `auto` strategy will pair the 10 backbone Conv/C3k2/SPPF layers
between base and custom (identical class sequence). The 4 head layers
(BiFPN × 3 + Detect) won't be paired since base has Conv/Concat/Detect
and custom has BiFPN/Detect — that's expected: with `transfer_scope=backbone`,
head is intentionally excluded.

If the user is pinning a non-standard base (`preferred_base_model: YOLO26X`),
copy the base's actual backbone YAML verbatim into `custom_yaml_template`'s
backbone section. The auto strategy needs class names to match the base.

### v1.9 — Resource impact tagging

When a module is likely to push GPU memory beyond the default `BATCH_SIZE`'s
budget, set `resource_impact` so autoresearch can preemptively halve
BATCH_SIZE. This avoids the silent CPU `TaskAlignedAssigner` fallback (real
run's Loop 7: EMA at batch=64 silently moved to CPU, training 3-7× slower,
"untrainable in equal-budget regime" — discoveries.md `resource_constraint`
category).

The 3 known tags:

| Tag | Meaning | Auto-action |
|---|---|---|
| `vram_4x` | ~4× baseline VRAM (P2 head, dense full-image attention, large transformer head) | autoresearch halves BATCH_SIZE twice (×0.25) |
| `vram_2x` | ~2× baseline VRAM (single attention layer added; small structural change) | autoresearch halves BATCH_SIZE once (×0.5) |
| `cpu_fallback_risk` | Known to trigger CPU assigner fallback at default batch even without big VRAM signature (e.g. matmul-heavy modules at high IMGSZ) | autoresearch halves BATCH_SIZE once (×0.5) and logs |

Set the field by inspecting the paper's design choices:

| Module description signal | Suggested `resource_impact` |
|---|---|
| "we add a P2 detection branch" / extra detection scale | `vram_4x` |
| "global self-attention over feature map" / dense attention | `vram_4x` |
| "transformer block on Pk feature" (k ≤ 4) | `vram_4x` (P2/P3 features are large) |
| "transformer block on Pk feature" (k ≥ 5) | `vram_2x` (smaller features) |
| Single SE/CBAM attention block per stage | `vram_2x` |
| "we wrap one Conv with attention" / single-layer change | (omit field — no auto-halve) |
| Loss-only changes (WIoU, Slide Loss, focal variants) | (omit field) |
| Hyperparameter changes (mixup, copy-paste, lr) | (omit field) |

When in doubt, **prefer to omit** the field — under-tagging means the
experiment runs at full batch and may CPU-fallback (loud signal: low it/s
in `logs/loop_<N>.log`); over-tagging unnecessarily halves batch and slows
the experiment without need. Loud failure beats silent over-correction.

Format in modules.md:

```
- **resource_impact**: vram_4x
```

Field is optional — modules without `resource_impact` get the default
behaviour (no auto-halve).


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
        "Location":    "backbone",                        # see Location vocabulary above
        "Improves":    "small objects",                   # free text; describes what it targets
        "Complexity":  "low",                             # low / medium / high
        "paper2code":  "yes",                             # yes / yes (GitHub repo: <url>) / no (not on arXiv) / no (no public repo)
        "pdf_path":    "<local pdf path>",                # only set for local PDFs
        "Status":      "pending",                         # default if omitted
        "Integration mode": "hook",                       # v1.7: hook / yaml_inject / full_yaml; default hook
    },
    "sections": {
        "What it does":     "<2–3 sentences from abstract, in your own words>",
        "Integration notes": (
            "Where does this go in train.py? Use the vocabulary from "
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
        ),
        "paper2code command": "/paper2code https://arxiv.org/abs/<id>\nExtract: `<ClassName>` from `src/model.py`",
    },
})
```

**yaml_inject variant (v1.7).** When `Integration mode: yaml_inject`, the
`Integration notes` section must instead carry the injection spec — the
hook-mode USE_* / Section ③ language doesn't apply. Template:

```python
mm.append_module("modules.md", {
    "name": "CBAM after backbone Convs",
    "fields": {
        "Paper":       "CBAM: Convolutional Block Attention Module",
        "arXiv":       "https://arxiv.org/abs/1807.06521",
        "Location":    "backbone",
        "Complexity":  "medium",
        "paper2code":  "yes",
        "Status":      "pending",
        "Integration mode": "yaml_inject",                # v1.7 — triggers weight_transfer path
    },
    "sections": {
        "What it does": "Channel + spatial attention block inserted after each Conv.",
        "Integration notes": (
            "yaml_inject spec — autoresearch extracts these fields verbatim to "
            "write arch_spec.json. All five are REQUIRED; omit any and "
            "autoresearch will discard the module.\n"
            " - module_class:  LazyCBAM\n"
            " - position:      after_class: Conv\n"
            " - scope:         backbone\n"
            " - yaml_args:     [64]\n"
            " - module_kwargs: {\"kernel_size\": 7}\n\n"
            "The lazy wrapper must conform to train-script-spec.md § "
            "Lazy-wrapper contract: side-effect-free __init__, inner module "
            "built on first forward from x.shape[1], .to(x.device) at build "
            "time. paper2code output goes into custom_modules.py; we add "
            "LazyCBAM as a thin wrapper around it."
        ),
        "paper2code command": "/paper2code https://arxiv.org/abs/1807.06521\nExtract: `CBAM` from `src/model.py`",
    },
})
```

Field reference for `position`:
- `after_class: <ClassName>` — insert after every layer of that class in `scope`
- `at_index: <N>` — insert after base-yaml index N (helper handles offset);
  N must be within `scope` or raises at train time

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

Base model selected: <n>
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
