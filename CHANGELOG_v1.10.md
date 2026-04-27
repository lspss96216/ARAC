# Changelog — v1.10

Release type: **Minor — paper-finder source diversification + dedup.**
Adds HuggingFace Papers as Source 6 and a Phase 2.5 cross-source dedup
step. Drop-in on top of v1.9.3.

## Background

This release is a conservative subset of a broader ml-intern integration
proposal that was reviewed and trimmed. The two items shipped here are
the lowest-risk, highest-confidence pieces:

- **HF Papers Source 6** — pure additive. Adds a 6th search source for
  ML-specialised paper ranking. PwC has known coverage gaps for
  HuggingFace-ecosystem models (real run thinking_log: "drone +
  transformer" returned mostly radar-based UAV detection); HF's
  ML-specific ranking improves precision.
- **Phase 2.5 cross-source dedup** — fixes a latent design issue: Phase 3
  scoring's "top 20" was previously polluted with duplicate variants of
  the same paper across sources, weakening the candidate pool. Real
  example: BiFPN paper would appear in arXiv search, Papers with Code,
  Semantic Scholar AND HF Papers — now collapses into one merged entry.

The deferred items from the broader proposal (Phase 3.5 read methodology,
Phase 5 Step 0 citation graph, hyperparam recipe injection) are NOT in
v1.10. Each of those needs a v1.9.3 implementation feedback signal
("paper-finder's current Integration notes are too thin", "stall
expansion can't find new ideas") before justifying the implementation
cost. v1.10 ships first, real-run feedback informs v1.11+ scope.

## What's in v1.10

### Phase 2 — HuggingFace Papers as Source 6

New section in `paper-finder/SKILL.md § Phase 2` after the Local PDF
source. Two endpoints:

```bash
# Targeted search
curl "https://huggingface.co/api/papers/search?q=<query>&limit=20"

# Trending fallback (manually-curated daily ML list)
curl "https://huggingface.co/api/daily_papers?limit=30"
```

Each result carries `ai_summary`, `ai_keywords`, and `githubRepo` —
fields that contribute directly to Phase 3 scoring without LLM calls.
`ai_summary` is treated as triage signal, NOT authoritative (it can be
wrong about specific numbers).

Documented integration notes:
- `daily_papers` is high-precision low-recall — use as fallback when
  Source 1-5 returns thin
- Single subagent invocation should batch all queries (HF API politeness)
- `ai_keywords` overlap with task description is a fast "task relevance"
  proxy

### Phase 2.5 — Cross-source dedup (NEW)

New phase between Phase 2 (collect) and Phase 3 (score). Runs once after
all 6 sources + S2 expansion populate the candidate pool.

Two-tier dedup key:
1. **Primary**: normalised arxiv_id. Each source provides this in a
   slightly different field; helper `normalize_arxiv_id()` collapses
   variants (URL prefix, `arXiv:` prefix, version suffix `v1`/`v2`).
2. **Fallback**: lowercased, whitespace-collapsed title. Less reliable
   but catches papers without arxiv presence.

Source priority for which version "wins" when duplicates collapse:
arXiv (5) > S2 (4) = S2 expansion (4) > PwC (3) > HF (2) > Local (1).
The lower-priority versions don't disappear entirely — their **fields
are merged up** into the surviving entry. So if PwC's record has the
GitHub URL but the arXiv version wins as "primary", the GitHub URL
still survives in the merged candidate.

Sanity checks logged after dedup:
- Candidate count went down (else `_source` tagging bug)
- arxiv_id-keyed dedup count > title-keyed dedup count (shape sanity)
- No survivor has missing `_source` (untagged-at-source bug)

When in doubt, dedup keeps both copies — false-positive dedup is much
worse than false-negative. Phase 3 scoring tolerates near-duplicates
(they get the same score, both make it to top 20, autoresearch picks
the first one and discards-or-keeps; the other lingers in modules.md
with no harm done).

### Phase 2 — `_source` tagging requirement

Every candidate appended in Phase 2 MUST carry `_source` field. Six
valid values: `arxiv` / `pwc` / `s2` / `local` / `hf` / `s2_expansion`.
This drives Phase 2.5's dedup; missing tag makes the candidate invisible
to dedup logic.

This was previously implicit (each subagent prompt mentioned the source)
but never enforced as a record-level field. v1.10 makes it explicit and
required.

## What's NOT in v1.10

Per the broader proposal review, these were deferred pending v1.9.3
real-run feedback:

| Item | Why deferred |
|---|---|
| Phase 3.5 — read methodology HTML | ROI depends on whether v1.9.3's full_yaml `custom_yaml_template` is too thin in practice. Wait for evidence. |
| Phase 5 Step 0 — citation graph expansion | ROI depends on whether stall expansion currently fails to find new ideas. Wait for evidence. |
| Hyperparam recipe injection into modules.md | Rejected — paper hyperparams (lr=0.01 on 2× RTX3090) don't transfer to user's setup (H100 80GB batch=48). Conflicts with v1.8 invariants philosophy + v1.9 resource_impact. |
| `shared/paper_utils.py` HTML parser module | Rejected — paper-finder Phase 5 already uses subagent delegation; subagent can `import httpx` directly. Adding bs4/httpx to `shared/` (currently CPU-deterministic only) breaks the layer's design. |

If v1.9.3 real-run feedback shows Phase 3.5 / 5 Step 0 ROI, those land
in v1.11. The architecture decisions (no `paper_utils.py`, no hyperparam
auto-apply) are settled regardless of feedback.

## Files changed

| File | Change |
|---|---|
| `paper-finder/SKILL.md` | New § HuggingFace Papers (Source 6) ~50 lines; new § Phase 2.5 — Cross-source dedup ~120 lines; Phase 2 opening adds `_source` tagging requirement table |
| `CHANGELOG_v1.10.md` | New (this file) |
| `README.md` | Banner + Versions list |

### Unchanged

Every `shared/` file. No new dependencies. No new test files. v1.10 is
SKILL-only — the changes live in subagent reference code, not in
deterministic Python modules.

`autoresearch/SKILL.md`, `research-orchestrator/SKILL.md`,
`dataset-hunter/SKILL.md` all unchanged.

## Test coverage

| Suite | v1.9.3 | v1.10 |
|---|---|---|
| `shared/test_modules_md.py` | 23 | 23 |
| `shared/test_templates.py` | 3 | 3 |
| `shared/test_weight_transfer.py` | 61 | 61 |
| `shared/test_hook_utils.py` | 13 | 13 |
| `shared/test_invariants.py` | 29 | 29 |
| **Total python tests** | **129** | **129** |
| SKILL.md python snippets | 100 | **102** (+2) |
| YAML examples | 2 | 2 |

No new tests because v1.10's changes live in paper-finder's subagent
delegation path, which isn't unit-testable from this repository (the
subagent runs in a different process, against live external APIs).

The two new SKILL python snippets (HF Papers `httpx` example, dedup
helper functions) are syntax-checked by the existing snippet sweep.

## Upgrade path

Drop-in. paper-finder is invoked at Stage 1 of orchestrator; v1.10's
new sources and dedup behavior take effect on the next paper-finder
run, no state migration needed.

For users with a populated `modules.md` from a v1.9.3 run:
- v1.10's dedup applies only to NEW paper-finder invocations
- Existing modules.md entries are not retroactively deduped
- If you want to clean up duplicates in an existing modules.md, the
  parser at `shared/modules_md.py` doesn't enforce uniqueness — manual
  edit is the path. autoresearch's Priority A iterates pending entries
  in order, so duplicates would just consume extra loops; not a
  correctness issue.

## Operational expectations

After upgrade, the next paper-finder run should produce a `modules.md`
with similar OR HIGHER paper diversity than v1.9.3. Higher because:
- Source 6 adds papers PwC missed
- Dedup makes top 20 more diverse (no wasted slots on duplicates)

If post-upgrade modules.md has LOWER diversity than v1.9.3, that's a
regression — most likely the dedup is over-aggressive (false-positive
title matching). Inspect the merged candidates' `_source` field
distribution; any paper showing as deduped from 4+ sources is a
suspicious signal.

If `daily_papers` returns 503/504 (HF API is occasionally flaky), the
Source 6 logic should fall through cleanly to the other 5 sources —
HF Papers is additive, not required. Verify via:
- Search for any `[paper-finder] HF Papers fetch failed` in
  discoveries.md
- Confirm paper count from other sources is unchanged

## v1.11 outlook

Three candidates, ROI depends on v1.9.3 + v1.10 real-run feedback:

1. **Phase 3.5 read methodology** — if `full_yaml` modules consistently
   discard because `custom_yaml_template` is too thin
2. **Phase 5 Step 0 citation graph** — if stall expansion finds same
   papers repeatedly without new ideas
3. **Per-machine resource calibration** for v1.9 auto-halve — if
   `vram_4x` halve count (currently fixed ×0.25) is wrong for users on
   different GPU configs

None of these have ROI evidence yet. Run v1.10 and see what surfaces.
