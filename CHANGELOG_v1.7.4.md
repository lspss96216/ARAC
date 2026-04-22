# Changelog — v1.7.4

Release type: **Bug fix (documentation).** Adds a Tie-breaking rule in
`autoresearch/SKILL.md` Step 2 so the loop always has a deterministic
fallback when multiple candidates look equally valid. No code changes,
no schema changes, no test changes. Drop-in on top of v1.7.3.

## The bug

Critical Rule #13 and the `§ Discoveries` section both tell
autoresearch not to stop and ask the user mid-loop. But in practice,
Claude-as-agent still paused to ask "which of these candidates should
I try first?" when Step 2 walked Priority A → E and found multiple
options tied on `find_pending`'s existing sort keys (complexity,
location rank, write order). The rules say "don't ask" but don't say
what to do **instead** when genuinely tied — so Claude's reflex was to
surface the ambiguity to the user.

Observed symptom: after an iteration completed, the agent wrote
something like "你想怎麼走？我有明確方向就直接啟動 iter 9" and waited.
Whole iterations of wall-clock time lost to a question that had an
obviously correct "just pick one" answer.

## The fix

New `§ Tie-breaking — pick one, never ask` subsection in Step 2, placed
right after Priority E and before § Forced architecture exploration.
Gives a **4-step deterministic tie-breaker** for modules.md picks (write
order → lower complexity → preferred_locations → alphabetical) and a
parallel 3-step ladder for non-modules.md ties (lightest change → more
basic setting → alphabetical).

Plus a short "these are NOT tie-breakers" list that names the reasoning
Claude has been using to justify asking — "user would know better",
"this is a strategic choice", "VRAM is tight, want confirmation",
"want to confirm my priority is right" — and marks each as not a valid
reason to stop.

The key framing change: **a deterministic wrong-seeming choice beats
the right choice obtained by asking**, because "asking" costs unbounded
wall-clock while a wrong pick costs one 20-minute iteration that Step 7
discards.

## Why this wasn't already in the skill

Rule #13 handled the "never ask" obligation but assumed the default
behaviour after declining to ask was obvious. It isn't — when you have
two tied modules the instinct to defer to the user is strong unless
the skill explicitly says "pick by this rule, keep going". v1.7.4 makes
the default explicit and names the specific rationalisations to watch
for.

## Files changed

| File | Change |
|---|---|
| `autoresearch/SKILL.md` | New subsection `§ Tie-breaking — pick one, never ask` (~60 lines) inserted between Priority E and § Forced architecture exploration |
| `CHANGELOG_v1.7.4.md` | New (this file) |
| `README.md` | Version bumped |

## Files NOT changed

Everything else byte-identical to v1.7.3 — all code, all templates,
all other SKILL.md, all tests, all examples. This is purely a
documentation fix to a SKILL.md.

## Test coverage

Unchanged from v1.7.3. All 57 python tests + 83 SKILL.md python
snippets still green.

## Upgrade path

Drop-in. No state migration, no behaviour change for code paths. The
next time autoresearch's agent considers asking "which should I try
first?", it now finds an explicit rule telling it to pick deterministically
and keep looping.

## Operational tip

If the agent **still** stops to ask mid-loop after this upgrade, a
valid user reply is simply `continue` (or `繼續` / `go`). Treat such
stops as a bug worth logging to `discoveries.md` yourself — the
Tie-breaking rule was supposed to catch this, and if the agent routed
around it, that's a symptom the rule needs further strengthening in a
future release.
