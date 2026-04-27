"""Verify train.py templates comply with train-script-spec.md.

D3 fix: templates directory is resolved relative to this test file, with
optional PIPELINE_TEMPLATES_DIR env var override. Previous version
hardcoded /home/claude/shared/templates which worked nowhere but the
author's machine.

v1.7 additions:
- ARCH_INJECTION_ENABLED / ARCH_INJECTION_SPEC_FILE in train.py Section ②
- conditional `if ARCH_INJECTION_ENABLED:` branch in train.py main()
"""
import os, pathlib, re, sys

# v1.7.5 — regex tolerates both Unicode circled digits and ASCII digits so
# a template like `# Section 2 — Tunables` is accepted alongside
# `# Section ② — Tunables`. Keeps tests passing when SKILL.md-level regex
# (check_spec_compliance / patch_section_2) likewise tolerates both.
REQUIRED_SECTION_PATS = [
    ("Section 1", re.compile(r"(?mi)^#?\s*Section\s*[①1]\b")),
    ("Section 2", re.compile(r"(?mi)^#?\s*Section\s*[②2]\b")),
    ("Section 3", re.compile(r"(?mi)^#?\s*Section\s*[③3]\b")),
    ("Section 4", re.compile(r"(?mi)^#?\s*Section\s*[④4]\b")),
]
REQUIRED_VARIABLES = ["TIME_BUDGET", "SEED", "BATCH_SIZE", "WEIGHTS",
                      "DATA_YAML", "NUM_CLASSES", "CKPT_DIR",
                      # v1.7 — architecture injection surface
                      "ARCH_INJECTION_ENABLED", "ARCH_INJECTION_SPEC_FILE",
                      # v1.7.7 — explicit optimizer (never 'auto')
                      "OPTIMIZER", "LR0", "MOMENTUM"]
REQUIRED_FUNCTIONS = ["def inject_modules", "def main"]
# v1.7 — main() must branch on ARCH_INJECTION_ENABLED so the weight-transfer
# path is reachable. Presence of the literal substring is a necessary but not
# sufficient check — template_smoke test covers the full flow end-to-end.
REQUIRED_BRANCHES = ["if ARCH_INJECTION_ENABLED", "build_custom_model_with_injection"]

# v1.9.2 — every train.py / track.py must write run.log sentinels so Step 6
# freshness check can detect cross-project run.log pollution.
REQUIRED_SENTINELS = ["__RUN_START__", "__RUN_END__"]
# track.py is the second leg of a tracking pipeline; train.py wrote the
# RUN_START at the top, so track.py only needs RUN_END.
REQUIRED_SENTINELS_TRACK_ONLY = ["__RUN_END__"]


def check(path, kind="train"):
    src = pathlib.Path(path).read_text()
    missing = []

    missing += [name for name, pat in REQUIRED_SECTION_PATS if not pat.search(src)]

    # track.py has a different hook name and no detector variables — check only
    # what the spec requires for its role.
    if kind == "train":
        missing += [v for v in REQUIRED_VARIABLES if not re.search(rf"(?m)^{v}\s*=", src)]
        missing += [f for f in REQUIRED_FUNCTIONS if f not in src]
        missing += [b for b in REQUIRED_BRANCHES if b not in src]
        missing += [s for s in REQUIRED_SENTINELS if s not in src]
    elif kind == "track":
        # track.py only needs SEED (for RNG) and TIME_BUDGET (spec placeholder)
        for v in ["SEED", "TIME_BUDGET"]:
            if not re.search(rf"(?m)^{v}\s*=", src):
                missing.append(v)
        if "def apply_tracker_modules" not in src:
            missing.append("def apply_tracker_modules")
        if "def main" not in src:
            missing.append("def main")
        missing += [s for s in REQUIRED_SENTINELS_TRACK_ONLY if s not in src]

    return missing


if __name__ == "__main__":
    # D3 — resolve templates dir relative to this file (shared/), with env override
    HERE = pathlib.Path(__file__).resolve().parent
    TEMPLATES = pathlib.Path(
        os.environ.get("PIPELINE_TEMPLATES_DIR", HERE / "templates")
    )

    failed = 0
    for path, kind in [
        (TEMPLATES / "train.py.detection", "train"),
        (TEMPLATES / "train.py.tracking",  "train"),
        (TEMPLATES / "track.py.tracking",  "track"),
    ]:
        if not path.exists():
            print(f"✗ {path.name}: file not found at {path}")
            failed += 1
            continue
        missing = check(str(path), kind)
        name = path.name
        if missing:
            print(f"✗ {name}: missing {missing}")
            failed += 1
        else:
            print(f"✓ {name}")
    sys.exit(failed)
