"""Verify train.py templates comply with train-script-spec.md.

D3 fix: templates directory is resolved relative to this test file, with
optional PIPELINE_TEMPLATES_DIR env var override. Previous version
hardcoded /home/claude/shared/templates which worked nowhere but the
author's machine.
"""
import os, pathlib, re, sys

REQUIRED_SECTIONS  = ["Section ①", "Section ②", "Section ③", "Section ④"]
REQUIRED_VARIABLES = ["TIME_BUDGET", "SEED", "BATCH_SIZE", "WEIGHTS",
                      "DATA_YAML", "NUM_CLASSES", "CKPT_DIR"]
REQUIRED_FUNCTIONS = ["def inject_modules", "def main"]


def check(path, kind="train"):
    src = pathlib.Path(path).read_text()
    missing = []

    missing += [s for s in REQUIRED_SECTIONS if s not in src]

    # track.py has a different hook name and no detector variables — check only
    # what the spec requires for its role.
    if kind == "train":
        missing += [v for v in REQUIRED_VARIABLES if not re.search(rf"(?m)^{v}\s*=", src)]
        missing += [f for f in REQUIRED_FUNCTIONS if f not in src]
    elif kind == "track":
        # track.py only needs SEED (for RNG) and TIME_BUDGET (spec placeholder)
        for v in ["SEED", "TIME_BUDGET"]:
            if not re.search(rf"(?m)^{v}\s*=", src):
                missing.append(v)
        if "def apply_tracker_modules" not in src:
            missing.append("def apply_tracker_modules")
        if "def main" not in src:
            missing.append("def main")

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
