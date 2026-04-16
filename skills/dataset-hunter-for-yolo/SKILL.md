---
name: dataset hunter for yolo
description: >
  Autonomous dataset search, download, format conversion, and pretrain loop for YOLO object
  detection models. Searches public sources (HuggingFace Datasets, Roboflow Universe, COCO,
  OpenImages, Papers with Code), downloads all available object detection datasets, converts
  them to YOLO format, merges them, runs pretrain, and self-evaluates the result. Trigger
  when the user wants to find public datasets, build a pretrain corpus, or says "hunt datasets",
  "find pretrain data", "search for object detection datasets", "download public datasets".
---

# Dataset Hunter for YOLO — Autonomous Search, Download, Convert, Pretrain

Autonomously searches public sources for object detection datasets, downloads them, converts to
YOLO format, merges into a pretrain corpus, runs pretrain on the target model, and self-evaluates.

**Objective:** Build the largest usable pretrain corpus from public sources, then pretrain the
YOLO model and report whether pretraining improved downstream performance.

Shared files read/written by this skill (`pipeline_state.json`, `base_model.md`,
`pretrain_eval.json`) have their schemas documented in
`<skills_dir>/shared/file-contracts.md` — read it once at Phase 1.

---

## Sources to search (in order)

1. **HuggingFace Datasets** — Hub API, tag `object-detection`
2. **Roboflow Universe** — public export API (requires free API key)
3. **COCO 2017** — direct URL download
4. **OpenImages v7** — `openimages` CLI downloader (lighter than fiftyone)
5. **Papers with Code** — only datasets that have a direct, parseable download URL

---

## Phase 1 — Setup

### 0. Load evaluation config

Before anything else, read the evaluation config so Phase 6 (self-eval) knows what
metric to compare. If run via orchestrator, most of this is already in
`pipeline_state.json`; load either or both as available:

```python
import yaml, json, pathlib

state = json.loads(pathlib.Path("pipeline_state.json").read_text()) \
    if pathlib.Path("pipeline_state.json").exists() else {}

cfg = yaml.safe_load(pathlib.Path("research_config.yaml").read_text()) \
    if pathlib.Path("research_config.yaml").exists() else {}

ev  = cfg.get("evaluation", {})
dhp = cfg.get("dataset_hunter", {}).get("pretrain", {})

# Dataset hunter's self-eval metric is deliberately independent of
# autoresearch's PRIMARY. Pretrain is about detector backbone quality, and the
# natural measurement of that is detection mAP — regardless of whether
# autoresearch's downstream task is detection, tracking, or something else.
# User can override via dataset_hunter.pretrain.eval_metric, but the default
# is val_mAP50_95.
EVAL_METRIC = dhp.get("eval_metric", "val_mAP50_95")

# Parsing machinery is shared with autoresearch — EVAL_METRIC must have a
# pattern defined in evaluation.parsing.patterns. If it does not, the user
# either needs to add it or override eval_metric to match an existing pattern.
PARSE_SOURCE   = ev.get("parsing", {}).get("source", "stdout")
PARSE_PATTERNS = ev.get("parsing", {}).get("patterns", {})
if EVAL_METRIC not in PARSE_PATTERNS:
    raise RuntimeError(
        f"dataset_hunter eval_metric {EVAL_METRIC!r} has no pattern in "
        f"evaluation.parsing.patterns. Add one, or set "
        f"dataset_hunter.pretrain.eval_metric to a metric that does."
    )

# Minimize set applies to the eval metric's direction too
MINIMIZE = set(ev.get("metrics", {}).get("minimize", []))

IMPROVEMENT_THRESHOLD = state.get("pretrain_improvement_threshold") \
    or dhp.get("improvement_threshold", 0.002)
```

**Why not use `evaluation.metrics.primary`?** For a tracking task, PRIMARY is
`HOTA`, but HOTA does not exist until `track.py` runs, and dataset hunter's
self-eval does not run `track.py` — it just runs the detector training twice
(with and without pretrained weights) and compares detector quality. Using
`val_mAP50_95` is the right question for "did pretrain improve the detector?".

Phases 5 and 6 derive `pretrain.py`, `eval_a.py`, and `eval_b.py` from the
project's `train.py` by patching Section ② variables. That machinery assumes
the four-section layout documented in
`<skills_dir>/shared/train-script-spec.md`. Read the spec once now so you can
trust the patch helper later:

```bash
cat "$SKILLS_DIR/shared/train-script-spec.md"
```

Orchestrator already scaffolded a spec-compliant `train.py` at its Stage 0
Step 6, so the file is valid on entry.

### 1. Confirm with user

- Target model weights path (e.g. `weights/yolo26x.pt`)
- Pretrain output directory (default: `pretrain_data/`)
- Disk budget in GB — **required, no default** — downloads can easily exceed 100 GB
- Roboflow API key (free at roboflow.com — needed for Universe exports)
- `TIME_BUDGET` for pretrain in seconds (suggest 21600 = 6 hours)
- `LOCAL_DATASETS_DIR` — path to a local folder of pre-converted YOLO datasets
  (optional; read from `pipeline_state.local_datasets_dir` if not provided directly)
  Each subdirectory must contain a `readme.md` describing the dataset.

### 2. Create directory structure

```
pretrain_data/
├── raw/          # original downloaded files, one subdir per dataset
├── converted/    # each dataset converted to YOLO format, one subdir per dataset
├── merged/
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   ├── labels/val/
│   └── classes.txt
└── dataset.yaml
```

```bash
mkdir -p pretrain_data/{raw,converted,merged/{images,labels}/{train,val}}
```

### 3. Check available disk space

```bash
DISK_FREE_GB=$(df -BG pretrain_data/ | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "Free space: ${DISK_FREE_GB}GB  Budget: ${DISK_BUDGET_GB}GB"
if [ "$DISK_FREE_GB" -lt "$DISK_BUDGET_GB" ]; then
  echo "WARNING: free space is less than requested budget — adjust budget"
fi
```

Store `DISK_BUDGET_GB` as a variable and check remaining space before each download.

### 4. Initialise `hunt_log.tsv` (never git add)

```
source	dataset_name	images	classes	size_mb	status	notes
```

---

## Phase 2 — Search & Download

For each source: search → check size → download → log.

**Timing expectations:** Individual dataset downloads routinely take 10–60 minutes
depending on size. COCO alone is ~25 GB and takes 20–40 min on a typical
connection. OpenImages subsets can take even longer. **This is normal — do not
treat a long-running download as a failure.** Only skip a dataset if the download
process exits with an error code or produces no output file after the timeout.

**Download timeout:** Use `timeout 3600` (1 hour) per individual dataset download.
If a download exceeds 1 hour, kill it and log `skip (download timeout 1h)`. Under
1 hour, **wait patiently and do not interrupt**.

```bash
# Example: HuggingFace git clone with 1-hour timeout
timeout 3600 git clone --depth=1 https://huggingface.co/datasets/<n> \
    pretrain_data/raw/<safe_name>
if [ $? -eq 124 ]; then
    echo "skip (download timeout 1h)"
fi
```

**Skip a dataset and log `skip` if:**
- Download fails after 2 retries
- Estimated size would exceed remaining disk budget
- No usable annotation file found (images only, no bounding boxes)

Check remaining budget before each download:
```bash
USED_GB=$(du -sB1G pretrain_data/raw/ | awk '{print $1}')
REMAINING_GB=$(( DISK_BUDGET_GB - USED_GB ))
```

---

### Source 0 — Local datasets (always runs first)

If `LOCAL_DATASETS_DIR` is set and the directory exists:

```bash
find "<LOCAL_DATASETS_DIR>" -mindepth 1 -maxdepth 1 -type d | sort
```

For each subdirectory found:

1. **Read `readme.md`** to extract dataset metadata:
   ```python
   import pathlib, re

   def parse_readme(ds_dir: pathlib.Path) -> dict:
       text = (ds_dir / "readme.md").read_text(errors="ignore")
       return {
           "name":   ds_dir.name,
           "path":   str(ds_dir),
           "task":   _extract(text, r"[Tt]ask[:\s]+(.+)"),
           "format": _extract(text, r"[Ff]ormat[:\s]+(.+)"),
           "labels": _extract_list(text, r"[Ll]abels?[:\s]+(.+)"),
           "splits": {
               "train": str(ds_dir / "images" / "train"),
               "val":   str(ds_dir / "images" / "val"),
           },
           "readme_text": text[:2000],   # keep first 2000 chars as context
       }
   ```
   If `readme.md` is missing → log `skip (no readme.md)` and continue.

2. **Check task compatibility** — compare the task/labels from readme.md against
   the current task description. Accept the dataset if:
   - Task mentions object detection, bounding box, or detection
   - At least one label name overlaps with the task's detection targets, OR
   - The task description mentions the same domain (aerial, drone, UAV, etc.)
   - If uncertain → accept anyway (pretrain on diverse data is generally beneficial)

3. **Verify YOLO structure exists:**
   ```bash
   [ -d "<ds_dir>/images/train" ] && [ -d "<ds_dir>/labels/train" ]      && echo "OK" || echo "MISSING — skip"
   ```
   If structure is missing → log `skip (no images/labels dirs)`.

4. **Count images and estimate size:**
   ```bash
   IMG_COUNT=$(find "<ds_dir>/images" -type f | wc -l)
   SIZE_MB=$(du -sm "<ds_dir>" | awk '{print $1}')
   ```

5. **Copy to `pretrain_data/converted/<dataset_name>/`** (symlink if disk is tight):
   ```python
   import shutil, pathlib

   src = pathlib.Path(ds_dir)
   dst = pathlib.Path(f"pretrain_data/converted/{src.name}")

   if not dst.exists():
       # Hard copy to keep pretrain_data self-contained
       shutil.copytree(src, dst)

   # Write data.yaml if not already present
   if not (dst / "data.yaml").exists():
       # Parse class names from labels in readme or from label files directly
       label_files = sorted(dst.glob("labels/train/*.txt"))[:100]
       class_ids = set()
       for lf in label_files:
           for line in lf.read_text().splitlines():
               parts = line.split()
               if parts: class_ids.add(int(parts[0]))
       # Use readme label names if available, else fallback to class_0, class_1 ...
       names = meta["labels"] if meta["labels"] else [f"class_{i}" for i in sorted(class_ids)]
       (dst / "data.yaml").write_text(
           f"path: {dst}\ntrain: images/train\nval: images/val\n"
           f"nc: {len(names)}\nnames: {names}\n"
       )
   ```

6. Log to `hunt_log.tsv`:
   ```
   local   <dataset_name>   <IMG_COUNT>   <nc>   <SIZE_MB>   local_copied   <task from readme>
   ```

Local datasets are processed **before** any network downloads. They do not count
against the disk budget (they are already on disk).

---

### Source 1 — HuggingFace Datasets

Search for object detection datasets via the Hub API:
```python
from huggingface_hub import list_datasets
results = list_datasets(tags="object-detection", limit=500)
candidates = [d for d in results if d.id]
```

For each candidate, check dataset card for annotation format before downloading.
Download with streaming to avoid loading everything into RAM:
```python
from datasets import load_dataset
try:
    ds = load_dataset(name, split="train", streaming=False, trust_remote_code=False)
    ds.save_to_disk(f"pretrain_data/raw/{safe_name}")
except Exception as e:
    # Fall back to git clone for large repos
    pass
```

If `load_dataset` fails or dataset is >10GB, use git clone with LFS:
```bash
GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/datasets/<name> \
    pretrain_data/raw/<safe_name> --depth=1
```

Handle datasets with no val split: if only `train` split exists, reserve 10% as val manually
after conversion (split by filename hash, not randomly, for reproducibility).

---

### Source 2 — Roboflow Universe

Roboflow's public export API requires a free API key. Without it, export URLs are not accessible.

Search for public object detection datasets:
```python
import requests
headers = {"Authorization": f"Bearer {ROBOFLOW_API_KEY}"}
# Search projects
resp = requests.get(
    "https://api.roboflow.com/",
    params={"api_key": ROBOFLOW_API_KEY}
)
```

For each public dataset found, export in `yolov8` format (already YOLO — skip conversion):
```python
export_url = f"https://app.roboflow.com/{workspace}/{project}/{version}/export/yolov8"
resp = requests.get(export_url, headers=headers)
download_link = resp.json()["export"]["link"]
# Download the zip
wget.download(download_link, f"pretrain_data/raw/roboflow_{project}.zip")
```

If `ROBOFLOW_API_KEY` is not provided: **skip this source entirely**, log `skip (no api key)`.

---

### Source 3 — COCO 2017

Estimated size: ~25 GB total. Check budget before starting.

```bash
mkdir -p pretrain_data/raw/coco && cd pretrain_data/raw/coco

# Download with resume support (-c flag)
wget -c http://images.cocodataset.org/zips/train2017.zip        # ~18 GB
wget -c http://images.cocodataset.org/zips/val2017.zip          # ~1 GB
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip  # ~250 MB

for f in *.zip; do unzip -q -o "$f" && rm "$f"; done
```

---

### Source 4 — OpenImages v7

Use the `openimages` pip package — much lighter than fiftyone, no MongoDB dependency:
```bash
pip install openimages --quiet
```

Calculate max samples from remaining disk budget (each image ~200KB average):
```python
MAX_SAMPLES = min(50000, int(REMAINING_GB * 1024 * 1024 / 200))
```

Download detection subset:
```bash
python -m openimages download \
    --dataset_dir pretrain_data/raw/openimages \
    --csv_dir pretrain_data/raw/openimages/csv \
    --classes "" \
    --annotation_types detections \
    --max_images ${MAX_SAMPLES}
```

If `openimages` package is unavailable, fall back to direct CSV download:
```bash
# Download class descriptions and bbox annotations
wget -c https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv \
    -P pretrain_data/raw/openimages/
wget -c https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv \
    -P pretrain_data/raw/openimages/
wget -c https://storage.googleapis.com/openimages/v6/validation-annotations-bbox.csv \
    -P pretrain_data/raw/openimages/
```
Then download images listed in the CSV using `aws s3 cp --no-sign-request`.

---

### Source 5 — Papers with Code

Only attempt datasets that have a **direct, parseable download link** (zip/tar).
Do not attempt to scrape arbitrary external sites.

```bash
# Fetch the dataset list API (JSON, not HTML)
curl "https://paperswithcode.com/api/v1/datasets/?task=object-detection&page=1" \
    -H "Accept: application/json" > /tmp/pwc_datasets.json
```

For each dataset in the response:
- Check if `url` field points to a direct file (ends in `.zip`, `.tar`, `.tar.gz`)
- If yes: download with `wget -c`
- If no (points to a webpage): log `skip (no direct download)` and move on

---

## Phase 3 — Format Conversion

Convert every downloaded dataset to YOLO format. Write converted files to
`pretrain_data/converted/<dataset_name>/images/{train,val}/` and `labels/{train,val}/`.

YOLO label format per line: `<class_id> <cx> <cy> <w> <h>` (normalised [0,1])
Class IDs are **local** at this stage — global remapping happens in Phase 4.

### Conversion table

| Source format | Approach |
|---------------|----------|
| COCO JSON | write and run `scripts/coco2yolo.py` inline |
| Pascal VOC XML | write and run `scripts/voc2yolo.py` inline |
| HuggingFace `datasets` | iterate rows, extract bbox + label fields |
| OpenImages CSV | join bbox CSV with class descriptions CSV |
| Already YOLO (Roboflow) | copy directly, no conversion needed |
| Already YOLO (local dataset) | already in `pretrain_data/converted/` from Source 0 — skip |

### Val split for datasets without val

If a dataset has no val split after conversion:
```python
import hashlib
all_images = sorted(Path("converted/<n>/images/train").glob("*"))
# Deterministic 90/10 split by filename hash
val_images = [p for p in all_images
              if int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 10 == 0]
# Move val images + labels to val/
```

### Write per-dataset data.yaml

```yaml
path: pretrain_data/converted/<dataset_name>
train: images/train
val: images/val
nc: <local_num_classes>
names: [<local class names>]
```

---

## Phase 4 — Merge

### Step 1 — Build global class list

Collect all class names from every `converted/*/data.yaml`
(includes local datasets copied in Source 0 — their data.yaml was written there):
```python
all_classes = set()
for yaml_path in Path("pretrain_data/converted").glob("*/data.yaml"):
    cfg = yaml.safe_load(yaml_path.read_text())
    all_classes.update(cfg["names"])
global_classes = sorted(all_classes)   # alphabetical, deterministic
global_id = {name: i for i, name in enumerate(global_classes)}
```

Write `pretrain_data/merged/classes.txt` (one class per line).

### Step 2 — Remap labels and copy files

For each dataset, remap local class IDs → global IDs **before** copying:

```python
for ds_dir in Path("pretrain_data/converted").iterdir():
    cfg = yaml.safe_load((ds_dir / "data.yaml").read_text())
    local_to_global = {i: global_id[name] for i, name in enumerate(cfg["names"])}

    for split in ("train", "val"):
        img_src = ds_dir / "images" / split
        lbl_src = ds_dir / "labels" / split
        img_dst = Path(f"pretrain_data/merged/images/{split}")
        lbl_dst = Path(f"pretrain_data/merged/labels/{split}")

        for img_path in img_src.glob("*"):
            # Prefix filename with dataset name to avoid collisions
            new_name = f"{ds_dir.name}__{img_path.name}"
            shutil.copy2(img_path, img_dst / new_name)

            lbl_path = lbl_src / (img_path.stem + ".txt")
            if lbl_path.exists():
                lines = lbl_path.read_text().strip().splitlines()
                remapped = []
                for line in lines:
                    parts = line.split()
                    local_id = int(parts[0])
                    global_id_val = local_to_global.get(local_id, local_id)
                    remapped.append(f"{global_id_val} {' '.join(parts[1:])}")
                (lbl_dst / new_name.replace(img_path.suffix, ".txt")).write_text(
                    "\n".join(remapped))
```

### Step 3 — Generate dataset.yaml

```python
yaml_content = f"""path: pretrain_data/merged
train: images/train
val: images/val
nc: {len(global_classes)}
names: {global_classes}
"""
Path("pretrain_data/merged/dataset.yaml").write_text(yaml_content)
```

### Step 4 — Print summary

```bash
echo "=== Pretrain Corpus Summary ==="
echo "Train images: $(ls pretrain_data/merged/images/train | wc -l)"
echo "Val images:   $(ls pretrain_data/merged/images/val | wc -l)"
echo "Classes:      $(wc -l < pretrain_data/merged/classes.txt)"
echo "Datasets:     $(ls pretrain_data/converted | wc -l)"
echo "  - local:    $(ls pretrain_data/converted | grep -c '__local__' || ls <LOCAL_DATASETS_DIR> 2>/dev/null | wc -l)"
echo "  - external: $(ls pretrain_data/converted | wc -l)"
echo "Total size:   $(du -sh pretrain_data/merged | cut -f1)"
```

Sanity check — warn if val set is empty:
```bash
VAL_COUNT=$(ls pretrain_data/merged/images/val | wc -l)
if [ "$VAL_COUNT" -eq 0 ]; then
  echo "WARNING: val set is empty — pretrain will run without validation"
fi
```

---

## Phase 5 — Pretrain

> **Direct entry point** — when called from orchestrator's optional pretrain trigger,
> start here and skip Phases 1–4. Inputs are already available:
> - Weights: read from `pipeline_state.base_weights_local` (do not re-resolve)
> - Corpus: `pretrain_data/merged/` already exists (do not re-download)

### Determine pretrain weights

If entering from orchestrator optional pretrain trigger:
- Use `pipeline_state.base_weights_local` directly as `PRETRAIN_WEIGHTS`
- Skip the `base_model.md` resolution block below

Otherwise (normal entry from Phase 4), check `base_model.md` in the project root:

```python
import yaml, pathlib

base_model_md = pathlib.Path("base_model.md")
if base_model_md.exists():
    text = base_model_md.read_text()
    # Extract Weights URL line
    import re
    m = re.search(r"Weights URL.*?:\s*(.+)", text)
    weights_url = m.group(1).strip() if m else None
    if weights_url and weights_url != "reconstruct via paper2code":
        # Download weights if URL is a direct link
        import subprocess
        subprocess.run(["wget", "-c", weights_url, "-P", "weights/"], check=True)
        PRETRAIN_WEIGHTS = f"weights/{pathlib.Path(weights_url).name}"
    elif weights_url == "reconstruct via paper2code":
        arxiv_m = re.search(r"arxiv\.org/abs/([\w.]+)", text)
        if arxiv_m:
            arxiv_id = arxiv_m.group(1)
            print(f"Running paper2code to reconstruct base model: {arxiv_id}")
            # Invoke paper2code skill for this arXiv ID, then find the checkpoint
            ckpts = sorted(pathlib.Path(".").glob("*/checkpoints/*.pt")) +                     sorted(pathlib.Path(".").glob("*/src/*.pt"))
            if ckpts:
                PRETRAIN_WEIGHTS = str(ckpts[-1])
                print(f"Base model reconstructed: {PRETRAIN_WEIGHTS}")
            else:
                print("WARNING: paper2code ran but no .pt found — falling back to yolo26x.pt")
                PRETRAIN_WEIGHTS = "weights/yolo26x.pt"
        else:
            print("WARNING: no arXiv ID in base_model.md — cannot reconstruct")
            PRETRAIN_WEIGHTS = "weights/yolo26x.pt"
    else:
        PRETRAIN_WEIGHTS = "weights/yolo26x.pt"
else:
    # base_model.md not found — paper finder has not run yet
    PRETRAIN_WEIGHTS = "weights/yolo26x.pt"

print(f"Pretrain weights: {PRETRAIN_WEIGHTS}")
```

### Write pretrain.py

Derive `pretrain.py` from the **detection template**, not the project's current
`train.py`. Reasoning:

- Pretrain is about building detector backbone features. Detection-mode training
  + eval is the natural measurement, regardless of what `task_type` the user's
  autoresearch loop is running.
- A tracking `train.py` produces no detection mAP, so patching it cannot yield a
  usable self-eval comparison. Using the detection template sidesteps this.
- The user's `train.py` may also accumulate experimental module state from
  autoresearch. Pretrain should use a clean baseline regardless.

```python
import shutil, pathlib, json

state = json.loads(pathlib.Path("pipeline_state.json").read_text())
DETECTION_TEMPLATE = pathlib.Path(state["skills_dir"]) / "shared" / "templates" / "train.py.detection"

# Stage src for patch_section_2 is always the detection template
SELF_EVAL_SOURCE = "_self_eval_base.py"
shutil.copy(DETECTION_TEMPLATE, SELF_EVAL_SOURCE)
```

The layout, `inject_modules()` hook, and Section ④ main flow of the detection
template are preserved unchanged — see
`<skills_dir>/shared/train-script-spec.md` for the contract.

The spec's note that `TIME_BUDGET` and `SEED` are "locked by orchestrator" refers
to **autoresearch's** loop on `train.py`, not dataset hunter's derived scripts.
Dataset hunter sets its own `TIME_BUDGET` / `SEED` on the copies without touching
the locked `train.py`.

#### Section ② patch helper

```python
import re, pathlib, yaml as _yaml

def patch_section_2(src_path: str, dst_path: str, overrides: dict[str, str]) -> None:
    """Copy src → dst, replacing Section ② top-level assignments.

    Each (name, value_expr) pair replaces a line matching `^<name>\\s*=.*$`
    inside Section ②. The value_expr is inserted verbatim — callers must
    quote strings (use repr()) and not quote numbers.

    Raises RuntimeError if src is not spec-compliant or any variable is
    missing from Section ②.
    """
    src = pathlib.Path(src_path).read_text()
    s2 = src.find("Section ②")
    s3 = src.find("Section ③")
    if s2 < 0 or s3 < 0:
        raise RuntimeError(f"{src_path} is not spec-compliant (see train-script-spec.md)")

    before, section2, after = src[:s2], src[s2:s3], src[s3:]
    for name, value_expr in overrides.items():
        pat = re.compile(rf"(?m)^{re.escape(name)}\s*=.*$")
        new_section2, n = pat.subn(f"{name} = {value_expr}", section2, count=1)
        if n == 0:
            raise RuntimeError(f"{name!r} not found in Section ② of {src_path}")
        section2 = new_section2

    pathlib.Path(dst_path).write_text(before + section2 + after)
```

#### Apply it

```python
# Resolve NUM_CLASSES from the merged pretrain corpus
pretrain_yaml = "pretrain_data/merged/dataset.yaml"
num_classes   = len(_yaml.safe_load(open(pretrain_yaml))["names"])

patch_section_2(SELF_EVAL_SOURCE, "pretrain.py", {
    "DATA_YAML":     repr(pretrain_yaml),
    "WEIGHTS":       repr(PRETRAIN_WEIGHTS),
    "FREEZE_LAYERS": "0",
    "CKPT_DIR":      f"Path({state['pretrain_ckpt_dir']!r})",
    "TIME_BUDGET":   str(state["pretrain_time_budget"]),
    "NUM_CLASSES":   str(num_classes),
})
```

The helper fails loudly if the source file is not spec-compliant. Since the
source is the project's pinned detection template, that failure would indicate
the template itself is broken — report and stop.

### Run pretrain

```bash
uv run pretrain.py > pretrain.log 2>&1 &
echo $! > pretrain.pid
echo "Pretrain started, PID=$(cat pretrain.pid)"
```

Monitor every 5 minutes:
```bash
tail -n 20 pretrain.log
```

### Find the final checkpoint after pretrain

```bash
PRETRAIN_CKPT=$(ls -t pretrain_ckpt/*.pt | head -1)
echo "Final checkpoint: $PRETRAIN_CKPT"
```

Use `$PRETRAIN_CKPT` in Phase 6.

---

## Phase 6 — Self-Evaluation

Compare finetuning from original weights vs pretrained weights on the real target dataset.

### Step 1 — Run two short finetune jobs

Both derive from the detection template (`SELF_EVAL_SOURCE` set up in Phase 5),
not from the user's `train.py`. This matters for tracking tasks: the user's
`train.py.tracking` has no eval step, so patching it would produce scripts
that train but never emit `val_mAP50_95`. Using the detection template
guarantees both eval scripts produce a comparable detection metric.

The only difference between `eval_a.py` and `eval_b.py` is `WEIGHTS`.

```python
# eval_a.py — finetune from original (pre-pretrain) weights
patch_section_2(SELF_EVAL_SOURCE, "eval_a.py", {
    "WEIGHTS":      repr(state["base_weights_local"]),
    "CKPT_DIR":     'Path("eval_baseline")',
    "TIME_BUDGET":  str(state["self_eval_time_budget"]),
})

# eval_b.py — finetune from the pretrain checkpoint
patch_section_2(SELF_EVAL_SOURCE, "eval_b.py", {
    "WEIGHTS":      repr(str(pretrain_ckpt)),   # from Phase 5
    "CKPT_DIR":     'Path("eval_pretrained")',
    "TIME_BUDGET":  str(state["self_eval_time_budget"]),
})
```

Run sequentially (GPU can only handle one at a time):
```bash
uv run eval_a.py > eval_baseline.log 2>&1
uv run eval_b.py > eval_pretrained.log 2>&1
```

Both logs will contain the same print format as `train.py`, so the parser in
Step 2 handles them uniformly via `evaluation.parsing.patterns`.

### Step 2 — Extract eval metric from both logs

Use the parsing config loaded in Phase 1 Step 0. Note this uses `EVAL_METRIC`
(dataset hunter's choice of detector-quality metric), not `PRIMARY`:

```python
import re, pathlib

def extract_eval_metric(log_path: str) -> float | None:
    """Extract EVAL_METRIC value from a run log using evaluation.parsing config."""
    if PARSE_SOURCE == "stdout":
        pattern = PARSE_PATTERNS.get(EVAL_METRIC)
        if not pattern:
            raise RuntimeError(f"No parsing pattern defined for {EVAL_METRIC}")
        log = pathlib.Path(log_path).read_text()
        m = re.search(pattern, log)
        return float(m.group(1)) if m else None
    elif PARSE_SOURCE == "json":
        # Use ev["parsing"]["json_file"] and ev["parsing"]["json_paths"][EVAL_METRIC]
        ...
    elif PARSE_SOURCE == "csv":
        # Use ev["parsing"]["csv_file"] and ev["parsing"]["csv_columns"][EVAL_METRIC]
        ...

baseline   = extract_eval_metric("eval_baseline.log")
pretrained = extract_eval_metric("eval_pretrained.log")
```

For quick visual inspection, tail each log:

```bash
echo "--- Baseline ---"
tail -n 30 eval_baseline.log
echo "--- Pretrained ---"
tail -n 30 eval_pretrained.log
```

### Step 3 — Decision

Compute delta respecting the metric's direction (larger-is-better vs smaller-is-better),
then compare against `IMPROVEMENT_THRESHOLD` from Phase 1 Step 0. The output
file `pretrain_eval.json` follows the schema in
`<skills_dir>/shared/file-contracts.md § pretrain_eval.json` — orchestrator
Stage 2 reads it and treats `inconclusive` as `use_original`.

```python
if baseline is None or pretrained is None:
    recommendation = "inconclusive"
    delta = None
else:
    if EVAL_METRIC in MINIMIZE:
        # smaller is better: pretrained helps if it went down
        delta = baseline - pretrained
    else:
        # larger is better: pretrained helps if it went up
        delta = pretrained - baseline

    if delta >= IMPROVEMENT_THRESHOLD:
        recommendation = "use_pretrained"
    elif delta <= -IMPROVEMENT_THRESHOLD:
        recommendation = "use_original"
    else:
        recommendation = "inconclusive"   # margin too small to trust

# Write recommendation for orchestrator to read
import json
pathlib.Path("pretrain_eval.json").write_text(json.dumps({
    "eval_metric":     EVAL_METRIC,
    "baseline":        baseline,
    "pretrained":      pretrained,
    "delta":           delta,
    "threshold":       IMPROVEMENT_THRESHOLD,
    "recommendation":  recommendation,
}, indent=2))
```

Decision table (for reference — implemented in the code above):

| delta vs threshold | Recommendation |
|-------|----------------|
| `delta ≥ +threshold` | ✅ `use_pretrained` |
| `-threshold < delta < +threshold` | `inconclusive` — use original to be safe |
| `delta ≤ -threshold` | ⚠️ `use_original` |

`inconclusive` is treated as `use_original` by orchestrator (safer default when
pretrain gave no clear signal).

### Step 4 — Print summary

```python
print(f"""
=== Pretrain Evaluation ===
Eval metric:     {EVAL_METRIC}
Baseline   {EVAL_METRIC}: {baseline}
Pretrained {EVAL_METRIC}: {pretrained}
Delta:                {delta:+.4f} (threshold: {IMPROVEMENT_THRESHOLD})
Recommendation: {recommendation.upper()}

Pretrain corpus:
  Datasets : {num_datasets}
  Images   : {num_images}
  Classes  : {num_classes}
  Disk used: {disk_used_gb:.1f} GB
  Checkpoint: {pretrain_ckpt}
""")
```

The recommendation is also written to `pretrain_eval.json` (Step 3) for the
orchestrator to read — orchestrator does not re-parse the summary text.

---

## Error handling

| Error | Action |
|-------|--------|
| Download 404 / timeout after 2 retries | Log `skip`, continue |
| Remaining disk < next dataset size | Stop downloading, proceed with what was collected |
| Conversion fails for a dataset | Log `conversion_failed`, skip that dataset |
| Val set empty after all merging | Proceed with train-only pretrain, skip val loss logging |
| Pretrain OOM | Halve `BATCH_SIZE` in `pretrain.py`, retry once |
| Pretrain loss = nan | Reduce LR 10×, retry once |
| eval_a or eval_b log missing/empty | Re-run that job before printing summary |
| Roboflow API key missing | Skip Roboflow source, log `skip (no api key)` |

---

## hunt_log.tsv columns

```
source         — huggingface / roboflow / coco / openimages / paperswithcode
dataset_name   — original dataset name
images         — images downloaded
classes        — classes in this dataset
size_mb        — disk size after download
status         — downloaded / converted / merged / skip / conversion_failed / crash
notes          — e.g. "no val split, auto-split 90/10" or "skip (no api key)"
```
