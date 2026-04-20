---
name: dataset hunter
description: >
  Autonomous dataset search, download, format conversion, and pretrain loop for object
  detection models. Searches public sources (HuggingFace Datasets, Roboflow Universe, COCO,
  OpenImages, Papers with Code), downloads all available object detection datasets, converts
  them to YOLO format, merges them, runs pretrain, and self-evaluates the result. Trigger
  when the user wants to find public datasets, build a pretrain corpus, or says "hunt datasets",
  "find pretrain data", "search for object detection datasets", "download public datasets".
---

# Dataset Hunter — Autonomous Search, Download, Convert, Pretrain

Autonomously searches public sources for object detection datasets, downloads them, converts to
YOLO format, merges into a pretrain corpus, runs pretrain on the target model, and self-evaluates.

**Objective:** Build the largest usable pretrain corpus from public sources, then pretrain the
model and report whether pretraining improved downstream performance.

Shared files read/written by this skill (`pipeline_state.json`, `base_model.md`,
`pretrain_eval.json`) have their schemas documented in
`<skills_dir>/shared/file-contracts.md` — read it once at Phase 1.

---

## Helpers used throughout this skill

### `safe_wget` (A4)

Same contract as orchestrator's helper — never raise on download failure,
clean up zero-byte files, return bool. Every `wget` in this skill must go
through `safe_wget` so a dead URL skips the dataset rather than crashing
the whole hunt.

```python
def safe_wget(url: str, dest: str, timeout_sec: int = 3600) -> bool:
    import subprocess, pathlib
    dest_p = pathlib.Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["wget", "-q", "-c", "--tries=2", "--timeout=60",
             url, "-O", dest],
            timeout=timeout_sec,
        )
        if r.returncode != 0:
            if dest_p.exists() and dest_p.stat().st_size == 0:
                dest_p.unlink()
            return False
    except subprocess.TimeoutExpired:
        if dest_p.exists() and dest_p.stat().st_size == 0:
            dest_p.unlink()
        return False
    return dest_p.exists() and dest_p.stat().st_size > 0
```

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
PARSING_CFG    = ev.get("parsing", {})   # full dict for B1 helper
MINIMIZE       = set(ev.get("metrics", {}).get("minimize", []))
IMPROVEMENT_THRESHOLD = dhp.get("improvement_threshold", 0.002)
```

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
           "readme_text": text[:2000],
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
       shutil.copytree(src, dst)

   if not (dst / "data.yaml").exists():
       label_files = sorted(dst.glob("labels/train/*.txt"))[:100]
       class_ids = set()
       for lf in label_files:
           for line in lf.read_text().splitlines():
               parts = line.split()
               if parts: class_ids.add(int(parts[0]))
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

#### Pre-flight check via Dataset Viewer API

Before downloading any candidate, query the HF Dataset Viewer API to inspect
its schema. This avoids downloading multi-GB datasets only to discover they
have no bounding box annotations — a common waste that previously cost 10–60
minutes per false hit.

```python
import requests

def preflight_check_bbox(repo_id: str) -> dict:
    """Query HF Dataset Viewer API to check dataset schema BEFORE downloading.
    Returns {"has_bbox": bool, "columns": list, "num_rows": int, "reason": str}.
    On API error, returns has_bbox=True (optimistic — don't skip on API failure)."""
    base = "https://datasets-server.huggingface.co"
    result = {"has_bbox": True, "columns": [], "num_rows": 0, "reason": ""}

    try:
        info = requests.get(f"{base}/info", params={"dataset": repo_id}, timeout=15)
        if info.status_code != 200:
            result["reason"] = f"Viewer API returned {info.status_code}"
            return result
        info_data = info.json()

        configs = info_data.get("dataset_info", {})
        if not configs:
            result["reason"] = "no configs in Viewer API"
            return result

        first_config = next(iter(configs))
        splits = configs[first_config].get("splits", {})
        first_split = "train" if "train" in splits else next(iter(splits), None)
        if not first_split:
            result["reason"] = "no splits found"
            return result

        rows_resp = requests.get(
            f"{base}/first-rows",
            params={"dataset": repo_id, "config": first_config, "split": first_split},
            timeout=15,
        )
        if rows_resp.status_code != 200:
            result["reason"] = f"first-rows returned {rows_resp.status_code}"
            return result

        rows_data = rows_resp.json()
        columns = [f["column"]["name"] for f in rows_data.get("features", [])]
        col_types = {f["column"]["name"]: f["column"].get("_type", "")
                     for f in rows_data.get("features", [])}
        result["columns"] = columns
        result["num_rows"] = splits.get(first_split, {}).get("num_examples", 0)

        bbox_names = {"bbox", "bboxes", "boxes", "objects", "annotations",
                      "image", "label", "labels"}
        found_by_name = set(c.lower() for c in columns) & bbox_names
        found_by_type = any("bbox" in str(v).lower() or "BoundingBox" in str(v)
                            for v in col_types.values())

        if not found_by_name and not found_by_type:
            result["has_bbox"] = False
            result["reason"] = f"no bbox columns in {columns}"

    except Exception as e:
        result["reason"] = f"preflight error: {e}"

    return result
```

Apply preflight to each candidate:

```python
for candidate in candidates:
    pf = preflight_check_bbox(candidate.id)
    if not pf["has_bbox"]:
        log_to_hunt_log(candidate.id, status="skip",
                        notes=f"preflight: {pf['reason']}")
        continue

    if pf["num_rows"] > 0:
        est_size_mb = pf["num_rows"] * 0.3   # rough 300KB/image estimate
        if est_size_mb > REMAINING_GB * 1024:
            log_to_hunt_log(candidate.id, status="skip",
                            notes=f"preflight: est {est_size_mb:.0f}MB > budget")
            continue
    # Preflight passed — proceed with download
```

This typically filters out 60–80% of false-hit candidates in seconds, saving
hours of wasted download time. The Viewer API is free, unauthenticated, and
rate-limited to ~100 req/min (sufficient for our use case).

#### Download

For each candidate that passes preflight:
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
GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/datasets/<n> \
    pretrain_data/raw/<safe_name> --depth=1
```

Handle datasets with no val split: if only `train` split exists, reserve 10% as val manually
after conversion (split by filename hash, not randomly, for reproducibility). See
Phase 3 § Val split for datasets without val for the full implementation (B5).

---

### Source 2 — Roboflow Universe

Roboflow's public export API requires a free API key. Without it, export URLs are not accessible.

```python
import os, requests

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")  # loaded from .env by orchestrator
if not ROBOFLOW_API_KEY:
    log_to_hunt_log("roboflow", status="skip", notes="no ROBOFLOW_API_KEY in .env")
else:
    headers = {"Authorization": f"Bearer {ROBOFLOW_API_KEY}"}
    resp = requests.get("https://api.roboflow.com/",
                        params={"api_key": ROBOFLOW_API_KEY})
```

For each public dataset found, export in `yolov8` format (already YOLO — skip conversion):
```python
export_url = f"https://app.roboflow.com/{workspace}/{project}/{version}/export/yolov8"
resp = requests.get(export_url, headers=headers)
download_link = resp.json()["export"]["link"]
safe_wget(download_link, f"pretrain_data/raw/roboflow_{project}.zip")   # A4
```

---

### Source 3 — COCO 2017

Estimated size: ~25 GB total. Check budget before starting.

```bash
mkdir -p pretrain_data/raw/coco && cd pretrain_data/raw/coco
```

Use `safe_wget` (A4) so a CDN hiccup doesn't crash the whole hunt:

```python
coco_urls = [
    "http://images.cocodataset.org/zips/train2017.zip",        # ~18 GB
    "http://images.cocodataset.org/zips/val2017.zip",          # ~1 GB
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",  # ~250 MB
]
for url in coco_urls:
    dest = f"pretrain_data/raw/coco/{pathlib.Path(url).name}"
    if not safe_wget(url, dest):
        log_to_hunt_log("coco", status="skip", notes=f"download failed: {url}")
```

```bash
for f in pretrain_data/raw/coco/*.zip; do unzip -q -o "$f" -d pretrain_data/raw/coco/ && rm "$f"; done
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

If `openimages` package is unavailable, fall back to direct CSV download via
`safe_wget` (A4):

```python
openimages_csvs = [
    "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv",
    "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv",
    "https://storage.googleapis.com/openimages/v6/validation-annotations-bbox.csv",
]
for url in openimages_csvs:
    dest = f"pretrain_data/raw/openimages/{pathlib.Path(url).name}"
    safe_wget(url, dest)
```
Then download images listed in the CSV using `aws s3 cp --no-sign-request`.

---

### Source 5 — Papers with Code

Fetch datasets tagged with object detection:

```bash
# Fetch the dataset list API (JSON, not HTML)
curl "https://paperswithcode.com/api/v1/datasets/?task=object-detection&page=1" \
    -H "Accept: application/json" > /tmp/pwc_datasets.json
```

For each dataset in the response:
- Check if `url` field points to a direct file (ends in `.zip`, `.tar`, `.tar.gz`)
- If yes: download with `safe_wget`
- If no (points to a webpage): try Firecrawl fallback below, else skip

#### Firecrawl fallback for non-direct URLs (optional)

Many PwC datasets link to a landing page rather than a direct download. If
`firecrawl` is installed (**see D1/D2 caveat in paper-finder**: there is no
official `firecrawl` CLI binary, so this block is usually skipped), scrape
the landing page to find the actual download link:

```python
import subprocess, re, shutil

def find_download_url_via_firecrawl(page_url: str) -> str | None:
    """Scrape a PwC dataset page to find a direct download link.
    D1/D2 — best-effort. Returns None silently when firecrawl is not present."""
    if not shutil.which("firecrawl"):
        return None
    try:
        r = subprocess.run(
            ["firecrawl", "scrape", page_url, "--format", "links"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return None
        for link in r.stdout.splitlines():
            link = link.strip()
            if re.search(r'\.(zip|tar|tar\.gz|tgz|rar)(\?|$)', link, re.IGNORECASE):
                return link
        return None
    except Exception:
        return None

# In the download loop:
for dataset in pwc_datasets:
    url = dataset.get("url", "")
    if re.search(r'\.(zip|tar|tar\.gz|tgz)(\?|$)', url):
        # Direct download — proceed normally
        dest = f"pretrain_data/raw/pwc_{dataset['name']}.{url.split('.')[-1]}"
        if not safe_wget(url, dest):
            log_to_hunt_log(dataset["name"], status="skip", notes="download failed")
    else:
        real_url = find_download_url_via_firecrawl(url)
        if real_url:
            log_to_hunt_log(dataset["name"], status="downloading",
                            notes=f"firecrawl found: {real_url}")
            dest = f"pretrain_data/raw/pwc_{dataset['name']}.zip"
            if not safe_wget(real_url, dest):
                log_to_hunt_log(dataset["name"], status="skip", notes="download failed")
        else:
            log_to_hunt_log(dataset["name"], status="skip",
                            notes="no direct download (firecrawl unavailable or no link found)")
```

Without Firecrawl, behaviour is unchanged from v1.2 — non-direct URLs are
skipped. With Firecrawl, ~30% more PwC datasets become downloadable.
Firecrawl CLI (if present) reads `FIRECRAWL_API_KEY` from `os.environ`
(loaded from `.env` by orchestrator Stage 0 Step 1.5).

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

### Val split for datasets without val (B5)

If a dataset has no val split after conversion, the hash-based 90/10 split
below handles small datasets gracefully:

- Previous version used `int(md5, 16) % 10 == 0` which, on small datasets
  (< 20 images), could produce zero val images and pretrain would run
  without validation. That also broke self-eval because val-less pretrain
  checkpoints can't be scored against.
- If the hash filter yields zero val images, we fall back to reserving the
  last 1 image (or 10% of total, whichever is larger).
- The move itself is now implemented (B5): previous docs only described
  what to do, without actual filesystem operations.

```python
import hashlib, shutil
from pathlib import Path

def make_val_split(converted_dir: Path, val_ratio: float = 0.1) -> int:
    """Split train into train+val for a converted dataset if val is empty.
    Returns number of val images after the split."""
    train_img = converted_dir / "images" / "train"
    train_lbl = converted_dir / "labels" / "train"
    val_img   = converted_dir / "images" / "val"
    val_lbl   = converted_dir / "labels" / "val"
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    # If val already populated, nothing to do
    if any(val_img.iterdir()):
        return sum(1 for _ in val_img.iterdir())

    all_train = sorted(train_img.iterdir())
    if not all_train:
        return 0

    # Primary: hash-based deterministic split
    to_move = [p for p in all_train
               if int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 10 == 0]

    # Fallback: if filter yielded zero (small dataset), take ceil(val_ratio * N)
    # or at least 1 image so val is never empty.
    if not to_move:
        import math
        n = max(1, math.ceil(len(all_train) * val_ratio))
        to_move = all_train[-n:]

    moved = 0
    for img_path in to_move:
        shutil.move(str(img_path), str(val_img / img_path.name))
        # C5 — pair each image with its label using with_suffix, not
        # str.replace(suffix, '.txt'). .replace() would match the suffix
        # anywhere in the filename (e.g. "cat.jpg.jpg") and produce
        # "cat..txt" — a silently broken move that leaves the label
        # behind.
        lbl_name = Path(img_path.name).with_suffix(".txt").name
        lbl_src = train_lbl / lbl_name
        if lbl_src.exists():
            shutil.move(str(lbl_src), str(val_lbl / lbl_name))
        moved += 1

    return moved

# In the per-dataset loop:
for ds_dir in Path("pretrain_data/converted").iterdir():
    if not ds_dir.is_dir():
        continue
    n_val = make_val_split(ds_dir)
    if n_val == 0:
        log_to_hunt_log(ds_dir.name, status="skip",
                        notes="could not produce val split (empty train)")
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
global_classes = sorted(all_classes)
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
                # C5 — use with_suffix(".txt") on the new image name.
                # `new_name.replace(img_path.suffix, ".txt")` would replace
                # the suffix anywhere in the string (e.g.
                # "roboflow__img.jpg.jpg" → "roboflow__img..txt").
                lbl_name = Path(new_name).with_suffix(".txt").name
                (lbl_dst / lbl_name).write_text("\n".join(remapped))
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
> start here and skip Phases 1–4.
>
> Inputs are already available:
> - Weights: read from `pipeline_state.base_weights_local` (do not re-resolve)
> - Corpus: `pretrain_data/merged/` already exists (do not re-download)

### Determine pretrain weights

If entering from orchestrator optional pretrain trigger:
- Use `pipeline_state.base_weights_local` directly as `PRETRAIN_WEIGHTS`
- Skip the `base_model.md` resolution block below

Otherwise (normal entry from Phase 4), check `base_model.md` in the project root.
Use `safe_wget` (A4) so a dead URL falls back rather than crashes the pretrain:

```python
import yaml, pathlib, re

base_model_md = pathlib.Path("base_model.md")
PRETRAIN_WEIGHTS = "weights/yolo26x.pt"   # default fallback

if base_model_md.exists():
    text = base_model_md.read_text()
    m = re.search(r"Weights URL.*?:\s*(.+)", text)
    weights_url = m.group(1).strip() if m else None

    if weights_url and weights_url.startswith("http"):
        dest = f"weights/{pathlib.Path(weights_url).name}"
        if pathlib.Path(dest).exists() and pathlib.Path(dest).stat().st_size > 0:
            PRETRAIN_WEIGHTS = dest
        elif safe_wget(weights_url, dest):
            PRETRAIN_WEIGHTS = dest
        else:
            print(f"WARNING: weights download failed from {weights_url}; "
                  f"using default {PRETRAIN_WEIGHTS}")

    elif weights_url == "reconstruct via paper2code":
        arxiv_m = re.search(r"arxiv\.org/abs/([\w.]+)", text)
        if arxiv_m:
            arxiv_id = arxiv_m.group(1)
            print(f"Running paper2code to reconstruct base model: {arxiv_id}")
            ckpts = (sorted(pathlib.Path(".").glob("*/checkpoints/*.pt"))
                     + sorted(pathlib.Path(".").glob("*/src/*.pt")))
            if ckpts:
                PRETRAIN_WEIGHTS = str(ckpts[-1])
                print(f"Base model reconstructed: {PRETRAIN_WEIGHTS}")
            else:
                print(f"WARNING: paper2code ran but no .pt found — using default")
        else:
            print("WARNING: no arXiv ID in base_model.md — cannot reconstruct")
    # else: weights_url is None or unrecognised — keep default

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

    Each (name, value_expr) pair replaces a line matching `^<n>\\s*=.*$`
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
pretrain_yaml = "pretrain_data/merged/dataset.yaml"
num_classes   = len(_yaml.safe_load(open(pretrain_yaml))["names"])

# v1.7.2 — pull IMGSZ from pipeline_state, where orchestrator Stage 3 Step 3
# canonically resolved it from research_config.yaml. Before v1.7.2,
# pretrain.py silently ran at the detection template's default IMGSZ
# (1920), ignoring user-configured 1280 / 640 / etc. — the self-eval
# comparison against the user's IMGSZ then operated at different
# resolutions in pretrain vs finetune, making the pretrain-improves-or-not
# signal unreliable.
imgsz = state.get("imgsz", 1920)   # fallback if orchestrator pre-v1.7.2

patch_section_2(SELF_EVAL_SOURCE, "pretrain.py", {
    "DATA_YAML":     repr(pretrain_yaml),
    "WEIGHTS":       repr(PRETRAIN_WEIGHTS),
    "IMGSZ":         str(imgsz),
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

D5 — use the runner from state, not hardcoded `uv run`:

```bash
RUNNER=$(python3 -c "import json; print(json.load(open('pipeline_state.json')).get('python_runner', 'uv run'))")
$RUNNER pretrain.py > pretrain.log 2>&1 &
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
    "WEIGHTS":      repr(str(pretrain_ckpt)),
    "CKPT_DIR":     'Path("eval_pretrained")',
    "TIME_BUDGET":  str(state["self_eval_time_budget"]),
})
```

Run sequentially (GPU can only handle one at a time). D5 runner:
```bash
RUNNER=$(python3 -c "import json; print(json.load(open('pipeline_state.json')).get('python_runner', 'uv run'))")
$RUNNER eval_a.py > eval_baseline.log   2>&1
$RUNNER eval_b.py > eval_pretrained.log 2>&1
```

Both logs will contain the same print format as `train.py`, so the parser in
Step 2 handles them uniformly via `evaluation.parsing.patterns`.

### Step 2 — Extract eval metric from both logs (B1)

Use the parsing config loaded in Phase 1 Step 0. Note this uses `EVAL_METRIC`
(dataset hunter's choice of detector-quality metric), not `PRIMARY`.

Both the shared helper and the inline implementation are valid — previously
the json/csv branches were `...` stubs that silently crashed on non-stdout
eval configs.

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(state["skills_dir"]) / "shared"))
import parse_metrics

def extract_eval_metric(log_path: str) -> float | None:
    """Extract EVAL_METRIC value. Delegates to shared parse_metrics helper
    so the json/csv branches are implemented (B1)."""
    if PARSE_SOURCE == "stdout":
        cfg = dict(PARSING_CFG)
        cfg["log_file"] = log_path
        # Reduce patterns to only EVAL_METRIC so we don't extract everything
        cfg["patterns"] = {EVAL_METRIC: PARSE_PATTERNS[EVAL_METRIC]}
        return parse_metrics.extract("stdout", cfg).get(EVAL_METRIC)

    if PARSE_SOURCE == "json":
        # Resolve json path. Eval scripts write their own per-run json file;
        # the parsing config lists the dotted path relative to that file.
        cfg = {"json_file": log_path.replace(".log", ".json"),
               "json_paths": {EVAL_METRIC: PARSING_CFG["json_paths"][EVAL_METRIC]}}
        return parse_metrics.extract("json", cfg).get(EVAL_METRIC)

    if PARSE_SOURCE == "csv":
        cfg = {"csv_file": log_path.replace(".log", ".csv"),
               "csv_columns": {EVAL_METRIC: PARSING_CFG["csv_columns"][EVAL_METRIC]}}
        return parse_metrics.extract("csv", cfg).get(EVAL_METRIC)

    raise RuntimeError(f"Unsupported parsing.source: {PARSE_SOURCE!r}")

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
        recommendation = "inconclusive"

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
