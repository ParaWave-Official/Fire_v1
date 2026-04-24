"""
generate_report.py

Reads completed inference results from benchmarks/final_results/ and produces:
  - README.md          (full benchmark report)
  - assets/fov_gradient_chart.png  (distance study chart)

Run:
    python generate_report.py
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ===========================================================================
# CONFIGURATION
# ===========================================================================

DATASETS: list[dict] = [
    {
        "name":   "Fire",
        "type":   "detection",
        "source": "https://universe.roboflow.com/sean-cftrp/fire-z2n21",
    },
    {
        "name":   "Fire_Thermal",
        "type":   "detection",
        "source": "https://universe.roboflow.com/surveillance-rtj0v/fire-thermal",
    },
    {
        "name":   "Wildfire",
        "type":   "detection",
        "source": "https://universe.roboflow.com/insa-ausjd/wildfire-v7dbx",
    },
    {
        "name":   "Fire_Distance",
        "type":   "distance",
        "source": None,   # proprietary / no public URL
    },
]

# Google Drive folder containing all benchmark images.
# Replace the placeholder below with the actual shared link once available.
GOOGLE_DRIVE_LINK: str = ""   # e.g. "https://drive.google.com/drive/folders/XXXXX"

RESULTS_DIR    = "final_results"
OUTPUT_DIR     = "."          # README.md is written here
ASSETS_DIR     = os.path.join(OUTPUT_DIR, "assets")
BENCHMARK_NAME = "Fire_v1"

# Detection thresholds
IOU_THRESHOLD = 0.0001        # minimum IOU to count as a true positive (effectively any overlap)

# Confidence tiers (cumulative order: each tier INCLUDES all tiers above it)
CONFIDENCE_TIERS = [
    ("Highest",              ["highest"]),
    ("High",       ["highest", "high"]),
    ("Medium", ["highest", "high", "medium"]),
    ("Low",                  ["highest", "high", "medium", "low"]),
]

# Distance study chart settings
CHART_FOV_ANGLES     = [20, 40, 60, 80, 100, 120]   # FOVs to plot as gradient lines
HIGHLIGHT_FOV        = 80                            # FOV used in the summary questions
HIGHLIGHT_SIZES_FT   = [1 / 12, 15]                 # 1 inch and 15 ft
CHART_MAX_DIST_FT    = 10560                         # x-axis max (2 miles)
CHART_MAX_SIZE_FT    = 50                            # y-axis max


# ===========================================================================
# UTILITIES
# ===========================================================================

def load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def iou(a: dict, b: dict) -> float:
    """Compute IOU between two boxes (x_min, y_min, x_max, y_max dicts)."""
    ix1 = max(a["x_min"], b["x_min"])
    iy1 = max(a["y_min"], b["y_min"])
    ix2 = min(a["x_max"], b["x_max"])
    iy2 = min(a["y_max"], b["y_max"])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, a["x_max"] - a["x_min"]) * max(0.0, a["y_max"] - a["y_min"])
    area_b = max(0.0, b["x_max"] - b["x_min"]) * max(0.0, b["y_max"] - b["y_min"])
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def get_gt_boxes(image_entry: dict) -> list[dict]:
    """Return all ground-truth boxes from raw_data (any class)."""
    boxes = []
    for cls_entry in image_entry.get("raw_data", []):
        boxes.extend(cls_entry.get("boxes", []))
    return boxes


def get_pred_boxes(image_entry: dict, active_levels: list[str]) -> list[dict]:
    """Return all predicted boxes whose level is in active_levels (any class)."""
    boxes = []
    for cls_entry in image_entry.get("predictions", []):
        for box in cls_entry.get("boxes", []):
            if box.get("level") in active_levels:
                boxes.append(box)
    return boxes


# ===========================================================================
# DETECTION METRICS
# ===========================================================================

def compute_tier_metrics(
    normal_data: dict | None,
    blackout_data: dict | None,
    active_levels: list[str],
) -> dict:
    """
    Compute DA, FAR, FNR for a single confidence tier across one dataset.

    DA  = TP / (TP + FN)   [computed on normal images against ground truth]
    FAR = blackout images with ≥1 predicted box / total blackout images
    FNR = FN / (TP + FN)   = 1 − DA
    """
    tp = fn = 0

    if normal_data:
        for img in normal_data.values():
            gt_boxes   = get_gt_boxes(img)
            pred_boxes = get_pred_boxes(img, active_levels)

            for gt in gt_boxes:
                hit = any(
                    iou(gt, pb) >= IOU_THRESHOLD
                    for pb in pred_boxes
                )
                if hit:
                    tp += 1
                else:
                    fn += 1

    blackout_total    = 0
    blackout_detected = 0

    if blackout_data:
        for img in blackout_data.values():
            blackout_total += 1
            pred_boxes = get_pred_boxes(img, active_levels)
            if pred_boxes:
                blackout_detected += 1

    da  = tp / (tp + fn) if (tp + fn) > 0 else None
    far = blackout_detected / blackout_total if blackout_total > 0 else None
    fnr = fn / (tp + fn) if (tp + fn) > 0 else None

    return {
        "tp":                tp,
        "fn":                fn,
        "blackout_total":    blackout_total,
        "blackout_detected": blackout_detected,
        "da":                da,
        "far":               far,
        "fnr":               fnr,
    }


def analyze_detection(dataset_name: str) -> dict:
    """Return per-tier metrics for a detection dataset."""
    normal_path   = os.path.join(RESULTS_DIR, f"{dataset_name}_predictions.json")
    blackout_path = os.path.join(RESULTS_DIR, f"{dataset_name}_predictions_blackout.json")

    normal_data   = load_json(normal_path)
    blackout_data = load_json(blackout_path)

    if normal_data is None and blackout_data is None:
        print(f"  [WARN] No results found for {dataset_name}, skipping.")
        return {}

    tiers = {}
    for label, levels in CONFIDENCE_TIERS:
        tiers[label] = compute_tier_metrics(normal_data, blackout_data, levels)

    return tiers


# ===========================================================================
# DISTANCE STUDY
# ===========================================================================

def any_iou_overlap(gt_box: dict, pred_boxes: list[dict]) -> bool:
    """Return True if gt_box has IOU > 0 with any predicted box."""
    return any(iou(gt_box, pb) > 0.0 for pb in pred_boxes)


def analyze_distance(dataset_name: str) -> dict:
    """
    Find the smallest GT box (by area) that has any IOU overlap with predictions.
    Returns a dict with the box dimensions and FOV analysis.
    """
    normal_path = os.path.join(RESULTS_DIR, f"{dataset_name}_predictions.json")
    data = load_json(normal_path)

    if data is None:
        print(f"  [WARN] No results found for distance study: {dataset_name}")
        return {}

    # Use ALL confidence levels for distance study
    all_levels = [lvl for _, lvls in CONFIDENCE_TIERS for lvl in lvls]
    active_levels = list(dict.fromkeys(all_levels))  # deduplicated, ordered

    candidate_boxes = []

    for img in data.values():
        pred_boxes = get_pred_boxes(img, active_levels)
        if not pred_boxes:
            continue

        for gt in get_gt_boxes(img):
            if any_iou_overlap(gt, pred_boxes):
                w = gt["x_max"] - gt["x_min"]
                h = gt["y_max"] - gt["y_min"]
                area = w * h
                candidate_boxes.append({"w": w, "h": h, "area": area, "box": gt})

    if not candidate_boxes:
        print(f"  [WARN] No overlapping predictions found for {dataset_name}")
        return {}

    smallest = min(candidate_boxes, key=lambda x: x["area"])
    w_gt = smallest["w"]
    h_gt = smallest["h"]

    print(f"  Smallest detected GT box — width: {w_gt:.6f}, height: {h_gt:.6f}, area: {smallest['area']:.8f}")

    return {
        "w_gt":    w_gt,
        "h_gt":    h_gt,
        "dataset": dataset_name,
    }


def fov_from_size_and_distance(
    object_size_ft: float,
    distance_ft: float,
    gt_fraction: float,
) -> float:
    """
    Return the maximum FOV (degrees) at which an object of `object_size_ft`
    at `distance_ft` still appears as at least `gt_fraction` of the image.

    Derivation:
        scene_width = 2 * D * tan(FOV/2)
        object_fraction = object_size / scene_width = object_size / (2*D*tan(FOV/2))
        Setting object_fraction = gt_fraction:
            tan(FOV/2) = object_size / (2 * D * gt_fraction)
            FOV = 2 * arctan(object_size / (2 * D * gt_fraction))
    """
    if distance_ft <= 0 or gt_fraction <= 0:
        return 0.0
    return math.degrees(2 * math.atan(object_size_ft / (2 * distance_ft * gt_fraction)))


def min_detectable_size_ft(
    distance_ft: float,
    fov_deg: float,
    gt_fraction: float,
) -> float:
    """
    Return the minimum object size (ft) detectable at `distance_ft` with
    a camera of `fov_deg` degrees FOV, given the minimum detectable image
    fraction `gt_fraction`.

        min_size = 2 * D * tan(FOV/2) * gt_fraction
    """
    return 2 * distance_ft * math.tan(math.radians(fov_deg / 2)) * gt_fraction


def max_detection_distance_ft(
    object_size_ft: float,
    fov_deg: float,
    gt_fraction: float,
) -> float:
    """
    Return the maximum distance (ft) at which an object of `object_size_ft`
    is detectable with a camera of `fov_deg` degrees FOV.

        D = object_size / (2 * tan(FOV/2) * gt_fraction)
    """
    denom = 2 * math.tan(math.radians(fov_deg / 2)) * gt_fraction
    return object_size_ft / denom if denom > 0 else float("inf")


def generate_gradient_chart(w_gt: float, h_gt: float, output_path: str) -> None:
    """
    Generate and save the FOV gradient chart.

    X: distance (ft)
    Y: minimum detectable fire size (ft)
    One line per FOV angle in CHART_FOV_ANGLES, colored by FOV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    distances = np.linspace(1, CHART_MAX_DIST_FT, 1000)

    # Use the SMALLER of w_gt / h_gt as the binding constraint
    binding_fraction = min(w_gt, h_gt)

    fig, ax = plt.subplots(figsize=(11, 7))
    cmap   = cm.plasma
    colors = [cmap(i / (len(CHART_FOV_ANGLES) - 1)) for i in range(len(CHART_FOV_ANGLES))]

    for fov, color in zip(CHART_FOV_ANGLES, colors):
        sizes = [min_detectable_size_ft(d, fov, binding_fraction) for d in distances]
        ax.plot(distances, sizes, color=color, linewidth=2, label=f"{fov}°")

    ax.set_xlabel("Distance from Camera (ft)", fontsize=12)
    ax.set_ylabel("Minimum Detectable Fire Size (ft)", fontsize=12)
    ax.set_title(f"Minimum Detectable Fire Size vs Distance\n(binding image fraction = {binding_fraction:.4f})", fontsize=13)
    ax.set_xlim(0, CHART_MAX_DIST_FT)
    ax.set_ylim(0, CHART_MAX_SIZE_FT)
    ax.legend(title="Camera FOV", fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [Chart] Saved to {output_path}")


# ===========================================================================
# README GENERATION
# ===========================================================================

def pct(val: float | None, decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def build_detection_table(tier_results: dict) -> str:
    """Build a markdown table for one dataset's per-tier metrics."""
    header = "| Confidence Tier | DA | FAR | FNR | TP | FN | Blackout Images | False Alarms |"
    divider = "|---|---|---|---|---|---|---|---|"
    rows = [header, divider]

    for label, m in tier_results.items():
        rows.append(
            f"| {label} | {pct(m['da'])} | {pct(m['far'])} | {pct(m['fnr'])} "
            f"| {m['tp']} | {m['fn']} | {m['blackout_total']} | {m['blackout_detected']} |"
        )
    return "\n".join(rows)


def build_cumulative_table(cumulative: dict) -> str:
    """Build a markdown table for cumulative metrics across all detection datasets."""
    header = "| Confidence Tier | DA | FAR | FNR | Total TP | Total FN |"
    divider = "|---|---|---|---|---|---|"
    rows = [header, divider]

    for label, m in cumulative.items():
        rows.append(
            f"| {label} | {pct(m['da'])} | {pct(m['far'])} | {pct(m['fnr'])} "
            f"| {m['tp']} | {m['fn']} |"
        )
    return "\n".join(rows)


def build_distance_table(
    w_gt: float, h_gt: float, fov_angles: list[int], sizes_ft: list[float]
) -> str:
    """Build a markdown table of max detection distances for multiple FOVs and sizes."""
    binding = min(w_gt, h_gt)

    size_labels = []
    for s in sizes_ft:
        if s < 1:
            size_labels.append(f'{round(s * 12)}" ({s:.4f} ft)')
        else:
            size_labels.append(f"{s:.0f} ft")

    header  = "| FOV (°) | " + " | ".join(size_labels) + " |"
    divider = "|---| " + " | ".join(["---"] * len(sizes_ft)) + " |"
    rows = [header, divider]

    for fov in fov_angles:
        cells = [f"{max_detection_distance_ft(s, fov, binding):,.0f} ft" for s in sizes_ft]
        rows.append(f"| {fov}° | " + " | ".join(cells) + " |")

    return "\n".join(rows)


def generate_readme(
    detection_results: dict[str, dict],
    distance_result:   dict,
    cumulative:        dict,
) -> str:
    """Assemble the full README.md content."""

    # -----------------------------------------------------------------------
    # Cumulative summary (all tiers row for the top-level numbers)
    all_tier_label = CONFIDENCE_TIERS[-1][0]   # "All"
    cum_all = cumulative.get(all_tier_label, {})
    cum_da  = pct(cum_all.get("da"))
    cum_far = pct(cum_all.get("far"))
    cum_fnr = pct(cum_all.get("fnr"))

    # Distance study headline numbers
    dist_lines = []
    if distance_result:
        w_gt    = distance_result["w_gt"]
        h_gt    = distance_result["h_gt"]
        binding = min(w_gt, h_gt)
        for size_ft in HIGHLIGHT_SIZES_FT:
            max_d = max_detection_distance_ft(size_ft, HIGHLIGHT_FOV, binding)
            if size_ft < 1:
                label = f'{round(size_ft * 12)}-inch fire'
            else:
                label = f"{size_ft:.0f} ft fire"
            dist_lines.append(f"- **{label}** at {HIGHLIGHT_FOV}° FOV: detectable up to **{max_d:,.0f} ft**")
    dist_summary = "\n".join(dist_lines) if dist_lines else "_Distance study results unavailable._"

    # -----------------------------------------------------------------------
    # Google Drive link
    if GOOGLE_DRIVE_LINK:
        drive_link = f"[{GOOGLE_DRIVE_LINK}]({GOOGLE_DRIVE_LINK})"
    else:
        drive_link = "_Link not yet available — will be added once the folder is shared._"

    # -----------------------------------------------------------------------
    # Dataset citations table rows
    citation_rows = "\n".join(
        f'| **{ds["name"]}** | [{ds["source"]}]({ds["source"]}) |'
        if ds.get("source") else
        f'| **{ds["name"]}** | _Proprietary / no public source_ |'
        for ds in DATASETS
    )

    # -----------------------------------------------------------------------
    # Per-dataset detection sections (with source citations)
    source_map = {ds["name"]: ds.get("source") for ds in DATASETS}

    dataset_sections = []
    for name, tiers in detection_results.items():
        if not tiers:
            continue
        table  = build_detection_table(tiers)
        source = source_map.get(name)
        citation = (
            f"\n_Source: [{source}]({source})_\n"
            if source else ""
        )
        dataset_sections.append(f"""\
### {name}
{citation}
{table}
""")
    datasets_md = "\n".join(dataset_sections)

    # -----------------------------------------------------------------------
    # Distance study section
    if distance_result:
        w_gt    = distance_result["w_gt"]
        h_gt    = distance_result["h_gt"]
        binding = min(w_gt, h_gt)

        fov_table = build_distance_table(w_gt, h_gt, CHART_FOV_ANGLES, HIGHLIGHT_SIZES_FT + [1, 5, 10, 30])

        fov_formula_md = f"""\
#### FOV Formula

Given:
- **Object size** = `S` (ft)  
- **Distance** = `D` (ft)  
- **Minimum detectable image fraction** = `f` (dimensionless, 0–1)

The **maximum FOV** at which the object still subtends at least `f` of the image:

```
FOV = 2 × arctan( S / (2 × D × f) )
```

Solving for the **maximum detection distance** at a given FOV:

```
D_max = S / (2 × tan(FOV/2) × f)
```

And the **minimum detectable object size** at a given FOV and distance:

```
S_min = 2 × D × tan(FOV/2) × f
```

This is derived independently for the horizontal (width, f = {w_gt:.6f}) and vertical  
(height, f = {h_gt:.6f}) dimensions. The **binding constraint is the smaller**:  
`f_binding = min(w_gt, h_gt) = {binding:.6f}`.

All table and chart values use `f_binding`.
"""

        distance_md = f"""\
## Distance Study — {distance_result['dataset']}

### Smallest Detected Ground Truth Box

The smallest ground-truth bounding box for which the model produced a prediction  
with IOU > 0 has the following normalised dimensions:

| Dimension | Value |
|---|---|
| Width  (fraction of image) | `{w_gt:.6f}` |
| Height (fraction of image) | `{h_gt:.6f}` |
| Binding constraint (`min`) | `{binding:.6f}` |

{fov_formula_md}

### Maximum Detection Distance Table

Rows = camera FOV; columns = fire size.

{fov_table}

### FOV Gradient Chart

Each line shows the **minimum detectable fire size** at a given distance for a specific camera FOV.  
Objects above the line for a given FOV can be detected; objects below cannot.

![FOV Gradient Chart](assets/fov_gradient_chart.png)
"""
    else:
        distance_md = "_Distance study results not available._\n"

    # -----------------------------------------------------------------------
    # Cumulative table
    cum_table = build_cumulative_table(cumulative)

    # -----------------------------------------------------------------------
    # Full README
    readme = f"""\
# {BENCHMARK_NAME} — Benchmark Report

## Executive Summary

| Metric | Value (All Confidence Tiers) |
|---|---|
| Cumulative Detection Accuracy (DA)   | **{cum_da}** |
| Cumulative False Alarm Rate (FAR)    | **{cum_far}** |
| Cumulative False Negative Rate (FNR) | **{cum_fnr}** |

### Image Dataset

> 📁 **All benchmark images** are available in the shared Google Drive folder:  
> {drive_link}

### Distance Study Highlights ({HIGHLIGHT_FOV}° FOV)

{dist_summary}

---

## Methodology

### Detection Study

Three datasets were evaluated as **detection studies** (Fire, Fire\\_Thermal, Wildfire).  
Each dataset provides two splits:

- **Final images** — original frames with fire/smoke present. Used to measure  
  **Detection Accuracy (DA)** and **False Negative Rate (FNR)**.
- **Blackout images** — the same frames with the labelled objects blacked out.  
  Used to measure the **False Alarm Rate (FAR)**, since any detection in these  
  images is by definition a false alarm.

#### Metrics

| Metric | Definition |
|---|---|
| **DA**  | `TP / (TP + FN)` — fraction of ground-truth boxes matched by at least one prediction with IOU ≥ {IOU_THRESHOLD} (effectively any overlap) |
| **FAR** | `images with ≥1 prediction / total blackout images` — proportion of blackout images that trigger at least one false detection |
| **FNR** | `FN / (TP + FN) = 1 − DA` — fraction of ground-truth boxes missed by the model |

Results are reported **cumulatively by confidence tier** — each tier includes all boxes  
at that level and above:

| Tier | Levels included |
|---|---|
| Highest              | `highest` only |
| Highest + High       | `highest`, `high` |
| Highest + High + Med | `highest`, `high`, `medium` |
| All                  | `highest`, `high`, `medium`, `low` |

A ground-truth box is considered **matched** (TP) if any predicted box in the same  
image has IOU ≥ {IOU_THRESHOLD} with it (effectively any bounding-box overlap),  
regardless of class label (since terms vary per dataset). A single predicted box may  
overlap multiple GT boxes, and each overlapping GT box is counted as a separate TP.  
DA and FNR are computed from normal images only; blackout images do not contribute  
to these metrics.

### Distance Study

One dataset (Fire\\_Distance) is treated as a **distance study**. The methodology is:

1. Identify the smallest ground-truth bounding box (by area) for which the model  
   produced at least one prediction with IOU > 0.
2. Record its normalised width and height as the **minimum detectable image fraction**.
3. Apply the FOV formula (see Distance Study section below) to compute maximum  
   detection distances for any combination of camera FOV and fire size.

---

## Detection Study Results

### Cumulative (All Datasets)

{cum_table}

---

### Per-Dataset Results

{datasets_md}

---

{distance_md}

---

## Dataset Citations

| Dataset | Source |
|---|---|
{citation_rows}

---

*Report generated by `generate_report.py` for benchmark **{BENCHMARK_NAME}**.*
"""
    return readme


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    os.makedirs(ASSETS_DIR, exist_ok=True)

    detection_results: dict[str, dict] = {}
    distance_result:   dict            = {}

    # Initialise cumulative accumulators per tier
    cumulative: dict[str, dict] = {
        label: {"tp": 0, "fn": 0, "blackout_total": 0, "blackout_detected": 0}
        for label, _ in CONFIDENCE_TIERS
    }

    for ds in DATASETS:
        name = ds["name"]
        kind = ds["type"]
        print(f"\n{'='*60}")
        print(f"  Processing: {name}  ({kind})")

        if kind == "detection":
            tiers = analyze_detection(name)
            if tiers:
                detection_results[name] = tiers
                for label, m in tiers.items():
                    cumulative[label]["tp"]                += m["tp"]
                    cumulative[label]["fn"]                += m["fn"]
                    cumulative[label]["blackout_total"]    += m["blackout_total"]
                    cumulative[label]["blackout_detected"] += m["blackout_detected"]

        elif kind == "distance":
            result = analyze_distance(name)
            if result:
                distance_result = result
                chart_path = os.path.join(ASSETS_DIR, "fov_gradient_chart.png")
                generate_gradient_chart(result["w_gt"], result["h_gt"], chart_path)

    # Derive DA / FAR / FNR for each cumulative tier
    for label, m in cumulative.items():
        total_gt = m["tp"] + m["fn"]
        m["da"]  = m["tp"] / total_gt if total_gt > 0 else None
        m["fnr"] = m["fn"] / total_gt if total_gt > 0 else None
        m["far"] = (
            m["blackout_detected"] / m["blackout_total"]
            if m["blackout_total"] > 0 else None
        )

    # Generate README
    readme_content = generate_readme(detection_results, distance_result, cumulative)
    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"\n{'='*60}")
    print(f"  [DONE] README written to {readme_path}")


if __name__ == "__main__":
    main()