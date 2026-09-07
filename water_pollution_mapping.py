"""
Water Pollution and Contamination Mapping from Drone Imagery
==============================================================
CIA 3 Project - Drone Technologies and Its Transformative Applications

Pipeline:
    1. Load & pre-process drone image
    2. Segment water region (HSV threshold + Otsu + morphology)
    3. Detect abnormal / contaminated surface patches within the water
    4. Estimate contamination coverage percentage
    5. Classify pollution severity (Low / Moderate / Severe)
    6. Save annotated overlay + write a CSV/JSON summary report

Usage (single image):
    python water_pollution_mapping.py --image path/to/drone_image.jpg --outdir results

Usage (batch - folder of images, e.g. a time series of the same lake):
    python water_pollution_mapping.py --folder path/to/images --outdir results

Dependencies: opencv-python, numpy, matplotlib (only used for the optional histogram plot)
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np


# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------

class Config:
    """Tunable thresholds. Adjust these to match your dataset's lighting
    and water colour after looking at a few sample images."""

    # Standard working resolution (keeps processing fast & consistent)
    RESIZE_WIDTH = 800
    RESIZE_HEIGHT = 600

    # HSV range that broadly captures "water" pixels (blue-green-cyan hues)
    WATER_HSV_LOWER = np.array([60, 15, 15])
    WATER_HSV_UPPER = np.array([140, 255, 255])

    # HSV range considered "clean / normal" water within the water mask.
    # Anything inside the water mask but OUTSIDE this band is flagged as
    # a possible contaminant (algae = green/brown, oil sheen = iridescent,
    # sediment/industrial discharge = grey/brown/white).
    CLEAN_WATER_HSV_LOWER = np.array([80, 30, 30])
    CLEAN_WATER_HSV_UPPER = np.array([130, 215, 235])

    # Morphology kernel size for cleaning up masks
    MORPH_KERNEL_SIZE = 5

    # Minimum contour area (in pixels) to keep as a genuine contamination
    # patch rather than sensor/compression noise
    MIN_CONTAMINATION_AREA = 40

    # Severity thresholds (percentage of water surface contaminated)
    SEVERITY_LOW_MAX = 10.0        # 0%   - <10%  -> Low
    SEVERITY_MODERATE_MAX = 35.0   # 10%  - <35%  -> Moderate
    #                                >=35%        -> Severe


# ----------------------------------------------------------------------
# 2. CORE PIPELINE FUNCTIONS
# ----------------------------------------------------------------------

def load_and_preprocess(image_path, cfg: Config):
    """Read an image from disk, resize it, and denoise with a light blur."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.resize(img, (cfg.RESIZE_WIDTH, cfg.RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
    denoised = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    return img, denoised, hsv


def segment_water(hsv, cfg: Config):
    """Segment the water region using HSV thresholding + Otsu refinement
    + morphological cleanup. Returns a binary mask (255 = water)."""

    # Step 1: broad colour-range threshold
    color_mask = cv2.inRange(hsv, cfg.WATER_HSV_LOWER, cfg.WATER_HSV_UPPER)

    # Step 2: Otsu thresholding on the Value channel as a secondary cue,
    # then combine with the colour mask (helps on overexposed/dark patches)
    v_channel = hsv[:, :, 2]
    _, otsu_mask = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(color_mask, cv2.bitwise_and(color_mask, otsu_mask))

    # Step 3: morphological cleanup - close small gaps, remove speckle noise
    kernel = np.ones((cfg.MORPH_KERNEL_SIZE, cfg.MORPH_KERNEL_SIZE), np.uint8)
    water_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 4: keep only the largest connected components (removes tiny
    # stray blobs classified as "water" by mistake)
    water_mask = _keep_significant_components(water_mask, min_area=500)

    return water_mask


def _keep_significant_components(mask, min_area=500):
    """Remove small connected components below `min_area` pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # skip background (0)
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def detect_contamination(hsv, water_mask, cfg: Config):
    """Within the water mask, flag pixels whose colour falls outside the
    'clean water' band as contaminated. Also runs a texture check so
    foam/scum (visually similar colour, rougher texture) is caught."""

    clean_mask = cv2.inRange(hsv, cfg.CLEAN_WATER_HSV_LOWER, cfg.CLEAN_WATER_HSV_UPPER)

    # Anything that IS water but is NOT "clean water" -> candidate contamination
    contamination_raw = cv2.bitwise_and(water_mask, cv2.bitwise_not(clean_mask))

    # Texture cue: compute local standard deviation (foam/scum/turbidity
    # tends to have higher local texture variance than smooth open water)
    gray = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (9, 9))
    sq_mean = cv2.blur(gray * gray, (9, 9))
    local_var = np.clip(sq_mean - mean ** 2, 0, None)
    texture_mask = (local_var > np.percentile(local_var[water_mask > 0], 94)).astype(np.uint8) * 255 \
        if np.any(water_mask > 0) else np.zeros_like(gray, dtype=np.uint8)
    texture_mask = cv2.bitwise_and(texture_mask, water_mask)

    # Combine colour-anomaly + texture-anomaly cues
    contamination_mask = cv2.bitwise_or(contamination_raw, texture_mask)

    # Clean up: remove tiny noise contours below MIN_CONTAMINATION_AREA
    contamination_mask = _filter_small_contours(contamination_mask, cfg.MIN_CONTAMINATION_AREA)

    # Safety: contamination can only exist *within* the water region.
    # (Filled-contour drawing can bleed a pixel or two past the original
    # mask boundary, so re-clip here to guarantee contam_px <= water_px.)
    contamination_mask = cv2.bitwise_and(contamination_mask, water_mask)

    return contamination_mask


def _filter_small_contours(mask, min_area):
    """Remove connected blobs smaller than `min_area` pixels to suppress
    noise. Uses connected-component filtering (not contour re-fill) so the
    *actual* mask pixels are kept rather than solid-filling each blob's
    outer boundary (which would wrongly fill in any holes)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # skip background (0)
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def compute_coverage(water_mask, contamination_mask):
    """Return (water_pixel_count, contaminated_pixel_count, coverage_pct)."""
    water_px = int(cv2.countNonZero(water_mask))
    contam_px = int(cv2.countNonZero(contamination_mask))
    coverage_pct = (contam_px / water_px * 100.0) if water_px > 0 else 0.0
    return water_px, contam_px, round(coverage_pct, 2)


def classify_severity(coverage_pct, cfg: Config):
    """Map a coverage percentage to a severity label."""
    if coverage_pct < cfg.SEVERITY_LOW_MAX:
        return "Low"
    elif coverage_pct < cfg.SEVERITY_MODERATE_MAX:
        return "Moderate"
    else:
        return "Severe"


def draw_overlay(original_img, water_mask, contamination_mask, coverage_pct, severity):
    """Produce a visual overlay: water outline in cyan, contamination in red,
    plus a text banner with coverage % and severity."""

    overlay = original_img.copy()

    # Water boundary outline (cyan)
    water_contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, water_contours, -1, (255, 255, 0), 2)

    # Contaminated regions filled in semi-transparent red
    red_layer = original_img.copy()
    red_layer[contamination_mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(red_layer, 0.45, overlay, 0.55, 0)

    # Text banner
    severity_color = {"Low": (0, 200, 0), "Moderate": (0, 165, 255), "Severe": (0, 0, 255)}[severity]
    banner_h = 46
    cv2.rectangle(blended, (0, 0), (blended.shape[1], banner_h), (30, 30, 30), thickness=cv2.FILLED)
    text = f"Contaminated Coverage: {coverage_pct:.2f}%   |   Severity: {severity}"
    cv2.putText(blended, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, severity_color, 2, cv2.LINE_AA)

    return blended


# ----------------------------------------------------------------------
# 3. SINGLE-IMAGE PIPELINE
# ----------------------------------------------------------------------

def process_image(image_path, outdir, cfg: Config = Config()):
    """Run the full pipeline on one image and save mask/overlay outputs.
    Returns a dict summary (used for the CSV/JSON batch report)."""

    os.makedirs(outdir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    original, _, hsv = load_and_preprocess(image_path, cfg)
    water_mask = segment_water(hsv, cfg)
    contamination_mask = detect_contamination(hsv, water_mask, cfg)
    water_px, contam_px, coverage_pct = compute_coverage(water_mask, contamination_mask)
    severity = classify_severity(coverage_pct, cfg)
    overlay = draw_overlay(original, water_mask, contamination_mask, coverage_pct, severity)

    # Save outputs
    cv2.imwrite(os.path.join(outdir, f"{basename}_water_mask.png"), water_mask)
    cv2.imwrite(os.path.join(outdir, f"{basename}_contamination_mask.png"), contamination_mask)
    cv2.imwrite(os.path.join(outdir, f"{basename}_overlay.jpg"), overlay)

    summary = {
        "image": os.path.basename(image_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "water_pixels": water_px,
        "contaminated_pixels": contam_px,
        "coverage_percent": coverage_pct,
        "severity": severity,
    }
    return summary


# ----------------------------------------------------------------------
# 4. BATCH PIPELINE (folder of images / drone flight image sequence)
# ----------------------------------------------------------------------

def process_folder(folder_path, outdir, cfg: Config = Config()):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = sorted(
        os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)
    )
    if not image_files:
        print(f"No images found in {folder_path}")
        return []

    summaries = []
    for path in image_files:
        print(f"Processing {os.path.basename(path)} ...")
        try:
            summary = process_image(path, outdir, cfg)
            summaries.append(summary)
            print(f"  -> Coverage: {summary['coverage_percent']}%  Severity: {summary['severity']}")
        except Exception as e:
            print(f"  !! Failed on {path}: {e}")

    _write_reports(summaries, outdir)
    return summaries


def _write_reports(summaries, outdir):
    if not summaries:
        return
    csv_path = os.path.join(outdir, "contamination_report.csv")
    json_path = os.path.join(outdir, "contamination_report.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nSummary report written to:\n  {csv_path}\n  {json_path}")


# ----------------------------------------------------------------------
# 5. COMMAND-LINE INTERFACE
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Drone-based water pollution & contamination mapping")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single drone image")
    group.add_argument("--folder", type=str, help="Path to a folder of drone images (batch mode)")
    parser.add_argument("--outdir", type=str, default="results", help="Directory to save outputs")
    args = parser.parse_args()

    cfg = Config()

    if args.image:
        summary = process_image(args.image, args.outdir, cfg)
        _write_reports([summary], args.outdir)
        print(json.dumps(summary, indent=2))
    else:
        process_folder(args.folder, args.outdir, cfg)


if __name__ == "__main__":
    sys.exit(main())
