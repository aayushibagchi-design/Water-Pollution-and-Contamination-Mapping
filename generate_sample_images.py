"""
Generates synthetic 'drone photo' test images of a lake/pond with varying
levels of simulated contamination (algae bloom / oil sheen patches), so the
pipeline in water_pollution_mapping.py can be demonstrated end-to-end even
without real UAV footage. Replace these with your own drone images for the
actual submission.

Usage:
    python generate_sample_images.py --outdir sample_images
"""

import argparse
import os

import cv2
import numpy as np


def make_lake_image(width=800, height=600, contamination_level=0.0, seed=0):
    """contamination_level: 0.0 (clean) to 1.0 (heavily polluted)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Background: green-brown land / vegetation
    img[:, :] = (60, 110, 70)  # BGR
    noise = rng.integers(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw an irregular lake shape (ellipse-ish water body) in blue-cyan
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (width // 3, height // 3)
    cv2.ellipse(mask, center, axes, 15, 0, 360, 255, thickness=-1)

    water_color = np.array([160, 110, 40], dtype=np.int16)  # BGR - blue-teal water
    water_noise = rng.integers(-8, 8, (height, width, 3))
    water_layer = np.clip(water_color + water_noise, 0, 255).astype(np.uint8)
    img[mask > 0] = water_layer[mask > 0]

    # Add contamination patches proportional to contamination_level
    if contamination_level > 0:
        num_patches = max(1, int(contamination_level * 6))
        ys, xs = np.where(mask > 0)
        for _ in range(num_patches):
            idx = rng.integers(0, len(xs))
            cx, cy = int(xs[idx]), int(ys[idx])
            radius = int(rng.integers(15, 15 + int(60 * contamination_level)))
            # BGR colours chosen to sit within the "water" hue range but
            # outside the "clean water" band used by the detector, so they
            # simulate algae bloom (greenish) / turbid-murky (brownish-red)
            # patches that are still classified as *water*, just polluted.
            patch_color = (73, 120, 26) if rng.random() > 0.5 else (100, 29, 65)
            cv2.circle(img, (cx, cy), radius, patch_color, thickness=-1)
            # Slight irregular texture on the patch
            for _ in range(6):
                jx = cx + int(rng.integers(-radius, radius))
                jy = cy + int(rng.integers(-radius, radius))
                if 0 <= jx < width and 0 <= jy < height and mask[jy, jx] > 0:
                    cv2.circle(img, (jx, jy), int(radius * 0.4), patch_color, thickness=-1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic drone lake images for testing")
    parser.add_argument("--outdir", type=str, default="sample_images")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    scenarios = {
        "lake_day1_clean": 0.03,
        "lake_day2_mild": 0.20,
        "lake_day3_moderate": 0.45,
        "lake_day4_severe": 0.85,
    }

    for name, level in scenarios.items():
        img = make_lake_image(contamination_level=level, seed=hash(name) % 1000)
        path = os.path.join(args.outdir, f"{name}.jpg")
        cv2.imwrite(path, img)
        print(f"Saved {path} (simulated contamination level={level})")


if __name__ == "__main__":
    main()
