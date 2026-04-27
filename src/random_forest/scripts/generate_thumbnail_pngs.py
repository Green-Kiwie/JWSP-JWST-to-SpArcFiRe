#!/usr/bin/env python3
"""
Generate annotated PNG thumbnails for JWST galaxies from inference results.

For each galaxy in a predict.csv, loads the corresponding .fits thumbnail,
renders it as a PNG with the predicted spirality score displayed below the
image, and saves it to the output directory named by its predicted score.

Usage:
    python scripts/generate_thumbnail_pngs.py \
        --predict inference/RUN-1/predict.csv \
        --fits-dir /extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails \
        --output-dir inference/RUN-1/images
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval

# Regex to parse CSV galaxy names like "0_471107_-15_482449_f090w_v1"
# into RA (0.471107), DEC (-15.482449), filter (f090w), version (v1).
_NAME_RE = re.compile(
    r'^(-?\d+)_(\d+)_(-?\d+)_(\d+)_(f\w+)_(v\d+)$'
)


def name_to_fits_filename(name: str) -> str:
    """Convert a predict.csv galaxy name to the corresponding .fits filename.

    Example:
        "0_471107_-15_482449_f090w_v1" -> "0.471107_-15.482449_f090w_v1.fits"
    """
    m = _NAME_RE.match(name)
    if not m:
        return None
    ra = f"{m.group(1)}.{m.group(2)}"
    dec = f"{m.group(3)}.{m.group(4)}"
    filt = m.group(5)
    ver = m.group(6)
    return f"{ra}_{dec}_{filt}_{ver}.fits"


def render_thumbnail(fits_path: Path, score: str, output_path: Path):
    """Load a FITS file, render it as an image with the score as a label."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        if data is None and len(hdul) > 1:
            data = hdul[1].data

    if data is None:
        print(f"  WARNING: No image data in {fits_path}, skipping")
        return False

    # Handle 3D+ data by taking first 2D slice
    while data.ndim > 2:
        data = data[0]

    data = data.astype(np.float64)

    # Use ZScale for astronomical image stretching
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(data)

    fig, ax = plt.subplots(figsize=(4, 4.6))
    ax.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    ax.set_title(score, fontsize=10, pad=8)

    fig.savefig(output_path, dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate annotated galaxy thumbnail PNGs from '
                    'inference predictions and FITS files.')
    parser.add_argument('--predict', required=True,
                        help='Path to predict.csv')
    parser.add_argument('--fits-dir', required=True,
                        help='Directory containing .fits thumbnails')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write output PNGs')
    args = parser.parse_args()

    predict_path = Path(args.predict)
    fits_dir = Path(args.fits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read predict.csv
    rows = []
    with open(predict_path, 'r') as f:
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            score = parts[1].strip()
            rows.append((name, score))

    print(f"Loaded {len(rows)} predictions from {predict_path}")

    created = 0
    skipped_name = 0
    skipped_missing = 0
    skipped_render = 0

    for i, (name, score) in enumerate(rows):
        fits_filename = name_to_fits_filename(name)
        if fits_filename is None:
            skipped_name += 1
            if skipped_name <= 5:
                print(f"  WARNING: Could not parse name '{name}', skipping")
            continue

        fits_path = fits_dir / fits_filename
        if not fits_path.exists():
            skipped_missing += 1
            if skipped_missing <= 5:
                print(f"  WARNING: FITS not found: {fits_path}, skipping")
            continue

        # Place into bin folder based on score (0.0-0.1, 0.1-0.2, …, 0.9-1.0)
        score_val = float(score)
        bin_lo = int(score_val * 10) / 10          # e.g. 0.18 -> 0.1
        bin_lo = min(bin_lo, 0.9)                   # clamp 1.0 into 0.9-1.0
        bin_hi = round(bin_lo + 0.1, 1)
        bin_folder = output_dir / f"{bin_lo:.1f}-{bin_hi:.1f}"
        bin_folder.mkdir(parents=True, exist_ok=True)

        output_path = bin_folder / f"{score}.png"
        if render_thumbnail(fits_path, score, output_path):
            created += 1

            if created % 1000 == 0:
                print(f"  Progress: {created} PNGs created "
                      f"({i + 1}/{len(rows)} processed)")
        else:
            skipped_render += 1

    print(f"\nDone: {created} PNGs created in {output_dir}")
    if skipped_name:
        print(f"  Skipped (unparseable name): {skipped_name}")
    if skipped_missing:
        print(f"  Skipped (FITS not found): {skipped_missing}")
    if skipped_render:
        print(f"  Skipped (render failed): {skipped_render}")


if __name__ == '__main__':
    main()
