#!/usr/bin/env python3
'''
View random thumbnail FITS files in a grid.

Usage:
    python validate_filter.py -i /path/to/thumbnail/fits/
    python validate_filter.py -i /path/to/thumbnail/fits/ -n 25
    python validate_filter.py -i /path/to/thumbnail/fits/ -n 25 -s grid.png
'''

import os
import sys
import glob
import random
import argparse
import math
import numpy as np
from astropy.io import fits

# Defer matplotlib import until after arg parsing so we can pick the right backend


def load_thumbnail(path: str):
    '''Read a single thumbnail FITS file and return normalized 2D float array.'''
    try:
        with fits.open(path, memmap=True) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    d = hdu.data.astype(np.float64)
                    d = np.nan_to_num(d, nan=0.0)
                    lo, hi = np.percentile(d, [1, 99])
                    if hi > lo:
                        d = np.clip((d - lo) / (hi - lo), 0, 1)
                    else:
                        d[:] = 0
                    return d
    except Exception as e:
        print(f"  Skipping {os.path.basename(path)}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description='View random thumbnail FITS files in a grid.')
    parser.add_argument('-i', '--input', required=True,
                        help='Directory containing thumbnail .fits files')
    parser.add_argument('-n', '--num', type=int, default=16,
                        help='Number of thumbnails to show (default: 16)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random)')
    parser.add_argument('-s', '--save', type=str, default=None,
                        help='Save grid to this file (e.g. grid.png) instead of displaying')
    args = parser.parse_args()

    # Pick matplotlib backend BEFORE importing pyplot
    import matplotlib
    if args.save:
        matplotlib.use('Agg')
    else:
        # Try interactive backends; fall back to Agg
        for _backend in ('TkAgg', 'Qt5Agg', 'GTK3Agg', 'Agg'):
            try:
                matplotlib.use(_backend)
                break
            except ImportError:
                continue
    import matplotlib.pyplot as plt

    if args.seed is not None:
        random.seed(args.seed)

    # Find FITS files
    fits_files = glob.glob(os.path.join(args.input, '*.fits'))
    if not fits_files:
        print(f"No .fits files found in {args.input}")
        sys.exit(1)

    # Sample
    n = min(args.num, len(fits_files))
    sample = random.sample(fits_files, n)

    # Load thumbnails
    images = []
    names = []
    for path in sample:
        img = load_thumbnail(path)
        if img is not None:
            images.append(img)
            names.append(os.path.basename(path))

    if not images:
        print("No valid thumbnails loaded.")
        sys.exit(1)

    # Layout grid
    cols = min(len(images), math.ceil(math.sqrt(len(images))))
    rows = math.ceil(len(images) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx, (img, name) in enumerate(zip(images, names)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.imshow(img, cmap='gray', origin='lower')
        ax.set_title(name, fontsize=6)
        ax.axis('off')

    # Turn off unused subplots
    for idx in range(len(images), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')

    fig.suptitle(f'{len(images)} thumbnails from {os.path.basename(args.input.rstrip("/"))}',
                 fontsize=10)
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show(block=True)


if __name__ == '__main__':
    main()
