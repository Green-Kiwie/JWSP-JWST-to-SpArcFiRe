#!/usr/bin/env python3
'''
Plot the pixel-area distribution of FITS thumbnail cutouts on a log-scaled x-axis.

Each thumbnail's pixel area (width × height) is computed and counted.  The result
is a bar chart where the x-axis shows distinct pixel-area values on a log scale
and the y-axis shows how many thumbnails fall under each value.

Usage:
    python plot_dimension_distribution.py -i /path/to/fits/ -o distribution.png
    python plot_dimension_distribution.py -i /path/to/fits/ -o distribution.png --instrument NIRCAM
'''

import os
import sys
import glob
import argparse
from collections import Counter

import numpy as np
from astropy.io import fits

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def read_thumbnail_info(path: str):
    '''Read a single FITS thumbnail header and return (pixel_area, instrument).

    Only the primary header is read (no pixel data), which is significantly
    faster when scanning thousands of files.

    Returns:
        Tuple of (int, str | None) on success, or None on failure.
    '''
    try:
        header = fits.getheader(path, 0)
        naxis1 = header.get('NAXIS1')
        naxis2 = header.get('NAXIS2')
        if naxis1 is None or naxis2 is None:
            print(f"  Warning: missing NAXIS dimensions in {os.path.basename(path)}")
            return None
        instrument = header.get('INSTRUME', None)
        if instrument is not None:
            instrument = str(instrument).strip()
        return (int(naxis1) * int(naxis2), instrument)
    except Exception as e:
        print(f"  Warning: could not read {os.path.basename(path)}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Plot pixel-area distribution of FITS thumbnail cutouts.')
    parser.add_argument('-i', '--input', required=True,
                        help='Directory containing thumbnail .fits files')
    parser.add_argument('-o', '--output', required=True,
                        help='Output path for the .png plot')
    parser.add_argument('-n', '--instrument', type=str, default=None,
                        help='Filter by JWST instrument (e.g. NIRCAM, NIRISS, MIRI). '
                             'Case-insensitive. If omitted, all files are included.')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: input directory does not exist: {args.input}")
        sys.exit(1)

    fits_files = sorted(glob.glob(os.path.join(args.input, '*.fits')))
    if not fits_files:
        print(f"No .fits files found in {args.input}")
        sys.exit(1)

    instrument_filter = args.instrument.upper().strip() if args.instrument else None

    areas = []
    skipped_no_instrument = 0
    skipped_mismatch = 0
    errors = 0

    for path in fits_files:
        info = read_thumbnail_info(path)
        if info is None:
            errors += 1
            continue

        pixel_area, instrument = info

        if instrument_filter is not None:
            if instrument is None:
                skipped_no_instrument += 1
                continue
            if instrument.upper() != instrument_filter:
                skipped_mismatch += 1
                continue

        areas.append(pixel_area)

    # Console summary
    print(f"Total .fits files scanned : {len(fits_files)}")
    print(f"Matched thumbnails        : {len(areas)}")
    if instrument_filter:
        print(f"Filtered by instrument    : {instrument_filter}")
        print(f"  Skipped (no INSTRUME)   : {skipped_no_instrument}")
        print(f"  Skipped (other instr.)  : {skipped_mismatch}")
    if errors:
        print(f"  Unreadable files        : {errors}")

    if not areas:
        print("No thumbnails matched the criteria. No plot generated.")
        sys.exit(0)

    counter = Counter(areas)
    unique_areas = sorted(counter.keys())
    counts = [counter[a] for a in unique_areas]

    print(f"\nPixel-area breakdown ({len(unique_areas)} distinct values):")
    for area, count in zip(unique_areas, counts):
        print(f"  {area:>6d} px²  —  {count} galaxies")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([str(a) for a in unique_areas], counts, color='steelblue', edgecolor='black',
           linewidth=0.4)

    ax.set_xlabel('Pixel Area (width × height)', fontsize=12)
    ax.set_ylabel('Galaxy Count', fontsize=12)

    title = 'Thumbnail Pixel-Area Distribution'
    if instrument_filter:
        title += f' — {instrument_filter}'
    ax.set_title(title, fontsize=14)

    ax.tick_params(axis='x', labelrotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=10)

    # Since the user wants a log-scale x-axis and the values are discrete,
    # we position bars at their true log-spaced locations instead of evenly.
    ax.cla()  # clear the categorical bars — replot with numeric positions

    bar_positions = np.array(unique_areas, dtype=float)
    # Compute bar widths that look reasonable on a log scale
    log_positions = np.log10(bar_positions)
    if len(log_positions) > 1:
        diffs = np.diff(log_positions)
        min_diff = diffs.min()
        bar_width_factor = 0.8 * min_diff
    else:
        bar_width_factor = 0.05

    # Width of each bar in log-space, converted back to linear for plotting
    widths = bar_positions * (10**bar_width_factor - 1) * 0.9

    ax.bar(bar_positions, counts, width=widths, color='steelblue', edgecolor='black',
           linewidth=0.4, align='center')
    ax.set_xscale('log')
    ax.set_xlim(300, 12000)

    # Format x-ticks as plain integers
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)

    # Add minor tick labels at all unique areas
    ax.set_xticks(unique_areas, minor=True)
    ax.set_xticklabels([str(a) for a in unique_areas], minor=True, fontsize=6, rotation=45)
    ax.tick_params(axis='x', which='minor', length=3)
    ax.tick_params(axis='x', which='major', length=5)

    ax.set_xlabel('Pixel Area (width × height)', fontsize=12)
    ax.set_ylabel('Galaxy Count', fontsize=12)
    ax.set_title(title, fontsize=14)

    fig.tight_layout()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nPlot saved to {args.output}")


if __name__ == '__main__':
    main()
