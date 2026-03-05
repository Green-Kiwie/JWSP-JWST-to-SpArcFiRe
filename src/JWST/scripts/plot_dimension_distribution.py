#!/usr/bin/env python3
'''
Plot the pixel-area distribution of FITS thumbnail cutouts on a log-scaled x-axis.

Each thumbnail's pixel area (width × height) is computed and grouped into ~20
logarithmically spaced bins spanning 400–10 000 px².  The result is a bar chart
where the x-axis is log-scaled and the y-axis shows counts per bin.

Uses raw binary FITS header parsing (no astropy) and multiprocessing for speed.

Usage:
    python plot_dimension_distribution.py -i /path/to/fits/ -o distribution.png
    python plot_dimension_distribution.py -i /path/to/fits/ -o distribution.png --instrument NIRCAM
    python plot_dimension_distribution.py -i /path/to/fits/ -o distribution.png -j 16
'''

import os
import sys
import glob
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ---------------------------------------------------------------------------
# Fast raw FITS header reader — bypasses astropy entirely
# ---------------------------------------------------------------------------

_CARD_LEN = 80             # Each header card is 80 ASCII characters
_MAX_HEADER_BYTES = 28800   # Read up to 10 blocks (~360 cards) — plenty for thumbnails


def _parse_header_value(card_bytes: bytes) -> str:
    '''Extract the value portion from an 80-byte FITS header card.'''
    raw = card_bytes[10:].decode('ascii', errors='replace')
    if "'" in raw:
        start = raw.index("'") + 1
        end = raw.index("'", start)
        return raw[start:end].strip()
    else:
        if '/' in raw:
            raw = raw[:raw.index('/')]
        return raw.strip()


def read_thumbnail_info_fast(path: str):
    '''Read NAXIS1, NAXIS2, INSTRUME from raw FITS binary header.

    Only the first ~30 KB of the file is read — no full parsing, no astropy.
    Returns (pixel_area, instrument_str_or_None) or None on error.
    '''
    try:
        with open(path, 'rb') as f:
            header_bytes = f.read(_MAX_HEADER_BYTES)

        naxis1 = None
        naxis2 = None
        instrument = None

        for offset in range(0, len(header_bytes), _CARD_LEN):
            card = header_bytes[offset:offset + _CARD_LEN]
            if len(card) < _CARD_LEN:
                break
            keyword = card[:8].decode('ascii', errors='replace').rstrip()

            if keyword == 'NAXIS1':
                naxis1 = int(_parse_header_value(card))
            elif keyword == 'NAXIS2':
                naxis2 = int(_parse_header_value(card))
            elif keyword == 'INSTRUME':
                instrument = _parse_header_value(card)
            elif keyword == 'END':
                break

            if naxis1 is not None and naxis2 is not None and instrument is not None:
                break

        if naxis1 is None or naxis2 is None:
            return None

        return (naxis1 * naxis2, instrument)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument('-j', '--jobs', type=int, default=None,
                        help='Number of parallel workers (default: all CPUs)')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: input directory does not exist: {args.input}")
        sys.exit(1)

    fits_files = sorted(glob.glob(os.path.join(args.input, '*.fits')))
    if not fits_files:
        print(f"No .fits files found in {args.input}")
        sys.exit(1)

    n_workers = args.jobs if args.jobs else cpu_count()
    instrument_filter = args.instrument.upper().strip() if args.instrument else None

    print(f"Scanning {len(fits_files):,} files with {n_workers} workers ...")

    areas = []
    skipped_no_instrument = 0
    skipped_mismatch = 0
    errors = 0
    processed = 0

    chunk_size = max(1, len(fits_files) // (n_workers * 4))

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(read_thumbnail_info_fast, fits_files,
                                          chunksize=chunk_size):
            processed += 1
            if processed % 100_000 == 0:
                print(f"  ... {processed:,} / {len(fits_files):,}")

            if result is None:
                errors += 1
                continue

            pixel_area, instrument = result

            if instrument_filter is not None:
                if instrument is None:
                    skipped_no_instrument += 1
                    continue
                if instrument.upper() != instrument_filter:
                    skipped_mismatch += 1
                    continue

            areas.append(pixel_area)

    # Console summary
    print(f"\nTotal .fits files scanned : {len(fits_files):,}")
    print(f"Matched thumbnails        : {len(areas):,}")
    if instrument_filter:
        print(f"Filtered by instrument    : {instrument_filter}")
        print(f"  Skipped (no INSTRUME)   : {skipped_no_instrument:,}")
        print(f"  Skipped (other instr.)  : {skipped_mismatch:,}")
    if errors:
        print(f"  Unreadable files        : {errors:,}")

    if not areas:
        print("No thumbnails matched the criteria. No plot generated.")
        sys.exit(0)

    # Log-spaced binning
    n_bins = 20
    areas_arr = np.array(areas, dtype=float)
    bin_edges = np.logspace(np.log10(400), np.log10(10000), n_bins + 1)
    counts, _ = np.histogram(areas_arr, bins=bin_edges)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    print(f"\nPixel-area distribution ({n_bins} log-spaced bins):")
    for i in range(n_bins):
        lo, hi = int(round(bin_edges[i])), int(round(bin_edges[i + 1]))
        print(f"  {lo:>5d} – {hi:>5d} px²  —  {counts[i]:,} galaxies")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    title = 'Thumbnail Pixel-Area Distribution'
    if instrument_filter:
        title += f' — {instrument_filter}'

    widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], counts, width=widths, align='edge',
           color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlim(350, 11000)

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)

    tick_labels = [int(round(e)) for e in bin_edges]
    ax.set_xticks(bin_edges)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)

    ax.set_xlabel('Pixel Area (width × height)', fontsize=12)
    ax.set_ylabel('Galaxy Count', fontsize=12)
    ax.set_title(title, fontsize=14)

    fig.tight_layout()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nPlot saved to {args.output}")


if __name__ == '__main__':
    main()
