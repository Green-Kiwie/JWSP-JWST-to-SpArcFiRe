#!/usr/bin/env python3
'''
Waveband Quality Inspection Tool
=================================
Extracts sample galaxy thumbnails per waveband from JWST all-sky images,
producing PNG montage grids for visual quality assessment.

Uses the same extraction pipeline as parallel_galaxy_extractor.py
(SEP detection at threshold=3σ, 20-100px size limits, 7-filter galaxy pipeline).

Usage:
    python inspect_wavebands.py
    python inspect_wavebands.py --input-dir /path/to/allsky/ --output-dir ./inspection/
    python inspect_wavebands.py --n-thumbnails 25 --max-per-file 5 --cols 5
    python inspect_wavebands.py --no-galaxy-filter   # skip morphological filtering

python inspect_wavebands.py \
    --input-dir /extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/outputs/JWST_full_all_sky_output/ \
    --output-dir /extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/outputs/waveband_inspection/ \
    --n-thumbnails 50 \
    --max-per-file 10
'''

import os
import sys
import glob
import random
import argparse
import math
import time
import traceback
import numpy as np

# Add modules directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'modules'))

from waveband_extraction import extract_waveband_from_filename, extract_instrument_from_filename

# ─── Constants ───────────────────────────────────────────────────────────────

DEFAULT_INPUT_DIR = '/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/outputs/JWST_full_all_sky_output/'
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'outputs', 'waveband_inspection')

# Extraction parameters
DETECTION_THRESHOLD = 3.0
MIN_SIZE = 20
MAX_SIZE = 100
PADDING = 1.5
EDGE_MARGIN = 10
MAX_NAN_FRACTION = 0.2
PIXEL_STACK_LIMIT = 5_000_000
SUB_OBJECT_LIMIT = 50_000


# ─── Helpers ─────────────────────────────────────────────────────────────────

def scan_and_group_files(input_dir: str) -> dict:
    '''Scan directory for *_i2d.fits files and group by (instrument, waveband).
    
    Returns:
        dict mapping (instrument, waveband) -> list of file paths
    '''
    pattern = os.path.join(input_dir, '*_i2d.fits')
    fits_files = sorted(glob.glob(pattern))

    groups = {}
    unclassified_count = 0

    for filepath in fits_files:
        filename = os.path.basename(filepath)
        waveband = extract_waveband_from_filename(filename)
        instrument = extract_instrument_from_filename(filename)

        if waveband == 'unclassified' or instrument is None:
            unclassified_count += 1
            continue

        key = (instrument, waveband)
        groups.setdefault(key, []).append(filepath)

    if unclassified_count > 0:
        print(f"  Skipped {unclassified_count} files with unrecognized naming.")

    return groups


def extract_thumbnails_from_file(filepath: str, max_thumbnails: int,
                                  apply_galaxy_filter: bool = True,
                                  verbosity: int = 0) -> list:
    '''Extract galaxy thumbnails from a single all-sky FITS file.
    
    Uses the same pipeline as parallel_galaxy_extractor.py:
      1. Load image data + NaN mask
      2. Background subtraction via SEP
      3. Source extraction at threshold=3σ
      4. (Optional) 7-filter morphological galaxy filtering
      5. Crop thumbnails with size/edge/NaN quality gates
    
    Args:
        filepath: Path to the all-sky FITS image
        max_thumbnails: Maximum number of thumbnails to return from this file
        apply_galaxy_filter: Whether to apply the 7-filter morphological pipeline
        verbosity: 0=quiet, 1=summary, 2=verbose
    
    Returns:
        List of 2D numpy arrays (cropped galaxy thumbnails)
    '''
    import sep_helpers as sh
    import sep_pjw as sep

    filename = os.path.basename(filepath)

    # Load FITS data
    try:
        image_data = sh.get_main_fits_data(filepath)
    except Exception as e:
        if verbosity >= 1:
            print(f"    Failed to load {filename}: {e}")
        return []

    if image_data is None or image_data.size == 0:
        return []

    # Compute NaN mask from raw SCI extension (before NaN filling)
    nan_mask = None
    try:
        from astropy.io import fits as _fits
        with _fits.open(filepath) as _hdul:
            try:
                _raw = _hdul['SCI'].data
            except (KeyError, IndexError):
                _raw = _hdul[0].data
            if _raw is not None:
                nan_mask = np.isnan(_raw)
    except Exception:
        pass

    # Background subtraction
    try:
        bkg = sh.get_image_background(image_data)
        bkgless_data = sh.subtract_bkg(image_data)
    except Exception as e:
        if verbosity >= 1:
            print(f"    Background subtraction failed for {filename}: {e}")
        return []

    # Source extraction (threshold=3σ, same limits as parallel_galaxy_extractor.py)
    try:
        celestial_objects = sh.extract_objects(
            bkgless_data, bkg,
            detection_threshold=DETECTION_THRESHOLD,
            pixel_stack_limit=PIXEL_STACK_LIMIT,
            sub_object_limit=SUB_OBJECT_LIMIT
        )
    except Exception as e:
        if verbosity >= 1:
            print(f"    SEP extraction failed for {filename}: {e}")
        return []

    # Apply 7-filter galaxy morphological pipeline
    if apply_galaxy_filter and len(celestial_objects) > 0:
        try:
            bkg_rms_val = sep.Background(image_data).globalrms
            celestial_objects = sh.filter_galaxies(
                bkgless_data, celestial_objects, bkg_rms_val,
                verbosity=verbosity
            )
        except Exception as e:
            if verbosity >= 1:
                print(f"    Galaxy filter failed for {filename} ({e}), using unfiltered detections")

    if len(celestial_objects) == 0:
        return []

    # Shuffle detections so we get a random sample from across the image
    indices = list(range(len(celestial_objects)))
    random.shuffle(indices)

    thumbnails = []
    for idx in indices:
        if len(thumbnails) >= max_thumbnails:
            break

        obj = celestial_objects[idx]
        # Crop from background-subtracted image for better contrast
        cropped = sh._extract_object(
            obj, bkgless_data,
            padding=PADDING,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            verbosity=0,
            nan_mask=nan_mask,
            edge_margin=EDGE_MARGIN,
            max_nan_fraction=MAX_NAN_FRACTION
        )

        if cropped is not None:
            thumbnails.append(cropped)

    return thumbnails


def normalize_thumbnail(data: np.ndarray) -> np.ndarray:
    '''Normalize a 2D array to [0, 1] using 1st/99th percentile clipping.
    Same approach as validate_filter.py.'''
    d = data.astype(np.float64)
    d = np.nan_to_num(d, nan=0.0)
    lo, hi = np.percentile(d, [1, 99])
    if hi > lo:
        d = np.clip((d - lo) / (hi - lo), 0, 1)
    else:
        d[:] = 0
    return d


def render_montage(thumbnails: list, title: str, save_path: str,
                   cols: int = 10, cell_size: float = 2.0) -> None:
    '''Render a grid of thumbnails and save as PNG.
    
    Args:
        thumbnails: List of 2D numpy arrays
        title: Figure title (shown at top)
        save_path: Output PNG path
        cols: Number of columns in the grid
        cell_size: Size of each cell in inches
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(thumbnails)
    if n == 0:
        print(f"    No thumbnails to render for: {title}")
        return

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cell_size * cols, cell_size * rows + 0.6))

    # Ensure axes is always 2D
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.atleast_2d(axes)
        if rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(-1, 1)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        if idx < n:
            img = normalize_thumbnail(thumbnails[idx])
            ax.imshow(img, cmap='gray', origin='lower')
            h, w = thumbnails[idx].shape
            ax.set_title(f'{w}x{h}', fontsize=5, pad=1)
        ax.axis('off')

    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Waveband quality inspection: extract sample thumbnails per waveband from JWST all-sky images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python inspect_wavebands.py
  python inspect_wavebands.py --input-dir /my/allsky/ --output-dir ./out/
  python inspect_wavebands.py --n-thumbnails 25 --max-per-file 5
  python inspect_wavebands.py --no-galaxy-filter
        '''
    )
    parser.add_argument('--input-dir', type=str, default=DEFAULT_INPUT_DIR,
                        help='Directory containing JWST all-sky *_i2d.fits files')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory where montage PNGs will be saved')
    parser.add_argument('--n-thumbnails', type=int, default=50,
                        help='Number of thumbnails to collect per waveband (default: 50)')
    parser.add_argument('--max-per-file', type=int, default=10,
                        help='Max thumbnails to take from a single file (default: 10)')
    parser.add_argument('--cols', type=int, default=10,
                        help='Number of columns in the montage grid (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no-galaxy-filter', action='store_true',
                        help='Skip the 7-filter morphological galaxy pipeline')
    parser.add_argument('--include-subarrays', action='store_true',
                        help='Include subarray variants (e.g., clear-f200w-sub64). '
                             'Excluded by default since subarrays are too small for galaxy extraction.')
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2],
                        help='Verbosity level (0=quiet, 1=normal, 2=verbose)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    apply_galaxy_filter = not args.no_galaxy_filter
    skip_subarrays = not args.include_subarrays

    # ── Phase 1: Scan & Group ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Waveband Quality Inspection Tool")
    print(f"{'='*70}")
    print(f"Input directory : {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Thumbnails/band : {args.n_thumbnails}")
    print(f"Max per file    : {args.max_per_file}")
    print(f"Galaxy filter   : {'ON (7-filter pipeline)' if apply_galaxy_filter else 'OFF (SEP + size only)'}")
    print(f"Skip subarrays  : {'YES' if skip_subarrays else 'NO'}")
    print(f"Random seed     : {args.seed}")
    print()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    print("Phase 1: Scanning and grouping all-sky images by instrument + waveband...")
    groups = scan_and_group_files(args.input_dir)

    # Filter out subarray variants (e.g., clear-f200w-sub64, clear-f200w-sub160p) unless requested
    if skip_subarrays:
        import re as _re
        subarray_pattern = _re.compile(r'-sub\d+p?$', _re.IGNORECASE)
        # Also filter "brightsky" calibration modes
        brightsky_pattern = _re.compile(r'-brightsky$', _re.IGNORECASE)
        filtered_groups = {
            k: v for k, v in groups.items()
            if not subarray_pattern.search(k[1]) and not brightsky_pattern.search(k[1])
        }
        n_removed = len(groups) - len(filtered_groups)
        if n_removed > 0:
            print(f"  Filtered out {n_removed} subarray/calibration groups (use --include-subarrays to keep them)")
        groups = filtered_groups

    if not groups:
        print("ERROR: No valid *_i2d.fits files found.")
        sys.exit(1)

    # Print summary table
    print(f"\n  {'Instrument':<12} {'Waveband':<20} {'Files':>6}")
    print(f"  {'-'*12} {'-'*20} {'-'*6}")
    for (instrument, waveband), files in sorted(groups.items()):
        print(f"  {instrument:<12} {waveband:<20} {len(files):>6}")
    print(f"\n  Total: {len(groups)} unique (instrument, waveband) groups, "
          f"{sum(len(f) for f in groups.values())} files\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Phase 2 & 3: Extract thumbnails per group ────────────────────────────
    print("Phase 2: Extracting thumbnails per waveband...")
    print(f"  (This may take a while — processing large FITS files)\n")

    results = []  # (instrument, waveband, n_files_available, n_files_processed, n_thumbnails)

    for group_idx, ((instrument, waveband), file_list) in enumerate(sorted(groups.items()), 1):
        group_label = f"{instrument}_{waveband}"
        print(f"  [{group_idx}/{len(groups)}] {group_label} ({len(file_list)} files available)")

        # Shuffle files so we don't always process the same ones
        shuffled_files = file_list.copy()
        random.shuffle(shuffled_files)

        collected_thumbnails = []
        files_processed = 0
        files_errored = 0

        for filepath in shuffled_files:
            if len(collected_thumbnails) >= args.n_thumbnails:
                break

            remaining = args.n_thumbnails - len(collected_thumbnails)
            # Cap how many we take from one file to avoid overrepresentation
            max_from_this_file = min(args.max_per_file, remaining)

            files_processed += 1
            filename = os.path.basename(filepath)

            if args.verbosity >= 2:
                print(f"    Processing {filename}...")

            t0 = time.time()
            try:
                thumbs = extract_thumbnails_from_file(
                    filepath,
                    max_thumbnails=max_from_this_file,
                    apply_galaxy_filter=apply_galaxy_filter,
                    verbosity=args.verbosity
                )
                collected_thumbnails.extend(thumbs)
                elapsed = time.time() - t0

                if args.verbosity >= 1:
                    print(f"    {filename}: {len(thumbs)} thumbnails ({elapsed:.1f}s) "
                          f"[total: {len(collected_thumbnails)}/{args.n_thumbnails}]")

            except Exception as e:
                files_errored += 1
                elapsed = time.time() - t0
                print(f"    ERROR processing {filename} ({elapsed:.1f}s): {e}")
                if args.verbosity >= 2:
                    traceback.print_exc()

        # ── Phase 4: Render montage ──────────────────────────────────────────
        n_collected = len(collected_thumbnails)
        montage_title = (f"{group_label}  ({n_collected} thumbnails from "
                         f"{files_processed} files)")
        save_path = os.path.join(args.output_dir, f"{group_label}_montage.png")

        if n_collected > 0:
            render_montage(
                collected_thumbnails,
                title=montage_title,
                save_path=save_path,
                cols=args.cols
            )
            print(f"    -> Saved montage: {save_path}")
        else:
            print(f"    -> No thumbnails extracted for {group_label} (skipping montage)")

        results.append((instrument, waveband, len(file_list), files_processed,
                         files_errored, n_collected))
        print()

    # ── Phase 5: Summary Report ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Summary Report")
    print(f"{'='*70}")
    print(f"\n  {'Instrument':<12} {'Waveband':<20} {'Files':>6} {'Processed':>10} {'Errors':>7} {'Thumbnails':>11}")
    print(f"  {'-'*12} {'-'*20} {'-'*6} {'-'*10} {'-'*7} {'-'*11}")

    total_thumbs = 0
    total_processed = 0
    for (instrument, waveband, n_files, n_proc, n_err, n_thumbs) in results:
        print(f"  {instrument:<12} {waveband:<20} {n_files:>6} {n_proc:>10} {n_err:>7} {n_thumbs:>11}")
        total_thumbs += n_thumbs
        total_processed += n_proc

    print(f"\n  Total: {total_thumbs} thumbnails from {total_processed} files processed")
    print(f"  Montages saved to: {os.path.abspath(args.output_dir)}")
    print()


if __name__ == '__main__':
    main()
