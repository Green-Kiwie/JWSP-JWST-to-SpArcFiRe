#!/usr/bin/env python3
'''
List all .fits thumbnails captured by a specified JWST instrument.

Uses raw binary FITS header parsing and multiprocessing for speed.

Usage:
    python list_by_instrument.py -i /path/to/fits/ -n NIRCAM -o nircam_files.txt
    python list_by_instrument.py -i /path/to/fits/ -n NIRISS -o niriss_files.txt -j 20
'''

import os
import sys
import glob
import argparse
from multiprocessing import Pool, cpu_count


# ---------------------------------------------------------------------------
# Fast raw FITS header reader
# ---------------------------------------------------------------------------

_CARD_LEN = 80
_MAX_HEADER_BYTES = 28800


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


def _read_instrument(path: str):
    '''Read INSTRUME from raw FITS header. Returns (basename, instrument) or None.'''
    try:
        with open(path, 'rb') as f:
            header_bytes = f.read(_MAX_HEADER_BYTES)

        for offset in range(0, len(header_bytes), _CARD_LEN):
            card = header_bytes[offset:offset + _CARD_LEN]
            if len(card) < _CARD_LEN:
                break
            keyword = card[:8].decode('ascii', errors='replace').rstrip()

            if keyword == 'INSTRUME':
                return (os.path.basename(path), _parse_header_value(card))
            elif keyword == 'END':
                break

        return (os.path.basename(path), None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='List .fits thumbnails captured by a specified JWST instrument.')
    parser.add_argument('-i', '--input', required=True,
                        help='Directory containing thumbnail .fits files')
    parser.add_argument('-n', '--instrument', required=True,
                        help='JWST instrument to filter by (e.g. NIRCAM, NIRISS, MIRI). '
                             'Case-insensitive.')
    parser.add_argument('-o', '--output', required=True,
                        help='Output .txt file path')
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
    instrument_filter = args.instrument.upper().strip()

    print(f"Scanning {len(fits_files):,} files with {n_workers} workers ...")
    print(f"Filtering for instrument: {instrument_filter}")

    matched = []
    errors = 0
    no_header = 0
    processed = 0

    chunk_size = max(1, len(fits_files) // (n_workers * 4))

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_read_instrument, fits_files,
                                          chunksize=chunk_size):
            processed += 1
            if processed % 100_000 == 0:
                print(f"  ... {processed:,} / {len(fits_files):,}")

            if result is None:
                errors += 1
                continue

            basename, instrument = result

            if instrument is None:
                no_header += 1
                continue

            if instrument.upper() == instrument_filter:
                matched.append(basename)

    matched.sort()

    # Console summary
    print(f"\nTotal .fits files scanned : {len(fits_files):,}")
    print(f"Matched ({instrument_filter})     : {len(matched):,}")
    if no_header:
        print(f"  Missing INSTRUME header : {no_header:,}")
    if errors:
        print(f"  Unreadable files        : {errors:,}")

    # Write output
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, 'w') as f:
        for name in matched:
            f.write(name + '\n')

    print(f"Wrote {len(matched):,} filenames to {args.output}")


if __name__ == '__main__':
    main()
