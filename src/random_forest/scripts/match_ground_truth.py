#!/usr/bin/env python3
"""
Match galaxies in a SpArcFiRe output CSV to Galaxy Zoo 2 data by RA/DEC
coordinates encoded in the galaxy name, then prepend P_CW, P_ACW, and a
0-based row index.

Galaxy names in SpArcFiRe follow the format:
    t{RA_int}_{RA_dec}[+-]{DEC_int}_{DEC_dec}_{band}
    e.g. t045_8870+0_5025_r  ->  RA=45.8870, DEC=+0.5025

Matching uses RA/DEC rounded to 4 decimal places against galaxy_zoo_2.csv.
Unmatched rows are dropped.

Usage:
    python scripts/match_ground_truth.py \\
        testing_data/SpArcFiRe_partial_PANSTARRS_overlap.csv \\
        testing_data/galaxy_zoo_2.csv \\
        -o testing_data/SpArcFiRe_with_ground_truth.csv
"""

import argparse
import re
import sys


def parse_name_to_radec(name):
    """Parse a SpArcFiRe galaxy name to extract RA and DEC.

    Args:
        name: Galaxy name like 't045_8870+0_5025_r'

    Returns:
        (ra, dec) as floats, or (None, None) if parsing fails.
    """
    name = name.strip()
    m = re.match(r'^t(\d+_\d+)([+-])(\d+_\d+)_[ri]$', name)
    if not m:
        return None, None
    ra = float(m.group(1).replace('_', '.'))
    dec = float(m.group(2) + m.group(3).replace('_', '.'))
    return ra, dec


def build_gz2_lookup(gz2_path):
    """Build a lookup dict from galaxy_zoo_2.csv.

    Keys are (round(ra, 4), round(dec, 4)).
    Values are the t04_spiral_a08_spiral_weighted_fraction string.

    Returns:
        dict, int: lookup dictionary and total rows loaded.
    """
    lookup = {}
    total = 0
    with open(gz2_path, 'r') as f:
        header = f.readline().strip().split(',')
        ra_idx = header.index('ra')
        dec_idx = header.index('dec')
        spiral_idx = header.index('t04_spiral_a08_spiral_weighted_fraction')

        for line in f:
            parts = line.strip().split(',')
            if len(parts) <= max(ra_idx, dec_idx, spiral_idx):
                continue
            try:
                ra = float(parts[ra_idx])
                dec = float(parts[dec_idx])
                spiral_frac = parts[spiral_idx].strip()
                key = (round(ra, 4), round(dec, 4))
                lookup[key] = spiral_frac
                total += 1
            except (ValueError, IndexError):
                continue

    return lookup, total


def strip_trailing_empty(fields):
    """Remove trailing empty/whitespace-only fields."""
    while fields and fields[-1].strip() == '':
        fields.pop()
    return fields


def main():
    parser = argparse.ArgumentParser(
        description='Match SpArcFiRe galaxies to Galaxy Zoo 2 ground truth '
                    'and prepend P_CW, P_ACW, and 0-based index columns.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('sparcfire_csv',
                        help='Path to SpArcFiRe output CSV '
                             '(e.g. SpArcFiRe_partial_PANSTARRS_overlap.csv)')
    parser.add_argument('gz2_csv',
                        help='Path to galaxy_zoo_2.csv')
    parser.add_argument('-o', '--output', required=True,
                        help='Output CSV path')
    args = parser.parse_args()

    # --- Load Galaxy Zoo 2 lookup ---
    print(f"Loading Galaxy Zoo 2 data from {args.gz2_csv}...", file=sys.stderr)
    lookup, gz2_total = build_gz2_lookup(args.gz2_csv)
    print(f"  Loaded {gz2_total} rows -> {len(lookup)} unique (RA, DEC) keys",
          file=sys.stderr)

    # --- Process SpArcFiRe CSV ---
    matched = 0
    unmatched = 0
    parse_errors = 0

    with open(args.sparcfire_csv, 'r') as fin, \
         open(args.output, 'w') as fout:

        # Read and write header
        header_line = fin.readline().rstrip('\n')
        headers = [h.strip() for h in header_line.split(',')]
        headers = strip_trailing_empty(headers)

        # Output header: unnamed index, P_CW, P_ACW, then all SpArcFiRe columns
        fout.write(',P_CW,P_ACW,' + ','.join(headers) + '\n')

        # Process data rows
        for line in fin:
            line = line.rstrip('\n')
            if not line.strip():
                continue

            fields = line.split(',')
            fields = [f.strip() for f in fields]
            fields = strip_trailing_empty(fields)

            if not fields:
                continue

            name = fields[0]
            ra, dec = parse_name_to_radec(name)

            if ra is None:
                print(f"  WARNING: Could not parse name '{name}', skipping.",
                      file=sys.stderr)
                parse_errors += 1
                continue

            key = (round(ra, 4), round(dec, 4))
            spiral_frac = lookup.get(key)

            if spiral_frac is None:
                print(f"  No match for '{name}' (RA={ra}, DEC={dec}), "
                      f"dropping row.", file=sys.stderr)
                unmatched += 1
                continue

            # Write: index, P_CW, P_ACW, then all original fields
            out_fields = [str(matched), spiral_frac, '0'] + fields
            fout.write(','.join(out_fields) + '\n')
            matched += 1

    # --- Summary ---
    total = matched + unmatched + parse_errors
    print(f"\nResults:", file=sys.stderr)
    print(f"  Total input rows:  {total}", file=sys.stderr)
    print(f"  Matched:           {matched}", file=sys.stderr)
    print(f"  Unmatched/dropped: {unmatched}", file=sys.stderr)
    print(f"  Parse errors:      {parse_errors}", file=sys.stderr)
    print(f"  Output written to: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
