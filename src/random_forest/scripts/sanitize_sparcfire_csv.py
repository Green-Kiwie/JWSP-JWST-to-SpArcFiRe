#!/usr/bin/env python3
"""
Sanitize a SpArcFiRe CSV (with P_CW/P_ACW already prepended by
match_ground_truth.py) to match the format of data_cleaned.csv used
for random forest training.

Transformations applied:
  - Drop extra columns not in data_cleaned (cropRad, fitQualityF1,
    fitQualityPCC, pa_avg_abs, pa_alenWtd_avg_abs, paErr_alenWtd_stdev_abs)
  - Rename chirality values: Z-wise -> Zwise, S-wise -> Swise
  - Rename mask values: unavailable -> _, aggressive-exclusive -> aggressiveexclusive, etc.
  - Capitalize booleans: true -> True, false -> False
  - Fix top2_chirality_agreement: 'one-long' -> 'onelong', 'all-short' -> 'allshort', etc.
  - Replace spaces with underscores in: chirality_votes_maj,
    chirality_votes_alenWtd, sorted_agreeing_pangs, sorted_agreeing_arclengths
  - Remove semicolons from covarFit (keep spaces)
  - Drop rows containing -Inf, Inf, or NaN field values
  - Drop rows with blank fields in key columns
  - Drop rows where fit_state is 'input rejected'
  - Re-number 0-based row index

Usage:
    python scripts/sanitize_sparcfire_csv.py \\
        testing_data/SpArcFiRe_with_ground_truth.csv \\
        -o testing_data/SpArcFiRe_cleaned.csv
"""

import argparse
import sys


# Columns present in SpArcFiRe output but NOT in data_cleaned.csv
COLUMNS_TO_DROP = {
    'cropRad',
    'fitQualityF1',
    'fitQualityPCC',
    'pa_avg_abs',
    'pa_alenWtd_avg_abs',
    'paErr_alenWtd_stdev_abs',
}

# Columns where Z-wise/S-wise should be renamed
CHIRALITY_COLUMNS = {
    'chirality_maj',
    'chirality_alenWtd',
    'chirality_wtdPangSum',
    'chirality_longestArc',
}

# Columns where 'unavailable' -> '_' and hyphenated names are fixed
MASK_COLUMNS = {
    'star_mask_used',
    'noise_mask_used',
}

# Columns where spaces should be replaced with underscores
SPACE_TO_UNDERSCORE_COLUMNS = {
    'chirality_votes_maj',
    'chirality_votes_alenWtd',
    'sorted_agreeing_pangs',
    'sorted_agreeing_arclengths',
}

# Boolean-valued columns: true -> True, false -> False
BOOLEAN_COLUMNS = {
    'badBulgeFitFlag',
    'hasDeletedCtrClus',
    'failed2revDuringMergeCheck',
    'failed2revDuringSecondaryMerging',
    'failed2revInOutput',
    'bar_candidate_available',
    'bar_used',
}

# Values that indicate a bad row when found as an exact field value
BAD_VALUES = {'-Inf', 'Inf', 'NaN', '-inf', 'inf', 'nan'}


def build_column_map(headers):
    """Build a dict mapping column name -> index."""
    return {name: i for i, name in enumerate(headers)}


def transform_row(fields, col_map):
    """Apply all value transformations to a row in-place.

    Returns:
        True if the row should be kept, False if it should be dropped.
    """
    # --- Check for bad values ---
    for val in fields:
        if val in BAD_VALUES:
            return False

    # --- Check fit_state ---
    fit_state_idx = col_map.get('fit_state')
    if fit_state_idx is not None and fields[fit_state_idx] == 'input rejected':
        return False

    # --- Check for blank key fields (columns 2-20 of SpArcFiRe, offset by 3) ---
    for i in range(4, min(23, len(fields))):
        if fields[i].strip() == '':
            return False

    # --- Chirality: Z-wise -> Zwise, S-wise -> Swise ---
    for col_name in CHIRALITY_COLUMNS:
        idx = col_map.get(col_name)
        if idx is not None and idx < len(fields):
            if fields[idx] == 'Z-wise':
                fields[idx] = 'Zwise'
            elif fields[idx] == 'S-wise':
                fields[idx] = 'Swise'

    # --- Mask columns: unavailable -> _, fix hyphenated names ---
    for col_name in MASK_COLUMNS:
        idx = col_map.get(col_name)
        if idx is not None and idx < len(fields):
            val = fields[idx]
            if val == 'unavailable':
                fields[idx] = 'none'
            elif val == 'aggressive-exclusive':
                fields[idx] = 'aggressiveexclusive'
            elif val == 'conservative-exclusive':
                fields[idx] = 'conservativeexclusive'

    # --- Booleans: true -> True, false -> False ---
    for col_name in BOOLEAN_COLUMNS:
        idx = col_map.get(col_name)
        if idx is not None and idx < len(fields):
            if fields[idx] == 'true':
                fields[idx] = 'True'
            elif fields[idx] == 'false':
                fields[idx] = 'False'

    # --- top2_chirality_agreement fixes ---
    idx = col_map.get('top2_chirality_agreement')
    if idx is not None and idx < len(fields):
        val = fields[idx]
        if val == "'one-long'":
            fields[idx] = "'onelong'"
        elif val == "'all-short'":
            fields[idx] = "'allshort'"
        elif val == '<2 arcs':
            fields[idx] = "'<2_arcs'"

    # --- Space -> underscore in array columns ---
    for col_name in SPACE_TO_UNDERSCORE_COLUMNS:
        idx = col_map.get(col_name)
        if idx is not None and idx < len(fields):
            fields[idx] = fields[idx].replace(' ', '_')

    # --- covarFit: remove semicolons ---
    idx = col_map.get('covarFit')
    if idx is not None and idx < len(fields):
        fields[idx] = fields[idx].replace(';', ' ')

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Sanitize SpArcFiRe CSV to match data_cleaned.csv format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input_csv',
                        help='Path to SpArcFiRe CSV with P_CW/P_ACW '
                             '(output of match_ground_truth.py)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output CSV path')
    args = parser.parse_args()

    kept = 0
    dropped = 0

    with open(args.input_csv, 'r') as fin, \
         open(args.output, 'w') as fout:

        # --- Process header ---
        header_line = fin.readline().rstrip('\n')
        headers = [h.strip() for h in header_line.split(',')]

        # Remove trailing empty columns
        while headers and headers[-1] == '':
            headers.pop()

        # Find indices of columns to drop
        drop_indices = set()
        for col_name in COLUMNS_TO_DROP:
            if col_name in headers:
                drop_indices.add(headers.index(col_name))

        # Build output headers (excluding dropped columns)
        out_headers = [h for i, h in enumerate(headers) if i not in drop_indices]

        # Build column map for transformation (BEFORE dropping columns)
        col_map = build_column_map(headers)

        # Write output header
        fout.write(','.join(out_headers) + '\n')

        # --- Process data rows ---
        for line in fin:
            line = line.rstrip('\n')
            if not line.strip():
                continue

            fields = [f.strip() for f in line.split(',')]

            # Remove trailing empty fields
            while fields and fields[-1] == '':
                fields.pop()

            # Pad fields if shorter than header
            while len(fields) < len(headers):
                fields.append('')

            # Apply transformations
            keep = transform_row(fields, col_map)

            if not keep:
                dropped += 1
                continue

            # Drop extra columns
            out_fields = [f for i, f in enumerate(fields)
                          if i < len(headers) and i not in drop_indices]

            # Update 0-based index (first column)
            out_fields[0] = str(kept)

            fout.write(','.join(out_fields) + '\n')
            kept += 1

    print(f"Results:", file=sys.stderr)
    print(f"  Rows kept:    {kept}", file=sys.stderr)
    print(f"  Rows dropped: {dropped}", file=sys.stderr)
    print(f"  Output:       {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
