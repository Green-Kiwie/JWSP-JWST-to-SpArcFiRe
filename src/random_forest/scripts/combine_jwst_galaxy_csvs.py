#!/usr/bin/env python3
"""
Combine per-galaxy SpArcFiRe CSV outputs into a single CSV file
suitable for use with infer_jwst.py.

Walks the SpArcFiRe output directory and collects every file matching
GALAXYNAME.csv (skipping *_arcs.csv files), then concatenates them
into one file with a single header row.

Usage:
    python scripts/combine_jwst_galaxy_csvs.py \
        --input  /path/to/sparcfire/output \
        --output /path/to/combined_galaxy.csv

The output path's parent directory is created automatically.
"""

import argparse
import os
import sys
from pathlib import Path


def iter_galaxy_csvs(root: Path):
    """Yield paths to every galaxy-level CSV under root (no _arcs.csv files)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith('.csv') and not fname.endswith('_arcs.csv'):
                yield Path(dirpath) / fname


def combine(input_dir: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_written = False
    count = 0

    with output_path.open('w', encoding='utf-8') as out:
        for csv_path in iter_galaxy_csvs(input_dir):
            try:
                with csv_path.open('r', encoding='utf-8') as f:
                    lines = f.readlines()
            except OSError as e:
                print(f"WARNING: skipping {csv_path}: {e}", file=sys.stderr)
                continue

            if not lines:
                continue

            # First non-empty line is the header; the rest are data rows.
            # Skip files that only have a header (no data rows).
            data_lines = [l for l in lines[1:] if l.strip()]
            if not data_lines:
                continue

            if not header_written:
                out.write(lines[0])
                header_written = True

            for line in data_lines:
                out.write(line)

            count += 1
            if count % 10_000 == 0:
                print(f"  {count} galaxies written...", flush=True)

    return count


def main():
    parser = argparse.ArgumentParser(description='Combine JWST galaxy-level SpArcFiRe CSVs')
    parser.add_argument('--input', required=True,
                        help='Root SpArcFiRe output directory (e.g. cosmos_jwst_data/output)')
    parser.add_argument('--output', required=True,
                        help='Destination combined CSV path')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    if output_path.suffix.lower() != '.csv':
        output_path = output_path.with_suffix('.csv')

    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {input_dir} for galaxy CSVs...")
    n = combine(input_dir, output_path)
    print(f"Done. {n} galaxies written to {output_path}")


if __name__ == '__main__':
    main()
