#!/usr/bin/env python3
"""Download JWST sky patch FITS products listed in a manifest CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, List, Tuple
import urllib.request


DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "inputs" / "jwst_L3_sky_patches.csv"
DEFAULT_DOWNLOAD_DIRECTORY = Path(__file__).resolve().parents[3] / "outputs" / "JWST_L3_sky_patches"


def load_manifest_rows(csv_file_path: str | Path) -> List[Tuple[str, str]]:
    """Load downloadable `(data_uri, product_filename)` rows from a manifest CSV."""
    csv_path = Path(csv_file_path)
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        preview_reader = csv.reader(file)
        first_row = next(preview_reader, None)

    if first_row is None:
        return []

    header_names = {value.strip() for value in first_row}
    if {"dataURI", "productFilename", "calib_level"}.issubset(header_names):
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            return [
                (row.get("dataURI", "").strip(), row.get("productFilename", "").strip())
                for row in reader
                if str(row.get("calib_level", "")).strip() == "3"
                and row.get("dataURI", "").strip()
                and row.get("productFilename", "").strip()
            ]

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        return [
            (row[6].strip(), row[14].strip())
            for row in reader
            if len(row) >= 19 and row[18].strip() == "3" and row[6].strip() and row[14].strip()
        ]


def download_sky_patches(
    csv_file_path: str | Path = DEFAULT_MANIFEST_PATH,
    download_directory: str | Path = DEFAULT_DOWNLOAD_DIRECTORY,
    downloader: Callable[[str, str], object] = urllib.request.urlretrieve,
    logger: Callable[[str], None] = print,
) -> dict:
    """Download all Level-3 sky patch files listed in a manifest CSV."""
    download_path = Path(download_directory)
    download_path.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest_rows(csv_file_path)
    total_files = len(manifest_rows)
    downloaded = 0
    skipped = 0
    failed = 0

    for index, (data_uri, file_name) in enumerate(manifest_rows, 1):
        download_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={data_uri}"
        local_file_path = download_path / file_name

        if local_file_path.exists():
            skipped += 1
            logger(f"[{index}/{total_files}] Skipping {file_name} (already downloaded)")
            continue

        logger(f"[{index}/{total_files}] Downloading {file_name}...")
        try:
            downloader(download_url, str(local_file_path))
            downloaded += 1
            logger(f"Successfully downloaded to {local_file_path}")
        except Exception as exc:
            failed += 1
            logger(f"Failed to download {file_name}: {exc}")

    logger(
        f"Complete. Downloaded {downloaded}, skipped {skipped}, failed {failed}, total manifest rows {total_files}."
    )
    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "total": total_files,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Download JWST sky patch FITS files from a manifest CSV.")
    parser.add_argument(
        "csv_file_path",
        nargs="?",
        default=DEFAULT_MANIFEST_PATH,
        help=f"Manifest CSV path (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--download-directory",
        default=DEFAULT_DOWNLOAD_DIRECTORY,
        help=f"Directory to write FITS downloads into (default: {DEFAULT_DOWNLOAD_DIRECTORY})",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    download_sky_patches(args.csv_file_path, download_directory=args.download_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())