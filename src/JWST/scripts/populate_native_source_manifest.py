"""
Populate native_source_manifest.csv with real JWST parent FITS products.

This first pass intentionally resolves only JWST rows for the
Hubble+SDSS+JWST overlap list. It queries MAST for COSMOS-WEB NIRCam
observations, downloads candidate `_i2d.fits` products, and records the
first product whose WCS covers each overlap coordinate.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning


SCRIPT_DIR = Path(__file__).resolve().parent
OVERLAP_CSV = SCRIPT_DIR / "../data/surveys/overlaps/Hubble+SDSS+JWST.csv"
MANIFEST_PATH = SCRIPT_DIR / "../data/surveys/overlaps/native_source_manifest.csv"
DOWNLOAD_ROOT = SCRIPT_DIR / "../data/surveys/native_parents/JWST"
QUERY_RADIUS = 0.01 * u.deg
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
JWST_FILTER_PRIORITY = ("F150W", "F115W", "F277W", "F444W")
MANIFEST_BASE_FIELDS = [
    "ra",
    "dec",
    "scope",
    "provider",
    "path",
    "ext",
    "product_filename",
    "obs_id",
    "target_name",
    "filter",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlap-csv",
        default=str(OVERLAP_CSV),
        help="Input overlap CSV with RA and DEC columns.",
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="Manifest CSV to create or update.",
    )
    parser.add_argument(
        "--download-root",
        default=str(DOWNLOAD_ROOT),
        help="Directory for downloaded native JWST parent FITS files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of overlap coordinates to resolve.",
    )
    return parser.parse_args()


def coord_key(ra, dec, scope):
    return (round(float(ra), 6), round(float(dec), 6), scope)


def format_coord(value):
    return f"{float(value):.6f}"


def manifest_relative_path(path):
    return str(Path("../data/surveys/native_parents/JWST") / path.name)


def resolve_local_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def load_existing_rows(manifest_path):
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        return {}

    with manifest_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            if not row.get("ra") or not row.get("dec") or not row.get("scope"):
                continue
            rows[coord_key(row["ra"], row["dec"], row["scope"])] = dict(row)
        return rows


def write_manifest(rows, manifest_path):
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    extras = []
    for row in rows.values():
        for field in row.keys():
            if field not in MANIFEST_BASE_FIELDS and field not in extras:
                extras.append(field)

    fieldnames = MANIFEST_BASE_FIELDS + extras
    ordered_rows = sorted(
        rows.values(),
        key=lambda row: (row.get("scope", ""), float(row.get("ra", 0.0)), float(row.get("dec", 0.0))),
    )

    with manifest_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered_rows:
            output_row = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(output_row)


def is_existing_jwst_row_usable(row):
    if row.get("scope") != "JWST":
        return False
    path_value = row.get("path")
    if not path_value:
        return False
    return resolve_local_path(path_value).exists()


def mast_download_file(data_uri, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    response = requests.get(
        MAST_DOWNLOAD_URL,
        params={"uri": data_uri},
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    temp_path = destination.with_suffix(destination.suffix + f".{os.getpid()}.part")
    with temp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)

    if destination.exists():
        temp_path.unlink(missing_ok=True)
        return destination

    temp_path.replace(destination)
    return destination


def covering_extension(path, coord):
    with fits.open(path, memmap=False) as hdul:
        for ext_index, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            if data is None:
                continue

            array = np.squeeze(np.asarray(data))
            if array.ndim != 2:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    wcs = WCS(hdu.header)
            except Exception:
                continue

            if not wcs.has_celestial:
                continue

            try:
                x_coord, y_coord = wcs.world_to_pixel(coord)
            except Exception:
                continue

            if not np.isfinite(x_coord) or not np.isfinite(y_coord):
                continue

            height, width = array.shape
            if -0.5 <= x_coord < (width - 0.5) and -0.5 <= y_coord < (height - 0.5):
                return True, ext_index

    return False, None


def query_jwst_candidates(coord):
    results = Observations.query_region(coord, radius=QUERY_RADIUS)
    filter_rank = {name: index for index, name in enumerate(JWST_FILTER_PRIORITY)}
    candidates = []
    seen = set()

    for row in results:
        if str(row["obs_collection"]).upper() != "JWST":
            continue
        if int(row["calib_level"]) != 3:
            continue

        obs_id = str(row["obs_id"])
        target_name = str(row["target_name"])
        filter_name = str(row["filters"]).upper()
        if "NIRCAM" not in obs_id.upper():
            continue
        if not target_name.upper().startswith("CWEBTILE-"):
            continue
        if filter_name not in filter_rank:
            continue

        key = (obs_id, target_name, filter_name)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(row)

    def separation_rank(row):
        try:
            center = SkyCoord(ra=float(row["s_ra"]) * u.deg, dec=float(row["s_dec"]) * u.deg)
        except Exception:
            return float("inf")
        return coord.separation(center).arcsecond

    candidates.sort(
        key=lambda row: (
            filter_rank[str(row["filters"]).upper()],
            separation_rank(row),
            str(row["obs_id"]),
        )
    )
    return candidates


def list_i2d_products(obs_row, product_cache):
    cache_key = str(obs_row["obs_id"])
    if cache_key in product_cache:
        return product_cache[cache_key]

    product_rows = []
    for row in Observations.get_product_list(obs_row):
        filename = str(row["productFilename"])
        if not filename.endswith("_i2d.fits"):
            continue
        if str(row["productType"]).upper() != "SCIENCE":
            continue
        product_rows.append(
            {
                "product_filename": filename,
                "data_uri": str(row["dataURI"]),
            }
        )

    preferred_filename = f"{cache_key}_i2d.fits"

    def product_rank(row):
        filename = row["product_filename"].lower()
        return (
            row["product_filename"] != preferred_filename,
            "_nrc" in filename,
            row["product_filename"],
        )

    product_rows.sort(key=product_rank)
    product_cache[cache_key] = product_rows
    return product_rows


def resolve_jwst_row(ra, dec, download_root, product_cache):
    coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)
    candidates = query_jwst_candidates(coord)
    if not candidates:
        return None

    for obs_row in candidates:
        for product in list_i2d_products(obs_row, product_cache):
            local_path = Path(download_root) / product["product_filename"]
            mast_download_file(product["data_uri"], local_path)
            covers_coord, ext_index = covering_extension(local_path, coord)
            if not covers_coord:
                continue

            manifest_row = {
                "ra": format_coord(ra),
                "dec": format_coord(dec),
                "scope": "JWST",
                "provider": "local_fits_cutout",
                "path": manifest_relative_path(local_path),
                "product_filename": product["product_filename"],
                "obs_id": str(obs_row["obs_id"]),
                "target_name": str(obs_row["target_name"]),
                "filter": str(obs_row["filters"]).upper(),
            }
            if ext_index not in (None, 0):
                manifest_row["ext"] = str(ext_index)
            return manifest_row

    return None


def main():
    args = parse_args()
    overlaps = pd.read_csv(args.overlap_csv)
    if args.limit is not None:
        overlaps = overlaps.head(args.limit)

    existing_rows = load_existing_rows(args.manifest)
    product_cache = {}
    resolved_count = 0
    skipped_count = 0
    unresolved = []

    for index, row in overlaps.iterrows():
        key = coord_key(row["RA"], row["DEC"], "JWST")
        if key in existing_rows and is_existing_jwst_row_usable(existing_rows[key]):
            skipped_count += 1
            print(
                f"[{index + 1}/{len(overlaps)}] Reusing existing JWST row for "
                f"{format_coord(row['RA'])}, {format_coord(row['DEC'])}"
            )
            continue

        print(
            f"[{index + 1}/{len(overlaps)}] Resolving JWST source for "
            f"{format_coord(row['RA'])}, {format_coord(row['DEC'])}"
        )
        manifest_row = resolve_jwst_row(row["RA"], row["DEC"], args.download_root, product_cache)
        if manifest_row is None:
            unresolved.append((format_coord(row["RA"]), format_coord(row["DEC"])))
            print("  No covering JWST parent product found")
            continue

        existing_rows[key] = manifest_row
        resolved_count += 1
        print(
            f"  Selected {manifest_row['product_filename']} "
            f"({manifest_row['filter']} / {manifest_row['target_name']})"
        )

    write_manifest(existing_rows, args.manifest)
    print(f"Wrote manifest: {args.manifest}")
    print(f"New JWST rows resolved: {resolved_count}")
    print(f"Existing JWST rows reused: {skipped_count}")
    if unresolved:
        print("Unresolved coordinates:")
        for ra_value, dec_value in unresolved:
            print(f"  {ra_value}, {dec_value}")


if __name__ == "__main__":
    main()