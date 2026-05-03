"""
Populate a staging Hubble manifest with real parent FITS products.

This pass resolves Hubble rows for the Hubble+SDSS+JWST overlap list and writes
them to a separate staging manifest so it can run concurrently with the JWST
population pass without both processes rewriting the same CSV.
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
from astropy.wcs import FITSFixedWarning, WCS


SCRIPT_DIR = Path(__file__).resolve().parent
OVERLAP_CSV = SCRIPT_DIR / "../data/surveys/overlaps/Hubble+SDSS+JWST.csv"
MANIFEST_PATH = SCRIPT_DIR / "../data/surveys/overlaps/native_source_manifest_hubble.csv"
DOWNLOAD_ROOT = SCRIPT_DIR / "../data/surveys/native_parents/Hubble"
QUERY_RADIUS = 0.01 * u.deg
MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
MANIFEST_BASE_FIELDS = [
    "ra",
    "dec",
    "scope",
    "provider",
    "path",
    "ext",
    "ext_name",
    "product_filename",
    "obs_id",
    "target_name",
    "filter",
]
HUBBLE_PRODUCT_SUFFIX_PRIORITY = (
    "_drc.fits",
    "_drz.fits",
    "_flc.fits",
    "_flt.fits",
)


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
        help="Staging manifest CSV to create or update.",
    )
    parser.add_argument(
        "--download-root",
        default=str(DOWNLOAD_ROOT),
        help="Directory for downloaded native Hubble parent FITS files.",
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
    return str(Path("../data/surveys/native_parents/Hubble") / path.name)


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


def is_existing_hubble_row_usable(row):
    if row.get("scope") != "Hubble":
        return False
    path_value = row.get("path")
    if not path_value:
        return False
    ext_name = str(row.get("ext_name") or "").upper()
    if ext_name != "SCI":
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


def covering_extension(path, coord, allowed_names=None):
    allowed = None if allowed_names is None else {name.upper() for name in allowed_names}
    best_candidate = None

    with fits.open(path, memmap=False) as hdul:
        for ext_index, hdu in enumerate(hdul):
            ext_name = str(getattr(hdu, "name", "") or "").upper()
            if allowed is not None and ext_name not in allowed:
                continue

            data = getattr(hdu, "data", None)
            if data is None:
                continue

            array = np.squeeze(np.asarray(data))
            if array.ndim != 2:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    wcs = WCS(hdu.header, fobj=hdul)
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
                center_x = (width - 1) / 2.0
                center_y = (height - 1) / 2.0
                center_distance = ((x_coord - center_x) / max(width, 1)) ** 2 + ((y_coord - center_y) / max(height, 1)) ** 2
                candidate = (center_distance, ext_index, ext_name)
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate

    if best_candidate is None:
        return False, None, None

    _, ext_index, ext_name = best_candidate
    return True, ext_index, ext_name


def query_hubble_candidates(coord):
    results = Observations.query_region(coord, radius=QUERY_RADIUS)
    candidates = []
    seen = set()

    for row in results:
        if str(row["obs_collection"]).upper() != "HST":
            continue
        if int(row["calib_level"]) != 3:
            continue

        obs_id = str(row["obs_id"])
        target_name = str(row["target_name"])
        filter_name = str(row["filters"]).upper()
        if "COSMOS" not in target_name.upper():
            continue
        if filter_name != "F814W":
            continue
        if "ACS" not in obs_id.upper():
            continue

        if obs_id in seen:
            continue
        seen.add(obs_id)
        candidates.append(row)

    def separation_rank(row):
        try:
            center = SkyCoord(ra=float(row["s_ra"]) * u.deg, dec=float(row["s_dec"]) * u.deg)
        except Exception:
            return float("inf")
        return coord.separation(center).arcsecond

    candidates.sort(key=lambda row: (separation_rank(row), str(row["obs_id"])))
    return candidates


def list_hubble_products(obs_row, product_cache):
    cache_key = str(obs_row["obs_id"])
    if cache_key in product_cache:
        return product_cache[cache_key]

    product_rows = []
    for row in Observations.get_product_list(obs_row):
        filename = str(row["productFilename"])
        if not filename.endswith(".fits"):
            continue
        if str(row["productType"]).upper() != "SCIENCE":
            continue
        calib_level = int(row["calib_level"])
        if calib_level not in {2, 3}:
            continue
        if any(filename.endswith(suffix) for suffix in HUBBLE_PRODUCT_SUFFIX_PRIORITY):
            product_rows.append(
                {
                    "product_filename": filename,
                    "data_uri": str(row["dataURI"]),
                    "calib_level": calib_level,
                }
            )

    product_rows.sort(key=_hubble_product_rank)
    product_cache[cache_key] = product_rows
    return product_rows


def _hubble_product_rank(row):
    filename = row["product_filename"].lower()

    suffix_rank = len(HUBBLE_PRODUCT_SUFFIX_PRIORITY)
    for index, suffix in enumerate(HUBBLE_PRODUCT_SUFFIX_PRIORITY):
        if filename.endswith(suffix):
            suffix_rank = index
            break

    calib_rank = 0 if int(row.get("calib_level", 99)) == 3 else 1
    filter_rank = 0 if "_f814w_" in filename else 1
    mosaic_rank = 1 if ("skycell" in filename or "_all_" in filename or "_total_" in filename) else 0
    return (suffix_rank, calib_rank, filter_rank, mosaic_rank, row["product_filename"])


def resolve_hubble_row(ra, dec, download_root, product_cache):
    coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)
    candidates = query_hubble_candidates(coord)
    if not candidates:
        return None

    for obs_row in candidates:
        for product in list_hubble_products(obs_row, product_cache):
            local_path = Path(download_root) / product["product_filename"]
            mast_download_file(product["data_uri"], local_path)
            covers_coord, ext_index, ext_name = covering_extension(local_path, coord, allowed_names={"SCI"})
            if not covers_coord:
                continue

            manifest_row = {
                "ra": format_coord(ra),
                "dec": format_coord(dec),
                "scope": "Hubble",
                "provider": "local_fits_cutout",
                "path": manifest_relative_path(local_path),
                "product_filename": product["product_filename"],
                "obs_id": str(obs_row["obs_id"]),
                "target_name": str(obs_row["target_name"]),
                "filter": str(obs_row["filters"]).upper(),
            }
            if ext_index is not None:
                manifest_row["ext"] = str(ext_index)
            if ext_name:
                manifest_row["ext_name"] = ext_name
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
        key = coord_key(row["RA"], row["DEC"], "Hubble")
        if key in existing_rows and is_existing_hubble_row_usable(existing_rows[key]):
            skipped_count += 1
            print(
                f"[{index + 1}/{len(overlaps)}] Reusing existing Hubble row for "
                f"{format_coord(row['RA'])}, {format_coord(row['DEC'])}"
            )
            continue

        print(
            f"[{index + 1}/{len(overlaps)}] Resolving Hubble source for "
            f"{format_coord(row['RA'])}, {format_coord(row['DEC'])}"
        )
        manifest_row = resolve_hubble_row(row["RA"], row["DEC"], args.download_root, product_cache)
        if manifest_row is None:
            unresolved.append((format_coord(row["RA"]), format_coord(row["DEC"])))
            print("  No covering Hubble parent product found")
            continue

        existing_rows[key] = manifest_row
        resolved_count += 1
        write_manifest(existing_rows, args.manifest)
        print(
            f"  Selected {manifest_row['product_filename']} "
            f"({manifest_row['filter']} / {manifest_row['target_name']})"
        )

    write_manifest(existing_rows, args.manifest)
    print(f"Wrote manifest: {args.manifest}")
    print(f"New Hubble rows resolved: {resolved_count}")
    print(f"Existing Hubble rows reused: {skipped_count}")
    if unresolved:
        print("Unresolved coordinates:")
        for ra_value, dec_value in unresolved:
            print(f"  {ra_value}, {dec_value}")


if __name__ == "__main__":
    main()