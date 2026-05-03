#!/usr/bin/env python3
"""Build a CSV manifest of JWST Level-3 sky mosaic FITS products from MAST.

The MAST missions API query mirrors the broad JWST mission search exposed in the
portal, then narrows the results to the sky mosaic image products this project
actually downloads:

- calibration level 3
- image data products
- FITS products whose filenames end in ``_i2d.fits``

The full live query can return hundreds of thousands of rows, so the script
pages through the mission results in batches and writes a normalized CSV that
the downloader can consume reliably.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
from astroquery.mast.missions import MastMissions


DEFAULT_OUTPUT_FILE = Path(__file__).resolve().parents[1] / "inputs" / "jwst_L3_sky_patches.csv"
DEFAULT_BATCH_SIZE = 500
DEFAULT_SLEEP_SECONDS = 5.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_BATCHES = None
JWST_MISSION = "jwst"
JWST_INSTRUMENTS = "MIRI,NIRCAM,NIRSPEC,NIRISS"
JWST_EXP_TYPES = (
    "MIR_LRS-FIXEDSLIT,MIR_LRS-SLITLESS,MIR_MRS,NRC_GRISM,NRC_WFSS,"
    "NIS_SOSS,NIS_WFSS,NRS_FIXEDSLIT,NRS_MSASPEC,NRC_TSGRISM,NRC_TSIMAGE,"
    "NRS_BRIGHTOBJ,MIR_4QPM,MIR_CORONCAL,MIR_IMAGE,MIR_LYOT,NRC_CORON,"
    "NRC_IMAGE,NIS_IMAGE,NRS_IFU,NRS_IMAGE,NRS_MIMF,NRS_MSATA,"
    "NRS_TACONFIRM,NRS_WATA"
)
OUTPUT_COLUMNS = [
    "dataURI",
    "productFilename",
    "obsID",
    "target_name",
    "obs_collection",
    "calib_level",
    "productType",
    "dataproduct_type",
    "instrume",
    "exp_type",
    "filters",
    "proposal_id",
    "size",
    "dataRights",
]
COLUMN_ALIASES = {
    "dataURI": ("dataURI", "data_uri", "uri"),
    "productFilename": ("productFilename", "filename"),
    "obsID": ("obsID", "obs_id", "observation_id", "dataset", "fileSetName"),
    "target_name": ("target_name", "target", "targname", "sci_targname", "target_name_obs"),
    "obs_collection": ("obs_collection",),
    "calib_level": ("calib_level", "productLevel", "calib_level_obs"),
    "productType": ("productType", "product_type", "category", "type"),
    "dataproduct_type": ("dataproduct_type",),
    "instrume": ("instrume", "instrument_name", "instrument", "instrume_obs"),
    "exp_type": ("exp_type", "exp_type_obs"),
    "filters": ("filters", "filter", "filters_obs", "opticalElements"),
    "proposal_id": ("proposal_id", "proposal", "proposal_id_obs"),
    "size": ("size",),
    "dataRights": ("dataRights", "data_rights", "access", "dataRights_obs"),
}


def build_query_kwargs() -> dict:
    """Return the portal-aligned query parameters for the JWST missions API."""
    return {
        "select_cols": [],
        "exp_type": JWST_EXP_TYPES,
        "instrume": JWST_INSTRUMENTS,
        "productLevel": "3",
    }


def _to_dataframe(result) -> pd.DataFrame:
    """Convert an astroquery result table or DataFrame into a pandas DataFrame."""
    if result is None:
        return pd.DataFrame()
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    return pd.DataFrame(result)


def _coalesce_column(frame: pd.DataFrame, aliases: Iterable[str]) -> pd.Series:
    """Return the first matching column from *frame* or a blank string series."""
    combined = None
    for alias in aliases:
        if alias in frame.columns:
            series = frame[alias]
            combined = series if combined is None else combined.fillna(series)
    if combined is not None:
        return combined
    return pd.Series([""] * len(frame), index=frame.index, dtype="object")


def _normalize_calib_level(value) -> str:
    """Return a stable string form for calibration levels across API schemas."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    try:
        numeric_value = float(text)
    except ValueError:
        return text

    if numeric_value.is_integer():
        return str(int(numeric_value))
    return text


def enrich_product_results(product_frame: pd.DataFrame, observation_frame: pd.DataFrame) -> pd.DataFrame:
    """Propagate observation metadata onto product rows when the product list lacks it."""
    if product_frame.empty or observation_frame is None or observation_frame.empty:
        return product_frame

    if "dataset" not in product_frame.columns or "fileSetName" not in observation_frame.columns:
        return product_frame

    metadata_columns = [
        "fileSetName",
        "targprop",
        "instrume",
        "exp_type",
        "program",
        "access",
        "productLevel",
        "opticalElements",
    ]
    available_columns = [column for column in metadata_columns if column in observation_frame.columns]
    if not available_columns:
        return product_frame

    metadata = observation_frame[available_columns].copy()
    metadata = metadata.rename(
        columns={
            "fileSetName": "dataset",
            "targprop": "target_name_obs",
            "instrume": "instrume_obs",
            "exp_type": "exp_type_obs",
            "program": "proposal_id_obs",
            "access": "dataRights_obs",
            "productLevel": "calib_level_obs",
            "opticalElements": "filters_obs",
        }
    )
    metadata = metadata.drop_duplicates(subset=["dataset"])
    return product_frame.merge(metadata, on="dataset", how="left")


def normalize_query_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw missions query batch into the downloader manifest schema."""
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    normalized = pd.DataFrame(index=frame.index)
    for column_name, aliases in COLUMN_ALIASES.items():
        normalized[column_name] = _coalesce_column(frame, aliases)

    normalized["obs_collection"] = normalized["obs_collection"].fillna("").replace("", "JWST")

    filename_series = normalized["productFilename"].astype("string").fillna("").str.strip()
    calib_series = normalized["calib_level"].map(_normalize_calib_level)
    if calib_series.eq("").all():
        calib_series = pd.Series(["3"] * len(normalized), index=normalized.index, dtype="object")
    normalized["calib_level"] = calib_series
    dataproduct_series = (
        normalized["dataproduct_type"].astype("string").fillna("").str.strip().str.lower()
    )

    normalized["dataURI"] = normalized["dataURI"].astype("string").fillna("").str.strip()
    missing_uri_mask = normalized["dataURI"] == ""
    normalized.loc[missing_uri_mask, "dataURI"] = "mast:JWST/product/" + filename_series[missing_uri_mask]

    keep_mask = filename_series.str.endswith("_i2d.fits")
    if calib_series.ne("").any():
        keep_mask &= calib_series.eq("3")
    if dataproduct_series.ne("").any():
        keep_mask &= dataproduct_series.eq("image")

    normalized = normalized.loc[keep_mask, OUTPUT_COLUMNS]
    normalized = normalized.drop_duplicates(subset=["productFilename"])
    normalized = normalized.sort_values("productFilename").reset_index(drop=True)
    return normalized


def export_sky_patch_manifest(
    query_client: Any,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_batches: int | None = DEFAULT_MAX_BATCHES,
    logger: Callable[[str], None] = print,
) -> int:
    """Query MAST in pages and write the normalized sky patch manifest CSV."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1 when provided")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    query_kwargs = build_query_kwargs()
    offset = 0
    batch_number = 0
    total_observation_rows = 0
    total_product_rows = 0
    total_written_rows = 0
    wrote_any_rows = False

    logger(
        "Querying MAST JWST missions API for Level-3 products "
        f"({query_kwargs['instrume']}; productLevel={query_kwargs['productLevel']})"
    )

    while True:
        observation_frame = None
        raw_observation_batch = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_observation_batch = query_client.query_criteria(
                    limit=batch_size,
                    offset=offset,
                    **query_kwargs,
                )
                observation_frame = _to_dataframe(raw_observation_batch)
                break
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                logger(
                    f"Query failed at offset {offset} (attempt {attempt}/{max_retries}): {exc}. "
                    f"Retrying in {sleep_seconds:.1f}s..."
                )
                time.sleep(sleep_seconds)

        if observation_frame is None or observation_frame.empty:
            break

        batch_number += 1
        observation_row_count = len(observation_frame)
        total_observation_rows += observation_row_count

        product_frame = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_product_batch = query_client.get_product_list(raw_observation_batch)
                product_frame = _to_dataframe(raw_product_batch)
                break
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                logger(
                    f"Product list query failed at offset {offset} (attempt {attempt}/{max_retries}): {exc}. "
                    f"Retrying in {sleep_seconds:.1f}s..."
                )
                time.sleep(sleep_seconds)

        if product_frame is None:
            product_frame = pd.DataFrame()

        product_row_count = len(product_frame)
        total_product_rows += product_row_count

        enriched_product_frame = enrich_product_results(product_frame, observation_frame)
        normalized_batch = normalize_query_results(enriched_product_frame)
        kept_row_count = len(normalized_batch)

        if kept_row_count:
            normalized_batch.to_csv(
                output_path,
                mode="a" if wrote_any_rows else "w",
                header=not wrote_any_rows,
                index=False,
            )
            wrote_any_rows = True
            total_written_rows += kept_row_count

        logger(
            f"Processed batch {batch_number}: {observation_row_count} observation rows, "
            f"{product_row_count} product rows, "
            f"{kept_row_count} sky mosaics kept (offset {offset})."
        )
        if batch_number == 1 and product_row_count > 0 and kept_row_count == 0:
            diagnostic = {
                "columns": list(enriched_product_frame.columns),
                "sample_product_level": enriched_product_frame.get("productLevel", pd.Series(dtype="object")).head(3).tolist(),
                "sample_calib_level": enriched_product_frame.get("calib_level", pd.Series(dtype="object")).head(3).tolist(),
                "sample_filename": enriched_product_frame.get("filename", pd.Series(dtype="object")).head(3).tolist(),
                "sample_product_filename": enriched_product_frame.get("productFilename", pd.Series(dtype="object")).head(3).tolist(),
                "sample_dataproduct_type": enriched_product_frame.get("dataproduct_type", pd.Series(dtype="object")).head(3).tolist(),
            }
            logger(
                "First batch kept zero rows; sample fields were "
                f"{diagnostic}"
            )
        offset += observation_row_count

        if max_batches is not None and batch_number >= max_batches:
            logger(f"Reached max batch limit ({max_batches}); stopping early.")
            break

    if not wrote_any_rows:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)

    logger(
        f"Complete. Wrote {total_written_rows} sky mosaic rows from {total_product_rows} product rows "
        f"across {total_observation_rows} observation rows "
        f"to {output_path}"
    )
    return total_written_rows


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build a CSV manifest of JWST Level-3 sky mosaic FITS products from MAST."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows to request per missions API page (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help=f"Delay before retrying a failed page request (default: {DEFAULT_SLEEP_SECONDS})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retries per failed page request (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=DEFAULT_MAX_BATCHES,
        help="Stop after this many observation batches. Useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    missions = MastMissions(mission=JWST_MISSION)
    export_sky_patch_manifest(
        missions,
        output_file=args.output,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        max_batches=args.max_batches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())