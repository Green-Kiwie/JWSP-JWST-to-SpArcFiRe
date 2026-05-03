"""
Find overlapping galaxies between Hubble, SDSS, and JWST catalogs.

Step 1 — Create master catalogs:
    data/surveys/Hubble.csv  (RA, DEC, survey)  from GZH.fits
    data/surveys/JWST.csv    (RA, DEC, survey)  from JWST_COSMOS-WEB.csv

Step 2 — Cross-match catalogs (default: 1 arcsecond tolerance):
    data/surveys/overlaps/Hubble+SDSS.csv
    data/surveys/overlaps/Hubble+JWST.csv
    data/surveys/overlaps/SDSS+JWST.csv
    data/surveys/overlaps/Hubble+SDSS+JWST.csv

All overlap CSVs contain two columns: RA, DEC (decimal degrees).
Pass a numeric threshold in arcseconds on the command line to override the
default match tolerance.
"""

import os
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

DEFAULT_TOLERANCE_ARCSEC = 1.0

# Input files
GZH_FITS = "../archive/GZH.fits"
JWST_CSV = "../archive/JWST_COSMOS-WEB.csv"
SDSS_CSV = "../data/surveys/SDSS.csv"

# Output master catalogs
HUBBLE_OUT = "../data/surveys/Hubble.csv"
JWST_OUT = "../data/surveys/JWST.csv"

# Output overlap directory
OVERLAP_DIR = "../data/surveys/overlaps"


def load_hubble():
    """Load GZH.fits → DataFrame with RA, DEC, survey columns."""
    print("Loading GZH.fits...")
    with fits.open(GZH_FITS) as hdu:
        data = hdu[1].data
        df = pd.DataFrame({
            "RA": data["RA"],
            "DEC": data["DEC"],
            "survey": ["GZH-" + img.strip() for img in data["imaging"]],
        })
    print(f"  {len(df):,} Hubble galaxies")
    return df


def load_jwst():
    """Load JWST_COSMOS-WEB.csv → DataFrame with RA, DEC, survey columns."""
    print("Loading JWST_COSMOS-WEB.csv...")
    raw = pd.read_csv(JWST_CSV)
    df = pd.DataFrame({
        "RA": raw["ra"],
        "DEC": raw["dec"],
        "survey": "COSMOS-WEB",
    })
    print(f"  {len(df):,} JWST sources")
    return df


def load_sdss():
    """Load SDSS.csv, convert sexagesimal RA/DEC to decimal degrees."""
    print("Loading SDSS.csv...")
    raw = pd.read_csv(SDSS_CSV)

    # Vectorised sexagesimal → decimal
    ra_parts = raw["RA"].str.split(":", expand=True).astype(float)
    ra_deg = (ra_parts[0] + ra_parts[1] / 60.0 + ra_parts[2] / 3600.0) * 15.0

    dec_sign = raw["DEC"].str.startswith("-").map({True: -1.0, False: 1.0})
    dec_parts = raw["DEC"].str.lstrip("+-").str.split(":", expand=True).astype(float)
    dec_deg = dec_sign * (dec_parts[0] + dec_parts[1] / 60.0 + dec_parts[2] / 3600.0)

    df = pd.DataFrame({"RA": ra_deg, "DEC": dec_deg})
    print(f"  {len(df):,} SDSS galaxies")
    return df


def cross_match(name1, coords1, name2, coords2, tolerance_arcsec):
    """Return (idx_into_1, idx_into_2) for all nearest matches within tolerance."""
    print(f"Cross-matching {name1} ({len(coords1):,}) × {name2} ({len(coords2):,})...")
    idx, sep, _ = coords1.match_to_catalog_sky(coords2)
    mask = sep <= tolerance_arcsec * u.arcsec
    idx1 = np.where(mask)[0]
    idx2 = idx[mask]
    print(f"  → {len(idx1):,} matches within {tolerance_arcsec} arcsec")
    return idx1, idx2


def save_overlap(ra, dec, name):
    """Save an overlap CSV with RA, DEC columns."""
    path = os.path.join(OVERLAP_DIR, f"{name}.csv")
    pd.DataFrame({"RA": ra, "DEC": dec}).to_csv(path, index=False)
    print(f"  Saved {path}")


def parse_tolerance_arcsec(argv):
    if len(argv) <= 1:
        return DEFAULT_TOLERANCE_ARCSEC

    try:
        tolerance_arcsec = float(argv[1])
    except ValueError as exc:
        raise SystemExit("Threshold must be a numeric value in arcseconds.") from exc

    if tolerance_arcsec <= 0:
        raise SystemExit("Threshold must be greater than zero.")

    return tolerance_arcsec


def main():
    tolerance_arcsec = parse_tolerance_arcsec(sys.argv)

    # ── Step 1: Load catalogs ──────────────────────────────────────────
    hubble = load_hubble()
    jwst = load_jwst()
    sdss = load_sdss()

    # Save master catalogs
    hubble.to_csv(HUBBLE_OUT, index=False)
    print(f"Saved {HUBBLE_OUT}")
    jwst.to_csv(JWST_OUT, index=False)
    print(f"Saved {JWST_OUT}")

    # ── Step 2: Build SkyCoord objects ─────────────────────────────────
    print("\nBuilding SkyCoord objects...")
    print(f"Using match tolerance: {tolerance_arcsec} arcsec")
    hubble_sc = SkyCoord(ra=hubble["RA"].values, dec=hubble["DEC"].values, unit="deg")
    sdss_sc = SkyCoord(ra=sdss["RA"].values, dec=sdss["DEC"].values, unit="deg")
    jwst_sc = SkyCoord(ra=jwst["RA"].values, dec=jwst["DEC"].values, unit="deg")

    # ── Step 3: Pairwise cross-matches ─────────────────────────────────
    os.makedirs(OVERLAP_DIR, exist_ok=True)

    # Hubble ∩ SDSS
    hs_i1, _ = cross_match("Hubble", hubble_sc, "SDSS", sdss_sc, tolerance_arcsec)
    save_overlap(hubble["RA"].values[hs_i1], hubble["DEC"].values[hs_i1], "Hubble+SDSS")

    # Hubble ∩ JWST
    hj_i1, _ = cross_match("Hubble", hubble_sc, "JWST", jwst_sc, tolerance_arcsec)
    save_overlap(hubble["RA"].values[hj_i1], hubble["DEC"].values[hj_i1], "Hubble+JWST")

    # SDSS ∩ JWST
    sj_i1, _ = cross_match("SDSS", sdss_sc, "JWST", jwst_sc, tolerance_arcsec)
    save_overlap(sdss["RA"].values[sj_i1], sdss["DEC"].values[sj_i1], "SDSS+JWST")

    # Triple: Hubble galaxies matched in BOTH SDSS and JWST
    triple_idx = sorted(set(hs_i1) & set(hj_i1))
    save_overlap(hubble["RA"].values[triple_idx], hubble["DEC"].values[triple_idx],
                 "Hubble+SDSS+JWST")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 40}")
    print(f"Hubble+SDSS:      {len(hs_i1):>7,}")
    print(f"Hubble+JWST:      {len(hj_i1):>7,}")
    print(f"SDSS+JWST:        {len(sj_i1):>7,}")
    print(f"Hubble+SDSS+JWST: {len(triple_idx):>7,}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
