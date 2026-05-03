"""
Find galaxies present in both GalaxyZoo1_DR_table2.csv and gzh_spiral_labels.csv
by cross-matching on RA/DEC coordinates (1 arcsecond tolerance).

GalaxyZoo1 uses sexagesimal RA (HH:MM:SS) and DEC (±DD:MM:SS).
gzh_spiral_labels uses decimal degrees.

Outputs: outputs/common_galaxies.csv with RA/DEC in decimal degrees.
"""

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

TOLERANCE_ARCSEC = 1.0

GZ1_PATH = "../inputs/GalaxyZoo1_DR_table2.csv"
GZH_PATH = "../inputs/gzh_spiral_labels.csv"
OUT_PATH = "../outputs/common_galaxies.csv"


def parse_gz1_ra(ra_str):
    """Convert HH:MM:SS.ss sexagesimal RA to decimal degrees."""
    parts = ra_str.strip().split(":")
    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    return (h + m / 60.0 + s / 3600.0) * 15.0


def parse_gz1_dec(dec_str):
    """Convert ±DD:MM:SS.s sexagesimal DEC to decimal degrees."""
    dec_str = dec_str.strip()
    sign = -1.0 if dec_str.startswith("-") else 1.0
    parts = dec_str.lstrip("+-").split(":")
    d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    return sign * (d + m / 60.0 + s / 3600.0)


def main():
    print("Loading GalaxyZoo1...")
    gz1 = pd.read_csv(GZ1_PATH)
    gz1["ra_deg"] = gz1["RA"].apply(parse_gz1_ra)
    gz1["dec_deg"] = gz1["DEC"].apply(parse_gz1_dec)

    print("Loading gzh spiral labels...")
    gzh = pd.read_csv(GZH_PATH)

    print(f"GZ1 galaxies: {len(gz1):,}")
    print(f"gzh galaxies: {len(gzh):,}")

    print("Building SkyCoord catalogs...")
    gz1_coords = SkyCoord(ra=gz1["ra_deg"].values * u.deg, dec=gz1["dec_deg"].values * u.deg)
    gzh_coords = SkyCoord(ra=gzh["ra"].values * u.deg, dec=gzh["dec"].values * u.deg)

    print(f"Cross-matching within {TOLERANCE_ARCSEC} arcsecond(s)...")
    # For each gzh galaxy, find the nearest GZ1 galaxy
    idx, sep, _ = gzh_coords.match_to_catalog_sky(gz1_coords)
    match_mask = sep <= TOLERANCE_ARCSEC * u.arcsec

    matched_gzh = gzh[match_mask].copy()
    matched_gz1_idx = idx[match_mask]

    print(f"Matched galaxies: {match_mask.sum():,}")

    output = pd.DataFrame({
        "ra": matched_gzh["ra"].values,
        "dec": matched_gzh["dec"].values,
        "gz1_ra_sexagesimal": gz1["RA"].values[matched_gz1_idx],
        "gz1_dec_sexagesimal": gz1["DEC"].values[matched_gz1_idx],
        "separation_arcsec": sep[match_mask].to(u.arcsec).value,
    })

    output.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
