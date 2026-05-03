"""
Generate galaxy composite PNGs and download raw FITS cutouts for every
overlap pair produced by find_overlapping_galaxies.py.

Outputs
-------
outputs/galaxy_overlap/galaxy_composites/<pair>/<ra>_<dec>.png
outputs/galaxy_overlap/fits/<pair>/<telescope>/<ra>_<dec>[_band].fits

Features
--------
* Resume-safe: skips files that already exist on disk.
* Parallel downloads via ThreadPoolExecutor.
* HST panels use the same asinh stretch as generate_galaxy_composites.py.
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from astropy.io import fits as pyfits
from astropy.visualization import AsinhStretch, PercentileInterval, make_lupton_rgb
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── HiPS cutout parameters ─────────────────────────────────────────────
HIPS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
FOV_DEG = 0.005        # ~18 arcsec
IMG_SIZE = 256         # pixels per panel
TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2

# HST stretch parameters
HST_ASINH_A = 0.01
HST_PCT_INTERVAL = 99.5

# Quality-check threshold: minimum 99.5th-0.5th percentile range in raw
# HST FITS data for a cutout to be considered "real signal" (not noise).
HST_MIN_RANGE = 0.15

# SDSS Lupton RGB parameters
SDSS_LUPTON_Q = 10
SDSS_LUPTON_STRETCH = 0.5
SDSS_RGB_BANDS = ["CDS/P/SDSS9/i", "CDS/P/SDSS9/r", "CDS/P/SDSS9/g"]

# Parallelism
WORKERS = 4

# ── Output root directories ────────────────────────────────────────────
BASE_COMPOSITE = "../outputs/galaxy_overlap/galaxy_composites"
BASE_FITS = "../outputs/galaxy_overlap/fits"

# ── Overlap configurations ──────────────────────────────────────────────
# Each entry defines the composite panel order and the FITS downloads.
# Composite panels are shown left-to-right in the listed order.
# FITS entries: list of (hips_id, suffix) — suffix is appended to filename.
SDSS_FITS_BANDS = [
    ("CDS/P/SDSS9/u", "_u"),
    ("CDS/P/SDSS9/g", "_g"),
    ("CDS/P/SDSS9/r", "_r"),
    ("CDS/P/SDSS9/i", "_i"),
    ("CDS/P/SDSS9/z", "_z"),
]

OVERLAPS = {
    "Hubble+SDSS+JWST": {
        "csv": "../data/surveys/overlaps/Hubble+SDSS+JWST.csv",
        "composite_panels": [
            ("SDSS", "CDS/P/SDSS9/color"),
            ("Hubble", "CDS/P/HLA/I"),
            ("JWST", "ESAVO/P/JWST/NIRCam_Imaging"),
        ],
        "fits": {
            "Hubble": [("CDS/P/HLA/I", "")],
            "SDSS": SDSS_FITS_BANDS,
            "JWST": [("ESAVO/P/JWST/NIRCam_Imaging", "")],
        },
    },
    "SDSS+JWST": {
        "csv": "../data/surveys/overlaps/SDSS+JWST.csv",
        "composite_panels": [
            ("SDSS", "CDS/P/SDSS9/color"),
            ("JWST", "ESAVO/P/JWST/NIRCam_Imaging"),
        ],
        "fits": {
            "SDSS": SDSS_FITS_BANDS,
            "JWST": [("ESAVO/P/JWST/NIRCam_Imaging", "")],
        },
    },
    "Hubble+SDSS": {
        "csv": "../data/surveys/overlaps/Hubble+SDSS.csv",
        "composite_panels": [
            ("Hubble", "CDS/P/HLA/I"),
            ("SDSS", "CDS/P/SDSS9/color"),
        ],
        "fits": {
            "Hubble": [("CDS/P/HLA/I", "")],
            "SDSS": SDSS_FITS_BANDS,
        },
    },
    "Hubble+JWST": {
        "csv": "../data/surveys/overlaps/Hubble+JWST.csv",
        "composite_panels": [
            ("Hubble", "CDS/P/HLA/I"),
            ("JWST", "ESAVO/P/JWST/NIRCam_Imaging"),
        ],
        "fits": {
            "Hubble": [("CDS/P/HLA/I", "")],
            "JWST": [("ESAVO/P/JWST/NIRCam_Imaging", "")],
        },
    },
}


# ── Cutout fetching ─────────────────────────────────────────────────────

def _hips_request(hips_id, ra, dec, fmt="jpg"):
    """Raw HTTP request to the hips2fits service. Returns bytes or None."""
    params = {
        "hips": hips_id,
        "ra": ra,
        "dec": dec,
        "fov": FOV_DEG,
        "width": IMG_SIZE,
        "height": IMG_SIZE,
        "format": fmt,
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(HIPS_URL, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    return None


def _fetch_sdss_rgb(ra, dec):
    """Build an SDSS RGB composite from i/r/g band FITS cutouts.

    Uses Lupton et al. (2004) colour-preserving stretch via
    astropy.visualization.make_lupton_rgb.  Maps i→R, r→G, g→B.
    Sky background is subtracted per-band using the median pixel value.

    Returns (PIL RGB Image, True) on success, (blank, False) on failure.
    """
    channels = []
    for band_id in SDSS_RGB_BANDS:
        raw = _hips_request(band_id, ra, dec, fmt="fits")
        if raw is None:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (30, 30, 30)), False
        hdu = pyfits.open(BytesIO(raw))
        data = hdu[0].data.astype(np.float64)
        data = np.nan_to_num(data, nan=0.0)
        data -= np.median(data)
        data = np.clip(data, 0, None)
        channels.append(data)
    rgb = make_lupton_rgb(*channels, Q=SDSS_LUPTON_Q,
                          stretch=SDSS_LUPTON_STRETCH)
    return Image.fromarray(rgb, "RGB"), True


def fetch_panel(hips_id, ra, dec):
    """Download a composite-panel image.

    Returns (PIL RGB Image, quality_ok).
    quality_ok is False when an HST cutout contains only noise.
    """
    if "SDSS" in hips_id and "color" in hips_id:
        return _fetch_sdss_rgb(ra, dec)

    is_hst = "HLA" in hips_id
    is_jwst = "JWST" in hips_id
    fmt = "fits" if is_hst else "jpg"
    raw = _hips_request(hips_id, ra, dec, fmt=fmt)
    if raw is None:
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (30, 30, 30)), False

    if is_hst:
        hdu = pyfits.open(BytesIO(raw))
        data = hdu[0].data.astype(np.float64)
        data = np.nan_to_num(data, nan=0.0)

        # Quality gate: reject cutouts where the dynamic range is too
        # small — these are noise-only tiles with no real galaxy signal.
        p_low, p_high = np.percentile(data, [0.5, 99.5])
        if (p_high - p_low) < HST_MIN_RANGE:
            return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (30, 30, 30)), False

        interval = PercentileInterval(HST_PCT_INTERVAL)
        vmin, vmax = interval.get_limits(data)
        if vmax == vmin:
            vmax = vmin + 1.0
        normed = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        stretched = AsinhStretch(a=HST_ASINH_A)(normed)
        gray = (stretched * 255).astype(np.uint8)
        return Image.fromarray(gray).convert("RGB"), True
    else:
        img = Image.open(BytesIO(raw)).convert("RGB")
        if is_jwst:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return img, True


# ── Composite assembly ──────────────────────────────────────────────────

def _get_font():
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        return ImageFont.load_default()

_FONT = _get_font()


def make_composite(images, labels, ra, dec):
    """Stitch panels into a single labelled composite."""
    n = len(images)
    label_h = 28
    pad = 4
    w = n * IMG_SIZE + (n - 1) * pad
    h = IMG_SIZE + label_h

    comp = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(comp)

    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * (IMG_SIZE + pad)
        comp.paste(img, (x, label_h))
        bbox = draw.textbbox((0, 0), label, font=_FONT)
        tw = bbox[2] - bbox[0]
        draw.text((x + (IMG_SIZE - tw) // 2, 4), label,
                  fill=(255, 255, 255), font=_FONT)

    coord = f"RA={ra:.5f}  DEC={dec:.5f}"
    bbox = draw.textbbox((0, 0), coord, font=_FONT)
    tw = bbox[2] - bbox[0]
    draw.text((w - tw - 6, label_h + IMG_SIZE - 18), coord,
              fill=(200, 200, 200), font=_FONT)
    return comp


# ── Per-galaxy worker functions ─────────────────────────────────────────

def _generate_composite(ra, dec, panels, out_dir):
    """Generate one composite PNG.

    Returns:
      'created'  – new composite saved
      'skipped'  – file already exists (resume)
      'rejected' – cutout failed quality check (noise-only)
    """
    filename = f"{ra:.6f}_{dec:.6f}.png"
    filepath = os.path.join(out_dir, filename)
    if os.path.exists(filepath):
        return "skipped"

    images, labels = [], []
    for label, hips_id in panels:
        img, quality_ok = fetch_panel(hips_id, ra, dec)
        if not quality_ok:
            return "rejected"
        images.append(img)
        labels.append(label)

    comp = make_composite(images, labels, ra, dec)
    comp.save(filepath)
    return "created"


def _download_fits(ra, dec, hips_id, suffix, out_dir):
    """Download one FITS cutout. Returns True if created, False if skipped."""
    filename = f"{ra:.6f}_{dec:.6f}{suffix}.fits"
    filepath = os.path.join(out_dir, filename)
    if os.path.exists(filepath):
        return False

    raw = _hips_request(hips_id, ra, dec, fmt="fits")
    if raw is None:
        return False

    with open(filepath, "wb") as f:
        f.write(raw)
    return True


# ── Process one overlap pair ────────────────────────────────────────────

def process_overlap(name, config):
    """Generate all composites + FITS for one overlap pair."""
    df = pd.read_csv(config["csv"])
    n = len(df)
    if n == 0:
        print(f"  {name}: empty — skipping")
        return

    panels = config["composite_panels"]
    fits_cfg = config["fits"]

    # Count total FITS files per galaxy
    fits_per_galaxy = sum(len(bands) for bands in fits_cfg.values())
    # Estimate: (panels + fits_per_galaxy) requests × avg 1.5s per request
    est_seq_sec = n * (len(panels) + fits_per_galaxy) * 1.5
    est_min = est_seq_sec / WORKERS / 60
    print(f"\n{'─' * 60}")
    print(f"  {name}: {n:,} galaxies")
    print(f"  Composites: {len(panels)} panels each")
    print(f"  FITS: {fits_per_galaxy} files/galaxy ({', '.join(fits_cfg.keys())})")
    print(f"  Estimated time: ~{est_min:.0f} minutes ({WORKERS} workers)")
    print(f"{'─' * 60}")

    # ── Create output directories ──
    comp_dir = os.path.join(BASE_COMPOSITE, name)
    os.makedirs(comp_dir, exist_ok=True)

    fits_dirs = {}
    for scope, bands in fits_cfg.items():
        d = os.path.join(BASE_FITS, name, scope)
        os.makedirs(d, exist_ok=True)
        fits_dirs[scope] = d

    # ── Composites ──
    print(f"  Generating composites...")
    created = 0
    t0 = time.time()

    def _comp_task(row):
        return _generate_composite(row["RA"], row["DEC"], panels, comp_dir)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_comp_task, row): (idx, row)
                   for idx, row in df.iterrows()}
        done = 0
        created = 0
        rejected = 0
        accepted_coords = []   # (RA, DEC) of galaxies that passed quality
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            idx, row = futures[fut]
            if result == "created":
                created += 1
                accepted_coords.append((row["RA"], row["DEC"]))
            elif result == "skipped":
                accepted_coords.append((row["RA"], row["DEC"]))
            elif result == "rejected":
                rejected += 1
            if done % 100 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (n - done) / rate / 60 if rate else 0
                print(f"    composites: {done:,}/{n:,}  "
                      f"({created} new, {rejected} rejected)  "
                      f"ETA {eta:.1f} min", end="\r")

    elapsed = time.time() - t0
    print(f"\n    composites done: {created:,} new, "
          f"{rejected:,} rejected (noise) in {elapsed / 60:.1f} min")

    # ── FITS downloads (only for galaxies that passed quality check) ──
    accepted_set = set(accepted_coords)
    print(f"  Downloading FITS cutouts for {len(accepted_set):,} accepted galaxies...")
    total_fits = len(accepted_set) * fits_per_galaxy
    created = 0
    t0 = time.time()

    def _fits_task(args):
        ra, dec, hips_id, suffix, out_dir = args
        return _download_fits(ra, dec, hips_id, suffix, out_dir)

    tasks = []
    for _, row in df.iterrows():
        ra, dec = row["RA"], row["DEC"]
        if (ra, dec) not in accepted_set:
            continue
        for scope, bands in fits_cfg.items():
            for hips_id, suffix in bands:
                tasks.append((ra, dec, hips_id, suffix, fits_dirs[scope]))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_fits_task, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if fut.result():
                created += 1
            if done % 200 == 0 or done == total_fits:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (total_fits - done) / rate / 60 if rate else 0
                print(f"    FITS: {done:,}/{total_fits:,}  "
                      f"({created} new)  "
                      f"ETA {eta:.1f} min", end="\r")

    elapsed = time.time() - t0
    print(f"\n    FITS done: {created:,} new in {elapsed / 60:.1f} min")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("Galaxy overlap output generator")
    print(f"Workers: {WORKERS}")

    # If a pair name is given on the command line, process only that pair.
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name not in OVERLAPS:
            print(f"Unknown overlap pair: {name}")
            print(f"Available: {', '.join(OVERLAPS.keys())}")
            sys.exit(1)
        process_overlap(name, OVERLAPS[name])
    else:
        for name, config in OVERLAPS.items():
            process_overlap(name, config)

    print("\nAll done.")


if __name__ == "__main__":
    main()
