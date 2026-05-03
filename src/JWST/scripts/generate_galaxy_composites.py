"""
Generate 3-panel composite PNGs for each of the 44 matched galaxies,
showing SDSS, HST (ACS), and JWST (NIRCam) views side by side.

Uses the CDS hips2fits cutout service for all three telescopes:
  - SDSS:  CDS/P/SDSS9/color
  - HST:   CDS/P/HLA/I  (Hubble Legacy Archive, F814W, 32-bit FITS)
  - JWST:  ESAVO/P/JWST/NIRCam_Imaging

Input:  outputs/common_galaxies.csv
Output: outputs/galaxy_composites/<ra>_<dec>.png
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from astropy.io import fits as pyfits
from astropy.visualization import AsinhStretch, PercentileInterval, make_lupton_rgb

INPUT_CSV = "../outputs/common_galaxies.csv"
OUTPUT_DIR = "../outputs/galaxy_composites"

HIPS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"

SURVEYS = {
    "SDSS": "CDS/P/SDSS9/color",
    "HST (HLA)": "CDS/P/HLA/I",
    "JWST (NIRCam)": "ESAVO/P/JWST/NIRCam_Imaging",
}

# Cutout parameters
FOV_DEG = 0.005       # field of view in degrees (~18 arcsec)
IMG_SIZE = 256        # pixels per panel
TIMEOUT = 30          # HTTP timeout in seconds
RETRY_DELAY = 2       # seconds between retries
MAX_RETRIES = 3

# HST HLA/I is fetched as 32-bit FITS and stretched with asinh to compress the
# bright core while preserving faint outer structure.
HST_ASINH_A = 0.01       # asinh softening (smaller = more core compression)
HST_PCT_INTERVAL = 99.5  # percentile interval for clipping before stretch

# SDSS Lupton RGB parameters
SDSS_LUPTON_Q = 10
SDSS_LUPTON_STRETCH = 0.5
SDSS_RGB_BANDS = ["CDS/P/SDSS9/i", "CDS/P/SDSS9/r", "CDS/P/SDSS9/g"]


def fetch_cutout(hips_id, ra, dec):
    """Fetch a cutout from the hips2fits service.

    For HST (HLA), requests FITS format to preserve the full 32-bit dynamic
    range, then applies an asinh stretch.  For SDSS, builds an RGB composite
    from i/r/g band FITS using Lupton et al. (2004) stretch.
    All other surveys use JPEG.
    """
    if "SDSS" in hips_id and "color" in hips_id:
        channels = []
        for band_id in SDSS_RGB_BANDS:
            params = {
                "hips": band_id,
                "ra": ra,
                "dec": dec,
                "fov": FOV_DEG,
                "width": IMG_SIZE,
                "height": IMG_SIZE,
                "format": "fits",
            }
            data = None
            for attempt in range(MAX_RETRIES):
                try:
                    r = requests.get(HIPS_URL, params=params, timeout=TIMEOUT)
                    if r.status_code == 200:
                        hdu = pyfits.open(BytesIO(r.content))
                        data = hdu[0].data.astype(np.float64)
                        data = np.nan_to_num(data, nan=0.0)
                        data -= np.median(data)
                        data = np.clip(data, 0, None)
                        break
                except requests.RequestException:
                    pass
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            if data is None:
                return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (30, 30, 30))
            channels.append(data)
        rgb = make_lupton_rgb(*channels, Q=SDSS_LUPTON_Q,
                              stretch=SDSS_LUPTON_STRETCH)
        return Image.fromarray(rgb, "RGB")

    is_hst = "HLA" in hips_id
    is_jwst = "JWST" in hips_id
    params = {
        "hips": hips_id,
        "ra": ra,
        "dec": dec,
        "fov": FOV_DEG,
        "width": IMG_SIZE,
        "height": IMG_SIZE,
        "format": "fits" if is_hst else "jpg",
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(HIPS_URL, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                if is_hst:
                    hdu = pyfits.open(BytesIO(r.content))
                    data = hdu[0].data.astype(np.float64)
                    data = np.nan_to_num(data, nan=0.0)
                    interval = PercentileInterval(HST_PCT_INTERVAL)
                    vmin, vmax = interval.get_limits(data)
                    normed = np.clip((data - vmin) / (vmax - vmin), 0, 1)
                    stretched = AsinhStretch(a=HST_ASINH_A)(normed)
                    gray = (stretched * 255).astype(np.uint8)
                    img = Image.fromarray(gray).convert("RGB")
                else:
                    img = Image.open(BytesIO(r.content)).convert("RGB")
                    if is_jwst:
                        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                return img
        except requests.RequestException:
            pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)

    # Return a blank placeholder on failure
    return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (30, 30, 30))


def make_composite(images, labels, ra, dec):
    """Combine cutout images into a single labeled composite."""
    n = len(images)
    label_h = 28
    padding = 4
    total_w = n * IMG_SIZE + (n - 1) * padding
    total_h = IMG_SIZE + label_h

    composite = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        x_offset = i * (IMG_SIZE + padding)
        composite.paste(img, (x_offset, label_h))

        # Draw label centered above each panel
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x_offset + (IMG_SIZE - text_w) // 2
        draw.text((text_x, 4), label, fill=(255, 255, 255), font=font)

    # Add coordinate info at the bottom-right
    coord_text = f"RA={ra:.5f}  DEC={dec:.5f}"
    bbox = draw.textbbox((0, 0), coord_text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text((total_w - text_w - 6, label_h + IMG_SIZE - 18), coord_text,
              fill=(200, 200, 200), font=font)

    return composite


def main():
    galaxies = pd.read_csv(INPUT_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Generating composites for {len(galaxies)} galaxies...")
    labels = list(SURVEYS.keys())
    hips_ids = list(SURVEYS.values())

    for idx, row in galaxies.iterrows():
        ra, dec = row["ra"], row["dec"]
        filename = f"{ra:.6f}_{dec:.6f}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        images = [fetch_cutout(hips_id, ra, dec) for hips_id in hips_ids]
        composite = make_composite(images, labels, ra, dec)
        composite.save(filepath)

        print(f"  [{idx + 1}/{len(galaxies)}] {filename}")

    print(f"Done. Saved {len(galaxies)} composites to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
