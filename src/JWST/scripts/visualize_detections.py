#!/usr/bin/env python3
"""Visualize galaxy detections on a FITS sky patch.

Runs SEP detection + the updated filter pipeline, then draws circles on the
image around every object that would survive through to the crop stage.
Saves the result as a PNG.
"""

import sys, os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.visualization import ZScaleInterval

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'modules'))
import sep_helpers as sh

FITS_PATH = (
    "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/outputs/"
    "JWST_L3_sky_patches/jw01571-c1011_t000_niriss_clear-f200w_i2d.fits"
)
OUTPUT_PNG = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                          'jw01571-c1011_niriss_f200w_detections.png')


def main():
    # Load FITS
    image_data = sh.get_main_fits_data(FITS_PATH)
    meta = sh.get_relevant_fits_meta_data(FITS_PATH)

    # Derive pixel scale
    pixel_scale = None
    pixar_a2 = meta.get('PIXAR_A2')
    if pixar_a2 is not None:
        val = pixar_a2[0] if isinstance(pixar_a2, (tuple, list)) else pixar_a2
        if val and float(val) > 0:
            pixel_scale = math.sqrt(float(val))

    instrument = 'niriss'
    waveband = 'clear-f200w'

    # SEP detection
    bkg = sh.get_image_background(image_data)
    bkgless = sh.subtract_bkg(image_data)
    objects = sh.extract_objects(bkgless, bkg, detection_threshold=5.0,
                                pixel_stack_limit=5_000_000,
                                sub_object_limit=50_000)
    print(f"Raw detections: {len(objects)}")

    # Filter
    import sep
    bkg_rms = sep.Background(image_data).globalrms
    galaxies = sh.filter_galaxies(
        bkgless, objects, bkg_rms,
        verbosity=1,
        instrument=instrument, waveband=waveband,
        pixel_scale_arcsec=pixel_scale,
    )
    print(f"After filter: {len(galaxies)}")

    # Further filter by _extract_object (padding=3.5, min_size=20, max_size=200)
    # to show only those that actually produce a crop
    with fits.open(FITS_PATH) as hdul:
        try:
            raw = hdul['SCI'].data
        except (KeyError, IndexError):
            raw = hdul[0].data
        nan_mask = np.isnan(raw) if raw is not None else None

    cropable = []
    for obj in galaxies:
        crop = sh._extract_object(obj, bkgless, padding=3.5, min_size=20,
                                  max_size=200, verbosity=0,
                                  nan_mask=nan_mask, max_nan_fraction=0.02)
        if crop is not None:
            # Also apply post-crop quality checks
            if sh.thumbnail_has_structure(crop) and sh.thumbnail_has_galaxy_profile(crop):
                cropable.append(obj)
    print(f"Cropable galaxies (would be saved): {len(cropable)}")

    # Plot
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(bkgless)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(bkgless, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)

    for obj in cropable:
        half = max(3.5 * max(obj['a'], obj['b']), 10)
        e = Ellipse(
            xy=(obj['x'], obj['y']),
            width=2 * half,
            height=2 * half,
            angle=0,
            linewidth=1.2,
            edgecolor='lime',
            facecolor='none',
        )
        ax.add_patch(e)

    ax.set_title(
        f"jw01571-c1011  NIRISS clear-f200w  —  {len(cropable)} galaxies detected",
        fontsize=13, color='white',
    )
    ax.tick_params(colors='white', labelsize=8)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == '__main__':
    main()
