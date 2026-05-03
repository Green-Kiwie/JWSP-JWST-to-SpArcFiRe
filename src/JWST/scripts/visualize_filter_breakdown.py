#!/usr/bin/env python3
"""Per-detection color-coded filter breakdown visualization.

For every raw SEP detection on a FITS sky patch, determines which filter
(if any) first rejects it, draws a colored circle on the image, and writes
a detailed .txt report explaining why each detection was accepted or rejected.

Color legend:
  lime    = PASSED all filters (would be saved)
  red     = Filter 1  – too few pixels (npix)
  orange  = Filter 2  – negative/zero flux
  gold    = Filter 3  – unresolved (semi-major too small)
  magenta = Filter 4  – cosmic ray / hot pixel (peak/flux)
  cyan    = Filter 5  – streak / diffraction spike (elongation)
  blue    = Filter 6  – low SNR
  yellow  = Filter 7  – concentration out of range
  violet  = Filter 8  – star-like (flux-radius CI)
  pink    = Filter 9  – PSF-unresolved
  white   = Filter 10 – peak saturation (flat-top)
  salmon  = Filter 11 – compact Kron radius
  gray    = _extract_object rejection (too small / too large / edge / NaN)
  silver  = post-crop quality (no structure or no galaxy profile)
"""

import sys, os, math
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from astropy.io import fits
from astropy.visualization import ZScaleInterval

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'modules'))
import sep_helpers as sh
import sep

# ── Configuration ────────────────────────────────────────────────────────────
FITS_PATH = (
    "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/outputs/"
    "JWST_L3_sky_patches/jw01571-c1011_t000_niriss_clear-f200w_i2d.fits"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PNG_OUT = os.path.join(OUTPUT_DIR, 'jw01571-c1011_niriss_f200w_filter_breakdown.png')
TXT_OUT = os.path.join(OUTPUT_DIR, 'jw01571-c1011_niriss_f200w_filter_breakdown.txt')

# Filter defaults (matching production code)
MIN_NPIX = 20
MIN_SEMI_MAJOR = 4.0
MAX_ELONGATION = 5.0
MAX_PEAK_FLUX_RATIO = 0.5
MIN_FLUX = 0.0
CONCENTRATION_RANGE = (0.03, 0.75)
MIN_SNR = 7.0
MAX_STAR_CI = 0.45
PSF_RESOLVED_THRESHOLD = 1.5
MIN_KRON_RADIUS = 1.0
PADDING = 3.5
MIN_SIZE = 20
MAX_SIZE = 200

# Color map: filter_name -> (color, short label)
FILTER_COLORS = {
    'pass':           ('lime',    'Passed all filters'),
    'npix':           ('red',     'F1: too few pixels'),
    'flux':           ('orange',  'F2: non-positive flux'),
    'resolved':       ('gold',    'F3: unresolved (a too small)'),
    'artifact':       ('magenta', 'F4: cosmic ray / hot pixel'),
    'elongation':     ('cyan',    'F5: streak / spike'),
    'snr':            ('blue',    'F6: low SNR'),
    'concentration':  ('yellow',  'F7: concentration out of range'),
    'star':           ('#EE82EE', 'F8: star-like (flux-radius CI)'),
    'psf':            ('hotpink', 'F9: PSF-unresolved'),
    'saturation':     ('white',   'F10: saturated (flat-top)'),
    'kron':           ('salmon',  'F11: compact Kron radius'),
    'extract_object': ('gray',    'Crop rejected (size/edge/NaN)'),
    'post_crop':      ('silver',  'Post-crop quality fail'),
}


def compute_all_filter_masks(image_data, objects, bkg_rms,
                             instrument, waveband, pixel_scale):
    """Replicate filter_galaxies logic but return individual masks + diagnostics."""
    n = len(objects)
    a = objects['a']
    b = objects['b']
    flux = objects['flux']
    peak = objects['peak']
    npix = objects['npix']

    rms_val = float(np.median(bkg_rms)) if isinstance(bkg_rms, np.ndarray) else float(bkg_rms)

    # ── Filter 1: npix ──
    mask_npix = npix >= MIN_NPIX
    diag_npix = [f"npix={int(npix[i])}, need>={MIN_NPIX}" for i in range(n)]

    # ── Filter 2: flux ──
    mask_flux = flux > MIN_FLUX
    diag_flux = [f"flux={flux[i]:.2f}, need>{MIN_FLUX}" for i in range(n)]

    # ── Filter 3: resolved ──
    mask_resolved = a > MIN_SEMI_MAJOR
    diag_resolved = [f"a={a[i]:.2f}px, need>{MIN_SEMI_MAJOR}" for i in range(n)]

    # ── Filter 4: artifact ──
    safe_flux = np.maximum(flux, 1e-10)
    peak_ratio = peak / safe_flux
    mask_artifact = peak_ratio < MAX_PEAK_FLUX_RATIO
    diag_artifact = [f"peak/flux={peak_ratio[i]:.4f}, need<{MAX_PEAK_FLUX_RATIO}" for i in range(n)]

    # ── Filter 5: elongation ──
    safe_b = np.maximum(b, 0.01)
    elongation = a / safe_b
    mask_elongation = elongation < MAX_ELONGATION
    diag_elongation = [f"a/b={elongation[i]:.2f}, need<{MAX_ELONGATION}" for i in range(n)]

    # ── Filter 6: SNR ──
    flux_err = np.sqrt(npix.astype(float)) * rms_val
    snr = flux / np.maximum(flux_err, 1e-10)
    mask_snr = snr >= MIN_SNR
    diag_snr = [f"SNR={snr[i]:.1f}, need>={MIN_SNR}" for i in range(n)]

    # ── Filter 7: concentration (PSF-adaptive) ──
    _conc_psf_fwhm = None
    if instrument and waveband and pixel_scale:
        _conc_psf_fwhm = sh.get_psf_fwhm_pixels(instrument, waveband, pixel_scale)
    if _conc_psf_fwhm is not None:
        r_inner = max(1.5, 1.5 * _conc_psf_fwhm)
        r_outer = max(4.0, 4.0 * _conc_psf_fwhm)
    else:
        r_inner, r_outer = 3.0, 8.0
    mask_concentration = np.ones(n, dtype=bool)
    diag_concentration = ["(skipped)" for _ in range(n)]
    try:
        flux_small, _, _ = sep.sum_circle(image_data, objects['x'], objects['y'],
                                           r=r_inner, err=rms_val)
        flux_large, _, _ = sep.sum_circle(image_data, objects['x'], objects['y'],
                                           r=r_outer, err=rms_val)
        safe_large = np.maximum(flux_large, 1e-10)
        concentration = flux_small / safe_large
        cmin, cmax = CONCENTRATION_RANGE
        mask_concentration = (concentration > cmin) & (concentration < cmax)
        diag_concentration = [
            f"C={concentration[i]:.3f} (r_in={r_inner:.1f},r_out={r_outer:.1f}), need ({cmin},{cmax})"
            for i in range(n)
        ]
    except Exception as e:
        diag_concentration = [f"(failed: {e})" for _ in range(n)]

    # ── Filter 8: star/galaxy CI ──
    mask_star = np.ones(n, dtype=bool)
    diag_star = ["(skipped)" for _ in range(n)]
    try:
        _img = np.ascontiguousarray(image_data, dtype=np.float64)
        rmax = np.clip(6.0 * a, 5.0, 200.0)
        r25, _ = sep.flux_radius(_img, objects['x'], objects['y'], rmax, 0.25, subpix=5)
        r75, _ = sep.flux_radius(_img, objects['x'], objects['y'], rmax, 0.75, subpix=5)
        safe_r75 = np.maximum(r75, 0.1)
        ci = r25 / safe_r75
        is_star_like = ci >= MAX_STAR_CI
        mask_star = ~is_star_like
        diag_star = [f"R25/R75={ci[i]:.3f}, need<{MAX_STAR_CI}" for i in range(n)]
    except Exception as e:
        diag_star = [f"(failed: {e})" for _ in range(n)]

    # ── Filter 9: PSF-adaptive resolved ──
    mask_psf = np.ones(n, dtype=bool)
    diag_psf = ["(skipped – no PSF data)" for _ in range(n)]
    psf_fwhm_px = None
    if instrument and waveband and pixel_scale:
        psf_fwhm_px = sh.get_psf_fwhm_pixels(instrument, waveband, pixel_scale)
    if psf_fwhm_px is not None:
        try:
            _img = np.ascontiguousarray(image_data, dtype=np.float64)
            rmax_psf = np.clip(8.0 * a, 5.0, 200.0)
            r50, _ = sep.flux_radius(_img, objects['x'], objects['y'],
                                      rmax_psf, 0.50, subpix=5)
            psf_r50 = 0.54 * psf_fwhm_px
            is_unresolved = r50 <= (PSF_RESOLVED_THRESHOLD * psf_r50)
            mask_psf = ~is_unresolved
            diag_psf = [
                f"R50={r50[i]:.2f}px, PSF_R50={psf_r50:.2f}px, threshold={PSF_RESOLVED_THRESHOLD}x → {'UNRESOLVED' if is_unresolved[i] else 'resolved'}"
                for i in range(n)
            ]
        except Exception as e:
            diag_psf = [f"(failed: {e})" for _ in range(n)]

    # ── Filter 10: saturation (flat-top) ──
    mask_saturation = np.ones(n, dtype=bool)
    diag_saturation = ["below bright floor" for _ in range(n)]
    try:
        img_h, img_w = image_data.shape
        pos_pixels = image_data[image_data > 0]
        bright_floor = np.percentile(pos_pixels, 99.0) if len(pos_pixels) > 0 else 0.0
        for _i in range(n):
            pk = float(peak[_i])
            if pk < bright_floor:
                diag_saturation[_i] = f"peak={pk:.2f} < p99={bright_floor:.2f}, not bright enough to check"
                continue
            _x, _y = int(objects[_i]['x']), int(objects[_i]['y'])
            _y0, _y1 = max(0, _y - 2), min(img_h, _y + 3)
            _x0, _x1 = max(0, _x - 2), min(img_w, _x + 3)
            patch = image_data[_y0:_y1, _x0:_x1]
            if patch.size == 0:
                continue
            tol = abs(pk) * 0.001 if abs(pk) > 1e-12 else 1e-12
            n_flat = int(np.sum(np.abs(patch - pk) < tol))
            if n_flat >= 4:
                mask_saturation[_i] = False
                diag_saturation[_i] = f"peak={pk:.2f}, {n_flat} flat-top pixels in 5x5 box (>=4 = SATURATED)"
            else:
                diag_saturation[_i] = f"peak={pk:.2f}, {n_flat} flat-top pixels in 5x5 box (<4 = OK)"
    except Exception as e:
        diag_saturation = [f"(failed: {e})" for _ in range(n)]

    # ── Filter 11: Kron radius ──
    mask_kron = np.ones(n, dtype=bool)
    diag_kron = ["(skipped)" for _ in range(n)]
    try:
        _img = np.ascontiguousarray(image_data, dtype=np.float64)
        kronrad, krflag = sep.kron_radius(
            _img, objects['x'], objects['y'],
            objects['a'], objects['b'], objects['theta'],
            r=6.0
        )
        is_compact = (kronrad < MIN_KRON_RADIUS) | ((krflag & 1) != 0)
        mask_kron = ~is_compact
        diag_kron = [
            f"Kron_r={kronrad[i]:.2f}, flag={krflag[i]}, need>={MIN_KRON_RADIUS}"
            for i in range(n)
        ]
    except Exception as e:
        diag_kron = [f"(failed: {e})" for _ in range(n)]

    # Return ordered list of (name, mask, diag_list)
    filters = [
        ('npix',          mask_npix,          diag_npix),
        ('flux',          mask_flux,          diag_flux),
        ('resolved',      mask_resolved,      diag_resolved),
        ('artifact',      mask_artifact,      diag_artifact),
        ('elongation',    mask_elongation,    diag_elongation),
        ('snr',           mask_snr,           diag_snr),
        ('concentration', mask_concentration, diag_concentration),
        ('star',          mask_star,          diag_star),
        ('psf',           mask_psf,           diag_psf),
        ('saturation',    mask_saturation,    diag_saturation),
        ('kron',          mask_kron,          diag_kron),
    ]
    return filters


def classify_extract_object(obj, bkgless, nan_mask):
    """Determine why _extract_object would accept/reject, returning (reason_str, crop|None)."""
    x_center, y_center = int(obj['x']), int(obj['y'])
    try:
        max_half_size = int(PADDING * max(obj['a'], obj['b']))
    except Exception:
        return "bounding box contains NaN", None

    img_h, img_w = bkgless.shape
    edge_margin = 10
    if (x_center < edge_margin or x_center >= img_w - edge_margin or
        y_center < edge_margin or y_center >= img_h - edge_margin):
        return f"too close to edge (x={x_center},y={y_center}, margin={edge_margin})", None

    x_min = max(0, x_center - max_half_size)
    x_max = min(img_w, x_center + max_half_size)
    y_min = max(0, y_center - max_half_size)
    y_max = min(img_h, y_center + max_half_size)
    w, h = abs(x_max - x_min), abs(y_max - y_min)

    if w < MIN_SIZE or h < MIN_SIZE:
        return f"crop too small ({w}x{h}px, min={MIN_SIZE})", None
    if w > MAX_SIZE or h > MAX_SIZE:
        return f"crop too large ({w}x{h}px, max={MAX_SIZE})", None

    if nan_mask is not None:
        crop_nan = nan_mask[y_min:y_max, x_min:x_max]
        nan_frac = np.sum(crop_nan) / max(crop_nan.size, 1)
        if nan_frac > 0.02:
            return f"too many NaN pixels ({nan_frac:.1%} > 2%)", None

    crop = bkgless[y_min:y_max, x_min:x_max]
    return "OK", crop


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    image_data = sh.get_main_fits_data(FITS_PATH)
    meta = sh.get_relevant_fits_meta_data(FITS_PATH)

    pixel_scale = None
    pixar_a2 = meta.get('PIXAR_A2')
    if pixar_a2 is not None:
        val = pixar_a2[0] if isinstance(pixar_a2, (tuple, list)) else pixar_a2
        if val and float(val) > 0:
            pixel_scale = math.sqrt(float(val))

    instrument, waveband = 'niriss', 'clear-f200w'

    # NaN mask
    with fits.open(FITS_PATH) as hdul:
        try:
            raw = hdul['SCI'].data
        except (KeyError, IndexError):
            raw = hdul[0].data
        nan_mask = np.isnan(raw) if raw is not None else None

    # ── SEP detection ────────────────────────────────────────────────────
    bkg = sh.get_image_background(image_data)
    bkgless = sh.subtract_bkg(image_data)
    objects = sh.extract_objects(bkgless, bkg, detection_threshold=5.0,
                                pixel_stack_limit=5_000_000,
                                sub_object_limit=50_000)
    n = len(objects)
    print(f"Raw detections: {n}")

    bkg_rms = sep.Background(image_data).globalrms

    # ── Compute all filter masks ─────────────────────────────────────────
    filters = compute_all_filter_masks(bkgless, objects, bkg_rms,
                                       instrument, waveband, pixel_scale)

    # ── Classify each detection ──────────────────────────────────────────
    # For each object: find the FIRST filter it fails (priority order).
    # If it passes all 11 filters, check _extract_object and post-crop.
    obj_labels = []      # filter name that killed it, or 'pass'
    obj_details = []     # list of detail strings per object

    combined_mask = np.ones(n, dtype=bool)
    for fname, fmask, _ in filters:
        combined_mask &= fmask

    for i in range(n):
        details = []
        details.append(f"  Position: x={objects[i]['x']:.1f}, y={objects[i]['y']:.1f}")
        details.append(f"  a={objects[i]['a']:.2f}, b={objects[i]['b']:.2f}, "
                        f"theta={objects[i]['theta']:.3f}")
        details.append(f"  flux={objects[i]['flux']:.2f}, peak={objects[i]['peak']:.2f}, "
                        f"npix={int(objects[i]['npix'])}")

        first_fail = None
        for fname, fmask, fdiag in filters:
            passed = bool(fmask[i])
            status = "PASS" if passed else "FAIL"
            label_color, label_text = FILTER_COLORS[fname]
            details.append(f"  [{status}] {label_text}: {fdiag[i]}")
            if not passed and first_fail is None:
                first_fail = fname

        if first_fail is not None:
            obj_labels.append(first_fail)
            obj_details.append(details)
            continue

        # Passed all 11 filters → check _extract_object
        eo_reason, crop = classify_extract_object(objects[i], bkgless, nan_mask)
        details.append(f"  [{'PASS' if crop is not None else 'FAIL'}] Crop extraction: {eo_reason}")
        if crop is None:
            obj_labels.append('extract_object')
            obj_details.append(details)
            continue

        # Check post-crop quality
        has_structure = sh.thumbnail_has_structure(crop)
        has_profile = sh.thumbnail_has_galaxy_profile(crop)
        details.append(f"  [{'PASS' if has_structure else 'FAIL'}] thumbnail_has_structure")
        details.append(f"  [{'PASS' if has_profile else 'FAIL'}] thumbnail_has_galaxy_profile")
        if not has_structure or not has_profile:
            obj_labels.append('post_crop')
            obj_details.append(details)
            continue

        obj_labels.append('pass')
        obj_details.append(details)

    # ── Summary counts ───────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(obj_labels)
    print(f"\n{'Label':<20} {'Count':>5}")
    print("-" * 28)
    for key in list(FILTER_COLORS.keys()):
        if counts[key] > 0:
            _, desc = FILTER_COLORS[key]
            print(f"  {desc:<26} {counts[key]:>3}")
    print(f"  {'TOTAL':<26} {n:>3}")

    # ── Write .txt report ────────────────────────────────────────────────
    with open(TXT_OUT, 'w') as f:
        f.write(f"Filter breakdown for: {os.path.basename(FITS_PATH)}\n")
        f.write(f"Raw detections: {n}\n\n")
        f.write("Summary:\n")
        for key in list(FILTER_COLORS.keys()):
            if counts[key] > 0:
                color, desc = FILTER_COLORS[key]
                f.write(f"  {desc:<36} {counts[key]:>3}  (color: {color})\n")
        f.write(f"  {'TOTAL':<36} {n:>3}\n")
        f.write("\n" + "=" * 72 + "\n")
        f.write("Per-detection details:\n")
        f.write("=" * 72 + "\n\n")

        for i in range(n):
            color, desc = FILTER_COLORS[obj_labels[i]]
            f.write(f"Detection #{i+1}: {desc}  (color: {color})\n")
            for line in obj_details[i]:
                f.write(line + "\n")
            f.write("\n")

    print(f"\nWrote: {TXT_OUT}")

    # ── Plot ─────────────────────────────────────────────────────────────
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(bkgless)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(bkgless, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)

    for i in range(n):
        color = FILTER_COLORS[obj_labels[i]][0]
        half = max(PADDING * max(objects[i]['a'], objects[i]['b']), 8)
        e = Ellipse(
            xy=(objects[i]['x'], objects[i]['y']),
            width=6 * objects[i]['a'],
            height=6 * objects[i]['b'],
            angle=objects[i]['theta'] * 180.0 / np.pi,
            linewidth=1.0 if obj_labels[i] != 'pass' else 2.0,
            edgecolor=color,
            facecolor='none',
            alpha=0.85 if obj_labels[i] != 'pass' else 1.0,
        )
        ax.add_patch(e)
        # Small index label
        ax.text(objects[i]['x'] + 3 * objects[i]['a'] + 2, objects[i]['y'],
                str(i + 1), fontsize=5, color=color, ha='left', va='center',
                alpha=0.7)

    # Legend
    legend_handles = []
    for key in list(FILTER_COLORS.keys()):
        if counts[key] > 0:
            color, desc = FILTER_COLORS[key]
            legend_handles.append(
                Line2D([0], [0], marker='o', color='none', markeredgecolor=color,
                       markerfacecolor='none', markersize=8, label=f"{desc} ({counts[key]})")
            )
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
              framealpha=0.8, facecolor='black', edgecolor='white',
              labelcolor='white')

    ax.set_title(
        f"jw01571-c1011  NIRISS clear-f200w  —  {n} detections, "
        f"{counts['pass']} passed",
        fontsize=12, color='white',
    )
    ax.tick_params(colors='white', labelsize=7)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    fig.savefig(PNG_OUT, dpi=180, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"Saved: {PNG_OUT}")


if __name__ == '__main__':
    main()
