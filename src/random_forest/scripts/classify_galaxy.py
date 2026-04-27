#!/usr/bin/env python3
"""Galaxy classification using the trained Random Forest model.

Provides a function to classify a single FITS thumbnail as galaxy or non-galaxy,
usable from other modules (e.g., during extraction in sep_helpers).

Usage standalone:
    python scripts/classify_galaxy.py /path/to/thumbnail.fits
"""

import os
import sys
import numpy as np
import joblib
from scipy import ndimage
from astropy.io import fits

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATHS = [
    os.path.join(ROOT_DIR, 'output', 'models', 'galaxy_classifier', 'galaxy_classifier_fits.joblib'),
    os.path.join(ROOT_DIR, 'output', 'models', 'galaxy_classifier', 'galaxy_classifier.joblib'),
    os.path.join(ROOT_DIR, 'output', 'galaxy_classifier_fits.joblib'),
    os.path.join(ROOT_DIR, 'output', 'galaxy_classifier.joblib'),
]

# Feature column order must match training
FEATURE_COLS = [
    'e1', 'e4', 'e16', 'coarse_to_fine', 'directional_ratio',
    'saturated_fraction', 'center_fraction', 'outer_fraction',
    'interior_zero_frac', 'quadrant_ratio', 'max_quadrant_frac',
    'n_sources', 'concentration', 'asymmetry', 'total_rise',
    'half_light_radius', 'peak_offset_frac', 'gini', 'm20',
    'snr', 'kurtosis_val', 'min_dim',
]

_model_cache = {}


def _load_model():
    """Load the trained model (cached after first call)."""
    if 'clf' not in _model_cache:
        model_path = next((path for path in MODEL_PATHS if os.path.exists(path)), None)
        if model_path is None:
            raise FileNotFoundError(
                "Model not found. Checked: " + ', '.join(MODEL_PATHS) +
                ". Run train_galaxy_classifier.py first."
            )
        data = joblib.load(model_path)
        _model_cache['clf'] = data['model']
        _model_cache['feature_cols'] = data['feature_cols']
    return _model_cache['clf']


def extract_features(data: np.ndarray) -> dict:
    """Extract morphological features from a 2D array (same as training)."""
    d = data.astype(np.float64)
    d = np.nan_to_num(d, nan=0.0)
    h, w = d.shape
    cy, cx = h / 2.0, w / 2.0

    feats = {}
    feats['min_dim'] = float(min(h, w))

    total_var = np.var(d)
    if total_var < 1e-12:
        for k in FEATURE_COLS:
            feats[k] = 0.0
        return feats

    min_dim = min(h, w)
    coarse_sigma = max(4.0, min(12.0, min_dim / 5.0))
    var_s1 = np.var(ndimage.gaussian_filter(d, sigma=1.0))
    var_s4 = np.var(ndimage.gaussian_filter(d, sigma=3.0))
    var_s16 = np.var(ndimage.gaussian_filter(d, sigma=coarse_sigma))

    e1 = var_s1 / total_var
    e4 = var_s4 / total_var
    e16 = var_s16 / total_var
    feats['e1'] = float(e1)
    feats['e4'] = float(e4)
    feats['e16'] = float(e16)
    feats['coarse_to_fine'] = float(e16 / e1) if e1 > 1e-12 else 0.0

    row_var = np.var(np.mean(d, axis=1))
    col_var = np.var(np.mean(d, axis=0))
    feats['directional_ratio'] = float(
        max(row_var, col_var) / max(min(row_var, col_var), 1e-12)
    )

    img_max = np.max(d)
    img_min = np.min(d)
    img_range = img_max - img_min
    if img_range > 1e-12:
        sat_thresh = img_max - 0.05 * img_range
        feats['saturated_fraction'] = float(np.sum(d >= sat_thresh)) / d.size
    else:
        feats['saturated_fraction'] = 1.0

    total_flux = np.sum(np.abs(d))
    y_idx, x_idx = np.ogrid[:h, :w]
    r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    r_max = np.sqrt(cx**2 + cy**2)

    inner_mask = r < r_max * 0.5
    n_inner = int(np.sum(inner_mask))
    feats['interior_zero_frac'] = (
        float(np.sum(d[inner_mask] == 0)) / n_inner if n_inner > 0 else 0.0
    )

    icy, icx = int(cy), int(cx)
    q_flux = [
        np.sum(np.abs(d[:icy, :icx])),
        np.sum(np.abs(d[:icy, icx:])),
        np.sum(np.abs(d[icy:, :icx])),
        np.sum(np.abs(d[icy:, icx:])),
    ]
    min_q, max_q = min(q_flux), max(q_flux)
    feats['quadrant_ratio'] = float(max_q / min_q) if min_q > 1e-12 else 100.0
    feats['max_quadrant_frac'] = float(max_q / total_flux) if total_flux > 1e-12 else 0.0

    threshold = np.mean(d) + 3.0 * np.std(d)
    binary = d > threshold
    _, n_sources = ndimage.label(binary)
    feats['n_sources'] = float(n_sources)

    n_bins = 8
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_flux = np.zeros(n_bins)
    bin_mean = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(mask):
            bin_flux[i] = np.sum(np.abs(d[mask]))
            bin_mean[i] = np.mean(np.abs(d[mask]))

    feats['center_fraction'] = float(bin_flux[0] / total_flux) if total_flux > 1e-12 else 0.0
    feats['outer_fraction'] = float(np.sum(bin_flux[4:]) / total_flux) if total_flux > 1e-12 else 0.0

    mask_r3 = r < 3
    mask_r8 = r < 8
    flux_r3 = np.sum(np.abs(d[mask_r3]))
    flux_r8 = np.sum(np.abs(d[mask_r8]))
    feats['concentration'] = float(flux_r3 / flux_r8) if flux_r8 > 1e-12 else 0.0

    rot180 = np.rot90(np.rot90(d))
    feats['asymmetry'] = float(np.sum(np.abs(d - rot180)) / (2.0 * total_flux)) if total_flux > 1e-12 else 0.0

    if bin_mean[0] > 1e-12:
        total_rise = 0.0
        for i in range(1, n_bins):
            if bin_mean[i] > bin_mean[i - 1]:
                total_rise += (bin_mean[i] - bin_mean[i - 1]) / bin_mean[0]
        feats['total_rise'] = float(total_rise)
    else:
        feats['total_rise'] = 0.0

    sorted_r = np.sort(r.ravel())
    sorted_flux = np.abs(d).ravel()[np.argsort(r.ravel())]
    cumflux = np.cumsum(sorted_flux)
    if cumflux[-1] > 1e-12:
        half_idx = np.searchsorted(cumflux, 0.5 * cumflux[-1])
        feats['half_light_radius'] = float(sorted_r[min(half_idx, len(sorted_r) - 1)])
    else:
        feats['half_light_radius'] = 0.0

    peak_y, peak_x = np.unravel_index(np.argmax(d), d.shape)
    peak_offset = np.sqrt((peak_x - cx)**2 + (peak_y - cy)**2)
    feats['peak_offset_frac'] = float(peak_offset / r_max) if r_max > 0 else 0.0

    abs_sorted = np.sort(np.abs(d).ravel())
    n = len(abs_sorted)
    cumsum = np.cumsum(abs_sorted)
    total = cumsum[-1]
    if total > 1e-12 and n > 1:
        feats['gini'] = float(
            (2.0 * np.sum((np.arange(1, n + 1) * abs_sorted))) / (n * total) - (n + 1) / n
        )
    else:
        feats['gini'] = 0.0

    abs_flat = np.abs(d).ravel()
    order = np.argsort(abs_flat)[::-1]
    cum = np.cumsum(abs_flat[order])
    total_moment = total_flux
    if total_moment > 1e-12:
        threshold_20 = 0.2 * total_moment
        bright_mask = cum <= threshold_20
        if not np.any(bright_mask):
            bright_mask[0] = True
        y_flat = np.repeat(np.arange(h), w)
        x_flat = np.tile(np.arange(w), h)
        m_total = np.sum(abs_flat * ((x_flat - cx)**2 + (y_flat - cy)**2))
        m_20 = np.sum(
            abs_flat[order[bright_mask]] *
            ((x_flat[order[bright_mask]] - cx)**2 + (y_flat[order[bright_mask]] - cy)**2)
        )
        feats['m20'] = float(np.log10(m_20 / m_total)) if m_total > 1e-12 and m_20 > 0 else 0.0
    else:
        feats['m20'] = 0.0

    bg_mask = r > r_max * 0.7
    if np.any(bg_mask):
        bg_std = np.std(d[bg_mask])
        feats['snr'] = float(img_max / bg_std) if bg_std > 1e-12 else 0.0
    else:
        feats['snr'] = 0.0

    mean_val = np.mean(d)
    std_val = np.std(d)
    if std_val > 1e-12:
        feats['kurtosis_val'] = float(np.mean(((d - mean_val) / std_val)**4) - 3.0)
    else:
        feats['kurtosis_val'] = 0.0

    return feats


def classify_thumbnail(data: np.ndarray, threshold: float = 0.5) -> tuple:
    """Classify a thumbnail as galaxy (True) or non-galaxy (False).

    Args:
        data: 2D numpy array of the thumbnail.
        threshold: Probability threshold for galaxy classification.

    Returns:
        (is_galaxy: bool, probability: float)
    """
    clf = _load_model()
    feats = extract_features(data)
    feature_vec = np.array([[feats[col] for col in FEATURE_COLS]])
    prob = clf.predict_proba(feature_vec)[0, 1]  # P(galaxy)
    return prob >= threshold, float(prob)


def main():
    """CLI: classify a single FITS thumbnail."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <thumbnail.fits> [threshold]")
        sys.exit(1)

    fits_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)
    data = np.nan_to_num(data, nan=0.0)

    is_galaxy, prob = classify_thumbnail(data, threshold)
    label = "GALAXY" if is_galaxy else "NON-GALAXY"
    print(f"{label}  (P={prob:.4f}, threshold={threshold})")


if __name__ == '__main__':
    main()
