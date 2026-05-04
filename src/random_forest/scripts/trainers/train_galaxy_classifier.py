#!/usr/bin/env python3
"""Train a Random Forest classifier to distinguish galaxies from non-galaxies
using features extracted from FITS thumbnails or PNG images.

Features mirror those computed by multi_scale_structure_score() and
thumbnail_has_galaxy_profile() in sep_helpers.py, plus additional morphological
statistics (Gini, M20, half-light radius, etc.).

Usage:
    # Train on FITS thumbnails (using hardcoded galaxy lists):
    python scripts/train_galaxy_classifier.py

    # Train on FITS thumbnails referenced by labeled galaxy-separator PNGs:
    python scripts/train_galaxy_classifier.py \
        --fits-training-root randomforest_training_data/galaxy_separator_RF

    # Train on the labeled galaxy-separator PNG bins:
    python scripts/train_galaxy_classifier.py \
        --png-training-root randomforest_training_data/galaxy_separator_RF

    # Train on explicit PNG directories:
    python scripts/train_galaxy_classifier.py \
        --png-galaxies inference/RUN-1/1-galaxies/ \
        --png-non-galaxies inference/RUN-1/1-not-galaxies/
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
from scipy import ndimage
from astropy.io import fits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
THUMB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(ROOT_DIR)),
    'outputs', 'JWST_galaxy_thumbnails'
)
MODEL_DIR = os.path.join(ROOT_DIR, 'output', 'models', 'galaxy_classifier')
FITS_MODEL_PATH = os.path.join(MODEL_DIR, 'galaxy_classifier_fits.joblib')
PNG_MODEL_PATH = os.path.join(MODEL_DIR, 'galaxy_classifier_png.joblib')
FITS_STATS_PATH = os.path.join(MODEL_DIR, 'galaxy_classifier_fits_stats.txt')
PNG_STATS_PATH = os.path.join(MODEL_DIR, 'galaxy_classifier_png_stats.txt')
LEGACY_MODEL_PATH = os.path.join(MODEL_DIR, 'galaxy_classifier.joblib')
OLD_FITS_MODEL_PATH = os.path.join(ROOT_DIR, 'output', 'galaxy_classifier_fits.joblib')
OLD_PNG_MODEL_PATH = os.path.join(ROOT_DIR, 'output', 'galaxy_classifier_png.joblib')
OLD_LEGACY_MODEL_PATH = os.path.join(ROOT_DIR, 'output', 'galaxy_classifier.joblib')
DEFAULT_FITS_MAPPING_CSV = os.path.join(
    ROOT_DIR,
    'output',
    'galaxy_separator_rf_raw_mapping.csv',
)
DEFAULT_PNG_TRAINING_ROOT = os.path.join(
    ROOT_DIR,
    'randomforest_training_data',
    'galaxy_separator_RF',
)
DEFAULT_PNG_GALAXY_DIR = os.path.join(DEFAULT_PNG_TRAINING_ROOT, '1-galaxies')
DEFAULT_PNG_NON_GALAXY_DIR = os.path.join(DEFAULT_PNG_TRAINING_ROOT, '1-not-galaxies')

sys.path.insert(0, os.path.join(ROOT_DIR, 'modules'))

from galaxy_separator_png_utils import load_png_morphology_array


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(data: np.ndarray) -> dict:
    """Extract all morphological features from a 2D FITS thumbnail array.

    Returns a dict of scalar feature values.
    """
    d = data.astype(np.float64)
    d = np.nan_to_num(d, nan=0.0)
    h, w = d.shape
    cy, cx = h / 2.0, w / 2.0

    feats = {}
    feats['min_dim'] = float(min(h, w))

    # --- Multi-scale energy fractions (mirrors multi_scale_structure_score) ---
    total_var = np.var(d)
    if total_var < 1e-12:
        # Flat image – fill with zeros
        for k in ['e1', 'e4', 'e16', 'coarse_to_fine', 'directional_ratio',
                   'saturated_fraction', 'center_fraction', 'outer_fraction',
                   'interior_zero_frac', 'quadrant_ratio', 'max_quadrant_frac',
                   'n_sources', 'concentration', 'asymmetry', 'total_rise',
                   'half_light_radius', 'peak_offset_frac', 'gini', 'm20',
                   'snr', 'kurtosis_val']:
            feats[k] = 0.0
        return feats

    min_dim = min(h, w)
    coarse_sigma = max(4.0, min(12.0, min_dim / 5.0))
    var_s1  = np.var(ndimage.gaussian_filter(d, sigma=1.0))
    var_s4  = np.var(ndimage.gaussian_filter(d, sigma=3.0))
    var_s16 = np.var(ndimage.gaussian_filter(d, sigma=coarse_sigma))

    e1  = var_s1  / total_var
    e4  = var_s4  / total_var
    e16 = var_s16 / total_var
    feats['e1']  = float(e1)
    feats['e4']  = float(e4)
    feats['e16'] = float(e16)
    feats['coarse_to_fine'] = float(e16 / e1) if e1 > 1e-12 else 0.0

    # Directional ratio
    row_var = np.var(np.mean(d, axis=1))
    col_var = np.var(np.mean(d, axis=0))
    feats['directional_ratio'] = float(
        max(row_var, col_var) / max(min(row_var, col_var), 1e-12)
    )

    # Saturated fraction
    img_max = np.max(d)
    img_min = np.min(d)
    img_range = img_max - img_min
    if img_range > 1e-12:
        sat_thresh = img_max - 0.05 * img_range
        feats['saturated_fraction'] = float(np.sum(d >= sat_thresh)) / d.size
    else:
        feats['saturated_fraction'] = 1.0

    # --- Radial profile features (mirrors thumbnail_has_galaxy_profile) ---
    total_flux = np.sum(np.abs(d))
    y_idx, x_idx = np.ogrid[:h, :w]
    r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    r_max = np.sqrt(cx**2 + cy**2)

    # Interior zero fraction
    inner_mask = r < r_max * 0.5
    n_inner = int(np.sum(inner_mask))
    feats['interior_zero_frac'] = (
        float(np.sum(d[inner_mask] == 0)) / n_inner if n_inner > 0 else 0.0
    )

    # Quadrant flux analysis
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

    # Number of bright sources
    threshold = np.mean(d) + 3.0 * np.std(d)
    binary = d > threshold
    _, n_sources = ndimage.label(binary)
    feats['n_sources'] = float(n_sources)

    # Radial bins
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

    # Concentration (r<3 / r<8)
    mask_r3 = r < 3
    mask_r8 = r < 8
    flux_r3 = np.sum(np.abs(d[mask_r3]))
    flux_r8 = np.sum(np.abs(d[mask_r8]))
    feats['concentration'] = float(flux_r3 / flux_r8) if flux_r8 > 1e-12 else 0.0

    # Rotational asymmetry
    rot180 = np.rot90(np.rot90(d))
    feats['asymmetry'] = float(np.sum(np.abs(d - rot180)) / (2.0 * total_flux)) if total_flux > 1e-12 else 0.0

    # Profile monotonicity: total fractional rise
    if bin_mean[0] > 1e-12:
        total_rise = 0.0
        for i in range(1, n_bins):
            if bin_mean[i] > bin_mean[i - 1]:
                total_rise += (bin_mean[i] - bin_mean[i - 1]) / bin_mean[0]
        feats['total_rise'] = float(total_rise)
    else:
        feats['total_rise'] = 0.0

    # Half-light radius
    sorted_r = np.sort(r.ravel())
    sorted_flux = np.abs(d).ravel()[np.argsort(r.ravel())]
    cumflux = np.cumsum(sorted_flux)
    if cumflux[-1] > 1e-12:
        half_idx = np.searchsorted(cumflux, 0.5 * cumflux[-1])
        feats['half_light_radius'] = float(sorted_r[min(half_idx, len(sorted_r) - 1)])
    else:
        feats['half_light_radius'] = 0.0

    # Peak offset from center
    peak_y, peak_x = np.unravel_index(np.argmax(d), d.shape)
    peak_offset = np.sqrt((peak_x - cx)**2 + (peak_y - cy)**2)
    feats['peak_offset_frac'] = float(peak_offset / r_max) if r_max > 0 else 0.0

    # --- Additional morphological features ---
    # Gini coefficient (light concentration inequality)
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

    # M20 (second-order moment of the brightest 20% of flux)
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

    # Signal-to-noise ratio (peak / std of background)
    bg_mask = r > r_max * 0.7
    if np.any(bg_mask):
        bg_std = np.std(d[bg_mask])
        feats['snr'] = float(img_max / bg_std) if bg_std > 1e-12 else 0.0
    else:
        feats['snr'] = 0.0

    # Kurtosis (peakedness of pixel distribution)
    mean_val = np.mean(d)
    std_val = np.std(d)
    if std_val > 1e-12:
        feats['kurtosis_val'] = float(np.mean(((d - mean_val) / std_val)**4) - 3.0)
    else:
        feats['kurtosis_val'] = 0.0

    return feats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_thumbnail(fits_path: str) -> np.ndarray:
    """Load FITS thumbnail as float64 array."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)
    return np.nan_to_num(data, nan=0.0)


def load_png(png_path: str) -> np.ndarray:
    """Load PNG image as a morphology-only 2D float64 grayscale array."""
    return load_png_morphology_array(png_path)


def source_name_to_fits_path(source_name: str) -> str:
    """Convert a mapping CSV source_name into the exact FITS thumbnail path."""
    parts = source_name.split('_')

    filter_idx = None
    for idx, part in enumerate(parts):
        if part.startswith('f') and any(char.isdigit() for char in part):
            filter_idx = idx
            break

    if filter_idx is None:
        raise ValueError(f"Cannot parse filter from source name: {source_name}")

    coord_parts = parts[:filter_idx]
    filter_parts = parts[filter_idx:]

    dec_start = None
    for idx in range(1, len(coord_parts)):
        if coord_parts[idx].startswith('-'):
            dec_start = idx
            break
    if dec_start is None:
        dec_start = 2

    ra_str = '.'.join(coord_parts[:dec_start])
    dec_str = '.'.join(coord_parts[dec_start:])
    if dec_str.startswith('-.'):
        dec_str = '-' + dec_str[2:]

    fits_name = f"{ra_str}_{dec_str}_{'_'.join(filter_parts)}.fits"
    return os.path.join(THUMB_DIR, fits_name)


def normalize_label_dir(label_dir: str) -> str:
    """Normalize a label directory path down to its terminal directory name."""
    return os.path.basename(os.path.normpath(str(label_dir)))


def load_fits_training_lists():
    """Load the FITS training lists only when FITS mode is actually used."""
    sys.path.insert(0, SCRIPT_DIR)
    from test_filter_sets import GALAXIES, NOT_GALAXIES, name_to_fits_path

    return GALAXIES, NOT_GALAXIES, name_to_fits_path


def build_dataset():
    """Build feature matrix and labels from GALAXIES and NOT_GALAXIES lists (FITS)."""
    rows = []
    galaxies, not_galaxies, name_to_fits_path = load_fits_training_lists()

    for label, name_list in [(1, galaxies), (0, not_galaxies)]:
        for name in name_list:
            fits_path = name_to_fits_path(name)
            if not os.path.exists(fits_path):
                continue
            data = load_thumbnail(fits_path)
            feats = extract_features(data)
            feats['name'] = name
            feats['label'] = label
            rows.append(feats)

    df = pd.DataFrame(rows)
    return df


def build_png_dataset(galaxy_dir: str, non_galaxy_dir: str):
    """Build feature matrix and labels from PNG directories."""
    rows = []

    for label, directory, label_name in [
        (1, galaxy_dir, 'galaxy'),
        (0, non_galaxy_dir, 'non-galaxy'),
    ]:
        if not os.path.isdir(directory):
            print(f"Warning: {directory} not found, skipping {label_name}")
            continue
        pngs = sorted(f for f in os.listdir(directory) if f.lower().endswith('.png'))
        print(f"  {label_name}: {len(pngs)} PNGs from {directory}")
        for fname in pngs:
            path = os.path.join(directory, fname)
            data = load_png(path)
            feats = extract_features(data)
            feats['name'] = fname
            feats['label'] = label
            rows.append(feats)

    df = pd.DataFrame(rows)
    return df


def build_mapped_fits_dataset(training_root: str, mapping_csv: str):
    """Build a FITS feature dataset using PNG labels plus the mapping CSV."""
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    mapping_df = pd.read_csv(mapping_csv, dtype=str).fillna('')
    required_cols = {'label_dir', 'score_filename', 'source_name'}
    missing_cols = required_cols - set(mapping_df.columns)
    if missing_cols:
        missing_text = ', '.join(sorted(missing_cols))
        raise ValueError(f"Mapping CSV is missing required columns: {missing_text}")

    mapping_index = {}
    for row in mapping_df.itertuples(index=False):
        key = (normalize_label_dir(row.label_dir), row.score_filename)
        mapping_index.setdefault(key, []).append(row.source_name)

    rows = []
    skipped_counts = {}
    warned_counts = {}

    def record_skip(reason: str, message: str):
        skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
        seen = warned_counts.get(reason, 0)
        if seen < 10:
            print(f"  Warning [{reason}]: {message}")
        elif seen == 10:
            print(f"  Warning [{reason}]: additional entries suppressed...")
        warned_counts[reason] = seen + 1

    for label, label_dir_name, label_name in [
        (1, '1-galaxies', 'galaxy'),
        (0, '1-not-galaxies', 'non-galaxy'),
    ]:
        directory = os.path.join(training_root, label_dir_name)
        if not os.path.isdir(directory):
            record_skip('missing_label_dir', f"{directory} not found; skipping {label_name} set")
            continue

        pngs = sorted(fname for fname in os.listdir(directory) if fname.lower().endswith('.png'))
        print(f"  {label_name}: {len(pngs)} labeled PNGs from {directory}")

        for fname in pngs:
            mapping_key = (label_dir_name, fname)
            source_names = mapping_index.get(mapping_key)
            if not source_names:
                record_skip('missing_mapping', f"{label_dir_name}/{fname} has no mapping row")
                continue
            if len(source_names) != 1:
                record_skip(
                    'duplicate_mapping',
                    f"{label_dir_name}/{fname} resolves to multiple source names: {source_names!r}",
                )
                continue

            source_name = source_names[0].strip()
            if not source_name:
                record_skip('missing_source_name', f"{label_dir_name}/{fname} has an empty source_name")
                continue

            try:
                fits_path = source_name_to_fits_path(source_name)
            except ValueError as exc:
                record_skip('invalid_source_name', f"{label_dir_name}/{fname}: {exc}")
                continue

            if not os.path.exists(fits_path):
                record_skip(
                    'missing_fits',
                    f"{label_dir_name}/{fname} mapped to missing FITS {fits_path}",
                )
                continue

            try:
                data = load_thumbnail(fits_path)
            except Exception as exc:
                record_skip(
                    'unreadable_fits',
                    f"{label_dir_name}/{fname} failed to load {fits_path}: {exc}",
                )
                continue

            feats = extract_features(data)
            feats['name'] = fname
            feats['label'] = label
            feats['label_dir'] = label_dir_name
            feats['source_name'] = source_name
            feats['fits_path'] = fits_path
            rows.append(feats)

    return pd.DataFrame(rows), skipped_counts


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    'e1', 'e4', 'e16', 'coarse_to_fine', 'directional_ratio',
    'saturated_fraction', 'center_fraction', 'outer_fraction',
    'interior_zero_frac', 'quadrant_ratio', 'max_quadrant_frac',
    'n_sources', 'concentration', 'asymmetry', 'total_rise',
    'half_light_radius', 'peak_offset_frac', 'gini', 'm20',
    'snr', 'kurtosis_val', 'min_dim',
]


def train_and_evaluate(
    df: pd.DataFrame,
    model_path: str,
    compatibility_paths: tuple[str, ...] = (),
    stats_path: str | None = None,
    run_label: str | None = None,
    skipped_counts: dict[str, int] | None = None,
):
    """Train RF classifier with stratified 5-fold CV and report results."""
    X = df[FEATURE_COLS].values
    y = df['label'].values

    report_lines = []

    def emit(line: str = ''):
        print(line)
        report_lines.append(line)

    if run_label:
        emit(f"Mode: {run_label}")

    emit(f"\nDataset: {len(df)} samples ({int(y.sum())} galaxies, {int((1-y).sum())} non-galaxies)")
    emit(f"Features: {len(FEATURE_COLS)}")

    if skipped_counts:
        emit('')
        emit('Skipped examples:')
        for reason in sorted(skipped_counts):
            emit(f"  {reason:20s} {skipped_counts[reason]}")

    # --- Cross-validation ---
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, X, y, cv=cv)

    emit("\n" + "="*60)
    emit("  5-Fold Stratified Cross-Validation Results")
    emit("="*60)
    emit(classification_report(y, y_pred_cv, target_names=['non-galaxy', 'galaxy']))

    cm = confusion_matrix(y, y_pred_cv)
    tn, fp, fn, tp = cm.ravel()
    emit("Confusion matrix:")
    emit(f"  TN={tn}  FP={fp}")
    emit(f"  FN={fn}  TP={tp}")
    emit(f"\nAccuracy:  {accuracy_score(y, y_pred_cv):.4f}")
    emit(f"Precision: {precision_score(y, y_pred_cv):.4f}")
    emit(f"Recall:    {recall_score(y, y_pred_cv):.4f}")
    emit(f"F1:        {f1_score(y, y_pred_cv):.4f}")

    # --- Misclassifications ---
    misclass = np.where(y != y_pred_cv)[0]
    if len(misclass) > 0:
        emit(f"\nMisclassified ({len(misclass)}):")
        for idx in misclass:
            true_label = 'galaxy' if y[idx] == 1 else 'non-galaxy'
            pred_label = 'galaxy' if y_pred_cv[idx] == 1 else 'non-galaxy'
            emit(f"  {df.iloc[idx]['name']:50s}  true={true_label}  pred={pred_label}")

    # --- Train final model on full data ---
    clf.fit(X, y)

    # Feature importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    emit(f"\nFeature importances:")
    for i in sorted_idx:
        emit(f"  {FEATURE_COLS[i]:25s}  {importances[i]:.4f}")

    model_bundle = {
        'model': clf,
        'feature_cols': FEATURE_COLS,
    }

    save_paths = list(dict.fromkeys((model_path, *compatibility_paths)))
    for save_path in save_paths:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model_bundle, save_path)

    emit("\nModels saved to:")
    for save_path in save_paths:
        emit(f"  {save_path}")

    if stats_path is not None:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        emit(f"\nStats saved to:")
        emit(f"  {stats_path}")
        with open(stats_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(report_lines) + '\n')

    return clf, y_pred_cv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Random Forest galaxy classifier on FITS or PNG data."
    )
    parser.add_argument(
        '--fits-training-root',
        default=None,
        help=(
            "Training root containing 1-galaxies/ and 1-not-galaxies/ "
            "subdirectories whose score-named PNGs are mapped back to source "
            "FITS thumbnails before feature extraction."
        ),
    )
    parser.add_argument(
        '--mapping-csv',
        default=None,
        help=(
            "Optional mapping CSV for --fits-training-root. Defaults to "
            "output/galaxy_separator_rf_raw_mapping.csv."
        ),
    )
    parser.add_argument(
        '--png-galaxies',
        default=None,
        help="Directory of galaxy PNG images (for PNG model training)."
    )
    parser.add_argument(
        '--png-non-galaxies',
        default=None,
        help="Directory of non-galaxy PNG images (for PNG model training)."
    )
    parser.add_argument(
        '--png-training-root',
        default=None,
        help=(
            "Training root containing 1-galaxies/ and 1-not-galaxies/ "
            "subdirectories."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mapping_csv and not args.fits_training_root:
        print(
            "Error: --mapping-csv can only be used with --fits-training-root.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.png_training_root and (args.png_galaxies or args.png_non_galaxies):
        print(
            "Error: use either --png-training-root or the explicit PNG directories, not both.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.fits_training_root and (
        args.png_training_root or args.png_galaxies or args.png_non_galaxies
    ):
        print(
            "Error: use either --fits-training-root or the PNG training arguments, not both.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.png_training_root:
        png_training_root = os.path.abspath(args.png_training_root)
        args.png_galaxies = os.path.join(png_training_root, '1-galaxies')
        args.png_non_galaxies = os.path.join(png_training_root, '1-not-galaxies')

    mapped_fits_mode = args.fits_training_root is not None
    png_mode = args.png_galaxies is not None or args.png_non_galaxies is not None

    if mapped_fits_mode:
        fits_training_root = os.path.abspath(args.fits_training_root)
        mapping_csv = os.path.abspath(args.mapping_csv or DEFAULT_FITS_MAPPING_CSV)

        print("=" * 60)
        print("  Galaxy vs Non-Galaxy Random Forest Classifier (Mapped FITS)")
        print("=" * 60)

        print("\nBuilding mapped FITS feature dataset...")
        print("Labeled PNGs provide classes, but features come from mapped FITS thumbnails.")
        print(f"Training root: {fits_training_root}")
        print(f"Mapping CSV:   {mapping_csv}")
        df, skipped_counts = build_mapped_fits_dataset(fits_training_root, mapping_csv)
        model_path = FITS_MODEL_PATH
        csv_name = 'galaxy_features_fits_mapped.csv'
        compatibility_paths = (LEGACY_MODEL_PATH,)
        stats_path = FITS_STATS_PATH
        run_label = 'Mapped FITS training from galaxy_separator_RF labels'
    elif png_mode:
        if not args.png_galaxies or not args.png_non_galaxies:
            print("Error: both --png-galaxies and --png-non-galaxies are required for PNG training.",
                  file=sys.stderr)
            sys.exit(1)

        print("=" * 60)
        print("  Galaxy vs Non-Galaxy Random Forest Classifier (PNG)")
        print("=" * 60)

        print("\nBuilding PNG feature dataset...")
        print("Annotated PNGs are auto-cropped to the morphology panel before feature extraction.")
        df = build_png_dataset(
            os.path.abspath(args.png_galaxies),
            os.path.abspath(args.png_non_galaxies),
        )
        model_path = PNG_MODEL_PATH
        csv_name = 'galaxy_features_png.csv'
        compatibility_paths = (OLD_PNG_MODEL_PATH,)
        skipped_counts = None
        stats_path = PNG_STATS_PATH
        run_label = 'PNG training from labeled separator dataset'
    else:
        print("=" * 60)
        print("  Galaxy vs Non-Galaxy Random Forest Classifier (FITS)")
        print("=" * 60)

        print("\nBuilding FITS feature dataset...")
        df = build_dataset()
        model_path = FITS_MODEL_PATH
        csv_name = 'galaxy_features_fits.csv'
        compatibility_paths = (
            LEGACY_MODEL_PATH
        )
        skipped_counts = None
        stats_path = FITS_STATS_PATH
        run_label = 'Legacy FITS training from hardcoded lists'

    if df.empty:
        print("Error: no training examples were loaded.", file=sys.stderr)
        sys.exit(1)

    # Save features CSV for inspection
    csv_path = os.path.join(ROOT_DIR, 'output', csv_name)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Features saved to: {csv_path}")

    train_and_evaluate(
        df,
        model_path,
        compatibility_paths=compatibility_paths,
        stats_path=stats_path,
        run_label=run_label,
        skipped_counts=skipped_counts,
    )


if __name__ == '__main__':
    main()
