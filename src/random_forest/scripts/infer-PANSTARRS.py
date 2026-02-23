#!/usr/bin/env python3
"""
Centralized PANSTARRS inference pipeline.

Takes a raw SpArcFiRe output CSV and a Random Forest model, then:
  1) Matches galaxies to Galaxy Zoo 2 ground truth by RA/DEC
  2) Sanitizes the CSV to match the RF training format
  3) Runs RF inference and generates Actual vs Predicted scatter plot
  4) (Optionally) downloads PANSTARRS thumbnails annotated with spirality

Outputs:
  a) Scatter plot PNG (predicted vs actual spirality)
  b) Cleaned CSV file
  c) Optional: PANSTARRS-inference-images/ directory with annotated PNGs

Usage:
    python infer-PANSTARRS.py \\
        --sparcfire-csv <raw SpArcFiRe output file> \\
        --gz2-csv <Galaxy Zoo 2 CSV file> \\
        --model <Random Forest model file> \\
        --output-dir <output directory> \\
        --[images]

Example Usage (as single line):
    python scripts/infer-PANSTARRS.py \
        --sparcfire-csv testing_data/SpArcFiRe_outputs_testing.csv \
        --gz2-csv testing_data/galaxy_zoo_2.csv \
        --model output/models/blurred_random_forest_models/150_110_1.pkl \
        --output-dir output/panstarrs_inference \
        --images

    The "--images" flag is optional and will download PANSTARRS thumbnails for each galaxy, so
    remove if visualization is unnecessary.

    IMPORTANT: Make sure the both model .pkl and its corresponding _le.pkl are in the same location!!!!!!!!!!!
"""

import argparse
import csv
import os
import re
import sys
import warnings
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import requests
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / 'modules'))

import reading_sparcfire as sparcfire


TRAINING_DATA_DIR = SCRIPT_DIR / 'randomforest_training_data'

DROP_COLS = {'P_spiral', 'P_CW', 'P_ACW', 'Unnamed: 0', 'name',
             'split', '_is_validation'}

ARRAY_COLS = {'iptSz', 'covarFit', 'chirality_votes_maj',
              'chirality_votes_alenWtd', 'sorted_agreeing_pangs',
              'sorted_agreeing_arclengths'}

COLUMNS_TO_DROP = {
    'cropRad', 'fitQualityF1', 'fitQualityPCC',
    'pa_avg_abs', 'pa_alenWtd_avg_abs', 'paErr_alenWtd_stdev_abs',
}
CHIRALITY_COLUMNS = {
    'chirality_maj', 'chirality_alenWtd',
    'chirality_wtdPangSum', 'chirality_longestArc',
}
MASK_COLUMNS = {'star_mask_used', 'noise_mask_used'}
SPACE_TO_UNDERSCORE_COLUMNS = {
    'chirality_votes_maj', 'chirality_votes_alenWtd',
    'sorted_agreeing_pangs', 'sorted_agreeing_arclengths',
}
BOOLEAN_COLUMNS = {
    'badBulgeFitFlag', 'hasDeletedCtrClus',
    'failed2revDuringMergeCheck', 'failed2revDuringSecondaryMerging',
    'failed2revInOutput', 'bar_candidate_available', 'bar_used',
}
BAD_VALUES = {'-Inf', 'Inf', 'NaN', '-inf', 'inf', 'nan'}


# STEP 1: Match Ground Truth

def parse_name_to_radec(name: str):
    """Parse SpArcFiRe galaxy name -> (RA, DEC, band) or (None, None, None)."""
    name = name.strip()
    m = re.match(r'^t(\d+_\d+)([+-])(\d+_\d+)_([ri])$', name)
    if not m:
        return None, None, None
    ra = float(m.group(1).replace('_', '.'))
    dec = float(m.group(2) + m.group(3).replace('_', '.'))
    band = m.group(4)
    return ra, dec, band


def build_gz2_lookup(gz2_path: str) -> dict:
    """Build {(round(ra,4), round(dec,4)): spiral_frac} from galaxy_zoo_2.csv."""
    lookup = {}
    with open(gz2_path, 'r') as f:
        header = f.readline().strip().split(',')
        ra_idx = header.index('ra')
        dec_idx = header.index('dec')
        spiral_idx = header.index('t04_spiral_a08_spiral_weighted_fraction')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) <= max(ra_idx, dec_idx, spiral_idx):
                continue
            try:
                ra = float(parts[ra_idx])
                dec = float(parts[dec_idx])
                key = (round(ra, 4), round(dec, 4))
                lookup[key] = parts[spiral_idx].strip()
            except (ValueError, IndexError):
                continue
    return lookup


def strip_trailing_empty(fields: list) -> list:
    """Remove trailing empty/whitespace-only fields."""
    while fields and fields[-1].strip() == '':
        fields.pop()
    return fields


def match_ground_truth(sparcfire_csv: str, gz2_csv: str, output_csv: str):
    """Step 1: Match SpArcFiRe CSV to GZ2 by RA/DEC, prepend P_CW/P_ACW.

    Also builds and returns a list of (name, ra, dec, band) for matched rows
    so we can later download images without re-parsing.
    """
    print("=" * 60)
    print("STEP 1: Matching ground truth (Galaxy Zoo 2)")
    print("=" * 60)

    lookup = build_gz2_lookup(gz2_csv)
    print(f"  GZ2 lookup: {len(lookup)} unique (RA, DEC) keys")

    matched = 0
    unmatched = 0
    parse_errors = 0
    galaxy_info = []  # (name, ra, dec, band, actual_spirality)

    with open(sparcfire_csv, 'r') as fin, open(output_csv, 'w') as fout:
        header_line = fin.readline().rstrip('\n')
        headers = [h.strip() for h in header_line.split(',')]
        headers = strip_trailing_empty(headers)
        fout.write(',P_CW,P_ACW,' + ','.join(headers) + '\n')

        for line in fin:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            fields = [f.strip() for f in line.split(',')]
            fields = strip_trailing_empty(fields)
            if not fields:
                continue

            name = fields[0]
            ra, dec, band = parse_name_to_radec(name)
            if ra is None:
                parse_errors += 1
                continue

            key = (round(ra, 4), round(dec, 4))
            spiral_frac = lookup.get(key)
            if spiral_frac is None:
                unmatched += 1
                continue

            out_fields = [str(matched), spiral_frac, '0'] + fields
            fout.write(','.join(out_fields) + '\n')
            galaxy_info.append((name, ra, dec, band, float(spiral_frac)))
            matched += 1

    print(f"  Matched: {matched}  |  Unmatched: {unmatched}  |  "
          f"Parse errors: {parse_errors}")
    print(f"  Output: {output_csv}")
    return galaxy_info


# STEP 2: Sanitize CSV

def _build_column_map(headers):
    return {name: i for i, name in enumerate(headers)}


def _transform_row(fields, col_map):
    """Apply all sanitization transformations. Returns True to keep row."""
    for val in fields:
        if val in BAD_VALUES:
            return False

    fit_state_idx = col_map.get('fit_state')
    if fit_state_idx is not None and fields[fit_state_idx] == 'input rejected':
        return False

    for i in range(4, min(23, len(fields))):
        if fields[i].strip() == '':
            return False

    for col_name in CHIRALITY_COLUMNS:
        idx = col_map.get(col_name)
        if idx is not None and idx < len(fields):
            if fields[idx] == 'Z-wise':
                fields[idx] = 'Zwise'
            elif fields[idx] == 'S-wise':
                fields[idx] = 'Swise'

    for col_name in MASK_COLUMNS:
        idx = col_map.get(col_name)
        if idx is not None and idx < len(fields):
            val = fields[idx]
            if val == 'unavailable':
                fields[idx] = 'none'
            elif val == 'aggressive-exclusive':
                fields[idx] = 'aggressiveexclusive'
            elif val == 'conservative-exclusive':
                fields[idx] = 'conservativeexclusive'

    for col_name in BOOLEAN_COLUMNS:
        idx = col_map.get(col_name)
        if idx is not None and idx < len(fields):
            if fields[idx] == 'true':
                fields[idx] = 'True'
            elif fields[idx] == 'false':
                fields[idx] = 'False'

    idx = col_map.get('top2_chirality_agreement')
    if idx is not None and idx < len(fields):
        val = fields[idx]
        if val == "'one-long'":
            fields[idx] = "'onelong'"
        elif val == "'all-short'":
            fields[idx] = "'allshort'"
        elif val == '<2 arcs':
            fields[idx] = "'<2_arcs'"

    idx = col_map.get('covarFit')
    if idx is not None and idx < len(fields):
        fields[idx] = fields[idx].replace(';', ' ')

    return True


def sanitize_csv(input_csv: str, output_csv: str) -> list[int]:
    """Step 2: Sanitize the ground-truth-matched CSV.

    Returns a list of original 0-based row indices that survived sanitization,
    so we can map back to galaxy_info from step 1.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Sanitizing CSV for RF model")
    print("=" * 60)

    kept = 0
    dropped = 0
    kept_original_indices = []

    with open(input_csv, 'r') as fin, open(output_csv, 'w') as fout:
        header_line = fin.readline().rstrip('\n')
        headers = [h.strip() for h in header_line.split(',')]
        while headers and headers[-1] == '':
            headers.pop()

        drop_indices = set()
        for col_name in COLUMNS_TO_DROP:
            if col_name in headers:
                drop_indices.add(headers.index(col_name))

        out_headers = [h for i, h in enumerate(headers) if i not in drop_indices]
        col_map = _build_column_map(headers)
        fout.write(','.join(out_headers) + '\n')

        for line in fin:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            fields = [f.strip() for f in line.split(',')]
            while fields and fields[-1] == '':
                fields.pop()
            while len(fields) < len(headers):
                fields.append('')

            original_idx = int(fields[0])

            if not _transform_row(fields, col_map):
                dropped += 1
                continue

            out_fields = [f for i, f in enumerate(fields)
                          if i < len(headers) and i not in drop_indices]
            out_fields[0] = str(kept)
            fout.write(','.join(out_fields) + '\n')
            kept_original_indices.append(original_idx)
            kept += 1

    print(f"  Kept: {kept}  |  Dropped: {dropped}")
    print(f"  Output: {output_csv}")
    return kept_original_indices


# STEP 3: RF Inference + Scatter Plot

def get_training_feature_cols(training_csv: Path) -> list[str]:
    """Read the header of the training CSV to get expected feature columns."""
    with open(training_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    header = [c.strip() for c in header]
    return [c for c in header if c not in DROP_COLS and not c.startswith('_')]


def run_inference(model_path: str, data_path: str, scatter_output: str,
                  training_csv: Path = None):
    """Step 3: Load model, run inference, generate scatter plot.

    Returns (actual, predicted, names) arrays for downstream use.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Random Forest Inference")
    print("=" * 60)

    model_path = Path(model_path)
    le_path = model_path.with_name(model_path.stem + '_le.pkl')
    data_path = Path(data_path)

    # Auto-detect training CSV
    if training_csv is None:
        model_dir = model_path.parent.name.lower()
        if 'original' in model_dir:
            training_csv = TRAINING_DATA_DIR / 'data_original_train.csv'
        else:
            training_csv = TRAINING_DATA_DIR / 'data_augmented_train.csv'

    print(f"  Model: {model_path.name}")
    print(f"  Label encoders: {le_path.name}")
    print(f"  Training reference: {training_csv.name}")

    # Read expected feature columns
    training_feature_cols = get_training_feature_cols(training_csv)
    uses_expanded = any(c.endswith('_0') for c in training_feature_cols)
    print(f"  Model expects {len(training_feature_cols)} features "
          f"({'expanded' if uses_expanded else 'un-expanded'} arrays)")

    # Load data
    if uses_expanded:
        data = sparcfire.load_inference_data(data_path)
    else:
        data = pd.read_csv(data_path)
    data.columns = [col.strip() for col in data.columns]

    actual = data['P_CW'] + data['P_ACW']
    names = data['name'].values if 'name' in data.columns else None
    print(f"  {len(data)} galaxies loaded")

    # Prepare feature matrix
    avail_feature_cols = [c for c in data.columns
                          if c not in DROP_COLS and not c.startswith('_')]
    X = data[avail_feature_cols].copy()

    # Load model + label encoders
    warnings.filterwarnings('ignore', category=UserWarning)
    rf_model = joblib.load(str(model_path))
    label_encoders = joblib.load(str(le_path))

    # Encode categoricals
    for col, le in label_encoders.items():
        if col in X.columns:
            known = set(le.classes_)
            unseen = set(X[col].dropna().astype(str).unique()) - known
            if unseen:
                fallback = le.classes_[0]
                X[col] = X[col].astype(str).apply(
                    lambda v, k=known, fb=fallback: v if v in k else fb)
            X[col] = le.transform(X[col].astype(str))

    pd.set_option('future.no_silent_downcasting', True)
    X = X.fillna(0)

    # Reindex to training column order
    missing = set(training_feature_cols) - set(X.columns)
    extra = set(X.columns) - set(training_feature_cols)
    if missing:
        print(f"  Padding {len(missing)} missing columns with 0")
    if extra:
        print(f"  Dropping {len(extra)} extra columns")

    X = X.reindex(columns=training_feature_cols, fill_value=0)
    X_np = X.values.astype(np.float32)

    assert X_np.shape[1] == rf_model.n_features_in_, (
        f"Feature mismatch: {X_np.shape[1]} vs {rf_model.n_features_in_}")

    # Predict
    print("  Running inference...")
    predicted = rf_model.predict(X_np)

    # Metrics
    r2 = r2_score(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = np.mean(np.abs(actual - predicted))
    print(f"  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actual, predicted, alpha=0.5, s=15, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual P_spiral', fontsize=13)
    ax.set_ylabel('Predicted P_spiral', fontsize=13)
    ax.set_title(
        f'PANSTARRS Overlap – {model_path.stem}.pkl\n'
        f'R²= {r2:.4f}  RMSE= {rmse:.4f}',
        fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(scatter_output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Scatter plot: {scatter_output}")

    return np.asarray(actual), predicted, names


# STEP 4: PANSTARRS Thumbnail Images (optional)

def _ps1_getimages(ra: float, dec: float, filters: str = "grizy"):
    """Query PS1 filenames service for available images at (ra, dec)."""
    from astropy.table import Table
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    table = Table.read(url, format='ascii')
    return table


def _ps1_geturl(ra: float, dec: float, size: int = 240,
                output_size: int = None, filt: str = "r",
                fmt: str = "png"):
    """Build the PS1 cutout URL for a single-filter grayscale image."""
    table = _ps1_getimages(ra, dec, filters=filt)
    if len(table) == 0:
        return None
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={fmt}")
    if output_size:
        url += f"&output_size={output_size}"
    url += f"&red={table['filename'][0]}"
    return url


def _sdss_get_image(ra: float, dec: float, size: int = 240):
    """Download an SDSS color cutout (used for Galaxy Zoo 2 imagery).

    Uses the SDSS SkyServer Image Cutout service.
    Returns a PIL Image or None on failure.
    """
    scale = 0.4  # arcsec/pixel – matches PS1 default of 0.25"/px at 240px
    url = (f"https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
           f"?ra={ra}&dec={dec}&scale={scale}&width={size}&height={size}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert('RGB')
    except Exception:
        return None


def download_and_annotate_image(name: str, ra: float, dec: float,
                                band: str, actual: float,
                                predicted: float, output_path: str,
                                size: int = 240, output_size: int = 240):
    """Download PANSTARRS and SDSS (GZ2) cutouts side-by-side, annotated
    with spirality info.  The PNG filename is the predicted spirality value.
    """
    try:
        url = _ps1_geturl(ra, dec, size=size, output_size=output_size,
                          filt=band, fmt="png")
        if url is None:
            print(f"    No PS1 image for {name} (band={band})")
            return False

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        ps1_img = Image.open(BytesIO(resp.content)).convert('RGB')
    except Exception as e:
        print(f"    PS1 download failed for {name}: {e}")
        return False

    # ── Download SDSS / Galaxy Zoo 2 image ──
    sdss_img = _sdss_get_image(ra, dec, size=output_size)

    # ── Build side-by-side figure ──
    has_sdss = sdss_img is not None
    ncols = 2 if has_sdss else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # SDSS (GZ2) panel – left
    if has_sdss:
        axes[0].imshow(np.array(sdss_img), origin='upper')
        axes[0].set_title('Galaxy Zoo 2 (SDSS)', fontsize=9, fontweight='bold')
        axes[0].axis('off')

    # PANSTARRS panel – right (or only)
    ps1_ax = axes[1] if has_sdss else axes[0]
    ps1_ax.imshow(np.array(ps1_img), origin='upper')
    ps1_ax.set_title(f'PANSTARRS ({band}-band)', fontsize=9, fontweight='bold')
    ps1_ax.axis('off')

    # Annotation text
    fig.suptitle(name, fontsize=10, fontweight='bold')
    text = (f"ACTUAL SPIRALITY: {actual:.4f}\n"
            f"PREDICTED SPIRALITY: {predicted:.4f}")
    fig.text(0.5, 0.02, text, ha='center', va='bottom', fontsize=8,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       alpha=0.9, edgecolor='gray'))

    plt.tight_layout(rect=[0, 0.1, 1, 0.93])
    plt.savefig(output_path, dpi=120, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    return True


def generate_images(galaxy_info, kept_indices, actual_arr, predicted_arr,
                    output_dir: str):
    """Step 4: Download PANSTARRS thumbnails for each inferred galaxy."""
    print("\n" + "=" * 60)
    print("STEP 4: Downloading PANSTARRS thumbnails")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    total = len(kept_indices)
    success = 0
    failed = 0

    for i, orig_idx in enumerate(kept_indices):
        name, ra, dec, band, _ = galaxy_info[orig_idx]
        actual_val = float(actual_arr[i])
        predicted_val = float(predicted_arr[i])

        # File name based on predicted spirality classification
        filename = f"{predicted_val:.4f}.png"
        out_path = os.path.join(output_dir, filename)

        ok = download_and_annotate_image(
            name, ra, dec, band, actual_val, predicted_val, out_path)

        if ok:
            success += 1
        else:
            failed += 1

        # Progress
        if (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"  Progress: {i + 1}/{total}  "
                  f"(success={success}, failed={failed})")

    print(f"  Done: {success} images saved, {failed} failed")
    print(f"  Output: {output_dir}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description='End-to-end PANSTARRS inference pipeline: '
                    'match GZ2 ground truth → sanitize → RF inference → '
                    'optional annotated thumbnails.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--sparcfire-csv', required=True,
                        help='Raw SpArcFiRe output CSV '
                             '(e.g. SpArcFiRe_partial_PANSTARRS_overlap.csv)')
    parser.add_argument('--gz2-csv', required=True,
                        help='Path to galaxy_zoo_2.csv')
    parser.add_argument('--model', required=True,
                        help='Path to RF model .pkl '
                             '(label encoder _le.pkl auto-derived)')
    parser.add_argument('--output-dir', default='output/panstarrs_inference',
                        help='Directory for all outputs '
                             '(default: output/panstarrs_inference)')
    parser.add_argument('--training-csv', default=None,
                        help='Training CSV for feature column order '
                             '(auto-detected from model path if omitted)')
    parser.add_argument('--images', action='store_true',
                        help='Download PANSTARRS thumbnails annotated with '
                             'actual/predicted spirality')
    parser.add_argument('--image-size', type=int, default=240,
                        help='PS1 cutout size in pixels '
                             '(0.25 arcsec/pixel, default: 240)')
    args = parser.parse_args()

    # Validate inputs
    for path, label in [(args.sparcfire_csv, 'SpArcFiRe CSV'),
                        (args.gz2_csv, 'Galaxy Zoo 2 CSV'),
                        (args.model, 'Model .pkl')]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    le_path = Path(args.model).with_name(
        Path(args.model).stem + '_le.pkl')
    if not le_path.is_file():
        print(f"ERROR: Label encoder not found: {le_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Define intermediate file paths
    matched_csv = os.path.join(args.output_dir,
                               'SpArcFiRe_with_ground_truth.csv')
    cleaned_csv = os.path.join(args.output_dir,
                               'SpArcFiRe_cleaned.csv')
    scatter_png = os.path.join(args.output_dir,
                               'panstarrs_scatter.png')

    # ── Step 1 ──
    galaxy_info = match_ground_truth(
        args.sparcfire_csv, args.gz2_csv, matched_csv)

    if not galaxy_info:
        print("ERROR: No galaxies matched. Check inputs.", file=sys.stderr)
        sys.exit(1)

    # ── Step 2 ──
    kept_indices = sanitize_csv(matched_csv, cleaned_csv)

    if not kept_indices:
        print("ERROR: All rows dropped during sanitization.", file=sys.stderr)
        sys.exit(1)

    # ── Step 3 ──
    training_csv = Path(args.training_csv) if args.training_csv else None
    actual, predicted, _ = run_inference(
        args.model, cleaned_csv, scatter_png, training_csv)

    # ── Step 4 (optional) ──
    if args.images:
        images_dir = os.path.join(args.output_dir,
                                  'PANSTARRS-inference-images')
        generate_images(galaxy_info, kept_indices, actual, predicted,
                        images_dir)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Scatter plot:  {scatter_png}")
    print(f"  Cleaned CSV:   {cleaned_csv}")
    if args.images:
        print(f"  Images:        {os.path.join(args.output_dir, 'PANSTARRS-inference-images')}")
    # Clean up intermediate file
    if os.path.isfile(matched_csv):
        os.remove(matched_csv)
        print(f"  (Removed intermediate: {os.path.basename(matched_csv)})")


if __name__ == '__main__':
    main()
