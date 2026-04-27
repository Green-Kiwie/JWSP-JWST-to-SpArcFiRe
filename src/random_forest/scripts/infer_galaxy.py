#!/usr/bin/env python3
"""Classify a directory tree of galaxy thumbnails (.png or .fits) using the trained
Random Forest model.

Recursively reads every image under the input directory, runs the classifier,
and copies each file into a ``galaxy/`` or ``non-galaxy/`` subfolder under the
output directory while preserving the relative directory layout.

Automatically selects the correct model based on the input file type:
  - .fits files → uses galaxy_classifier_fits.joblib
  - .png  files → uses galaxy_classifier_png.joblib

Annotated PNG exports are cropped to the morphology panel before features are
computed, so text labels on the canvas do not leak into the separator.

Usage:
    python scripts/infer_galaxy.py <input_dir> [--output <output_dir>] [--threshold <0-1>]

Examples:
    # PNGs: output defaults to <input_dir>/classified/ and scans recursively
    python scripts/infer_galaxy.py inference/RUN-1/1-galaxies/

    # FITS: custom output dir and threshold
    python scripts/infer_galaxy.py /path/to/fits_thumbs --output results/ --threshold 0.6
"""

import argparse
import os
import shutil
import sys

import numpy as np
import joblib
from astropy.io import fits

# Ensure scripts/ and modules/ are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'modules'))

FITS_MODEL_PATH = os.path.join(ROOT_DIR, 'output', 'models', 'galaxy_classifier', 'galaxy_classifier_fits.joblib')
PNG_MODEL_PATH = os.path.join(ROOT_DIR, 'output', 'models', 'galaxy_classifier', 'galaxy_classifier_png.joblib')
LEGACY_MODEL_PATH = os.path.join(ROOT_DIR, 'output', 'models', 'galaxy_classifier', 'galaxy_classifier.joblib')
OLD_FITS_MODEL_PATH = os.path.join(ROOT_DIR, 'output', 'galaxy_classifier_fits.joblib')
OLD_PNG_MODEL_PATH = os.path.join(ROOT_DIR, 'output', 'galaxy_classifier_png.joblib')
OLD_LEGACY_MODEL_PATH = os.path.join(ROOT_DIR, 'output', 'galaxy_classifier.joblib')

# Feature extraction is imported from classify_galaxy to avoid duplication
from classify_galaxy import extract_features, FEATURE_COLS 
from galaxy_separator_png_utils import load_png_morphology_array


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model_cache = {}


def _get_model(file_type: str):
    """Load the appropriate model for the given file type."""
    if file_type not in _model_cache:
        if file_type == 'fits':
            paths = [
                FITS_MODEL_PATH,
                LEGACY_MODEL_PATH,
                OLD_FITS_MODEL_PATH,
                OLD_LEGACY_MODEL_PATH,
            ]
        else:
            paths = [PNG_MODEL_PATH, OLD_PNG_MODEL_PATH]

        model_data = None
        for p in paths:
            if os.path.exists(p):
                model_data = joblib.load(p)
                print(f"Using model: {p}")
                break

        if model_data is None:
            if file_type == 'png':
                print(
                    "Error: PNG model not found. Train it first:\n"
                    "  python scripts/train_galaxy_classifier.py "
                    "--png-training-root randomforest_training_data/galaxy_separator_RF",
                    file=sys.stderr,
                )
            else:
                print(
                    "Error: FITS model not found. Train it first:\n"
                    "  python scripts/train_galaxy_classifier.py",
                    file=sys.stderr,
                )
            sys.exit(1)

        _model_cache[file_type] = model_data['model']

    return _model_cache[file_type]


def classify(data: np.ndarray, file_type: str, threshold: float = 0.5):
    """Classify a 2D array as galaxy or non-galaxy.

    Returns (is_galaxy: bool, probability: float).
    """
    clf = _get_model(file_type)
    feats = extract_features(data)
    feature_vec = np.array([[feats[col] for col in FEATURE_COLS]])
    prob = clf.predict_proba(feature_vec)[0, 1]  # P(galaxy)
    return prob >= threshold, float(prob)


# ---------------------------------------------------------------------------
# Image loaders
# ---------------------------------------------------------------------------

def load_fits(path: str) -> np.ndarray:
    """Load a FITS thumbnail as a 2D float64 array."""
    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None:
            # Try the next extension
            for ext in hdul[1:]:
                if ext.data is not None:
                    data = ext.data
                    break
        if data is None:
            raise ValueError(f"No image data found in {path}")
        data = data.astype(np.float64)
    return np.nan_to_num(data, nan=0.0)


def load_png(path: str) -> np.ndarray:
    """Load a PNG image as a morphology-only 2D float64 array."""
    return load_png_morphology_array(path)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def discover_images(input_dir: str) -> tuple:
    """Find all .png or .fits files under *input_dir* recursively.

    Returns (file_list, file_type) where file_type is 'png' or 'fits'.
    Raises if the directory contains no supported files.
    """
    pngs = []
    fits_files = []

    for root, _, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        for fname in files:
            rel_path = fname if rel_root == '.' else os.path.join(rel_root, fname)
            lower_name = fname.lower()
            if lower_name.endswith('.png'):
                pngs.append(rel_path)
            elif lower_name.endswith(('.fits', '.fit', '.fits.gz')):
                fits_files.append(rel_path)

    pngs.sort()
    fits_files.sort()

    if pngs and fits_files:
        # Use whichever has more files; warn about the other
        if len(pngs) >= len(fits_files):
            print(f"Warning: ignoring {len(fits_files)} FITS file(s) in a directory tree with {len(pngs)} PNGs")
            return pngs, 'png'
        else:
            print(f"Warning: ignoring {len(pngs)} PNG file(s) in a directory tree with {len(fits_files)} FITS files")
            return fits_files, 'fits'
    elif pngs:
        return pngs, 'png'
    elif fits_files:
        return fits_files, 'fits'
    else:
        raise FileNotFoundError(f"No .png or .fits files found under {input_dir}")


def run_inference(input_dir: str, output_dir: str, threshold: float = 0.5):
    """Classify all images recursively and sort them into output folders."""
    files, file_type = discover_images(input_dir)
    loader = load_fits if file_type == 'fits' else load_png

    galaxy_dir = os.path.join(output_dir, 'galaxy')
    non_galaxy_dir = os.path.join(output_dir, 'non-galaxy')
    os.makedirs(galaxy_dir, exist_ok=True)
    os.makedirs(non_galaxy_dir, exist_ok=True)

    n_galaxy = 0
    n_non_galaxy = 0
    n_errors = 0

    print(f"Input:     {input_dir}  ({len(files)} {file_type} files)")
    print(f"Output:    {output_dir}")
    print(f"Threshold: {threshold}")
    print()

    for i, fname in enumerate(files, 1):
        src = os.path.join(input_dir, fname)
        try:
            data = loader(src)
            is_galaxy, prob = classify(data, file_type, threshold=threshold)
        except Exception as exc:
            print(f"  [{i}/{len(files)}] ERROR  {fname}: {exc}")
            n_errors += 1
            continue

        dst_root = galaxy_dir if is_galaxy else non_galaxy_dir
        dst = os.path.join(dst_root, fname)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if is_galaxy:
            n_galaxy += 1
            label = 'galaxy'
        else:
            n_non_galaxy += 1
            label = 'non-galaxy'

        shutil.copy2(src, dst)
        print(f"  [{i}/{len(files)}] {label:11s}  P={prob:.4f}  {fname}")

    print()
    print("=" * 60)
    print(f"  Galaxy:     {n_galaxy}")
    print(f"  Non-galaxy: {n_non_galaxy}")
    if n_errors:
        print(f"  Errors:     {n_errors}")
    print("=" * 60)
    print(f"\nResults in: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify galaxy thumbnails using the trained Random Forest model."
    )
    parser.add_argument(
        'input_dir',
        help="Directory containing .png or .fits thumbnails to classify."
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help="Output directory (default: <input_dir>/classified/)."
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help="Probability threshold for galaxy classification (default: 0.5)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'classified')
    output_dir = os.path.abspath(output_dir)

    run_inference(input_dir, output_dir, threshold=args.threshold)


if __name__ == '__main__':
    main()
