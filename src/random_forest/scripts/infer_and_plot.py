#!/usr/bin/env python3
"""
Run inference on a cleaned SpArcFiRe CSV using a saved Random Forest model
and generate an Actual vs Predicted P_spiral scatter plot.

Follows the same data preparation pipeline as train_grid_search.py:
load data via reading_sparcfire, drop non-feature columns, encode
categoricals with the saved label encoders, fill NaN, then predict.

Usage:
    python scripts/infer_and_plot.py \\
        --model output/blurred_random_forest_model_output/150_110_1.pkl \\
        --le output/blurred_random_forest_model_output/150_110_1_le.pkl \\
        --data testing_data/SpArcFiRe_cleaned.csv \\
        --output testing_data/panstarrs_scatter.png
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

# Add modules/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'modules'))

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

import reading_sparcfire as sparcfire

# Columns to drop before inference (same as train_grid_search.py)
DROP_COLS = {'P_spiral', 'P_CW', 'P_ACW', 'Unnamed: 0', 'name',
             'split', '_is_validation'}

SCRIPT_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA_DIR = SCRIPT_DIR / 'randomforest_training_data'

# Array columns that reading_sparcfire expands into sub-columns
ARRAY_COLS = {'iptSz', 'covarFit', 'chirality_votes_maj',
              'chirality_votes_alenWtd', 'sorted_agreeing_pangs',
              'sorted_agreeing_arclengths'}


def get_training_feature_cols(training_csv: Path) -> list[str]:
    """Read just the header of the augmented training CSV and return
    the feature column names (after dropping non-feature columns),
    preserving the exact order the model was trained on."""
    with open(training_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    header = [c.strip() for c in header]
    return [c for c in header if c not in DROP_COLS and not c.startswith('_')]


def main():
    parser = argparse.ArgumentParser(
        description='Run RF inference on a cleaned SpArcFiRe CSV and plot '
                    'Actual vs Predicted P_spiral.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--model', required=True,
                        help='Path to the model .pkl file')
    parser.add_argument('--le', required=True,
                        help='Path to the label encoder .pkl file')
    parser.add_argument('--data', required=True,
                        help='Path to cleaned SpArcFiRe CSV '
                             '(output of sanitize_sparcfire_csv.py)')
    parser.add_argument('--output', default='scatter_actual_vs_predicted.png',
                        help='Output path for scatter plot PNG '
                             '(default: scatter_actual_vs_predicted.png)')
    parser.add_argument('--training-csv', default=None,
                        help='Path to training CSV (used only to read the '
                             'header for feature column order). Auto-detected '
                             'from model path if omitted.')
    args = parser.parse_args()

    model_path = Path(args.model)
    le_path = Path(args.le)
    data_path = Path(args.data)

    # --- Auto-detect training CSV if not provided ---
    if args.training_csv:
        training_csv = Path(args.training_csv)
    else:
        # Heuristic: if model path contains 'original', use original data
        model_dir = model_path.parent.name.lower()
        if 'original' in model_dir:
            training_csv = TRAINING_DATA_DIR / 'data_original_train.csv'
        else:
            training_csv = TRAINING_DATA_DIR / 'data_augmented_train.csv'

    # --- Read expected feature column order from training data header ---
    print(f"Reading training feature columns from {training_csv.name}...")
    training_feature_cols = get_training_feature_cols(training_csv)
    print(f"  Model expects {len(training_feature_cols)} feature columns")

    # --- Detect whether training used expanded array columns ---
    uses_expanded = any(c.endswith('_0') for c in training_feature_cols)
    print(f"  Array columns {'expanded' if uses_expanded else 'un-expanded'}")

    # --- Load data ---
    print(f"Loading data from {data_path}...")
    if uses_expanded:
        data = sparcfire.load_inference_data(data_path)
    else:
        data = pd.read_csv(data_path)
    data.columns = [col.strip() for col in data.columns]

    # Compute actual P_spiral
    actual = data['P_CW'] + data['P_ACW']
    print(f"  {len(data)} galaxies loaded")
    print(f"  P_spiral (actual) range: [{actual.min():.4f}, {actual.max():.4f}]")

    # --- Prepare feature matrix ---
    # Start with all non-drop columns in the data
    avail_feature_cols = [c for c in data.columns
                          if c not in DROP_COLS and not c.startswith('_')]
    X = data[avail_feature_cols].copy()
    print(f"  {len(avail_feature_cols)} feature columns from data")

    # --- Load model and label encoders ---
    print(f"Loading model {model_path.name}...")
    warnings.filterwarnings('ignore', category=UserWarning)
    rf_model = joblib.load(str(model_path))
    label_encoders = joblib.load(str(le_path))

    # --- Encode categorical columns with saved label encoders ---
    for col, le in label_encoders.items():
        if col in X.columns:
            known = set(le.classes_)
            unseen = set(X[col].dropna().astype(str).unique()) - known
            if unseen:
                print(f"  WARNING: unseen values in '{col}': {unseen}")
                fallback = le.classes_[0]
                X[col] = X[col].astype(str).apply(
                    lambda v: v if v in known else fallback)
            X[col] = le.transform(X[col].astype(str))

    # --- Fill NaN and reindex to match training column order ---
    pd.set_option('future.no_silent_downcasting', True)
    X = X.fillna(0)

    # Reindex to exact training feature columns, filling missing with 0
    missing = set(training_feature_cols) - set(X.columns)
    extra = set(X.columns) - set(training_feature_cols)
    if missing:
        print(f"  Padding {len(missing)} missing columns with 0: "
              f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        print(f"  Dropping {len(extra)} extra columns not in training: "
              f"{sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")

    X = X.reindex(columns=training_feature_cols, fill_value=0)
    X_np = X.values.astype(np.float32)

    assert X_np.shape[1] == rf_model.n_features_in_, (
        f"Feature count mismatch: reindexed {X_np.shape[1]} vs "
        f"model {rf_model.n_features_in_}"
    )

    # --- Predict ---
    print("Running inference...")
    predicted = rf_model.predict(X_np)

    # --- Compute metrics ---
    r2 = r2_score(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = np.mean(np.abs(actual - predicted))
    print(f"\nMetrics:")
    print(f"  R²   = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

    # --- Scatter plot (matching train_grid_search.py style) ---
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(actual, predicted, alpha=0.5, s=15, edgecolors='none')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel('Actual P_spiral', fontsize=13)
    ax.set_ylabel('Predicted P_spiral', fontsize=13)
    ax.set_title(
        f'PANSTARRS Overlap\n'
        f'Model: {model_path.stem}\n'
        f'R²= {r2:.4f}  RMSE= {rmse:.4f}',
        fontsize=12
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nScatter plot saved to {args.output}")


if __name__ == '__main__':
    main()
