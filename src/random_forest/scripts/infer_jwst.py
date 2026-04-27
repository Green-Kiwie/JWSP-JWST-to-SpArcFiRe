#!/usr/bin/env python3
"""
Run inference on a raw SpArcFiRe CSV (no ground truth) using a saved
Random Forest model and save the predicted spirality scores.

Applies the same sanitization and data preparation pipeline as
sanitize_sparcfire_csv.py + infer_and_plot.py, but without requiring
P_CW/P_ACW ground truth columns.

Usage:
    python scripts/infer_jwst.py \\
        --data testing_data/combined.csv \\
        --model inference-starter-kit/150_110_1.pkl \\
        --output output/jwst_predictions.csv
"""

import argparse
import csv
import io
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'modules'))

import numpy as np
import pandas as pd
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA_DIR = SCRIPT_DIR / 'randomforest_training_data'

DROP_COLS = {'P_spiral', 'P_CW', 'P_ACW', 'Unnamed: 0', 'name',
             'split', '_is_validation'}

# Columns present in raw SpArcFiRe output but NOT in training data
COLUMNS_TO_DROP = {
    'cropRad', 'fitQualityF1', 'fitQualityPCC',
    'pa_avg_abs', 'pa_alenWtd_avg_abs', 'paErr_alenWtd_stdev_abs',
}

CHIRALITY_COLUMNS = {
    'chirality_maj', 'chirality_alenWtd',
    'chirality_wtdPangSum', 'chirality_longestArc',
}
MASK_COLUMNS = {'star_mask_used', 'noise_mask_used'}
BOOLEAN_COLUMNS = {
    'badBulgeFitFlag', 'hasDeletedCtrClus',
    'failed2revDuringMergeCheck', 'failed2revDuringSecondaryMerging',
    'failed2revInOutput', 'bar_candidate_available', 'bar_used',
}
SPACE_TO_UNDERSCORE_COLUMNS = {
    'chirality_votes_maj', 'chirality_votes_alenWtd',
    'sorted_agreeing_pangs', 'sorted_agreeing_arclengths',
}
BAD_VALUES = {'-Inf', 'Inf', 'NaN', '-inf', 'inf', 'nan'}

# Matches complex numbers like '0+5.63e-09i' or '1.2-3.4i'
_COMPLEX_RE = re.compile(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)[+-]\d*\.?\d+(?:[eE][+-]?\d+)?i$')

_PREFILTER_MIN_DISK_AXIS = 5.0        # diskMinAxsLen below this → tiny/absent disk
_PREFILTER_MIN_TOTAL_ARC = 50.0       # totalArcLength below this → no spiral arms detected
_PREFILTER_MIN_LIK_OF_CTR = -10.0     # likOfCtr below this → poor center fit

# Confidence thresholds for flagging uncertain predictions
_HIGH_VARIANCE_QUANTILE = 0.90        # predictions above this variance quantile are flagged


def _strip_complex(val: str) -> str:
    """If val looks like a complex number, return just its real part as a string."""
    m = _COMPLEX_RE.match(val)
    return m.group(1) if m else val


def _flag_non_galaxies(df: pd.DataFrame) -> np.ndarray:
    """Return a boolean mask where True = likely non-galaxy (should get P_spiral=0).

    An input is flagged when ALL of the following are simultaneously true:
      - diskMinAxsLen is very small (no extended disk detected)
      - totalArcLength is near zero (no spiral arms found)
      - likOfCtr is very negative (SpArcFiRe couldn't fit a convincing center)

    This catches stars, noise, and artifacts that produce degenerate SpArcFiRe
    outputs while preserving real galaxies (which satisfy at least one criterion).
    """
    n = len(df)
    mask = np.zeros(n, dtype=bool)

    disk_col = 'diskMinAxsLen'
    arc_col = 'totalArcLength'
    lik_col = 'likOfCtr'

    has_disk = disk_col in df.columns
    has_arc = arc_col in df.columns
    has_lik = lik_col in df.columns

    if has_disk and has_arc and has_lik:
        disk_vals = pd.to_numeric(df[disk_col], errors='coerce').fillna(0.0)
        arc_vals = pd.to_numeric(df[arc_col], errors='coerce').fillna(0.0)
        lik_vals = pd.to_numeric(df[lik_col], errors='coerce').fillna(-999.0)

        mask = (
            (disk_vals < _PREFILTER_MIN_DISK_AXIS) &
            (arc_vals < _PREFILTER_MIN_TOTAL_ARC) &
            (lik_vals < _PREFILTER_MIN_LIK_OF_CTR)
        ).values

    return mask


def sanitize_to_buffer(input_path: Path) -> io.StringIO:
    """Apply the same transformations as sanitize_sparcfire_csv.py and
    return the result as an in-memory CSV buffer. Galaxy names are preserved
    in the 'name' column."""
    buf = io.StringIO()
    kept = 0
    dropped = 0

    with open(input_path, 'r') as fin:
        header_line = fin.readline().rstrip('\n')
        headers = [h.strip() for h in header_line.split(',')]
        while headers and headers[-1] == '':
            headers.pop()

        drop_indices = {headers.index(c) for c in COLUMNS_TO_DROP if c in headers}
        col_map = {name: i for i, name in enumerate(headers)}

        out_headers = [h for i, h in enumerate(headers) if i not in drop_indices]
        buf.write(','.join(out_headers) + '\n')

        for line in fin:
            line = line.rstrip('\n')
            if not line.strip():
                continue

            fields = [f.strip() for f in line.split(',')]
            while fields and fields[-1] == '':
                fields.pop()
            while len(fields) < len(headers):
                fields.append('')

            # Drop rows with bad values
            if any(v in BAD_VALUES for v in fields):
                dropped += 1
                continue

            # Drop rows with fit_state == 'input rejected'
            fs_idx = col_map.get('fit_state')
            if fs_idx is not None and fields[fs_idx] == 'input rejected':
                dropped += 1
                continue

            # Drop rows with blank key fields (positional check)
            if any(fields[i].strip() == '' for i in range(4, min(23, len(fields)))):
                dropped += 1
                continue

            # Chirality: Z-wise -> Zwise, S-wise -> Swise
            for col_name in CHIRALITY_COLUMNS:
                idx = col_map.get(col_name)
                if idx is not None and idx < len(fields):
                    if fields[idx] == 'Z-wise':
                        fields[idx] = 'Zwise'
                    elif fields[idx] == 'S-wise':
                        fields[idx] = 'Swise'

            # Mask columns
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

            # Booleans: true -> True, false -> False
            for col_name in BOOLEAN_COLUMNS:
                idx = col_map.get(col_name)
                if idx is not None and idx < len(fields):
                    if fields[idx] == 'true':
                        fields[idx] = 'True'
                    elif fields[idx] == 'false':
                        fields[idx] = 'False'

            # top2_chirality_agreement fixes
            idx = col_map.get('top2_chirality_agreement')
            if idx is not None and idx < len(fields):
                val = fields[idx]
                if val == "'one-long'":
                    fields[idx] = "'onelong'"
                elif val == "'all-short'":
                    fields[idx] = "'allshort'"
                elif val == '<2 arcs':
                    fields[idx] = "'<2_arcs'"

            # Space -> underscore in array columns
            for col_name in SPACE_TO_UNDERSCORE_COLUMNS:
                idx = col_map.get(col_name)
                if idx is not None and idx < len(fields):
                    fields[idx] = fields[idx].replace(' ', '_')

            # covarFit: remove semicolons
            idx = col_map.get('covarFit')
            if idx is not None and idx < len(fields):
                fields[idx] = fields[idx].replace(';', ' ')

            out_fields = [f for i, f in enumerate(fields)
                          if i < len(headers) and i not in drop_indices]
            # Clean up any complex-number artifacts (e.g. '0+5.6e-09i' -> '0')
            out_fields = [_strip_complex(f) for f in out_fields]
            buf.write(','.join(out_fields) + '\n')
            kept += 1

    print(f"  Sanitization: {kept} rows kept, {dropped} rows dropped")
    buf.seek(0)
    return buf, kept


def _parse_array_str(s):
    """Parse a SpArcFiRe array string like '1_2_3' or '1 2 3' into a list of floats."""
    if not isinstance(s, str):
        return [float(s)] if s is not None and s == s else []
    s = re.sub(r'[\[\]]', '', s).strip()
    if not s:
        return []
    parts = re.split(r'[_\s]+', s)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            try:
                out.append(float(p))
            except ValueError:
                out.append(np.nan)
    return out


def _load_and_expand(buf) -> pd.DataFrame:
    """Load CSV from buffer and expand array columns into numbered sub-columns.
    Mirrors augment_and_train_rf.expand_list_columns but without logging."""
    df = pd.read_csv(buf, skipinitialspace=True, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    ARRAY_COLS = ['iptSz', 'covarFit', 'chirality_votes_maj',
                  'chirality_votes_alenWtd', 'sorted_agreeing_pangs',
                  'sorted_agreeing_arclengths']

    for col in ARRAY_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(_parse_array_str)
        if col == 'covarFit':
            df[col] = df[col].apply(
                lambda x: list(x)[:7] + [None] * (7 - len(x))
                if isinstance(x, (list, np.ndarray)) else [None] * 7
            )
            expanded = pd.DataFrame(df[col].tolist(),
                                    columns=[f'{col}_{i}' for i in range(7)],
                                    index=df.index)
        else:
            expanded = df[col].apply(pd.Series)
            expanded.columns = [f'{col}_{i}' for i in expanded.columns]
            expanded.index = df.index
        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    return df


def get_training_feature_cols(training_csv: Path) -> list:
    with open(training_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    header = [c.strip() for c in header]
    return [c for c in header if c not in DROP_COLS and not c.startswith('_')]


def main():
    parser = argparse.ArgumentParser(
        description='Run RF inference on raw JWST SpArcFiRe CSV (no ground truth).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data', required=True,
                        help='Path to raw SpArcFiRe CSV (e.g. combined.csv)')
    parser.add_argument('--model', required=True,
                        help='Path to the Random Forest model .pkl file')
    parser.add_argument('--output', default='output/jwst_predictions.csv',
                        help='Output CSV path (default: output/jwst_predictions.csv)')
    parser.add_argument('--training-csv', default=None,
                        help='Training CSV used to determine feature column order '
                             '(auto-detected from model path if omitted)')
    args = parser.parse_args()

    model_path = Path(args.model)
    le_path = model_path.with_name(model_path.stem + '_le.pkl')
    data_path = Path(args.data)
    output_path = Path(args.output)
    if output_path.suffix.lower() != '.csv':
        output_path = output_path.with_suffix('.csv')

    for path, label in [(model_path, "model"), (le_path, "label encoder"), (data_path, "data file")]:
        try:
            exists = path.exists()
        except PermissionError:
            print(f"ERROR: permission denied accessing {label}: {path}", file=sys.stderr)
            sys.exit(1)
        if not exists:
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Auto-detect training CSV ---
    if args.training_csv:
        training_csv = Path(args.training_csv)
    else:
        model_dir = model_path.parent.name.lower()
        if 'original' in model_dir:
            training_csv = TRAINING_DATA_DIR / 'data_original_train.csv'
        else:
            training_csv = TRAINING_DATA_DIR / 'data_augmented_train.csv'

    print(f"Reading training feature columns from {training_csv.name}...")
    training_feature_cols = get_training_feature_cols(training_csv)
    print(f"  Model expects {len(training_feature_cols)} feature columns")

    uses_expanded = any(c.endswith('_0') for c in training_feature_cols)
    print(f"  Array columns {'expanded' if uses_expanded else 'un-expanded'}")

    # --- Sanitize the raw CSV ---
    print(f"Sanitizing {data_path.name}...")
    sanitized_buf, n_rows = sanitize_to_buffer(data_path)

    # --- Load sanitized data ---
    print(f"Loading sanitized data ({n_rows} rows)...")
    if uses_expanded:
        data = _load_and_expand(sanitized_buf)
    else:
        data = pd.read_csv(sanitized_buf, skipinitialspace=True, low_memory=False)
    data.columns = [col.strip() for col in data.columns]

    galaxy_names = data['name'].astype(str) if 'name' in data.columns else pd.Series(range(len(data)))
    print(f"  {len(data)} galaxies loaded")

    # --- Pre-filter: flag likely non-galaxies ---
    non_galaxy_mask = _flag_non_galaxies(data)
    n_flagged = int(np.sum(non_galaxy_mask))
    print(f"  Pre-filter: {n_flagged} rows flagged as non-galaxy (will get P_spiral=0.0)")

    # --- Prepare feature matrix ---
    avail_feature_cols = [c for c in data.columns
                          if c not in DROP_COLS and not c.startswith('_')]
    X = data[avail_feature_cols].copy()
    print(f"  {len(avail_feature_cols)} feature columns available")

    # --- Load model and label encoders ---
    print(f"Loading model {model_path.name}...")
    warnings.filterwarnings('ignore', category=UserWarning)
    rf_model = joblib.load(str(model_path))
    label_encoders = joblib.load(str(le_path))

    # --- Encode categoricals ---
    for col, le in label_encoders.items():
        if col in X.columns:
            known = set(le.classes_)
            unseen = set(X[col].dropna().astype(str).unique()) - known
            if unseen:
                print(f"  WARNING: unseen values in '{col}': {unseen}")
                fallback = le.classes_[0]
                X[col] = X[col].astype(str).apply(
                    lambda v, k=known, fb=fallback: v if v in k else fb)
            X[col] = le.transform(X[col].astype(str))

    # --- Fill NaN and reindex ---
    pd.set_option('future.no_silent_downcasting', True)
    X = X.fillna(0)

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

    # --- Predict with per-tree variance ---
    print("Running inference...")
    predicted = np.zeros(len(X_np), dtype=np.float64)
    tree_std = np.zeros(len(X_np), dtype=np.float64)

    galaxy_indices = ~non_galaxy_mask
    if np.any(galaxy_indices):
        X_gal = X_np[galaxy_indices]
        predicted[galaxy_indices] = rf_model.predict(X_gal)

        # Compute per-tree predictions to get variance (confidence metric)
        tree_preds = np.array([t.predict(X_gal) for t in rf_model.estimators_])
        tree_std[galaxy_indices] = tree_preds.std(axis=0)
    # Non-galaxy rows remain at 0.0

    # Flag low-confidence predictions: high tree variance + suspect P_spiral range
    gal_stds = tree_std[galaxy_indices]
    if len(gal_stds) > 0:
        var_threshold = np.quantile(gal_stds, _HIGH_VARIANCE_QUANTILE)
    else:
        var_threshold = 0.0
    low_confidence = (tree_std > var_threshold) & (predicted >= 0.5) & (predicted < 0.75)

    print(f"\nPrediction summary:")
    print(f"  Total inputs:   {len(predicted)}")
    print(f"  Non-galaxies:   {n_flagged} (pre-filtered, P_spiral=0.0)")
    print(f"  Galaxies:       {len(predicted) - n_flagged}")
    if np.any(galaxy_indices):
        galaxy_preds = predicted[galaxy_indices]
        print(f"  P_spiral range: [{galaxy_preds.min():.4f}, {galaxy_preds.max():.4f}]")
        print(f"  Mean P_spiral:  {galaxy_preds.mean():.4f} (galaxies only)")
        print(f"  Tree std range: [{gal_stds.min():.4f}, {gal_stds.max():.4f}]")
        print(f"  Variance threshold (p{int(_HIGH_VARIANCE_QUANTILE*100)}): {var_threshold:.4f}")
    n_low_conf = int(low_confidence.sum())
    print(f"  Low confidence: {n_low_conf} ({100*n_low_conf/len(predicted):.1f}%)")

    # --- Save results ---
    results = pd.DataFrame({
        'name': galaxy_names.values,
        'predicted_P_spiral': predicted,
        'tree_std': tree_std,
        'flagged_non_galaxy': non_galaxy_mask,
        'low_confidence': low_confidence,
    })
    results.to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} predictions to {output_path}")


if __name__ == '__main__':
    main()
