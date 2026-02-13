#!/usr/bin/env python3
"""
Train Random Forest Grid Search

This script loads pre-augmented data ONCE, then trains 560 models in parallel.
Data is loaded into memory before spawning workers, so each worker only trains.

Usage:
    python train_grid_search.py
    python train_grid_search.py --train-csv path/to/train.csv --val-csv path/to/val.csv
    python train_grid_search.py --dry-run  # Test with 3 models
"""

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_TRAIN_CSV = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/randomforest_training_data/data_augmented_train.csv"
DEFAULT_VAL_CSV = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/randomforest_training_data/data_augmented_val.csv"
DEFAULT_MODEL_OUTPUT = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/random_forest_model_output"
DEFAULT_RESULTS_OUTPUT = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/random_forest_output"

# Hyperparameter grid
TREES_GRID = [10, 30, 50, 70, 90, 110, 130, 150]
FEATURES_GRID = [10, 30, 50, 70, 90, 110, 130]
BINS_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Dry-run settings
DRY_RUN_MODELS = [(10, 10, 1), (10, 10, 2), (10, 10, 3)]

# Global variable for shared data (set by initializer)
SHARED_DATA = {}


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file: str = "train_grid_search.log") -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("train_grid_search")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare_data(train_csv: str, val_csv: str, logger: logging.Logger) -> Dict:
    """
    Load and prepare all data for training.
    Returns a dict with all prepared numpy arrays and metadata.
    """
    logger.info(f"Loading training data from {train_csv}")
    train_df = pd.read_csv(train_csv)
    
    logger.info(f"Loading validation data from {val_csv}")
    val_df = pd.read_csv(val_csv)
    
    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Compute P_spiral
    train_df['P_spiral'] = train_df['P_CW'] + train_df['P_ACW']
    val_df['P_spiral'] = val_df['P_CW'] + val_df['P_ACW']
    
    # Get feature columns
    drop_cols = ['P_spiral', 'P_CW', 'P_ACW', 'Unnamed: 0', 'name', 'split', '_is_validation']
    feature_cols = [c for c in train_df.columns if c not in drop_cols and not c.startswith('_')]
    
    logger.info(f"Number of features: {len(feature_cols)}")
    
    X_train_full = train_df[feature_cols].copy()
    y_train_full = train_df['P_spiral'].copy()
    
    X_val = val_df[feature_cols].copy()
    y_val = val_df['P_spiral'].copy()
    
    # Encode categorical columns
    categorical_cols = X_train_full.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    label_encoders = {}
    
    logger.info(f"Encoding {len(categorical_cols)} categorical columns")
    
    for col in categorical_cols:
        le = LabelEncoder()
        all_values = pd.concat([X_train_full[col], X_val[col]]).astype(str).unique()
        le.fit(all_values)
        
        X_train_full[col] = le.transform(X_train_full[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        label_encoders[col] = le
    
    # Fill NaN
    X_train_full = X_train_full.fillna(0)
    X_val = X_val.fillna(0)
    
    # Convert to numpy for faster access
    X_train_np = X_train_full.values.astype(np.float32)
    y_train_np = y_train_full.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    y_val_np = y_val.values.astype(np.float32)
    
    logger.info("Data preparation complete")
    
    return {
        'X_train': X_train_np,
        'y_train': y_train_np,
        'X_val': X_val_np,
        'y_val': y_val_np,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders,
        'n_features': len(feature_cols)
    }


def bucket_based_split(
    X: np.ndarray,
    y: np.ndarray,
    num_buckets: int,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bucket-based train/test split (numpy version for speed)."""
    np.random.seed(random_state)
    
    bucket_step = 1 / num_buckets
    
    # Find smallest bucket size
    bucket_sizes = []
    for i in range(num_buckets):
        bucket_min = round(i * bucket_step, 3)
        bucket_max = round((i + 1) * bucket_step, 3)
        
        if i == num_buckets - 1:
            mask = y >= bucket_min
        else:
            mask = (y >= bucket_min) & (y < bucket_max)
        
        count = np.sum(mask)
        if count > 0:
            bucket_sizes.append(count)
    
    if not bucket_sizes:
        # Fallback: simple split
        n_train = int(len(y) * (1 - test_size))
        indices = np.random.permutation(len(y))
        return X[indices[:n_train]], X[indices[n_train:]], y[indices[:n_train]], y[indices[n_train:]]
    
    smallest = min(bucket_sizes)
    num_to_select = int(smallest * (1 - test_size))
    
    train_indices = []
    test_indices = []
    
    for i in range(num_buckets):
        bucket_min = round(i * bucket_step, 3)
        bucket_max = round((i + 1) * bucket_step, 3)
        
        if i == num_buckets - 1:
            mask = y >= bucket_min
        else:
            mask = (y >= bucket_min) & (y < bucket_max)
        
        bucket_indices = np.where(mask)[0]
        
        if len(bucket_indices) == 0:
            continue
        
        n_select = min(num_to_select, len(bucket_indices))
        np.random.shuffle(bucket_indices)
        
        train_indices.extend(bucket_indices[:n_select])
        test_indices.extend(bucket_indices[n_select:])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# ============================================================================
# WORKER INITIALIZER
# ============================================================================

def init_worker(shared_data: Dict):
    """Initialize worker with shared data."""
    global SHARED_DATA
    SHARED_DATA = shared_data


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_single_model(args: Tuple) -> Dict:
    """
    Train a single Random Forest model.
    Uses global SHARED_DATA set by init_worker.
    """
    trees, features, bins, model_output_dir, results_output_dir, model_idx, total_models = args
    
    try:
        global SHARED_DATA
        
        X_train_full = SHARED_DATA['X_train']
        y_train_full = SHARED_DATA['y_train']
        X_val = SHARED_DATA['X_val']
        y_val = SHARED_DATA['y_val']
        n_features = SHARED_DATA['n_features']
        label_encoders = SHARED_DATA['label_encoders']
        
        # Bucket-based split within training data
        X_train, X_test, y_train, y_test = bucket_based_split(
            X_train_full, y_train_full, num_buckets=bins, test_size=0.2, random_state=42
        )
        
        # Limit max_features
        actual_features = min(features, n_features)
        
        # Train model
        rf = RandomForestRegressor(
            n_estimators=trees,
            max_features=actual_features,
            random_state=42,
            n_jobs=1
        )
        rf.fit(X_train, y_train)
        
        # Evaluate
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)
        val_pred = rf.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        # Save model
        model_name = f"{trees}_{features}_{bins}"
        model_path = Path(model_output_dir) / f"{model_name}.pkl"
        le_path = Path(model_output_dir) / f"{model_name}_le.pkl"
        
        joblib.dump(rf, str(model_path))
        joblib.dump(label_encoders, str(le_path))
        
        # Generate scatterplot
        plot_path = Path(results_output_dir) / f"{model_name}.png"
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].scatter(y_train, train_pred, alpha=0.3, s=1)
        axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[0].set_xlabel('Actual P_spiral')
        axes[0].set_ylabel('Predicted P_spiral')
        axes[0].set_title(f'Training (R²={train_r2:.4f}, RMSE={train_rmse:.4f})')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        
        axes[1].scatter(y_test, test_pred, alpha=0.3, s=1)
        axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[1].set_xlabel('Actual P_spiral')
        axes[1].set_ylabel('Predicted P_spiral')
        axes[1].set_title(f'Test (R²={test_r2:.4f}, RMSE={test_rmse:.4f})')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        
        axes[2].scatter(y_val, val_pred, alpha=0.3, s=1)
        axes[2].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[2].set_xlabel('Actual P_spiral')
        axes[2].set_ylabel('Predicted P_spiral')
        axes[2].set_title(f'Validation (R²={val_r2:.4f}, RMSE={val_rmse:.4f})')
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        
        plt.suptitle(f'Trees={trees}, Features={features}, Bins={bins}')
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=100)
        plt.close(fig)
        
        # Progress message
        print(f"[{model_idx}/{total_models}] Completed: trees={trees}, features={features}, bins={bins}, val_RMSE={val_rmse:.4f}")
        
        return {
            'trees': trees,
            'features': features,
            'bins': bins,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'val_r2': val_r2,
            'model_path': str(model_path),
            'plot_path': str(plot_path),
            'status': 'success'
        }
        
    except Exception as e:
        print(f"[{model_idx}/{total_models}] FAILED: trees={trees}, features={features}, bins={bins}: {e}")
        return {
            'trees': trees,
            'features': features,
            'bins': bins,
            'status': 'error',
            'error': str(e)
        }


def run_grid_search(
    shared_data: Dict,
    model_output_dir: str,
    results_output_dir: str,
    dry_run: bool,
    n_workers: int,
    logger: logging.Logger
) -> List[Dict]:
    """Run parallel grid search over hyperparameters."""
    
    if dry_run:
        param_combos = DRY_RUN_MODELS
        logger.info(f"DRY RUN: Training {len(param_combos)} models")
    else:
        param_combos = [(t, f, b) for t in TREES_GRID for f in FEATURES_GRID for b in BINS_GRID]
        logger.info(f"Training {len(param_combos)} models ({len(TREES_GRID)} trees × {len(FEATURES_GRID)} features × {len(BINS_GRID)} bins)")
    
    total_models = len(param_combos)
    
    # Prepare arguments
    args_list = [
        (trees, features, bins, model_output_dir, results_output_dir, idx + 1, total_models)
        for idx, (trees, features, bins) in enumerate(param_combos)
    ]
    
    logger.info(f"Starting parallel training with {n_workers} workers")
    start_time = time.time()
    
    # Use initializer to share data with workers
    with Pool(processes=n_workers, initializer=init_worker, initargs=(shared_data,)) as pool:
        results = pool.map(train_single_model, args_list)
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Random Forest models via grid search (fast version)'
    )
    
    parser.add_argument('--train-csv', type=str, default=DEFAULT_TRAIN_CSV,
                        help='Path to augmented training CSV')
    parser.add_argument('--val-csv', type=str, default=DEFAULT_VAL_CSV,
                        help='Path to validation CSV')
    parser.add_argument('--model-output-dir', type=str, default=DEFAULT_MODEL_OUTPUT,
                        help='Directory to save trained models')
    parser.add_argument('--results-output-dir', type=str, default=DEFAULT_RESULTS_OUTPUT,
                        help='Directory to save results and scatterplots')
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run on small subset for testing')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Train Grid Search - Starting")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("*** DRY RUN MODE ***")
    
    n_workers = args.n_workers or max(1, cpu_count() - 1)
    logger.info(f"Workers: {n_workers}")
    logger.info(f"Train CSV: {args.train_csv}")
    logger.info(f"Val CSV: {args.val_csv}")
    
    Path(args.model_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data ONCE
    logger.info("\n" + "=" * 40)
    logger.info("STEP 1: Load and prepare data")
    logger.info("=" * 40)
    
    shared_data = load_and_prepare_data(args.train_csv, args.val_csv, logger)
    
    # Run grid search
    logger.info("\n" + "=" * 40)
    logger.info("STEP 2: Grid search training")
    logger.info("=" * 40)
    
    results = run_grid_search(
        shared_data, args.model_output_dir, args.results_output_dir,
        args.dry_run, n_workers, logger
    )
    
    # Save results
    logger.info("\n" + "=" * 40)
    logger.info("STEP 3: Save results")
    logger.info("=" * 40)
    
    results_df = pd.DataFrame(results)
    results_path = Path(args.results_output_dir) / "model_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Summary
    successful = results_df[results_df['status'] == 'success'].copy()
    successful = successful.sort_values('val_rmse')
    
    logger.info("\n" + "=" * 40)
    logger.info("TOP 10 MODELS (by validation RMSE)")
    logger.info("=" * 40)
    
    top10 = successful.head(10)
    for _, row in top10.iterrows():
        logger.info(f"Trees={row['trees']}, Features={row['features']}, Bins={row['bins']}: "
                   f"Val RMSE={row['val_rmse']:.4f}, Val R²={row['val_r2']:.4f}")
    
    top10_path = Path(args.results_output_dir) / "top10_models.csv"
    top10.to_csv(top10_path, index=False)
    
    logger.info("\n" + "=" * 40)
    logger.info("SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Total models: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(results) - len(successful)}")
    if len(successful) > 0:
        logger.info(f"Best val RMSE: {successful['val_rmse'].min():.4f}")
        logger.info(f"Best val R²: {successful['val_r2'].max():.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Train Grid Search - Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
