#!/usr/bin/env python3
"""
Augment Random Forest Training with Blurred Galaxy Data

This script:
1. Parses blurred galaxy SpArcFiRe outputs from 7000images/
2. Balances P_spiral bins by augmenting underrepresented bins with blurred variations
3. Cleans the data using the same transformations as scripts_for_sparcfire_output.txt
4. Trains 560 Random Forest models via parallel grid search
5. Saves all models, scatterplots, and metrics

Usage:
    python augment_and_train_rf.py [options]
    python augment_and_train_rf.py --dry-run  # Test on small subset
"""

import argparse
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, Manager, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Suppress warnings during parallel training
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

DEFAULT_IMAGES_DIR = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/7000images"
DEFAULT_ZOO_CSV = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/zoo2MainSpecz.csv"
DEFAULT_TRAINING_CSV = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/randomforest_training_data/data_cleaned5.csv"
DEFAULT_MODEL_OUTPUT = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/random_forest_model_output"
DEFAULT_RESULTS_OUTPUT = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/random_forest_output"
DEFAULT_AUGMENTED_DATA = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/randomforest_training_data"

# Hyperparameter grid
TREES_GRID = [10, 30, 50, 70, 90, 110, 130, 150]
FEATURES_GRID = [10, 30, 50, 70, 90, 110, 130]
BINS_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Dry-run settings
DRY_RUN_GALAXIES = 100
DRY_RUN_MODELS = [(10, 10, 1), (10, 10, 2), (10, 10, 3)]  # (trees, features, bins)

# Number of P_spiral bins for augmentation
NUM_SPIRALITY_BINS = 10

# Column data types for SpArcFiRe output (matching reading_sparcfire.py)
SPARCFIRE_DTYPE = {
    "name": "string",
    "fit_state": "string",
    "warnings": "string",
    "star_mask_used": "string",
    "noise_mask_used": "string",
    "chirality_maj": "string",
    "chirality_alenWtd": "string",
    "chirality_wtdPangSum": "string",
    "chirality_longestArc": "string",
    "badBulgeFitFlag": "string",  # Will convert to bool after cleaning
    "bulgeAxisRatio": "float64",
    "bulgeMajAxsLen": "float64",
    "bulgeMajAxsAngle": "float64",
    "bulgeAvgBrt": "float64",
    "bulgeDiskBrtRatio": "float64",
    "numElpsRefits": "float64",
    "diskAxisRatio": "float64",
    "diskMinAxsLen": "float64",
    "diskMajAxsLen": "float64",
    "diskMajAxsAngleRadians": "float64",
    "inputCenterR": "float64",
    "inputCenterC": "float64",
    "iptSz": "string",
    "muDist": "float64",
    "muDistProp": "float64",
    "covarFit": "string",
    "wtdLik": "float64",
    "likOfCtr": "float64",
    "brtUnifScore": "float64",
    "gaussLogLik": "float64",
    "contourBrtRatio": "float64",
    "standardizedCenterR": "float64",
    "standardizedCenterC": "float64",
    "hasDeletedCtrClus": "string",
    "failed2revDuringMergeCheck": "string",
    "failed2revDuringSecondaryMerging": "string",
    "failed2revInOutput": "string",
    "cpu_time": "float64",
    "wall_time": "float64",
    "bar_candidate_available": "string",
    "bar_used": "string",
    "alenAt25pct": "float64",
    "alenAt50pct": "float64",
    "alenAt75pct": "float64",
    "rankAt25pct": "float64",
    "rankAt50pct": "float64",
    "rankAt75pct": "float64",
    "avgArcLength": "float64",
    "minArcLength": "float64",
    "lowerQuartileArcLength": "float64",
    "medianArcLength": "float64",
    "upperQuartileArcLength": "float64",
    "maxArcLength": "float64",
    "totalArcLength": "float64",
    "totalNumArcs": "float64",
    "chirality_votes_maj": "string",
    "chirality_votes_alenWtd": "string",
    "alenWtdPangSum": "float64",
    "top2_chirality_agreement": "string",
    "pa_longest": "float64",
    "pa_avg": "float64",
    "pa_avg_abs": "float64",
    "pa_avg_domChiralityOnly": "float64",
    "pa_alenWtd_avg": "float64",
    "paErr_alenWtd_stdev": "float64",
    "pa_alenWtd_avg_abs": "float64",
    "paErr_alenWtd_stdev_abs": "float64",
    "pa_alenWtd_avg_domChiralityOnly": "float64",
    "paErr_alenWtd_stdev_domChiralityOnly": "float64",
    "pa_totBrtWtd": "float64",
    "pa_avgBrtWtd": "float64",
    "pa_alenWtd_median": "float64",
    "pa_alenWtd_lowQuartile": "float64",
    "pa_alenWtd_highQuartile": "float64",
    "pa_alenWtd_lowDecile": "float64",
    "pa_alenWtd_highDecile": "float64",
    "pa_alenWtd_median_domChiralityOnly": "float64",
    "pa_alenWtd_lowQuartile_domChiralityOnly": "float64",
    "pa_alenWtd_highQuartile_domChiralityOnly": "float64",
    "pa_alenWtd_lowDecile_domChiralityOnly": "float64",
    "pa_alenWtd_highDecile_domChiralityOnly": "float64",
    "sorted_agreeing_pangs": "string",
    "sorted_agreeing_arclengths": "string",
    "numArcs_largest_length_gap": "float64",
    "numDcoArcs_largest_length_gap": "float64",
    "numArcs_arclength_function_flattening": "float64",
    "numDcoArcs_arclength_function_flattening": "float64",
    "bar_score_img": "float64",
    "bar_cand_score_orifld": "float64",
    "bar_angle_input_img": "float64",
    "bar_half_length_input_img": "float64",
    "bar_angle_standardized_img": "float64",
    "bar_half_length_standardized_img": "float64",
}

# Add numArcsGE and numDcoArcsGE columns
for suffix in ["000", "010", "020", "040", "050", "055", "060", "065", "070", "075", 
               "080", "085", "090", "095", "100", "120", "140", "160", "180", "200",
               "220", "240", "260", "280", "300", "350", "400", "450", "500", "550", "600"]:
    SPARCFIRE_DTYPE[f"numArcsGE{suffix}"] = "float64"
    SPARCFIRE_DTYPE[f"numDcoArcsGE{suffix}"] = "float64"


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str = "augment_train.log") -> logging.Logger:
    """Configure logging with timestamp to file and console."""
    logger = logging.getLogger("augment_train")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# DATA PARSING FUNCTIONS
# ============================================================================

def parse_7000images(images_dir: str, dry_run: bool = False, logger: logging.Logger = None) -> Dict[str, pd.DataFrame]:
    """
    Parse all successful SpArcFiRe outputs from 7000images/ folder.
    
    Returns:
        Dict mapping objectID -> DataFrame of successful variation rows
    """
    images_path = Path(images_dir)
    galaxy_data = {}
    processed = 0
    errors = 0
    
    galaxy_folders = sorted([f for f in images_path.iterdir() if f.is_dir()])
    
    if dry_run:
        galaxy_folders = galaxy_folders[:DRY_RUN_GALAXIES]
        logger.info(f"DRY RUN: Processing only {len(galaxy_folders)} galaxies")
    
    total = len(galaxy_folders)
    logger.info(f"Parsing {total} galaxy folders from {images_dir}")
    
    for folder in galaxy_folders:
        object_id = folder.name
        galaxy_csv = folder / "2-OUT" / "galaxy.csv"
        
        if not galaxy_csv.exists():
            errors += 1
            continue
        
        try:
            # Read CSV with flexible parsing
            df = pd.read_csv(galaxy_csv, skipinitialspace=True, on_bad_lines='skip')
            df.columns = df.columns.str.strip()
            
            # Filter for successful runs only
            if 'fit_state' in df.columns:
                df_ok = df[df['fit_state'].str.strip() == 'OK'].copy()
                
                if len(df_ok) > 0:
                    galaxy_data[object_id] = df_ok
            
            processed += 1
            
            if processed % 500 == 0:
                logger.info(f"Processed {processed}/{total} galaxies ({len(galaxy_data)} with successful runs)")
                
        except Exception as e:
            errors += 1
            if processed < 10:  # Only log first few errors
                logger.warning(f"Error parsing {galaxy_csv}: {e}")
    
    logger.info(f"Completed parsing: {processed} processed, {errors} errors, {len(galaxy_data)} galaxies with successful SpArcFiRe runs")
    
    return galaxy_data


def load_zoo_pspiral(zoo_csv: str, logger: logging.Logger) -> Dict[str, float]:
    """
    Load Galaxy Zoo P_spiral values from zoo2MainSpecz.csv.
    
    Returns:
        Dict mapping dr8objid -> P_spiral (t04_spiral_a08_spiral_weighted_fraction)
    """
    logger.info(f"Loading P_spiral labels from {zoo_csv}")
    
    # Read with dr8objid as string to preserve full 19-digit integer
    df = pd.read_csv(zoo_csv, usecols=['dr8objid', 't04_spiral_a08_spiral_weighted_fraction'],
                     dtype={'dr8objid': str})
    
    # Remove any NaN values and create mapping
    df = df.dropna(subset=['dr8objid'])
    pspiral_map = dict(zip(df['dr8objid'].str.strip(), df['t04_spiral_a08_spiral_weighted_fraction']))
    
    logger.info(f"Loaded P_spiral values for {len(pspiral_map)} galaxies")
    
    # Log a sample for verification
    sample_keys = list(pspiral_map.keys())[:3]
    logger.info(f"Sample objectIDs: {sample_keys}")
    
    return pspiral_map


def analyze_bin_distribution(training_csv: str, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[int, int], int]:
    """
    Analyze current training data bin distribution.
    
    Returns:
        (full_dataframe, bin_counts_dict, max_bin_count)
    """
    logger.info(f"Loading training data from {training_csv}")
    
    df = pd.read_csv(training_csv)
    df.columns = df.columns.str.strip()
    
    # Compute P_spiral
    df['P_spiral'] = df['P_CW'] + df['P_ACW']
    
    # Bin the data
    bin_counts = {}
    for i in range(NUM_SPIRALITY_BINS):
        bin_min = i / NUM_SPIRALITY_BINS
        bin_max = (i + 1) / NUM_SPIRALITY_BINS
        
        if i == NUM_SPIRALITY_BINS - 1:
            count = len(df[df['P_spiral'] >= bin_min])
        else:
            count = len(df[(df['P_spiral'] >= bin_min) & (df['P_spiral'] < bin_max)])
        
        bin_counts[i] = count
        logger.info(f"Bin {i} ({bin_min:.1f}-{bin_max:.1f}): {count} galaxies")
    
    max_count = max(bin_counts.values())
    logger.info(f"Max bin count (target): {max_count}")
    
    return df, bin_counts, max_count


# ============================================================================
# DATA CLEANING FUNCTIONS (Python port of scripts_for_sparcfire_output.txt)
# ============================================================================

def clean_sparcfire_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Clean SpArcFiRe output data using same transformations as shell scripts.
    
    Applies:
    - Fix categorical names (Z-wise -> Zwise, etc.)
    - Replace spaces with underscores in array columns
    - Fix boolean casing (true -> True)
    - Remove rows with -Inf values
    """
    original_len = len(df)
    logger.info(f"Cleaning {original_len} rows of SpArcFiRe data")
    
    if original_len == 0:
        return df
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # 1. Fix categorical names
    categorical_fixes = {
        'star_mask_used': {'unavailable': 'none'},
        'noise_mask_used': {'unavailable': 'none', 'aggressive-exclusive': 'aggressiveexclusive', 
                           'conservative-exclusive': 'conservativeexclusive'},
        'chirality_maj': {'Z-wise': 'Zwise', 'S-wise': 'Swise'},
        'chirality_alenWtd': {'Z-wise': 'Zwise', 'S-wise': 'Swise'},
        'chirality_wtdPangSum': {'Z-wise': 'Zwise', 'S-wise': 'Swise'},
        'chirality_longestArc': {'Z-wise': 'Zwise', 'S-wise': 'Swise'},
        'top2_chirality_agreement': {'<2 arcs': '<2_arcs', 'one-long': 'onelong', 'all-short': 'allshort'}
    }
    
    for col, replacements in categorical_fixes.items():
        if col in df.columns:
            for old, new in replacements.items():
                df[col] = df[col].astype(str).str.replace(old, new, regex=False)
    
    # 2. Replace spaces with underscores in array-like columns
    array_cols = ['iptSz', 'covarFit', 'chirality_votes_maj', 'chirality_votes_alenWtd',
                  'sorted_agreeing_arclengths', 'sorted_agreeing_pangs']
    for col in array_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(' ', '_', regex=False)
    
    # 3. Fix boolean casing (only for string columns)
    bool_cols = ['badBulgeFitFlag', 'hasDeletedCtrClus', 'failed2revDuringMergeCheck',
                 'failed2revDuringSecondaryMerging', 'failed2revInOutput',
                 'bar_candidate_available', 'bar_used']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('true', 'True', regex=False)
            df[col] = df[col].astype(str).str.replace('false', 'False', regex=False)
    
    # 4. Remove rows with -Inf, Inf in numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df = df[~df[col].isin([np.inf, -np.inf])]
    
    # 5. Check for string representations of inf in string columns
    for col in df.select_dtypes(include=['object']).columns:
        mask = df[col].str.contains('-Inf|Inf', na=False, regex=True)
        if mask.any():
            df = df[~mask]
    
    final_len = len(df)
    logger.info(f"Cleaning complete: {original_len} -> {final_len} rows ({original_len - final_len} removed)")
    
    return df


# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

def create_balanced_augmentation(
    galaxy_data: Dict[str, pd.DataFrame],
    pspiral_map: Dict[str, float],
    bin_counts: Dict[int, int],
    target_count: int,
    validation_object_ids: set,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Create balanced augmentation set by sampling blurred variations.
    
    Args:
        galaxy_data: Dict of objectID -> DataFrame of successful variations
        pspiral_map: Dict of objectID -> P_spiral value
        bin_counts: Current count per bin
        target_count: Target count for each bin
        validation_object_ids: Set of objectIDs to exclude (validation set)
    
    Returns:
        DataFrame of augmented rows with P_CW and P_ACW columns added
    """
    logger.info("Creating balanced augmentation set")
    
    # Organize galaxies by bin
    galaxies_by_bin = {i: [] for i in range(NUM_SPIRALITY_BINS)}
    
    for object_id, df in galaxy_data.items():
        # Skip if in validation set
        if object_id in validation_object_ids:
            continue
        
        # Get P_spiral for this galaxy
        pspiral = pspiral_map.get(object_id)
        if pspiral is None:
            continue
        
        # Determine bin
        bin_idx = min(int(pspiral * NUM_SPIRALITY_BINS), NUM_SPIRALITY_BINS - 1)
        galaxies_by_bin[bin_idx].append((object_id, pspiral, df))
    
    # Log available galaxies per bin
    for bin_idx, galaxies in galaxies_by_bin.items():
        logger.info(f"Bin {bin_idx}: {len(galaxies)} galaxies available for augmentation")
    
    # Sample variations to balance bins
    augmented_rows = []
    
    for bin_idx in range(NUM_SPIRALITY_BINS):
        current_count = bin_counts[bin_idx]
        deficit = target_count - current_count
        
        if deficit <= 0:
            logger.info(f"Bin {bin_idx}: No augmentation needed (already at target)")
            continue
        
        available_galaxies = galaxies_by_bin[bin_idx]
        if not available_galaxies:
            logger.warning(f"Bin {bin_idx}: No galaxies available for augmentation (deficit: {deficit})")
            continue
        
        logger.info(f"Bin {bin_idx}: Augmenting {deficit} samples from {len(available_galaxies)} galaxies")
        
        # Sample variations randomly
        samples_added = 0
        np.random.seed(42 + bin_idx)  # Reproducible per bin
        
        # Create pool of all variations
        variation_pool = []
        for object_id, pspiral, df in available_galaxies:
            for idx, row in df.iterrows():
                variation_pool.append((object_id, pspiral, row))
        
        # Shuffle and sample
        np.random.shuffle(variation_pool)
        
        for object_id, pspiral, row in variation_pool:
            if samples_added >= deficit:
                break
            
            row_dict = row.to_dict()
            row_dict['P_CW'] = pspiral  # Synthetic: P_CW = P_spiral
            row_dict['P_ACW'] = 0.0     # Synthetic: P_ACW = 0
            row_dict['_original_object_id'] = object_id  # For tracking
            augmented_rows.append(row_dict)
            samples_added += 1
        
        logger.info(f"Bin {bin_idx}: Added {samples_added} augmented samples")
    
    if not augmented_rows:
        logger.warning("No augmented rows created!")
        return pd.DataFrame()
    
    augmented_df = pd.DataFrame(augmented_rows)
    logger.info(f"Total augmented samples: {len(augmented_df)}")
    
    return augmented_df


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def parse_str_to_np(s: str) -> np.ndarray:
    """Parse string representation of array to numpy array."""
    if pd.isna(s) or s == 'nan':
        return np.array([])
    s = re.sub(r'[\[\]]', '', str(s))
    s = s.replace('_', ' ')
    try:
        return np.fromstring(s, sep=' ', dtype=float)
    except:
        return np.array([])


def expand_list_columns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Expand list-like columns into separate columns (matching reading_sparcfire.py)."""
    list_cols = ["iptSz", "covarFit", "chirality_votes_maj", "chirality_votes_alenWtd",
                 "sorted_agreeing_pangs", "sorted_agreeing_arclengths"]
    
    for col in list_cols:
        if col not in df.columns:
            continue
        
        try:
            df[col] = df[col].apply(parse_str_to_np)
            
            if col == "covarFit":
                # Fixed 7 elements for covarFit
                df[col] = df[col].apply(
                    lambda x: list(x)[:7] + [None] * (7 - len(x)) if isinstance(x, (list, np.ndarray)) else [None] * 7
                )
                expanded_cols = pd.DataFrame(df[col].tolist(), columns=[f"{col}_{i}" for i in range(7)], index=df.index)
            else:
                expanded_cols = df[col].apply(pd.Series)
                expanded_cols.columns = [f"{col}_{i}" for i in expanded_cols.columns]
                expanded_cols.index = df.index
            
            df = pd.concat([df, expanded_cols], axis=1)
            df.drop(columns=[col], inplace=True)
            
        except Exception as e:
            logger.warning(f"Error expanding column {col}: {e}")
    
    return df


def prepare_training_data(
    original_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    validation_object_ids: set,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare training and validation datasets.
    
    Returns:
        (training_df, validation_df)
    """
    logger.info("Preparing training and validation datasets")
    
    # Split original data
    original_df['_is_validation'] = original_df['name'].astype(str).isin(validation_object_ids)
    
    train_original = original_df[~original_df['_is_validation']].copy()
    val_df = original_df[original_df['_is_validation']].copy()
    
    train_original.drop(columns=['_is_validation'], inplace=True)
    val_df.drop(columns=['_is_validation'], inplace=True)
    
    logger.info(f"Original training: {len(train_original)}, Validation: {len(val_df)}")
    
    # Merge augmented data with training
    if len(augmented_df) > 0:
        # Align columns
        augmented_df = augmented_df.drop(columns=['_original_object_id'], errors='ignore')
        
        # Add missing columns to augmented data
        for col in train_original.columns:
            if col not in augmented_df.columns:
                augmented_df[col] = np.nan
        
        # Keep only columns that exist in original
        augmented_df = augmented_df[train_original.columns]
        
        train_df = pd.concat([train_original, augmented_df], ignore_index=True)
        logger.info(f"Combined training: {len(train_df)} (original: {len(train_original)}, augmented: {len(augmented_df)})")
    else:
        train_df = train_original
    
    return train_df, val_df


def stratified_validation_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    logger: logging.Logger = None
) -> set:
    """
    Perform stratified split by P_spiral bin.
    
    Returns:
        Set of object IDs for validation set
    """
    logger.info(f"Performing stratified {test_size*100:.0f}% validation split")
    
    df = df.copy()
    df['P_spiral'] = df['P_CW'] + df['P_ACW']
    df['_bin'] = pd.cut(df['P_spiral'], bins=NUM_SPIRALITY_BINS, labels=False)
    
    validation_ids = set()
    
    for bin_idx in range(NUM_SPIRALITY_BINS):
        bin_data = df[df['_bin'] == bin_idx]
        n_val = max(1, int(len(bin_data) * test_size))
        
        val_sample = bin_data.sample(n=min(n_val, len(bin_data)), random_state=42)
        validation_ids.update(val_sample['name'].astype(str).tolist())
    
    logger.info(f"Validation set: {len(validation_ids)} unique galaxies")
    
    return validation_ids


def bucket_based_split(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    num_buckets: int,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Bucket-based train/test split matching original implementation."""
    x_data = x_data.copy()
    x_data['P_spiral'] = y_data
    
    x_test = pd.DataFrame()
    x_train = pd.DataFrame()
    
    bucket_step = 1 / num_buckets
    
    # Find smallest bucket size
    smallest = float('inf')
    for i in range(num_buckets):
        bucket_min = round(i * bucket_step, 3)
        bucket_max = round((i + 1) * bucket_step, 3)
        
        if i == num_buckets - 1:
            count = len(x_data[x_data['P_spiral'] >= bucket_min])
        else:
            count = len(x_data[(x_data['P_spiral'] >= bucket_min) & (x_data['P_spiral'] < bucket_max)])
        
        if count > 0:
            smallest = min(smallest, count)
    
    num_to_select = int(smallest * (1 - test_size))
    
    for i in range(num_buckets):
        bucket_min = round(i * bucket_step, 3)
        bucket_max = round((i + 1) * bucket_step, 3)
        
        if i == num_buckets - 1:
            bucket_values = x_data[x_data['P_spiral'] >= bucket_min]
        else:
            bucket_values = x_data[(x_data['P_spiral'] >= bucket_min) & (x_data['P_spiral'] < bucket_max)]
        
        if len(bucket_values) == 0:
            continue
        
        n_select = min(num_to_select, len(bucket_values))
        x_bucket_train = bucket_values.sample(n=n_select, random_state=random_state)
        x_bucket_test = bucket_values.drop(index=x_bucket_train.index)
        
        x_test = pd.concat([x_test, x_bucket_test], axis=0)
        x_train = pd.concat([x_train, x_bucket_train], axis=0)
    
    x_train = x_train.drop(columns=['P_spiral'])
    x_test = x_test.drop(columns=['P_spiral'])
    
    y_train = y_data.loc[x_train.index]
    y_test = y_data.loc[x_test.index]
    
    return x_train, x_test, y_train, y_test


def train_single_model(args: Tuple) -> Dict:
    """
    Train a single Random Forest model.
    
    This function is designed to be called from a multiprocessing Pool.
    """
    (trees, features, bins, train_df, val_df, 
     model_output_dir, results_output_dir, 
     progress_counter, progress_lock, total_models) = args
    
    try:
        # Prepare data
        train_df = train_df.copy()
        val_df = val_df.copy()
        
        # Compute P_spiral
        train_df['P_spiral'] = train_df['P_CW'] + train_df['P_ACW']
        val_df['P_spiral'] = val_df['P_CW'] + val_df['P_ACW']
        
        # Get feature columns (drop non-feature columns)
        drop_cols = ['P_spiral', 'P_CW', 'P_ACW', 'Unnamed: 0', 'name', 'split']
        feature_cols = [c for c in train_df.columns if c not in drop_cols]
        
        X_train_full = train_df[feature_cols]
        y_train_full = train_df['P_spiral']
        
        X_val = val_df[feature_cols]
        y_val = val_df['P_spiral']
        
        # Bucket-based split within training data
        X_train, X_test, y_train, y_test = bucket_based_split(
            X_train_full, y_train_full, num_buckets=bins, test_size=0.2, random_state=42
        )
        
        # Encode categorical columns
        categorical_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on all possible values
            all_values = pd.concat([X_train[col], X_test[col], X_val[col]]).astype(str).unique()
            le.fit(all_values)
            
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            X_val[col] = le.transform(X_val[col].astype(str))
            label_encoders[col] = le
        
        # Handle NaN values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        X_val = X_val.fillna(0)
        
        # Limit max_features to available features
        actual_features = min(features, len(feature_cols))
        
        # Train model
        rf = RandomForestRegressor(
            n_estimators=trees,
            max_features=actual_features,
            random_state=42,
            n_jobs=1  # Single thread per model since we're parallelizing models
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
        
        # Training scatter
        axes[0].scatter(y_train, train_pred, alpha=0.3, s=1)
        axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[0].set_xlabel('Actual P_spiral')
        axes[0].set_ylabel('Predicted P_spiral')
        axes[0].set_title(f'Training (R²={train_r2:.4f}, RMSE={train_rmse:.4f})')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        
        # Test scatter
        axes[1].scatter(y_test, test_pred, alpha=0.3, s=1)
        axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[1].set_xlabel('Actual P_spiral')
        axes[1].set_ylabel('Predicted P_spiral')
        axes[1].set_title(f'Test (R²={test_r2:.4f}, RMSE={test_rmse:.4f})')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        
        # Validation scatter
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
        
        # Update progress (thread-safe)
        with progress_lock:
            progress_counter.value += 1
            current = progress_counter.value
            if current % 10 == 0 or current == total_models:
                print(f"Progress: {current}/{total_models} models completed")
        
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
        # Update progress even on failure
        with progress_lock:
            progress_counter.value += 1
        
        return {
            'trees': trees,
            'features': features,
            'bins': bins,
            'status': 'error',
            'error': str(e)
        }


def run_grid_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_output_dir: str,
    results_output_dir: str,
    dry_run: bool,
    logger: logging.Logger
) -> List[Dict]:
    """
    Run parallel grid search over hyperparameters.
    """
    # Generate parameter combinations
    if dry_run:
        param_combos = DRY_RUN_MODELS
        logger.info(f"DRY RUN: Training {len(param_combos)} models")
    else:
        param_combos = [(t, f, b) for t in TREES_GRID for f in FEATURES_GRID for b in BINS_GRID]
        logger.info(f"Training {len(param_combos)} models ({len(TREES_GRID)} trees × {len(FEATURES_GRID)} features × {len(BINS_GRID)} bins)")
    
    total_models = len(param_combos)
    
    # Setup multiprocessing manager for thread-safe progress
    manager = Manager()
    progress_counter = manager.Value('i', 0)
    progress_lock = manager.Lock()
    
    # Prepare arguments for each model
    args_list = [
        (trees, features, bins, train_df, val_df,
         model_output_dir, results_output_dir,
         progress_counter, progress_lock, total_models)
        for trees, features, bins in param_combos
    ]
    
    # Run parallel training
    n_workers = max(1, cpu_count() - 1)
    logger.info(f"Starting parallel training with {n_workers} workers")
    
    start_time = time.time()
    
    with Pool(processes=n_workers) as pool:
        results = pool.map(train_single_model, args_list)
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    return results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Augment training data and train Random Forest models for galaxy spirality prediction'
    )
    
    parser.add_argument('--images-dir', type=str, default=DEFAULT_IMAGES_DIR,
                        help='Path to 7000images/ folder with blurred SpArcFiRe outputs')
    parser.add_argument('--zoo-csv', type=str, default=DEFAULT_ZOO_CSV,
                        help='Path to zoo2MainSpecz.csv with P_spiral labels')
    parser.add_argument('--training-csv', type=str, default=DEFAULT_TRAINING_CSV,
                        help='Path to existing training data CSV')
    parser.add_argument('--model-output-dir', type=str, default=DEFAULT_MODEL_OUTPUT,
                        help='Directory to save trained models')
    parser.add_argument('--results-output-dir', type=str, default=DEFAULT_RESULTS_OUTPUT,
                        help='Directory to save results and scatterplots')
    parser.add_argument('--augmented-data-dir', type=str, default=DEFAULT_AUGMENTED_DATA,
                        help='Directory to save augmented training data')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run on small subset for testing')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Augment and Train Random Forest - Starting")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("*** DRY RUN MODE ***")
    
    logger.info(f"Images dir: {args.images_dir}")
    logger.info(f"Zoo CSV: {args.zoo_csv}")
    logger.info(f"Training CSV: {args.training_csv}")
    logger.info(f"Model output: {args.model_output_dir}")
    logger.info(f"Results output: {args.results_output_dir}")
    
    # Create output directories
    Path(args.model_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.augmented_data_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Parse 7000images/
    logger.info("\n" + "=" * 40)
    logger.info("STEP 1: Parse blurred galaxy data")
    logger.info("=" * 40)
    galaxy_data = parse_7000images(args.images_dir, args.dry_run, logger)
    
    # Step 2: Load P_spiral labels
    logger.info("\n" + "=" * 40)
    logger.info("STEP 2: Load P_spiral labels")
    logger.info("=" * 40)
    pspiral_map = load_zoo_pspiral(args.zoo_csv, logger)
    
    # Step 3: Analyze current bin distribution
    logger.info("\n" + "=" * 40)
    logger.info("STEP 3: Analyze bin distribution")
    logger.info("=" * 40)
    original_df, bin_counts, target_count = analyze_bin_distribution(args.training_csv, logger)
    
    # Step 4: Stratified validation split
    logger.info("\n" + "=" * 40)
    logger.info("STEP 4: Stratified validation split")
    logger.info("=" * 40)
    validation_ids = stratified_validation_split(original_df, test_size=0.2, logger=logger)
    
    # Step 5: Create balanced augmentation
    logger.info("\n" + "=" * 40)
    logger.info("STEP 5: Create balanced augmentation")
    logger.info("=" * 40)
    augmented_df = create_balanced_augmentation(
        galaxy_data, pspiral_map, bin_counts, target_count, validation_ids, logger
    )
    
    # Step 6: Clean blurred data
    logger.info("\n" + "=" * 40)
    logger.info("STEP 6: Clean blurred data")
    logger.info("=" * 40)
    if len(augmented_df) > 0:
        augmented_df = clean_sparcfire_data(augmented_df, logger)
        augmented_df = expand_list_columns(augmented_df, logger)
    
    # Step 7: Prepare training and validation data
    logger.info("\n" + "=" * 40)
    logger.info("STEP 7: Prepare datasets")
    logger.info("=" * 40)
    
    # Expand list columns in original data too
    original_df = expand_list_columns(original_df, logger)
    
    train_df, val_df = prepare_training_data(original_df, augmented_df, validation_ids, logger)
    
    # Save augmented data
    augmented_path = Path(args.augmented_data_dir) / "data_augmented.csv"
    train_df.to_csv(augmented_path, index=False)
    logger.info(f"Saved augmented training data to {augmented_path}")
    
    # Step 8: Grid search training
    logger.info("\n" + "=" * 40)
    logger.info("STEP 8: Grid search training")
    logger.info("=" * 40)
    results = run_grid_search(train_df, val_df, args.model_output_dir, args.results_output_dir, args.dry_run, logger)
    
    # Step 9: Save results and summarize
    logger.info("\n" + "=" * 40)
    logger.info("STEP 9: Save results")
    logger.info("=" * 40)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_path = Path(args.results_output_dir) / "model_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved all results to {results_path}")
    
    # Filter successful models and sort by validation RMSE
    successful = results_df[results_df['status'] == 'success'].copy()
    successful = successful.sort_values('val_rmse')
    
    # Log top 10
    logger.info("\n" + "=" * 40)
    logger.info("TOP 10 MODELS (by validation RMSE)")
    logger.info("=" * 40)
    
    top10 = successful.head(10)
    for i, row in top10.iterrows():
        logger.info(f"Trees={row['trees']}, Features={row['features']}, Bins={row['bins']}: "
                   f"Val RMSE={row['val_rmse']:.4f}, Val R²={row['val_r2']:.4f}")
    
    # Save top 10 summary
    top10_path = Path(args.results_output_dir) / "top10_models.csv"
    top10.to_csv(top10_path, index=False)
    logger.info(f"\nSaved top 10 models to {top10_path}")
    
    # Summary statistics
    logger.info("\n" + "=" * 40)
    logger.info("SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Total models trained: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(results) - len(successful)}")
    logger.info(f"Best validation RMSE: {successful['val_rmse'].min():.4f}")
    logger.info(f"Best validation R²: {successful['val_r2'].max():.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Augment and Train Random Forest - Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
