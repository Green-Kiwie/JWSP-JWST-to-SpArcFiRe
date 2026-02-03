#!/usr/bin/env python3
"""
Augment Training Data with Blurred Galaxy Variations

This script performs data augmentation ONLY (no training):
1. Parses blurred galaxy SpArcFiRe outputs from 7000images/
2. Balances P_spiral bins by augmenting underrepresented bins
3. Cleans and prepares the data
4. Saves augmented training + validation CSVs for use by train_grid_search.py

Usage:
    python augment_data_only.py
    python augment_data_only.py --dry-run  # Test on small subset
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

DEFAULT_IMAGES_DIR = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/7000images"
DEFAULT_ZOO_CSV = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/zoo2MainSpecz.csv"
DEFAULT_TRAINING_CSV = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/randomforest_training_data/data_cleaned5.csv"
DEFAULT_OUTPUT_DIR = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/randomforest_training_data"

# Dry-run settings
DRY_RUN_GALAXIES = 100

# Number of P_spiral bins for augmentation
NUM_SPIRALITY_BINS = 10

# Column data types for SpArcFiRe output
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
    "badBulgeFitFlag": "string",
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

def setup_logging(log_file: str = "augment_data.log") -> logging.Logger:
    """Configure logging with timestamp to file and console."""
    logger = logging.getLogger("augment_data")
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
            if processed < 10:
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
# DATA CLEANING FUNCTIONS
# ============================================================================

def clean_sparcfire_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Clean SpArcFiRe output data using same transformations as shell scripts.
    """
    original_len = len(df)
    logger.info(f"Cleaning {original_len} rows of SpArcFiRe data")
    
    if original_len == 0:
        return df
    
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
    
    # 3. Fix boolean casing
    bool_cols = ['badBulgeFitFlag', 'hasDeletedCtrClus', 'failed2revDuringMergeCheck',
                 'failed2revDuringSecondaryMerging', 'failed2revInOutput',
                 'bar_candidate_available', 'bar_used']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('true', 'True', regex=False)
            df[col] = df[col].astype(str).str.replace('false', 'False', regex=False)
    
    # 4. Remove rows with -Inf, Inf in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df = df[~df[col].isin([np.inf, -np.inf])]
    
    # 5. Check for string representations of inf
    for col in df.select_dtypes(include=['object']).columns:
        mask = df[col].str.contains('-Inf|Inf', na=False, regex=True)
        if mask.any():
            df = df[~mask]
    
    final_len = len(df)
    logger.info(f"Cleaning complete: {original_len} -> {final_len} rows ({original_len - final_len} removed)")
    
    return df


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
    """Expand list-like columns into separate columns."""
    list_cols = ["iptSz", "covarFit", "chirality_votes_maj", "chirality_votes_alenWtd",
                 "sorted_agreeing_pangs", "sorted_agreeing_arclengths"]
    
    for col in list_cols:
        if col not in df.columns:
            continue
        
        try:
            df[col] = df[col].apply(parse_str_to_np)
            
            if col == "covarFit":
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


# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

def stratified_validation_split(df: pd.DataFrame, test_size: float = 0.2, logger: logging.Logger = None) -> set:
    """Perform stratified split by P_spiral bin."""
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


def create_balanced_augmentation(
    galaxy_data: Dict[str, pd.DataFrame],
    pspiral_map: Dict[str, float],
    bin_counts: Dict[int, int],
    target_count: int,
    validation_object_ids: set,
    logger: logging.Logger
) -> pd.DataFrame:
    """Create balanced augmentation set by sampling blurred variations."""
    logger.info("Creating balanced augmentation set")
    
    # Organize galaxies by bin
    galaxies_by_bin = {i: [] for i in range(NUM_SPIRALITY_BINS)}
    
    for object_id, df in galaxy_data.items():
        if object_id in validation_object_ids:
            continue
        
        pspiral = pspiral_map.get(object_id)
        if pspiral is None:
            continue
        
        bin_idx = min(int(pspiral * NUM_SPIRALITY_BINS), NUM_SPIRALITY_BINS - 1)
        galaxies_by_bin[bin_idx].append((object_id, pspiral, df))
    
    for bin_idx, galaxies in galaxies_by_bin.items():
        logger.info(f"Bin {bin_idx}: {len(galaxies)} galaxies available for augmentation")
    
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
        
        samples_added = 0
        np.random.seed(42 + bin_idx)
        
        variation_pool = []
        for object_id, pspiral, df in available_galaxies:
            for idx, row in df.iterrows():
                variation_pool.append((object_id, pspiral, row))
        
        np.random.shuffle(variation_pool)
        
        for object_id, pspiral, row in variation_pool:
            if samples_added >= deficit:
                break
            
            row_dict = row.to_dict()
            row_dict['P_CW'] = pspiral
            row_dict['P_ACW'] = 0.0
            row_dict['_original_object_id'] = object_id
            augmented_rows.append(row_dict)
            samples_added += 1
        
        logger.info(f"Bin {bin_idx}: Added {samples_added} augmented samples")
    
    if not augmented_rows:
        logger.warning("No augmented rows created!")
        return pd.DataFrame()
    
    augmented_df = pd.DataFrame(augmented_rows)
    logger.info(f"Total augmented samples: {len(augmented_df)}")
    
    return augmented_df


def prepare_training_data(
    original_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    validation_object_ids: set,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare training and validation datasets."""
    logger.info("Preparing training and validation datasets")
    
    original_df['_is_validation'] = original_df['name'].astype(str).isin(validation_object_ids)
    
    train_original = original_df[~original_df['_is_validation']].copy()
    val_df = original_df[original_df['_is_validation']].copy()
    
    train_original.drop(columns=['_is_validation'], inplace=True)
    val_df.drop(columns=['_is_validation'], inplace=True)
    
    logger.info(f"Original training: {len(train_original)}, Validation: {len(val_df)}")
    
    if len(augmented_df) > 0:
        augmented_df = augmented_df.drop(columns=['_original_object_id'], errors='ignore')
        
        for col in train_original.columns:
            if col not in augmented_df.columns:
                augmented_df[col] = np.nan
        
        augmented_df = augmented_df[train_original.columns]
        
        train_df = pd.concat([train_original, augmented_df], ignore_index=True)
        logger.info(f"Combined training: {len(train_df)} (original: {len(train_original)}, augmented: {len(augmented_df)})")
    else:
        train_df = train_original
    
    return train_df, val_df


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Augment training data with blurred galaxy variations'
    )
    
    parser.add_argument('--images-dir', type=str, default=DEFAULT_IMAGES_DIR,
                        help='Path to 7000images/ folder with blurred SpArcFiRe outputs')
    parser.add_argument('--zoo-csv', type=str, default=DEFAULT_ZOO_CSV,
                        help='Path to zoo2MainSpecz.csv with P_spiral labels')
    parser.add_argument('--training-csv', type=str, default=DEFAULT_TRAINING_CSV,
                        help='Path to existing training data CSV')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save augmented data CSVs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run on small subset for testing')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Augment Data - Starting")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("*** DRY RUN MODE ***")
    
    logger.info(f"Images dir: {args.images_dir}")
    logger.info(f"Zoo CSV: {args.zoo_csv}")
    logger.info(f"Training CSV: {args.training_csv}")
    logger.info(f"Output dir: {args.output_dir}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
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
    original_df = expand_list_columns(original_df, logger)
    train_df, val_df = prepare_training_data(original_df, augmented_df, validation_ids, logger)
    
    # Step 8: Save output CSVs
    logger.info("\n" + "=" * 40)
    logger.info("STEP 8: Save augmented datasets")
    logger.info("=" * 40)
    
    train_path = Path(args.output_dir) / "data_augmented_train.csv"
    val_path = Path(args.output_dir) / "data_augmented_val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    logger.info(f"Saved training data ({len(train_df)} rows) to {train_path}")
    logger.info(f"Saved validation data ({len(val_df)} rows) to {val_path}")
    
    # Final bin distribution
    logger.info("\n" + "=" * 40)
    logger.info("FINAL BIN DISTRIBUTION (Training)")
    logger.info("=" * 40)
    train_df['P_spiral'] = train_df['P_CW'] + train_df['P_ACW']
    for i in range(NUM_SPIRALITY_BINS):
        bin_min = i / NUM_SPIRALITY_BINS
        bin_max = (i + 1) / NUM_SPIRALITY_BINS
        if i == NUM_SPIRALITY_BINS - 1:
            count = len(train_df[train_df['P_spiral'] >= bin_min])
        else:
            count = len(train_df[(train_df['P_spiral'] >= bin_min) & (train_df['P_spiral'] < bin_max)])
        logger.info(f"Bin {i} ({bin_min:.1f}-{bin_max:.1f}): {count} galaxies")
    
    logger.info("\n" + "=" * 60)
    logger.info("Augment Data - Complete")
    logger.info("=" * 60)
    logger.info(f"\nNext step: Run train_grid_search.py with:")
    logger.info(f"  python train_grid_search.py --train-csv {train_path} --val-csv {val_path}")


if __name__ == "__main__":
    main()
