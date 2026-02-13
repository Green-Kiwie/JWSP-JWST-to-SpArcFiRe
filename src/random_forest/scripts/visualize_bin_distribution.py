#!/usr/bin/env python3
"""
Visualize P_spiral Bin Distribution Before and After Augmentation

Creates bar charts showing:
1. Original training data distribution 
2. After augmentation with blurred galaxy variations
3. Side-by-side comparison

Usage:
    python visualize_bin_distribution.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
ORIGINAL_CSV = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/randomforest_training_data/data_cleaned5.csv"
AUGMENTED_CSV = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/randomforest_training_data/data_augmented_train.csv"
OUTPUT_DIR = "/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/random_forest/random_forest_output"

NUM_BINS = 10

def calculate_bin_distribution(df: pd.DataFrame, label: str) -> tuple:
    """Calculate P_spiral bin distribution."""
    print(f"\nAnalyzing {label} data:")
    print(f"Total samples: {len(df):,}")
    
    # Calculate P_spiral
    df['P_spiral'] = df['P_CW'] + df['P_ACW']
    
    bins = []
    counts = []
    
    for i in range(NUM_BINS):
        bin_min = i / NUM_BINS
        bin_max = (i + 1) / NUM_BINS
        
        if i == NUM_BINS - 1:
            mask = df['P_spiral'] >= bin_min
        else:
            mask = (df['P_spiral'] >= bin_min) & (df['P_spiral'] < bin_max)
        
        count = np.sum(mask)
        bins.append(f"Bin {i}\n({bin_min:.1f}-{bin_max:.1f})")
        counts.append(count)
        
        print(f"  Bin {i} ({bin_min:.1f}-{bin_max:.1f}): {count:,} galaxies")
    
    return bins, counts

def main():
    print("Loading datasets...")
    
    # Load original data
    print(f"Loading original data from {ORIGINAL_CSV}")
    original_df = pd.read_csv(ORIGINAL_CSV)
    original_bins, original_counts = calculate_bin_distribution(original_df, "Original")
    
    # Load augmented data
    print(f"\nLoading augmented data from {AUGMENTED_CSV}")
    augmented_df = pd.read_csv(AUGMENTED_CSV)
    augmented_bins, augmented_counts = calculate_bin_distribution(augmented_df, "Augmented")
    
    # Check if augmented data exists
    if not Path(AUGMENTED_CSV).exists():
        print(f"ERROR: Augmented data file not found: {AUGMENTED_CSV}")
        print("Please run 'python augment_data_only.py' first")
        return
    
    # Create visualization
    print(f"\nCreating visualizations...")
    
    # Figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Colors
    original_color = '#2E86AB'
    augmented_color = '#A23B72'
    
    x_positions = np.arange(NUM_BINS)
    
    # Plot 1: Original distribution
    axes[0].bar(x_positions, original_counts, color=original_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Original Training Data\nP_spiral Bin Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('P_spiral Bins', fontsize=12)
    axes[0].set_ylabel('Number of Galaxies', fontsize=12)
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels([f'Bin {i}\n({i/10:.1f}-{(i+1)/10:.1f})' for i in range(NUM_BINS)], rotation=0)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(original_counts):
        axes[0].text(i, count + max(original_counts) * 0.01, f'{count:,}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Augmented distribution
    axes[1].bar(x_positions, augmented_counts, color=augmented_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1].set_title('After Augmentation with Blurred Galaxies\nP_spiral Bin Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('P_spiral Bins', fontsize=12)
    axes[1].set_ylabel('Number of Galaxies', fontsize=12)
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([f'Bin {i}\n({i/10:.1f}-{(i+1)/10:.1f})' for i in range(NUM_BINS)], rotation=0)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(augmented_counts):
        axes[1].text(i, count + max(augmented_counts) * 0.01, f'{count:,}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Side-by-side comparison
    width = 0.35
    x1 = x_positions - width/2
    x2 = x_positions + width/2
    
    axes[2].bar(x1, original_counts, width, label='Original', color=original_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[2].bar(x2, augmented_counts, width, label='Augmented', color=augmented_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[2].set_title('Before vs After Augmentation\nComparison', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('P_spiral Bins', fontsize=12)
    axes[2].set_ylabel('Number of Galaxies', fontsize=12)
    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels([f'Bin {i}' for i in range(NUM_BINS)], rotation=0)
    axes[2].legend(fontsize=12)
    axes[2].grid(axis='y', alpha=0.3)
    
    # Calculate and display augmentation statistics
    total_original = sum(original_counts)
    total_augmented = sum(augmented_counts)
    added_samples = total_augmented - total_original
    
    # Add summary statistics as text
    stats_text = f"""Summary Statistics:
Original: {total_original:,} samples
Augmented: {total_augmented:,} samples  
Added: +{added_samples:,} samples (+{added_samples/total_original*100:.1f}%)

Most balanced bin: {max(original_counts):,} → {max(augmented_counts):,}
Least balanced bin: {min(original_counts):,} → {min(augmented_counts):,}"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for stats text
    
    # Save plot
    output_path = Path(OUTPUT_DIR) / "pspiral_bin_distribution_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # Also create a simple histogram showing P_spiral distribution
    plt.figure(figsize=(12, 8))
    
    # Load P_spiral values
    original_pspiral = original_df['P_CW'] + original_df['P_ACW']
    augmented_pspiral = augmented_df['P_CW'] + augmented_df['P_ACW']
    
    plt.subplot(2, 1, 1)
    plt.hist(original_pspiral, bins=50, alpha=0.7, color=original_color, edgecolor='black', linewidth=0.5)
    plt.title('Original P_spiral Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('P_spiral Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.hist(augmented_pspiral, bins=50, alpha=0.7, color=augmented_color, edgecolor='black', linewidth=0.5)
    plt.title('Augmented P_spiral Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('P_spiral Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save histogram
    hist_path = Path(OUTPUT_DIR) / "pspiral_histogram_comparison.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"Saved P_spiral histogram to: {hist_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("BIN DISTRIBUTION ANALYSIS COMPLETE")
    print("="*60)
    print(f"Visualizations saved to:")
    print(f"  - {output_path}")
    print(f"  - {hist_path}")

if __name__ == "__main__":
    main()