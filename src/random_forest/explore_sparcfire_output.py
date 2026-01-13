import pandas as pd
from pathlib import Path
import io
import sys
import matplotlib.pyplot as plt
import numpy as np

# Load data with low_memory=False to avoid DtypeWarning
# full_data = pd.read_csv("full_sparcfire_output.csv", low_memory=False)
full_data = pd.read_csv("randomforest_training_data/data_cleaned5.csv", low_memory=False)

full_data["P_spiral"] = full_data["P_CW"]+ full_data["P_ACW"]

# full_data['P_spiral'].plot(kind='hist', bins=10, edgecolor='black')

# counts, bin_edges = np.histogram(full_data['P_spiral'], bins=10)

# # Print counts for each bin
# for i in range(len(counts)):
#     print(f"Bin {i+1}: {bin_edges[i]:.3f} to {bin_edges[i+1]:.3f} → count = {counts[i]}")

# Define number of bins
num_bins = 5

# Cut data into bins
full_data["P_spiral_bin"] = pd.cut(full_data["P_spiral"], bins=num_bins)

# Create an empty list to hold sampled data
sampled_data = []

# Loop through each bin and sample
for bin_label, group in full_data.groupby("P_spiral_bin"):
    sample_size = 400000  #pick at least 400000 datapoints for each bin to avoid undersampling
    first_sample_size = min(sample_size, len(group))
    sampled_group1 = group.sample(first_sample_size, random_state=42)

    # after each datapoint picked at least one, do sampling with replacement.
    remaining = max(0, sample_size - len(sampled_group1))
    extra = group.sample(remaining, replace=True, random_state=42)

    sampled_group = pd.concat([sampled_group1, extra], ignore_index=True)

    
    sampled_data.append(sampled_group)

# Combine all samples
final_sample = pd.concat(sampled_data, ignore_index=True)

final_sample.drop(columns="P_spiral_bin", inplace=True)

# Save to a single CSV file
final_sample.to_csv("randomforest_training_data/data_equal_oversample.csv", index=False)

print(f"Saved {len(final_sample)} total rows to sampled_spirality_data.csv")
# # Add labels and title
# plt.xlabel('P_spiral')
# plt.ylabel('Frequency')
# plt.title('Distribution of spirality')

# # Show the plot
# plt.savefig('spiralityHistogram.jpg', format='jpg', dpi=300, bbox_inches='tight')

# # Close the figure to free memory
# plt.close()

# # Redirect output properly
# temp_output = io.StringIO()
# original_stdout = sys.stdout
# sys.stdout = temp_output

# # Basic exploration
# print("Full data info:")
# print(full_data.info())
# print("\nColumns:")
# print(full_data.columns)
# print("\nFirst few rows:")
# print(full_data.head())

# # Missing or zero values
# print("\nMissing values per column:")
# print(full_data.isna().sum())

# print("\nZero values per column:")
# print((full_data == 0).sum(numeric_only=True))

# # Value counts for key columns (cleaned column names)
# key_columns = [
#     " fit_state", " chirality_maj", " chirality_alenWtd",
#     " chirality_wtdPangSum", " chirality_longestArc"
# ]

# print("\nValue counts for key columns:")
# for col in key_columns:
#     if col in full_data.columns:
#         print(f"\nColumn: {col}")
#         print(full_data[col].value_counts(dropna=False))
#     else:
#         print(f"Column {col} not found.")

# # Restore stdout and write to file
# sys.stdout = original_stdout
# with open("sparcfire_data_explore.txt", 'w') as f:
#     f.write(temp_output.getvalue())