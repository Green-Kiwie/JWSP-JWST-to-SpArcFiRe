import pandas as pd
from pathlib import Path
import io
import sys

# Load data with low_memory=False to avoid DtypeWarning
# full_data = pd.read_csv("full_sparcfire_output.csv", low_memory=False)
full_data = pd.read_csv("randomforest_training_data/data.csv", low_memory=False)

# Redirect output properly
temp_output = io.StringIO()
original_stdout = sys.stdout
sys.stdout = temp_output

# Basic exploration
print("Full data info:")
print(full_data.info())
print("\nColumns:")
print(full_data.columns)
print("\nFirst few rows:")
print(full_data.head())

# Missing or zero values
print("\nMissing values per column:")
print(full_data.isna().sum())

print("\nZero values per column:")
print((full_data == 0).sum(numeric_only=True))

# Value counts for key columns (cleaned column names)
key_columns = [
    " fit_state", " chirality_maj", " chirality_alenWtd",
    " chirality_wtdPangSum", " chirality_longestArc"
]

print("\nValue counts for key columns:")
for col in key_columns:
    if col in full_data.columns:
        print(f"\nColumn: {col}")
        print(full_data[col].value_counts(dropna=False))
    else:
        print(f"Column {col} not found.")

# Restore stdout and write to file
sys.stdout = original_stdout
with open("sparcfire_data_explore.txt", 'w') as f:
    f.write(temp_output.getvalue())