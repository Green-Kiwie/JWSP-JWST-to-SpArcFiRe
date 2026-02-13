"""
Quick utility: split data_cleaned.csv into stratified 80/20 train/val sets.
Outputs: data_original_train.csv, data_original_val.csv
"""
import pandas as pd
import os

DATA_DIR = "randomforest_training_data"
INPUT_CSV = os.path.join(DATA_DIR, "data_cleaned.csv")

print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, low_memory=False)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Compute P_spiral and bin for stratified split
df["P_spiral"] = df["P_CW"] + df["P_ACW"]
df["_bin"] = pd.cut(df["P_spiral"], bins=10, labels=False)

# Stratified 20% validation split
val_idx = df.groupby("_bin", group_keys=False).apply(
    lambda x: x.sample(frac=0.2, random_state=42)
).index

train_mask = ~df.index.isin(val_idx)
val_mask = df.index.isin(val_idx)

# Drop helper columns before saving
cols_to_drop = ["P_spiral", "_bin"]
train_df = df.loc[train_mask].drop(columns=cols_to_drop)
val_df = df.loc[val_mask].drop(columns=cols_to_drop)

train_path = os.path.join(DATA_DIR, "data_original_train.csv")
val_path = os.path.join(DATA_DIR, "data_original_val.csv")

print(f"Saving train ({len(train_df)} rows) -> {train_path}")
train_df.to_csv(train_path, index=False)
print(f"Saving val ({len(val_df)} rows) -> {val_path}")
val_df.to_csv(val_path, index=False)

print("Done!")
