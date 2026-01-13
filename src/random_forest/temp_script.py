import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np


sys.path[0] += ('/modules')
import reading_sparcfire as sparcfire
import random_forest_trainer as rf

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

try:

    # --- SETTINGS ---
    pred_csv = "test_bayonet_prediction.csv"   # CSV containing predicted values
    true_csv = "actual.csv"        # CSV containing actual/true values

    pred_col = "predicted"         # column name in predictions.csv
    true_col = "actual"            # column name in actual.csv

    # --- LOAD FILES ---
    df_pred = pd.read_csv(pred_csv)
    df_true = pd.read_csv(true_csv)

    # --- CHECK COLUMNS EXIST ---
    if pred_col not in df_pred.columns:
        raise ValueError(f"Column '{pred_col}' not found in {pred_csv}.")
    if true_col not in df_true.columns:
        raise ValueError(f"Column '{true_col}' not found in {true_csv}.")

    # --- EXTRACT VALUES ---
    y_pred = df_pred[pred_col].astype(float).values
    y_true = df_true[true_col].astype(float).values

    # --- CHECK LENGTH MATCH ---
    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Length mismatch: predicted has {len(y_pred)} rows, "
            f"actual has {len(y_true)} rows."
        )

    # --- COMPUTE RMSE ---
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    print(f"RMSE between actual ({true_col}) and predicted ({pred_col}): {rmse:.5f}")

    #

    # # test_data = pd.read_csv(infer_filepath)
    # # print(test_data.columns)
    # # for col in test_data.columns:
    # #     print(f"{col}: {test_data[col].dtype}")

    # training_data = sparcfire.load_training_data(data_filepath)
    # print(", ".join(training_data.columns))
    # print(training_data[["covarFit_0"]].head(10))


    # # input_data = sparcfire.load_training_data(data_filepath)
    # infer_data = sparcfire.load_inference_data(infer_filepath)
    # print(", ".join(infer_data.columns))
    # print(infer_data[["covarFit_0", "covarFit_1"]].head(10))

    # trained_model = rf.RandomForestTrainer(150, 110, filepath = data_filepath, split_test_train_function = rf.bucket_based_split_test, split_test_inputs = {"num_buckets": 1, "random_state": 42, "test_size": 0.2})

    # print("Input Data Columns and Types:")
    # for col in input_data.columns:
    #     print(f"{col}: {input_data[col].dtype}")

    # print("\nInference Data Columns and Types:")
    # for col in infer_data.columns:
    #     print(f"{col}: {infer_data[col].dtype}")

    # print(input_data.head(5))
    # print(infer_data.head(5))


except Exception as e:
    print(f"error: {e}")
    traceback.print_exc()
