import sys
import traceback
from pathlib import Path
import pandas as pd

sys.path[0] += ('/modules')
import reading_sparcfire as sparcfire
import random_forest_trainer as rf

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

try:
    arguments = sys.argv

    data_filepath = Path(str(arguments[1]))
    infer_filepath = Path(str(arguments[2]))

    test_data = pd.read_csv(infer_filepath)
    print(test_data.columns)
    for col in test_data.columns:
        print(f"{col}: {test_data[col].dtype}")

    # input_data = sparcfire.load_training_data(data_filepath)
    infer_data = sparcfire.load_inference_data(infer_filepath, data_filepath)

    trained_model = rf.RandomForestTrainer(150, 110, filepath = data_filepath, split_test_train_function = rf.bucket_based_split_test, split_test_inputs = {"num_buckets": 1, "random_state": 42, "test_size": 0.2})

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
