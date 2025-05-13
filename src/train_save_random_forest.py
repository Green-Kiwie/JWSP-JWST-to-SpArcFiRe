import sys
import traceback
from pathlib import Path

sys.path[0] += ('/modules')
import random_forest_trainer as rf

try:
    arguments = sys.argv

    trees = int(arguments[1])
    features = int(arguments[2])
    save_file = Path(str(arguments[3]))
    save_le = Path(str(arguments[4]))

    print(f"starting training for {trees} trees, {features} features")
    trained_model = rf.RandomForestTrainer(trees, features, split_test_train_function = rf.bucket_based_split_test, split_test_inputs = {"num_buckets": 1, "random_state": 42, "test_size": 0.2})
    print(trained_model.summary_msg())

    trained_model.save_model(save_file)
    del trained_model

except Exception as e:
    print(f"error: {e}")
    traceback.print_exc()