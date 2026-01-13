import sys
import traceback
import numpy as np

sys.path[0] += ('/modules')
print(sys.path[0])
import xg_boost_training as xgb


try:
    arguments = sys.argv

    trees_start = int(arguments[1])
    trees_end = int(arguments[2])
    trees_step = int(arguments[3])

    features_start = float(arguments[4])
    features_end = float(arguments[5])
    features_step = float(arguments[6])

    for tree in range(trees_start, trees_end, trees_step):
        for feature in np.arange(features_start, features_end, features_step):
            print(f"starting training for {tree} trees and {feature}% of features")
            trained_model = xgb.XGBoostTrainer(tree, feature)
            print(trained_model.summary_msg())
            with open(f"xg_boost.txt", 'a') as file:
                file.write(trained_model.summary_msg())
            
            print(trained_model.full_dataset().columns)
            trained_model.full_dataset().to_csv(f"random_forest_output/xg_boos_output_{tree}_{feature}.csv")

            del trained_model
except Exception as e:
    print(f"error: {e}")
    traceback.print_exc()




