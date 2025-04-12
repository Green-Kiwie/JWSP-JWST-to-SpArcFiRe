import sys
import traceback

sys.path[0] += ('/modules')
print(sys.path[0])
import random_forest_trainer as rf


try:
    arguments = sys.argv

    trees_start = int(arguments[1])
    trees_end = int(arguments[2])
    trees_step = int(arguments[3])

    feature_start = int(arguments[4])
    feature_end = int(arguments[5])
    feature_step = int(arguments[6])

    num_buckets_input = int(arguments[7])

    for tree in range(trees_start, trees_end, trees_step):
        for feature in range(feature_start, feature_end, feature_step):
            print(f"starting training for {tree} trees and {feature} features")
            trained_model = rf.RandomForestTrainer(tree, feature, split_test_train_function = rf.bucket_based_split_test, split_test_inputs = {"num_buckets": num_buckets_input})
            print(trained_model.summary_msg())
            with open(f"random_forest_output.txt", 'a') as file:
                file.write(trained_model.summary_msg())
            
            print(trained_model.full_dataset().columns)
            trained_model.full_dataset().to_csv(f"random_forest_output/random_forest_output_{tree}_{feature}.csv")

            del trained_model
except Exception as e:
    print(f"error: {e}")
    traceback.print_exc()




