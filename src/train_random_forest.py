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

    num_buckets_start = int(arguments[7])
    num_buckets_end = int(arguments[8])
    num_buckets_step = int(arguments[9])

    filepath = arguments[10]
    
    # print(num_buckets_start)
    # print(num_buckets_end)
    # print(num_buckets_step)
    # print(list(range(num_buckets_start, num_buckets_end, num_buckets_step)))

    for tree in range(trees_start, trees_end, trees_step):
        for feature in range(feature_start, feature_end, feature_step):
            for buckets in range(num_buckets_start, num_buckets_end, num_buckets_step):
                print(f"starting training for {tree} trees, {feature} features and {buckets} buckets")
                trained_model = rf.RandomForestTrainer(tree, feature, filepath = filepath, split_test_train_function = rf.bucket_based_split_test, split_test_inputs = {"num_buckets": buckets, "random_state": 42, "test_size": 0.2})
                print(trained_model.summary_msg())
                with open(f"random_forest_output.txt", 'a') as file:
                    file.write(trained_model.summary_msg())
                
                print(trained_model.full_dataset().columns)
                trained_model.full_dataset().to_csv(f"random_forest_output/random_forest_output_{tree}_{feature}_{buckets}.csv")

                del trained_model
except Exception as e:
    print(f"error: {e}")
    traceback.print_exc()




