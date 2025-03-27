import sys

sys.path[0] += ('/modules')
print(sys.path[0])
import random_forest_trainer as rf


arguments = sys.argv

trees_start = int(arguments[1])
trees_end = int(arguments[2])
trees_step = int(arguments[3])

feature_start = int(arguments[4])
feature_end = int(arguments[5])
feature_step = int(arguments[6])

for tree in range(trees_start, trees_end, trees_step):
    for feature in range(feature_start, feature_end, feature_step):
        print(f"starting training for {tree} trees and {feature} features")
        training_output = rf.read_train(tree, feature)
        print(training_output)
        with open("random_forest_output.txt", 'a') as file:
            file.write(training_output)




