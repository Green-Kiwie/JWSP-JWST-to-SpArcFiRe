import sys
import randomforest_training_data as rf


arguments = sys.argv

trees_start = int(arguments[1])
trees_end = int(arguments[2])
trees_step = int(arguments[3])

feature_start = int(arguments[4])
feature_end = int(arguments[5])
feature_step = int(arguments[6])

for tree in range(trees_start, trees_end, trees_step):
    for feature in range(feature_start, feature_end, feature_step):
        training_output = rf.read_train(tree, feature)
        with open("random_forest_output.txt", 'w') as file:
            file.write(training_output)




