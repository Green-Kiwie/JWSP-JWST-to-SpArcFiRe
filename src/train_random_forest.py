import sys
import randomforest_training_data as rf

<<<<<<< HEAD
arguments = sys.args
=======
arguments = sys.argv
>>>>>>> 766ed5d (added random forest training code)
trees_start = arguments[0]
trees_end = arguments[1]
trees_step = arguments[2]

feature_start = arguments[3]
feature_end = arguments[4]
feature_step = arguments[5]

for tree in range(trees_start, trees_end, trees_step):
    for feature in range(feature_start, feature_end, feature_step):
        training_output = rf.read_train(tree, feature)
        with open("random_forest_output.txt", 'w') as file:
            file.write(training_output)




