import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from typing import Callable

class RandomForestTrainer:
    def target_values(self) -> pd.Series:
        """returns the target values"""
        return self._full_dataset["P_spiral"]
    
    def full_dataset(self) -> pd.DataFrame:
        """returns the full target values"""
        return self._full_dataset

    def train_x(self) -> pd.DataFrame:
        """returns the training dataset"""
        return self.full_dataset()[self.full_dataset()["split_type"] == "train"].drop(["P_spiral"])
    
    def test_x(self, percentage = 1) -> pd.DataFrame:
        """returns a certain percentage of the testing dataset, default 100%"""
        dataset = self.full_dataset()[self.full_dataset()["split_type"] == "test"].drop(["P_spiral"])
        return dataset.sample(frac=0.3, random_state=1)
        
    def train_y(self) -> pd.DataFrame:
        """returns the training dataset"""
        return self.full_dataset()[self.full_dataset()["split_type"] == "train"]["P_spiral"]
    
    def test_y(self, percentage = 1) -> pd.DataFrame:
        """returns a certain percentage of the testing dataset, default 100%"""
        dataset = self.full_dataset()[self.full_dataset()["split_type"] == "test"]["P_spiral"]
        return dataset.sample(frac=percentage, random_state=1)
    
    def num_trees(self) -> int:
        """returns the number of trees to train the model"""
        return self._num_trees

    def num_features(self) -> int:
        """returns the number of features to train the model"""
        return self._num_features
    
    def training_r2(self) -> float:
        """returns the r^2 measure (accuracy measure) of the random forest model on training data"""
        return self._rf_model.score(self.train_x(), self.train_y)

    def testing_r2(self) -> float:
        """returns the r^2 measure (accuracy measure) of the random forest model on the testing data"""
        return self._rf_model.score(self.test_x(), self.test_y)
        
    def summary_msg(self) -> str:
        """returns a string of format: 
        'for {num of trees} and {num of features} features, training r^2: {accuracy score}, testing r^2: {accuracy score}'"""
        return f"for {self.num_trees()} trees and {self.num_features()} features, training data accuracy: {self.training_r2():.2f}, testing data accuracy: {self.testing_r2():.2f} \n"
    
    def __init__(self, num_trees: int, num_features: int, split_test_train_function: Callable[[pd.DataFrame, pd.Series]] = train_test_split, split_test_inputs: dict = {"random_state": 1, "test_size": 0.2}):
        self._num_trees = num_trees
        self._num_features = num_features
        self._full_dataset = self._load_training_data(split_test_train_function, split_test_inputs) 
        self._transformed_dataset, self._data_transformer = self.transform_data()
        self._rf_model = self._train_random_forest(self.train_x(), self.train_y(), num_trees, num_features)


    def train_random_forest(x_train, y_train, num_tress, num_features) -> RandomForestRegressor:
        """trains model and returns string value of training data and testing data accuracy"""
        rf = RandomForestRegressor(n_estimators=num_tress, random_state=42, max_features=num_features)
        rf.fit(x_train, y_train)
        return rf

# def read_train(num_tress, num_features) -> str:
#     training_data = read_training_data()
#     feature_set, target_values = get_target_training(training_data)
#     encoded_data = transform_data(feature_set)
#     output = train_random_forest(encoded_data, target_values, num_tress, num_features)

#     return output 


    def _load_training_data(self, split_test_train_function: Callable[[pd.DataFrame, pd.Series]], split_test_train_kwargs: dict, filepath: str = "randomforest_training_data/data.csv") -> pd.DataFrame:
        """reads csv filepath and loads into dataframe object, adds P_spiral value"""
        training_data = pd.read_csv(filepath)
        training_data = self._get_target(training_data)
        self._split_training_testing_data(split_test_train_function, split_test_train_kwargs)
        return training_data
    
    def _split_training_testing_data(self, split_function: Callable, split_function_kwargs: dict) -> None:
        """splits training and testing data then notates it in the original dataframe"""
        X_train, X_test, _, _ = split_function(self.full_dataset(), self.target_values(), **split_function_kwargs)

        self.full_dataset()["split"] = None
        self.full_dataset().loc[X_train.index, "split"] = "train"
        self.full_dataset().loc[X_test.index, "split"] = "test"
    
    def transform_data(self) -> tuple[pd.DataFrame, LabelEncoder]:
        """scales featureset with label encoders. Returns a dataframe of scaled featuresets"""
        categorical_cols = self._get_relevant_training_values().select_dtypes(include=["object", "category"]).columns
        encoded_data = self._get_relevant_training_values()

        le = LabelEncoder()
        for col in categorical_cols:
            encoded_data[col] = le.fit_transform(encoded_data[col])

        return encoded_data, le
    

    @staticmethod
    def _get_target(training_data: pd.DataFrame) -> pd.Series:
        """adds p_spiral metric to training dataframe"""
        training_data['P_spiral'] = training_data["P_CW"] + training_data["P_ACW"]
        return training_data
    
    def _get_relevant_training_values(self) -> pd.DataFrame:
        """returns only columns that are important for training"""
        return self._full_dataset.drop(columns = ["P_spiral", "P_CW", "P_ACW", "Unnamed: 0", "name", "split"])
        


    

    

