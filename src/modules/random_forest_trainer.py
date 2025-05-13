import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from pathlib import Path

from typing import Callable

class RandomForestTrainer:
    def target_values(self) -> pd.Series:
        """returns the target values"""
        return self._full_dataset["P_spiral"]
    
    def full_dataset(self) -> pd.DataFrame:
        """returns the full target values"""
        return self._full_dataset
    
    def _get_relevant_training_values(self) -> pd.DataFrame:
        """returns only columns that are important for training"""
        return self._full_dataset.drop(columns = ["P_spiral", "P_CW", "P_ACW", "Unnamed: 0", "name", "split"])

    def train_x(self) -> pd.DataFrame:
        """returns the training dataset"""
        indexes =  self.full_dataset()[self.full_dataset()["split"] == "train"].index
        return self._transformed_dataset.loc[indexes]

    def test_x(self, percentage = 1) -> pd.DataFrame:
        """returns a certain percentage of the testing dataset, default 100%"""
        test_indices = self.full_dataset()[self.full_dataset()["split"] == "test"].index
        sampled_indices = self._full_dataset.loc[test_indices].sample(frac=percentage, random_state=42).index
        return self._transformed_dataset.loc[sampled_indices]
        
    def train_y(self) -> pd.DataFrame:
        """returns the training dataset"""
        return self.full_dataset()[self.full_dataset()["split"] == "train"]["P_spiral"]
    
    def test_y(self, percentage=1) -> pd.Series:
        """returns a certain percentage of the testing targets, aligned with test_x"""
        test_indices = self.full_dataset()[self.full_dataset()["split"] == "test"].index
        sampled_indices = self._full_dataset.loc[test_indices].sample(frac=percentage, random_state=42).index
        return self._full_dataset.loc[sampled_indices, "P_spiral"]
    
    def num_trees(self) -> int:
        """returns the number of trees to train the model"""
        return self._num_trees

    def num_features(self) -> int:
        """returns the number of features to train the model"""
        return self._num_features
    
    def training_r2(self) -> float:
        """returns the r^2 measure (accuracy measure) of the random forest model on training data"""
        return self._rf_model.score(self.train_x(), self.train_y())

    def testing_r2(self) -> float:
        """returns the r^2 measure (accuracy measure) of the random forest model on the testing data"""
        return self._rf_model.score(self.test_x(), self.test_y())
        
    def training_rmse(self) -> float:
        """returns the rmse measure (accuracy measure) of the random forest model on training data"""
        predictions = self._rf_model.predict(self.train_x())
        rmse = np.sqrt(mean_squared_error(self.train_y(), predictions))
        return rmse

    def testing_rmse(self) -> float:
        """returns the rmse measure (accuracy measure) of the random forest model on the testing data"""
        predictions = self._rf_model.predict(self.test_x())
        rmse = np.sqrt(mean_squared_error(self.test_y(), predictions))
        return rmse

    
    def summary_msg(self) -> str:
        """returns a string of format: 
        'for {num of trees} and {num of features} features, training r^2: {accuracy score}, testing r^2: {accuracy score}'"""
        return f"for {self.num_trees()} trees, {self.num_features()} features and {self._buckets} buckets, training data r^2: {self.training_r2():.2f}, testing data r^2: {self.testing_r2():.2f}, training data rmse: {self.training_rmse():.2f}, testing data rmse: {self.testing_rmse():.2f} \n"
    
    def __init__(self, num_trees: int, num_features: int, split_test_train_function: Callable = train_test_split, split_test_inputs: dict = {"random_state": 42, "test_size": 0.2, "num_buckets": 1}):
        self._num_trees = num_trees
        self._num_features = num_features
        self._buckets = split_test_inputs["num_buckets"]

        self._full_dataset = self._load_training_data(split_test_train_function, split_test_inputs)
        self._transformed_dataset, self._data_transformer = self.transform_data()

        self._rf_model = self._train_random_forest(self.train_x(), self.train_y(), num_trees, num_features)
        
        # predictions = self._rf_model.predict(self._transformed_dataset)
        # self._full_dataset["P_spiral_predicted"] = predictions

        predictions = self._rf_model.predict(self._transformed_dataset)
        self._full_dataset["P_spiral_predicted"] = pd.Series(predictions, index=self._transformed_dataset.index)
        del predictions


    def _train_random_forest(self, x_train, y_train, num_tress, num_features) -> RandomForestRegressor:
        """trains model and returns string value of training data and testing data accuracy"""
        rf = RandomForestRegressor(n_estimators=num_tress, random_state=42, max_features=num_features)
        rf.fit(x_train, y_train)
        return rf


    def _load_training_data(self, split_test_train_function: Callable, split_test_train_kwargs: dict, filepath: str = "randomforest_training_data/data.csv") -> pd.DataFrame:
        """reads csv filepath and loads into dataframe object, adds P_spiral value"""
        dataset = pd.read_csv(filepath)
        dataset.columns = [col.strip() for col in dataset.columns]
        dataset = self._get_target(dataset)
        dataset = self._split_training_testing_data(dataset, split_test_train_function, split_test_train_kwargs)
        return dataset
    
    def _split_training_testing_data(self, full_dataset, split_function: Callable, split_function_kwargs: dict) -> pd.DataFrame:
        """splits training and testing data then notates it in the original dataframe"""
        X_train, X_test, _, _ = split_function(full_dataset, full_dataset["P_spiral"], **split_function_kwargs)

        full_dataset["split"] = None
        full_dataset.loc[X_train.index, "split"] = "train"
        full_dataset.loc[X_test.index, "split"] = "test"
        return full_dataset
    
    def transform_data(self) -> tuple[pd.DataFrame, LabelEncoder]:
        """scales featureset with label encoders. Returns a dataframe of scaled featuresets"""
        categorical_cols = self._get_relevant_training_values().select_dtypes(include=["object", "category"]).columns
        encoded_data = self._get_relevant_training_values()

        le = LabelEncoder()
        for col in categorical_cols:
            encoded_data[col] = le.fit_transform(encoded_data[col])

        return encoded_data, le
    
    def save_model(self, path: Path) -> None:
        """saves model to a .pkl file"""
        if path.suffix != '.pkl':
            raise ValueError("Filepath must be a pkl file!")
        joblib.dump(self._rf_model, str(path))
        print(f"Model saved to {path}.")

    def save_label_encoder(self, path: Path) -> None:
        """saves label encoder to a .pkl file"""
        if path.suffix != '.pkl':
            raise ValueError("Filepath must be a pkl file!")

        joblib.dump(self._data_transformer, str(path))
        print(f"Label encoder saved to {path}.")
    
    @staticmethod
    def _get_target(training_data: pd.DataFrame) -> pd.Series:
        """adds p_spiral metric to training dataframe"""
        training_data['P_spiral'] = training_data["P_CW"] + training_data["P_ACW"]
        return training_data
    

def bucket_based_split_test(x_data: pd.DataFrame, y_data: pd.Series, num_buckets = 10, random_state = 42, test_size = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, None, None]:
    """function that splits data based on buckets"""
    x_test = pd.DataFrame()
    x_train = pd.DataFrame()
    print(f"total items: {len(x_data)}")

    bucket_step = 1/num_buckets
    smallest_bucket_size = _get_smallest_bucket_size(x_data, bucket_step, num_buckets)
    num_to_select = int(smallest_bucket_size*(1-test_size))

    for i in range(num_buckets):
        
        bucket_min = round(i*bucket_step, 3)
        bucket_max = round((i+1)*bucket_step, 3)
        
        if (i == num_buckets-1):
            bucket_values = x_data[(x_data["P_spiral"] >= bucket_min)]
        else:
            bucket_values = x_data[(x_data["P_spiral"] >= bucket_min) & (x_data["P_spiral"] < bucket_max)]
        print(f"bucket range: {bucket_min} to {bucket_max}, {len(bucket_values)} items")

        x_bucket_train = bucket_values.sample(n=num_to_select, random_state=random_state)
        x_bucket_test = bucket_values.drop(index=x_bucket_train.index)
        
        x_test = pd.concat([x_test, x_bucket_test], axis=0).copy()
        x_train = pd.concat([x_train, x_bucket_train], axis=0).copy()

        del x_bucket_test, x_bucket_train, bucket_values

    x_test.sort_index()
    x_train.sort_index()
    print(f"x_test size: {len(x_test)}")
    print(f"x_train size: {len(x_train)}")
    return x_train, x_test, None, None

def _get_smallest_bucket_size(dataframe: pd.DataFrame, bucket_size: float, num_buckets: int) -> int:
    """returns the smallest bucket size in the dataframe"""
    smallest_bucket_size = 99999999999
    for i in range(num_buckets):
        
        bucket_min = round(i*bucket_size, 3)
        bucket_max = round((i+1)*bucket_size, 3)

        bucket_values = dataframe[(dataframe["P_spiral"] >= bucket_min) & (dataframe["P_spiral"] < bucket_max)]
        if smallest_bucket_size > len(bucket_values):
            smallest_bucket_size = len(bucket_values)
    
    print(f"smallest bucket size: {smallest_bucket_size}")
    return smallest_bucket_size



        


    

    

