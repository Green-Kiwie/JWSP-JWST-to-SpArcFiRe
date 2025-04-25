import pandas as pd
import numpy as np
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from typing import Callable

class XGBoostTrainer: 
    def target_values(self) -> pd.Series:
        return self._full_dataset["P_spiral"]
    
    def full_dataset(self) -> pd.DataFrame:
        return self._full_dataset
    
    def _get_relevant_training_values(self) -> pd.DataFrame:
        return self._full_dataset.drop(columns=["P_spiral", "P_CW", "P_ACW", "Unnamed: 0", "name", "split"])

    def train_x(self) -> pd.DataFrame:
        indexes = self.full_dataset()[self.full_dataset()["split"] == "train"].index
        return self._transformed_dataset.loc[indexes]

    def test_x(self, percentage=1) -> pd.DataFrame:
        test_indices = self.full_dataset()[self.full_dataset()["split"] == "test"].index
        sampled_indices = self._full_dataset.loc[test_indices].sample(frac=percentage, random_state=42).index
        return self._transformed_dataset.loc[sampled_indices]
        
    def train_y(self) -> pd.DataFrame:
        return self.full_dataset()[self.full_dataset()["split"] == "train"]["P_spiral"]
    
    def test_y(self, percentage=1) -> pd.Series:
        test_indices = self.full_dataset()[self.full_dataset()["split"] == "test"].index
        sampled_indices = self._full_dataset.loc[test_indices].sample(frac=percentage, random_state=42).index
        return self._full_dataset.loc[sampled_indices, "P_spiral"]
    
    def num_trees(self) -> int:
        return self._num_trees

    def num_features(self) -> int:
        return self._num_features
    
    def training_r2(self) -> float:
        return self._xgb_model.score(self.train_x(), self.train_y())

    def testing_r2(self) -> float:
        return self._xgb_model.score(self.test_x(), self.test_y())
        
    def training_rmse(self) -> float:
        predictions = self._xgb_model.predict(self.train_x())
        rmse = np.sqrt(mean_squared_error(self.train_y(), predictions))
        return rmse

    def testing_rmse(self) -> float:
        predictions = self._xgb_model.predict(self.test_x())
        rmse = np.sqrt(mean_squared_error(self.test_y(), predictions))
        return rmse

    def summary_msg(self) -> str:
        return f"for {self.num_trees()} trees, training data r^2: {self.training_r2():.10f}, testing data r^2: {self.testing_r2():.10f}, training data rmse: {self.training_rmse():.10f}, testing data rmse: {self.testing_rmse():.10f} \n"
    
    def __init__(self, num_trees: int, num_features: int, split_test_train_function: Callable = train_test_split, split_test_inputs: dict = {"random_state": 42, "test_size": 0.2}):
        self._num_trees = num_trees
        self._num_features = num_features

        self._full_dataset = self._load_training_data(split_test_train_function, split_test_inputs)
        self._transformed_dataset, self._data_transformer = self.transform_data()

        self._xgb_model = self._train_xgboost(self.train_x(), self.train_y(), num_trees, num_features) 

        predictions = self._xgb_model.predict(self._transformed_dataset)
        self._full_dataset["P_spiral_predicted"] = pd.Series(predictions, index=self._transformed_dataset.index)
        del predictions

    def _train_xgboost(self, x_train, y_train, num_trees, num_features) -> XGBRegressor: 
        """trains XGBoost model"""
        xgb = XGBRegressor(n_estimators=num_trees, random_state=42, verbosity=0) 
        #use either colsample_bytree, colsample_bylevel or colsample_bynode to simulate the num_features in random forest
        xgb.fit(x_train, y_train)
        return xgb

    def _load_training_data(self, split_test_train_function: Callable, split_test_train_kwargs: dict, filepath: str = "randomforest_training_data/data.csv") -> pd.DataFrame:
        dataset = pd.read_csv(filepath)
        dataset = self._get_target(dataset)
        dataset = self._split_training_testing_data(dataset, split_test_train_function, split_test_train_kwargs)
        return dataset
    
    def _split_training_testing_data(self, full_dataset, split_function: Callable, split_function_kwargs: dict) -> pd.DataFrame:
        X_train, X_test, _, _ = split_function(full_dataset, full_dataset["P_spiral"], **split_function_kwargs)

        full_dataset["split"] = None
        full_dataset.loc[X_train.index, "split"] = "train"
        full_dataset.loc[X_test.index, "split"] = "test"
        return full_dataset
    
    def transform_data(self) -> tuple[pd.DataFrame, LabelEncoder]:
        categorical_cols = self._get_relevant_training_values().select_dtypes(include=["object", "category"]).columns
        encoded_data = self._get_relevant_training_values()

        le = LabelEncoder()
        for col in categorical_cols:
            encoded_data[col] = le.fit_transform(encoded_data[col])

        return encoded_data, le
    
    @staticmethod
    def _get_target(training_data: pd.DataFrame) -> pd.Series:
        training_data['P_spiral'] = training_data["P_CW"] + training_data["P_ACW"]
        return training_data
    

def bucket_based_split_test(x_data: pd.DataFrame, y_data: pd.Series, num_buckets=10, random_state=42, test_size=0.2) -> tuple[pd.DataFrame, pd.DataFrame, None, None]:
    x_test = pd.DataFrame()
    x_train = pd.DataFrame()
    print(f"total items: {len(x_data)}")

    bucket_step = 1/num_buckets
    smallest_bucket_size = _get_smallest_bucket_size(x_data, bucket_step, num_buckets)
    num_to_select = int(smallest_bucket_size * (1 - test_size))

    for i in range(num_buckets):
        bucket_min = round(i * bucket_step, 3)
        bucket_max = round((i + 1) * bucket_step, 3)

        if i == num_buckets - 1:
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
    smallest_bucket_size = 99999999999
    for i in range(num_buckets):
        bucket_min = round(i * bucket_size, 3)
        bucket_max = round((i + 1) * bucket_size, 3)

        bucket_values = dataframe[(dataframe["P_spiral"] >= bucket_min) & (dataframe["P_spiral"] < bucket_max)]
        if smallest_bucket_size > len(bucket_values):
            smallest_bucket_size = len(bucket_values)

    print(f"smallest bucket size: {smallest_bucket_size}")
    return smallest_bucket_size