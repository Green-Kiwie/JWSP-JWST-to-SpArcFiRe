import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def read_training_data(filepath: str = "randomforest_training_data/data.csv") -> pd.DataFrame:
    """reads csv filepath and loads into dataframe object"""
    training_data = pd.read_csv(filepath)
    return training_data

def get_target_training(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """split dataset into features and y values"""
    data["P_spiral"] = data["P_CW"] + data["P_ACW"]
    training_values = data.drop(columns = ["P_spiral", "P_CW", "P_ACW", "Unnamed: 0", "name"])
    target_values = data["P_spiral"]
    return training_values, target_values

def transform_data(feature_set: pd.DataFrame) -> pd.DataFrame:
    """scales featureset with label encoders. Returns a dataframe of scaled featuresets"""
    categorical_cols = feature_set.select_dtypes(include=["object", "category"]).columns
    encoded_data = feature_set

    le = LabelEncoder()
    for col in categorical_cols:
        encoded_data[col] = le.fit_transform(encoded_data[col])

    return encoded_data

def train_random_forest(encoded_data, target_value, num_tress, num_features) -> str:
    """trains model and returns string value of training data and testing data accuracy"""
    X_train, X_test, y_train, y_test = train_test_split(encoded_data, target_value, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=num_tress, random_state=42, max_features=num_features)
    rf.fit(X_train, y_train)
    training_accuracy = rf.score(X_train, y_train)
    testing_accuracy = rf.score(X_test, y_test)

    output_string = f"for {num_tress} trees and {num_features} features, training data accuracy: {training_accuracy:.2f}, testing data accuracy: {testing_accuracy:.2f} \n"
    return output_string

def read_train(num_tress, num_features) -> str:
    training_data = read_training_data()
    feature_set, target_values = get_target_training(training_data)
    encoded_data = transform_data(feature_set)
    output = train_random_forest(encoded_data, target_values, num_tress, num_features)

    return output 