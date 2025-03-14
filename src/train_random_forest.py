import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys

with open('randomforest_training_data/output.txt', 'w') as sys.stdout:
    training_data = pd.read_csv("randomforest_training_data/data.csv")
    training_data["P_spiral"] = training_data["P_CW"] + training_data["P_ACW"]
    training_values = training_data.drop(columns = ["P_spiral", "P_CW", "P_ACW", "Unnamed: 0", "name"])
    target_value = training_data["P_spiral"]

    #encode columns
    categorical_cols = training_values.select_dtypes(include=["object", "category"]).columns
    print(list(categorical_cols))

    encoded_data = training_values
    le = LabelEncoder()
    for col in categorical_cols:
        encoded_data[col] = le.fit_transform(encoded_data[col])

    #splitting training and testing data
    X_train, X_test, y_train, y_test = train_test_split(encoded_data, target_value, test_size=0.2, random_state=42)
    

    rf = RandomForestRegressor(n_estimators=150, random_state=42, max_features=40)
    rf.fit(X_train, y_train)


    training_accuracy = rf.score(X_train, y_train)
    print(f"model accuracy with training data: {training_accuracy:.2f}")

    accuracy = rf.score(X_test, y_test)
    print(f"Model Accuracy with unseen: {accuracy:.2f}")

