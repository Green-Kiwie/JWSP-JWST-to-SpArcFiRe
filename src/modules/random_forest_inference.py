import pandas as pd
import joblib
from pathlib import Path

import numpy as np

class random_forest_inferer:
    def __init__(self, model_path: Path, le_path: Path, data: pd.DataFrame):
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")
        if not le_path.exists():
            raise ValueError(f"Label encoder file not found: {le_path}")

        
        self._rf_model = joblib.load(model_path)
        self._le_model = joblib.load(le_path)
        self._data = data

    def _print_debug(self) -> None:
        """prints issues between data to be inferred and the training data"""
        input_data = self._data
        input_data.columns = [col.strip() for col in input_data.columns]
        input_data_filtered = input_data[self._rf_model.feature_names_in_].copy()

        print(list(self._rf_model.feature_names_in_))
        print(list(input_data_filtered))

        model_features = set(self._rf_model.feature_names_in_)
        input_features = set(input_data_filtered.columns)

        missing_from_input = model_features - input_features
        print("In model but not in input:", list(missing_from_input))

        unexpected_in_input = input_features - model_features
        print("In input but not in model:", list(unexpected_in_input))


        for col, encoder in self._le_model.items():
            
            if type(input_data_filtered[col][0]) != str:
                nan_count = input_data_filtered[col].isna().sum()
                if nan_count > 0:
                    print(f"float value column {col}:")
                    print(f"  - NaN: {nan_count} occurrence(s)")
                continue

            # Drop nulls before comparing to encoder classes
            unique_values = input_data_filtered[col].dropna().unique()
            unseen_values = set(unique_values) - set(encoder.classes_)

            if unseen_values:
                unseen_counts = input_data_filtered[col].value_counts().loc[list(unseen_values)]
                print(f"Unseen labels in column '{col}':")
                for value, count in unseen_counts[:10].items():
                    print(f"  - {value!r}: {count} occurrence(s)")

            nan_count = input_data_filtered[col].isna().sum()
            if nan_count > 0:
                print(f"  - NaN: {nan_count} occurrence(s)")

            print(f"    potential corresponding data on {col}: ", list(set(encoder.classes_) - set(unique_values))[:10])
        
    def predict(self) -> pd.DataFrame:
        """Infers using the dataframe and returns the dataframe with a new inferred column.
        Column name is "predicted_spirality"."""  
        input_data = self._data
        input_data.columns = [col.strip() for col in input_data.columns]

        self._print_debug()

        input_data_filtered = input_data[self._rf_model.feature_names_in_].copy()

        for col, encoder in self._le_model.items():
            input_data_filtered[col] = encoder.transform(input_data_filtered[col])

        # for col in input_data_filtered.columns:
        #     if (input_data_filtered[col].isin([np.inf, -np.inf]).any()):
        #         print(f"data is np.inf or -np.inf in column {col}")
        
        predictions = self._rf_model.predict(input_data_filtered)
        input_data["predicted_spirality"] = predictions

        return input_data

    def list_model_features(self) -> list:
        """Lists all features used in the Random Forest model."""
        return list(self._rf_model.feature_names_in_)
    
    def list_data_features(self) -> list:
        """Lists all data used in the Random Forest model."""
        return list(self._data.columns)
    
    