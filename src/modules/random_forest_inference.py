import pandas as pd
import joblib
from pathlib import Path

class random_forest_inferer:
    def __init__(self, model_path: Path, le_path: Path):
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")
        if not le_path.exists():
            raise ValueError(f"Label encoder file not found: {le_path}")

        
        self._rf_model = joblib.load(model_path)
        self._le_model = joblib.load(le_path)

    # def predict(self, input_data: pd.DataFrame):
    #     """infers using the dataframe and returns the dataframe that has a new inferred column. 
    #     column name is "predicted_spirality" """
    #     input_data.columns = [col.strip() for col in input_data.columns]
    #     input_data_filtered = input_data[self._rf_model.feature_names_in_]

    #     predictions = self._rf_model.predict(input_data_filtered)
        
    #     input_data["predicted_spirality"] = predictions
        
    #     return input_data
        
    def predict(self, input_data: pd.DataFrame):
        """Infers using the dataframe and returns the dataframe with a new inferred column.
        Column name is "predicted_spirality"."""    
        input_data.columns = [col.strip() for col in input_data.columns]

        input_data_filtered = input_data[self._rf_model.feature_names_in_].copy()

        for col in input_data_filtered.columns:
            if col in self._le_model.classes_:

                input_data_filtered[col] = self._le_model.transform(input_data_filtered[col])
        
        predictions = self._rf_model.predict(input_data_filtered)
        input_data["predicted_spirality"] = predictions

        return input_data