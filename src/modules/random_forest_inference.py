import pandas as pd
import joblib
from pathlib import Path

class random_forest_inferer:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")
        
        # Load the trained RandomForest model from the provided file path
        self._rf_model = joblib.load(model_path)

    def predict(self, input_data: pd.DataFrame):
        """infers using the dataframe and returns the dataframe that has a new inferred column. 
        column name is "predicted_spirality" """
        predictions = self._rf_model.predict(input_data)
        
        input_data["predicted_spirality"] = predictions
        
        return input_data