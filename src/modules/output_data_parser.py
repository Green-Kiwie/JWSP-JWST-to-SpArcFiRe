import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


def read_csv(file: Path) -> pd.DataFrame:
    """reads a csv file"""
    return pd.read_csv(file)

def plot_fig(data: pd.DataFrame) -> plt.Figure:
    """plts a figure of actual P-spiral against prediction p_spiral"""
    x_col = 'P_spiral'
    y_col = 'P_spiral_predicted'

    testing_rmse = get_testing_rmse(data)
    high_p_rmse = get_high_p_rmse(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[x_col], data[y_col], alpha=0.7, s=0.2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col} \n testing RMSE: {testing_rmse:.10f} \n high P RMSE: {high_p_rmse:.10f}")
    ax.grid(True)

    return fig

def _get_testing_data(data: pd.DataFrame) -> pd.DataFrame:
    """returns the data for testing"""
    test_df = data[data["split"] == "test"]
    return test_df

def _calculate_rmse(data: pd.DataFrame) -> float:
    """calculates the rmse value on the dataframe"""
    predictions = data["P_spiral_predicted"]
    rmse = np.sqrt(mean_squared_error(data["P_spiral"], predictions))
    return rmse

def get_testing_rmse(data: pd.DataFrame) -> float:
    """returns the rmse measure (accuracy measure) of the random forest model on the testing data"""
    testing_data = _get_testing_data(data)
    return _calculate_rmse(testing_data)
    

def _get_sample_data(data: pd.DataFrame) -> pd.DataFrame:
    """gets only the high p_spirals to test model for bias"""
    testing_data = _get_testing_data(data)
    high_p_spiral_data = testing_data[testing_data["P_spiral"] >= 0.8]
    return high_p_spiral_data

def get_high_p_rmse(data: pd.DataFrame) -> pd.DataFrame:
    """calculates rmse of high p_spirals"""
    high_p_data = _get_sample_data(data)
    return _calculate_rmse(high_p_data)