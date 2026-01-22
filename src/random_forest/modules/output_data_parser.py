import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


def read_csv(file: Path) -> pd.DataFrame:
    """reads a csv file"""
    return pd.read_csv(file)

def read_csvs(files: list[Path]) -> pd.DataFrame:
    """reads csv of multiple random tree outputs"""
    output_pd = _get_testing_data(pd.read_csv(files[0]))
    count = 0
    for filepaths in files[1:]:
        temp_pd = _get_testing_data(pd.read_csv(filepaths))
        count += 1
        new_column_name = f'P_spiral_predicted_{count}'
        temp_pd = temp_pd.rename(columns={'P_spiral_predicted': new_column_name})
        output_pd = pd.merge(output_pd, temp_pd[['name', new_column_name]], on='name', how='left')
    
    predicted_cols = [col for col in output_pd.columns if col.startswith('P_spiral_predicted')]
    output_pd["P_spiral_predicted"] = output_pd[predicted_cols].mean(axis=1)

    return output_pd

def plot_fig(data: pd.DataFrame) -> plt.Figure:
    """plts a figure of actual P-spiral against prediction p_spiral"""
    x_col = 'P_spiral'
    y_col = 'P_spiral_predicted'

    testing_rmse = get_testing_rmse(data)
    p_8_rmse = get_8_p_rmse(data)
    p_7_rmse = get_7_p_rmse(data)
    p_6_rmse = get_6_p_rmse(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[x_col], data[y_col], alpha=0.7, s=0.2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col} \n testing RMSE: {testing_rmse:.10f} \n 8 P RMSE: {p_8_rmse:.10f} \n 7 P RMSE: {p_7_rmse:.10f} \n 6 P RMSE: {p_6_rmse:.10f} ")
    ax.grid(True)

    return fig

# def plot_meta_tree_fig(data: pd.DataFrame) -> plt.Figure:
#     """Plots P_spiral vs all predicted P_spiral columns on the same scatter plot."""
#     x_col = 'P_spiral'

#     predicted_cols = [col for col in data.columns if col.startswith('P_spiral_predicted')]

#     fig, ax = plt.subplots(figsize=(8, 6))

#     testing_rmse = _get_meta_rmse(predicted_cols, data)

#     for col in predicted_cols:
#         ax.scatter(data[x_col], data[col], alpha=0.5, s=0.4, label=col)

#     ax.set_xlabel(x_col)
#     ax.set_ylabel("Predicted P_spiral")
#     ax.set_title(f"Predicted P_spiral vs Actual P_spiral \n RMSE: {testing_rmse}")
#     ax.grid(True)
#     ax.legend(markerscale=5, fontsize='small')  # Adjust legend size for small markers

#     return fig

# def _get_meta_rmse(predicted_cols: list[str], data) -> float:
#     y_pred = pd.concat([data[col] for col in predicted_cols], ignore_index=True)
#     y_true = pd.concat([data["P_spiral"]] * len(predicted_cols), ignore_index=True)
#     return np.sqrt(mean_squared_error(y_true, y_pred))

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
    

def _get_sample_data(data: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """gets only the high p_spirals to test model for bias"""
    testing_data = _get_testing_data(data)
    high_p_spiral_data = testing_data[testing_data["P_spiral_predicted"] >= threshold]
    return high_p_spiral_data

def get_8_p_rmse(data: pd.DataFrame) -> pd.DataFrame:
    """calculates rmse of high p_spirals"""
    high_p_data = _get_sample_data(data, 0.8)
    return _calculate_rmse(high_p_data)

def get_7_p_rmse(data: pd.DataFrame) -> pd.DataFrame:
    """calculates rmse of high p_spirals"""
    high_p_data = _get_sample_data(data, 0.7)
    return _calculate_rmse(high_p_data)

def get_6_p_rmse(data: pd.DataFrame) -> pd.DataFrame:
    """calculates rmse of high p_spirals"""
    high_p_data = _get_sample_data(data, 0.6)
    return _calculate_rmse(high_p_data)