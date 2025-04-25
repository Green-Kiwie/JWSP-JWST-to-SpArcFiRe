import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

def plot_fig(data: pd.DataFrame) -> plt.Figure:
    x_col = 'P_spiral'
    y_col = 'P_spiral_predicted'

    rmse = testing_rmse(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[x_col], data[y_col], alpha=0.7, s=0.2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col} \n RMSE: {rmse:.10f}")
    ax.grid(True)

    return fig

def testing_data(data: pd.DataFrame):
    """returns the data for testing"""
    test_df = data[data["split"] == "test"]
    return test_df

def testing_rmse(data: pd.DataFrame) -> float:
    """returns the rmse measure (accuracy measure) of the random forest model on the testing data"""
    predictions = data["P_spiral_predicted"]
    rmse = np.sqrt(mean_squared_error(data["P_spiral"], predictions))
    return rmse
    
def get_plt_name(filepath: Path) -> Path:
    output = filepath.parent
    output /= filepath.stem
    return Path(output)

for file in Path("random_forest_output/").iterdir():
    if file.suffix == ".csv":
        data = pd.read_csv(file)
        fig = plot_fig(data)
        fig.savefig(get_plt_name(file), dpi = 300, bbox_inches='tight')
        print("plotted for", file)
        del data, fig

