import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_fig(filepath: Path) -> plt.Figure:
    data = pd.read_csv(filepath)

    x_col = 'P_spiral'
    y_col = 'P_spiral_predicted'

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[x_col], data[y_col], alpha=0.7, s=0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(True)

    return fig

def get_plt_name(filepath: Path) -> Path:
    output = filepath.parent
    output /= filepath.stem
    return Path(output)

for file in Path("random_forest_output/").iterdir():
    if file.suffix == ".csv":
        fig = plot_fig(file)
        fig.savefig(get_plt_name(file), dpi = 300, bbox_inches='tight')
        print("plotted for", file)

