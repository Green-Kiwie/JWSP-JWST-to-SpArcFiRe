import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

filepath = Path("random_forest_output/to_plot.csv")

data = pd.read_csv(filepath)
print(data.columns)

x_col = 'P_spiral'
y_col = 'P_spiral_predicted'

plt.figure(figsize=(8, 6))
plt.scatter(data[x_col], data[y_col], alpha=0.7)
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(f"{y_col} vs {x_col}")
plt.grid(True)
plt.show()