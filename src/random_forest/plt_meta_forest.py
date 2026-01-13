from pathlib import Path
import sys
import traceback

sys.path[0] += ('/modules')
print(sys.path[0])
import output_data_parser as dp

meta_tree_path = "random_forest_output/random_forest_output_"
meta_tree_path_end = "_1.csv"

files = [Path(f"{meta_tree_path}150_110{meta_tree_path_end}"),
         Path(f"{meta_tree_path}150_90{meta_tree_path_end}"),
         Path(f"{meta_tree_path}150_70{meta_tree_path_end}"),
         Path(f"{meta_tree_path}150_50{meta_tree_path_end}"),
         Path(f"{meta_tree_path}150_30{meta_tree_path_end}"),
         Path(f"{meta_tree_path}130_110{meta_tree_path_end}"),
         Path(f"{meta_tree_path}130_90{meta_tree_path_end}"),
         Path(f"{meta_tree_path}130_70{meta_tree_path_end}"),
         Path(f"{meta_tree_path}130_50{meta_tree_path_end}")]

data = dp.read_csvs(files)
fig = dp.plot_fig(data)
fig.savefig("meta_tree.png", dpi=300, bbox_inches='tight')
