from pathlib import Path
import sys
import traceback
import matplotlib.pyplot as plt

sys.path[0] += ('/modules')
print(sys.path[0])
import output_data_parser as dp

arguments = sys.argv

# filepath = arguments[1]

    
def get_plt_name(filepath: Path) -> Path:
    output = filepath.parent
    output /= filepath.stem
    return Path(output)

for file in Path("random_forest_output/").iterdir():
    if file.suffix == ".csv":
        if file.stem == "random_forest_output_110_130_1":
            file_name = file.with_suffix("")  # remove .csv
            file_name = file_name.parent / (file_name.name + "_testing.png")
            # png_file = file.with_suffix(".png")  # Change .csv -> .png
            data = dp.read_csv(file)
            # data_testing = data[data["split"] == "test"]
            data_testing = data
            print("size: "+ str(len(data_testing)))
            fig = dp.plot_fig(data_testing)
            fig.savefig(file_name, dpi=300, bbox_inches='tight')
            print("plotted for", file)
            plt.close(fig)
            del data, fig
