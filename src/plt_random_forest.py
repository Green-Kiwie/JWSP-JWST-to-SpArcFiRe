from pathlib import Path
import sys
import traceback

sys.path[0] += ('/modules')
print(sys.path[0])
import output_data_parser as dp

    
def get_plt_name(filepath: Path) -> Path:
    output = filepath.parent
    output /= filepath.stem
    return Path(output)

for file in Path("random_forest_output/").iterdir():
    if file.suffix == ".csv":
        png_file = file.with_suffix(".png")  # Change .csv -> .png
        if not png_file.exists():  # use pathlib's .exists()
            data = dp.read_csv(file)
            fig = dp.plot_fig(data)
            fig.savefig(png_file, dpi=300, bbox_inches='tight')
            print("plotted for", file)
            del data, fig
