import pandas as pd
import sys
import traceback
from pathlib import Path
import matplotlib.pyplot as plt

sys.path[0] += ('/modules')
print(sys.path[0])

import sep_helpers as sh

try:
    arguments = sys.argv
    csv_filepath = Path(str(arguments[1]))
    image_directory = Path(str(arguments[2]))
    output_directory = Path(str(arguments[3]))
    sample_number = int(arguments[4])

    csv_data = pd.read_csv(csv_filepath)
    sample_data = csv_data.sample(n=sample_number)

    copied_files = 0
    for row in sample_data[["name", "predicted_spirality"]].itertuples(index=False):
        file, p_spiral = row
        # print(f"searching for: {file}")
        for image_path in image_directory.rglob(file+".fits"):
            # print("found file")
            fits_data = sh.get_main_fits_data(image_path)
            new_filepath = str(output_directory.parent) + "/" + output_directory.name + "/" + str(p_spiral) + "_" + file +".png"
            sh.save_fits_as_png(fits_data, new_filepath)
            
            copied_files += 1
            break

except Exception as e:
    print(f"Error: {e}")