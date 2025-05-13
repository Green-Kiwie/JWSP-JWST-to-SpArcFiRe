import sys
import traceback
from pathlib import Path
import pandas as pd

sys.path[0] += ('/modules')
import random_forest_inference as rfi

try:
    arguments = sys.argv

    data_filepath = Path(str(arguments[1]))
    output_filepath = Path(str(arguments[2]))
    model_filepath = Path(str(arguments[3]))
    le_filepath = Path(str(arguments[3]))

    input_data = pd.read_csv(data_filepath)

    inferer = rfi.random_forest_inferer(model_filepath, le_filepath)

    result_df = inferer.predict(input_data)

    result_df.to_csv(output_filepath)
    print(f"output saved to {output_filepath}")

except Exception as e:
    print(f"error: {e}")
    traceback.print_exc()