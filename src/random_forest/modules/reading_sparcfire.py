import pandas as pd
from pathlib import Path
import numpy as np
import re

dtype = {
    "name": "string",
    "fit_state": "string",
    "warnings": "string",
    "star_mask_used": "string",
    "noise_mask_used": "string",
    "chirality_maj": "string",
    "chirality_alenWtd": "string",
    "chirality_wtdPangSum": "string",
    "chirality_longestArc": "string",
    "badBulgeFitFlag": "bool",
    "bulgeAxisRatio": "float64",
    "bulgeMajAxsLen": "float64",
    "bulgeMajAxsAngle": "float64",
    "bulgeAvgBrt": "float64",
    "bulgeDiskBrtRatio": "float64",
    "numElpsRefits": "int64",
    "diskAxisRatio": "float64",
    "diskMinAxsLen": "float64",
    "diskMajAxsLen": "float64",
    "diskMajAxsAngleRadians": "float64",
    "inputCenterR": "float64",
    "inputCenterC": "float64",
    "iptSz": "string",
    "muDist": "float64",
    "muDistProp": "float64",
    "covarFit": "string",
    "wtdLik": "float64",
    "likOfCtr": "float64",
    "brtUnifScore": "float64",
    "gaussLogLik": "float64",
    "contourBrtRatio": "float64",
    "standardizedCenterR": "float64",
    "standardizedCenterC": "float64",
    'hasDeletedCtrClus': "bool",
    "failed2revDuringMergeCheck": "bool",
    "failed2revDuringSecondaryMerging": "bool",
    "failed2revInOutput": "bool",
    "cpu_time": "float64",
    "wall_time": "float64",
    "bar_candidate_available": "bool",
    "bar_used": "bool",
    "alenAt25pct": "float64",
    "alenAt50pct": "float64",
    "alenAt75pct": "float64",
    "rankAt25pct": "float64",
    "rankAt50pct": "float64",
    "rankAt75pct": "float64",
    "avgArcLength": "float64",
    "minArcLength": "float64",
    "lowerQuartileArcLength": "float64",
    "medianArcLength": "float64",
    "upperQuartileArcLength": "float64",
    "maxArcLength": "float64",
    "totalArcLength": "float64",
    "totalNumArcs": "int64",
    "chirality_votes_maj": "string",
    "chirality_votes_alenWtd": "string",
    "alenWtdPangSum": "float64",
    "top2_chirality_agreement": "string",
    "pa_longest": "float64",
    "pa_avg": "float64",
    "pa_avg_domChiralityOnly": "float64",
    "paErr_alenWtd_stdev_domChiralityOnly": "float64",
    "pa_alenWtd_avg": "float64",
    "paErr_alenWtd_stdev": "float64",
    "pa_alenWtd_avg_abs": "float64",
    "paErr_alenWtd_stdev_abs": "float64",
    "pa_alenWtd_avg_domChiralityOnly": "float64",
    "pa_totBrtWtd": "float64",
    "pa_avgBrtWtd": "float64",
    "pa_alenWtd_median": "float64",
    "pa_alenWtd_lowQuartile": "float64",
    "pa_alenWtd_highQuartile": "float64",
    "pa_alenWtd_lowDecile": "float64",
    "pa_alenWtd_highDecile": "float64",
    "pa_alenWtd_median_domChiralityOnly": "float64",
    "pa_alenWtd_lowQuartile_domChiralityOnly": "float64",
    "pa_alenWtd_highQuartile_domChiralityOnly": "float64",
    "pa_alenWtd_lowDecile_domChiralityOnly": "float64",
    "pa_alenWtd_highDecile_domChiralityOnly": "float64",
    "sorted_agreeing_pangs": "string",
    "sorted_agreeing_arclengths": "string",
    "numArcs_largest_length_gap": "int64",
    "numDcoArcs_largest_length_gap": "int64",
    "numArcs_arclength_function_flattening": "int64",
    "numDcoArcs_arclength_function_flattening": "int64",
    "bar_score_img": "float64",
    "bar_cand_score_orifld": "float64",
    "bar_angle_input_img": "float64",
    "bar_half_length_input_img": "float64",
    "bar_angle_standardized_img": "float64",
    "bar_half_length_standardized_img": "float64",
    "numArcsGE000": "float64",
    "numArcsGE010": "float64",
    "numArcsGE020": "float64",
    'numArcsGE040': "int64",
    "numArcsGE050": "int64",
    "numArcsGE055": "int64",
    "numArcsGE060": "int64",
    "numArcsGE065": "int64",
    "numArcsGE070": "int64",
    "numArcsGE075": "int64",
    "numArcsGE080": "int64",
    "numArcsGE085": "int64",
    "numArcsGE090": "int64",
    "numArcsGE095": "int64",
    "numArcsGE100": "int64",
    "numArcsGE120": "int64",
    "numArcsGE140": "int64",
    "numArcsGE160": "int64",
    "numArcsGE180": "int64",
    "numArcsGE200": "int64",
    "numArcsGE220": "int64",
    "numArcsGE240": "int64",
    "numArcsGE260": "int64",
    "numArcsGE280": "int64",
    "numArcsGE300": "int64",
    "numArcsGE350": "int64",
    "numArcsGE400": "int64",
    "numArcsGE450": "int64",
    "numArcsGE500": "int64",
    "numArcsGE550": "int64",
    "numArcsGE600": "int64",
    "numDcoArcsGE000": "int64",
    "numDcoArcsGE010": "int64",
    "numDcoArcsGE020": "int64",
    "numDcoArcsGE040": "int64",
    "numDcoArcsGE050": "int64",
    "numDcoArcsGE055": "int64",
    "numDcoArcsGE060": "int64",
    "numDcoArcsGE065": "int64",
    "numDcoArcsGE070": "int64",
    "numDcoArcsGE075": "int64",
    "numDcoArcsGE080": "int64",
    "numDcoArcsGE085": "int64",
    "numDcoArcsGE090": "int64",
    "numDcoArcsGE095": "int64",
    "numDcoArcsGE100": "int64",
    "numDcoArcsGE120": "int64",
    "numDcoArcsGE140": "int64",
    "numDcoArcsGE160": "int64",
    "numDcoArcsGE180": "int64",
    "numDcoArcsGE200": "int64",
    "numDcoArcsGE220": "int64",
    "numDcoArcsGE240": "int64",
    "numDcoArcsGE260": "int64",
    "numDcoArcsGE280": "int64",
    "numDcoArcsGE300": "int64",
    "numDcoArcsGE350": "int64",
    "numDcoArcsGE400": "int64",
    "numDcoArcsGE450": "int64",
    "numDcoArcsGE500": "int64",
    "numDcoArcsGE550": "int64",
    "numDcoArcsGE600": "int64",
}

def _parse_str_to_np(s: str) -> np.array:
    """used to parse data when loading data"""
    # print("s:" + s)
    s = re.sub(r'[\[\]]', '', s)
    # print("s stripped:" + s)
    # print(np.fromstring(s, sep=' ', dtype=float))
    # exit()
    return np.fromstring(s, sep=' ', dtype=float)

def _load_data(filepath: Path) -> pd.DataFrame:
    """loads a csv according to format"""
    main_data = pd.read_csv(filepath, dtype=dtype)

    main_data.columns = main_data.columns.str.strip()

    list_cols = [
        "iptSz", "covarFit", "chirality_votes_maj",
        "chirality_votes_alenWtd", "sorted_agreeing_pangs", "sorted_agreeing_arclengths"
    ]

    for col in list_cols:
        if col == "covarFit":
            main_data[col] = main_data[col].apply(_parse_str_to_np)

            main_data[col] = main_data[col].apply(
                lambda x: list(x)[:7] + [None] * (7 - len(x)) if isinstance(x, (list, np.ndarray)) else [None] * 7
            )
            expanded_cols = pd.DataFrame(main_data[col].tolist(), columns=[f"{col}_{i}" for i in range(7)])
    
        else:
            main_data[col] = main_data[col].apply(_parse_str_to_np)
            expanded_cols = main_data[col].apply(pd.Series)
            expanded_cols.columns = [f"{col}_{i}" for i in expanded_cols.columns]

        main_data = pd.concat([main_data, expanded_cols], axis=1)
        main_data.drop(columns=[col], inplace=True)

    
    print("columns")
    print(list(main_data.columns))

    return main_data


def load_training_data(data_path: Path = "randomforest_training_data/data_cleaned.csv") -> pd.DataFrame:
    """loads path based on hardcoded datatypes"""
    return _load_data(data_path)

def load_inference_data(data_path: Path) -> pd.DataFrame:
    """
    Loads data with columns based on training data column types
    """
    return _load_data(data_path)
    
    

