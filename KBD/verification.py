import os
from collections import OrderedDict
from concurrent.futures import as_completed, ThreadPoolExecutor

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from .constants import (
    AVG_DISP_NAME,
    GT_DIST_NAME,
    GT_ERROR_NAME,
)

from .helpers import preprocessing

from .plotters import plot_linear2
from .utils import json_reader


def verify_cpp(
    path: str,
    table_path: str,
    save_path: str,
    json_path: str,
    disjoint_depth_range: tuple,
    compensate_dist: float = 200,
    scaling_factor: float = 10,
    apply_global: bool = False,
):
    df, focal, baseline = preprocessing(path=path, table_path=table_path)

    actual_depth = df[GT_DIST_NAME].values
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
    error = df[GT_ERROR_NAME].values

    params = json_reader(json_path)
    param_matrix = np.array(params["kbd_params"])[::-1, :]

    linear_model1 = LinearRegression()
    linear_model1.coef_ = np.array([param_matrix[1, 3]])
    linear_model1.intercept_ = param_matrix[1, 4]
    linear_model2 = LinearRegression()
    linear_model2.coef_ = np.array([param_matrix[3, 3]])
    linear_model2.intercept_ = param_matrix[3, 4]

    res = (param_matrix[2, 0], param_matrix[2, 1], param_matrix[2, 2])

    plot_linear2(
        actual_depth,
        avg_50x50_anchor_disp,
        error,
        focal,
        baseline,
        (linear_model1, res, linear_model2),
        disjoint_depth_range,
        compensate_dist=compensate_dist,
        scaling_factor=scaling_factor,
        apply_global=apply_global,
        save_path=save_path,
    )


if __name__ == "__main___":
    root_dir = "D:/william/data/KBD/0723"
    camera_types = [f for f in os.listdir(root_dir)]
    disjoint_depth_ranges = [600, 3000]
    sf = 10
    cd = 400
    for camera_type in camera_types:
        base_path = os.path.join(root_dir, camera_type)
        file_path = os.path.join(base_path, "image_data")
        table_name = [f for f in os.listdir(base_path) if f.endswith('.xlsx') and os.path.isfile(os.path.join(base_path, f))][0]
        table_path = os.path.join(base_path, table_name)
        json_path = os.path.join(base_path, "segmented_linear_KBD_params_local.json")
        verify_cpp(file_path, table_path, base_path, json_path, disjoint_depth_ranges, cd, sf, False)

