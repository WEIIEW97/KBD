import os
from collections import OrderedDict
from concurrent.futures import as_completed, ThreadPoolExecutor

import numpy as np

from KBD.constants import *

from KBD.helpers import preprocessing

from KBD.plotters import plot_linear2
from KBD.utils import json_reader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def verify_cpp(
    path: str,
    table_path: str,
    save_path: str,
    json_path: str,
    scaling_factor: float = 10,
    apply_global: bool = False,
):
    df, focal, baseline = preprocessing(path=path, table_path=table_path)

    actual_depth = df[GT_DIST_NAME].values
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
    error = (df[AVG_DIST_NAME]-df[GT_DIST_NAME]).values

    params = json_reader(json_path)
    if len(params) != 4:
        raise ValueError("Don't have matched keys to be parsed!")
    param_matrix = np.array(params["kbd_params"])[::-1, :]

    range_start = params["optimal_range_start"]
    cd = params["optimal_cd"]

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
        (int(range_start), 3000),
        compensate_dist=cd,
        scaling_factor=scaling_factor,
        apply_global=apply_global,
        save_path=save_path,
    )


if __name__ == "__main__":
    root_dir = "/home/william/extdisk/data/KBD"
    camera_types = [
        f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))
    ]
    sf = 10
    for camera_type in camera_types:
        print(f"processing {camera_type} ...")
        if camera_type != "N09ASH24DH0082":
            continue
        base_path = os.path.join(root_dir, camera_type)
        fig_path = os.path.join(base_path, "cpp_verify")
        os.makedirs(fig_path, exist_ok=True)
        file_path = os.path.join(base_path, "image_data")
        table_name = [
            f
            for f in os.listdir(base_path)
            if f.endswith(".xlsx") and os.path.isfile(os.path.join(base_path, f))
        ]
        if len(table_name) == 0:
            table_name = [
                f
                for f in os.listdir(base_path)
                if f.endswith(".csv") and os.path.isfile(os.path.join(base_path, f))
            ]

        table_name = table_name[0]
        table_path = os.path.join(base_path, table_name)
        json_path = os.path.join(base_path, "segmented_linear_KBD_params_local.json")
        verify_cpp(file_path, table_path, fig_path, json_path, sf, False)
