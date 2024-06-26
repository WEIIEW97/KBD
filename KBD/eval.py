import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from KBD.models import linear_KBD_piecewise_func
from .helpers import preprocessing
from .constants import MAPPED_PAIR_DICT, AVG_DIST_NAME, GT_DIST_NAME, GT_ERROR_NAME
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def eval(path, table_path, pair_dict=MAPPED_PAIR_DICT, stage=200):
    df, _, _ = preprocessing(path, table_path, pair_dict)
    df["absolute_error_rate"] = df[GT_ERROR_NAME] / df[GT_DIST_NAME]
    max_stage = np.max(df[GT_DIST_NAME].values)
    n_stage = int(max_stage / stage)
    stages = [stage + i * stage for i in range(n_stage)]
    eval_res = dict()
    for s in stages:
        dt = df[(df["actual_depth"] <= s) & (df["actual_depth"] > s - stage)]
        mape = np.mean(np.abs(dt["absolute_error_rate"]))
        eval_res[s] = mape

    total_bins = len(eval_res)
    accept = 0
    for k, v in eval_res.items():
        if k <= 1000 and v < 0.01:
            accept += 1
        if k <= 2000 and v < 0.02:
            accept += 1

    acceptance = accept / total_bins
    return eval_res, acceptance


def _is_monotonically_increasing(lst):
    """
    Check if a list of numbers is monotonically increasing.

    Parameters:
    lst (list): A list of numbers.

    Returns:
    bool: True if the list is monotonically increasing, False otherwise.
    """
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            print(f"Monotonicity fails at point:  y={lst[i-10:i+10]}), index is {i}")
            return False
    return True


def check_monotonicity(
    minv,
    maxv,
    focal,
    baseline,
    param_matrix,
    disjoint_depth_range,
    compensate_dist,
    scaling_factor,
):
    x_values = np.arange(minv, maxv, step=1)
    y_values = [
        linear_KBD_piecewise_func(
            x,
            focal,
            baseline,
            param_matrix,
            disjoint_depth_range,
            compensate_dist,
            scaling_factor,
        )
        for x in x_values
    ]
    return _is_monotonically_increasing(y_values)


if __name__ == "__main__":
    cwd = os.getcwd()
    root_dir = cwd + "/data/N09ASH24DH0054/image_data"
    tablepath = cwd + "/data/N09ASH24DH0054/depthquality-2024-05-22.xlsx"

    res = eval(root_dir, tablepath)
    print(res)
