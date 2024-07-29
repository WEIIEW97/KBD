import os

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from .constants import GT_DIST_NAME, GT_ERROR_NAME, MAPPED_PAIR_DICT
from .core import modify_linear_vectorize2
from .helpers import preprocessing

from .models import linear_KBD_piecewise_func


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
        if k <= 1000 and v < 0.02:
            accept += 1
        elif k <= 2000 and v < 0.04:
            accept += 1
    acceptance = accept / total_bins
    return eval_res, acceptance


def pass_or_not(path, table_path, pair_dict=MAPPED_PAIR_DICT):
    df, _, _ = preprocessing(path, table_path, pair_dict)
    df["absolute_error_rate"] = df[GT_ERROR_NAME] / df[GT_DIST_NAME]
    metric_dist = [300, 500, 600, 1000, 1500, 2000]

    for metric in metric_dist:
        quali = df[df[GT_DIST_NAME] == metric]["absolute_error_rate"]

        if metric in (300, 500, 600, 1000):
            if not (quali < 0.2).all():
                return False
        elif metric in (1500, 2000):
            if not (quali < 0.4).all():
                return False
    return True


def evaluate_target(
    focal,
    baseline,
    param_matrix,
    disjoint_depth_range,
    compensate_dist,
    scaling_factor,
    z=[300, 500, 600, 1000, 1500, 2000],
):
    z_array = np.array(z)
    d_array = focal * baseline / z_array
    z_after = modify_linear_vectorize2(
        d_array,
        focal,
        baseline,
        param_matrix,
        disjoint_depth_range,
        compensate_dist,
        scaling_factor,
    )
    z_error_rate = np.abs((z_after - z_array)/z_array)
    print(f"z before is {z_array}")
    print(f"z after is {z_after}")
    return mean_squared_error(z_array, z_after), z_error_rate


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
