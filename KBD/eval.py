import numpy as np
import pandas as pd
import os

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


if __name__ == "__main__":
    cwd = os.getcwd()
    root_dir = cwd + "/data/N09ASH24DH0054/image_data"
    tablepath = cwd + "/data/N09ASH24DH0054/depthquality-2024-05-22.xlsx"

    res = eval(root_dir, tablepath)
    print(res)
