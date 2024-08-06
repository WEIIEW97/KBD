import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from .models import global_KBD_func, linear_KBD_piecewise_func
from .constants import *


def read_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_table(path: str, pair_dict: dict) -> pd.DataFrame:
    pos = path.find("csv")
    if pos != -1:
        df = read_csv(path)
    else:
        df = read_excel(path)
    df_sel = df[list(pair_dict.keys())]
    needed_df = df_sel.rename(columns=pair_dict)
    return needed_df


def load_raw(path: str, h: int, w: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    return data.reshape((h, w))


def depth2disp(m: np.ndarray, focal: float, baseline: float) -> np.ndarray:
    fb = focal * baseline
    m = m.astype(np.float32)
    d = np.divide(fb, m, where=(m != 0), out=np.zeros_like(m))
    return d


def normalize(x: pd.DataFrame):
    scaler = MinMaxScaler()
    return scaler.fit_transform(x.values.reshape(-1, 1)).flatten()


def get_linear_model_params(linear_model):
    """Extract parameters from a linear regression model."""
    params = OrderedDict(
        [
            ("alpha", linear_model.coef_.tolist()),
            ("beta", linear_model.intercept_.tolist()),
        ]
    )
    return params


def json_dumper(data, savepath):
    with open(savepath, "w") as f:
        json.dump(data, f, indent=4)


def json_reader(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def json_to_numpy(data):
    values = []

    for _, value in data.items():
        values.extend(
            [
                value["k"],
                value["delta"],
                value["b"],
                value["linear_model_params"]["alpha"][0],
                value["linear_model_params"]["beta"],
            ]
        )
    matrix = np.array(values).reshape(5, 5)

    return matrix


def generate_linear_KBD_data(
    focal,
    baseline,
    params_matrix,
    disjoint_depth_range,
    range_start,
    range_end,
    step=1,
):
    x_values = np.arange(range_start, range_end + 1, step)
    y_values = [
        linear_KBD_piecewise_func(
            x, focal, baseline, params_matrix, disjoint_depth_range
        )
        for x in x_values
    ]
    y_values = np.array(y_values).astype(np.uint16)
    return x_values, y_values


def generate_global_KBD_data(
    focal, baseline, k, delta, b, range_start, range_end, step=1
):
    x_values = np.arange(range_start, range_end + 1, step)
    y_values = [global_KBD_func(x, focal, baseline, k, delta, b) for x in x_values]
    y_values = np.array(y_values).astype(np.uint16)
    return x_values, y_values


def save_arrays_to_txt(savepath, arr1d, arr2d):
    sp = Path(savepath)
    os.makedirs(sp.parent, exist_ok=True)
    with open(savepath, "w") as f:
        f.write("disp_nodes:\n")
        np.savetxt(f, arr1d, fmt="%d", newline=" ")
        f.write("\n\n")
        f.write("kbd_params:\n")
        np.savetxt(f, arr2d, fmt="%.16f")
    print(f"Arrays have been saved to {savepath}")


def save_arrays_to_json(savepath, arr1d, arr2d):
    sp = Path(savepath)
    os.makedirs(sp.parent, exist_ok=True)
    arr1d_lst = arr1d.tolist()
    arr2d_lst = arr2d.tolist()

    params = {"disp_nodes": arr1d_lst, "kbd_params": arr2d_lst}
    with open(savepath, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Arrays have been saved to {savepath}")


def save_arrays_to_txt2(savepath, arr1, arr2):
    sp = Path(savepath)
    os.makedirs(sp.parent, exist_ok=True)
    with open(savepath, "w") as f:
        f.write("optimal depth joint range is :\n")
        np.savetxt(f, arr1, fmt="%d", delimiter=",")
        f.write("\n")
        f.write("z error rate according to designed distance : \n")
        np.savetxt(f, arr2, fmt="%.16f", delimiter=",")
    print(f"Arrays have been saved to {savepath}")


def export_default_settings(path, focal, baseline, compensate_dist, scaling_factor):
    default_range = (600, 3000)
    extra_range = [
        default_range[0] - compensate_dist,
        default_range[0],
        default_range[1],
        default_range[1] + compensate_dist * scaling_factor,
    ]
    disp_nodes_fp32 = focal * baseline / (np.array(extra_range))
    disp_nodes_uint16 = (disp_nodes_fp32 * 64).astype(np.uint16)
    disp_nodes_uint16 = np.sort(disp_nodes_uint16)
    disp_nodes_uint16 = np.append(disp_nodes_uint16, DISP_VAL_MAX_UINT16)
    default_param = np.array([1, 0, 0, 1, 0])
    matrix = np.tile(default_param, (5, 1))
    matrix_param_by_disp = matrix[::-1, :]
    save_arrays_to_json(path, disp_nodes_uint16, matrix_param_by_disp)
    return default_range, matrix