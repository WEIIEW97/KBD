import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

from .models import global_KBD_func, linear_KBD_piecewise_func


def ordered_dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


yaml.add_representer(OrderedDict, ordered_dict_representer)


def read_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_table(path: str, pair_dict: dict) -> pd.DataFrame:
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


def yaml_dumper(data, savepath):
    with open(savepath, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)


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
    with open(savepath, "w") as f:
        f.write("disp_nodes:\n")
        np.savetxt(f, arr1d, fmt="%d", newline=" ")
        f.write("\n\n")
        f.write("kbd_params:\n")
        np.savetxt(f, arr2d, fmt="%.16f")

    print(f"Arrays have been saved to {savepath}")


def save_arrays_to_json(savepath, arr1d, arr2d):
    arr1d_lst = arr1d.tolist()
    arr2d_lst = arr2d.tolist()

    params = {"disp_nodes": arr1d_lst, "kbd_params": arr2d_lst}

    with open(savepath, "w") as f:
        json.dump(params, f, indent=4)

    print(f"Arrays have been saved to {savepath}")
