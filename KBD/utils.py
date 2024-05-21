import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict

from .models import global_KBD_func, linear_KBD_piecewise_func


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
