import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
