import numpy as np
from numba import jit

@jit(nopython=True)
def modify(
    m: np.ndarray,
    h: int,
    w: int,
    k: float,
    delta: float,
    b: float,
    focal: float,
    baseline: float,
    epsilon: float,
) -> np.ndarray:
    fb = focal * baseline
    out = np.zeros_like(m)
    for i in range(h):
        for j in range(w):
            quali = m[i, j] + delta
            if quali <= epsilon or m[i, j] == 0:
                continue
            out[i, j] = max(k * fb / quali + b, 0)
    return out


def modify_vectorize(
    m: np.ndarray,
    k: float,
    delta: float,
    b: float,
    focal: float,
    baseline: float,
    epsilon: float,
) -> None:
    fb = focal * baseline
    quali = m + delta
    mask = quali > epsilon
    modified_values = np.maximum(k * fb / quali + b, 0)
    m[:] = np.where(mask, modified_values, m)
    return m
