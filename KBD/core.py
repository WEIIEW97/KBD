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


@jit(nopython=True)
def modify_linear(
    m: np.ndarray,
    h: int,
    w: int,
    focal: float,
    baseline: float,
    param_matrix: np.ndarray,
    disjoint_depth_range: tuple,
) -> np.ndarray:
    fb = focal * baseline
    out = np.zeros_like(m)
    for i in range(h):
        for j in range(w):
            if m[i, j] < disjoint_depth_range[0]:
                out[i, j] = m[i, j]
            elif (
                m[i, j] >= disjoint_depth_range[0] and m[i, j] < disjoint_depth_range[1]
            ):
                disp0 = fb / m[i, j]
                k_, delta_, b_, alpha_, beta_ = param_matrix[1, :]
                disp1 = alpha_ * disp0 + beta_
                out[i, j] = fb / disp1
            elif (
                m[i, j] >= disjoint_depth_range[1] and m[i, j] < disjoint_depth_range[2]
            ):
                disp0 = fb / m[i, j]
                k_, delta_, b_, alpha_, beta_ = param_matrix[2, :]
                out[i, j] = k_ * fb / (disp0 + delta_) + b_
            elif (
                m[i, j] >= disjoint_depth_range[2] and m[i, j] < disjoint_depth_range[3]
            ):
                disp0 = fb / m[i, j]
                k_, delta_, b_, alpha_, beta_ = param_matrix[3, :]
                disp1 = alpha_ * disp0 + beta_
                out[i, j] = fb / disp1
            elif m[i, j] >= disjoint_depth_range[3]:
                out[i, j] = m[i, j]
    return out