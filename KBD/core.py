import numpy as np
from numba import jit


@jit(nopython=True, parallel=True)
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


@jit(nopython=True, parallel=True)
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
            depth = m[i, j]
            if depth < disjoint_depth_range[0]:
                out[i, j] = depth
            elif depth < disjoint_depth_range[1]:
                disp0 = fb / depth
                alpha_, beta_ = param_matrix[1, 3:5]
                disp1 = alpha_ * disp0 + beta_
                out[i, j] = fb / disp1
            elif depth < disjoint_depth_range[2]:
                disp0 = fb / depth
                k_, delta_, b_ = param_matrix[2, :3]
                out[i, j] = k_ * fb / (disp0 + delta_) + b_
            elif depth < disjoint_depth_range[3]:
                disp0 = fb / depth
                alpha_, beta_ = param_matrix[3, 3:5]
                disp1 = alpha_ * disp0 + beta_
                out[i, j] = fb / disp1
            else:
                out[i, j] = depth

    return out