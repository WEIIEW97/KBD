import numpy as np
from numba import jit

from .constants import EPSILON


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
    disjoint_depth_range: tuple | list,
    compensate_dist: float,
    scaling_factor: float,
) -> np.ndarray:
    fb = focal * baseline
    out = np.zeros_like(m)

    for i in range(h):
        for j in range(w):
            depth = m[i, j]
            if depth >= 0 and depth < 0 + EPSILON:
                continue
            elif depth < disjoint_depth_range[0] - compensate_dist:
                out[i, j] = depth
            elif depth < disjoint_depth_range[0]:
                disp0 = fb / depth
                alpha_, beta_ = param_matrix[1, 3:5]
                disp1 = alpha_ * disp0 + beta_
                out[i, j] = fb / disp1
            elif depth < disjoint_depth_range[1]:
                disp0 = fb / depth
                k_, delta_, b_ = param_matrix[2, :3]
                out[i, j] = k_ * fb / (disp0 + delta_) + b_
            elif depth < disjoint_depth_range[1] + compensate_dist * scaling_factor:
                disp0 = fb / depth
                alpha_, beta_ = param_matrix[3, 3:5]
                disp1 = alpha_ * disp0 + beta_
                out[i, j] = fb / disp1
            else:
                out[i, j] = depth

    return out


def modify_linear_vectorize(
    m: np.ndarray,
    focal: float,
    baseline: float,
    param_matrix: np.ndarray,
    disjoint_depth_range: tuple | list,
    compensate_dist: float,
    scaling_factor: float,
) -> np.ndarray:
    r"""
    input m is disparity
    output depth follow the formula below:
    D = k*fb/(alpha*d + beta + delta) + b
    """
    fb = focal * baseline
    out = np.zeros_like(m)

    depth_ = np.where(m != 0, fb / m, 0)

    mask0 = np.where((depth_ >= 0) & (depth_ < 0 + EPSILON))
    mask1 = np.where(
        (depth_ >= 0 + EPSILON) & (depth_ < disjoint_depth_range[0] - compensate_dist)
    )
    mask2 = np.where(
        (depth_ >= disjoint_depth_range[0] - compensate_dist)
        & (depth_ < disjoint_depth_range[0])
    )
    mask3 = np.where(
        (depth_ >= disjoint_depth_range[0]) & (depth_ < disjoint_depth_range[1])
    )
    mask4 = np.where(
        (depth_ >= disjoint_depth_range[1])
        & (depth_ < disjoint_depth_range[1] + compensate_dist * scaling_factor)
    )
    mask5 = np.where(
        (depth_) >= disjoint_depth_range[1] + compensate_dist * scaling_factor
    )

    out[mask0] = 0
    out[mask1] = fb / m[mask1]
    out[mask2] = fb / (param_matrix[1, 3] * m[mask2] + param_matrix[1, 4])
    out[mask3] = (
        param_matrix[2, 0] * fb / (m[mask3] + param_matrix[2, 1]) + param_matrix[2, 2]
    )
    out[mask4] = fb / (param_matrix[3, 3] * m[mask4] + param_matrix[3, 4])
    out[mask5] = fb / m[mask5]

    return out
