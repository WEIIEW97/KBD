import numpy as np
from bayes_opt import BayesianOptimization
from scipy.optimize import differential_evolution, minimize
from sklearn.linear_model import LinearRegression

from .constants import EPSILON


def fit_linear_model(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model


def model(disp, focal, baseline, k, delta, b):
    return k * focal * baseline / (disp + delta) + b

def reverse_z(z, k, delta, b, fb):
    return 1/(((k/(z-b))-delta/fb))

def linear_KBD_piecewise_func(
    x,
    focal,
    baseline,
    params_matrix,
    disjoint_depth_range,
    compensate_dist=200,
    scaling_factor=10,
) -> float:
    k1, delta1, b1, coef1, intercept1 = params_matrix[1]
    k2, delta2, b2, coef2, intercept2 = params_matrix[2]
    k3, delta3, b3, coef3, intercept3 = params_matrix[3]

    FB = focal * baseline
    if x == 0:
        return x
    disp = FB / x

    if x < disjoint_depth_range[0] - compensate_dist:
        return x
    if disjoint_depth_range[0] - compensate_dist <= x < disjoint_depth_range[0]:
        return FB / (coef1 * disp + intercept1)
    if disjoint_depth_range[0] <= x < disjoint_depth_range[1]:
        return k2 * FB / (disp + delta2) + b2
    if (
        disjoint_depth_range[1]
        <= x
        < disjoint_depth_range[1] + compensate_dist * scaling_factor
    ):
        return FB / (coef3 * disp + intercept3)
    else:
        return x


def global_KBD_func(x, focal, baseline, k, delta, b):
    FB = focal * baseline
    if x <= EPSILON:
        return x
    disp = FB / x
    return k * FB / (disp + delta) + b
