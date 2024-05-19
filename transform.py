from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import yaml
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from typing import Union, Callable, Any
from tqdm import tqdm
from numba import jit
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict


def ordered_dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


yaml.add_representer(OrderedDict, ordered_dict_representer)

# constants
SUBFIX = "DEPTH/raw"
CAMERA_TYPE = "0050"
BASEDIR = f"data/{CAMERA_TYPE}/image_data"
H = 480
W = 640
EPSILON = 1e-6
UINT16_MIN = 0
UINT16_MAX = 65535

ANCHOR_POINT = [H // 2, W // 2]

AVG_DIST_NAME = "avg_depth_50x50_anchor"
AVG_DISP_NAME = "avg_disp_50x50_anchor"
GT_DIST_NAME = "actual_depth"
GT_ERROR_NAME = "absolute_error"
FOCAL_NAME = "focal"
BASLINE_NAME = "baseline"

OUT_PARAMS_FILE_NAME = "KBD_model_fitted_params.yaml"
LINEAR_OUT_PARAMS_FILE_NAME = "linear_" + OUT_PARAMS_FILE_NAME
OUT_FIG_COMP_FILE_NAME = "compare.jpg"
OUT_FIG_RESIDUAL_FILE_NAME = "fitted_residual.jpg"
OUT_FIG_ERROR_RATE_FILE_NAME = "error_rate.jpg"
MAPPED_COLUMN_NAMES = ["actual_depth", "focal", "baseline", "absolute_error"]
MAPPED_PAIR_DICT = {
    "距离(mm)": "actual_depth",
    "相机焦距": "focal",
    "相机基线": "baseline",
    "绝对误差/mm": "absolute_error",
}


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


def helper_save_data_to_csv(path: str, table_path: str, save_path: str):
    all_distances = retrive_folder_names(path)
    mean_dists = calculate_mean_value(path, all_distances)
    df = read_table(table_path, pair_dict=MAPPED_PAIR_DICT)
    _ = map_table(df, mean_dists)
    df.to_csv(save_path)


def fit_linear_model(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model


def crop_center(array: np.ndarray, crop_size: Union[tuple, list]):
    if array.ndim != 2:
        raise ValueError("Input array must be a 2D array")
    height, width = array.shape
    center_y, center_x = height // 2, width // 2
    half_crop_size = crop_size // 2

    # Calculate start and end indices
    start_y = max(0, center_y - half_crop_size)
    end_y = min(height, center_y + half_crop_size)
    start_x = max(0, center_x - half_crop_size)
    end_x = min(width, center_x + half_crop_size)

    # Crop and return the central square
    return array[start_y:end_y, start_x:end_x]


def copy_files_in_directory(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    files = retrive_file_names(src)

    for file in files:
        source = os.path.join(src, file)
        destination = os.path.join(dst, file)
        shutil.copy2(source, destination)


def copy_all_subfolders(src: str, dst: str) -> None:
    folders = retrive_folder_names(src)

    for folder in tqdm(folders):
        source_path = os.path.join(src, folder, SUBFIX)
        destination_path = os.path.join(dst, folder, SUBFIX)
        copy_files_in_directory(source_path, destination_path)

    print("Copying done ...")


def parallel_copy(src: str, dst: str) -> None:
    folders = retrive_folder_names(src)

    with ThreadPoolExecutor() as executor:
        for folder in tqdm(folders, desc="Copying subfolders ..."):
            source_path = os.path.join(src, folder, SUBFIX)
            destination_path = os.path.join(dst, folder, SUBFIX)
            executor.submit(copy_files_in_directory, source_path, destination_path)

    print("Copying done ...")


def retrive_folder_names(path: str) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def retrive_file_names(path: str) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def calculate_mean_value(rootpath: str, folders: list[str]) -> dict[str, float]:
    dist_dict = {}
    for folder in folders:
        distance = folder.split("_")[0]
        rawpath = os.path.join(rootpath, folder, SUBFIX)
        paths = [
            f for f in os.listdir(rawpath) if os.path.isfile(os.path.join(rawpath, f))
        ]
        mean_dist_holder = []
        for path in paths:
            path = os.path.join(rawpath, path)
            raw = load_raw(path, H, W)
            valid_raw = raw[
                ANCHOR_POINT[0] - 25 : ANCHOR_POINT[0] + 25,
                ANCHOR_POINT[1] - 25 : ANCHOR_POINT[1] + 25,
            ]
            mu = np.mean(valid_raw)
            mean_dist_holder.append(mu)
        final_mu = np.mean(mean_dist_holder)
        dist_dict[distance] = final_mu
    return dist_dict


def map_table(df: pd.DataFrame, dist_dict: dict) -> tuple[float, float]:
    df[AVG_DIST_NAME] = df[GT_DIST_NAME].astype(str).map(dist_dict)
    focal = df[FOCAL_NAME].iloc[0]  # assume focal value is the same
    baseline = df[BASLINE_NAME].iloc[0]  # assume basline value is the same

    df[AVG_DISP_NAME] = focal * baseline / df[AVG_DIST_NAME]

    return focal, baseline


def model_kbd(
    actual_depth: np.ndarray, disp: np.ndarray, focal: float, baseline: float
):
    # disp_norm = normalize(disp)
    disp_norm = disp

    def mfunc(params, disp_norm, baseline, focal):
        k, delta, b = params
        y_hat = k * focal * baseline / (disp_norm + delta) + b
        return y_hat

    # Define the cost function (MSE)
    def cost_func(params, disp_norm, baseline, focal, actual_depth):
        predictions = mfunc(params, disp_norm, baseline, focal)
        # return np.mean((actual_depth - predictions) ** 2)
        return np.mean((actual_depth - predictions) ** 2)

    # Initial guess for the parameters and bounds
    initial_params = [1.0, 0.01, 10]  # Starting values for k, delta, b
    # bounds = [(0.1, 100), (0, 1), (-50, 50)]  # Expanded bounds for parameters

    result = minimize(
        cost_func,
        initial_params,
        args=(disp_norm, baseline, focal, actual_depth),
        method="Nelder-Mead",
    )

    print("Optimization Results:")
    print("Parameters (k, delta, b):", result.x)
    print("Minimum MSE:", result.fun)
    if result.success:
        print("The optimization converged successfully.")
    else:
        print("The optimization did not converge:", result.message)

    return result


def model_kbd_further_optimized(
    actual_depth: np.ndarray,
    disp: np.ndarray,
    focal: float,
    baseline: float,
    reg_lambda: float = 0.001,
):
    # disp_norm = normalize(disp)
    disp_norm = disp

    def mfunc(params, disp_norm, baseline, focal):
        k, delta, b = params
        y_hat = k * focal * baseline / (disp_norm + delta) + b
        return y_hat

    def cost_func(params, disp_norm, baseline, focal, actual_depth):
        predictions = mfunc(params, disp_norm, baseline, focal)
        mse = np.mean((predictions - actual_depth) ** 2)
        # Adding L2 regularization
        regularization = reg_lambda * np.sum(np.square(params))
        return mse + regularization

    # Adjusting initial parameters and bounds based on previous results
    initial_params = [1.0, 0.01, 0]  # Modified initial values for k, delta, b
    # bounds = [(None, None), (0, 1), (None, None)]  # Expanded bounds for parameters

    # Using a different optimization method: 'TNC'
    result = minimize(
        cost_func,
        initial_params,
        args=(disp_norm, baseline, focal, actual_depth),
        method="Nelder-Mead",
    )

    print("Optimization Results:")
    print("Parameters (k, delta, b):", result.x)
    print("Minimum MSE:", result.fun)
    if result.success:
        print("The optimization converged successfully.")
    else:
        print("The optimization did not converge:", result.message)

    return result


def model_kbd_segmented(actual_depth, disp, focal, baseline, depth_ranges):
    """
    Perform piecewise optimization on depth data using given segments, ensuring continuity at joint points.

    Parameters:
    actual_depth (np.ndarray): The actual depth measurements.
    disp (np.ndarray): The disparity measurements corresponding to the actual depths.
    focal (float): Focal length of the camera.
    baseline (float): Baseline distance between cameras.
    depth_ranges (list of tuples): List of tuples specifying the depth ranges for each segment.

    Returns:
    dict: A dictionary containing optimization results for each segment.
    """

    def mfunc(params, disp, baseline, focal):
        k, delta, b = params
        return k * focal * baseline / (disp + delta) + b

    def cost_func(params, disp, baseline, focal, actual_depth):
        predictions = mfunc(params, disp, baseline, focal)
        return np.mean((actual_depth - predictions) ** 2)

    results = {}
    initial_params = [1.0, 0.01, 10]  # Reasonable starting values

    for idx, (start, end) in enumerate(depth_ranges):
        # Find indices within the specified depth range
        indices = np.where((actual_depth >= start) & (actual_depth <= end))[0]
        segment_disp = disp[indices]
        segment_depth = actual_depth[indices]

        result = minimize(
            cost_func,
            initial_params,
            args=(segment_disp, baseline, focal, segment_depth),
            method="Nelder-Mead",
        )

        results[(start, end)] = result
        initial_params = (
            result.x
        )  # Use optimized parameters as initial for next segment

    return results


def model_kbd_joint_linear(
    actual_depth, disp, focal, baseline, disjoint_depth_range
):
    """
    Fit the KBD model to the data where actual_depth >= 500.

    Parameters:
    actual_depth (np.ndarray): The actual depth measurements.
    disp (np.ndarray): The disparity measurements corresponding to the actual depths.
    focal (float): Focal length of the camera.
    baseline (float): Baseline distance between cameras.

    Returns:
    tuple: A tuple containing the linear model for the joint point and the optimization result.
    """

    # find the range to calculate KBD params within
    KBD_mask = np.where(
        (actual_depth >= disjoint_depth_range[1])
        & (actual_depth <= disjoint_depth_range[2])
    )
    KBD_disp = disp[KBD_mask]
    KBD_detph = actual_depth[KBD_mask]

    res = model_kbd(KBD_detph, KBD_disp, focal, baseline)
    k_, delta_, b_ = res.x
    FB = focal * baseline
    # now find the prediction within KBD_disp with KBD_res parameters
    KBD_disp_min = np.min(KBD_disp)
    KBD_disp_max = np.max(KBD_disp)

    KBD_pred_depth_max = k_ * FB / (KBD_disp_min + delta_) + b_
    KBD_pred_depth_min = k_ * FB / (KBD_disp_max + delta_) + b_

    KBD_disp_gt_max = FB / KBD_pred_depth_min
    KBD_disp_gt_min = FB / KBD_pred_depth_max
    max_joint_disp_gt = FB / disjoint_depth_range[0]
    min_joint_disp_gt = FB / disjoint_depth_range[3]
    max_joint_disp = disp[np.where(actual_depth == disjoint_depth_range[0])]
    min_joint_disp = disp[np.where(actual_depth == disjoint_depth_range[3])]

    linear_model1 = LinearRegression()
    X1 = np.array([max_joint_disp[0], KBD_disp_max])
    y1 = np.array([max_joint_disp_gt, KBD_disp_gt_max])
    linear_model1.fit(X1.reshape(-1, 1), y1)

    # if (KBD_disp_gt_max - max_joint_disp_gt) <= 0 or (KBD_disp_max - max_joint_disp) <= 0:
    #     linear_model1.coef_ = np.array([1])
    #     linear_model1.intercept_ = np.array(0)

    linear_model2 = LinearRegression()
    X2 = np.array([min_joint_disp[0], KBD_disp_min])
    y2 = np.array([min_joint_disp_gt, KBD_disp_gt_min])
    linear_model2.fit(X2.reshape(-1, 1), y2)
    

    # if (KBD_disp_gt_min - min_joint_disp_gt) <= 0 or (KBD_disp_min - min_joint_disp) <= 0:
    #     linear_model2.coef_ = np.array([1])
    #     linear_model2.intercept_ = np.array(0)

    return linear_model1, res, linear_model2


def generate_parameters(
    path: str,
    tabel_path: str,
    save_path: str,
    use_l2: bool = False,
    reg_lambda: float = 0.01,
):
    all_distances = retrive_folder_names(path)
    mean_dists = calculate_mean_value(path, all_distances)
    df = read_table(tabel_path, pair_dict=MAPPED_PAIR_DICT)
    focal, baseline = map_table(df, mean_dists)

    actual_depth = df[GT_DIST_NAME].values
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
    error = df[GT_ERROR_NAME].values

    if not use_l2:
        res = model_kbd(actual_depth, avg_50x50_anchor_disp, focal, baseline)
    else:
        res = model_kbd_further_optimized(
            actual_depth, avg_50x50_anchor_disp, focal, baseline, reg_lambda=reg_lambda
        )

    param_path = os.path.join(save_path, OUT_PARAMS_FILE_NAME)
    comp_path = os.path.join(save_path, OUT_FIG_COMP_FILE_NAME)
    residual_path = os.path.join(save_path, OUT_FIG_RESIDUAL_FILE_NAME)
    error_rate_path = os.path.join(save_path, OUT_FIG_ERROR_RATE_FILE_NAME)

    if use_l2:
        common_prefix = "l2_"
        param_path = os.path.join(save_path, common_prefix + OUT_PARAMS_FILE_NAME)
        comp_path = os.path.join(save_path, common_prefix + OUT_FIG_COMP_FILE_NAME)
        residual_path = os.path.join(
            save_path, common_prefix + OUT_FIG_RESIDUAL_FILE_NAME
        )
        error_rate_path = os.path.join(
            save_path, common_prefix + OUT_FIG_ERROR_RATE_FILE_NAME
        )

    # params_dict = {"k": str(res.x[0]), "delta": str(res.x[1]), "b": str(res.x[2])}
    k_ = float(np.float64(res.x[0]))
    delta_ = float(np.float64(res.x[1]))
    b_ = float(np.float64(res.x[2]))
    params_dict = {
        "k": k_,
        "delta": delta_,
        "b": b_,
    }
    print(params_dict)

    with open(param_path, "w") as f:
        yaml.dump(params_dict, f, default_flow_style=False)
    print("Generating done...")

    pred = k_ * focal * baseline / (avg_50x50_anchor_disp + delta_) + b_
    residual = pred - actual_depth
    plot_residuals(residual, error, actual_depth, residual_path)
    plot_error_rate(residual, error, actual_depth, error_rate_path)
    plot_comparison(
        actual_depth, focal * baseline / avg_50x50_anchor_disp, pred, comp_path
    )

    return k_, delta_, b_, focal, baseline


def get_linear_model_params(linear_model):
    """Extract parameters from a linear regression model."""
    params = OrderedDict(
        [
            ("alpha", linear_model.coef_.tolist()),
            ("beta", linear_model.intercept_.tolist()),
        ]
    )
    return params


def generate_parameters_linear(
    path: str,
    tabel_path: str,
    save_path: str,
    disjoint_depth_range: tuple,
):
    all_distances = retrive_folder_names(path)
    mean_dists = calculate_mean_value(path, all_distances)
    df = read_table(tabel_path, pair_dict=MAPPED_PAIR_DICT)
    focal, baseline = map_table(df, mean_dists)

    actual_depth = df[GT_DIST_NAME].values
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
    error = df[GT_ERROR_NAME].values

    linear_model1, res, linear_model2 = model_kbd_joint_linear(
        actual_depth, avg_50x50_anchor_disp, focal, baseline, disjoint_depth_range
    )

    param_path = os.path.join(save_path, LINEAR_OUT_PARAMS_FILE_NAME)
    # comp_path = os.path.join(save_path, OUT_FIG_COMP_FILE_NAME)
    # residual_path = os.path.join(save_path, OUT_FIG_RESIDUAL_FILE_NAME)
    # error_rate_path = os.path.join(save_path, OUT_FIG_ERROR_RATE_FILE_NAME)

    # params_dict = {"k": str(res.x[0]), "delta": str(res.x[1]), "b": str(res.x[2])}
    k_ = float(np.float64(res.x[0]))
    delta_ = float(np.float64(res.x[1]))
    b_ = float(np.float64(res.x[2]))

    linear_model1_params = get_linear_model_params(linear_model1)
    linear_model2_params = get_linear_model_params(linear_model2)

    ### do not support the shared pointer
    # default_linear_model = LinearRegression()
    # default_linear_model.coef_ = np.array([1.0])
    # default_linear_model.intercept_ = np.array([0.0])
    # default_linear_model_params = get_linear_model_params(default_linear_model)

    def create_default_linear_model_params():
        default_linear_model = LinearRegression()
        default_linear_model.coef_ = np.array([1.0])
        default_linear_model.intercept_ = np.array(0.0)
        return get_linear_model_params(default_linear_model)

    params_dict = OrderedDict(
        [
            (
                f"{0}-{disjoint_depth_range[0]}",
                OrderedDict(
                    [
                        ("k", 1),
                        ("delta", 0),
                        ("b", 0),
                        ("linear_model_params", create_default_linear_model_params()),
                    ]
                ),
            ),
            (
                f"{disjoint_depth_range[0]}-{disjoint_depth_range[1]}",
                OrderedDict(
                    [
                        ("k", 1),
                        ("delta", 0),
                        ("b", 0),
                        ("linear_model_params", linear_model1_params),
                    ]
                ),
            ),
            (
                f"{disjoint_depth_range[1]}-{disjoint_depth_range[2]}",
                OrderedDict(
                    [
                        ("k", k_),
                        ("delta", delta_),
                        ("b", b_),
                        ("linear_model_params", create_default_linear_model_params()),
                    ]
                ),
            ),
            (
                f"{disjoint_depth_range[2]}-{disjoint_depth_range[3]}",
                OrderedDict(
                    [
                        ("k", 1),
                        ("delta", 0),
                        ("b", 0),
                        ("linear_model_params", linear_model2_params),
                    ]
                ),
            ),
            (
                f"{disjoint_depth_range[3]}-{np.inf}",
                OrderedDict(
                    [
                        ("k", 1),
                        ("delta", 0),
                        ("b", 0),
                        ("linear_model_params", create_default_linear_model_params()),
                    ]
                ),
            ),
        ]
    )

    print(params_dict)

    with open(param_path, "w") as f:
        yaml.dump(params_dict, f, default_flow_style=None, sort_keys=False)
    print("Generating done...")

    # pred = k_ * focal * baseline / (avg_50x50_anchor_disp + delta_) + b_
    # residual = pred - actual_depth
    # plot_residuals(residual, error, actual_depth, residual_path)
    # plot_error_rate(residual, error, actual_depth, error_rate_path)
    # plot_comparison(
    #     actual_depth, focal * baseline / avg_50x50_anchor_disp, pred, comp_path
    # )

    params_matrix = np.zeros((5, 5), dtype=np.float32)
    params_matrix[0, :] = np.array([1, 0, 0, 1, 0])
    params_matrix[1, :] = np.array(
        [1, 0, 0, linear_model1.coef_[0], linear_model1.intercept_]
    )
    params_matrix[2, :] = np.array([k_, delta_, b_, 1, 0])
    params_matrix[3, :] = np.array(
        [1, 0, 0, linear_model2.coef_[0], linear_model2.intercept_]
    )
    params_matrix[4, :] = np.array([1, 0, 0, 1, 0])

    return params_matrix


def depth2disp(m: np.ndarray, focal: float, baseline: float) -> np.ndarray:
    fb = focal * baseline
    m = m.astype(np.float32)
    d = np.divide(fb, m, where=(m != 0), out=np.zeros_like(m))
    return d


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


def apply_transformation(
    path: str,
    k: float,
    delta: float,
    b: float,
    focal: float,
    baseline: float,
    epislon: float = EPSILON,
) -> None:
    folders = retrive_folder_names(path)

    for folder in tqdm(folders):
        paths = retrive_file_names(os.path.join(path, folder, SUBFIX))
        for p in paths:
            full_path = os.path.join(path, folder, SUBFIX, p)
            raw = load_raw(full_path, H, W)
            disp = depth2disp(raw, focal, baseline)
            depth = modify(disp, H, W, k, delta, b, focal, baseline, epislon)
            # make sure raw value is within range(0, 65535)
            depth = np.clip(depth, UINT16_MIN, UINT16_MAX)
            depth = depth.astype(np.uint16)
            with open(full_path, "wb") as f:
                depth.tofile(f)

    print("Transformating data done ...")


def transformer_impl(full_path, H, W, k, delta, b, focal, baseline, epislon) -> None:
    raw = load_raw(full_path, H, W)
    disp = depth2disp(raw, focal, baseline)
    depth = modify(disp, H, W, k, delta, b, focal, baseline, epislon)
    depth = np.clip(depth, UINT16_MIN, UINT16_MAX)  # Ensure within range
    depth = depth.astype(np.uint16)
    with open(full_path, "wb") as f:
        depth.tofile(f)


def apply_transformation_parallel(
    path: str,
    k: float,
    delta: float,
    b: float,
    focal: float,
    baseline: float,
    epislon: float = EPSILON,
):
    folders = retrive_folder_names(path)

    def process_folder(folder):
        paths = retrive_file_names(os.path.join(path, folder, SUBFIX))
        full_paths = [os.path.join(path, folder, SUBFIX, p) for p in paths]

        with ThreadPoolExecutor() as executor:
            executor.map(
                transformer_impl,
                full_paths,
                [H] * len(full_paths),
                [W] * len(full_paths),
                [k] * len(full_paths),
                [delta] * len(full_paths),
                [b] * len(full_paths),
                [focal] * len(full_paths),
                [baseline] * len(full_paths),
                [epislon] * len(full_paths),
            )

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_folder, folders), total=len(folders)))

    print("Transformation data done ...")


def plot_residuals(
    residuals: np.ndarray, error: np.ndarray, gt: np.ndarray, save_path: str = None
):
    plt.figure(figsize=(10, 6))
    plt.scatter(gt, residuals, alpha=0.5, color="blue", label="fitted residuals")
    plt.scatter(gt, error, alpha=0.5, color="green", label="actual residuals")
    plt.hlines(
        0,
        xmin=0,
        xmax=np.max(gt),
        colors="red",
        linestyles="dashed",
        label="Zero Error Line",
    )
    plt.xlabel("Ground truth distance (mm)")
    plt.ylabel("Residuals (Error) vs original error")
    plt.title("Residuals Plot")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    # Print the mean of the residuals
    mean_residuals = np.mean(residuals)
    print("Mean of residuals:", mean_residuals)


def plot_error_rate(
    residuals: np.ndarray,
    error: np.ndarray,
    nominator: np.ndarray,
    save_path: str = None,
):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        nominator,
        residuals / nominator * 100,
        alpha=0.5,
        color="blue",
        label="fitted residuals",
    )
    plt.scatter(
        nominator,
        error / nominator * 100,
        alpha=0.5,
        color="green",
        label="actual residuals",
    )
    plt.hlines(
        0,
        xmin=0,
        xmax=np.max(nominator),
        colors="red",
        linestyles="dashed",
        label="Zero Error Line",
    )
    plt.hlines(
        2,
        xmin=0,
        xmax=np.max(nominator),
        colors="green",
        linestyles="dashed",
        label="2(%) error Line",
    )
    plt.hlines(
        -2,
        xmin=0,
        xmax=np.max(nominator),
        colors="green",
        linestyles="dashed",
        label="2(%) error Line",
    )
    plt.hlines(
        4,
        xmin=0,
        xmax=np.max(nominator),
        colors="blue",
        linestyles="dashed",
        label="4(%) error Line",
    )
    plt.hlines(
        -4,
        xmin=0,
        xmax=np.max(nominator),
        colors="blue",
        linestyles="dashed",
        label="4(%) error Line",
    )
    plt.xlabel("Ground truth distance (mm)")
    plt.ylabel("Residuals (Error) rate vs original error rate (%)")
    plt.title("Error rate Plot (%)")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_comparison(x, y1, y2, save_path):
    fig, ax = plt.subplots()
    ax.plot(x, y1, label="measured data", marker="o")
    ax.plot(x, y2, label="fitted data", marker="x")

    ax.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_metric(ax, data, metric, title, xlabel, ylabel, zero_line=True, legend=True):
    """Helper function to plot a specific metric."""
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for idx, ((start, end), color) in enumerate(zip(data.keys(), colors)):
        indices, segment_depth, segment_disp, pred, residual, error, focal, baseline = (
            data[(start, end)]
        )
        if metric == "residual":
            values = residual
        elif metric == "error rate":
            values = residual / segment_depth * 100
            error = error / segment_depth * 100
        elif metric == "depth comparison":
            ax.plot(
                segment_depth,
                pred,
                label=f"Fitted {start}-{end}m",
                marker="x",
                linestyle="None",
                color=color,
            )
            continue
        elif metric == "unified comparison":
            ax.plot(
                segment_depth,
                focal * baseline / segment_disp,
                label="Measured Data",
                marker="o",
                linestyle="None",
                color="black",
            )
            ax.plot(
                segment_depth,
                pred,
                label=f"Fitted {start}-{end}m",
                marker="x",
                linestyle="None",
                color=color,
            )
            continue
        ax.scatter(
            segment_depth, values, color=color, alpha=0.5, label=f"{start}-{end}m"
        )
        ax.scatter(
            segment_depth,
            error,
            color="black",
            alpha=0.5,
            label=f"{start}-{end}m actual residuals",
        )
        if zero_line:
            ax.hlines(
                0,
                xmin=np.min(segment_depth),
                xmax=np.max(segment_depth),
                colors="red",
                linestyles="dashed",
                label="Zero Error Line" if idx == 0 else "",
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()


def plot_unified_results1(gt, est, error, focal, baseline, depth_ranges, res):
    # Prepare data for plotting
    plot_data = {}
    for start, end in depth_ranges:
        indices = np.where((gt >= start) & (gt <= end))[0]
        segment_disp = est[indices]
        segment_depth = gt[indices]
        segment_error = error[indices]
        optimized_params = res[(start, end)].x
        pred = (
            optimized_params[0]
            * focal
            * baseline
            / (segment_disp + optimized_params[1])
            + optimized_params[2]
        )
        residual = pred - segment_depth
        plot_data[(start, end)] = (
            indices,
            segment_depth,
            segment_disp,
            pred,
            residual,
            segment_error,
            focal,
            baseline,
        )

    # Plot configurations
    metrics = [
        ("residual", "Residuals Plot", "Ground Truth Depth (m)", "Residuals"),
        (
            "error rate",
            "Error Rate Plot (%)",
            "Ground Truth Depth (m)",
            "Error Rate (%)",
        ),
        (
            "depth comparison",
            "Depth Comparison by Segment",
            "Ground Truth Depth (m)",
            "Predicted Depth",
        ),
        (
            "unified comparison",
            "Unified Depth Comparison",
            "Ground Truth Depth (m)",
            "Depth",
        ),
    ]

    # Generate plots
    for metric, title, xlabel, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_metric(
            ax,
            plot_data,
            metric,
            title,
            xlabel,
            ylabel,
            zero_line=(metric in ["residual", "error rate"]),
            legend=(metric in ["depth comparison", "unified comparison"]),
        )
        plt.show()


def plot_unified_results(gt, est, error, focal, baseline, depth_ranges, res):
    # Create a figure with 3 subplots (one row, three columns)
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(
        1, 4, figsize=(24, 6)
    )  # Adjust the figsize accordingly

    # Extended color palette to accommodate more segments
    colors = plt.cm.viridis(
        np.linspace(0, 1, len(depth_ranges))
    )  # Using a colormap for more segments

    for idx, ((start, end), color) in enumerate(zip(depth_ranges, colors)):
        # Select data for the current segment
        indices = np.where((gt >= start) & (gt <= end))[0]
        segment_disp = est[indices]
        segment_depth = gt[indices]
        segment_error = error[indices]

        # Use optimized parameters to predict depths
        optimized_params = res[(start, end)].x
        pred = (
            optimized_params[0]
            * focal
            * baseline
            / (segment_disp + optimized_params[1])
            + optimized_params[2]
        )
        residual = pred - segment_depth

        # Plot residuals
        ax1.scatter(
            segment_depth, residual, color=color, alpha=0.5, label=f"{start}-{end}m"
        )
        ax1.scatter(
            segment_depth,
            segment_error,
            color="black",
            alpha=0.5,
            label=f"{start}-{end}m actual residuals",
        )

        ax1.hlines(
            0,
            xmin=np.min(segment_depth),
            xmax=np.max(segment_depth),
            colors="red",
            linestyles="dashed",
            label="Zero Error Line" if idx == 0 else "",
        )

        # Plot error rate
        ax2.scatter(
            segment_depth,
            residual / segment_depth * 100,
            color=color,
            alpha=0.5,
            label=f"{start}-{end}m",
        )
        ax2.scatter(
            segment_depth,
            segment_error / segment_depth * 100,
            color="black",
            alpha=0.5,
            label=f"{start}-{end}m actual residuals",
        )
        ax2.hlines(
            0,
            xmin=np.min(segment_depth),
            xmax=np.max(segment_depth),
            colors="red",
            linestyles="dashed",
            label="Zero Error Line" if idx == 0 else "",
        )

        # Plot comparison
        ax3.scatter(
            segment_depth,
            pred,
            color=color,
            alpha=0.5,
            label=f"{start}-{end}m Predicted",
        )
        ax3.plot(segment_depth, segment_depth, "k--", alpha=0.5)  # Actual depth line

        ax4.plot(
            segment_depth,
            focal * baseline / segment_disp,
            label="Measured Data",
            marker="o",
            linestyle="None",
            color="black",
        )
        ax4.plot(
            segment_depth,
            pred,
            label=f"Fitted {start}-{end}m",
            marker="x",
            linestyle="None",
            color=color,
        )

    # Set titles and labels
    ax1.set_title("Residuals Plot")
    ax1.set_xlabel("Ground Truth Depth (m)")
    ax1.set_ylabel("Residuals")
    ax1.legend()

    ax2.set_title("Error Rate Plot (%)")
    ax2.set_xlabel("Ground Truth Depth (m)")
    ax2.set_ylabel("Error Rate (%)")
    ax2.legend()

    ax3.set_title("Depth Comparison")
    ax3.set_xlabel("Ground Truth Depth (m)")
    ax3.set_ylabel("Predicted Depth")
    ax3.legend()

    ax4.set_title("Unified Depth Comparison")
    ax4.set_xlabel("Ground Truth Depth (m)")
    ax4.set_ylabel("Depth")
    ax4.legend()

    plt.tight_layout()
    plt.show()


def plot_linear(gt, est, error, focal, baseline, res, disjoint_depth_range):
    linear_model, optimization_result, linear_model2 = res

    # Filter data where actual_depth >= 600
    mask0 = np.where(gt < disjoint_depth_range[0])
    mask1 = np.where((gt >= disjoint_depth_range[0]) & (gt <= disjoint_depth_range[1]))
    mask2 = (gt > disjoint_depth_range[1]) & (gt <= disjoint_depth_range[2])
    mask3 = (gt > disjoint_depth_range[2]) & (gt <= disjoint_depth_range[3])
    mask4 = gt > disjoint_depth_range[3]

    filtered_disp0 = est[mask0]
    filtered_depth0 = gt[mask0]
    error0 = error[mask0]
    pred0 = gt[mask0]
    residual0 = pred0 - filtered_depth0

    filtered_disp1 = est[mask1]
    filtered_depth1 = gt[mask1]
    error1 = error[mask1]
    pred_1_disp = linear_model.predict(filtered_disp1.reshape(-1, 1))
    pred1 = focal * baseline / pred_1_disp
    residual1 = pred1 - filtered_depth1

    filtered_disp2 = est[mask2]
    filtered_depth2 = gt[mask2]
    error2 = error[mask2]

    # Get optimized parameters
    optimized_params = optimization_result.x
    pred2 = (
        optimized_params[0] * focal * baseline / (filtered_disp2 + optimized_params[1])
        + optimized_params[2]
    )
    residual2 = pred2 - filtered_depth2

    filtered_disp3 = est[mask3]
    filtered_depth3 = gt[mask3]
    error3 = error[mask3]
    pred_3_disp = linear_model2.predict(filtered_disp3.reshape(-1, 1))
    pred3 = focal * baseline / pred_3_disp
    residual3 = pred3 - filtered_depth3

    filtered_disp4 = est[mask4]
    filtered_depth4 = gt[mask4]
    error4 = error[mask4]
    pred4 = gt[mask4]
    residual4 = pred4 - filtered_depth4

    # all_gt = np.concatenate([gt[mask0], filtered_depth1, filtered_depth2])
    # all_pred = np.concatenate([pred0, pred1, pred2])
    # all_residuals = np.concatenate([np.zeros_like(pred0), residual1, residual2])
    # all_errors = np.concatenate([np.zeros_like(pred0), error1, error2])

    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Residuals plot
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(
        filtered_depth1,
        residual1,
        color="blue",
        alpha=0.5,
        label="Linear Model 1 Residuals",
    )
    ax1.scatter(
        filtered_depth1, error1, color="black", alpha=0.5, label="Actual Residuals"
    )

    ax1.scatter(
        filtered_depth2,
        residual2,
        color="green",
        alpha=0.5,
        label="Optimized Model Residuals",
    )
    ax1.scatter(
        filtered_depth2, error2, color="black", alpha=0.5, label="Actual Residuals"
    )

    ax1.scatter(
        filtered_depth3,
        residual3,
        color="red",
        alpha=0.5,
        label="Linear Model 2 Residuals",
    )
    ax1.scatter(
        filtered_depth3, error3, color="black", alpha=0.5, label="Actual Residuals"
    )
    ax1.hlines(
        0,
        xmin=np.min(filtered_depth2),
        xmax=np.max(filtered_depth2),
        colors="red",
        linestyles="dashed",
    )
    ax1.set_title("Residuals Plot")
    ax1.set_xlabel("Ground Truth Depth (m)")
    ax1.set_ylabel("Residuals")
    ax1.legend()
    plt.show()

    # Error rate plot
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(
        filtered_depth0,
        residual0 / filtered_depth0 * 100,
        color="pink",
        alpha=0.5,
        label="Unchanged Error Rate",
    )
    ax2.scatter(
        filtered_depth0,
        error0 / filtered_depth0 * 100,
        color="black",
        alpha=0.5,
        label="Actual Error Rate",
    )
    ax2.scatter(
        filtered_depth1,
        residual1 / filtered_depth1 * 100,
        color="blue",
        alpha=0.5,
        label="Linear Model 1 Error Rate",
    )
    ax2.scatter(
        filtered_depth1,
        error1 / filtered_depth1 * 100,
        color="black",
        alpha=0.5,
        label="Actual Error Rate",
    )
    ax2.scatter(
        filtered_depth2,
        residual2 / filtered_depth2 * 100,
        color="green",
        alpha=0.5,
        label="Optimized Model Error Rate",
    )
    ax2.scatter(
        filtered_depth2,
        error2 / filtered_depth2 * 100,
        color="black",
        alpha=0.5,
        label="Actual Error Rate",
    )
    ax2.scatter(
        filtered_depth3,
        residual3 / filtered_depth3 * 100,
        color="gray",
        alpha=0.5,
        label="Linear Model 2 Error Rate",
    )
    ax2.scatter(
        filtered_depth3,
        error3 / filtered_depth3 * 100,
        color="black",
        alpha=0.5,
        label="Actual Error Rate",
    )
    ax2.scatter(
        filtered_depth4,
        residual4 / filtered_depth4 * 100,
        color="cyan",
        alpha=0.5,
        label="Unchanged Error Rate",
    )
    ax2.scatter(
        filtered_depth4,
        error4 / filtered_depth4 * 100,
        color="black",
        alpha=0.5,
        label="Actual Error Rate",
    )
    ax2.hlines(
        0,
        xmin=np.min(filtered_depth1),
        xmax=np.max(filtered_depth2),
        colors="red",
        linestyles="dashed",
    )
    ax2.hlines(
        2,
        xmin=0,
        xmax=np.max(filtered_depth4),
        colors="pink",
        linestyles="dashed",
        label="2(%) error Line",
    )
    ax2.hlines(
        -2,
        xmin=0,
        xmax=np.max(filtered_depth4),
        colors="pink",
        linestyles="dashed",
        label="2(%) error Line",
    )
    ax2.hlines(
        4,
        xmin=0,
        xmax=np.max(filtered_depth4),
        colors="cyan",
        linestyles="dashed",
        label="4(%) error Line",
    )
    ax2.hlines(
        -4,
        xmin=0,
        xmax=np.max(filtered_depth4),
        colors="cyan",
        linestyles="dashed",
        label="4(%) error Line",
    )
    ax2.set_title("Error Rate Plot (%)")
    ax2.set_xlabel("Ground Truth Depth (m)")
    ax2.set_ylabel("Error Rate (%)")
    ax2.legend()
    plt.show()

    # Depth comparison plot
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.plot(
        gt, focal * baseline / est, label="Measured Data", marker="o", color="black"
    )
    ax3.plot(gt[mask0], pred0, label="Measured Data (< 400)", marker="o", color="red")
    ax3.plot(
        filtered_depth1, pred1, label="Linear Model (400-600)", marker="x", color="blue"
    )
    ax3.plot(
        filtered_depth2,
        pred2,
        label="Optimized Model (> 600)",
        marker="x",
        color="green",
    )
    ax3.plot(
        filtered_depth3,
        pred3,
        label="Linear Model (2700-2900)",
        marker="x",
        color="cyan",
    )
    ax3.plot(gt[mask4], pred4, label="Measured Data (>2900)", marker="o", color="red")

    ax3.set_xlabel("Ground Truth Depth (m)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_title("Comparison of Measured and Fitted Depths")
    ax3.legend()
    plt.show()


if __name__ == "__main__":
    cwd = os.getcwd()
    rootdir = f"{cwd}/data/{CAMERA_TYPE}/image_data"
    copydir = f"{cwd}/data/{CAMERA_TYPE}/image_data_transformed"
    table_path = f"{cwd}/data/{CAMERA_TYPE}/depthquality-2024-05-18.xlsx"
    params_save_path = f"{cwd}/data/{CAMERA_TYPE}"
    l2_regularization_param = (0.01,)
    disjoint_depth_range = (400, 600, 2700, 2904)

    ################# save new df to csv
    # all_distances = retrive_folder_names(rootdir)
    # mean_dists = calculate_mean_value(rootdir, all_distances)
    # df = read_table(table_path, pair_dict=MAPPED_PAIR_DICT)
    # focal, baseline = map_table(
    #     df,
    #     mean_dists,
    #     "D:/william/codes/depth-quality-fitting/data/N9_concat/dq_0513.csv",
    # )
    #################

    k, delta, b, focal, baseline = generate_parameters(
        path=rootdir,
        tabel_path=table_path,
        save_path=params_save_path,
        use_l2=False,
    )

    # params_matrix = generate_parameters_linear(
    #     path=rootdir,
    #     tabel_path=table_path,
    #     save_path=params_save_path,
    #     disjoint_depth_range=disjoint_depth_range,
    # )
    # print(params_matrix)
    # copy_all_subfolders(rootdir, copydir)
    parallel_copy(rootdir, copydir)

    apply_transformation_parallel(copydir, k, delta, b, focal, baseline)
