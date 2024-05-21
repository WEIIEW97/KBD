import numpy as np
import os
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from sklearn.linear_model import LinearRegression

from .helpers import (
    retrive_folder_names,
    calculate_mean_value,
    map_table,
    retrive_file_names,
)
from .utils import read_table, load_raw, depth2disp, get_linear_model_params
from .models import model_kbd, model_kbd_further_optimized, model_kbd_joint_linear
from .constants import (
    UINT16_MIN,
    UINT16_MAX,
    H,
    W,
    SUBFIX,
    EPSILON,
    MAPPED_PAIR_DICT,
    GT_DIST_NAME,
    AVG_DISP_NAME,
    GT_ERROR_NAME,
    OUT_PARAMS_FILE_NAME,
    OUT_FIG_RESIDUAL_FILE_NAME,
    OUT_FIG_COMP_FILE_NAME,
    OUT_FIG_ERROR_RATE_FILE_NAME,
    LINEAR_OUT_PARAMS_FILE_NAME,
)
from .core import modify, modify_linear
from .plotters import plot_error_rate, plot_comparison, plot_residuals, plot_linear


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

    actual_depth = df[GT_DIST_NAME]
    avg_50x50_anchor_disp = df[AVG_DISP_NAME]
    error = df[GT_ERROR_NAME]

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

    plot_linear(
        actual_depth,
        avg_50x50_anchor_disp,
        error,
        focal,
        baseline,
        (linear_model1, res, linear_model2),
        disjoint_depth_range,
        save_path=save_path,
    )

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

    return params_matrix, focal, baseline


def apply_transformation_linear(
    path: str,
    params_matrix: np.ndarray,
    focal: float,
    baseline: float,
    disjoint_depth_range: tuple,
) -> None:
    folders = retrive_folder_names(path)

    for folder in tqdm(folders):
        paths = retrive_file_names(os.path.join(path, folder, SUBFIX))
        for p in paths:
            full_path = os.path.join(path, folder, SUBFIX, p)
            raw = load_raw(full_path, H, W)
            depth = modify_linear(
                raw, H, W, focal, baseline, params_matrix, disjoint_depth_range
            )
            # make sure raw value is within range(0, 65535)
            depth = np.clip(depth, UINT16_MIN, UINT16_MAX)
            depth = depth.astype(np.uint16)
            with open(full_path, "wb") as f:
                depth.tofile(f)

    print("Transformating data done ...")


def transformer_linear_impl(
    full_path, H, W, focal, baseline, params_matrix, disjoint_depth_range
):
    raw = load_raw(full_path, H, W)
    depth = modify_linear(
        raw, H, W, focal, baseline, params_matrix, disjoint_depth_range
    )
    depth = np.clip(depth, UINT16_MIN, UINT16_MAX)
    depth = depth.astype(np.uint16)
    with open(full_path, "wb") as f:
        depth.tofile(f)


def apply_transformation_linear_parallel(
    path: str,
    params_matrix: np.ndarray,
    focal: float,
    baseline: float,
    disjoint_depth_range: tuple,
    max_workers: int = 8,
) -> None:
    folders = retrive_folder_names(path)
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for folder in folders:
            paths = retrive_file_names(os.path.join(path, folder, SUBFIX))
            for p in paths:
                full_path = os.path.join(path, folder, SUBFIX, p)
                tasks.append(
                    executor.submit(
                        transformer_linear_impl,
                        full_path,
                        H,
                        W,
                        focal,
                        baseline,
                        params_matrix,
                        disjoint_depth_range,
                    )
                )

        for future in tqdm(as_completed(tasks), total=len(tasks)):
            future.result()  # Ensure any exceptions are raised

    print("Transforming data done...")
