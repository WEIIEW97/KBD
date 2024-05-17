import numpy as np
import os
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from .helpers import (
    retrive_folder_names,
    calculate_mean_value,
    map_table,
    retrive_file_names,
)
from .utils import read_table, load_raw, depth2disp
from .models import model_kbd, model_kbd_further_optimized
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
)
from .core import modify
from .plotters import plot_error_rate, plot_comparison, plot_residuals


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
