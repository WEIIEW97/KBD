import os
from concurrent.futures import as_completed, ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from .constants import *
from .core import modify, modify_linear, modify_linear_vectorize2
from .eval import evaluate_target

from .helpers import retrive_file_names, retrive_folder_names
from .optimizers import (
    JointLinearSmoothingOptimizer,
    NelderMeadOptimizer,
    TrustRegionReflectiveOptimizer,
)
from .plotters import plot_comparison, plot_error_rate, plot_linear2, plot_residuals
from .utils import depth2disp, json_dumper, load_raw


def generate_parameters(
    df: pd.DataFrame,
    focal: float,
    baseline: float,
    save_path: str,
    use_l2: bool = False,
    reg_lambda: float = 0.01,
    plot: bool = False,
):

    actual_depth = df[GT_DIST_NAME].values  # make sure is np.ndarray
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
    error = df[GT_ERROR_NAME].values

    nelder = NelderMeadOptimizer(
        actual_depth,
        avg_50x50_anchor_disp,
        focal,
        baseline,
        apply_l2=use_l2,
        reg_lambda=reg_lambda,
    )

    res = nelder.run()

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
    k_ = float(np.float64(res[0]))
    delta_ = float(np.float64(res[1]))
    b_ = float(np.float64(res[2]))
    params_dict = {
        "k": k_,
        "delta": delta_,
        "b": b_,
    }
    print(params_dict)

    json_dumper(params_dict, param_path)
    print("Generating done...")

    if plot:
        pred = k_ * focal * baseline / (avg_50x50_anchor_disp + delta_) + b_
        residual = pred - actual_depth
        plot_residuals(residual, error, actual_depth, residual_path)
        plot_error_rate(residual, error, actual_depth, error_rate_path)
        plot_comparison(
            actual_depth, focal * baseline / avg_50x50_anchor_disp, pred, comp_path
        )
    return k_, delta_, b_


def generate_parameters_trf(
    df: pd.DataFrame,
    focal: float,
    baseline: float,
    save_path: str,
    plot: bool = False,
):

    actual_depth = df[GT_DIST_NAME].values
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
    error = df[GT_ERROR_NAME].values

    trf = TrustRegionReflectiveOptimizer(
        actual_depth, avg_50x50_anchor_disp, focal, baseline
    )
    k_, delta_, b_ = trf.run()

    common_prefix = "TRF_"
    param_path = os.path.join(save_path, common_prefix + OUT_PARAMS_FILE_NAME)
    comp_path = os.path.join(save_path, common_prefix + OUT_FIG_COMP_FILE_NAME)
    residual_path = os.path.join(save_path, common_prefix + OUT_FIG_RESIDUAL_FILE_NAME)
    error_rate_path = os.path.join(
        save_path, common_prefix + OUT_FIG_ERROR_RATE_FILE_NAME
    )

    # params_dict = {"k": str(res.x[0]), "delta": str(res.x[1]), "b": str(res.x[2])}

    params_dict = {
        "k": k_,
        "delta": delta_,
        "b": b_,
    }
    print(params_dict)

    json_dumper(params_dict, param_path)
    print("Generating done...")

    if plot:
        pred = k_ * focal * baseline / (avg_50x50_anchor_disp + delta_) + b_
        residual = pred - actual_depth
        plot_residuals(residual, error, actual_depth, residual_path)
        plot_error_rate(residual, error, actual_depth, error_rate_path)
        plot_comparison(
            actual_depth, focal * baseline / avg_50x50_anchor_disp, pred, comp_path
        )
    return k_, delta_, b_



def construct_param_matrix(linear_model1, kbd_params, linear_model2):
    k_, delta_, b_ = kbd_params
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


class GridSearch2D:
    def __init__(self, df, focal, baseline, scaling_factor, save_path=None, engine="Nelder-Mead", apply_global=False, is_plot=False):
        # Validate inputs
        if df.empty:
            raise ValueError("DataFrame is empty.")
        if not all(x in df.columns for x in [GT_DIST_NAME, AVG_DISP_NAME, GT_ERROR_NAME]):
            raise ValueError("DataFrame missing required columns.")

        self.df = df
        self.focal = focal
        self.baseline = baseline
        self.engine = engine
        self.apply_global = apply_global
        self.scaling_factor = scaling_factor
        self.save_path = save_path
        self.is_plot = is_plot

        # Calculate fb only once since it's a fixed value based on focal and baseline
        self.fb = focal * baseline

        # Convert to numpy for faster calculations
        self.actual_depth = df[GT_DIST_NAME].values
        self.avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
        self.error = df[GT_ERROR_NAME].values

    def eval_parameters(self, range_start, compensate_dist):
        disjoint_depth_range = [range_start, 3000]
        jlm = JointLinearSmoothingOptimizer(
            self.actual_depth,
            self.avg_50x50_anchor_disp,
            self.focal,
            self.baseline,
            disjoint_depth_range,
            compensate_dist,
            scaling_factor=self.scaling_factor,
            engine=self.engine,
            apply_global=self.apply_global,
        )
        lm1, kbd, lm2 = jlm.run()
        pm = construct_param_matrix(lm1, kbd, lm2)
        mse, _ = evaluate_target(
            self.focal, self.baseline, pm, disjoint_depth_range, compensate_dist, self.scaling_factor, TARGET_POINTS
        )
        return mse, pm, (lm1, kbd, lm2)

    def optimize_parameters(self, search_range, cd_range, max_iter=1000, tol=1e-6):
        def objective(x):
            mse, pm, params = self.eval_parameters(x[0], x[1])
            if params[0].coef_[0] <= 0:
                mse = 10e7
            return mse

        initial_guess = [search_range[0], cd_range[0]]
        bounds = [search_range, cd_range]

        self.result = minimize(
            objective,
            initial_guess,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'xatol': tol}
        )

    def get_results(self):
        if self.result.success:
            optimized_range_start, optimized_compensate_dist = self.result.x
            best_mse, best_pm, best_params = self.eval_parameters(optimized_range_start, optimized_compensate_dist)
            print("Optimization successful.")
            print(f"Optimized range start: {optimized_range_start}")
            print(f"Optimized compensate distance: {optimized_compensate_dist}")
            print(f"Minimum MSE: {best_mse}")

            if self.save_path is not None and self.is_plot:
                plot_linear2(
                    self.actual_depth,
                    self.avg_50x50_anchor_disp,
                    self.error,
                    self.focal,
                    self.baseline,
                    best_params,
                    (int(optimized_range_start), 3000),
                    int(optimized_compensate_dist),
                    scaling_factor=self.scaling_factor,
                    apply_global=self.apply_global,
                    save_path=self.save_path,
                )

            return (best_pm, optimized_range_start, optimized_compensate_dist)
        else:
            print("Optimization failed: " + self.result.message)
            return (-1, -1, -1)
    

def generate_parameters_linear_search(
    df: pd.DataFrame,
    focal: float,
    baseline: float,
    save_path: str,
    search_range: tuple,
    compensate_dist: float = 200,
    scaling_factor: float = 10,
    engine="Nelder-Mead",
    apply_global=False,
    plot: bool = False,
):
    actual_depth = df[GT_DIST_NAME].values
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
    error = df[GT_ERROR_NAME].values

    STEP = 50

    lowest_mse = np.inf
    ranges = []
    pms = []
    lm1s = []
    kbds = []
    lm2s = []
    mses = []
    z_error_rates = []
    for start in range(search_range[0], search_range[1] + STEP, STEP):
        disjoint_depth_range = [start, 3000]
        jlm = JointLinearSmoothingOptimizer(
            actual_depth,
            avg_50x50_anchor_disp,
            focal,
            baseline,
            disjoint_depth_range,
            compensate_dist,
            scaling_factor,
            engine=engine,
            apply_global=apply_global,
            apply_weights=False,
            apply_l2=False,
        )

        lm1, kbd, lm2 = jlm.run()
        pm = construct_param_matrix(lm1, kbd, lm2)
        mse, z_error_rate = evaluate_target(
            focal, baseline, pm, disjoint_depth_range, compensate_dist, scaling_factor
        )
        print(f"z_error_rate is {z_error_rate}")
        ranges.append(disjoint_depth_range)
        pms.append(pm)
        lm1s.append(lm1)
        kbds.append(kbd)
        lm2s.append(lm2)
        mses.append(mse)
        z_error_rates.append(z_error_rate)

    # After collecting all results, analyze them based on the given conditions
    for i in range(len(mses)):
        z_er = z_error_rates[i]
        if z_er[0:4].all() < 0.02 and z_er[4:].all() < 0.04:
            if mses[i] < lowest_mse:
                lowest_mse = mses[i]
                best_range = ranges[i]
                best_pm = pms[i]
                best_lm1 = lm1s[i]
                best_kbd = kbds[i]
                best_lm2 = lm2s[i]
                best_z_error_rate = z_error_rates[i]

    # If no suitable mse is found, pick the smallest overall
    if lowest_mse == np.inf:
        lowest_mse = min(mses)
        index = mses.index(lowest_mse)
        best_range = ranges[index]
        best_pm = pms[index]
        best_lm1 = lm1s[index]
        best_kbd = kbds[index]
        best_lm2 = lm2s[index]
        best_z_error_rate = z_error_rates[index]

    print("=" * 50)
    print("Best ranges:", best_range)
    print("*" * 50)
    print("Best z error rate: ", best_z_error_rate)
    print("=" * 50)

    if plot:
        plot_linear2(
            actual_depth,
            avg_50x50_anchor_disp,
            error,
            focal,
            baseline,
            (best_lm1, best_kbd, best_lm2),
            best_range,
            compensate_dist=compensate_dist,
            scaling_factor=scaling_factor,
            apply_global=apply_global,
            save_path=save_path,
        )

    return best_pm, best_range, best_z_error_rate


def generate_parameters_linear(
    df: pd.DataFrame,
    focal: float,
    baseline: float,
    save_path: str,
    disjoint_depth_range: tuple,
    compensate_dist: float = 200,
    scaling_factor: float = 10,
    apply_global=False,
    plot: bool = False,
):
    actual_depth = df[GT_DIST_NAME].values
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
    error = df[GT_ERROR_NAME].values

    jlm = JointLinearSmoothingOptimizer(
        actual_depth,
        avg_50x50_anchor_disp,
        focal,
        baseline,
        disjoint_depth_range,
        compensate_dist,
        scaling_factor,
        apply_global=apply_global,
        apply_weights=False,
        apply_l2=False,
    )

    linear_model1, res, linear_model2 = jlm.run()

    k_ = float(np.float64(res[0]))
    delta_ = float(np.float64(res[1]))
    b_ = float(np.float64(res[2]))

    if plot:
        plot_linear2(
            actual_depth,
            avg_50x50_anchor_disp,
            error,
            focal,
            baseline,
            (linear_model1, res, linear_model2),
            disjoint_depth_range,
            compensate_dist=compensate_dist,
            scaling_factor=scaling_factor,
            apply_global=apply_global,
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

    return params_matrix


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


def apply_transformation_linear(
    path: str,
    params_matrix: np.ndarray,
    focal: float,
    baseline: float,
    disjoint_depth_range: tuple,
    compensate_dist: float,
    scaling_factor: float,
) -> None:
    folders = retrive_folder_names(path)

    for folder in tqdm(folders):
        paths = retrive_file_names(os.path.join(path, folder, SUBFIX))
        for p in paths:
            full_path = os.path.join(path, folder, SUBFIX, p)
            raw = load_raw(full_path, H, W)
            depth = modify_linear(
                raw,
                H,
                W,
                focal,
                baseline,
                params_matrix,
                disjoint_depth_range,
                compensate_dist,
                scaling_factor,
            )
            # make sure raw value is within range(0, 65535)

            depth = np.clip(depth, UINT16_MIN, UINT16_MAX)
            depth = depth.astype(np.uint16)
            with open(full_path, "wb") as f:
                depth.tofile(f)
    print("Transformating data done ...")


def transformer_linear_impl(
    full_path,
    H,
    W,
    focal,
    baseline,
    params_matrix,
    disjoint_depth_range,
    compensate_dist,
    scaling_factor,
):
    raw = load_raw(full_path, H, W)
    depth = modify_linear(
        raw,
        H,
        W,
        focal,
        baseline,
        params_matrix,
        disjoint_depth_range,
        compensate_dist,
        scaling_factor,
    )
    depth = np.clip(depth, UINT16_MIN, UINT16_MAX)
    depth = depth.astype(np.uint16)
    with open(full_path, "wb") as f:
        depth.tofile(f)


def transformer_linear_vectorize_impl(
    full_path,
    H,
    W,
    focal,
    baseline,
    params_matrix,
    disjoint_depth_range,
    compensate_dist,
    scaling_factor,
):
    raw = load_raw(full_path, H, W).astype(np.float64)
    disp_ = np.divide(focal * baseline, raw, out=np.zeros_like(raw), where=(raw != 0))
    # disp_ = np.where(raw!=0, focal*baseline/raw, 0)

    depth = modify_linear_vectorize2(
        disp_,
        focal,
        baseline,
        params_matrix,
        disjoint_depth_range,
        compensate_dist,
        scaling_factor,
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
    disjoint_depth_range: tuple | list,
    compensate_dist: float,
    scaling_factor: float,
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
                        compensate_dist,
                        scaling_factor,
                    )
                )
        for future in tqdm(as_completed(tasks), total=len(tasks)):
            future.result()  # Ensure any exceptions are raised
    print("Transforming data done...")


def apply_transformation_linear_vectorize_parallel(
    path: str,
    params_matrix: np.ndarray,
    focal: float,
    baseline: float,
    disjoint_depth_range: tuple | list,
    compensate_dist: float,
    scaling_factor: float,
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
                        transformer_linear_vectorize_impl,
                        full_path,
                        H,
                        W,
                        focal,
                        baseline,
                        params_matrix,
                        disjoint_depth_range,
                        compensate_dist,
                        scaling_factor,
                    )
                )
        for future in tqdm(as_completed(tasks), total=len(tasks)):
            future.result()  # Ensure any exceptions are raised
    print("Transforming data done...")
