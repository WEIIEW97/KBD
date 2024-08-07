from __future__ import annotations

import json

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from numba import jit

from scipy.optimize import least_squares, minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

##################### predefined constants #####################

DISP_VAL_MAX_UINT16 = 32767
SUBFIX = "DEPTH/raw"
H = 480
W = 640
EPSILON = 1e-6
UINT16_MIN = 0
UINT16_MAX = 65535
EVAL_WARNING_RATE = 0.5

ANCHOR_POINT = [H // 2, W // 2]
TARGET_POINTS = [300, 500, 600, 1000, 1500, 2000]
TARGET_THRESHOLDS = [0.02, 0.02, 0.02, 0.02, 0.04, 0.04]

AVG_DIST_NAME = "avg_depth_50x50_anchor"
AVG_DISP_NAME = "avg_disp_50x50_anchor"
MEDIAN_DIST_NAME = "median_depth_50x50_anchor"
MEDIAN_DISP_NAME = "median_disp_50x50_anchor"
GT_DIST_NAME = "actual_depth"
GT_ERROR_NAME = "absolute_error"
GT_DISP_ERROR_NAME = "absolute_disp_error"
GT_DIST_ERROR_NAME = "absolute_depth_error"
KBD_ERROR_NAME = "absolute_kbd_error"
FOCAL_NAME = "focal"
BASLINE_NAME = "baseline"
KBD_PRED_NAME = "kbd_pred"
MAPPED_PAIR_DICT = {
    "距离(mm)": "actual_depth",
    "相机焦距": "focal",
    "相机基线": "baseline",
    "绝对误差/mm": "absolute_error",
}


##################### functions #####################


def load_raw(path: str, h: int, w: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    return data.reshape((h, w))


def retrive_folder_names(path: str) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def retrive_file_names(path: str) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def copy_files_in_directory(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    files = retrive_file_names(src)
    for file in files:
        source = os.path.join(src, file)
        destination = os.path.join(dst, file)
        shutil.copy2(source, destination)


def calculate_mean_value(
    rootpath: str, folders: list[str], is_median: bool = False
) -> dict[str, float]:
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
            if is_median:
                mu = np.median(valid_raw)
            else:
                mu = np.mean(valid_raw)
            mean_dist_holder.append(mu)
        final_mu = np.mean(mean_dist_holder)
        dist_dict[distance] = final_mu
    return dist_dict


def read_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_table(path: str, pair_dict: dict) -> pd.DataFrame:
    pos = path.find("csv")
    if pos != -1:
        df = read_csv(path)
    else:
        df = read_excel(path)
    df_sel = df[list(pair_dict.keys())]
    needed_df = df_sel.rename(columns=pair_dict)
    return needed_df


def map_table(df: pd.DataFrame, dist_dict: dict) -> tuple[float, float]:
    df[AVG_DIST_NAME] = df[GT_DIST_NAME].astype(str).map(dist_dict)
    focal = df[FOCAL_NAME].iloc[0]  # assume focal value is the same
    focal *= 1.6
    baseline = df[BASLINE_NAME].iloc[0]  # assume basline value is the same

    df[AVG_DISP_NAME] = focal * baseline / df[AVG_DIST_NAME]

    return focal, baseline


def preprocessing(path, table_path, paid_dict=MAPPED_PAIR_DICT):
    all_distances = retrive_folder_names(path)
    mean_dists = calculate_mean_value(path, all_distances)
    df = read_table(table_path, pair_dict=paid_dict)
    focal, baseline = map_table(df, mean_dists)
    return df, focal, baseline


def eval(df: pd.DataFrame, stage=100):
    df["absolute_error_rate"] = df[GT_ERROR_NAME] / df[GT_DIST_NAME]
    max_stage = np.max(df[GT_DIST_NAME].values)
    n_stage = int(max_stage / stage)
    stages = [stage + i * stage for i in range(n_stage)]
    eval_res = dict()
    for s in stages:
        dt = df[(df["actual_depth"] <= s) & (df["actual_depth"] > s - stage)]
        mape = np.mean(np.abs(dt["absolute_error_rate"]))
        eval_res[s] = mape
    total_bins = len(eval_res)
    accept = 0
    for k, v in eval_res.items():
        if k <= 1000 and v < 0.02:
            accept += 1
        elif k <= 2000 and v < 0.04:
            accept += 1
    acceptance = accept / total_bins
    return eval_res, acceptance


def pass_or_not(df: pd.DataFrame):
    df["absolute_error_rate"] = df[GT_ERROR_NAME] / df[GT_DIST_NAME]
    metric_dist = [300, 500, 600, 1000, 1500, 2000]

    for metric in metric_dist:
        quali = df[df[GT_DIST_NAME] == metric]["absolute_error_rate"]
        quali = np.abs(quali)
        if metric in (300, 500, 600, 1000):
            if not (quali <= 0.02).all():
                return False
        elif metric in (1500, 2000):
            if not (quali <= 0.04).all():
                return False
    return True


def ratio_evaluate(alpha: float, df: pd.DataFrame, min_offset: int = 500):
    z = df[GT_DIST_NAME].values
    focal = df[FOCAL_NAME].values[0]
    baseline = df[BASLINE_NAME].values[0]
    d = focal * baseline / z
    indices = np.where(df[GT_DIST_NAME] >= min_offset)
    reciprocal_d = 1 / d
    ratio = 1 / (1 - alpha * reciprocal_d) - 1
    df["bound_ratio"] = ratio
    err_rate = np.abs(df[GT_ERROR_NAME] / df[GT_DIST_NAME])
    df["abs_error_rate"] = err_rate
    df["delta"] = ratio - err_rate
    ratio = ratio[indices]
    err_rate = np.array(err_rate)[indices]
    if ((ratio - err_rate) < 0).any():
        return False
    else:
        return True


def first_check(df: pd.DataFrame, max_thr: float = 0.07, mean_thr: float = 0.05):
    err_rate = np.abs(df[GT_ERROR_NAME] / df[GT_DIST_NAME])
    if np.max(err_rate) < max_thr and 0 <= np.mean(err_rate) < mean_thr:
        return True
    else:
        return False


def evaluate_target(
    focal,
    baseline,
    param_matrix,
    disjoint_depth_range,
    compensate_dist,
    scaling_factor,
    z=[300, 500, 600, 1000, 1500, 2000],
):
    z_array = np.array(z)
    d_array = focal * baseline / z_array
    z_after = modify_linear_vectorize2(
        d_array,
        focal,
        baseline,
        param_matrix,
        disjoint_depth_range,
        compensate_dist,
        scaling_factor,
    )
    z_error_rate = np.abs((z_after - z_array) / z_array)
    # print(f"z before is {z_array}")
    # print(f"z after is {z_after}")
    return mean_squared_error(z_array, z_after), z_error_rate


def model(disp, focal, baseline, k, delta, b):
    return k * focal * baseline / (disp + delta) + b


class NelderMeadOptimizer:
    def __init__(
        self,
        gt,
        est,
        focal,
        baseline,
        local_restriction_weights=1000,
        restriction_loc=1000,
        target_rate=0.02,
        apply_weights=False,
        apply_l2=False,
        reg_lambda=0.001,
    ):
        self.gt = gt
        self.est = est
        self.focal = focal
        self.baseline = baseline
        self.local_restriction_weights = local_restriction_weights
        self.restriction_loc = restriction_loc
        self.target_rate = target_rate
        self.reg_lambda = reg_lambda

        self.initial_params = [1.0, 0.01, 10]
        self.bounds = None

        self.apply_weights = apply_weights
        self.apply_l2 = apply_l2

    def loss(self, params):
        k, delta, b = params
        pred = model(self.est, self.focal, self.baseline, k, delta, b)
        residuals = self.gt - pred
        mse = np.mean(residuals**2)
        if self.apply_weights:
            local_restric = np.mean(
                np.abs(
                    (
                        pred[self.gt < self.restriction_loc]
                        - self.gt[self.gt < self.restriction_loc]
                    )
                    / self.gt[self.gt < self.restriction_loc]
                )
            )
        else:
            local_restric = 0
        if self.apply_l2:
            l2_reg = self.reg_lambda * np.sum(np.square(params))
        else:
            l2_reg = 0
        return (
            mse
            + self.local_restriction_weights * max(0, local_restric - self.target_rate)
            + l2_reg
        )

    def optimize(self, initial_params, bounds):
        result = minimize(
            self.loss, initial_params, bounds=bounds, method="Nelder-Mead"
        )
        return result

    def run(self):
        result = self.optimize(self.initial_params, self.bounds)
        print("Optimization Result:")

        if result.success:
            optimized_params = result.x
            k, delta, b = optimized_params

            pred = model(self.est, self.focal, self.baseline, k, delta, b)

            mse = np.mean((pred - self.gt) ** 2)
            if self.apply_weights:
                local_restric = np.mean(
                    np.abs(
                        (
                            pred[self.gt < self.restriction_loc]
                            - self.gt[self.gt < self.restriction_loc]
                        )
                        / self.gt[self.gt < self.restriction_loc]
                    )
                )
            print("MSE:", mse)
            if self.apply_weights:
                print(f"Error less than {self.restriction_loc}:", local_restric)
            print("Optimized Parameters:", optimized_params)

            return k, delta, b
        else:
            print("Optimization failed.")


class TrustRegionReflectiveOptimizer:
    def __init__(
        self,
        gt,
        est,
        focal,
        baseline,
        local_restriction_weights=1000,
        restriction_loc=1000,
        target_rate=0.02,
    ):
        self.gt = gt
        self.est = est
        self.focal = focal
        self.baseline = baseline
        self.local_restriction_weights = local_restriction_weights
        self.restriction_loc = restriction_loc
        self.target_rate = target_rate

        self.initial_params = [1.0, 0.01, 10]
        self.bounds = ([0, -10, -100], [10, 10, 100])

    def loss(self, params):
        k, delta, b = params
        pred = model(self.est, self.focal, self.baseline, k, delta, b)
        residuals = pred - self.gt
        # mse = np.mean(residuals**2)

        local_restric = np.abs(
            (
                pred[self.gt < self.restriction_loc]
                - self.gt[self.gt < self.restriction_loc]
            )
            / self.gt[self.gt < self.restriction_loc]
        )
        return np.concatenate(
            (
                residuals,
                self.local_restriction_weights
                * np.maximum(0, local_restric - self.target_rate),
            )
        )

    def optimize(self, initial_params, bounds):
        result = least_squares(self.loss, initial_params, bounds=bounds)
        return result

    def run(self):
        result = self.optimize(self.initial_params, self.bounds)
        print("Optimization Result:", result)

        if result.success:
            optimized_params = result.x
            k, delta, b = optimized_params

            pred = model(self.est, self.focal, self.baseline, k, delta, b)

            mse = np.mean((pred - self.gt) ** 2)
            local_restric = np.mean(
                np.abs(
                    (
                        pred[self.gt < self.restriction_loc]
                        - self.gt[self.gt < self.restriction_loc]
                    )
                    / self.gt[self.gt < self.restriction_loc]
                )
            )

            print("MSE:", mse)
            print(f"Error less than {self.restriction_loc}:", local_restric)
            print("Optimized Parameters:", optimized_params)

            return k, delta, b
        else:
            print("Optimization failed.")


class JointLinearSmoothingOptimizer:
    def __init__(
        self,
        gt,
        est,
        focal,
        baseline,
        disjoint_depth_range,
        compensate_dist=200,
        scaling_factor=10,
        engine="Nelder-Mead",
        local_restriction_weights=1000,
        target_rate=0.02,
        apply_global=False,
        apply_weights=False,
        apply_l2=False,
        reg_lambda=0.001,
    ):
        assert engine in (
            "Nelder-Mead",
            "Trust-Region",
        ), f"optimize engine {engine} is not supported!"
        self.gt = gt
        self.est = est
        self.focal = focal
        self.baseline = baseline
        self.local_restriction_weights = local_restriction_weights
        self.restriction_loc = disjoint_depth_range[0]
        self.target_rate = target_rate
        self.disjoint_depth_range = disjoint_depth_range
        self.compensate_dist = compensate_dist
        self.scaling_factor = scaling_factor
        self.apply_global = apply_global
        self.apply_weights = apply_weights
        self.apply_l2 = apply_l2
        self.reg_lambda = reg_lambda

        self.fb = focal * baseline
        self.engine = engine

        self.initial_params = [1.0, 0.01, 10]
        self.bounds = ([0, -10, -100], [10, 10, 100])

    def segment(self):
        # find the range to calculate KBD params within

        mask = np.where(
            (self.gt > self.disjoint_depth_range[0])
            & (self.gt < self.disjoint_depth_range[1])
        )
        if not self.apply_global:
            self.kbd_x = self.est[mask]
            self.kbd_y = self.gt[mask]
        else:
            self.kbd_x = self.est
            self.kbd_y = self.gt

        kbd_base_optimizer = None

        if self.engine == "Nelder-Mead":
            kbd_base_optimizer = NelderMeadOptimizer(
                self.kbd_y,
                self.kbd_x,
                self.focal,
                self.baseline,
                self.local_restriction_weights,
                self.restriction_loc,
                self.target_rate,
                self.apply_weights,
                self.apply_l2,
                self.reg_lambda,
            )
        elif self.engine == "Trust-Region":
            kbd_base_optimizer = TrustRegionReflectiveOptimizer(
                self.kbd_y,
                self.kbd_x,
                self.focal,
                self.baseline,
                restriction_loc=self.restriction_loc,
            )

        kbd_result = kbd_base_optimizer.run()
        return kbd_result

    def run(self):
        kbd_result = self.segment()
        if kbd_result is not None:
            k_, delta_, b_ = kbd_result
            x_min = np.min(self.kbd_x)
            x_max = np.max(self.kbd_x)

            y_hat_max = k_ * self.fb / (x_min + delta_) + b_
            y_hat_min = k_ * self.fb / (x_max + delta_) + b_

            x_hat_min = self.fb / y_hat_max
            x_hat_max = self.fb / y_hat_min

            pre_y = y_hat_min - self.compensate_dist
            after_y = y_hat_max + self.compensate_dist * self.scaling_factor

            pre_x = self.fb / pre_y
            after_x = self.fb / after_y

            lm1 = LinearRegression()
            x1 = np.array([pre_x, x_max])
            y1 = np.array([pre_x, x_hat_max])
            lm1.fit(x1.reshape(-1, 1), y1)

            lm2 = LinearRegression()
            x2 = np.array([x_min, after_x])
            y2 = np.array([x_hat_min, after_x])
            lm2.fit(x2.reshape(-1, 1), y2)

            return lm1, kbd_result, lm2
        return


def generate_parameters_linear(
    df: pd.DataFrame,
    focal: float,
    baseline: float,
    disjoint_depth_range: tuple,
    compensate_dist: float = 200,
    scaling_factor: float = 10,
    engine="Nelder-Mead",
    apply_global=False,
    apply_weights=False,
    apply_l2=False,
):

    actual_depth = df[GT_DIST_NAME].values
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values

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
        apply_weights=apply_weights,
        apply_l2=apply_l2,
    )

    linear_model1, res, linear_model2 = jlm.run()

    k_ = float(np.float64(res[0]))
    delta_ = float(np.float64(res[1]))
    b_ = float(np.float64(res[2]))

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


def generate_parameters_linear_search(
    df: pd.DataFrame,
    focal: float,
    baseline: float,
    search_range: tuple,
    compensate_dist: float = 200,
    scaling_factor: float = 10,
    engine="Nelder-Mead",
    apply_global=False,
):
    actual_depth = df[GT_DIST_NAME].values
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
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
                best_z_error_rate = z_error_rates[i]

    # If no suitable mse is found, pick the smallest overall
    if lowest_mse == np.inf:
        lowest_mse = min(mses)
        index = mses.index(lowest_mse)
        best_range = ranges[index]
        best_pm = pms[index]
        best_z_error_rate = z_error_rates[index]

    print("=" * 50)
    print("Best ranges:", best_range)
    print("*" * 50)
    print("Best z error rate: ", best_z_error_rate)
    print("=" * 50)

    return best_pm, best_range, best_z_error_rate

class GridSearch2D:
    def __init__(
        self,
        df,
        focal,
        baseline,
        scaling_factor,
        engine="Nelder-Mead",
        apply_global=False,
    ):
        # Validate inputs
        if df.empty:
            raise ValueError("DataFrame is empty.")
        if not all(
            x in df.columns for x in [GT_DIST_NAME, AVG_DISP_NAME, GT_ERROR_NAME]
        ):
            raise ValueError("DataFrame missing required columns.")

        self.df = df
        self.focal = focal
        self.baseline = baseline
        self.engine = engine
        self.apply_global = apply_global
        self.scaling_factor = scaling_factor

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
            self.focal,
            self.baseline,
            pm,
            disjoint_depth_range,
            compensate_dist,
            self.scaling_factor,
            TARGET_POINTS,
        )
        return mse, pm, (lm1, kbd, lm2)

    def optimize_parameters(self, search_range, cd_range, max_iter=1000, tol=1e-6):
        def objective(x):
            mse, pm, params = self.eval_parameters(x[0], x[1])
            # if params[0].coef_[0] <= 0:
            #     mse = 10e7
            return mse

        initial_guess = [search_range[0], cd_range[0]]
        bounds = [search_range, cd_range]

        self.result = minimize(
            objective,
            initial_guess,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False, "xatol": tol},
        )

    def get_results(self):
        if self.result.success:
            optimized_range_start, optimized_compensate_dist = self.result.x
            best_mse, best_pm, best_params = self.eval_parameters(
                optimized_range_start, optimized_compensate_dist
            )
            print("Optimization successful.")
            print(f"Optimized range start: {optimized_range_start}")
            print(f"Optimized compensate distance: {optimized_compensate_dist}")
            print(f"Minimum MSE: {best_mse}")

            return (best_pm, optimized_range_start, optimized_compensate_dist)
        else:
            print("Optimization failed: " + self.result.message)
            return (-1, -1, -1)


def save_arrays_to_json(savepath, arr1d, arr2d):
    arr1d_lst = arr1d.tolist()
    arr2d_lst = arr2d.tolist()

    params = {"disp_nodes": arr1d_lst, "kbd_params": arr2d_lst}

    with open(savepath, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Arrays have been saved to {savepath}")


def parallel_copy(src: str, dst: str) -> None:
    folders = retrive_folder_names(src)
    cam_name = "camparam.txt"

    with ThreadPoolExecutor() as executor:
        for folder in tqdm(folders, desc="Copying subfolders ..."):
            source_path = os.path.join(src, folder, SUBFIX)
            destination_path = os.path.join(dst, folder, SUBFIX)
            cam_source = os.path.join(src, folder, cam_name)
            cam_dest = os.path.join(dst, folder, cam_name)
            executor.submit(copy_files_in_directory, source_path, destination_path)
            executor.submit(shutil.copy2, cam_source, cam_dest)
    print("Copying done ...")


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


def modify_linear_vectorize2(
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
    output depth follows the formula below:
    D = k*fb/(alpha*d + beta + delta) + b
    """
    fb = focal * baseline
    out = np.zeros_like(m)
    mask0 = np.zeros_like(m)
    mask1 = np.zeros_like(m)
    mask2 = np.zeros_like(m)
    mask3 = np.zeros_like(m)
    mask4 = np.zeros_like(m)

    lb = disjoint_depth_range[0]
    ub = disjoint_depth_range[1]
    thr = compensate_dist
    sf = scaling_factor

    conditions = [
        (m != 0) & (m > fb / (lb - thr)),
        (m != 0) & (m <= fb / (lb - thr)) & (m > fb / lb),
        (m != 0) & (m <= fb / lb) & (m > fb / ub),
        (m != 0) & (m <= fb / ub) & (m > fb / (ub + thr * sf)),
        (m != 0) & (m <= fb / (ub + thr * sf)),
    ]

    # Apply conditions to masks
    mask0[conditions[0]] = 1
    mask1[conditions[1]] = 1
    mask2[conditions[2]] = 1
    mask3[conditions[3]] = 1
    mask4[conditions[4]] = 1

    out = (
        mask0 * np.divide(fb, m, out=np.zeros_like(m), where=m != 0)
        + mask1
        * np.divide(
            fb,
            param_matrix[1, 3] * m + param_matrix[1, 4],
            out=np.zeros_like(m),
            where=m != 0,
        )
        + mask2
        * (
            param_matrix[2, 0]
            * np.divide(fb, m + param_matrix[2, 1], out=np.zeros_like(m), where=m != 0)
            + param_matrix[2, 2]
        )
        + mask3
        * np.divide(
            fb,
            param_matrix[3, 3] * m + param_matrix[3, 4],
            out=np.zeros_like(m),
            where=m != 0,
        )
        + mask4 * np.divide(fb, m, out=np.zeros_like(m), where=m != 0)
    )

    return out


def export_default_settings(path, focal, baseline, compensate_dist, scaling_factor):
    default_range = (600, 3000)
    extra_range = [
        default_range[0] - compensate_dist,
        default_range[0],
        default_range[1],
        default_range[1] + compensate_dist * scaling_factor,
    ]
    disp_nodes_fp32 = focal * baseline / (np.array(extra_range))
    disp_nodes_uint16 = (disp_nodes_fp32 * 64).astype(np.uint16)
    disp_nodes_uint16 = np.sort(disp_nodes_uint16)
    disp_nodes_uint16 = np.append(disp_nodes_uint16, DISP_VAL_MAX_UINT16)
    default_param = np.array([1, 0, 0, 1, 0])
    matrix = np.tile(default_param, (5, 1))
    matrix_param_by_disp = matrix[::-1, :]
    save_arrays_to_json(path, disp_nodes_uint16, matrix_param_by_disp)
    return default_range, matrix


if __name__ == "__main__":
    cwd = "/home/william/extdisk/data/20240803"
    # please make adjustments to them accordingly
    # disjoint_depth_range = [600, 3000]
    engine = "Nelder-Mead"  # Nelder-Mead or Trust-Region
    camera_type = "N9LAZG24GN0293"
    table_name = "depthquality_2024-08-02.xlsx"
    apply_global = False
    bound_ratio_alpha = 0.7
    sample_weights_factor = 3.0
    scaling_factor = 10
    search_range = (600, 1100)
    cd_range = (100, 400)
    global_judge = "global" if apply_global else "local"
    optimizer_judge = "nelder-mead" if engine == "Nelder-Mead" else "trust-region"
    status = -1
    print(f"processing {camera_type} now with {table_name} ...")

    root_dir = f"{cwd}/{camera_type}/image_data"
    copy_dir = (
        f"{cwd}/{camera_type}/image_data_transformed_linear_{global_judge}_scale1.6"
    )
    save_dir = f"{cwd}/{camera_type}"
    tablepath = f"{cwd}/{camera_type}/{table_name}"
    save_params_path = (
        save_dir + f"/params_{global_judge}_{optimizer_judge}_scale1.6.json"
    )

    df, focal, baseline = preprocessing(root_dir, tablepath)
    export_original = False
    if not first_check(df) or not pass_or_not(df):
        # trial = ratio_evaluate(bound_ratio_alpha, df)
        # if not trial:
        #     print("Ratio bound evaluation test failed.")
        #     print("Will not push to further process ...")
        #     raise ValueError("Ratio-Bound test failed!")

        eval_res, acceptance_rate = eval(df)

        if acceptance_rate < EVAL_WARNING_RATE:
            print("*********** WARNING *************")
            print(
                f"Please be really cautious since the acceptance rate is {acceptance_rate},"
            )
            print("This may not be the ideal data to be tackled with.")
            print("*********** END OF WARNING *************")
        print("Begin to generate parameters with line searching...")

        GridSearchOptimizer = GridSearch2D(df, focal, baseline, scaling_factor, engine=engine, apply_global=apply_global)
        GridSearchOptimizer.optimize_parameters(search_range, cd_range)
        matrix, best_range_start, best_cd = GridSearchOptimizer.get_results()
        best_range = (best_range_start, 3000)
        ## add before/after kbd comparsion
        X = df[AVG_DISP_NAME]
        Y = df[GT_DIST_NAME]
        pred = modify_linear_vectorize2(
            X.values,
            focal,
            baseline,
            matrix,
            best_range,
            best_cd,
            scaling_factor,
        )
        df[KBD_PRED_NAME] = pred
        df[GT_DIST_ERROR_NAME] = np.abs(df[GT_ERROR_NAME]/Y)
        df[KBD_ERROR_NAME] = np.abs((Y-pred)/Y)
        df_criteria = df[df[GT_DIST_NAME].isin(np.array(TARGET_POINTS))]
        weights = Y.apply(lambda x: sample_weights_factor if x in TARGET_POINTS else 1.0)
        previous_mse = mean_squared_error(Y, X, sample_weight=weights)
        after_mse = mean_squared_error(Y, df[KBD_PRED_NAME], sample_weight=weights)
        ##################################
        if after_mse < previous_mse:
            range_raw = best_range
            extra_range = [
                range_raw[0] - best_cd,
                range_raw[0],
                range_raw[1],
                range_raw[1] + best_cd * scaling_factor,
            ]
            disp_nodes_fp32 = focal * baseline / (np.array(extra_range))
            disp_nodes_uint16 = (disp_nodes_fp32 * 64).astype(np.uint16)
            disp_nodes_uint16 = np.sort(disp_nodes_uint16)
            disp_nodes_uint16 = np.append(disp_nodes_uint16, DISP_VAL_MAX_UINT16)

            matrix_param_by_disp = matrix[::-1, :]
            save_arrays_to_json(
                save_params_path, disp_nodes_uint16, matrix_param_by_disp
            )
            cond = (df_criteria[KBD_ERROR_NAME] < TARGET_THRESHOLDS)
            if cond.all():
                status = 0
            else:
                status = 1
        else:
            best_range, matrix = export_default_settings(
                save_params_path, focal, baseline, best_cd, scaling_factor
            )
            export_original = True
            cond = (df_criteria[KBD_ERROR_NAME] < TARGET_THRESHOLDS)
            if cond.all():
                status = 0
            else:
                status = 2
    else:
        best_cd = 100
        best_range, matrix = export_default_settings(
            save_params_path, focal, baseline, best_cd, scaling_factor
        )
        export_original = True
        status = 0
    print("===> Working done! Parameters are being generated.")

    print("Beging to copy and transform depth raws ...")
    parallel_copy(root_dir, copy_dir)
    if not export_original:
        apply_transformation_linear(
            copy_dir,
            matrix,
            focal,
            baseline,
            best_range,
            best_cd,
            scaling_factor,
        )
    print("===> Working done! Transformed raw depths are being generated.")
