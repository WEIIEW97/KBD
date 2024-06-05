import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from sklearn.linear_model import LinearRegression

from .helpers import (
    preprocessing,
    retrive_file_names,
    retrive_folder_names,
)
from .utils import load_raw, depth2disp, get_linear_model_params, json_dumper
from .models import (
    model_kernel_fit,
    model_kbd_v3,
    model_kbd_bayes,
)
from .constants import (
    UINT16_MIN,
    UINT16_MAX,
    H,
    W,
    SUBFIX,
    EPSILON,
    GT_DIST_NAME,
    AVG_DISP_NAME,
    GT_ERROR_NAME,
    OUT_PARAMS_FILE_NAME,
    OUT_FIG_RESIDUAL_FILE_NAME,
    OUT_FIG_COMP_FILE_NAME,
    OUT_FIG_ERROR_RATE_FILE_NAME,
    LINEAR_OUT_PARAMS_FILE_NAME,
)
from .core import modify, modify_linear, modify_linear_vectorize, modify_linear_vectorize2
from .plotters import plot_error_rate, plot_comparison, plot_residuals, plot_linear
from .kernels import gaussian_kernel, polynomial_kernel_n2, laplacian_kernel
from .optimizers import (
    TrustRegionReflectiveOptimizer,
    JointLinearSmoothingOptimizer,
    NelderMeadOptimizer,
)


def generate_parameters(
    path: str,
    table_path: str,
    save_path: str,
    use_l2: bool = False,
    reg_lambda: float = 0.01,
):
    df, focal, baseline = preprocessing(path=path, table_path=table_path)

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

    pred = k_ * focal * baseline / (avg_50x50_anchor_disp + delta_) + b_
    residual = pred - actual_depth
    plot_residuals(residual, error, actual_depth, residual_path)
    plot_error_rate(residual, error, actual_depth, error_rate_path)
    plot_comparison(
        actual_depth, focal * baseline / avg_50x50_anchor_disp, pred, comp_path
    )

    return k_, delta_, b_, focal, baseline


def generate_parameters_trf(
    path: str,
    table_path: str,
    save_path: str,
):
    df, focal, baseline = preprocessing(path=path, table_path=table_path)

    actual_depth = df[GT_DIST_NAME].values
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
    error = df[GT_ERROR_NAME].values

    trf = TrustRegionReflectiveOptimizer(
        actual_depth, avg_50x50_anchor_disp, focal, baseline
    )
    k_, delta_, b_, residuals_ = trf.run()

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

    pred = k_ * focal * baseline / (avg_50x50_anchor_disp + delta_) + b_
    residual = pred - actual_depth
    plot_residuals(residual, error, actual_depth, residual_path)
    plot_error_rate(residual, error, actual_depth, error_rate_path)
    plot_comparison(
        actual_depth, focal * baseline / avg_50x50_anchor_disp, pred, comp_path
    )

    return k_, delta_, b_, focal, baseline


def generate_parameters_adv(path: str, table_path: str, save_path: str, method="evo"):
    df, focal, baseline = preprocessing(path=path, table_path=table_path)

    actual_depth = df[GT_DIST_NAME].values
    avg_50x50_anchor_disp = df[AVG_DISP_NAME].values
    error = df[GT_ERROR_NAME].values

    assert method in ("evo", "bayes")

    res = None

    if method == "evo":
        res = model_kbd_v3(actual_depth, avg_50x50_anchor_disp, focal, baseline)
    elif method == "bayes":
        res = model_kbd_bayes(actual_depth, avg_50x50_anchor_disp, focal, baseline)

    common_prefix = f"{method}_"
    param_path = os.path.join(save_path, common_prefix + OUT_PARAMS_FILE_NAME)
    comp_path = os.path.join(save_path, common_prefix + OUT_FIG_COMP_FILE_NAME)
    residual_path = os.path.join(save_path, common_prefix + OUT_FIG_RESIDUAL_FILE_NAME)
    error_rate_path = os.path.join(
        save_path, common_prefix + OUT_FIG_ERROR_RATE_FILE_NAME
    )

    # params_dict = {"k": str(res.x[0]), "delta": str(res.x[1]), "b": str(res.x[2])}
    if method == "evo":
        k_ = float(np.float64(res.x[0]))
        delta_ = float(np.float64(res.x[1]))
        b_ = float(np.float64(res.x[2]))
    elif method == "bayes":
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

    pred = k_ * focal * baseline / (avg_50x50_anchor_disp + delta_) + b_
    residual = pred - actual_depth
    plot_residuals(residual, error, actual_depth, residual_path)
    plot_error_rate(residual, error, actual_depth, error_rate_path)
    plot_comparison(
        actual_depth, focal * baseline / avg_50x50_anchor_disp, pred, comp_path
    )
    error_less_than_1000_bayes = np.mean(
        np.abs(
            (pred[actual_depth < 1000] - actual_depth[actual_depth < 1000])
            / actual_depth[actual_depth < 1000]
        )
    )

    print(f"error in <1000 is {error_less_than_1000_bayes}")

    return k_, delta_, b_, focal, baseline


def generate_parameters_linear(
    path: str,
    table_path: str,
    save_path: str,
    disjoint_depth_range: tuple,
    compensate_dist: float = 200,
    scaling_factor: float = 10,
):
    df, focal, baseline = preprocessing(path=path, table_path=table_path)

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
    )

    linear_model1, res, linear_model2 = jlm.run()

    param_path = os.path.join(save_path, LINEAR_OUT_PARAMS_FILE_NAME)

    k_ = float(np.float64(res[0]))
    delta_ = float(np.float64(res[1]))
    b_ = float(np.float64(res[2]))

    linear_model1_params = get_linear_model_params(linear_model1)
    linear_model2_params = get_linear_model_params(linear_model2)

    ### do not support the shared pointer

    def create_default_linear_model_params():
        default_linear_model = LinearRegression()
        default_linear_model.coef_ = np.array([1.0])
        default_linear_model.intercept_ = np.array(0.0)
        return get_linear_model_params(default_linear_model)

    params_dict = OrderedDict(
        [
            (
                f"{0}-{disjoint_depth_range[0]-compensate_dist}",
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
                f"{disjoint_depth_range[0]-compensate_dist}-{disjoint_depth_range[0]}",
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
                f"{disjoint_depth_range[0]}-{disjoint_depth_range[1]}",
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
                f"{disjoint_depth_range[1]}-{disjoint_depth_range[1]+compensate_dist*scaling_factor}",
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
                f"{disjoint_depth_range[1]+compensate_dist*scaling_factor}-{np.inf}",
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

    json_dumper(params_dict, param_path)
    print("Generating done...")

    plot_linear(
        actual_depth,
        avg_50x50_anchor_disp,
        error,
        focal,
        baseline,
        (linear_model1, res, linear_model2),
        disjoint_depth_range,
        compensate_dist=compensate_dist,
        scaling_factor=scaling_factor,
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


def generate_parameters_kernel(
    path: str,
    table_path: str,
    save_path: str,
    method="gaussian",
):
    df, focal, baseline = preprocessing(path=path, table_path=table_path)

    actual_depth = df[GT_DIST_NAME]
    avg_50x50_anchor_disp = df[AVG_DISP_NAME]
    error = df[GT_ERROR_NAME]

    res = model_kernel_fit(
        actual_depth, avg_50x50_anchor_disp, focal, baseline, method=method
    )

    common_prefix = f"{method}_"
    param_path = os.path.join(save_path, common_prefix + OUT_PARAMS_FILE_NAME)
    comp_path = os.path.join(save_path, common_prefix + OUT_FIG_COMP_FILE_NAME)
    residual_path = os.path.join(save_path, common_prefix + OUT_FIG_RESIDUAL_FILE_NAME)
    error_rate_path = os.path.join(
        save_path, common_prefix + OUT_FIG_ERROR_RATE_FILE_NAME
    )

    if method == "gaussian":
        k = float(np.float64(res.x[0]))
        b = float(np.float64(res.x[1]))
        mu = float(np.float64(res.x[2]))
        sigma = float(np.float64(res.x[3]))
        params_dict = {
            "k": k,
            "b": b,
            "mu": mu,
            "sigma": sigma,
        }
        print(params_dict)

        json_dumper(params_dict, param_path)
        print("Generating done...")

        pred = (
            k
            * focal
            * baseline
            / (
                avg_50x50_anchor_disp
                + gaussian_kernel(avg_50x50_anchor_disp, mu, sigma)
            )
            + b
        )
        residual = pred - actual_depth
        plot_residuals(residual, error, actual_depth, residual_path)
        plot_error_rate(residual, error, actual_depth, error_rate_path)
        plot_comparison(
            actual_depth, focal * baseline / avg_50x50_anchor_disp, pred, comp_path
        )

        return k, b, mu, sigma, focal, baseline

    if method == "polynomial":
        k = float(np.float64(res.x[0]))
        b_ = float(np.float64(res.x[1]))
        a = float(np.float64(res.x[2]))
        b = float(np.float64(res.x[3]))
        c = float(np.float64(res.x[4]))

        params_dict = {
            "k": k,
            "b_": b_,
            "a": a,
            "b": b,
            "c": c,
        }
        print(params_dict)

        json_dumper(params_dict, param_path)
        print("Generating done...")

        pred = (
            k
            * focal
            * baseline
            / (
                avg_50x50_anchor_disp
                + polynomial_kernel_n2(avg_50x50_anchor_disp, a, b, c)
            )
            + b_
        )
        residual = pred - actual_depth
        plot_residuals(residual, error, actual_depth, residual_path)
        plot_error_rate(residual, error, actual_depth, error_rate_path)
        plot_comparison(
            actual_depth, focal * baseline / avg_50x50_anchor_disp, pred, comp_path
        )

        return k, b_, a, b, c, focal, baseline

    if method == "laplacian":
        k = float(np.float64(res.x[0]))
        b = float(np.float64(res.x[1]))
        mu = float(np.float64(res.x[2]))
        sigma = float(np.float64(res.x[3]))
        params_dict = {
            "k": k,
            "b": b,
            "mu": mu,
            "sigma": sigma,
        }
        print(params_dict)

        json_dumper(params_dict, param_path)
        print("Generating done...")

        pred = (
            k
            * focal
            * baseline
            / (
                avg_50x50_anchor_disp
                + laplacian_kernel(avg_50x50_anchor_disp, mu, sigma)
            )
            + b
        )
        residual = pred - actual_depth
        plot_residuals(residual, error, actual_depth, residual_path)
        plot_error_rate(residual, error, actual_depth, error_rate_path)
        plot_comparison(
            actual_depth, focal * baseline / avg_50x50_anchor_disp, pred, comp_path
        )

        return k, b, mu, sigma, focal, baseline


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
    disp_ = np.divide(focal*baseline, raw, out=np.zeros_like(raw), where=(raw!=0))
    # disp_ = np.where(raw!=0, focal*baseline/raw, 0)
    depth = modify_linear_vectorize2(
        disp_, focal, baseline, params_matrix, disjoint_depth_range, compensate_dist, scaling_factor
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
