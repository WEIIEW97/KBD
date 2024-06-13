import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from scipy.optimize import differential_evolution
from bayes_opt import BayesianOptimization

from .constants import EPSILON
from .kernels import gaussian_kernel, polynomial_kernel_n2, laplacian_kernel


def fit_linear_model(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model


def model(disp, focal, baseline, k, delta, b):
    return k * focal * baseline / (disp + delta) + b


def model_kb(x, x_hat):

    def mfunc(params, x_hat):
        k, b = params
        y_hat = k * x_hat + b
        return y_hat

    def cost_func(params, x_hat, x):
        pred = mfunc(params, x_hat)
        return np.mean((x - pred) ** 2)

    initial_params = [1.0, 0]  # Starting values for k, b

    result = minimize(cost_func, initial_params, args=(x_hat, x), method="Nelder-Mead")

    print("Optimization Results:")
    print("Parameters (k, b):", result.x)
    print("Minimum MSE:", result.fun)
    if result.success:
        print("The optimization converged successfully.")
    else:
        print("The optimization did not converge:", result.message)

    return result


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
        mse = np.mean((actual_depth - predictions) ** 2)
        return mse

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


def model_kbd_joint_linear(
    actual_depth,
    disp,
    focal,
    baseline,
    disjoint_depth_range,
    compensate_dist=200,
    scaling_factor=10,
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
        (actual_depth > disjoint_depth_range[0])
        & (actual_depth < disjoint_depth_range[1])
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

    KBD_pred_disp_min = FB / KBD_pred_depth_max
    KBD_pred_disp_max = FB / KBD_pred_depth_min

    print(
        f"KBD model prediction on {KBD_disp_max} is {KBD_pred_depth_min}, where GT detph is {FB / KBD_disp_max} and prediction disp is {KBD_pred_disp_max}"
    )
    print(
        f"KBD model prediction on {KBD_disp_min} is {KBD_pred_depth_max}, where GT detph is {FB / KBD_disp_min} and prediction disp is {KBD_pred_disp_min}"
    )

    smooth_dist = compensate_dist

    pre_depth_joint = KBD_pred_depth_min - smooth_dist
    after_depth_joint = KBD_pred_depth_max + smooth_dist * scaling_factor

    pre_disp_joint = FB / pre_depth_joint
    after_disp_joint = FB / after_depth_joint

    linear_model1 = LinearRegression()
    X1 = np.array([pre_disp_joint, KBD_disp_max])
    y1 = np.array([pre_disp_joint, KBD_pred_disp_max])
    linear_model1.fit(X1.reshape(-1, 1), y1)

    linear_model2 = LinearRegression()
    X2 = np.array([KBD_disp_min, after_disp_joint])
    y2 = np.array([KBD_pred_disp_min, after_disp_joint])
    linear_model2.fit(X2.reshape(-1, 1), y2)

    return linear_model1, res, linear_model2


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


def model_poly_n2(actual_depth, disp, focal, baseline, reg_lambda=0.001):
    def mfunc(params, disp, baseline, focal):
        a, b, c, d = params
        y_hat = focal * baseline / (a * disp**2 + b * disp + c) + d
        return y_hat

    def cost_func(params, disp, baseline, focal, actual_depth):
        predictions = mfunc(params, disp, baseline, focal)
        mse = np.mean((predictions - actual_depth) ** 2)
        regularization = reg_lambda * np.sum(np.square(params))
        return mse + regularization

    initial_params = [0, 1.0, 0, 0]
    result = minimize(
        cost_func,
        initial_params,
        args=(disp, baseline, focal, actual_depth),
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


def model_kernel_fit(actual_depth, disp, focal, baseline, method="gaussian"):
    assert method in ("gaussian", "polynomial", "laplacian")

    def mfunc(params, disp, baseline, focal):
        if method == "gaussian":
            k, b, mu, sigma = params
            y_hat = k * focal * baseline / (disp + gaussian_kernel(disp, mu, sigma)) + b
            return y_hat
        elif method == "polynomial":
            k, b_, a, b, c = params
            y_hat = (
                k * focal * baseline / (disp + polynomial_kernel_n2(disp, a, b, c)) + b_
            )
            return y_hat
        elif method == "laplacian":
            k, b, mu, sigma = params
            y_hat = (
                k * focal * baseline / (disp + laplacian_kernel(disp, mu, sigma)) + b
            )
            return y_hat

    def cost_func(params, disp, baseline, focal, actual_depth):
        predictions = mfunc(params, disp, baseline, focal)
        mse = np.mean((predictions - actual_depth) ** 2)
        return mse

    initial_params = [1, 0, 0, 0]

    if method == "gaussian":
        initial_params = [1, 0, 0.1, 0.1]
    elif method == "polynomial":
        initial_params = [1, 0, 0.1, 0.1, 0.1]
    elif method == "laplacian":
        initial_params = [1, 0, 0.1, 0.1]

    result = minimize(
        cost_func,
        initial_params,
        args=(disp, baseline, focal, actual_depth),
        method="Nelder-Mead",
    )

    print("Optimization Results:")
    print("Parameters:", result.x)
    print("Minimum MSE:", result.fun)
    if result.success:
        print("The optimization converged successfully.")
    else:
        print("The optimization did not converge:", result.message)

    return result


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


def model_kbd_v2(
    actual_depth: np.ndarray,
    disp: np.ndarray,
    focal: float,
    baseline: float,
    weights=100,
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
        mse = np.mean((actual_depth - predictions) ** 2)
        error_less_than_1k = np.mean(
            np.abs(
                (predictions[actual_depth < 1000] - actual_depth[actual_depth < 1000])
                / actual_depth[actual_depth < 1000]
            )
        )
        return mse + weights * max(0, error_less_than_1k - 0.02)

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


def model_kbd_v3(
    actual_depth: np.ndarray,
    disp: np.ndarray,
    focal: float,
    baseline: float,
    weights=500,
):

    def mfunc(params, disp_norm, baseline, focal):
        k, delta, b = params
        y_hat = k * focal * baseline / (disp_norm + delta) + b
        return y_hat

    # Define the cost function (MSE)
    def cost_func(params, disp_norm, baseline, focal, actual_depth):
        predictions = mfunc(params, disp_norm, baseline, focal)
        mse = np.mean((actual_depth - predictions) ** 2)
        error_less_than_1k = np.mean(
            np.abs(
                (predictions[actual_depth < 1000] - actual_depth[actual_depth < 1000])
                / actual_depth[actual_depth < 1000]
            )
        )
        return mse + weights * max(0, error_less_than_1k - 0.02)

    # Initial guess for the parameters and bounds
    bounds = [(-1, 1), (-1, 1), (-100, 100)]
    result = differential_evolution(
        cost_func, bounds, args=(actual_depth, disp, focal, baseline), maxiter=2000
    )

    print("Optimization Results:")
    print("Parameters (k, delta, b):", result.x)
    print("Minimum MSE:", result.fun)
    if result.success:
        print("The optimization converged successfully.")
    else:
        print("The optimization did not converge:", result.message)

    return result


def model_kbd_v4(
    actual_depth: np.ndarray,
    disp: np.ndarray,
    focal: float,
    baseline: float,
    weights=500,
):
    def mfunc(params, disp_norm, baseline, focal):
        k, delta, b = params
        y_hat = k * focal * baseline / (disp_norm + delta) + b
        return y_hat

    # Define the cost function (MSE)
    def cost_func(params, disp_norm, baseline, focal, actual_depth):
        predictions = mfunc(params, disp_norm, baseline, focal)
        mse = np.mean((actual_depth - predictions) ** 2)
        error_less_than_1k = np.mean(
            np.abs(
                (predictions[actual_depth < 1000] - actual_depth[actual_depth < 1000])
                / actual_depth[actual_depth < 1000]
            )
        )
        return mse + weights * max(0, error_less_than_1k - 0.02)

    # Initial guess for the parameters and bounds
    bounds = [(-1000, 1000), (-100, 100), (-1000, 1000)]
    result_de = differential_evolution(
        cost_func, bounds, args=(actual_depth, disp, focal, baseline), maxiter=2000
    )

    optimized_params_de = result_de.x

    result_local = minimize(
        cost_func, optimized_params_de, args=(actual_depth, disp, focal, baseline)
    )

    print("Optimization Results:")
    print("Parameters (k, delta, b):", result_local.x)
    print("Minimum MSE:", result_local.fun)
    if result_local.success:
        print("The optimization converged successfully.")
    else:
        print("The optimization did not converge:", result_local.message)

    return result_local


def model_kbd_bayes(
    actual_depth: np.ndarray,
    disp: np.ndarray,
    focal: float,
    baseline: float,
    weights=500,
):
    def model_updated(disp, focal, baseline, k, delta, b):
        return k * (focal * baseline) / (disp + delta) + b

    def make_bayes_opt_function(avg_disp, focal, baseline, actual_depth, weights):
        def bayes_opt_function(k, delta, b):
            predicted_depth = model_updated(avg_disp, focal, baseline, k, delta, b)
            mse = np.mean((predicted_depth - actual_depth) ** 2)
            error_less_than_1000 = np.mean(
                np.abs(
                    (
                        predicted_depth[actual_depth < 1000]
                        - actual_depth[actual_depth < 1000]
                    )
                    / actual_depth[actual_depth < 1000]
                )
            )
            return -(mse + weights * max(0, error_less_than_1000 - 0.02))

        return bayes_opt_function

    bayes_opt_function = make_bayes_opt_function(
        disp, focal, baseline, actual_depth, weights
    )

    pbounds = {"k": (-1000, 1000), "delta": (-100, 100), "b": (-1000, 1000)}

    optimizer = BayesianOptimization(
        f=bayes_opt_function,
        pbounds=pbounds,
        random_state=1,
    )

    print("Optimizer Max:", optimizer.max)
    result = None

    if "params" in optimizer.max:
        optimized_params_bayes = optimizer.max["params"]
        k_bayes, delta_bayes, b_bayes = (
            optimized_params_bayes["k"],
            optimized_params_bayes["delta"],
            optimized_params_bayes["b"],
        )
        print("Optimized Parameters:", optimized_params_bayes)
        result = (
            optimized_params_bayes["k"],
            optimized_params_bayes["delta"],
            optimized_params_bayes["b"],
        )
    else:
        print("Optimizer max does not contain 'params' key.")

    return result
