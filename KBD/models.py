import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


def fit_linear_model(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model


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


def model_kbd_joint_linear(actual_depth, disp, focal, baseline, disjoint_depth_range=(500, 600)):
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
    def mfunc(params, disp, baseline, focal):
        k, delta, b = params
        return k * focal * baseline / (disp + delta) + b

    def cost_func(params, disp, baseline, focal, actual_depth):
        predictions = mfunc(params, disp, baseline, focal)
        return np.mean((actual_depth - predictions) ** 2)

    # Filter data where actual_depth >= 500
    mask = np.where(actual_depth >= disjoint_depth_range[0])
    filtered_disp = disp[mask]
    filtered_depth = actual_depth[mask]

    # Fit the model on the filtered data
    initial_params = [1.0, 0.01, 10]  # Reasonable starting values
    result = minimize(
        cost_func,
        initial_params,
        args=(filtered_disp, baseline, focal, filtered_depth),
        method="Nelder-Mead"
    )

    # find the estimiated disparity to depth range within [500, 600]
    fb = focal*baseline
    k_, delta_, b_ = result.x
    d_sup = fb/disjoint_depth_range[0]
    d_inf = fb/disjoint_depth_range[1]

    depth_estimated_d_sup = k_*fb/(d_inf+delta_) + b_
    actual_disp = fb / actual_depth

    # Fit linear model for the range [500, 600] to ensure continuity
    mask_linear = np.where((actual_depth >= 500) & (actual_depth <= depth_estimated_d_sup))
    x_linear = disp[mask_linear]
    y_linear = actual_disp[mask_linear]
    linear_model = fit_linear_model(x_linear, y_linear)

    return linear_model, result


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
