import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


def load_raw(path, h, w):
    data = np.fromfile(path, dtype=np.uint16)
    return data.reshape((h, w))


def normalize(x: pd.DataFrame):
    scaler = MinMaxScaler()
    return scaler.fit_transform(x.values.reshape(-1, 1)).flatten()


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


def model_kbd(actual_depth: pd.DataFrame, disp: pd.DataFrame, focal, baseline):
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


def model_kbd_further_optimized(actual_depth, disp, focal, baseline, reg_lambda=0.001):
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
        cost_func, initial_params, args=(disp_norm, baseline, focal, actual_depth),
        method='Nelder-Mead'
    )

    print("Optimization Results:")
    print("Parameters (k, delta, b):", result.x)
    print("Minimum MSE:", result.fun)
    if result.success:
        print("The optimization converged successfully.")
    else:
        print("The optimization did not converge:", result.message)

    return result


def plot_residuals(residuals, error, gt):
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
    plt.show()

    # Print the mean of the residuals
    mean_residuals = np.mean(residuals)
    print("Mean of residuals:", mean_residuals)


def plot_error_rate(residuals, error, nominator):
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
    plt.xlabel("Ground truth distance (mm)")
    plt.ylabel("Residuals (Error) rate vs original error rate (%)")
    plt.title("Error rate Plot (%)")
    plt.legend()
    plt.show()


def plot_comparison(x, y1, y2):
    fig, ax = plt.subplots()
    ax.plot(x, y1, label="measured data", marker="o")
    ax.plot(x, y2, label="fitted data", marker="x")

    ax.legend()
    plt.show()


def plot_illustration_fig(lraw_path, depth_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    lraw = mpimg.imread(lraw_path)
    h, w = lraw.shape[0], lraw.shape[1]
    depth = load_raw(depth_path, h, w)

    axs[0].imshow(lraw)
    axs[0].axis("off")
    axs[0].set_title("left raw image")

    # Show second image
    axs[1].imshow(depth)
    axs[1].axis("off")
    axs[1].set_title("depth image")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_comparison4(x, y1, y2, y3, y4):
    fig, ax = plt.subplots()
    ax.plot(x, y1, label="fitted data", marker="o")
    ax.plot(x, y2, label="center half size crop", marker="x")
    ax.plot(x, y3, label="center 50x50 crop", marker="x")
    ax.plot(x, y4, label="center 50x50 crop with anchor", marker="x")

    ax.legend()
    plt.show()


def linear_fitting(X, y):
    model = LinearRegression()
    model.fit(X, y)
    k = model.coef_
    b = model.intercept_

    print("Slope (k):", k)
    print("Intercept (b):", b)
    return model


def pipeline(gt, est, error, focal=None, baseline=None, method="KB", desc=""):
    assert method in ("KB", "KBD")
    if method == "KB":
        print(f"Starting optimization for KB model with {desc} ...")
        res = model_kb(gt, est)
        params = res.x
        pred = params[0] * est + params[1]
        residual = pred - gt
        plot_residuals(residual, error, gt)
        plot_error_rate(residual, error, gt)
        plot_comparison(gt, est, pred)

    elif method == "KBD" and focal is not None and baseline is not None:
        print(f"Starting optimization for KBD model with {desc} ...")
        res = model_kbd(gt, est, focal, baseline)
        params = res.x
        pred = params[0] * focal * baseline / (est + params[1]) + params[2]
        residual = pred - gt
        plot_residuals(residual, error, gt)
        plot_error_rate(residual, error, gt)
        plot_comparison(gt, focal * baseline / est, pred)


if __name__ == "__main__":
    # Load the data
    data_path = "data/dq_0507.csv"
    data = pd.read_csv(data_path)

    # Extracting necessary columns
    actual_depth = data["actual_depth"]
    fit_depth = data["fit_depth"]
    fit_disp = data["fit_disp"]
    error_rate = data["error_percentage"]
    error = data["absolute_error"]

    baseline = data["baseline"].iloc[0]  # Assuming constant for all data points
    focal = data["focal"].iloc[0]  # Assuming constant for all data points

    print("Starting optimization for KB model ...")
    kb_model_res = model_kb(actual_depth, fit_depth)
    kb_model_params = kb_model_res.x
    kb_save_path = "data/kb_res.jpg"
    kb_residuals = kb_model_params[0] * fit_depth + kb_model_params[1] - actual_depth
    data["kb_depth"] = kb_model_params[0] * fit_depth + kb_model_params[1]
    data["kb_error"] = kb_residuals
    kb_error_rate = kb_residuals / actual_depth
    plot_residuals(kb_error_rate, error)

    print("Starting optimization for KBD model ...")
    kbd_model_res = model_kbd(actual_depth, fit_disp, focal, baseline)
    kbd_model_params = kbd_model_res.x
    kbd_save_path = "data/kbd_res.jpg"
    kbd_residuals = (
        kbd_model_params[0]
        * focal
        * baseline
        / (normalize(fit_disp) + kbd_model_params[1])
        + kbd_model_params[2]
        - actual_depth
    )
    data["kbd_depth"] = (
        kbd_model_params[0]
        * focal
        * baseline
        / (normalize(fit_disp) + kbd_model_params[1])
        + kbd_model_params[2]
    )
    data["kbd_error"] = kbd_residuals
    kbd_error_rate = kbd_residuals / actual_depth
    plot_residuals(kbd_error_rate, error)
    # data.to_csv('data/output_0507.csv', index=False)
