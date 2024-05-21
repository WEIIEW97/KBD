import os
import numpy as np
import matplotlib.pyplot as plt

from .constants import (
    OUT_FIG_ERROR_RATE_FILE_NAME,
    OUT_FIG_COMP_FILE_NAME,
    OUT_FIG_RESIDUAL_FILE_NAME,
)


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
    linear_model, optimization_result = res

    # Filter data where actual_depth >= 600
    mask0 = np.where(gt < disjoint_depth_range[0])
    mask1 = np.where((gt >= disjoint_depth_range[0]) & (gt <= disjoint_depth_range[1]))
    mask2 = np.where(gt > disjoint_depth_range[1])

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
        label="Linear Model Residuals",
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
        label="Linear Model Error Rate",
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
        xmax=np.max(filtered_depth2),
        colors="pink",
        linestyles="dashed",
        label="2(%) error Line",
    )
    ax2.hlines(
        -2,
        xmin=0,
        xmax=np.max(filtered_depth2),
        colors="pink",
        linestyles="dashed",
        label="2(%) error Line",
    )
    ax2.hlines(
        4,
        xmin=0,
        xmax=np.max(filtered_depth2),
        colors="cyan",
        linestyles="dashed",
        label="4(%) error Line",
    )
    ax2.hlines(
        -4,
        xmin=0,
        xmax=np.max(filtered_depth2),
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
        gt,
        focal * baseline / est,
        label="Measured Data",
        marker="o",
        linestyle="None",
        color="black",
    )
    ax3.plot(
        gt[mask0],
        pred0,
        label="Measured Data (< 500)",
        marker="o",
        linestyle="None",
        color="red",
    )
    ax3.plot(
        filtered_depth1,
        pred1,
        label="Linear Model (500-600)",
        marker="x",
        linestyle="None",
        color="blue",
    )
    ax3.plot(
        filtered_depth2,
        pred2,
        label="Optimized Model (> 600)",
        marker="x",
        linestyle="None",
        color="green",
    )
    ax3.set_xlabel("Ground Truth Depth (m)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_title("Comparison of Measured and Fitted Depths")
    ax3.legend()
    plt.show()


def plot_all_in_one(gt, est, focal, baseline, depth_ranges, res):
    # Create a figure with subplots
    num_segments = len(depth_ranges)
    fig, axes = plt.subplots(
        num_segments, 3, figsize=(18, 4 * num_segments)
    )  # 3 plots per segment

    for idx, ((start, end), ax_row) in enumerate(zip(depth_ranges, axes)):
        # Select data for the current segment
        indices = np.where((gt >= start) & (gt <= end))[0]
        segment_disp = est[indices]
        segment_depth = gt[indices]

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
        error = segment_depth - pred  # This is a placeholder, adjust based on your data

        # Plot residuals
        ax_row[0].scatter(segment_depth, residual, color="blue", label="Residuals")
        ax_row[0].hlines(
            0,
            xmin=np.min(segment_depth),
            xmax=np.max(segment_depth),
            colors="red",
            linestyles="dashed",
        )
        ax_row[0].set_title(f"Residuals for {start}-{end}m")
        ax_row[0].set_xlabel("Ground Truth Depth (m)")
        ax_row[0].set_ylabel("Residual")

        # Plot error rate
        ax_row[1].scatter(
            segment_depth,
            residual / segment_depth * 100,
            color="green",
            label="Error Rate (%)",
        )
        ax_row[1].hlines(
            0,
            xmin=np.min(segment_depth),
            xmax=np.max(segment_depth),
            colors="red",
            linestyles="dashed",
        )
        ax_row[1].set_title(f"Error Rate for {start}-{end}m")
        ax_row[1].set_xlabel("Ground Truth Depth (m)")
        ax_row[1].set_ylabel("Error Rate (%)")

        # Plot comparison
        ax_row[2].plot(segment_depth, segment_depth, "k--", label="Actual Depth")
        ax_row[2].scatter(
            segment_depth, pred, color="red", label="Predicted Depth", marker="x"
        )
        ax_row[2].set_title(f"Comparison for {start}-{end}m")
        ax_row[2].set_xlabel("Ground Truth Depth (m)")
        ax_row[2].set_ylabel("Depth")

        # Set legends
        for ax in ax_row:
            ax.legend()

    plt.tight_layout()
    plt.show()


def plot_linear(
    gt, est, error, focal, baseline, res, disjoint_depth_range, save_path=None
):
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

    if save_path is not None:
        LINEAR_COMMON = "linear_"
        comp_path = os.path.join(save_path, LINEAR_COMMON + OUT_FIG_COMP_FILE_NAME)
        residual_path = os.path.join(
            save_path, LINEAR_COMMON + OUT_FIG_RESIDUAL_FILE_NAME
        )
        error_rate_path = os.path.join(
            save_path, LINEAR_COMMON + OUT_FIG_ERROR_RATE_FILE_NAME
        )

    # Residuals plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(
        filtered_depth0,
        residual0,
        color="cyan",
        alpha=0.5,
        label="Actual Residuals",
    )
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
        xmin=0,
        xmax=np.max(filtered_depth4),
        colors="red",
        linestyles="dashed",
    )
    ax1.set_title("Residuals Plot")
    ax1.set_xlabel("Ground Truth Depth (m)")
    ax1.set_ylabel("Residuals")
    ax1.legend()
    if save_path is not None:
        plt.savefig(residual_path)
    plt.show()

    # Error rate plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
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
        xmin=0,
        xmax=np.max(filtered_depth4),
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
    ax2.legend(fontsize=5)
    if save_path is not None:
        plt.savefig(error_rate_path)
    plt.show()

    # Depth comparison plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(
        gt, focal * baseline / est, label="Measured Data", marker="o", color="black"
    )
    ax3.plot(
        gt[mask0],
        pred0,
        label=f"Measured Data (< {disjoint_depth_range[0]})",
        marker="o",
        color="red",
    )
    ax3.plot(
        filtered_depth1,
        pred1,
        label=f"Linear Model ({disjoint_depth_range[0]}-{disjoint_depth_range[1]})",
        marker="x",
        color="blue",
    )
    ax3.plot(
        filtered_depth2,
        pred2,
        label=f"Optimized Model (> {disjoint_depth_range[1]})",
        marker="x",
        color="green",
    )
    ax3.plot(
        filtered_depth3,
        pred3,
        label=f"Linear Model ({disjoint_depth_range[2]}-{disjoint_depth_range[3]})",
        marker="x",
        color="cyan",
    )
    ax3.plot(
        gt[mask4],
        pred4,
        label=f"Measured Data (>{disjoint_depth_range[3]})",
        marker="o",
        color="red",
    )

    ax3.set_xlabel("Ground Truth Depth (m)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_title("Comparison of Measured and Fitted Depths")
    ax3.legend()
    if save_path is not None:
        plt.savefig(comp_path)
    plt.show()


def plot_prediction_curve(
    func,
    func_args,
    save_path=None,
):
    x_values, y_values = func(*func_args)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="Predicted Depth", color="blue")
    plt.xlabel("Distance (mm)")
    plt.ylabel("Predicted Value")
    plt.title("Prediction Curve")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()