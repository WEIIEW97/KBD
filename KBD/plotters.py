import os

import matplotlib.pyplot as plt
import numpy as np

from .constants import (
    OUT_FIG_COMP_FILE_NAME,
    OUT_FIG_ERROR_RATE_FILE_NAME,
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
    # # plt.show()

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
    # # plt.show()


def plot_comparison(x, y1, y2, save_path=None):
    fig, ax = plt.subplots()
    ax.plot(x, y1, label="measured data", marker="o")
    ax.plot(x, y2, label="fitted data", marker="x")

    ax.legend()
    if save_path:
        plt.savefig(save_path)
    # # plt.show()

def plot_linear2(
    gt,
    est,
    error,
    focal,
    baseline,
    res,
    disjoint_depth_range,
    compensate_dist,
    scaling_factor,
    apply_global=False,
    save_path=None,
):  
    linear_model, optimization_result, linear_model2 = res
    fb = focal * baseline
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    # Filter data where actual_depth >= 600
    mask0 = np.where(gt < disjoint_depth_range[0] - compensate_dist)
    mask1 = np.where(
        (gt >= (disjoint_depth_range[0] - compensate_dist))
        & (gt <= disjoint_depth_range[0])
    )
    mask2 = (gt > disjoint_depth_range[0]) & (gt <= disjoint_depth_range[1])
    mask3 = (gt > disjoint_depth_range[1]) & (
        gt <= disjoint_depth_range[1] + compensate_dist * scaling_factor
    )
    mask4 = gt > (disjoint_depth_range[1] + compensate_dist * scaling_factor)

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
    optimized_params = optimization_result
    pred2 = (
        optimized_params[0] * focal * baseline / (filtered_disp2 + optimized_params[1])
        + optimized_params[2]
    )
    residual2 = pred2 - filtered_depth2

    plot_fig_3 = False
    if np.sum(mask3) > 0:
        plot_fig_3 = True
        filtered_disp3 = est[mask3]
        filtered_depth3 = gt[mask3]
        error3 = error[mask3]
        pred_3_disp = linear_model2.predict(filtered_disp3.reshape(-1, 1))
        pred3 = focal * baseline / pred_3_disp
        residual3 = pred3 - filtered_depth3

    plot_fig_4 = False
    if np.sum(mask4) > 0:
        plot_fig_4 = True
        filtered_disp4 = est[mask4]
        filtered_depth4 = gt[mask4]
        error4 = error[mask4]
        pred4 = gt[mask4]
        residual4 = pred4 - filtered_depth4

    if save_path is not None and not apply_global:
        LINEAR_COMMON = "linear_local_"
        comp_path = os.path.join(save_path, LINEAR_COMMON + OUT_FIG_COMP_FILE_NAME)
        residual_path = os.path.join(
            save_path, LINEAR_COMMON + OUT_FIG_RESIDUAL_FILE_NAME
        )
        error_rate_path = os.path.join(
            save_path, LINEAR_COMMON + OUT_FIG_ERROR_RATE_FILE_NAME
        )
    elif save_path is not None and apply_global:
        LINEAR_COMMON = "linear_global_"
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
    ax1.plot(
        filtered_depth0,
        residual0,
        color="cyan",
        alpha=0.5,
    )
    ax1.scatter(
        filtered_depth1,
        residual1,
        color="blue",
        alpha=0.5,
        label="Linear Model 1 Residuals",
    )
    ax1.plot(
        filtered_depth1,
        residual1,
        color="blue",
        alpha=0.5,
    )
    ax1.scatter(
        filtered_depth1, error1, color="black", alpha=0.5, label="Actual Residuals"
    )
    ax1.plot(
        filtered_depth1, error1, color="black", alpha=0.5,
    )
    ax1.scatter(
        filtered_depth2,
        residual2,
        color="green",
        alpha=0.5,
        label="Optimized Model Residuals",
    )
    ax1.plot(
        filtered_depth2,
        residual2,
        color="green",
        alpha=0.5,
    )
    ax1.scatter(
        filtered_depth2, error2, color="black", alpha=0.5, label="Actual Residuals"
    )
    ax1.plot(
        filtered_depth2, error2, color="black", alpha=0.5,
    )

    if plot_fig_3:
        ax1.scatter(
            filtered_depth3,
            residual3,
            color="red",
            alpha=0.5,
            label="Linear Model 2 Residuals",
        )
        ax1.plot(
           filtered_depth3,
            residual3,
            color="red",
            alpha=0.5, 
        )
        ax1.scatter(
            filtered_depth3, error3, color="black", alpha=0.5, label="Actual Residuals"
        )
        ax1.plot(
            filtered_depth3, error3, color="black", alpha=0.5
        )
        ax1.hlines(
            0,
            xmin=0,
            xmax=np.max(filtered_depth4) if plot_fig_4 else np.max(filtered_depth3),
            colors="red",
            linestyles="dashed",
        )
    ax1.hlines(
        0,
        xmin=0,
        xmax=np.max(filtered_depth2),
        colors="red",
        linestyles="dashed",
    )
    ax1.set_title("Residuals Plot")
    ax1.set_xlabel("Ground Truth Depth (mm)")
    ax1.set_ylabel("Residuals")
    ax1.legend()
    ax1.set_xlim(0, 3000)
    depth_ticks = ax1.get_xticks()
    disparity_ticks = np.divide(fb, depth_ticks, out=np.zeros_like(depth_ticks), where=depth_ticks != 0)

    ax1.set_xticks(depth_ticks)
    ax1.set_xticklabels([f'{d:.3f}\n({dp:.3f})' for d, dp in zip(depth_ticks, disparity_ticks)])

    if save_path is not None:
        plt.savefig(residual_path)
    # plt.show()

    # Error rate plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(
        filtered_depth0,
        residual0 / filtered_depth0 * 100,
        color="pink",
        alpha=0.5,
        label="Unchanged Error Rate",
    )
    ax2.plot(
       filtered_depth0,
        residual0 / filtered_depth0 * 100,
        color="pink",
        alpha=0.5, 
    )
    ax2.scatter(
        filtered_depth0,
        error0 / filtered_depth0 * 100,
        color="black",
        alpha=0.5,
        label="Actual Error Rate",
    )
    ax2.plot(
        filtered_depth0,
        error0 / filtered_depth0 * 100,
        color="black",
        alpha=0.5,
    )
    ax2.scatter(
        filtered_depth1,
        residual1 / filtered_depth1 * 100,
        color="blue",
        alpha=0.5,
        label="Linear Model 1 Error Rate",
    )
    ax2.plot(
      filtered_depth1,
        residual1 / filtered_depth1 * 100,
        color="blue",
        alpha=0.5,  
    )
    ax2.scatter(
        filtered_depth1,
        error1 / filtered_depth1 * 100,
        color="black",
        alpha=0.5,
        label="Actual Error Rate",
    )
    ax2.plot(
        filtered_depth1,
        error1 / filtered_depth1 * 100,
        color="black",
        alpha=0.5,
    )
    ax2.scatter(
        filtered_depth2,
        residual2 / filtered_depth2 * 100,
        color="green",
        alpha=0.5,
        label="Optimized Model Error Rate",
    )
    ax2.plot(
        filtered_depth2,
        residual2 / filtered_depth2 * 100,
        color="green",
        alpha=0.5,
    )
    ax2.scatter(
        filtered_depth2,
        error2 / filtered_depth2 * 100,
        color="black",
        alpha=0.5,
        label="Actual Error Rate",
    )
    ax2.plot(
        filtered_depth2,
        error2 / filtered_depth2 * 100,
        color="black",
        alpha=0.5,
    )
    if plot_fig_3:
        ax2.scatter(
            filtered_depth3,
            residual3 / filtered_depth3 * 100,
            color="gray",
            alpha=0.5,
            label="Linear Model 2 Error Rate",
        )
        ax2.plot(
            filtered_depth3,
            residual3 / filtered_depth3 * 100,
            color="gray",
            alpha=0.5,
        )
        ax2.scatter(
            filtered_depth3,
            error3 / filtered_depth3 * 100,
            color="black",
            alpha=0.5,
            label="Actual Error Rate",
        )
        ax2.plot(
           filtered_depth3,
            error3 / filtered_depth3 * 100,
            color="black",
            alpha=0.5, 
        )
        ax2.hlines(
            0,
            xmin=0,
            xmax=np.max(filtered_depth4) if plot_fig_4 else np.max(filtered_depth3),
            colors="red",
            linestyles="dashed",
        )
        ax2.hlines(
            2,
            xmin=0,
            xmax=np.max(filtered_depth4) if plot_fig_4 else np.max(filtered_depth3),
            colors="pink",
            linestyles="dashed",
            label="2(%) error Line",
        )
        ax2.hlines(
            -2,
            xmin=0,
            xmax=np.max(filtered_depth4) if plot_fig_4 else np.max(filtered_depth3),
            colors="pink",
            linestyles="dashed",
            label="2(%) error Line",
        )
        ax2.hlines(
            4,
            xmin=0,
            xmax=np.max(filtered_depth4) if plot_fig_4 else np.max(filtered_depth3),
            colors="cyan",
            linestyles="dashed",
            label="4(%) error Line",
        )
        ax2.hlines(
            -4,
            xmin=0,
            xmax=np.max(filtered_depth4) if plot_fig_4 else np.max(filtered_depth3),
            colors="cyan",
            linestyles="dashed",
            label="4(%) error Line",
        )
    if plot_fig_4:
        ax2.scatter(
            filtered_depth4,
            residual4 / filtered_depth4 * 100,
            color="cyan",
            alpha=0.5,
            label="Unchanged Error Rate",
        )
        ax2.plot(
            filtered_depth4,
            residual4 / filtered_depth4 * 100,
            color="cyan",
            alpha=0.5,
        )
        ax2.scatter(
            filtered_depth4,
            error4 / filtered_depth4 * 100,
            color="black",
            alpha=0.5,
            label="Actual Error Rate",
        )
        ax2.plot(
            filtered_depth4,
            error4 / filtered_depth4 * 100,
            color="black",
            alpha=0.5,
        )
    ax2.hlines(
        0,
        xmin=0,
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
    ax2.set_xlabel("Ground Truth Depth (mm)")
    ax2.set_ylabel("Error Rate (%)")
    ax2.legend(fontsize=5)
    ax2.set_xlim(0, 3000)
    depth_ticks2 = ax2.get_xticks()
    disparity_ticks2 = np.divide(fb, depth_ticks2, out=np.zeros_like(depth_ticks2), where=depth_ticks2 != 0)

    ax2.set_xticks(depth_ticks2)
    ax2.set_xticklabels([f'{d:.3f}\n({dp:.3f})'  for d, dp in zip(depth_ticks2, disparity_ticks2)])
    if save_path is not None:
        plt.savefig(error_rate_path)
    # plt.show()

    # Depth comparison plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(
        gt, focal * baseline / est, label="Measured Data", marker="o", color="black"
    )
    ax3.plot(
        gt[mask0],
        pred0,
        label=f"Measured Data (< {disjoint_depth_range[0]-compensate_dist})",
        marker="o",
        color="red",
    )
    ax3.plot(
        filtered_depth1,
        pred1,
        label=f"Linear Model ({disjoint_depth_range[0]-compensate_dist}-{disjoint_depth_range[0]})",
        marker="x",
        color="blue",
    )
    ax3.plot(
        filtered_depth2,
        pred2,
        label=f"Optimized Model ({disjoint_depth_range[0]}-{disjoint_depth_range[1]})",
        marker="x",
        color="green",
    )
    if plot_fig_3:
        ax3.plot(
            filtered_depth3,
            pred3,
            label=f"Linear Model ({disjoint_depth_range[1]}-{disjoint_depth_range[1]+compensate_dist})",
            marker="x",
            color="cyan",
        )
    if plot_fig_4:
        ax3.plot(
            gt[mask4],
            pred4,
            label=f"Measured Data (>{disjoint_depth_range[1]+compensate_dist})",
            marker="o",
            color="red",
        )

    ax3.set_xlabel("Ground Truth Depth (mm)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_title("Comparison of Measured and Fitted Depths")
    ax3.set_xlim(0, 3000)
    depth_ticks3 = ax3.get_xticks()
    disparity_ticks3 = np.divide(fb, depth_ticks3, out=np.zeros_like(depth_ticks3), where=depth_ticks3 != 0)

    ax3.set_xticks(depth_ticks3)
    ax3.set_xticklabels([f'{d:.3f}\n({dp:.3f})'  for d, dp in zip(depth_ticks3, disparity_ticks3)])
    ax3.legend()
    if save_path is not None:
        plt.savefig(comp_path)
    # plt.show()