import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from KBD.constants import UINT16_MAX, UINT16_MIN
from KBD.core import modify_linear_vectorize2
from KBD.models import linear_KBD_piecewise_func
from KBD.utils import json_reader, json_to_numpy, load_raw


def create_lut(
    max_disparity,
    scale,
    focal,
    baseline,
    param_matrix,
    disjoint_depth_range,
    compensate_dist,
    scaling_factor,
):
    fb = focal * baseline
    lb = disjoint_depth_range[0]
    ub = disjoint_depth_range[1]
    thr = compensate_dist
    sf = scaling_factor

    lut = np.zeros(max_disparity, dtype=np.float64)

    print(fb / (lb - thr))
    print(fb / lb)
    print(fb / ub)
    print(fb / (ub + thr * sf))

    for d in range(1, max_disparity):
        d_scale = d / scale
        if d_scale > fb / (lb - thr):
            lut[d] = fb / d_scale
        elif fb / (lb - thr) >= d_scale > fb / lb:
            lut[d] = fb / (param_matrix[1, 3] * d_scale + param_matrix[1, 4])
        elif fb / lb >= d_scale > fb / ub:
            lut[d] = (
                param_matrix[2, 0] * (fb / (d_scale + param_matrix[2, 1]))
                + param_matrix[2, 2]
            )
        elif fb / ub >= d_scale > fb / (ub + thr * sf):
            lut[d] = fb / (param_matrix[3, 3] * d_scale + param_matrix[3, 4])
        else:
            lut[d] = fb / d_scale

    lut[lut > 65535] = 65535
    return lut.astype(np.uint16)


def plot_prediction(
    minv,
    maxv,
    focal,
    baseline,
    param_matrix,
    disjoint_depth_range,
    compensate_dist,
    scaling_factor,
):
    x_values = np.arange(minv, maxv, step=1)
    y_values = [
        linear_KBD_piecewise_func(
            x,
            focal,
            baseline,
            param_matrix,
            disjoint_depth_range,
            compensate_dist,
            scaling_factor,
        )
        for x in x_values
    ]
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="Predicted depth", color="blue")
    plt.xlabel("Distance (mm)")
    plt.ylabel("Predicted Value")
    plt.title("Prediction Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    cwd = os.getcwd()
    export_path = os.path.join(cwd, "data/export")
    os.makedirs(export_path, exist_ok=True)
    before_path = os.path.join(export_path, "before")
    after_path = os.path.join(export_path, "after")
    dsp_path = os.path.join(export_path, "dsp_ref")
    input_path = os.path.join(export_path, "input")
    os.makedirs(before_path, exist_ok=True)
    os.makedirs(after_path, exist_ok=True)
    os.makedirs(input_path, exist_ok=True)

    json_path = cwd + "/data/N09ASH24DH0015/linear_KBD_model_fitted_params.json"
    data = json_reader(json_path)
    param_matrix = json_to_numpy(data)

    lst = [1, 0, 0, 1, 0]
    arr = np.array(lst)
    param_matrix2 = np.tile(arr, (5, 1))
    print(param_matrix2)

    example_path1 = (
        cwd
        + "/data/N09ASH24DH0015/image_data/1499_N09ASH24DH0015_2024_05_31_16_53_52/DEPTH/raw/Depth-2024-5-31-16-54-1-907-14-33794-2261583962.raw"
    )
    example_path2 = (
        cwd
        + "/data/N09ASH24DH0015/image_data/2099_N09ASH24DH0015_2024_05_31_17_06_02/DEPTH/raw/Depth-2024-5-31-17-6-11-467-16-44741-2991376665.raw"
    )
    example_path3 = (
        cwd
        + "/data/N09ASH24DH0015/image_data/3000_N09ASH24DH0015_2024_05_31_17_19_18/DEPTH/raw/Depth-2024-5-31-17-19-23-601-10-5720-385271576.raw"
    )

    compensate_dist = 400
    compensate_dist2 = 0
    scaling_factor = 10
    scaling_factor2 = 1
    focal = 166.371384
    baseline = 55.693584

    disjoint_depth_range = [601, 3000]
    disjoint_depth_range2 = [int(focal * baseline / (32768 / 64)), int(1e7)]

    x1 = 599
    y1 = linear_KBD_piecewise_func(
        x1,
        focal,
        baseline,
        param_matrix,
        disjoint_depth_range,
        compensate_dist,
        scaling_factor,
    )
    x3 = 600
    y3 = linear_KBD_piecewise_func(
        x3,
        focal,
        baseline,
        param_matrix,
        disjoint_depth_range,
        compensate_dist,
        scaling_factor,
    )
    x2 = 601
    y2 = linear_KBD_piecewise_func(
        x2,
        focal,
        baseline,
        param_matrix,
        disjoint_depth_range,
        compensate_dist,
        scaling_factor,
    )

    print(f"x1, y1 = ({x1, y1})")
    print(f"x3, y3 = ({x3, y3})")
    print(f"x2, y2 = ({x2, y2})")

    plot_prediction(
        0,
        5000,
        focal,
        baseline,
        param_matrix,
        disjoint_depth_range,
        compensate_dist,
        scaling_factor,
    )

    # paths = [example_path1, example_path2, example_path3]

    # for path in paths:
    #     filename = path.split('/')[-1]
    #     print(filename)

    #     #### for test data generation
    #     shutil.copy2(path, os.path.join(before_path, filename))
    #     m = load_raw(path, 480, 640).astype(np.float64)
    #     md = np.divide(focal*baseline, m, out=np.zeros_like(m), where=m!=0)
    #     # m = (m * 64).astype(np.uint16)
    #     # m.tofile(os.path.join(input_path, filename.replace("raw", "bin")))
    #     # m = m.astype(np.float64) / 64
    #     mm = modify_linear_vectorize2(md, focal, baseline, param_matrix2, disjoint_depth_range, compensate_dist, scaling_factor)
    #     # mm1 = mm.astype(np.uint16)
    #     # mm1.tofile(os.path.join(after_path, filename))
    #     print(f"default model sum of absolute error is {np.sum(np.abs(m - mm))}")

    #### for test data verification
    # alg_path = os.path.join(after_path, filename)
    # dspref_path = os.path.join(dsp_path, filename)

    # alg = load_raw(alg_path, 480, 640)
    # dsp = load_raw(dspref_path, 480, 640)

    # diff = alg - dsp

    # print(f"sum of absolute error is: {np.sum(np.abs(diff))}")
    # print(f"alg value at (240, 320) = {alg[240, 320]}")
    # print(f"dsp value at (240, 320) = {dsp[240, 320]}")
