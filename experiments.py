from KBD.utils import load_raw, json_reader, json_to_numpy
from KBD.core import modify_linear_vectorize2
from KBD.constants import UINT16_MIN, UINT16_MAX
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt



if __name__ == "__main__":

    cwd = os.getcwd()
    export_path = os.path.join(cwd, "data/export")
    os.makedirs(export_path, exist_ok=True)
    before_path = os.path.join(export_path, 'before')
    after_path = os.path.join(export_path, 'after')
    dsp_path = os.path.join(export_path, 'dsp_ref')
    input_path = os.path.join(export_path, 'input')
    os.makedirs(before_path, exist_ok=True)
    os.makedirs(after_path, exist_ok=True)
    os.makedirs(input_path, exist_ok=True)


    json_path = "/home/william/Codes/KBD/data/N09ASH24DH0015/linear_KBD_model_fitted_params.json"
    data = json_reader(json_path)
    param_matrix = json_to_numpy(data)

    example_path1 = "/home/william/Codes/KBD/data/N09ASH24DH0015/image_data/1499_N09ASH24DH0015_2024_05_31_16_53_52/DEPTH/raw/Depth-2024-5-31-16-54-1-907-14-33794-2261583962.raw"
    example_path2 = "/home/william/Codes/KBD/data/N09ASH24DH0015/image_data/2099_N09ASH24DH0015_2024_05_31_17_06_02/DEPTH/raw/Depth-2024-5-31-17-6-11-467-16-44741-2991376665.raw"
    example_path3 = "/home/william/Codes/KBD/data/N09ASH24DH0015/image_data/3000_N09ASH24DH0015_2024_05_31_17_19_18/DEPTH/raw/Depth-2024-5-31-17-19-23-601-10-5720-385271576.raw"

    compensate_dist = 200
    scaling_factor = 10
    focal = 166.371384
    baseline = 55.693584

    disjoint_depth_range = [601, 3000]


    paths = [example_path1, example_path2, example_path3]

    for path in paths:
        filename = path.split('/')[-1]
        print(filename)

        #### for test data generation
        shutil.copy2(path, os.path.join(before_path, filename))
        m = load_raw(path, 480, 640).astype(np.float64)
        m = np.divide(focal*baseline, m, out=np.zeros_like(m), where=m!=0)
        m = (m * 64).astype(np.uint16)
        m.tofile(os.path.join(input_path, filename.replace("raw", "bin")))
        m = m.astype(np.float64) / 64
        mm = modify_linear_vectorize2(m, focal, baseline, param_matrix, disjoint_depth_range, compensate_dist, scaling_factor)
        mm1 = mm.astype(np.uint16)
        mm1.tofile(os.path.join(after_path, filename))


        #### for test data verification
        # alg_path = os.path.join(after_path, filename)
        # dspref_path = os.path.join(dsp_path, filename)

        # alg = load_raw(alg_path, 480, 640)
        # dsp = load_raw(dspref_path, 480, 640)

        # diff = alg - dsp

        # print(f"sum of absolute error is: {np.sum(np.abs(diff))}")
        # print(f"alg value at (240, 320) = {alg[240, 320]}")
        # print(f"dsp value at (240, 320) = {dsp[240, 320]}")
        


    