import os

import numpy as np
from KBD.apis import (
    apply_transformation_linear_parallel,
    apply_transformation_linear_vectorize_parallel,
    generate_parameters,
    generate_parameters_linear,
    generate_parameters_linear_search,
    generate_parameters_trf,
)
from KBD.constants import (
    CAMERA_TYPE,
    OUT_FIG_GLOBAL_PREDICTION_FILE_NAME,
    OUT_FIG_LINEAR_PREDICTION_FILE_NAME,
)
from KBD.eval import check_monotonicity, eval
from KBD.helpers import parallel_copy, sampling_strategy_criterion
from KBD.plotters import plot_linear2
from KBD.utils import (
    generate_global_KBD_data,
    generate_linear_KBD_data,
    save_arrays_to_json,
    save_arrays_to_txt,
)

DISP_VAL_MAX_UINT16 = 32767


if __name__ == "__main__":
    cwd = os.getcwd()
    compensate_dist = 400
    scaling_factor = 10
    # apply_global = False

    # camera_types = [
    #     "N09ASH24DH0015",
    #     "N09ASH24DH0043",
    #     "N09ASH24DH0050",
    #     "N09ASH24DH0054",
    #     "N09ASH24DH0055",
    #     "N09ASH24DH0056",
    #     "N09ASH24DH0058",
    #     "test",
    # ]
    # table_names = [
    #     "depthquality-2024-05-20.xlsx",
    #     "depthquality_2024-07-08.xlsx",
    #     "depthquality_2024-07-09.xlsx",
    #     "depthquality-2024-05-22.xlsx",
    #     "depthquality-2024-05-22_55.xlsx",
    #     "depthquality-2024-05-22_56.xlsx",
    #     "depthquality-2024-05-22.xlsx",
    #     "depthquality_2024-07-19.xlsx",
    # ]
    # disjoint_depth_ranges = [
    #     [601, 3000],
    #     [600, 3000],
    #     [600, 3000],
    #     [600, 2999],
    #     [1008, 2908],
    #     [600, 3000],
    #     [600, 2999],
    #     [600, 3000],
    # ]
    root_dir = "D:/william/data/KBD/0723"
    camera_types = [f for f in os.listdir(root_dir)]
    disjoint_depth_ranges = [600, 3000]

    # for type, table_name, range_ in zip(
    #     camera_types, table_names, disjoint_depth_ranges
    # ):
    #     print(f"processing {type} now with {table_name} ...")
    #     if type != "test":
    #         continue
    for apply_global in [True, False]:
        if apply_global:
            continue
        for camera_type in camera_types:
            if camera_type != "N09ALC247H0046":
                continue
            base_path = os.path.join(root_dir, camera_type)
            file_path = os.path.join(base_path, "image_data")
            table_name = [f for f in os.listdir(base_path) if f.endswith('.xlsx') and os.path.isfile(os.path.join(base_path, f))][0]
            table_path = os.path.join(base_path, table_name)
            copy_path = os.path.join(base_path, "image_data_l")

            # save_params_path = save_dir + "/segmented_linear_KBD_params.json"
            # sampling_strategy_criterion(root_dir, tablepath, tablepath.replace("depthquality","sampling-criterion"))
            eval_res, acceptance_rate = eval(file_path, table_path)
            print(f"acceptance rate is {acceptance_rate}")
            matrix, focal, baseline = generate_parameters_linear(
                path=file_path,
                table_path=table_path,
                save_path=base_path,
                disjoint_depth_range=disjoint_depth_ranges,
                compensate_dist=compensate_dist,
                scaling_factor=scaling_factor,
                apply_global=apply_global,
                plot=True,
            )

            # matrix, focal, baseline = generate_parameters_linear_search(
            #     path=file_path,
            #     table_path=table_path,
            #     save_path=base_path,
            #     search_range=(600, 1100),
            #     compensate_dist=compensate_dist,
            #     scaling_factor=scaling_factor,
            #     apply_global=apply_global,
            #     plot=True,
            # )
        # k, delta, b, _, _ = generate_parameters(
        #     root_dir, tablepath, save_dir, plot=True
        # )
        # print(f"k, delta, b = {k}, {delta}, {b}")
        # k, delta, b, _, _ = generate_parameters_trf(
        #     root_dir, tablepath, save_dir, plot=True
        # )
        # print(f"k, delta, b = {k}, {delta}, {b}")

        # range_raw = range_
        # extra_range = [
        #     range_raw[0] - compensate_dist,
        #     range_raw[0],
        #     range_raw[1],
        #     range_raw[1] + compensate_dist * scaling_factor,
        # ]
        # disp_nodes_fp32 = focal * baseline / (np.array(extra_range))
        # disp_nodes_uint16 = (disp_nodes_fp32 * 64).astype(np.uint16)
        # disp_nodes_uint16 = np.sort(disp_nodes_uint16)
        # disp_nodes_uint16 = np.append(disp_nodes_uint16, DISP_VAL_MAX_UINT16)

        # matrix_param_by_disp = matrix[::-1, :]
        # # save_arrays_to_txt(save_params_path, disp_nodes_uint16, matrix_param_by_disp)
        # save_arrays_to_json(save_params_path, disp_nodes_uint16, matrix_param_by_disp)
        # judgement = check_monotonicity(0, 5000, focal, baseline, matrix, [601, 3000], compensate_dist, scaling_factor)
        # if judgement:
        #     print("Optimization task succeeded!")
        # else:
        #     raise ValueError("Optimization task failed. Cannot gurantee the monotonicity...")
