import os

import numpy as np
from KBD.apis import (
    apply_transformation_linear_parallel,
    apply_transformation_linear_vectorize_parallel,
    generate_parameters_linear,
    generate_parameters_linear_search,
)
from KBD.constants import *
from KBD.eval import check_monotonicity, eval, ratio_evaluate
from KBD.helpers import parallel_copy, preprocessing, sampling_strategy_criterion
from KBD.utils import (
    save_arrays_to_json,
    save_arrays_to_txt,
    save_arrays_to_txt2,
)


if __name__ == "__main__":
    cwd = os.getcwd()
    compensate_dist = 400
    scaling_factor = 10

    root_dir = "/home/william/extdisk/data/KBD_analysis"
    # camera_types = [
    #     f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))
    # ]
    disjoint_depth_ranges = [600, 3000]
    engine = "Nelder-Mead"
    apply_global = False
    genres = [
        f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))
    ]
    for genre in genres:
        # if apply_global:
        #     continue
        entry_dir = os.path.join(root_dir, genre)
        camera_types = [
            f for f in os.listdir(entry_dir) if os.path.isdir(os.path.join(entry_dir, f))
        ]
        for camera_type in camera_types:
            # if camera_type != "N09ALC247H0046":
            #     continue
            print(f"begin to process {camera_type}")
            base_path = os.path.join(entry_dir, camera_type)
            file_path = os.path.join(base_path, "image_data")
            table_name = [
                f
                for f in os.listdir(base_path)
                if f.endswith(".xlsx") and os.path.isfile(os.path.join(base_path, f))
            ][0]
            table_path = os.path.join(base_path, table_name)
            copy_path = os.path.join(base_path, "image_data_l")

            global_judge = "global" if apply_global else "local"
            optimizer_judge = (
                "nelder-mead" if engine == "Nelder-Mead" else "trust-region"
            )
            save_params_path = (
                base_path
                + f"/{optimizer_judge}"
                + f"/segmented_linear_KBD_params_{global_judge}_scale1.6.json"
            )
            save_txt_path = (
                base_path
                + f"/{optimizer_judge}"
                + f"/segmented_linear_KBD_eval_{global_judge}_scale1.6.txt"
            )
            # sampling_strategy_criterion(root_dir, tablepath, tablepath.replace("depthquality","sampling-criterion"))

            df, focal, baseline = preprocessing(file_path, table_path)
            trial = ratio_evaluate(0.5, df)
            if not trial:
                print("ratio bound evaluation test failed.")
                print("will not push to further process ...")
            eval_res, acceptance_rate = eval(df)
            print(f"acceptance rate is {acceptance_rate}")

            # matrix = generate_parameters_linear(
            #     df,
            #     focal,
            #     baseline,
            #     save_path=base_path,
            #     disjoint_depth_range=disjoint_depth_ranges,
            #     compensate_dist=compensate_dist,
            #     scaling_factor=scaling_factor,
            #     apply_global=apply_global,
            #     plot=True,
            # )

            matrix, best_range, best_z_err = generate_parameters_linear_search(
                df,
                focal,
                baseline,
                save_path=os.path.join(base_path, optimizer_judge),
                search_range=(600, 1100),
                engine=engine,
                compensate_dist=compensate_dist,
                scaling_factor=scaling_factor,
                apply_global=apply_global,
                plot=True,
            )

            range_raw = best_range
            extra_range = [
                range_raw[0] - compensate_dist,
                range_raw[0],
                range_raw[1],
                range_raw[1] + compensate_dist * scaling_factor,
            ]
            disp_nodes_fp32 = focal * baseline / (np.array(extra_range))
            disp_nodes_uint16 = (disp_nodes_fp32 * 64).astype(np.uint16)
            disp_nodes_uint16 = np.sort(disp_nodes_uint16)
            disp_nodes_uint16 = np.append(disp_nodes_uint16, DISP_VAL_MAX_UINT16)

            matrix_param_by_disp = matrix[::-1, :]
            # save_arrays_to_txt(save_params_path, disp_nodes_uint16, matrix_param_by_disp)

            save_arrays_to_json(
                save_params_path, disp_nodes_uint16, matrix_param_by_disp
            )
            save_arrays_to_txt2(save_txt_path, best_range, best_z_err)
        # judgement = check_monotonicity(0, 5000, focal, baseline, matrix, [601, 3000], compensate_dist, scaling_factor)
        # if judgement:
        #     print("Optimization task succeeded!")
        # else:
        #     raise ValueError("Optimization task failed. Cannot gurantee the monotonicity...")
