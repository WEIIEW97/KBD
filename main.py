import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from KBD.apis import (
    apply_transformation_linear,
    generate_parameters_linear,
    generate_parameters_linear_search,
    GridSearch2D
)
from KBD.constants import *
from KBD.eval import check_monotonicity, eval, ratio_evaluate, first_check, pass_or_not
from KBD.helpers import parallel_copy, preprocessing, sampling_strategy_criterion
from KBD.utils import (
    save_arrays_to_json,
    save_arrays_to_txt,
    save_arrays_to_txt2,
    export_default_settings,
)
from KBD.core import modify_linear_vectorize2


if __name__ == "__main__":
    cwd = os.getcwd()
    compensate_dist = 400
    scaling_factor = 10

    root_dir = "/home/william/extdisk/data/KBD_ACCURACY"
    # camera_types = [
    #     f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))
    # ]
    disjoint_depth_ranges = [600, 3000]
    engine = "Nelder-Mead"
    sample_weights_factor = 3.0
    export_original = False
    # apply_global = False
    # genres = [
    #     f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))
    # ]
    # for genre in genres:
    failed_devices = []
    for apply_global in [True, False]:
        if apply_global:
            continue
        # entry_dir = os.path.join(root_dir, genre)
        camera_types = [
            f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))
        ]
        N = len(camera_types)
        p = 0
        for camera_type in camera_types:
            # if camera_type == "N9LAZG24GN0294":
            #     continue
            print(f"begin to process {camera_type}")
            base_path = os.path.join(root_dir, camera_type)
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
            optimizer_judge = optimizer_judge + "-grid-restrict"
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

            df, focal, baseline = preprocessing(file_path, table_path)
            if not first_check(df) or not pass_or_not(df):
                # trial = ratio_evaluate(bound_ratio_alpha, df)
                # if not trial:
                #     print("Ratio bound evaluation test failed.")
                #     print("Will not push to further process ...")
                #     raise ValueError("Ratio-Bound test failed!")

                eval_res, acceptance_rate = eval(df)

                if acceptance_rate < EVAL_WARNING_RATE:
                    print("*********** WARNING *************")
                    print(
                        f"Please be really cautious since the acceptance rate is {acceptance_rate},"
                    )
                    print("This may not be the ideal data to be tackled with.")
                    print("*********** END OF WARNING *************")
                print("Begin to generate parameters with line searching...")

                # matrix, best_range, best_z_err = generate_parameters_linear_search(
                #     df,
                #     focal,
                #     baseline,
                #     save_path=os.path.join(base_path, optimizer_judge),
                #     search_range=(600, 1100),
                #     engine=engine,
                #     compensate_dist=compensate_dist,
                #     scaling_factor=scaling_factor,
                #     apply_global=apply_global,
                #     plot=True,
                # )

                # matrix = generate_parameters_linear(
                #     df,
                #     focal,
                #     baseline,
                #     os.path.join(base_path, optimizer_judge),
                #     disjoint_depth_range=(1100, 3000),
                #     compensate_dist=400,
                #     scaling_factor=10,
                #     plot=True
                # )

                search_range = (600, 1100)
                cd_range = (10, 400)
                GridSearchOptimizer = GridSearch2D(df, focal, baseline, scaling_factor,save_path=os.path.join(base_path, optimizer_judge), engine=engine, apply_global=apply_global, is_plot=True)
                GridSearchOptimizer.optimize_parameters(search_range, cd_range)
                matrix, best_range_start, best_cd = GridSearchOptimizer.get_results()
                best_range = (best_range_start, 3000)
                # best_range = (1100, 3000)
                # best_cd = 400
                # add before/after kbd comparsion
                X = df[AVG_DISP_NAME]
                Y = df[GT_DIST_NAME]
                pred = modify_linear_vectorize2(
                    X.values,
                    focal,
                    baseline,
                    matrix,
                    best_range,
                    best_cd,
                    scaling_factor,
                )
                df[KBD_PRED_NAME] = pred
                df[GT_DIST_ERROR_NAME] = np.abs(df[GT_ERROR_NAME]/Y)
                df[KBD_ERROR_NAME] = np.abs((Y-pred)/Y)
                df_criteria = df[df[GT_DIST_NAME].isin(np.array(TARGET_POINTS))]
                weights = Y.apply(lambda x: sample_weights_factor if x in TARGET_POINTS else 1.0)
                previous_mse = mean_squared_error(Y, X, sample_weight=weights)
                after_mse = mean_squared_error(Y, df[KBD_PRED_NAME], sample_weight=weights)
                ##################################
                if after_mse < previous_mse:
                    range_raw = best_range
                    extra_range = [
                        range_raw[0] - best_cd,
                        range_raw[0],
                        range_raw[1],
                        range_raw[1] + best_cd * scaling_factor,
                    ]
                    disp_nodes_fp32 = focal * baseline / (np.array(extra_range))
                    disp_nodes_uint16 = (disp_nodes_fp32 * 64).astype(np.uint16)
                    disp_nodes_uint16 = np.sort(disp_nodes_uint16)
                    disp_nodes_uint16 = np.append(disp_nodes_uint16, DISP_VAL_MAX_UINT16)

                    matrix_param_by_disp = matrix[::-1, :]
                    save_arrays_to_json(
                        save_params_path, disp_nodes_uint16, matrix_param_by_disp
                    )
                    cond = (df_criteria[KBD_ERROR_NAME] < TARGET_THRESHOLDS)
                    if cond.all():
                        p += 1
                    else:
                        failed_devices.append(camera_type)
                else:
                    best_range, matrix = export_default_settings(
                        save_params_path, focal, baseline, best_cd, scaling_factor
                    )
                    export_original = True
                    cond = (df_criteria[GT_DIST_ERROR_NAME] < TARGET_THRESHOLDS)
                    if cond.all():
                        p += 1
                    else:
                        failed_devices.append(camera_type)
            else:
                best_range, matrix = export_default_settings(
                    save_params_path, focal, baseline, best_cd, scaling_factor
                )
                export_original = True
                p += 1
            print("===> Working done! Parameters are being generated.")

            # print("Beging to copy and transform depth raws ...")
            # parallel_copy(file_path, copy_path)
            # if not export_original:
            #     apply_transformation_linear(
            #         copy_path,
            #         matrix,
            #         focal,
            #         baseline,
            #         best_range,
            #         compensate_dist,
            #         scaling_factor,
            #     )
            # print("===> Working done! Transformed raw depths are being generated.")
        
        print(f"The passing rate is {p/N} ...")
        failed_df = pd.DataFrame()
        failed_df["devices"] = failed_devices
        failed_df.to_csv(os.path.join(root_dir, "failed_devices_grid_restrict.csv"))