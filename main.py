from KBD.apis import (
    generate_parameters_linear,
    apply_transformation_linear_parallel,
    apply_transformation_linear_vectorize_parallel,
)
from KBD.helpers import parallel_copy, sampling_strategy_criterion
from KBD.utils import generate_linear_KBD_data, generate_global_KBD_data
from KBD.constants import (
    CAMERA_TYPE,
    OUT_FIG_GLOBAL_PREDICTION_FILE_NAME,
    OUT_FIG_LINEAR_PREDICTION_FILE_NAME,
)
from KBD.eval import eval
import os


if __name__ == "__main__":
    cwd = os.getcwd()
    # rootdir = f"{cwd}/data/{CAMERA_TYPE}/image_data"
    # copydir = f"{cwd}/data/{CAMERA_TYPE}/image_data_transformed_trf"
    # table_path = f"{cwd}/data/{CAMERA_TYPE}/depthquality-2024-05-22.xlsx"
    # params_save_path = f"{cwd}/data/{CAMERA_TYPE}"
    # l2_regularization_param = (0.01,)
    # disjoint_depth_range = [1000, 2900]
    # pseudo_range = (100, 5000)
    compensate_dist = 400
    scaling_factor = 10

    camera_types = [
        "N09ASH24DH0015",
        "N09ASH24DH0054",
        "N09ASH24DH0055",
        "N09ASH24DH0056",
        "N09ASH24DH0058",
    ]
    table_names = [
        "depthquality-2024-05-20.xlsx",
        "depthquality-2024-05-22.xlsx",
        "depthquality-2024-05-22_55.xlsx",
        "depthquality-2024-05-22_56.xlsx",
        "depthquality-2024-05-22.xlsx",
    ]
    disjoint_depth_ranges = [[601, 3000], [600, 2999], [1008, 2908], [600, 3000], [600, 2999]]
    for type, table_name, range in zip(
        camera_types, table_names, disjoint_depth_ranges
    ):
        print(f"processing {type} now with {table_name} ...")
        if type != "N09ASH24DH0015":
            continue
        root_dir = f"{cwd}/data/{type}/image_data"
        copy_dir = f"{cwd}/data/{type}/image_data_transformed_linear"
        copy_dir2 = f"{cwd}/data/{type}/image_data_transformed_linear2"
        save_dir = f"{cwd}/data/{type}"
        tablepath = f"{cwd}/data/{type}/{table_name}"
        # sampling_strategy_criterion(root_dir, tablepath, tablepath.replace("depthquality","sampling-criterion"))
        eval_res, acceptance_rate = eval(root_dir, tablepath)
        matrix, focal, baseline = generate_parameters_linear(
            path=root_dir,
            table_path=tablepath,
            save_path=save_dir,
            disjoint_depth_range=range,
            compensate_dist=compensate_dist,
            scaling_factor=scaling_factor,
        )
        # parallel_copy(root_dir, copy_dir)
        # parallel_copy(root_dir, copy_dir2)
        # apply_transformation_linear_parallel(copy_dir, matrix, focal, baseline, range, compensate_dist, scaling_factor)
        # apply_transformation_linear_vectorize_parallel(copy_dir2, matrix, focal, baseline, range, compensate_dist, scaling_factor)
    # methods = ["gaussian", "polynomial", "laplacian"]
    # for method in methods:
    #     ret = generate_parameters_adv(
    #         path=rootdir,
    #         tabel_path=table_path,
    #         save_path=params_save_path,
    #         method=method
    #     )

    # params_matrix, focal, baseline = generate_parameters_linear(
    #     path=rootdir,
    #     tabel_path=table_path,
    #     save_path=params_save_path,
    #     disjoint_depth_range=disjoint_depth_range,
    # )
    # print(params_matrix)

    # plot_prediction_curve(
    #     generate_linear_KBD_data,
    #     (
    #         focal,
    #         baseline,
    #         params_matrix,
    #         disjoint_depth_range,
    #         pseudo_range[0],
    #         pseudo_range[1],
    #     ),
    #     os.path.join(params_save_path, OUT_FIG_LINEAR_PREDICTION_FILE_NAME),
    # )

    # plot_prediction_curve(
    #     generate_global_KBD_data,
    #     (focal, baseline, k, delta, b, pseudo_range[0], pseudo_range[1]),
    #     os.path.join(params_save_path, OUT_FIG_GLOBAL_PREDICTION_FILE_NAME),
    # )

    # copy_all_subfolders(rootdir, copydir)
    # parallel_copy(rootdir, copydir)
    # apply_transformation_parallel(copydir, k, delta, b, focal, baseline)
    # apply_transformation_linear_parallel(
    #     copydir, params_matrix, focal, baseline, disjoint_depth_range
    # )
