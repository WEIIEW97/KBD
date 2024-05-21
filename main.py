from KBD.apis import (
    generate_parameters_linear,
    apply_transformation_linear_parallel,
    generate_parameters,
)
from KBD.plotters import plot_prediction_curve
from KBD.helpers import parallel_copy
from KBD.utils import generate_linear_KBD_data, generate_global_KBD_data
from KBD.constants import (
    CAMERA_TYPE,
    OUT_FIG_GLOBAL_PREDICTION_FILE_NAME,
    OUT_FIG_LINEAR_PREDICTION_FILE_NAME,
)
import os


if __name__ == "__main__":
    cwd = os.getcwd()
    rootdir = f"{cwd}/data/{CAMERA_TYPE}/image_data"
    copydir = f"{cwd}/data/{CAMERA_TYPE}/image_data_transformed_linear"
    table_path = f"{cwd}/data/{CAMERA_TYPE}/depthquality-2024-05-17.xlsx"
    params_save_path = f"{cwd}/data/{CAMERA_TYPE}"
    l2_regularization_param = (0.01,)
    disjoint_depth_range = (403, 600, 2700, 2907)
    pseudo_range = (100, 5000)

    k, delta, b, focal, baseline = generate_parameters(
        path=rootdir,
        tabel_path=table_path,
        save_path=params_save_path,
        use_l2=False,
    )

    # params_matrix, focal, baseline = generate_parameters_linear(
    #     path=rootdir,
    #     tabel_path=table_path,
    #     save_path=params_save_path,
    #     disjoint_depth_range=disjoint_depth_range,
    # )
    # print(params_matrix)

    # plot_prediction_curve(
    #     generate_linear_KBD_data,
    #     (focal, baseline, params_matrix, disjoint_depth_range),
    #     os.path.join(params_save_path, OUT_FIG_PREDICTION_FILE_NAME),
    #     pseudo_range[0],
    #     pseudo_range[1],
    # )

    plot_prediction_curve(
        generate_global_KBD_data,
        (focal, baseline, k, delta, b, pseudo_range[0], pseudo_range[1]),
        os.path.join(params_save_path, OUT_FIG_GLOBAL_PREDICTION_FILE_NAME),
    )

    # copy_all_subfolders(rootdir, copydir)
    # parallel_copy(rootdir, copydir)

    # # apply_transformation_parallel(copydir, k, delta, b, focal, baseline)
    # apply_transformation_linear_parallel(
    #     copydir, params_matrix, focal, baseline, disjoint_depth_range
    # )
