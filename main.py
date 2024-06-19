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
from KBD.eval import eval, check_monotonicity
import os


if __name__ == "__main__":
    cwd = os.getcwd()
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
        judgement = check_monotonicity(0, 5000, focal, baseline, matrix, [601, 3000], compensate_dist, scaling_factor)
        if judgement:
            print("Optimization task succeeded!")
        else:
            raise ValueError("Optimization task failed. Cannot gurantee the monotonicity...")
