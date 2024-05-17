import numpy as np
import pandas as pd
import os
import shutil

from .constants import (
    MAPPED_PAIR_DICT,
    SUBFIX,
    ANCHOR_POINT,
    H,
    W,
    AVG_DIST_NAME,
    AVG_DISP_NAME,
    GT_DIST_NAME,
    FOCAL_NAME,
    BASLINE_NAME,
)
from .utils import read_table, load_raw

from typing import Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def helper_save_data_to_csv(path: str, table_path: str, save_path: str):
    all_distances = retrive_folder_names(path)
    mean_dists = calculate_mean_value(path, all_distances)
    df = read_table(table_path, pair_dict=MAPPED_PAIR_DICT)
    _ = map_table(df, mean_dists)
    df.to_csv(save_path)


def crop_center(array: np.ndarray, crop_size: Union[tuple, list]):
    if array.ndim != 2:
        raise ValueError("Input array must be a 2D array")
    height, width = array.shape
    center_y, center_x = height // 2, width // 2
    half_crop_size = crop_size // 2

    # Calculate start and end indices
    start_y = max(0, center_y - half_crop_size)
    end_y = min(height, center_y + half_crop_size)
    start_x = max(0, center_x - half_crop_size)
    end_x = min(width, center_x + half_crop_size)

    # Crop and return the central square
    return array[start_y:end_y, start_x:end_x]


def copy_files_in_directory(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    files = retrive_file_names(src)

    for file in files:
        source = os.path.join(src, file)
        destination = os.path.join(dst, file)
        shutil.copy2(source, destination)


def copy_all_subfolders(src: str, dst: str) -> None:
    folders = retrive_folder_names(src)

    for folder in tqdm(folders):
        source_path = os.path.join(src, folder, SUBFIX)
        destination_path = os.path.join(dst, folder, SUBFIX)
        copy_files_in_directory(source_path, destination_path)

    print("Copying done ...")


def parallel_copy(src: str, dst: str) -> None:
    folders = retrive_folder_names(src)

    with ThreadPoolExecutor() as executor:
        for folder in tqdm(folders, desc="Copying subfolders ..."):
            source_path = os.path.join(src, folder, SUBFIX)
            destination_path = os.path.join(dst, folder, SUBFIX)
            executor.submit(copy_files_in_directory, source_path, destination_path)

    print("Copying done ...")


def retrive_folder_names(path: str) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def retrive_file_names(path: str) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def calculate_mean_value(rootpath: str, folders: list[str]) -> dict[str, float]:
    dist_dict = {}
    for folder in folders:
        distance = folder.split("_")[0]
        rawpath = os.path.join(rootpath, folder, SUBFIX)
        paths = [
            f for f in os.listdir(rawpath) if os.path.isfile(os.path.join(rawpath, f))
        ]
        mean_dist_holder = []
        for path in paths:
            path = os.path.join(rawpath, path)
            raw = load_raw(path, H, W)
            valid_raw = raw[
                ANCHOR_POINT[0] - 25 : ANCHOR_POINT[0] + 25,
                ANCHOR_POINT[1] - 25 : ANCHOR_POINT[1] + 25,
            ]
            mu = np.mean(valid_raw)
            mean_dist_holder.append(mu)
        final_mu = np.mean(mean_dist_holder)
        dist_dict[distance] = final_mu
    return dist_dict


def map_table(df: pd.DataFrame, dist_dict: dict) -> tuple[float, float]:
    df[AVG_DIST_NAME] = df[GT_DIST_NAME].astype(str).map(dist_dict)
    focal = df[FOCAL_NAME].iloc[0]  # assume focal value is the same
    baseline = df[BASLINE_NAME].iloc[0]  # assume basline value is the same

    df[AVG_DISP_NAME] = focal * baseline / df[AVG_DIST_NAME]

    return focal, baseline
