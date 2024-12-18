import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .constants import *
from .utils import load_raw, read_table


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
    # by need of QA team

    cam_name = "camparam.txt"
    for folder in tqdm(folders):
        source_path = os.path.join(src, folder, SUBFIX)
        destination_path = os.path.join(dst, folder, SUBFIX)
        copy_files_in_directory(source_path, destination_path)
        cam_source = os.path.join(src, folder, cam_name)
        cam_dest = os.path.join(dst, folder, cam_name)
        shutil.copy2(cam_source, cam_dest)
    print("Copying done ...")


def parallel_copy(src: str, dst: str) -> None:
    folders = retrive_folder_names(src)
    cam_name = "camparam.txt"

    with ThreadPoolExecutor() as executor:
        for folder in tqdm(folders, desc="Copying subfolders ..."):
            source_path = os.path.join(src, folder, SUBFIX)
            destination_path = os.path.join(dst, folder, SUBFIX)
            cam_source = os.path.join(src, folder, cam_name)
            cam_dest = os.path.join(dst, folder, cam_name)
            executor.submit(copy_files_in_directory, source_path, destination_path)
            executor.submit(shutil.copy2, cam_source, cam_dest)
    print("Copying done ...")


def retrive_folder_names(path: str) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def retrive_file_names(path: str) -> list[str]:
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def calculate_mean_value(
    rootpath: str, folders: list[str], is_median: bool = False
) -> dict[str, float]:
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
            if is_median:
                mu = np.median(valid_raw)
            else:
                mu = np.mean(valid_raw, where=valid_raw != 0)
            mean_dist_holder.append(mu)
        final_mu = np.mean(mean_dist_holder)
        dist_dict[distance] = final_mu
    return dist_dict


def get_anchor_point_data(rootpath: str, folders: list[str]):
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
            valid_raw = raw[ANCHOR_POINT[0], ANCHOR_POINT[1]]
            mu = valid_raw
            mean_dist_holder.append(mu)
        final_mu = np.mean(mean_dist_holder)
        dist_dict[distance] = final_mu
    return dist_dict


def map_table(df: pd.DataFrame, dist_dict: dict) -> tuple[float, float]:
    df[AVG_DIST_NAME] = df[GT_DIST_NAME].astype(str).map(dist_dict)
    focal = df[FOCAL_NAME].iloc[0]  # assume focal value is the same
    baseline = df[BASLINE_NAME].iloc[0]  # assume basline value is the same

    focal *= FOCAL_MULTIPLIER  # very dirty hack
    df[AVG_DISP_NAME] = focal * baseline / df[AVG_DIST_NAME]
    # df[GT_DISP_NAME] = focal * baseline / df[GT_DIST_NAME]
    # df[GT_DISP_ERROR_NAME] = df[GT_DISP_NAME] - df[AVG_DISP_NAME]

    return focal, baseline

def map_table_debug(df: pd.DataFrame, dist_dict1: dict, dist_dict2: dict) -> tuple[float, float]:
    df[AVG_DIST_NAME] = df[GT_DIST_NAME].astype(str).map(dist_dict1)
    focal = df[FOCAL_NAME].iloc[0]  # assume focal value is the same
    baseline = df[BASLINE_NAME].iloc[0]  # assume basline value is the same

    focal *= FOCAL_MULTIPLIER  # very dirty hack
    df[AVG_DISP_NAME] = focal * baseline / df[AVG_DIST_NAME]

    df["ANCHOR_POINT_DATA"] = df[GT_DIST_NAME].astype(str).map(dist_dict2)
    # df[GT_DISP_NAME] = focal * baseline / df[GT_DIST_NAME]
    # df[GT_DISP_ERROR_NAME] = df[GT_DISP_NAME] - df[AVG_DISP_NAME]
    return focal, baseline


def preprocessing(path, table_path, paid_dict=MAPPED_PAIR_DICT):
    all_distances = retrive_folder_names(path)
    mean_dists = calculate_mean_value(path, all_distances)
    # s_dists = get_anchor_point_data(path, all_distances)
    df = read_table(table_path, pair_dict=paid_dict)
    focal, baseline = map_table(df, mean_dists)
    # focal, baseline = map_table_debug(df, mean_dists, s_dists)
    return df, focal, baseline
