from KBD.utils import load_raw

import numpy as np

if __name__ == "__main__":
    path1 = "/home/william/Codes/KBD/data/N09ASH24DH0015/image_data_transformed_linear/800_N09ASH24DH0015_2024_05_31_16_38_41/DEPTH/raw/Depth-2024-5-31-16-38-49-549-12-20107-1349126441.raw"
    path2 = "/home/william/Codes/KBD/data/N09ASH24DH0015/image_data_transformed_linear2/800_N09ASH24DH0015_2024_05_31_16_38_41/DEPTH/raw/Depth-2024-5-31-16-38-49-549-12-20107-1349126441.raw"

    depth1 = load_raw(path1, 480, 640)
    depth2 = load_raw(path2, 480, 640)

    print(depth1[177, 177])
    print(depth2[177, 177])

    print(np.sum(depth1-depth2))