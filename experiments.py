from KBD.utils import load_raw

import numpy as np
import matplotlib.pyplot as plt    

if __name__ == "__main__":
    path1 = "/home/william/Codes/KBD/data/N09ASH24DH0015/image_data_transformed_linear/1600_N09ASH24DH0015_2024_05_31_16_55_02/DEPTH/raw/Depth-2024-5-31-16-55-20-751-25-34966-2339716515.raw"
    path2 = "/home/william/Codes/KBD/data/N09ASH24DH0015/image_data_transformed_linear2/1600_N09ASH24DH0015_2024_05_31_16_55_02/DEPTH/raw/Depth-2024-5-31-16-55-20-751-25-34966-2339716515.raw"

    depth1 = load_raw(path1, 480, 640)
    depth2 = load_raw(path2, 480, 640)

    print(depth1[177, 177])
    print(depth2[177, 177])

    diff = depth1 - depth2
    print(diff)

    print(np.sum(diff))

    plt.figure()
    plt.imshow(diff)
    plt.show()
