import os

import cv2
import numpy as np


if __name__ == "__main__":
    for rear_image_path, front_image_path in zip(
        os.listdir("./data/scene1/cam0_clean"),
        os.listdir("./data/scene1/cam1_clean"),
    ):
        left = cv2.imread(f"./data/scene1/cam0_clean/{rear_image_path}")
        right = cv2.imread(f"./data/scene1/cam1_clean/{front_image_path}")

        cv2.imwrite(
            f"./data/scene1/merged_clean/{rear_image_path}", np.hstack((left, right))
        )
