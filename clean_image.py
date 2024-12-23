import os

import cv2

from camera360.remove_outer_noise import remove_outer_noise


REAR_CAMERA_IMAGES_PATH = "./data/imx477/cam0"
FRONT_CAMERA_IMAGES_PATH = "./data/imx477/cam1"

REAR_CAMERA_IMAGES_OUT_PATH = "./data/imx477/cam0_clean"
FRONT_CAMERA_IMAGES_OUT_PATH = "./data/imx477/cam1_clean"

radius = 1510
rear_image_center_x = 2028
rear_image_center_y = 1562

front_image_center_x = 1950
front_image_center_y = 1556

if __name__ == "__main__":
    os.makedirs(REAR_CAMERA_IMAGES_OUT_PATH)
    os.makedirs(FRONT_CAMERA_IMAGES_OUT_PATH)

    for rear_image_path in os.listdir(REAR_CAMERA_IMAGES_PATH):
        if not rear_image_path.endswith(".jpg"):
            continue

        image = cv2.imread(f"{REAR_CAMERA_IMAGES_PATH}/{rear_image_path}")
        clean_image = remove_outer_noise(
            image, radius, (rear_image_center_x, rear_image_center_y)
        )
        cropped_image = clean_image[
            :, rear_image_center_x - radius : rear_image_center_x + radius
        ]
        cv2.imwrite(
            f"{REAR_CAMERA_IMAGES_OUT_PATH}/{rear_image_path}",
            cropped_image,
        )

    for front_image_path in os.listdir(FRONT_CAMERA_IMAGES_PATH):
        if not front_image_path .endswith(".jpg"):
            continue

        image = cv2.imread(f"{FRONT_CAMERA_IMAGES_PATH}/{front_image_path}")
        clean_image = remove_outer_noise(
            image, radius, (front_image_center_x, front_image_center_y)
        )
        cropped_image = clean_image[
            : , front_image_center_x - radius : front_image_center_x + radius
        ]
        cv2.imwrite(
            f"{FRONT_CAMERA_IMAGES_OUT_PATH}/{front_image_path}",
            cropped_image,
        )
