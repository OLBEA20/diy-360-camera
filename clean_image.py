import os

import cv2

from diy_camera_360.remove_outer_noise import remove_outer_noise


REAR_CAMERA_IMAGES_PATH = "./data/imx477/scene6/cam0"
FRONT_CAMERA_IMAGES_PATH = "./data/imx477/scene6/cam1"

REAR_CAMERA_IMAGES_OUT_PATH = "./data/imx477/scene6/cam0_clean"
FRONT_CAMERA_IMAGES_OUT_PATH = "./data/imx477/scene6/cam1_clean"

# rear_radius = 1510
rear_radius = 1430
front_radius = 1510
# rear_image_center_x = 1954
# rear_image_center_y = 1575
rear_image_center_x = 1960
rear_image_center_y = 1603

front_image_center_x = 2012
front_image_center_y = 1560

if __name__ == "__main__":
    os.makedirs(REAR_CAMERA_IMAGES_OUT_PATH)
    os.makedirs(FRONT_CAMERA_IMAGES_OUT_PATH)

    for rear_image_path in os.listdir(REAR_CAMERA_IMAGES_PATH):
        if not rear_image_path.endswith(".jpg"):
            continue

        image = cv2.imread(f"{REAR_CAMERA_IMAGES_PATH}/{rear_image_path}")
        clean_image = remove_outer_noise(
            image, rear_radius, (rear_image_center_x, rear_image_center_y)
        )
        cropped_image = clean_image[
            rear_image_center_y - rear_radius : rear_image_center_y + rear_radius,
            rear_image_center_x - rear_radius : rear_image_center_x + rear_radius,
        ]
        cv2.imwrite(
            f"{REAR_CAMERA_IMAGES_OUT_PATH}/{rear_image_path}",
            cropped_image,
        )

    for front_image_path in os.listdir(FRONT_CAMERA_IMAGES_PATH):
        if not front_image_path.endswith(".jpg"):
            continue

        image = cv2.imread(f"{FRONT_CAMERA_IMAGES_PATH}/{front_image_path}")
        clean_image = remove_outer_noise(
            image, front_radius, (front_image_center_x, front_image_center_y)
        )
        cropped_image = clean_image[
            front_image_center_y - front_radius : front_image_center_y + front_radius :,
            front_image_center_x - front_radius : front_image_center_x + front_radius,
        ]
        cv2.imwrite(
            f"{FRONT_CAMERA_IMAGES_OUT_PATH}/{front_image_path}",
            cropped_image,
        )
