import os

import cv2

from diy_camera_360.remove_outer_noise import remove_outer_noise


def clean_images(
    rear_camera_images_path,
    rear_camera_images_out_path,
    rear_radius,
    rear_image_center_x,
    rear_image_center_y,
    front_camera_images_path,
    front_camera_images_out_path,
    front_radius,
    front_image_center_x,
    front_image_center_y,
):
    os.makedirs(rear_camera_images_out_path)
    os.makedirs(front_camera_images_out_path)

    for rear_image_path in os.listdir(rear_camera_images_path):
        if not rear_image_path.endswith(".jpg"):
            continue

        image = cv2.imread(f"{rear_camera_images_path}/{rear_image_path}")
        clean_image = remove_outer_noise(
            image, rear_radius, (rear_image_center_x, rear_image_center_y)
        )
        cropped_image = clean_image[
            rear_image_center_y - rear_radius : rear_image_center_y + rear_radius,
            rear_image_center_x - rear_radius : rear_image_center_x + rear_radius,
        ]
        cv2.imwrite(
            f"{rear_camera_images_out_path}/{rear_image_path}",
            cropped_image,
        )

    for front_image_path in os.listdir(front_camera_images_path):
        if not front_image_path.endswith(".jpg"):
            continue

        image = cv2.imread(f"{front_camera_images_path}/{front_image_path}")
        clean_image = remove_outer_noise(
            image, front_radius, (front_image_center_x, front_image_center_y)
        )
        cropped_image = clean_image[
            front_image_center_y - front_radius : front_image_center_y + front_radius :,
            front_image_center_x - front_radius : front_image_center_x + front_radius,
        ]
        cv2.imwrite(
            f"{front_camera_images_out_path}/{front_image_path}",
            cropped_image,
        )


def clean_imx477():
    rear_camera_images_path = "./data/imx477/scene10/cam0"
    front_camera_images_path = "./data/imx477/scene10/cam1"

    rear_camera_images_out_path = "./data/imx477/scene10/cam0_clean"
    front_camera_images_out_path = "./data/imx477/scene10/cam1_clean"

    # rear_radius = 1510
    rear_radius = 1430
    front_radius = 1430
    # rear_image_center_x = 1954
    # rear_image_center_y = 1575
    rear_image_center_x = 1960
    rear_image_center_y = 1603

    # front_image_center_x = 2012
    # front_image_center_y = 1560
    front_image_center_x = 1933
    front_image_center_y = 1631

    clean_images(
        rear_camera_images_path,
        rear_camera_images_out_path,
        rear_radius,
        rear_image_center_x,
        rear_image_center_y,
        front_camera_images_path,
        front_camera_images_out_path,
        front_radius,
        front_image_center_x,
        front_image_center_y,
    )


def clean_gopro():
    rear_camera_images_path = "./data/gopro/scene1/cam0"
    front_camera_images_path = "./data/gopro/scene1/cam1"

    rear_camera_images_out_path = "./data/gopro/scene1/cam0_clean"
    front_camera_images_out_path = "./data/gopro/scene1/cam1_clean"

    rear_radius = 352
    front_radius = 352
    rear_image_center_x = 352
    rear_image_center_y = 352

    front_image_center_x = 352
    front_image_center_y = 352

    clean_images(
        rear_camera_images_path,
        rear_camera_images_out_path,
        rear_radius,
        rear_image_center_x,
        rear_image_center_y,
        front_camera_images_path,
        front_camera_images_out_path,
        front_radius,
        front_image_center_x,
        front_image_center_y,
    )


if __name__ == "__main__":
    clean_imx477()
