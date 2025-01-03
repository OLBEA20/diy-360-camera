import cv2
import numpy as np
from find_parameters import CameraParameters

from camera360.fisheye_to_equirect_converter import FishEyeToEquirectConverter

REAR_IMAGE_PATH = "./data/imx477/scene2/cam0_clean/frame_00214.jpg"
CAM0_FISHEYE_IMAGE = cv2.imread(REAR_IMAGE_PATH)

FRONT_IMAGE_PATH = "./data/imx477/scene2/cam1_clean/frame_00214.jpg"
CAM1_FISHEYE_IMAGE = cv2.imread(FRONT_IMAGE_PATH)

RADIUS = 1510


def process_parameters(
    cam0_parameters: CameraParameters, cam1_parameters: CameraParameters
):
    cam0_mapper = FishEyeToEquirectConverter(RADIUS, cam0_parameters)
    cam1_mapper = FishEyeToEquirectConverter(RADIUS, cam1_parameters)

    rear_equirect_image = cam0_mapper.fisheye_to_equirectangular(
        CAM0_FISHEYE_IMAGE, out_shape=(960, 1920)
    )
    front_equirect_image = cam1_mapper.fisheye_to_equirectangular(
        CAM1_FISHEYE_IMAGE, out_shape=(960, 1920)
    )

    left_left = 470
    left_right = 490

    right_left = 1430
    right_right = 1450

    vertical_height = 530

    left = np.hstack(
        (
            rear_equirect_image[0:vertical_height, left_left:left_right],
            rear_equirect_image[0:vertical_height, right_left:right_right],
        )
    )
    right = np.hstack(
        (
            front_equirect_image[0:vertical_height, right_left:right_right],
            front_equirect_image[0:vertical_height, left_left:left_right],
        )
    )

    return cv2.imwrite("overlap.png", np.vstack((left, right)))


if __name__ == "__main__":
    process_parameters(
        CameraParameters(188, 0, 0, (0, 0, 0)), CameraParameters(188, 0, 0, (0, 0, 0))
    )
