import cv2
import numpy as np
from camera360.blend_image_with_mask import blend_images_with_mask
from camera360.fisheye_to_equirect_converter import CameraParameters, FishEyeToEquirectConverter
from camera360.horizontal_fade import fade_horizontal_edges
from camera360.swap_image_halves import swap_image_halves

REAR_IMAGE_PATH = "./data/imx477/scene2/cam0_clean/frame_00810.jpg"

FRONT_IMAGE_PATH = "./data/imx477/scene2/cam1_clean/frame_00810.jpg"

if __name__ == "__main__":
    cam0_params = CameraParameters(aperture=185.4, del_x=45, del_y=-25, rotation=(2.44, -1.05, -2.36)) 
    cam1_params = CameraParameters(aperture=185.9, del_x=-64, del_y=-13, rotation=(3.31, -0.17, 3.08))

    _cam0_mapper = FishEyeToEquirectConverter(1510, cam0_params)
    _cam1_mapper = FishEyeToEquirectConverter(1510, cam1_params)

    fisheye_image = cv2.imread(REAR_IMAGE_PATH)
    rear_equirect_image = fade_horizontal_edges(
        _cam0_mapper.fisheye_to_equirectangular(fisheye_image, (3040, 6080))
    )

    fisheye_image = cv2.imread(FRONT_IMAGE_PATH)
    front_equirect_image = fade_horizontal_edges(
        _cam1_mapper.fisheye_to_equirectangular(fisheye_image, (3040, 6080))
    )

    cv2.imwrite("merged_temp.jpg", np.vstack((rear_equirect_image, swap_image_halves(front_equirect_image))))

    merged = np.clip(rear_equirect_image + swap_image_halves(front_equirect_image), 0, 255).astype(np.uint8)

    cv2.imwrite("merged.jpg", merged)