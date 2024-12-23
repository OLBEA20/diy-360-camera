import cv2

from omnicv import fisheyeImgConv
from camera360.fisheye_to_equirect_converter import FishEyeToEquirectConverterCalibrator
from main import blend_images_with_mask, swap_image_halves

rear_image_path = "./data/scene1/cam0_clean/test00735.jpeg"
front_image_path = "./data/scene1/cam1_clean/test00735.jpeg"

cam0_mapper = FishEyeToEquirectConverterCalibrator(
    fisheyeImgConv("./cam0_fisheye_params.txt")
)
cam1_mapper = FishEyeToEquirectConverterCalibrator(
    fisheyeImgConv("./cam1_fisheye_params.txt")
)


def process_parameters(
    rear_del_x,
    rear_del_y,
    rear_aperture_offset,
    front_del_x,
    front_del_y,
    front_aperture_offset,
):
    fisheye_image = cv2.imread(rear_image_path)
    rear_equirect_image = cam0_mapper.fisheye_to_equirectangular(
        fisheye_image,
        200 + rear_aperture_offset,
        rear_del_x,
        rear_del_y,
    )

    fisheye_image = cv2.imread(front_image_path)
    front_equirect_image = swap_image_halves(
        cam1_mapper.fisheye_to_equirectangular(
            fisheye_image,
            200 + front_aperture_offset,
            front_del_x,
            front_del_y,
        )
    )

    return blend_images_with_mask(rear_equirect_image, front_equirect_image, 1, 3)


if __name__ == "__main__":
    rear_delX, rear_delY, rear_aperture_offset = 0, -10, -4
    front_delX, front_delY, front_aperture_offset = 0, 10, -4
    cv2.imwrite(
        f"result_{rear_delX}_{rear_delY}_{rear_aperture_offset}_{front_delX}_{front_delY}_{front_aperture_offset}.jpeg",
        process_parameters(
            rear_delX,
            rear_delY,
            rear_aperture_offset,
            front_delX,
            front_delY,
            front_aperture_offset,
        ),
    )
