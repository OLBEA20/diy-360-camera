import os
from tempfile import TemporaryDirectory
import cv2
from omnicv import fisheyeImgConv
from camera360.fisheye_to_equirect_converter import FishEyeToEquirectConverter
import numpy as np
import subprocess

from camera360.remove_outer_noise import remove_outer_noise


def swap_image_halves(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]

    mid = width // 2

    result = np.zeros_like(image)

    result[:, :mid] = image[:, mid:width]

    result[:, mid:width] = image[:, :mid]

    return result


REAR_CAMERA_IMAGES_PATH = "./data/cam0"
FRONT_CAMERA_IMAGES_PATH = "./data/cam1"


def blend_images_with_mask(image1, image2, weight=0.5, threshold=10):
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    mask1 = np.sum(image1, axis=2) > (threshold * 3)
    mask2 = np.sum(image2, axis=2) > (threshold * 3)

    both_color_mask = mask1 & mask2

    result = np.zeros_like(image1)

    # Where both images have color, blend them
    result[both_color_mask] = image1[both_color_mask] + image2[both_color_mask]

    # Where only img1 has color, use img1
    only_img1_mask = mask1 & ~mask2
    result[only_img1_mask] = image1[only_img1_mask]

    # Where only img2 has color, use img2
    only_img2_mask = mask2 & ~mask1
    result[only_img2_mask] = image2[only_img2_mask]

    # Convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


if __name__ == "__main__":
    param_file_path = "./fisheye_params.txt"
    cam0_mapper = FishEyeToEquirectConverter(
        fisheyeImgConv("./cam0_fisheye_params.txt")
    )
    cam1_mapper = FishEyeToEquirectConverter(
        fisheyeImgConv("./cam1_fisheye_params.txt")
    )

    with TemporaryDirectory() as back_temp_dir:
        with TemporaryDirectory() as front_temp_dir:
            with TemporaryDirectory() as merged_temp_dir:
                for image_path in os.listdir(REAR_CAMERA_IMAGES_PATH):
                    if image_path.endswith(".jpeg"):
                        fisheye_image = cv2.imread(
                            os.path.join(REAR_CAMERA_IMAGES_PATH, image_path)
                        )
                        equirect_image = cam0_mapper.fisheye_to_equirectangular(
                            remove_outer_noise(fisheye_image, 1205, -19, 5)
                        )
                        cv2.imwrite(
                            os.path.join(back_temp_dir, image_path), equirect_image
                        )

                for image_path in os.listdir(FRONT_CAMERA_IMAGES_PATH):
                    if image_path.endswith(".jpeg"):
                        fisheye_image = cv2.imread(
                            os.path.join(FRONT_CAMERA_IMAGES_PATH, image_path)
                        )
                        equirect_image = cam1_mapper.fisheye_to_equirectangular(
                            remove_outer_noise(fisheye_image, 1205, 26, 5)
                        )
                        cv2.imwrite(
                            os.path.join(front_temp_dir, image_path),
                            swap_image_halves(equirect_image),
                        )

                for path_a, path_b in zip(
                    os.listdir(back_temp_dir), os.listdir(front_temp_dir)
                ):
                    if path_a.endswith(".jpeg") and path_b.endswith(".jpeg"):
                        a = cv2.imread(os.path.join(back_temp_dir, path_a))
                        b = cv2.imread(os.path.join(front_temp_dir, path_b))

                        merged = blend_images_with_mask(a, b, 0.5, 1)

                        cv2.imwrite(os.path.join(merged_temp_dir, path_a), merged)

                command = [
                    "ffmpeg",
                    "-framerate",
                    "15",
                    "-i",
                    os.path.join(merged_temp_dir, "test%05d.jpeg"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "merged.mp4",
                ]
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                process.wait()
                if process.returncode != 0:
                    print(process.returncode)
                    print(process.stdout.read())
                    print(process.stderr.read())
