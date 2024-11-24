import os
from tempfile import TemporaryDirectory
from typing import Tuple
import cv2
from omnicv import fisheyeImgConv
import numpy as np
import subprocess

CV2_BLACK = 0
CV2_WHITE = 255
CV2_FILLED = -1


class FishEyeToEquirectConverter:
    def __init__(self, mapper: fisheyeImgConv):
        self._mapper = mapper
        self._map_created = False
    
    def fisheye_to_equirectangular(self, image: cv2.Mat, outShape=(960, 1920)):
        if not self._map_created :
            equirect_image = self._mapper.fisheye2equirect(image, outShape)
            self._map_created = True
        else:
            equirect_image = self._mapper.applyMap(0, image)

        return equirect_image

def create_image(height: int, width: int, value: int):
    shape = (height, width)
    return np.full(shape, value, np.uint8)

def get_annulus_filter_image(
    image_height: int,
    image_width: int,
    center: Tuple[int, int],
    outer_radius: int | None = None,
) -> np.ndarray:
    annulus_image = create_image(image_height, image_width, CV2_BLACK)

    center_y = int(annulus_image.shape[0] / 2) if center is None else center[1]
    center_x = int(annulus_image.shape[1] / 2) if center is None else center[0]
    annulus_image = cv2.circle(
        annulus_image, (center_x, center_y), outer_radius, CV2_WHITE, CV2_FILLED
    )

    return annulus_image

def remove_outer_noise(image, fisheye_radius: int):
    height, width = image.shape[:2]
    annulus_filter_image = get_annulus_filter_image(
        height, width, (width//2, height//2), fisheye_radius
    )
    return cv2.bitwise_and(
        image, image, mask=annulus_filter_image
    )

def swap_image_halves(image: np.ndarray | cv2.Mat) -> np.ndarray | cv2.Mat:
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
    result[both_color_mask] = (
        image1[both_color_mask] * weight + 
        image2[both_color_mask] * (1 - weight)
    )
    
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
    cam0_mapper = FishEyeToEquirectConverter(fisheyeImgConv("./cam0_fisheye_params.txt"))
    cam1_mapper = FishEyeToEquirectConverter(fisheyeImgConv("./cam1_fisheye_params.txt"))

    with TemporaryDirectory() as back_temp_dir:
        with TemporaryDirectory() as front_temp_dir:
            with TemporaryDirectory() as merged_temp_dir:
                for image_path in os.listdir(REAR_CAMERA_IMAGES_PATH):
                    if image_path.endswith(".jpeg"):
                        fisheye_image = cv2.imread(os.path.join(REAR_CAMERA_IMAGES_PATH, image_path))
                        equirect_image = cam0_mapper.fisheye_to_equirectangular(remove_outer_noise(fisheye_image, 1205))
                        cv2.imwrite(os.path.join(back_temp_dir, image_path), equirect_image)

                for image_path in os.listdir(FRONT_CAMERA_IMAGES_PATH):
                    if image_path.endswith(".jpeg"):
                        fisheye_image = cv2.imread(os.path.join(FRONT_CAMERA_IMAGES_PATH, image_path))
                        equirect_image = cam1_mapper.fisheye_to_equirectangular(remove_outer_noise(fisheye_image, 1205))
                        cv2.imwrite(os.path.join(front_temp_dir, image_path), swap_image_halves(equirect_image))

                for path_a, path_b in zip(os.listdir(back_temp_dir), os.listdir(front_temp_dir)):
                    if path_a.endswith(".jpeg") and path_b.endswith(".jpeg"):
                        a = cv2.imread(os.path.join(back_temp_dir, path_a))
                        b = cv2.imread(os.path.join(front_temp_dir, path_b))

                        merged = blend_images_with_mask(a, b, 0.5, 1)

                        cv2.imwrite(os.path.join(merged_temp_dir, path_a), merged)
            
                command = ["ffmpeg", "-framerate", "15", "-i", os.path.join(merged_temp_dir, "test%05d.jpeg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "merged.mp4"]
                process = subprocess.Popen(command , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    print(process.returncode)
                    print(process.stdout.read())
                    print(process.stderr.read())
