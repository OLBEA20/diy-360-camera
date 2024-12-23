from typing import Tuple

import cv2
import numpy as np

CV2_BLACK = 0
CV2_WHITE = 255
CV2_FILLED = -1


def create_image(height: int, width: int, value: int):
    shape = (height, width)
    return np.full(shape, value, np.uint8)


def get_annulus_filter_image(
    image_height: int,
    image_width: int,
    center: Tuple[int, int],
    radius: int | None = None,
) -> np.ndarray:
    annulus_image = create_image(image_height, image_width, CV2_BLACK)

    center_x, center_y =center
    annulus_image = cv2.circle(
        annulus_image, (center_x, center_y), radius, CV2_WHITE, CV2_FILLED
    )

    return annulus_image


def remove_outer_noise(image, fisheye_radius: int, center: Tuple[int, int]):
    height, width = image.shape[:2]
    annulus_filter_image = get_annulus_filter_image(
        height, width, center, fisheye_radius
    )
    return cv2.bitwise_and(image, image, mask=annulus_filter_image)
