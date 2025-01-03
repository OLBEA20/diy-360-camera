import cv2
import numpy as np

def fade_horizontal_edges(image, fade_angle=3.75):
    height, width, _ = image.shape
    fade_width = round((fade_angle / 360) * width)

    fade_mask = np.ones((height, width), dtype=float)

    left = round((90 / 360) * width - fade_width / 2)
    for x in range(left, left + fade_width + 1):
        fade_value = (x - left) / fade_width
        fade_mask[:, x] *= fade_value + (1/fade_width)

    for x in range(0, left):
        fade_mask[:, x] *= 0

    right = round((270 / 360) * width - fade_width / 2)
    for x in range(right - 1, right + fade_width):
        fade_value = (right + fade_width - x - 1) / fade_width
        fade_mask[:, x] *= fade_value + (1/fade_width)

    for x in range(right + fade_width, width):
        fade_mask[:, x] *= 0

    fade_mask = np.stack([fade_mask] * 3, axis=-1)

    return image * fade_mask


def fade_horizontal_edges_2(image, fade_angle=3.75):
    img_array = add_alpha(image)

    height, width, _ = img_array.shape
    fade_width = round((fade_angle / 360) * width)

    fade_mask = np.ones((height, width), dtype=float)

    left = round((90 / 360) * width - fade_width / 2)
    for x in range(left, left + fade_width):
        fade_value = (x - left) / fade_width
        fade_mask[:, x] *= fade_value

    for x in range(0, left):
        fade_mask[:, x] *= 0

    right = round((270 / 360) * width - fade_width / 2) 
    for x in range(right, right + fade_width):
        fade_value = ((right + fade_width) - x - 1) / fade_width
        fade_mask[:, x] *= fade_value

    for x in range(right + fade_width, width):
        fade_mask[:, x] *= 0

    fade_mask = np.stack([fade_mask] * 4, axis=-1)

    return (img_array * fade_mask).astype(np.uint8)


def add_alpha(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    height, width = image.shape[:2]
    alpha_channel = np.ones((height, width), dtype=image.dtype) * 255

    return cv2.merge((image[:, :, 0], image[:, :, 1], image[:, :, 2], alpha_channel))
