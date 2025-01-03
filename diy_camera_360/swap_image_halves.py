import numpy as np


def swap_image_halves(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    mid = width // 2
    result = np.zeros_like(image)
    result[:, :mid] = image[:, mid:width]
    result[:, mid:width] = image[:, :mid]

    return result
