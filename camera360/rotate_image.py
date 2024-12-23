import cv2
import numpy as np


def rotate_image(image, angle_deg):
    (h, w) = image.shape[:2]

    new_w = int(
        w * abs(np.cos(np.radians(angle_deg))) + h * abs(np.sin(np.radians(angle_deg)))
    )
    new_h = int(
        h * abs(np.cos(np.radians(angle_deg))) + w * abs(np.sin(np.radians(angle_deg)))
    )

    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

    return rotated_image
