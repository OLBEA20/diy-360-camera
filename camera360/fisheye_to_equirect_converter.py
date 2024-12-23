import math
import cv2
import numpy as np


class FishEyeToEquirectConverter:
    def __init__(
        self, radius: int, aperture: int, del_x: int, del_y: int, rotation_y: float
    ):
        self._map_x = None
        self._map_y = None
        self._radius = radius
        self._aperture = aperture
        self._del_x = del_x
        self._del_y = del_y
        self._rotation_y = rotation_y

    def fisheye_to_equirectangular(self, image, out_shape=(960, 1920)):
        if self._map_x is not None and self._map_y is not None:
            return self.applyMap(image)

        return self._fisheye_to_equirectangular(image, out_shape)

    def _fisheye_to_equirectangular(self, source_frame, out_shape):
        in_height, in_width = source_frame.shape[:2]
        out_height, out_width = out_shape

        self._map_x = np.zeros((out_height, out_width), np.float32)
        self._map_y = np.zeros((out_height, out_width), np.float32)

        center_x = in_width // 2 - self._del_x
        center_y = in_height // 2 - self._del_y

        i, j = np.meshgrid(np.arange(0, int(out_height)), np.arange(0, int(out_width)))

        xyz = np.zeros((out_height, out_width, 3))
        x, y, z = np.split(xyz, 3, axis=-1)

        x = (
            self._radius
            * np.cos((i * 1.0 / out_height - 0.5) * np.pi)
            * np.cos((j * 1.0 / out_height - 0.5) * np.pi)
        )
        y = (
            self._radius
            * np.cos((i * 1.0 / out_height - 0.5) * np.pi)
            * np.sin((j * 1.0 / out_height - 0.5) * np.pi)
        )
        z = self._radius * np.sin((i * 1.0 / out_height - 0.5) * np.pi)

        x, y, z = rotate_around_y((x, y, z), self._rotation_y)

        r = (
            2
            * np.arctan2(np.sqrt(x**2 + z**2), y)
            / np.pi
            * 180
            / self._aperture
            * self._radius
        )
        theta = np.arctan2(z, x)

        self._map_x = np.multiply(r, np.cos(theta)).T.astype(np.float32) + center_x
        self._map_y = np.multiply(r, np.sin(theta)).T.astype(np.float32) + center_y

        return cv2.remap(
            source_frame,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def applyMap(self, srcFrame):
        return cv2.remap(
            srcFrame,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def rotate_around_y(point, angle_degrees):
    x, y, z = point
    angle_radians = math.radians(angle_degrees)

    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)

    x_new = x * cos_theta + z * sin_theta
    y_new = y
    z_new = -x * sin_theta + z * cos_theta

    return (x_new, y_new, z_new)
