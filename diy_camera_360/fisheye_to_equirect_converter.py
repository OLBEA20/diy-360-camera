from dataclasses import dataclass
import math
import cv2
import numpy as np

@dataclass
class CameraParameters:
    aperture: float
    del_x: int
    del_y: int
    rotation: tuple[float, float, float]

    def with_aperture(self, aperture: float):
        return CameraParameters(
            aperture=aperture,
            del_x=self.del_x,
            del_y=self.del_y,
            rotation=self.rotation,
        )

    def with_del_x(self, del_x: float):
        return CameraParameters(
            aperture=self.aperture,
            del_x=del_x,
            del_y=self.del_y,
            rotation=self.rotation,
        )

    def with_del_y(self, del_y: float):
        return CameraParameters(
            aperture=self.aperture,
            del_x=self.del_x,
            del_y=del_y,
            rotation=self.rotation,
        )

    def with_rotation(self, rotation: tuple[float, float, float]):
        return CameraParameters(
            aperture=self.aperture,
            del_x=self.del_x,
            del_y=self.del_y,
            rotation=rotation,
        )

class FishEyeToEquirectConverter:
    def __init__(
        self,
        radius: int,
        camera_parameters: CameraParameters
    ):
        self._map_x = None
        self._map_y = None
        self._radius = radius
        self._aperture = camera_parameters.aperture
        self._del_x = camera_parameters.del_x
        self._del_y = camera_parameters.del_y
        self._rotation = camera_parameters.rotation

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

        x, y, z = rotate_point_separately((x, y, z), *self._rotation)

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


def rotate_point(point, angle_x, angle_y, angle_z):
    angle_x = np.radians(angle_x)
    angle_y = np.radians(angle_y)
    angle_z = np.radians(angle_z)

    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )

    R_y = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ]
    )

    R_z = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    return np.dot(R, point)


def rotate_point_separately(point, angle_x, angle_y, angle_z):
    """
    Rotates a 3D point around the x, y, and z axes sequentially.

    Parameters:
        point (list or np.array): The 3D point as [x, y, z].
        angle_x (float): Rotation angle around the x-axis in degrees.
        angle_y (float): Rotation angle around the y-axis in degrees.
        angle_z (float): Rotation angle around the z-axis in degrees.

    Returns:
        np.array: The rotated 3D point.
    """
    # Convert angles to radians
    angle_x = np.radians(angle_x)
    angle_y = np.radians(angle_y)
    angle_z = np.radians(angle_z)

    # Unpack the original point
    x, y, z = point

    # Step 1: Rotate around X-axis
    y_new = y * np.cos(angle_x) - z * np.sin(angle_x)
    z_new = y * np.sin(angle_x) + z * np.cos(angle_x)
    y, z = y_new, z_new  # Update y and z

    # Step 2: Rotate around Y-axis
    x_new = x * np.cos(angle_y) + z * np.sin(angle_y)
    z_new = -x * np.sin(angle_y) + z * np.cos(angle_y)
    x, z = x_new, z_new  # Update x and z

    # Step 3: Rotate around Z-axis
    x_new = x * np.cos(angle_z) - y * np.sin(angle_z)
    y_new = x * np.sin(angle_z) + y * np.cos(angle_z)
    x, y = x_new, y_new  # Update x and y

    # Return the rotated point
    return x, y, z


def rotate_meshgrid(x, y, z, angle_x, angle_y, angle_z):
    """
    Rotates a 3D meshgrid around the x, y, and z axes.

    Parameters:
        x, y, z (np.array): The 3D meshgrid coordinate arrays.
        angle_x (float): Rotation angle around the x-axis in degrees.
        angle_y (float): Rotation angle around the y-axis in degrees.
        angle_z (float): Rotation angle around the z-axis in degrees.

    Returns:
        x_rot, y_rot, z_rot (np.array): Rotated 3D meshgrid coordinate arrays.
    """
    # Convert angles to radians
    angle_x = np.radians(angle_x)
    angle_y = np.radians(angle_y)
    angle_z = np.radians(angle_z)

    # Rotation matrices
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)],
        ]
    )

    R_y = np.array(
        [
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)],
        ]
    )

    R_z = np.array(
        [
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1],
        ]
    )

    # Combine rotations: R = Rz * Ry * Rx
    R = np.dot(R_z, np.dot(R_y, R_x))

    # Flatten the meshgrid into a list of points (N x 3)
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()])  # Shape: (3, N)

    # Apply the rotation
    rotated_points = np.dot(R, points)  # Shape: (3, N)

    # Reshape back to the original meshgrid shape
    x_rot = rotated_points[0, :].reshape(x.shape)
    y_rot = rotated_points[1, :].reshape(y.shape)
    z_rot = rotated_points[2, :].reshape(z.shape)

    return x_rot, y_rot, z_rot
