#!/usr/bin/env/python
import cv2
import sys
from camera360.fisheye_to_equirect_converter import (
    FishEyeToEquirectConverter,
)
from camera360.swap_image_halves import swap_image_halves
import numpy as np


def hard_merge(image1, image2):
    left = image2[:, :480]
    middle = image1[:, 480:1440]
    right = image2[:, 1440:]

    return np.hstack((left, middle, right))


def process_parameters(
    rear_radius,
    rear_del_x,
    rear_del_y,
    rear_aperture,
    rear_rotation,
    front_radius,
    front_del_x,
    front_del_y,
    front_aperture,
    front_rotation,
    out_shape,
):
    fisheye_image = cv2.imread(rear_image_path)

    cam0_mapper = FishEyeToEquirectConverter(
        rear_radius, rear_aperture, rear_del_x, rear_del_y, rear_rotation
    )
    rear_equirect_image = cam0_mapper.fisheye_to_equirectangular(
        fisheye_image,
        out_shape,
    )

    fisheye_image = cv2.imread(front_image_path)
    cam1_mapper = FishEyeToEquirectConverter(
        front_radius, front_aperture, front_del_x, front_del_y, front_rotation
    )
    front_equirect_image = swap_image_halves(
        cam1_mapper.fisheye_to_equirectangular(
            fisheye_image,
            out_shape,
        )
    )

    return hard_merge(rear_equirect_image, front_equirect_image)


def nothing(x):
    pass


TRACKBAR_WINDOW = "image"
IMAGE_WINDOW = "output image"
rear_image_path = sys.argv[1]
front_image_path = sys.argv[2]
REAR_RADIUS = None
FRONT_RADIUS = None

# Example of using the converter class
frame = cv2.imread(rear_image_path)
cv2.namedWindow(TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(TRACKBAR_WINDOW, 800, 800)
height, width = frame.shape[:2]
cv2.createTrackbar("rear_radius", TRACKBAR_WINDOW, 1050, width, nothing)
cv2.createTrackbar("rear_Cx", TRACKBAR_WINDOW, width // 2, width, nothing)
cv2.createTrackbar("rear_Cy", TRACKBAR_WINDOW, height // 2, width, nothing)

while True:
    if True:
        frame = cv2.imread(rear_image_path)
        rear_radius = cv2.getTrackbarPos("rear_radius", TRACKBAR_WINDOW)
        rear_Cx = cv2.getTrackbarPos("rear_Cx", TRACKBAR_WINDOW)
        rear_Cy = cv2.getTrackbarPos("rear_Cy", TRACKBAR_WINDOW)
        frame = cv2.circle(frame, (rear_Cx, rear_Cy), rear_radius, (0, 200, 0), 2)
        cv2.imshow(TRACKBAR_WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            REAR_RADIUS = rear_radius
            cv2.destroyAllWindows()
            break

frame = cv2.imread(front_image_path)
cv2.namedWindow(TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(TRACKBAR_WINDOW, 800, 800)
cv2.createTrackbar("front_radius", TRACKBAR_WINDOW, 1050, width, nothing)
cv2.createTrackbar("front_Cx", TRACKBAR_WINDOW, width // 2, width, nothing)
cv2.createTrackbar("front_Cy", TRACKBAR_WINDOW, height // 2, width, nothing)

while True:
    if True:
        frame = cv2.imread(front_image_path)
        front_radius = cv2.getTrackbarPos("front_radius", TRACKBAR_WINDOW)
        front_Cx = cv2.getTrackbarPos("front_Cx", TRACKBAR_WINDOW)
        front_Cy = cv2.getTrackbarPos("front_Cy", TRACKBAR_WINDOW)
        frame = cv2.circle(frame, (front_Cx, front_Cy), front_radius, (0, 200, 0), 2)
        cv2.imshow(TRACKBAR_WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            FRONT_RADIUS = front_radius
            cv2.destroyAllWindows()
            break

TRACKBAR_WINDOW = "set aperture"

cv2.namedWindow(TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
cv2.namedWindow(IMAGE_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(TRACKBAR_WINDOW, 600, 1200)
cv2.resizeWindow(IMAGE_WINDOW, 600, 1200)

cv2.createTrackbar("rear aperture", TRACKBAR_WINDOW, 196, 1000, nothing)
cv2.createTrackbar("rear del Cx", TRACKBAR_WINDOW, 500, 1000, nothing)
cv2.createTrackbar("rear del Cy", TRACKBAR_WINDOW, 500, 1000, nothing)
cv2.createTrackbar("rear rotation x", TRACKBAR_WINDOW, 50, 100, nothing)
cv2.createTrackbar("rear rotation y", TRACKBAR_WINDOW, 50, 100, nothing)
cv2.createTrackbar("rear rotation z", TRACKBAR_WINDOW, 50, 100, nothing)

cv2.createTrackbar("front aperture", TRACKBAR_WINDOW, 196, 1000, nothing)
cv2.createTrackbar("front del Cx", TRACKBAR_WINDOW, 500, 1000, nothing)
cv2.createTrackbar("front del Cy", TRACKBAR_WINDOW, 500, 1000, nothing)
cv2.createTrackbar("front rotation x", TRACKBAR_WINDOW, 50, 100, nothing)
cv2.createTrackbar("front rotation y", TRACKBAR_WINDOW, 50, 100, nothing)
cv2.createTrackbar("front rotation z", TRACKBAR_WINDOW, 50, 100, nothing)

# Example of using the converter class
frame = cv2.imread(rear_image_path)
frame = cv2.circle(
    frame, (frame.shape[1] // 2, frame.shape[0] // 2), 4, (255, 255, 255), -1
)
frame = cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 2, (0, 0, 0), -1)

outShape = [960, 1920]
inShape = frame.shape[:2]

while True:
    if True:
        rear_aperture = cv2.getTrackbarPos("rear aperture", TRACKBAR_WINDOW)
        rear_delx = cv2.getTrackbarPos("rear del Cx", TRACKBAR_WINDOW) - 500
        rear_dely = cv2.getTrackbarPos("rear del Cy", TRACKBAR_WINDOW) - 500
        rotation_x = (cv2.getTrackbarPos("rear rotation x", TRACKBAR_WINDOW) - 50) / 10
        rotation_y = (cv2.getTrackbarPos("rear rotation y", TRACKBAR_WINDOW) - 50) / 10
        rotation_z = (cv2.getTrackbarPos("rear rotation z", TRACKBAR_WINDOW) - 50) / 10

        front_aperture = cv2.getTrackbarPos("front aperture", TRACKBAR_WINDOW)
        front_delx = cv2.getTrackbarPos("front del Cx", TRACKBAR_WINDOW) - 500
        front_dely = cv2.getTrackbarPos("front del Cy", TRACKBAR_WINDOW) - 500
        front_rotation_x = (
            cv2.getTrackbarPos("front rotation x", TRACKBAR_WINDOW) - 50
        ) / 10
        front_rotation_y = (
            cv2.getTrackbarPos("front rotation y", TRACKBAR_WINDOW) - 50
        ) / 10
        front_rotation_z = (
            cv2.getTrackbarPos("front rotation z", TRACKBAR_WINDOW) - 50
        ) / 10

        frame2 = process_parameters(
            REAR_RADIUS,
            rear_delx,
            rear_dely,
            rear_aperture,
            (rotation_x, rotation_y, rotation_z),
            FRONT_RADIUS,
            front_delx,
            front_dely,
            front_aperture,
            (front_rotation_x, front_rotation_y, front_rotation_z),
            outShape,
        )

        frame2 = cv2.line(
            frame2,
            (int(frame2.shape[1] * 0.25), 0),
            (int(frame2.shape[1] * 0.25), int(frame2.shape[0])),
            (0, 180, 0),
            1,
        )
        frame2 = cv2.line(
            frame2,
            (int(frame2.shape[1] * 0.75), 0),
            (int(frame2.shape[1] * 0.75), int(frame2.shape[0])),
            (0, 180, 0),
            1,
        )
        frame2 = cv2.line(
            frame2,
            (0, int(frame2.shape[0] * 0.5)),
            (int(frame2.shape[1]), int(frame2.shape[0] * 0.5)),
            (0, 180, 0),
            1,
        )
        frame2 = cv2.line(
            frame2,
            (int(frame2.shape[1] * 0.5), 0),
            (int(frame2.shape[1] * 0.5), int(frame2.shape[0])),
            (0, 180, 0),
            1,
        )
        cv2.imshow(IMAGE_WINDOW, frame2)
        if cv2.waitKey(1) & 0xFF == 27:
            with open("./rear_cam_params.txt", "w") as f:
                f.write(str(REAR_RADIUS) + "\n")
                f.write(str(rear_aperture) + "\n")
                f.write(str(rear_delx) + "\n")
                f.write(str(rear_dely) + "\n")

            with open("./front_cam_params.txt", "w") as f:
                f.write(str(REAR_RADIUS) + "\n")
                f.write(str(front_aperture) + "\n")
                f.write(str(front_delx) + "\n")
                f.write(str(front_dely) + "\n")

            break
        frame2[:, :, :] = 0
        cv2.imshow("image", frame2)
