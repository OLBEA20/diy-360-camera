from __future__ import annotations
from datetime import datetime
from multiprocessing import Queue
from queue import Empty
import random
from statistics import mean
import time
import traceback
from typing import Callable
import cv2
import numpy as np
from diy_camera_360.fisheye_to_equirect_converter import (
    CameraParameters,
    FishEyeToEquirectConverter,
)
from diy_camera_360.horizontal_fade import fade_horizontal_edges
from diy_camera_360.mean_square_error import mean_square_error
from diy_camera_360.multi_processing.task import ResultTask, Task, TaskStatus
from diy_camera_360.multi_processing.task_queue import TaskQueue
from diy_camera_360.swap_image_halves import swap_image_halves

REAR_IMAGE_PATH = "./data/imx477/scene10/cam0_clean/frame_00200.jpg"
# REAR_IMAGE_PATH = "./data/gopro/scene2/cam0_clean/frame_00665.jpg"
CAM0_FISHEYE_IMAGE = cv2.imread(REAR_IMAGE_PATH)

FRONT_IMAGE_PATH = "./data/imx477/scene10/cam1_clean/frame_00200.jpg"
# FRONT_IMAGE_PATH = "./data/gopro/scene2/cam1_clean/frame_00665.jpg"
CAM1_FISHEYE_IMAGE = cv2.imread(FRONT_IMAGE_PATH)

REAR_RADIUS = 1430
FRONT_RADIUS = 1430

# REAR_RADIUS = 352
# FRONT_RADIUS = 352

OUT_WIDTH = 1920
OUT_HEIGHT = 960
# OUT_WIDTH = 1408
# OUT_HEIGHT = 704

OVERLAP_PIXELS = int(4 / 360 * OUT_WIDTH)


def process_parameters(
    cam0_parameters: CameraParameters, cam1_parameters: CameraParameters
):
    cam0_mapper = FishEyeToEquirectConverter(REAR_RADIUS, cam0_parameters)
    cam1_mapper = FishEyeToEquirectConverter(FRONT_RADIUS, cam1_parameters)

    rear_equirect_image = cam0_mapper.fisheye_to_equirectangular(
        CAM0_FISHEYE_IMAGE, out_shape=(OUT_HEIGHT, OUT_WIDTH)
    )
    front_equirect_image = cam1_mapper.fisheye_to_equirectangular(
        CAM1_FISHEYE_IMAGE, out_shape=(OUT_HEIGHT, OUT_WIDTH)
    )

    left_left = 450
    left_right = 510

    right_left = 1410
    right_right = 1470

    vertical_height = int(0.8 * OUT_HEIGHT)

    # left = np.hstack(
    #    (
    #        rear_equirect_image[0:vertical_height, left_left:left_right],
    #        rear_equirect_image[0:vertical_height, right_left:right_right],
    #    )
    # )
    # right = np.hstack(
    #    (
    #        front_equirect_image[0:vertical_height, right_left:right_right],
    #        front_equirect_image[0:vertical_height, left_left:left_right],
    #    )
    # )
    left = np.hstack(
        (
            rear_equirect_image[0:vertical_height, 406:446],
            rear_equirect_image[0:vertical_height, 1473:1513],
        )
    )
    right = np.hstack(
        (
            front_equirect_image[0:vertical_height, 1366:1406],
            front_equirect_image[0:vertical_height, 513:553],
        )
    )

    # cv2.imwrite("overlap.png", np.vstack((left, right)))

    return mean_square_error(left, right)


class ParameterOptimizer:
    def __init__(
        self,
        build_params: Callable[[CameraParameters, int], CameraParameters],
        steps: list[float],
    ):
        self._build_params = build_params
        self._steps = steps
        self.iterations = 0

    def optimize(
        self,
        reference_score: float,
        cam0_params: CameraParameters,
        cam1_params: CameraParameters,
    ):
        sign = None
        for step in self._steps:
            positive_cam0_param, positive_cam1_param = self._build_params(
                cam0_params, cam1_params, step
            )
            negative_cam0_param, negative_cam1_param = self._build_params(
                cam0_params, cam1_params, -step
            )

            positive_score = process_parameters(
                positive_cam0_param, positive_cam1_param
            )
            negative_score = process_parameters(
                negative_cam0_param, negative_cam1_param
            )
            self.iterations += 2

            if positive_score > reference_score and negative_score > reference_score:
                continue
            else:
                sign = 1 if positive_score < negative_score else -1
                reference_score = (
                    positive_score
                    if positive_score < negative_score
                    else negative_score
                )
                cam0_params, cam1_params = (
                    (positive_cam0_param, positive_cam1_param)
                    if positive_score < negative_score
                    else (negative_cam0_param, negative_cam1_param)
                )
                break

        if sign is None:
            return cam0_params, cam1_params, reference_score

        new_score = reference_score
        best_score = reference_score
        new_cam0_params, new_cam1_params = cam0_params, cam1_params

        for step in self._steps:
            while new_score <= best_score:
                best_score = new_score
                cam0_best_params, cam1_best_params = new_cam0_params, new_cam1_params

                new_cam0_params, new_cam1_params = self._build_params(
                    cam0_best_params, cam1_best_params, sign * step
                )
                new_score = process_parameters(new_cam0_params, new_cam1_params)
                self.iterations += 1

            new_score = best_score
            new_cam0_params, new_cam1_params = cam0_best_params, cam1_best_params

        return cam0_best_params, cam1_best_params, best_score


class OneStepParameterOptimizer:
    def __init__(
        self,
        build_params: Callable[
            [CameraParameters, CameraParameters, int], CameraParameters
        ],
        steps: float,
    ):
        self._build_params = build_params
        self._steps = steps
        self.iterations = 0

    def optimize(
        self,
        reference_score: float,
        cam0_params: CameraParameters,
        cam1_params: CameraParameters,
    ):
        for step in self._steps:
            positive_cam0_param, positive_cam1_param = self._build_params(
                cam0_params, cam1_params, step
            )
            positive_score = process_parameters(
                positive_cam0_param, positive_cam1_param
            )
            self.iterations += 1

            if positive_score < reference_score:
                return positive_cam0_param, positive_cam1_param, positive_score

            negative_cam0_param, negative_cam1_param = self._build_params(
                cam0_params, cam1_params, step
            )

            negative_score = process_parameters(
                negative_cam0_param, negative_cam1_param
            )
            self.iterations += 1

            if negative_score < reference_score:
                return negative_cam0_param, negative_cam1_param, negative_score

        return cam0_params, cam1_params, reference_score


def print_calibration_image(
    cam0_params: CameraParameters,
    cam1_params: CameraParameters,
    destination_file_path: str,
):
    cam0_mapper = FishEyeToEquirectConverter(REAR_RADIUS, cam0_params)
    cam1_mapper = FishEyeToEquirectConverter(FRONT_RADIUS, cam1_params)

    fisheye_image = cv2.imread(REAR_IMAGE_PATH)
    rear_equirect_image = fade_horizontal_edges(
        cam0_mapper.fisheye_to_equirectangular(fisheye_image, (3040, 6080)),
        3.75,
        80,
        280,
    )

    fisheye_image = cv2.imread(FRONT_IMAGE_PATH)
    front_equirect_image = fade_horizontal_edges(
        cam1_mapper.fisheye_to_equirectangular(fisheye_image, (3040, 6080)),
        3.75,
        100,
        260,
    )

    merged = np.clip(
        rear_equirect_image + swap_image_halves(front_equirect_image), 0, 255
    ).astype(np.uint8)

    cv2.imwrite(destination_file_path, merged)


class ParametersProcessorFactory:
    def create_processor(self, task_queue: Queue, result_queue: Queue, stop_event):
        return ParametersProcessor(task_queue, result_queue, stop_event)


class ParametersProcessor:
    def __init__(self, task_queue: Queue, result_queue: Queue, stop_event):
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._stop_event = stop_event
        self._optimizers = [
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params.with_aperture(cam0_params.aperture + delta),
                    cam1_params,
                ),
                [1, 0.5, 0.1],
            ),
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params.with_del_y(cam0_params.del_y + delta),
                    cam1_params,
                ),
                [1],
            ),
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params.with_del_x(cam0_params.del_x + delta),
                    cam1_params,
                ),
                [3, 2, 1],
            ),
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params.with_rotation(
                        (
                            cam0_params.rotation[0],
                            cam0_params.rotation[1] + delta,
                            cam0_params.rotation[2],
                        )
                    ),
                    cam1_params,
                ),
                [0.25, 0.1, 0.05],
            ),
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params.with_rotation(
                        (
                            cam0_params.rotation[0] + delta,
                            cam0_params.rotation[1],
                            cam0_params.rotation[2],
                        )
                    ),
                    cam1_params,
                ),
                [0.25, 0.1, 0.05],
            ),
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params.with_rotation(
                        (
                            cam0_params.rotation[0],
                            cam0_params.rotation[1],
                            cam0_params.rotation[2] + delta,
                        )
                    ),
                    cam1_params,
                ),
                [0.25, 0.1, 0.05],
            ),
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params,
                    cam1_params.with_aperture(cam1_params.aperture + delta),
                ),
                [1, 0.5, 0.1],
            ),
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params,
                    cam1_params.with_del_y(cam1_params.del_y + delta),
                ),
                [1],
            ),
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params,
                    cam1_params.with_del_x(cam1_params.del_x + delta),
                ),
                [3, 2, 1],
            ),
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params,
                    cam1_params.with_rotation(
                        (
                            cam1_params.rotation[0],
                            cam1_params.rotation[1] + delta,
                            cam1_params.rotation[2],
                        )
                    ),
                ),
                [0.25, 0.1, 0.05],
            ),
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params,
                    cam1_params.with_rotation(
                        (
                            cam1_params.rotation[0] + delta,
                            cam1_params.rotation[1],
                            cam1_params.rotation[2],
                        )
                    ),
                ),
                [0.25, 0.1, 0.05],
            ),
            ParameterOptimizer(
                lambda cam0_params, cam1_params, delta: (
                    cam0_params,
                    cam1_params.with_rotation(
                        (
                            cam1_params.rotation[0],
                            cam1_params.rotation[1],
                            cam1_params.rotation[2] + delta,
                        )
                    ),
                ),
                [0.25, 0.1, 0.05],
            ),
        ]

    def process(self, process_id):
        print(f"Worker {process_id} started")
        while not self._stop_event.is_set():
            try:
                task: Task = self._task_queue.get(timeout=1)
                result_task = ResultTask(id=task.id)

                result_task.status = TaskStatus.RUNNING
                result_task.started_at = datetime.now()
                self._result_queue.put(("status_update", result_task))
                try:
                    result = self._process_parameters(task.id, *task.args)

                    result_task.status = TaskStatus.COMPLETED
                    result_task.completed_at = datetime.now()
                    result_task.func = None
                    result_task.result = (result, task.args)

                except Exception as e:
                    result_task.status = TaskStatus.FAILED
                    result_task.error = f"{str(e)}\n{traceback.format_exc()}"
                    result_task.completed_at = datetime.now()

                self._result_queue.put(("task_complete", result_task))

            except Empty:
                continue
            except Exception as e:
                print(f"Worker {process_id} error: {str(e)}")

        print(f"Worker {process_id} stopped")

    def _process_parameters(
        self,
        task_id,
        initial_cam0_params: CameraParameters,
        initial_cam1_params: CameraParameters,
    ):
        cam0_param, cam1_param = initial_cam0_params, initial_cam1_params

        last_score = process_parameters(cam0_param, cam1_param)
        best_score = last_score + 1

        for optimizer in self._optimizers:
            optimizer.iterations = 0

        with open(f"data/logs/{task_id}.txt", "w") as log_file:
            while last_score < best_score:
                best_score = last_score
                for optimizer in self._optimizers:
                    cam0_param, cam1_param, last_score = optimizer.optimize(
                        last_score,
                        cam0_param,
                        cam1_param,
                    )
                    log_file.write(f"{last_score} {cam0_param} {cam1_param}\n")

        print_calibration_image(
            cam0_param,
            cam1_param,
            f"./data/calibration/{int(last_score)}_{task_id}.jpg",
        )

        return (
            best_score,
            sum([o.iterations for o in self._optimizers]),
            cam0_param,
            cam1_param,
        )


if __name__ == "__main__":
    queue = TaskQueue(10, ParametersProcessorFactory())
    queue.start()
    try:
        task_id = 0
        for iteration in range(20):
            initial_aperture = random.randint(208, 218)
            initial_cam0_param = CameraParameters(
                initial_aperture,
                random.randint(-10, 10),
                random.randint(-10, -10),
                (
                    round(random.uniform(-2.5, 2.5), 2),
                    round(random.uniform(-2.5, 2.5), 2),
                    round(random.uniform(-2.5, 2.5), 2),
                ),
            )
            initial_cam1_param = CameraParameters(
                initial_aperture,
                random.randint(-10, 10),
                random.randint(-10, 10),
                (
                    round(random.uniform(-2.5, 2.5), 2),
                    round(random.uniform(-2.5, 2.5), 2),
                    round(random.uniform(-2.5, 2.5), 2),
                ),
            )
            queue.enqueue(task_id, (initial_cam0_param, initial_cam1_param))
            task_id += 1
        while any(
            task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            for task in queue.tasks.values()
        ):
            queue.process_results()
            time.sleep(1)

        queue.process_results()

        failed_task = [
            task for task in queue.tasks.values() if task.status == TaskStatus.FAILED
        ]
        for task in failed_task:
            print(task.error)

        successful_tasks = [
            task for task in queue.tasks.values() if task.status == TaskStatus.COMPLETED
        ]
        average_duration = mean(
            (task.completed_at - task.started_at).total_seconds()
            for task in successful_tasks
        )

        task_with_result = [
            task for task in successful_tasks if task.result[0] is not None
        ]

        iterations_sum = 0
        with open("parameters_result.txt", "w") as file:
            for task in sorted(task_with_result, key=lambda x: x.result[0][0]):
                score = f"{task.result[0][0]:.2f}"
                iterations = f"{task.result[0][1]}"
                iterations_sum += int(iterations)
                cam0 = f"{task.result[0][2]}"
                cam1 = f"{task.result[0][3]}"

                file.write(f"{score} {iterations} {cam0} {cam1}")
                file.write("\n")

        print(20 * "-")
        print(f"Successful Tasks: {len(successful_tasks)}")
        print(f"Failed Tasks: {len(failed_task)}")
        print(f"Mean duration: {average_duration:.2f}s")
        print(f"Iterations: {iterations_sum}")
        print(20 * "-")

    finally:
        queue.stop()
