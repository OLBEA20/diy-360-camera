from datetime import datetime
from multiprocessing import Queue
from queue import Empty
import traceback
import cv2
import numpy as np
from diy_camera_360.fisheye_to_equirect_converter import (
    FishEyeToEquirectConverter,
)
from diy_camera_360.mean_square_error import mean_square_error

from diy_camera_360.multi_processing.task import ResultTask, Task, TaskStatus

# REAR_IMAGE_PATH = "./data/scene1/cam0_clean/test00735.jpeg"
# FRONT_IMAGE_PATH = "./data/scene1/cam1_clean/test00735.jpeg"
# REAR_IMAGE_PATH = "./data/scene4/cam0_clean/test00208.jpeg"
# FRONT_IMAGE_PATH = "./data/scene4/cam1_clean/test00208.jpeg"
REAR_IMAGE_PATH = "./data/imx477/scene1/cam0_clean/frame_00358.jpg"
FRONT_IMAGE_PATH = "./data/imx477/scene1/cam1_clean/frame_00358.jpg"


class ParametersProcessor:
    def __init__(self, task_queue: Queue, result_queue: Queue, stop_event):
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._stop_event = stop_event

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
                    result = self._process_parameters(*task.args)

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

    def _process_parameters(self, del_x, del_y, aperture, rotation_y):
        cam0_mapper = FishEyeToEquirectConverter(1510, 190, -5, 3, (0, -1, 0))
        cam1_mapper = FishEyeToEquirectConverter(1510, 189, -10, -10, (0, 0.25, 0))

        fisheye_image = cv2.imread(REAR_IMAGE_PATH)
        rear_equirect_image = cam0_mapper.fisheye_to_equirectangular(
            fisheye_image, out_shape=(960, 1920)
        )

        fisheye_image = cv2.imread(FRONT_IMAGE_PATH)
        front_equirect_image = cam1_mapper.fisheye_to_equirectangular(
            fisheye_image, out_shape=(960, 1920)
        )

        left = np.hstack(
            (rear_equirect_image[0:680, 460:500], rear_equirect_image[0:680, 1420:1460])
        )
        right = np.hstack(
            (
                front_equirect_image[0:680, 1420:1460],
                front_equirect_image[0:680, 460:500],
            )
        )

        return mean_square_error(left, right)


class ParametersProcessorFactory:
    def create_processor(self, task_queue: Queue, result_queue: Queue, stop_event):
        return ParametersProcessor(task_queue, result_queue, stop_event)
