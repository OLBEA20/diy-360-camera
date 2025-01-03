from datetime import datetime
from queue import Empty
import traceback
import cv2
from multiprocessing import Queue

import numpy as np
from camera360.blend_image_with_mask import blend_images_with_mask
from camera360.fisheye_to_equirect_converter import CameraParameters, FishEyeToEquirectConverter
from camera360.horizontal_fade import fade_horizontal_edges
from camera360.mean_square_error import mean_square_error
from camera360.multi_processing.task import ResultTask, Task, TaskStatus
from camera360.swap_image_halves import swap_image_halves


class ImageProcessor:
    def __init__(self, task_queue: Queue, result_queue: Queue, stop_event):
        #cam0_params = CameraParameters(aperture=187.5, del_x=-9, del_y=-24, rotation=(1.47, -0.82, 2.05)) 
        #cam1_params = CameraParameters(aperture=183.9, del_x=4, del_y=-2, rotation=(4.46, -0.8, -0.09))
        #cam0_params = CameraParameters(aperture=185.4, del_x=45, del_y=-25, rotation=(2.44, -1.05, -2.36)) 
        #cam1_params = CameraParameters(aperture=185.9, del_x=-64, del_y=-13, rotation=(3.31, -0.17, 3.08))
        cam0_params = CameraParameters(aperture=187.4, del_x=-24, del_y=-30, rotation=(0.49, 0.18000000000000005, 0.6)) 
        cam1_params = CameraParameters(aperture=184.4, del_x=11, del_y=-5, rotation=(3.97, -0.85, -0.5))

        self._cam0_mapper = FishEyeToEquirectConverter(1510, cam0_params)
        self._cam1_mapper = FishEyeToEquirectConverter(1510, cam1_params)

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
                    mean_square_error = self._process_image(*task.args)

                    result_task.result = (task.args[0].split(".")[0][-5:],mean_square_error)
                    result_task.status = TaskStatus.COMPLETED
                    result_task.completed_at = datetime.now()
                    result_task.func = None

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

    def _process_image(self, rear_image_path, front_image_path, output_image_path):
        fisheye_image = cv2.imread(rear_image_path)
        rear_equirect_image = self._cam0_mapper.fisheye_to_equirectangular(fisheye_image, (3040, 6080))

        fisheye_image = cv2.imread(front_image_path)
        front_equirect_image = self._cam1_mapper.fisheye_to_equirectangular(fisheye_image, (3040, 6080))

        left_left = 470
        left_right = 490

        right_left = 1430
        right_right = 1450

        vertical_height = 530

        left = np.hstack(
            (
                rear_equirect_image[0:vertical_height, left_left:left_right],
                rear_equirect_image[0:vertical_height, right_left:right_right],
            )
        )
        right = np.hstack(
            (
                front_equirect_image[0:vertical_height, right_left:right_right],
                front_equirect_image[0:vertical_height, left_left:left_right],
            )
        )

        merged = np.clip(fade_horizontal_edges(rear_equirect_image) + swap_image_halves(fade_horizontal_edges(front_equirect_image)), 0, 255).astype(np.uint8)

        cv2.imwrite(output_image_path, merged)

        return mean_square_error(left, right)


class ImageProcessorFactory:
    def create_processor(self, task_queue: Queue, result_queue: Queue, stop_event):
        return ImageProcessor(task_queue, result_queue, stop_event)
