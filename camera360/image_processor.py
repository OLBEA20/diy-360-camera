from datetime import datetime
from queue import Empty
import traceback
import cv2
from multiprocessing import Queue
from camera360.fisheye_to_equirect_converter import FishEyeToEquirectConverter
from camera360.horizontal_fade import fade_horizontal_edges
from camera360.multi_processing.task import ResultTask, Task, TaskStatus
from main import (
    blend_images_with_mask,
    swap_image_halves,
)


class ImageProcessor:
    def __init__(self, task_queue: Queue, result_queue: Queue, stop_event):
        self._cam0_mapper = FishEyeToEquirectConverter(1050, 199, 4, -10, -2.25)
        self._cam1_mapper = FishEyeToEquirectConverter(1050, 197, 4, -10, 0.5)
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
                    self._process_image(*task.args)

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
        rear_equirect_image = fade_horizontal_edges(
            self._cam0_mapper.fisheye_to_equirectangular(fisheye_image, (1944, 3888))
        )

        fisheye_image = cv2.imread(front_image_path)
        front_equirect_image = fade_horizontal_edges(
            self._cam1_mapper.fisheye_to_equirectangular(fisheye_image, (1944, 3888))
        )

        def merge_equirect_images(rear, front):
            return blend_images_with_mask(rear, swap_image_halves(front), 1, 3)

        merged = merge_equirect_images(rear_equirect_image, front_equirect_image)

        cv2.imwrite(output_image_path, merged)


class ImageProcessorFactory:
    def create_processor(self, task_queue: Queue, result_queue: Queue, stop_event):
        return ImageProcessor(task_queue, result_queue, stop_event)
