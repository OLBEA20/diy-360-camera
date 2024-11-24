from multiprocessing import Process, Queue, Event, cpu_count
from statistics import mean
import subprocess
from tempfile import TemporaryDirectory
from omnicv import fisheyeImgConv
from queue import Empty
from typing import Callable, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime
import time
import logging
import traceback
import signal
import sys
from enum import Enum
import os
import cv2

from main import (
    FRONT_CAMERA_IMAGES_PATH,
    REAR_CAMERA_IMAGES_PATH,
    FishEyeToEquirectConverter,
    blend_images_with_mask,
    remove_outer_noise,
    swap_image_halves,
)

T = TypeVar("T")  # Type for task result
I = TypeVar("I")  # Type for task input


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task(Generic[I, T]):
    """Task container with metadata"""

    id: str
    args: I
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[T] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ResultTask(Generic[I, T]):
    id: str
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskQueue:
    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize task queue system.

        Args:
            num_workers: Number of worker processes. Defaults to CPU count.
        """
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.stop_event = Event()
        self.num_workers = num_workers or cpu_count()
        self.workers: List[Process] = []
        self.tasks: Dict[str, Task] = {}

        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)


    def start(self):
        """Start the worker processes"""
        print(f"Starting {self.num_workers} workers")

        for i in range(self.num_workers):
            image_convert = ImageProcessor(self.task_queue, self.result_queue, self.stop_event)
            worker = Process(target=image_convert.process, args=(i,))
            worker.start()
            self.workers.append(worker)

    def stop(self):
        """Stop all workers and clean up"""
        print("Stopping workers")
        self.stop_event.set()

        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        self.workers.clear()
        print("All workers stopped")

    def enqueue(self, task_id: str, args: I) -> str:
        """
        Add a task to the queue

        Args:
            task_id: Unique identifier for the task
            func: Function to execute
            args: Arguments for the function

        Returns:
            task_id: The task identifier
        """
        task = Task(id=task_id, args=args)
        self.tasks[task_id] = task
        self.task_queue.put(task)
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get the current status of a task"""
        return self.tasks.get(task_id)

    def process_results(self):
        """Process any completed tasks in the result queue"""
        try:
            while True:
                message_type, task = self.result_queue.get_nowait()

                if message_type == "status_update":
                    self.tasks[task.id] = task
                elif message_type == "task_complete":
                    self.tasks[task.id] = task
                    #print(f"Task {task.id} completed with status {task.status}")

        except Empty:
            pass

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print("Received interrupt signal, shutting down...")
        self.stop()
        sys.exit(0)


class ImageProcessor:
    def __init__(self, task_queue: Queue, result_queue: Queue, stop_event):
        self._cam0_mapper = FishEyeToEquirectConverter(
            fisheyeImgConv("./cam0_fisheye_params.txt")
        )
        self._cam1_mapper = FishEyeToEquirectConverter(
            fisheyeImgConv("./cam1_fisheye_params.txt")
        )
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._stop_event = stop_event

    def process_image(self, rear_image_path, front_image_path, output_image_path):
        fisheye_image = cv2.imread(rear_image_path)
        rear_equirect_image = self._cam0_mapper.fisheye_to_equirectangular(
            remove_outer_noise(fisheye_image, 1205)
        )

        fisheye_image = cv2.imread(front_image_path)
        front_equirect_image = swap_image_halves(
            self._cam1_mapper.fisheye_to_equirectangular(
                remove_outer_noise(fisheye_image, 1205)
            )
        )

        merged = blend_images_with_mask(
            rear_equirect_image, front_equirect_image, 0.5, 1
        )

        cv2.imwrite(output_image_path, merged)
    
    def process(self, process_id):
        print(f"Worker {process_id} started")

        while not self._stop_event.is_set():
            try:
                # Try to get a task with timeout to allow checking stop_event
                task: Task = self._task_queue.get(timeout=1)
                result_task = ResultTask(id=task.id)

                # Update task status
                result_task.status = TaskStatus.RUNNING
                result_task.started_at = datetime.now()
                self._result_queue.put(("status_update", result_task))

                try:
                    self.process_image(*task.args)

                    # Update task with result
                    result_task.status = TaskStatus.COMPLETED
                    result_task.completed_at = datetime.now()
                    result_task.func = None

                except Exception as e:
                    # Handle task failure
                    result_task.status = TaskStatus.FAILED
                    result_task.error = f"{str(e)}\n{traceback.format_exc()}"
                    result_task.completed_at = datetime.now()

                # Send result back
                self._result_queue.put(("task_complete", result_task))

            except Empty:
                continue
            except Exception as e:
                print(f"Worker {process_id} error: {str(e)}")

        print(f"Worker {process_id} stopped")


# Example usage
if __name__ == "__main__":
    queue = TaskQueue(12)
    queue.start()

    with TemporaryDirectory() as merged_temp_dir:
        try:
            # Enqueue some tasks
            for rear_image_path, front_image_path in zip(
                os.listdir(REAR_CAMERA_IMAGES_PATH),
                os.listdir(FRONT_CAMERA_IMAGES_PATH),
            ):
                if not rear_image_path.endswith(
                    ".jpeg"
                ) or not front_image_path.endswith(".jpeg"):
                    continue

                task_id = f"task_{rear_image_path}"
                queue.enqueue(
                    task_id,
                    (
                        os.path.join(REAR_CAMERA_IMAGES_PATH, rear_image_path),
                        os.path.join(FRONT_CAMERA_IMAGES_PATH, front_image_path),
                        os.path.join(merged_temp_dir, rear_image_path),
                    ),
                )

            # Monitor tasks until all complete
            while any(
                task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                for task in queue.tasks.values()
            ):
                queue.process_results()
                time.sleep(1)

            # Final results processing
            queue.process_results()

            # Print results
            failed_task = [task for task in queue.tasks.values() if task.status == TaskStatus.FAILED]
            successful_tasks = [task for task in queue.tasks.values() if task.status == TaskStatus.COMPLETED]
            average_duration = mean((task.completed_at - task.started_at).total_seconds() for task in successful_tasks)

            print(20*"-")
            print(f"Successful Tasks: {len(successful_tasks)}")
            print(f"Failed Tasks: {len(failed_task)}")
            print(f"Mean duration: {average_duration:.2f}s")
            print(20*"-")


        finally:
            # Clean up
            queue.stop()

        print("Merge image into video...")
        command = [
            "ffmpeg",
            "-framerate",
            "15",
            "-i",
            os.path.join(merged_temp_dir, "test%05d.jpeg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "merged.mp4",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("Wating for completion")
        process.wait()
        if process.returncode != 0:
            print(process.returncode)
            print(process.stdout.read())
            print(process.stderr.read())
