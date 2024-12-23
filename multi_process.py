from statistics import mean
import subprocess
from tempfile import TemporaryDirectory
import time
import os
from camera360.image_processor import ImageProcessorFactory
from camera360.multi_processing.task import TaskStatus
from camera360.multi_processing.task_queue import TaskQueue

scene_name = "scene4"

REAR_CAMERA_IMAGES_PATH = f"./data/{scene_name}/cam0_clean"
FRONT_CAMERA_IMAGES_PATH = f"./data/{scene_name}/cam1_clean"

# Example usage
if __name__ == "__main__":
    queue = TaskQueue(10, ImageProcessorFactory())
    queue.start()

    with TemporaryDirectory() as merged_temp_dir:
        try:
            for rear_image_file_name in os.listdir(REAR_CAMERA_IMAGES_PATH):
                front_image_path = os.path.join(
                    FRONT_CAMERA_IMAGES_PATH, rear_image_file_name
                )
                if (
                    not rear_image_file_name.endswith(".jpeg")
                    or not front_image_path.endswith(".jpeg")
                    and os.path.exists(front_image_path)
                ):
                    continue

                task_id = f"task_{rear_image_file_name}"
                queue.enqueue(
                    task_id,
                    (
                        os.path.join(REAR_CAMERA_IMAGES_PATH, rear_image_file_name),
                        front_image_path,
                        os.path.join(merged_temp_dir, rear_image_file_name),
                    ),
                )

            while any(
                task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                for task in queue.tasks.values()
            ):
                queue.process_results()
                time.sleep(1)

            queue.process_results()

            failed_task = [
                task
                for task in queue.tasks.values()
                if task.status == TaskStatus.FAILED
            ]
            for task in failed_task:
                print(task.error)

            successful_tasks = [
                task
                for task in queue.tasks.values()
                if task.status == TaskStatus.COMPLETED
            ]

            average_duration = mean(
                (task.completed_at - task.started_at).total_seconds()
                for task in successful_tasks
            )

            print(20 * "-")
            print(f"Successful Tasks: {len(successful_tasks)}")
            print(f"Failed Tasks: {len(failed_task)}")
            print(f"Mean duration: {average_duration:.2f}s")
            print(20 * "-")

        finally:
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
            f"{scene_name}.mp4",
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
