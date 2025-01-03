from statistics import mean
import time
from camera360.multi_processing.task import TaskStatus
from camera360.multi_processing.task_queue import TaskQueue
from camera360.parameters_processor import ParametersProcessorFactory

if __name__ == "__main__":
    queue = TaskQueue(10, ParametersProcessorFactory())
    queue.start()
    try:
        task_id = 0
        for delX in range(-10, 10):
            for delY in range(-10, 10):
                for aperture in range(187, 192):
                    for rotation_y in range(-12, 12, 1):
                        queue.enqueue(
                            task_id,
                            (delX, delY, aperture, rotation_y / 4),
                        )
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

        print(20 * "-")
        print(f"Successful Tasks: {len(successful_tasks)}")
        print(f"Failed Tasks: {len(failed_task)}")
        print(f"Mean duration: {average_duration:.2f}s")
        print(20 * "-")

        task_with_result = [
            task for task in successful_tasks if task.result[0] is not None
        ]

        with open("parameters_result.txt", "w") as file:
            for task in sorted(task_with_result, key=lambda x: x.result[0]):
                file.write(f"{task.result[0]:.2f} {task.result[1]}")
                file.write("\n")

    finally:
        queue.stop()
