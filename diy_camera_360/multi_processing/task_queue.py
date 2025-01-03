from multiprocessing import Process, Queue, Event
from queue import Empty
import signal
from typing import Dict, List, Optional
import sys

from camera360.multi_processing.processor_factory import ProcessorFactory
from camera360.multi_processing.task import I, Task, TaskStatus


class TaskQueue:
    def __init__(self, num_workers: int, processor_factory: ProcessorFactory):
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.stop_event = Event()
        self.num_workers = num_workers
        self.workers: List[Process] = []
        self.tasks: Dict[str, Task] = {}
        self._processor_factory = processor_factory

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def start(self):
        print(f"Starting {self.num_workers} workers")

        for i in range(self.num_workers):
            processor = self._processor_factory.create_processor(
                self.task_queue, self.result_queue, self.stop_event
            )
            worker = Process(target=processor.process, args=(i,))
            worker.start()
            self.workers.append(worker)

    def stop(self):
        print("Stopping workers")
        self.stop_event.set()

        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        self.workers.clear()
        print("All workers stopped")

    def enqueue(self, task_id: str, args: I) -> str:
        task = Task(id=task_id, args=args)
        self.tasks[task_id] = task
        self.task_queue.put(task)
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def process_results(self):
        try:
            while True:
                message_type, task = self.result_queue.get_nowait()

                if message_type == "status_update":
                    self.tasks[task.id] = task
                elif message_type == "task_complete":
                    self.tasks[task.id] = task
                    # print(f"Task {task.id} completed with status {task.status}")

                print(
                    f"{(len([ task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED ]) / len(self.tasks.values())) * 100:.2f}%",
                    end="\r",
                )

        except Empty:
            pass

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print("Received interrupt signal, shutting down...")
        self.stop()
        sys.exit(0)
