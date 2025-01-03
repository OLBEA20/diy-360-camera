from multiprocessing import Queue
from typing import Protocol


class ProcessorFactory(Protocol):
    def create_processor(self, task_queue: Queue, result_queue: Queue, stop_event):
        pass
