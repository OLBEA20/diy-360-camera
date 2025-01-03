from typing import Protocol


class Processor(Protocol):
    def process(self, process_id: int):
        pass
