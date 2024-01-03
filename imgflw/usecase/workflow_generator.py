from abc import ABC, abstractmethod


class WorkflowGenerator(ABC):
    @abstractmethod
    def generate(self, request_content: str) -> str:
        pass
