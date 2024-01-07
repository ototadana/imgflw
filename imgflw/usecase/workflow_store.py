from abc import ABC, abstractmethod
from typing import List


class WorkflowStore(ABC):
    @abstractmethod
    def save(self, request: str, workflow: str) -> None:
        pass

    @abstractmethod
    def get(self, request: str) -> str:
        pass

    @abstractmethod
    def find(self, request: str) -> List[str]:
        pass
