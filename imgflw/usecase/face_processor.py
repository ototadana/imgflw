from abc import ABC, abstractmethod
from typing import List

from imgflw.entities import DebugImage, Face


class FaceProcessor(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def process(self, face: Face, intermediate_steps: List[DebugImage], **kwargs) -> None:
        pass
