from abc import ABC, abstractmethod
from typing import List

from imgflw.entities import Image, Rect


class FaceDetector(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def detect(self, image: Image, **kwargs) -> List[Rect]:
        pass
