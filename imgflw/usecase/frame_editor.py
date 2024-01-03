from abc import ABC, abstractmethod
from typing import List, Tuple

from imgflw.entities import DebugImage, Image, Rect


class FrameEditor(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def edit(
        self, image: Image, mask_image: Image, faces: List[Rect], intermediate_steps: List[DebugImage], **kwargs
    ) -> Tuple[Image, Image]:
        pass
