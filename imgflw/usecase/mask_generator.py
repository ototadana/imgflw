from abc import ABC, abstractmethod
from typing import List

import numpy as np

from imgflw.entities import DebugImage, Face


class MaskGenerator(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_mask(self, face: Face, intermediate_steps: List[DebugImage], **kwargs) -> None:
        pass

    def add_debug_image(self, face: Face, intermediate_steps: List[DebugImage], top_message: str = None) -> None:
        mask = face.mask_image.array
        masked_image = self.to_masked_image(mask, face.face_image.array)
        coverage = face.calculate_mask_coverage(mask)
        intermediate_steps.append(
            DebugImage(masked_image, bottom_message=f"Coverage: {coverage * 100:.0f}%", top_message=top_message)
        )

    @staticmethod
    def to_masked_image(mask_image: np.ndarray, image: np.ndarray) -> np.ndarray:
        gray_mask = mask_image / 255.0
        return (image * gray_mask).astype("uint8")
