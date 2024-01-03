from typing import List

from imgflw.entities import DebugImage, Face
from imgflw.usecase import MaskGenerator


class RectMaskGenerator(MaskGenerator):
    def name(self) -> str:
        return "Rect"

    def generate_mask(self, face: Face, intermediate_steps: List[DebugImage], **kwargs) -> None:
        face.mask_non_face_areas()

        if intermediate_steps is not None:
            self.add_debug_image(face, intermediate_steps)
