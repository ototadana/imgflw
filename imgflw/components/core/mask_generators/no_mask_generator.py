from typing import List

import numpy as np

from imgflw.entities import DebugImage, Face, Image
from imgflw.usecase import MaskGenerator


class NoMaskGenerator(MaskGenerator):
    def name(self) -> str:
        return "NoMask"

    def generate_mask(
        self,
        face: Face,
        intermediate_steps: List[DebugImage],
        use_minimal_area: bool = False,
        **kwargs,
    ) -> None:
        face.mask_image = Image(np.ones((face.height, face.width, 3), np.uint8) * 255)
        if use_minimal_area:
            face.mask_non_face_areas()
        if intermediate_steps is not None:
            self.add_debug_image(face, intermediate_steps)
