from typing import List

import cv2
import numpy as np

from imgflw.entities import DebugImage, Face, Image
from imgflw.usecase import MaskGenerator


class VignetteMaskGenerator(MaskGenerator):
    def name(self) -> str:
        return "Vignette"

    def generate_mask(
        self,
        face: Face,
        intermediate_steps: List[DebugImage],
        use_minimal_area: bool = False,
        sigma: float = -1,
        keep_safe_area: bool = False,
        **kwargs
    ) -> None:
        (left, top, right, bottom) = face.face_area_on_face_image
        w, h = right - left, bottom - top
        face_image = face.face_image.array
        mask = np.zeros((face_image.shape[0], face_image.shape[1]), dtype=np.uint8)
        if use_minimal_area:
            sigma = 120 if sigma == -1 else sigma
            mask[top : top + h, left : left + w] = 255
        else:
            sigma = 180 if sigma == -1 else sigma
            h, w = face_image.shape[0], face_image.shape[1]
            mask[:, :] = 255

        Y = np.linspace(0, h, h, endpoint=False)
        X = np.linspace(0, w, w, endpoint=False)
        Y, X = np.meshgrid(Y, X)
        Y -= h / 2
        X -= w / 2

        gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        gaussian_mask = np.uint8(255 * gaussian.T)
        if use_minimal_area:
            mask[top : top + h, left : left + w] = gaussian_mask
        else:
            mask[:, :] = gaussian_mask

        if keep_safe_area:
            mask = cv2.ellipse(mask, ((left + right) // 2, (top + bottom) // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        face.mask_image = Image(mask)

        if intermediate_steps is not None:
            self.add_debug_image(face, intermediate_steps)
