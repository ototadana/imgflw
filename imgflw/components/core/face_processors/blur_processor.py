from typing import List

from PIL import ImageFilter

from imgflw.entities import DebugImage, Face, Image
from imgflw.usecase import FaceProcessor


class BlurProcessor(FaceProcessor):
    def name(self) -> str:
        return "Blur"

    def process(self, face: Face, intermediate_steps: List[DebugImage], radius: float = 20, **kwargs) -> None:
        face.face_image = Image(face.face_image.pil_image.filter(ImageFilter.GaussianBlur(radius)))
        if intermediate_steps is not None:
            intermediate_steps.append(DebugImage(face.face_image, bottom_message=f"Radius: {radius}"))
