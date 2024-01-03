from typing import List

from imgflw.entities import DebugImage, Face
from imgflw.usecase import FaceProcessor


class NoOpProcessor(FaceProcessor):
    def name(self) -> str:
        return "NoOp"

    def process(self, face: Face, intermediate_steps: List[DebugImage], **kwargs) -> None:
        if intermediate_steps is not None:
            intermediate_steps.append(DebugImage(face.face_image, bottom_message="NoOp"))
