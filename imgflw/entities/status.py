from typing import List

from PIL.Image import Image as PILImage

from imgflw.entities import DebugImage


class Status:
    def __init__(self):
        self.canceled = False
        self.intermediate_steps: List[DebugImage] = None

    def get_images(self, size: int) -> List[PILImage]:
        if self.intermediate_steps is None:
            return []
        return [step.get_image(size).pil_image for step in self.intermediate_steps]
