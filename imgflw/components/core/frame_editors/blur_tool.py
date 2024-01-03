from typing import List, Tuple

from PIL import ImageFilter

from imgflw.entities import DebugImage, Image, Rect
from imgflw.usecase import FrameEditor


class BlurTool(FrameEditor):
    def name(self) -> str:
        return "Blur"

    def edit(
        self,
        image: Image,
        mask_image: Image,
        faces: List[Rect],
        intermediate_steps: List[DebugImage],
        radius: float = 2,
        **kwargs,
    ) -> Tuple[Image, Image]:
        image = Image(image.pil_image.filter(ImageFilter.GaussianBlur(radius)))

        if intermediate_steps is not None:
            intermediate_steps.append(DebugImage(image, bottom_message=f"Radius: {radius}"))

        return image, mask_image
