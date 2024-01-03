from typing import List, Tuple

from PIL import ImageEnhance

from imgflw.entities import DebugImage, Image, Rect
from imgflw.usecase import FrameEditor


class ContrastTool(FrameEditor):
    def name(self) -> str:
        return "Contrast"

    def edit(
        self,
        image: Image,
        mask_image: Image,
        faces: List[Rect],
        intermediate_steps: List[DebugImage],
        contrast: float = 1.0,
        **kwargs,
    ) -> Tuple[Image, Image]:
        enhancer = ImageEnhance.Contrast(image.pil_image)
        img_contrasted = Image(enhancer.enhance(contrast))

        if intermediate_steps is not None:
            intermediate_steps.append(DebugImage(img_contrasted, bottom_message=f"Contrast: {contrast}"))

        return img_contrasted, mask_image
