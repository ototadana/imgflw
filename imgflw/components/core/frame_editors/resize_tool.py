from typing import List, Tuple

from imgflw.entities import DebugImage, Image, Rect, default
from imgflw.usecase import FrameEditor
from imgflw.usecase.image_processing_util import resize


class ResizeTool(FrameEditor):
    def name(self) -> str:
        return "Resize"

    def edit(
        self,
        image: Image,
        mask_image: Image,
        faces: List[Rect],
        intermediate_steps: List[DebugImage],
        scale: float = None,
        width: int = 512,
        height: int = None,
        upscaler: str = default.UPSCALER,
        **kwargs,
    ) -> Tuple[Image, Image]:
        w, h = self.__get_size(scale, width, height, image)
        if w == image.width and h == image.height:
            return image, mask_image
        resized_image = resize(image, w, h, upscaler=upscaler)
        mask_image = resize(mask_image, w, h)

        if intermediate_steps is not None:
            intermediate_steps.append(
                DebugImage(resized_image, bottom_message=f"{image.width}x{image.height} -> {w}x{h}")
            )

        return resized_image, mask_image

    def __get_size(self, scale: float, width: int, height: int, image: Image) -> Tuple[int, int]:
        if scale is not None:
            return round(image.width * scale), round(image.height * scale)
        if width is not None and height is not None:
            return width, height
        if width is not None:
            return width, round(image.height * width / image.width)
        if height is not None:
            return round(image.width * height / image.height), height
        return image.width, image.height
