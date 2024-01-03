from typing import List, Tuple

import cv2
import numpy as np

from imgflw.entities import DebugImage, Image, Rect
from imgflw.usecase import FrameEditor


class RgbTool(FrameEditor):
    def name(self) -> str:
        return "RGB"

    def edit(
        self,
        image: Image,
        mask_image: Image,
        faces: List[Rect],
        intermediate_steps: List[DebugImage],
        red: float = 1.0,
        green: float = 1.0,
        blue: float = 1.0,
        **kwargs,
    ) -> Tuple[Image, Image]:
        red_channel, green_channel, blue_channel = cv2.split(image.array.copy())

        red_channel = np.clip(red_channel * red, 0, 255).astype(np.uint8)
        green_channel = np.clip(green_channel * green, 0, 255).astype(np.uint8)
        blue_channel = np.clip(blue_channel * blue, 0, 255).astype(np.uint8)

        image = Image(cv2.merge([red_channel, green_channel, blue_channel]))

        if intermediate_steps is not None:
            intermediate_steps.append(DebugImage(image, bottom_message=f"R: {red}, G: {green}, B: {blue}"))

        return image, mask_image
