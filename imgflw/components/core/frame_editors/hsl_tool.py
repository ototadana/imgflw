from typing import List, Tuple

import cv2
import numpy as np

from imgflw.entities import DebugImage, Image, Rect
from imgflw.usecase import FrameEditor


class HslTool(FrameEditor):
    def name(self) -> str:
        return "HSL"

    def edit(
        self,
        image: Image,
        mask_image: Image,
        faces: List[Rect],
        intermediate_steps: List[DebugImage],
        hue: int = 0,
        saturation: float = 1.0,
        lightness: float = 1.0,
        **kwargs,
    ) -> Tuple[Image, Image]:
        img = cv2.cvtColor(image.array.copy(), cv2.COLOR_RGB2HLS)
        hue_adjust = hue / 360
        hv, lv, sv = cv2.split(img)

        hv = np.mod(hv + hue_adjust * 180, 180).astype(np.uint8)
        sv = np.clip(sv * saturation, 0, 255).astype(np.uint8)
        lv = np.clip(lv * lightness, 0, 255).astype(np.uint8)

        adjusted_img = cv2.cvtColor(cv2.merge([hv, lv, sv]), cv2.COLOR_HLS2RGB)
        image = Image(adjusted_img)

        if intermediate_steps is not None:
            intermediate_steps.append(DebugImage(image, bottom_message=f"H: {hue}, S: {saturation}, L: {lightness}"))

        return image, mask_image
