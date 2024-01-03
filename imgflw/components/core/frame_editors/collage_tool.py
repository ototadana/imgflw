from typing import Dict, List, Tuple

import numpy as np

from imgflw.components.core.frame_editors.crop_tool import CropTool
from imgflw.entities import DebugImage, Image, Rect
from imgflw.usecase import FrameEditor


class CollageTool(FrameEditor):
    def name(self) -> str:
        return "Collage"

    def edit(
        self,
        frame: Image,
        image_mask: Image,
        faces: List[Rect],
        intermediate_steps: List[DebugImage],
        crop_params: List[Dict[str, str]] = [],
        **kwargs,
    ) -> Tuple[Image, Image]:
        crop_tool = CropTool()
        min_top = frame.height
        max_bottom = 0
        min_margin = frame.width

        for crop_param in crop_params:
            crop_param["margin"] = (
                2.0 if crop_param.get("margin", None) is None else max(float(crop_param["margin"]), 1.2)
            )
            _, _, rect = crop_tool.edit_(frame, image_mask, faces, intermediate_steps, dry_run=True, **crop_param)
            if rect is not None:
                min_top = min(min_top, rect.top)
                max_bottom = max(max_bottom, rect.bottom)
                margin = (rect.width - round(rect.width / crop_param["margin"])) // 2
                min_margin = min(min_margin, margin)

        cropped_images = []
        for crop_param in crop_params:
            crop_param["top"] = min_top
            crop_param["bottom"] = max_bottom
            cropped, cropped_mask, rect = crop_tool.edit_(frame, image_mask, faces, intermediate_steps, **crop_param)
            if rect is not None:
                cropped_images.append((cropped, cropped_mask, rect))

        if len(cropped_images) == 0:
            return frame, image_mask
        new_frame, new_mask = self.__concat_images(cropped_images, min_margin)

        if intermediate_steps is not None:
            intermediate_steps.append(DebugImage(new_frame, bottom_message=f"Collage: {len(cropped_images)} images"))
        return new_frame, new_mask

    def __concat_images(self, images: List[Tuple[Image, Image, Rect]], margin: int) -> (Image, Image):
        new_frame = images[0][0]
        new_mask = images[0][1]
        for image, mask, _ in images[1:]:
            new_frame = self.__concat_image(new_frame, image, margin)
            if new_mask is not None:
                new_mask = self.__concat_image(new_mask, mask, margin)
        return new_frame, new_mask

    def __concat_image(self, frame: Image, image: Image, overlap_width: int) -> Image:
        new_frame = frame.array.copy()
        new_image = image.array.copy()

        pad_height = new_frame.shape[0] - new_image.shape[0]
        if pad_height > 0:
            pad_before = pad_height // 2
            pad_after = pad_height - pad_before
            new_image = np.pad(new_image, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")

        alpha = np.linspace(1, 0, overlap_width).reshape(1, -1, 1)
        blended = alpha * new_frame[:, -overlap_width:] + (1 - alpha) * new_image[:, :overlap_width]
        new_frame = np.concatenate((new_frame[:, :-overlap_width], blended, new_image[:, overlap_width:]), axis=1)
        return Image(new_frame.astype(np.uint8))
