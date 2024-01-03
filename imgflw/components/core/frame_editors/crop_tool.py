from typing import Dict, List, Tuple

import cv2
import numpy as np

from imgflw.entities import Condition, DebugImage, Image, Rect
from imgflw.usecase import FrameEditor, condition_matcher


class CropTool(FrameEditor):
    def name(self) -> str:
        return "Crop"

    def edit(
        self,
        image: Image,
        mask_image: Image,
        faces: List[Rect],
        intermediate_steps: List[DebugImage],
        mode: str = "keep",
        reference_face: Dict[str, str] = {},
        aspect_ratio: str = "auto",
        margin: float = 2.0,
        **kwargs,
    ) -> Tuple[Image, Image]:
        image, mask_image, _ = self.edit_(
            image,
            mask_image,
            faces,
            intermediate_steps,
            mode=mode,
            reference_face=reference_face,
            aspect_ratio=aspect_ratio,
            margin=margin,
            **kwargs,
        )
        return image, mask_image

    def edit_(
        self,
        image: Image,
        mask_image: Image,
        faces: List[Rect],
        intermediate_steps: List[DebugImage],
        mode: str = "keep",
        reference_face: Dict[str, str] = {},
        aspect_ratio: str = "auto",
        margin: float = 2.0,
        top: int = 0,
        bottom: int = 0,
        dry_run: bool = False,
        **kwargs,
    ) -> Tuple[Image, Image, Rect]:
        mode = mode.lower() if mode else "keep"
        reference_face = reference_face if reference_face else {}
        aspect_ratio = aspect_ratio.lower() if aspect_ratio else "auto"

        frame: Image = image
        condition = Condition.model_validate(reference_face) if reference_face is not None else None
        target_faces: List[Rect] = []
        for face in faces:
            if condition_matcher.check_condition(condition, faces, face, frame.width, frame.height):
                target_faces.append(face)

        if len(target_faces) == 0:
            return image, mask_image, None

        overlay = frame.array.copy()
        output_image = frame.array.copy()

        face_areas = self.__get_area(target_faces, frame)
        rect = self.__add_margin(face_areas, margin, frame)
        if not dry_run:
            self.__rectangle(overlay, rect, (0, 255, 0), -1)
            self.__rectangle(overlay, face_areas, (255, 0, 0), 10)

        if mode == "remove":
            if condition.is_left():
                rect = Rect(rect.right, 0, frame.width, frame.height)
            if condition.is_right():
                rect = Rect(0, 0, rect.left, frame.height)
            if condition.is_top():
                rect = Rect(0, rect.bottom, frame.width, frame.height)
            if condition.is_bottom():
                rect = Rect(0, 0, frame.width, rect.top)

        if rect.width == 0 or rect.height == 0:
            return image, mask_image, None

        rect = self.__get_rect_by_aspect_ratio(rect, aspect_ratio, frame)
        if dry_run:
            return image, mask_image, rect

        if bottom - top > 0:
            rect = Rect(rect.left, top, rect.right, bottom)

        self.__rectangle(overlay, rect, (0, 255, 255), 10)
        image = Image(frame.pil_image.crop(rect.to_tuple()))
        mask_image = Image(mask_image.pil_image.crop(rect.to_tuple()))

        if intermediate_steps is not None:
            mask = np.zeros_like(output_image)
            self.__rectangle(mask, rect, (255, 255, 255), -1)
            cropped_image = cv2.bitwise_and(output_image, mask)
            output_image = cv2.addWeighted(output_image, 0.3, cropped_image, 0.7, 0)
            output_image = cv2.addWeighted(output_image, 0.7, overlay, 0.3, 0)
            mode_text = (
                f"{mode} {condition.criteria if condition.criteria else ''} {condition.tag if condition.tag else ''}"
            )
            intermediate_steps.append(
                DebugImage(
                    output_image,
                    bottom_message=f"crop: {rect.width}x{rect.height} ({aspect_ratio})",
                    top_message=mode_text,
                )
            )

        return image, mask_image, rect

    def __rectangle(self, image: np.ndarray, rect: Rect, color: Tuple[int, int, int], thickness: int = 10) -> None:
        left, top, right, bottom = rect.to_tuple()
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness)

    def __get_area(self, faces: List[Rect], frame: Image) -> Rect:
        rightmost = 0
        leftmost = frame.width
        topmost = frame.height
        bottommost = 0

        for face in faces:
            left, top, right, bottom = face.to_tuple()
            rightmost = max(rightmost, right)
            leftmost = min(leftmost, left)
            topmost = min(topmost, top)
            bottommost = max(bottommost, bottom)

        return Rect(leftmost, topmost, rightmost, bottommost)

    def __get_rect_by_aspect_ratio(self, rect: Rect, aspect_ratio: str, frame: Image) -> Rect:
        if aspect_ratio == "auto":
            return rect

        width = rect.width
        height = rect.height

        (wr, hr) = self.__get_aspect_ratio(aspect_ratio, width, height)

        if width * hr > height * wr:
            height = round(width * hr / wr)
        else:
            width = round(height * wr / hr)

        left = max(0, rect.left - round((width - rect.width) / 2))
        top = max(0, rect.top - round((height - rect.height) / 2))
        right = min(frame.width, left + width)
        bottom = min(frame.height, top + height)

        return Rect(left, top, right, bottom)

    def __add_margin(self, rect: Rect, margin: float, frame: Image) -> Rect:
        if margin == 0:
            return rect
        size = min(rect.width, rect.height)
        margin = (round(size * margin) - size) // 2
        left = max(0, rect.left - margin)
        top = max(0, rect.top - margin)
        right = min(frame.width, rect.right + margin)
        bottom = min(frame.height, rect.bottom + margin)
        return Rect(left, top, right, bottom)

    def __get_aspect_ratio(self, aspect_ratio: str, width: int, height: int) -> Tuple[int, int]:
        if aspect_ratio == "square":
            aspect_ratio = "1:1"
        if aspect_ratio == "portrait":
            aspect_ratio = "3:4"
        if aspect_ratio == "landscape":
            aspect_ratio = "4:3"

        ratio = aspect_ratio.split(":")
        return int(ratio[0]), int(ratio[1]) if len(ratio) > 1 else (width, height)
