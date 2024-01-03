from typing import Union

import cv2
import numpy as np
from PIL import Image as PILImage

from imgflw.entities.image import Image


class DebugImage:
    def __init__(self, image: Union[np.ndarray, PILImage.Image], bottom_message: str = None, top_message: str = None):
        image = image if isinstance(image, Image) else Image(image)
        self.image = image.copy().array
        self.bottom_message = bottom_message
        self.top_message = top_message

    def get_image(self, size: int) -> Image:
        image = self.__resize(size)

        if self.bottom_message:
            image = self.__add_comment(image, self.bottom_message, top=False)
        if self.top_message:
            image = self.__add_comment(image, self.top_message, top=True)

        return Image(image)

    def __add_comment(self, image: np.ndarray, comment: str, top: bool = False) -> np.ndarray:
        h, _, _ = image.shape
        lines = comment.split("\n")
        dy = 40
        for i, line in enumerate(reversed(lines) if not top else lines):
            y = (48 + i * dy) if top else (h - 16 - i * dy)
            pos = (10, y)
            cv2.putText(
                image,
                text=line,
                org=pos,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=(0, 0, 0),
                thickness=10,
            )
            cv2.putText(
                image,
                text=line,
                org=pos,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=(255, 255, 255),
                thickness=2,
            )
        return image

    def __resize(self, size: int) -> np.ndarray:
        height, width = self.image.shape[:2]
        if height == width:
            return cv2.resize(self.image, dsize=(size, size))

        aspect_ratio = width / height

        if width > height:
            new_width = size
            new_height = int(size / aspect_ratio)
        else:
            new_height = size
            new_width = int(size * aspect_ratio)

        resized_image = cv2.resize(self.image, (new_width, new_height))
        image = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - new_height) // 2
        x_offset = (size - new_width) // 2
        image[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized_image
        return image
