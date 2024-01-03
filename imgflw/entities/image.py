from typing import Union

import numpy as np
from PIL import Image as PILImage


class Image:
    def __init__(self, image: Union[np.ndarray, PILImage.Image, "Image"]):
        assert image is not None
        if isinstance(image, Image):
            image = image.array

        if isinstance(image, PILImage.Image):
            self.__array = None
            self.__pil_image = image
        else:
            self.__array = image
            self.__pil_image = None

    def copy(self) -> "Image":
        return Image(self.array.copy())

    @property
    def array(self) -> np.ndarray:
        if self.__array is None:
            self.__array = np.array(self.__pil_image, dtype=np.uint8)
        return self.__array

    @property
    def pil_image(self) -> PILImage.Image:
        if self.__pil_image is None:
            self.__pil_image = PILImage.fromarray(self.__array)
        return self.__pil_image

    @property
    def width(self) -> int:
        return self.array.shape[1] if self.array is not None else self.pil_image.width

    @property
    def height(self) -> int:
        return self.array.shape[0] if self.array is not None else self.pil_image.height
