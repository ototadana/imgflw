from abc import ABC, abstractmethod

from imgflw.entities import Image


class Upscaler(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def upscale(self, image: Image, **kwargs) -> Image:
        pass
