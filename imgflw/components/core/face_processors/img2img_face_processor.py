from typing import List

from imgflw.components.core.frame_editors.img2img_tool import Img2ImgTool
from imgflw.entities import DebugImage, Face, default
from imgflw.usecase import FaceProcessor
from imgflw.usecase.image_processing_util import resize, rotate


class Img2ImgFaceProcessor(FaceProcessor):
    def __init__(self):
        self.__img2img_tool = Img2ImgTool()

    def name(self) -> str:
        return "img2img"

    def process(
        self,
        face: Face,
        intermediate_steps: List[DebugImage],
        model: str = default.IMG2IMG_MODEL,
        pp: str = "",
        np: str = "",
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 0.4,
        img2img_size: int = default.IMG2IMG_SIZE,
        seed: int = 2,
        steps: int = 20,
        ignore_larger_faces=False,
        upscaler: str = default.UPSCALER,
        **kwargs,
    ) -> None:
        if ignore_larger_faces and face.width > img2img_size:
            message = f"ignore larger face:\n {face.width}x{face.height} > {img2img_size}x{img2img_size}"
            print(message, flush=True)
            if intermediate_steps is not None:
                intermediate_steps.append(DebugImage(face.face_image, bottom_message=message))
            return

        pp = pp or prompt
        np = np or negative_prompt

        angle = face.get_angle()
        new_image = rotate(face.face_image, angle)
        new_image = resize(new_image, img2img_size, upscaler=upscaler)

        new_image = self.__img2img_tool.img2img(model, new_image, None, pp, np, strength, seed, steps)

        new_image = resize(new_image, face.width)
        new_image = rotate(new_image, -angle)
        face.face_image = new_image

        if intermediate_steps is not None:
            intermediate_steps.append(
                DebugImage(face.face_image, bottom_message=f"Prompt: {pp}", top_message=f"Strength: {strength}")
            )
