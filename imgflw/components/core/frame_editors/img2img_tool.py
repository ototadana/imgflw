import math
from typing import List, Tuple

import torch
from diffusers import AutoPipelineForImage2Image

from imgflw.entities import DebugImage, Image, Rect, default
from imgflw.usecase import FrameEditor, Settings
from imgflw.usecase.image_processing_util import resize
from imgflw.usecase.mask_generator import MaskGenerator


class Img2ImgTool(FrameEditor):
    def __init__(self):
        self.__pipeline: AutoPipelineForImage2Image = None
        self.__model: str = None

    def name(self) -> str:
        return "img2img"

    def edit(
        self,
        image: Image,
        mask_image: Image,
        faces: List[Rect],
        intermediate_steps: List[DebugImage],
        model: str = "stabilityai/sd-turbo",
        pp: str = "",
        np: str = "",
        prompt: str = "",
        negative_prompt: str = "",
        strength: float = 0.3,
        no_mask: bool = False,
        img2img_size: int = 512,
        seed: int = 2,
        steps: int = 20,
        upscaler: str = default.UPSCALER,
        **kwargs,
    ) -> Tuple[Image, Image]:
        if strength == 0:
            return image, mask_image

        pp = pp or prompt
        np = np or negative_prompt

        if image.width < img2img_size and image.height < img2img_size:
            image = resize(image, img2img_size, upscaler=upscaler)
            mask_image = resize(mask_image, img2img_size)

        rounded_width = int(image.width // 8 * 8)
        rounded_height = int(image.height // 8 * 8)
        if image.width != rounded_width or image.height != rounded_height:
            left = (image.width - rounded_width) // 2
            top = (image.height - rounded_height) // 2
            right = left + rounded_width
            bottom = top + rounded_height
            image = Image(image.pil_image.crop((left, top, right, bottom)))
            mask_image = Image(mask_image.pil_image.crop((left, top, right, bottom)))

        mask = mask_image if not no_mask else None
        image = self.img2img(model, image, mask, pp, np, strength, seed, steps)

        if intermediate_steps is not None:
            masked_image = MaskGenerator.to_masked_image(mask_image.array, image.array)
            intermediate_steps.append(DebugImage(masked_image, bottom_message="img2img mask"))
            intermediate_steps.append(
                DebugImage(image, bottom_message=f"Prompt: {pp}", top_message=f"Strength: {strength}")
            )

        return image, mask_image

    def img2img(
        self, model: str, image: Image, mask: Image, pp: str, np: str, strength: int, seed: int, steps: int
    ) -> Image:
        if steps * strength < 1:
            steps = math.ceil(1 / strength)

        pipeline = self.__get_pipeline(model)
        generator = torch.Generator(Settings.device).manual_seed(seed)
        new_image = pipeline(
            pp,
            negative_prompt=np,
            image=image.pil_image,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=0.7,
            generator=generator,
        ).images[0]

        if mask is None:
            return Image(new_image)

        mask = mask.array
        original = image.array
        generated = Image(new_image).array

        foreground = (original * (mask / 255.0)).astype("uint8")
        background = (generated * (1 - (mask / 255.0))).astype("uint8")
        return Image(foreground + background)

    def __get_pipeline(self, model: str) -> AutoPipelineForImage2Image:
        if self.__pipeline is None or self.__model != model:
            self.__pipeline = self.__create_pipeline(model)
            self.__model = model

        return self.__pipeline

    def __create_pipeline(self, model: str) -> AutoPipelineForImage2Image:
        pipeline = AutoPipelineForImage2Image.from_pretrained(model, torch_dtype=torch.float16, variant="fp16")
        pipeline.to(Settings.device)
        pipeline.enable_model_cpu_offload()
        return pipeline
