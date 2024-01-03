import cv2
from PIL import Image as PILImage

from imgflw.entities import Image
from imgflw.usecase import component_registry as registry


def rotate(image: Image, angle: float) -> Image:
    if angle == 0:
        return image

    h, w = image.array.shape[:2]
    center = (w // 2, h // 2)

    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return Image(cv2.warpAffine(image.array, m, (w, h)))


def resize(image: Image, width: int, height: int = None, upscaler: str = None) -> Image:
    if image.width == width:
        return image

    width = int(width)
    if height is None:
        height = round(image.height * width / image.width)

    if image.width < width:
        return upscale(image, width, height, upscaler)
    else:
        return downscale(image, width, height)


def downscale(image: Image, width: int, height: int) -> Image:
    return Image(image.pil_image.resize((width, height), resample=PILImage.LANCZOS))


def upscale(image: Image, width: int, height: int, upscaler_name: str = None) -> Image:
    width = int(width // 8 * 8)
    height = int(height // 8 * 8)

    if upscaler_name:
        upscaler = registry.get_upscaler(upscaler_name)

        for _ in range(3):
            if image.width >= width and image.height >= height:
                break

            original_size = (image.width, image.height)

            image = upscaler.upscale(image)

            if original_size == (image.width, image.height):
                break

    if image.width != width or image.height != height:
        image = image.pil_image.resize((width, height), resample=PILImage.LANCZOS)

    return Image(image)
