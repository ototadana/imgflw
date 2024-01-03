from typing import List

import cv2
import numpy as np
import torch
from facexlib.parsing import init_parsing_model
from facexlib.utils.misc import img2tensor
from torchvision.transforms.functional import normalize

from imgflw.components.core.mask_generators.vignette_mask_generator import VignetteMaskGenerator
from imgflw.entities import DebugImage, Face, Image
from imgflw.usecase import MaskGenerator, Settings


class BiSeNetMaskGenerator(MaskGenerator):
    def __init__(self):
        self.__fallback_mask_generator = VignetteMaskGenerator()
        self.__mask_model = None

    def name(self) -> str:
        return "BiSeNet"

    def generate_mask(
        self,
        face: Face,
        intermediate_steps: List[DebugImage],
        affected_areas: List[str] = ["Face"],
        mask_size: int = 0,
        use_minimal_area: bool = False,
        fallback_ratio: float = 0.5,
        mask_blur: int = 12,
        use_convex_hull: bool = False,
        **kwargs,
    ) -> None:
        face_image = face.face_image.array.copy()
        face_image = face_image[:, :, ::-1]

        h, w, _ = face_image.shape

        if w != 512 or h != 512:
            rw = (int(w * (512 / w)) // 8) * 8
            rh = (int(h * (512 / h)) // 8) * 8
            face_image = cv2.resize(face_image, dsize=(rw, rh))

        face_tensor = img2tensor(face_image.astype("float32") / 255.0, float32=True)
        normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        face_tensor = torch.unsqueeze(face_tensor, 0).to(Settings.device)

        if self.__mask_model is None:
            self.__mask_model = init_parsing_model(device=Settings.device)

        with torch.no_grad():
            face_image = self.__mask_model(face_tensor)[0]

        face_image = face_image.squeeze(0).cpu().numpy().argmax(0)
        face_image = face_image.copy().astype(np.uint8)

        mask = self.__to_mask(face_image, affected_areas, use_convex_hull)
        if mask_size > 0:
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=mask_size)

        if w != 512 or h != 512:
            mask = cv2.resize(mask, dsize=(w, h))

        if mask_blur > 0:
            mask = cv2.blur(mask, (mask_blur, mask_blur))

        mask_coverage = face.calculate_mask_coverage(mask)
        if mask_coverage < fallback_ratio:
            print(f"BiSeNetMaskGenerator: mask_coverage={mask_coverage * 100:.0f}% < {fallback_ratio * 100:.0f}%")
            self.__fallback_mask_generator.generate_mask(face, intermediate_steps, use_minimal_area=True)
            return

        face.mask_image = Image(mask)
        if use_minimal_area:
            face.mask_non_face_areas()

        if intermediate_steps is not None:
            self.add_debug_image(face, intermediate_steps)

    def __to_mask(self, face: np.ndarray, affected_areas: List[str], use_convex_hull: bool) -> np.ndarray:
        keep_face = "Face" in affected_areas
        keep_neck = "Neck" in affected_areas
        keep_hair = "Hair" in affected_areas
        keep_hat = "Hat" in affected_areas

        mask = np.zeros((face.shape[0], face.shape[1], 3), dtype=np.uint8)
        num_of_class = np.max(face)

        points = []
        for i in range(1, num_of_class + 1):
            index = np.where(face == i)
            if i < 14 and keep_face:
                mask[index[0], index[1], :] = [255, 255, 255]
                points.extend(zip(index[1], index[0]))
            elif i == 14 and keep_neck:
                mask[index[0], index[1], :] = [255, 255, 255]
                points.extend(zip(index[1], index[0]))
            elif i == 17 and keep_hair:
                mask[index[0], index[1], :] = [255, 255, 255]
                points.extend(zip(index[1], index[0]))
            elif i == 18 and keep_hat:
                mask[index[0], index[1], :] = [255, 255, 255]
                points.extend(zip(index[1], index[0]))
        if use_convex_hull and points:
            points = cv2.convexHull(np.array(points, dtype=np.float32).reshape(-1, 1, 2))
            cv2.fillConvexPoly(mask, np.int32(points), (255, 255, 255))
        return mask
