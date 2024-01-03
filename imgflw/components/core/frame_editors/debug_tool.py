from typing import List, Tuple

import cv2
import numpy as np

from imgflw.entities import DebugImage, Image, Rect
from imgflw.usecase import FrameEditor


class DebugTool(FrameEditor):
    def name(self) -> str:
        return "Debug"

    def edit(
        self, image: Image, mask_image: Image, faces: List[Rect], intermediate_steps: List[DebugImage], **kwargs
    ) -> Tuple[Image, Image]:
        if intermediate_steps is None:
            return image, mask_image

        image: np.ndarray = image.copy().array
        overlay = image.copy()
        color = (0, 0, 0)
        alpha = 0.3

        for face in faces:
            cv2.rectangle(overlay, (face.left, face.top), (face.right, face.bottom), color, 4)
            if face.landmarks is not None:
                for landmark in face.landmarks:
                    cv2.circle(overlay, (int(landmark.x), int(landmark.y)), 6, color, 4)

        output = Image(cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0))
        message = f"Faces: {len(faces)}" if len(faces) > 0 else "No faces detected"
        intermediate_steps.append(DebugImage(output, bottom_message=message))
        print(message, flush=True)

        return image, mask_image
