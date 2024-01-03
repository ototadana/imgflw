from typing import List

import torch
from facexlib.detection import init_detection_model, retinaface

from imgflw.entities import Image, Landmarks, Point, Rect
from imgflw.usecase import FaceDetector, Settings


class RetinafaceDetector(FaceDetector):
    def __init__(self) -> None:
        if hasattr(retinaface, "device"):
            retinaface.device = Settings.device
        self.detection_model = None

    def name(self):
        return "RetinaFace"

    def detect(self, image: Image, confidence: float = 0.9, **kwargs) -> List[Rect]:
        if self.detection_model is None:
            self.detection_model = init_detection_model("retinaface_resnet50", device=Settings.device)

        with torch.no_grad():
            boxes_landmarks = self.detection_model.detect_faces(image.array, confidence)

        faces = []
        for box_landmark in boxes_landmarks:
            face_box = box_landmark[:5]
            landmark = box_landmark[5:]
            face = Rect.from_ndarray(face_box)

            eye1 = Point(int(landmark[0]), int(landmark[1]))
            eye2 = Point(int(landmark[2]), int(landmark[3]))
            nose = Point(int(landmark[4]), int(landmark[5]))
            mouth2 = Point(int(landmark[6]), int(landmark[7]))
            mouth1 = Point(int(landmark[8]), int(landmark[9]))

            face.landmarks = Landmarks(eye1, eye2, nose, mouth1, mouth2)
            faces.append(face)

        return faces
