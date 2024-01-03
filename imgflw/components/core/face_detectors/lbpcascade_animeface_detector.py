import os
from typing import List, Sequence

import cv2

from imgflw.entities import Image, Rect
from imgflw.usecase import FaceDetector


class LbpcascadeAnimefaceDetector(FaceDetector):
    def __init__(self) -> None:
        self.cascade_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lbpcascade_animeface.xml")

    def name(self):
        return "lbpcascade_animeface"

    def detect(self, image: Image, min_neighbors: int = 5, **kwargs) -> List[Rect]:
        cascade = cv2.CascadeClassifier(self.cascade_file)
        gray = cv2.cvtColor(image.array, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        xywhs = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(24, 24))
        return self.__xywh_to_ltrb(xywhs)

    def __xywh_to_ltrb(self, xywhs: Sequence) -> List[Rect]:
        ltrbs = []
        for xywh in xywhs:
            x, y, w, h = xywh
            ltrbs.append(Rect(x, y, x + w, y + h))
        return ltrbs
