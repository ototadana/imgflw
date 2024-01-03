import traceback
from typing import Tuple

import cv2
import numpy as np

from .image import Image
from .rect import Point, Rect


class Face:
    def __init__(self, entire_image: Image, face_area: Rect, face_margin: float):
        self.face_area = face_area
        self.center = face_area.center
        left, top, right, bottom = face_area.to_square()

        self.left, self.top, self.right, self.bottom = self.__ensure_margin(
            left, top, right, bottom, entire_image, face_margin
        )

        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.face_image = Image(entire_image.array[self.top : self.bottom, self.left : self.right, :])
        self.face_area_on_face_image = self.__get_face_area_on_face_image()
        self.landmarks_on_face_image = self.__get_landmarks_on_face_image()
        l, t, r, b = self.face_area_on_face_image
        self.face_area_total_pixels = (r - l) * (b - t)
        self.mask_image = Image(np.ones((self.height, self.width, 3), np.uint8) * 255)

    def __get_face_area_on_face_image(self):
        left = int((self.face_area.left - self.left))
        top = int((self.face_area.top - self.top))
        right = int((self.face_area.right - self.left))
        bottom = int((self.face_area.bottom - self.top))
        return self.__clip_values(left, top, right, bottom)

    def __get_landmarks_on_face_image(self):
        landmarks = []
        if self.face_area.landmarks is not None:
            for landmark in self.face_area.landmarks:
                landmarks.append(
                    Point(
                        int((landmark.x - self.left)),
                        int((landmark.y - self.top)),
                    )
                )
        return landmarks

    def __ensure_margin(
        self, left: int, top: int, right: int, bottom: int, entire_image: Image, margin: float
    ) -> Tuple[int, int, int, int]:
        entire_height, entire_width = entire_image.array.shape[:2]

        side_length = right - left
        margin = min(min(entire_height, entire_width) / side_length, margin)
        diff = int((side_length * margin - side_length) / 2)

        top = top - diff
        bottom = bottom + diff
        left = left - diff
        right = right + diff

        if top < 0:
            bottom = bottom - top
            top = 0
        if left < 0:
            right = right - left
            left = 0

        if bottom > entire_height:
            top = top - (bottom - entire_height)
            bottom = entire_height
        if right > entire_width:
            left = left - (right - entire_width)
            right = entire_width

        return left, top, right, bottom

    def get_angle(self) -> float:
        landmarks = getattr(self.face_area, "landmarks", None)
        if landmarks is None:
            return 0

        eye1 = getattr(landmarks, "eye1", None)
        eye2 = getattr(landmarks, "eye2", None)
        if eye2 is None or eye1 is None:
            return 0

        try:
            dx = eye2.x - eye1.x
            dy = eye2.y - eye1.y
            if dx == 0:
                dx = 1
            angle = np.arctan(dy / dx) * 180 / np.pi

            if dx < 0:
                angle = (angle + 180) % 360
            return angle
        except Exception:
            print(traceback.format_exc())
            return 0

    def rotate_face_area_on_cropped_image(self, angle: float):
        center = [
            (self.face_area_on_face_image[0] + self.face_area_on_face_image[2]) / 2,
            (self.face_area_on_face_image[1] + self.face_area_on_face_image[3]) / 2,
        ]

        points = [
            [self.face_area_on_face_image[0], self.face_area_on_face_image[1]],
            [self.face_area_on_face_image[2], self.face_area_on_face_image[3]],
        ]

        angle = np.radians(angle)
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        points = np.array(points) - center
        points = np.dot(points, rot_matrix.T)
        points += center
        left, top, right, bottom = (int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]))

        left, right = (right, left) if left > right else (left, right)
        top, bottom = (bottom, top) if top > bottom else (top, bottom)

        width, height = right - left, bottom - top
        if width < height:
            left, right = left - (height - width) // 2, right + (height - width) // 2
        elif height < width:
            top, bottom = top - (width - height) // 2, bottom + (width - height) // 2
        return self.__clip_values(left, top, right, bottom)

    def __clip_values(self, left, top, right, bottom):
        left = min(self.width, max(0, left))
        top = min(self.height, max(0, top))
        right = min(self.width, max(0, right))
        bottom = min(self.height, max(0, bottom))
        return left, top, right, bottom

    def merge(self, entire_image: Image, entire_mask_image: Image, use_minimal_area: bool) -> (Image, Image):
        face_image = self.face_image.array
        mask_image = self.mask_image.array

        top = self.top
        left = self.left
        bottom = self.bottom
        right = self.right

        if use_minimal_area:
            left, top, right, bottom = self.face_area.to_tuple()
            face_image = face_image[top - self.top : bottom - self.top, left - self.left : right - self.left]
            mask_image = mask_image[top - self.top : bottom - self.top, left - self.left : right - self.left]

        face_background = entire_image.array[top:bottom, left:right]
        face_fg = (face_image * (mask_image / 255.0)).astype("uint8")
        face_bg = (face_background * (1 - (mask_image / 255.0))).astype("uint8")
        face_image = face_fg + face_bg

        entire_image.array[top:bottom, left:right] = face_image
        entire_mask_image.array[top:bottom, left:right] = mask_image
        return Image(entire_image.array), Image(entire_mask_image.array)

    def mask_non_face_areas(self) -> None:
        self.mask_image = Image(self.get_mask_non_face_areas())

    def get_mask_non_face_areas(self) -> np.ndarray:
        image = self.mask_image.array
        left, top, right, bottom = self.face_area_on_face_image

        image = image.copy()
        image[:top, :] = 0
        image[bottom:, :] = 0
        image[:, :left] = 0
        image[:, right:] = 0
        return image

    def calculate_mask_coverage(self, mask: np.ndarray) -> float:
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        non_black_pixels = np.count_nonzero(gray_mask)
        return non_black_pixels / self.face_area_total_pixels
