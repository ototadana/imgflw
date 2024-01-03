import importlib.util
import inspect
import os
import traceback
from typing import Dict, List, Type

from imgflw.io import util as io_util
from imgflw.usecase import FaceDetector, FaceProcessor, FrameEditor, MaskGenerator, Upscaler


def load_classes_from_file(file_path: str, base_class: Type) -> List[Type]:
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    classes = []

    try:
        spec.loader.exec_module(module)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, base_class) and cls is not base_class:
                classes.append(cls)
    except Exception as e:
        print(file_path, ":", e, flush=True)
        raise e

    return classes


def load_classes_from_directory(base_class: Type, dir: str) -> List[Type]:
    all_classes = []
    dir = io_util.get_module(dir)
    for file in os.listdir(dir):
        if file.endswith(".py") and file != os.path.basename(__file__):
            file_path = os.path.join(dir, file)
            try:
                classes = load_classes_from_file(file_path, base_class)
                if classes:
                    all_classes.extend(classes)
            except Exception as e:
                print(f"Can't load {file_path}", flush=True)
                print(str(e), flush=True)
                print(traceback.format_exc(), flush=True)

    return all_classes


def create(all_classes, type: str) -> Dict:
    d = {}
    for cls in all_classes:
        try:
            c = cls()
            d[c.name().lower()] = c
        except Exception as e:
            print(traceback.format_exc())
            print(f"Face Editor: {cls}, Error: {e}")
    return d


face_detectors = create(load_classes_from_directory(FaceDetector, "components/core/face_detectors"), "FaceDetector")
face_processors = create(load_classes_from_directory(FaceProcessor, "components/core/face_processors"), "FaceProcessor")
mask_generators = create(load_classes_from_directory(MaskGenerator, "components/core/mask_generators"), "MaskGenerator")
frame_editors = create(load_classes_from_directory(FrameEditor, "components/core/frame_editors"), "FrameEditor")
upscalers = create(load_classes_from_directory(Upscaler, "components/core/upscalers"), "Upscaler")

face_detector_names = list(face_detectors.keys())
face_processor_names = list(face_processors.keys())
mask_generator_names = list(mask_generators.keys())
frame_editor_names = list(frame_editors.keys())
upscaler_names = list(upscalers.keys())


def get_face_detector(name: str) -> FaceDetector:
    return face_detectors[name.lower()]


def get_face_processor(name: str) -> FaceProcessor:
    return face_processors[name.lower()]


def get_mask_generator(name: str) -> MaskGenerator:
    return mask_generators[name.lower()]


def get_frame_editor(name: str) -> FrameEditor:
    return frame_editors[name.lower()]


def get_upscaler(name: str) -> Upscaler:
    return upscalers[name.lower()]


def has_face_detector(name: str) -> bool:
    return name.lower() in face_detectors


def has_face_processor(name: str) -> bool:
    return name.lower() in face_processors


def has_mask_generator(name: str) -> bool:
    return name.lower() in mask_generators


def has_frame_editor(name: str) -> bool:
    return name.lower() in frame_editors


def has_upscaler(name: str) -> bool:
    return name.lower() in upscalers
