from operator import attrgetter
from typing import List, Tuple, Union

import numpy as np
from PIL import Image as PILImage

from imgflw.entities import Config, DebugImage, Face, Image, Job, Rect, Rule, Status, Worker, Workflow
from imgflw.usecase import component_registry as registry
from imgflw.usecase import condition_matcher, query_matcher


class ImageProcessor:
    def validate_workflow(self, workflow: Union[Workflow, str]) -> Workflow:
        if isinstance(workflow, str):
            workflow: Workflow = Workflow.model_validate_json(workflow)
        self.__validate_workflow(workflow)
        return workflow

    def process(
        self,
        image: Union[np.ndarray, PILImage.Image, Image],
        workflow: Union[Workflow, str],
        config: Union[Workflow, str],
        status: Status,
    ) -> Image:
        workflow = self.validate_workflow(workflow)

        if isinstance(config, str):
            config: Config = Config.model_validate_json(config)

        image = Image(image)
        mask_image = Image(np.zeros_like(image.array))
        status.intermediate_steps: List[DebugImage] = [] if config.show_intermediate_steps else None
        faces = self.__detect_faces(workflow, image, config, status)

        if self.__has_preprocessors(workflow):
            image, mask_image = self.__preprocess(workflow, image, mask_image, faces, config, status)
            faces = self.__detect_faces(workflow, image, config, status)

        for i, _ in enumerate(faces):
            if status.canceled:
                return image

            image, mask_image = self.__process_face_area(workflow, image, mask_image, faces, i, config, status)

        if status.canceled:
            return image

        if self.__has_postprocessors(workflow):
            image, mask_image = self.__postprocess(workflow, image, mask_image, faces, config, status)

        return image

    def __process_face_area(
        self,
        workflow: Workflow,
        entire_image: Image,
        entire_mask_image: Image,
        face_areas: List[Rect],
        index: int,
        config: Config,
        status: Status,
    ) -> Tuple[Image, Image]:
        rule = self.__select_rule(workflow, face_areas, index, entire_image.width, entire_image.height)
        if rule is None or len(rule.then) == 0:
            return entire_image, entire_mask_image

        face_area = face_areas[index]
        face = Face(entire_image, face_area, config.face_margin)
        for job in rule.then:
            if status.canceled:
                return entire_image, entire_mask_image

            face_intermediate_steps: List[DebugImage] = [] if status.intermediate_steps is not None else None
            if face_intermediate_steps is not None:
                face_intermediate_steps.append(self.__create_debug_image_for_detected(face_area, rule, face))

            self.__process_face(job, face, config, face_intermediate_steps)
            self.__generate_mask(job, face, config, face_intermediate_steps)
            entire_image, entire_mask_image = face.merge(entire_image, entire_mask_image, config.use_minimal_area)

            if status.intermediate_steps is not None:
                face_image = Face(entire_image, face_area, config.face_margin).face_image.copy()
                face_intermediate_steps.append(DebugImage(face_image))
                status.intermediate_steps.append(self.__create_debug_image_for_face(face_intermediate_steps, config))
        return entire_image, entire_mask_image

    def __create_debug_image_for_face(self, face_intermediate_steps: List[DebugImage], config: Config) -> DebugImage:
        img = np.zeros((config.img2img_size * 2, config.img2img_size * 2, 3), dtype=np.uint8)
        size = config.img2img_size

        img[0:size, 0:size] = face_intermediate_steps[0].get_image(size).array
        if len(face_intermediate_steps) > 1:
            img[0:size, size:] = face_intermediate_steps[1].get_image(size).array
        if len(face_intermediate_steps) > 2:
            img[size:, 0:size] = face_intermediate_steps[2].get_image(size).array
        if len(face_intermediate_steps) > 3:
            img[size:, size:] = face_intermediate_steps[3].get_image(size).array

        return DebugImage(Image(img))

    def __create_debug_image_for_detected(self, face_area: Rect, rule: Rule, face: Face) -> DebugImage:
        top_message = f"{face_area.tag} ({face_area.width}x{face_area.height})"
        criteria = rule.when.criteria if rule.when is not None and rule.when.criteria is not None else ""
        attributes = str(face_area.attributes) if face_area.attributes else ""
        if criteria != "" and attributes != "":
            bottom_message = f"{criteria}\n{attributes}"
        else:
            bottom_message = f"{criteria}{attributes}"

        face_image = face.face_image.copy()
        return DebugImage(face_image, top_message=top_message, bottom_message=bottom_message)

    def __generate_mask(self, job: Job, face: Face, config: Config, intermediate_steps: List[DebugImage]) -> None:
        mg = job.mask_generator
        mask_generator = registry.get_mask_generator(mg.name)
        params = config.model_dump().copy()
        params.update(mg.params)
        mask_generator.generate_mask(face, intermediate_steps, **params)

    def __process_face(
        self,
        job: Job,
        face: Face,
        config: Config,
        intermediate_steps: List[DebugImage],
    ) -> None:
        fp = job.face_processor
        face_processor = registry.get_face_processor(fp.name)
        params = config.model_dump().copy()
        params.update(fp.params)
        face_processor.process(face, intermediate_steps, **params)

    def __select_rule(self, workflow: Workflow, faces: List[Rect], index: int, width: int, height: int) -> Rule:
        face = faces[index]

        rules = workflow.rules if workflow.rules is not None else []
        for rule in rules:
            if rule.when is None:
                return rule
            if condition_matcher.check_condition(rule.when, faces, face, width, height):
                return rule

        return None

    def __validate_workflow(self, workflow: Workflow):
        for face_detector in workflow.face_detector:
            if not registry.has_face_detector(face_detector.name):
                raise KeyError(f"face_detector `{face_detector.name}` does not exist")

        rules = workflow.rules if workflow.rules is not None else []
        for rule in rules:
            for job in rule.then:
                if not registry.has_face_processor(job.face_processor.name):
                    raise KeyError(f"face_processor `{job.face_processor.name}` does not exist")
                if not registry.has_mask_generator(job.mask_generator.name):
                    raise KeyError(f"mask_generator `{job.mask_generator.name}` does not exist")
            if rule.when is not None and rule.when.tag is not None and "?" in rule.when.tag:
                _, query = condition_matcher.parse_tag(rule.when.tag)
                if len(query) > 0:
                    query_matcher.validate(query)

        self.__validate_frame_editors(workflow.preprocessors)
        self.__validate_frame_editors(workflow.postprocessors)

    def __validate_frame_editors(self, frame_editors: List[Worker]) -> List[Worker]:
        if frame_editors is None:
            return

        for frame_editor in frame_editors:
            if not registry.has_frame_editor(frame_editor.name):
                raise KeyError(f"frame_editor `{frame_editor.name}` does not exist")

    def __detect_faces(self, workflow: Workflow, image: Image, config: Config, status: Status) -> List[Rect]:
        results = []

        for fd in workflow.face_detector:
            face_detector = registry.get_face_detector(fd.name)
            params = config.model_dump().copy()
            params.update(fd.params)
            results.extend(face_detector.detect(image, **params))

        faces = sorted(results, key=attrgetter("height"), reverse=True)
        faces = faces[: config.max_face_count]
        faces = sorted(faces, key=attrgetter("center"))

        if status.intermediate_steps is not None:
            debug_tool = registry.get_frame_editor("debug")
            debug_tool.edit(image, None, faces, status.intermediate_steps)

        return faces

    def __has_preprocessors(self, workflow: Workflow) -> bool:
        return workflow.preprocessors is not None and len(workflow.preprocessors) > 0

    def __has_postprocessors(self, workflow: Workflow) -> bool:
        return workflow.postprocessors is not None and len(workflow.postprocessors) > 0

    def __preprocess(
        self, workflow: Workflow, image: Image, mask_image: Image, config: Config, faces: List[Rect], status: Status
    ) -> Tuple[Image, Image]:
        if workflow.preprocessors is None:
            return image, mask_image

        return self.__edit(workflow.preprocessors, image, mask_image, config, faces, status)

    def __postprocess(
        self, workflow: Workflow, image: Image, mask_image: Image, faces: List[Rect], config: Config, status: Status
    ) -> Tuple[Image, Image]:
        if workflow.postprocessors is None:
            return image, mask_image

        return self.__edit(workflow.postprocessors, image, mask_image, faces, config, status)

    def __edit(
        self,
        frame_editors: List[Worker],
        image: Image,
        mask_image: Image,
        faces: List[Rect],
        config: Config,
        status: Status,
    ) -> Tuple[Image, Image]:
        for fe in frame_editors:
            if status.canceled:
                return image
            print(f"frame_editor: {fe.name}", flush=True)
            frame_editor = registry.get_frame_editor(fe.name)
            params = config.model_dump().copy()
            params.update(fe.params)
            image, mask_image = frame_editor.edit(image, mask_image, faces, status.intermediate_steps, **params)
        return image, mask_image
