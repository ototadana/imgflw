import json
import traceback
import uuid
from typing import List

import gradio as gr
from PIL import Image as PILImage
from pydantic import ValidationError

from imgflw.entities import Config, Status
from imgflw.usecase import Settings, WorkflowGenerator, WorkflowStore
from imgflw.usecase.image_processor import ImageProcessor


class ImageInfo:
    def __init__(self, image: PILImage.Image) -> None:
        self.image = image

    def set_info(self, request: str, workflow: str, config: str) -> None:
        self.image.info["imgflw_request"] = request
        self.image.info["imgflw_workflow"] = workflow
        self.image.info["imgflw_config"] = config
        self.image.info["imgflw_id"] = uuid.uuid4().hex

    def get_info(self, key) -> str:
        if self.image is None or not hasattr(self.image, "info"):
            return ""
        return self.image.info.get(key, "")

    @property
    def request(self):
        return self.get_info("imgflw_request")

    @property
    def workflow(self):
        return self.get_info("imgflw_workflow")

    @property
    def config(self):
        return self.get_info("imgflw_config")

    @property
    def id(self):
        return self.get_info("imgflw_id")


class History:
    empty = ImageInfo(None)

    def __init__(self, capacity: int = 10) -> None:
        self.capacity = capacity
        self.clear()

    def undo(self) -> ImageInfo:
        if self.index < 0:
            return self.empty
        if self.index > 0:
            self.index -= 1
        return self.items[self.index]

    def redo(self) -> ImageInfo:
        if self.index < 0:
            return self.empty
        if self.index < len(self.items) - 1:
            self.index += 1
        return self.items[self.index]

    def clear(self) -> None:
        self.items: List[ImageInfo] = []
        self.index = -1

    def put(self, item: ImageInfo) -> None:
        if item is None:
            return
        if self.index > -1 and item.id == self.items[self.index].id:
            return
        self.items = self.items[: self.index + 1]
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)
        self.index = len(self.items) - 1

    def peek(self) -> ImageInfo:
        if self.index < 0:
            return self.empty
        return self.items[self.index]


class UIBuilder:
    def __init__(self, generator: WorkflowGenerator, store: WorkflowStore):
        self.generator = generator
        self.store = store
        self.status = None
        self.error_message = None
        self.intermediate_steps = None
        self.history = History()

    def setup_buttons(self):
        return gr.Button(visible=False), gr.Button(visible=True)

    def click_save_workflow(self, request: str, workflow: str):
        if not request or not workflow:
            return gr.TextArea("Request and workflow must be specified", visible=True)
        _, error = self.generate_and_validate_workflow(request, workflow, save=True)
        return error

    def generate_and_validate_workflow(self, request: str, workflow: str, save: bool = False):
        self.status = None
        self.error_message = None
        self.intermediate_steps = None

        try:
            if request:
                if not workflow:
                    workflow = self.generator.generate(request)
                    save = True
            ImageProcessor().validate_workflow(workflow)
        except json.JSONDecodeError as e:
            self.error_message = f"Error in JSON: {str(e)}"
        except ValidationError as e:
            errors = e.errors()
            if len(errors) == 0:
                self.error_message = f"{str(e)}\n\n{traceback.format_exc()}"
            else:
                err = errors[-1]
                self.error_message = f"{' -> '.join(str(er) for er in err['loc'])} {err['msg']}\n--\n{str(e)}"
        except Exception as e:
            self.error_message = f"{str(e)}\n\n{traceback.format_exc()}"

        error = gr.TextArea(self.error_message, visible=self.error_message is not None)

        if self.error_message is None and save:
            self.save_workflow(request, workflow)

        return workflow, error

    def generate_workflow(self, request: str, workflow: str):
        workflow, error = self.generate_and_validate_workflow(request, workflow, save=False)
        return workflow, error, gr.Tabs(selected=1)

    def save_workflow(self, request: str, workflow: str) -> None:
        self.store.save(request, workflow)

    def edit(self, request: str, workflow: str, image: PILImage.Image):
        self.status = Status()
        if self.error_message:
            return None, gr.TextArea(self.error_message, visible=True)

        config = Config(
            model=Settings.get("model"),
            prompt=Settings.get("prompt"),
            negative_prompt=Settings.get("negative_prompt"),
            use_minimal_area=Settings.get("use_minimal_area"),
            img2img_size=Settings.get("img2img_size"),
        )

        img = image
        output_img = None
        self.error_message = None
        try:
            if image is not None:
                output_img = ImageProcessor().process(image, workflow, config, self.status)
        except Exception as e:
            self.error_message = f"{str(e)}\n\n{traceback.format_exc()}"
        error = gr.TextArea(self.error_message, visible=self.error_message is not None)
        if output_img is not None:
            img = output_img.pil_image
            ImageInfo(img).set_info(request, workflow, config.model_dump_json())
        return img, error

    def save_settings(
        self,
        openai_api_key: str,
        model: str,
        prompt: str,
        negative_prompt: str,
        use_minimal_area: bool,
        img2img_size: int,
    ):
        Settings.set("openai_api_key", openai_api_key)
        Settings.set("model", model)
        Settings.set("prompt", prompt)
        Settings.set("negative_prompt", negative_prompt)
        Settings.set("use_minimal_area", use_minimal_area)
        Settings.set("img2img_size", img2img_size)
        Settings.save()

    def get_undo_button(self):
        return gr.Button(interactive=self.history.index > 0)

    def get_redo_button(self):
        return gr.Button(interactive=self.history.index < len(self.history.items) - 1)

    def change_image(self, image: PILImage.Image):
        info = ImageInfo(image)
        self.history.put(info)
        return (
            gr.Button(interactive=image is not None),
            self.get_undo_button(),
            self.get_redo_button(),
            gr.Image(interactive=image is None),
            gr.Button(interactive=image is not None),
        )

    def change_request(self, request: str):
        return request, self.store.get(request)

    def input_request(self, request: str):
        if not request or len(request) < 2:
            return gr.Radio(visible=False), ""
        choices = self.store.find(request) if request else []
        choices = choices if choices else []
        import time

        time.sleep(0.5)
        return gr.Radio(choices=choices, value=None, visible=len(choices) > 0), ""

    def get_intermediate_steps(self):
        if self.status is None or self.status.intermediate_steps is None or len(self.status.intermediate_steps) == 0:
            return []
        if self.intermediate_steps is None or len(self.status.intermediate_steps) != len(self.intermediate_steps):
            self.intermediate_steps = self.status.get_images(512)
        return self.intermediate_steps

    def cancel(self):
        self.status.canceled = True
        return gr.Button(visible=True), gr.Button(visible=False)

    def undo(self):
        self.status = None
        current = self.history.peek()
        item = self.history.undo()
        return current.request, current.workflow, item.image

    def redo(self):
        self.status = None
        item = self.history.redo()
        return item.request, item.workflow, item.image

    def clear_image(self):
        self.status = None
        self.history.clear()
        return (
            gr.Image(value=None, interactive=True),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            gr.Button(interactive=False),
        )

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="ImageFlow") as blocks:
            with gr.Tab("Canvas"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            clear_image_button = gr.Button(
                                value="🗑️ Clear",
                                scale=1,
                                size="sm",
                                variant="secondary",
                                interactive=False,
                            )
                            undo_button = gr.Button(
                                value="↩️ Undo", scale=1, size="sm", variant="secondary", interactive=False
                            )
                            redo_button = gr.Button(
                                value="↪️ Redo", scale=1, size="sm", variant="secondary", interactive=False
                            )
                        input_img = gr.Image(show_label=False, type="pil", show_download_button=True)
                        with gr.Row():
                            request = gr.Textbox(
                                elem_id="request",
                                scale=2,
                                label="Input your request here:",
                            )
                            edit_button = gr.Button(
                                value="▶️ Edit", scale=0, size="lg", variant="primary", interactive=False
                            )
                            cancel_button = gr.Button(
                                value="🚫 Cancel", scale=0, size="lg", variant="stop", visible=False
                            )
                        requests = gr.Radio(interactive=True, label="Did you mean?", visible=False)

                    with gr.Column():
                        with gr.Row():
                            with gr.Tabs(selected=1) as tabs:
                                with gr.Tab("Workflow Definition", id=1):
                                    workflow = gr.Code(language="json", container=False, label="Workflow Definition")
                                    with gr.Row():
                                        clear_workflow_button = gr.Button(
                                            value="🗑️ Clear", scale=1, size="lg", variant="secondary"
                                        )
                                        save_workflow_button = gr.Button(
                                            value="💾 Save", scale=1, size="lg", variant="secondary"
                                        )
                                with gr.Tab("Intermediate Steps", id=2):
                                    gr.Gallery(every=1, value=self.get_intermediate_steps, preview=True)
                        error_message = gr.Code(visible=False, label="Error Message")
            with gr.Tab("Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("##### Workflow Generator")
                        openai_api_key = gr.Textbox(label="OpenAI API Key", value=Settings.get("openai_api_key"))
                    with gr.Column():
                        gr.Markdown("##### Workflow Processor")
                        model = gr.Textbox(label="Model", value=Settings.get("model"))
                        prompt = gr.Textbox(label="Prompt", value=Settings.get("prompt"))
                        negative_prompt = gr.Textbox(label="Negative prompt", value=Settings.get("negative_prompt"))
                        use_minimal_area = gr.Checkbox(
                            label="Use minimal area (for close faces)", value=Settings.get("use_minimal_area")
                        )
                        img2img_size = gr.Dropdown(
                            label="Img2img size", choices=[512, 1024], value=Settings.get("img2img_size")
                        )
                with gr.Row():
                    save_settings_button = gr.Button(value="💾 Save", scale=0, size="lg", variant="primary")

            request.input(
                fn=self.input_request, inputs=[request], outputs=[requests, workflow], trigger_mode="always_last"
            )

            requests.select(fn=self.change_request, inputs=[requests], outputs=[request, workflow])

            input_img.change(
                fn=self.change_image,
                inputs=[input_img],
                outputs=[edit_button, undo_button, redo_button, input_img, clear_image_button],
            )

            edit_button.click(
                fn=self.setup_buttons,
                outputs=[edit_button, cancel_button],
                queue=False,
            ).then(
                fn=self.generate_workflow,
                inputs=[request, workflow],
                outputs=[workflow, error_message, tabs],
            ).then(
                fn=self.edit,
                inputs=[request, workflow, input_img],
                outputs=[input_img, error_message],
            ).then(
                fn=self.setup_buttons,
                outputs=[cancel_button, edit_button],
                queue=False,
            )

            undo_button.click(fn=self.undo, outputs=[request, workflow, input_img])
            redo_button.click(fn=self.redo, outputs=[request, workflow, input_img])
            clear_image_button.click(
                fn=self.clear_image, outputs=[input_img, clear_image_button, undo_button, redo_button]
            )
            clear_workflow_button.click(
                fn=lambda: ("", gr.TextArea("", visible=False)), outputs=[workflow, error_message]
            )
            save_workflow_button.click(fn=self.click_save_workflow, inputs=[request, workflow], outputs=[error_message])

            cancel_button.click(fn=self.cancel, outputs=[edit_button, cancel_button])
            save_settings_button.click(
                fn=self.save_settings,
                inputs=[openai_api_key, model, prompt, negative_prompt, use_minimal_area, img2img_size],
            )
            return blocks
