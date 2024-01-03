import json
import traceback
from collections import OrderedDict
from typing import List

import gradio as gr
from PIL import Image as PILImage
from pydantic import ValidationError

from imgflw.entities import Config, Status
from imgflw.usecase import Settings, WorkflowGenerator
from imgflw.usecase.image_processor import ImageProcessor


class LRU:
    def __init__(self, items: List[str] = [], capacity: int = 20) -> None:
        self.capacity = capacity
        self.cache = OrderedDict()
        self.put_all(items)

    def put(self, key: str) -> None:
        if not key:
            return
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = key
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def put_all(self, items: List[str]) -> None:
        for item in items:
            self.put(item)

    def items(self) -> List[str]:
        return [key for key in self.cache.keys()]


class UIBuilder:
    def __init__(self, llm_generator: WorkflowGenerator, file_generator: WorkflowGenerator):
        self.llm_generator = llm_generator
        self.file_generator = file_generator
        self.status = None
        self.error_message = None
        self.requests = LRU(Settings.get("requests", []))
        self.intermediate_steps = None

    def setup_buttons(self):
        return gr.Button(visible=False), gr.Button(visible=True)

    def generate_workflow(self, request: str, workflow_json: str):
        self.status = None
        self.error_message = None
        try:
            if request:
                if request.endswith(".json") or request.startswith("http"):
                    generator = self.file_generator
                else:
                    generator = self.llm_generator
                workflow_json = generator.generate(request)
            ImageProcessor().validate_workflow(workflow_json)
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

        if self.error_message is None:
            self.requests.put(request)
            Settings.set("request", request)
            Settings.set("requests", self.requests.items())
            Settings.save()

        return gr.Dropdown(choices=self.requests.items(), value=request), workflow_json, error, gr.Tabs(selected=2)

    def setup_tabs(self):
        if self.error_message:
            return gr.Tabs(selected=2)
        self.status = Status()
        return gr.Tabs(selected=3)

    def edit(self, workflow_json: str, image: PILImage.Image):
        if self.error_message:
            return None, gr.TextArea(self.error_message, visible=True), gr.Tabs(selected=2)

        config = Config(
            model=Settings.get("model"),
            prompt=Settings.get("prompt"),
            negative_prompt=Settings.get("negative_prompt"),
            use_minimal_area=Settings.get("use_minimal_area"),
            img2img_size=Settings.get("img2img_size"),
        )

        output_img = None
        self.error_message = None
        try:
            if image is not None:
                output_img = ImageProcessor().process(image, workflow_json, config, self.status)
        except Exception as e:
            self.error_message = f"{str(e)}\n\n{traceback.format_exc()}"
        error = gr.TextArea(self.error_message, visible=self.error_message is not None)
        if output_img is not None:
            img = output_img.pil_image
            img.info["imgflw_workflow"] = workflow_json
            img.info["imgflw_config"] = config.model_dump_json()
        return img, error, gr.Tabs(selected=1 if self.error_message is None else 3)

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

    def change_image(self, image: PILImage.Image):
        workflow = image.info.get("imgflw_workflow", "")
        return gr.Button(interactive=image is not None), workflow

    def get_intermediate_steps(self):
        if self.status is None or self.status.intermediate_steps is None or len(self.status.intermediate_steps) == 0:
            return []
        if self.intermediate_steps is None or len(self.status.intermediate_steps) != len(self.intermediate_steps):
            self.intermediate_steps = self.status.get_images(512)
        return self.intermediate_steps

    def cancel(self):
        self.status.canceled = True
        return gr.Button(visible=True), gr.Button(visible=False)

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="ImageFlow") as blocks:
            with gr.Tab("Canvas"):
                with gr.Row():
                    with gr.Column():
                        input_img = gr.Image(show_label=False, type="pil")
                        with gr.Row():
                            request = gr.Dropdown(
                                allow_custom_value=True,
                                choices=self.requests.items(),
                                label="Input your request here:",
                                value=Settings.get("request", ""),
                            )
                            edit_button = gr.Button(
                                value="▶️ Edit", scale=0, size="lg", variant="primary", interactive=False
                            )
                            cancel_button = gr.Button(
                                value="🚫 Cancel",
                                scale=0,
                                size="lg",
                                variant="secondary",
                                visible=False,
                            )
                    with gr.Column():
                        with gr.Row():
                            with gr.Tabs(selected=1) as tabs:
                                with gr.Tab("Output Image", id=1):
                                    output_img = gr.Image(show_label=False, type="pil")
                                with gr.Tab("Workflow Definition", id=2):
                                    workflow_definition = gr.Code(
                                        language="json", container=False, label="Workflow Definition"
                                    )
                                with gr.Tab("Intermediate Steps", id=3):
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

            input_img.change(fn=self.change_image, inputs=[input_img], outputs=[edit_button, workflow_definition])
            edit_button.click(
                fn=self.setup_buttons,
                outputs=[edit_button, cancel_button],
                queue=False,
            ).then(
                fn=self.generate_workflow,
                inputs=[request, workflow_definition],
                outputs=[request, workflow_definition, error_message, tabs],
            ).then(
                fn=self.setup_tabs,
                outputs=[tabs],
                queue=False,
            ).then(
                fn=self.edit,
                inputs=[workflow_definition, input_img],
                outputs=[output_img, error_message, tabs],
            ).then(
                fn=self.setup_buttons,
                outputs=[cancel_button, edit_button],
                queue=False,
            )

            cancel_button.click(fn=self.cancel, outputs=[edit_button, cancel_button])
            save_settings_button.click(
                fn=self.save_settings,
                inputs=[openai_api_key, model, prompt, negative_prompt, use_minimal_area, img2img_size],
            )
            return blocks
